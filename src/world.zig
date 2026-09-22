//! Große Welten: Streaming von Chunks mit LOD, erzeugt und gebaut auf der GPU.
//!
//! Je Update: Planer (world_plan.zig) bestimmt benötigte Chunks um die Kamera
//! → Generator-Kernel schreibt Voxel direkt in der Auflösung der LOD-Stufe
//! (nur die Oberflächenhaut, nie ein volles Volumen) → ein GPU-DAG-Bau für
//! bis zu chunks_per_update Chunks → je Chunk eine Geometrie mit gemeinsamen
//! Pool-Bereichen, alle GAS in einem Zug → Instanzen; nicht sichtbare Chunks
//! haben Maske 0, lange nicht gebrauchte werden freigegeben.
//!
//! Auf dem Host liegen nur Chunk-Schlüssel und Handles; Voxel sehen ihn nie.

const std = @import("std");
const types = @import("pyrit_device").types;
const api = @import("api.zig");
const cuda = @import("cuda.zig");
const diag = @import("diag.zig");
const gpu_build = @import("gpu_build.zig");
const world_plan = @import("world_plan.zig");
const Context = @import("context.zig").Context;

const Error = diag.Error;
const fail = diag.fail;
const Key = world_plan.Key;

fn oom(v: anytype) Error!@typeInfo(@TypeOf(v)).error_union.payload {
    return v catch return fail(error.OutOfMemory, "Host-Speicher", .{});
}

pub fn defaultTerrain() types.TerrainParams {
    return .{
        .seed = 1,
        .octaves = 9,
        .base_height = 40,
        .amplitude = 260,
        .wavelength = 1400,
        .sea_level = 56,
        .snow_height = 250,
        .rock_slope = 1.1,
        .attr_grass = 0,
        .attr_dirt = 0,
        .attr_rock = 0,
        .attr_snow = 0,
        .attr_sand = 0,
        // Wasser und Bäume: 0 lässt sie weg. Die Materialien legt die
        // Anwendung an (Wasser mit PYR_MATERIAL_TRANSPARENT).
        .attr_water = 0,
        .attr_leaves = 0,
        .attr_wood = 0,
        .tree_density = 0.004,
    };
}

const Chunk = struct {
    key: Key,
    geometry: usize,
    instance: usize,
    /// zuletzt gesetzte Instanzmaske
    mask: u32,
    /// Frame der feinen bzw. groben Auswahl
    stamp: u64,
    stamp_coarse: u64,
    voxels: u32,
};

pub const World = struct {
    ctx: *Context,
    plan: world_plan.Plan,
    chunk_log2: u32,
    batch_max: u32,
    capacity: u32,
    mask: u32,
    rt_log2: u32,
    generate: api.WorldGenFn,
    user: ?*anyopaque,
    terrain: types.TerrainParams,

    // Gerätepuffer (bleiben über alle Updates)
    keys_dev: cuda.CUdeviceptr = 0,
    voxels_dev: cuda.CUdeviceptr = 0,
    counts_dev: cuda.CUdeviceptr = 0,
    offsets_dev: cuda.CUdeviceptr = 0,
    /// gepinnter Host-Puffer: Chunkliste, Zähler, Präfixe
    pinned: ?*anyopaque = null,

    chunks: std.ArrayList(Chunk) = .empty,
    chunk_free: std.ArrayList(u32) = .empty,
    requests: std.ArrayList(Key) = .empty,
    evicted: std.ArrayList(world_plan.Node) = .empty,
    origin: [3]f64 = .{ 0, 0, 0 },
    stats: api.WorldStats = std.mem.zeroes(api.WorldStats),
    /// Maske und Vergröberung der Auswahl für Sekundärstrahlen (0 = aus)
    secondary_mask: u32 = 0,
    secondary_factor: f64 = 4,
    /// Zielgröße eines Voxels in Pixeln (Vorgabe) und Budget
    voxel_pixels: f64 = 4,
    memory_budget: u64 = 0,
    /// gleitende Anpassung an Budget und Platzgrenzen (1 = volle Feinheit)
    lod_scale: f64 = 1,
    /// so viele Chunks dürfen höchstens liegen (aus max_geometries)
    chunk_limit: u32 = 0,
    /// Pixel je Grundvoxel in Entfernung 1 (aus der Kamera)
    pixels_per_unit: f64 = 1000,

    // Auftrag an den Arbeiter (ein Batch zur Zeit)
    thread: ?std.Thread = null,
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
    quit: bool = false,
    job_state: enum { idle, submitted, done } = .idle,
    /// nur Hauptthread: kein Auftrag unterwegs
    job_idle: bool = true,
    job_count: u32 = 0,
    job_total: u32 = 0,
    job_overflow: u32 = 0,
    job_built: ?gpu_build.Built = null,
    job_roots: []u32 = &.{},
    job_error: ?Error = null,
    job_message: [512]u8 = undefined,

    pub fn create(ctx: *Context, info: *const api.WorldInfo) Error!*World {
        const cl: u32 = if (info.chunk_log2 == 0) 5 else info.chunk_log2;
        if (cl < 3 or cl > 8) return fail(error.InvalidArgument, "chunk_log2 muss in [3, 8] liegen", .{});
        const max_lod: u32 = if (info.max_lod == 0) 20 else info.max_lod;
        if (max_lod > 20) return fail(error.InvalidArgument, "max_lod höchstens 20", .{});
        const voxel_px: f64 = if (info.voxel_pixels > 0) info.voxel_pixels else 4;
        const view_distance: f64 = if (info.view_distance > 0) info.view_distance else 16384;
        const terrain = if (info.terrain) |t| t.* else defaultTerrain();
        var y_min = info.y_min;
        var y_max = info.y_max;
        if (y_min == 0 and y_max == 0) {
            if (info.generate != null) return fail(error.InvalidArgument, "mit eigenem Generator y_min/y_max angeben", .{});
            y_min = @intFromFloat(@floor(terrain.base_height - 16));
            y_max = @intFromFloat(@ceil(terrain.base_height + terrain.amplitude * 1.6 + 16));
        }
        if (y_max <= y_min) return fail(error.InvalidArgument, "y_max muss größer als y_min sein", .{});
        const n: u32 = @as(u32, 1) << @intCast(cl);
        const batch: u32 = if (info.chunks_per_update == 0) 64 else info.chunks_per_update;
        if (batch > 4096) return fail(error.InvalidArgument, "chunks_per_update höchstens 4096", .{});
        const cap: u32 = if (info.chunk_capacity == 0) @min(n * n * 8, n * n * n) else @min(info.chunk_capacity, n * n * n);
        // gemessen: kleinere AABBs sind in der Welt schneller (dünne Oberflächen,
        // die Hardware-BVH trennt sie besser als der DAG-Lauf im Shader)
        const rt_log2: u32 = if (info.rt_leaf_log2 == 0) @max(cl -| 2, 3) else std.math.clamp(info.rt_leaf_log2, 3, cl);

        const w = try oom(ctx.gpa.create(World));
        errdefer ctx.gpa.destroy(w);
        w.* = .{
            .ctx = ctx,
            .plan = .{ .gpa = ctx.gpa, .cfg = .{
                .chunk_log2 = cl,
                .max_lod = max_lod,
                .view_chunks = if (info.view_chunks == 0) 2 else info.view_chunks,
                .view_distance = view_distance,
                // bis zur ersten Kamera: 1000 px je Einheit in Entfernung 1 (≈ 1080p, 60°)
                .refine_k = 1000.0 / voxel_px,
                .y_min = y_min,
                .y_max = y_max,
                .evict_frames = if (info.keep_frames == 0) 8 else info.keep_frames,
            } },
            .chunk_log2 = cl,
            .batch_max = batch,
            .capacity = cap,
            .mask = if (info.mask == 0) 0x1 else info.mask,
            .rt_log2 = rt_log2,
            .generate = info.generate,
            .user = info.user,
            .terrain = terrain,
            .voxel_pixels = voxel_px,
            .secondary_mask = info.secondary_mask,
            .secondary_factor = if (info.secondary_pixels > 0) @as(f64, info.secondary_pixels) / voxel_px else 4,
            .memory_budget = if (info.memory_budget == 0) 256 << 20 else info.memory_budget,
        };
        // Platzgrenze: die Welt bleibt unter den Geometrieplätzen des Kontexts
        w.chunk_limit = @intCast(ctx.geometries.len * 3 / 4);
        errdefer w.freeBuffers();
        w.keys_dev = try ctx.devAlloc(@as(u64, batch) * @sizeOf(types.ChunkKey), "Welt: Chunkliste");
        w.voxels_dev = try ctx.devAlloc(@as(u64, batch) * cap * 16, "Welt: Generatorpuffer");
        w.counts_dev = try ctx.devAlloc(@as(u64, batch) * 4, "Welt: Zähler");
        w.offsets_dev = try ctx.devAlloc(@as(u64, batch + 1) * 4, "Welt: Präfixe");
        try ctx.check(ctx.drv.cuMemHostAlloc(&w.pinned, @as(u64, batch) * (@sizeOf(types.ChunkKey) + 8) + 4, 0), "cuMemHostAlloc");
        w.job_roots = try oom(ctx.gpa.alloc(u32, 3 * @as(usize, batch)));
        if (info.flags & api.world_sync == 0) {
            w.thread = std.Thread.spawn(.{}, workerMain, .{w}) catch return fail(error.OutOfMemory, "Welt: Arbeiter-Thread nicht startbar", .{});
        }
        ctx.logf(3, "Welt: Chunks {d}^3, Ziel {d:.1} px je Voxel, Sichtweite {d:.0}, bis {d} Chunks je Update, Generatorpuffer {d} MiB", .{ n, voxel_px, view_distance, batch, @as(u64, batch) * cap * 16 >> 20 });
        return w;
    }

    fn freeBuffers(self: *World) void {
        if (self.job_roots.len != 0) self.ctx.gpa.free(self.job_roots);
        self.job_roots = &.{};
        for ([_]cuda.CUdeviceptr{ self.keys_dev, self.voxels_dev, self.counts_dev, self.offsets_dev }) |p| {
            if (p != 0) _ = self.ctx.drv.cuMemFree_v2(p);
        }
        if (self.pinned != null) _ = self.ctx.drv.cuMemFreeHost(self.pinned);
    }

    pub fn destroy(self: *World) void {
        if (self.thread) |t| {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            self.quit = true;
            _ = std.c.pthread_cond_broadcast(&self.cond);
            _ = std.c.pthread_mutex_unlock(&self.mutex);
            t.join();
            self.thread = null;
        }
        if (self.job_built) |*b| b.free(self.ctx.gpuExecAux());
        for (self.chunks.items) |c| {
            if (c.instance == 0) continue;
            self.ctx.instanceDestroy(c.instance) catch {};
            self.ctx.geometryDestroy(c.geometry) catch {};
        }
        // Puffer erst freigeben, wenn die GPU mit laufenden Generatoren fertig ist
        _ = self.ctx.drv.cuStreamSynchronize(self.ctx.aux_stream);
        self.freeBuffers();
        self.plan.deinit();
        self.chunks.deinit(self.ctx.gpa);
        self.chunk_free.deinit(self.ctx.gpa);
        self.requests.deinit(self.ctx.gpa);
        self.evicted.deinit(self.ctx.gpa);
        self.ctx.gpa.destroy(self);
    }

    fn chunkSize(self: *const World, lod: u32) f64 {
        return @floatFromInt(@as(u64, 1) << @intCast(self.chunk_log2 + lod));
    }

    fn transformOf(self: *const World, k: Key) [12]f32 {
        const s = self.chunkSize(k.lod);
        const step: f32 = @floatFromInt(@as(u32, 1) << @intCast(k.lod));
        const t = [3]f32{
            @floatCast(@as(f64, @floatFromInt(k.x)) * s - self.origin[0]),
            @floatCast(@as(f64, @floatFromInt(k.y)) * s - self.origin[1]),
            @floatCast(@as(f64, @floatFromInt(k.z)) * s - self.origin[2]),
        };
        return .{ step, 0, 0, t[0], 0, step, 0, t[1], 0, 0, step, t[2] };
    }

    /// Ein Frame: fertige Chunks übernehmen, neue beauftragen, zeigen, verdrängen.
    /// `camera` in Weltkoordinaten (Grundvoxel), `origin` = der Render-Ursprung,
    /// der auch an pyr_commit geht (Instanzen liegen relativ dazu).
    /// Erzeugen und Bauen laufen auf einem Hintergrund-Thread und -Stream; der
    /// Aufrufer wartet nie auf die GPU (außer mit world_sync).
    pub fn update(self: *World, camera: [3]f64, origin: [3]f64, cam: ?*const types.Camera) Error!void {
        const ctx = self.ctx;
        // LOD aus dem Bildschirmmaß: Pixel je Einheit in Entfernung 1
        if (cam) |c| {
            const h: f64 = @floatFromInt(@max(c.height, 1));
            // orthografisch schrumpft nichts mit der Entfernung: überall fein
            self.pixels_per_unit = if (c.projection == types.projection_orthographic)
                1e9
            else
                0.5 * h / @max(c.scale[1], 1e-6);
        }
        // Budget: gleitend gröber werden, wenn Speicher oder Plätze knapp werden
        var pressure: f64 = 0;
        if (self.memory_budget > 0 and self.stats.bytes > 0)
            pressure = @as(f64, @floatFromInt(self.stats.bytes)) / @as(f64, @floatFromInt(self.memory_budget));
        if (self.chunk_limit > 0)
            pressure = @max(pressure, @as(f64, @floatFromInt(self.stats.resident_chunks)) / @as(f64, @floatFromInt(self.chunk_limit)));
        if (pressure > 1) {
            self.lod_scale = @max(self.lod_scale * @max(std.math.pow(f64, 1 / pressure, 0.25), 0.9), 0.02);
        } else if (pressure < 0.8) {
            self.lod_scale = @min(self.lod_scale * 1.02, 1); // wieder feiner werden
        }
        const target = self.voxel_pixels / self.lod_scale;
        self.plan.cfg.refine_k = self.pixels_per_unit / target;
        self.stats.voxel_pixels = @floatCast(target);
        self.stats.top_lod = self.plan.topLod();
        // Voxel der groben Fassung sind in Entfernung t etwa so groß
        self.stats.secondary_bias = if (self.secondary_mask != 0)
            @floatCast(target * self.secondary_factor / self.pixels_per_unit)
        else
            0;
        if (!std.mem.eql(f64, &origin, &self.origin)) {
            self.origin = origin;
            for (self.chunks.items) |c| {
                if (c.instance != 0) try ctx.instanceSetTransform(c.instance, &self.transformOf(c.key));
            }
        }
        try oom(self.plan.update(camera));
        self.stats.built_chunks = 0;
        self.stats.built_voxels = 0;

        if (self.pollJob()) try self.finishJob();
        if (self.job_idle) {
            try oom(self.plan.takeRequests(camera, self.batch_max, &self.requests));
            if (self.requests.items.len > 0) {
                self.submitJob();
                if (self.thread == null) {
                    self.gpuWork();
                    try self.finishJob();
                }
            }
        }
        if (self.stats.built_chunks > 0) try oom(self.plan.collect(camera));
        // gröbere Auswahl für Schatten und GI (Vorfahren, schon im Speicher)
        if (self.secondary_mask != 0) try oom(self.plan.collectCoarse(camera, self.secondary_factor));
        try self.applyVisibility();
        try self.evict();
    }

    /// Wartet, bis der laufende Auftrag fertig ist, und übernimmt ihn
    /// (Ladebildschirm, Teleport). Ohne laufenden Auftrag sofort zurück.
    pub fn wait(self: *World, camera: [3]f64) Error!void {
        if (self.job_idle) return;
        if (self.thread != null) {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            while (self.job_state != .done) _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
            _ = std.c.pthread_mutex_unlock(&self.mutex);
        }
        try self.finishJob();
        try oom(self.plan.collect(camera));
        try self.applyVisibility();
    }

    // -----------------------------------------------------------------------
    // Auftrag: Hauptthread schreibt Schlüssel -> Arbeiter erzeugt und baut
    // -> Hauptthread übernimmt (Pools, GAS ohne Synchronisation, Instanzen)
    // -----------------------------------------------------------------------

    fn pinnedKeys(self: *World) [*]types.ChunkKey {
        return @ptrCast(@alignCast(self.pinned.?));
    }

    fn pinnedCounts(self: *World) [*]u32 {
        const pin: [*]u8 = @ptrCast(self.pinned.?);
        return @ptrCast(@alignCast(pin + @as(usize, self.batch_max) * @sizeOf(types.ChunkKey)));
    }

    fn submitJob(self: *World) void {
        const k = self.requests.items.len;
        for (self.requests.items, self.pinnedKeys()[0..k]) |r, *d| d.* = .{ .x = r.x, .y = r.y, .z = r.z, .lod = r.lod };
        self.job_count = @intCast(k);
        self.job_idle = false;
        if (self.thread != null) {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            self.job_state = .submitted;
            _ = std.c.pthread_cond_broadcast(&self.cond);
            _ = std.c.pthread_mutex_unlock(&self.mutex);
        } else self.job_state = .submitted;
    }

    fn pollJob(self: *World) bool {
        if (self.job_idle) return false;
        if (self.thread == null) return self.job_state == .done;
        _ = std.c.pthread_mutex_lock(&self.mutex);
        defer _ = std.c.pthread_mutex_unlock(&self.mutex);
        return self.job_state == .done;
    }

    fn workerMain(self: *World) void {
        self.ctx.enter() catch {};
        defer self.ctx.leave();
        while (true) {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            while (self.job_state != .submitted and !self.quit) _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
            const quit = self.quit;
            _ = std.c.pthread_mutex_unlock(&self.mutex);
            if (quit) return;
            self.gpuWork();
            _ = std.c.pthread_mutex_lock(&self.mutex);
            self.job_state = .done;
            _ = std.c.pthread_cond_broadcast(&self.cond);
            _ = std.c.pthread_mutex_unlock(&self.mutex);
        }
    }

    /// GPU-Teil eines Auftrags (Arbeiter-Thread oder inline): Erzeugen, Zählen,
    /// Bauen. Berührt keinen Kontextzustand außer dem Hintergrund-Stream.
    fn gpuWork(self: *World) void {
        self.job_error = null;
        self.gpuWorkImpl() catch |e| {
            self.job_error = e;
            const m = std.mem.span(diag.message());
            const n = @min(m.len, self.job_message.len - 1);
            @memcpy(self.job_message[0..n], m[0..n]);
            self.job_message[n] = 0;
        };
    }

    fn gpuWorkImpl(self: *World) Error!void {
        const ctx = self.ctx;
        const k = self.job_count;
        const aux = ctx.aux_stream;
        const counts = self.pinnedCounts();
        const sums = counts + self.batch_max;
        self.job_total = 0;
        self.job_overflow = 0;
        self.job_built = null;
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.keys_dev, self.pinnedKeys(), @as(u64, k) * @sizeOf(types.ChunkKey), aux), "cuMemcpyHtoDAsync");
        try ctx.check(ctx.drv.cuMemsetD8Async(self.counts_dev, 0, @as(u64, k) * 4, aux), "cuMemsetD8Async");

        // 1. Erzeugen
        var gp = types.WorldGenParams{
            .chunks = self.keys_dev,
            .voxels = self.voxels_dev,
            .counts = self.counts_dev,
            .count = k,
            .capacity = self.capacity,
            .chunk_log2 = self.chunk_log2,
            .reserved = 0,
            .user = @intFromPtr(self.user),
        };
        if (self.generate) |gen| {
            gen(self.user, &gp, @ptrCast(aux));
        } else {
            var tp = self.terrain;
            const threads = k << @intCast(2 * self.chunk_log2);
            const params = [_]?*anyopaque{ @ptrCast(&gp), @ptrCast(&tp) };
            const b = types.gen_block;
            try ctx.check(ctx.drv.cuLaunchKernel(ctx.fn_gen_terrain, (threads + b - 1) / b, 1, 1, b, 1, 1, 0, aux, @constCast(&params), null), "cuLaunchKernel(Gelände)");
        }

        // 2. Belegung lesen, dichte Präfixe
        const e = ctx.gpuExecAux();
        try e.read(e.ctx, std.mem.sliceAsBytes(counts[0..k]), self.counts_dev);
        var total: u64 = 0;
        sums[0] = 0;
        for (0..k) |c| {
            var n = counts[c];
            if (n > self.capacity) {
                self.job_overflow += 1;
                n = self.capacity;
            }
            total += n;
            sums[c + 1] = @intCast(total);
        }
        self.job_total = @intCast(total);
        if (total == 0) return;

        // 3. DAG-Bau aller Chunks in einem Zug
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.offsets_dev, sums, (@as(u64, k) + 1) * 4, aux), "cuMemcpyHtoDAsync");
        self.job_built = try gpu_build.buildChunks(e, self.chunk_log2, self.rt_log2, .{
            .count = k,
            .capacity = self.capacity,
            .voxels = self.voxels_dev,
            .offsets = self.offsets_dev,
            .total = @intCast(total),
        }, self.jobOut());
    }

    fn jobOut(self: *World) gpu_build.ChunkOut {
        const k = self.batch_max;
        return .{ .roots = self.job_roots[0..k], .first_voxel = self.job_roots[k .. 2 * k], .first_prim = self.job_roots[2 * k ..] };
    }

    /// Hauptthread: Ergebnis übernehmen
    fn finishJob(self: *World) Error!void {
        const ctx = self.ctx;
        const k = self.job_count;
        self.job_idle = true;
        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.job_state = .idle;
        _ = std.c.pthread_mutex_unlock(&self.mutex);
        if (self.job_error) |err| {
            // Chunks bleiben angefragt und werden später erneut versucht
            for (self.requests.items) |key| {
                if (self.plan.nodes.getPtr(key)) |n| n.state = .requested;
            }
            if (self.job_built) |b| b.free(ctx.gpuExecAux());
            self.job_built = null;
            return fail(err, "Welt: {s}", .{std.mem.sliceTo(&self.job_message, 0)});
        }
        if (self.job_overflow > 0) ctx.logf(2, "Welt: Generatorpuffer bei {d} Chunks übergelaufen (chunk_capacity erhöhen)", .{self.job_overflow});
        self.stats.overflow_chunks += self.job_overflow;
        self.stats.built_voxels = self.job_total;

        const handles = try oom(ctx.gpa.alloc(usize, k));
        defer ctx.gpa.free(handles);
        @memset(handles, 0);
        if (self.job_built) |built| {
            var b = built;
            self.job_built = null;
            defer b.free(ctx.gpuExecAux());
            ctx.installChunkBatch(&b, self.jobOut(), k, handles) catch |e| {
                if (e != error.Capacity) return e;
                // Geometrieplätze erschöpft: Batch verwerfen, Welt wird gröber
                self.lod_scale = @max(self.lod_scale * 0.8, 0.02);
                ctx.logf(2, "Welt: keine Geometrieplätze mehr, LOD wird gröber (max_geometries erhöhen)", .{});
                for (self.requests.items) |key| {
                    if (self.plan.nodes.getPtr(key)) |nd| nd.state = .requested;
                }
                return;
            };
        }

        // Instanzen anlegen (zunächst unsichtbar; applyVisibility schaltet sie ein)
        const sums = self.pinnedCounts() + self.batch_max;
        for (self.requests.items, 0..) |key, c| {
            if (handles[c] == 0) {
                self.plan.finish(key, true, 0);
                continue;
            }
            const inst = ctx.instanceCreate(handles[c]) catch |e| {
                // keine Plätze mehr: Chunk verwerfen, Welt wird gröber
                if (e != error.Capacity) return e;
                ctx.geometryDestroy(handles[c]) catch {};
                self.plan.finish(key, true, 0);
                self.lod_scale = @max(self.lod_scale * 0.9, 0.02);
                continue;
            };
            try ctx.instanceSetTransform(inst, &self.transformOf(key));
            try ctx.instanceSetMask(inst, 0);
            try ctx.instanceKeepHistory(inst);
            const idx: u32 = self.chunk_free.pop() orelse blk: {
                try oom(self.chunks.append(ctx.gpa, undefined));
                break :blk @intCast(self.chunks.items.len - 1);
            };
            self.chunks.items[idx] = .{ .key = key, .geometry = handles[c], .instance = inst, .mask = 0, .stamp = 0, .stamp_coarse = 0, .voxels = sums[c + 1] - sums[c] };
            self.plan.finish(key, false, @as(u64, idx) + 1);
            self.stats.built_chunks += 1;
        }
    }

    fn applyVisibility(self: *World) Error!void {
        const ctx = self.ctx;
        const frame = self.plan.frame;
        for (self.plan.visible.items) |k| {
            const node = self.plan.nodes.get(k) orelse continue;
            if (node.user == 0) continue;
            self.chunks.items[node.user - 1].stamp = frame;
        }
        for (self.plan.visible_coarse.items) |k| {
            const node = self.plan.nodes.get(k) orelse continue;
            if (node.user == 0) continue;
            self.chunks.items[node.user - 1].stamp_coarse = frame;
        }
        var visible: u32 = 0;
        var resident: u32 = 0;
        for (self.chunks.items) |*c| {
            if (c.instance == 0) continue;
            resident += 1;
            var mask: u32 = 0;
            if (c.stamp == frame) {
                mask |= self.mask;
                visible += 1;
            }
            if (c.stamp_coarse == frame) mask |= self.secondary_mask;
            if (mask != c.mask) {
                try ctx.instanceSetMask(c.instance, mask);
                c.mask = mask;
            }
        }
        self.stats.visible_chunks = visible;
        self.stats.resident_chunks = resident;
        var pending: u32 = 0;
        var it = self.plan.nodes.valueIterator();
        while (it.next()) |n| {
            if (n.state == .requested or n.state == .building) pending += 1;
        }
        self.stats.pending_chunks = pending;
        self.stats.bytes = self.ctx.batchBytes();
    }

    fn evict(self: *World) Error!void {
        try oom(self.plan.evict(&self.evicted));
        for (self.evicted.items) |n| {
            if (n.user == 0) continue;
            const idx: u32 = @intCast(n.user - 1);
            const c = &self.chunks.items[idx];
            try self.ctx.instanceDestroy(c.instance);
            try self.ctx.geometryDestroy(c.geometry);
            c.* = .{ .key = c.key, .geometry = 0, .instance = 0, .mask = 0, .stamp = 0, .stamp_coarse = 0, .voxels = 0 };
            try oom(self.chunk_free.append(self.ctx.gpa, idx));
        }
    }
};
