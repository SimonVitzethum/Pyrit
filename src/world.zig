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

/// Ein geänderter Grundvoxel
pub const EditPos = struct { x: i64, y: i64, z: i64 };

/// Zelle einer gröberen Stufe (für die Zählung entfernter Grundvoxel)
const CellKey = struct { lod: u8, x: i64, y: i64, z: i64 };

/// Chunk, in dem ein Grundvoxel auf Stufe `lod` liegt; null, wenn die Position
/// außerhalb des darstellbaren Bereichs liegt.
fn chunkKeyFor(chunk_log2: u32, p: EditPos, lod: u32) ?Key {
    const shift: u6 = @intCast(chunk_log2 + lod);
    const cx = p.x >> shift;
    const cy = p.y >> shift;
    const cz = p.z >> shift;
    if (cx < std.math.minInt(i32) or cx > std.math.maxInt(i32) or
        cy < std.math.minInt(i32) or cy > std.math.maxInt(i32) or
        cz < std.math.minInt(i32) or cz > std.math.maxInt(i32)) return null;
    return .{ .lod = @intCast(lod), .x = @intCast(cx), .y = @intCast(cy), .z = @intCast(cz) };
}

/// Chunk-lokale Zelle auf Stufe `lod`
fn localCell(chunk_log2: u32, p: EditPos, lod: u32) [3]u32 {
    const mask: i64 = (@as(i64, 1) << @intCast(chunk_log2)) - 1;
    const s: u6 = @intCast(lod);
    return .{
        @intCast((p.x >> s) & mask),
        @intCast((p.y >> s) & mask),
        @intCast((p.z >> s) & mask),
    };
}

/// Zelle der Stufe `lod`, in der ein Grundvoxel liegt
fn cellOf(p: EditPos, lod: u32) CellKey {
    const s: u6 = @intCast(lod);
    return .{ .lod = @intCast(lod), .x = p.x >> s, .y = p.y >> s, .z = p.z >> s };
}

/// Wie viele Grundvoxel eine Zelle der Stufe `lod` enthält
fn cellVoxels(lod: u32) u64 {
    return @as(u64, 1) << @intCast(3 * lod);
}

/// Entfernen schlägt auf eine gröbere Stufe erst durch, wenn *alle* darin
/// liegenden Grundvoxel entfernt sind. Ab dieser Stufe sind das mehr als 2^24
/// Voxel – das kommt nicht vor, also wird dort nicht mehr gezählt.
const rm_track_max: u32 = 8;

/// Neu planen erst nach so viel Kamerabewegung (Grundvoxel)
const replan_distance: f64 = 1;

pub const World = struct {
    ctx: *Context,
    plan: world_plan.Plan,
    chunk_log2: u32,
    batch_max: u32,
    capacity: u32,
    mask: u32,
    rt_log2: u32,
    generate: *const fn (user: ?*anyopaque, params: *const types.WorldGenParams, stream: ?*anyopaque) callconv(.c) void,
    user: ?*anyopaque,

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
    /// Auftrag war zu groß für die neue Kapazität: verwerfen und kleiner erneut
    job_retry: bool = false,

    // Änderungen an der Welt (Setzen und Entfernen einzelner Grundvoxel).
    // Die Überlagerung ist die Wahrheit: sie wird bei jeder Erzeugung eines
    // Chunks erneut angewandt, überlebt also Verdrängung und LOD-Wechsel.
    edits: std.AutoHashMapUnmanaged(EditPos, u32) = .empty,
    /// je Chunk (über alle Stufen) die darin liegenden Änderungen
    edit_chunks: std.AutoHashMapUnmanaged(Key, std.ArrayListUnmanaged(EditPos)) = .empty,
    /// entfernte Grundvoxel je gröberer Zelle
    rm_counts: std.AutoHashMapUnmanaged(CellKey, u32) = .empty,
    /// Chunks, die wegen einer Änderung neu gebaut werden müssen
    dirty: std.ArrayList(Key) = .empty,
    dirty_set: std.AutoHashMapUnmanaged(Key, void) = .empty,
    /// vom Hauptthread für den laufenden Auftrag vorbereitet (der Arbeiter
    /// fasst die Überlagerung nicht an: sie gehört dem Hauptthread)
    job_edits: std.ArrayList([4]u32) = .empty,
    job_edit_off: std.ArrayList(u32) = .empty,
    /// Gerätepuffer der Änderungen, wachsen nach Bedarf
    edits_dev: cuda.CUdeviceptr = 0,
    edit_off_dev: cuda.CUdeviceptr = 0,
    edit_used_dev: cuda.CUdeviceptr = 0,
    edits_dev_cap: u32 = 0,
    /// zweiter Voxelpuffer: Ziel der Verdichtung nach dem Anwenden
    voxels2_dev: cuda.CUdeviceptr = 0,
    counts2_dev: cuda.CUdeviceptr = 0,
    /// Chunks je Auftrag; sinkt, wenn die Kapazität je Chunk wachsen muss,
    /// damit der Generatorpuffer nie neu (und größer) angelegt werden muss
    job_limit: u32 = 0,
    /// Gesamtplätze im Generatorpuffer (batch_max · Anfangskapazität)
    voxel_slots: u64 = 0,
    /// letzter Zustand, um unveränderte Frames zu erkennen
    last_cam: [3]f64 = .{ std.math.nan(f64), 0, 0 },
    /// Seit dem letzten Planerlauf ist ein Auftrag fertig geworden. Dann muss
    /// der Planer noch einmal laufen: erst mit den neuen Chunks kennt er die
    /// nächstfeinere Stufe darunter.
    replan: bool = true,
    last_refine_k: f64 = 0,
    /// Kameraposition beim letzten Planerlauf (siehe replan_distance)
    plan_cam: [3]f64 = .{ std.math.nan(f64), 0, 0 },
    plan_refine_k: f64 = 0,
    quiet_frames: u64 = 0,
    /// Zeiten im Auftrag (nur mit PYRIT_WORLD_PROFILE)
    job_gen_ms: f64 = 0,
    job_read_ms: f64 = 0,
    job_pre_ms: f64 = 0,
    job_build_ms: f64 = 0,
    job_runs: u64 = 0,
    /// in diesem Frame neu fertig gewordene Chunks (ohne Ersetzungen)
    built_new: u32 = 0,
    /// Zeitmessung der Update-Phasen (PYRIT_WORLD_PROFILE=1)
    prof: ?*[6]f64 = null,
    prof_frames: u64 = 0,

    pub fn create(ctx: *Context, info: *const api.WorldInfo) Error!*World {
        const cl: u32 = if (info.chunk_log2 == 0) 5 else info.chunk_log2;
        if (cl < 3 or cl > 8) return fail(error.InvalidArgument, "chunk_log2 muss in [3, 8] liegen", .{});
        const max_lod: u32 = if (info.max_lod == 0) 20 else info.max_lod;
        if (max_lod > 20) return fail(error.InvalidArgument, "max_lod höchstens 20", .{});
        const voxel_px: f64 = if (info.voxel_pixels > 0) info.voxel_pixels else 4;
        const view_distance: f64 = if (info.view_distance > 0) info.view_distance else 16384;
        const generate = info.generate orelse return fail(error.InvalidArgument, "generate fehlt: Pyrit bringt kein Gelände mit, die Anwendung liefert den Generator", .{});
        const y_min = info.y_min;
        const y_max = info.y_max;
        if (y_max <= y_min) return fail(error.InvalidArgument, "y_max muss größer als y_min sein", .{});
        const n: u32 = @as(u32, 1) << @intCast(cl);
        // Wie viele Chunks ein Auftrag umfasst, bestimmt, wie schnell eine
        // frisch betretene Welt ihre volle Schärfe erreicht: je Update läuft
        // genau ein Auftrag. Gemessen bis zur vollen Schärfe: 64 -> 112
        // Updates, 256 -> 29, 1024 -> 17. Die Vorgabe richtet sich deshalb
        // nicht nach einer festen Zahl, sondern danach, wie viele Chunks in
        // den Generatorpuffer passen (128 MiB, also bei der Standardkapazität
        // von 8192 Voxeln rund 1024 Chunks auf einmal).
        const gen_budget: u64 = 128 << 20;
        const cap_pre: u32 = if (info.chunk_capacity == 0) @min(n * n * 8, n * n * n) else @min(info.chunk_capacity, n * n * n);
        const auto_batch: u32 = @intCast(std.math.clamp(gen_budget / (@as(u64, cap_pre) * 16), 16, 4096));
        const batch: u32 = if (info.chunks_per_update == 0) auto_batch else info.chunks_per_update;
        if (batch > 4096) return fail(error.InvalidArgument, "chunks_per_update höchstens 4096", .{});
        const cap: u32 = cap_pre;
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
            .generate = generate,
            .user = info.user,
            .voxel_pixels = voxel_px,
            .secondary_mask = info.secondary_mask,
            .secondary_factor = if (info.secondary_pixels > 0) @as(f64, info.secondary_pixels) / voxel_px else 4,
            .memory_budget = if (info.memory_budget == 0) 256 << 20 else info.memory_budget,
        };
        // Platzgrenze: die Welt bleibt unter den Geometrieplätzen des Kontexts
        w.chunk_limit = @intCast(ctx.geometries.len * 3 / 4);
        errdefer w.freeBuffers();
        w.job_limit = batch;
        w.voxel_slots = @as(u64, batch) * cap;
        w.keys_dev = try ctx.devAlloc(@as(u64, batch) * @sizeOf(types.ChunkKey), "Welt: Chunkliste");
        w.voxels_dev = try ctx.devAlloc(w.voxel_slots * 16, "Welt: Generatorpuffer");
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
        for ([_]cuda.CUdeviceptr{
            self.keys_dev,      self.voxels_dev,   self.counts_dev,    self.offsets_dev,
            self.voxels2_dev,   self.counts2_dev,  self.edits_dev,     self.edit_off_dev,
            self.edit_used_dev,
        }) |p| {
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
        if (self.prof) |p| self.ctx.gpa.destroy(p);
        self.plan.deinit();
        self.chunks.deinit(self.ctx.gpa);
        self.chunk_free.deinit(self.ctx.gpa);
        self.requests.deinit(self.ctx.gpa);
        self.evicted.deinit(self.ctx.gpa);
        self.edits.deinit(self.ctx.gpa);
        var it = self.edit_chunks.valueIterator();
        while (it.next()) |l| l.deinit(self.ctx.gpa);
        self.edit_chunks.deinit(self.ctx.gpa);
        self.rm_counts.deinit(self.ctx.gpa);
        self.dirty.deinit(self.ctx.gpa);
        self.dirty_set.deinit(self.ctx.gpa);
        self.job_edits.deinit(self.ctx.gpa);
        self.job_edit_off.deinit(self.ctx.gpa);
        self.ctx.gpa.destroy(self);
    }

    // -----------------------------------------------------------------------
    // Änderungen an der Welt
    //
    // Die Überlagerung in Grundvoxel-Koordinaten ist die Wahrheit. Ein Chunk
    // bekommt seine Änderungen bei *jeder* Erzeugung aufgeprägt, also überleben
    // sie Verdrängung, LOD-Wechsel und Neustart (mit save/load).
    //
    // Auf gröberen Stufen: Hinzufügen setzt die Zelle immer (eine Zelle gilt
    // als gefüllt, sobald irgendetwas darin liegt). Entfernen wirkt erst, wenn
    // *alle* Grundvoxel der Zelle entfernt sind – sonst würde ein einzelnes
    // abgebautes Voxel in der Ferne ein ganzes Loch reißen.
    // -----------------------------------------------------------------------

    fn chunkKeyOf(self: *const World, p: EditPos, lod: u32) Error!Key {
        return chunkKeyFor(self.chunk_log2, p, lod) orelse
            fail(error.InvalidArgument, "Position liegt außerhalb der Welt", .{});
    }

    fn markDirty(self: *World, key: Key) Error!void {
        if (self.plan.nodes.get(key) == null) return; // noch nicht gebaut
        const gop = try oom(self.dirty_set.getOrPut(self.ctx.gpa, key));
        if (gop.found_existing) return;
        try oom(self.dirty.append(self.ctx.gpa, key));
    }

    /// Grundvoxel setzen (`attribute` != 0) oder entfernen (`attribute` == 0)
    pub fn edit(self: *World, list: []const api.WorldEdit) Error!void {
        for (list) |ed| {
            const pos = EditPos{ .x = ed.x, .y = ed.y, .z = ed.z };
            const gop = try oom(self.edits.getOrPut(self.ctx.gpa, pos));
            const prev: ?u32 = if (gop.found_existing) gop.value_ptr.* else null;
            if (prev) |q| if (q == ed.attribute) continue;
            gop.value_ptr.* = ed.attribute;

            // Entfernte Grundvoxel je grober Zelle mitzählen
            const was_removed = if (prev) |q| q == 0 else false;
            const now_removed = ed.attribute == 0;
            if (was_removed != now_removed) {
                var lod: u32 = 1;
                while (lod <= rm_track_max) : (lod += 1) {
                    const cell = cellOf(pos, lod);
                    const c = try oom(self.rm_counts.getOrPut(self.ctx.gpa, cell));
                    if (!c.found_existing) c.value_ptr.* = 0;
                    if (now_removed) c.value_ptr.* += 1 else c.value_ptr.* -|= 1;
                }
            }

            var lod: u32 = 0;
            while (lod <= self.plan.cfg.max_lod) : (lod += 1) {
                const key = try self.chunkKeyOf(pos, lod);
                if (!gop.found_existing) {
                    const l = try oom(self.edit_chunks.getOrPut(self.ctx.gpa, key));
                    if (!l.found_existing) l.value_ptr.* = .empty;
                    try oom(l.value_ptr.append(self.ctx.gpa, pos));
                }
                // Eine Entfernung, die auf dieser Stufe gar nicht wirkt (die
                // grobe Zelle ist noch nicht ganz leer), braucht auch keinen
                // Neubau. Hinzufügen wirkt dagegen auf jeder Stufe.
                if (now_removed and lod > 0) {
                    if (lod > rm_track_max) continue;
                    if ((self.rm_counts.get(cellOf(pos, lod)) orelse 0) < cellVoxels(lod)) continue;
                }
                try self.markDirty(key);
            }
        }
    }

    // Speicherformat der Änderungen: Kopf "PYRE", Version, Anzahl, dann je
    // Eintrag x, y, z (i64) und Attribut (u32) mit Füllwort. Das Gelände selbst
    // steht nicht darin – der Generator liefert es jederzeit wieder.
    const edits_magic: u32 = 0x45525950; // "PYRE"
    const edits_version: u32 = 1;
    const edits_header: usize = 16;
    const edits_entry: usize = 32;

    pub fn editsBytes(self: *const World) u64 {
        return edits_header + @as(u64, self.edits.count()) * edits_entry;
    }

    pub fn editsSave(self: *const World, dst: []u8) Error!void {
        if (dst.len < self.editsBytes()) return fail(error.Capacity, "Puffer zu klein: {d} Bytes nötig", .{self.editsBytes()});
        std.mem.writeInt(u32, dst[0..4], edits_magic, .little);
        std.mem.writeInt(u32, dst[4..8], edits_version, .little);
        std.mem.writeInt(u64, dst[8..16], self.edits.count(), .little);
        var o: usize = edits_header;
        var it = self.edits.iterator();
        while (it.next()) |kv| {
            std.mem.writeInt(i64, dst[o..][0..8], kv.key_ptr.x, .little);
            std.mem.writeInt(i64, dst[o + 8 ..][0..8], kv.key_ptr.y, .little);
            std.mem.writeInt(i64, dst[o + 16 ..][0..8], kv.key_ptr.z, .little);
            std.mem.writeInt(u32, dst[o + 24 ..][0..4], kv.value_ptr.*, .little);
            std.mem.writeInt(u32, dst[o + 28 ..][0..4], 0, .little);
            o += edits_entry;
        }
    }

    pub fn editsLoad(self: *World, src: []const u8) Error!void {
        if (src.len < edits_header) return fail(error.InvalidArgument, "Änderungsdaten zu kurz", .{});
        if (std.mem.readInt(u32, src[0..4], .little) != edits_magic)
            return fail(error.InvalidArgument, "keine Pyrit-Änderungsdaten", .{});
        const version = std.mem.readInt(u32, src[4..8], .little);
        if (version != edits_version)
            return fail(error.InvalidArgument, "Änderungsdaten Version {d}, erwartet {d}", .{ version, edits_version });
        const n = std.mem.readInt(u64, src[8..16], .little);
        if (src.len < edits_header + n * edits_entry)
            return fail(error.InvalidArgument, "Änderungsdaten unvollständig ({d} Einträge angekündigt)", .{n});
        var o: usize = edits_header;
        var i: u64 = 0;
        while (i < n) : (i += 1) {
            const one = [1]api.WorldEdit{.{
                .x = std.mem.readInt(i64, src[o..][0..8], .little),
                .y = std.mem.readInt(i64, src[o + 8 ..][0..8], .little),
                .z = std.mem.readInt(i64, src[o + 16 ..][0..8], .little),
                .attribute = std.mem.readInt(u32, src[o + 24 ..][0..4], .little),
            }};
            try self.edit(&one);
            o += edits_entry;
        }
    }

    /// Änderungen des Auftrags in Chunk-lokale Einträge umrechnen
    fn prepareEdits(self: *World) Error!void {
        self.job_edits.clearRetainingCapacity();
        self.job_edit_off.clearRetainingCapacity();
        try oom(self.job_edit_off.append(self.ctx.gpa, 0));
        for (self.requests.items) |key| {
            if (self.edit_chunks.get(key)) |list| {
                for (list.items) |pos| {
                    const attr = self.edits.get(pos) orelse continue;
                    if (attr == 0 and key.lod > 0) {
                        // Entfernen wirkt grob erst, wenn die Zelle ganz leer ist
                        if (key.lod > rm_track_max) continue;
                        const need = cellVoxels(key.lod);
                        const cell = cellOf(pos, key.lod);
                        if ((self.rm_counts.get(cell) orelse 0) < need) continue;
                    }
                    const cell = localCell(self.chunk_log2, pos, key.lod);
                    try oom(self.job_edits.append(self.ctx.gpa, .{ cell[0], cell[1], cell[2], attr }));
                }
            }
            try oom(self.job_edit_off.append(self.ctx.gpa, @intCast(self.job_edits.items.len)));
        }
    }

    /// Geänderte Chunks zum Neubau vormerken (vor den neuen Chunks)
    fn takeDirty(self: *World, max: u32) Error!void {
        while (self.requests.items.len < max) {
            const key = self.dirty.pop() orelse break;
            _ = self.dirty_set.remove(key);
            if (self.plan.nodes.get(key) == null) continue; // inzwischen verdrängt
            try oom(self.requests.append(self.ctx.gpa, key));
        }
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
    fn tick() i64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(i64, ts.sec) * 1_000_000_000 + ts.nsec;
    }

    fn lap(self: *World, slot: usize, t0: *i64) void {
        const p = self.prof orelse return;
        const now = tick();
        p[slot] += @as(f64, @floatFromInt(now - t0.*)) / 1e6;
        t0.* = now;
    }

    pub fn update(self: *World, camera: [3]f64, origin: [3]f64, cam: ?*const types.Camera) Error!void {
        const ctx = self.ctx;
        if (self.prof == null and std.c.getenv("PYRIT_WORLD_PROFILE") != null) {
            self.prof = ctx.gpa.create([6]f64) catch null;
            if (self.prof) |p| p.* = .{ 0, 0, 0, 0, 0, 0 };
        }
        var t0: i64 = if (self.prof != null) tick() else 0;
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
        // Ruhepfad: hat sich weder Kamera noch Ziel-LOD geändert und ist nichts
        // in Arbeit, dann ist auch die Auswahl dieselbe. Dann darf der Frame
        // gar nicht erst weiterzählen, sonst altern die Knoten und die
        // Verdrängung würde eine ruhende Welt abräumen.
        // Das macht Einzeländerungen von der CPU praktisch kostenlos: sie
        // setzen `dirty`, und nur dann läuft der volle Durchlauf.
        const same_view = std.mem.eql(f64, &camera, &self.last_cam) and
            std.mem.eql(f64, &origin, &self.origin) and
            self.plan.cfg.refine_k == self.last_refine_k;
        const quiet = same_view and
            !self.replan and
            self.job_idle and
            self.dirty.items.len == 0 and
            self.stats.pending_chunks == 0;
        if (quiet) {
            self.quiet_frames += 1;
            self.stats.built_chunks = 0;
            self.stats.built_voxels = 0;
            return;
        }
        self.last_cam = camera;
        self.last_refine_k = self.plan.cfg.refine_k;

        if (!std.mem.eql(f64, &origin, &self.origin)) {
            self.origin = origin;
            for (self.chunks.items) |c| {
                if (c.instance != 0) try ctx.instanceSetTransform(c.instance, &self.transformOf(c.key));
            }
        }
        self.lap(0, &t0); // Budget, Instanz-Transformationen
        // Der Planerlauf bleibt: er stempelt auch die *inneren* Knoten des
        // Octrees als gebraucht. Ohne ihn verdrängt die Welt die Vorfahren der
        // sichtbaren Chunks und baut deren Kinder dauernd neu (gemessen: 1250
        // Chunks blieben dann dauerhaft offen). Übersprungen wird nur der
        // vollständig ruhende Frame weiter oben.
        // Der Planer läuft nur, wenn sich die Kamera merklich bewegt hat.
        // Seine Auswahl kippt erst, wenn ein Chunk eine Schwelle überquert;
        // die liegen alle mindestens refine_k weit weg, ein Block Weg
        // verschiebt sie unsichtbar wenig. Vorher lief er bei jeder Bewegung
        // über alle Knoten – im Flug 11 ms je Frame auf dem Hauptthread
        // (18 600 Chunks), in denen die GPU wartete.
        var moved: f64 = 0;
        for (0..3) |a| moved = @max(moved, @abs(camera[a] - self.plan_cam[a]));
        const planned = self.replan or !(moved < replan_distance) or self.plan.cfg.refine_k != self.plan_refine_k;
        if (planned) {
            try oom(self.plan.update(camera));
            self.plan_cam = camera;
            self.plan_refine_k = self.plan.cfg.refine_k;
        }
        self.replan = false;
        self.lap(1, &t0); // Planer
        self.stats.built_chunks = 0;
        self.stats.built_voxels = 0;
        self.built_new = 0;

        if (self.pollJob()) try self.finishJob();
        self.lap(2, &t0); // Ergebnis übernehmen (Pools, GAS, Instanzen)
        if (self.job_idle) {
            // Geänderte Chunks zuerst, aber nur bis zur Hälfte des Auftrags:
            // sonst hungert ein Strom von Änderungen das Nachladen aus (die
            // Welt wurde dann nie fertig geladen – gemessen: 384 Chunks blieben
            // dauerhaft offen).
            const limit = @min(self.batch_max, self.job_limit);
            self.requests.clearRetainingCapacity();
            try self.takeDirty(@max(limit / 2, 1));
            if (self.requests.items.len < limit)
                try oom(self.plan.takeRequests(camera, limit - self.requests.items.len, &self.requests));
            if (self.requests.items.len > 0) {
                try self.submitJob();
                if (self.thread == null) {
                    self.gpuWork();
                    try self.finishJob();
                }
            }
        }
        self.lap(3, &t0); // Auftrag stellen
        // Ein ersetzter Chunk (Änderung) verändert die Auswahl nicht – nur
        // neu fertig gewordene tun das.
        if (self.built_new > 0) try oom(self.plan.collect(self.plan_cam));
        // Auswahl und Sichtbarkeit ändern sich nur mit dem Plan oder neuen Chunks
        if (planned or self.built_new > 0) {
            // gröbere Auswahl für Schatten und GI (Vorfahren, schon im Speicher)
            if (self.secondary_mask != 0) try oom(self.plan.collectCoarse(self.plan_cam, self.secondary_factor));
            try self.applyVisibility();
        }
        self.lap(4, &t0); // Auswahl und Sichtbarkeit
        try self.evict();
        self.lap(5, &t0); // Verdrängung
        if (self.prof) |p| {
            self.prof_frames += 1;
            if (self.prof_frames % 30 == 0) {
                const n: f64 = @floatFromInt(self.prof_frames);
                ctx.logf(1, "Welt je Update: Budget {d:.2} ms, Planer {d:.2}, Übernahme {d:.2}, Auftrag {d:.2}, Sichtbarkeit {d:.2}, Verdrängung {d:.2} ms", .{ p[0] / n, p[1] / n, p[2] / n, p[3] / n, p[4] / n, p[5] / n });
                if (self.job_runs > 0) {
                    const jr: f64 = @floatFromInt(self.job_runs);
                    ctx.logf(1, "Auftrag auf dem Arbeiter ({d} Laeufe): Erzeugen {d:.2} ms, Zaehler zurueckl. {d:.2}, Vorbereiten {d:.2}, Bauen {d:.2}", .{ self.job_runs, self.job_gen_ms / jr, self.job_read_ms / jr, self.job_pre_ms / jr, self.job_build_ms / jr });
                }
            }
        }
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

    fn submitJob(self: *World) Error!void {
        // Die Überlagerung gehört dem Hauptthread: hier umrechnen, der
        // Arbeiter lädt nur noch hoch.
        try self.prepareEdits();
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

    /// Kapazität je Chunk erhöhen, ohne den Generatorpuffer zu vergrößern:
    /// dafür passen weniger Chunks in einen Auftrag. Reicht der Puffer nicht
    /// einmal für einen Chunk, wird er (einmalig) vergrößert.
    /// Rückgabe: true, wenn der laufende Auftrag mit `k` Chunks weitermachen kann.
    /// Freier Grafikspeicher in Bytes (0, wenn nicht ermittelbar)
    fn vramFree(self: *World) u64 {
        var free_b: usize = 0;
        var total_b: usize = 0;
        if (self.ctx.drv.cuMemGetInfo_v2(&free_b, &total_b) != cuda.CUDA_SUCCESS) return 0;
        // Ein Polster bleibt frei: der Renderer braucht selbst noch Speicher.
        const margin: u64 = @max(@as(u64, total_b) / 8, 128 << 20);
        return if (free_b > margin) @as(u64, free_b) - margin else 0;
    }

    /// Generator- (und ggf. Änderungs-)Puffer auf `slots` Voxelplätze bringen.
    /// Erst anlegen, dann freigeben: schlägt es fehl, bleibt alles wie es war.
    fn resizeVoxelBuffers(self: *World, slots: u64) bool {
        const drv = &self.ctx.drv;
        const two = self.voxels2_dev != 0;
        var a: cuda.CUdeviceptr = 0;
        var b: cuda.CUdeviceptr = 0;
        if (drv.cuMemAlloc_v2(&a, slots * 16) != cuda.CUDA_SUCCESS) return false;
        if (two and drv.cuMemAlloc_v2(&b, slots * 16) != cuda.CUDA_SUCCESS) {
            _ = drv.cuMemFree_v2(a);
            return false;
        }
        _ = drv.cuMemFree_v2(self.voxels_dev);
        self.voxels_dev = a;
        if (two) {
            _ = drv.cuMemFree_v2(self.voxels2_dev);
            self.voxels2_dev = b;
        }
        self.voxel_slots = slots;
        return true;
    }

    /// Kapazität je Chunk erhöhen. Solange Grafikspeicher da ist, wächst der
    /// Puffer mit, damit weiterhin viele Chunks je Auftrag durchlaufen –
    /// erst wenn der Speicher knapp wird, sinkt die Zahl der Chunks.
    /// Abgebrochen wird nie: im schlechtesten Fall läuft ein Chunk je Auftrag.
    /// Rückgabe: true, wenn der laufende Auftrag mit `k` Chunks weitermachen kann.
    fn growCapacity(self: *World, need: u32, k: u32) Error!bool {
        const n: u32 = @as(u32, 1) << @intCast(self.chunk_log2);
        const max_cap: u64 = @as(u64, n) * n * n;
        var new_cap: u64 = @max(@as(u64, self.capacity) * 2, need);
        new_cap = @min(new_cap, max_cap);
        if (new_cap <= self.capacity) return true; // schon am Maximum: unmöglich
        self.capacity = @intCast(new_cap);

        const buffers: u64 = if (self.voxels2_dev != 0) 2 else 1;
        const want = @as(u64, self.batch_max) * new_cap; // volle Chunkzahl behalten
        var grew = false;
        if (want > self.voxel_slots and self.vramFree() > want * 16 * buffers) {
            grew = self.resizeVoxelBuffers(want);
        }
        var fit: u64 = self.voxel_slots / new_cap;
        if (fit == 0) {
            // Ein Chunk muss hineinpassen, auch wenn es eng wird.
            if (!self.resizeVoxelBuffers(new_cap))
                return fail(error.OutOfMemory, "Welt: kein Grafikspeicher für Chunks mit {d} Voxeln", .{new_cap});
            fit = 1;
        }
        self.job_limit = @intCast(@max(@min(fit, self.batch_max), 1));
        self.ctx.logf(3, "Welt: Kapazität je Chunk auf {d} erhöht, {d} Chunks je Auftrag, Puffer {d} MiB{s}", .{
            self.capacity,
            self.job_limit,
            self.voxel_slots * 16 * buffers >> 20,
            if (grew) " (vergrößert)" else "",
        });
        return k <= self.job_limit;
    }

    fn ensureEditBuffers(self: *World, entries: u32) Error!void {
        if (self.voxels2_dev == 0) {
            self.voxels2_dev = try self.ctx.devAlloc(self.voxel_slots * 16, "Welt: Änderungspuffer");
            self.counts2_dev = try self.ctx.devAlloc(@as(u64, self.batch_max) * 4, "Welt: Änderungszähler");
        }
        if (self.edit_off_dev == 0)
            self.edit_off_dev = try self.ctx.devAlloc(@as(u64, self.batch_max + 1) * 4, "Welt: Änderungsgrenzen");
        if (entries > self.edits_dev_cap) {
            const cap = @max(entries, @max(self.edits_dev_cap * 2, 256));
            if (self.edits_dev != 0) _ = self.ctx.drv.cuMemFree_v2(self.edits_dev);
            if (self.edit_used_dev != 0) _ = self.ctx.drv.cuMemFree_v2(self.edit_used_dev);
            self.edits_dev = 0;
            self.edit_used_dev = 0;
            self.edits_dev = try self.ctx.devAlloc(@as(u64, cap) * 16, "Welt: Änderungen");
            self.edit_used_dev = try self.ctx.devAlloc(@as(u64, cap) * 4, "Welt: Änderungen (verbraucht)");
            self.edits_dev_cap = cap;
        }
    }

    /// Änderungen des Auftrags anwenden; liefert den Voxelpuffer für den Bau.
    /// Die Zähler des Ziels entstehen dabei neu (`counts` wird überschrieben).
    fn applyEdits(self: *World, k: u32, counts: [*]u32) Error!cuda.CUdeviceptr {
        const ctx = self.ctx;
        const aux = ctx.aux_stream;
        const entries: u32 = @intCast(self.job_edits.items.len);
        try self.ensureEditBuffers(entries);
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.edits_dev, self.job_edits.items.ptr, @as(u64, entries) * 16, aux), "cuMemcpyHtoDAsync");
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.edit_off_dev, self.job_edit_off.items.ptr, (@as(u64, k) + 1) * 4, aux), "cuMemcpyHtoDAsync");
        try ctx.check(ctx.drv.cuMemsetD8Async(self.edit_used_dev, 0, @as(u64, entries) * 4, aux), "cuMemsetD8Async");
        try ctx.check(ctx.drv.cuMemsetD8Async(self.counts2_dev, 0, @as(u64, k) * 4, aux), "cuMemsetD8Async");

        var p = types.WorldEditParams{
            .voxels = self.voxels_dev,
            .counts = self.counts_dev,
            .out_voxels = self.voxels2_dev,
            .out_counts = self.counts2_dev,
            .edits = self.edits_dev,
            .edit_offsets = self.edit_off_dev,
            .edit_used = self.edit_used_dev,
            .count = k,
            .capacity = self.capacity,
        };
        const params = [_]?*anyopaque{@ptrCast(&p)};
        const b = types.edit_block;
        const cells = @as(u64, k) * self.capacity;
        const grid_cells: u32 = @intCast((cells + b - 1) / b);
        const grid_edits: u32 = @intCast((@as(u64, entries) + b - 1) / b);
        try ctx.check(ctx.drv.cuLaunchKernel(ctx.fn_edit_apply, grid_cells, 1, 1, b, 1, 1, 0, aux, @constCast(&params), null), "cuLaunchKernel(Änderungen)");
        try ctx.check(ctx.drv.cuLaunchKernel(ctx.fn_edit_compact, grid_cells, 1, 1, b, 1, 1, 0, aux, @constCast(&params), null), "cuLaunchKernel(Änderungen: verdichten)");
        if (grid_edits > 0)
            try ctx.check(ctx.drv.cuLaunchKernel(ctx.fn_edit_append, grid_edits, 1, 1, b, 1, 1, 0, aux, @constCast(&params), null), "cuLaunchKernel(Änderungen: anhängen)");
        const e = ctx.gpuExecAux();
        try e.read(e.ctx, std.mem.sliceAsBytes(counts[0..k]), self.counts2_dev);
        return self.voxels2_dev;
    }

    fn gpuWorkImpl(self: *World) Error!void {
        const ctx = self.ctx;
        const prof = self.prof != null;
        var t_phase: i64 = if (prof) tick() else 0;
        const k = self.job_count;
        const aux = ctx.aux_stream;
        const counts = self.pinnedCounts();
        const sums = counts + self.batch_max;
        self.job_total = 0;
        self.job_overflow = 0;
        self.job_retry = false;
        self.job_built = null;
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.keys_dev, self.pinnedKeys(), @as(u64, k) * @sizeOf(types.ChunkKey), aux), "cuMemcpyHtoDAsync");

        const e = ctx.gpuExecAux();
        var voxels = self.voxels_dev;
        while (true) {
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
            self.generate(self.user, &gp, @ptrCast(aux));

            // 2. Belegung lesen. Passt ein Chunk nicht in die Kapazität, wird
            //    sie erhöht und der Auftrag wiederholt – abgeschnitten wird nie.
            if (prof) {
                self.job_gen_ms += @as(f64, @floatFromInt(tick() - t_phase)) / 1e6;
                t_phase = tick();
            }
            try e.read(e.ctx, std.mem.sliceAsBytes(counts[0..k]), self.counts_dev);
            if (prof) {
                self.job_read_ms += @as(f64, @floatFromInt(tick() - t_phase)) / 1e6;
                t_phase = tick();
            }
            var need: u32 = 0;
            for (0..k) |c| need = @max(need, counts[c]);

            // 3. Änderungen anwenden (verdichtet in den zweiten Puffer)
            if (need <= self.capacity and self.job_edits.items.len > 0) {
                voxels = try self.applyEdits(k, counts);
                for (0..k) |c| need = @max(need, counts[c]);
            } else voxels = self.voxels_dev;

            if (need <= self.capacity) break;
            self.job_overflow += 1;
            if (!try self.growCapacity(need, k)) {
                // Auftrag passt nicht mehr: verwerfen, der Planer fragt die
                // Chunks im nächsten Update in kleineren Häppchen erneut an.
                self.job_retry = true;
                return;
            }
        }

        var total: u64 = 0;
        sums[0] = 0;
        for (0..k) |c| {
            total += counts[c];
            sums[c + 1] = @intCast(total);
        }
        self.job_total = @intCast(total);
        if (total == 0) return;

        // 4. DAG-Bau aller Chunks in einem Zug
        try ctx.check(ctx.drv.cuMemcpyHtoDAsync_v2(self.offsets_dev, sums, (@as(u64, k) + 1) * 4, aux), "cuMemcpyHtoDAsync");
        if (prof) {
            self.job_pre_ms += @as(f64, @floatFromInt(tick() - t_phase)) / 1e6;
            t_phase = tick();
        }
        self.job_built = try gpu_build.buildChunks(e, self.chunk_log2, self.rt_log2, .{
            .count = k,
            .capacity = self.capacity,
            .voxels = voxels,
            .offsets = self.offsets_dev,
            .total = @intCast(total),
        }, self.jobOut());
        if (prof) {
            self.job_build_ms += @as(f64, @floatFromInt(tick() - t_phase)) / 1e6;
            self.job_runs += 1;
        }
    }

    fn jobOut(self: *World) gpu_build.ChunkOut {
        const k = self.batch_max;
        return .{ .roots = self.job_roots[0..k], .first_voxel = self.job_roots[k .. 2 * k], .first_prim = self.job_roots[2 * k ..] };
    }

    /// Hauptthread: Ergebnis übernehmen
    fn finishJob(self: *World) Error!void {
        self.replan = true;
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
        if (self.job_retry) {
            // Kapazität wurde erhöht; dieselben Chunks kommen kleiner zurück.
            for (self.requests.items) |key| {
                if (self.plan.nodes.getPtr(key)) |n| n.state = .requested;
            }
            self.stats.overflow_chunks += self.job_overflow;
            return;
        }
        if (self.job_overflow > 0) ctx.logf(3, "Welt: Kapazität je Chunk {d} Mal erhöht (kein Voxel ging verloren)", .{self.job_overflow});
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
            // Neubau nach einer Änderung: den alten Chunk erst jetzt freigeben,
            // damit nie ein Loch entsteht.
            var replaced = false;
            if (self.plan.nodes.get(key)) |nd| {
                if (nd.user != 0) {
                    replaced = true;
                    const old: u32 = @intCast(nd.user - 1);
                    const oc = &self.chunks.items[old];
                    if (oc.instance != 0) {
                        try ctx.instanceDestroy(oc.instance);
                        try ctx.geometryDestroy(oc.geometry);
                    }
                    oc.* = .{ .key = oc.key, .geometry = 0, .instance = 0, .mask = 0, .stamp = 0, .stamp_coarse = 0, .voxels = 0 };
                    try oom(self.chunk_free.append(ctx.gpa, old));
                }
            }
            const idx: u32 = self.chunk_free.pop() orelse blk: {
                try oom(self.chunks.append(ctx.gpa, undefined));
                break :blk @intCast(self.chunks.items.len - 1);
            };
            self.chunks.items[idx] = .{ .key = key, .geometry = handles[c], .instance = inst, .mask = 0, .stamp = 0, .stamp_coarse = 0, .voxels = sums[c + 1] - sums[c] };
            self.plan.finish(key, false, @as(u64, idx) + 1);
            self.stats.built_chunks += 1;
            if (!replaced) self.built_new += 1;
        }
    }

    fn applyVisibility(self: *World) Error!void {
        const ctx = self.ctx;
        const frame = self.plan.frame;
        // Ein Nachschlagen, zwei Stempel: der Knoten gilt als gebraucht (sonst
        // verdrängt) und der Chunk als sichtbar. Damit braucht die Verdrängung
        // keinen eigenen Planerlauf mehr.
        for (self.plan.visible.items) |k| {
            const node = self.plan.nodes.getPtr(k) orelse continue;
            node.last_used = frame;
            if (node.user == 0) continue;
            self.chunks.items[node.user - 1].stamp = frame;
        }
        for (self.plan.visible_coarse.items) |k| {
            const node = self.plan.nodes.getPtr(k) orelse continue;
            node.last_used = frame;
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

// ---------------------------------------------------------------------------
// Tests der reinen Host-Logik (ohne GPU)
// ---------------------------------------------------------------------------

test "Änderungen: Abbildung auf Chunk und Zelle über alle Stufen" {
    const testing = std.testing;
    const cl: u32 = 5; // 32^3 Voxel je Chunk

    // Ein Voxel bei (100, 70, -3): auf jeder Stufe muss er in dem Chunk liegen,
    // der seine Position überdeckt, und die lokale Zelle muss dazu passen.
    const p = EditPos{ .x = 100, .y = 70, .z = -3 };
    var lod: u32 = 0;
    while (lod <= 8) : (lod += 1) {
        const key = chunkKeyFor(cl, p, lod).?;
        const cell = localCell(cl, p, lod);
        // Rückrechnung: Weltposition der Zelle muss p enthalten
        const size: i64 = @as(i64, 1) << @intCast(cl + lod);
        const step: i64 = @as(i64, 1) << @intCast(lod);
        const wx = @as(i64, key.x) * size + @as(i64, cell[0]) * step;
        const wy = @as(i64, key.y) * size + @as(i64, cell[1]) * step;
        const wz = @as(i64, key.z) * size + @as(i64, cell[2]) * step;
        try testing.expect(p.x >= wx and p.x < wx + step);
        try testing.expect(p.y >= wy and p.y < wy + step);
        try testing.expect(p.z >= wz and p.z < wz + step);
        try testing.expect(cell[0] < @as(u32, 1) << @intCast(cl));
    }

    // Negative Koordinaten runden nach unten, nicht zur Null
    try testing.expectEqual(@as(i32, -1), chunkKeyFor(cl, .{ .x = -1, .y = 0, .z = 0 }, 0).?.x);
    try testing.expectEqual(@as(u32, 31), localCell(cl, .{ .x = -1, .y = 0, .z = 0 }, 0)[0]);
}

test "Änderungen: Entfernen wirkt grob erst, wenn die Zelle ganz leer ist" {
    const testing = std.testing;
    // Eine Zelle der Stufe 2 umfasst 8^3 = 512 Grundvoxel
    try testing.expectEqual(@as(u64, 1), cellVoxels(0));
    try testing.expectEqual(@as(u64, 512), cellVoxels(3));
    try testing.expectEqual(@as(u64, 4096), cellVoxels(4));

    // Alle Grundvoxel einer Zelle der Stufe 2 (4^3 = 64) gehören zu derselben
    // groben Zelle – erst der 64. Abbau darf sie leeren.
    const lod: u32 = 2;
    var removed: u64 = 0;
    var x: i64 = 0;
    while (x < 4) : (x += 1) {
        var y: i64 = 0;
        while (y < 4) : (y += 1) {
            var z: i64 = 0;
            while (z < 4) : (z += 1) {
                const cell = cellOf(.{ .x = x, .y = y, .z = z }, lod);
                try testing.expectEqual(CellKey{ .lod = 2, .x = 0, .y = 0, .z = 0 }, cell);
                removed += 1;
                // vorher greift die Entfernung auf dieser Stufe nicht
                try testing.expectEqual(removed == cellVoxels(lod), removed >= cellVoxels(lod));
            }
        }
    }
    try testing.expectEqual(cellVoxels(lod), removed);
}
