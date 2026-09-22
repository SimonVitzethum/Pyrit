//! Steuerung des DAG-Baus auf der GPU (Operationen in src/device/gbuild.zig).
//!
//! `Exec` abstrahiert Speicher und Kernelstarts: in der Laufzeit die CUDA-
//! Treiber-API, in Tests die CPU. Bis auf einen Rücklesevorgang am Ende (die
//! Größen für die Pools) bleibt alles auf der GPU.

const std = @import("std");
const types = @import("pyrit_device").types;
const gb = @import("pyrit_device").gbuild;
const diag = @import("diag.zig");

pub const Error = diag.Error;

pub const Exec = struct {
    ctx: *anyopaque,
    alloc: *const fn (ctx: *anyopaque, bytes: u64) Error!u64,
    free: *const fn (ctx: *anyopaque, ptr: u64) void,
    memset: *const fn (ctx: *anyopaque, ptr: u64, value: u8, bytes: u64) Error!void,
    copy: *const fn (ctx: *anyopaque, dst: u64, src: u64, bytes: u64) Error!void,
    launch: *const fn (ctx: *anyopaque, p: *const types.BuildParams, threads: u32) Error!void,
    /// synchron
    read: *const fn (ctx: *anyopaque, dst: []u8, src: u64) Error!void,
};

/// Ergebnis; alle Puffer gehören dem Aufrufer (Gerätespeicher).
pub const Built = struct {
    log2_size: u32,
    rt_log2: u32,
    nodes: u64,
    node_words: u32,
    root: u32,
    leaves: u64,
    leaf_count: u32,
    /// sortierte Voxel: Schlüssel (für spätere Änderungen) und Attribute
    keys: u64,
    attrs: u64,
    voxel_count: u32,
    prims: u64,
    aabbs: u64,
    prim_count: u32,

    pub fn free(self: *const Built, e: Exec) void {
        for ([_]u64{ self.nodes, self.leaves, self.keys, self.attrs, self.prims, self.aabbs }) |p| {
            if (p != 0) e.free(e.ctx, p);
        }
    }
};

fn nextPow2(v: u64) u32 {
    var c: u64 = 1024;
    while (c < v) c *= 2;
    return @intCast(c);
}

const Scratch = struct {
    e: Exec,
    list: [48]u64 = undefined,
    n: usize = 0,

    fn get(self: *Scratch, bytes: u64) Error!u64 {
        const p = try self.e.alloc(self.e.ctx, @max(bytes, 16));
        self.list[self.n] = p;
        self.n += 1;
        return p;
    }

    fn release(self: *Scratch) void {
        for (self.list[0..self.n]) |p| self.e.free(self.e.ctx, p);
        self.n = 0;
    }
};

fn launch(e: Exec, p: *types.BuildParams, op: u32, threads: u32) Error!void {
    p.op = op;
    try e.launch(e.ctx, p, @max(threads, 1));
}

/// Exklusiver Scan über p.scan_data mit fester Länge (len) oder Länge aus einem Zähler.
fn scan(e: Exec, p: *types.BuildParams, data: u64, len_slot: u32, len: u32, upper: u32, result_slot: u32) Error!void {
    p.scan_data = data;
    p.scan_len_slot = len_slot;
    p.scan_len = len;
    p.scan_result_slot = result_slot;
    const segs = (upper + gb.scan_segment - 1) / gb.scan_segment;
    try launch(e, p, gb.op_scan_local, segs);
    try launch(e, p, gb.op_scan_totals, 1);
    try launch(e, p, gb.op_scan_add, segs);
}

/// Baut eine DAG aus `n_old` bereits sortierten, eindeutigen Voxeln (old_keys,
/// old_vals; z. B. die vorige Fassung einer Geometrie) und `n_new` Voxeln im
/// Format [x, y, z, attribut] (edits). Neue Einträge überschreiben alte,
/// Attribut 0 löscht.
pub fn build(e: Exec, log2_size: u32, rt_log2_req: u32, old_keys: u64, old_vals: u64, n_old: u32, edits: u64, n_new: u32) Error!Built {
    return buildImpl(e, log2_size, rt_log2_req, old_keys, old_vals, n_old, edits, n_new, null, null);
}

/// Mehrere Chunks in einem Durchlauf (Welt-Streaming). `voxels` hat K Segmente
/// zu `capacity` Einträgen [x, y, z, attribut] in Chunk-lokalen Koordinaten;
/// Segment c ist bis `offsets[c + 1] - offsets[c]` belegt (Gerät, u32[K + 1],
/// exklusive Präfixsumme, jede Länge <= capacity), `total` = offsets[K].
/// Der Zwischenspeicher richtet sich nach `total`, nicht nach K · capacity.
/// Alle Chunks teilen sich Knoten-, Blatt- und Attributpuffer; `out` bekommt
/// je Chunk Wurzel, erstes Voxel und erstes Primitiv (0xFFFFFFFF = leer).
pub const ChunkBatch = struct {
    count: u32,
    capacity: u32,
    voxels: u64,
    offsets: u64,
    total: u32,
};

pub const ChunkOut = struct {
    roots: []u32,
    first_voxel: []u32,
    first_prim: []u32,
};

pub fn buildChunks(e: Exec, log2_size: u32, rt_log2_req: u32, batch: ChunkBatch, out: ChunkOut) Error!Built {
    if (batch.count == 0 or batch.capacity == 0) return diag.fail(error.InvalidArgument, "leerer Chunk-Batch", .{});
    return buildImpl(e, log2_size, rt_log2_req, 0, 0, 0, batch.voxels, batch.total, null, .{ .batch = batch, .out = out });
}

const Chunks = struct { batch: ChunkBatch, out: ChunkOut };

/// Verkleinerte Fassung (LOD): Kantenlänge 2^(log2_src - shift), je grober Zelle
/// ein Voxel. Quelle sind die sortierten Voxel einer änderbaren Geometrie.
pub fn downsample(e: Exec, log2_src: u32, rt_log2_req: u32, src_keys: u64, src_vals: u64, n_src: u32, shift: u32) Error!Built {
    if (shift == 0 or shift + 3 > log2_src) return diag.fail(error.InvalidArgument, "Verkleinerung um 2^{d} bei 2^{d} nicht möglich (Ergebnis mindestens 2^3)", .{ shift, log2_src });
    return buildImpl(e, log2_src - shift, rt_log2_req, 0, 0, 0, 0, n_src, .{ .keys = src_keys, .vals = src_vals, .shift = shift }, null);
}

const Downsample = struct { keys: u64, vals: u64, shift: u32 };

fn buildImpl(e: Exec, log2_size: u32, rt_log2_req: u32, old_keys: u64, old_vals: u64, n_old: u32, edits: u64, n_new: u32, ds: ?Downsample, chunks: ?Chunks) Error!Built {
    if (log2_size < 3 or log2_size > 20) return diag.fail(error.InvalidArgument, "log2_size muss in [3, 20] liegen", .{});
    const n_total = n_old + n_new;
    const nmax: u32 = @max(n_total, 1);
    const rt_log2 = std.math.clamp(rt_log2_req, 3, log2_size);
    const chunk_factor: u64 = if (chunks) |cb| cb.batch.count else 1;

    // Obere Schranken je Ebene: ab den Bricks gibt es nie mehr Elemente als
    // Zellen dieser Größe (und nie mehr als Voxel). Das spart den größten Teil
    // des Zwischenspeichers, sobald die Geometrie dünn besetzt ist.
    const cellsOf = struct {
        fn at(log2_size_: u32, lv: u32, factor: u64, cap: u32) u32 {
            const per = @as(u64, 1) << @intCast(3 * (log2_size_ - @min(lv, log2_size_)));
            return @intCast(@max(@min(@as(u64, cap), per * factor), 1));
        }
    };
    // Bricks (4^3-Zellen) – auch Blätter, Hash und Slots
    const n_brick = cellsOf.at(log2_size, 2, chunk_factor, nmax);
    // Knoten der ersten Ebene darüber (8^3-Zellen); höhere Ebenen sind kleiner
    const n_node = cellsOf.at(log2_size, 3, chunk_factor, nmax);

    var s = Scratch{ .e = e };
    defer s.release();
    var p = std.mem.zeroes(types.BuildParams);
    p.log2_size = log2_size;
    p.rt_log2 = rt_log2;
    p.n_old = n_old;
    p.n_new = n_new;
    p.n_total = n_total;
    if (chunks) |cb| {
        p.chunk_count = cb.batch.count;
        p.chunk_capacity = cb.batch.capacity;
        p.chunk_offsets = cb.batch.offsets;
        var bits: u32 = 0;
        while ((@as(u32, 1) << @intCast(bits)) < cb.batch.count) bits += 1;
        p.chunk_bits = bits;
        if (gb.keyBits(log2_size, bits) > 64) return diag.fail(error.InvalidArgument, "zu viele Chunks für diese Chunkgröße", .{});
        const k: u64 = cb.batch.count;
        p.chunk_first = try s.get(k * 4);
        p.roots_out = try s.get(k * 4);
        p.prim_start = try s.get(k * 4);
        for ([_]u64{ p.chunk_first, p.roots_out, p.prim_start }) |b| try e.memset(e.ctx, b, 0xFF, k * 4);
    }
    p.counts = try s.get(gb.c_count * 4);
    try e.memset(e.ctx, p.counts, 0, gb.c_count * 4);

    // 1. Schlüssel
    const keys = [2]u64{ try s.get(@as(u64, nmax) * 8), try s.get(@as(u64, nmax) * 8) };
    const vals = [2]u64{ try s.get(@as(u64, nmax) * 4), try s.get(@as(u64, nmax) * 4) };
    if (n_old > 0) {
        try e.copy(e.ctx, keys[0], old_keys, @as(u64, n_old) * 8);
        try e.copy(e.ctx, vals[0], old_vals, @as(u64, n_old) * 4);
    }
    p.edits = edits;
    p.keys_out = keys[0];
    p.vals_out = vals[0];
    if (chunks != null) {
        if (n_new > 0) try launch(e, &p, gb.op_pack_chunks, n_new);
    } else if (ds) |d| {
        p.src_keys = d.keys;
        p.src_vals = d.vals;
        p.shift = d.shift;
        if (n_new > 0) try launch(e, &p, gb.op_downsample, n_new);
    } else if (n_new > 0) try launch(e, &p, gb.op_encode, n_new);

    // 2. Radix-Sort (stabil, 8 Bit je Durchgang)
    // Elemente je Thread: 128 ist gemessen der beste Kompromiss aus Parallelität
    // und Speicher für die Histogramme (256: 3,3 ms/4 B je Voxel, 128: 2,7 ms/8 B,
    // 64: 2,85 ms/16 B bei 312k Voxeln)
    p.chunk = 128;
    p.chunks = (nmax + p.chunk - 1) / p.chunk;
    p.hist = try s.get(@as(u64, p.chunks) * 256 * 4);
    p.scan_totals = try s.get(@as(u64, (@max(p.chunks * 256, nmax) + gb.scan_segment - 1) / gb.scan_segment) * 4 + 16);
    var cur: usize = 0;
    var pass: u32 = 0;
    while (pass < gb.radixPasses(log2_size, p.chunk_bits)) : (pass += 1) {
        p.pass = pass;
        p.keys_in = keys[cur];
        p.vals_in = vals[cur];
        p.keys_out = keys[cur ^ 1];
        p.vals_out = vals[cur ^ 1];
        try launch(e, &p, gb.op_radix_hist, p.chunks);
        try scan(e, &p, p.hist, gb.no_slot, p.chunks * 256, p.chunks * 256, gb.c_scan);
        try launch(e, &p, gb.op_radix_scatter, p.chunks);
        cur ^= 1;
    }
    p.keys_in = keys[cur];
    p.vals_in = vals[cur];

    // 3. Eindeutige Voxel (Ergebnis bleibt als Quelle für spätere Änderungen)
    p.flags = try s.get(@as(u64, nmax) * 4);
    try launch(e, &p, gb.op_unique_flags, n_total);
    try scan(e, &p, p.flags, gb.no_slot, n_total, nmax, gb.c_unique);
    const ukey = try e.alloc(e.ctx, @as(u64, nmax) * 8);
    errdefer e.free(e.ctx, ukey);
    const uval = try e.alloc(e.ctx, @as(u64, nmax) * 4);
    errdefer e.free(e.ctx, uval);
    p.ukey = ukey;
    p.uval = uval;
    try launch(e, &p, gb.op_unique_scatter, n_total);
    if (chunks != null) try launch(e, &p, gb.op_chunk_first, nmax);

    // 4. Bricks: ch hält Bricks, par die Knoten darüber (beide wechseln je Ebene
    // die Rolle; die Größen nehmen nach oben hin ab)
    var ch = [5]u64{ try s.get(@as(u64, n_brick) * 8), try s.get(@as(u64, n_brick) * 4), try s.get(@as(u64, n_brick) * 4), try s.get(@as(u64, n_brick) * 8), try s.get(@as(u64, n_brick) * 8) };
    const setLevel = struct {
        fn f(pp: *types.BuildParams, c: [5]u64, q: [5]u64) void {
            pp.ch_key = c[0];
            pp.ch_ref = c[1];
            pp.ch_count = c[2];
            pp.ch_lo = c[3];
            pp.ch_hi = c[4];
            pp.par_key = q[0];
            pp.par_ref = q[1];
            pp.par_count = q[2];
            pp.par_lo = q[3];
            pp.par_hi = q[4];
        }
    }.f;
    p.ch_key = ch[0];
    p.ch_ref = ch[1];
    p.ch_count = ch[2];
    p.ch_lo = ch[3];
    p.ch_hi = ch[4];
    p.masks = try s.get(@as(u64, n_brick) * 8);
    try launch(e, &p, gb.op_brick_heads, nmax);
    try scan(e, &p, p.flags, gb.c_unique, 0, nmax, gb.c_children);
    try launch(e, &p, gb.op_brick_build, nmax);

    // Einmal die tatsächliche Brickzahl lesen: alles Weitere (Hash, Knoten,
    // Primitive) richtet sich danach statt nach der Zahl der Voxel. Bei dünn
    // besetzten Geometrien spart das den größten Teil des Zwischenspeichers.
    var c0: [gb.c_count]u32 = undefined;
    try e.read(e.ctx, std.mem.sliceAsBytes(&c0), p.counts);
    const n_brick_real: u32 = @max(c0[gb.c_children], 1);
    const n_node_real = @min(n_brick_real, n_node);
    var par = [5]u64{ try s.get(@as(u64, n_node_real) * 8), try s.get(@as(u64, n_node_real) * 4), try s.get(@as(u64, n_node_real) * 4), try s.get(@as(u64, n_node_real) * 8), try s.get(@as(u64, n_node_real) * 8) };
    setLevel(&p, ch, par);

    // 5. Blätter deduplizieren
    // Auslastung höchstens 0,8: lineares Sondieren bleibt kurz, spart aber
    // gegenüber Faktor 2 die Hälfte des Tabellenspeichers
    p.hash_cap = nextPow2(@as(u64, n_brick_real) * 5 / 4);
    p.hash_keys = try s.get(@as(u64, p.hash_cap) * 8);
    p.hash_owner = try s.get(@as(u64, p.hash_cap) * 4);
    p.slot = try s.get(@as(u64, n_brick_real) * 4);
    const leaves = try e.alloc(e.ctx, @as(u64, n_brick_real) * 8);
    errdefer e.free(e.ctx, leaves);
    p.leaves_out = leaves;
    try launch(e, &p, gb.op_hash_clear, p.hash_cap);
    try launch(e, &p, gb.op_leaf_insert, nmax);
    try launch(e, &p, gb.op_leaf_owner, nmax);
    try launch(e, &p, gb.op_leaf_first, nmax);
    try scan(e, &p, p.flags, gb.c_children, 0, nmax, gb.c_leaves);
    try launch(e, &p, gb.op_leaf_write, nmax);

    // 6. Knotenebenen; obere Schranke der Worte: je Ebene min(Voxel, Würfelanzahl) Knoten
    var word_bound: u64 = 2;
    var prim_bound: u64 = 1;
    var lv: u32 = 2;
    while (lv < log2_size) : (lv += 1) {
        const cells_per_axis = @as(u64, 1) << @intCast(log2_size - lv - 1);
        const cap = @min(@as(u64, n_brick_real), cells_per_axis * cells_per_axis * cells_per_axis * chunk_factor);
        word_bound += cap * gb.node_stride;
        if (lv + 1 == rt_log2) prim_bound = @max(cap, 1);
    }
    const nodes = try e.alloc(e.ctx, word_bound * 4);
    errdefer e.free(e.ctx, nodes);
    const prims = try e.alloc(e.ctx, prim_bound * @sizeOf(types.RtPrim));
    errdefer e.free(e.ctx, prims);
    const aabbs = try e.alloc(e.ctx, prim_bound * 24);
    errdefer e.free(e.ctx, aabbs);
    p.nodes_out = nodes;
    p.prims_out = prims;
    p.aabbs_out = aabbs;
    p.node_tmp = try s.get(@as(u64, n_node_real) * gb.node_stride * 4);
    p.node_len = try s.get(@as(u64, n_node_real) * 4);

    lv = 2;
    while (lv < log2_size) : (lv += 1) {
        p.level = lv;
        setLevel(&p, ch, par);
        try launch(e, &p, gb.op_node_heads, nmax);
        try scan(e, &p, p.flags, gb.c_children, 0, nmax, gb.c_parents);
        try launch(e, &p, gb.op_node_build, nmax);
        try launch(e, &p, gb.op_hash_clear, p.hash_cap);
        try launch(e, &p, gb.op_node_insert, nmax);
        try launch(e, &p, gb.op_node_owner, nmax);
        try launch(e, &p, gb.op_node_first, nmax);
        try scan(e, &p, p.flags, gb.c_parents, 0, nmax, gb.c_scan);
        try launch(e, &p, gb.op_node_write, nmax);
        if (lv + 1 == rt_log2) {
            try launch(e, &p, gb.op_prim_setup, nmax);
            try scan(e, &p, p.flags, gb.c_parents, 0, nmax, gb.c_tmp);
            try launch(e, &p, gb.op_prim_emit, nmax);
            if (chunks != null) try launch(e, &p, gb.op_prim_start, nmax);
        }
        if (chunks != null and lv + 1 == log2_size) try launch(e, &p, gb.op_roots, nmax);
        try launch(e, &p, gb.op_level_advance, 1);
        std.mem.swap([5]u64, &ch, &par);
    }
    try launch(e, &p, gb.op_finalize, 1);

    var c: [gb.c_count]u32 = undefined;
    try e.read(e.ctx, std.mem.sliceAsBytes(&c), p.counts);
    if (chunks) |cb| {
        try e.read(e.ctx, std.mem.sliceAsBytes(cb.out.roots[0..cb.batch.count]), p.roots_out);
        try e.read(e.ctx, std.mem.sliceAsBytes(cb.out.first_voxel[0..cb.batch.count]), p.chunk_first);
        try e.read(e.ctx, std.mem.sliceAsBytes(cb.out.first_prim[0..cb.batch.count]), p.prim_start);
    }
    return .{
        .log2_size = log2_size,
        .rt_log2 = rt_log2,
        .nodes = nodes,
        .node_words = c[gb.c_cursor],
        .root = c[gb.c_root],
        .leaves = leaves,
        .leaf_count = c[gb.c_leaves],
        .keys = ukey,
        .attrs = uval,
        .voxel_count = c[gb.c_unique],
        .prims = prims,
        .aabbs = aabbs,
        .prim_count = c[gb.c_prims],
    };
}

// ---------------------------------------------------------------------------
// CPU-Ausführung (Tests, Werkzeuge)
// ---------------------------------------------------------------------------

pub const CpuExec = struct {
    gpa: std.mem.Allocator,
    live: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn exec(self: *CpuExec) Exec {
        return .{ .ctx = self, .alloc = alloc, .free = free, .memset = memset, .copy = copy, .launch = launchCpu, .read = read };
    }

    pub fn deinit(self: *CpuExec) void {
        var it = self.live.iterator();
        while (it.next()) |kv| self.gpa.free(@as([*]align(16) u8, @ptrFromInt(kv.key_ptr.*))[0..kv.value_ptr.*]);
        self.live.deinit(self.gpa);
    }

    fn cast(c: *anyopaque) *CpuExec {
        return @ptrCast(@alignCast(c));
    }

    fn alloc(c: *anyopaque, bytes: u64) Error!u64 {
        const self = cast(c);
        const mem = self.gpa.alignedAlloc(u8, .@"16", bytes) catch return error.OutOfMemory;
        @memset(mem, 0xCD);
        self.live.put(self.gpa, @intFromPtr(mem.ptr), bytes) catch return error.OutOfMemory;
        return @intFromPtr(mem.ptr);
    }

    fn free(c: *anyopaque, p: u64) void {
        const self = cast(c);
        const len = self.live.fetchRemove(p) orelse return;
        self.gpa.free(@as([*]align(16) u8, @ptrFromInt(p))[0..len.value]);
    }

    fn memset(_: *anyopaque, p: u64, v: u8, bytes: u64) Error!void {
        @memset(@as([*]u8, @ptrFromInt(p))[0..bytes], v);
    }

    fn copy(_: *anyopaque, dst: u64, src: u64, bytes: u64) Error!void {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..bytes], @as([*]const u8, @ptrFromInt(src))[0..bytes]);
    }

    fn launchCpu(_: *anyopaque, p: *const types.BuildParams, threads: u32) Error!void {
        var i: u32 = 0;
        while (i < threads) : (i += 1) gb.run(p, i);
    }

    fn read(_: *anyopaque, dst: []u8, src: u64) Error!void {
        @memcpy(dst, @as([*]const u8, @ptrFromInt(src))[0..dst.len]);
    }
};
