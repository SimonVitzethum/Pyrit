//! DAG-Bau auf der GPU. Jede Operation ist eine Funktion pro Thread
//! (`run(p, i)`); auf der GPU ein Kernel, auf der CPU eine Schleife.
//!
//! Ablauf (Steuerung in src/gpu_build.zig):
//!  1. encode         Voxel (x,y,z,a) -> Schlüssel (Brick-Morton << 6 | Bit)
//!  2. radix_*        stabiler Radix-Sort der Paare (Schlüssel, Attribut)
//!  3. unique_*       gleiche Schlüssel: der letzte gewinnt; Attribut 0 löscht
//!  4. brick_*        Voxel zu 4x4x4-Bricks (Maske, Anzahl, Hülle)
//!  5. leaf_*         Bricks deduplizieren (Hashtabelle) -> Blätter
//!  6. node_* je Ebene: Kinder zu Knoten, deduplizieren, Wortoffsets vergeben
//!  7. prim_emit      auf Ebene rt_log2: AABB-Primitive für die RT-Cores
//!  8. finalize       leere Geometrie
//!
//! Längen stehen in `counts` auf der GPU; Kernel werden mit Obergrenzen
//! gestartet und prüfen selbst. Die Knoten sind relativ zum Geometriebeginn
//! und passen damit direkt in die Pools (gleiches Format wie dag.zig).

const types = @import("types.zig");
const rt = @import("rt.zig");

pub const op_encode: u32 = 0;
pub const op_radix_hist: u32 = 1;
pub const op_radix_scatter: u32 = 2;
pub const op_scan_local: u32 = 3;
pub const op_scan_totals: u32 = 4;
pub const op_scan_add: u32 = 5;
pub const op_unique_flags: u32 = 6;
pub const op_unique_scatter: u32 = 7;
pub const op_brick_heads: u32 = 8;
pub const op_brick_build: u32 = 9;
pub const op_leaf_insert: u32 = 10;
pub const op_leaf_owner: u32 = 11;
pub const op_leaf_first: u32 = 12;
pub const op_leaf_write: u32 = 13;
pub const op_node_heads: u32 = 14;
pub const op_node_build: u32 = 15;
pub const op_node_insert: u32 = 16;
pub const op_node_owner: u32 = 17;
pub const op_node_first: u32 = 18;
pub const op_node_write: u32 = 19;
pub const op_prim_setup: u32 = 20;
pub const op_prim_emit: u32 = 21;
pub const op_level_advance: u32 = 22;
pub const op_finalize: u32 = 23;
pub const op_hash_clear: u32 = 24;
pub const op_downsample: u32 = 25;
pub const op_pack_chunks: u32 = 26;
pub const op_chunk_first: u32 = 27;
pub const op_roots: u32 = 28;
pub const op_prim_start: u32 = 29;

/// Zähler in `counts`
pub const c_unique = 0;
pub const c_children = 1;
pub const c_parents = 2;
pub const c_leaves = 3;
pub const c_cursor = 4;
pub const c_scan = 5;
pub const c_prims = 6;
pub const c_root = 7;
pub const c_tmp = 8;
pub const c_count = 9;

pub const scan_segment: u32 = 256;
pub const no_slot: u32 = 0xFFFF_FFFF;
pub const node_stride = 10;

inline fn ptr(comptime T: type, a: u64) [*]T {
    return @ptrFromInt(a);
}

inline fn cptr(comptime T: type, a: u64) [*]const T {
    return @ptrFromInt(a);
}

inline fn counts(p: *const types.BuildParams) [*]u32 {
    return ptr(u32, p.counts);
}

/// Bits des Schlüssels innerhalb eines Chunks (Brick-Morton << 6 | Bit)
pub inline fn localBits(log2_size: u32) u32 {
    return 3 * (log2_size - 2) + 6;
}

/// Anzahl Schlüsselbits inklusive Chunknummer und Ungültig-Bit
pub fn keyBits(log2_size: u32, chunk_bits: u32) u32 {
    return localBits(log2_size) + chunk_bits + 1;
}

pub fn radixPasses(log2_size: u32, chunk_bits: u32) u32 {
    return (keyBits(log2_size, chunk_bits) + 7) / 8;
}

inline fn invalidKeyP(p: *const types.BuildParams) u64 {
    return @as(u64, 1) << @intCast(keyBits(p.log2_size, p.chunk_bits) - 1);
}

inline fn localKey(x: u32, y: u32, z: u32) u64 {
    const brick = spread3(x >> 2) | (spread3(y >> 2) << 1) | (spread3(z >> 2) << 2);
    return (brick << 6) | ((x & 3) + 4 * (y & 3) + 16 * (z & 3));
}

fn spread3(v: u32) u64 {
    var x: u64 = v & 0x1FFFFF;
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
}

fn compact3(v: u64) u32 {
    var x = v & 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
    x = (x | (x >> 4)) & 0x100F00F00F00F00F;
    x = (x | (x >> 8)) & 0x1F0000FF0000FF;
    x = (x | (x >> 16)) & 0x1F00000000FFFF;
    x = (x | (x >> 32)) & 0x1FFFFF;
    return @intCast(x);
}

fn hash64(v: u64) u64 {
    var x = v +% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn scanLen(p: *const types.BuildParams) u32 {
    return if (p.scan_len_slot == no_slot) p.scan_len else counts(p)[p.scan_len_slot];
}

/// Hülle packen: min/max je Achse mit 21 Bit
inline fn packBox(v: [3]u32) u64 {
    return rt.packCell(v[0], v[1], v[2]);
}

inline fn satAdd(a: u32, b: u32) u32 {
    const r = @addWithOverflow(a, b);
    return if (r[1] != 0) 0xFFFF_FFFF else r[0];
}

pub fn run(p: *const types.BuildParams, i: u32) void {
    switch (p.op) {
        op_encode => encode(p, i),
        op_radix_hist => radixHist(p, i),
        op_radix_scatter => radixScatter(p, i),
        op_scan_local => scanLocal(p, i),
        op_scan_totals => if (i == 0) scanTotals(p),
        op_scan_add => scanAdd(p, i),
        op_unique_flags => uniqueFlags(p, i),
        op_unique_scatter => uniqueScatter(p, i),
        op_brick_heads => brickHeads(p, i),
        op_brick_build => brickBuild(p, i),
        op_leaf_insert => leafInsert(p, i),
        op_leaf_owner => owner(p, i, c_children),
        op_leaf_first => leafFirst(p, i),
        op_leaf_write => leafWrite(p, i),
        op_node_heads => nodeHeads(p, i),
        op_node_build => nodeBuild(p, i),
        op_node_insert => nodeInsert(p, i),
        op_node_owner => owner(p, i, c_parents),
        op_node_first => nodeFirst(p, i),
        op_node_write => nodeWrite(p, i),
        op_prim_setup => primSetup(p, i),
        op_prim_emit => primEmit(p, i),
        op_level_advance => if (i == 0) levelAdvance(p),
        op_finalize => if (i == 0) finalize(p),
        op_hash_clear => hashClear(p, i),
        op_downsample => downsample(p, i),
        op_pack_chunks => packChunks(p, i),
        op_chunk_first => chunkFirst(p, i),
        op_roots => roots(p, i),
        op_prim_start => primStart(p, i),
        else => {},
    }
}

// ---------------------------------------------------------------------------
// Schlüssel und Sortierung
// ---------------------------------------------------------------------------

fn encode(p: *const types.BuildParams, i: u32) void {
    if (i >= p.n_new) return;
    const v = cptr([4]u32, p.edits)[i];
    const n: u32 = @as(u32, 1) << @intCast(p.log2_size);
    const x: i32 = @bitCast(v[0]);
    const y: i32 = @bitCast(v[1]);
    const z: i32 = @bitCast(v[2]);
    const dst = p.n_old + i;
    if (x < 0 or y < 0 or z < 0 or x >= n or y >= n or z >= n) {
        ptr(u64, p.keys_out)[dst] = invalidKeyP(p);
        ptr(u32, p.vals_out)[dst] = 0;
        return;
    }
    ptr(u64, p.keys_out)[dst] = localKey(@intCast(x), @intCast(y), @intCast(z));
    ptr(u32, p.vals_out)[dst] = v[3];
}

/// Chunk-Batch: Eingabe ist in Segmente zu chunk_capacity Voxeln aufgeteilt
/// (lokale Koordinaten je Chunk); ungenutzte Plätze werden ungültig.
fn packChunks(p: *const types.BuildParams, i: u32) void {
    if (i >= p.n_total) return;
    // Chunk per Binärsuche in den Präfixen: offsets[c] <= i < offsets[c + 1]
    const offs = cptr(u32, p.chunk_offsets);
    var lo: u32 = 0;
    var hi: u32 = p.chunk_count;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (offs[mid] <= i) lo = mid else hi = mid;
    }
    const c = lo;
    const j = i - offs[c];
    const n: u32 = @as(u32, 1) << @intCast(p.log2_size);
    var key = invalidKeyP(p);
    var val: u32 = 0;
    if (j < p.chunk_capacity) {
        const v = cptr([4]u32, p.edits)[@as(u64, c) * p.chunk_capacity + j];
        const x: i32 = @bitCast(v[0]);
        const y: i32 = @bitCast(v[1]);
        const z: i32 = @bitCast(v[2]);
        if (x >= 0 and y >= 0 and z >= 0 and x < n and y < n and z < n) {
            key = (@as(u64, c) << @intCast(localBits(p.log2_size))) | localKey(@intCast(x), @intCast(y), @intCast(z));
            val = v[3];
        }
    }
    ptr(u64, p.keys_out)[i] = key;
    ptr(u32, p.vals_out)[i] = val;
}

inline fn chunkOf(p: *const types.BuildParams, key: u64) u32 {
    return @intCast(key >> @intCast(localBits(p.log2_size)));
}

/// Erstes Voxel (Attributrang) je Chunk; leere Chunks behalten 0xFFFFFFFF
fn chunkFirst(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_unique]) return;
    const ukey = cptr(u64, p.ukey);
    const c = chunkOf(p, ukey[i]);
    if (i == 0 or chunkOf(p, ukey[i - 1]) != c) ptr(u32, p.chunk_first)[c] = i;
}

/// Wurzel je Chunk (auf der obersten Ebene ist par_key = Chunknummer)
fn roots(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    ptr(u32, p.roots_out)[@intCast(cptr(u64, p.par_key)[i])] = cptr(u32, p.par_ref)[i];
}

/// Erstes Primitiv je Chunk (Primitive sind nach Chunk sortiert)
fn primStart(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    const key = cptr(u64, p.par_key);
    const shift: u6 = @intCast(3 * (p.log2_size - p.rt_log2));
    const c: u32 = @intCast(key[i] >> shift);
    if (i == 0 or (key[i - 1] >> shift) != c) ptr(u32, p.prim_start)[c] = i;
}

/// Verkleinern: Quellschlüssel -> Koordinaten -> um `shift` verkleinert -> neuer
/// Schlüssel. Nach dem stabilen Sortieren gewinnt pro grober Zelle das Quellvoxel
/// mit dem größten Schlüssel (in y/z oben liegend).
fn downsample(p: *const types.BuildParams, i: u32) void {
    if (i >= p.n_new) return;
    const key = cptr(u64, p.src_keys)[i];
    const brick = key >> 6;
    const bit: u32 = @intCast(key & 63);
    const sh: u5 = @intCast(p.shift);
    const x = ((compact3(brick) << 2) | (bit & 3)) >> sh;
    const y = ((compact3(brick >> 1) << 2) | ((bit >> 2) & 3)) >> sh;
    const z = ((compact3(brick >> 2) << 2) | (bit >> 4)) >> sh;
    const nb = spread3(x >> 2) | (spread3(y >> 2) << 1) | (spread3(z >> 2) << 2);
    ptr(u64, p.keys_out)[i] = (nb << 6) | ((x & 3) + 4 * (y & 3) + 16 * (z & 3));
    ptr(u32, p.vals_out)[i] = cptr(u32, p.src_vals)[i];
}

/// Ein Thread pro Block von `chunk` Elementen; Histogramm nach Ziffer sortiert
fn radixHist(p: *const types.BuildParams, c: u32) void {
    if (c >= p.chunks) return;
    const keys = cptr(u64, p.keys_in);
    const hist = ptr(u32, p.hist);
    const shift: u6 = @intCast(8 * p.pass);
    var d: u32 = 0;
    while (d < 256) : (d += 1) hist[d * p.chunks + c] = 0;
    const start = c * p.chunk;
    const end = @min(start + p.chunk, p.n_total);
    var k = start;
    while (k < end) : (k += 1) {
        const digit: u32 = @intCast((keys[k] >> shift) & 0xFF);
        hist[digit * p.chunks + c] += 1;
    }
}

/// Stabil: jeder Thread verteilt seinen Block der Reihe nach
fn radixScatter(p: *const types.BuildParams, c: u32) void {
    if (c >= p.chunks) return;
    const keys = cptr(u64, p.keys_in);
    const vals = cptr(u32, p.vals_in);
    const ko = ptr(u64, p.keys_out);
    const vo = ptr(u32, p.vals_out);
    const hist = cptr(u32, p.hist);
    const shift: u6 = @intCast(8 * p.pass);
    var offs: [256]u32 = undefined;
    var d: u32 = 0;
    while (d < 256) : (d += 1) offs[d] = hist[d * p.chunks + c];
    const start = c * p.chunk;
    const end = @min(start + p.chunk, p.n_total);
    var k = start;
    while (k < end) : (k += 1) {
        const digit: u32 = @intCast((keys[k] >> shift) & 0xFF);
        const dst = offs[digit];
        offs[digit] = dst + 1;
        ko[dst] = keys[k];
        vo[dst] = vals[k];
    }
}

// ---------------------------------------------------------------------------
// Exklusiver Scan (in-place) in drei Schritten
// ---------------------------------------------------------------------------

fn scanLocal(p: *const types.BuildParams, t: u32) void {
    const len = scanLen(p);
    const start = t * scan_segment;
    if (start >= len) return;
    const data = ptr(u32, p.scan_data);
    const end = @min(start + scan_segment, len);
    var sum: u32 = 0;
    var k = start;
    while (k < end) : (k += 1) {
        const v = data[k];
        data[k] = sum;
        sum +%= v;
    }
    ptr(u32, p.scan_totals)[t] = sum;
}

fn scanTotals(p: *const types.BuildParams) void {
    const len = scanLen(p);
    const segs = (len + scan_segment - 1) / scan_segment;
    const totals = ptr(u32, p.scan_totals);
    var sum: u32 = 0;
    var k: u32 = 0;
    while (k < segs) : (k += 1) {
        const v = totals[k];
        totals[k] = sum;
        sum +%= v;
    }
    counts(p)[p.scan_result_slot] = sum;
}

fn scanAdd(p: *const types.BuildParams, t: u32) void {
    const len = scanLen(p);
    const start = t * scan_segment;
    if (start >= len or t == 0) return;
    const add = cptr(u32, p.scan_totals)[t];
    const data = ptr(u32, p.scan_data);
    const end = @min(start + scan_segment, len);
    var k = start;
    while (k < end) : (k += 1) data[k] +%= add;
}

// ---------------------------------------------------------------------------
// Eindeutige Voxel
// ---------------------------------------------------------------------------

fn keep(p: *const types.BuildParams, i: u32) bool {
    const keys = cptr(u64, p.keys_in);
    const k = keys[i];
    if (k & invalidKeyP(p) != 0) return false;
    if (cptr(u32, p.vals_in)[i] == 0) return false; // Löschen
    return i + 1 == p.n_total or keys[i + 1] != k; // der letzte gleiche Schlüssel gewinnt
}

fn uniqueFlags(p: *const types.BuildParams, i: u32) void {
    if (i >= p.n_total) return;
    // der letzte Eintrag eines Schlüssels entscheidet, auch wenn er löscht
    const keys = cptr(u64, p.keys_in);
    const last = i + 1 == p.n_total or keys[i + 1] != keys[i];
    const valid = keys[i] & invalidKeyP(p) == 0 and cptr(u32, p.vals_in)[i] != 0;
    ptr(u32, p.flags)[i] = @intFromBool(last and valid);
}

fn uniqueScatter(p: *const types.BuildParams, i: u32) void {
    if (i >= p.n_total or !keep(p, i)) return;
    const pos = cptr(u32, p.flags)[i];
    ptr(u64, p.ukey)[pos] = cptr(u64, p.keys_in)[i];
    ptr(u32, p.uval)[pos] = cptr(u32, p.vals_in)[i];
}

// ---------------------------------------------------------------------------
// Bricks und Blätter
// ---------------------------------------------------------------------------

inline fn isBrickHead(ukey: [*]const u64, i: u32) bool {
    return i == 0 or (ukey[i] >> 6) != (ukey[i - 1] >> 6);
}

fn brickHeads(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_unique]) return;
    ptr(u32, p.flags)[i] = @intFromBool(isBrickHead(cptr(u64, p.ukey), i));
}

fn brickBuild(p: *const types.BuildParams, i: u32) void {
    const u = counts(p)[c_unique];
    if (i >= u) return;
    const ukey = cptr(u64, p.ukey);
    if (!isBrickHead(ukey, i)) return;
    const b = cptr(u32, p.flags)[i];
    const brick = ukey[i] >> 6;
    var mask: u64 = 0;
    var lo = [3]u32{ 4, 4, 4 };
    var hi = [3]u32{ 0, 0, 0 };
    var j = i;
    while (j < u and (ukey[j] >> 6) == brick) : (j += 1) {
        const bit: u32 = @intCast(ukey[j] & 63);
        mask |= @as(u64, 1) << @intCast(bit);
        const v = [3]u32{ bit & 3, (bit >> 2) & 3, bit >> 4 };
        inline for (0..3) |a| {
            lo[a] = @min(lo[a], v[a]);
            hi[a] = @max(hi[a], v[a] + 1);
        }
    }
    ptr(u64, p.ch_key)[b] = brick;
    ptr(u64, p.masks)[b] = mask;
    ptr(u32, p.ch_count)[b] = j - i;
    ptr(u64, p.ch_lo)[b] = packBox(lo);
    ptr(u64, p.ch_hi)[b] = packBox(hi);
}

fn hashClear(p: *const types.BuildParams, i: u32) void {
    if (i >= p.hash_cap) return;
    ptr(u64, p.hash_keys)[i] = 0;
    ptr(u32, p.hash_owner)[i] = 0xFFFF_FFFF;
}

fn leafInsert(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_children]) return;
    const mask = cptr(u64, p.masks)[i];
    const table = ptr(u64, p.hash_keys);
    const m = p.hash_cap - 1;
    var h: u32 = @truncate(hash64(mask) & m);
    while (true) : (h = (h + 1) & m) {
        const prev = @cmpxchgStrong(u64, &table[h], 0, mask, .monotonic, .monotonic);
        if (prev == null or prev.? == mask) break;
    }
    ptr(u32, p.slot)[i] = h;
}

/// Kleinster Index gleicher Inhalte wird Besitzer (deterministisch)
fn owner(p: *const types.BuildParams, i: u32, comptime count_slot: u32) void {
    if (i >= counts(p)[count_slot]) return;
    const s = cptr(u32, p.slot)[i];
    _ = @atomicRmw(u32, &ptr(u32, p.hash_owner)[s], .Min, i, .monotonic);
}

fn leafFirst(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_children]) return;
    const s = cptr(u32, p.slot)[i];
    ptr(u32, p.flags)[i] = @intFromBool(cptr(u32, p.hash_owner)[s] == i);
}

fn leafWrite(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_children]) return;
    const s = cptr(u32, p.slot)[i];
    const o = cptr(u32, p.hash_owner)[s];
    const pos = cptr(u32, p.flags);
    if (o == i) ptr(u64, p.leaves_out)[pos[i]] = cptr(u64, p.masks)[i];
    ptr(u32, p.ch_ref)[i] = pos[o];
}

// ---------------------------------------------------------------------------
// Knotenebenen
// ---------------------------------------------------------------------------

inline fn isNodeHead(key: [*]const u64, i: u32) bool {
    return i == 0 or (key[i] >> 3) != (key[i - 1] >> 3);
}

fn nodeHeads(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_children]) return;
    ptr(u32, p.flags)[i] = @intFromBool(isNodeHead(cptr(u64, p.ch_key), i));
}

fn nodeBuild(p: *const types.BuildParams, i: u32) void {
    const n = counts(p)[c_children];
    if (i >= n) return;
    const key = cptr(u64, p.ch_key);
    if (!isNodeHead(key, i)) return;
    const par = cptr(u32, p.flags)[i];
    const parent_key = key[i] >> 3;
    const child_size = @as(u32, 1) << @intCast(p.level);
    const words = ptr(u32, p.node_tmp) + @as(u64, par) * node_stride;
    var mask: u32 = 0;
    var count: u32 = 0;
    var k: u32 = 0;
    var lo = [3]u32{ 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF };
    var hi = [3]u32{ 0, 0, 0 };
    var j = i;
    while (j < n and (key[j] >> 3) == parent_key) : (j += 1) {
        const idx: u32 = @intCast(key[j] & 7);
        mask |= @as(u32, 1) << @intCast(idx);
        words[2 + k] = cptr(u32, p.ch_ref)[j];
        k += 1;
        count = satAdd(count, cptr(u32, p.ch_count)[j]);
        const off = [3]u32{ (idx & 1) * child_size, ((idx >> 1) & 1) * child_size, (idx >> 2) * child_size };
        const clo = rt.unpackCell(cptr(u64, p.ch_lo)[j]);
        const chi = rt.unpackCell(cptr(u64, p.ch_hi)[j]);
        inline for (0..3) |a| {
            lo[a] = @min(lo[a], clo[a] + off[a]);
            hi[a] = @max(hi[a], chi[a] + off[a]);
        }
    }
    words[0] = mask;
    words[1] = count;
    ptr(u32, p.node_len)[par] = 2 + k;
    ptr(u64, p.par_key)[par] = parent_key;
    ptr(u32, p.par_count)[par] = count;
    ptr(u64, p.par_lo)[par] = packBox(lo);
    ptr(u64, p.par_hi)[par] = packBox(hi);
}

fn nodeWords(p: *const types.BuildParams, i: u32) []const u32 {
    const len = cptr(u32, p.node_len)[i];
    return (cptr(u32, p.node_tmp) + @as(u64, i) * node_stride)[0..len];
}

fn nodeInsert(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    const words = nodeWords(p, i);
    var h64: u64 = words.len;
    for (words) |w| h64 = hash64(h64 ^ w);
    // Tabelle speichert Index + 1 (0 = frei)
    const table = ptr(u64, p.hash_keys);
    const m = p.hash_cap - 1;
    var h: u32 = @truncate(h64 & m);
    while (true) : (h = (h + 1) & m) {
        const prev = @cmpxchgStrong(u64, &table[h], 0, @as(u64, i) + 1, .monotonic, .monotonic);
        if (prev == null) break;
        const other: u32 = @intCast(prev.? - 1);
        const ow = nodeWords(p, other);
        if (ow.len == words.len) {
            var same = true;
            for (ow, words) |a, b| {
                if (a != b) same = false;
            }
            if (same) break;
        }
    }
    ptr(u32, p.slot)[i] = h;
}

fn nodeFirst(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    const s = cptr(u32, p.slot)[i];
    const own = cptr(u32, p.hash_owner)[s] == i;
    ptr(u32, p.flags)[i] = if (own) cptr(u32, p.node_len)[i] else 0;
}

fn nodeWrite(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    const s = cptr(u32, p.slot)[i];
    const o = cptr(u32, p.hash_owner)[s];
    const pos = cptr(u32, p.flags);
    const base = counts(p)[c_cursor];
    if (o == i) {
        const words = nodeWords(p, i);
        const dst = ptr(u32, p.nodes_out) + base + pos[i];
        for (words, 0..) |w, k| dst[k] = w;
    }
    ptr(u32, p.par_ref)[i] = base + pos[o];
}

/// Scan-Eingabe für die Attributränge der Primitive
fn primSetup(p: *const types.BuildParams, i: u32) void {
    if (i >= counts(p)[c_parents]) return;
    ptr(u32, p.flags)[i] = cptr(u32, p.par_count)[i];
}

fn primEmit(p: *const types.BuildParams, i: u32) void {
    const n = counts(p)[c_parents];
    if (i >= n) return;
    const full_key = cptr(u64, p.par_key)[i];
    const local_shift: u6 = @intCast(3 * (p.log2_size - p.rt_log2));
    const key = full_key & ((@as(u64, 1) << local_shift) - 1);
    const chunk: u32 = @intCast(full_key >> local_shift);
    const cell = [3]u32{ compact3(key), compact3(key >> 1), compact3(key >> 2) };
    const first = if (p.chunk_bits > 0) cptr(u32, p.chunk_first)[chunk] else 0;
    ptr(types.RtPrim, p.prims_out)[i] = .{
        .node = cptr(u32, p.par_ref)[i],
        .attr_base = cptr(u32, p.flags)[i] - first,
        .cell = rt.packCell(cell[0], cell[1], cell[2]),
    };
    const size = @as(u32, 1) << @intCast(p.rt_log2);
    const lo = rt.unpackCell(cptr(u64, p.par_lo)[i]);
    const hi = rt.unpackCell(cptr(u64, p.par_hi)[i]);
    const eps: f32 = 1.0 / 256.0;
    var box: [6]f32 = undefined;
    inline for (0..3) |a| {
        const base: f32 = @floatFromInt(cell[a] * size);
        box[a] = base + @as(f32, @floatFromInt(lo[a])) - eps;
        box[3 + a] = base + @as(f32, @floatFromInt(hi[a])) + eps;
    }
    ptr([6]f32, p.aabbs_out)[i] = box;
    if (i == 0) counts(p)[c_prims] = n;
}

fn levelAdvance(p: *const types.BuildParams) void {
    const c = counts(p);
    c[c_cursor] += c[c_scan];
    c[c_children] = c[c_parents];
    if (p.level + 1 == p.log2_size) c[c_root] = if (c[c_parents] > 0) cptr(u32, p.par_ref)[0] else 0;
}

/// Leere Geometrie: Wurzel ohne Kinder
fn finalize(p: *const types.BuildParams) void {
    const c = counts(p);
    if (c[c_unique] != 0) return;
    const nodes = ptr(u32, p.nodes_out);
    nodes[c[c_cursor]] = 0;
    nodes[c[c_cursor] + 1] = 0;
    c[c_root] = c[c_cursor];
    c[c_cursor] += 2;
    c[c_prims] = 0;
}
