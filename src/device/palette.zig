//! Attribute kompakt: je Chunk eine Palette aus höchstens 16 Werten und je
//! Voxel ein 4-Bit-Index (8 je Wort) statt 32 Bit (types.geometry_palette).
//!
//! Gemessen in der Demo-Welt: höchstens 12 verschiedene Attribute je Chunk,
//! sobald die Farbschwankung je Säule aus dem Attribut in den Shader gewandert
//! ist (Material.variation). Attribute waren 94 % des Weltspeichers.
//!
//! Zwei Kernel, je ein Block pro Chunk:
//!   scan: verschiedene Werte sammeln (Hash in gemeinsamem Speicher);
//!         mehr als 16 -> Chunk behält 32 Bit je Voxel
//!   pack: Indizes schreiben (ein Thread je Ausgabewort, keine Konflikte)

const builtin = @import("builtin");
const types = @import("types.zig");

const nvptx = builtin.cpu.arch == .nvptx64 or builtin.cpu.arch == .nvptx;

pub const block: u32 = 256;
pub const max_entries: u32 = 16;
/// Zähler für "zu viele verschiedene Werte"
pub const overflow: u32 = 0xFFFF_FFFF;

const slots = 64; // offene Adressierung, Füllgrad <= 1/4
const empty: u32 = 0xFFFF_FFFF;

var sh_keys: [slots]u32 addrspace(.shared) = undefined;
var sh_count: u32 addrspace(.shared) = undefined;
var sh_full: u32 addrspace(.shared) = undefined;

inline fn barrier() void {
    if (comptime nvptx) asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn hash(v: u32) u32 {
    var h = v *% 0x9E37_79B1;
    h ^= h >> 15;
    return h;
}

/// Parameter beider Kernel
pub const Params = extern struct {
    /// u32-Attribute des Baus (Chunks hintereinander)
    attrs: u64,
    /// u32[count + 1]: erstes Voxel je Chunk (letzter Eintrag = Gesamtzahl)
    first: u64,
    /// scan: Ausgabe; pack: Eingabe. [count][16] Paletten, [count] Anzahl
    palettes: u64,
    counts: u64,
    /// pack: Ziel je Chunk (Wortoffset im Pool) und der Pool
    dst: u64,
    pool: u64,
    count: u32,
    reserved: u32 = 0,
};

pub fn scan(p: *const Params, c: u32, tid: u32) void {
    if (c >= p.count) return;
    const first: [*]const u32 = @ptrFromInt(p.first);
    const attrs: [*]const u32 = @ptrFromInt(p.attrs);
    const lo = first[c];
    const hi = first[c + 1];
    var k = tid;
    while (k < slots) : (k += block) sh_keys[k] = empty;
    if (tid == 0) {
        sh_count = 0;
        sh_full = 0;
    }
    barrier();
    var i = lo + tid;
    while (i < hi) : (i += block) {
        const v = attrs[i];
        var s = hash(v) & (slots - 1);
        var probes: u32 = 0;
        while (probes < slots) : (probes += 1) {
            const cur = @atomicLoad(u32, &sh_keys[s], .monotonic);
            if (cur == v) break;
            if (cur == empty) {
                // einfügen: ein Vergleich-und-Tausch, nur der Gewinner zählt
                const prev = @cmpxchgStrong(u32, &sh_keys[s], empty, v, .monotonic, .monotonic);
                if (prev == null) {
                    const n = @atomicRmw(u32, &sh_count, .Add, 1, .monotonic);
                    if (n >= max_entries) sh_full = 1;
                    break;
                }
                if (prev.? == v) break;
            }
            s = (s + 1) & (slots - 1);
        }
        if (probes == slots) sh_full = 1;
    }
    barrier();
    if (tid == 0) {
        const counts: [*]u32 = @ptrFromInt(p.counts);
        const pal: [*]u32 = @as([*]u32, @ptrFromInt(p.palettes)) + @as(u64, c) * max_entries;
        if (sh_full != 0 or sh_count > max_entries) {
            counts[c] = overflow;
        } else {
            var n: u32 = 0;
            var q: u32 = 0;
            while (q < slots) : (q += 1) {
                if (sh_keys[q] != empty) {
                    pal[n] = sh_keys[q];
                    n += 1;
                }
            }
            // Rest mit dem ersten Wert füllen: jede Palette hat 16 gültige Einträge
            while (n < max_entries) : (n += 1) pal[n] = if (sh_count > 0) pal[0] else 0;
            counts[c] = sh_count;
        }
    }
}

pub fn pack(p: *const Params, c: u32, tid: u32) void {
    if (c >= p.count) return;
    const first: [*]const u32 = @ptrFromInt(p.first);
    const attrs: [*]const u32 = @ptrFromInt(p.attrs);
    const counts: [*]const u32 = @ptrFromInt(p.counts);
    const pool: [*]u32 = @ptrFromInt(p.pool);
    const dst = @as([*]const u32, @ptrFromInt(p.dst))[c];
    const lo = first[c];
    const n = first[c + 1] - lo;
    if (counts[c] == overflow) {
        // zu viele Werte: 32 Bit je Voxel wie bisher
        var i = tid;
        while (i < n) : (i += block) pool[dst + i] = attrs[lo + i];
        return;
    }
    const pal: [*]const u32 = @as([*]const u32, @ptrFromInt(p.palettes)) + @as(u64, c) * max_entries;
    // Palette vor die Indizes
    if (tid < max_entries) pool[dst + tid] = pal[tid];
    const words = (n + 7) / 8;
    var w = tid;
    while (w < words) : (w += block) {
        var word: u32 = 0;
        var j: u32 = 0;
        while (j < 8) : (j += 1) {
            const i = w * 8 + j;
            if (i >= n) break;
            const v = attrs[lo + i];
            var idx: u32 = 0;
            var q: u32 = 0;
            while (q < max_entries) : (q += 1) {
                if (pal[q] == v) {
                    idx = q;
                    break;
                }
            }
            word |= idx << @intCast(j * 4);
        }
        pool[dst + max_entries + w] = word;
    }
}
