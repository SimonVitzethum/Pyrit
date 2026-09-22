//! Format und Traversierung einer Sparse Voxel DAG (Host und GPU).
//!
//! Knoten (32-Bit-Worte, Verweise relativ zum Geometriebeginn):
//!   [0]   Kindmaske in Bits 0..7; Kindindex = x | y << 1 | z << 2
//!   [1]   Anzahl gesetzter Voxel im Teilbaum (sättigend bei 2^32 - 1)
//!   [2..] ein Verweis pro gesetztem Kind, aufsteigend nach Kindindex.
//!         Kinder der Kantenlänge 4 sind Blätter (Index ins Blattarray),
//!         alle größeren Kinder sind Knoten (Wortoffset ins Knotenarray).
//! Blatt: 4x4x4-Brick als 64-Bit-Maske, Bit = x + 4 * y + 16 * z.
//!
//! Attribute (optional) liegen in Tiefensuch-Reihenfolge nach Kindindex,
//! innerhalb eines Bricks in Bitreihenfolge. Der Rang eines Voxels ergibt sich
//! aus den Voxelzahlen der vorangehenden Geschwister (Dado et al. 2016).
//!
//! Objektraum: die Geometrie füllt [0, 2^log2_size)^3, ein Voxel ist 1 groß.

const vec = @import("vec.zig");
const Vec3 = vec.Vec3;

pub const min_log2: u32 = 3;
pub const max_log2: u32 = 20;
pub const max_steps: u32 = 65536;

pub const Dag = struct {
    nodes: [*]const u32,
    leaves: [*]const u64,
    /// null: default_attribute für alle Voxel
    attributes: ?[*]const u32,
    root: u32,
    log2_size: u32,
    default_attribute: u32,
};

pub const DagHit = struct {
    t: f32,
    /// face_* im Objektraum
    face: u32,
    voxel: [3]i32,
    attribute: u32,
    /// Strahl beginnt im getroffenen Voxel
    inside: bool,
};

pub inline fn brickBit(lx: u32, ly: u32, lz: u32) u32 {
    return lx + 4 * ly + 16 * lz;
}

/// Maske des 2x2x2-Blocks mit Ursprung (lx, ly, lz), jeweils 0 oder 2
inline fn brickBlockMask(lx: u32, ly: u32, lz: u32) u64 {
    return @as(u64, 0x0033_0033) << @intCast(brickBit(lx, ly, lz));
}

inline fn bitOf(v: u32, s: u32) u32 {
    return (v >> @intCast(s)) & 1;
}

/// Rang eines Voxels in der Attributreihenfolge. stk[s] ist der Knoten auf dem
/// Pfad, dessen Kinder Kantenlänge 2^s haben.
fn attributeRank(g: *const Dag, stk: *const [max_log2 + 1]u32, v: [3]u32, brick: u64) u32 {
    var rank: u32 = 0;
    var s: u32 = g.log2_size - 1;
    while (s >= 2) : (s -= 1) {
        const node = stk[s];
        const mask = g.nodes[node] & 0xFF;
        const idx = bitOf(v[0], s) | (bitOf(v[1], s) << 1) | (bitOf(v[2], s) << 2);
        const before: u32 = @popCount(mask & ((@as(u32, 1) << @intCast(idx)) - 1));
        var k: u32 = 0;
        while (k < before) : (k += 1) {
            const ref = g.nodes[node + 2 + k];
            rank +%= if (s == 2) @popCount(g.leaves[ref]) else g.nodes[ref + 1];
        }
    }
    const bit = brickBit(v[0] & 3, v[1] & 3, v[2] & 3);
    rank +%= @popCount(brick & ((@as(u64, 1) << @intCast(bit)) - 1));
    return rank;
}

inline fn exitAt(t: f32, axis: u32, oct: u32) DagHit {
    // Strahl läuft (gespiegelt) in +axis: Außennormale zeigt in Laufrichtung
    const mirrored = (oct >> @intCast(axis)) & 1 != 0;
    return .{ .t = t, .face = 2 * axis + @as(u32, if (mirrored) 1 else 0), .voxel = .{ 0, 0, 0 }, .attribute = 0, .inside = false };
}

/// Erster Treffer des Strahls o + t * d mit t in [tmin, tmax).
///
/// Verfahren: Die Strahlrichtung wird in den positiven Oktanten gespiegelt,
/// sodass Zellen nur in aufsteigender Koordinatenrichtung verlassen werden.
/// Die aktuelle Zelle ist ein ganzzahliger Ursprung plus Ebene (scale = log2
/// der Zellgröße). Beim Verlassen einer Zelle zeigt das höchste geänderte Bit
/// direkt die Ebene, bis zu der aufgestiegen werden muss (ESVO-Prinzip).
/// Die beiden untersten Ebenen liegen im 64-Bit-Brick.
///
/// Achsen werden mit `inline for` behandelt, damit alles in Registern bleibt.
pub fn trace(g: *const Dag, o: Vec3, d: Vec3, tmin: f32, tmax: f32, want_attribute: bool) ?DagHit {
    return walk(false, g, o, d, tmin, tmax, want_attribute, null);
}

/// Wie `trace`, überspringt aber Voxel, deren Material in `skip` steht (ein Bit
/// je Materialindex). So laufen Primär- und Schattenstrahlen durch Wasser oder
/// Glas hindurch, ohne die Traversierung mehrfach zu starten.
pub fn traceSkipping(g: *const Dag, o: Vec3, d: Vec3, tmin: f32, tmax: f32, want_attribute: bool, skip: *const [4]u64) ?DagHit {
    return walk(false, g, o, d, tmin, tmax, want_attribute, skip);
}

/// Austritt aus festem Material: erster Punkt ab tmin, an dem der Strahl in
/// eine leere Zelle übergeht (für transparente Medien). `face` ist die dabei
/// durchquerte Fläche als Außennormale des Materials. Startet der Strahl im
/// Leeren, ist das Ergebnis t = tmin. null, wenn der Strahl bis tmax im
/// Material bleibt; Verlassen des Würfels zählt als Austritt.
pub fn traceExit(g: *const Dag, o: Vec3, d: Vec3, tmin: f32, tmax: f32) ?DagHit {
    return walk(true, g, o, d, tmin, tmax, false, null);
}

inline fn skipped(skip: ?*const [4]u64, attribute: u32) bool {
    const m = skip orelse return false;
    const idx = attribute & 0xFF;
    return (m[idx >> 6] >> @intCast(idx & 63)) & 1 != 0;
}

fn walk(comptime exit_mode: bool, g: *const Dag, o: Vec3, d: Vec3, tmin: f32, tmax: f32, want_attribute: bool, skip: ?*const [4]u64) ?DagHit {
    const n: u32 = @as(u32, 1) << @intCast(g.log2_size);
    const nf: f32 = @floatFromInt(n);

    var oct: u32 = 0;
    var od: [3]f32 = undefined;
    var dd: [3]f32 = undefined;
    var inv: [3]f32 = undefined;
    inline for (0..3) |a| {
        var oa = o[a];
        var da = d[a];
        if (da < 0) {
            oa = nf - oa;
            da = -da;
            oct |= 1 << a;
        }
        od[a] = oa;
        dd[a] = da;
        inv[a] = 1.0 / @max(da, 1e-30);
    }

    // Eintritt in den Würfel [0, n)^3
    var axis: u32 = 0;
    var tn = -od[0] * inv[0];
    inline for (1..3) |a| {
        const ta = -od[a] * inv[a];
        if (ta > tn) {
            tn = ta;
            axis = a;
        }
    }
    const tf = @min((nf - od[0]) * inv[0], (nf - od[1]) * inv[1], (nf - od[2]) * inv[2]);
    var t = @max(tn, tmin);
    const tend = @min(tf, tmax);
    if (exit_mode and tn > tmin) return exitAt(tmin, axis, oct); // Start außerhalb: leer
    if (!(t < tend)) return if (exit_mode and tf <= tmax and t >= tf) exitAt(@max(tf, tmin), axis, oct) else null;
    const starts_inside = tn < tmin;
    var moved = false;

    var scale: u32 = g.log2_size - 1;
    var cell: [3]u32 = undefined;
    inline for (0..3) |a| {
        const p = @min(@max(@floor(od[a] + dd[a] * t), 0.0), nf - 1.0);
        cell[a] = @as(u32, @intFromFloat(p)) & ~((@as(u32, 1) << @intCast(scale)) - 1);
    }

    var stk: [max_log2 + 1]u32 = undefined;
    stk[scale] = g.root;
    var brick: u64 = 0;

    var step: u32 = 0;
    while (step < max_steps) : (step += 1) {
        // Ist die aktuelle Zelle belegt?
        var exists: bool = undefined;
        var child: u32 = 0;
        if (scale >= 2) {
            const node = stk[scale];
            const mask = g.nodes[node] & 0xFF;
            const idx = (bitOf(cell[0], scale) | (bitOf(cell[1], scale) << 1) | (bitOf(cell[2], scale) << 2)) ^ oct;
            const bit = @as(u32, 1) << @intCast(idx);
            exists = mask & bit != 0;
            if (exists) child = g.nodes[node + 2 + @popCount(mask & (bit - 1))];
        } else {
            const size = @as(u32, 1) << @intCast(scale);
            var l: [3]u32 = undefined;
            inline for (0..3) |a| {
                l[a] = (if ((oct >> a) & 1 != 0) n - size - cell[a] else cell[a]) & 3;
            }
            exists = if (scale == 1)
                brick & brickBlockMask(l[0], l[1], l[2]) != 0
            else
                (brick >> @intCast(brickBit(l[0], l[1], l[2]))) & 1 != 0;
        }

        if (exit_mode and !exists) return exitAt(t, axis, oct);
        if (exists and !(exit_mode and scale == 0)) {
            if (!exit_mode and scale == 0) {
                var v: [3]u32 = undefined;
                var voxel: [3]i32 = undefined;
                inline for (0..3) |a| {
                    v[a] = if ((oct >> a) & 1 != 0) n - 1 - cell[a] else cell[a];
                    voxel[a] = @intCast(v[a]);
                }
                var attribute = g.default_attribute;
                if (want_attribute or skip != null) {
                    if (g.attributes) |attrs| attribute = attrs[attributeRank(g, &stk, v, brick)];
                }
                // durchsichtiges Material: weiterlaufen statt treffen
                if (skipped(skip, attribute)) {
                    scale = 0;
                    // wie "Zelle leer": zur nächsten Zelle derselben Größe
                    const size0: u32 = 1;
                    var tnext0 = (@as(f32, @floatFromInt(cell[0] + size0)) - od[0]) * inv[0];
                    axis = 0;
                    inline for (1..3) |a| {
                        const ta = (@as(f32, @floatFromInt(cell[a] + size0)) - od[a]) * inv[a];
                        if (ta < tnext0) {
                            tnext0 = ta;
                            axis = a;
                        }
                    }
                    if (!(tnext0 < tend)) return null;
                    t = @max(t, tnext0);
                    moved = true;
                    var old0: u32 = 0;
                    inline for (0..3) |a| {
                        if (axis == a) old0 = cell[a];
                    }
                    const next0 = old0 + size0;
                    if (next0 >= n) return null;
                    inline for (0..3) |a| {
                        if (axis == a) cell[a] = next0;
                    }
                    const h0: u32 = 31 - @as(u32, @clz(old0 ^ next0));
                    if (h0 > scale) {
                        scale = h0;
                        const m0 = ~((@as(u32, 1) << @intCast(scale)) - 1);
                        inline for (0..3) |a| cell[a] &= m0;
                    }
                    continue;
                }
                const mirrored = (oct >> @intCast(axis)) & 1 != 0;
                return .{
                    .t = t,
                    .face = 2 * axis + @as(u32, if (mirrored) 0 else 1),
                    .voxel = voxel,
                    .attribute = attribute,
                    .inside = starts_inside and !moved,
                };
            }
            if (scale >= 3) {
                stk[scale - 1] = child;
            } else if (scale == 2) {
                brick = g.leaves[child];
            }

            // Abstieg in das Kind, das den aktuellen Punkt enthält
            scale -= 1;
            const half = @as(u32, 1) << @intCast(scale);
            const halff: f32 = @floatFromInt(half);
            inline for (0..3) |a| {
                const p = od[a] + dd[a] * t;
                if (p - @as(f32, @floatFromInt(cell[a])) >= halff) cell[a] += half;
            }
            continue;
        }

        // Zur nächsten Zelle gleicher Größe
        const size = @as(u32, 1) << @intCast(scale);
        var tnext = (@as(f32, @floatFromInt(cell[0] + size)) - od[0]) * inv[0];
        axis = 0;
        inline for (1..3) |a| {
            const ta = (@as(f32, @floatFromInt(cell[a] + size)) - od[a]) * inv[a];
            if (ta < tnext) {
                tnext = ta;
                axis = a;
            }
        }
        if (!(tnext < tend)) {
            // Würfelrand erreicht: im Austrittsmodus zählt das als Austritt
            if (exit_mode and tf <= tmax and tnext >= tf) return exitAt(tf, axis, oct);
            return null;
        }
        t = @max(t, tnext);
        moved = true;

        var old: u32 = 0;
        inline for (0..3) |a| {
            if (axis == a) old = cell[a];
        }
        const next = old + size;
        if (next >= n) return if (exit_mode) exitAt(t, axis, oct) else null;
        inline for (0..3) |a| {
            if (axis == a) cell[a] = next;
        }
        // Höchstes geänderte Bit = Ebene, bis zu der aufgestiegen wird
        const h: u32 = 31 - @as(u32, @clz(old ^ next));
        if (h > scale) {
            scale = h;
            const m = ~((@as(u32, 1) << @intCast(scale)) - 1);
            inline for (0..3) |a| cell[a] &= m;
        }
    }
    return null;
}
