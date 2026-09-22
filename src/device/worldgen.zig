//! Eingebauter Geländegenerator für das Welt-Streaming. Die Höhe hängt nur von
//! der Weltposition ab, daher passen alle LOD-Stufen zusammen; jede Stufe
//! tastet in ihrem eigenen Voxelabstand ab, so dass nie mehr Voxel entstehen
//! als die Stufe darstellen kann.

const std = @import("std");
const types = @import("types.zig");
const fm = @import("fmath.zig");

inline fn hash(x: i32, z: i32, seed: u32) u32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 0x8da6b343 +% @as(u32, @bitCast(z)) *% 0xd8163841 +% seed *% 0xcb1ab31f;
    h ^= h >> 15;
    h *%= 0x2c1b3c6d;
    h ^= h >> 12;
    h *%= 0x297a2d39;
    h ^= h >> 15;
    return h;
}

inline fn lattice(x: i32, z: i32, seed: u32) f32 {
    return @as(f32, @floatFromInt(hash(x, z, seed) >> 8)) * (1.0 / 16777216.0);
}

/// Wertrauschen in [0, 1] mit C1-stetiger Interpolation
fn valueNoise(x: f32, z: f32, seed: u32) f32 {
    const fx = @floor(x);
    const fz = @floor(z);
    const ix: i32 = @intFromFloat(fx);
    const iz: i32 = @intFromFloat(fz);
    const tx = x - fx;
    const tz = z - fz;
    const sx = tx * tx * (3 - 2 * tx);
    const sz = tz * tz * (3 - 2 * tz);
    const a = lattice(ix, iz, seed);
    const b = lattice(ix + 1, iz, seed);
    const c = lattice(ix, iz + 1, seed);
    const d = lattice(ix + 1, iz + 1, seed);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz;
}

/// Geländehöhe in Grundvoxeln an (x, z). `min_wavelength`: Oktaven, deren
/// Wellenlänge darunter liegt, trägt die Stufe nicht mehr (kein Aliasing).
pub fn height(t: *const types.TerrainParams, x: f32, z: f32, min_wavelength: f32) f32 {
    var wl = t.wavelength;
    var amp: f32 = 1;
    var sum: f32 = 0;
    var norm: f32 = 0;
    var o: u32 = 0;
    while (o < t.octaves) : (o += 1) {
        if (wl < min_wavelength) break;
        // jede Oktave gedreht: sonst fallen die Kerben des Rauschens auf
        // dieselben Gitterachsen und die Landschaft bekommt ein Karomuster
        const ca = fm.cos(0.7 * @as(f32, @floatFromInt(o)));
        const sa = fm.sin(0.7 * @as(f32, @floatFromInt(o)));
        const rx = (x * ca - z * sa) / wl + @as(f32, @floatFromInt(o)) * 37.0;
        const rz = (x * sa + z * ca) / wl - @as(f32, @floatFromInt(o)) * 19.0;
        const n = valueNoise(rx, rz, t.seed +% o *% 0x9e3779b9);
        // leicht gratartig in den oberen Oktaven
        const v = if (o == 0) n else 1 - @abs(2 * n - 1);
        sum += v * amp;
        norm += amp;
        amp *= 0.5;
        wl *= 0.5;
    }
    // restliche Amplitude als Mittelwert, damit die Höhe über die Stufen stabil bleibt
    while (o < t.octaves) : (o += 1) {
        sum += 0.5 * amp;
        norm += amp;
        amp *= 0.5;
    }
    const h = sum / @max(norm, 1e-6);
    // Täler flacher, Gipfel steiler
    return t.base_height + t.amplitude * h * h * 1.6;
}

fn pick(a: u32, default: u32) u32 {
    return if (a != 0) a else default;
}

fn rgb(r: u32, g: u32, b: u32) u32 {
    return (r << 24) | (g << 16) | (b << 8);
}

/// Ein Thread je Spalte (Chunk c, x, z)
pub fn terrainColumn(g: *const types.WorldGenParams, t: *const types.TerrainParams, i: u32) void {
    const cl: u5 = @intCast(g.chunk_log2);
    const n: u32 = @as(u32, 1) << cl;
    const c = i >> (2 * cl);
    if (c >= g.count) return;
    const col = i & ((n * n) - 1);
    const lx = col & (n - 1);
    const lz = col >> cl;
    const key = @as([*]const types.ChunkKey, @ptrFromInt(g.chunks))[c];
    const step_i: i32 = @as(i32, 1) << @intCast(key.lod);
    const step: f32 = @floatFromInt(step_i);
    const ni: i32 = @intCast(n);

    // Weltposition der Spaltenmitte (Grundvoxel)
    const wx = @as(f32, @floatFromInt((key.x * ni + @as(i32, @intCast(lx))) * step_i)) + 0.5 * step;
    const wz = @as(f32, @floatFromInt((key.z * ni + @as(i32, @intCast(lz))) * step_i)) + 0.5 * step;
    const min_wl = 2 * step;
    const h = height(t, wx, wz, min_wl);
    const hx0 = height(t, wx - step, wz, min_wl);
    const hx1 = height(t, wx + step, wz, min_wl);
    const hz0 = height(t, wx, wz - step, min_wl);
    const hz1 = height(t, wx, wz + step, min_wl);
    // Haut: bis zur niedrigsten Nachbarhöhe hinab, damit keine Lücken an Hängen entstehen
    // etwas tiefer als der niedrigste Nachbar: an Chunk-Rändern treffen
    // unterschiedliche Auflösungen aufeinander, sonst bleiben dort Nähte
    const low = @min(@min(hx0, hx1), @min(hz0, hz1)) - 2.5 * step;
    const slope = @max(@max(@abs(hx1 - hx0), @abs(hz1 - hz0)) / (2 * step), 0);

    const y0w = @as(f32, @floatFromInt(key.y * ni * step_i));
    // Zellen [y0w + ly*step, ...) mit Mitte <= h sind fest
    const top_f = (h - y0w) / step - 0.5;
    const bot_f = (low - y0w) / step - 0.5;
    const counts: [*]u32 = @ptrFromInt(g.counts);
    const out: [*][4]u32 = @ptrFromInt(g.voxels);
    const base = @as(u64, c) * g.capacity;
    const emit = struct {
        fn f(gg: *const types.WorldGenParams, cc: u32, bb: u64, cnt: [*]u32, o: [*][4]u32, vx: u32, vy: i32, vz: u32, attr: u32) void {
            const k = @atomicRmw(u32, &cnt[cc], .Add, 1, .monotonic);
            if (k < gg.capacity) o[bb + k] = .{ vx, @intCast(vy), vz, attr };
        }
    }.f;

    // Wasser bis zum Meeresspiegel (transparentes Material, gleiche Geometrie)
    if (t.attr_water != 0 and h < t.sea_level) {
        const wtop_f = (t.sea_level - y0w) / step - 0.5;
        // nur der Chunk, in dem die Wasseroberfläche liegt, bekommt Wasser
        if (wtop_f >= 0 and wtop_f < @as(f32, @floatFromInt(n))) {
            // Nur die Oberfläche: das Medium reicht ohnehin bis zum Grund
            // (die Absorption rechnet mit dem Weg bis zum Untergrund), und eine
            // dicke Haut würde an den Rändern ihre Seitenflächen zeigen.
            const wy_i: i32 = @intFromFloat(@floor(wtop_f));
            if (wy_i >= 0 and wy_i < ni) emit(g, c, base, counts, out, lx, wy_i, lz, t.attr_water);
        }
    }

    if (top_f < 0 or bot_f >= @as(f32, @floatFromInt(n))) return;
    const top: i32 = @min(@as(i32, @intFromFloat(@floor(top_f))), ni - 1);
    const bot: i32 = @max(@as(i32, @intFromFloat(@floor(bot_f))), 0);
    if (bot > top) return;

    const surface_y = @floor((h - y0w) / step - 0.5); // lokale Höhe der obersten Zelle
    var y = top;
    while (y >= bot) : (y -= 1) {
        const yf: f32 = @floatFromInt(y);
        const depth = (surface_y - yf) * step; // Grundvoxel unter der Oberfläche
        const wy = y0w + (yf + 0.5) * step;
        const a: u32 = blk: {
            if (slope > t.rock_slope) break :blk pick(t.attr_rock, rgb(118, 112, 106));
            if (wy > t.snow_height) break :blk pick(t.attr_snow, rgb(236, 240, 245));
            if (wy < t.sea_level + 2) break :blk pick(t.attr_sand, rgb(214, 196, 142));
            if (depth < 1.0) break :blk pick(t.attr_grass, rgb(84, 140, 58));
            if (depth < 4.0) break :blk pick(t.attr_dirt, rgb(122, 92, 62));
            break :blk pick(t.attr_rock, rgb(118, 112, 106));
        };
        emit(g, c, base, counts, out, lx, y, lz, a);
    }

    // Vegetation: Grasbüschel und Bäume auf der obersten Zelle (nur in feiner
    // Auflösung sichtbar, darüber zu klein)
    if (t.attr_leaves == 0 or key.lod > 1 or slope > t.rock_slope) return;
    const wy_top = y0w + (surface_y + 0.5) * step;
    if (wy_top < t.sea_level + 1 or wy_top > t.snow_height) return;
    const r = lattice(@intFromFloat(wx), @intFromFloat(wz), t.seed ^ 0x51ed);
    if (r > t.tree_density) return;
    const trunk = pick(t.attr_wood, rgb(96, 68, 44));
    const leaf = t.attr_leaves;
    const trunk_h: i32 = 4 + @mod(@as(i32, @intFromFloat(r * 400)), 3);
    var ty: i32 = 1;
    while (ty <= trunk_h) : (ty += 1) {
        const vy = @as(i32, @intFromFloat(surface_y)) + ty;
        if (vy >= 0 and vy < ni) emit(g, c, base, counts, out, lx, vy, lz, trunk);
    }
    // Krone: Kugelschale um die Stammspitze, innerhalb des Chunks
    var dz: i32 = -2;
    while (dz <= 2) : (dz += 1) {
        var dx: i32 = -2;
        while (dx <= 2) : (dx += 1) {
            var dy: i32 = -1;
            while (dy <= 2) : (dy += 1) {
                if (dx * dx + dy * dy + dz * dz > 5) continue;
                const vx = @as(i32, @intCast(lx)) + dx;
                const vz = @as(i32, @intCast(lz)) + dz;
                const vy = @as(i32, @intFromFloat(surface_y)) + trunk_h + dy;
                if (vx < 0 or vz < 0 or vx >= ni or vz >= ni or vy < 0 or vy >= ni) continue;
                emit(g, c, base, counts, out, @intCast(vx), vy, @intCast(vz), leaf);
            }
        }
    }
}
