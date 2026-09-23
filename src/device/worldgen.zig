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
///
/// Aufbau: erst wird die Abtastposition von einem groben Rauschen verschoben
/// (Domain-Warping) – das biegt die Grate und nimmt der Landschaft das
/// Gitterhafte. Darauf ein gratartiges Multifraktal: jede Oktave wird mit dem
/// Ergebnis der vorigen gewichtet, wodurch sich feine Grate auf den
/// Höhenzügen sammeln statt gleichmäßig über die Ebene zu streuen. Genau das
/// unterscheidet ein Gebirge von verrauschten Hügeln.
pub fn height(t: *const types.TerrainParams, x: f32, z: f32, min_wavelength: f32) f32 {
    // 1. Domain-Warping mit der doppelten Grundwellenlänge
    const wwl = t.wavelength * 2;
    var wx = x;
    var wz = z;
    if (min_wavelength < wwl) {
        const w1 = valueNoise(x / wwl + 11.3, z / wwl - 4.7, t.seed ^ 0x2f1d) - 0.5;
        const w2 = valueNoise(x / wwl - 8.1, z / wwl + 15.9, t.seed ^ 0x7c3b) - 0.5;
        const amount = t.wavelength * 0.75;
        wx += w1 * amount;
        wz += w2 * amount;
    }

    // 2. Gratartiges Multifraktal
    var wl = t.wavelength;
    var amp: f32 = 1;
    var weight: f32 = 1;
    var sum: f32 = 0;
    var norm: f32 = 0;
    var o: u32 = 0;
    while (o < t.octaves) : (o += 1) {
        if (wl < min_wavelength) break;
        // jede Oktave gedreht: sonst fallen die Kerben des Rauschens auf
        // dieselben Gitterachsen und die Landschaft bekommt ein Karomuster
        const ca = fm.cos(0.7 * @as(f32, @floatFromInt(o)));
        const sa = fm.sin(0.7 * @as(f32, @floatFromInt(o)));
        const rx = (wx * ca - wz * sa) / wl + @as(f32, @floatFromInt(o)) * 37.0;
        const rz = (wx * sa + wz * ca) / wl - @as(f32, @floatFromInt(o)) * 19.0;
        const n = valueNoise(rx, rz, t.seed +% o *% 0x9e3779b9);
        // Grat: Spitze dort, wo das Rauschen die Mitte kreuzt
        var r = 1 - @abs(2 * n - 1);
        r *= r; // schärfer
        // Die feinen Oktaven bekommen mehr Gewicht als das reine Halbieren,
        // sonst liegt die Oberfläche als glatte Terrasse da und die Stufen
        // lesen sich wie Höhenlinien.
        const v = r * weight;
        // Die nächste Oktave zählt nur dort voll, wo diese schon hoch liegt
        weight = @min(@max(r * 1.8, 0), 1);
        sum += v * amp;
        norm += amp;
        amp *= 0.52;
        wl *= 0.5;
    }
    // Restliche Amplitude als Erwartungswert, damit die Höhe über die
    // LOD-Stufen stabil bleibt (der Grat hat im Mittel etwa 1/3)
    while (o < t.octaves) : (o += 1) {
        sum += 0.333 * weight * amp;
        norm += amp;
        amp *= 0.5;
    }
    const h = sum / @max(norm, 1e-6);
    // Täler flacher, Gipfel steiler
    return t.base_height + t.amplitude * h * h * 1.6;
}

/// Höhe, ab der Schnee liegt – keine Linie, sondern ein von Rauschen und
/// Hangneigung aufgelöster Übergang. Eine feste Grenze sieht an einem Berg
/// sofort künstlich aus.
fn snowLine(t: *const types.TerrainParams, x: f32, z: f32) f32 {
    const band = @max(t.amplitude * 0.12, 24);
    const n = valueNoise(x / (t.wavelength * 0.22), z / (t.wavelength * 0.22), t.seed ^ 0x51a7);
    const n2 = valueNoise(x / (t.wavelength * 0.05) + 5.5, z / (t.wavelength * 0.05) - 3.3, t.seed ^ 0x9e12);
    return t.snow_height + (n - 0.5) * band * 2 + (n2 - 0.5) * band * 0.6;
}

fn pick(a: u32, default: u32) u32 {
    return if (a != 0) a else default;
}

fn rgb(r: u32, g: u32, b: u32) u32 {
    return (r << 24) | (g << 16) | (b << 8);
}

/// Farbanteil eines Attributs skalieren, Materialindex behalten.
/// `warm` verschiebt zusätzlich ins Gelbliche (trockene Stellen).
fn tint(a: u32, f: f32, warm: f32) u32 {
    const mat = a & 0xFF;
    const r0: f32 = @floatFromInt((a >> 24) & 0xFF);
    const g0: f32 = @floatFromInt((a >> 16) & 0xFF);
    const b0: f32 = @floatFromInt((a >> 8) & 0xFF);
    const r: u32 = @intFromFloat(@min(@max(r0 * (f + warm * 0.35), 0), 255));
    const g: u32 = @intFromFloat(@min(@max(g0 * (f + warm * 0.12), 0), 255));
    const b: u32 = @intFromFloat(@min(@max(b0 * (f - warm * 0.25), 0), 255));
    return (r << 24) | (g << 16) | (b << 8) | mat;
}

/// Farbschwankung des Bodens: eine grobe Lage für große Flecken, eine feine
/// für die Sprenkelung. Ohne sie ist die ganze Landschaft exakt eine Farbe
/// und sieht wie ein Teppich aus.
fn groundVariation(t: *const types.TerrainParams, x: f32, z: f32) [2]f32 {
    const coarse = valueNoise(x / 260 + 3.1, z / 260 - 7.4, t.seed ^ 0x3c19);
    const fine = valueNoise(x / 21 - 1.7, z / 21 + 9.2, t.seed ^ 0xb72d);
    const f = 0.62 + coarse * 0.62 + fine * 0.26;
    // trockene Flecken dort, wo die grobe Lage hoch liegt
    const warm = @max(coarse - 0.45, 0) * 2.4;
    return .{ f, warm };
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

    if (top_f < 0 or bot_f >= @as(f32, @floatFromInt(n))) return;
    // Zelle der Wasseroberfläche in diesem Chunk (-1 = keine). Sie gehört dem
    // Wasser; das Gelände endet darunter. Beides in dieselbe Zelle zu legen
    // hieße, dem Bau die Wahl zu lassen – auf der Wasserfläche standen dann
    // einzelne Sandwürfel verstreut herum.
    var water_y: i32 = -1;
    if (t.attr_water != 0 and h < t.sea_level) {
        const wtop_f = (t.sea_level - y0w) / step - 0.5;
        if (wtop_f >= 0 and wtop_f < @as(f32, @floatFromInt(n))) water_y = @intFromFloat(@floor(wtop_f));
    }
    var top: i32 = @min(@as(i32, @intFromFloat(@floor(top_f))), ni - 1);
    if (water_y >= 0 and top >= water_y) top = water_y - 1;
    const bot: i32 = @max(@as(i32, @intFromFloat(@floor(bot_f))), 0);
    if (bot > top) return;

    const snow_h = snowLine(t, wx, wz);
    const gv = groundVariation(t, wx, wz);
    const var_f = gv[0];
    const var_warm = gv[1];
    const surface_y = @floor((h - y0w) / step - 0.5); // lokale Höhe der obersten Zelle
    var y = top;
    while (y >= bot) : (y -= 1) {
        const yf: f32 = @floatFromInt(y);
        const depth = (surface_y - yf) * step; // Grundvoxel unter der Oberfläche
        const wy = y0w + (yf + 0.5) * step;
        const a: u32 = blk: {
            if (slope > t.rock_slope) break :blk tint(pick(t.attr_rock, rgb(118, 112, 106)), var_f * 0.95 + 0.1, 0);
            // Schnee: über der aufgelösten Grenze, und je steiler der Hang,
            // desto höher muss es dafür sein – auf einer Felswand hält er nicht.
            if (wy > snow_h + slope * t.amplitude * 0.05) break :blk tint(pick(t.attr_snow, rgb(236, 240, 245)), 0.95 + var_f * 0.08, 0);
            if (wy < t.sea_level + 2) break :blk tint(pick(t.attr_sand, rgb(214, 196, 142)), var_f * 0.9 + 0.15, 0);
            if (depth < 1.0) break :blk tint(pick(t.attr_grass, rgb(84, 140, 58)), var_f, var_warm);
            if (depth < 4.0) break :blk tint(pick(t.attr_dirt, rgb(122, 92, 62)), var_f, 0);
            break :blk tint(pick(t.attr_rock, rgb(118, 112, 106)), var_f * 0.95 + 0.1, 0);
        };
        emit(g, c, base, counts, out, lx, y, lz, a);
    }

    // Nur die Oberfläche: das Medium reicht ohnehin bis zum Grund (die
    // Absorption rechnet mit dem Weg bis zum Untergrund), und eine dicke Haut
    // würde an den Rändern ihre Seitenflächen zeigen.
    if (water_y >= 0) emit(g, c, base, counts, out, lx, water_y, lz, t.attr_water);

    // Bäume. Die Maße stehen in Grundvoxeln und werden auf die Voxelgröße der
    // Stufe umgerechnet – so stehen sie auf *jeder* Stufe, nur eben gröber.
    // (Vorher gab es sie nur auf Stufe 0 und 1, also nur nahe der Kamera:
    // beim Näherkommen wuchs plötzlich ein Wald aus dem Nichts.)
    if (t.attr_leaves == 0 or slope > t.rock_slope) return;
    const wy_top = y0w + (surface_y + 0.5) * step;
    if (wy_top < t.sea_level + 1 or wy_top > snow_h) return;
    const r = lattice(@intFromFloat(wx), @intFromFloat(wz), t.seed ^ 0x51ed);
    // Eine Spalte deckt step² Grundspalten ab; so viele Bäume stünden darin.
    const expect = t.tree_density * step * step;

    // Ist ein Baum kleiner als eine Zelle, kann man ihn nicht mehr setzen –
    // jede Spalte zu bewalden würde den Generatorpuffer sprengen. Stattdessen
    // färbt sich der Boden zum Laub hin, anteilig zur Kronendeckung. Aus der
    // Ferne ist genau das der richtige Anblick: eine Waldfläche, keine
    // einzelnen Bäume. Vorher verschwand der Wald ab Stufe 3 einfach.
    if (step > 4) {
        const cover = @min(expect * 7, 1.0); // eine Krone deckt ~7 Grundspalten
        if (cover <= 0.02) return;
        // Eine Zelle ÜBER dem Boden: in dieselbe zu schreiben hieße, sich mit
        // dem Geländevoxel um die Zelle zu streiten (derselbe Fehler, der beim
        // Wasser einzelne Würfel auf die Oberfläche gestreut hat).
        const ly: i32 = @as(i32, @intFromFloat(surface_y)) + 1;
        if (ly < 0 or ly >= ni) return;
        const canopy = tint(t.attr_leaves, 0.75 + r * 0.5, 0);
        // gemischt: je dichter der Wald, desto eher gewinnt das Laub
        if (r < cover) emit(g, c, base, counts, out, lx, ly, lz, canopy);
        return;
    }
    if (r > @min(expect, 1.0)) return;

    const trunk = pick(t.attr_wood, rgb(96, 68, 44));
    // Zweite Zufallszahl je Baum: Art, Größe und Grünton
    const r2 = lattice(@intFromFloat(wx * 1.7), @intFromFloat(wz * 1.3), t.seed ^ 0x2ab1);
    const conifer = r2 > 0.55; // Nadelbaum: schmal und spitz, sonst rundkronig
    const leaf = tint(t.attr_leaves, 0.70 + r2 * 0.55, if (conifer) 0 else 0.10);

    // Maße in Grundvoxeln, dann auf die Voxelgröße der Stufe umgerechnet
    const trunk_base: f32 = if (conifer) 7 + r * 260 else 4 + r * 160;
    const crown_base: f32 = if (conifer) 2.2 + r2 * 0.8 else 2.6 + r2 * 1.2;
    const trunk_h: i32 = @max(@as(i32, @intFromFloat(@round(trunk_base / step))), 1);
    const crown_r: i32 = @max(@as(i32, @intFromFloat(@round(crown_base / step))), 1);

    var ty: i32 = 1;
    while (ty <= trunk_h) : (ty += 1) {
        const vy = @as(i32, @intFromFloat(surface_y)) + ty;
        if (vy >= 0 and vy < ni) emit(g, c, base, counts, out, lx, vy, lz, trunk);
    }

    // Krone: Nadelbaum kegelförmig von unten nach oben schmaler, Laubbaum
    // kugelig. Beide mit leicht unregelmäßigem Rand, sonst sieht man die Form.
    const top_y = @as(i32, @intFromFloat(surface_y)) + trunk_h;
    const h_lo: i32 = if (conifer) 0 - @max(@divTrunc(trunk_h * 2, 3), @as(i32, 1)) else 0 - @divTrunc(crown_r, 2);
    const h_hi: i32 = if (conifer) @max(@divTrunc(crown_r * 3, 2), @as(i32, 1)) else crown_r;
    var dy: i32 = h_lo;
    while (dy <= h_hi) : (dy += 1) {
        // Radius auf dieser Höhe
        var rad: f32 = undefined;
        if (conifer) {
            const f = @as(f32, @floatFromInt(dy - h_lo)) / @as(f32, @floatFromInt(@max(h_hi - h_lo, 1)));
            rad = @as(f32, @floatFromInt(crown_r)) * (1 - f) + 0.3;
        } else {
            const f = @as(f32, @floatFromInt(dy)) / @as(f32, @floatFromInt(@max(crown_r, 1)));
            rad = @as(f32, @floatFromInt(crown_r)) * @sqrt(@max(1 - f * f * 0.75, 0));
        }
        const ri: i32 = @intFromFloat(@floor(rad));
        var dz: i32 = -ri;
        while (dz <= ri) : (dz += 1) {
            var dx: i32 = -ri;
            while (dx <= ri) : (dx += 1) {
                const d2 = @as(f32, @floatFromInt(dx * dx + dz * dz));
                // unregelmäßiger Rand
                const jitter = lattice(@as(i32, @intFromFloat(wx)) + dx * 7, @as(i32, @intFromFloat(wz)) + dz * 11 + dy * 3, t.seed ^ 0x77b5);
                if (d2 > rad * rad * (0.75 + jitter * 0.5)) continue;
                const vx = @as(i32, @intCast(lx)) + dx;
                const vz = @as(i32, @intCast(lz)) + dz;
                const vy = top_y + dy;
                if (vx < 0 or vz < 0 or vx >= ni or vz >= ni or vy < 0 or vy >= ni) continue;
                emit(g, c, base, counts, out, @intCast(vx), vy, @intCast(vz), leaf);
            }
        }
    }
}
