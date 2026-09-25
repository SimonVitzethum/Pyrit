//! Kleine 32x32-Texturen für die Blockarten der Demo, auf dem Host erzeugt.
//!
//! Eine Kachel deckt genau einen Grundvoxel ab (`texture_scale = 1`), die
//! Projektion auf Voxelflächen ist achsenparallel – das ergibt den vertrauten
//! Blick auf würfelförmige Blöcke. Für gröbere LOD-Stufen baut Pyrit beim
//! Anlegen eine Verkleinerungskette, sonst würde die Kachel in der Ferne
//! flimmern.

const std = @import("std");

pub const size: u32 = 32;

pub const Kind = enum { grass, rock, sand, snow, wood, leaves, dirt, gravel };

fn hash(x: u32, y: u32, s: u32) u32 {
    var h: u32 = x *% 0x8da6b343 +% y *% 0xd8163841 +% s *% 0xcb1ab31f;
    h ^= h >> 15;
    h *%= 0x2c1b3c6d;
    h ^= h >> 12;
    h *%= 0x297a2d39;
    h ^= h >> 15;
    return h;
}

/// Zufallswert in [0, 1)
fn rnd(x: u32, y: u32, s: u32) f32 {
    return @as(f32, @floatFromInt(hash(x, y, s) >> 8)) * (1.0 / 16777216.0);
}

/// geglättetes Rauschen mit Kachelgrenze (wiederholt sich sauber)
fn smooth(x: u32, y: u32, s: u32, cell: u32) f32 {
    const gx = x / cell;
    const gy = y / cell;
    const fx = @as(f32, @floatFromInt(x % cell)) / @as(f32, @floatFromInt(cell));
    const fy = @as(f32, @floatFromInt(y % cell)) / @as(f32, @floatFromInt(cell));
    const n = size / cell;
    const a = rnd(gx % n, gy % n, s);
    const b = rnd((gx + 1) % n, gy % n, s);
    const c = rnd(gx % n, (gy + 1) % n, s);
    const d = rnd((gx + 1) % n, (gy + 1) % n, s);
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    const top = a + (b - a) * sx;
    const bot = c + (d - c) * sx;
    return top + (bot - top) * sy;
}

/// Schreibt eine 32x32-RGBA8-Kachel. Die Farben sind Faktoren um 1 herum: das
/// Material bringt seine Grundfarbe mit, die Textur moduliert sie nur.
pub fn make(kind: Kind, out: []u8) void {
    std.debug.assert(out.len >= size * size * 4);
    for (0..size) |yy| {
        for (0..size) |xx| {
            const x: u32 = @intCast(xx);
            const y: u32 = @intCast(yy);
            var r: f32 = 1;
            var g: f32 = 1;
            var b: f32 = 1;
            switch (kind) {
                .grass => {
                    // kurze senkrechte Halme: feines Rauschen, in y gestreckt
                    const blade = rnd(x, y / 3, 11);
                    const patch = smooth(x, y, 12, 8);
                    const v = 0.70 + blade * 0.38 + patch * 0.20;
                    r = v * 0.92;
                    g = v;
                    b = v * 0.72;
                },
                .rock => {
                    // Sprünge: dunkle Adern entlang einer groben Rauschkante
                    const n = smooth(x, y, 13, 8);
                    const fine = rnd(x, y, 14);
                    const crack: f32 = if (@abs(n - 0.5) < 0.045) 0.55 else 1.0;
                    const v = (0.82 + n * 0.26 + fine * 0.10) * crack;
                    r = v;
                    g = v * 0.99;
                    b = v * 0.97;
                },
                .sand => {
                    // feine Körnung, dazu flache Rippel in x
                    const grain = rnd(x, y, 15);
                    const ripple = @sin(@as(f32, @floatFromInt(x)) * 0.6) * 0.5 + 0.5;
                    const v = 0.88 + grain * 0.14 + ripple * 0.08;
                    r = v;
                    g = v * 0.985;
                    b = v * 0.93;
                },
                .snow => {
                    // verwehte Mulden und einzelne glitzernde Körner
                    const n = smooth(x, y, 16, 8);
                    const drift = smooth(x, y, 18, 16);
                    const spark: f32 = if (rnd(x, y, 17) > 0.985) 1.15 else 1.0;
                    const v = (0.84 + n * 0.14 + drift * 0.12) * spark;
                    r = v * 0.99;
                    g = v * 0.995;
                    b = v;
                },
                .wood => {
                    // senkrechte Rinde: Streifen in x, leicht verzogen
                    const warp = smooth(x, y, 18, 16) * 3;
                    const fx = @as(f32, @floatFromInt(x)) + warp;
                    const stripe = @sin(fx * 1.5) * 0.5 + 0.5;
                    const fine = rnd(x, y, 19);
                    const v = 0.74 + stripe * 0.34 + fine * 0.10;
                    r = v;
                    g = v * 0.92;
                    b = v * 0.84;
                },
                .dirt => {
                    // Erde: Krümel, dazwischen einzelne helle Steinchen
                    const n = smooth(x, y, 22, 8);
                    const fine = rnd(x, y, 23);
                    const pebble: f32 = if (rnd(x / 2, y / 2, 24) > 0.94) 1.2 else 1.0;
                    const v = (0.80 + n * 0.20 + fine * 0.14) * pebble;
                    r = v;
                    g = v * 0.97;
                    b = v * 0.93;
                },
                .gravel => {
                    // Kies: grobe, deutlich verschieden helle Körner
                    const stone = rnd(x / 3, y / 3, 25);
                    const edge: f32 = if (x % 3 == 0 or y % 3 == 0) 0.82 else 1.0;
                    const v = (0.70 + stone * 0.45) * edge;
                    r = v;
                    g = v * 0.98;
                    b = v * 0.96;
                },
                .leaves => {
                    // Büschel: grobe Flecken, dazwischen dunkle Lücken
                    const clump = smooth(x, y, 20, 8);
                    const fine = rnd(x, y, 21);
                    const gap: f32 = if (clump < 0.30) 0.68 else 1.0;
                    const v = (0.80 + clump * 0.30 + fine * 0.12) * gap;
                    r = v * 0.86;
                    g = v;
                    b = v * 0.70;
                },
            }
            // Die Faktoren liegen um 1 und reichen bis gut 1,2. Ohne diese
            // Skalierung wurde alles über 1 bei 255 abgeschnitten – genau die
            // helle Hälfte der Zeichnung ging verloren, Gras wirkte flach.
            const k: f32 = 0.82;
            const o = (yy * size + xx) * 4;
            out[o + 0] = @intFromFloat(@min(@max(r * k * 255, 0), 255));
            out[o + 1] = @intFromFloat(@min(@max(g * k * 255, 0), 255));
            out[o + 2] = @intFromFloat(@min(@max(b * k * 255, 0), 255));
            out[o + 3] = 255;
        }
    }
}

/// Normalentextur zu einer Kachel: die Helligkeit gilt als Höhe (hell =
/// erhaben), ihre Steigung wird in Rot (u) und Grün (v) um 128 abgelegt.
/// Fugen im Stein, Halme im Gras und Rinde fangen damit das Streiflicht.
pub fn normalMap(tile: []const u8, out: []u8, strength: f32) void {
    const n = size;
    for (0..n) |yy| {
        for (0..n) |xx| {
            const h = struct {
                fn at(t: []const u8, x: usize, y: usize) f32 {
                    const o = ((y % size) * size + (x % size)) * 4;
                    return (0.2126 * @as(f32, @floatFromInt(t[o])) + 0.7152 * @as(f32, @floatFromInt(t[o + 1])) + 0.0722 * @as(f32, @floatFromInt(t[o + 2]))) / 255.0;
                }
            }.at;
            const du = (h(tile, xx + 1, yy) - h(tile, xx + n - 1, yy)) * strength;
            const dv = (h(tile, xx, yy + 1) - h(tile, xx, yy + n - 1)) * strength;
            const o = (yy * n + xx) * 4;
            out[o + 0] = @intFromFloat(@min(@max(128 + du * 127, 0), 255));
            out[o + 1] = @intFromFloat(@min(@max(128 + dv * 127, 0), 255));
            out[o + 2] = 255;
            out[o + 3] = 255;
        }
    }
}

/// Seite eines Grasblocks: Erde, oben ein ausgefranster Grasrand. Anders als
/// die übrigen Kacheln trägt sie ihre Farbe selbst (Material.side_color =
/// 1,1,1), als lineares Albedo: Erde um 0,17/0,10/0,05, Gras um 0,05/0,10/0,02.
/// Zeile size-1 liegt oben am Block.
pub fn grassSide(out: []u8) void {
    std.debug.assert(out.len >= size * size * 4);
    for (0..size) |yy| {
        for (0..size) |xx| {
            const x: u32 = @intCast(xx);
            const y: u32 = @intCast(yy);
            // Tiefe des Grasrands je Spalte: 3 bis 7 Zeilen, ausgefranst
            const fringe: u32 = 3 + @as(u32, @intFromFloat(rnd(x, 0, 31) * 3.0)) + @as(u32, @intFromFloat(smooth(x, 0, 32, 8) * 2.0));
            const from_top = size - 1 - y;
            const n = smooth(x, y, 33, 8);
            const fine = rnd(x, y, 34);
            var r: f32 = undefined;
            var g: f32 = undefined;
            var b: f32 = undefined;
            if (from_top < fringe) {
                const v = 0.8 + fine * 0.35 + n * 0.15;
                r = 0.048 * v;
                g = 0.098 * v;
                b = 0.022 * v;
            } else {
                const pebble: f32 = if (rnd(x / 2, y / 2, 35) > 0.93) 1.35 else 1.0;
                const v = (0.78 + n * 0.3 + fine * 0.2) * pebble;
                r = 0.17 * v;
                g = 0.10 * v;
                b = 0.052 * v;
            }
            const o = (yy * size + xx) * 4;
            out[o + 0] = @intFromFloat(@min(@max(r * 255, 0), 255));
            out[o + 1] = @intFromFloat(@min(@max(g * 255, 0), 255));
            out[o + 2] = @intFromFloat(@min(@max(b * 255, 0), 255));
            out[o + 3] = 255;
        }
    }
}
