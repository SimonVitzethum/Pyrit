//! Himmel der Demo: Einfachstreuung in der Erdatmosphäre (Rayleigh für das
//! Blau, Mie für den hellen Hof um die Sonne und den Dunst am Horizont),
//! einmal auf dem Host in eine Umgebungskarte gerechnet.
//!
//! Die Sonne selbst ist nicht in der Karte – sie bleibt ein eigenes Licht.
//! Ihre Farbe ergibt sich aus der Durchlässigkeit der Atmosphäre entlang des
//! Sonnenstrahls: je tiefer sie steht, desto wärmer.

const std = @import("std");

const earth_r: f32 = 6360e3;
const atmo_r: f32 = 6420e3;
/// Streukoeffizienten auf Meereshöhe (1/m) und Skalenhöhen (m)
const beta_r = [3]f32{ 5.8e-6, 13.5e-6, 33.1e-6 };
/// Aerosol: 21e-6 wäre ein diesiger Tag (grauer Himmel), 8e-6 ist klar
const beta_m: f32 = 8e-6;
/// Einfachstreuung unterschätzt die Helligkeit des Himmels; die mehrfach
/// gestreute Hälfte fehlt. Als Näherung wird sie pauschal aufgeschlagen.
const multiple_scattering: f32 = 1.9;
const h_r: f32 = 8000;
const h_m: f32 = 1200;
const mie_g: f32 = 0.76;

const V = [3]f32;

fn dot(a: V, b: V) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Strecke vom Punkt o (Abstand vom Erdmittelpunkt) in Richtung d bis zum
/// Rand der Atmosphäre
fn toTop(o: V, d: V) f32 {
    const b = dot(o, d);
    const c = dot(o, o) - atmo_r * atmo_r;
    return -b + @sqrt(@max(b * b - c, 0));
}

/// Trifft der Strahl die Erde?
fn hitsGround(o: V, d: V) bool {
    const b = dot(o, d);
    const c = dot(o, o) - earth_r * earth_r;
    return b < 0 and b * b - c > 0;
}

/// Optische Tiefe (Rayleigh, Mie) von o in Richtung d bis zum Rand
fn opticalDepth(o: V, d: V, steps: u32) [2]f32 {
    const len = toTop(o, d);
    const ds = len / @as(f32, @floatFromInt(steps));
    var r: f32 = 0;
    var m: f32 = 0;
    for (0..steps) |i| {
        const t = (@as(f32, @floatFromInt(i)) + 0.5) * ds;
        const p = V{ o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t };
        const hgt = @sqrt(dot(p, p)) - earth_r;
        r += @exp(-hgt / h_r) * ds;
        m += @exp(-hgt / h_m) * ds;
    }
    return .{ r, m };
}

fn transmittance(depth: [2]f32) V {
    var t: V = undefined;
    for (0..3) |k| t[k] = @exp(-(beta_r[k] * depth[0] + beta_m * 1.1 * depth[1]));
    return t;
}

// ---------------------------------------------------------------------------
// Wolken: eine Schicht in fester Höhe, aus Rauschen. Keine Volumen – aus der
// Entfernung einer Landschaft reicht eine beleuchtete Decke, und sie macht den
// leeren Verlauf erst zu einem Himmel.
// ---------------------------------------------------------------------------

fn hash(x: i32, y: i32, seed: u32) f32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 0x8da6b343 +% @as(u32, @bitCast(y)) *% 0xd8163841 +% seed *% 0xcb1ab31f;
    h ^= h >> 15;
    h *%= 0x2c1b3c6d;
    h ^= h >> 12;
    return @as(f32, @floatFromInt(h >> 8)) * (1.0 / 16777216.0);
}

fn noise(x: f32, y: f32, seed: u32) f32 {
    const fx = @floor(x);
    const fy = @floor(y);
    const ix: i32 = @intFromFloat(fx);
    const iy: i32 = @intFromFloat(fy);
    const tx = x - fx;
    const ty = y - fy;
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const a = hash(ix, iy, seed);
    const b = hash(ix + 1, iy, seed);
    const c = hash(ix, iy + 1, seed);
    const d = hash(ix + 1, iy + 1, seed);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}

fn fbm(x: f32, y: f32, seed: u32) f32 {
    var sum: f32 = 0;
    var amp: f32 = 0.5;
    var f: f32 = 1;
    for (0..6) |o| {
        sum += amp * noise(x * f + @as(f32, @floatFromInt(o)) * 7.1, y * f, seed +% @as(u32, @intCast(o)));
        amp *= 0.5;
        f *= 2.03;
    }
    return sum;
}

/// Deckung (0..1) der Wolkenschicht in Richtung d (y oben)
fn cloudCover(d: V) f32 {
    if (d[1] <= 0.01) return 0;
    const height: f32 = 1500;
    const t = height / d[1];
    // Koordinaten auf der Schicht in Kilometern
    const x = d[0] * t / 1000;
    const z = d[2] * t / 1000;
    // Grundform grob, Rand mit feinerem Rauschen ausgefranst
    const base = fbm(x * 0.35, z * 0.35, 17);
    const detail = fbm(x * 2.1 + 5.3, z * 2.1 - 1.7, 29);
    const n = base + (detail - 0.5) * 0.22;
    // Schönwetter-Cumulus: etwa ein Drittel Deckung, deutliche Ränder
    const c = @min(@max((n - 0.53) * 7, 0), 1);
    // Zum Horizont hin verschwimmen die Wolken im Dunst
    const fade = @min(@max((d[1] - 0.02) / 0.12, 0), 1);
    return c * c * (3 - 2 * c) * fade;
}

pub const Sky = struct {
    /// RGBA-Karte (equirektangulär, w x h)
    env: []f32,
    /// Beleuchtungsstärke der Sonne am Boden, schon durch die Atmosphäre
    sun_color: V,
};

/// `sun_dir`: Richtung zur Sonne (y oben). `sun_illuminance`: Helligkeit der
/// Sonne am Boden (Luminanz); die Karte wird im selben Maßstab gerechnet.
pub fn build(gpa: std.mem.Allocator, w: u32, h: u32, sun_dir: V, sun_illuminance: f32) !Sky {
    const env = try gpa.alloc(f32, @as(usize, w) * h * 4);
    const eye = V{ 0, earth_r + 100, 0 };
    const sl = @sqrt(dot(sun_dir, sun_dir));
    const sd = V{ sun_dir[0] / sl, sun_dir[1] / sl, sun_dir[2] / sl };

    // Sonne am Boden: Durchlässigkeit entlang ihres Strahls. Der Maßstab e0
    // (außerhalb der Atmosphäre) folgt aus der gewünschten Helligkeit unten.
    const t_sun = transmittance(opticalDepth(eye, sd, 32));
    const lum_t = 0.2126 * t_sun[0] + 0.7152 * t_sun[1] + 0.0722 * t_sun[2];
    const e0 = sun_illuminance / @max(lum_t, 1e-4);
    const sun_color = V{ e0 * t_sun[0], e0 * t_sun[1], e0 * t_sun[2] };

    const pi = std.math.pi;
    for (0..h) |y| {
        const theta = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(h)) * pi;
        for (0..w) |x| {
            const phi = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, @floatFromInt(w)) * 2 * pi;
            // dieselbe Abbildung wie envRadiance: phi = atan2(z, x), theta = acos(y)
            const d = V{ @sin(theta) * @cos(phi), @cos(theta), @sin(theta) * @sin(phi) };
            var c = V{ 0, 0, 0 };
            // Unter dem Horizont: Boden aus der Ferne (dunkel, leicht bläulich
            // vom Dunst davor). Die Landschaft deckt das ohnehin fast immer.
            const dd = if (d[1] < 0.0) V{ d[0], 0.0, d[2] } else d;
            const len = toTop(eye, dd);
            const steps: u32 = 24;
            const ds = len / @as(f32, @floatFromInt(steps));
            var dr: f32 = 0;
            var dm: f32 = 0;
            var sum_r = V{ 0, 0, 0 };
            var sum_m = V{ 0, 0, 0 };
            for (0..steps) |i| {
                const t = (@as(f32, @floatFromInt(i)) + 0.5) * ds;
                const p = V{ eye[0] + dd[0] * t, eye[1] + dd[1] * t, eye[2] + dd[2] * t };
                const hgt = @sqrt(dot(p, p)) - earth_r;
                const rr = @exp(-hgt / h_r) * ds;
                const mm = @exp(-hgt / h_m) * ds;
                dr += rr;
                dm += mm;
                if (hitsGround(p, sd)) continue; // Punkt liegt im Erdschatten
                const ls = opticalDepth(p, sd, 8);
                for (0..3) |k| {
                    const tau = beta_r[k] * (dr + ls[0]) + beta_m * 1.1 * (dm + ls[1]);
                    const att = @exp(-tau);
                    sum_r[k] += rr * att;
                    sum_m[k] += mm * att;
                }
            }
            const mu = dot(dd, sd);
            const phase_r = 3.0 / (16.0 * pi) * (1 + mu * mu);
            const g2 = mie_g * mie_g;
            const phase_m = 3.0 / (8.0 * pi) * ((1 - g2) * (1 + mu * mu)) / ((2 + g2) * std.math.pow(f32, 1 + g2 - 2 * mie_g * mu, 1.5));
            // Pyrit führt Lichtstärken durch π geteilt (Lambert = Albedo · E ·
            // cos); die Strahldichte des Himmels braucht die ungeteilte.
            for (0..3) |k| c[k] = pi * e0 * multiple_scattering * (sum_r[k] * beta_r[k] * phase_r + sum_m[k] * beta_m * phase_m);
            // Wolken: von der Sonne beschienen, Unterseite vom Himmel erhellt,
            // gegen die Sonne mit hellem Rand (Vorwärtsstreuung)
            const cov = cloudCover(d);
            if (cov > 0) {
                const fwd = std.math.pow(f32, @max(dot(d, sd), 0), 8);
                // dichte Mitte dunkler (Selbstschatten), Ränder hell
                const shade_f = 0.45 + 0.55 * (1 - cov * cov);
                for (0..3) |k| {
                    const lit = sun_color[k] * (0.95 * shade_f + 1.5 * fwd) + 0.5 * c[k] + 0.2;
                    c[k] = c[k] + (lit - c[k]) * @min(cov * 1.1, 0.96);
                }
            }
            if (d[1] < 0) {
                // Unter dem Horizont: der Dunst davor, zum Boden hin nur
                // langsam dunkler. Ein harter Wechsel zeigte sich als dunkles
                // Band zwischen Himmel und Landschaft.
                const f = @max(1 + d[1] * 1.5, 0.35);
                for (0..3) |k| c[k] *= f;
            }
            const o = (y * w + x) * 4;
            env[o + 0] = c[0];
            env[o + 1] = c[1];
            env[o + 2] = c[2];
            env[o + 3] = 0;
        }
    }
    if (std.c.getenv("DEMO_SKY_INFO") != null) {
        // Beleuchtungsstärke auf einer waagerechten Fläche: Himmel gegen Sonne
        var e_sky = V{ 0, 0, 0 };
        for (0..h / 2) |y| {
            const theta = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(h)) * pi;
            const dw = (pi / @as(f32, @floatFromInt(h))) * (2 * pi / @as(f32, @floatFromInt(w))) * @sin(theta);
            for (0..w) |x| {
                const o = (y * w + x) * 4;
                for (0..3) |k| e_sky[k] += env[o + k] * @cos(theta) * dw;
            }
        }
        for ([_]f32{ 0.02, 0.25, 0.5, 1.0, 1.5 }) |el| {
            // Richtung gegenüber der Sonne auf dieser Höhe
            const th = 0.5 * pi - el;
            const ph = std.math.atan2(-sd[2], -sd[0]);
            var u = ph / (2 * pi);
            u -= @floor(u);
            const xi: usize = @min(@as(usize, @intFromFloat(u * @as(f32, @floatFromInt(w)))), w - 1);
            const yi: usize = @min(@as(usize, @intFromFloat(th / pi * @as(f32, @floatFromInt(h)))), h - 1);
            const o = (yi * w + xi) * 4;
            std.debug.print("Himmel gegenüber der Sonne, Höhe {d:.2} rad: {d:.3} {d:.3} {d:.3}\n", .{ el, env[o], env[o + 1], env[o + 2] });
        }
        std.debug.print("Auf einer waagerechten Fläche (Einheiten von Pyrit): Himmel {d:.3} {d:.3} {d:.3}, Sonne {d:.3} {d:.3} {d:.3}\n", .{ e_sky[0], e_sky[1], e_sky[2], sun_color[0] * sd[1], sun_color[1] * sd[1], sun_color[2] * sd[1] });
    }
    return .{ .env = env, .sun_color = sun_color };
}

/// Wolkendeckung als kachelbare Textur für die Wolkenschatten am Boden
/// (PyrLighting.sun_shadow_texture). Periodisches Wertrauschen: die
/// Gitterzellen wiederholen sich exakt mit der Kachel, es gibt keine Naht.
pub fn cloudShadowMap(gpa: std.mem.Allocator, n: u32) ![]u8 {
    const out = try gpa.alloc(u8, @as(usize, n) * n * 4);
    for (0..n) |y| {
        for (0..n) |x| {
            const u = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(n));
            const v = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(n));
            var sum: f32 = 0;
            var amp: f32 = 0.5;
            var cells: u32 = 4;
            for (0..6) |o| {
                sum += amp * periodicNoise(u * @as(f32, @floatFromInt(cells)), v * @as(f32, @floatFromInt(cells)), cells, 41 + @as(u32, @intCast(o)));
                amp *= 0.5;
                cells *= 2;
            }
            // dieselbe Schwelle und Randschärfe wie die Wolken am Himmel
            const c = @min(@max((sum - 0.53) * 7, 0), 1);
            const cov = c * c * (3 - 2 * c);
            const o = (y * n + x) * 4;
            out[o + 0] = @intFromFloat(cov * 255);
            out[o + 1] = out[o + 0];
            out[o + 2] = out[o + 0];
            out[o + 3] = 255;
        }
    }
    return out;
}

fn periodicNoise(x: f32, y: f32, period: u32, seed: u32) f32 {
    const fx = @floor(x);
    const fy = @floor(y);
    const p: i32 = @intCast(period);
    const ix: i32 = @mod(@as(i32, @intFromFloat(fx)), p);
    const iy: i32 = @mod(@as(i32, @intFromFloat(fy)), p);
    const tx = x - fx;
    const ty = y - fy;
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const a = hash(ix, iy, seed);
    const b = hash(@mod(ix + 1, p), iy, seed);
    const c = hash(ix, @mod(iy + 1, p), seed);
    const d = hash(@mod(ix + 1, p), @mod(iy + 1, p), seed);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}
