//! Nachbearbeitung pro Pixel (GPU und CPU):
//!
//! 1. temporal:  Verlauf über die Motion Vectors reprojizieren, gegen Normale
//!               und Tiefe prüfen, mit dem aktuellen Frame mischen. Mit
//!               Jitter ist das zugleich TAA; mit GI die Rauschreduktion.
//! 2. atrous:    kantenerhaltender À-trous-Filter (5x5, Schrittweite 2^i),
//!               gewichtet nach Normale, Tiefe und Helligkeit.
//! 3. resolve:   Albedo wieder aufmultiplizieren, Belichtung, Tonemapping,
//!               Ausgabe als HDR (4 x f32) und/oder RGBA8 (sRGB).
//!
//! Gefiltert wird die Beleuchtung ohne Albedo (Farbe / Albedo), damit
//! Voxelfarben scharf bleiben.

const types = @import("types.zig");
const fm = @import("fmath.zig");

const V4 = [4]f32;

inline fn ld4(ptr: u64, i: u64) V4 {
    return @as([*]const V4, @ptrFromInt(ptr))[i];
}

inline fn st4(ptr: u64, i: u64, v: V4) void {
    @as([*]V4, @ptrFromInt(ptr))[i] = v;
}

/// Interne Puffer (Verlauf, Filter, HDR-Zwischenbild) liegen halbgenau: halb
/// so viel Speicher und Bandbreite. Farben und Tiefen brauchen die Genauigkeit
/// von f32 hier nicht.
pub const H4 = [4]f16;

pub inline fn ldh(ptr: u64, i: u64) V4 {
    const h = @as([*]const H4, @ptrFromInt(ptr))[i];
    return .{ h[0], h[1], h[2], h[3] };
}

pub inline fn sth(ptr: u64, i: u64, v: V4) void {
    var h: H4 = undefined;
    inline for (0..4) |k| h[k] = @floatCast(@min(@max(v[k], -65504.0), 65504.0));
    @as([*]H4, @ptrFromInt(ptr))[i] = h;
}

/// Momente (Summe und Quadratsumme der Helligkeit) und Varianz liegen ebenfalls
/// halbgenau: 4 bzw. 2 Bytes je Pixel.
pub inline fn ld2(ptr: u64, i: u64) [2]f32 {
    const h = @as([*]const [2]f16, @ptrFromInt(ptr))[i];
    return .{ h[0], h[1] };
}

pub inline fn st2(ptr: u64, i: u64, v: [2]f32) void {
    @as([*][2]f16, @ptrFromInt(ptr))[i] = .{
        @floatCast(@min(@max(v[0], -65504.0), 65504.0)),
        @floatCast(@min(@max(v[1], 0), 65504.0)),
    };
}

pub inline fn ld1(ptr: u64, i: u64) f32 {
    return @as([*]const f16, @ptrFromInt(ptr))[i];
}

pub inline fn st1(ptr: u64, i: u64, v: f32) void {
    @as([*]f16, @ptrFromInt(ptr))[i] = @floatCast(@min(@max(v, 0), 65504.0));
}

inline fn lum(c: V4) f32 {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

inline fn isHit(n: V4) bool {
    return n[3] < 1e30;
}

/// Beleuchtung ohne Albedo
fn demodulate(color: V4, albedo: V4) V4 {
    if (color[3] < 0.5) return .{ color[0], color[1], color[2], 1 };
    var r: V4 = undefined;
    inline for (0..3) |k| r[k] = color[k] / @max(albedo[k], 0.01);
    r[3] = 1;
    return r;
}

pub fn temporal(p: *const types.PostParams, x: u32, y: u32) void {
    const w = p.width;
    const h = p.height;
    const i = @as(u64, y) * w + x;
    const n = ld4(p.normal, i);
    const cur = demodulate(ld4(p.color, i), ld4(p.albedo, i));
    sth(p.out_normal, i, n);

    var reset = p.reset != 0;
    const meta = @as([*]const types.Hit, @ptrFromInt(p.hits))[i].meta;
    if (meta & (types.hit_no_history | types.hit_new) != 0) reset = true;

    var hist: V4 = .{ 0, 0, 0, 0 };
    var hmom: [2]f32 = .{ 0, 0 };
    // Anteil des Verlaufs, der von einer *anderen* Voxelfläche stammt
    var edge_w: f32 = 0;
    if (!reset) {
        const mv = @as([*]const [2]f32, @ptrFromInt(p.motion))[i];
        const px = @as(f32, @floatFromInt(x)) + mv[0];
        const py = @as(f32, @floatFromInt(y)) + mv[1];
        const fx0 = @floor(px);
        const fy0 = @floor(py);
        const fx = px - fx0;
        const fy = py - fy0;
        const hit = isHit(n);
        var wsum: f32 = 0;
        inline for (0..4) |k| {
            const tx = fx0 + @as(f32, @floatFromInt(k & 1));
            const ty = fy0 + @as(f32, @floatFromInt(k >> 1));
            const bw = (if (k & 1 != 0) fx else 1 - fx) * (if (k >> 1 != 0) fy else 1 - fy);
            if (tx >= 0 and ty >= 0 and tx < @as(f32, @floatFromInt(w)) and ty < @as(f32, @floatFromInt(h)) and bw > 0) {
                const j = @as(u64, @intFromFloat(ty)) * w + @as(u64, @intFromFloat(tx));
                const hn = ldh(p.hist_normal, j);
                // Gültigkeit nicht als Ja/Nein, sondern als Gewicht. An einer
                // Voxelkante stehen die beiden Flächen 90° zueinander; mit
                // Jitter springt die Abtastung jeden Frame über die Kante.
                // Eine harte Prüfung verwirft dort den Verlauf *jedes Mal*,
                // das Pixel zeigt abwechselnd die eine und die andere Fläche
                // und der Schatten scheint zu wandern. Mit einem weichen
                // Übergang behält es einen Teil seiner Vorgeschichte und
                // mittelt beide Flächen, statt zwischen ihnen zu springen.
                var vw: f32 = 0;
                var same_face = false;
                if (hit) {
                    if (isHit(hn) and @abs(hn[3] - n[3]) < 0.05 * n[3] + 0.01) {
                        const nd = n[0] * hn[0] + n[1] * hn[1] + n[2] * hn[2];
                        same_face = nd > 0.9;
                        vw = (nd - p.normal_reject) / (1 - p.normal_reject);
                        vw = @min(@max(vw, 0), 1);
                        vw *= vw;
                    }
                } else if (!isHit(hn)) {
                    vw = 1;
                    same_face = true;
                }
                if (vw > 0 and !same_face) edge_w += bw;
                if (vw > 0) {
                    const c = ldh(p.hist_color, j);
                    const bwv = bw * vw;
                    inline for (0..4) |q| hist[q] += c[q] * bwv;
                    if (p.hist_moments != 0) {
                        const m = ld2(p.hist_moments, j);
                        hmom[0] += m[0] * bwv;
                        hmom[1] += m[1] * bwv;
                    }
                    wsum += bwv;
                }
            }
        }
        if (wsum < 1e-3) {
            reset = true;
        } else {
            inline for (0..4) |q| hist[q] /= wsum;
            hmom[0] /= wsum;
            hmom[1] /= wsum;
        }
    }

    // 3x3-Statistik des aktuellen Frames: für die Varianzbegrenzung und als
    // Varianzschätzung, solange zu wenige Frames akkumuliert sind.
    const want_moments = p.out_moments != 0;
    const count_prev = if (reset) 0 else hist[3];
    const need_spatial = (!reset and p.clamp_sigma > 0) or (want_moments and count_prev < 4);
    var mean: [3]f32 = .{ 0, 0, 0 };
    var sd: [3]f32 = .{ 0, 0, 0 };
    var var_spatial: f32 = 0;
    if (need_spatial) {
        var m1: [3]f32 = .{ 0, 0, 0 };
        var m2: [3]f32 = .{ 0, 0, 0 };
        var l1: f32 = 0;
        var l2: f32 = 0;
        var cnt: f32 = 0;
        var dy: i32 = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: i32 = -1;
            while (dx <= 1) : (dx += 1) {
                const sx = @as(i32, @intCast(x)) + dx;
                const sy = @as(i32, @intCast(y)) + dy;
                if (sx < 0 or sy < 0 or sx >= w or sy >= h) continue;
                const j = @as(u64, @intCast(sy)) * w + @as(u64, @intCast(sx));
                const c = demodulate(ld4(p.color, j), ld4(p.albedo, j));
                inline for (0..3) |k| {
                    m1[k] += c[k];
                    m2[k] += c[k] * c[k];
                }
                const l = lum(c);
                l1 += l;
                l2 += l * l;
                cnt += 1;
            }
        }
        inline for (0..3) |k| {
            mean[k] = m1[k] / cnt;
            sd[k] = @sqrt(@max(m2[k] / cnt - mean[k] * mean[k], 0));
        }
        const lm = l1 / cnt;
        var_spatial = @max(l2 / cnt - lm * lm, 0);
    }

    var out: V4 = undefined;
    if (reset) {
        out = .{ cur[0], cur[1], cur[2], 1 };
    } else {
        if (p.clamp_sigma > 0) {
            // Varianzbegrenzung gegen Nachziehen (TAA)
            inline for (0..3) |k| {
                hist[k] = @min(@max(hist[k], mean[k] - p.clamp_sigma * sd[k]), mean[k] + p.clamp_sigma * sd[k]);
            }
        }
        // Stammt der Verlauf von einer anderen Fläche, liegt das Pixel auf
        // einer Voxelkante: der Jitter schiebt die Abtastung jeden Frame
        // hin und her. Den Verlauf zu verwerfen lässt das Pixel zwischen
        // beiden Flächen springen (gemessen: Ausreißer 20,7 Stufen), ihn voll
        // zu übernehmen zieht nach (33 % Schärfeverlust bei Bewegung).
        // Also beides: übernehmen, aber die Mittelung kurz halten, damit es
        // beide Flächen mittelt und trotzdem schnell folgt.
        var limit = 1.0 / @max(p.alpha_min, 1e-4);
        if (edge_w > 0.25) limit = @min(limit, p.edge_frames);
        const count = @min(hist[3] + 1, limit);
        const a = 1.0 / count;
        inline for (0..3) |k| out[k] = hist[k] + (cur[k] - hist[k]) * a;
        out[3] = count;
    }
    sth(p.out_color, i, out);

    // Momente der Helligkeit mitführen; ihre Varianz steuert den À-trous-Filter.
    if (want_moments) {
        const l = lum(cur);
        var m: [2]f32 = .{ l, l * l };
        if (!reset) {
            const a = 1.0 / out[3];
            m[0] = hmom[0] + (l - hmom[0]) * a;
            m[1] = hmom[1] + (l * l - hmom[1]) * a;
        }
        st2(p.out_moments, i, m);
        // Wenige Frames: der zeitliche Schätzer ist noch blind, räumlich messen.
        var variance = @max(m[1] - m[0] * m[0], 0);
        if (out[3] < 4) variance = @max(variance, var_spatial);
        st1(p.out_var, i, variance);
    }
}

const kernel5 = [5]f32{ 1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

pub fn atrous(p: *const types.PostParams, x: u32, y: u32) void {
    const w = p.width;
    const h = p.height;
    const i = @as(u64, y) * w + x;
    const c = ldh(p.src, i);
    const n = ld4(p.normal, i);
    if (!isHit(n)) {
        sth(p.dst, i, c);
        if (p.var_dst != 0) st1(p.var_dst, i, if (p.var_src != 0) ld1(p.var_src, i) else 0);
        return;
    }
    const step: i32 = @intCast(p.step);
    const lc = lum(c);

    // Helligkeitstoleranz: aus der gemessenen Varianz (3x3-Gauß geglättet), wie
    // bei SVGF. Ohne Varianzpuffer bleibt die alte Schätzung über die Framezahl.
    var sigma_l: f32 = 4.0 * @max(lc, 0.02) / @sqrt(@max(c[3], 1));
    if (p.var_src != 0) {
        const gk = [3]f32{ 0.25, 0.5, 0.25 };
        var vg: f32 = 0;
        var gsum: f32 = 0;
        inline for (0..3) |b| {
            inline for (0..3) |a| {
                const sx = @as(i32, @intCast(x)) + @as(i32, a) - 1;
                const sy = @as(i32, @intCast(y)) + @as(i32, b) - 1;
                if (sx >= 0 and sy >= 0 and sx < w and sy < h) {
                    const g = gk[a] * gk[b];
                    vg += ld1(p.var_src, @as(u64, @intCast(sy)) * w + @as(u64, @intCast(sx))) * g;
                    gsum += g;
                }
            }
        }
        if (gsum > 0) vg /= gsum;
        sigma_l = @max(p.phi_lum, 1e-3) * @sqrt(vg) + 1e-4;
    }

    var sum: V4 = .{ 0, 0, 0, 0 };
    var vsum: f32 = 0;
    var wsum: f32 = 0;
    inline for (0..5) |b| {
        inline for (0..5) |a| {
            const sx = @as(i32, @intCast(x)) + (@as(i32, a) - 2) * step;
            const sy = @as(i32, @intCast(y)) + (@as(i32, b) - 2) * step;
            if (sx >= 0 and sy >= 0 and sx < w and sy < h) {
                const j = @as(u64, @intCast(sy)) * w + @as(u64, @intCast(sx));
                const nq = ld4(p.normal, j);
                if (isHit(nq)) {
                    const cq = ldh(p.src, j);
                    const nd = @max(n[0] * nq[0] + n[1] * nq[1] + n[2] * nq[2], 0);
                    const n2 = nd * nd;
                    const n4 = n2 * n2;
                    const n8 = n4 * n4;
                    // ^32. Der Exponent ist bei Voxelgeometrie ohne Belang:
                    // achsenparallele Flächen haben n·n' von exakt 0 oder 1,
                    // da ändert ein weicherer Verlauf nichts (nachgemessen).
                    const wn = n8 * n8 * n8 * n8;
                    const wd = fm.exp(-@abs(nq[3] - n[3]) / (0.02 * n[3] * @as(f32, @floatFromInt(step)) + 1e-3));
                    const wl = fm.exp(-@abs(lum(cq) - lc) / sigma_l);
                    const wgt = kernel5[a] * kernel5[b] * wn * wd * wl;
                    inline for (0..3) |k| sum[k] += cq[k] * wgt;
                    // Varianz mitfiltern (Gewichte quadriert): sie sinkt mit der Glättung
                    if (p.var_src != 0) vsum += ld1(p.var_src, j) * wgt * wgt;
                    wsum += wgt;
                }
            }
        }
    }
    if (wsum > 1e-8) {
        sth(p.dst, i, .{ sum[0] / wsum, sum[1] / wsum, sum[2] / wsum, c[3] });
        if (p.var_dst != 0) st1(p.var_dst, i, vsum / (wsum * wsum));
    } else {
        sth(p.dst, i, c);
        if (p.var_dst != 0) st1(p.var_dst, i, if (p.var_src != 0) ld1(p.var_src, i) else 0);
    }
}

pub fn tonemapChannel(x: f32, mode: u32) f32 {
    return switch (mode) {
        types.tonemap_aces => @min(@max((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0), 1),
        types.tonemap_reinhard => x / (1 + x),
        else => @min(@max(x, 0), 1),
    };
}

pub fn linearToSrgb(x: f32) f32 {
    return if (x <= 0.0031308) 12.92 * x else 1.055 * fm.pow(x, 1.0 / 2.4) - 0.055;
}

pub fn resolve(p: *const types.PostParams, x: u32, y: u32) void {
    const i = @as(u64, y) * p.width + x;
    const irr = ldh(p.src, i);
    const color = ld4(p.color, i);
    const alb = ld4(p.albedo, i);
    var hdr: V4 = .{ irr[0], irr[1], irr[2], color[3] };
    if (color[3] >= 0.5) {
        inline for (0..3) |k| hdr[k] *= @max(alb[k], 0.01);
    }
    if (p.out_hdr != 0) {
        if (p.hdr_half != 0) sth(p.out_hdr, i, hdr) else st4(p.out_hdr, i, hdr);
    }
    if (p.out_ldr != 0) {
        var px: [4]u8 = undefined;
        inline for (0..3) |k| {
            const v = linearToSrgb(tonemapChannel(@max(hdr[k] * p.exposure, 0), p.tonemap));
            px[if (p.bgra != 0) 2 - k else k] = @intFromFloat(@min(@max(v * 255.0 + 0.5, 0), 255));
        }
        px[3] = 255;
        @as([*][4]u8, @ptrFromInt(p.out_ldr))[i] = px;
    }
}
