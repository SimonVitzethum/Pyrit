//! Temporales Hochskalieren (TAAU) und Frame Generation, pro Ausgabepixel.
//!
//! TAAU: die gejitterten Renderpixel werden an ihrer echten Subpixelposition
//! mit einem Blackman-Harris-ähnlichen Kern in die Ausgabeauflösung
//! rekonstruiert und mit dem reprojizierten Verlauf (Catmull-Rom) gemischt.
//! Der Verlauf wird in YCoCg auf die Varianz der Nachbarschaft begrenzt; der
//! MV stammt vom nächstgelegenen (vordersten) Nachbarn, damit Kanten sauber
//! bleiben. Bei Faktor 1 ist das gewöhnliches, gutes TAA.
//!
//! Frame Generation: Zwischenbild aus zwei Ausgabeframes und den MVs per
//! Fixpunkt-Rückwärtssuche (p + (1 - t) · mv(p) = q).

const types = @import("types.zig");
const post = @import("post.zig");
const fm = @import("fmath.zig");

const V4 = [4]f32;

inline fn ld4(ptr: u64, i: u64) V4 {
    return @as([*]const V4, @ptrFromInt(ptr))[i];
}

inline fn st4(ptr: u64, i: u64, v: V4) void {
    @as([*]V4, @ptrFromInt(ptr))[i] = v;
}

/// Verlauf und Zwischenbilder liegen halbgenau (siehe post.zig)
const ldh = post.ldh;
const sth = post.sth;

inline fn ldColor(p: *const types.UpscaleParams, i: u64) V4 {
    return if (p.color_half != 0) ldh(p.color, i) else ld4(p.color, i);
}

inline fn toYCoCg(c: V4) [3]f32 {
    return .{
        0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2],
        0.5 * c[0] - 0.5 * c[2],
        -0.25 * c[0] + 0.5 * c[1] - 0.25 * c[2],
    };
}

inline fn fromYCoCg(y: [3]f32) [3]f32 {
    return .{ y[0] + y[1] - y[2], y[0] + y[2], y[0] - y[1] - y[2] };
}

/// Helligkeit komprimieren, damit einzelne helle Samples nicht dominieren
inline fn tonemapWeight(c: V4) f32 {
    return 1.0 / (1.0 + @max(c[0], @max(c[1], c[2])));
}

/// Catmull-Rom-Abtastung (4x4) mit Randbehandlung; null außerhalb
fn sampleCatmull(buf: u64, w: u32, h: u32, px: f32, py: f32) ?V4 {
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    if (px < 0 or py < 0 or px >= fw or py >= fh) return null;
    const x = px - 0.5;
    const y = py - 0.5;
    const x0 = @floor(x);
    const y0 = @floor(y);
    const tx = x - x0;
    const ty = y - y0;
    const wx = catmull(tx);
    const wy = catmull(ty);
    var sum: V4 = .{ 0, 0, 0, 0 };
    var wsum: f32 = 0;
    inline for (0..4) |j| {
        inline for (0..4) |i| {
            const sx = @as(i32, @intFromFloat(x0)) + @as(i32, i) - 1;
            const sy = @as(i32, @intFromFloat(y0)) + @as(i32, j) - 1;
            const cx: u32 = @intCast(@min(@max(sx, 0), @as(i32, @intCast(w)) - 1));
            const cy: u32 = @intCast(@min(@max(sy, 0), @as(i32, @intCast(h)) - 1));
            const wt = wx[i] * wy[j];
            const c = ldh(buf, @as(u64, cy) * w + cx);
            inline for (0..4) |k| sum[k] += c[k] * wt;
            wsum += wt;
        }
    }
    inline for (0..4) |k| sum[k] /= wsum;
    // Catmull-Rom kann negativ überschwingen
    inline for (0..3) |k| sum[k] = @max(sum[k], 0);
    return sum;
}

inline fn catmull(t: f32) [4]f32 {
    const t2 = t * t;
    const t3 = t2 * t;
    return .{
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    };
}

fn writeOutput(out_hdr: u64, out_ldr: u64, i: u64, c: V4, exposure: f32, tonemap: u32) void {
    writeOutputFmt(out_hdr, out_ldr, i, c, exposure, tonemap, false);
}

fn writeOutputFmt(out_hdr: u64, out_ldr: u64, i: u64, c: V4, exposure: f32, tonemap: u32, bgra: bool) void {
    if (out_hdr != 0) st4(out_hdr, i, .{ c[0], c[1], c[2], 1 });
    if (out_ldr != 0) {
        var px: [4]u8 = undefined;
        const tm = post.tonemap(.{ @max(c[0] * exposure, 0), @max(c[1] * exposure, 0), @max(c[2] * exposure, 0) }, tonemap);
        inline for (0..3) |k| {
            const v = post.linearToSrgb(tm[k]);
            px[if (bgra) 2 - k else k] = @intFromFloat(@min(@max(v * 255.0 + 0.5, 0), 255));
        }
        px[3] = 255;
        @as([*][4]u8, @ptrFromInt(out_ldr))[i] = px;
    }
}

pub fn taau(p: *const types.UpscaleParams, ox: u32, oy: u32) void {
    const iw = p.in_width;
    const ih = p.in_height;
    const oi = @as(u64, oy) * p.out_width + ox;
    const rx = @as(f32, @floatFromInt(iw)) / @as(f32, @floatFromInt(p.out_width));
    const ry = @as(f32, @floatFromInt(ih)) / @as(f32, @floatFromInt(p.out_height));
    // Ausgabepixelmitte in Renderpixeln
    const sx = (@as(f32, @floatFromInt(ox)) + 0.5) * rx;
    const sy = (@as(f32, @floatFromInt(oy)) + 0.5) * ry;
    const cx: i32 = @intFromFloat(@floor(sx - p.jitter[0]));
    const cy: i32 = @intFromFloat(@floor(sy - p.jitter[1]));

    var sum: V4 = .{ 0, 0, 0, 0 };
    var wsum: f32 = 0;
    var wmax: f32 = 0;
    var m1: [3]f32 = .{ 0, 0, 0 };
    var m2: [3]f32 = .{ 0, 0, 0 };
    var lo: [3]f32 = .{ 1e30, 1e30, 1e30 };
    var hi: [3]f32 = .{ -1e30, -1e30, -1e30 };
    var cnt: f32 = 0;
    var best_depth: f32 = types.flt_max;
    var best: u64 = @as(u64, @intCast(@min(@max(cy, 0), @as(i32, @intCast(ih)) - 1))) * iw + @as(u64, @intCast(@min(@max(cx, 0), @as(i32, @intCast(iw)) - 1)));
    // Kernbreite: bei starkem Hochskalieren etwas breiter, sonst Treppen
    const kscale = 1.0 / @max(@max(rx, ry), 0.5);
    var dy: i32 = -1;
    while (dy <= 1) : (dy += 1) {
        var dx: i32 = -1;
        while (dx <= 1) : (dx += 1) {
            const x = cx + dx;
            const y = cy + dy;
            if (x < 0 or y < 0 or x >= iw or y >= ih) continue;
            const j = @as(u64, @intCast(y)) * iw + @as(u64, @intCast(x));
            const c = ldColor(p, j);
            const ddx = (@as(f32, @floatFromInt(x)) + 0.5 + p.jitter[0] - sx) * kscale;
            const ddy = (@as(f32, @floatFromInt(y)) + 0.5 + p.jitter[1] - sy) * kscale;
            const wgt = fm.exp(-p.kernel_sharp * (ddx * ddx + ddy * ddy)) * tonemapWeight(c);
            inline for (0..3) |k| sum[k] += c[k] * wgt;
            wsum += wgt;
            wmax = @max(wmax, fm.exp(-p.kernel_sharp * (ddx * ddx + ddy * ddy)));
            const yc = toYCoCg(c);
            inline for (0..3) |k| {
                m1[k] += yc[k];
                m2[k] += yc[k] * yc[k];
                lo[k] = @min(lo[k], yc[k]);
                hi[k] = @max(hi[k], yc[k]);
            }
            cnt += 1;
            const d = ld4(p.normal, j)[3];
            if (d < best_depth) {
                best_depth = d;
                best = j;
            }
        }
    }
    var cur: V4 = .{ 0, 0, 0, 1 };
    if (wsum > 0) {
        inline for (0..3) |k| cur[k] = sum[k] / wsum;
    }

    const mv_in = @as([*]const [2]f32, @ptrFromInt(p.motion))[best];
    const mv = [2]f32{ mv_in[0] / rx, mv_in[1] / ry };
    st4(p.mvd_out, oi, .{ mv[0], mv[1], best_depth, 0 });

    var reset = p.reset != 0;
    const meta = @as([*]const types.Hit, @ptrFromInt(p.hits))[best].meta;
    if (meta & (types.hit_no_history | types.hit_new) != 0) reset = true;

    // Anteil des aktuellen Frames: je näher ein Sample an der Pixelmitte, desto mehr
    const conf = @max(wmax, 0.05);
    var out: V4 = undefined;
    const hist_opt = if (reset) null else sampleCatmull(p.hist_in, p.out_width, p.out_height, @as(f32, @floatFromInt(ox)) + 0.5 + mv[0], @as(f32, @floatFromInt(oy)) + 0.5 + mv[1]);
    if (hist_opt) |hist| {
        // Verlauf auf die Farbverteilung der Nachbarschaft begrenzen (YCoCg)
        var hy = toYCoCg(hist);
        inline for (0..3) |k| {
            const mean = m1[k] / cnt;
            const sd = @sqrt(@max(m2[k] / cnt - mean * mean, 0));
            const a = @max(lo[k], mean - p.clamp_sigma * sd);
            const b = @min(hi[k], mean + p.clamp_sigma * sd);
            hy[k] = @min(@max(hy[k], a), @max(a, b));
        }
        const hc = fromYCoCg(hy);
        // Jede Umabtastung bei halbem Pixelversatz weicht auf: dann weniger Verlauf
        const frx = @abs(mv[0] - @round(mv[0]));
        const fry = @abs(mv[1] - @round(mv[1]));
        const soften = @max(0.2, 1.0 - 3.0 * @max(frx, fry));
        const hw = @min(hist[3], p.max_weight * soften);
        const tw = hw + conf;
        inline for (0..3) |k| out[k] = (hc[k] * hw + cur[k] * conf) / tw;
        out[3] = tw;
    } else {
        out = .{ cur[0], cur[1], cur[2], conf };
    }
    sth(p.hist_out, oi, out);
    writeOutputFmt(p.out_hdr, p.out_ldr, oi, out, p.exposure, p.tonemap, p.bgra != 0);
}

inline fn sampleBilinear(buf: u64, w: u32, h: u32, px: f32, py: f32) V4 {
    const x = @min(@max(px - 0.5, 0), @as(f32, @floatFromInt(w - 1)));
    const y = @min(@max(py - 0.5, 0), @as(f32, @floatFromInt(h - 1)));
    const x0: u32 = @intFromFloat(@floor(x));
    const y0: u32 = @intFromFloat(@floor(y));
    const x1 = @min(x0 + 1, w - 1);
    const y1 = @min(y0 + 1, h - 1);
    const fx = x - @as(f32, @floatFromInt(x0));
    const fy = y - @as(f32, @floatFromInt(y0));
    const a = ldh(buf, @as(u64, y0) * w + x0);
    const b = ldh(buf, @as(u64, y0) * w + x1);
    const c = ldh(buf, @as(u64, y1) * w + x0);
    const d = ldh(buf, @as(u64, y1) * w + x1);
    var r: V4 = undefined;
    inline for (0..4) |k| r[k] = (a[k] + (b[k] - a[k]) * fx) + ((c[k] + (d[k] - c[k]) * fx) - (a[k] + (b[k] - a[k]) * fx)) * fy;
    return r;
}

inline fn nearestMv(p: *const types.FrameGenParams, px: f32, py: f32) V4 {
    const x: u32 = @intFromFloat(@min(@max(px, 0), @as(f32, @floatFromInt(p.width - 1))));
    const y: u32 = @intFromFloat(@min(@max(py, 0), @as(f32, @floatFromInt(p.height - 1))));
    return ld4(p.motion_depth, @as(u64, y) * p.width + x);
}

/// Vorwärtsprojektion, Schritt 1: jedes Pixel des aktuellen Frames wandert an
/// seinen Platz im Zwischenbild; die kleinste Tiefe gewinnt (Verdeckung).
pub fn frameGenSplat(p: *const types.FrameGenParams, x: u32, y: u32) void {
    const i = @as(u64, y) * p.width + x;
    const m = ld4(p.motion_depth, i);
    const back = 1 - p.t;
    const qx: i32 = @intFromFloat(@round(@as(f32, @floatFromInt(x)) - back * m[0]));
    const qy: i32 = @intFromFloat(@round(@as(f32, @floatFromInt(y)) - back * m[1]));
    if (qx < 0 or qy < 0 or qx >= p.width or qy >= p.height) return;
    const j = @as(u64, @intCast(qy)) * p.width + @as(u32, @intCast(qx));
    const bits: u32 = @bitCast(m[2]);
    _ = @atomicRmw(u32, &@as([*]u32, @ptrFromInt(p.depth_mid))[j], .Min, bits, .monotonic);
}

/// Schritt 2: der Gewinner schreibt seinen Bewegungsvektor ins Zwischenbild.
pub fn frameGenSplatMv(p: *const types.FrameGenParams, x: u32, y: u32) void {
    const i = @as(u64, y) * p.width + x;
    const m = ld4(p.motion_depth, i);
    const back = 1 - p.t;
    const qx: i32 = @intFromFloat(@round(@as(f32, @floatFromInt(x)) - back * m[0]));
    const qy: i32 = @intFromFloat(@round(@as(f32, @floatFromInt(y)) - back * m[1]));
    if (qx < 0 or qy < 0 or qx >= p.width or qy >= p.height) return;
    const j = @as(u64, @intCast(qy)) * p.width + @as(u32, @intCast(qx));
    const bits: u32 = @bitCast(m[2]);
    if (@as([*]const u32, @ptrFromInt(p.depth_mid))[j] != bits) return;
    @as([*][2]f32, @ptrFromInt(p.mv_mid))[j] = .{ m[0], m[1] };
}

pub fn frameGen(p: *const types.FrameGenParams, x: u32, y: u32) void {
    const i = @as(u64, y) * p.width + x;
    const qx = @as(f32, @floatFromInt(x)) + 0.5;
    const qy = @as(f32, @floatFromInt(y)) + 0.5;
    const back = 1 - p.t;
    // Vorwärtsprojektion: der MV am Zwischenbildpixel ist schon bekannt, wenn
    // ihn ein Pixel des aktuellen Frames dorthin geworfen hat (Verdeckung
    // korrekt nach Tiefe). Sonst Fixpunktsuche.
    var px = qx;
    var py = qy;
    var m = nearestMv(p, px, py);
    if (p.mv_mid != 0 and @as([*]const u32, @ptrFromInt(p.depth_mid))[i] != 0xFFFF_FFFF) {
        const sm = @as([*]const [2]f32, @ptrFromInt(p.mv_mid))[i];
        m = .{ sm[0], sm[1], m[2], 0 };
        px = qx - back * m[0];
        py = qy - back * m[1];
        const fw2: f32 = @floatFromInt(p.width);
        const fh2: f32 = @floatFromInt(p.height);
        const in_cur2 = px >= 0 and py >= 0 and px < fw2 and py < fh2;
        const prx2 = px + m[0];
        const pry2 = py + m[1];
        const in_prev2 = prx2 >= 0 and pry2 >= 0 and prx2 < fw2 and pry2 < fh2;
        var c2: V4 = undefined;
        if (in_cur2 and in_prev2) {
            const c1 = sampleBilinear(p.cur_color, p.width, p.height, px, py);
            const c0 = sampleBilinear(p.prev_color, p.width, p.height, prx2, pry2);
            inline for (0..4) |k| c2[k] = c0[k] * back + c1[k] * p.t;
        } else if (in_cur2) {
            c2 = sampleBilinear(p.cur_color, p.width, p.height, px, py);
        } else {
            c2 = sampleBilinear(p.prev_color, p.width, p.height, qx, qy);
        }
        writeOutput(p.out_hdr, p.out_ldr, i, c2, p.exposure, p.tonemap);
        return;
    }
    inline for (0..3) |_| {
        px = qx - back * m[0];
        py = qy - back * m[1];
        m = nearestMv(p, px, py);
    }
    // Konsistenz: landet die Suche wirklich bei q? Sonst (Verdeckung) die
    // nähere von zwei Lösungen: Start mit dem MV am Zielpunkt selbst
    const m_here = nearestMv(p, qx, qy);
    const err = @abs(px + back * m[0] - qx) + @abs(py + back * m[1] - qy);
    if (err > 1.0 and m_here[2] < m[2]) {
        m = m_here;
        px = qx - back * m[0];
        py = qy - back * m[1];
    }
    const fw: f32 = @floatFromInt(p.width);
    const fh: f32 = @floatFromInt(p.height);
    const in_cur = px >= 0 and py >= 0 and px < fw and py < fh;
    const prx = px + m[0];
    const pry = py + m[1];
    const in_prev = prx >= 0 and pry >= 0 and prx < fw and pry < fh;
    var c: V4 = undefined;
    if (in_cur and in_prev) {
        const c1 = sampleBilinear(p.cur_color, p.width, p.height, px, py);
        const c0 = sampleBilinear(p.prev_color, p.width, p.height, prx, pry);
        inline for (0..4) |k| c[k] = c0[k] * back + c1[k] * p.t;
    } else if (in_cur) {
        c = sampleBilinear(p.cur_color, p.width, p.height, px, py);
    } else if (in_prev) {
        c = sampleBilinear(p.prev_color, p.width, p.height, prx, pry);
    } else {
        c = sampleBilinear(p.cur_color, p.width, p.height, qx, qy);
    }
    writeOutputFmt(p.out_hdr, p.out_ldr, i, c, p.exposure, p.tonemap, p.bgra != 0);
}

/// Nach einem externen Upscaler (DLSS): `p.color` liegt in Ausgabeauflösung
/// vor. Schreibt Ausgaben, Verlauf (für die Frame Generation) und MV + Tiefe.
pub fn present(p: *const types.UpscaleParams, ox: u32, oy: u32) void {
    const iw = p.in_width;
    const ih = p.in_height;
    const oi = @as(u64, oy) * p.out_width + ox;
    const rx = @as(f32, @floatFromInt(iw)) / @as(f32, @floatFromInt(p.out_width));
    const ry = @as(f32, @floatFromInt(ih)) / @as(f32, @floatFromInt(p.out_height));
    const cx: i32 = @intFromFloat(@floor((@as(f32, @floatFromInt(ox)) + 0.5) * rx - p.jitter[0]));
    const cy: i32 = @intFromFloat(@floor((@as(f32, @floatFromInt(oy)) + 0.5) * ry - p.jitter[1]));
    var best_depth: f32 = types.flt_max;
    var best: u64 = @as(u64, @intCast(@min(@max(cy, 0), @as(i32, @intCast(ih)) - 1))) * iw + @as(u64, @intCast(@min(@max(cx, 0), @as(i32, @intCast(iw)) - 1)));
    var dy: i32 = -1;
    while (dy <= 1) : (dy += 1) {
        var dx: i32 = -1;
        while (dx <= 1) : (dx += 1) {
            const x = cx + dx;
            const y = cy + dy;
            if (x < 0 or y < 0 or x >= iw or y >= ih) continue;
            const j = @as(u64, @intCast(y)) * iw + @as(u64, @intCast(x));
            const d = ld4(p.normal, j)[3];
            if (d < best_depth) {
                best_depth = d;
                best = j;
            }
        }
    }
    const mv_in = @as([*]const [2]f32, @ptrFromInt(p.motion))[best];
    st4(p.mvd_out, oi, .{ mv_in[0] / rx, mv_in[1] / ry, best_depth, 0 });
    const c = ldColor(p, oi);
    const out: V4 = .{ c[0], c[1], c[2], 1 };
    sth(p.hist_out, oi, out);
    writeOutputFmt(p.out_hdr, p.out_ldr, oi, out, p.exposure, p.tonemap, p.bgra != 0);
}
