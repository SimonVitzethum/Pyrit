//! Kamera- und Bildeffekte auf dem fertigen HDR-Bild in Ausgabeauflösung.
//!
//! Reihenfolge (jede Stufe einzeln abschaltbar):
//!
//!   1. dof         Tiefenschärfe: Zerstreuungskreis aus der Tiefe, Sammelfilter
//!   2. motionBlur  Bewegungsunschärfe entlang des Bewegungsvektors
//!   3. bloomPre    Lichter abschöpfen und halbieren
//!      bloomDown   weitere Halbierungen (Pyramide)
//!      bloomUp     wieder hoch und dazuaddieren (weiche, breite Lichter)
//!   4. exposeScan  mittlere Log-Helligkeit messen (Belichtungsautomatik)
//!      exposeApply Belichtung nachführen (zeitlich gedämpft)
//!   5. resolveFx   Weißabgleich, Kontrast, Sättigung, Lift/Gamma/Gain,
//!                  Tonemapping, optional 3D-LUT, Ausgabe als HDR und/oder LDR
//!
//! Bewegung und Tiefe kommen aus demselben Puffer, den TAAU für die Frame
//! Generation schreibt (`mvd`: mv.xy, Tiefe, 0) – in Ausgabeauflösung, also
//! ohne zusätzliche Abtastung.

const warp = @import("warp.zig");
const types = @import("types.zig");
const fm = @import("fmath.zig");
const post = @import("post.zig");

const V4 = [4]f32;

inline fn ld4(ptr: u64, i: u64) V4 {
    return @as([*]const V4, @ptrFromInt(ptr))[i];
}

inline fn st4(ptr: u64, i: u64, v: V4) void {
    @as([*]V4, @ptrFromInt(ptr))[i] = v;
}

inline fn ldh(ptr: u64, i: u64) V4 {
    const h = @as([*]const [4]f16, @ptrFromInt(ptr))[i];
    return .{ h[0], h[1], h[2], h[3] };
}

inline fn sth(ptr: u64, i: u64, v: V4) void {
    var h: [4]f16 = undefined;
    inline for (0..4) |k| h[k] = @floatCast(@min(@max(v[k], -65504.0), 65504.0));
    @as([*][4]f16, @ptrFromInt(ptr))[i] = h;
}

inline fn lum(c: V4) f32 {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

inline fn coords(w: u32, h: u32, i: u64) ?[2]u32 {
    const x: u32 = @intCast(i % w);
    const y: u32 = @intCast(i / w);
    if (y >= h) return null;
    return .{ x, y };
}

/// Feste Abtastpunkte auf der Einheitsscheibe (Vogel-Spirale, vorberechnet):
/// gleichmäßig verteilt, ohne Zufall – dadurch flimmert die Unschärfe nicht.
const disk = [16][2]f32{
    .{ 0.1768, 0.0000 },   .{ -0.1227, 0.2126 },  .{ -0.1545, -0.3455 }, .{ 0.4157, 0.1854 },
    .{ -0.2621, 0.4996 },  .{ -0.2848, -0.5301 }, .{ 0.6708, 0.1099 },   .{ -0.5934, 0.4718 },
    .{ 0.0653, -0.7627 },  .{ 0.5934, 0.5533 },   .{ -0.8434, -0.0623 }, .{ 0.4744, -0.7628 },
    .{ 0.2447, 0.8981 },   .{ -0.8215, -0.4931 }, .{ 0.9186, -0.2807 },  .{ -0.3441, 0.9070 },
};

/// Bewegung und Tiefe aus der Renderauflösung in die Ausgabeauflösung packen
/// (nur nötig, wenn nicht TAAU sie ohnehin schon liefert).
pub fn packMvd(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.width, p.height, i) orelse return;
    const sw = p.dst_width;
    const sh = p.dst_height;
    const rx = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(p.width));
    const ry = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(p.height));
    const sx: u32 = @min(@as(u32, @intFromFloat((@as(f32, @floatFromInt(c[0])) + 0.5) * rx)), sw - 1);
    const sy: u32 = @min(@as(u32, @intFromFloat((@as(f32, @floatFromInt(c[1])) + 0.5) * ry)), sh - 1);
    const j = @as(u64, sy) * sw + sx;
    const mv = @as([*]const [2]f32, @ptrFromInt(p.mv_src))[j];
    const nd = ld4(p.normal_src, j);
    st4(p.mvd, i, .{ mv[0] / rx, mv[1] / ry, nd[3], 0 });
}

// ---------------------------------------------------------------------------
// 1. Tiefenschärfe
// ---------------------------------------------------------------------------

/// Zerstreuungskreis in Pixeln. `focus` ist die Entfernung der Schärfeebene,
/// `strength` fasst Blende und Brennweite zusammen.
/// `dof_strength` ist der Zerstreuungskreis in Pixeln für unendlich weit
/// entfernte Punkte; nähere Punkte bekommen entsprechend weniger.
inline fn cocOf(p: *const types.PostFxParams, depth: f32, focus: f32) f32 {
    // Himmel liegt im Unendlichen: (d-f)/d -> 1
    if (!(depth < 1e30)) return @min(p.dof_strength, p.dof_max_coc);
    const d = @max(depth, 1e-3);
    const f = @max(focus, 1e-3);
    // Vor der Schärfeebene auf Wunsch scharf lassen: in einem Spiel steht die
    // Kamera meist dicht an Geometrie, und ein unscharfer Vordergrund stört
    // dort mehr als er nützt.
    if (p.dof_far_only != 0 and d <= f) return 0;
    const c = p.dof_strength * (d - f) / d;
    return @min(@abs(c), p.dof_max_coc);
}

pub fn dof(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.width, p.height, i) orelse return;
    const center = ld4(p.color, i);
    const mvd = ld4(p.mvd, i);
    // Autofokus: die Tiefe in der Bildmitte. Ein zusätzlicher Lesezugriff je
    // Thread (aus dem Cache), dafür bleibt die Kette ohne Rückkanal zum Host.
    const focus = if (p.dof_autofocus != 0) blk: {
        const ci = @as(u64, p.height / 2) * p.width + p.width / 2;
        const d = ld4(p.mvd, ci)[2];
        break :blk if (d < 1e30) d else 1e4;
    } else p.dof_focus;
    const coc = cocOf(p, mvd[2], focus);
    if (coc < 0.75) { // schärfer als ein Pixel: nichts zu tun
        st4(p.dst, i, center);
        return;
    }
    var sum: V4 = .{ center[0], center[1], center[2], 0 };
    var wsum: f32 = 1;
    for (disk) |d| {
        const sx = @as(f32, @floatFromInt(c[0])) + d[0] * coc;
        const sy = @as(f32, @floatFromInt(c[1])) + d[1] * coc;
        if (sx < 0 or sy < 0 or sx >= @as(f32, @floatFromInt(p.width)) or sy >= @as(f32, @floatFromInt(p.height))) continue;
        const j = @as(u64, @intFromFloat(sy)) * p.width + @as(u64, @intFromFloat(sx));
        const sc = ld4(p.color, j);
        const sd = ld4(p.mvd, j)[2];
        const scoc = cocOf(p, sd, focus);
        // Ein scharfes Vordergrundpixel darf nicht in den unscharfen
        // Hintergrund bluten: es zählt nur, wenn es selbst unscharf genug ist
        // oder vor der Schärfeebene liegt.
        const near = sd < mvd[2];
        const wgt: f32 = if (scoc >= coc * 0.5 or near) 1 else 0;
        inline for (0..3) |k| sum[k] += sc[k] * wgt;
        wsum += wgt;
    }
    inline for (0..3) |k| sum[k] /= wsum;
    sum[3] = center[3];
    st4(p.dst, i, sum);
}

// ---------------------------------------------------------------------------
// 2. Bewegungsunschärfe
// ---------------------------------------------------------------------------

pub fn motionBlur(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.width, p.height, i) orelse return;
    const center = ld4(p.color, i);
    const mvd = ld4(p.mvd, i);
    // Der Bewegungsvektor zeigt zum Vorframe; die Verwischung läuft entlang
    // dieser Strecke, skaliert mit der Verschlusszeit.
    var vx = -mvd[0] * p.blur_scale;
    var vy = -mvd[1] * p.blur_scale;
    const len = @sqrt(vx * vx + vy * vy);
    if (len < 0.5) {
        st4(p.dst, i, center);
        return;
    }
    if (len > p.blur_max) {
        vx *= p.blur_max / len;
        vy *= p.blur_max / len;
    }
    const n: u32 = @min(@max(p.blur_samples, 2), 32);
    var sum: V4 = .{ 0, 0, 0, 0 };
    var wsum: f32 = 0;
    var s: u32 = 0;
    while (s < n) : (s += 1) {
        // von -0.5 bis +0.5 der Strecke, damit die Unschärfe symmetrisch liegt
        const t = (@as(f32, @floatFromInt(s)) + 0.5) / @as(f32, @floatFromInt(n)) - 0.5;
        const sx = @as(f32, @floatFromInt(c[0])) + vx * t;
        const sy = @as(f32, @floatFromInt(c[1])) + vy * t;
        if (sx < 0 or sy < 0 or sx >= @as(f32, @floatFromInt(p.width)) or sy >= @as(f32, @floatFromInt(p.height))) continue;
        const j = @as(u64, @intFromFloat(sy)) * p.width + @as(u64, @intFromFloat(sx));
        const sc = ld4(p.color, j);
        inline for (0..3) |k| sum[k] += sc[k];
        wsum += 1;
    }
    if (wsum < 1) {
        st4(p.dst, i, center);
        return;
    }
    var out: V4 = undefined;
    inline for (0..3) |k| out[k] = sum[k] / wsum;
    out[3] = center[3];
    st4(p.dst, i, out);
}

// ---------------------------------------------------------------------------
// 3. Bloom (Pyramide aus Halbierungen, danach gewichtet wieder hoch)
// ---------------------------------------------------------------------------

inline fn tap(buf: u64, w: u32, h: u32, x: i32, y: i32, half: bool) V4 {
    const cx: u32 = @intCast(@min(@max(x, 0), @as(i32, @intCast(w - 1))));
    const cy: u32 = @intCast(@min(@max(y, 0), @as(i32, @intCast(h - 1))));
    const j = @as(u64, cy) * w + cx;
    return if (half) ldh(buf, j) else ld4(buf, j);
}

/// Lichter abschöpfen und dabei halbieren. Der Knick (`knee`) macht den
/// Übergang weich, sonst flackern Kanten an der Schwelle.
pub fn bloomPrefilter(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.dst_width, p.dst_height, i) orelse return;
    var sum: V4 = .{ 0, 0, 0, 0 };
    inline for (0..2) |dy| {
        inline for (0..2) |dx| {
            const s = tap(p.color, p.width, p.height, @as(i32, @intCast(c[0])) * 2 + @as(i32, dx), @as(i32, @intCast(c[1])) * 2 + @as(i32, dy), p.src_half != 0);
            inline for (0..3) |k| sum[k] += s[k] * 0.25;
        }
    }
    const l = lum(sum);
    const knee = @max(p.bloom_knee, 1e-4);
    const soft = @min(@max(l - p.bloom_threshold + knee, 0), 2 * knee);
    const contrib = @max(soft * soft / (4 * knee), l - p.bloom_threshold) / @max(l, 1e-4);
    var out: V4 = .{ 0, 0, 0, 1 };
    inline for (0..3) |k| out[k] = sum[k] * @max(contrib, 0);
    sth(p.dst, i, out);
}

/// Halbieren (4 Abtastungen)
pub fn bloomDown(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.dst_width, p.dst_height, i) orelse return;
    var sum: V4 = .{ 0, 0, 0, 0 };
    inline for (0..2) |dy| {
        inline for (0..2) |dx| {
            const s = tap(p.color, p.width, p.height, @as(i32, @intCast(c[0])) * 2 + @as(i32, dx), @as(i32, @intCast(c[1])) * 2 + @as(i32, dy), true);
            inline for (0..3) |k| sum[k] += s[k] * 0.25;
        }
    }
    sum[3] = 1;
    sth(p.dst, i, sum);
}

/// Verdoppeln und dazuaddieren (3x3-Zelt, dadurch weich statt blockig)
pub fn bloomUp(p: *const types.PostFxParams, i: u64) void {
    const c = coords(p.dst_width, p.dst_height, i) orelse return;
    const hx = @as(i32, @intCast(c[0])) >> 1;
    const hy = @as(i32, @intCast(c[1])) >> 1;
    const kern = [3]f32{ 0.25, 0.5, 0.25 };
    var sum: V4 = .{ 0, 0, 0, 0 };
    var wsum: f32 = 0;
    inline for (0..3) |dy| {
        inline for (0..3) |dx| {
            const wgt = kern[dx] * kern[dy];
            const s = tap(p.color, p.width, p.height, hx + @as(i32, dx) - 1, hy + @as(i32, dy) - 1, true);
            inline for (0..3) |k| sum[k] += s[k] * wgt;
            wsum += wgt;
        }
    }
    const base = ldh(p.dst, i);
    var out: V4 = .{ 0, 0, 0, 1 };
    inline for (0..3) |k| out[k] = base[k] + sum[k] / wsum;
    sth(p.dst, i, out);
}

// ---------------------------------------------------------------------------
// 4. Belichtungsautomatik
// ---------------------------------------------------------------------------

/// Festkomma-Streuung der Log-Helligkeit: die GPU-Backends hier haben keinen
/// verlässlichen f32-Atomic, ein u32-Zähler tut es genauso.
const expose_scale: f32 = 256;
const expose_offset: f32 = 16; // log2-Helligkeiten von -16 bis +16

pub fn exposeScan(p: *const types.PostFxParams, i: u64) void {
    // Nur jedes vierte Pixel in beiden Richtungen: ein Sechzehntel der Arbeit,
    // für einen Mittelwert mehr als genug.
    const step: u32 = 4;
    const w = (p.width + step - 1) / step;
    const h = (p.height + step - 1) / step;
    const c = coords(w, h, i) orelse return;
    const x = c[0] * step;
    const y = c[1] * step;
    if (x >= p.width or y >= p.height) return;
    const col = ld4(p.color, @as(u64, y) * p.width + x);
    const l = @max(lum(col), 1e-4);
    const v = @min(@max(fm.log2(l) + expose_offset, 0), 2 * expose_offset);
    const acc: [*]u32 = @ptrFromInt(p.expose_acc);
    warp.add(&acc[0], @intFromFloat(v * expose_scale));
    warp.add(&acc[1], 1);
}

/// Aus dem Mittelwert die Belichtung bilden und zeitlich gedämpft nachführen.
/// Ein Thread genügt; das Ergebnis bleibt auf der GPU, damit die Kette nicht
/// auf den Host warten muss.
pub fn exposeApply(p: *const types.PostFxParams, i: u64) void {
    if (i != 0) return;
    const acc: [*]u32 = @ptrFromInt(p.expose_acc);
    const state: [*]f32 = @ptrFromInt(p.expose_state);
    const n = acc[1];
    var target = state[0];
    if (n > 0) {
        const mean = @as(f32, @floatFromInt(acc[0])) / (expose_scale * @as(f32, @floatFromInt(n))) - expose_offset;
        // Mittleres Grau auf 0.18 legen, dann Korrektur des Anwenders
        const want = 0.18 / @max(fm.exp2(mean), 1e-6);
        target = @min(@max(want, p.expose_min), p.expose_max) * fm.exp2(p.expose_compensation);
    }
    const prev = state[1];
    // gedämpft: exponentiell mit der Zeitkonstante `expose_speed`
    // Große Sprünge (Laden, Szenenwechsel) holt sie schneller ein: ab einer
    // Blende Abstand wächst die Rate mit, sonst lag das Bild nach dem Start
    // sekundenlang fast weiß im Dunst
    const stops = if (prev > 0 and target > 0) @abs(fm.log2(target / prev)) else 0;
    const a = if (prev > 0) @min(@max(p.expose_speed, 0) * @max(1, stops * stops), 1) else 1;
    const cur = if (prev > 0) prev + (target - prev) * a else target;
    state[0] = target;
    state[1] = cur;
    acc[0] = 0;
    acc[1] = 0;
}

// ---------------------------------------------------------------------------
// 5. Farbkorrektur und Ausgabe
// ---------------------------------------------------------------------------

/// Weißabgleich über Temperatur (in Mired) und Tint, im linearen Raum.
inline fn whiteBalance(c: V4, temperature: f32, tint: f32) V4 {
    if (temperature == 0 and tint == 0) return c;
    // Näherung: warme Temperatur hebt Rot und senkt Blau, Tint dreht Grün/Magenta
    const t = temperature * 0.01;
    const g = tint * 0.01;
    return .{
        c[0] * (1 + t),
        c[1] * (1 + g),
        c[2] * (1 - t),
        c[3],
    };
}

inline fn gradeLinear(p: *const types.PostFxParams, c: V4) V4 {
    var r = whiteBalance(c, p.temperature, p.tint);
    // Sättigung um die Helligkeit
    const l = lum(r);
    inline for (0..3) |k| r[k] = l + (r[k] - l) * p.saturation;
    // Kontrast um mittleres Grau
    inline for (0..3) |k| r[k] = 0.18 + (r[k] - 0.18) * p.contrast;
    // Lift / Gamma / Gain
    inline for (0..3) |k| {
        const v = @max(r[k] * p.gain[k] + p.lift[k], 0);
        r[k] = if (p.gamma[k] == 1) v else fm.pow(v, 1.0 / @max(p.gamma[k], 1e-3));
    }
    return r;
}

/// 3D-LUT im Anzeigeraum (Kantenlänge `lut_size`, RGBA8, trilinear)
inline fn applyLut(p: *const types.PostFxParams, c: [3]f32) [3]f32 {
    const n = p.lut_size;
    if (p.lut == 0 or n < 2) return c;
    const fn_f: f32 = @floatFromInt(n - 1);
    const lut: [*]const [4]u8 = @ptrFromInt(p.lut);
    var out: [3]f32 = .{ 0, 0, 0 };
    var fx: [3]f32 = undefined;
    var base_idx: [3]u32 = undefined;
    inline for (0..3) |k| {
        const v = @min(@max(c[k], 0), 1) * fn_f;
        const f = @floor(v);
        base_idx[k] = @intFromFloat(f);
        fx[k] = v - f;
    }
    inline for (0..8) |corner| {
        var idx: [3]u32 = undefined;
        var wgt: f32 = 1;
        inline for (0..3) |k| {
            const up = (corner >> k) & 1 == 1;
            idx[k] = @min(base_idx[k] + @as(u32, if (up) 1 else 0), n - 1);
            wgt *= if (up) fx[k] else 1 - fx[k];
        }
        if (wgt > 0) {
            const e = lut[(@as(u64, idx[2]) * n + idx[1]) * n + idx[0]];
            inline for (0..3) |k| out[k] += @as(f32, @floatFromInt(e[k])) * (1.0 / 255.0) * wgt;
        }
    }
    return out;
}

/// Farbe eines Quellpixels samt Bloom
inline fn resolvePixel(p: *const types.PostFxParams, i: u64) V4 {
    var col = ld4(p.color, i);

    // Bloom dazu (halbe Auflösung, bilinear hoch)
    if (p.bloom != 0) {
        const bw = p.bloom_width;
        const bh = p.bloom_height;
        const x = (@as(f32, @floatFromInt(i % p.width)) + 0.5) * 0.5 - 0.5;
        const y = (@as(f32, @floatFromInt(i / p.width)) + 0.5) * 0.5 - 0.5;
        const x0: i32 = @intFromFloat(@floor(x));
        const y0: i32 = @intFromFloat(@floor(y));
        const tx = x - @floor(x);
        const ty = y - @floor(y);
        var b: V4 = .{ 0, 0, 0, 0 };
        inline for (0..2) |dy| {
            inline for (0..2) |dx| {
                const s = tap(p.bloom, bw, bh, x0 + @as(i32, dx), y0 + @as(i32, dy), true);
                const wgt = (if (dx == 1) tx else 1 - tx) * (if (dy == 1) ty else 1 - ty);
                inline for (0..3) |k| b[k] += s[k] * wgt;
            }
        }
        inline for (0..3) |k| col[k] += b[k] * p.bloom_strength;
    }

    return col;
}

pub fn resolveFx(p: *const types.PostFxParams, i: u64) void {
    const ss: u32 = @max(p.supersample, 1);
    const ow = if (ss > 1) p.out_width else p.width;
    const oh = if (ss > 1) p.out_height else p.height;
    const c = coords(ow, oh, i) orelse return;

    // Supersampling: über supersample² Quellpixel mitteln. Genau hier
    // entscheidet sich die Deckung einer Voxelkante *innerhalb* eines Frames,
    // statt sie über die Zeit auszumitteln – das ist der einzige Hebel, der
    // gegen das Wandern an Kanten wirklich hilft (gemessen: unruhige Pixel
    // 1,27 % -> 0,24 %). Er kostet allerdings supersample² mal Strahlen.
    var col: V4 = .{ 0, 0, 0, 0 };
    if (ss == 1) {
        col = resolvePixel(p, i);
    } else {
        var n: f32 = 0;
        var sy: u32 = 0;
        while (sy < ss) : (sy += 1) {
            var sx: u32 = 0;
            while (sx < ss) : (sx += 1) {
                const px = c[0] * ss + sx;
                const py = c[1] * ss + sy;
                if (px >= p.width or py >= p.height) continue;
                const s4 = resolvePixel(p, @as(u64, py) * p.width + px);
                inline for (0..3) |k| col[k] += s4[k];
                n += 1;
            }
        }
        if (n > 0) inline for (0..3) |k| {
            col[k] /= n;
        };
    }

    // Belichtung: fest oder aus der Automatik
    var exposure = p.exposure;
    if (p.expose_state != 0) {
        const state: [*]const f32 = @ptrFromInt(p.expose_state);
        exposure = state[1];
    }
    inline for (0..3) |k| col[k] = @max(col[k] * exposure, 0);

    if (p.grade != 0) col = gradeLinear(p, col);

    var disp: [3]f32 = undefined;
    const tm = post.tonemap(.{ @max(col[0], 0), @max(col[1], 0), @max(col[2], 0) }, p.tonemap);
    inline for (0..3) |k| disp[k] = post.linearToSrgb(tm[k]);
    disp = applyLut(p, disp);

    if (p.out_hdr != 0) st4(p.out_hdr, i, .{ col[0], col[1], col[2], 1 });
    if (p.out_ldr != 0) {
        var px: [4]u8 = undefined;
        inline for (0..3) |k| px[if (p.bgra != 0) 2 - k else k] = @intFromFloat(@min(@max(disp[k] * 255.0 + 0.5, 0), 255));
        px[3] = 255;
        @as([*][4]u8, @ptrFromInt(p.out_ldr))[i] = px;
    }
}
