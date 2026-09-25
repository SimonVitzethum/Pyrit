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

/// Verlauf bikubisch (Catmull-Rom) an (px, py) in Pixelmitten-Koordinaten
/// lesen. Bilineares Nachschlagen glättet bei jeder Bewegung ein wenig, und
/// weil der Verlauf Frame für Frame neu nachgeschlagen wird, summiert sich das
/// zu deutlicher Unschärfe (gemessen: rund 40 % weniger Schärfe nach zehn
/// Frames Flug). Catmull-Rom erhält die Details. Randpixel werden geklemmt.
fn historyCubic(p: *const types.PostParams, px: f32, py: f32) [3]f32 {
    const fx0 = @floor(px);
    const fy0 = @floor(py);
    const tx = px - fx0;
    const ty = py - fy0;
    const wx = [4]f32{
        tx * (-0.5 + tx * (1 - 0.5 * tx)),
        1 + tx * tx * (-2.5 + 1.5 * tx),
        tx * (0.5 + tx * (2 - 1.5 * tx)),
        tx * tx * (-0.5 + 0.5 * tx),
    };
    const wy = [4]f32{
        ty * (-0.5 + ty * (1 - 0.5 * ty)),
        1 + ty * ty * (-2.5 + 1.5 * ty),
        ty * (0.5 + ty * (2 - 1.5 * ty)),
        ty * ty * (-0.5 + 0.5 * ty),
    };
    const ix: i32 = @intFromFloat(fx0);
    const iy: i32 = @intFromFloat(fy0);
    const wm: i32 = @intCast(p.width - 1);
    const hm: i32 = @intCast(p.height - 1);
    var out: [3]f32 = .{ 0, 0, 0 };
    inline for (0..4) |b| {
        const sy: u64 = @intCast(@min(@max(iy + @as(i32, b) - 1, 0), hm));
        inline for (0..4) |a| {
            const sx: u64 = @intCast(@min(@max(ix + @as(i32, a) - 1, 0), wm));
            const c = ldh(p.hist_color, sy * p.width + sx);
            const w = wx[a] * wy[b];
            inline for (0..3) |k| out[k] += c[k] * w;
        }
    }
    return out;
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
        // Alle vier Nachbarn gültig und auf derselben Fläche? Dann darf
        // bikubisch nachgeschlagen werden (siehe historyCubic). Auch über
        // Flächenkanten hinweg wurde gemessen: schlechter (Fehler gegen die
        // Referenz 3,8 -> 4,3 %).
        var all_same: u32 = 0;
        var lo: [3]f32 = .{ 65504, 65504, 65504 };
        var hi: [3]f32 = .{ -65504, -65504, -65504 };
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
                if (vw >= 0.999 and same_face) all_same += 1;
                if (vw > 0) {
                    const c = ldh(p.hist_color, j);
                    inline for (0..3) |q| {
                        lo[q] = @min(lo[q], c[q]);
                        hi[q] = @max(hi[q], c[q]);
                    }
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
            // Nur wenn sich wirklich etwas bewegt: bei stehendem Bild trifft
            // die Reprojektion die Pixelmitte, dann ist bilinear exakt.
            const moving = @abs(fx - @round(fx)) > 0.01 or @abs(fy - @round(fy)) > 0.01;
            if (all_same == 4 and moving and hit) {
                const cr = historyCubic(p, px, py);
                // Überschwinger auf den Bereich der vier Nachbarn begrenzen
                inline for (0..3) |q| hist[q] = @min(@max(cr[q], lo[q]), hi[q]);
            }
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
            // Varianzbegrenzung gegen Nachziehen (TAA) – nur bei Bewegung.
            //
            // Steht das Bild, ist die Reprojektion exakt und der Verlauf kann
            // gar nicht nachziehen. Die Grenze misst sich dann aber an der
            // 3x3-Umgebung *eines* verrauschten Frames: sie reißt den gut
            // gemittelten Verlauf jeden Frame ein Stück zum Rauschen zurück.
            // Gemessen war das bei stehender Kamera die Hauptquelle der
            // Unruhe (0,48 % unruhige Pixel mit Grenze, 0,015 % ohne). Deshalb
            // öffnet sie sich unter 1/10 Pixel Bewegung ganz und greift erst ab
            // einem Pixel je Frame voll.
            const mv = @as([*]const [2]f32, @ptrFromInt(p.motion))[i];
            const speed = @sqrt(mv[0] * mv[0] + mv[1] * mv[1]);
            const t = @min(@max((speed - 0.1) / 0.9, 0), 1);
            if (t > 0) {
                const sigma = p.clamp_sigma / t;
                inline for (0..3) |k| {
                    hist[k] = @min(@max(hist[k], mean[k] - sigma * sd[k]), mean[k] + sigma * sd[k]);
                }
            }
        }
        // Stammt der Verlauf von einer anderen Fläche, liegt das Pixel auf
        // einer Voxelkante: der Jitter schiebt die Abtastung jeden Frame hin
        // und her. Den Verlauf zu verwerfen lässt das Pixel zwischen beiden
        // Flächen springen, ihn voll zu übernehmen zieht bei Bewegung nach.
        //
        // Entscheidend ist die Bewegung: steht das Bild, ist die Reprojektion
        // exakt und eine lange Mittelung völlig unbedenklich – sie ergibt
        // genau den Deckungsgrad der beiden Flächen, also saubere
        // Kantenglättung. Erst bei Bewegung muss sie kurz werden.
        var limit = 1.0 / @max(p.alpha_min, 1e-4);
        const mv = @as([*]const [2]f32, @ptrFromInt(p.motion))[i];
        const speed = @sqrt(mv[0] * mv[0] + mv[1] * mv[1]);
        // Steht das Bild, ist die Reprojektion exakt: dann darf der Verlauf
        // länger werden (bis gut dreimal so lang). Das Restrauschen der
        // indirekten Beleuchtung sinkt damit um fast die Hälfte, ohne dass
        // bei Bewegung irgendetwas nachzieht – dort gilt wieder die Vorgabe.
        const still = 1 - @min(@max((speed - 0.02) / 0.1, 0), 1);
        limit *= 1 + 2.2 * still;
        // Schnelle Bewegung: jedes Nachschlagen im Verlauf glättet ein wenig,
        // lange Verläufe summieren das zu Unschärfe. Ab 2 Pixeln je Frame
        // bleibt deshalb kaum Verlauf, dort trägt der räumliche Filter.
        // Gemessen gegen die eingeschwungene Referenz (Flug, 10 Frames):
        // halbiert 9,1 % Abweichung und 19 % Schärfeverlust, auf 5 % gekürzt
        // 6,6 % und 6 % – das Rauschen im Einzelframe ist kleiner als der
        // Fehler, den ein langer, verwischter Verlauf mitbringt.
        const fast = @min(@max((speed - 0.5) / 1.5, 0), 1);
        limit = @max(limit * (1 - 0.95 * fast), 8);
        // Hinter Wasser und Glas: Spiegelung und Brechung bewegen sich nicht
        // mit dem Untergrund, nach dessen Motion Vector reprojiziert wird, und
        // Wellen ändern sich mit der Zeit. Lange gemittelt verschmierten die
        // Spiegelungen; die Spiegelung selbst rauscht kaum, fünf Frames genügen.
        if (meta & types.hit_through_transparent != 0) limit = @min(limit, 5);
        if (edge_w > 0.25) {
            // unter 1/10 Pixel Bewegung: volle Mittelung, darüber gleitend
            // hinunter auf edge_frames
            const t = @min(@max((speed - 0.1) / 0.9, 0), 1);
            limit = @min(limit, limit + (p.edge_frames - limit) * t);
        }
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

/// Tonemapping der ganzen Farbe. Die kanalweisen Kurven übersättigen helle
/// Farben (ein grelles Grün bleibt grell, bis ein Kanal abschneidet); die
/// angepasste ACES-Kurve mischt über ihre Eingangs- und Ausgangsmatrix zum
/// Weiß hin, wie Film es tut.
pub fn tonemap(c: [3]f32, mode: u32) [3]f32 {
    if (mode == types.tonemap_neutral) return tonemapNeutral(c);
    if (mode != types.tonemap_aces_fitted) {
        return .{ tonemapChannel(c[0], mode), tonemapChannel(c[1], mode), tonemapChannel(c[2], mode) };
    }
    const in_m = [3][3]f32{ .{ 0.59719, 0.35458, 0.04823 }, .{ 0.07600, 0.90834, 0.01566 }, .{ 0.02840, 0.13383, 0.83777 } };
    const out_m = [3][3]f32{ .{ 1.60475, -0.53108, -0.07367 }, .{ -0.10208, 1.10813, -0.00605 }, .{ -0.00327, -0.07276, 1.07602 } };
    var v: [3]f32 = undefined;
    inline for (0..3) |r| v[r] = in_m[r][0] * c[0] + in_m[r][1] * c[1] + in_m[r][2] * c[2];
    inline for (0..3) |k| {
        const x = @max(v[k], 0);
        v[k] = (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510) + 0.238081);
    }
    var o: [3]f32 = undefined;
    inline for (0..3) |r| o[r] = @min(@max(out_m[r][0] * v[0] + out_m[r][1] * v[1] + out_m[r][2] * v[2], 0), 1);
    return o;
}

/// Khronos PBR Neutral. ACES hat einen starken Fuß: ein Schatten, der im
/// Licht ein Fünftel der besonnten Fläche hat, landete auf dem Schirm bei
/// einem Achtundzwanzigstel – fast schwarz. Diese Kurve lässt alles unter
/// 0,76 unverändert und rollt nur die Lichter ab.
fn tonemapNeutral(c: [3]f32) [3]f32 {
    const start = 0.8 - 0.04;
    const desat = 0.15;
    const x = @min(@min(c[0], c[1]), c[2]);
    const offset = if (x < 0.08) x - 6.25 * x * x else 0.04;
    var v = [3]f32{ c[0] - offset, c[1] - offset, c[2] - offset };
    const peak = @max(@max(v[0], v[1]), v[2]);
    if (peak < start) return .{ @max(v[0], 0), @max(v[1], 0), @max(v[2], 0) };
    const d = 1 - start;
    const new_peak = 1 - d * d / (peak + d - start);
    inline for (0..3) |k| v[k] *= new_peak / peak;
    const g = 1 - 1 / (desat * (peak - new_peak) + 1);
    inline for (0..3) |k| v[k] = @min(@max(v[k] + (new_peak - v[k]) * g, 0), 1);
    return v;
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
        const tm = tonemap(.{ @max(hdr[0] * p.exposure, 0), @max(hdr[1] * p.exposure, 0), @max(hdr[2] * p.exposure, 0) }, p.tonemap);
        inline for (0..3) |k| {
            const v = linearToSrgb(tm[k]);
            px[if (p.bgra != 0) 2 - k else k] = @intFromFloat(@min(@max(v * 255.0 + 0.5, 0), 255));
        }
        px[3] = 255;
        @as([*][4]u8, @ptrFromInt(p.out_ldr))[i] = px;
    }
}
