//! Primärstrahlen, Tiefe und Motion Vectors pro Pixel.
//!
//! Motion Vector = Position im Vorframe − Position jetzt, in Pixeln, ohne Jitter.
//! Er wird aus dem exakten Trefferpunkt im Objektraum berechnet: dieselbe
//! Ruheposition wird mit der Vorframe-Transformation und -Kamera projiziert.

const types = @import("types.zig");
const vec = @import("vec.zig");
const tr = @import("trace.zig");
const shade = @import("shade.zig");
const Vec3 = vec.Vec3;

pub const CameraRay = struct { o: Vec3, d: Vec3, tmin: f32, tmax: f32 };

/// Strahl durch die Bildposition (px, py) in Pixeln; t misst Weltdistanz.
pub fn cameraRay(c: *const types.Camera, px: f32, py: f32) CameraRay {
    const nx = 2.0 * px / @as(f32, @floatFromInt(c.width)) - 1.0;
    const ny = 1.0 - 2.0 * py / @as(f32, @floatFromInt(c.height));
    const vx = (nx + c.shift[0]) * c.scale[0];
    const vy = (ny + c.shift[1]) * c.scale[1];
    const ortho = c.projection == types.projection_orthographic;
    const ov: Vec3 = if (ortho) .{ vx, vy, 0 } else .{ 0, 0, 0 };
    const dv: Vec3 = if (ortho) .{ 0, 0, -1 } else .{ vx, vy, -1 };
    const dw = vec.xformVector(&c.view_to_world, dv);
    const len = vec.length(dw);
    return .{
        .o = vec.xformPoint(&c.view_to_world, ov),
        .d = dw * vec.splat(1.0 / len),
        .tmin = c.near_plane * len,
        .tmax = if (c.far_plane > 0) c.far_plane * len else types.flt_max,
    };
}

pub const Projected = struct { x: f32, y: f32, depth: f32 };

inline fn toPixels(c: *const types.CameraData, x: f32, y: f32) [2]f32 {
    const nx = x / c.camera.scale[0] - c.camera.shift[0];
    const ny = y / c.camera.scale[1] - c.camera.shift[1];
    return .{
        (nx + 1.0) * 0.5 * @as(f32, @floatFromInt(c.camera.width)),
        (1.0 - ny) * 0.5 * @as(f32, @floatFromInt(c.camera.height)),
    };
}

/// Weltpunkt -> Bildposition in Pixeln; null hinter der Kamera.
pub fn projectPoint(c: *const types.CameraData, pw: Vec3) ?Projected {
    const v = vec.xformPoint(&c.world_to_view, pw);
    const depth = -v[2];
    var x = v[0];
    var y = v[1];
    if (c.camera.projection != types.projection_orthographic) {
        if (!(depth > 0)) return null;
        x /= depth;
        y /= depth;
    }
    const p = toPixels(c, x, y);
    return .{ .x = p[0], .y = p[1], .depth = depth };
}

/// Richtung (Punkt im Unendlichen) -> Bildposition; nur perspektivisch.
pub fn projectDirection(c: *const types.CameraData, dw: Vec3) ?[2]f32 {
    if (c.camera.projection == types.projection_orthographic) return null;
    const v = vec.xformVector(&c.world_to_view, dw);
    const depth = -v[2];
    if (!(depth > 0)) return null;
    return toPixels(c, v[0] / depth, v[1] / depth);
}

pub const PixelResult = struct {
    hit: types.Hit,
    depth: f32,
    motion: [2]f32,
    /// nur mit Shading (p.color != 0)
    color: [4]f32 = .{ 0, 0, 0, 0 },
    normal: [4]f32 = .{ 0, 0, 0, types.flt_max },
    albedo: [4]f32 = .{ 1, 1, 1, 0 },
    /// Rauheit und Metallanteil (für DLSS Ray Reconstruction)
    material: [2]f32 = .{ 1, 0 },
};

/// Schreibt die angeforderten Ausgaben eines Pixels.
pub fn writePixel(p: *const types.RenderParams, i: u64, r: *const PixelResult) void {
    if (p.hits != 0) @as([*]types.Hit, @ptrFromInt(p.hits))[i] = r.hit;
    if (p.depth != 0) @as([*]f32, @ptrFromInt(p.depth))[i] = r.depth;
    if (p.motion != 0) @as([*][2]f32, @ptrFromInt(p.motion))[i] = r.motion;
    if (p.color != 0) @as([*][4]f32, @ptrFromInt(p.color))[i] = r.color;
    if (p.normal != 0) @as([*][4]f32, @ptrFromInt(p.normal))[i] = r.normal;
    if (p.albedo != 0) @as([*][4]f32, @ptrFromInt(p.albedo))[i] = r.albedo;
    if (p.material != 0) @as([*][2]f32, @ptrFromInt(p.material))[i] = r.material;
}

/// Software-Traversierung (CUDA-Kerne); Alternative: RT-Cores in rt_kernels.zig.
pub const SoftwareTracer = struct {
    pub inline fn trace(_: SoftwareTracer, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?tr.TraceHit {
        return tr.traceScene(s, o, d, tmin, tmax, ray_mask, flags);
    }
};

pub fn renderPixel(p: *const types.RenderParams, s: *const types.Scene, x: u32, y: u32) PixelResult {
    return renderPixelWith(SoftwareTracer{}, p, s, x, y);
}

/// Primärstrahl, Tiefe und Motion Vector eines Pixels. `tracer.trace` liefert
/// den nächsten Treffer samt exakter Ruheposition (p_object).
pub fn renderPixelWith(tracer: anytype, p: *const types.RenderParams, s: *const types.Scene, x: u32, y: u32) PixelResult {
    const cam = &p.cur.camera;
    const px = @as(f32, @floatFromInt(x)) + 0.5 + cam.jitter[0];
    const py = @as(f32, @floatFromInt(y)) + 0.5 + cam.jitter[1];
    const ray = cameraRay(cam, px, py);
    const opaque_mask = p.ray_mask & ~p.transparent_mask;
    // Schatten, GI und Reflexionen dürfen eine andere (gröbere) Auswahl sehen
    const secondary_mask = if (p.secondary_mask != 0) p.secondary_mask else opaque_mask;
    // durchsichtige Voxel überspringt die Traversierung selbst
    const trans_mask = p.ray_mask & p.transparent_mask;
    const found = tracer.trace(s, ray.o, ray.d, ray.tmin, ray.tmax, opaque_mask, p.flags | types.trace_skip_transparent);

    var r = PixelResult{ .hit = tr.toHit(found), .depth = types.flt_max, .motion = .{ 0, 0 } };
    const history = p.history_valid != 0;

    if (found) |h| {
        const pw = ray.o + ray.d * vec.splat(h.t);
        r.depth = -vec.xformPoint(&p.cur.world_to_view, pw)[2];

        const cur = &tr.instances(s)[h.instance];
        const prev = &tr.instancesPrev(s)[h.instance];
        const fresh = prev.flags & types.instance_active == 0 or prev.history != cur.history;
        if (fresh and cur.flags & types.instance_keep_history == 0) r.hit.meta |= types.hit_new;

        if (history) {
            // Ruheposition mit dem Vorframe-Zustand abbilden; ohne Vorgeschichte nur Kamerabewegung
            const pw_prev = if (fresh) pw else vec.xformPoint(&prev.object_to_world, h.p_object);
            if (projectPoint(&p.prev, pw_prev)) |q| r.motion = .{ q.x - px, q.y - py };
        }
    } else if (history) {
        if (projectDirection(&p.prev, ray.d)) |q| r.motion = .{ q[0] - px, q[1] - py };
    }
    if (!history) r.hit.meta |= types.hit_no_history;

    if (p.color != 0 or p.normal != 0 or p.albedo != 0) {
        if (found) |h| {
            var rng = shade.Rng.init(x, y, p.frame_index, 0);
            // Größe eines Bildschirmpixels in Welteinheiten am Treffer.
            // Daraus wählt das Shading die Verkleinerungsstufe der Texturen
            // und blendet die Detailnormale aus, bevor sie flimmern kann.
            const cm = &p.cur.camera;
            const px_per_unit = 2 * cm.scale[1] / @as(f32, @floatFromInt(@max(cm.height, 1)));
            const footprint = if (cm.projection == types.projection_orthographic) px_per_unit else h.t * px_per_unit;
            const sh = shade.shadeHit(tracer, s, ray.o, ray.d, h, &rng, secondary_mask, trans_mask, footprint);
            r.color = .{ sh.color[0], sh.color[1], sh.color[2], 1 };
            r.normal = .{ sh.normal[0], sh.normal[1], sh.normal[2], r.depth };
            // w trägt den diffusen Anteil für den indirekten Durchgang
            r.albedo = .{ sh.albedo[0], sh.albedo[1], sh.albedo[2], sh.diffuse };
            r.material = .{ sh.roughness, 1 - sh.diffuse };
        } else {
            const c = shade.sky(@ptrFromInt(s.lighting), ray.d, true);
            r.color = .{ c[0], c[1], c[2], 0 };
        }
        // Transparente Flächen (mehrere, mit Brechung) vor dem Untergrund
        if (p.color != 0) {
            const t_behind = if (found) |h| h.t else ray.tmax;
            var rng = shade.Rng.init(x, y, p.frame_index, 1);
            const behind = vec.Vec3{ r.color[0], r.color[1], r.color[2] };
            const cm2 = &p.cur.camera;
            const fp_scale = 2 * cm2.scale[1] / @as(f32, @floatFromInt(@max(cm2.height, 1)));
            var c = shade.transparentLayers(tracer, s, ray.o, ray.d, ray.tmin, t_behind, behind, p.ray_mask, trans_mask, opaque_mask, secondary_mask, &rng, fp_scale);
            // Nebel ganz zum Schluss: er dämpft alles dahinter, auch die
            // transparenten Schichten, und steuert die Lichtschächte bei.
            const lg: *const types.Lighting = @ptrFromInt(s.lighting);
            if (shade.hasFog(lg)) {
                var frng = shade.Rng.init(x, y, p.frame_index, 3);
                const cv = vec.Vec3{ c[0], c[1], c[2] };
                const fogged = shade.applyFog(tracer, s, lg, ray.o, ray.d, t_behind, cv, &frng, secondary_mask);
                c = fogged;
            }
            r.color = .{ c[0], c[1], c[2], r.color[3] };
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Indirekte Beleuchtung in halber Auflösung (lighting_gi_half)
//
// 1. Hauptdurchgang: alles außer der indirekten Beleuchtung; albedo.w trägt
//    den diffusen Anteil (1 − metallic).
// 2. giPixel: je 2x2-Block ein Strahl, Abtastpunkt wandert mit dem Frame.
// 3. combinePixel: kantenbewusst hochskalieren (Normale, Tiefe) und dazurechnen.
// ---------------------------------------------------------------------------

const H4 = [4]f16;

inline fn giStore(ptr: u64, i: u64, v: [4]f32) void {
    var h: H4 = undefined;
    inline for (0..4) |k| h[k] = @floatCast(@min(@max(v[k], -65504.0), 65504.0));
    @as([*]H4, @ptrFromInt(ptr))[i] = h;
}

inline fn giLoad(ptr: u64, i: u64) [4]f32 {
    const h = @as([*]const H4, @ptrFromInt(ptr))[i];
    return .{ h[0], h[1], h[2], h[3] };
}

/// Abtastpixel eines 2x2-Blocks; der Versatz wandert mit dem Frame.
/// (Ein fester Versatz wurde gemessen: am Flimmern ändert er nichts – 0,887
/// gegen 0,891 –, kostet aber die Abdeckung über die Zeit.)
pub inline fn giSample(p: *const types.RenderParams, hx: u32, hy: u32) [2]u32 {
    const ox = p.frame_index & 1;
    const oy = (p.frame_index >> 1) & 1;
    return .{
        @min(hx * 2 + ox, p.cur.camera.width - 1),
        @min(hy * 2 + oy, p.cur.camera.height - 1),
    };
}

/// Ein Strahl je 2x2-Block: Treffer und Normale kommen aus dem Hauptdurchgang.
pub fn giPixel(tracer: anytype, p: *const types.RenderParams, s: *const types.Scene, hx: u32, hy: u32) void {
    const gi_index = @as(u64, hy) * p.gi_width + hx;
    const c = giSample(p, hx, hy);
    const i = @as(u64, c[1]) * p.cur.camera.width + c[0];
    const hit = @as([*]const types.Hit, @ptrFromInt(p.hits))[i];
    if (hit.instance == types.no_hit) {
        giStore(p.gi, gi_index, .{ 0, 0, 0, 0 });
        return;
    }
    const nw = @as([*]const [4]f32, @ptrFromInt(p.normal))[i];
    const n = Vec3{ nw[0], nw[1], nw[2] };
    const cam = &p.cur.camera;
    const ray = cameraRay(cam, @as(f32, @floatFromInt(c[0])) + 0.5 + cam.jitter[0], @as(f32, @floatFromInt(c[1])) + 0.5 + cam.jitter[1]);
    const l: *const types.Lighting = @ptrFromInt(s.lighting);
    const inst = &tr.instances(s)[hit.instance];
    const pw = ray.o + ray.d * vec.splat(hit.t) + n * vec.splat(1e-3 * shade.voxelSize(inst));
    const ps = if (l.secondary_bias > 0) pw + n * vec.splat(l.secondary_bias * hit.t) else pw;
    const opaque_mask = p.ray_mask & ~p.transparent_mask;
    const mask = if (p.secondary_mask != 0) p.secondary_mask else opaque_mask;
    var rng = shade.Rng.init(c[0], c[1], p.frame_index, 2);
    const li = shade.indirect(tracer, s, l, ps, n, &rng, mask, p.ray_mask & p.transparent_mask);
    giStore(p.gi, gi_index, .{ li[0], li[1], li[2], 1 });
}

/// Hochskalieren und dazurechnen: Gewichte aus Normale und Tiefe, damit die
/// indirekte Beleuchtung nicht über Kanten läuft.
pub fn combinePixel(p: *const types.RenderParams, x: u32, y: u32) void {
    const w = p.cur.camera.width;
    const i = @as(u64, y) * w + x;
    const nw = @as([*]const [4]f32, @ptrFromInt(p.normal))[i];
    if (!(nw[3] < types.flt_max)) return; // Himmel
    const n = Vec3{ nw[0], nw[1], nw[2] };
    const depth = nw[3];

    const fx = (@as(f32, @floatFromInt(x)) - 0.5) * 0.5;
    const fy = (@as(f32, @floatFromInt(y)) - 0.5) * 0.5;
    const bx: i32 = @intFromFloat(@floor(fx));
    const by: i32 = @intFromFloat(@floor(fy));
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);

    var sum: Vec3 = @splat(0);
    var wsum: f32 = 0;
    var best: Vec3 = @splat(0);
    var best_w: f32 = -1;
    inline for (0..4) |k| {
        const hx = bx + @as(i32, k & 1);
        const hy = by + @as(i32, k >> 1);
        if (hx >= 0 and hy >= 0 and hx < p.gi_width and hy < p.gi_height) {
            const g = giLoad(p.gi, @as(u64, @intCast(hy)) * p.gi_width + @as(u32, @intCast(hx)));
            if (g[3] > 0) {
                const c = giSample(p, @intCast(hx), @intCast(hy));
                const sn = @as([*]const [4]f32, @ptrFromInt(p.normal))[@as(u64, c[1]) * w + c[0]];
                const nd = @max(n[0] * sn[0] + n[1] * sn[1] + n[2] * sn[2], 0);
                const n4 = nd * nd * nd * nd;
                const wd = 1.0 / (1.0 + 20.0 * @abs(sn[3] - depth) / @max(depth, 1e-3));
                const bw = (if (k & 1 != 0) tx else 1 - tx) * (if (k >> 1 != 0) ty else 1 - ty);
                const wgt = bw * n4 * n4 * wd;
                const li = Vec3{ g[0], g[1], g[2] };
                sum += li * vec.splat(wgt);
                wsum += wgt;
                if (n4 * wd > best_w) {
                    best_w = n4 * wd;
                    best = li;
                }
            }
        }
    }
    if (best_w < 0) return; // kein gültiger Nachbar
    const li = if (wsum > 1e-4) sum * vec.splat(1.0 / wsum) else best;
    const alb = @as([*]const [4]f32, @ptrFromInt(p.albedo))[i];
    const add = Vec3{ alb[0], alb[1], alb[2] } * vec.splat(alb[3]) * li;
    const col = @as([*][4]f32, @ptrFromInt(p.color));
    col[i] = .{ col[i][0] + add[0], col[i][1] + add[1], col[i][2] + add[2], col[i][3] };
}
