//! OptiX-Programme (RT-Cores). Wird für nvptx64 übersetzt und von OptiX als
//! Modul geladen.
//!
//! - Hardware-BVH: Instanzen (IAS) und AABBs der belegten Teilbäume (GAS)
//! - __intersection__dag: verfolgt den Teil-DAG im getroffenen AABB
//! - __closesthit__dag:   schreibt t, Instanz, Attribut, Fläche in die Payload
//! - __raygen__render:    Primärstrahlen, Tiefe, Motion Vectors (wie der CUDA-Pfad)
//! - __raygen__gi:        indirekte Beleuchtung in halber Auflösung
//! - __raygen__trace:     beliebige Strahlen (pyr_trace)

const std = @import("std");
const pyr = @import("pyrit_device");
const types = pyr.types;
const vec = pyr.vec;
const ox = pyr.optix;
const Vec3 = vec.Vec3;

pub const panic = std.debug.no_panic;

/// Launch-Parameter im Konstantenspeicher. Zig erlaubt dort nur `extern const`;
/// tools/ptx_fixup.zig macht daraus die Definition, die OptiX erwartet.
extern const pyr_rt_params: types.RtParams addrspace(.constant);

inline fn params() *const types.RtParams {
    return @addrSpaceCast(&pyr_rt_params);
}

/// Größtes tmax, das an die Hardware geht
const tmax_limit: f32 = 1e16;

const RtTracer = struct {
    handle: u64,
    flags: u32,

    pub inline fn trace(self: RtTracer, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?pyr.scene.TraceHit {
        if (self.handle == 0) return null;
        // p[3] trägt hin die Absicht "durchsichtige Voxel überspringen" und
        // zurück die getroffene Fläche
        var p = [4]u32{ @bitCast(types.flt_max), types.no_hit, 0, @intFromBool((self.flags | flags) & types.trace_skip_transparent != 0) };
        var rf = ox.ray_flag_disable_anyhit;
        if ((self.flags | flags) & types.trace_any_hit != 0) rf |= ox.ray_flag_terminate_on_first_hit;
        ox.trace4(self.handle, o, d, tmin, @min(tmax, tmax_limit), ray_mask & 0xFF, rf, 0, 1, 0, &p);
        if (p[1] == types.no_hit) return null;
        const t: f32 = @bitCast(p[0]);
        const inst = &pyr.scene.instances(s)[p[1]];
        return .{
            .t = t,
            .instance = p[1],
            .attribute = p[2],
            .face = p[3],
            .voxel = .{ 0, 0, 0 },
            // Ruheposition aus dem Weltpunkt (für den Motion Vector)
            .p_object = vec.xformPoint(&inst.world_to_object, o + d * vec.splat(t)),
        };
    }
};

export fn __raygen__render() callconv(.nvptx_kernel) void {
    const prm = params();
    const idx = ox.launchIndex();
    const rp = &prm.render;
    const s: *const types.Scene = @ptrFromInt(prm.scene);
    const r = pyr.render.renderPixelWith(RtTracer{ .handle = prm.handle, .flags = prm.flags }, rp, s, idx[0], idx[1]);

    pyr.render.writePixel(rp, @as(u64, idx[1]) * rp.cur.camera.width + idx[0], &r);
}

/// Indirekte Beleuchtung in halber Auflösung (ein Strahl je 2x2-Block)
export fn __raygen__gi() callconv(.nvptx_kernel) void {
    const prm = params();
    const idx = ox.launchIndex();
    const rp = &prm.render;
    if (idx[0] >= rp.gi_width or idx[1] >= rp.gi_height) return;
    const s: *const types.Scene = @ptrFromInt(prm.scene);
    pyr.render.giPixel(RtTracer{ .handle = prm.handle, .flags = prm.flags }, rp, s, idx[0], idx[1]);
}

export fn __raygen__trace() callconv(.nvptx_kernel) void {
    const prm = params();
    const i = ox.launchIndex()[0];
    const tp = &prm.trace;
    if (i >= tp.count) return;
    const s: *const types.Scene = @ptrFromInt(prm.scene);
    const ray = @as([*]const types.Ray, @ptrFromInt(tp.rays))[i];
    const tracer = RtTracer{ .handle = prm.handle, .flags = prm.flags };
    const h = tracer.trace(s, ray.origin, ray.direction, ray.tmin, ray.tmax, tp.ray_mask, tp.flags);
    if (tp.flags & types.trace_extended != 0) {
        @as([*]types.HitEx, @ptrFromInt(tp.hits))[i] = pyr.scene.extendedHit(s, ray.origin, ray.direction, h);
    } else {
        @as([*]types.Hit, @ptrFromInt(tp.hits))[i] = pyr.scene.toHit(h);
    }
}

export fn __intersection__dag() callconv(.nvptx_kernel) void {
    const g: *const types.RtGeometry = @ptrFromInt(ox.sbtDataPointer());
    const prim = @as([*]const types.RtPrim, @ptrFromInt(g.prims))[ox.primitiveIndex()];
    const want_attribute = params().flags & types.trace_no_attribute == 0;
    const s: *const types.Scene = @ptrFromInt(params().scene);
    const skip: ?*const [4]u64 = if (ox.getPayload(3) != 0) &s.transparent_materials else null;
    const h = pyr.rt.traceSubtree(g, prim, ox.objectRayOrigin(), ox.objectRayDirection(), ox.rayTmin(), ox.rayTmax(), want_attribute, skip) orelse return;
    const kind = h.face | (if (h.inside) types.hit_inside else 0);
    _ = ox.reportIntersection1(h.t, kind, h.attribute);
}

export fn __closesthit__dag() callconv(.nvptx_kernel) void {
    ox.setPayload(0, @bitCast(ox.rayTmax()));
    ox.setPayload(1, ox.instanceId());
    ox.setPayload(2, ox.attribute0());
    ox.setPayload(3, ox.hitKind());
}

export fn __miss__none() callconv(.nvptx_kernel) void {}
