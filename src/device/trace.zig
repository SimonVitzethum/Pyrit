//! Strahlverfolgung durch die ganze Szene (alle Instanzen).

const std = @import("std");
const types = @import("types.zig");
const vec = @import("vec.zig");
const dag = @import("dag.zig");
const Vec3 = vec.Vec3;

pub const TraceHit = struct {
    t: f32,
    instance: u32,
    attribute: u32,
    /// face_* im Objektraum | hit_inside
    face: u32,
    /// Voxelkoordinate im Objektraum
    voxel: [3]i32,
    /// exakter Trefferpunkt im Objektraum (= Ruheposition)
    p_object: Vec3,
};

pub inline fn instances(s: *const types.Scene) [*]const types.InstanceData {
    return @ptrFromInt(s.instances);
}

pub inline fn instancesPrev(s: *const types.Scene) [*]const types.InstanceData {
    return @ptrFromInt(s.instances_prev);
}

pub inline fn geometries(s: *const types.Scene) [*]const types.GeometryData {
    return @ptrFromInt(s.geometries);
}

pub fn dagOf(s: *const types.Scene, g: *const types.GeometryData) dag.Dag {
    const nodes: [*]const u32 = @ptrFromInt(s.nodes);
    const leaves: [*]const u64 = @ptrFromInt(s.leaves);
    const attributes: ?[*]const u32 = if (g.flags & types.geometry_has_attributes != 0)
        @as([*]const u32, @ptrFromInt(s.attributes)) + g.attribute_offset
    else
        null;
    return .{
        .nodes = nodes + g.node_offset,
        .leaves = leaves + g.leaf_offset,
        .attributes = attributes,
        .root = g.root,
        .log2_size = g.log2_size,
        .default_attribute = g.default_attribute,
    };
}

/// Kehrwert ohne Division durch 0; Vorzeichen bleibt erhalten
inline fn safeRcp(x: f32) f32 {
    return 1.0 / (if (@abs(x) < 1e-30) (if (x < 0) @as(f32, -1e-30) else 1e-30) else x);
}

/// Schnittintervall mit einer AABB oder null
inline fn rayAabb(o: Vec3, inv: Vec3, bmin: [3]f32, bmax: [3]f32) ?[2]f32 {
    const lo: Vec3 = (@as(Vec3, bmin) - o) * inv;
    const hi: Vec3 = (@as(Vec3, bmax) - o) * inv;
    const t0 = @reduce(.Max, @min(lo, hi));
    const t1 = @reduce(.Min, @max(lo, hi));
    return if (t0 <= t1) .{ t0, t1 } else null;
}

/// Nächster Treffer (bzw. irgendein Treffer mit trace_any_hit) im Intervall
/// [tmin, tmax). Nur Instanzen mit (mask & ray_mask) != 0 werden getroffen.
/// Derzeit linear über alle Instanzen mit AABB-Test; die Instanz-BVH folgt.
pub fn traceScene(s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?TraceHit {
    const inst = instances(s);
    const geo = geometries(s);
    const inv = Vec3{ safeRcp(d[0]), safeRcp(d[1]), safeRcp(d[2]) };
    const want_attribute = flags & types.trace_no_attribute == 0;
    var best = tmax;
    var result: ?TraceHit = null;

    var i: u32 = 0;
    while (i < s.instance_count) : (i += 1) {
        const in = &inst[i];
        if (in.flags & types.instance_active == 0 or in.mask & ray_mask == 0) continue;
        const range = rayAabb(o, inv, in.bounds_min, in.bounds_max) orelse continue;
        if (!(@max(range[0], tmin) < @min(range[1], best))) continue;

        const g = dagOf(s, &geo[in.geometry]);
        const oo = vec.xformPoint(&in.world_to_object, o);
        const dd = vec.xformVector(&in.world_to_object, d);
        // durchsichtige Materialien überspringt schon die Traversierung
        const hit = if (flags & types.trace_skip_transparent != 0)
            dag.traceSkipping(&g, oo, dd, tmin, best, want_attribute, &s.transparent_materials)
        else
            dag.trace(&g, oo, dd, tmin, best, want_attribute);
        if (hit) |h| {
            best = h.t;
            result = .{
                .t = h.t,
                .instance = i,
                .attribute = h.attribute,
                .face = h.face | (if (h.inside) types.hit_inside else 0),
                .voxel = h.voxel,
                .p_object = oo + dd * vec.splat(h.t),
            };
            if (flags & types.trace_any_hit != 0) return result;
        }
    }
    return result;
}

pub fn toHit(r: ?TraceHit) types.Hit {
    if (r) |h| return .{ .t = h.t, .instance = h.instance, .attribute = h.attribute, .meta = h.face };
    return .{ .t = types.flt_max, .instance = types.no_hit, .attribute = 0, .meta = 0 };
}

/// Öffentliche Gerätefunktion für eigene Kernel
pub fn trace(s: *const types.Scene, ray: types.Ray, ray_mask: u32, flags: u32) types.Hit {
    return toHit(traceScene(s, ray.origin, ray.direction, ray.tmin, ray.tmax, ray_mask, flags));
}

/// Erweiterter Treffer: Voxel aus der exakten Ruheposition (auf der Fläche
/// einen halben Voxel nach innen), damit CUDA- und RT-Pfad gleich rechnen.
pub fn extendedHit(s: *const types.Scene, o: Vec3, d: Vec3, found: ?TraceHit) types.HitEx {
    var r = std.mem.zeroes(types.HitEx);
    r.hit = toHit(found);
    const h = found orelse return r;
    const face = h.face & types.hit_face_mask;
    const axis = face >> 1;
    const inside = h.face & types.hit_inside != 0;
    const po: [3]f32 = h.p_object;
    inline for (0..3) |a| {
        var p = po[a];
        if (!inside and a == axis) p += if (face & 1 != 0) 0.5 else -0.5;
        r.voxel[a] = @intFromFloat(@floor(p));
    }
    const pw = o + d * vec.splat(h.t);
    r.position = pw;
    r.normal = hitNormal(s, r.hit);
    return r;
}

/// Weltnormale eines Treffers (aktueller Frame)
pub fn hitNormal(s: *const types.Scene, h: types.Hit) Vec3 {
    const face = h.meta & types.hit_face_mask;
    const sign: f32 = if (face & 1 != 0) -1 else 1;
    const axis = face >> 1;
    const n = Vec3{
        if (axis == 0) sign else 0,
        if (axis == 1) sign else 0,
        if (axis == 2) sign else 0,
    };
    return vec.normalize(vec.xformNormal(&instances(s)[h.instance].world_to_object, n));
}
