//! Gemeinsame Hilfen der Tests: Testszene, Referenz-Traversierung, Host-Szene.

const std = @import("std");
const pyr = @import("pyrit_device");
const pyrit = @import("pyrit");
const types = pyr.types;

pub const api = pyrit.api;

// ---------------------------------------------------------------------------
// Testszene
// ---------------------------------------------------------------------------

pub fn attrOf(x: i32, y: i32, z: i32) u32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 73856093 ^ @as(u32, @bitCast(y)) *% 19349663 ^ @as(u32, @bitCast(z)) *% 83492791;
    h ^= h >> 13;
    h *%= 0x5bd1e995;
    return (h ^ (h >> 15)) | 1;
}

pub const SceneFn = struct { n: i32 };

/// Kugel, Boden, dünne Säulen und verstreute Einzelvoxel; Attribut = attrOf.
pub fn sceneVoxel(user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32 {
    const n: f32 = @floatFromInt(@as(*SceneFn, @ptrCast(@alignCast(user.?))).n);
    const c = n * 0.5;
    const dx = @as(f32, @floatFromInt(x)) + 0.5 - c * 0.8;
    const dy = @as(f32, @floatFromInt(y)) + 0.5 - c;
    const dz = @as(f32, @floatFromInt(z)) + 0.5 - c;
    var in = dx * dx + dy * dy + dz * dz < (n * 0.3) * (n * 0.3);
    in = in or y < 3;
    in = in or (@mod(x, 16) == 3 and @mod(z, 16) == 5);
    in = in or attrOf(x, y, z) % 97 == 0;
    return if (in) attrOf(x, y, z) else 0;
}

pub fn denseScene(gpa: std.mem.Allocator, n: usize) ![]u32 {
    var f = SceneFn{ .n = @intCast(n) };
    const v = try gpa.alloc(u32, n * n * n);
    for (v, 0..) |*e, i| {
        e.* = sceneVoxel(&f, @intCast(i % n), @intCast((i / n) % n), @intCast(i / (n * n)));
    }
    return v;
}

pub const HostDag = struct {
    dag: pyrit.dag_builder.Dag,
    geometry: types.GeometryData,

    pub fn deinit(self: *HostDag, gpa: std.mem.Allocator) void {
        self.dag.deinit(gpa);
    }

    pub fn device(self: *const HostDag) pyr.dag.Dag {
        return .{
            .nodes = self.dag.nodes.ptr,
            .leaves = self.dag.leaves.ptr,
            .attributes = if (self.dag.attributes) |a| a.ptr else null,
            .root = self.dag.root,
            .log2_size = self.dag.log2_size,
            .default_attribute = 1,
        };
    }
};

pub fn buildSceneDag(gpa: std.mem.Allocator, log2: u32) !HostDag {
    var f = SceneFn{ .n = @as(i32, 1) << @intCast(log2) };
    const dag = try pyrit.dag_builder.buildFn(gpa, log2, sceneVoxel, null, &f, true);
    return .{
        .dag = dag,
        .geometry = .{
            .node_offset = 0,
            .leaf_offset = 0,
            .attribute_offset = 0,
            .root = dag.root,
            .log2_size = dag.log2_size,
            .flags = types.geometry_has_attributes,
            .default_attribute = 1,
            .reserved = 0,
        },
    };
}

/// Szene mit Host-Zeigern; dieselbe Struktur wie auf der GPU.
pub fn hostScene(h: *const HostDag, geos: [*]const types.GeometryData, cur: [*]const types.InstanceData, prev: [*]const types.InstanceData, count: u32) types.Scene {
    var s = std.mem.zeroes(types.Scene);
    s.nodes = @intFromPtr(h.dag.nodes.ptr);
    s.leaves = @intFromPtr(h.dag.leaves.ptr);
    s.attributes = if (h.dag.attributes) |a| @intFromPtr(a.ptr) else 0;
    s.geometries = @intFromPtr(geos);
    s.instances = @intFromPtr(cur);
    s.instances_prev = @intFromPtr(prev);
    s.instance_count = count;
    s.geometry_count = 1;
    return s;
}

// ---------------------------------------------------------------------------
// Referenz: Amanatides-Woo-DDA auf dem dichten Gitter, in f64
// ---------------------------------------------------------------------------

pub const RefHit = struct { t: f64, v: [3]i32, face: i32 };

pub fn refTrace(g: []const u32, n: usize, o: [3]f64, d: [3]f64, tmin: f64, tmax: f64) ?RefHit {
    const nf: f64 = @floatFromInt(n);
    var t0 = tmin;
    var t1 = tmax;
    var entry: i32 = -1;
    for (0..3) |a| {
        if (d[a] == 0) {
            if (o[a] < 0 or o[a] >= nf) return null;
            continue;
        }
        var ta = (0 - o[a]) / d[a];
        var tb = (nf - o[a]) / d[a];
        if (ta > tb) std.mem.swap(f64, &ta, &tb);
        if (ta > t0) {
            t0 = ta;
            entry = @intCast(a);
        }
        t1 = @min(t1, tb);
    }
    if (!(t0 < t1)) return null;

    const ni: i32 = @intCast(n);
    var cell: [3]i32 = undefined;
    var step: [3]i32 = undefined;
    var tnext: [3]f64 = undefined;
    var tdelta: [3]f64 = undefined;
    for (0..3) |a| {
        const p = o[a] + d[a] * t0;
        cell[a] = std.math.clamp(@as(i32, @intFromFloat(@max(@min(@floor(p), 1e9), -1e9))), 0, ni - 1);
        step[a] = if (d[a] > 0) 1 else -1;
        tdelta[a] = if (d[a] != 0) @abs(1.0 / d[a]) else std.math.inf(f64);
        const bound: f64 = @floatFromInt(cell[a] + @as(i32, if (d[a] > 0) 1 else 0));
        tnext[a] = if (d[a] != 0) (bound - o[a]) / d[a] else std.math.inf(f64);
    }
    var axis: usize = if (entry < 0) 0 else @intCast(entry);
    var t = t0;
    while (true) {
        const idx: usize = @intCast(cell[0] + ni * (cell[1] + ni * cell[2]));
        if (g[idx] != 0) {
            const face: i32 = if (entry < 0 and t == t0) -1 else @as(i32, @intCast(2 * axis)) + @as(i32, if (d[axis] > 0) 1 else 0);
            return .{ .t = t, .v = cell, .face = face };
        }
        axis = if (tnext[0] < tnext[1]) (if (tnext[0] < tnext[2]) 0 else 2) else (if (tnext[1] < tnext[2]) 1 else 2);
        t = tnext[axis];
        if (t >= t1) return null;
        cell[axis] += step[axis];
        if (cell[axis] < 0 or cell[axis] >= ni) return null;
        tnext[axis] += tdelta[axis];
    }
}

// ---------------------------------------------------------------------------
// Affine Hilfen
// ---------------------------------------------------------------------------

pub fn inverse(m: [12]f32) [12]f32 {
    return pyrit.xform.inverse(&m).?;
}

pub fn applyF64(m: [12]f32, p: [3]f64, w: f64) [3]f64 {
    var out: [3]f64 = undefined;
    for (0..3) |i| out[i] = m[i * 4] * p[0] + m[i * 4 + 1] * p[1] + m[i * 4 + 2] * p[2] + m[i * 4 + 3] * w;
    return out;
}

/// Rotation um die Achse (ax, ay, az) mit Winkel a, Skalierung s, Verschiebung t
pub fn makeTransform(axis_in: [3]f64, a: f64, s: f64, t: [3]f64) [12]f32 {
    const l = @sqrt(axis_in[0] * axis_in[0] + axis_in[1] * axis_in[1] + axis_in[2] * axis_in[2]);
    const ax = axis_in[0] / l;
    const ay = axis_in[1] / l;
    const az = axis_in[2] / l;
    const c = @cos(a);
    const sn = @sin(a);
    const k = 1 - c;
    const r = [9]f64{
        c + ax * ax * k,      ax * ay * k - az * sn, ax * az * k + ay * sn,
        ay * ax * k + az * sn, c + ay * ay * k,      ay * az * k - ax * sn,
        az * ax * k - ay * sn, az * ay * k + ax * sn, c + az * az * k,
    };
    var out: [12]f32 = undefined;
    for (0..3) |i| {
        out[i * 4 + 0] = @floatCast(r[i * 3 + 0] * s);
        out[i * 4 + 1] = @floatCast(r[i * 3 + 1] * s);
        out[i * 4 + 2] = @floatCast(r[i * 3 + 2] * s);
        out[i * 4 + 3] = @floatCast(t[i]);
    }
    return out;
}

pub fn makeInstance(geometry: u32, n: u32, m: [12]f32, history: u32) types.InstanceData {
    var d = std.mem.zeroes(types.InstanceData);
    d.object_to_world = m;
    d.world_to_object = inverse(m);
    const box = pyrit.xform.transformedBox(&m, @floatFromInt(n));
    d.bounds_min = box.min;
    d.bounds_max = box.max;
    d.geometry = geometry;
    d.mask = 0xFF;
    d.history = history;
    d.flags = types.instance_active;
    return d;
}

pub fn cameraData(c: types.Camera) types.CameraData {
    return .{ .camera = c, .world_to_view = inverse(c.view_to_world) };
}

pub fn lookAt(cam: *types.Camera, eye: [3]f32, target: [3]f32, up: [3]f32) void {
    pyrit.pyr_camera_look_at(cam, &eye, &target, &up);
}
