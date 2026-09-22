//! Zerlegt eine DAG in AABB-Primitive für die Hardware-BVH der RT-Cores.
//!
//! Jeder belegte Teilbaum der Kantenlänge 2^k wird ein Primitiv. Die AABB ist
//! so knapp wie möglich (Hülle der tatsächlich belegten Voxel), damit die
//! RT-Cores möglichst wenige Intersection-Aufrufe auslösen. Knoten sind in der
//! DAG geteilt, daher werden Hüllen pro (Knoten, Ebene) zwischengespeichert.

const std = @import("std");
const types = @import("pyrit_device").types;
const rt = @import("pyrit_device").rt;
const dag_builder = @import("dag_builder.zig");
const Allocator = std.mem.Allocator;

/// Layout von OptixAabb
pub const Aabb = extern struct { min: [3]f32, max: [3]f32 };

pub const Prims = struct {
    rt_log2: u32,
    prims: []types.RtPrim,
    aabbs: []Aabb,

    pub fn deinit(self: *Prims, gpa: Allocator) void {
        gpa.free(self.prims);
        gpa.free(self.aabbs);
    }
};

/// Gemessen (RTX 5070 Laptop, 128^3-Geometrie): große Teilbäume sind am
/// schnellsten, die DAG-Traversierung im Shader schlägt eine feinere Hardware-BVH.
pub const default_log2: u32 = 7;

/// Voxelgenaue Hülle, max exklusiv
const Box = struct {
    min: [3]u32,
    max: [3]u32,

    const empty = Box{ .min = .{ std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32) }, .max = .{ 0, 0, 0 } };

    fn merge(a: *Box, b: Box, off: [3]u32) void {
        for (0..3) |i| {
            a.min[i] = @min(a.min[i], b.min[i] + off[i]);
            a.max[i] = @max(a.max[i], b.max[i] + off[i]);
        }
    }
};

fn leafBox(mask: u64) Box {
    var b = Box.empty;
    var m = mask;
    while (m != 0) : (m &= m - 1) {
        const bit: u32 = @ctz(m);
        const p = [3]u32{ bit & 3, (bit >> 2) & 3, bit >> 4 };
        b.merge(.{ .min = p, .max = .{ p[0] + 1, p[1] + 1, p[2] + 1 } }, .{ 0, 0, 0 });
    }
    return b;
}

const Extractor = struct {
    gpa: Allocator,
    dag: *const dag_builder.Dag,
    k: u32,
    boxes: std.AutoHashMapUnmanaged(u64, Box) = .empty,
    prims: std.ArrayList(types.RtPrim) = .empty,
    aabbs: std.ArrayList(Aabb) = .empty,

    fn childOffset(idx: u32, size: u32) [3]u32 {
        return .{ (idx & 1) * size, ((idx >> 1) & 1) * size, (idx >> 2) * size };
    }

    /// Hülle eines Knotens, der einen Würfel der Kantenlänge 2^level abdeckt
    fn box(self: *Extractor, node: u32, level: u32) Allocator.Error!Box {
        const key = (@as(u64, level) << 32) | node;
        if (self.boxes.get(key)) |b| return b;
        var b = Box.empty;
        const nodes = self.dag.nodes;
        const mask = nodes[node] & 0xFF;
        const child_size = @as(u32, 1) << @intCast(level - 1);
        var j: u32 = 0;
        var idx: u32 = 0;
        while (idx < 8) : (idx += 1) {
            if (mask & (@as(u32, 1) << @intCast(idx)) == 0) continue;
            const ref = nodes[node + 2 + j];
            j += 1;
            const cb = if (level == 3) leafBox(self.dag.leaves[ref]) else try self.box(ref, level - 1);
            b.merge(cb, childOffset(idx, child_size));
        }
        try self.boxes.put(self.gpa, key, b);
        return b;
    }

    fn visit(self: *Extractor, node: u32, level: u32, origin: [3]u32, rank: u32) Allocator.Error!void {
        const nodes = self.dag.nodes;
        if (level == self.k) {
            const b = try self.box(node, level);
            if (b.min[0] >= b.max[0]) return; // leer
            const shift: u5 = @intCast(self.k);
            try self.prims.append(self.gpa, .{
                .node = node,
                .attr_base = rank,
                .cell = rt.packCell(origin[0] >> shift, origin[1] >> shift, origin[2] >> shift),
            });
            // kleiner Rand gegen Rundung im Strahl-Box-Test der Hardware
            const eps = 1.0 / 256.0;
            var a: Aabb = undefined;
            for (0..3) |i| {
                a.min[i] = @as(f32, @floatFromInt(origin[i] + b.min[i])) - eps;
                a.max[i] = @as(f32, @floatFromInt(origin[i] + b.max[i])) + eps;
            }
            try self.aabbs.append(self.gpa, a);
            return;
        }
        const mask = nodes[node] & 0xFF;
        const child_size = @as(u32, 1) << @intCast(level - 1);
        var r = rank;
        var j: u32 = 0;
        var idx: u32 = 0;
        while (idx < 8) : (idx += 1) {
            if (mask & (@as(u32, 1) << @intCast(idx)) == 0) continue;
            const ref = nodes[node + 2 + j];
            j += 1;
            const off = childOffset(idx, child_size);
            try self.visit(ref, level - 1, .{ origin[0] + off[0], origin[1] + off[1], origin[2] + off[2] }, r);
            r +%= nodes[ref + 1];
        }
    }
};

/// Primitive der Kantenlänge 2^k (k wird auf [3, log2_size] begrenzt).
pub fn extract(gpa: Allocator, dag: *const dag_builder.Dag, requested_log2: u32) Allocator.Error!Prims {
    const k = std.math.clamp(requested_log2, 3, dag.log2_size);
    var e = Extractor{ .gpa = gpa, .dag = dag, .k = k };
    defer e.boxes.deinit(gpa);
    errdefer e.prims.deinit(gpa);
    errdefer e.aabbs.deinit(gpa);
    try e.visit(dag.root, dag.log2_size, .{ 0, 0, 0 }, 0);
    return .{ .rt_log2 = k, .prims = try e.prims.toOwnedSlice(gpa), .aabbs = try e.aabbs.toOwnedSlice(gpa) };
}

test "Primitive decken genau die belegten Teilbäume ab" {
    const gpa = std.testing.allocator;
    const pts = [_]dag_builder.Point{
        .{ .x = 1, .y = 2, .z = 3, .attribute = 7 },
        .{ .x = 30, .y = 31, .z = 17, .attribute = 8 },
        .{ .x = 31, .y = 31, .z = 17, .attribute = 9 },
    };
    var dag = try dag_builder.buildPoints(gpa, 5, &pts, true);
    defer dag.deinit(gpa);
    var p = try extract(gpa, &dag, 4);
    defer p.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), p.prims.len);
    // erstes Primitiv: Zelle (0,0,0), Hülle genau um (1,2,3)
    try std.testing.expectApproxEqAbs(@as(f32, 1), p.aabbs[0].min[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4), p.aabbs[0].max[2], 0.01);
    try std.testing.expectEqual(@as(u32, 0), p.prims[0].attr_base);
    // zweites: Zelle (1,1,1), zwei Voxel, Attributrang beginnt bei 1
    try std.testing.expectEqual(rt.packCell(1, 1, 1), p.prims[1].cell);
    try std.testing.expectEqual(@as(u32, 1), p.prims[1].attr_base);
    try std.testing.expectApproxEqAbs(@as(f32, 30), p.aabbs[1].min[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 32), p.aabbs[1].max[0], 0.01);
}
