//! RT-Pfad: Verfolgung innerhalb eines AABB-Primitivs (Teil-DAG).
//!
//! Die Hardware-BVH der RT-Cores ersetzt die oberen DAG-Ebenen: jedes belegte
//! Teilstück der Kantenlänge 2^rt_log2 ist ein AABB. Trifft ein Strahl dessen
//! Box, verfolgt der Intersection-Shader nur den Teil-DAG ab `prim.node`.
//! Dieselbe Funktion läuft auf der CPU (Tests) und im OptiX-Shader.

const types = @import("types.zig");
const vec = @import("vec.zig");
const dag = @import("dag.zig");
const Vec3 = vec.Vec3;

pub const cell_bits = 21;
const cell_mask: u64 = (1 << cell_bits) - 1;

pub inline fn packCell(x: u32, y: u32, z: u32) u64 {
    return @as(u64, x) | (@as(u64, y) << cell_bits) | (@as(u64, z) << (2 * cell_bits));
}

pub inline fn unpackCell(c: u64) [3]u32 {
    return .{ @intCast(c & cell_mask), @intCast((c >> cell_bits) & cell_mask), @intCast(c >> (2 * cell_bits)) };
}

/// Treffer im Objektraum der Geometrie (Voxelkoordinaten global).
pub fn traceSubtree(g: *const types.RtGeometry, prim: types.RtPrim, o: Vec3, d: Vec3, tmin: f32, tmax: f32, want_attribute: bool, skip: ?*const [4]u64, cut: ?*const [4]u64) ?dag.DagHit {
    const cell = unpackCell(prim.cell);
    const shift: u5 = @intCast(g.rt_log2);
    const base = [3]u32{ cell[0] << shift, cell[1] << shift, cell[2] << shift };
    const origin = Vec3{ @floatFromInt(base[0]), @floatFromInt(base[1]), @floatFromInt(base[2]) };
    const sub = dag.Dag{
        .nodes = @ptrFromInt(g.nodes),
        .leaves = @ptrFromInt(g.leaves),
        // Rang über attr_base: mit Palette sind es 4-Bit-Indizes, ein Zeiger
        // ließe sich nicht auf die Hälfte eines Bytes versetzen
        .attributes = if (g.attributes != 0) @as([*]const u32, @ptrFromInt(g.attributes)) else null,
        .palette = if (g.palette != 0) @as([*]const u32, @ptrFromInt(g.palette)) else null,
        .attr_base = prim.attr_base,
        .root = prim.node,
        .log2_size = g.rt_log2,
        .default_attribute = g.default_attribute,
    };
    const hit = if (skip != null or cut != null)
        dag.traceSkipping(&sub, o - origin, d, tmin, tmax, want_attribute, skip, cut, .{ @intCast(base[0]), @intCast(base[1]), @intCast(base[2]) })
    else
        dag.trace(&sub, o - origin, d, tmin, tmax, want_attribute);
    var h = hit orelse return null;
    inline for (0..3) |a| h.voxel[a] += @intCast(base[a]);
    return h;
}
