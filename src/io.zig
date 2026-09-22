//! Ein- und Ausgabe: MagicaVoxel-Import (.vox) und ein eigenes Binärformat
//! für fertige DAGs. Nur Datenumwandlung auf dem Host; gebaut und gerendert
//! wird auf der GPU.

const std = @import("std");
const dag_builder = @import("dag_builder.zig");
const api = @import("api.zig");

pub const Error = error{ InvalidData, OutOfMemory };

// ---------------------------------------------------------------------------
// MagicaVoxel (.vox)
// ---------------------------------------------------------------------------

pub const VoxModel = struct {
    size: [3]u32,
    /// Voxel mit Attribut = PYR_VOXEL(0, r, g, b); y zeigt nach oben
    voxels: []api.Voxel,
    log2_size: u32,
};

fn rd32(d: []const u8, o: usize) Error!u32 {
    if (o + 4 > d.len) return error.InvalidData;
    return std.mem.readInt(u32, d[o..][0..4], .little);
}

/// Liest Modell `model` einer .vox-Datei. Koordinaten werden von z-oben
/// (MagicaVoxel) nach y-oben umgestellt, ohne die Händigkeit zu spiegeln.
pub fn parseVox(gpa: std.mem.Allocator, d: []const u8, model: u32) Error!VoxModel {
    if (d.len < 8 or !std.mem.eql(u8, d[0..4], "VOX ")) return error.InvalidData;
    var palette: [256][4]u8 = undefined;
    for (&palette, 0..) |*c, i| {
        const g: u8 = @intCast(i);
        c.* = .{ g, g, g, 255 };
    }
    var size: ?[3]u32 = null;
    var xyzi: ?[]const u8 = null;
    var models_seen: u32 = 0;

    // Kapitel flach durchlaufen (MAIN enthält alle anderen als Kinder)
    var o: usize = 8;
    while (o + 12 <= d.len) {
        const id = d[o..][0..4];
        const content = try rd32(d, o + 4);
        const children = try rd32(d, o + 8);
        const body = o + 12;
        if (body + content > d.len) return error.InvalidData;
        if (std.mem.eql(u8, id, "MAIN")) {
            o = body + content; // Kinder folgen direkt
            continue;
        }
        if (std.mem.eql(u8, id, "SIZE")) {
            if (models_seen == model) size = .{ try rd32(d, body), try rd32(d, body + 4), try rd32(d, body + 8) };
        } else if (std.mem.eql(u8, id, "XYZI")) {
            if (models_seen == model) {
                const n = try rd32(d, body);
                if (body + 4 + @as(usize, n) * 4 > d.len) return error.InvalidData;
                xyzi = d[body + 4 ..][0 .. @as(usize, n) * 4];
            }
            models_seen += 1;
        } else if (std.mem.eql(u8, id, "RGBA")) {
            if (content < 1024) return error.InvalidData;
            // Paletteneintrag i gehört zu Farbindex i + 1
            for (0..255) |i| palette[i + 1] = d[body + i * 4 ..][0..4].*;
        }
        o = body + content + children;
    }
    const sz = size orelse return error.InvalidData;
    const raw = xyzi orelse return error.InvalidData;
    const extent = @max(sz[0], @max(sz[1], sz[2]));
    var log2: u32 = 3;
    while ((@as(u32, 1) << @intCast(log2)) < extent) log2 += 1;
    if (log2 > dag_builder.max_log2) return error.InvalidData;

    const voxels = try gpa.alloc(api.Voxel, raw.len / 4);
    for (voxels, 0..) |*v, i| {
        const e = raw[i * 4 ..][0..4];
        const c = palette[e[3]];
        var a = (@as(u32, c[0]) << 24) | (@as(u32, c[1]) << 16) | (@as(u32, c[2]) << 8);
        if (a == 0) a = 1 << 8; // Schwarz wäre sonst "leer"
        // (x, y, z)_MV mit z oben -> (x, z, size_y - 1 - y)
        v.* = .{ .x = e[0], .y = e[2], .z = @as(i32, @intCast(sz[1])) - 1 - @as(i32, e[1]), .attribute = a };
    }
    return .{ .size = .{ sz[0], sz[2], sz[1] }, .voxels = voxels, .log2_size = log2 };
}

// ---------------------------------------------------------------------------
// DAG-Binärformat
//
//   "PYRD", Version, log2_size, root, Flags (Bit 0: Attribute),
//   voxel_count (u64), node_words (u64), leaf_count (u64), attr_count (u64),
//   nodes (u32[]), leaves (u64[]), attributes (u32[]); little-endian
// ---------------------------------------------------------------------------

const magic = "PYRD";
const format_version: u32 = 1;
const header_size = 4 + 4 * 4 + 8 * 4;

pub fn savedSize(dag: *const dag_builder.Dag) usize {
    const attrs: usize = if (dag.attributes) |a| a.len else 0;
    return header_size + dag.nodes.len * 4 + dag.leaves.len * 8 + attrs * 4;
}

pub fn save(dag: *const dag_builder.Dag, out: []u8) void {
    std.debug.assert(out.len >= savedSize(dag));
    const attrs: []const u32 = dag.attributes orelse &.{};
    @memcpy(out[0..4], magic);
    var o: usize = 4;
    for ([_]u32{ format_version, dag.log2_size, dag.root, @intFromBool(dag.attributes != null) }) |v| {
        std.mem.writeInt(u32, out[o..][0..4], v, .little);
        o += 4;
    }
    for ([_]u64{ dag.voxel_count, dag.nodes.len, dag.leaves.len, attrs.len }) |v| {
        std.mem.writeInt(u64, out[o..][0..8], v, .little);
        o += 8;
    }
    for (dag.nodes) |w| {
        std.mem.writeInt(u32, out[o..][0..4], w, .little);
        o += 4;
    }
    for (dag.leaves) |w| {
        std.mem.writeInt(u64, out[o..][0..8], w, .little);
        o += 8;
    }
    for (attrs) |w| {
        std.mem.writeInt(u32, out[o..][0..4], w, .little);
        o += 4;
    }
}

pub fn load(gpa: std.mem.Allocator, d: []const u8) Error!dag_builder.Dag {
    if (d.len < header_size or !std.mem.eql(u8, d[0..4], magic)) return error.InvalidData;
    const version = std.mem.readInt(u32, d[4..8], .little);
    if (version != format_version) return error.InvalidData;
    const log2 = std.mem.readInt(u32, d[8..12], .little);
    const root = std.mem.readInt(u32, d[12..16], .little);
    const has_attrs = std.mem.readInt(u32, d[16..20], .little) & 1 != 0;
    const voxel_count = std.mem.readInt(u64, d[20..28], .little);
    const nw = std.mem.readInt(u64, d[28..36], .little);
    const nl = std.mem.readInt(u64, d[36..44], .little);
    const na = std.mem.readInt(u64, d[44..52], .little);
    if (log2 < dag_builder.min_log2 or log2 > dag_builder.max_log2) return error.InvalidData;
    const need = std.math.add(u64, header_size, nw * 4 + nl * 8 + na * 4) catch return error.InvalidData;
    if (need > d.len or root >= nw or (has_attrs and na != voxel_count)) return error.InvalidData;

    const nodes = try gpa.alloc(u32, nw);
    errdefer gpa.free(nodes);
    const leaves = try gpa.alloc(u64, nl);
    errdefer gpa.free(leaves);
    const attrs: ?[]u32 = if (has_attrs) try gpa.alloc(u32, na) else null;
    var o: usize = header_size;
    for (nodes) |*w| {
        w.* = std.mem.readInt(u32, d[o..][0..4], .little);
        o += 4;
    }
    for (leaves) |*w| {
        w.* = std.mem.readInt(u64, d[o..][0..8], .little);
        o += 8;
    }
    if (attrs) |a| for (a) |*w| {
        w.* = std.mem.readInt(u32, d[o..][0..4], .little);
        o += 4;
    };
    return .{ .log2_size = log2, .root = root, .voxel_count = voxel_count, .nodes = nodes, .leaves = leaves, .attributes = attrs };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "DAG speichern und laden" {
    const gpa = testing.allocator;
    const pts = [_]dag_builder.Point{ .{ .x = 1, .y = 2, .z = 3, .attribute = 7 }, .{ .x = 9, .y = 9, .z = 9, .attribute = 8 } };
    var dag = try dag_builder.buildPoints(gpa, 4, &pts, true);
    defer dag.deinit(gpa);
    const buf = try gpa.alloc(u8, savedSize(&dag));
    defer gpa.free(buf);
    save(&dag, buf);
    var back = try load(gpa, buf);
    defer back.deinit(gpa);
    try testing.expectEqualSlices(u32, dag.nodes, back.nodes);
    try testing.expectEqualSlices(u64, dag.leaves, back.leaves);
    try testing.expectEqual(@as(?u32, 8), back.lookup(9, 9, 9));
    buf[30] ^= 0xFF; // beschädigt
    try testing.expectError(error.InvalidData, load(gpa, buf));
}

test "MagicaVoxel-Import" {
    const gpa = testing.allocator;
    // Minimaldatei: SIZE 2x3x4, zwei Voxel, Palette mit Rot an Index 1
    var d: std.ArrayList(u8) = .empty;
    defer d.deinit(gpa);
    const w = struct {
        fn u32le(l: *std.ArrayList(u8), a: std.mem.Allocator, v: u32) !void {
            var b: [4]u8 = undefined;
            std.mem.writeInt(u32, &b, v, .little);
            try l.appendSlice(a, &b);
        }
    };
    try d.appendSlice(gpa, "VOX ");
    try w.u32le(&d, gpa, 150);
    try d.appendSlice(gpa, "MAIN");
    try w.u32le(&d, gpa, 0);
    try w.u32le(&d, gpa, 12 + 12 + 12 + 12 + 1024 + 12);
    try d.appendSlice(gpa, "SIZE");
    try w.u32le(&d, gpa, 12);
    try w.u32le(&d, gpa, 0);
    for ([_]u32{ 2, 3, 4 }) |v| try w.u32le(&d, gpa, v);
    try d.appendSlice(gpa, "XYZI");
    try w.u32le(&d, gpa, 12);
    try w.u32le(&d, gpa, 0);
    try w.u32le(&d, gpa, 2);
    try d.appendSlice(gpa, &[_]u8{ 0, 0, 0, 1, 1, 2, 3, 2 });
    try d.appendSlice(gpa, "RGBA");
    try w.u32le(&d, gpa, 1024);
    try w.u32le(&d, gpa, 0);
    var pal = [_]u8{0} ** 1024;
    pal[0..4].* = .{ 255, 0, 0, 255 }; // Index 1
    try d.appendSlice(gpa, &pal);

    const m = try parseVox(gpa, d.items, 0);
    defer gpa.free(m.voxels);
    try testing.expectEqual([3]u32{ 2, 4, 3 }, m.size);
    try testing.expectEqual(@as(u32, 3), m.log2_size);
    try testing.expectEqual(@as(usize, 2), m.voxels.len);
    // (0,0,0) -> (0, 0, 2); Rot
    try testing.expectEqual(api.Voxel{ .x = 0, .y = 0, .z = 2, .attribute = 0xFF00_0000 }, m.voxels[0]);
    // (1,2,3) -> (1, 3, 0); Index 2 ist schwarz -> Ersatz
    try testing.expectEqual(api.Voxel{ .x = 1, .y = 3, .z = 0, .attribute = 1 << 8 }, m.voxels[1]);
}
