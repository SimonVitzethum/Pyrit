//! Bau einer Sparse Voxel DAG auf der CPU (Format siehe include/pyrit/dag.h).
//!
//! Alle Quellen (dicht, Punktliste, Funktion) liefern 4x4x4-Bricks in
//! Morton-Reihenfolge. Daraus entsteht die DAG strombasiert von unten nach
//! oben: pro Ebene gibt es genau einen offenen Knoten; wechselt das
//! Morton-Präfix, wird er abgeschlossen, dedupliziert und im Elternknoten
//! eingetragen. Die Morton-Reihenfolge (x im niedrigsten Bit) entspricht der
//! Tiefensuche nach Kindindex, daher liegen die Attribute direkt richtig.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const min_log2: u32 = 3;
pub const max_log2: u32 = 20;

pub const Error = error{ InvalidArgument, OutOfMemory };

pub const Dag = struct {
    log2_size: u32,
    root: u32,
    voxel_count: u64,
    nodes: []u32,
    leaves: []u64,
    attributes: ?[]u32,

    pub fn deinit(self: *Dag, gpa: Allocator) void {
        gpa.free(self.nodes);
        gpa.free(self.leaves);
        if (self.attributes) |a| gpa.free(a);
        self.* = undefined;
    }

    /// Attribut des Voxels (x, y, z) oder null, wenn leer. Referenzimplementierung
    /// derselben Rangberechnung wie in dag.h; für Tests und Werkzeuge.
    pub fn lookup(self: *const Dag, x: u32, y: u32, z: u32) ?u32 {
        const n = @as(u32, 1) << @intCast(self.log2_size);
        if (x >= n or y >= n or z >= n) return null;
        var node = self.root;
        var rank: u64 = 0;
        var s: u32 = self.log2_size - 1;
        while (true) : (s -= 1) {
            const mask = self.nodes[node] & 0xFF;
            const idx = childIndex(x, y, z, s);
            if ((mask >> @intCast(idx)) & 1 == 0) return null;
            const before = @popCount(mask & ((@as(u32, 1) << @intCast(idx)) - 1));
            var k: u32 = 0;
            while (k < before) : (k += 1) {
                const ref = self.nodes[node + 2 + k];
                rank += if (s == 2) @popCount(self.leaves[ref]) else self.nodes[ref + 1];
            }
            const child = self.nodes[node + 2 + before];
            if (s == 2) {
                const brick = self.leaves[child];
                const bit = brickBit(x & 3, y & 3, z & 3);
                if ((brick >> @intCast(bit)) & 1 == 0) return null;
                rank += @popCount(brick & ((@as(u64, 1) << @intCast(bit)) - 1));
                return if (self.attributes) |a| a[@intCast(rank)] else 1;
            }
            node = child;
        }
    }
};

pub fn brickBit(lx: u32, ly: u32, lz: u32) u32 {
    return lx + 4 * ly + 16 * lz;
}

fn childIndex(x: u32, y: u32, z: u32, s: u32) u32 {
    const sh: u5 = @intCast(s);
    return ((x >> sh) & 1) | (((y >> sh) & 1) << 1) | (((z >> sh) & 1) << 2);
}

/// Bit i von v landet auf Bit 3i.
fn spread3(v: u32) u64 {
    var x: u64 = v & 0x1FFFFF;
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
}

fn compact3(v: u64) u32 {
    var x = v & 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
    x = (x | (x >> 4)) & 0x100F00F00F00F00F;
    x = (x | (x >> 8)) & 0x1F0000FF0000FF;
    x = (x | (x >> 16)) & 0x1F00000000FFFF;
    x = (x | (x >> 32)) & 0x1FFFFF;
    return @intCast(x);
}

pub fn morton(x: u32, y: u32, z: u32) u64 {
    return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}

pub fn mortonDecode(m: u64) [3]u32 {
    return .{ compact3(m), compact3(m >> 1), compact3(m >> 2) };
}

// ---------------------------------------------------------------------------
// Deduplizierungstabellen
// ---------------------------------------------------------------------------

/// Offene Adressierung über Offsets in `nodes`; Schlüssel ist der Knoteninhalt.
const NodeTable = struct {
    slots: []u32 = &.{},
    used: usize = 0,
    const empty_slot = std.math.maxInt(u32);

    fn deinit(self: *NodeTable, gpa: Allocator) void {
        gpa.free(self.slots);
    }

    fn hashWords(words: []const u32) u64 {
        return std.hash.Wyhash.hash(0x9E3779B97F4A7C15, std.mem.sliceAsBytes(words));
    }

    fn nodeLen(nodes: []const u32, off: u32) usize {
        return 2 + @popCount(nodes[off] & 0xFF);
    }

    fn grow(self: *NodeTable, gpa: Allocator, nodes: []const u32) Allocator.Error!void {
        const new_cap = if (self.slots.len == 0) 1024 else self.slots.len * 2;
        const new_slots = try gpa.alloc(u32, new_cap);
        @memset(new_slots, empty_slot);
        for (self.slots) |off| {
            if (off == empty_slot) continue;
            var i = hashWords(nodes[off..][0..nodeLen(nodes, off)]) & (new_cap - 1);
            while (new_slots[i] != empty_slot) i = (i + 1) & (new_cap - 1);
            new_slots[i] = off;
        }
        gpa.free(self.slots);
        self.slots = new_slots;
    }

    /// Liefert den Offset eines gleichen Knotens oder hängt `words` an `nodes` an.
    fn intern(self: *NodeTable, gpa: Allocator, nodes: *std.ArrayList(u32), words: []const u32) Error!u32 {
        if ((self.used + 1) * 4 > self.slots.len * 3) try self.grow(gpa, nodes.items);
        const cap = self.slots.len;
        var i = hashWords(words) & (cap - 1);
        while (self.slots[i] != empty_slot) : (i = (i + 1) & (cap - 1)) {
            const off = self.slots[i];
            const len = nodeLen(nodes.items, off);
            if (len == words.len and std.mem.eql(u32, nodes.items[off..][0..len], words)) return off;
        }
        if (nodes.items.len + words.len > std.math.maxInt(u32)) return error.OutOfMemory;
        const off: u32 = @intCast(nodes.items.len);
        try nodes.appendSlice(gpa, words);
        self.slots[i] = off;
        self.used += 1;
        return off;
    }
};

// ---------------------------------------------------------------------------
// Strombasierter Bau
// ---------------------------------------------------------------------------

const Pending = struct {
    valid: bool = false,
    key: u64 = 0,
    mask: u32 = 0,
    count: u64 = 0,
    refs: [8]u32 = undefined,
};

pub const Builder = struct {
    gpa: Allocator,
    log2_size: u32,
    store_attributes: bool,
    nodes: std.ArrayList(u32) = .empty,
    leaves: std.ArrayList(u64) = .empty,
    attributes: std.ArrayList(u32) = .empty,
    node_table: NodeTable = .{},
    leaf_table: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    pending: [max_log2]Pending = [_]Pending{.{}} ** max_log2,
    last_brick: ?u64 = null,
    voxel_count: u64 = 0,

    pub fn init(gpa: Allocator, log2_size: u32, store_attributes: bool) Error!Builder {
        if (log2_size < min_log2 or log2_size > max_log2) return error.InvalidArgument;
        return .{ .gpa = gpa, .log2_size = log2_size, .store_attributes = store_attributes };
    }

    pub fn deinit(self: *Builder) void {
        self.nodes.deinit(self.gpa);
        self.leaves.deinit(self.gpa);
        self.attributes.deinit(self.gpa);
        self.node_table.deinit(self.gpa);
        self.leaf_table.deinit(self.gpa);
    }

    /// Schließt den offenen Knoten der Ebene s ab und trägt ihn im Elternknoten ein.
    fn finalize(self: *Builder, s: u32) Error!void {
        const p = &self.pending[s];
        var words: [10]u32 = undefined;
        const n = @popCount(p.mask);
        words[0] = p.mask;
        words[1] = @intCast(@min(p.count, std.math.maxInt(u32)));
        @memcpy(words[2..][0..n], p.refs[0..n]);
        const off = try self.node_table.intern(self.gpa, &self.nodes, words[0 .. 2 + n]);
        if (s + 1 < self.log2_size) {
            const parent = &self.pending[s + 1];
            std.debug.assert(parent.valid and parent.key == p.key >> 3);
            addChild(parent, @intCast(p.key & 7), off, p.count);
        } else {
            self.pending[s].refs[0] = off; // Wurzel merken
        }
        p.valid = false;
    }

    fn addChild(p: *Pending, idx: u32, ref: u32, count: u64) void {
        // Kinder kommen in aufsteigender Reihenfolge, daher genügt Anhängen.
        const pos = @popCount(p.mask);
        p.refs[pos] = ref;
        p.mask |= @as(u32, 1) << @intCast(idx);
        p.count += count;
    }

    /// Nimmt einen nicht leeren Brick mit Morton-Code `m` (Brick-Koordinaten) auf.
    /// `attrs` enthält die Attribute der gesetzten Bits in Bitreihenfolge.
    pub fn addBrick(self: *Builder, m: u64, mask: u64, attrs: []const u32) Error!void {
        std.debug.assert(mask != 0);
        if (self.last_brick) |last| {
            if (m <= last) return error.InvalidArgument;
        }
        self.last_brick = m;
        const top = self.log2_size - 1;

        // Offene Knoten abschließen, deren Präfix nicht mehr passt
        var s: u32 = 2;
        while (s <= top) : (s += 1) {
            const key = m >> @intCast(3 * (s - 1));
            if (self.pending[s].valid and self.pending[s].key != key) {
                try self.finalize(s);
            } else break;
        }
        // Fehlende offene Knoten anlegen
        s = 2;
        while (s <= top) : (s += 1) {
            if (!self.pending[s].valid) {
                self.pending[s] = .{ .valid = true, .key = m >> @intCast(3 * (s - 1)) };
            }
        }

        const gop = try self.leaf_table.getOrPut(self.gpa, mask);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(self.leaves.items.len);
            try self.leaves.append(self.gpa, mask);
        }
        const cnt = @popCount(mask);
        addChild(&self.pending[2], @intCast(m & 7), gop.value_ptr.*, cnt);
        self.voxel_count += cnt;
        if (self.store_attributes) try self.attributes.appendSlice(self.gpa, attrs[0..cnt]);
    }

    /// Schließt den Bau ab; der Builder ist danach leer.
    pub fn finish(self: *Builder) Error!Dag {
        const top = self.log2_size - 1;
        if (!self.pending[top].valid) {
            // Leere Geometrie: Wurzel ohne Kinder
            self.pending[top] = .{ .valid = true, .key = 0 };
        }
        var s: u32 = 2;
        while (s <= top) : (s += 1) {
            if (self.pending[s].valid) try self.finalize(s);
        }
        const root = self.pending[top].refs[0];
        const nodes = try self.nodes.toOwnedSlice(self.gpa);
        errdefer self.gpa.free(nodes);
        const leaves = try self.leaves.toOwnedSlice(self.gpa);
        errdefer self.gpa.free(leaves);
        const attributes: ?[]u32 = if (self.store_attributes) try self.attributes.toOwnedSlice(self.gpa) else null;
        return .{
            .log2_size = self.log2_size,
            .root = root,
            .voxel_count = self.voxel_count,
            .nodes = nodes,
            .leaves = leaves,
            .attributes = attributes,
        };
    }
};

// ---------------------------------------------------------------------------
// Quellen
// ---------------------------------------------------------------------------

/// voxels[x + n * (y + n * z)], 0 = leer.
pub fn buildDense(gpa: Allocator, log2_size: u32, voxels: []const u32, store_attributes: bool) Error!Dag {
    var b = try Builder.init(gpa, log2_size, store_attributes);
    defer b.deinit();
    const n: u64 = @as(u64, 1) << @intCast(log2_size);
    if (voxels.len != n * n * n) return error.InvalidArgument;
    const bricks_per_axis = n / 4;
    const brick_count = bricks_per_axis * bricks_per_axis * bricks_per_axis;
    var attrs: [64]u32 = undefined;
    var m: u64 = 0;
    while (m < brick_count) : (m += 1) {
        const c = mortonDecode(m);
        var mask: u64 = 0;
        var k: usize = 0;
        var bit: u32 = 0;
        while (bit < 64) : (bit += 1) {
            const x = c[0] * 4 + (bit & 3);
            const y = c[1] * 4 + ((bit >> 2) & 3);
            const z = c[2] * 4 + (bit >> 4);
            const v = voxels[@intCast(x + n * (y + n * @as(u64, z)))];
            if (v != 0) {
                mask |= @as(u64, 1) << @intCast(bit);
                attrs[k] = v;
                k += 1;
            }
        }
        if (mask != 0) try b.addBrick(m, mask, attrs[0..k]);
    }
    return b.finish();
}

pub const Point = extern struct { x: i32, y: i32, z: i32, attribute: u32 };

/// Punktliste in beliebiger Reihenfolge; bei doppelten Koordinaten gewinnt der letzte Eintrag.
pub fn buildPoints(gpa: Allocator, log2_size: u32, points: []const Point, store_attributes: bool) Error!Dag {
    var b = try Builder.init(gpa, log2_size, store_attributes);
    defer b.deinit();
    const n: i64 = @as(i64, 1) << @intCast(log2_size);

    const Entry = struct { key: u64, order: u32, attribute: u32 };
    const entries = try gpa.alloc(Entry, points.len);
    defer gpa.free(entries);
    for (points, 0..) |p, i| {
        if (p.x < 0 or p.y < 0 or p.z < 0 or p.x >= n or p.y >= n or p.z >= n or p.attribute == 0)
            return error.InvalidArgument;
        const x: u32 = @intCast(p.x);
        const y: u32 = @intCast(p.y);
        const z: u32 = @intCast(p.z);
        const brick = morton(x >> 2, y >> 2, z >> 2);
        entries[i] = .{ .key = (brick << 6) | brickBit(x & 3, y & 3, z & 3), .order = @intCast(i), .attribute = p.attribute };
    }
    std.mem.sort(Entry, entries, {}, struct {
        fn lessThan(_: void, a: Entry, bb: Entry) bool {
            return a.key < bb.key or (a.key == bb.key and a.order < bb.order);
        }
    }.lessThan);

    var attrs: [64]u32 = undefined;
    var i: usize = 0;
    while (i < entries.len) {
        const brick = entries[i].key >> 6;
        var mask: u64 = 0;
        var k: usize = 0;
        while (i < entries.len and entries[i].key >> 6 == brick) : (i += 1) {
            const bit: u6 = @intCast(entries[i].key & 63);
            const bitmask = @as(u64, 1) << bit;
            if (mask & bitmask != 0) {
                attrs[k - 1] = entries[i].attribute; // Duplikat: letzter gewinnt
            } else {
                mask |= bitmask;
                attrs[k] = entries[i].attribute;
                k += 1;
            }
        }
        try b.addBrick(brick, mask, attrs[0..k]);
    }
    return b.finish();
}

pub const VoxelFn = *const fn (user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32;
pub const RegionEmptyFn = *const fn (user: ?*anyopaque, x: i32, y: i32, z: i32, size: u32) callconv(.c) i32;

pub fn buildFn(gpa: Allocator, log2_size: u32, voxel: VoxelFn, region_empty: ?RegionEmptyFn, user: ?*anyopaque, store_attributes: bool) Error!Dag {
    var b = try Builder.init(gpa, log2_size, store_attributes);
    defer b.deinit();
    const Walker = struct {
        b: *Builder,
        voxel: VoxelFn,
        region_empty: ?RegionEmptyFn,
        user: ?*anyopaque,

        fn walk(w: *@This(), x: u32, y: u32, z: u32, size: u32) Error!void {
            if (w.region_empty) |f| {
                if (f(w.user, @intCast(x), @intCast(y), @intCast(z), size) != 0) return;
            }
            if (size == 4) {
                var attrs: [64]u32 = undefined;
                var mask: u64 = 0;
                var k: usize = 0;
                var bit: u32 = 0;
                while (bit < 64) : (bit += 1) {
                    const v = w.voxel(w.user, @intCast(x + (bit & 3)), @intCast(y + ((bit >> 2) & 3)), @intCast(z + (bit >> 4)));
                    if (v != 0) {
                        mask |= @as(u64, 1) << @intCast(bit);
                        attrs[k] = v;
                        k += 1;
                    }
                }
                if (mask != 0) try w.b.addBrick(morton(x >> 2, y >> 2, z >> 2), mask, attrs[0..k]);
                return;
            }
            const h = size / 2;
            var idx: u32 = 0;
            while (idx < 8) : (idx += 1) {
                try w.walk(x + (idx & 1) * h, y + ((idx >> 1) & 1) * h, z + (idx >> 2) * h, h);
            }
        }
    };
    var w = Walker{ .b = &b, .voxel = voxel, .region_empty = region_empty, .user = user };
    try w.walk(0, 0, 0, @as(u32, 1) << @intCast(log2_size));
    return b.finish();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "morton roundtrip" {
    const c = [_][3]u32{ .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 5, 9, 1023 }, .{ 0x1FFFFF, 3, 0x12345 } };
    for (c) |v| try testing.expectEqual(v, mortonDecode(morton(v[0], v[1], v[2])));
    try testing.expectEqual(@as(u64, 1), morton(1, 0, 0));
    try testing.expectEqual(@as(u64, 2), morton(0, 1, 0));
    try testing.expectEqual(@as(u64, 4), morton(0, 0, 1));
}

test "leere Geometrie" {
    var dag = try buildPoints(testing.allocator, 3, &.{}, true);
    defer dag.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 0), dag.nodes[dag.root] & 0xFF);
    try testing.expectEqual(@as(?u32, null), dag.lookup(1, 2, 3));
}

test "einzelnes Voxel" {
    const pts = [_]Point{.{ .x = 5, .y = 6, .z = 7, .attribute = 42 }};
    var dag = try buildPoints(testing.allocator, 4, &pts, true);
    defer dag.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 1), dag.voxel_count);
    try testing.expectEqual(@as(?u32, 42), dag.lookup(5, 6, 7));
    try testing.expectEqual(@as(?u32, null), dag.lookup(5, 6, 6));
}

test "dicht = Punkte = Funktion, Attribute pro Voxel" {
    const gpa = testing.allocator;
    const log2: u32 = 5;
    const n: u32 = 1 << log2;
    var prng = std.Random.DefaultPrng.init(1234);
    const rnd = prng.random();

    const dense = try gpa.alloc(u32, n * n * n);
    defer gpa.free(dense);
    var points: std.ArrayList(Point) = .empty;
    defer points.deinit(gpa);
    for (dense, 0..) |*v, i| {
        const x: u32 = @intCast(i % n);
        const y: u32 = @intCast((i / n) % n);
        const z: u32 = @intCast(i / (n * n));
        // Kugel plus Rauschen, damit sowohl gleiche als auch verschiedene Teilbäume entstehen
        const dx = @as(f32, @floatFromInt(x)) - 15.5;
        const dy = @as(f32, @floatFromInt(y)) - 15.5;
        const dz = @as(f32, @floatFromInt(z)) - 15.5;
        const inside = dx * dx + dy * dy + dz * dz < 12.0 * 12.0 or rnd.uintLessThan(u32, 50) == 0;
        v.* = if (inside) 1 + rnd.uintLessThan(u32, 1000) else 0;
        if (v.* != 0) try points.append(gpa, .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z), .attribute = v.* });
    }
    // Punktliste mischen: Reihenfolge darf keine Rolle spielen
    rnd.shuffle(Point, points.items);

    var a = try buildDense(gpa, log2, dense, true);
    defer a.deinit(gpa);
    var b = try buildPoints(gpa, log2, points.items, true);
    defer b.deinit(gpa);

    const Ctx = struct {
        fn voxel(user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32 {
            const d: [*]const u32 = @ptrCast(@alignCast(user.?));
            return d[@intCast(x + 32 * (y + 32 * z))];
        }
    };
    var c = try buildFn(gpa, log2, Ctx.voxel, null, @ptrCast(dense.ptr), true);
    defer c.deinit(gpa);

    try testing.expectEqualSlices(u32, a.nodes, b.nodes);
    try testing.expectEqualSlices(u64, a.leaves, b.leaves);
    try testing.expectEqualSlices(u32, a.attributes.?, b.attributes.?);
    try testing.expectEqualSlices(u32, a.nodes, c.nodes);
    try testing.expectEqual(a.root, c.root);

    for (dense, 0..) |v, i| {
        const x: u32 = @intCast(i % n);
        const y: u32 = @intCast((i / n) % n);
        const z: u32 = @intCast(i / (n * n));
        const expected: ?u32 = if (v == 0) null else v;
        try testing.expectEqual(expected, a.lookup(x, y, z));
    }
    try testing.expectEqual(@as(u64, points.items.len), a.voxel_count);
}

test "Deduplizierung: identische Teilbäume werden geteilt" {
    const gpa = testing.allocator;
    const log2: u32 = 6;
    const n: u32 = 1 << log2;
    const dense = try gpa.alloc(u32, n * n * n);
    defer gpa.free(dense);
    // Periodisches Muster mit Periode 8: alle 8^3-Blöcke gleich
    for (dense, 0..) |*v, i| {
        const x: u32 = @intCast(i % n);
        const y: u32 = @intCast((i / n) % n);
        const z: u32 = @intCast(i / (n * n));
        v.* = if ((x % 8) + (y % 8) < (z % 8) + 2) 1 else 0;
    }
    var dag = try buildDense(gpa, log2, dense, false);
    defer dag.deinit(gpa);
    // Vier Knotenebenen (64, 32, 16, 8), je Ebene genau ein eindeutiger Knoten mit höchstens 8 Kindern
    try testing.expect(dag.nodes.len <= 4 * 10);
    for (dense, 0..) |v, i| {
        const expected: ?u32 = if (v == 0) null else 1;
        try testing.expectEqual(expected, dag.lookup(@intCast(i % n), @intCast((i / n) % n), @intCast(i / (n * n))));
    }
}
