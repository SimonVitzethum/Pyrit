//! Bereichsverwaltung für die GPU-Pools (Knoten, Blätter, Attribute).
//! Reine Host-Buchhaltung: First-Fit über eine sortierte Freiliste mit
//! Zusammenfassen benachbarter Bereiche.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Range = struct { offset: u64, size: u64 };

pub const RangeAlloc = struct {
    capacity: u64,
    free: std.ArrayList(Range) = .empty, // nach offset sortiert, nicht überlappend, nicht benachbart
    used: u64 = 0,

    pub fn init(gpa: Allocator, capacity: u64) Allocator.Error!RangeAlloc {
        var r = RangeAlloc{ .capacity = capacity };
        if (capacity > 0) try r.free.append(gpa, .{ .offset = 0, .size = capacity });
        return r;
    }

    pub fn deinit(self: *RangeAlloc, gpa: Allocator) void {
        self.free.deinit(gpa);
    }

    pub fn alloc(self: *RangeAlloc, size: u64) ?u64 {
        if (size == 0) return 0;
        for (self.free.items, 0..) |*r, i| {
            if (r.size < size) continue;
            const off = r.offset;
            r.offset += size;
            r.size -= size;
            if (r.size == 0) _ = self.free.orderedRemove(i);
            self.used += size;
            return off;
        }
        return null;
    }

    pub fn release(self: *RangeAlloc, gpa: Allocator, offset: u64, size: u64) Allocator.Error!void {
        if (size == 0) return;
        self.used -= size;
        const items = self.free.items;
        var i: usize = 0;
        while (i < items.len and items[i].offset < offset) i += 1;
        const merge_prev = i > 0 and items[i - 1].offset + items[i - 1].size == offset;
        const merge_next = i < items.len and offset + size == items[i].offset;
        if (merge_prev and merge_next) {
            items[i - 1].size += size + items[i].size;
            _ = self.free.orderedRemove(i);
        } else if (merge_prev) {
            items[i - 1].size += size;
        } else if (merge_next) {
            items[i].offset = offset;
            items[i].size += size;
        } else {
            try self.free.insert(gpa, i, .{ .offset = offset, .size = size });
        }
    }
};

test "RangeAlloc: vergeben, freigeben, zusammenfassen" {
    const gpa = std.testing.allocator;
    var r = try RangeAlloc.init(gpa, 100);
    defer r.deinit(gpa);
    const a = r.alloc(30).?;
    const b = r.alloc(30).?;
    const c = r.alloc(30).?;
    try std.testing.expectEqual(@as(?u64, null), r.alloc(20));
    try r.release(gpa, b, 30);
    try r.release(gpa, a, 30);
    try std.testing.expectEqual(@as(usize, 2), r.free.items.len);
    try std.testing.expectEqual(@as(?u64, 0), r.alloc(60));
    try r.release(gpa, 0, 60);
    try r.release(gpa, c, 30);
    try std.testing.expectEqual(@as(usize, 1), r.free.items.len);
    try std.testing.expectEqual(@as(u64, 100), r.free.items[0].size);
    try std.testing.expectEqual(@as(u64, 0), r.used);
}
