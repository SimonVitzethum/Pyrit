//! Planer für große Welten: Chunk-Octree um die Kamera, LOD nach Bildschirmmaß.
//!
//! Ein Chunk (lod, x, y, z) hat immer 2^chunk_log2 Voxel pro Kante; ein Voxel
//! der Stufe lod ist 2^lod Grundvoxel groß. Feste Stufenzahlen und feste
//! Abstände gibt es nicht: Verfeinert wird, solange ein Voxel auf dem Bildschirm
//! größer als `voxel_pixels` wäre. Der Schwellabstand folgt damit aus Kamera,
//! Auflösung und Blickwinkel (`pixels_per_unit`), die Tiefe des Baums aus der
//! Sichtweite, und ein Speicherbudget skaliert alles gleitend (`lod_scale`).
//! Kinder werden erst gezeigt, wenn alle acht bereit sind (gebaut oder leer),
//! sonst bleibt der Elternknoten sichtbar – so entstehen beim Nachladen keine
//! Löcher.
//!
//! Der Planer ist reine Host-Logik ohne GPU (testbar); die Welt (world.zig)
//! baut angefragte Chunks auf der GPU und meldet sie als bereit.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Key = struct {
    lod: u8,
    x: i32,
    y: i32,
    z: i32,

    pub fn parent(k: Key) Key {
        return .{ .lod = k.lod + 1, .x = k.x >> 1, .y = k.y >> 1, .z = k.z >> 1 };
    }

    pub fn child(k: Key, i: u32) Key {
        return .{
            .lod = k.lod - 1,
            .x = k.x * 2 + @as(i32, @intCast(i & 1)),
            .y = k.y * 2 + @as(i32, @intCast((i >> 1) & 1)),
            .z = k.z * 2 + @as(i32, @intCast(i >> 2)),
        };
    }
};

pub const State = enum { requested, building, ready, empty };

pub const Node = struct {
    state: State,
    /// letzter Frame, in dem der Knoten gebraucht wurde (für die Verdrängung)
    last_used: u64,
    /// von der Welt vergeben (Geometrie, Instanz); vom Planer nicht angefasst
    user: u64 = 0,
};

pub const Config = struct {
    chunk_log2: u32,
    /// Grenze der Baumtiefe (Sicherheitsnetz, keine feste Stufenzahl)
    max_lod: u32 = 20,
    /// Radius der gröbsten Stufe in Chunks
    view_chunks: u32 = 2,
    /// Sichtweite in Grundvoxeln; bestimmt auch die gröbste Stufe
    view_distance: f64,
    /// Verfeinern, solange 2^lod · refine_k > Abstand.
    /// refine_k = Pixel je Grundvoxel in Entfernung 1 / Zielgröße in Pixeln.
    refine_k: f64,
    /// vertikaler Bereich in Grundvoxeln [y_min, y_max)
    y_min: i32,
    y_max: i32,
    /// Knoten, die so viele Frames nicht gebraucht wurden, werden verdrängt
    evict_frames: u64 = 8,
};

pub const Plan = struct {
    cfg: Config,
    gpa: Allocator,
    nodes: std.AutoHashMapUnmanaged(Key, Node) = .empty,
    visible: std.ArrayList(Key) = .empty,
    /// zweite, gröbere Auswahl für Sekundärstrahlen (Schatten, GI)
    visible_coarse: std.ArrayList(Key) = .empty,
    /// Ziel des laufenden Durchlaufs (visible oder visible_coarse)
    target: ?*std.ArrayList(Key) = null,
    /// offene Anfragen (auch aus früheren Frames); wird beim Abholen gefiltert
    requests: std.ArrayList(Key) = .empty,
    frame: u64 = 0,
    /// Verdrängung ist teuer (ganze Tabelle) und eilt nicht
    evict_frame: u64 = 0,

    pub fn deinit(self: *Plan) void {
        self.nodes.deinit(self.gpa);
        self.visible.deinit(self.gpa);
        self.visible_coarse.deinit(self.gpa);
        self.requests.deinit(self.gpa);
    }

    /// Chunkgröße in Grundvoxeln
    pub fn chunkSize(self: *const Plan, lod: u32) f64 {
        return @floatFromInt(@as(u64, 1) << @intCast(self.cfg.chunk_log2 + lod));
    }

    /// Gröbste Stufe: so tief, dass der Ring aus view_chunks die Sichtweite deckt
    pub fn topLod(self: *const Plan) u8 {
        var l: u32 = 0;
        const r: f64 = @floatFromInt(self.cfg.view_chunks);
        while (l + 1 < self.cfg.max_lod and r * self.chunkSize(l) < self.cfg.view_distance) l += 1;
        return @intCast(l);
    }

    /// Größe eines Voxels dieser Stufe in Pixeln, in Entfernung `dist`
    pub fn voxelPixels(_: *const Plan, lod: u32, dist: f64, pixels_per_unit: f64) f64 {
        const size = @as(f64, @floatFromInt(@as(u64, 1) << @intCast(lod)));
        return size * pixels_per_unit / @max(dist, 1e-3);
    }

    fn distance(self: *const Plan, k: Key, cam: [3]f64) f64 {
        const s = self.chunkSize(k.lod);
        const lo = [3]f64{ @as(f64, @floatFromInt(k.x)) * s, @as(f64, @floatFromInt(k.y)) * s, @as(f64, @floatFromInt(k.z)) * s };
        var d2: f64 = 0;
        for (0..3) |a| {
            const v = @max(lo[a] - cam[a], 0, cam[a] - (lo[a] + s));
            d2 += v * v;
        }
        return @sqrt(d2);
    }

    fn inRange(self: *const Plan, k: Key) bool {
        const s = self.chunkSize(k.lod);
        const y0 = @as(f64, @floatFromInt(k.y)) * s;
        return y0 + s > @as(f64, @floatFromInt(self.cfg.y_min)) and y0 < @as(f64, @floatFromInt(self.cfg.y_max));
    }

    fn touch(self: *Plan, k: Key) Allocator.Error!*Node {
        const gop = try self.nodes.getOrPut(self.gpa, k);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{ .state = .requested, .last_used = self.frame };
            try self.requests.append(self.gpa, k);
        }
        gop.value_ptr.last_used = self.frame;
        return gop.value_ptr;
    }

    fn visit(self: *Plan, k: Key, cam: [3]f64) Allocator.Error!void {
        if (!self.inRange(k)) return;
        const state = (try self.touch(k)).state; // Zeiger nicht halten: Tabelle wächst
        if (state == .empty) return; // leerer Bereich: Kinder ebenfalls leer
        const dist = self.distance(k, cam);
        if (dist > self.cfg.view_distance) return; // hinter der Sichtweite
        // Bildschirmmaß: verfeinern, solange ein Voxel zu groß erschiene
        const refine = k.lod > 0 and dist < self.cfg.refine_k * @as(f64, @floatFromInt(@as(u64, 1) << @intCast(k.lod)));
        if (refine and state == .ready) {
            var all_ready = true;
            var i: u32 = 0;
            while (i < 8) : (i += 1) {
                const ck = k.child(i);
                if (!self.inRange(ck)) continue;
                const c = try self.touch(ck);
                if (c.state != .ready and c.state != .empty) all_ready = false;
            }
            if (all_ready) {
                i = 0;
                while (i < 8) : (i += 1) try self.visit(k.child(i), cam);
                return;
            }
        }
        if (state == .ready) try self.target.?.append(self.gpa, k);
    }

    /// Neuer Frame: sichtbare Chunks und neue Anfragen bestimmen.
    /// `cam` in Grundvoxeln. Danach: `visible`, `requests` (neu, noch nicht gebaut).
    pub fn update(self: *Plan, cam: [3]f64) Allocator.Error!void {
        self.frame += 1;
        try self.collect(cam);
    }

    /// Sichtbare Chunks neu bestimmen, ohne einen neuen Frame zu beginnen
    /// (z. B. nachdem im selben Frame Chunks fertig wurden).
    pub fn collect(self: *Plan, cam: [3]f64) Allocator.Error!void {
        return self.collectInto(cam, &self.visible);
    }

    fn collectInto(self: *Plan, cam: [3]f64, out: *std.ArrayList(Key)) Allocator.Error!void {
        self.target = out;
        defer self.target = null;
        out.clearRetainingCapacity();
        const top = self.topLod();
        const s = self.chunkSize(top);
        const cx: i32 = @intFromFloat(@floor(cam[0] / s));
        const cz: i32 = @intFromFloat(@floor(cam[2] / s));
        const y0: i32 = @intFromFloat(@floor(@as(f64, @floatFromInt(self.cfg.y_min)) / s));
        const y1: i32 = @intFromFloat(@floor(@as(f64, @floatFromInt(self.cfg.y_max - 1)) / s));
        const r: i32 = @intCast(self.cfg.view_chunks);
        var z = cz - r;
        while (z <= cz + r) : (z += 1) {
            var x = cx - r;
            while (x <= cx + r) : (x += 1) {
                var y = y0;
                while (y <= y1) : (y += 1) try self.visit(.{ .lod = top, .x = x, .y = y, .z = z }, cam);
            }
        }
    }

    /// Zweite Auswahl mit gröberem Ziel (refine_k / factor) für Sekundärstrahlen.
    /// Die Knoten sind Vorfahren der feinen Auswahl und liegen schon im Speicher.
    pub fn collectCoarse(self: *Plan, cam: [3]f64, factor: f64) Allocator.Error!void {
        const fine = self.cfg.refine_k;
        self.cfg.refine_k = fine / @max(factor, 1);
        defer self.cfg.refine_k = fine;
        try self.collectInto(cam, &self.visible_coarse);
    }

    /// Nächste Anfragen, grob vor fein, nahe vor fern; markiert sie als in Arbeit.
    pub fn takeRequests(self: *Plan, cam: [3]f64, max: usize, out: *std.ArrayList(Key)) Allocator.Error!void {
        out.clearRetainingCapacity();
        // Liste der offenen Anfragen aufräumen: gebaute, verdrängte und gerade
        // nicht gebrauchte fallen heraus (kein Durchlauf über alle Knoten)
        var w: usize = 0;
        for (self.requests.items) |k| {
            const n = self.nodes.get(k) orelse continue;
            if (n.state != .requested or n.last_used != self.frame) continue;
            self.requests.items[w] = k;
            w += 1;
        }
        self.requests.shrinkRetainingCapacity(w);
        const Ctx = struct {
            plan: *const Plan,
            cam: [3]f64,
            fn less(c: @This(), a: Key, b: Key) bool {
                if (a.lod != b.lod) return a.lod > b.lod;
                return c.plan.distance(a, c.cam) < c.plan.distance(b, c.cam);
            }
        };
        std.mem.sort(Key, self.requests.items, Ctx{ .plan = self, .cam = cam }, Ctx.less);
        for (self.requests.items[0..@min(max, self.requests.items.len)]) |k| {
            self.nodes.getPtr(k).?.state = .building;
            try out.append(self.gpa, k);
        }
    }

    pub fn finish(self: *Plan, k: Key, empty: bool, user: u64) void {
        const n = self.nodes.getPtr(k) orelse return;
        n.state = if (empty) .empty else .ready;
        n.user = user;
    }

    /// Knoten, die lange nicht gebraucht wurden, entfernen; `out` erhält ihre
    /// Nutzdaten (zum Freigeben durch die Welt).
    pub fn evict(self: *Plan, out: *std.ArrayList(Node)) Allocator.Error!void {
        out.clearRetainingCapacity();
        // nur gelegentlich: der Durchlauf geht über alle Knoten
        if (self.frame - self.evict_frame < @max(self.cfg.evict_frames / 2, 1)) return;
        self.evict_frame = self.frame;
        var dead: std.ArrayList(Key) = .empty;
        defer dead.deinit(self.gpa);
        var it = self.nodes.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.state == .building) continue;
            if (self.frame - kv.value_ptr.last_used > self.cfg.evict_frames) try dead.append(self.gpa, kv.key_ptr.*);
        }
        for (dead.items) |k| {
            try out.append(self.gpa, self.nodes.get(k).?);
            _ = self.nodes.remove(k);
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const print = std.debug.print;

fn covers(a: Key, b: Key) bool {
    // a (gröber oder gleich) enthält b
    if (a.lod < b.lod) return false;
    const sh: u5 = @intCast(a.lod - b.lod);
    return (b.x >> sh) == a.x and (b.y >> sh) == a.y and (b.z >> sh) == a.z;
}

/// Simuliert: alle Anfragen sofort bauen; leer, wenn der Chunk über y = 100 liegt
fn settle(p: *Plan, cam: [3]f64, rounds: usize) !void {
    var req: std.ArrayList(Key) = .empty;
    defer req.deinit(testing.allocator);
    for (0..rounds) |_| {
        try p.update(cam);
        try p.takeRequests(cam, 1 << 20, &req);
        for (req.items) |k| {
            const s = p.chunkSize(k.lod);
            p.finish(k, @as(f64, @floatFromInt(k.y)) * s >= 100, 0);
        }
    }
    try p.update(cam);
}

test "Weltplaner: Bildschirmmaß statt fester Stufen, lückenlos, ohne Überlappung" {
    // 16^3-Chunks, Ziel 4 px je Voxel bei 500 px je Einheit
    const pixels_per_unit: f64 = 500;
    const target_px: f64 = 4;
    var p = Plan{ .gpa = testing.allocator, .cfg = .{ .chunk_log2 = 4, .max_lod = 6, .view_chunks = 2, .view_distance = 1500, .refine_k = pixels_per_unit / target_px, .y_min = 0, .y_max = 256 } };
    defer p.deinit();
    const cam = [3]f64{ 300, 40, -120 };
    try settle(&p, cam, 12);

    // keine zwei sichtbaren Knoten überlappen sich
    for (p.visible.items, 0..) |a, i| for (p.visible.items[i + 1 ..]) |b| {
        try testing.expect(!covers(a, b) and !covers(b, a));
    };
    // Bildschirmmaß: kein sichtbares Voxel ist größer als das Ziel, und
    // gröber ginge es nicht (der Elternknoten wäre zu grob)
    var nearest_lod: u8 = 255;
    var farthest_lod: u8 = 0;
    var worst_px: f64 = 0;
    var coarsest_ok: bool = true;
    for (p.visible.items) |k| {
        const d = p.distance(k, cam);
        // Stufe 0 ist die feinste vorhandene Auflösung; dort gilt das Ziel nicht
        if (k.lod > 0) worst_px = @max(worst_px, p.voxelPixels(k.lod, d, pixels_per_unit));
        // Der Elternknoten wurde verfeinert, weil er an *seinem* Abstand zu grob war
        if (k.lod < p.topLod()) {
            const par = k.parent();
            if (p.voxelPixels(par.lod, p.distance(par, cam), pixels_per_unit) <= target_px) coarsest_ok = false;
        }
        if (d == 0) nearest_lod = @min(nearest_lod, k.lod);
        if (d > 600) farthest_lod = @max(farthest_lod, k.lod);
    }
    try testing.expect(worst_px <= target_px + 1e-6);
    try testing.expect(coarsest_ok);
    try testing.expectEqual(@as(u8, 0), nearest_lod);
    try testing.expect(farthest_lod >= 3);
    print("  {d} sichtbare Chunks, gröbste Stufe {d}, größtes Voxel {d:.2} px (Ziel {d:.1})\n", .{ p.visible.items.len, p.topLod(), worst_px, target_px });

    // Ziel verdoppeln: gröber und weniger Chunks (dynamisch, ohne feste Stufen)
    const fine_count = p.visible.items.len;
    p.cfg.refine_k = pixels_per_unit / (2 * target_px);
    try settle(&p, cam, 12);
    var coarse_px: f64 = 0;
    for (p.visible.items) |k| {
        if (k.lod > 0) coarse_px = @max(coarse_px, p.voxelPixels(k.lod, p.distance(k, cam), pixels_per_unit));
    }
    print("  Ziel {d:.1} px: {d} Chunks statt {d}, größtes Voxel {d:.2} px\n", .{ 2 * target_px, p.visible.items.len, fine_count, coarse_px });
    try testing.expect(p.visible.items.len * 2 < fine_count);
    try testing.expect(coarse_px <= 2 * target_px + 1e-6);
    p.cfg.refine_k = pixels_per_unit / target_px;
    try settle(&p, cam, 12);

    // Grundfläche (y < 100) vollständig abgedeckt: Stichproben innerhalb des Sichtbereichs
    const top_size = p.chunkSize(4);
    var prng = std.Random.DefaultPrng.init(2);
    const rnd = prng.random();
    for (0..500) |_| {
        const px = cam[0] + (rnd.float(f64) * 2 - 1) * top_size * 1.9;
        const pz = cam[2] + (rnd.float(f64) * 2 - 1) * top_size * 1.9;
        const py = rnd.float(f64) * 99;
        const v = Key{ .lod = 0, .x = @intFromFloat(@floor(px / 16)), .y = @intFromFloat(@floor(py / 16)), .z = @intFromFloat(@floor(pz / 16)) };
        var hit = false;
        for (p.visible.items) |k| {
            if (covers(k, v)) hit = true;
        }
        try testing.expect(hit);
    }

    // Kamerawechsel: alte Knoten werden nach evict_frames verdrängt
    const far = [3]f64{ 20000, 40, 20000 };
    try settle(&p, far, 12);
    var gone: std.ArrayList(Node) = .empty;
    defer gone.deinit(testing.allocator);
    try p.evict(&gone);
    try testing.expect(gone.items.len > 0);
    var near_old: usize = 0;
    var it = p.nodes.iterator();
    while (it.next()) |kv| {
        if (p.distance(kv.key_ptr.*, cam) < 50 and kv.key_ptr.lod == 0) near_old += 1;
    }
    try testing.expectEqual(@as(usize, 0), near_old);
}
