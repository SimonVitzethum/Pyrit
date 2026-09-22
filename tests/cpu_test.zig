//! CPU-Tests: der Gerätecode (src/device) läuft auf der CPU und wird gegen
//! unabhängige Referenzen geprüft. Braucht keine GPU.

const std = @import("std");
const pyr = @import("pyrit_device");
const pyrit = @import("pyrit");
const common = @import("common.zig");
const types = pyr.types;
const vec = pyr.vec;
const testing = std.testing;
const gpa = testing.allocator;
const print = std.debug.print;

test "einzelnes Voxel, achsenparallele Strahlen" {
    const pts = [_]pyrit.dag_builder.Point{.{ .x = 5, .y = 6, .z = 7, .attribute = 42 }};
    var dag = try pyrit.dag_builder.buildPoints(gpa, 3, &pts, true);
    defer dag.deinit(gpa);
    const g = pyr.dag.Dag{ .nodes = dag.nodes.ptr, .leaves = dag.leaves.ptr, .attributes = dag.attributes.?.ptr, .root = dag.root, .log2_size = 3, .default_attribute = 1 };

    const Case = struct { o: vec.Vec3, d: vec.Vec3, t: f32, face: u32 };
    const cases = [_]Case{
        .{ .o = .{ -10, 6.5, 7.5 }, .d = .{ 1, 0, 0 }, .t = 15, .face = types.face_neg_x },
        .{ .o = .{ 20, 6.5, 7.5 }, .d = .{ -1, 0, 0 }, .t = 14, .face = types.face_pos_x },
        .{ .o = .{ 5.5, -3, 7.5 }, .d = .{ 0, 1, 0 }, .t = 9, .face = types.face_neg_y },
        .{ .o = .{ 5.5, 6.5, 30 }, .d = .{ 0, 0, -2 }, .t = 11, .face = types.face_pos_z },
    };
    for (cases) |c| {
        const h = pyr.dag.trace(&g, c.o, c.d, 0, types.flt_max, true) orelse return error.KeinTreffer;
        try testing.expectApproxEqAbs(c.t, h.t, 1e-5);
        try testing.expectEqual(c.face, h.face);
        try testing.expectEqual([3]i32{ 5, 6, 7 }, h.voxel);
        try testing.expectEqual(@as(u32, 42), h.attribute);
        try testing.expect(!h.inside);
    }
    try testing.expect(pyr.dag.trace(&g, .{ -10, 6.5, 6.5 }, .{ 1, 0, 0 }, 0, types.flt_max, true) == null);
    // Start im Voxel
    const inside = pyr.dag.trace(&g, .{ 5.5, 6.5, 7.5 }, .{ 1, 0, 0 }, 0, types.flt_max, true).?;
    try testing.expect(inside.inside);
}

fn randomRays(log2: u32, rays: usize) !void {
    const n: usize = @as(usize, 1) << @intCast(log2);
    var h = try common.buildSceneDag(gpa, log2);
    defer h.deinit(gpa);
    const dense = try common.denseScene(gpa, n);
    defer gpa.free(dense);
    const g = h.device();
    const nf: f64 = @floatFromInt(n);

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var hits: usize = 0;
    var mismatch: usize = 0;
    var face_mismatch: usize = 0;
    var attr_mismatch: usize = 0;
    var i: usize = 0;
    while (i < rays) : (i += 1) {
        var o: [3]f64 = undefined;
        var d: [3]f64 = undefined;
        for (0..3) |a| {
            o[a] = nf * 0.5 + (rnd.float(f64) * 2 - 1) * nf * 1.5;
            d[a] = nf * 0.5 + (rnd.float(f64) * 2 - 1) * nf * 0.45 - o[a];
        }
        if (i % 8 == 0) d[i % 3] = 0; // achsparallel in einer Ebene
        if (i % 16 == 1) {
            d[0] = 0;
            d[1] = 0;
        }
        if (d[0] == 0 and d[1] == 0 and d[2] == 0) continue;

        const r = common.refTrace(dense, n, o, d, 0, 1e30);
        const of = vec.Vec3{ @floatCast(o[0]), @floatCast(o[1]), @floatCast(o[2]) };
        const df = vec.Vec3{ @floatCast(d[0]), @floatCast(d[1]), @floatCast(d[2]) };
        const dh = pyr.dag.trace(&g, of, df, 0, types.flt_max, true);
        if ((r == null) != (dh == null) or (r != null and !std.mem.eql(i32, &r.?.v, &dh.?.voxel))) {
            mismatch += 1;
            if (mismatch <= 3) print("  Abweichung bei Strahl {d}\n", .{i});
            continue;
        }
        const rh = r orelse continue;
        const hh = dh.?;
        hits += 1;
        const len = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        try testing.expect(@abs(hh.t - rh.t) * len < 1e-3 * nf);
        if (rh.face >= 0 and @as(i32, @intCast(hh.face)) != rh.face) face_mismatch += 1;
        if (hh.attribute != common.attrOf(hh.voxel[0], hh.voxel[1], hh.voxel[2])) attr_mismatch += 1;
    }
    print("  {d}^3: {d} Treffer, {d} Abweichungen, {d} Flächen-, {d} Attributfehler\n", .{ n, hits, mismatch, face_mismatch, attr_mismatch });
    // f32 gegen f64: Strahlen, die exakt eine Kante streifen, dürfen selten abweichen
    try testing.expect(mismatch * 10000 <= rays);
    try testing.expect(face_mismatch * 10000 <= rays);
    try testing.expectEqual(@as(usize, 0), attr_mismatch);
    try testing.expect(hits > rays / 4);
}

test "DAG-Traversierung gegen Referenz-DDA 64^3" {
    try randomRays(6, 200_000);
}

test "DAG-Traversierung gegen Referenz-DDA 256^3" {
    try randomRays(8, 50_000);
}

test "Szene mit transformierten Instanzen gegen Referenz" {
    const log2 = 6;
    const n = 64;
    var h = try common.buildSceneDag(gpa, log2);
    defer h.deinit(gpa);
    const dense = try common.denseScene(gpa, n);
    defer gpa.free(dense);

    var inst = [3]types.InstanceData{
        common.makeInstance(0, n, common.makeTransform(.{ 0, 1, 0 }, 0.0, 1.0, .{ -80, 0, 0 }), 1),
        common.makeInstance(0, n, common.makeTransform(.{ 1, 2, 3 }, 0.7, 0.5, .{ 20, 10, -5 }), 1),
        common.makeInstance(0, n, common.makeTransform(.{ -1, 0.3, 0.2 }, 2.1, 1.7, .{ 0, -40, 60 }), 1),
    };
    inst[2].mask = 0x2;
    const s = common.hostScene(&h, @ptrCast(&h.geometry), &inst, &inst, 3);

    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var mismatch: usize = 0;
    var hits: usize = 0;
    var inside: usize = 0;
    const rays = 50_000;
    var i: usize = 0;
    while (i < rays) : (i += 1) {
        var o: [3]f64 = undefined;
        var d: [3]f64 = undefined;
        for (0..3) |a| {
            o[a] = (rnd.float(f64) * 2 - 1) * 200;
            d[a] = (rnd.float(f64) * 2 - 1) * 70 - o[a];
        }
        const ray_mask: u32 = if (i & 1 != 0) 0xFF else 0x1;

        // Referenz: jede Instanz im Objektraum, nächster Treffer
        var best: ?common.RefHit = null;
        var best_inst: u32 = types.no_hit;
        for (inst, 0..) |in, k| {
            if (in.mask & ray_mask == 0) continue;
            const r = common.refTrace(dense, n, common.applyF64(in.world_to_object, o, 1), common.applyF64(in.world_to_object, d, 0), 0, if (best) |b| b.t else 1e30) orelse continue;
            if (best == null or r.t < best.?.t) {
                best = r;
                best_inst = @intCast(k);
            }
        }
        const of = vec.Vec3{ @floatCast(o[0]), @floatCast(o[1]), @floatCast(o[2]) };
        const df = vec.Vec3{ @floatCast(d[0]), @floatCast(d[1]), @floatCast(d[2]) };
        const th = pyr.traceScene(&s, of, df, 0, types.flt_max, ray_mask, 0);
        if ((best == null) != (th == null) or (th != null and (th.?.instance != best_inst or !std.mem.eql(i32, &th.?.voxel, &best.?.v)))) {
            mismatch += 1;
            continue;
        }
        const hit = th orelse continue;
        hits += 1;
        try testing.expectEqual(common.attrOf(hit.voxel[0], hit.voxel[1], hit.voxel[2]), hit.attribute);
        if (hit.face & types.hit_inside != 0) {
            inside += 1;
            continue;
        }
        // exakter Trefferpunkt liegt auf der Fläche des Voxels
        const ax = hit.face >> 1;
        const face_coord: f32 = @as(f32, @floatFromInt(hit.voxel[ax])) + @as(f32, if (hit.face & 1 != 0) 0 else 1);
        const p_object: [3]f32 = hit.p_object;
        try testing.expect(@abs(p_object[ax] - face_coord) < 2e-3);
    }
    print("  {d} Treffer ({d} mit Start im Voxel), {d} Abweichungen\n", .{ hits, inside, mismatch });
    try testing.expect(mismatch * 5000 <= rays);
    try testing.expect(hits > rays / 20);
}

test "Motion Vectors: verschobene Instanz, orthografische Kamera" {
    const n = 64;
    var h = try common.buildSceneDag(gpa, 6);
    defer h.deinit(gpa);
    const prev = common.makeInstance(0, n, common.makeTransform(.{ 0, 1, 0 }, 0.3, 1.0, .{ -32, -32, -32 }), 5);
    const cur = common.makeInstance(0, n, common.makeTransform(.{ 0, 1, 0 }, 0.3, 1.0, .{ -32 + 3.25, -32 - 1.5, -32 }), 5);
    const s = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&cur), @ptrCast(&prev), 1);

    var cam = std.mem.zeroes(types.Camera);
    common.lookAt(&cam, .{ 0, 0, 200 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_orthographic(&cam, 60, 160, 128);
    cam.jitter = .{ 0.25, -0.125 };

    var p = std.mem.zeroes(types.RenderParams);
    p.cur = common.cameraData(cam);
    p.prev = common.cameraData(cam);
    p.history_valid = 1;
    p.ray_mask = 0xFFFF_FFFF;

    const ppu: f32 = 128.0 / (2.0 * 60.0); // Pixel pro Welteinheit
    const ex = -3.25 * ppu; // vorher − jetzt; Bild-y zeigt nach unten
    const ey = -1.5 * ppu;
    var hits: usize = 0;
    var worst: f32 = 0;
    for (0..cam.height) |y| for (0..cam.width) |x| {
        const r = pyr.render.renderPixel(&p, &s, @intCast(x), @intCast(y));
        if (r.hit.instance == types.no_hit) {
            try testing.expectEqual([2]f32{ 0, 0 }, r.motion);
            continue;
        }
        hits += 1;
        try testing.expect(r.hit.meta & (types.hit_new | types.hit_no_history) == 0);
        worst = @max(worst, @abs(r.motion[0] - ex) + @abs(r.motion[1] - ey));
        try testing.expectApproxEqAbs(r.hit.t, r.depth, 1e-3); // Bildebene z = 200
    };
    print("  {d} Treffer, größter MV-Fehler {e:.2} Pixel\n", .{ hits, worst });
    try testing.expect(hits > 1000);
    try testing.expect(worst < 2e-3);

    // Ohne Vorgeschichte: Flag und MV nur aus der (stehenden) Kamera
    var fresh = cur;
    fresh.history = 6;
    const s2 = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&fresh), @ptrCast(&prev), 1);
    const r = pyr.render.renderPixel(&p, &s2, 80, 64);
    try testing.expect(r.hit.instance == 0 and r.hit.meta & types.hit_new != 0);
    try testing.expectEqual([2]f32{ 0, 0 }, r.motion);
    p.history_valid = 0;
    try testing.expect(pyr.render.renderPixel(&p, &s, 80, 64).hit.meta & types.hit_no_history != 0);
}

// Validator-Prinzip: den Vorframe-Strahl durch (Pixel + MV) verfolgen; er muss
// denselben Flächenpunkt treffen (außer bei Disocclusion).
test "Motion Vectors: Kamera- und Objektbewegung, Reprojektion" {
    const n = 64;
    var h = try common.buildSceneDag(gpa, 6);
    defer h.deinit(gpa);
    const prev = common.makeInstance(0, n, common.makeTransform(.{ 0.2, 1, 0.1 }, 0.40, 1.0, .{ -30, -34, -28 }), 9);
    const cur = common.makeInstance(0, n, common.makeTransform(.{ 0.2, 1, 0.1 }, 0.43, 1.0, .{ -29.2, -34, -28.5 }), 9);
    const s = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&cur), @ptrCast(&prev), 1);
    const s_prev = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&prev), @ptrCast(&prev), 1);

    const w = 192;
    const hh = 128;
    var cp = std.mem.zeroes(types.Camera);
    var cc = std.mem.zeroes(types.Camera);
    common.lookAt(&cp, .{ 90, 55, 110 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    common.lookAt(&cc, .{ 86, 57, 112 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_perspective(&cp, 0.9, w, hh, 0.1);
    pyrit.pyr_camera_perspective(&cc, 0.9, w, hh, 0.1);
    cc.jitter = .{ -0.3, 0.2 };

    var p = std.mem.zeroes(types.RenderParams);
    p.cur = common.cameraData(cc);
    p.prev = common.cameraData(cp);
    p.history_valid = 1;
    p.ray_mask = 0xFFFF_FFFF;

    var hits: usize = 0;
    var same: usize = 0;
    var misses: usize = 0;
    var misses_ok: usize = 0;
    var worst: f32 = 0;
    for (0..hh) |y| for (0..w) |x| {
        const r = pyr.render.renderPixel(&p, &s, @intCast(x), @intCast(y));
        const px = @as(f32, @floatFromInt(x)) + 0.5 + cc.jitter[0];
        const py = @as(f32, @floatFromInt(y)) + 0.5 + cc.jitter[1];
        const prev_ray = pyr.render.cameraRay(&cp, px + r.motion[0], py + r.motion[1]);
        const cur_ray = pyr.render.cameraRay(&cc, px, py);
        if (r.hit.instance == types.no_hit) {
            misses += 1; // Punkt im Unendlichen: gleiche Richtung wie jetzt
            if (vec.dot(prev_ray.d, cur_ray.d) > 1.0 - 1e-6) misses_ok += 1;
            continue;
        }
        hits += 1;
        const a = pyr.traceScene(&s, cur_ray.o, cur_ray.d, cur_ray.tmin, cur_ray.tmax, 0xFFFF_FFFF, 0).?;
        const b = pyr.traceScene(&s_prev, prev_ray.o, prev_ray.d, prev_ray.tmin, prev_ray.tmax, 0xFFFF_FFFF, 0) orelse continue;
        // gleicher Punkt = gleiches Voxel und gleiche Fläche; sonst verdeckt
        if (std.mem.eql(i32, &a.voxel, &b.voxel) and a.face == b.face) {
            same += 1;
            worst = @max(worst, vec.length(a.p_object - b.p_object));
        }
    };
    print("  {d} Treffer, {d} auf denselben Flächenpunkt, größter Fehler {e:.2} Voxel; Hintergrund {d}/{d}\n", .{ hits, same, worst, misses_ok, misses });
    try testing.expect(same * 100 >= hits * 75);
    try testing.expect(worst < 1e-2);
    try testing.expectEqual(misses, misses_ok);
}

test "Strahl-Flags: any_hit, no_attribute, Masken, tmax" {
    var h = try common.buildSceneDag(gpa, 5);
    defer h.deinit(gpa);
    const inst = common.makeInstance(0, 32, common.makeTransform(.{ 0, 1, 0 }, 0, 1, .{ 0, 0, 0 }), 1);
    const s = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&inst), @ptrCast(&inst), 1);
    var ray = types.Ray{ .origin = .{ 16.5, 40, 16.5 }, .tmin = 0, .direction = .{ 0, -1, 0 }, .tmax = types.flt_max };
    const a = pyr.trace(&s, ray, 0xFF, 0);
    const b = pyr.trace(&s, ray, 0xFF, types.trace_any_hit | types.trace_no_attribute);
    try testing.expect(a.instance == 0 and b.instance == 0);
    try testing.expectEqual(@as(u32, 1), b.attribute);
    try testing.expectEqual(types.no_hit, pyr.trace(&s, ray, 0x100, 0).instance);
    ray.tmax = 1;
    try testing.expectEqual(types.no_hit, pyr.trace(&s, ray, 0xFF, 0).instance);
}

test "DAG-Bau über die C-API" {
    const n = 16;
    var dense = [_]u32{0} ** (n * n * n);
    var pts: std.ArrayList(common.api.Voxel) = .empty;
    defer pts.deinit(gpa);
    for (0..n) |z| for (0..n) |y| for (0..n) |x| {
        if ((x * y + z) % 7 == 0) {
            const xi: i32 = @intCast(x);
            const yi: i32 = @intCast(y);
            const zi: i32 = @intCast(z);
            dense[x + n * (y + n * z)] = common.attrOf(xi, yi, zi);
            try pts.append(gpa, .{ .x = xi, .y = yi, .z = zi, .attribute = common.attrOf(xi, yi, zi) });
        }
    };
    var a: ?*anyopaque = null;
    var b: ?*anyopaque = null;
    try testing.expectEqual(common.api.ok, pyrit.pyr_dag_build_dense(4, &dense, 0, @ptrCast(&a)));
    try testing.expectEqual(common.api.ok, pyrit.pyr_dag_build_points(4, pts.items.ptr, pts.items.len, 0, @ptrCast(&b)));
    defer pyrit.pyr_dag_destroy(@ptrCast(a));
    defer pyrit.pyr_dag_destroy(@ptrCast(b));
    var ia: common.api.DagInfo = undefined;
    var ib: common.api.DagInfo = undefined;
    pyrit.pyr_dag_get_info(@ptrCast(a), &ia);
    pyrit.pyr_dag_get_info(@ptrCast(b), &ib);
    try testing.expectEqual(ia.node_words, ib.node_words);
    try testing.expectEqual(@as(u64, pts.items.len), ia.voxel_count);
    try testing.expectEqualSlices(u32, ia.nodes.?[0..ia.node_words], ib.nodes.?[0..ib.node_words]);

    const bad = common.api.Voxel{ .x = 16, .y = 0, .z = 0, .attribute = 1 };
    var c: ?*anyopaque = null;
    try testing.expectEqual(common.api.error_invalid_argument, pyrit.pyr_dag_build_points(4, @ptrCast(&bad), 1, 0, @ptrCast(&c)));
    try testing.expect(std.mem.span(pyrit.pyr_error_message()).len > 0);
}

/// Emulation der RT-Cores auf der CPU: Strahl-Box-Test gegen alle AABBs, im
/// Treffer der Teil-DAG wie im Intersection-Shader; Ergebnis muss der vollen
/// DAG-Traversierung entsprechen (Voxel, t, Fläche, Attribut).
fn rtEquivalence(log2: u32, rt_log2: u32, rays: usize) !void {
    var h = try common.buildSceneDag(gpa, log2);
    defer h.deinit(gpa);
    var prims = try pyrit.rt_prims.extract(gpa, &h.dag, rt_log2);
    defer prims.deinit(gpa);
    const g = h.device();
    const rg = types.RtGeometry{
        .nodes = @intFromPtr(h.dag.nodes.ptr),
        .leaves = @intFromPtr(h.dag.leaves.ptr),
        .attributes = @intFromPtr(h.dag.attributes.?.ptr),
        .prims = @intFromPtr(prims.prims.ptr),
        .rt_log2 = prims.rt_log2,
        .default_attribute = 1,
        .reserved = .{ 0, 0 },
    };
    const nf: f32 = @floatFromInt(@as(u32, 1) << @intCast(log2));

    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var hits: usize = 0;
    var mismatch: usize = 0;
    var box_tests: usize = 0;
    var is_calls: usize = 0;
    for (0..rays) |_| {
        var o: vec.Vec3 = undefined;
        var d: vec.Vec3 = undefined;
        inline for (0..3) |a| {
            o[a] = nf * 0.5 + (rnd.float(f32) * 2 - 1) * nf * 1.5;
            d[a] = nf * 0.5 + (rnd.float(f32) * 2 - 1) * nf * 0.45 - o[a];
        }
        const ref = pyr.dag.trace(&g, o, d, 0, types.flt_max, true);

        // "Hardware": alle Boxen, jeweils mit dem aktuell nächsten t als tmax
        var best: ?pyr.dag.DagHit = null;
        var tmax: f32 = types.flt_max;
        const inv = vec.splat(1) / d;
        for (prims.prims, prims.aabbs) |prim, box| {
            box_tests += 1;
            const lo = (@as(vec.Vec3, box.min) - o) * inv;
            const hi = (@as(vec.Vec3, box.max) - o) * inv;
            const t0 = @max(@reduce(.Max, @min(lo, hi)), 0);
            const t1 = @min(@reduce(.Min, @max(lo, hi)), tmax);
            if (!(t0 <= t1)) continue;
            is_calls += 1;
            if (pyr.rt.traceSubtree(&rg, prim, o, d, 0, tmax, true, null)) |hh| {
                if (hh.t < tmax) {
                    tmax = hh.t;
                    best = hh;
                }
            }
        }
        if ((ref == null) != (best == null)) {
            mismatch += 1;
            if (mismatch <= 4) print("    Treffer voll={} RT={}\n", .{ ref != null, best != null });
            continue;
        }
        const a = ref orelse continue;
        const b = best.?;
        hits += 1;
        // Beginnt der Strahl im Voxel, ist die Fläche undefiniert; beide müssen es melden
        const face_ok = if (a.inside) b.inside else (!b.inside and a.face == b.face);
        if (!std.mem.eql(i32, &a.voxel, &b.voxel) or !face_ok or a.attribute != b.attribute) {
            mismatch += 1;
            if (mismatch <= 4) print("    voll: {any} Fläche {d} t={d:.5} | RT: {any} Fläche {d} t={d:.5}\n", .{ a.voxel, a.face, a.t, b.voxel, b.face, b.t });
            continue;
        }
        try testing.expectApproxEqAbs(a.t, b.t, 1e-3 * nf);
    }
    print("  {d}^3, Teilbäume 2^{d}: {d} Primitive, {d} Treffer, {d} Abweichungen, {d:.1} Shader-Aufrufe/Strahl\n", .{
        @as(u32, 1) << @intCast(log2), prims.rt_log2, prims.prims.len, hits, mismatch, @as(f64, @floatFromInt(is_calls)) / @as(f64, @floatFromInt(rays)),
    });
    try testing.expect(mismatch * 2000 <= rays);
    try testing.expect(hits > rays / 4);
}

test "RT-Pfad (CPU-Emulation) = volle DAG-Traversierung" {
    try rtEquivalence(6, 3, 4000);
    try rtEquivalence(6, 4, 4000);
    try rtEquivalence(7, 5, 3000);
    try rtEquivalence(5, 9, 2000); // größer als die Geometrie: ein Primitiv
}

// ---------------------------------------------------------------------------
// Shading und Nachbearbeitung
// ---------------------------------------------------------------------------

const white = (@as(u32, 255) << 24) | (@as(u32, 255) << 16) | (@as(u32, 255) << 8);

/// Boden (y < 4) und ein schwebender Würfel [24,40) x [20,36) x [24,40); Material 0 weiß,
/// Material 1 emittierend bei x < 4 auf dem Boden.
fn lightScene(user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32 {
    _ = user;
    if (y < 4) return if (x < 4) white | 1 else white;
    if (x >= 24 and x < 40 and y >= 20 and y < 36 and z >= 24 and z < 40) return white;
    return 0;
}

const LightFixture = struct {
    dag: pyrit.dag_builder.Dag,
    geo: types.GeometryData,
    inst: types.InstanceData,
    mats: [types.max_materials]types.Material,
    light: types.Lighting,
    scene: types.Scene,

    fn init(self: *LightFixture) !void {
        self.dag = try pyrit.dag_builder.buildFn(gpa, 6, lightScene, null, null, true);
        self.geo = .{ .node_offset = 0, .leaf_offset = 0, .attribute_offset = 0, .root = self.dag.root, .log2_size = 6, .flags = types.geometry_has_attributes, .default_attribute = 1, .reserved = 0 };
        self.inst = common.makeInstance(0, 64, common.makeTransform(.{ 0, 1, 0 }, 0, 1, .{ 0, 0, 0 }), 1);
        for (&self.mats) |*m| pyrit.pyr_material_default(m);
        self.mats[1].emission = .{ 5, 4, 3 };
        pyrit.pyr_lighting_default(&self.light);
        self.light.sun_direction = .{ 1, 1, 0 };
        self.light.sun_angular_radius = 0;
        self.light.sun_color = .{ 1, 1, 1 };
        self.light.sky_intensity = 0;
        self.light.flags = types.lighting_shadows;
        var h = common.HostDag{ .dag = self.dag, .geometry = self.geo };
        self.scene = common.hostScene(&h, @ptrCast(&self.geo), @ptrCast(&self.inst), @ptrCast(&self.inst), 1);
        self.scene.materials = @intFromPtr(&self.mats);
        self.scene.lighting = @intFromPtr(&self.light);
    }

    fn deinit(self: *LightFixture) void {
        self.dag.deinit(gpa);
    }

    /// Farbe des Bodens bei (x, z), von oben gesehen
    fn floor(self: *LightFixture, x: f32, z: f32, frame: u32) [3]f32 {
        const tracer = pyr.render.SoftwareTracer{};
        const o = vec.Vec3{ x, 60, z };
        const d = vec.Vec3{ 0, -1, 0 };
        const h = pyr.traceScene(&self.scene, o, d, 0, types.flt_max, 0xFFFF_FFFF, 0).?;
        var rng = pyr.shade.Rng.init(@intFromFloat(x), @intFromFloat(z), frame, 0);
        const sh = pyr.shade.shadeHit(tracer, &self.scene, o, d, h, &rng, 0xFFFF_FFFF, 0);
        return sh.color;
    }
};

test "Shading: Lambert, Schatten der Sonne, Emission" {
    var f: LightFixture = undefined;
    try f.init();
    defer f.deinit();

    // Unbeschatteter Boden unter 45° Sonne: Albedo · E · cos45 (+ kleiner Glanzanteil)
    const lit = f.floor(10.5, 8.5, 0);
    try testing.expect(lit[1] > 0.65 and lit[1] < 0.85);
    // Schatten des Würfels fällt nach -x: Boden bei x in [0,24), z in [24,40)
    const shadow = f.floor(16.5, 30.5, 0);
    try testing.expect(shadow[1] < 0.01);
    // Rechts vom Würfel (x > 40) kein Schatten
    const right = f.floor(50.5, 30.5, 0);
    try testing.expect(right[1] > 0.65);
    // Emission (Material 1 bei x < 4) addiert sich
    const glow = f.floor(1.5, 8.5, 0);
    try testing.expect(glow[0] > 5.0 and glow[0] > glow[2]);
    print("  Licht {d:.3}, Schatten {d:.4}, Emission {d:.2}\n", .{ lit[1], shadow[1], glow[0] });

    // Mit GI und Himmel wird der Schatten aufgehellt, bleibt aber dunkler
    f.light.flags = types.lighting_shadows | types.lighting_gi;
    f.light.sky_intensity = 0.5;
    var gi_sum: f32 = 0;
    for (0..256) |k| gi_sum += f.floor(16.5, 30.5, @intCast(k))[1];
    const gi = gi_sum / 256;
    try testing.expect(gi > 0.02 and gi < lit[1]);
    print("  Schatten mit GI (Mittel über 256 Frames): {d:.3}\n", .{gi});
}

test "Nachbearbeitung: temporale Akkumulation, À-trous, Tonemapping" {
    const w = 32;
    const h = 32;
    const n = w * h;
    const V4 = [4]f32;
    var color: [n]V4 = undefined;
    var normal: [n]V4 = undefined;
    var albedo: [n]V4 = undefined;
    var motion = [_][2]f32{.{ 0, 0 }} ** n;
    var hits = [_]types.Hit{.{ .t = 1, .instance = 0, .attribute = 1, .meta = 0 }} ** n;
    // interne Puffer der Nachbearbeitung liegen halbgenau
    const H4 = [4]f16;
    var hist = [2][n]H4{ undefined, undefined };
    var hist_n = [2][n]H4{ undefined, undefined };
    var tmp = [2][n]H4{ undefined, undefined };
    var ldr: [n][4]u8 = undefined;
    for (&normal) |*v| v.* = .{ 0, 1, 0, 10 };
    for (&albedo) |*v| v.* = .{ 0.5, 0.5, 0.5, 1 };

    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    var p = std.mem.zeroes(types.PostParams);
    p.width = w;
    p.height = h;
    p.color = @intFromPtr(&color);
    p.normal = @intFromPtr(&normal);
    p.albedo = @intFromPtr(&albedo);
    p.motion = @intFromPtr(&motion);
    p.hits = @intFromPtr(&hits);
    p.alpha_min = 0.02;
    p.exposure = 1;

    // Rauschen um den Mittelwert 0.5 (Beleuchtung 1.0 bei Albedo 0.5)
    const frames = 64;
    var cur: usize = 0;
    for (0..frames) |fi| {
        for (&color) |*c| {
            const v = 0.5 * (1 + (rnd.float(f32) * 2 - 1) * 0.8);
            c.* = .{ v, v, v, 1 };
        }
        p.hist_color = @intFromPtr(&hist[cur ^ 1]);
        p.hist_normal = @intFromPtr(&hist_n[cur ^ 1]);
        p.out_color = @intFromPtr(&hist[cur]);
        p.out_normal = @intFromPtr(&hist_n[cur]);
        p.reset = @intFromBool(fi == 0);
        for (0..h) |y| for (0..w) |x| pyr.post.temporal(&p, @intCast(x), @intCast(y));
        cur ^= 1;
    }
    const acc = &hist[cur ^ 1];
    var mean: f32 = 0;
    var m2: f32 = 0;
    for (acc) |c| {
        const v: f32 = c[0];
        mean += v;
        m2 += v * v;
    }
    mean /= n;
    const sd_temporal = @sqrt(m2 / n - mean * mean);
    // ein Frame: Standardabweichung der Beleuchtung 0.8/sqrt(3) ~ 0.46
    try testing.expect(@abs(mean - 1.0) < 0.05);
    try testing.expect(sd_temporal < 0.1);

    // À-trous glättet weiter
    p.src = @intFromPtr(acc);
    var it: u32 = 0;
    while (it < 3) : (it += 1) {
        p.dst = @intFromPtr(&tmp[it & 1]);
        p.step = @as(u32, 1) << @intCast(it);
        for (0..h) |y| for (0..w) |x| pyr.post.atrous(&p, @intCast(x), @intCast(y));
        p.src = p.dst;
    }
    const filtered = @as(*const [n]H4, @ptrFromInt(p.src));
    mean = 0;
    m2 = 0;
    for (filtered) |c| {
        const v: f32 = c[0];
        mean += v;
        m2 += v * v;
    }
    mean /= n;
    const sd_spatial = @sqrt(m2 / n - mean * mean);
    try testing.expect(sd_spatial < sd_temporal);

    // Tonemapping: Albedo 0.5 · Beleuchtung 1.0 = 0.5 linear, ACES -> ~0.6 -> sRGB ~200
    p.out_ldr = @intFromPtr(&ldr);
    for (0..h) |y| for (0..w) |x| pyr.post.resolve(&p, @intCast(x), @intCast(y));
    try testing.expect(ldr[0][0] > 150 and ldr[0][0] < 230 and ldr[0][3] == 255);
    print("  Abweichung: temporal {d:.3}, zusätzlich räumlich {d:.3}; LDR {d}\n", .{ sd_temporal, sd_spatial, ldr[0][0] });

    // Neue Instanz verwirft den Verlauf
    hits[5].meta = types.hit_new;
    color[5] = .{ 2, 2, 2, 1 };
    p.hist_color = @intFromPtr(acc);
    p.hist_normal = @intFromPtr(&hist_n[cur ^ 1]);
    p.out_color = @intFromPtr(&hist[cur]);
    p.out_normal = @intFromPtr(&hist_n[cur]);
    p.reset = 0;
    pyr.post.temporal(&p, 5, 0);
    try testing.expectApproxEqAbs(@as(f32, 4.0), @as(f32, hist[cur][5][0]), 1e-2);
    try testing.expectEqual(@as(f32, 1), @as(f32, hist[cur][5][3]));
}

test "Nachbearbeitung: varianzgeführter Filter entrauscht dunkle Flächen" {
    // Nachbau des gemeldeten Falls: dunkle, indirekt beleuchtete Fläche mit
    // 1-Sample-Rauschen. Die alte Helligkeitstoleranz (4 * Helligkeit / sqrt(n))
    // hält dort das Rauschen für Kanten; die gemessene Varianz nicht.
    const w = 48;
    const h = 48;
    const n = w * h;
    const V4 = [4]f32;
    const H4 = [4]f16;
    var color: [n]V4 = undefined;
    var normal: [n]V4 = undefined;
    var albedo: [n]V4 = undefined;
    var motion = [_][2]f32{.{ 0, 0 }} ** n;
    var hits = [_]types.Hit{.{ .t = 1, .instance = 0, .attribute = 1, .meta = 0 }} ** n;
    var hist = [2][n]H4{ undefined, undefined };
    var hist_n = [2][n]H4{ undefined, undefined };
    var tmp = [2][n]H4{ undefined, undefined };
    var mom = [2][n][2]f16{ undefined, undefined };
    var vbuf = [3][n]f16{ undefined, undefined, undefined };
    for (&normal) |*v| v.* = .{ 0, 1, 0, 10 };
    for (&albedo) |*v| v.* = .{ 0.5, 0.5, 0.5, 1 };

    // Referenz: linke Hälfte dunkel (0.08), rechte Hälfte hell (0.5)
    var truth: [n]f32 = undefined;
    for (0..h) |y| for (0..w) |x| {
        truth[y * w + x] = if (x < w / 2) 0.08 else 0.5;
    };

    const frames = 64;
    var results: [2]struct { sd: f32, edge: f32 } = undefined;
    for (0..2) |variant| {
        const guided = variant == 1;
        var prng = std.Random.DefaultPrng.init(17);
        const rnd = prng.random();
        var p = std.mem.zeroes(types.PostParams);
        p.width = w;
        p.height = h;
        p.color = @intFromPtr(&color);
        p.normal = @intFromPtr(&normal);
        p.albedo = @intFromPtr(&albedo);
        p.motion = @intFromPtr(&motion);
        p.hits = @intFromPtr(&hits);
        // wenige Frames wie bei bewegter Kamera: hier überlebt das Rauschen
        p.alpha_min = 0.2;
        p.exposure = 1;
        p.phi_lum = 4;

        var cur: usize = 0;
        for (0..frames) |fi| {
            for (&color, 0..) |*c, k| {
                // volles Einzelsample-Rauschen (Faktor 0 .. 2 der Referenz)
                const v = truth[k] * (rnd.float(f32) * 2);
                c.* = .{ v * 0.5, v * 0.5, v * 0.5, 1 };
            }
            p.hist_color = @intFromPtr(&hist[cur ^ 1]);
            p.hist_normal = @intFromPtr(&hist_n[cur ^ 1]);
            p.out_color = @intFromPtr(&hist[cur]);
            p.out_normal = @intFromPtr(&hist_n[cur]);
            if (guided) {
                p.hist_moments = @intFromPtr(&mom[cur ^ 1]);
                p.out_moments = @intFromPtr(&mom[cur]);
                p.out_var = @intFromPtr(&vbuf[0]);
            }
            p.reset = @intFromBool(fi == 0);
            for (0..h) |y| for (0..w) |x| pyr.post.temporal(&p, @intCast(x), @intCast(y));
            cur ^= 1;
        }

        p.src = @intFromPtr(&hist[cur ^ 1]);
        if (guided) p.var_src = @intFromPtr(&vbuf[0]);
        var it: u32 = 0;
        while (it < 3) : (it += 1) {
            p.dst = @intFromPtr(&tmp[it & 1]);
            if (guided) p.var_dst = @intFromPtr(&vbuf[1 + (it & 1)]);
            p.step = @as(u32, 1) << @intCast(it);
            for (0..h) |y| for (0..w) |x| pyr.post.atrous(&p, @intCast(x), @intCast(y));
            p.src = p.dst;
            if (guided) p.var_src = p.var_dst;
        }
        const out = @as(*const [n]H4, @ptrFromInt(p.src));

        // Restrauschen auf der dunklen Fläche (Rand und Kante ausgenommen)
        var m1: f32 = 0;
        var m2: f32 = 0;
        var cnt: f32 = 0;
        for (4..h - 4) |y| for (4..w / 2 - 4) |x| {
            const v: f32 = out[y * w + x][0];
            m1 += v;
            m2 += v * v;
            cnt += 1;
        };
        const mean = m1 / cnt;
        const sd = @sqrt(@max(m2 / cnt - mean * mean, 0));
        // Kantenschärfe: Sprung über die Mitte, bezogen auf den wahren Sprung
        var lo: f32 = 0;
        var hi: f32 = 0;
        for (4..h - 4) |y| {
            lo += out[y * w + w / 2 - 2][0];
            hi += out[y * w + w / 2 + 1][0];
        }
        const rows: f32 = @floatFromInt(h - 8);
        results[variant] = .{ .sd = sd, .edge = (hi - lo) / rows / (0.5 - 0.08) };
    }

    print("  dunkle Fläche: Restrauschen alt {d:.4} -> geführt {d:.4}; Kantensprung {d:.2} -> {d:.2}\n", .{
        results[0].sd, results[1].sd, results[0].edge, results[1].edge,
    });
    // Deutlich weniger Rauschen, Kante bleibt erhalten
    try testing.expect(results[1].sd < results[0].sd * 0.7);
    try testing.expect(results[1].edge > 0.8);
}

test "Halton-Jitter" {
    var j: [2]f32 = undefined;
    pyrit.pyr_jitter_halton(0, &j);
    try testing.expectApproxEqAbs(@as(f32, 0.0), j[0], 1e-6); // 1/2 - 0.5
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0 - 0.5), j[1], 1e-6);
    for (0..100) |k| {
        pyrit.pyr_jitter_halton(@intCast(k), &j);
        try testing.expect(j[0] >= -0.5 and j[0] < 0.5 and j[1] >= -0.5 and j[1] < 0.5);
    }
}

// ---------------------------------------------------------------------------
// DAG-Bau auf der GPU (hier auf der CPU ausgeführt)
// ---------------------------------------------------------------------------

const gpu_build = pyrit.gpu_build;

fn builtView(b: *const gpu_build.Built) pyrit.dag_builder.Dag {
    return .{
        .log2_size = b.log2_size,
        .root = b.root,
        .voxel_count = b.voxel_count,
        .nodes = @as([*]u32, @ptrFromInt(b.nodes))[0..b.node_words],
        .leaves = @as([*]u64, @ptrFromInt(b.leaves))[0..b.leaf_count],
        .attributes = @as([*]u32, @ptrFromInt(b.attrs))[0..b.voxel_count],
    };
}

fn expectSameVoxels(a: *const pyrit.dag_builder.Dag, b: *const pyrit.dag_builder.Dag) !void {
    const n = @as(u32, 1) << @intCast(a.log2_size);
    var z: u32 = 0;
    while (z < n) : (z += 1) {
        var y: u32 = 0;
        while (y < n) : (y += 1) {
            var x: u32 = 0;
            while (x < n) : (x += 1) try testing.expectEqual(a.lookup(x, y, z), b.lookup(x, y, z));
        }
    }
}

test "GPU-DAG-Bau (CPU-Ausführung) = CPU-Builder, inklusive RT-Primitive" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    const e = cpu.exec();
    const log2 = 6;
    const n = 64;

    // Szene als Punktliste, gemischt, mit Duplikaten (der letzte gewinnt) und Ausreißern
    var pts: std.ArrayList([4]u32) = .empty;
    defer pts.deinit(gpa);
    var f = common.SceneFn{ .n = n };
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    for (0..n) |z| for (0..n) |y| for (0..n) |x| {
        const a = common.sceneVoxel(&f, @intCast(x), @intCast(y), @intCast(z));
        if (a != 0) try pts.append(gpa, .{ @intCast(x), @intCast(y), @intCast(z), a });
    };
    rnd.shuffle([4]u32, pts.items);
    const dup_count = 200;
    for (0..dup_count) |k| {
        var d = pts.items[k];
        d[3] = common.attrOf(@intCast(d[0]), @intCast(d[1]), @intCast(d[2]));
        try pts.append(gpa, d); // gleicher Wert, andere Reihenfolge
    }
    try pts.append(gpa, .{ @bitCast(@as(i32, -1)), 0, 0, 5 }); // außerhalb: wird verworfen
    try pts.append(gpa, .{ 64, 0, 0, 5 });

    var built = try gpu_build.build(e, log2, 4, 0, 0, 0, @intFromPtr(pts.items.ptr), @intCast(pts.items.len));
    defer built.free(e);
    const view = builtView(&built);

    var ref = try common.buildSceneDag(gpa, log2);
    defer ref.deinit(gpa);
    try expectSameVoxels(&ref.dag, &view);
    try testing.expectEqual(ref.dag.voxel_count, built.voxel_count);
    try testing.expectEqualSlices(u32, ref.dag.attributes.?, view.attributes.?);
    print("  GPU-Bau: {d} Voxel, {d} Worte (CPU {d}), {d} Blätter (CPU {d})\n", .{ built.voxel_count, built.node_words, ref.dag.nodes.len, built.leaf_count, ref.dag.leaves.len });

    // RT-Primitive wie rt_prims.extract (gleiche Reihenfolge: Morton = Tiefensuche)
    var prims = try pyrit.rt_prims.extract(gpa, &ref.dag, 4);
    defer prims.deinit(gpa);
    try testing.expectEqual(prims.prims.len, built.prim_count);
    const gp = @as([*]const types.RtPrim, @ptrFromInt(built.prims))[0..built.prim_count];
    const ga = @as([*]const pyrit.rt_prims.Aabb, @ptrFromInt(built.aabbs))[0..built.prim_count];
    for (prims.prims, prims.aabbs, gp, ga) |cp, ca, g, gab| {
        try testing.expectEqual(cp.cell, g.cell);
        try testing.expectEqual(cp.attr_base, g.attr_base);
        try testing.expectEqual(ca, gab);
    }
}

test "GPU-DAG-Bau: Änderungen (hinzufügen, überschreiben, löschen) = Neubau" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    const e = cpu.exec();

    var base: std.ArrayList([4]u32) = .empty;
    defer base.deinit(gpa);
    for (0..32) |z| for (0..32) |x| try base.append(gpa, .{ @intCast(x), 0, @intCast(z), 0x10 });
    var b1 = try gpu_build.build(e, 5, 3, 0, 0, 0, @intFromPtr(base.items.ptr), @intCast(base.items.len));
    defer b1.free(e);

    // Bearbeitung: Turm hinzufügen, Bodenstück löschen, ein Voxel umfärben
    var edits: std.ArrayList([4]u32) = .empty;
    defer edits.deinit(gpa);
    for (1..20) |y| try edits.append(gpa, .{ 10, @intCast(y), 10, 0x20 });
    for (0..8) |x| try edits.append(gpa, .{ @intCast(x), 0, 3, 0 });
    try edits.append(gpa, .{ 20, 0, 20, 0x30 });
    var b2 = try gpu_build.build(e, 5, 3, b1.keys, b1.attrs, b1.voxel_count, @intFromPtr(edits.items.ptr), @intCast(edits.items.len));
    defer b2.free(e);

    // Referenz: Endmenge direkt
    var final: std.ArrayList(pyrit.dag_builder.Point) = .empty;
    defer final.deinit(gpa);
    for (base.items) |v| {
        if (v[2] == 3 and v[0] < 8) continue;
        const a: u32 = if (v[0] == 20 and v[2] == 20) 0x30 else v[3];
        try final.append(gpa, .{ .x = @intCast(v[0]), .y = @intCast(v[1]), .z = @intCast(v[2]), .attribute = a });
    }
    for (1..20) |y| try final.append(gpa, .{ .x = 10, .y = @intCast(y), .z = 10, .attribute = 0x20 });
    var ref = try pyrit.dag_builder.buildPoints(gpa, 5, final.items, true);
    defer ref.deinit(gpa);
    const view = builtView(&b2);
    try expectSameVoxels(&ref, &view);
    try testing.expectEqual(@as(u32, @intCast(final.items.len)), b2.voxel_count);

    // Leere Geometrie
    var b3 = try gpu_build.build(e, 4, 4, 0, 0, 0, 0, 0);
    defer b3.free(e);
    const v3 = builtView(&b3);
    try testing.expectEqual(@as(?u32, null), v3.lookup(1, 1, 1));
    try testing.expectEqual(@as(u32, 0), b3.prim_count);
}

test "Erweiterte Treffer (Picking): Voxel, Position, Normale" {
    var h = try common.buildSceneDag(gpa, 6);
    defer h.deinit(gpa);
    const inst = common.makeInstance(0, 64, common.makeTransform(.{ 0, 1, 0 }, 0.5, 2.0, .{ 3, -1, 2 }), 1);
    const s = common.hostScene(&h, @ptrCast(&h.geometry), @ptrCast(&inst), @ptrCast(&inst), 1);
    var prng = std.Random.DefaultPrng.init(8);
    const rnd = prng.random();
    var checked: usize = 0;
    for (0..20_000) |_| {
        var o: vec.Vec3 = undefined;
        var d: vec.Vec3 = undefined;
        inline for (0..3) |a| {
            o[a] = (rnd.float(f32) * 2 - 1) * 200;
            d[a] = (rnd.float(f32) * 2 - 1) * 40 - o[a];
        }
        const th = pyr.traceScene(&s, o, d, 0, types.flt_max, 0xFFFF_FFFF, 0) orelse continue;
        const ex = pyr.scene.extendedHit(&s, o, d, th);
        if (th.face & types.hit_inside != 0) continue;
        // Voxel aus der Ruheposition = Voxel der Traversierung
        try testing.expectEqual(th.voxel, ex.voxel);
        // Normale zeigt gegen den Strahl
        try testing.expect(vec.dot(@as(vec.Vec3, ex.normal), d) < 0);
        checked += 1;
    }
    try testing.expect(checked > 1000);
}

fn waterVoxels(user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32 {
    _ = user;
    _ = x;
    _ = z;
    return if (y >= 4 and y < 12) (white | 3) else 0; // Material 3: Wasser
}

test "Transparente Ebene (Wasser) und Reflexionen" {
    var f: LightFixture = undefined;
    try f.init();
    defer f.deinit();
    var water = try pyrit.dag_builder.buildFn(gpa, 6, waterVoxels, null, null, true);
    defer water.deinit(gpa);

    // Beide Geometrien in gemeinsame Pools (Verweise sind relativ zum Geometriebeginn)
    const nodes = try std.mem.concat(gpa, u32, &.{ f.dag.nodes, water.nodes });
    defer gpa.free(nodes);
    const leaves = try std.mem.concat(gpa, u64, &.{ f.dag.leaves, water.leaves });
    defer gpa.free(leaves);
    const attrs = try std.mem.concat(gpa, u32, &.{ f.dag.attributes.?, water.attributes.? });
    defer gpa.free(attrs);
    var geos = [2]types.GeometryData{ f.geo, f.geo };
    geos[1].node_offset = @intCast(f.dag.nodes.len);
    geos[1].leaf_offset = @intCast(f.dag.leaves.len);
    geos[1].attribute_offset = @intCast(f.dag.attributes.?.len);
    geos[1].root = water.root;
    var inst = [2]types.InstanceData{ f.inst, f.inst };
    inst[0].mask = 0x1;
    inst[1].geometry = 1;
    inst[1].mask = 0x2;
    f.scene.nodes = @intFromPtr(nodes.ptr);
    f.scene.leaves = @intFromPtr(leaves.ptr);
    f.scene.attributes = @intFromPtr(attrs.ptr);
    f.scene.geometries = @intFromPtr(&geos);
    f.scene.instances = @intFromPtr(&inst);
    f.scene.instances_prev = @intFromPtr(&inst);
    f.scene.instance_count = 2;

    f.mats[3] = f.mats[0];
    f.mats[3].base_color = .{ 0.2, 0.6, 0.9 };
    f.mats[3].flags = 0;
    f.mats[3].density = 0.15;
    f.mats[3].roughness = 0.05;
    f.mats[3].ior = 1.33;
    f.light.sky_intensity = 0.5;
    f.light.flags = types.lighting_shadows;

    // Kamera senkrecht von oben
    var cam = std.mem.zeroes(types.Camera);
    common.lookAt(&cam, .{ 50, 60, 10 }, .{ 50, 0, 10 }, .{ 0, 0, -1 });
    pyrit.pyr_camera_orthographic(&cam, 4, 8, 8);
    var p = std.mem.zeroes(types.RenderParams);
    p.cur = common.cameraData(cam);
    p.prev = p.cur;
    p.history_valid = 1;
    p.color = 1;
    const tracer = pyr.render.SoftwareTracer{};

    p.ray_mask = 0x1; // ohne Wasser
    const dry = pyr.render.renderPixelWith(tracer, &p, &f.scene, 4, 4).color;
    p.ray_mask = 0xFF;
    p.transparent_mask = 0x2; // Wasser als transparente Ebene
    const wet = pyr.render.renderPixelWith(tracer, &p, &f.scene, 4, 4);
    try testing.expectEqual(@as(u32, 0), wet.hit.instance); // Treffer/MV vom Untergrund
    try testing.expect(wet.color[1] < dry[1]); // abgedunkelt
    try testing.expect(wet.color[2] / wet.color[0] > dry[2] / dry[0] * 1.5); // blau getönt
    print("  Boden trocken {d:.3},{d:.3},{d:.3} / unter Wasser {d:.3},{d:.3},{d:.3}\n", .{ dry[0], dry[1], dry[2], wet.color[0], wet.color[1], wet.color[2] });

    // Spiegelnder Boden: Reflexionsstrahlen zeigen den Himmel
    f.mats[0].roughness = 0.05;
    f.mats[0].metallic = 1;
    p.transparent_mask = 0;
    p.ray_mask = 0x1;
    var cam2 = std.mem.zeroes(types.Camera);
    common.lookAt(&cam2, .{ 60, 30, 60 }, .{ 50, 4, 10 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_perspective(&cam2, 0.3, 8, 8, 0.1);
    p.cur = common.cameraData(cam2);
    p.prev = p.cur;
    const no_refl = pyr.render.renderPixelWith(tracer, &p, &f.scene, 4, 4).color;
    f.light.flags |= types.lighting_reflections;
    const refl = pyr.render.renderPixelWith(tracer, &p, &f.scene, 4, 4).color;
    try testing.expect(refl[2] > no_refl[2] + 0.1);
    print("  Metallboden ohne/mit Reflexion (blau): {d:.3} / {d:.3}\n", .{ no_refl[2], refl[2] });
}

test "GPU-Verkleinerung (LOD) = Referenz" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    const e = cpu.exec();
    var pts: std.ArrayList([4]u32) = .empty;
    defer pts.deinit(gpa);
    var f = common.SceneFn{ .n = 64 };
    for (0..64) |z| for (0..64) |y| for (0..64) |x| {
        const a = common.sceneVoxel(&f, @intCast(x), @intCast(y), @intCast(z));
        if (a != 0) try pts.append(gpa, .{ @intCast(x), @intCast(y), @intCast(z), a });
    };
    var full = try gpu_build.build(e, 6, 3, 0, 0, 0, @intFromPtr(pts.items.ptr), @intCast(pts.items.len));
    defer full.free(e);
    for ([_]u32{ 1, 2, 3 }) |shift| {
        var lod = try gpu_build.downsample(e, 6, 3, full.keys, full.attrs, full.voxel_count, shift);
        defer lod.free(e);
        // Referenz: je grober Zelle das Quellvoxel mit dem größten Morton-Schlüssel
        var best: std.AutoHashMapUnmanaged([3]u32, struct { key: u64, attr: u32 }) = .empty;
        defer best.deinit(gpa);
        for (pts.items) |v| {
            const c = [3]u32{ v[0] >> @intCast(shift), v[1] >> @intCast(shift), v[2] >> @intCast(shift) };
            const key = (pyrit.dag_builder.morton(v[0] >> 2, v[1] >> 2, v[2] >> 2) << 6) | pyrit.dag_builder.brickBit(v[0] & 3, v[1] & 3, v[2] & 3);
            const gop = try best.getOrPut(gpa, c);
            if (!gop.found_existing or key > gop.value_ptr.key) gop.value_ptr.* = .{ .key = key, .attr = v[3] };
        }
        var ref_pts: std.ArrayList(pyrit.dag_builder.Point) = .empty;
        defer ref_pts.deinit(gpa);
        var it = best.iterator();
        while (it.next()) |kv| try ref_pts.append(gpa, .{ .x = @intCast(kv.key_ptr[0]), .y = @intCast(kv.key_ptr[1]), .z = @intCast(kv.key_ptr[2]), .attribute = kv.value_ptr.attr });
        var ref = try pyrit.dag_builder.buildPoints(gpa, 6 - shift, ref_pts.items, true);
        defer ref.deinit(gpa);
        const view = builtView(&lod);
        try expectSameVoxels(&ref, &view);
        print("  LOD 2^{d}: {d} Voxel\n", .{ 6 - shift, lod.voxel_count });
    }
}

test "GPU-Bau im Chunk-Batch = Einzelbau je Chunk" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    const e = cpu.exec();
    const log2 = 5;
    const n = 32;
    const k = 5;
    const cap = 8000;
    // Segmente: Chunk c enthält eine Kugel mit anderem Radius; Chunk 3 bleibt leer
    const vox = try gpa.alloc([4]u32, k * cap);
    defer gpa.free(vox);
    var cnt = [_]u32{0} ** k;
    for (0..k) |c| {
        if (c == 3) continue;
        const r: f32 = 6 + 2 * @as(f32, @floatFromInt(c));
        for (0..n) |z| for (0..n) |y| for (0..n) |x| {
            const dx = @as(f32, @floatFromInt(x)) - 15.5;
            const dy = @as(f32, @floatFromInt(y)) - 15.5;
            const dz = @as(f32, @floatFromInt(z)) - 15.5;
            const d2 = dx * dx + dy * dy + dz * dz;
            // nur Schale (Oberfläche), wie ein Generator es liefern soll
            if (d2 < r * r and d2 > (r - 2) * (r - 2) and cnt[c] < cap) {
                vox[c * cap + cnt[c]] = .{ @intCast(x), @intCast(y), @intCast(z), common.attrOf(@intCast(x + c), @intCast(y), @intCast(z)) };
                cnt[c] += 1;
            }
        };
    }
    var roots: [k]u32 = undefined;
    var first: [k]u32 = undefined;
    var prim: [k]u32 = undefined;
    var offs = [_]u32{0} ** (k + 1);
    for (0..k) |c| offs[c + 1] = offs[c] + cnt[c];
    var b = try gpu_build.buildChunks(e, log2, 4, .{ .count = k, .capacity = cap, .voxels = @intFromPtr(vox.ptr), .offsets = @intFromPtr(&offs), .total = offs[k] }, .{ .roots = &roots, .first_voxel = &first, .first_prim = &prim });
    defer b.free(e);
    try testing.expectEqual(@as(u32, 0xFFFF_FFFF), roots[3]);

    const nodes = @as([*]u32, @ptrFromInt(b.nodes))[0..b.node_words];
    const leaves = @as([*]u64, @ptrFromInt(b.leaves))[0..b.leaf_count];
    const attrs = @as([*]u32, @ptrFromInt(b.attrs))[0..b.voxel_count];
    const all_prims = @as([*]const types.RtPrim, @ptrFromInt(b.prims))[0..b.prim_count];
    for (0..k) |c| {
        if (c == 3) continue;
        // Einzelbau desselben Chunks
        var single = try gpu_build.build(e, log2, 4, 0, 0, 0, @intFromPtr(vox.ptr + c * cap), cnt[c]);
        defer single.free(e);
        const sv = builtView(&single);
        // Sicht auf den Chunk im Batch: gemeinsame Knoten, eigene Wurzel, Attribute ab first[c]
        const count_c = nodes[roots[c] + 1];
        const bv = pyrit.dag_builder.Dag{ .log2_size = log2, .root = roots[c], .voxel_count = count_c, .nodes = nodes, .leaves = leaves, .attributes = attrs[first[c]..] };
        try expectSameVoxels(&sv, &bv);
        // Primitive des Chunks: gleiche Zellen und Attributränge
        const sp = @as([*]const types.RtPrim, @ptrFromInt(single.prims))[0..single.prim_count];
        for (sp, 0..) |q, qi| {
            const g = all_prims[prim[c] + qi];
            try testing.expectEqual(q.cell, g.cell);
            try testing.expectEqual(q.attr_base, g.attr_base);
        }
    }
    print("  Batch aus {d} Chunks: {d} Worte, {d} Blätter, {d} Primitive\n", .{ k, b.node_words, b.leaf_count, b.prim_count });
}

/// Erzeugt und baut einen Chunk-Batch wie die Welt (Generator auf der CPU)
fn worldBatch(e: gpu_build.Exec, t: *const types.TerrainParams, keys: []const types.ChunkKey, cl: u32, cap: u32, out: gpu_build.ChunkOut) !struct { b: gpu_build.Built, total: u32 } {
    const k: u32 = @intCast(keys.len);
    const vox = try gpa.alloc([4]u32, k * cap);
    defer gpa.free(vox);
    const cnt = try gpa.alloc(u32, k);
    defer gpa.free(cnt);
    @memset(cnt, 0);
    const gp = types.WorldGenParams{ .chunks = @intFromPtr(keys.ptr), .voxels = @intFromPtr(vox.ptr), .counts = @intFromPtr(cnt.ptr), .count = k, .capacity = cap, .chunk_log2 = cl, .reserved = 0, .user = 0 };
    const threads = k << @intCast(2 * cl);
    var i: u32 = 0;
    while (i < threads) : (i += 1) pyr.worldgen.terrainColumn(&gp, t, i);
    const offs = try gpa.alloc(u32, k + 1);
    defer gpa.free(offs);
    offs[0] = 0;
    for (0..k) |c| {
        try testing.expect(cnt[c] <= cap);
        offs[c + 1] = offs[c] + cnt[c];
    }
    const b = try gpu_build.buildChunks(e, cl, cl - 1, .{ .count = k, .capacity = cap, .voxels = @intFromPtr(vox.ptr), .offsets = @intFromPtr(offs.ptr), .total = offs[k] }, out);
    return .{ .b = b, .total = offs[k] };
}

test "Welt: GPU-Gelände je LOD-Stufe, lückenlose Oberfläche" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    const e = cpu.exec();
    var t = pyrit.world_mod.defaultTerrain();
    t.base_height = 20;
    t.amplitude = 25;
    t.wavelength = 200;
    const cl = 5;
    const n = 32;
    for ([_]u32{ 0, 2 }) |lod| {
        const step: f32 = @floatFromInt(@as(u32, 1) << @intCast(lod));
        // 3 x 3 Spalten, y-Chunks über den ganzen Höhenbereich
        var keys: std.ArrayList(types.ChunkKey) = .empty;
        defer keys.deinit(gpa);
        const y_chunks: i32 = @intFromFloat(@ceil((t.base_height + t.amplitude * 1.6 + 2) / (n * step)));
        for (0..3) |z| for (0..3) |x| {
            var y: i32 = 0;
            while (y < y_chunks) : (y += 1) try keys.append(gpa, .{ .x = @intCast(x), .y = y, .z = @intCast(z), .lod = lod });
        };
        const k = keys.items.len;
        const roots = try gpa.alloc(u32, 3 * k);
        defer gpa.free(roots);
        const out = gpu_build.ChunkOut{ .roots = roots[0..k], .first_voxel = roots[k .. 2 * k], .first_prim = roots[2 * k ..] };
        var r = try worldBatch(e, &t, keys.items, cl, n * n * 8, out);
        defer r.b.free(e);
        const nodes = @as([*]u32, @ptrFromInt(r.b.nodes));
        const leaves = @as([*]u64, @ptrFromInt(r.b.leaves));
        const attrs = @as([*]u32, @ptrFromInt(r.b.attrs));

        // senkrechte Strahlen von oben: jede Spalte trifft, Höhe passt zum Höhenfeld
        var prng = std.Random.DefaultPrng.init(5 + lod);
        const rnd = prng.random();
        var worst: f32 = 0;
        for (0..2000) |_| {
            const wx = rnd.float(f32) * 3 * n * step;
            const wz = rnd.float(f32) * 3 * n * step;
            var best: ?f32 = null;
            for (keys.items, 0..) |key, c| {
                if (roots[c] == 0xFFFF_FFFF) continue;
                const cs = n * step;
                const ox = @as(f32, @floatFromInt(key.x)) * cs;
                const oy = @as(f32, @floatFromInt(key.y)) * cs;
                const oz = @as(f32, @floatFromInt(key.z)) * cs;
                if (wx < ox or wx >= ox + cs or wz < oz or wz >= oz + cs) continue;
                const g = pyr.dag.Dag{ .nodes = nodes, .leaves = leaves, .attributes = attrs + roots[k + c], .root = roots[c], .log2_size = cl, .default_attribute = 1 };
                const o = vec.Vec3{ (wx - ox) / step, n + 1, (wz - oz) / step };
                const h = pyr.dag.trace(&g, o, .{ 0, -1, 0 }, 0, types.flt_max, true) orelse continue;
                try testing.expect(h.attribute != 0);
                const y = oy + (n + 1 - h.t) * step;
                if (best == null or y > best.?) best = y;
            }
            const y = best orelse return error.LochImGelaende;
            // Höhenfeld in voller Auflösung an der Spaltenmitte der Stufe
            const cx = (@floor(wx / step) + 0.5) * step;
            const cz = (@floor(wz / step) + 0.5) * step;
            const ref = pyr.worldgen.height(&t, cx, cz, 2 * step);
            worst = @max(worst, @abs(y - ref));
        }
        const cols: f32 = 9 * n * n;
        print("  Stufe {d}: {d} Chunks, {d} Voxel ({d:.1} je Spalte), größte Höhenabweichung {d:.2} (Voxel {d})\n", .{ lod, k, r.total, @as(f32, @floatFromInt(r.total)) / cols, worst, step });
        try testing.expect(worst <= 0.5 * step + 1e-3);
        try testing.expect(@as(f32, @floatFromInt(r.total)) / cols < 8); // nur die Haut
    }
}

/// Testbild in Ausgabepixeln: feine Streifen (über der Nyquist-Grenze der
/// Renderauflösung) und eine scharfe Kante
fn pattern(x: f32, y: f32) f32 {
    const stripes = 0.5 + 0.5 * @sin(x * 1.3 + y * 0.4);
    const edge: f32 = if (x * 0.8 + y * 0.6 > 40) 1.0 else 0.2;
    return stripes * edge;
}

test "TAAU: 2x Hochskalieren rekonstruiert Details, auch bei Bewegung" {
    const iw = 48;
    const ih = 32;
    const ow = 96;
    const oh = 64;
    const color = try gpa.alloc([4]f32, iw * ih);
    defer gpa.free(color);
    const normal = try gpa.alloc([4]f32, iw * ih);
    defer gpa.free(normal);
    const motion = try gpa.alloc([2]f32, iw * ih);
    defer gpa.free(motion);
    const hits = try gpa.alloc(types.Hit, iw * ih);
    defer gpa.free(hits);
    var hist: [2][]([4]f16) = undefined;
    for (&hist) |*hb| hb.* = try gpa.alloc([4]f16, ow * oh);
    defer for (hist) |hb| gpa.free(hb);
    const mvd = try gpa.alloc([4]f32, ow * oh);
    defer gpa.free(mvd);
    const out = try gpa.alloc([4]f32, ow * oh);
    defer gpa.free(out);
    @memset(normal, .{ 0, 1, 0, 10 });
    @memset(hits, .{ .t = 1, .instance = 0, .attribute = 1, .meta = 0 });

    for ([_]f32{ 0, 0.75 }) |speed| {
        var err_taau: f64 = 0;
        var err_single: f64 = 0;
        const frames = 48;
        for (0..frames) |f| {
            var j: [2]f32 = undefined;
            pyrit.pyr_jitter_halton(@intCast(f), &j);
            const shift = speed * @as(f32, @floatFromInt(f)); // Ausgabepixel nach rechts
            // Renderpixel: Punktabtastung an der gejitterten Position
            for (0..ih) |y| for (0..iw) |x| {
                const sx = (@as(f32, @floatFromInt(x)) + 0.5 + j[0]) * 2;
                const sy = (@as(f32, @floatFromInt(y)) + 0.5 + j[1]) * 2;
                const v = pattern(sx - shift, sy);
                color[y * iw + x] = .{ v, v, v, 1 };
                // MV in Renderpixeln: der Punkt lag im Vorframe um speed/2 weiter links
                motion[y * iw + x] = .{ -speed / 2, 0 };
                hits[y * iw + x].meta = if (f == 0) types.hit_no_history else 0;
            };
            const cur = f & 1;
            const u = types.UpscaleParams{
                .in_width = iw, .in_height = ih, .out_width = ow, .out_height = oh,
                .color = @intFromPtr(color.ptr), .color_half = 0, .reserved = 0, .normal = @intFromPtr(normal.ptr), .motion = @intFromPtr(motion.ptr), .hits = @intFromPtr(hits.ptr),
                .jitter = j, .hist_in = @intFromPtr(hist[cur ^ 1].ptr), .hist_out = @intFromPtr(hist[cur].ptr), .mvd_out = @intFromPtr(mvd.ptr),
                .reset = @intFromBool(f == 0), .max_weight = 12, .exposure = 1, .tonemap = types.tonemap_none,
                .out_hdr = @intFromPtr(out.ptr), .out_ldr = 0, .bgra = 0, .reserved_bgra = 0,
            };
            for (0..oh) |y| for (0..ow) |x| pyr.upscale.taau(&u, @intCast(x), @intCast(y));
            if (f == 0 or f == frames - 1) {
                var e: f64 = 0;
                for (0..oh) |y| for (0..ow) |x| {
                    const ref = pattern(@as(f32, @floatFromInt(x)) + 0.5 - shift, @as(f32, @floatFromInt(y)) + 0.5);
                    const d = out[y * ow + x][0] - ref;
                    e += d * d;
                };
                const rmse = @sqrt(e / (ow * oh));
                if (f == 0) err_single = rmse else err_taau = rmse;
            }
        }
        print("  TAAU 2x, Bewegung {d:.2} px/Frame: Fehler erster Frame {d:.3}, nach {d} Frames {d:.3}\n", .{ speed, err_single, frames, err_taau });
        try testing.expect(err_taau < 0.75 * err_single);
    }
    // MV für die Frame Generation in Ausgabepixeln
    try testing.expectApproxEqAbs(@as(f32, -0.75), mvd[10 * ow + 10][0], 1e-5);
}

test "Frame Generation: Zwischenbild einer Verschiebung" {
    const w = 64;
    const h = 32;
    const prev = try gpa.alloc([4]f16, w * h);
    defer gpa.free(prev);
    const cur = try gpa.alloc([4]f16, w * h);
    defer gpa.free(cur);
    const mvd = try gpa.alloc([4]f32, w * h);
    defer gpa.free(mvd);
    const out = try gpa.alloc([4]f32, w * h);
    defer gpa.free(out);
    const img = struct {
        fn f(x: f32, y: f32) f32 {
            return 0.5 + 0.5 * @sin(x * 0.35) * @cos(y * 0.2);
        }
    };
    // Bild bewegt sich um 8 Pixel nach rechts; MV (jetzt -> vorher) = -8
    for (0..h) |y| for (0..w) |x| {
        const fx = @as(f32, @floatFromInt(x)) + 0.5;
        const fy = @as(f32, @floatFromInt(y)) + 0.5;
        const a = img.f(fx, fy);
        const b = img.f(fx - 8, fy);
        prev[y * w + x] = .{ @floatCast(a), @floatCast(a), @floatCast(a), 1 };
        cur[y * w + x] = .{ @floatCast(b), @floatCast(b), @floatCast(b), 1 };
        mvd[y * w + x] = .{ -8, 0, 10, 0 };
    };
    const p = types.FrameGenParams{ .width = w, .height = h, .t = 0.5, .exposure = 1, .prev_color = @intFromPtr(prev.ptr), .cur_color = @intFromPtr(cur.ptr), .motion_depth = @intFromPtr(mvd.ptr), .tonemap = types.tonemap_none, .reserved = 0, .out_hdr = @intFromPtr(out.ptr), .out_ldr = 0, .user = 0, .depth_mid = 0, .mv_mid = 0, .bgra = 0, .reserved_bgra = 0 };
    for (0..h) |y| for (0..w) |x| pyr.upscale.frameGen(&p, @intCast(x), @intCast(y));
    var e: f64 = 0;
    var e_blend: f64 = 0;
    var n: f64 = 0;
    for (0..h) |y| for (12..w - 12) |x| {
        const ref = img.f(@as(f32, @floatFromInt(x)) + 0.5 - 4, @as(f32, @floatFromInt(y)) + 0.5);
        const d = out[y * w + x][0] - ref;
        const blend = 0.5 * (@as(f32, prev[y * w + x][0]) + @as(f32, cur[y * w + x][0])) - ref;
        e += d * d;
        e_blend += blend * blend;
        n += 1;
    };
    const rmse = @sqrt(e / n);
    print("  Frame Generation t=0.5: Fehler {d:.4} (einfaches Überblenden {d:.4})\n", .{ rmse, @sqrt(e_blend / n) });
    try testing.expect(rmse < 0.01);
}

test "DAG-Austritt (transparente Medien) gegen Referenz-Abtastung" {
    const log2 = 6;
    const n = 64;
    var h = try common.buildSceneDag(gpa, log2);
    defer h.deinit(gpa);
    const dense = try common.denseScene(gpa, n);
    defer gpa.free(dense);
    const g = h.device();
    const solid = struct {
        fn at(dn: []const u32, p: vec.Vec3) bool {
            inline for (0..3) |a| if (p[a] < 0 or p[a] >= n) return false;
            const x: usize = @intFromFloat(p[0]);
            const y: usize = @intFromFloat(p[1]);
            const z: usize = @intFromFloat(p[2]);
            return dn[(z * n + y) * n + x] != 0;
        }
    };
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    var checked: usize = 0;
    var worst: f32 = 0;
    while (checked < 3000) {
        const o = vec.Vec3{ rnd.float(f32) * n, rnd.float(f32) * n, rnd.float(f32) * n };
        if (!solid.at(dense, o)) continue;
        const d = vec.normalize(vec.Vec3{ rnd.float(f32) * 2 - 1, rnd.float(f32) * 2 - 1, rnd.float(f32) * 2 - 1 });
        const ex = pyr.dag.traceExit(&g, o, d, 0, types.flt_max) orelse return error.KeinAustritt;
        // Referenz: feines Abtasten bis zum ersten leeren Punkt
        var t: f32 = 0;
        while (solid.at(dense, o + d * vec.splat(t))) t += 1e-3;
        worst = @max(worst, @abs(ex.t - t));
        // Außennormale zeigt in Laufrichtung
        const ax = ex.face >> 1;
        const sign: f32 = if (ex.face & 1 != 0) -1 else 1;
        const da: [3]f32 = d;
        try testing.expect(da[ax] * sign > 0);
        checked += 1;
    }
    print("  {d} Austritte, größte Abweichung {d:.4}\n", .{ checked, worst });
    try testing.expect(worst < 2e-3);
    // Start im Leeren: sofort
    try testing.expectEqual(@as(f32, 5), pyr.dag.traceExit(&g, .{ -10, 3, 3 }, .{ 1, 0, 0 }, 5, types.flt_max).?.t);
}

fn glassCube(user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32 {
    _ = user;
    return if (x >= 4 and x < 20 and y >= 8 and y < 20 and z >= 4 and z < 20) (white | 4) else 0;
}

/// Tracer, der den letzten Strahl gegen `mask` mitschreibt
const SpyTracer = struct {
    mask: u32,
    last: *[2]vec.Vec3,
    pub fn trace(self: SpyTracer, s: *const types.Scene, o: vec.Vec3, d: vec.Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?pyr.scene.TraceHit {
        if (ray_mask == self.mask) self.last.* = .{ o, d };
        return pyr.traceScene(s, o, d, tmin, tmax, ray_mask, flags);
    }
};

/// Fixture + zweite Geometrie als transparente Instanz (Maske 0x2)
const TransFixture = struct {
    f: LightFixture,
    extra: pyrit.dag_builder.Dag,
    nodes: []u32,
    leaves: []u64,
    attrs: []u32,
    geos: [2]types.GeometryData,
    inst: [2]types.InstanceData,

    fn init(self: *TransFixture, vf: pyrit.dag_builder.VoxelFn) !void {
        try self.f.init();
        self.extra = try pyrit.dag_builder.buildFn(gpa, 6, vf, null, null, true);
        const f = &self.f;
        self.nodes = try std.mem.concat(gpa, u32, &.{ f.dag.nodes, self.extra.nodes });
        self.leaves = try std.mem.concat(gpa, u64, &.{ f.dag.leaves, self.extra.leaves });
        self.attrs = try std.mem.concat(gpa, u32, &.{ f.dag.attributes.?, self.extra.attributes.? });
        self.geos = .{ f.geo, f.geo };
        self.geos[1].node_offset = @intCast(f.dag.nodes.len);
        self.geos[1].leaf_offset = @intCast(f.dag.leaves.len);
        self.geos[1].attribute_offset = @intCast(f.dag.attributes.?.len);
        self.geos[1].root = self.extra.root;
        self.inst = .{ f.inst, f.inst };
        self.inst[0].mask = 0x1;
        self.inst[1].geometry = 1;
        self.inst[1].mask = 0x2;
        f.scene.nodes = @intFromPtr(self.nodes.ptr);
        f.scene.leaves = @intFromPtr(self.leaves.ptr);
        f.scene.attributes = @intFromPtr(self.attrs.ptr);
        f.scene.geometries = @intFromPtr(&self.geos);
        f.scene.instances = @intFromPtr(&self.inst);
        f.scene.instances_prev = @intFromPtr(&self.inst);
        f.scene.instance_count = 2;
    }

    fn deinit(self: *TransFixture) void {
        gpa.free(self.nodes);
        gpa.free(self.leaves);
        gpa.free(self.attrs);
        self.extra.deinit(gpa);
        self.f.deinit();
    }

    /// Farbe entlang (o, d) mit transparenter Ebene; last = letzter Untergrundstrahl
    fn shade(self: *TransFixture, o: vec.Vec3, d: vec.Vec3, last: *[2]vec.Vec3) vec.Vec3 {
        const s = &self.f.scene;
        const spy = SpyTracer{ .mask = 0x1, .last = last };
        const oh = pyr.traceScene(s, o, d, 0, types.flt_max, 0x1, 0);
        var rng = pyr.shade.Rng.init(1, 1, 0, 0);
        const behind = if (oh) |h| pyr.shade.shadeHit(spy, s, o, d, h, &rng, 0x1, 0).color else pyr.shade.sky(@ptrFromInt(s.lighting), d, true);
        last.* = .{ o, d };
        return pyr.shade.transparentLayers(spy, s, o, d, 0, if (oh) |h| h.t else types.flt_max, behind, 0xFFFF_FFFF, 0x2, 0x1, 0x1, &rng);
    }
};

test "Transparenz: Brechung nach Snell, parallel versetzter Durchgang, unsichtbares Medium" {
    // Wasserfläche y = 12, Boden darunter
    var w: TransFixture = undefined;
    try w.init(waterVoxels);
    defer w.deinit();
    w.f.mats[3] = w.f.mats[0];
    w.f.mats[3].flags = types.material_refract;
    w.f.mats[3].ior = 1.33;
    w.f.light.flags = 0;
    var last: [2]vec.Vec3 = undefined;
    const d45 = vec.normalize(.{ 1, -1, 0 });
    _ = w.shade(.{ 20, 30, 10 }, d45, &last);
    // Untergrundstrahl nach der Brechung: sin(t) = sin(45°) / 1.33
    const dt = last[1];
    const sin_t = @sqrt(dt[0] * dt[0] + dt[2] * dt[2]);
    try testing.expectApproxEqAbs(@sqrt(0.5) / 1.33, sin_t, 1e-4);
    try testing.expect(dt[1] < 0);
    print("  Wasser: Einfall 45°, gebrochen {d:.2}° (Snell {d:.2}°)\n", .{ std.math.radiansToDegrees(std.math.asin(sin_t)), std.math.radiansToDegrees(std.math.asin(@as(f32, @sqrt(0.5) / 1.33))) });

    // Glasblock in der Luft: Richtung danach wie vorher, Strahl parallel versetzt
    var gl: TransFixture = undefined;
    try gl.init(glassCube);
    defer gl.deinit();
    gl.f.mats[4] = gl.f.mats[0];
    gl.f.mats[4].flags = types.material_refract;
    gl.f.mats[4].ior = 1.5;
    gl.f.light.flags = 0;
    const d = vec.normalize(.{ 1, -0.25, 0.1 });
    const o = vec.Vec3{ 0, 16, 10 };
    _ = gl.shade(o, d, &last);
    try testing.expect(vec.length(last[1] - d) < 1e-4);
    // Abstand der Geraden: Versatz > 0 (Strahl ging durch den Block)
    const off = (last[0] - o) - d * vec.splat(vec.dot(last[0] - o, d));
    try testing.expect(vec.length(off) > 0.5);
    print("  Glasblock: Richtung erhalten, seitlicher Versatz {d:.2} Voxel\n", .{vec.length(off)});

    // Kamera im Medium: Blick von unter Wasser nach oben ist getönt und dunkler
    {
        w.f.mats[3].density = 0.2;
        w.f.mats[3].base_color = .{ 0.2, 0.6, 0.9 };
        w.f.mats[3].flags = types.material_refract | types.material_voxel_color;
        w.f.light.flags = 0;
        w.f.light.sky_intensity = 1;
        const up = vec.Vec3{ 0, 1, 0 };
        const inside = w.shade(.{ 20, 8, 10 }, up, &last); // y=8 liegt im Wasser (4..12)
        const above = w.shade(.{ 20, 20, 10 }, up, &last); // darüber: freier Himmel
        print("  unter Wasser nach oben {d:.3},{d:.3},{d:.3} / über Wasser {d:.3},{d:.3},{d:.3}\n", .{ inside[0], inside[1], inside[2], above[0], above[1], above[2] });
        try testing.expect(inside[2] < above[2]); // abgeschwächt
        try testing.expect(inside[2] / @max(inside[0], 1e-4) > above[2] / @max(above[0], 1e-4) * 1.2); // blauer
    }

    // unsichtbares Medium: ior 1, keine Dichte, keine Deckkraft -> Farbe unverändert
    gl.f.mats[4].flags = 0;
    gl.f.mats[4].ior = 1.0;
    gl.f.mats[4].density = 0;
    gl.f.mats[4].opacity = 0;
    const s = &gl.f.scene;
    const oh = pyr.traceScene(s, o, d, 0, types.flt_max, 0x1, 0).?;
    var rng = pyr.shade.Rng.init(1, 1, 0, 0);
    const tracer = pyr.render.SoftwareTracer{};
    const behind = pyr.shade.shadeHit(tracer, s, o, d, oh, &rng, 0x1, 0).color;
    const c = pyr.shade.transparentLayers(tracer, s, o, d, 0, oh.t, behind, 0xFFFF_FFFF, 0x2, 0x1, 0x1, &rng);
    try testing.expect(vec.length(c - behind) < 1e-5);
}

fn quatAxis(axis: [3]f32, angle: f32) [4]f32 {
    const s = @sin(angle / 2);
    return .{ axis[0] * s, axis[1] * s, axis[2] * s, @cos(angle / 2) };
}

/// Referenz: Rotationsmatrix um eine Achse (f64), 3x4 zeilenweise mit Verschiebung
fn refRot(axis: [3]f64, angle: f64, t: [3]f64) [12]f64 {
    const c = @cos(angle);
    const s = @sin(angle);
    const x = axis[0];
    const y = axis[1];
    const z = axis[2];
    return .{
        c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, t[0],
        y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s, t[1],
        z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c),     t[2],
    };
}

fn refMul(a: [12]f64, b: [12]f64) [12]f64 {
    var r: [12]f64 = undefined;
    for (0..3) |i| for (0..4) |j| {
        var v = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j];
        if (j == 3) v += a[i * 4 + 3];
        r[i * 4 + j] = v;
    };
    return r;
}

test "Skelettanimation: Kette, Interpolation, Schleife, Überblenden, Inverse, AABB" {
    const pa = pyr.anim;
    const id = pa.identity;
    // Arm aus 3 Knochen entlang +x, Teile 4^3 Voxel (Kante 8 nach Log2 3)
    const bones = [_]types.AnimBone{
        .{ .parent = -1, .part_size = 8, .sway = 0, .reserved = 0, .rest = id, .part = id },
        .{ .parent = 0, .part_size = 8, .sway = 0, .reserved = 0, .rest = .{ 1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0 }, .part = .{ 1, 0, 0, -4, 0, 1, 0, 0, 0, 0, 1, 0 } },
        .{ .parent = 1, .part_size = 8, .sway = 0, .reserved = 0, .rest = .{ 1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0 }, .part = id },
    };
    const z = [3]f32{ 0, 0, 1 };
    const pi = std.math.pi;
    // Clip A: Schulter dreht in 1 s von 0 auf 90° um z; Ellbogen fest 30°
    const keys = [_]types.Keyframe{
        .{ .bone = 0, .time = 0, .translation = .{ 0, 0, 0 }, .scale = 1, .rotation = quatAxis(z, 0) },
        .{ .bone = 0, .time = 1, .translation = .{ 0, 0, 0 }, .scale = 1, .rotation = quatAxis(z, pi / 2.0) },
        .{ .bone = 1, .time = 0, .translation = .{ 10, 0, 0 }, .scale = 1, .rotation = quatAxis(z, pi / 6.0) },
        // Clip B (Knochen 0): um y gedreht
        .{ .bone = 0, .time = 0, .translation = .{ 0, 2, 0 }, .scale = 1, .rotation = quatAxis(.{ 0, 1, 0 }, pi / 2.0) },
    };
    const tracks = [_][2]u32{ .{ 0, 2 }, .{ 2, 1 }, .{ 0, 0 }, .{ 3, 1 }, .{ 0, 0 }, .{ 0, 0 } };
    const clips = [_]types.AnimClip{
        .{ .track_offset = 0, .bone_count = 3, .duration = 2, .flags = types.clip_loop },
        .{ .track_offset = 3, .bone_count = 3, .duration = 1, .flags = 0 },
    };
    var actors = [_]types.AnimActor{.{ .root = .{ 1, 0, 0, 100, 0, 1, 0, 50, 0, 0, 1, 0 }, .bone_offset = 0, .bone_count = 3, .clip = 0, .blend_clip = types.no_clip, .start_time = 10, .speed = 1, .blend_start_time = 0, .blend_speed = 1, .blend = 0, .wind_amplitude = 0, .wind_frequency = 0, .wind_phase = 0 }};
    const jobs = [_]types.AnimJob{ .{ .actor = 0, .bone = 0, .instance = 0, .reserved = 0 }, .{ .actor = 0, .bone = 1, .instance = 1, .reserved = 0 }, .{ .actor = 0, .bone = 2, .instance = 2, .reserved = 0 } };
    var inst: [3]types.InstanceData = undefined;
    var p = types.AnimParams{ .bones = @intFromPtr(&bones), .tracks = @intFromPtr(&tracks), .keys = @intFromPtr(&keys), .clips = @intFromPtr(&clips), .actors = @intFromPtr(&actors), .jobs = @intFromPtr(&jobs), .instances = @intFromPtr(&inst), .job_count = 3, .reserved = 0, .time = 0 };

    const root = [12]f64{ 1, 0, 0, 100, 0, 1, 0, 50, 0, 0, 1, 0 };
    const zd = [3]f64{ 0, 0, 1 };
    // Zeiten: Mitte des Clips, nach einer Schleife, vor dem Start (gehalten)
    for ([_]f64{ 10.5, 12.25, 9.0 }) |time| {
        p.time = time;
        for (0..3) |i| pa.run(&p, @intCast(i));
        var ct = time - 10;
        ct -= @floor(ct / 2) * 2;
        const ang = std.math.clamp(ct, 0, 1) * pi / 2.0;
        const b0 = refMul(root, refRot(zd, ang, .{ 0, 0, 0 }));
        const b1 = refMul(b0, refRot(zd, pi / 6.0, .{ 10, 0, 0 }));
        const b2 = refMul(b1, refRot(zd, 0, .{ 10, 0, 0 }));
        const expect = [3][12]f64{ b0, refMul(b1, refRot(zd, 0, .{ -4, 0, 0 })), b2 };
        for (0..3) |k| {
            for (0..12) |e| try testing.expectApproxEqAbs(expect[k][e], inst[k].object_to_world[e], 2e-4);
            // Inverse
            const back = pa.mul(&inst[k].world_to_object, &inst[k].object_to_world);
            for (0..12) |e| try testing.expectApproxEqAbs(id[e], back[e], 1e-4);
            // AABB enthält alle Ecken des Teils
            for (0..8) |c| {
                const q = [3]f32{ if (c & 1 != 0) 8 else 0, if (c & 2 != 0) 8 else 0, if (c & 4 != 0) 8 else 0 };
                const m = inst[k].object_to_world;
                for (0..3) |r| {
                    const v = m[r * 4] * q[0] + m[r * 4 + 1] * q[1] + m[r * 4 + 2] * q[2] + m[r * 4 + 3];
                    try testing.expect(v >= inst[k].bounds_min[r] - 1e-3 and v <= inst[k].bounds_max[r] + 1e-3);
                }
            }
        }
    }
    // Überblenden 50 %: Rotation zwischen A (45° um z bei t=0.5) und B (90° um y), Verschiebung halb
    actors[0].blend_clip = 1;
    actors[0].blend = 0.5;
    p.time = 10.5;
    pa.run(&p, 0);
    const qa = quatAxis(z, pi / 4.0);
    const qb = quatAxis(.{ 0, 1, 0 }, pi / 2.0);
    var qm: [4]f32 = undefined;
    var len: f32 = 0;
    for (0..4) |i| {
        qm[i] = 0.5 * qa[i] + 0.5 * qb[i];
        len += qm[i] * qm[i];
    }
    for (&qm) |*v| v.* /= @sqrt(len);
    const ref = (pa.Trs{ .t = .{ 0, 1, 0 }, .s = 1, .q = qm }).matrix();
    const want = pa.mul(&actors[0].root, &ref);
    for (0..12) |e| try testing.expectApproxEqAbs(want[e], inst[0].object_to_world[e], 1e-5);
    print("  3 Knochen: Kette, Schleife, Halten, Überblenden gegen f64-Referenz\n", .{});
}

/// Exec, der den Zwischenspeicher des Baus mitzählt
const CountingExec = struct {
    inner: gpu_build.Exec,
    live: u64 = 0,
    peak: u64 = 0,
    sizes: std.AutoHashMapUnmanaged(u64, u64) = .empty,

    fn exec(self: *CountingExec) gpu_build.Exec {
        const E = struct {
            fn cast(c: *anyopaque) *CountingExec {
                return @ptrCast(@alignCast(c));
            }
            fn alloc(c: *anyopaque, bytes: u64) gpu_build.Error!u64 {
                const s = cast(c);
                const p = try s.inner.alloc(s.inner.ctx, bytes);
                s.live += bytes;
                s.peak = @max(s.peak, s.live);
                s.sizes.put(gpa, p, bytes) catch return error.OutOfMemory;
                return p;
            }
            fn free(c: *anyopaque, p: u64) void {
                const s = cast(c);
                if (s.sizes.fetchRemove(p)) |kv| s.live -= kv.value;
                s.inner.free(s.inner.ctx, p);
            }
            fn memset(c: *anyopaque, p: u64, v: u8, bytes: u64) gpu_build.Error!void {
                const s = cast(c);
                return s.inner.memset(s.inner.ctx, p, v, bytes);
            }
            fn copy(c: *anyopaque, dst: u64, src: u64, bytes: u64) gpu_build.Error!void {
                const s = cast(c);
                return s.inner.copy(s.inner.ctx, dst, src, bytes);
            }
            fn launch(c: *anyopaque, p: *const types.BuildParams, threads: u32) gpu_build.Error!void {
                const s = cast(c);
                return s.inner.launch(s.inner.ctx, p, threads);
            }
            fn read(c: *anyopaque, dst: []u8, src: u64) gpu_build.Error!void {
                const s = cast(c);
                return s.inner.read(s.inner.ctx, dst, src);
            }
        };
        return .{ .ctx = self, .alloc = E.alloc, .free = E.free, .memset = E.memset, .copy = E.copy, .launch = E.launch, .read = E.read };
    }
};

test "Zwischenspeicher des GPU-Baus bleibt klein" {
    var cpu = gpu_build.CpuExec{ .gpa = gpa };
    defer cpu.deinit();
    var counting = CountingExec{ .inner = cpu.exec() };
    defer counting.sizes.deinit(gpa);
    const e = counting.exec();

    // dünn besetzte 256^3-Geometrie (Oberfläche einer Kugel)
    var pts: std.ArrayList([4]u32) = .empty;
    defer pts.deinit(gpa);
    const n = 256;
    for (0..n) |z| for (0..n) |x| {
        const dx = @as(f32, @floatFromInt(x)) - 127.5;
        const dz = @as(f32, @floatFromInt(z)) - 127.5;
        const r2 = 110.0 * 110.0 - dx * dx - dz * dz;
        if (r2 <= 0) continue;
        const y: u32 = @intFromFloat(127.5 + @sqrt(r2));
        try pts.append(gpa, .{ @intCast(x), y, @intCast(z), 0x00FF_00FF });
    };
    var b = try gpu_build.build(e, 8, 7, 0, 0, 0, @intFromPtr(pts.items.ptr), @intCast(pts.items.len));
    defer b.free(e);
    const per_voxel = @as(f64, @floatFromInt(counting.peak)) / @as(f64, @floatFromInt(pts.items.len));
    print("  {d} Voxel (256^3): Spitze {d:.1} MiB, {d:.0} Byte je Voxel\n", .{ pts.items.len, @as(f64, @floatFromInt(counting.peak)) / (1 << 20), per_voxel });
    try testing.expect(per_voxel < 130);
    try testing.expect(b.voxel_count == pts.items.len);
}

test "Indirekte Beleuchtung in halber Auflösung ~ volle Auflösung" {
    var f: LightFixture = undefined;
    try f.init();
    defer f.deinit();
    f.light.flags = types.lighting_shadows | types.lighting_gi;
    f.light.sky_intensity = 0.4;

    const w = 48;
    const h = 48;
    const n = w * h;
    var cam = std.mem.zeroes(types.Camera);
    common.lookAt(&cam, .{ 70, 45, 70 }, .{ 32, 8, 32 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_perspective(&cam, 0.9, w, h, 0.1);

    const color = try gpa.alloc([4]f32, n);
    defer gpa.free(color);
    const normal = try gpa.alloc([4]f32, n);
    defer gpa.free(normal);
    const albedo = try gpa.alloc([4]f32, n);
    defer gpa.free(albedo);
    const hits = try gpa.alloc(types.Hit, n);
    defer gpa.free(hits);
    const gi = try gpa.alloc([4]f16, ((w + 1) / 2) * ((h + 1) / 2));
    defer gpa.free(gi);

    var p = std.mem.zeroes(types.RenderParams);
    p.scene = @intFromPtr(&f.scene);
    p.cur = common.cameraData(cam);
    p.prev = p.cur;
    p.history_valid = 1;
    p.ray_mask = 0xFFFF_FFFF;
    p.hits = @intFromPtr(hits.ptr);
    p.color = @intFromPtr(color.ptr);
    p.normal = @intFromPtr(normal.ptr);
    p.albedo = @intFromPtr(albedo.ptr);
    p.gi = @intFromPtr(gi.ptr);
    p.gi_width = (w + 1) / 2;
    p.gi_height = (h + 1) / 2;
    const tracer = pyr.render.SoftwareTracer{};

    // Mittelwerte über mehrere Frames (ein Strahl je Pixel ist verrauscht)
    var mean_full: [3]f64 = .{ 0, 0, 0 };
    var mean_half: [3]f64 = .{ 0, 0, 0 };
    var hit_pixels: f64 = 0;
    const frames = 24;
    for (0..frames) |fi| {
        p.frame_index = @intCast(fi);
        // volle Auflösung
        f.light.flags = types.lighting_shadows | types.lighting_gi;
        for (0..h) |y| for (0..w) |x| {
            const r = pyr.render.renderPixelWith(tracer, &p, &f.scene, @intCast(x), @intCast(y));
            pyr.render.writePixel(&p, @as(u64, @intCast(y)) * w + @as(u64, @intCast(x)), &r);
        };
        for (0..n) |i| {
            if (color[i][3] < 0.5) continue;
            inline for (0..3) |k| mean_full[k] += color[i][k];
            if (fi == 0) hit_pixels += 1;
        }
        // halbe Auflösung: drei Durchgänge wie in pyr_render
        f.light.flags = types.lighting_shadows | types.lighting_gi | types.lighting_gi_half;
        for (0..h) |y| for (0..w) |x| {
            const r = pyr.render.renderPixelWith(tracer, &p, &f.scene, @intCast(x), @intCast(y));
            pyr.render.writePixel(&p, @as(u64, @intCast(y)) * w + @as(u64, @intCast(x)), &r);
        };
        for (0..p.gi_height) |y| for (0..p.gi_width) |x| pyr.render.giPixel(tracer, &p, &f.scene, @intCast(x), @intCast(y));
        for (0..h) |y| for (0..w) |x| pyr.render.combinePixel(&p, @intCast(x), @intCast(y));
        for (0..n) |i| {
            if (color[i][3] < 0.5) continue;
            inline for (0..3) |k| mean_half[k] += color[i][k];
        }
    }
    const norm = hit_pixels * frames;
    inline for (0..3) |k| {
        mean_full[k] /= norm;
        mean_half[k] /= norm;
    }
    const rel = @abs(mean_half[1] - mean_full[1]) / mean_full[1];
    print("  GI voll {d:.3},{d:.3},{d:.3} / halb {d:.3},{d:.3},{d:.3} (Abweichung {d:.1} %)\n", .{ mean_full[0], mean_full[1], mean_full[2], mean_half[0], mean_half[1], mean_half[2], rel * 100 });
    try testing.expect(rel < 0.05);
    try testing.expect(hit_pixels > @as(f64, n) * 0.3);
}

test "Wind: Kette biegt sich weich, Motion Vectors bleiben exakt" {
    const pa = pyr.anim;
    const id = pa.identity;
    // drei Knochen übereinander, jeder biegt sich ein Stück weiter
    const shift = [12]f32{ 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1, 0 };
    const bones = [_]types.AnimBone{
        .{ .parent = -1, .part_size = 4, .sway = 0.2, .reserved = 0, .rest = id, .part = id },
        .{ .parent = 0, .part_size = 4, .sway = 0.5, .reserved = 0, .rest = shift, .part = id },
        .{ .parent = 1, .part_size = 4, .sway = 1.0, .reserved = 0, .rest = shift, .part = id },
    };
    const tracks = [_][2]u32{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
    const clips = [_]types.AnimClip{.{ .track_offset = 0, .bone_count = 3, .duration = 0, .flags = 0 }};
    var actors = [_]types.AnimActor{.{ .root = id, .bone_offset = 0, .bone_count = 3, .clip = types.no_clip, .blend_clip = types.no_clip, .start_time = 0, .speed = 1, .blend_start_time = 0, .blend_speed = 0, .blend = 0, .wind_amplitude = 0.25, .wind_frequency = 0.5, .wind_phase = 0 }};
    const jobs = [_]types.AnimJob{ .{ .actor = 0, .bone = 0, .instance = 0, .reserved = 0 }, .{ .actor = 0, .bone = 1, .instance = 1, .reserved = 0 }, .{ .actor = 0, .bone = 2, .instance = 2, .reserved = 0 } };
    var inst: [3]types.InstanceData = undefined;
    var p = types.AnimParams{ .bones = @intFromPtr(&bones), .tracks = @intFromPtr(&tracks), .keys = 0, .clips = @intFromPtr(&clips), .actors = @intFromPtr(&actors), .jobs = @intFromPtr(&jobs), .instances = @intFromPtr(&inst), .job_count = 3, .reserved = 0, .time = 0.5 };

    // Auslenkung der Spitze bei maximalem Wind (t = 0,5 s -> sin = 1)
    for (0..3) |i| pa.run(&p, @intCast(i));
    const tip_x = inst[2].object_to_world[3];
    const mid_x = inst[1].object_to_world[3];
    // Auslenkung gegen die Drehrichtung; oben mehr als unten
    try testing.expect(@abs(tip_x) > @abs(mid_x) and @abs(mid_x) > 0);
    // Ruhelage bei sin = 0
    p.time = 1.0;
    for (0..3) |i| pa.run(&p, @intCast(i));
    try testing.expectApproxEqAbs(@as(f32, 0), inst[2].object_to_world[3], 1e-5);
    // Motion Vector: Lage zweier Zeitpunkte unterscheidet sich stetig
    p.time = 1.02;
    for (0..3) |i| pa.run(&p, @intCast(i));
    const moved = @abs(inst[2].object_to_world[3]);
    try testing.expect(moved > 0 and moved < 1);
    print("  Wind: Spitze {d:.3} bei voller Auslenkung, {d:.4} nach 20 ms\n", .{ @abs(tip_x), moved });
}
