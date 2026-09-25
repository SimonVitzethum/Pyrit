//! GPU-Test: Render und Trace über die öffentliche API auf der GPU, Ergebnis
//! Pixel für Pixel gegen denselben Gerätecode auf der CPU; danach Durchsatz.
//!
//!   zig build gpu-test      (nur bei freier GPU starten)

const std = @import("std");
const pyr = @import("pyrit_device");
const pyrit = @import("pyrit");
const demo = @import("demo");
const common = @import("common.zig");
const types = pyr.types;
const api = pyrit.api;
const cuda = pyrit.cuda;
const print = std.debug.print;

var drv: cuda.Driver = undefined;
var failures: usize = 0;

fn check(ok: bool, comptime msg: []const u8, args: anytype) void {
    if (ok) return;
    print("  FEHLER: " ++ msg ++ "\n", args);
    failures += 1;
}

fn cu(r: cuda.CUresult) void {
    if (r != cuda.CUDA_SUCCESS) std.debug.panic("CUDA: {s}", .{drv.errorString(r)});
}

fn req(r: api.Result) void {
    if (r != api.ok) std.debug.panic("{s}: {s}", .{ pyrit.pyr_result_string(r), pyrit.pyr_error_message() });
}

fn download(comptime T: type, gpa: std.mem.Allocator, p: u64, count: usize) ![]T {
    const v = try gpa.alloc(T, count);
    if (count > 0) cu(drv.cuMemcpyDtoH_v2(v.ptr, p, count * @sizeOf(T)));
    return v;
}

/// Upload auf den Stream des Kontexts: garantiert die Reihenfolge vor pyr_trace.
/// (cuMemcpyHtoD auf dem Standard-Stream kann zurückkehren, bevor die Daten auf
/// der GPU sind; Pyrits Stream wartet nicht auf den Standard-Stream.)
fn uploadOn(ctx: ?*anyopaque, dst: u64, src: *const anyopaque, bytes: usize) void {
    const s: cuda.CUstream = @ptrCast(pyrit.pyr_cuda_stream(@ptrCast(ctx)));
    cu(drv.cuMemcpyHtoDAsync_v2(dst, src, bytes, s));
}

fn devAlloc(bytes: usize) u64 {
    var p: cuda.CUdeviceptr = 0;
    cu(drv.cuMemAlloc_v2(&p, bytes));
    return p;
}

fn logFn(_: ?*anyopaque, _: i32, msg: [*:0]const u8) callconv(.c) void {
    print("  [pyrit] {s}\n", .{msg});
}

/// Liest die Szene von der GPU und baut eine gleichwertige Host-Szene.
const Mirror = struct {
    scene: types.Scene,
    arena: std.heap.ArenaAllocator,

    fn load(backing: std.mem.Allocator, ctx: ?*anyopaque, info: api.DagInfo) !Mirror {
        var m = Mirror{ .scene = undefined, .arena = .init(backing) };
        const a = m.arena.allocator();
        var dev: types.Scene = undefined;
        cu(drv.cuMemcpyDtoH_v2(&dev, pyrit.pyr_scene_device(@ptrCast(ctx)), @sizeOf(types.Scene)));
        m.scene = dev;
        // eine Geometrie am Poolanfang
        m.scene.nodes = @intFromPtr((try download(u32, a, dev.nodes, info.node_words)).ptr);
        m.scene.leaves = @intFromPtr((try download(u64, a, dev.leaves, info.leaf_count)).ptr);
        m.scene.attributes = @intFromPtr((try download(u32, a, dev.attributes, info.attribute_count)).ptr);
        m.scene.geometries = @intFromPtr((try download(types.GeometryData, a, dev.geometries, dev.geometry_count)).ptr);
        m.scene.instances = @intFromPtr((try download(types.InstanceData, a, dev.instances, dev.instance_count)).ptr);
        m.scene.instances_prev = @intFromPtr((try download(types.InstanceData, a, dev.instances_prev, dev.instance_count)).ptr);
        return m;
    }

    fn instancesEqual(m: *const Mirror) bool {
        const n = m.scene.instance_count;
        const a = std.mem.sliceAsBytes(pyr.scene.instances(&m.scene)[0..n]);
        const b = std.mem.sliceAsBytes(pyr.scene.instancesPrev(&m.scene)[0..n]);
        return std.mem.eql(u8, a, b);
    }
};

const Targets = struct {
    w: u32,
    h: u32,
    hits: u64,
    depth: u64,
    motion: u64,

    fn init(w: u32, h: u32) Targets {
        const n: usize = @as(usize, w) * h;
        return .{ .w = w, .h = h, .hits = devAlloc(n * 16), .depth = devAlloc(n * 4), .motion = devAlloc(n * 8) };
    }
    fn deinit(t: Targets) void {
        _ = drv.cuMemFree_v2(t.hits);
        _ = drv.cuMemFree_v2(t.depth);
        _ = drv.cuMemFree_v2(t.motion);
    }
    fn api_(t: Targets) api.Targets {
        return .{ .hits = t.hits, .depth = t.depth, .motion = t.motion, .color = 0, .normal = 0, .albedo = 0, .material = 0, .ray_mask = 0, .flags = 0, .transparent_mask = 0, .secondary_mask = 0, .coverage = 0 };
    }
};

fn compareFrame(gpa: std.mem.Allocator, ctx: ?*anyopaque, info: api.DagInfo, t: Targets, cam: types.Camera, prev_cam: ?types.Camera) !void {
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const n: usize = @as(usize, t.w) * t.h;
    const hits = try download(types.Hit, gpa, t.hits, n);
    defer gpa.free(hits);
    const depth = try download(f32, gpa, t.depth, n);
    defer gpa.free(depth);
    const motion = try download([2]f32, gpa, t.motion, n);
    defer gpa.free(motion);

    var m = try Mirror.load(gpa, ctx, info);
    defer m.arena.deinit();
    var p = std.mem.zeroes(types.RenderParams);
    p.cur = common.cameraData(cam);
    p.prev = common.cameraData(prev_cam orelse cam);
    p.history_valid = @intFromBool(prev_cam != null);
    p.ray_mask = 0xFFFF_FFFF;

    var same: usize = 0;
    var hit_count: usize = 0;
    var worst_mv: f32 = 0;
    var worst_depth: f32 = 0;
    for (0..t.h) |y| for (0..t.w) |x| {
        const i = y * t.w + x;
        const r = pyr.render.renderPixel(&p, &m.scene, @intCast(x), @intCast(y));
        if (r.hit.instance != types.no_hit) hit_count += 1;
        if (r.hit.instance != hits[i].instance or r.hit.attribute != hits[i].attribute or r.hit.meta != hits[i].meta) continue;
        same += 1;
        worst_mv = @max(worst_mv, @abs(r.motion[0] - motion[i][0]) + @abs(r.motion[1] - motion[i][1]));
        if (r.hit.instance != types.no_hit) worst_depth = @max(worst_depth, @abs(r.depth - depth[i]) / @max(1.0, r.depth));
    };
    print("  {d}/{d} Pixel identisch ({d} Treffer), MV-Abweichung max {e:.2} px, Tiefe rel. {e:.2}\n", .{ same, n, hit_count, worst_mv, worst_depth });
    check(same * 1000 >= n * 998, "GPU weicht zu oft von der CPU ab", .{});
    check(worst_mv < 1e-2, "MV-Abweichung GPU/CPU zu groß", .{});
    check(worst_depth < 1e-4, "Tiefenabweichung GPU/CPU zu groß", .{});
    check(hit_count > n / 20, "zu wenige Treffer", .{});
}

pub fn main(init: std.process.Init) !void {
    drv = cuda.Driver.load() catch {
        print("libcuda nicht gefunden\n", .{});
        return error.NoCuda;
    };
    cu(drv.cuInit(0));
    var dev: cuda.CUdevice = 0;
    cu(drv.cuDeviceGet(&dev, 0));
    var cu_ctx: cuda.CUcontext = null;
    cu(drv.cuDevicePrimaryCtxRetain(&cu_ctx, dev));
    cu(drv.cuCtxSetCurrent(cu_ctx));

    try animCheck(init, api.create_force_rt, "RT-Cores");
    try animCheck(init, api.create_no_rt, "CUDA");
    try worldCheck(init, 0, "RT-Cores");
    try worldCheck(init, api.create_no_rt, "CUDA");
    const rt_ms = try suite(init, api.create_force_rt, "RT-Cores (OptiX), erzwungen");
    const cuda_ms = try suite(init, api.create_no_rt, "CUDA-Traversierung");
    if (rt_ms > 0 and cuda_ms > 0) print("\nRT-Cores / CUDA: {d:.2}x schneller\n", .{cuda_ms / rt_ms});

    print("\n=== Volle Pipeline 1920x1080: Primärstrahl + Sonne/Punktlicht mit Schatten + GI + TAA/Denoiser/Tonemapping ===\n", .{});
    for ([_]u32{ 3, 1000 }) |count| {
        const cu_ms = try benchFull(init, api.create_no_rt, count);
        const rt_ms2 = try benchFull(init, api.create_force_rt, count);
        const auto_ms = try benchFull(init, 0, count);
        print("  {d:>5} Instanzen: CUDA {d:.2} ms, RT {d:.2} ms, automatisch {d:.2} ms\n", .{ count, cu_ms, rt_ms2, auto_ms });
    }

    print("\n=== Durchsatz 1920x1080 nach Szenengröße (ms pro Frame) ===\n", .{});
    print("  Instanzen | CUDA   | RT 2^3 | RT 2^4 | RT 2^5 | RT 2^6 | RT 2^7 | Standard\n", .{});
    for ([_]u32{ 3, 64, 1000 }) |count| {
        print("  {d:>9} |", .{count});
        print(" {d:>6.3} |", .{try bench(init, api.create_no_rt, 0, count)});
        for ([_]u32{ 3, 4, 5, 6, 7 }) |k| print(" {d:>6.3} |", .{try bench(init, api.create_force_rt, k, count)});
        print(" auto {d:>6.3}", .{try bench(init, 0, 0, count)});
        print("\n", .{});
    }

    _ = drv.cuDevicePrimaryCtxRelease_v2(dev);
    if (failures > 0) {
        print("{d} Fehler\n", .{failures});
        return error.TestFehlgeschlagen;
    }
    print("Alle GPU-Tests bestanden\n", .{});
}

/// Kompletter Durchlauf mit den gegebenen Kontext-Flags; liefert ms pro Frame (1080p).
fn suite(init: std.process.Init, extra_flags: u32, label: []const u8) !f32 {
    const gpa = init.gpa;
    print("\n=== {s} ===\nKontext anlegen\n", .{label});
    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = api.create_debug | extra_flags;
    ci.max_instances = 1024;
    ci.max_geometries = 64;
    ci.node_pool_bytes = 64 << 20;
    ci.leaf_pool_bytes = 64 << 20;
    ci.attribute_pool_bytes = 64 << 20;
    ci.staging_bytes = 8 << 20;
    ci.log = logFn;
    const start = std.Io.Timestamp.now(init.io, .awake);
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    const elapsed = start.untilNow(init.io, .awake).toNanoseconds();
    const rt = pyrit.pyr_features(@ptrCast(ctx)) & api.feature_rt_cores != 0;
    print("  {d:.0} ms, RT-Cores: {s}\n", .{ @as(f64, @floatFromInt(elapsed)) / 1e6, if (rt) "ja" else "nein" });
    if (extra_flags & api.create_no_rt != 0) check(!rt, "create_no_rt ignoriert", .{});
    const log2 = 7;
    var f = common.SceneFn{ .n = 1 << log2 };
    var dag: ?*anyopaque = null;
    req(pyrit.pyr_dag_build_fn(log2, common.sceneVoxel, null, &f, 0, @ptrCast(&dag)));
    defer pyrit.pyr_dag_destroy(@ptrCast(dag));
    var info: api.DagInfo = undefined;
    pyrit.pyr_dag_get_info(@ptrCast(dag), &info);
    print("DAG 128^3: {d} Worte, {d} Bricks, {d} Voxel\n", .{ info.node_words, info.leaf_count, info.voxel_count });

    var geo: api.Handle = null;
    req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(dag), &geo));
    var m = [3][12]f32{
        common.makeTransform(.{ 0, 1, 0 }, 0.0, 1.0, .{ -64, -64, -64 }),
        common.makeTransform(.{ 1, 2, 3 }, 0.7, 0.5, .{ 70, 10, -5 }),
        common.makeTransform(.{ -1, 0.3, 0.2 }, 2.1, 0.8, .{ -40, 50, 30 }),
    };
    var inst: [3]api.Handle = .{ null, null, null };
    for (0..3) |i| {
        req(pyrit.pyr_instance_create(@ptrCast(ctx), geo, &inst[i]));
        req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst[i], &m[i]));
    }
    check(pyrit.pyr_geometry_destroy(@ptrCast(ctx), geo) == api.error_in_use, "Geometrie in Benutzung nicht erkannt", .{});

    var view: api.Handle = null;
    req(pyrit.pyr_view_create(@ptrCast(ctx), &view));
    const t = Targets.init(320, 200);
    defer t.deinit();
    var cam1 = std.mem.zeroes(types.Camera);
    common.lookAt(&cam1, .{ 150, 90, 170 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_perspective(&cam1, 0.9, 320, 200, 0.1);
    var cam2 = cam1;
    common.lookAt(&cam2, .{ 146, 93, 172 }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    cam2.jitter = .{ 0.3, -0.2 };
    const tg = t.api_();

    print("Frame 1: GPU gegen CPU\n", .{});
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_render(@ptrCast(ctx), view, &cam1, &tg));
    try compareFrame(gpa, ctx, info, t, cam1, null);

    print("Frame 2: Bewegung, gelöschte und neue Instanz\n", .{});
    m[1] = common.makeTransform(.{ 1, 2, 3 }, 0.75, 0.5, .{ 72, 9, -5 });
    req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst[1], &m[1]));
    req(pyrit.pyr_instance_destroy(@ptrCast(ctx), inst[2]));
    var extra: api.Handle = null;
    req(pyrit.pyr_instance_create(@ptrCast(ctx), geo, &extra));
    m[2] = common.makeTransform(.{ 0, 0, 1 }, 0.3, 0.6, .{ 20, -80, 20 });
    req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), extra, &m[2]));
    check(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst[2], &m[2]) == api.error_invalid_handle, "gelöschte Instanz noch gültig", .{});
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_render(@ptrCast(ctx), view, &cam2, &tg));
    try compareFrame(gpa, ctx, info, t, cam2, cam1);

    print("Frame 3: ohne Änderungen (Copy-Forward)\n", .{});
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_render(@ptrCast(ctx), view, &cam2, &tg));
    try compareFrame(gpa, ctx, info, t, cam2, cam2);
    {
        var mm = try Mirror.load(gpa, ctx, info);
        defer mm.arena.deinit();
        check(mm.instancesEqual(), "Zustand nach Frame ohne Änderungen nicht identisch", .{});
    }

    print("Frame 4: nur Transformationen (IAS-Refit)\n", .{});
    m[1] = common.makeTransform(.{ 1, 2, 3 }, 0.8, 0.5, .{ 74, 8, -6 });
    req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst[1], &m[1]));
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_render(@ptrCast(ctx), view, &cam2, &tg));
    try compareFrame(gpa, ctx, info, t, cam2, cam2);

    print("Erweiterte Treffer (Picking): GPU gegen CPU\n", .{});
    for (0..5) |_| try extendedCheck(gpa, ctx, info);

    print("Shading (Sonne, Schatten, GI) + Nachbearbeitung: GPU gegen CPU\n", .{});
    try shadingCheck(gpa, ctx, info, view, cam2);

    print("DAG-Bau und Änderung auf der GPU\n", .{});
    try buildCheck(init, ctx);

    print("pyr_trace: 100000 Strahlen gegen CPU\n", .{});
    {
        const count = 100_000;
        const rays = try gpa.alloc(types.Ray, count);
        defer gpa.free(rays);
        var prng = std.Random.DefaultPrng.init(5);
        const rnd = prng.random();
        for (rays) |*r| {
            r.* = .{ .origin = undefined, .tmin = 0, .direction = undefined, .tmax = types.flt_max };
            for (0..3) |a| {
                r.origin[a] = (rnd.float(f32) * 2 - 1) * 250;
                r.direction[a] = (rnd.float(f32) * 2 - 1) * 60 - r.origin[a];
            }
        }
        const d_rays = devAlloc(count * @sizeOf(types.Ray));
        const d_hits = devAlloc(count * @sizeOf(types.Hit));
        defer _ = drv.cuMemFree_v2(d_rays);
        defer _ = drv.cuMemFree_v2(d_hits);
        uploadOn(ctx, d_rays, rays.ptr, count * @sizeOf(types.Ray));
        req(pyrit.pyr_trace(@ptrCast(ctx), d_rays, d_hits, count, 0, 0));
        req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const gpu = try download(types.Hit, gpa, d_hits, count);
        defer gpa.free(gpu);
        var mm = try Mirror.load(gpa, ctx, info);
        defer mm.arena.deinit();
        var same: usize = 0;
        var hitc: usize = 0;
        var diffs: usize = 0;
        for (rays, gpu) |r, g| {
            const h = pyr.trace(&mm.scene, r, 0xFFFF_FFFF, 0);
            if (h.instance != types.no_hit) hitc += 1;
            if (h.instance == g.instance and h.attribute == g.attribute and h.meta == g.meta) {
                same += 1;
            } else if (h.instance == g.instance and h.attribute == g.attribute and h.meta & g.meta & types.hit_inside != 0) {
                same += 1; // Start im Voxel: Fläche undefiniert
            } else if (diffs < 6) {
                diffs += 1;
                print("    CPU: inst {d} attr {x} meta {x} t {d:.4} | GPU: inst {d} attr {x} meta {x} t {d:.4}\n", .{ h.instance, h.attribute, h.meta, h.t, g.instance, g.attribute, g.meta, g.t });
            }
        }
        print("  {d}/{d} identisch, {d} Treffer\n", .{ same, count, hitc });
        check(same * 1000 >= count * 999, "pyr_trace weicht ab", .{});
    }

    print("Durchsatz 1920x1080 (Primärstrahlen + MV)\n", .{});
    var frame_ms: f32 = 0;
    {
        // Debug-Kontext synchronisiert nach jedem Kernel; für die Messung ohne Debug neu anlegen
        pyrit.pyr_destroy(@ptrCast(ctx));
        ci.flags = extra_flags;
        req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
        req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(dag), &geo));
        for (0..3) |i| {
            req(pyrit.pyr_instance_create(@ptrCast(ctx), geo, &inst[i]));
            req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst[i], &m[i]));
        }
        req(pyrit.pyr_view_create(@ptrCast(ctx), &view));
        const big = Targets.init(1920, 1080);
        defer big.deinit();
        const bt = big.api_();
        var c = cam2;
        pyrit.pyr_camera_perspective(&c, 0.9, 1920, 1080, 0.1);
        const s: cuda.CUstream = @ptrCast(pyrit.pyr_cuda_stream(@ptrCast(ctx)));
        var e0: cuda.CUevent = null;
        var e1: cuda.CUevent = null;
        cu(drv.cuEventCreate(&e0, 0));
        cu(drv.cuEventCreate(&e1, 0));
        const frames = 50;
        for (0..frames + 5) |fi| {
            if (fi == 5) cu(drv.cuEventRecord(e0, s));
            req(pyrit.pyr_commit(@ptrCast(ctx), null));
            req(pyrit.pyr_render(@ptrCast(ctx), view, &c, &bt));
        }
        cu(drv.cuEventRecord(e1, s));
        cu(drv.cuEventSynchronize(e1));
        var ms: f32 = 0;
        cu(drv.cuEventElapsedTime(&ms, e0, e1));
        const per = ms / frames;
        print("  {d:.3} ms pro Frame, {d:.0} Mrays/s\n", .{ per, 1920.0 * 1080.0 / (per * 1e3) });
        frame_ms = per;
        _ = drv.cuEventDestroy_v2(e0);
        _ = drv.cuEventDestroy_v2(e1);
    }

    pyrit.pyr_destroy(@ptrCast(ctx));
    return frame_ms;
}

/// Frame-Zeit bei 1080p für `count` Instanzen der 128^3-Testgeometrie auf einem Gitter.
fn bench(init: std.process.Init, flags: u32, rt_log2: u32, count: u32) !f32 {
    return benchImpl(init, flags, rt_log2, count, false);
}

fn benchFull(init: std.process.Init, flags: u32, count: u32) !f32 {
    return benchImpl(init, flags, 0, count, true);
}

fn benchImpl(init: std.process.Init, flags: u32, rt_log2: u32, count: u32, full: bool) !f32 {
    _ = init;
    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = flags;
    ci.max_instances = 4096;
    ci.max_geometries = 16;
    ci.rt_leaf_log2 = rt_log2;
    ci.node_pool_bytes = 64 << 20;
    ci.leaf_pool_bytes = 64 << 20;
    ci.attribute_pool_bytes = 64 << 20;
    ci.staging_bytes = 16 << 20;
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    defer pyrit.pyr_destroy(@ptrCast(ctx));

    var f = common.SceneFn{ .n = 128 };
    var dag: ?*anyopaque = null;
    req(pyrit.pyr_dag_build_fn(7, common.sceneVoxel, null, &f, 0, @ptrCast(&dag)));
    defer pyrit.pyr_dag_destroy(@ptrCast(dag));
    var geo: api.Handle = null;
    req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(dag), &geo));

    // Instanzen auf einem Gitter, leicht gedreht, Kantenlänge ~ Wurzel(count)
    const side: u32 = @intFromFloat(@ceil(@sqrt(@as(f32, @floatFromInt(count)))));
    const spacing: f64 = 150;
    for (0..count) |i| {
        var inst: api.Handle = null;
        req(pyrit.pyr_instance_create(@ptrCast(ctx), geo, &inst));
        const gx: f64 = @floatFromInt(i % side);
        const gz: f64 = @floatFromInt(i / side);
        const half = @as(f64, @floatFromInt(side)) * spacing * 0.5;
        const m = common.makeTransform(.{ 0, 1, 0 }, 0.3 * gx + 0.1 * gz, 1.0, .{ gx * spacing - half, -64, gz * spacing - half });
        req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst, &m));
    }
    var view: api.Handle = null;
    req(pyrit.pyr_view_create(@ptrCast(ctx), &view));
    const t = Targets.init(1920, 1080);
    defer t.deinit();
    var tg = t.api_();
    const npx: usize = 1920 * 1080;
    const extra = [_]u64{ devAlloc(npx * 16), devAlloc(npx * 16), devAlloc(npx * 16), devAlloc(npx * 4) };
    defer for (extra) |b| {
        _ = drv.cuMemFree_v2(b);
    };
    var post = std.mem.zeroes(api.PostInfo);
    if (full) {
        tg.color = extra[0];
        tg.normal = extra[1];
        tg.albedo = extra[2];
        post.output_ldr = extra[3];
        post.denoise_iterations = 4;
        post.clamp_sigma = 1.5;
        var l: types.Lighting = undefined;
        pyrit.pyr_lighting_default(&l);
        l.light_count = 1;
        l.lights[0] = std.mem.zeroes(types.Light);
        l.lights[0].position = .{ 0, 200, 0 };
        l.lights[0].radius = 5;
        l.lights[0].color = .{ 20000, 18000, 15000 };
        req(pyrit.pyr_set_lighting(@ptrCast(ctx), &l));
    }
    var cam = std.mem.zeroes(types.Camera);
    const dist: f32 = @as(f32, @floatFromInt(side)) * 150.0 * 0.6 + 150;
    common.lookAt(&cam, .{ dist * 0.7, dist * 0.45, dist }, .{ 0, 0, 0 }, .{ 0, 1, 0 });
    pyrit.pyr_camera_perspective(&cam, 0.9, 1920, 1080, 0.1);

    const s: cuda.CUstream = @ptrCast(pyrit.pyr_cuda_stream(@ptrCast(ctx)));
    var e0: cuda.CUevent = null;
    var e1: cuda.CUevent = null;
    cu(drv.cuEventCreate(&e0, 0));
    cu(drv.cuEventCreate(&e1, 0));
    defer _ = drv.cuEventDestroy_v2(e0);
    defer _ = drv.cuEventDestroy_v2(e1);
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    const frames = 40;
    for (0..frames + 5) |fi| {
        if (fi == 5) cu(drv.cuEventRecord(e0, s));
        if (full) {
            var j: [2]f32 = undefined;
            pyrit.pyr_jitter_halton(@intCast(fi), &j);
            cam.jitter = j;
            req(pyrit.pyr_commit(@ptrCast(ctx), null));
        }
        req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tg));
        if (full) req(pyrit.pyr_postprocess(@ptrCast(ctx), view, &tg, &post));
    }
    cu(drv.cuEventRecord(e1, s));
    cu(drv.cuEventSynchronize(e1));
    var ms: f32 = 0;
    cu(drv.cuEventElapsedTime(&ms, e0, e1));
    return ms / frames;
}

/// Rendert mit Shading, vergleicht die Farbe mit dem CPU-Code (gleiche Zufallszahlen)
/// und prüft die Nachbearbeitung.
fn shadingCheck(gpa: std.mem.Allocator, ctx: ?*anyopaque, info: api.DagInfo, view: api.Handle, cam: types.Camera) !void {
    const w = cam.width;
    const h = cam.height;
    const n: usize = @as(usize, w) * h;
    const color = devAlloc(n * 16);
    const normal = devAlloc(n * 16);
    const albedo = devAlloc(n * 16);
    const hits = devAlloc(n * 16);
    const motion = devAlloc(n * 8);
    const ldr = devAlloc(n * 4);
    defer for ([_]u64{ color, normal, albedo, hits, motion, ldr }) |b| {
        _ = drv.cuMemFree_v2(b);
    };
    var tg = std.mem.zeroes(api.Targets);
    tg.hits = hits;
    tg.motion = motion;
    tg.color = color;
    tg.normal = normal;
    tg.albedo = albedo;

    // Materialien mit Farbe: Attribute der Testszene sind Hashwerte (beliebige Farben)
    var m: types.Material = undefined;
    pyrit.pyr_material_default(&m);
    m.emission = .{ 0.5, 0.2, 0.1 };
    req(pyrit.pyr_material_set(@ptrCast(ctx), 7, &m));
    var l: types.Lighting = undefined;
    pyrit.pyr_lighting_default(&l);
    l.light_count = 1;
    l.lights[0] = std.mem.zeroes(types.Light);
    l.lights[0].position = .{ 40, 90, 40 };
    l.lights[0].radius = 3;
    l.lights[0].color = .{ 3000, 2500, 2000 };
    req(pyrit.pyr_set_lighting(@ptrCast(ctx), &l));

    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tg));
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const gpu_color = try download([4]f32, gpa, color, n);
    defer gpa.free(gpu_color);

    var mm = try Mirror.load(gpa, ctx, info);
    defer mm.arena.deinit();
    // Materialien und Licht auf den Host spiegeln
    var mats: [types.max_materials]types.Material = undefined;
    var dev_scene: types.Scene = undefined;
    cu(drv.cuMemcpyDtoH_v2(&dev_scene, pyrit.pyr_scene_device(@ptrCast(ctx)), @sizeOf(types.Scene)));
    cu(drv.cuMemcpyDtoH_v2(&mats, dev_scene.materials, @sizeOf(@TypeOf(mats))));
    var host_light: types.Lighting = undefined;
    cu(drv.cuMemcpyDtoH_v2(&host_light, dev_scene.lighting, @sizeOf(types.Lighting)));
    mm.scene.materials = @intFromPtr(&mats);
    mm.scene.lighting = @intFromPtr(&host_light);
    mm.scene.frame = dev_scene.frame;

    var p = std.mem.zeroes(types.RenderParams);
    p.cur = common.cameraData(cam);
    p.prev = p.cur;
    p.history_valid = 1;
    p.ray_mask = 0xFFFF_FFFF;
    p.color = 1; // Shading anfordern (Zeiger wird hier nicht benutzt)
    p.frame_index = @truncate(dev_scene.frame);

    var close: usize = 0;
    var checked: usize = 0;
    var sum_gpu: f64 = 0;
    var sum_cpu: f64 = 0;
    var y: u32 = 0;
    while (y < h) : (y += 3) {
        var x: u32 = 0;
        while (x < w) : (x += 3) {
            const r = pyr.render.renderPixel(&p, &mm.scene, x, y);
            const g = gpu_color[@as(usize, y) * w + x];
            checked += 1;
            var ok = true;
            inline for (0..3) |k| {
                if (!std.math.isFinite(g[k])) ok = false;
                if (@abs(g[k] - r.color[k]) > 0.02 + 0.05 * @abs(r.color[k])) ok = false;
            }
            close += @intFromBool(ok);
            sum_gpu += g[1];
            sum_cpu += r.color[1];
        }
    }
    const cn: f64 = @floatFromInt(checked);
    print("  Farbe: {d}/{d} Pixel innerhalb 5 %, Mittel GPU {d:.4} / CPU {d:.4}\n", .{ close, checked, sum_gpu / cn, sum_cpu / cn });
    // Sekundärstrahlen weichen durch schnelle GPU-Näherungen (sin/cos/log) gelegentlich ab
    check(close * 100 >= checked * 95, "Shading GPU/CPU weicht zu stark ab", .{});
    check(@abs(sum_gpu - sum_cpu) / cn < 0.02 * @max(sum_cpu / cn, 0.1), "mittlere Helligkeit GPU/CPU verschieden", .{});

    // Nachbearbeitung über mehrere Frames
    var post = std.mem.zeroes(api.PostInfo);
    post.output_ldr = ldr;
    post.denoise_iterations = 4;
    post.clamp_sigma = 0;
    for (0..8) |_| {
        req(pyrit.pyr_commit(@ptrCast(ctx), null));
        req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tg));
        req(pyrit.pyr_postprocess(@ptrCast(ctx), view, &tg, &post));
    }
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const img = try download([4]u8, gpa, ldr, n);
    defer gpa.free(img);
    var lit: usize = 0;
    for (img) |px| lit += @intFromBool(px[0] > 5 or px[1] > 5 or px[2] > 5);
    print("  Nachbearbeitung: {d}/{d} Pixel mit Helligkeit\n", .{ lit, n });
    check(lit * 2 > n, "Nachbearbeitung liefert ein leeres Bild", .{});
}

/// Traced dieselben Strahlen gegen zwei Instanzen (Masken a, b) und zählt gleiche Treffer.
fn sameHits(gpa: std.mem.Allocator, ctx: ?*anyopaque, mask_a: u32, mask_b: u32) !usize {
    const count = 100_000;
    const rays = try gpa.alloc(types.Ray, count);
    defer gpa.free(rays);
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    for (rays) |*r| {
        r.* = .{ .origin = undefined, .tmin = 0, .direction = undefined, .tmax = types.flt_max };
        for (0..3) |a| {
            r.origin[a] = 500 + (rnd.float(f32) * 2 - 1) * 250;
            r.direction[a] = 500 + (rnd.float(f32) * 2 - 1) * 60 - r.origin[a];
        }
    }
    const d_rays = devAlloc(count * @sizeOf(types.Ray));
    const d_a = devAlloc(count * @sizeOf(types.Hit));
    const d_b = devAlloc(count * @sizeOf(types.Hit));
    defer for ([_]u64{ d_rays, d_a, d_b }) |b| {
        _ = drv.cuMemFree_v2(b);
    };
    uploadOn(ctx, d_rays, rays.ptr, count * @sizeOf(types.Ray));
    req(pyrit.pyr_trace(@ptrCast(ctx), d_rays, d_a, count, mask_a, 0));
    req(pyrit.pyr_trace(@ptrCast(ctx), d_rays, d_b, count, mask_b, 0));
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const a = try download(types.Hit, gpa, d_a, count);
    defer gpa.free(a);
    const b = try download(types.Hit, gpa, d_b, count);
    defer gpa.free(b);
    var same: usize = 0;
    var hits: usize = 0;
    for (a, b) |x, y| {
        hits += @intFromBool(x.instance != types.no_hit);
        const both_miss = x.instance == types.no_hit and y.instance == types.no_hit;
        const both_hit = x.instance != types.no_hit and y.instance != types.no_hit;
        if (both_miss or (both_hit and x.attribute == y.attribute and x.meta == y.meta and @abs(x.t - y.t) <= 1e-4 * @max(1, x.t))) same += 1;
    }
    print("    {d}/{d} Strahlen gleich ({d} Treffer)\n", .{ same, count, hits });
    return same;
}

fn buildCheck(init: std.process.Init, ctx: ?*anyopaque) !void {
    const gpa = init.gpa;
    const log2 = 7;
    const n = 1 << log2;
    var f = common.SceneFn{ .n = n };

    // Voxelliste auf dem Host
    var vox: std.ArrayList(api.Voxel) = .empty;
    defer vox.deinit(gpa);
    for (0..n) |z| for (0..n) |y| for (0..n) |x| {
        const a = common.sceneVoxel(&f, @intCast(x), @intCast(y), @intCast(z));
        if (a != 0) try vox.append(gpa, .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z), .attribute = a });
    };
    var prng = std.Random.DefaultPrng.init(4);
    prng.random().shuffle(api.Voxel, vox.items);

    var dag_cpu: ?*anyopaque = null;
    req(pyrit.pyr_dag_build_fn(log2, common.sceneVoxel, null, &f, 0, @ptrCast(&dag_cpu)));
    defer pyrit.pyr_dag_destroy(@ptrCast(dag_cpu));
    var g_cpu: api.Handle = null;
    req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(dag_cpu), &g_cpu));

    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const t0 = std.Io.Timestamp.now(init.io, .awake);
    var g_gpu: api.Handle = null;
    req(pyrit.pyr_geometry_build(@ptrCast(ctx), log2, vox.items.ptr, @intCast(vox.items.len), api.build_host_input | api.build_editable, &g_gpu));
    const ms = @as(f64, @floatFromInt(t0.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
    print("  GPU-Bau 128^3: {d} Voxel in {d:.1} ms (inkl. Upload, GAS)\n", .{ vox.items.len, ms });

    const m = common.makeTransform(.{ 0, 1, 0 }, 0.4, 3.0, .{ 350, 330, 360 });
    var ia: api.Handle = null;
    var ib: api.Handle = null;
    req(pyrit.pyr_instance_create(@ptrCast(ctx), g_cpu, &ia));
    req(pyrit.pyr_instance_create(@ptrCast(ctx), g_gpu, &ib));
    for ([_]api.Handle{ ia, ib }) |i| req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), i, &m));
    req(pyrit.pyr_instance_set_mask(@ptrCast(ctx), ia, 0x40));
    req(pyrit.pyr_instance_set_mask(@ptrCast(ctx), ib, 0x80));
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    check(try sameHits(gpa, ctx, 0x40, 0x80) == 100_000, "GPU-Bau weicht vom CPU-Bau ab", .{});

    // Änderung: Turm hinzufügen, Kugelhälfte entfernen, einige Voxel umfärben
    var edits: std.ArrayList(api.Voxel) = .empty;
    defer edits.deinit(gpa);
    for (0..100) |y| for (0..6) |z| for (0..6) |x| try edits.append(gpa, .{ .x = @intCast(100 + x), .y = @intCast(y), .z = @intCast(20 + z), .attribute = 0x7777_7700 });
    for (vox.items) |v| {
        if (v.y > 3 and v.x > 64) try edits.append(gpa, .{ .x = v.x, .y = v.y, .z = v.z, .attribute = 0 });
        if (v.y == 1) try edits.append(gpa, .{ .x = v.x, .y = v.y, .z = v.z, .attribute = 0x3300_0001 });
    }
    const t1 = std.Io.Timestamp.now(init.io, .awake);
    req(pyrit.pyr_geometry_edit(@ptrCast(ctx), g_gpu, edits.items.ptr, @intCast(edits.items.len), api.build_host_input));
    const ems = @as(f64, @floatFromInt(t1.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
    print("  Änderung: {d} Einträge in {d:.1} ms\n", .{ edits.items.len, ems });

    // Referenz: Änderungen in derselben Reihenfolge anwenden (der letzte Eintrag gewinnt)
    var map: std.AutoArrayHashMapUnmanaged([3]i32, u32) = .empty;
    defer map.deinit(gpa);
    for (vox.items) |v| try map.put(gpa, .{ v.x, v.y, v.z }, v.attribute);
    for (edits.items) |v| {
        if (v.attribute == 0) {
            _ = map.swapRemove(.{ v.x, v.y, v.z });
        } else try map.put(gpa, .{ v.x, v.y, v.z }, v.attribute);
    }
    var final: std.ArrayList(api.Voxel) = .empty;
    defer final.deinit(gpa);
    var it = map.iterator();
    while (it.next()) |kv| try final.append(gpa, .{ .x = kv.key_ptr[0], .y = kv.key_ptr[1], .z = kv.key_ptr[2], .attribute = kv.value_ptr.* });
    var dag_ref: ?*anyopaque = null;
    req(pyrit.pyr_dag_build_points(log2, final.items.ptr, final.items.len, 0, @ptrCast(&dag_ref)));
    defer pyrit.pyr_dag_destroy(@ptrCast(dag_ref));
    var g_ref: api.Handle = null;
    req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(dag_ref), &g_ref));
    var ic: api.Handle = null;
    req(pyrit.pyr_instance_create(@ptrCast(ctx), g_ref, &ic));
    req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), ic, &m));
    req(pyrit.pyr_instance_set_mask(@ptrCast(ctx), ic, 0x20));
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    check(try sameHits(gpa, ctx, 0x20, 0x80) == 100_000, "Änderung auf der GPU weicht vom Neubau ab", .{});

    // LOD auf der GPU gegen denselben Code auf der CPU
    var lod: api.Handle = null;
    req(pyrit.pyr_geometry_downsample(@ptrCast(ctx), g_gpu, 2, 0, &lod));
    {
        var cpu = pyrit.gpu_build.CpuExec{ .gpa = gpa };
        defer cpu.deinit();
        const e = cpu.exec();
        var full_list: std.ArrayList([4]u32) = .empty;
        defer full_list.deinit(gpa);
        for (final.items) |v| try full_list.append(gpa, .{ @bitCast(v.x), @bitCast(v.y), @bitCast(v.z), v.attribute });
        var full = try pyrit.gpu_build.build(e, log2, 7, 0, 0, 0, @intFromPtr(full_list.items.ptr), @intCast(full_list.items.len));
        defer full.free(e);
        var ref_lod = try pyrit.gpu_build.downsample(e, log2, 7, full.keys, full.attrs, full.voxel_count, 2);
        defer ref_lod.free(e);
        var view_dag = pyrit.dag_builder.Dag{
            .log2_size = ref_lod.log2_size,
            .root = ref_lod.root,
            .voxel_count = ref_lod.voxel_count,
            .nodes = @as([*]u32, @ptrFromInt(ref_lod.nodes))[0..ref_lod.node_words],
            .leaves = @as([*]u64, @ptrFromInt(ref_lod.leaves))[0..ref_lod.leaf_count],
            .attributes = @as([*]u32, @ptrFromInt(ref_lod.attrs))[0..ref_lod.voxel_count],
        };
        var g_lod_ref: api.Handle = null;
        req(pyrit.pyr_geometry_create(@ptrCast(ctx), @ptrCast(&view_dag), &g_lod_ref));
        const m4 = common.makeTransform(.{ 0, 1, 0 }, 0.4, 12.0, .{ 350, 330, 360 });
        var la: api.Handle = null;
        var lb: api.Handle = null;
        req(pyrit.pyr_instance_create(@ptrCast(ctx), lod, &la));
        req(pyrit.pyr_instance_create(@ptrCast(ctx), g_lod_ref, &lb));
        for ([_]api.Handle{ la, lb }) |i| req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), i, &m4));
        req(pyrit.pyr_instance_set_mask(@ptrCast(ctx), la, 0x10));
        req(pyrit.pyr_instance_set_mask(@ptrCast(ctx), lb, 0x08));
        req(pyrit.pyr_commit(@ptrCast(ctx), null));
        print("  LOD (2^5 aus 2^7): GPU gegen CPU\n", .{});
        check(try sameHits(gpa, ctx, 0x10, 0x08) == 100_000, "LOD auf der GPU weicht ab", .{});
        for ([_]api.Handle{ la, lb }) |i| req(pyrit.pyr_instance_destroy(@ptrCast(ctx), i));
        req(pyrit.pyr_commit(@ptrCast(ctx), null));
        req(pyrit.pyr_geometry_destroy(@ptrCast(ctx), g_lod_ref));
    }
    req(pyrit.pyr_geometry_destroy(@ptrCast(ctx), lod));

    for ([_]api.Handle{ ia, ib, ic }) |i| req(pyrit.pyr_instance_destroy(@ptrCast(ctx), i));
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    for ([_]api.Handle{ g_cpu, g_gpu, g_ref }) |g| req(pyrit.pyr_geometry_destroy(@ptrCast(ctx), g));
    req(pyrit.pyr_commit(@ptrCast(ctx), null));
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
}

fn extendedCheck(gpa: std.mem.Allocator, ctx: ?*anyopaque, info: api.DagInfo) !void {
    const count = 20_000;
    const rays = try gpa.alloc(types.Ray, count);
    defer gpa.free(rays);
    var prng = std.Random.DefaultPrng.init(12);
    const rnd = prng.random();
    for (rays) |*r| {
        r.* = .{ .origin = undefined, .tmin = 0, .direction = undefined, .tmax = types.flt_max };
        for (0..3) |a| {
            r.origin[a] = (rnd.float(f32) * 2 - 1) * 250;
            r.direction[a] = (rnd.float(f32) * 2 - 1) * 60 - r.origin[a];
        }
    }
    const d_rays = devAlloc(count * @sizeOf(types.Ray));
    const d_hits = devAlloc(count * @sizeOf(types.HitEx));
    defer for ([_]u64{ d_rays, d_hits }) |b| {
        _ = drv.cuMemFree_v2(b);
    };
    uploadOn(ctx, d_rays, rays.ptr, count * @sizeOf(types.Ray));
    req(pyrit.pyr_trace(@ptrCast(ctx), d_rays, d_hits, count, 0, types.trace_extended));
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const gpu = try download(types.HitEx, gpa, d_hits, count);
    defer gpa.free(gpu);
    var mm = try Mirror.load(gpa, ctx, info);
    defer mm.arena.deinit();
    var same: usize = 0;
    var dbg: usize = 0;
    for (rays, gpu) |r, g| {
        const h = pyr.traceScene(&mm.scene, r.origin, r.direction, r.tmin, r.tmax, 0xFFFF_FFFF, 0);
        const c = pyr.scene.extendedHit(&mm.scene, r.origin, r.direction, h);
        const ok = c.hit.instance == g.hit.instance and std.mem.eql(i32, &c.voxel, &g.voxel) and
            @abs(c.normal[0] - g.normal[0]) + @abs(c.normal[1] - g.normal[1]) + @abs(c.normal[2] - g.normal[2]) < 1e-3;
        const good = ok or (c.hit.meta & g.hit.meta & types.hit_inside != 0);
        same += @intFromBool(good);
        if (!good and same + 3 > 0 and dbg < 3) {
            dbg += 1;
            print("    CPU inst {d} voxel {any} n {any} | GPU inst {d} voxel {any} n {any} t {d}/{d}\n", .{ c.hit.instance, c.voxel, c.normal, g.hit.instance, g.voxel, g.normal, c.hit.t, g.hit.t });
        }
    }
    print("  {d}/{d} gleich (Voxel, Normale)\n", .{ same, count });
    check(same * 1000 >= count * 999, "erweiterte Treffer weichen ab", .{});
}

/// Welt: Gelände auf der GPU, LOD-Streaming; senkrechte Strahlen nahe der
/// Kamera müssen das Höhenfeld auf einen halben Voxel genau treffen.
fn worldCheck(init: std.process.Init, flags: u32, label: []const u8) !void {
    const gpa = init.gpa;
    print("\n=== Welt ({s}) ===\n", .{label});
    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = api.create_debug | flags;
    ci.log = logFn;
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    defer pyrit.pyr_destroy(@ptrCast(ctx));

    // Generator der Demo, ohne Wasser und Bäume: dann ist die Oberfläche
    // genau das Höhenfeld
    var gen = try demo.Generator.init(&drv, .{ .water = 0, .tree_density = 0 });
    defer gen.deinit();
    var wi = gen.worldInfo();
    wi.view_distance = 3000;
    wi.voxel_pixels = 8;
    wi.keep_frames = 2;
    wi.flags = api.world_sync;
    var world: ?*anyopaque = null;
    req(pyrit.pyr_world_create(@ptrCast(ctx), &wi, @ptrCast(&world)));
    defer _ = pyrit.pyr_world_destroy(@ptrCast(ctx), @ptrCast(world));

    for ([_][2]f64{ .{ 5000, 5000 }, .{ -123_456, 98_765 } }) |xz| {
        const ground = gen.height(xz[0], xz[1]);
        const cam = [3]f64{ xz[0], ground + 20, xz[1] };
        const origin = [3]f64{ @floor(xz[0] / 1024) * 1024, 0, @floor(xz[1] / 1024) * 1024 };
        var st: api.WorldStats = undefined;
        var updates: u32 = 0;
        const t0 = std.Io.Timestamp.now(init.io, .awake);
        while (updates < 200) : (updates += 1) {
            req(pyrit.pyr_world_update(@ptrCast(ctx), @ptrCast(world), &cam, &origin, null));
            req(pyrit.pyr_world_stats(@ptrCast(world), &st));
            if (st.pending_chunks == 0) break;
        }
        req(pyrit.pyr_commit(@ptrCast(ctx), &.{ .time = 0, .origin = origin }));
        req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const ms = @as(f64, @floatFromInt(t0.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
        print("  bei ({d:.0}, {d:.0}): {d} Updates, {d:.1} ms, {d} Chunks resident, {d} sichtbar, {d:.2} MiB\n", .{ xz[0], xz[1], updates + 1, ms, st.resident_chunks, st.visible_chunks, @as(f64, @floatFromInt(st.bytes)) / (1 << 20) });
        check(st.pending_chunks == 0 and st.visible_chunks > 0, "Welt nicht fertig", .{});

        // senkrechte Strahlen im Umkreis von 24 Voxeln: dort sicher feinste
        // Stufe (mit 8 px je Voxel beginnt Stufe 2 schon gut 50 Blöcke weit)
        const count = 4096;
        const rays = try gpa.alloc(types.Ray, count);
        defer gpa.free(rays);
        const cols = try gpa.alloc([2]f64, count);
        defer gpa.free(cols);
        var prng = std.Random.DefaultPrng.init(3);
        const rnd = prng.random();
        for (rays, cols) |*r, *c| {
            // Spaltenmitte, damit das Höhenfeld dort exakt ausgewertet werden kann
            c.* = .{ @floor(cam[0] + (rnd.float(f64) * 2 - 1) * 24) + 0.5, @floor(cam[2] + (rnd.float(f64) * 2 - 1) * 24) + 0.5 };
            r.* = .{ .origin = .{ @floatCast(c[0] - origin[0]), 2000, @floatCast(c[1] - origin[2]) }, .tmin = 0, .direction = .{ 0, -1, 0 }, .tmax = types.flt_max };
        }
        const d_rays = devAlloc(count * @sizeOf(types.Ray));
        const d_hits = devAlloc(count * @sizeOf(types.Hit));
        defer for ([_]u64{ d_rays, d_hits }) |b| {
            _ = drv.cuMemFree_v2(b);
        };
        uploadOn(ctx, d_rays, rays.ptr, count * @sizeOf(types.Ray));
        req(pyrit.pyr_trace(@ptrCast(ctx), d_rays, d_hits, count, 0xFFFF_FFFF, 0));
        req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const hits = try download(types.Hit, gpa, d_hits, count);
        defer gpa.free(hits);
        var miss: usize = 0;
        var worst: f64 = 0;
        for (hits, cols) |h, c| {
            if (h.instance == types.no_hit or h.attribute == 0) {
                miss += 1;
                continue;
            }
            const y = 2000 - @as(f64, h.t);
            // Über flachem Ufer trifft der Strahl die Wasseroberfläche, nicht den Grund
            const land = gen.heightFinest(c[0], c[1]);
            const ref = if (gen.params.water != 0) @max(land, gen.params.sea_level) else land;
            worst = @max(worst, @abs(y - ref));
        }
        print("  {d} senkrechte Strahlen: {d} ohne Treffer, größte Höhenabweichung {d:.3}\n", .{ count, miss, worst });
        check(miss == 0, "Löcher im Gelände", .{});
        check(worst < 0.75, "Gelände weicht vom Höhenfeld ab", .{});
    }
    // nach dem Sprung werden die alten Chunks verdrängt
    var st: api.WorldStats = undefined;
    req(pyrit.pyr_world_stats(@ptrCast(world), &st));
    check(st.resident_chunks < st.visible_chunks * 2 + 64, "alte Chunks nicht verdrängt ({d} resident)", .{st.resident_chunks});
}

fn quatZ(angle: f32) [4]f32 {
    return .{ 0, 0, @sin(angle / 2), @cos(angle / 2) };
}

/// Skelettanimation: GPU-Kernel gegen dieselbe Rechnung auf der CPU, Zeit je Commit
fn animCheck(init: std.process.Init, flags: u32, label: []const u8) !void {
    const gpa = init.gpa;
    print("\n=== Skelettanimation ({s}) ===\n", .{label});
    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = flags;
    ci.max_instances = 65536;
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    defer pyrit.pyr_destroy(@ptrCast(ctx));

    // Teil: voller Würfel 8^3
    var vox: [512]api.Voxel = undefined;
    for (&vox, 0..) |*v, i| v.* = .{ .x = @intCast(i & 7), .y = @intCast((i >> 3) & 7), .z = @intCast(i >> 6), .attribute = 0xC0C0C000 };
    var geo: api.Handle = null;
    req(pyrit.pyr_geometry_build(@ptrCast(ctx), 3, &vox, vox.len, api.build_host_input, &geo));

    const id = pyr.anim.identity;
    const shift = [12]f32{ 1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0 };
    const bones = [_]pyrit.anim_mod.Bone{
        .{ .parent = -1, .reserved = 0, .geometry = geo, .rest = id, .part = id, .sway = 0, .reserved2 = 0 },
        .{ .parent = 0, .reserved = 0, .geometry = geo, .rest = shift, .part = .{ 1, 0, 0, -4, 0, 1, 0, 0, 0, 0, 1, 0 }, .sway = 0, .reserved2 = 0 },
        .{ .parent = 1, .reserved = 0, .geometry = geo, .rest = shift, .part = id, .sway = 0, .reserved2 = 0 },
    };
    var skel: api.Handle = null;
    req(pyrit.pyr_skeleton_create(@ptrCast(ctx), &bones, bones.len, &skel));
    const keys = [_]types.Keyframe{
        .{ .bone = 0, .time = 0, .translation = .{ 0, 0, 0 }, .scale = 1, .rotation = quatZ(0) },
        .{ .bone = 0, .time = 1, .translation = .{ 0, 0, 0 }, .scale = 1, .rotation = quatZ(1.2) },
        .{ .bone = 0, .time = 2, .translation = .{ 0, 0, 0 }, .scale = 1, .rotation = quatZ(0) },
        .{ .bone = 1, .time = 0, .translation = .{ 10, 0, 0 }, .scale = 1, .rotation = quatZ(0.3) },
        .{ .bone = 1, .time = 1.5, .translation = .{ 10, 0, 0 }, .scale = 1, .rotation = quatZ(-0.9) },
    };
    var clip: api.Handle = null;
    req(pyrit.pyr_clip_create(@ptrCast(ctx), skel, &keys, keys.len, 2, types.clip_loop, &clip));

    const n_actors = 2000;
    const infos = try gpa.alloc(pyrit.anim_mod.ActorInfo, n_actors);
    defer gpa.free(infos);
    for (infos, 0..) |*a, i| {
        a.* = std.mem.zeroes(pyrit.anim_mod.ActorInfo);
        const fi: f32 = @floatFromInt(i);
        a.root = .{ 1, 0, 0, @mod(fi, 50) * 40, 0, 1, 0, 0, 0, 0, 1, @floor(fi / 50) * 40 };
        a.clip = clip;
        a.start_time = fi * 0.013;
        a.speed = 1 + @mod(fi, 7) * 0.1;
        a.user = @intCast(i);
        var actor: api.Handle = null;
        req(pyrit.pyr_actor_create(@ptrCast(ctx), skel, a, &actor));
    }

    // CPU-Spiegel derselben Daten (Reihenfolge wie in anim.zig)
    const abones = [_]types.AnimBone{
        .{ .parent = -1, .part_size = 8, .sway = 0, .reserved = 0, .rest = id, .part = id },
        .{ .parent = 0, .part_size = 8, .sway = 0, .reserved = 0, .rest = shift, .part = bones[1].part },
        .{ .parent = 1, .part_size = 8, .sway = 0, .reserved = 0, .rest = shift, .part = id },
    };
    const tracks = [_][2]u32{ .{ 0, 3 }, .{ 3, 2 }, .{ 5, 0 } };
    const clips = [_]types.AnimClip{.{ .track_offset = 0, .bone_count = 3, .duration = 2, .flags = types.clip_loop }};
    var cpu_actor: [1]types.AnimActor = undefined;
    const job = [_]types.AnimJob{.{ .actor = 0, .bone = 0, .instance = 0, .reserved = 0 }};
    var cpu_inst: [1]types.InstanceData = undefined;

    const scene_dev = pyrit.pyr_scene_device(@ptrCast(ctx));
    var worst: f32 = 0;
    var total_ms: f64 = 0;
    const frames = 60;
    for (0..frames) |f| {
        const time = 10 + @as(f64, @floatFromInt(f)) * 0.37;
        const t0 = std.Io.Timestamp.now(init.io, .awake);
        req(pyrit.pyr_commit(@ptrCast(ctx), &.{ .time = time, .origin = .{ 0, 0, 0 } }));
        req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        total_ms += @as(f64, @floatFromInt(t0.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
        if (f % 20 != 5) continue;
        var scene: types.Scene = undefined;
        cu(drv.cuMemcpyDtoH_v2(&scene, scene_dev, @sizeOf(types.Scene)));
        const gi = try download(types.InstanceData, gpa, scene.instances, 3 * n_actors);
        defer gpa.free(gi);
        for (0..n_actors) |a| {
            cpu_actor[0] = .{ .root = infos[a].root, .bone_offset = 0, .bone_count = 3, .clip = 0, .blend_clip = types.no_clip, .start_time = infos[a].start_time, .speed = infos[a].speed, .blend_start_time = 0, .blend_speed = 0, .blend = 0, .wind_amplitude = 0, .wind_frequency = 0, .wind_phase = 0 };
            for (0..3) |b| {
                var j = job;
                j[0].bone = @intCast(b);
                const p = types.AnimParams{ .bones = @intFromPtr(&abones), .tracks = @intFromPtr(&tracks), .keys = @intFromPtr(&keys), .clips = @intFromPtr(&clips), .actors = @intFromPtr(&cpu_actor), .jobs = @intFromPtr(&j), .instances = @intFromPtr(&cpu_inst), .job_count = 1, .reserved = 0, .time = time };
                pyr.anim.run(&p, 0);
                const g = gi[3 * a + b];
                check(g.user == a, "Instanzreihenfolge", .{});
                for (0..12) |e| worst = @max(worst, @abs(g.object_to_world[e] - cpu_inst[0].object_to_world[e]));
            }
        }
    }
    print("  {d} Akteure x 3 Teile: größte Abweichung GPU/CPU {e:.2}, Commit + Animation {d:.3} ms/Frame\n", .{ n_actors, worst, total_ms / frames });
    check(worst < 5e-3, "Animation weicht ab", .{});
}
