//! C-API der Pyrit-Runtime (include/pyrit.h beschreibt sie für C-Aufrufer;
//! tests/abi_test.zig prüft, dass beide Seiten übereinstimmen).

const std = @import("std");
pub const api = @import("api.zig");
pub const types = @import("pyrit_device").types;
pub const diag = @import("diag.zig");
pub const dag_builder = @import("dag_builder.zig");
pub const context = @import("context.zig");
pub const cuda = @import("cuda.zig");
pub const xform = @import("xform.zig");
pub const rt_prims = @import("rt_prims.zig");
pub const gpu_build = @import("gpu_build.zig");
pub const io = @import("io.zig");
pub const optix = @import("optix.zig");
pub const optix_device = @import("pyrit_device").optix;
pub const world_mod = @import("world.zig");
pub const world_plan = @import("world_plan.zig");
const pyr_device = @import("pyrit_device");
const Context = context.Context;
const World = world_mod.World;

const gpa = std.heap.c_allocator;

comptime {
    std.debug.assert(@sizeOf(types.Hit) == 16);
    std.debug.assert(@sizeOf(types.Ray) == 32);
    std.debug.assert(@sizeOf(api.Voxel) == @sizeOf(dag_builder.Point));
    std.debug.assert(@sizeOf(types.InstanceData) % 16 == 0);
}

// Kurzformen für die Signaturen
const Result = api.Result;
const PyrContext = opaque {};
const PyrDag = opaque {};
const Handle = api.Handle;

fn code(e: diag.Error) Result {
    return switch (e) {
        error.InvalidArgument => api.error_invalid_argument,
        error.InvalidHandle => api.error_invalid_handle,
        error.OutOfMemory => api.error_out_of_memory,
        error.Capacity => api.error_capacity,
        error.InUse => api.error_in_use,
        error.Cuda => api.error_cuda,
        error.Compile => api.error_compile,
        error.NotFound => api.error_not_found,
        error.Version => api.error_version,
    };
}

fn result(r: diag.Error!void) Result {
    r catch |e| return code(e);
    return api.ok;
}

fn ctxOf(ctx: ?*PyrContext) ?*Context {
    return @ptrCast(@alignCast(ctx));
}

/// Führt `f` mit aktuellem CUDA-Kontext aus.
fn call(ctx: ?*PyrContext, comptime f: anytype, args: anytype) Result {
    diag.clear();
    const self = ctxOf(ctx) orelse return code(diag.fail(error.InvalidArgument, "Kontext ist NULL", .{}));
    self.enter() catch |e| return code(e);
    defer self.leave();
    return result(@call(.auto, f, .{self} ++ args));
}

fn callHandle(ctx: ?*PyrContext, out: anytype, comptime f: anytype, args: anytype) Result {
    diag.clear();
    const self = ctxOf(ctx) orelse return code(diag.fail(error.InvalidArgument, "Kontext ist NULL", .{}));
    const dst = out orelse return code(diag.fail(error.InvalidArgument, "Ausgabezeiger ist NULL", .{}));
    self.enter() catch |e| return code(e);
    defer self.leave();
    const h = @call(.auto, f, .{self} ++ args) catch |e| return code(e);
    dst.* = @ptrFromInt(h);
    return api.ok;
}

fn handle(h: anytype) usize {
    return @intFromPtr(h);
}

// ---------------------------------------------------------------------------
// Allgemein
// ---------------------------------------------------------------------------

pub export fn pyr_result_string(r: Result) [*:0]const u8 {
    return switch (r) {
        api.ok => "PYR_OK",
        api.error_invalid_argument => "PYR_ERROR_INVALID_ARGUMENT",
        api.error_invalid_handle => "PYR_ERROR_INVALID_HANDLE",
        api.error_out_of_memory => "PYR_ERROR_OUT_OF_MEMORY",
        api.error_capacity => "PYR_ERROR_CAPACITY",
        api.error_in_use => "PYR_ERROR_IN_USE",
        api.error_cuda => "PYR_ERROR_CUDA",
        api.error_compile => "PYR_ERROR_COMPILE",
        api.error_not_found => "PYR_ERROR_NOT_FOUND",
        api.error_version => "PYR_ERROR_VERSION",
        else => "unbekannter PyrResult",
    };
}

pub export fn pyr_error_message() [*:0]const u8 {
    return diag.message();
}

// ---------------------------------------------------------------------------
// Kontext
// ---------------------------------------------------------------------------

pub export fn pyr_create(info: ?*const api.CreateInfo, out: ?*?*PyrContext) Result {
    diag.clear();
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    const dst = out orelse return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    const ctx = Context.create(gpa, i) catch |e| return code(e);
    dst.* = @ptrCast(ctx);
    return api.ok;
}

pub export fn pyr_destroy(ctx: ?*PyrContext) void {
    if (ctxOf(ctx)) |self| self.destroy();
}

pub export fn pyr_cuda_stream(ctx: ?*PyrContext) ?*anyopaque {
    const self = ctxOf(ctx) orelse return null;
    return @ptrCast(self.stream);
}

pub export fn pyr_features(ctx: ?*PyrContext) u32 {
    const self = ctxOf(ctx) orelse return 0;
    return self.features();
}

pub export fn pyr_synchronize(ctx: ?*PyrContext) Result {
    return call(ctx, Context.synchronize, .{});
}

// ---------------------------------------------------------------------------
// DAG-Bau
// ---------------------------------------------------------------------------

fn dagOut(out: ?*?*PyrDag, built: dag_builder.Error!dag_builder.Dag) Result {
    var dag = built catch |e| return switch (e) {
        error.InvalidArgument => code(diag.fail(error.InvalidArgument, "ungültige DAG-Eingabe (log2_size 3..20, Koordinaten im Würfel, Attribut != 0)", .{})),
        error.OutOfMemory => code(diag.fail(error.OutOfMemory, "Host-Speicher beim DAG-Bau", .{})),
    };
    const p = gpa.create(dag_builder.Dag) catch {
        dag.deinit(gpa);
        return code(diag.fail(error.OutOfMemory, "Host-Speicher", .{}));
    };
    p.* = dag;
    out.?.* = @ptrCast(p);
    return api.ok;
}

fn dagOf(dag: ?*const PyrDag) ?*const dag_builder.Dag {
    return @ptrCast(@alignCast(dag));
}

pub export fn pyr_dag_build_dense(log2_size: u32, voxels: ?[*]const u32, flags: u32, out: ?*?*PyrDag) Result {
    diag.clear();
    if (out == null or voxels == null or log2_size < dag_builder.min_log2 or log2_size > 10)
        return code(diag.fail(error.InvalidArgument, "pyr_dag_build_dense: log2_size 3..10, voxels und out dürfen nicht NULL sein", .{}));
    const n: usize = @as(usize, 1) << @intCast(log2_size);
    return dagOut(out, dag_builder.buildDense(gpa, log2_size, voxels.?[0 .. n * n * n], flags & api.dag_no_attributes == 0));
}

pub export fn pyr_dag_build_points(log2_size: u32, voxels: ?[*]const api.Voxel, count: usize, flags: u32, out: ?*?*PyrDag) Result {
    diag.clear();
    if (out == null or (voxels == null and count != 0))
        return code(diag.fail(error.InvalidArgument, "pyr_dag_build_points: voxels und out dürfen nicht NULL sein", .{}));
    const pts: []const dag_builder.Point = if (count == 0) &.{} else @as([*]const dag_builder.Point, @ptrCast(voxels.?))[0..count];
    return dagOut(out, dag_builder.buildPoints(gpa, log2_size, pts, flags & api.dag_no_attributes == 0));
}

pub export fn pyr_dag_build_fn(log2_size: u32, voxel: api.VoxelFn, region_empty: api.RegionEmptyFn, user: ?*anyopaque, flags: u32, out: ?*?*PyrDag) Result {
    diag.clear();
    const f = voxel orelse return code(diag.fail(error.InvalidArgument, "pyr_dag_build_fn: voxel ist NULL", .{}));
    if (out == null) return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    return dagOut(out, dag_builder.buildFn(gpa, log2_size, @ptrCast(f), @ptrCast(region_empty), user, flags & api.dag_no_attributes == 0));
}

pub export fn pyr_dag_get_info(dag: ?*const PyrDag, out: ?*api.DagInfo) void {
    const d = dagOf(dag) orelse return;
    const o = out orelse return;
    o.* = .{
        .log2_size = d.log2_size,
        .root = d.root,
        .voxel_count = d.voxel_count,
        .nodes = d.nodes.ptr,
        .node_words = d.nodes.len,
        .leaves = d.leaves.ptr,
        .leaf_count = d.leaves.len,
        .attributes = if (d.attributes) |a| a.ptr else null,
        .attribute_count = if (d.attributes) |a| a.len else 0,
    };
}

/// Liest Modell `model` einer MagicaVoxel-Datei. Mit out == NULL werden nur
/// count und log2_size geliefert (zweistufig: erst fragen, dann Speicher geben).
pub export fn pyr_vox_parse(data: ?*const anyopaque, size: usize, model: u32, out: ?[*]api.Voxel, count: ?*u32, log2_size: ?*u32) Result {
    diag.clear();
    const d = data orelse return code(diag.fail(error.InvalidArgument, "data ist NULL", .{}));
    const cnt = count orelse return code(diag.fail(error.InvalidArgument, "count ist NULL", .{}));
    const m = io.parseVox(gpa, @as([*]const u8, @ptrCast(d))[0..size], model) catch |e| return switch (e) {
        error.OutOfMemory => code(diag.fail(error.OutOfMemory, "Host-Speicher", .{})),
        error.InvalidData => code(diag.fail(error.InvalidArgument, "keine gültige .vox-Datei oder Modell {d} fehlt", .{model})),
    };
    defer gpa.free(m.voxels);
    if (log2_size) |l| l.* = m.log2_size;
    if (out) |o| {
        if (cnt.* < m.voxels.len) return code(diag.fail(error.InvalidArgument, "Ausgabe zu klein: {d} Voxel nötig", .{m.voxels.len}));
        @memcpy(o[0..m.voxels.len], m.voxels);
    }
    cnt.* = @intCast(m.voxels.len);
    return api.ok;
}

/// Serialisiert eine DAG. buffer == NULL: nur *size setzen.
pub export fn pyr_dag_save(dag: ?*const PyrDag, buffer: ?*anyopaque, size: ?*usize) Result {
    diag.clear();
    const d = dagOf(dag) orelse return code(diag.fail(error.InvalidArgument, "dag ist NULL", .{}));
    const sz = size orelse return code(diag.fail(error.InvalidArgument, "size ist NULL", .{}));
    const need = io.savedSize(d);
    if (buffer) |b| {
        if (sz.* < need) return code(diag.fail(error.InvalidArgument, "Puffer zu klein: {d} Bytes nötig", .{need}));
        io.save(d, @as([*]u8, @ptrCast(b))[0..need]);
    }
    sz.* = need;
    return api.ok;
}

pub export fn pyr_dag_load(data: ?*const anyopaque, size: usize, out: ?*?*PyrDag) Result {
    diag.clear();
    const d = data orelse return code(diag.fail(error.InvalidArgument, "data ist NULL", .{}));
    if (out == null) return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    const loaded = io.load(gpa, @as([*]const u8, @ptrCast(d))[0..size]) catch |e| return switch (e) {
        error.OutOfMemory => code(diag.fail(error.OutOfMemory, "Host-Speicher", .{})),
        error.InvalidData => code(diag.fail(error.InvalidArgument, "keine gültigen DAG-Daten", .{})),
    };
    return dagOut(out, loaded);
}

pub export fn pyr_geometry_download(ctx: ?*PyrContext, g: Handle, out: ?*?*PyrDag) Result {
    diag.clear();
    const self = ctxOf(ctx) orelse return code(diag.fail(error.InvalidArgument, "Kontext ist NULL", .{}));
    if (out == null) return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    self.enter() catch |e| return code(e);
    defer self.leave();
    const dag = self.geometryDownload(handle(g)) catch |e| return code(e);
    return dagOut(out, dag);
}

pub export fn pyr_dag_destroy(dag: ?*PyrDag) void {
    const d: *dag_builder.Dag = @ptrCast(@alignCast(dag orelse return));
    d.deinit(gpa);
    gpa.destroy(d);
}

// ---------------------------------------------------------------------------
// Geometrie und Instanzen
// ---------------------------------------------------------------------------

pub export fn pyr_geometry_create(ctx: ?*PyrContext, dag: ?*const PyrDag, out: ?*Handle) Result {
    const d = dagOf(dag) orelse return code(diag.fail(error.InvalidArgument, "dag ist NULL", .{}));
    return callHandle(ctx, out, Context.geometryCreate, .{d});
}

pub export fn pyr_geometry_build(ctx: ?*PyrContext, log2_size: u32, voxels: ?*const anyopaque, count: u32, flags: u32, out: ?*Handle) Result {
    return callHandle(ctx, out, Context.geometryBuild, .{ log2_size, voxels, count, flags });
}

pub export fn pyr_geometry_edit(ctx: ?*PyrContext, g: Handle, voxels: ?*const anyopaque, count: u32, flags: u32) Result {
    return call(ctx, Context.geometryEdit, .{ handle(g), voxels, count, flags });
}

pub export fn pyr_geometry_downsample(ctx: ?*PyrContext, g: Handle, shift: u32, flags: u32, out: ?*Handle) Result {
    return callHandle(ctx, out, Context.geometryDownsample, .{ handle(g), shift, flags });
}

pub export fn pyr_geometry_destroy(ctx: ?*PyrContext, g: Handle) Result {
    return call(ctx, Context.geometryDestroy, .{handle(g)});
}

pub export fn pyr_geometry_index(g: Handle) u32 {
    return context.decodeHandle(handle(g)).index;
}

pub export fn pyr_instance_create(ctx: ?*PyrContext, g: Handle, out: ?*Handle) Result {
    return callHandle(ctx, out, Context.instanceCreate, .{handle(g)});
}

pub export fn pyr_instance_destroy(ctx: ?*PyrContext, i: Handle) Result {
    return call(ctx, Context.instanceDestroy, .{handle(i)});
}

pub export fn pyr_instance_set_transform(ctx: ?*PyrContext, i: Handle, m: ?*const [12]f32) Result {
    const mm = m orelse return code(diag.fail(error.InvalidArgument, "Matrix ist NULL", .{}));
    return call(ctx, Context.instanceSetTransform, .{ handle(i), mm });
}

pub export fn pyr_instance_set_geometry(ctx: ?*PyrContext, i: Handle, g: Handle) Result {
    return call(ctx, Context.instanceSetGeometry, .{ handle(i), handle(g) });
}

pub export fn pyr_instance_set_mask(ctx: ?*PyrContext, i: Handle, mask: u32) Result {
    return call(ctx, Context.instanceSetMask, .{ handle(i), mask });
}

pub export fn pyr_instance_set_user(ctx: ?*PyrContext, i: Handle, user: u32) Result {
    return call(ctx, Context.instanceSetUser, .{ handle(i), user });
}

pub export fn pyr_instance_reset_history(ctx: ?*PyrContext, i: Handle) Result {
    return call(ctx, Context.instanceResetHistory, .{handle(i)});
}

pub export fn pyr_instance_index(i: Handle) u32 {
    return context.decodeHandle(handle(i)).index;
}

// ---------------------------------------------------------------------------
// Frames, Ansichten, Strahlen
// ---------------------------------------------------------------------------

pub export fn pyr_commit(ctx: ?*PyrContext, info: ?*const api.FrameInfo) Result {
    return call(ctx, Context.commit, .{info});
}

pub export fn pyr_scene_device(ctx: ?*PyrContext) u64 {
    const self = ctxOf(ctx) orelse return 0;
    return self.scene_dev;
}

pub export fn pyr_view_create(ctx: ?*PyrContext, out: ?*Handle) Result {
    return callHandle(ctx, out, Context.viewCreate, .{});
}

pub export fn pyr_view_destroy(ctx: ?*PyrContext, v: Handle) Result {
    return call(ctx, Context.viewDestroy, .{handle(v)});
}

pub export fn pyr_view_reset_history(ctx: ?*PyrContext, v: Handle) Result {
    return call(ctx, Context.viewResetHistory, .{handle(v)});
}

pub export fn pyr_render(ctx: ?*PyrContext, v: Handle, camera: ?*const types.Camera, targets: ?*const api.Targets) Result {
    const cam = camera orelse return code(diag.fail(error.InvalidArgument, "camera ist NULL", .{}));
    const tg = targets orelse return code(diag.fail(error.InvalidArgument, "targets ist NULL", .{}));
    return call(ctx, Context.render, .{ handle(v), cam, tg });
}

pub export fn pyr_trace(ctx: ?*PyrContext, rays: u64, hits: u64, count: u32, ray_mask: u32, flags: u32) Result {
    return call(ctx, Context.trace, .{ rays, hits, count, ray_mask, flags });
}

// ---------------------------------------------------------------------------
// Materialien, Licht, Nachbearbeitung, Statistik
// ---------------------------------------------------------------------------

pub export fn pyr_material_default(m: ?*types.Material) void {
    if (m) |p| p.* = Context.defaultMaterial();
}

pub export fn pyr_material_set(ctx: ?*PyrContext, index: u32, m: ?*const types.Material) Result {
    const mm = m orelse return code(diag.fail(error.InvalidArgument, "Material ist NULL", .{}));
    return call(ctx, Context.materialSet, .{ index, mm });
}

pub export fn pyr_lighting_default(l: ?*types.Lighting) void {
    if (l) |p| p.* = Context.defaultLighting();
}

pub export fn pyr_set_lighting(ctx: ?*PyrContext, l: ?*const types.Lighting) Result {
    const ll = l orelse return code(diag.fail(error.InvalidArgument, "Licht ist NULL", .{}));
    return call(ctx, Context.setLighting, .{ll});
}

pub export fn pyr_voxel_attribute(material: u32, r: u32, g: u32, b: u32) u32 {
    return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (material & 0xFF);
}

pub export fn pyr_postprocess(ctx: ?*PyrContext, v: Handle, in: ?*const api.Targets, info: ?*const api.PostInfo) Result {
    const t = in orelse return code(diag.fail(error.InvalidArgument, "Eingaben sind NULL", .{}));
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    return call(ctx, Context.postprocess, .{ handle(v), t, i });
}

pub export fn pyr_get_stats(ctx: ?*PyrContext, out: ?*api.Stats) Result {
    diag.clear();
    const self = ctxOf(ctx) orelse return code(diag.fail(error.InvalidArgument, "Kontext ist NULL", .{}));
    const o = out orelse return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    o.* = self.stats();
    return api.ok;
}

/// Subpixel-Versatz aus der Halton-Folge (2, 3), im Bereich [-0.5, 0.5)
pub export fn pyr_frame_generate(ctx: ?*PyrContext, view: Handle, info: ?*const api.FrameGenInfo) Result {
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    return call(ctx, Context.frameGenerate, .{ handle(view), i });
}

pub export fn pyr_jitter_halton(frame: u32, out: ?*[2]f32) void {
    const o = out orelse return;
    const i = (frame % 1024) + 1;
    o.* = .{ halton(i, 2) - 0.5, halton(i, 3) - 0.5 };
}

fn halton(index: u32, base: u32) f32 {
    var f: f32 = 1;
    var r: f32 = 0;
    var i = index;
    while (i > 0) {
        f /= @floatFromInt(base);
        r += f * @as(f32, @floatFromInt(i % base));
        i /= base;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Kamera-Helfer
// ---------------------------------------------------------------------------

fn normalize3(v: [3]f32) [3]f32 {
    const l = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return .{ v[0] / l, v[1] / l, v[2] / l };
}

fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

pub export fn pyr_camera_look_at(cam: ?*types.Camera, eye: ?*const [3]f32, target: ?*const [3]f32, up: ?*const [3]f32) void {
    const cm = cam orelse return;
    const e = (eye orelse return).*;
    const t = (target orelse return).*;
    const u = (up orelse return).*;
    const back = normalize3(.{ e[0] - t[0], e[1] - t[1], e[2] - t[2] }); // Kamera blickt entlang -Z
    const right = normalize3(cross(u, back));
    const true_up = cross(back, right);
    cm.view_to_world = .{
        right[0], true_up[0], back[0], e[0],
        right[1], true_up[1], back[1], e[1],
        right[2], true_up[2], back[2], e[2],
    };
}

pub export fn pyr_camera_perspective(cam: ?*types.Camera, fov_y: f32, width: u32, height: u32, near_plane: f32) void {
    const cm = cam orelse return;
    const ty = @tan(fov_y * 0.5);
    cm.projection = types.projection_perspective;
    cm.width = width;
    cm.height = height;
    cm.scale = .{ ty * @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height)), ty };
    cm.near_plane = near_plane;
}

pub export fn pyr_camera_orthographic(cam: ?*types.Camera, half_height: f32, width: u32, height: u32) void {
    const cm = cam orelse return;
    cm.projection = types.projection_orthographic;
    cm.width = width;
    cm.height = height;
    cm.scale = .{ half_height * @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height)), half_height };
}

// ---------------------------------------------------------------------------
// Skelettanimation
// ---------------------------------------------------------------------------

pub const anim_mod = @import("anim.zig");

fn skeletonCreate(self: *Context, bones: [*]const anim_mod.Bone, count: u32) diag.Error!usize {
    return self.anim.skeletonCreate(self, bones[0..count]);
}

pub export fn pyr_skeleton_create(ctx: ?*PyrContext, bones: ?[*]const anim_mod.Bone, count: u32, out: ?*Handle) Result {
    const b = bones orelse return code(diag.fail(error.InvalidArgument, "bones ist NULL", .{}));
    return callHandle(ctx, out, skeletonCreate, .{ b, count });
}

fn clipCreate(self: *Context, skeleton: usize, keys: [*]const types.Keyframe, count: u32, duration: f32, flags: u32) diag.Error!usize {
    return self.anim.clipCreate(self, skeleton, keys[0..count], duration, flags);
}

pub export fn pyr_clip_create(ctx: ?*PyrContext, skeleton: Handle, keys: ?[*]const types.Keyframe, count: u32, duration: f32, flags: u32, out: ?*Handle) Result {
    const k = keys orelse return code(diag.fail(error.InvalidArgument, "keys ist NULL", .{}));
    return callHandle(ctx, out, clipCreate, .{ handle(skeleton), k, count, duration, flags });
}

fn skeletonDestroy(self: *Context, skeleton: usize) diag.Error!void {
    return self.anim.skeletonDestroy(self, skeleton);
}

pub export fn pyr_skeleton_destroy(ctx: ?*PyrContext, skeleton: Handle) Result {
    return call(ctx, skeletonDestroy, .{handle(skeleton)});
}

fn clipDestroy(self: *Context, clip: usize) diag.Error!void {
    return self.anim.clipDestroy(self, clip);
}

pub export fn pyr_clip_destroy(ctx: ?*PyrContext, clip: Handle) Result {
    return call(ctx, clipDestroy, .{handle(clip)});
}

fn actorCreate(self: *Context, skeleton: usize, info: *const anim_mod.ActorInfo) diag.Error!usize {
    return self.anim.actorCreate(self, skeleton, info);
}

pub export fn pyr_actor_create(ctx: ?*PyrContext, skeleton: Handle, info: ?*const anim_mod.ActorInfo, out: ?*Handle) Result {
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    return callHandle(ctx, out, actorCreate, .{ handle(skeleton), i });
}

fn actorSet(self: *Context, actor: usize, info: *const anim_mod.ActorInfo) diag.Error!void {
    return self.anim.actorSet(self, actor, info);
}

pub export fn pyr_actor_set(ctx: ?*PyrContext, actor: Handle, info: ?*const anim_mod.ActorInfo) Result {
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    return call(ctx, actorSet, .{ handle(actor), i });
}

fn actorDestroy(self: *Context, actor: usize) diag.Error!void {
    return self.anim.actorDestroy(self, actor);
}

pub export fn pyr_actor_destroy(ctx: ?*PyrContext, actor: Handle) Result {
    return call(ctx, actorDestroy, .{handle(actor)});
}

// ---------------------------------------------------------------------------
// Große Welten
// ---------------------------------------------------------------------------

const PyrWorld = opaque {};

fn worldOf(w: ?*PyrWorld) ?*World {
    return @ptrCast(@alignCast(w));
}

fn worldCreate(self: *Context, info: *const api.WorldInfo) diag.Error!usize {
    return @intFromPtr(try World.create(self, info));
}

pub export fn pyr_world_create(ctx: ?*PyrContext, info: ?*const api.WorldInfo, out: ?*?*PyrWorld) Result {
    const i = info orelse return code(diag.fail(error.InvalidArgument, "info ist NULL", .{}));
    return callHandle(ctx, out, worldCreate, .{i});
}

fn worldDestroy(self: *Context, w: *World) diag.Error!void {
    if (w.ctx != self) return diag.fail(error.InvalidArgument, "Welt gehört zu einem anderen Kontext", .{});
    w.destroy();
}

pub export fn pyr_world_destroy(ctx: ?*PyrContext, world: ?*PyrWorld) Result {
    const w = worldOf(world) orelse return api.ok;
    return call(ctx, worldDestroy, .{w});
}

fn worldUpdate(self: *Context, w: *World, position: [3]f64, origin: [3]f64, camera: ?*const types.Camera) diag.Error!void {
    if (w.ctx != self) return diag.fail(error.InvalidArgument, "Welt gehört zu einem anderen Kontext", .{});
    try w.update(position, origin, camera);
}

pub export fn pyr_world_update(ctx: ?*PyrContext, world: ?*PyrWorld, position: ?*const [3]f64, origin: ?*const [3]f64, camera: ?*const types.Camera) Result {
    diag.clear();
    const w = worldOf(world) orelse return code(diag.fail(error.InvalidArgument, "world ist NULL", .{}));
    const p = position orelse return code(diag.fail(error.InvalidArgument, "position ist NULL", .{}));
    const o: [3]f64 = if (origin) |q| q.* else .{ 0, 0, 0 };
    return call(ctx, worldUpdate, .{ w, p.*, o, camera });
}

fn worldWait(self: *Context, w: *World, camera: [3]f64) diag.Error!void {
    if (w.ctx != self) return diag.fail(error.InvalidArgument, "Welt gehört zu einem anderen Kontext", .{});
    try w.wait(camera);
}

pub export fn pyr_world_wait(ctx: ?*PyrContext, world: ?*PyrWorld, camera: ?*const [3]f64) Result {
    diag.clear();
    const w = worldOf(world) orelse return code(diag.fail(error.InvalidArgument, "world ist NULL", .{}));
    const cam = camera orelse return code(diag.fail(error.InvalidArgument, "camera ist NULL", .{}));
    return call(ctx, worldWait, .{ w, cam.* });
}

pub export fn pyr_world_stats(world: ?*PyrWorld, out: ?*api.WorldStats) Result {
    const w = worldOf(world) orelse return code(diag.fail(error.InvalidArgument, "world ist NULL", .{}));
    const o = out orelse return code(diag.fail(error.InvalidArgument, "out ist NULL", .{}));
    o.* = w.stats;
    return api.ok;
}

/// Höhe des eingebauten Geländes an (x, z) in Grundvoxeln (volle Auflösung),
/// z. B. um die Kamera über den Boden zu setzen.
pub export fn pyr_terrain_height(terrain: ?*const api.TerrainInfo, x: f64, z: f64) f32 {
    const t = if (terrain) |p| p.* else world_mod.defaultTerrain();
    return pyr_device.worldgen.height(&t, @floatCast(x), @floatCast(z), 0);
}

pub export fn pyr_terrain_default(out: ?*api.TerrainInfo) void {
    if (out) |o| o.* = world_mod.defaultTerrain();
}

test {
    _ = @import("dag_builder.zig");
    _ = @import("range_alloc.zig");
    _ = @import("xform.zig");
    _ = @import("context.zig");
    _ = @import("rt_prims.zig");
    _ = @import("io.zig");
    _ = @import("world_plan.zig");
}
