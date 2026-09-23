//! Laufzeitkontext: GPU-Ressourcen, Zustandsverwaltung und Frame-Ablauf.
//!
//! Zustand (Instanzen) liegt auf der GPU doppelt vor. Beim Commit wird die
//! ältere Hälfte zur aktuellen: Einträge, die sich im Vorframe geändert haben,
//! werden aus der Vorframe-Hälfte kopiert (Copy-Forward), neue Werte kommen
//! aus dem Upload. Damit unterscheiden sich beide Hälften immer nur in den
//! Einträgen des letzten Frames.

const std = @import("std");
const types = @import("pyrit_device").types;
const api = @import("api.zig");
const cuda = @import("cuda.zig");
const dag_builder = @import("dag_builder.zig");
const xform = @import("xform.zig");
const diag = @import("diag.zig");
const RangeAlloc = @import("range_alloc.zig").RangeAlloc;
const Rt = @import("rt.zig").Rt;
const rt_prims = @import("rt_prims.zig");
const gpu_build = @import("gpu_build.zig");
const dlss = @import("dlss.zig");
const anim = @import("anim.zig");

const Error = diag.Error;

/// PTX der Zig-Kernel (src/gpu_kernels.zig), beim Bauen erzeugt
const ptx: [:0]const u8 = @embedFile("pyrit_ptx");
const fail = diag.fail;
const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------------
// Handles: Index + Generation, als undurchsichtiger Zeiger an C übergeben
// ---------------------------------------------------------------------------

pub fn encodeHandle(index: u32, generation: u32) usize {
    return (@as(usize, generation) << 32) | (@as(usize, index) + 1);
}

pub fn decodeHandle(h: usize) struct { index: u32, generation: u32 } {
    return .{ .index = @intCast((h & 0xFFFFFFFF) -% 1), .generation = @intCast(h >> 32) };
}

const GeometrySlot = struct {
    generation: u32 = 1,
    alive: bool = false,
    refs: u32 = 0,
    data: types.GeometryData = std.mem.zeroes(types.GeometryData),
    node_words: u64 = 0,
    leaf_count: u64 = 0,
    attribute_count: u64 = 0,
    /// änderbar (GPU-Bau): sortierte Voxelliste auf der GPU
    vox_keys: cuda.CUdeviceptr = 0,
    vox_attrs: cuda.CUdeviceptr = 0,
    vox_count: u32 = 0,
    /// Chunk einer Weltbatch: Pool-Bereiche gehören der Batch (siehe ChunkBatch)
    batch: ?u32 = null,
};

/// Gemeinsame Pool-Bereiche eines Chunk-Batches (Welt-Streaming). Alle Chunks
/// eines GPU-Baus teilen Knoten, Blätter, Attribute und Primitive (Teilbäume
/// sind batchweit dedupliziert); freigegeben wird, wenn der letzte Chunk geht.
const ChunkBatch = struct {
    refs: u32 = 0,
    node_off: u64 = 0,
    node_words: u64 = 0,
    leaf_off: u64 = 0,
    leaf_count: u64 = 0,
    attr_off: u64 = 0,
    attr_count: u64 = 0,
    prims: cuda.CUdeviceptr = 0,
};

const InstanceSlot = struct {
    generation: u32 = 1,
    alive: bool = false,
    geometry: u32 = 0,
    dirty_mark: u64 = 0,
    data: types.InstanceData = std.mem.zeroes(types.InstanceData),
};

const ViewSlot = struct {
    generation: u32 = 1,
    alive: bool = false,
    /// Puffer der Nachbearbeitung (Verlauf, Filter), bei Bedarf angelegt
    post_w: u32 = 0,
    post_h: u32 = 0,
    post_buf: [6]cuda.CUdeviceptr = .{ 0, 0, 0, 0, 0, 0 },
    /// Helligkeitsmomente je Parität und Varianz (Akkumulation + Filter-Pingpong)
    post_mom: [2]cuda.CUdeviceptr = .{ 0, 0 },
    /// Kamera- und Bildeffekte: zwei HDR-Puffer, Bewegung/Tiefe, Bloom-Pyramide,
    /// Belichtungszustand (bleibt über Frames stehen)
    fx_buf: [2]cuda.CUdeviceptr = .{ 0, 0 },
    fx_mvd: cuda.CUdeviceptr = 0,
    fx_bloom: [8]cuda.CUdeviceptr = .{0} ** 8,
    fx_bloom_w: [8]u32 = .{0} ** 8,
    fx_bloom_h: [8]u32 = .{0} ** 8,
    fx_levels: u32 = 0,
    fx_expose: cuda.CUdeviceptr = 0,
    fx_w: u32 = 0,
    fx_h: u32 = 0,
    post_var: [3]cuda.CUdeviceptr = .{ 0, 0, 0 },
    post_parity: u1 = 0,
    post_valid: bool = false,
    post_frame: u64 = 0,
    last_frame: u64 = 0,
    last: types.CameraData = undefined,
    has_last: bool = false,
    before_frame: u64 = 0,
    before: types.CameraData = undefined,
    has_before: bool = false,
    /// TAAU in Ausgabeauflösung: Verlauf 2x, MV + Tiefe (für die Frame Generation)
    up_w: u32 = 0,
    up_h: u32 = 0,
    up_buf: [3]cuda.CUdeviceptr = .{ 0, 0, 0 },
    up_parity: u1 = 0,
    up_valid: bool = false,
    up_frames: u32 = 0,
    up_frame: u64 = 0,
    up_exposure: f32 = 1,
    up_tonemap: u32 = 0,
    up_bgra: u32 = 0,
    /// entrauschte HDR-Farbe in Renderauflösung (Eingang des TAAU)
    lr_buf: cuda.CUdeviceptr = 0,
    /// DLSS (NGX) je Ansicht: Tiefe, Rauheit und spiegelnde Albedo als Eingaben
    dlss_feature: ?*dlss.Feature = null,
    depth_buf: cuda.CUdeviceptr = 0,
    rough_buf: cuda.CUdeviceptr = 0,
    spec_buf: cuda.CUdeviceptr = 0,
    /// Frame Generation: vorwärts projizierte Tiefe und Bewegung
    fg_depth: cuda.CUdeviceptr = 0,
    fg_mv: cuda.CUdeviceptr = 0,
    /// indirekte Beleuchtung in halber Auflösung ([4]f16)
    gi_buf: cuda.CUdeviceptr = 0,
    gi_w: u32 = 0,
    gi_h: u32 = 0,
};

/// Verzögerte Freigabe einer Geometrie. Bis zum nächsten Commit kann die GPU
/// sie noch über den alten Zustand (und die alte IAS) erreichen; erst danach
/// wird ein Event aufgenommen, und erst wenn die GPU daran vorbei ist, wird
/// der Speicher freigegeben.
const Deferred = struct {
    frame: u64,
    event: cuda.CUevent = null,
    node_off: u64 = 0,
    node_words: u64 = 0,
    leaf_off: u64 = 0,
    leaf_count: u64 = 0,
    attr_off: u64 = 0,
    attr_count: u64 = 0,
    /// Gerätepuffer (GAS, Primitive, Voxelliste)
    buffers: [4]cuda.CUdeviceptr = .{ 0, 0, 0, 0 },
    /// Geometrieindex nach der Freigabe wieder vergeben
    free_slot: ?u32 = null,
    /// Chunk-Batch, dessen Referenz beim Freigeben abgegeben wird
    batch: ?u32 = null,
    /// buffers stammen aus cuMemAllocAsync: stream-geordnet freigeben
    async_free: bool = false,
};

/// Ausführung eines GPU-Baus auf einem bestimmten Stream
const ExecState = struct {
    ctx: *Context,
    stream: cuda.CUstream,
};

pub const Context = struct {
    gpa: Allocator,
    drv: cuda.Driver,
    device: cuda.CUdevice,
    cu_ctx: cuda.CUcontext,
    owns_primary: bool,
    stream: cuda.CUstream,
    owns_stream: bool,
    /// zweiter Stream für Hintergrundbauten (Welt): wartet nie auf das Rendern
    aux_stream: cuda.CUstream = null,
    /// Nachbearbeitung läuft auf einem eigenen Stream: DLSS und TAA arbeiten
    /// auf den Tensorkernen, während der nächste Frame schon auf den
    /// Shader-Einheiten rendert. Ereignisse halten die Reihenfolge ein.
    post_stream: cuda.CUstream = null,
    render_done: cuda.CUevent = null,
    post_done: cuda.CUevent = null,
    post_pending: bool = false,
    async_post: bool = false,
    /// true, solange die Nachbearbeitung Kernel einreiht
    on_post: bool = false,
    aux_event: cuda.CUevent = null,
    exec_main: ExecState = undefined,
    exec_aux: ExecState = undefined,
    log_fn: api.LogFn,
    log_user: ?*anyopaque,
    debug: bool,
    force_rt: bool,

    module: cuda.CUmodule = null,
    fn_update: cuda.CUfunction = null,
    fn_render: cuda.CUfunction = null,
    fn_trace: cuda.CUfunction = null,
    fn_rt_instances: cuda.CUfunction = null,
    fn_temporal: cuda.CUfunction = null,
    fn_atrous: cuda.CUfunction = null,
    fn_resolve: cuda.CUfunction = null,
    fn_taau: cuda.CUfunction = null,
    fn_present: cuda.CUfunction = null,
    fn_animate: cuda.CUfunction = null,
    fn_gi: cuda.CUfunction = null,
    fn_dlss_prepare: cuda.CUfunction = null,
    fn_gi_combine: cuda.CUfunction = null,
    /// Skelettanimation (auf der GPU abgespielt)
    anim: anim.Animation = .{},
    /// NVIDIA NGX (DLSS), beim ersten Gebrauch geladen
    ngx: ?*dlss.Ngx = null,
    fn_framegen: cuda.CUfunction = null,
    fn_fg_splat: cuda.CUfunction = null,
    fn_fg_splat_mv: cuda.CUfunction = null,
    fn_build: cuda.CUfunction = null,
    fn_gen_terrain: cuda.CUfunction = null,
    fn_fx_pack_mvd: cuda.CUfunction = null,
    fn_fx_dof: cuda.CUfunction = null,
    fn_fx_motion: cuda.CUfunction = null,
    fn_fx_bloom_pre: cuda.CUfunction = null,
    fn_fx_bloom_down: cuda.CUfunction = null,
    fn_fx_bloom_up: cuda.CUfunction = null,
    fn_fx_expose_scan: cuda.CUfunction = null,
    fn_fx_expose_apply: cuda.CUfunction = null,
    fn_fx_resolve: cuda.CUfunction = null,
    fn_edit_apply: cuda.CUfunction = null,
    fn_edit_compact: cuda.CUfunction = null,
    fn_edit_append: cuda.CUfunction = null,

    // Materialien und Licht (Host-Kopie + Gerät)
    materials: [types.max_materials]types.Material = undefined,
    lighting: types.Lighting = undefined,
    materials_dev: cuda.CUdeviceptr = 0,
    lighting_dev: cuda.CUdeviceptr = 0,

    /// RT-Cores (OptiX); null = CUDA-Traversierung
    rt: ?*Rt = null,

    // GPU-Speicher
    node_pool: cuda.CUdeviceptr = 0,
    leaf_pool: cuda.CUdeviceptr = 0,
    attr_pool: cuda.CUdeviceptr = 0,
    geometry_table: cuda.CUdeviceptr = 0,
    /// Texturen: Tabelle auf dem Gerät und die Datenpuffer je Platz
    texture_table: cuda.CUdeviceptr = 0,
    texture_data: []cuda.CUdeviceptr = &.{},
    texture_high: u32 = 0,
    /// Umgebungskarte und ihre Verteilung
    env_data: cuda.CUdeviceptr = 0,
    env_cond: cuda.CUdeviceptr = 0,
    env_marginal: cuda.CUdeviceptr = 0,
    env_w: u32 = 0,
    env_h: u32 = 0,
    env_total: f32 = 0,
    env_mean: f32 = 0,
    instance_buf: [2]cuda.CUdeviceptr = .{ 0, 0 },
    update_scratch: cuda.CUdeviceptr = 0,
    scene_dev: cuda.CUdeviceptr = 0,

    node_alloc: RangeAlloc = .{ .capacity = 0 },
    leaf_alloc: RangeAlloc = .{ .capacity = 0 },
    attr_alloc: RangeAlloc = .{ .capacity = 0 },

    // gepinnter Upload-Puffer (linear, beim Umlauf wird auf den Stream gewartet)
    staging: []u8 = &.{},
    staging_head: usize = 0,
    staging_event: cuda.CUevent = null,

    // Host-Tabellen
    geometries: []GeometrySlot = &.{},
    geometry_free: std.ArrayList(u32) = .empty,
    geometry_high: u32 = 0,
    instances: []InstanceSlot = &.{},
    instance_free: std.ArrayList(u32) = .empty,
    instance_high: u32 = 0,
    views: []ViewSlot = &.{},
    view_free: std.ArrayList(u32) = .empty,
    view_high: u32 = 0,

    dirty_cur: std.ArrayList(u32) = .empty,
    dirty_prev: std.ArrayList(u32) = .empty,
    deferred: std.ArrayList(Deferred) = .empty,
    batches: std.ArrayList(ChunkBatch) = .empty,
    batch_free: std.ArrayList(u32) = .empty,
    history_counter: u32 = 0,
    /// seit dem letzten Commit strukturelle Instanzänderung (für die IAS)
    instances_structural: bool = false,

    // Frame
    frame: u64 = 0,
    parity: u1 = 0,
    time: f64 = 0,
    time_prev: f64 = 0,
    origin: [3]f64 = .{ 0, 0, 0 },
    origin_prev: [3]f64 = .{ 0, 0, 0 },

    const scratch_write_bytes = @sizeOf(types.InstanceWrite);

    // -----------------------------------------------------------------------
    // Erzeugen und Zerstören
    // -----------------------------------------------------------------------

    pub fn create(gpa: Allocator, info: *const api.CreateInfo) Error!*Context {
        if (info.struct_size < @sizeOf(api.CreateInfo)) return fail(error.InvalidArgument, "PyrCreateInfo.struct_size zu klein", .{});
        if ((info.version >> 16) != api.version_major) return fail(error.Version, "Header-Version {d}.{d} passt nicht zur Bibliothek {d}.{d}", .{ info.version >> 16, info.version & 0xFFFF, api.version_major, api.version_minor });

        var drv = cuda.Driver.load() catch return fail(error.NotFound, "libcuda nicht gefunden oder unvollständig", .{});
        errdefer drv.lib.close();
        if (drv.cuInit(0) != cuda.CUDA_SUCCESS) return fail(error.Cuda, "cuInit fehlgeschlagen", .{});

        var device: cuda.CUdevice = 0;
        if (drv.cuDeviceGet(&device, info.device) != cuda.CUDA_SUCCESS) return fail(error.InvalidArgument, "CUDA-Gerät {d} nicht vorhanden", .{info.device});

        var cu_ctx: cuda.CUcontext = @ptrCast(info.cuda_context);
        var owns_primary = false;
        if (cu_ctx == null) {
            if (drv.cuDevicePrimaryCtxRetain(&cu_ctx, device) != cuda.CUDA_SUCCESS) return fail(error.Cuda, "primärer CUDA-Kontext nicht verfügbar", .{});
            owns_primary = true;
        }
        errdefer if (owns_primary) {
            _ = drv.cuDevicePrimaryCtxRelease_v2(device);
        };

        const self = gpa.create(Context) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .drv = drv,
            .device = device,
            .cu_ctx = cu_ctx,
            .owns_primary = owns_primary,
            .stream = @ptrCast(info.cuda_stream),
            .owns_stream = info.cuda_stream == null,
            .log_fn = info.log,
            .log_user = info.log_user,
            .debug = info.flags & api.create_debug != 0,
            .force_rt = info.flags & api.create_force_rt != 0,
        };

        try self.enter();
        defer self.leave();
        self.initResources(info) catch |e| {
            self.freeResources();
            return e;
        };
        return self;
    }

    fn initResources(self: *Context, info: *const api.CreateInfo) Error!void {
        const drv = &self.drv;
        if (self.owns_stream) try self.check(drv.cuStreamCreate(&self.stream, cuda.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        try self.check(drv.cuStreamCreate(&self.aux_stream, cuda.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        try self.check(drv.cuStreamCreate(&self.post_stream, cuda.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        try self.check(drv.cuEventCreate(&self.render_done, cuda.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        try self.check(drv.cuEventCreate(&self.post_done, cuda.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        self.async_post = info.flags & api.create_async_post != 0;
        try self.check(drv.cuEventCreate(&self.aux_event, cuda.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        self.exec_main = .{ .ctx = self, .stream = self.stream };
        self.exec_aux = .{ .ctx = self, .stream = self.aux_stream };
        // Bauspeicher im Pool behalten statt nach jedem Sync ans System zurückzugeben
        var pool: cuda.CUmemoryPool = null;
        try self.check(drv.cuDeviceGetDefaultMemPool(&pool, self.device), "cuDeviceGetDefaultMemPool");
        var threshold: u64 = std.math.maxInt(u64);
        try self.check(drv.cuMemPoolSetAttribute(pool, cuda.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, @ptrCast(&threshold)), "cuMemPoolSetAttribute");

        var major: c_int = 0;
        var minor: c_int = 0;
        try self.check(drv.cuDeviceGetAttribute(&major, cuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device), "cuDeviceGetAttribute");
        try self.check(drv.cuDeviceGetAttribute(&minor, cuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device), "cuDeviceGetAttribute");
        var name_buf: [256]u8 = undefined;
        @memset(&name_buf, 0);
        _ = drv.cuDeviceGetName(&name_buf, name_buf.len, self.device);
        try self.loadKernels();
        self.logf(1, "Pyrit: {s} (sm_{d}{d})", .{ std.mem.sliceTo(&name_buf, 0), major, minor });

        // Kapazitäten
        const max_instances: u32 = if (info.max_instances == 0) 65536 else info.max_instances;
        const max_geometries: u32 = if (info.max_geometries == 0) 4096 else info.max_geometries;
        const max_views: u32 = if (info.max_views == 0) 16 else info.max_views;
        const node_bytes = if (info.node_pool_bytes == 0) 256 << 20 else info.node_pool_bytes;
        const leaf_bytes = if (info.leaf_pool_bytes == 0) 256 << 20 else info.leaf_pool_bytes;
        const attr_bytes = if (info.attribute_pool_bytes == 0) 256 << 20 else info.attribute_pool_bytes;
        const staging_bytes: usize = @intCast(if (info.staging_bytes == 0) 64 << 20 else info.staging_bytes);
        if (node_bytes / 4 > std.math.maxInt(u32) or leaf_bytes / 8 > std.math.maxInt(u32) or attr_bytes / 4 > std.math.maxInt(u32))
            return fail(error.InvalidArgument, "Pools dürfen höchstens 2^32 Einträge haben", .{});
        if (staging_bytes < 1 << 20) return fail(error.InvalidArgument, "staging_bytes muss mindestens 1 MiB sein", .{});

        self.node_alloc = try oom(RangeAlloc.init(self.gpa, node_bytes / 4));
        self.leaf_alloc = try oom(RangeAlloc.init(self.gpa, leaf_bytes / 8));
        self.attr_alloc = try oom(RangeAlloc.init(self.gpa, attr_bytes / 4));

        self.node_pool = try self.devAlloc(node_bytes, "Knotenpool");
        self.leaf_pool = try self.devAlloc(leaf_bytes, "Blattpool");
        self.attr_pool = try self.devAlloc(attr_bytes, "Attributpool");
        self.geometry_table = try self.devAlloc(@as(u64, max_geometries) * @sizeOf(types.GeometryData), "Geometrietabelle");
        const max_textures: u32 = if (info.max_textures == 0) 256 else info.max_textures;
        // Platz 0 bleibt frei: 0 bedeutet im Material "keine Textur"
        self.texture_table = try self.devAlloc(@as(u64, max_textures + 1) * @sizeOf(types.TextureData), "Texturtabelle");
        try self.check(self.drv.cuMemsetD8Async(self.texture_table, 0, @as(u64, max_textures + 1) * @sizeOf(types.TextureData), self.stream), "cuMemsetD8Async");
        self.texture_data = try oom(self.gpa.alloc(cuda.CUdeviceptr, max_textures + 1));
        @memset(self.texture_data, 0);
        const inst_bytes = @as(u64, max_instances) * @sizeOf(types.InstanceData);
        self.instance_buf[0] = try self.devAlloc(inst_bytes, "Instanzzustand");
        self.instance_buf[1] = try self.devAlloc(inst_bytes, "Instanzzustand");
        self.update_scratch = try self.devAlloc(@as(u64, max_instances) * (scratch_write_bytes + 4), "Aktualisierungspuffer");
        self.scene_dev = try self.devAlloc(@sizeOf(types.Scene), "Szene");
        try self.check(drv.cuMemsetD8Async(self.instance_buf[0], 0, inst_bytes, self.stream), "cuMemsetD8Async");
        try self.check(drv.cuMemsetD8Async(self.instance_buf[1], 0, inst_bytes, self.stream), "cuMemsetD8Async");

        var host: ?*anyopaque = null;
        try self.check(drv.cuMemHostAlloc(&host, staging_bytes, cuda.CU_MEMHOSTALLOC_PORTABLE | cuda.CU_MEMHOSTALLOC_WRITECOMBINED), "cuMemHostAlloc");
        self.staging = @as([*]u8, @ptrCast(host.?))[0..staging_bytes];
        try self.check(drv.cuEventCreate(&self.staging_event, cuda.CU_EVENT_DISABLE_TIMING), "cuEventCreate");

        self.geometries = try oom(self.gpa.alloc(GeometrySlot, max_geometries));
        @memset(self.geometries, .{});
        self.instances = try oom(self.gpa.alloc(InstanceSlot, max_instances));
        @memset(self.instances, .{});
        self.views = try oom(self.gpa.alloc(ViewSlot, max_views));
        @memset(self.views, .{});

        self.materials_dev = try self.devAlloc(@sizeOf(@TypeOf(self.materials)), "Materialien");
        self.lighting_dev = try self.devAlloc(@sizeOf(types.Lighting), "Licht");
        for (&self.materials) |*m| m.* = defaultMaterial();
        self.lighting = defaultLighting();
        try self.uploadValue(self.materials_dev, &self.materials);
        try self.uploadValue(self.lighting_dev, &self.lighting);
        try self.uploadScene();

        if (info.flags & api.create_no_rt == 0) {
            self.rt = Rt.init(self, info.rt_leaf_log2) catch |e| blk: {
                if (e == error.OutOfMemory) return e;
                self.logf(2, "Pyrit: RT-Cores werden nicht genutzt ({s}); CUDA-Traversierung", .{diag.message()});
                diag.clear();
                break :blk null;
            };
            if (self.rt) |r| self.logf(3, "Pyrit: RT-Cores aktiv (RTCore {d}, Teilbäume 2^{d})", .{ r.rtcore_version, r.rt_log2 });
        }
    }

    /// Lädt die eingebetteten Zig-Kernel (PTX). Der Treiber übersetzt sie beim
    /// ersten Laden für die vorhandene GPU und legt das Ergebnis in seinem Cache ab.
    fn loadKernels(self: *Context) Error!void {
        var err_log: [8192]u8 = undefined;
        @memset(&err_log, 0);
        const opts = [_]c_int{ cuda.CU_JIT_ERROR_LOG_BUFFER, cuda.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
        var vals = [_]?*anyopaque{ @ptrCast(&err_log), @ptrFromInt(err_log.len) };
        const r = self.drv.cuModuleLoadDataEx(&self.module, ptx.ptr, opts.len, &opts, &vals);
        if (r != cuda.CUDA_SUCCESS)
            return fail(error.Compile, "PTX laden: {s}\n{s}", .{ self.drv.errorString(r), std.mem.sliceTo(&err_log, 0) });
        try self.check(self.drv.cuModuleGetFunction(&self.fn_update, self.module, "pyr_k_update_instances"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_render, self.module, "pyr_k_render"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_trace, self.module, "pyr_k_trace"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_rt_instances, self.module, "pyr_k_build_rt_instances"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_temporal, self.module, "pyr_k_temporal"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_atrous, self.module, "pyr_k_atrous"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_resolve, self.module, "pyr_k_resolve"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_taau, self.module, "pyr_k_taau"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_present, self.module, "pyr_k_present"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_animate, self.module, "pyr_k_animate"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_gi, self.module, "pyr_k_gi"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_gi_combine, self.module, "pyr_k_gi_combine"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_dlss_prepare, self.module, "pyr_k_dlss_prepare"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_framegen, self.module, "pyr_k_framegen"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_fg_splat, self.module, "pyr_k_fg_splat"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_fg_splat_mv, self.module, "pyr_k_fg_splat_mv"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_build, self.module, "pyr_k_build"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_gen_terrain, self.module, "pyr_k_gen_terrain"), "cuModuleGetFunction");
        inline for (.{
            .{ "fn_fx_pack_mvd", "pyr_k_fx_pack_mvd" },
            .{ "fn_fx_dof", "pyr_k_fx_dof" },
            .{ "fn_fx_motion", "pyr_k_fx_motion" },
            .{ "fn_fx_bloom_pre", "pyr_k_fx_bloom_pre" },
            .{ "fn_fx_bloom_down", "pyr_k_fx_bloom_down" },
            .{ "fn_fx_bloom_up", "pyr_k_fx_bloom_up" },
            .{ "fn_fx_expose_scan", "pyr_k_fx_expose_scan" },
            .{ "fn_fx_expose_apply", "pyr_k_fx_expose_apply" },
            .{ "fn_fx_resolve", "pyr_k_fx_resolve" },
        }) |f| {
            try self.check(self.drv.cuModuleGetFunction(&@field(self, f[0]), self.module, f[1]), "cuModuleGetFunction");
        }
        try self.check(self.drv.cuModuleGetFunction(&self.fn_edit_apply, self.module, "pyr_k_edit_apply"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_edit_compact, self.module, "pyr_k_edit_compact"), "cuModuleGetFunction");
        try self.check(self.drv.cuModuleGetFunction(&self.fn_edit_append, self.module, "pyr_k_edit_append"), "cuModuleGetFunction");
    }

    pub fn destroy(self: *Context) void {
        self.enter() catch {};
        if (self.stream != null) _ = self.drv.cuStreamSynchronize(self.stream);
        if (self.post_stream != null) _ = self.drv.cuStreamSynchronize(self.post_stream);
        if (self.aux_stream != null) _ = self.drv.cuStreamSynchronize(self.aux_stream);
        self.freeResources();
        self.leave();
        if (self.owns_primary) _ = self.drv.cuDevicePrimaryCtxRelease_v2(self.device);
        self.drv.lib.close();
        self.gpa.destroy(self);
    }

    fn freeResources(self: *Context) void {
        const drv = &self.drv;
        if (self.rt) |r| r.deinit(self);
        self.rt = null;
        self.anim.deinit(self);
        for (self.deferred.items) |d| {
            if (d.event != null) _ = drv.cuEventDestroy_v2(d.event);
            for (d.buffers) |b| {
                if (b != 0) _ = drv.cuMemFree_v2(b);
            }
        }
        for (self.geometries) |g| {
            if (g.vox_keys != 0) _ = drv.cuMemFree_v2(g.vox_keys);
            if (g.vox_attrs != 0) _ = drv.cuMemFree_v2(g.vox_attrs);
        }
        self.deferred.deinit(self.gpa);
        for (self.batches.items) |b| {
            if (b.prims != 0) _ = drv.cuMemFree_v2(b.prims);
        }
        self.batches.deinit(self.gpa);
        self.batch_free.deinit(self.gpa);
        for (self.views) |*v| {
            freePost(drv, v);
            self.freeDlss(v);
        }
        if (self.ngx) |n| n.deinit(self.gpa);
        self.ngx = null;
        for ([_]cuda.CUdeviceptr{ self.materials_dev, self.lighting_dev }) |p| {
            if (p != 0) _ = drv.cuMemFree_v2(p);
        }
        self.freeEnv();
        for (self.texture_data) |t| {
            if (t != 0) _ = self.drv.cuMemFree_v2(t);
        }
        if (self.texture_data.len != 0) self.gpa.free(self.texture_data);
        for ([_]cuda.CUdeviceptr{ self.node_pool, self.leaf_pool, self.attr_pool, self.geometry_table, self.texture_table, self.instance_buf[0], self.instance_buf[1], self.update_scratch, self.scene_dev }) |p| {
            if (p != 0) _ = drv.cuMemFree_v2(p);
        }
        if (self.staging.len != 0) _ = drv.cuMemFreeHost(self.staging.ptr);
        if (self.staging_event != null) _ = drv.cuEventDestroy_v2(self.staging_event);
        if (self.module != null) _ = drv.cuModuleUnload(self.module);
        if (self.owns_stream and self.stream != null) _ = drv.cuStreamDestroy_v2(self.stream);
        if (self.aux_stream != null) _ = drv.cuStreamDestroy_v2(self.aux_stream);
        if (self.post_stream != null) _ = drv.cuStreamDestroy_v2(self.post_stream);
        if (self.render_done != null) _ = drv.cuEventDestroy_v2(self.render_done);
        if (self.post_done != null) _ = drv.cuEventDestroy_v2(self.post_done);
        if (self.aux_event != null) _ = drv.cuEventDestroy_v2(self.aux_event);
        self.node_alloc.deinit(self.gpa);
        self.leaf_alloc.deinit(self.gpa);
        self.attr_alloc.deinit(self.gpa);
        self.gpa.free(self.geometries);
        self.gpa.free(self.instances);
        self.gpa.free(self.views);
        self.geometry_free.deinit(self.gpa);
        self.instance_free.deinit(self.gpa);
        self.view_free.deinit(self.gpa);
        self.dirty_cur.deinit(self.gpa);
        self.dirty_prev.deinit(self.gpa);
    }

    // -----------------------------------------------------------------------
    // Hilfen
    // -----------------------------------------------------------------------

    /// Macht den Kontext auf dem aufrufenden Thread aktuell (für jeden API-Aufruf).
    pub fn enter(self: *Context) Error!void {
        try self.check(self.drv.cuCtxPushCurrent_v2(self.cu_ctx), "cuCtxPushCurrent");
    }

    pub fn leave(self: *Context) void {
        var old: cuda.CUcontext = null;
        _ = self.drv.cuCtxPopCurrent_v2(&old);
    }

    pub fn check(self: *Context, r: cuda.CUresult, what: []const u8) Error!void {
        if (r == cuda.CUDA_SUCCESS) return;
        return fail(error.Cuda, "{s}: {s} ({d})", .{ what, self.drv.errorString(r), r });
    }

    pub fn devAlloc(self: *Context, bytes: u64, what: []const u8) Error!cuda.CUdeviceptr {
        var p: cuda.CUdeviceptr = 0;
        const r = self.drv.cuMemAlloc_v2(&p, @intCast(@max(bytes, 256)));
        if (r != cuda.CUDA_SUCCESS) return fail(error.OutOfMemory, "{s}: {d} Bytes Gerätespeicher nicht verfügbar ({s})", .{ what, bytes, self.drv.errorString(r) });
        return p;
    }

    pub fn logf(self: *Context, level: i32, comptime fmt: []const u8, args: anytype) void {
        const f = self.log_fn orelse return;
        var buf: [512]u8 = undefined;
        const s = std.fmt.bufPrintZ(&buf, fmt, args) catch return;
        f(self.log_user, level, s.ptr);
    }

    fn oom(v: anytype) Error!@typeInfo(@TypeOf(v)).error_union.payload {
        return v catch return fail(error.OutOfMemory, "Host-Speicher", .{});
    }

    /// Platz im gepinnten Upload-Puffer. Reicht der Rest nicht, wird auf alle
    /// bisherigen Kopien gewartet und vorn neu begonnen.
    fn stagingAlloc(self: *Context, n: usize) Error![]u8 {
        if (n > self.staging.len) return fail(error.OutOfMemory, "Upload von {d} Bytes größer als staging_bytes", .{n});
        var head = std.mem.alignForward(usize, self.staging_head, 256);
        if (head + n > self.staging.len) {
            try self.check(self.drv.cuEventRecord(self.staging_event, self.stream), "cuEventRecord");
            try self.check(self.drv.cuEventSynchronize(self.staging_event), "cuEventSynchronize");
            head = 0;
        }
        self.staging_head = head + n;
        return self.staging[head..][0..n];
    }

    pub fn upload(self: *Context, dst: cuda.CUdeviceptr, src: []const u8) Error!void {
        var off: usize = 0;
        const max_chunk = self.staging.len / 2;
        while (off < src.len) {
            const n = @min(src.len - off, max_chunk);
            const buf = try self.stagingAlloc(n);
            @memcpy(buf, src[off..][0..n]);
            try self.check(self.drv.cuMemcpyHtoDAsync_v2(dst + off, buf.ptr, n, self.stream), "cuMemcpyHtoDAsync");
            off += n;
        }
    }

    pub fn uploadValue(self: *Context, dst: cuda.CUdeviceptr, value: anytype) Error!void {
        try self.upload(dst, std.mem.asBytes(value));
    }

    /// Stream, auf den die Kernel gerade laufen (die Nachbearbeitung schaltet
    /// ihn auf post_stream um, damit sie mit dem nächsten Frame überlappt)
    /// Stream der Nachbearbeitung (für DLSS und andere Fremdaufrufe)
    pub fn postStream(self: *Context) cuda.CUstream {
        return self.activeStream();
    }

    fn activeStream(self: *Context) cuda.CUstream {
        return if (self.on_post) self.post_stream else self.stream;
    }

    pub fn launch(self: *Context, f: cuda.CUfunction, grid: [3]u32, block: [3]u32, params: []const ?*anyopaque) Error!void {
        const st = self.activeStream();
        try self.check(self.drv.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, st, @constCast(params.ptr), null), "cuLaunchKernel");
        if (self.debug) try self.check(self.drv.cuStreamSynchronize(st), "Kernel");
    }

    fn uploadScene(self: *Context) Error!void {
        const cur = self.instance_buf[self.parity];
        const prev = self.instance_buf[self.parity ^ 1];
        const scene = types.Scene{
            .nodes = self.node_pool,
            .leaves = self.leaf_pool,
            .attributes = self.attr_pool,
            .geometries = self.geometry_table,
            .instances = cur,
            .instances_prev = prev,
            .instance_count = self.instance_high,
            .geometry_count = self.geometry_high,
            .frame = self.frame,
            .time = self.time,
            .time_prev = self.time_prev,
            .origin = self.origin,
            .origin_prev = self.origin_prev,
            .materials = self.materials_dev,
            .lighting = self.lighting_dev,
            .transparent_materials = self.transparentMaterials(),
            .textures = self.texture_table,
            .texture_count = self.texture_high,
            .reserved_tex = 0,
        };
        try self.uploadValue(self.scene_dev, &scene);
    }

    fn allocSlot(free: *std.ArrayList(u32), high: *u32, capacity: usize) ?u32 {
        if (free.pop()) |i| return i;
        if (high.* >= capacity) return null;
        high.* += 1;
        return high.* - 1;
    }

    // -----------------------------------------------------------------------
    // Geometrie
    // -----------------------------------------------------------------------

    pub fn geometryLog2(self: *Context, h: usize) Error!u32 {
        return (try self.geometrySlot(h)).data.log2_size;
    }

    fn geometrySlot(self: *Context, h: usize) Error!*GeometrySlot {
        const d = decodeHandle(h);
        if (h == 0 or d.index >= self.geometries.len) return fail(error.InvalidHandle, "ungültige Geometrie", .{});
        const s = &self.geometries[d.index];
        if (!s.alive or s.generation != d.generation) return fail(error.InvalidHandle, "Geometrie existiert nicht mehr", .{});
        return s;
    }

    /// Geometrie aus einer Host-DAG (CPU-Bau)
    pub fn geometryCreate(self: *Context, dag: *const dag_builder.Dag) Error!usize {
        try self.releaseDeferred(false);
        const idx = allocSlot(&self.geometry_free, &self.geometry_high, self.geometries.len) orelse
            return fail(error.Capacity, "max_geometries erreicht", .{});
        errdefer self.geometry_free.append(self.gpa, idx) catch {};

        const attr_len: u64 = if (dag.attributes) |a| a.len else 0;
        const r = try self.allocRanges(dag.nodes.len, dag.leaves.len, attr_len);
        errdefer self.releaseRanges(r) catch {};
        try self.upload(self.node_pool + r.node_off * 4, std.mem.sliceAsBytes(dag.nodes));
        try self.upload(self.leaf_pool + r.leaf_off * 8, std.mem.sliceAsBytes(dag.leaves));
        if (dag.attributes) |a| try self.upload(self.attr_pool + r.attr_off * 4, std.mem.sliceAsBytes(a));

        const data = geometryData(r, dag.root, dag.log2_size, dag.attributes != null);
        if (self.rt) |rt| try rt.buildGeometry(self, idx, dag, data);
        try self.installGeometry(idx, r, data);
        return encodeHandle(idx, self.geometries[idx].generation);
    }

    const Ranges = struct { node_off: u64, node_words: u64, leaf_off: u64, leaf_count: u64, attr_off: u64, attr_count: u64 };

    fn allocRanges(self: *Context, node_words: u64, leaf_count: u64, attr_count: u64) Error!Ranges {
        const node_off = self.node_alloc.alloc(node_words) orelse return fail(error.OutOfMemory, "Knotenpool voll ({d} Worte benötigt)", .{node_words});
        errdefer self.node_alloc.release(self.gpa, node_off, node_words) catch {};
        const leaf_off = self.leaf_alloc.alloc(leaf_count) orelse return fail(error.OutOfMemory, "Blattpool voll ({d} Bricks benötigt)", .{leaf_count});
        errdefer self.leaf_alloc.release(self.gpa, leaf_off, leaf_count) catch {};
        const attr_off = self.attr_alloc.alloc(attr_count) orelse return fail(error.OutOfMemory, "Attributpool voll ({d} Werte benötigt)", .{attr_count});
        return .{ .node_off = node_off, .node_words = node_words, .leaf_off = leaf_off, .leaf_count = leaf_count, .attr_off = attr_off, .attr_count = attr_count };
    }

    fn releaseRanges(self: *Context, r: Ranges) Error!void {
        try oom(self.node_alloc.release(self.gpa, r.node_off, r.node_words));
        try oom(self.leaf_alloc.release(self.gpa, r.leaf_off, r.leaf_count));
        try oom(self.attr_alloc.release(self.gpa, r.attr_off, r.attr_count));
    }

    fn geometryData(r: Ranges, root: u32, log2_size: u32, has_attributes: bool) types.GeometryData {
        return .{
            .node_offset = @intCast(r.node_off),
            .leaf_offset = @intCast(r.leaf_off),
            .attribute_offset = @intCast(r.attr_off),
            .root = root,
            .log2_size = log2_size,
            .flags = if (has_attributes) types.geometry_has_attributes else 0,
            .default_attribute = 1,
            .reserved = 0,
        };
    }

    fn installGeometry(self: *Context, idx: u32, r: Ranges, data: types.GeometryData) Error!void {
        const slot = &self.geometries[idx];
        slot.alive = true;
        slot.node_words = r.node_words;
        slot.leaf_count = r.leaf_count;
        slot.attribute_count = r.attr_count;
        slot.data = data;
        try self.uploadValue(self.geometry_table + @as(u64, idx) * @sizeOf(types.GeometryData), &slot.data);
    }

    // -----------------------------------------------------------------------
    // DAG-Bau und Änderungen auf der GPU
    // -----------------------------------------------------------------------

    pub fn gpuExec(self: *Context) gpu_build.Exec {
        return execOn(&self.exec_main);
    }

    /// Bau auf dem Hintergrund-Stream (synchronisiert nur mit sich selbst)
    pub fn gpuExecAux(self: *Context) gpu_build.Exec {
        return execOn(&self.exec_aux);
    }

    /// Speicher kommt stream-geordnet aus dem Gerätepool: Anlegen und
    /// Freigeben kosten fast nichts und synchronisieren das Gerät nicht.
    fn execOn(state: *ExecState) gpu_build.Exec {
        const E = struct {
            fn cast(c: *anyopaque) *ExecState {
                return @ptrCast(@alignCast(c));
            }
            fn alloc(c: *anyopaque, bytes: u64) Error!u64 {
                const st = cast(c);
                var p: cuda.CUdeviceptr = 0;
                const r = st.ctx.drv.cuMemAllocAsync(&p, @max(bytes, 16), st.stream);
                if (r != cuda.CUDA_SUCCESS) return fail(error.OutOfMemory, "GPU-Speicher für den DAG-Bau ({d} Bytes): {s}", .{ bytes, st.ctx.drv.errorString(r) });
                return p;
            }
            fn free(c: *anyopaque, p: u64) void {
                const st = cast(c);
                _ = st.ctx.drv.cuMemFreeAsync(p, st.stream);
            }
            fn memset(c: *anyopaque, p: u64, v: u8, bytes: u64) Error!void {
                const st = cast(c);
                try st.ctx.check(st.ctx.drv.cuMemsetD8Async(p, v, bytes, st.stream), "cuMemsetD8Async");
            }
            fn copy(c: *anyopaque, dst: u64, src: u64, bytes: u64) Error!void {
                const st = cast(c);
                if (bytes > 0) try st.ctx.check(st.ctx.drv.cuMemcpyDtoDAsync_v2(dst, src, bytes, st.stream), "cuMemcpyDtoDAsync");
            }
            fn launch(c: *anyopaque, p: *const types.BuildParams, threads: u32) Error!void {
                const st = cast(c);
                var pp = p.*;
                const params = [_]?*anyopaque{@ptrCast(&pp)};
                const b = types.build_block;
                try st.ctx.check(st.ctx.drv.cuLaunchKernel(st.ctx.fn_build, (threads + b - 1) / b, 1, 1, b, 1, 1, 0, st.stream, @constCast(&params), null), "cuLaunchKernel(Bau)");
            }
            fn read(c: *anyopaque, dst: []u8, src: u64) Error!void {
                const st = cast(c);
                try st.ctx.check(st.ctx.drv.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, st.stream), "cuMemcpyDtoHAsync");
                try st.ctx.check(st.ctx.drv.cuStreamSynchronize(st.stream), "cuStreamSynchronize");
            }
        };
        return .{ .ctx = state, .alloc = E.alloc, .free = E.free, .memset = E.memset, .copy = E.copy, .launch = E.launch, .read = E.read };
    }

    fn rtLog2(self: *const Context) u32 {
        return if (self.rt) |r| r.rt_log2 else rt_prims.default_log2;
    }

    /// Voxel nach Gerätespeicher (bei Host-Eingabe hochladen)
    fn voxelInput(self: *Context, voxels: ?*const anyopaque, count: u32, flags: u32) Error!struct { ptr: u64, owned: bool } {
        if (count == 0) return .{ .ptr = 0, .owned = false };
        const v = voxels orelse return fail(error.InvalidArgument, "voxels ist NULL", .{});
        if (flags & api.build_host_input == 0) return .{ .ptr = @intFromPtr(v), .owned = false };
        const bytes = @as(u64, count) * @sizeOf(api.Voxel);
        const dev = try self.devAlloc(bytes, "Voxel-Upload");
        errdefer _ = self.drv.cuMemFree_v2(dev);
        try self.upload(dev, @as([*]const u8, @ptrCast(v))[0..bytes]);
        return .{ .ptr = dev, .owned = true };
    }

    /// Übernimmt ein GPU-Bauergebnis in die Pools und baut die GAS.
    fn installBuilt(self: *Context, idx: u32, b: *gpu_build.Built, editable: bool) Error!void {
        const r = try self.allocRanges(b.node_words, b.leaf_count, b.voxel_count);
        errdefer self.releaseRanges(r) catch {};
        const e = self.gpuExec();
        try e.copy(e.ctx, self.node_pool + r.node_off * 4, b.nodes, @as(u64, b.node_words) * 4);
        try e.copy(e.ctx, self.leaf_pool + r.leaf_off * 8, b.leaves, @as(u64, b.leaf_count) * 8);
        try e.copy(e.ctx, self.attr_pool + r.attr_off * 4, b.attrs, @as(u64, b.voxel_count) * 4);
        const data = geometryData(r, b.root, b.log2_size, true);
        if (self.rt) |rt| {
            try rt.buildGas(self, idx, if (b.prim_count > 0) b.prims else 0, b.aabbs, b.prim_count, b.rt_log2, data);
            if (b.prim_count > 0) b.prims = 0; // gehört jetzt der GAS
        }
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        try self.installGeometry(idx, r, data);
        const slot = &self.geometries[idx];
        if (editable) {
            slot.vox_keys = b.keys;
            slot.vox_attrs = b.attrs;
            slot.vox_count = b.voxel_count;
            b.keys = 0;
            b.attrs = 0;
        }
    }

    pub fn geometryBuild(self: *Context, log2_size: u32, voxels: ?*const anyopaque, count: u32, flags: u32) Error!usize {
        try self.releaseDeferred(false);
        const idx = allocSlot(&self.geometry_free, &self.geometry_high, self.geometries.len) orelse
            return fail(error.Capacity, "max_geometries erreicht", .{});
        errdefer self.geometry_free.append(self.gpa, idx) catch {};
        const in = try self.voxelInput(voxels, count, flags);
        defer if (in.owned) {
            _ = self.drv.cuMemFree_v2(in.ptr);
        };
        const e = self.gpuExec();
        var b = try gpu_build.build(e, log2_size, self.rtLog2(), 0, 0, 0, in.ptr, count);
        defer b.free(e);
        try self.installBuilt(idx, &b, flags & api.build_editable != 0);
        self.geometries[idx].refs = 0;
        return encodeHandle(idx, self.geometries[idx].generation);
    }

    /// Setzt/entfernt Voxel (Attribut 0 = entfernen). Neubau auf der GPU; die
    /// alte Fassung wird nach dem nächsten Commit freigegeben.
    pub fn geometryEdit(self: *Context, h: usize, voxels: ?*const anyopaque, count: u32, flags: u32) Error!void {
        const s = try self.geometrySlot(h);
        if (s.vox_keys == 0) return fail(error.InvalidArgument, "Geometrie ist nicht änderbar (mit pyr_geometry_build und PYR_BUILD_EDITABLE anlegen)", .{});
        const idx = decodeHandle(h).index;
        try oom(self.deferred.ensureUnusedCapacity(self.gpa, 1));
        const in = try self.voxelInput(voxels, count, flags);
        defer if (in.owned) {
            _ = self.drv.cuMemFree_v2(in.ptr);
        };
        const e = self.gpuExec();
        var b = try gpu_build.build(e, s.data.log2_size, self.rtLog2(), s.vox_keys, s.vox_attrs, s.vox_count, in.ptr, count);
        defer b.free(e);

        // alte Fassung vormerken
        var old = Deferred{
            .frame = self.frame,
            .node_off = s.data.node_offset,
            .node_words = s.node_words,
            .leaf_off = s.data.leaf_offset,
            .leaf_count = s.leaf_count,
            .attr_off = s.data.attribute_offset,
            .attr_count = s.attribute_count,
            .buffers = .{ s.vox_keys, s.vox_attrs, 0, 0 },
        };
        if (self.rt) |rt| {
            const g = rt.takeGas(idx);
            old.buffers[2] = g[0];
            old.buffers[3] = g[1];
        }
        s.vox_keys = 0;
        s.vox_attrs = 0;
        try self.installBuilt(idx, &b, true);
        self.deferred.appendAssumeCapacity(old);
    }

    /// Kopiert eine Geometrie von der GPU in eine Host-DAG (z. B. zum Speichern).
    pub fn geometryDownload(self: *Context, h: usize) Error!dag_builder.Dag {
        const s = try self.geometrySlot(h);
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        const nodes = try oom(self.gpa.alloc(u32, s.node_words));
        errdefer self.gpa.free(nodes);
        const leaves = try oom(self.gpa.alloc(u64, s.leaf_count));
        errdefer self.gpa.free(leaves);
        const has_attrs = s.data.flags & types.geometry_has_attributes != 0;
        const attrs: ?[]u32 = if (has_attrs) try oom(self.gpa.alloc(u32, s.attribute_count)) else null;
        errdefer if (attrs) |a| self.gpa.free(a);
        try self.check(self.drv.cuMemcpyDtoH_v2(nodes.ptr, self.node_pool + @as(u64, s.data.node_offset) * 4, nodes.len * 4), "cuMemcpyDtoH");
        try self.check(self.drv.cuMemcpyDtoH_v2(leaves.ptr, self.leaf_pool + @as(u64, s.data.leaf_offset) * 8, leaves.len * 8), "cuMemcpyDtoH");
        if (attrs) |a| try self.check(self.drv.cuMemcpyDtoH_v2(a.ptr, self.attr_pool + @as(u64, s.data.attribute_offset) * 4, a.len * 4), "cuMemcpyDtoH");
        return .{
            .log2_size = s.data.log2_size,
            .root = s.data.root,
            .voxel_count = nodes[s.data.root + 1],
            .nodes = nodes,
            .leaves = leaves,
            .attributes = attrs,
        };
    }

    /// Gröbere Fassung (LOD) einer änderbaren Geometrie, auf der GPU: Kantenlänge
    /// 2^(log2 - shift). Für dieselbe Weltgröße die Instanz um 2^shift skalieren.
    pub fn geometryDownsample(self: *Context, h: usize, shift: u32, flags: u32) Error!usize {
        const src = try self.geometrySlot(h);
        if (src.vox_keys == 0) return fail(error.InvalidArgument, "Quelle ist nicht änderbar (mit PYR_BUILD_EDITABLE anlegen)", .{});
        try self.releaseDeferred(false);
        const idx = allocSlot(&self.geometry_free, &self.geometry_high, self.geometries.len) orelse
            return fail(error.Capacity, "max_geometries erreicht", .{});
        errdefer self.geometry_free.append(self.gpa, idx) catch {};
        const e = self.gpuExec();
        const s = &self.geometries[decodeHandle(h).index];
        var b = try gpu_build.downsample(e, s.data.log2_size, self.rtLog2(), s.vox_keys, s.vox_attrs, s.vox_count, shift);
        defer b.free(e);
        try self.installBuilt(idx, &b, flags & api.build_editable != 0);
        self.geometries[idx].refs = 0;
        return encodeHandle(idx, self.geometries[idx].generation);
    }

    /// Übernimmt einen Chunk-Batch (gpu_build.buildChunks): je nicht leerem
    /// Chunk eine Geometrie, alle mit gemeinsamen Pool-Bereichen. handles[c]
    /// ist 0 für leere Chunks. Die GAS aller Chunks entstehen in einem Zug.
    /// Läuft auf dem Hintergrund-Stream (b muss dort gebaut sein); der
    /// Render-Stream wartet per Event, bevor er die neuen Chunks sieht.
    pub fn installChunkBatch(self: *Context, b: *gpu_build.Built, out: gpu_build.ChunkOut, count: u32, handles: []usize) Error!void {
        @memset(handles[0..count], 0);
        var nonempty: u32 = 0;
        for (out.roots[0..count]) |r| {
            if (r != 0xFFFF_FFFF) nonempty += 1;
        }
        if (nonempty == 0) return;
        try self.releaseDeferred(false);

        // Geometrieplätze
        const idx = try oom(self.gpa.alloc(u32, nonempty));
        defer self.gpa.free(idx);
        var got: u32 = 0;
        errdefer for (idx[0..got]) |i| self.geometry_free.append(self.gpa, i) catch {};
        while (got < nonempty) : (got += 1) {
            idx[got] = allocSlot(&self.geometry_free, &self.geometry_high, self.geometries.len) orelse
                return fail(error.Capacity, "max_geometries erreicht (Welt braucht eine Geometrie je Chunk)", .{});
        }
        const bi: u32 = self.batch_free.pop() orelse blk: {
            try oom(self.batches.append(self.gpa, .{}));
            break :blk @intCast(self.batches.items.len - 1);
        };
        errdefer self.batch_free.append(self.gpa, bi) catch {};

        const r = try self.allocRanges(b.node_words, b.leaf_count, b.voxel_count);
        errdefer self.releaseRanges(r) catch {};
        const e = self.gpuExecAux();
        try e.copy(e.ctx, self.node_pool + r.node_off * 4, b.nodes, @as(u64, b.node_words) * 4);
        try e.copy(e.ctx, self.leaf_pool + r.leaf_off * 8, b.leaves, @as(u64, b.leaf_count) * 8);
        try e.copy(e.ctx, self.attr_pool + r.attr_off * 4, b.attrs, @as(u64, b.voxel_count) * 4);

        const datas = try oom(self.gpa.alloc(types.GeometryData, nonempty));
        defer self.gpa.free(datas);
        const jobs = try oom(self.gpa.alloc(Rt.GasJob, nonempty));
        defer self.gpa.free(jobs);
        var k: u32 = 0;
        var c: u32 = 0;
        while (c < count) : (c += 1) {
            if (out.roots[c] == 0xFFFF_FFFF) continue;
            var d = geometryData(r, out.roots[c], b.log2_size, true);
            d.attribute_offset += out.first_voxel[c];
            datas[k] = d;
            // Primitive liegen nach Chunks sortiert: bis zum nächsten nicht leeren Chunk
            var end: u32 = b.prim_count;
            var n = c + 1;
            while (n < count) : (n += 1) {
                if (out.roots[n] != 0xFFFF_FFFF) {
                    end = out.first_prim[n];
                    break;
                }
            }
            const first = out.first_prim[c];
            if (first == 0xFFFF_FFFF or end <= first or end > b.prim_count)
                return fail(error.InvalidArgument, "Chunk {d}/{d}: Primitive [{d}, {d}) von {d} (Wurzel {d})", .{ c, count, first, end, b.prim_count, out.roots[c] });
            jobs[k] = .{
                .index = idx[k],
                .prims = b.prims + @as(u64, first) * @sizeOf(types.RtPrim),
                .aabbs = b.aabbs + @as(u64, first) * 24,
                .count = end - first,
                .data = d,
            };
            k += 1;
        }
        if (self.rt) |rt| try rt.buildGasMany(self, jobs, b.rt_log2, self.aux_stream);
        // Render-Stream wartet auf Pools und GAS
        try self.check(self.drv.cuEventRecord(self.aux_event, self.aux_stream), "cuEventRecord");
        try self.check(self.drv.cuStreamWaitEvent(self.stream, self.aux_event, 0), "cuStreamWaitEvent");

        const batch = &self.batches.items[bi];
        batch.* = .{
            .refs = nonempty,
            .node_off = r.node_off,
            .node_words = r.node_words,
            .leaf_off = r.leaf_off,
            .leaf_count = r.leaf_count,
            .attr_off = r.attr_off,
            .attr_count = r.attr_count,
            .prims = if (self.rt != null) b.prims else 0,
        };
        if (self.rt != null) b.prims = 0; // gehört jetzt der Batch
        k = 0;
        c = 0;
        while (c < count) : (c += 1) {
            if (out.roots[c] == 0xFFFF_FFFF) continue;
            try self.installGeometry(idx[k], r, datas[k]);
            const slot = &self.geometries[idx[k]];
            slot.refs = 0;
            slot.batch = bi;
            handles[c] = encodeHandle(idx[k], slot.generation);
            k += 1;
        }
    }

    pub fn geometryDestroy(self: *Context, h: usize) Error!void {
        const s = try self.geometrySlot(h);
        if (s.refs != 0) return fail(error.InUse, "Geometrie wird noch von {d} Instanz(en) verwendet", .{s.refs});
        const idx = decodeHandle(h).index;
        try oom(self.deferred.ensureUnusedCapacity(self.gpa, 1));
        var d = Deferred{
            .frame = self.frame,
            .node_off = s.data.node_offset,
            .node_words = s.node_words,
            .leaf_off = s.data.leaf_offset,
            .leaf_count = s.leaf_count,
            .attr_off = s.data.attribute_offset,
            .attr_count = s.attribute_count,
            .buffers = .{ s.vox_keys, s.vox_attrs, 0, 0 },
            .free_slot = idx,
            .batch = s.batch,
        };
        if (s.batch != null) {
            d.async_free = true;
            d.node_words = 0;
            d.leaf_count = 0;
            d.attr_count = 0;
        }
        if (self.rt) |r| {
            try r.forgetGeometry(self, idx);
            const g = r.takeGas(idx);
            d.buffers[2] = g[0];
            d.buffers[3] = g[1];
        }
        s.alive = false;
        s.generation +%= 1;
        s.vox_keys = 0;
        s.vox_attrs = 0;
        s.vox_count = 0;
        s.batch = null;
        self.deferred.appendAssumeCapacity(d);
    }

    /// Nach einem Commit: Freigaben vormerken, die der neue Zustand nicht mehr erreicht.
    fn fenceDeferred(self: *Context) Error!void {
        for (self.deferred.items) |*d| {
            if (d.event != null or d.frame >= self.frame) continue;
            try self.check(self.drv.cuEventCreate(&d.event, cuda.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
            try self.check(self.drv.cuEventRecord(d.event, self.stream), "cuEventRecord");
        }
    }

    /// GPU-Bytes aller Chunk-Batches (Knoten, Blätter, Attribute)
    pub fn batchBytes(self: *const Context) u64 {
        var sum: u64 = 0;
        for (self.batches.items) |b| {
            if (b.refs != 0) sum += b.node_words * 4 + b.leaf_count * 8 + b.attr_count * 4;
        }
        return sum;
    }

    fn batchRelease(self: *Context, bi: u32) Error!void {
        const b = &self.batches.items[bi];
        b.refs -= 1;
        if (b.refs != 0) return;
        try self.releaseRanges(.{ .node_off = b.node_off, .node_words = b.node_words, .leaf_off = b.leaf_off, .leaf_count = b.leaf_count, .attr_off = b.attr_off, .attr_count = b.attr_count });
        if (b.prims != 0) _ = self.drv.cuMemFreeAsync(b.prims, self.stream);
        b.* = .{};
        try oom(self.batch_free.append(self.gpa, bi));
    }

    fn releaseDeferred(self: *Context, wait: bool) Error!void {
        var i: usize = 0;
        while (i < self.deferred.items.len) {
            const d = self.deferred.items[i];
            if (d.event == null) {
                i += 1;
                continue;
            }
            const r = if (wait) self.drv.cuEventSynchronize(d.event) else self.drv.cuEventQuery(d.event);
            if (r == cuda.CUDA_ERROR_NOT_READY) {
                i += 1;
                continue;
            }
            try self.check(r, "cuEventQuery");
            _ = self.drv.cuEventDestroy_v2(d.event);
            try self.releaseRanges(.{ .node_off = d.node_off, .node_words = d.node_words, .leaf_off = d.leaf_off, .leaf_count = d.leaf_count, .attr_off = d.attr_off, .attr_count = d.attr_count });
            for (d.buffers) |b| {
                if (b == 0) continue;
                _ = if (d.async_free) self.drv.cuMemFreeAsync(b, self.stream) else self.drv.cuMemFree_v2(b);
            }
            if (d.free_slot) |slot| try oom(self.geometry_free.append(self.gpa, slot));
            if (d.batch) |bi| try self.batchRelease(bi);
            _ = self.deferred.swapRemove(i);
        }
    }

    // -----------------------------------------------------------------------
    // Instanzen
    // -----------------------------------------------------------------------

    fn instanceSlot(self: *Context, h: usize) Error!*InstanceSlot {
        const d = decodeHandle(h);
        if (h == 0 or d.index >= self.instances.len) return fail(error.InvalidHandle, "ungültige Instanz", .{});
        const s = &self.instances[d.index];
        if (!s.alive or s.generation != d.generation) return fail(error.InvalidHandle, "Instanz existiert nicht mehr", .{});
        return s;
    }

    fn markDirty(self: *Context, index: u32) Error!void {
        const s = &self.instances[index];
        if (s.dirty_mark == self.frame + 1) return;
        s.dirty_mark = self.frame + 1;
        try oom(self.dirty_cur.append(self.gpa, index));
    }

    fn updateBounds(self: *Context, s: *InstanceSlot) void {
        const size: f32 = @floatFromInt(@as(u32, 1) << @intCast(self.geometries[s.geometry].data.log2_size));
        const box = xform.transformedBox(&s.data.object_to_world, size);
        s.data.bounds_min = box.min;
        s.data.bounds_max = box.max;
    }

    pub fn instanceCreate(self: *Context, geometry: usize) Error!usize {
        self.instances_structural = true;
        const g = try self.geometrySlot(geometry);
        const idx = allocSlot(&self.instance_free, &self.instance_high, self.instances.len) orelse
            return fail(error.Capacity, "max_instances erreicht", .{});
        const s = &self.instances[idx];
        s.alive = true;
        s.geometry = decodeHandle(geometry).index;
        self.history_counter +%= 1;
        const identity = [12]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
        s.data = std.mem.zeroes(types.InstanceData);
        s.data.object_to_world = identity;
        s.data.world_to_object = identity;
        s.data.geometry = s.geometry;
        s.data.mask = 0xFF;
        s.data.history = self.history_counter;
        s.data.flags = types.instance_active;
        self.updateBounds(s);
        g.refs += 1;
        self.markDirty(idx) catch |e| {
            s.alive = false;
            g.refs -= 1;
            return e;
        };
        return encodeHandle(idx, s.generation);
    }

    pub fn instanceDestroy(self: *Context, h: usize) Error!void {
        self.instances_structural = true;
        const s = try self.instanceSlot(h);
        const idx = decodeHandle(h).index;
        try self.markDirty(idx);
        self.geometries[s.geometry].refs -= 1;
        s.alive = false;
        s.generation +%= 1;
        s.data.flags = 0;
        try oom(self.instance_free.append(self.gpa, idx));
    }

    pub fn instanceSetTransform(self: *Context, h: usize, m: *const [12]f32) Error!void {
        const s = try self.instanceSlot(h);
        const inv = xform.inverse(m) orelse return fail(error.InvalidArgument, "Transformation ist nicht invertierbar", .{});
        s.data.object_to_world = m.*;
        s.data.world_to_object = inv;
        self.updateBounds(s);
        try self.markDirty(decodeHandle(h).index);
    }

    pub fn instanceSetGeometry(self: *Context, h: usize, geometry: usize) Error!void {
        self.instances_structural = true;
        const s = try self.instanceSlot(h);
        const g = try self.geometrySlot(geometry);
        self.geometries[s.geometry].refs -= 1;
        g.refs += 1;
        s.geometry = decodeHandle(geometry).index;
        s.data.geometry = s.geometry;
        self.history_counter +%= 1;
        s.data.history = self.history_counter; // andere Form: keine Vorgeschichte
        self.updateBounds(s);
        try self.markDirty(decodeHandle(h).index);
    }

    /// Nur die Sichtbarkeitsmaske: die IAS wird nachgeführt (Refit), nicht neu
    /// gebaut – wichtig, wenn viele Instanzen je Frame ein- und ausgeblendet
    /// werden (Welt-Chunks).
    pub fn instanceSetMask(self: *Context, h: usize, mask: u32) Error!void {
        const s = try self.instanceSlot(h);
        s.data.mask = mask;
        try self.markDirty(decodeHandle(h).index);
    }

    pub fn instanceSetUser(self: *Context, h: usize, user: u32) Error!void {
        const s = try self.instanceSlot(h);
        s.data.user = user;
        try self.markDirty(decodeHandle(h).index);
    }

    /// intern (Welt): Instanz erbt den Verlauf ihrer Umgebung
    pub fn instanceKeepHistory(self: *Context, h: usize) Error!void {
        const s = try self.instanceSlot(h);
        s.data.flags |= types.instance_keep_history;
        try self.markDirty(decodeHandle(h).index);
    }

    pub fn instanceResetHistory(self: *Context, h: usize) Error!void {
        const s = try self.instanceSlot(h);
        self.history_counter +%= 1;
        s.data.history = self.history_counter;
        try self.markDirty(decodeHandle(h).index);
    }

    // -----------------------------------------------------------------------
    // Frame
    // -----------------------------------------------------------------------

    pub fn commit(self: *Context, info: ?*const api.FrameInfo) Error!void {
        const next_frame = self.frame + 1;

        // Einträge des Vorframes, die jetzt nicht neu geschrieben werden, nach vorn kopieren
        var copy_count: u32 = 0;
        const write_count: u32 = @intCast(self.dirty_cur.items.len);
        const total = write_count + self.dirty_prev.items.len;
        if (total > 0) {
            const writes_bytes = @as(usize, write_count) * scratch_write_bytes;
            const buf = try self.stagingAlloc(writes_bytes + self.dirty_prev.items.len * 4);
            const writes: [*]align(1) types.InstanceWrite = @ptrCast(buf.ptr);
            for (self.dirty_cur.items, 0..) |idx, i| {
                writes[i] = .{ .index = idx, .reserved = .{ 0, 0, 0 }, .data = self.instances[idx].data };
            }
            const copies: [*]align(1) u32 = @ptrCast(buf.ptr + writes_bytes);
            for (self.dirty_prev.items) |idx| {
                if (self.instances[idx].dirty_mark == next_frame) continue;
                copies[copy_count] = idx;
                copy_count += 1;
            }
            const used = writes_bytes + @as(usize, copy_count) * 4;
            try self.check(self.drv.cuMemcpyHtoDAsync_v2(self.update_scratch, buf.ptr, used, self.stream), "cuMemcpyHtoDAsync");
        }

        self.frame = next_frame;
        self.parity ^= 1;
        self.time_prev = self.time;
        self.origin_prev = self.origin;
        if (info) |fi| {
            self.time = fi.time;
            self.origin = fi.origin;
        }

        if (write_count + copy_count > 0) {
            var cur = self.instance_buf[self.parity];
            var prev = self.instance_buf[self.parity ^ 1];
            var writes_ptr: u64 = self.update_scratch;
            var wc = write_count;
            var copies_ptr: u64 = self.update_scratch + @as(u64, write_count) * scratch_write_bytes;
            var cc = copy_count;
            const params = [_]?*anyopaque{ @ptrCast(&cur), @ptrCast(&prev), @ptrCast(&writes_ptr), @ptrCast(&wc), @ptrCast(&copies_ptr), @ptrCast(&cc) };
            const n = write_count + copy_count;
            try self.launch(self.fn_update, .{ (n + types.update_block - 1) / types.update_block, 1, 1 }, .{ types.update_block, 1, 1 }, &params);
        }

        // Skelette: Knochen auf der GPU direkt in die Instanzen des neuen Frames
        const animated = try self.anim.apply(self, self.instance_buf[self.parity], self.time);

        std.mem.swap(std.ArrayList(u32), &self.dirty_cur, &self.dirty_prev);
        self.dirty_cur.clearRetainingCapacity();
        try self.uploadScene();
        if (self.rt) |r| {
            if (write_count > 0 or animated) r.markInstancesDirty(self.instances_structural);
            self.instances_structural = false;
            try r.buildInstances(self, self.instance_buf[self.parity], self.instance_high);
        }
        try self.fenceDeferred();
        try self.releaseDeferred(false);
    }

    // -----------------------------------------------------------------------
    // Ansichten, Rendern, Strahlen
    // -----------------------------------------------------------------------

    fn viewSlot(self: *Context, h: usize) Error!*ViewSlot {
        const d = decodeHandle(h);
        if (h == 0 or d.index >= self.views.len) return fail(error.InvalidHandle, "ungültige Ansicht", .{});
        const s = &self.views[d.index];
        if (!s.alive or s.generation != d.generation) return fail(error.InvalidHandle, "Ansicht existiert nicht mehr", .{});
        return s;
    }

    pub fn viewCreate(self: *Context) Error!usize {
        const idx = allocSlot(&self.view_free, &self.view_high, self.views.len) orelse
            return fail(error.Capacity, "max_views erreicht", .{});
        const s = &self.views[idx];
        const gen = s.generation;
        s.* = .{ .generation = gen, .alive = true };
        return encodeHandle(idx, gen);
    }

    pub fn viewDestroy(self: *Context, h: usize) Error!void {
        const s = try self.viewSlot(h);
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        freePost(&self.drv, s);
        self.freeDlss(s);
        s.alive = false;
        s.generation +%= 1;
        try oom(self.view_free.append(self.gpa, decodeHandle(h).index));
    }

    pub fn viewResetHistory(self: *Context, h: usize) Error!void {
        const s = try self.viewSlot(h);
        s.has_last = false;
        s.has_before = false;
    }

    pub fn render(self: *Context, view: usize, camera: *const types.Camera, targets: *const api.Targets) Error!void {
        const v = try self.viewSlot(view);
        if (camera.width == 0 or camera.height == 0) return fail(error.InvalidArgument, "Kamera ohne Auflösung", .{});
        if (!(camera.scale[0] > 0 and camera.scale[1] > 0)) return fail(error.InvalidArgument, "Kamera: scale muss positiv sein", .{});
        const w2v = xform.inverse(&camera.view_to_world) orelse return fail(error.InvalidArgument, "Kamerapose ist nicht invertierbar", .{});
        const cur = types.CameraData{ .camera = camera.*, .world_to_view = w2v };

        // Kamera des Vorframes bestimmen (mehrfaches Rendern im selben Frame erlaubt)
        if (!(v.has_last and v.last_frame == self.frame)) {
            v.before = v.last;
            v.before_frame = v.last_frame;
            v.has_before = v.has_last;
            v.last_frame = self.frame;
            v.has_last = true;
        }
        v.last = cur;
        const prev_ok = v.has_before and v.before_frame + 1 == self.frame and
            v.before.camera.width == camera.width and v.before.camera.height == camera.height;

        var params = types.RenderParams{
            .scene = self.scene_dev,
            .cur = cur,
            .prev = if (prev_ok) v.before else cur,
            .history_valid = @intFromBool(prev_ok),
            .ray_mask = if (targets.ray_mask == 0) 0xFFFFFFFF else targets.ray_mask,
            .flags = targets.flags,
            .reserved = 0,
            .hits = targets.hits,
            .depth = targets.depth,
            .motion = targets.motion,
            .color = targets.color,
            .normal = targets.normal,
            .albedo = targets.albedo,
            .material = targets.material,
            .frame_index = @truncate(self.frame),
            .transparent_mask = targets.transparent_mask,
            .secondary_mask = targets.secondary_mask,
            .gi = 0,
            .gi_width = 0,
            .gi_height = 0,
        };
        // Indirekte Beleuchtung in halber Auflösung: eigener Durchgang und
        // kantenbewusstes Hochskalieren (braucht color, normal, albedo, hits)
        const half_gi = self.lighting.flags & types.lighting_gi_half != 0 and params.color != 0;
        if (half_gi) {
            if (params.normal == 0 or params.albedo == 0 or params.hits == 0)
                return fail(error.InvalidArgument, "PYR_LIGHTING_GI_HALF braucht color, normal, albedo und hits", .{});
            const gw = (camera.width + 1) / 2;
            const gh = (camera.height + 1) / 2;
            if (v.gi_w != gw or v.gi_h != gh) {
                try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
                if (v.gi_buf != 0) _ = self.drv.cuMemFree_v2(v.gi_buf);
                v.gi_buf = try self.devAlloc(@as(u64, gw) * gh * 8, "indirekte Beleuchtung");
                v.gi_w = gw;
                v.gi_h = gh;
            }
            params.gi = v.gi_buf;
            params.gi_width = gw;
            params.gi_height = gh;
        }

        const bx = types.render_block_x;
        const by = types.render_block_y;
        const block = [3]u32{ bx, by, 1 };
        const params_ptr = [_]?*anyopaque{@ptrCast(&params)};
        if (self.useRt(params.color != 0 or params.normal != 0 or params.albedo != 0)) |r| {
            try r.render(self, &params);
            if (half_gi) try r.renderGi(self, &params);
        } else {
            const grid = [3]u32{ (camera.width + bx - 1) / bx, (camera.height + by - 1) / by, 1 };
            try self.launch(self.fn_render, grid, block, &params_ptr);
            if (half_gi) {
                const gg = [3]u32{ (params.gi_width + bx - 1) / bx, (params.gi_height + by - 1) / by, 1 };
                try self.launch(self.fn_gi, gg, block, &params_ptr);
            }
        }
        if (half_gi) {
            const grid = [3]u32{ (camera.width + bx - 1) / bx, (camera.height + by - 1) / by, 1 };
            try self.launch(self.fn_gi_combine, grid, block, &params_ptr);
        }
    }

    pub fn trace(self: *Context, rays: u64, hits: u64, count: u32, ray_mask: u32, flags: u32) Error!void {
        if (count == 0) return;
        if (rays == 0 or hits == 0) return fail(error.InvalidArgument, "rays und hits müssen Gerätezeiger sein", .{});
        var params = types.TraceParams{
            .scene = self.scene_dev,
            .rays = rays,
            .hits = hits,
            .count = count,
            .ray_mask = if (ray_mask == 0) 0xFFFFFFFF else ray_mask,
            .flags = flags,
            .reserved = 0,
        };
        if (self.useRt(false)) |r| return r.trace(self, &params);
        const params_ptr = [_]?*anyopaque{@ptrCast(&params)};
        const b = types.trace_block;
        try self.launch(self.fn_trace, .{ (count + b - 1) / b, 1, 1 }, .{ b, 1, 1 }, &params_ptr);
    }

    /// Nur Primärstrahlen: unterhalb dieser Instanzzahl ist die CUDA-Traversierung
    /// schneller (gemessen: 3 Instanzen CUDA 1,4x schneller, 64 Instanzen RT 1,5x).
    /// Mit Shading (Schatten-/GI-Strahlen) gewinnen die RT-Cores immer
    /// (3 Instanzen: RT 10,2 ms gegen CUDA 12,2 ms bei 1080p).
    pub const rt_min_instances: u32 = 16;

    fn useRt(self: *Context, shading: bool) ?*Rt {
        const r = self.rt orelse return null;
        return if (self.force_rt or shading or self.instance_high >= rt_min_instances) r else null;
    }

    // -----------------------------------------------------------------------
    // Materialien, Licht
    // -----------------------------------------------------------------------

    pub fn defaultMaterial() types.Material {
        return .{
            .base_color = .{ 1, 1, 1 },
            .roughness = 0.8,
            .emission = .{ 0, 0, 0 },
            .metallic = 0,
            .flags = types.material_voxel_color,
            .opacity = 0,
            .ior = 1.5,
            .density = 0,
            .wave_height = 0.15,
            .wave_length = 12,
            .wave_speed = 0.35,
            .clearcoat = 0,
            .clearcoat_roughness = 0.1,
            .subsurface = 0,
            .subsurface_color = .{ 1, 1, 1 },
            .texture = 0,
            .normal_texture = 0,
            .texture_scale = 1,
            .normal_strength = 0,
            .normal_scale = 1,
            .reserved = 0,
        };
    }

    pub fn defaultLighting() types.Lighting {
        var l = std.mem.zeroes(types.Lighting);
        l.sun_direction = .{ 0.45, 0.8, 0.35 };
        l.sun_angular_radius = 0.02;
        l.sun_color = .{ 2.6, 2.45, 2.2 };
        l.flags = types.lighting_shadows | types.lighting_gi | types.lighting_sun_disk | types.lighting_reflections;
        l.sky_zenith = .{ 0.25, 0.45, 0.9 };
        l.sky_horizon = .{ 0.75, 0.82, 0.95 };
        l.ground_color = .{ 0.3, 0.27, 0.24 };
        l.sky_intensity = 1.0;
        l.ao_radius = 8;
        l.gi_bounces = 1;
        l.fog_color = .{ 1, 1, 1 };
        l.fog_anisotropy = 0.6;
        return l;
    }

    /// Bitmaske der Materialien mit material_transparent (für die Traversierung)
    fn transparentMaterials(self: *const Context) [4]u64 {
        var m = [4]u64{ 0, 0, 0, 0 };
        for (self.materials, 0..) |mat, i| {
            if (mat.flags & types.material_transparent != 0) m[i >> 6] |= @as(u64, 1) << @intCast(i & 63);
        }
        return m;
    }

    pub fn materialSet(self: *Context, index: u32, m: *const types.Material) Error!void {
        if (index >= types.max_materials) return fail(error.InvalidArgument, "Materialindex {d} >= {d}", .{ index, types.max_materials });
        self.materials[index] = m.*;
        try self.uploadValue(self.materials_dev + @as(u64, index) * @sizeOf(types.Material), m);
    }

    /// Textur anlegen: RGBA8, dicht gepackt, `width * height * 4` Bytes.
    /// Der Index geht 1-basiert ins Material (0 = keine Textur).
    pub fn textureCreate(self: *Context, w: u32, h: u32, pixels: []const u8) Error!u32 {
        if (w == 0 or h == 0) return fail(error.InvalidArgument, "Textur braucht Breite und Höhe", .{});
        const need = @as(u64, w) * h * 4;
        if (pixels.len < need) return fail(error.InvalidArgument, "Textur {d}x{d} braucht {d} Bytes, bekommen {d}", .{ w, h, need, pixels.len });
        var slot: u32 = 1;
        while (slot < self.texture_data.len and self.texture_data[slot] != 0) slot += 1;
        if (slot >= self.texture_data.len) return fail(error.Capacity, "keine Texturplätze mehr (max_textures erhöhen)", .{});

        // Verkleinerungsstufen erzeugen (Mittel über je 2x2). Ohne sie
        // flimmert jede Textur in der Ferne, sobald sich etwas bewegt: dann
        // fallen viele Texel auf ein Pixel und es wird jedes Mal ein anderes
        // getroffen.
        var levels: u32 = 1;
        {
            var lw = w;
            var lh = h;
            while (lw > 1 or lh > 1) : (levels += 1) {
                lw = @max(lw / 2, 1);
                lh = @max(lh / 2, 1);
            }
        }
        var total: u64 = 0;
        {
            var lw = w;
            var lh = h;
            var k: u32 = 0;
            while (k < levels) : (k += 1) {
                total += @as(u64, lw) * lh;
                lw = @max(lw / 2, 1);
                lh = @max(lh / 2, 1);
            }
        }
        const chain = try oom(self.gpa.alloc([4]u8, @intCast(total)));
        defer self.gpa.free(chain);
        @memcpy(std.mem.sliceAsBytes(chain[0..@intCast(@as(u64, w) * h)]), pixels[0..@intCast(need)]);
        {
            var src_off: u64 = 0;
            var lw = w;
            var lh = h;
            var k: u32 = 1;
            while (k < levels) : (k += 1) {
                const dst_off = src_off + @as(u64, lw) * lh;
                const nw = @max(lw / 2, 1);
                const nh = @max(lh / 2, 1);
                for (0..nh) |y| {
                    for (0..nw) |x| {
                        var acc: [4]u32 = .{ 0, 0, 0, 0 };
                        var n: u32 = 0;
                        for (0..2) |dy| {
                            for (0..2) |dx| {
                                const sx = @min(x * 2 + dx, lw - 1);
                                const sy = @min(y * 2 + dy, lh - 1);
                                const p4 = chain[@intCast(src_off + @as(u64, sy) * lw + sx)];
                                inline for (0..4) |c| acc[c] += p4[c];
                                n += 1;
                            }
                        }
                        var out: [4]u8 = undefined;
                        inline for (0..4) |c| out[c] = @intCast(acc[c] / n);
                        chain[@intCast(dst_off + @as(u64, y) * nw + x)] = out;
                    }
                }
                src_off = dst_off;
                lw = nw;
                lh = nh;
            }
        }

        const buf = try self.devAlloc(total * 4, "Textur");
        errdefer _ = self.drv.cuMemFree_v2(buf);
        try self.upload(buf, std.mem.sliceAsBytes(chain));
        const entry = types.TextureData{ .data = buf, .width = w, .height = h, .levels = levels, .reserved_tex_data = 0 };
        try self.uploadValue(self.texture_table + @as(u64, slot) * @sizeOf(types.TextureData), &entry);
        self.texture_data[slot] = buf;
        if (slot >= self.texture_high) self.texture_high = slot + 1;
        self.logf(3, "Textur {d}: {d}x{d}, {d} Stufen, {d} KiB", .{ slot, w, h, levels, total * 4 >> 10 });
        return slot;
    }

    pub fn textureDestroy(self: *Context, index: u32) Error!void {
        if (index == 0 or index >= self.texture_data.len or self.texture_data[index] == 0)
            return fail(error.InvalidHandle, "Textur {d} gibt es nicht", .{index});
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        _ = self.drv.cuMemFree_v2(self.texture_data[index]);
        self.texture_data[index] = 0;
        const entry = std.mem.zeroes(types.TextureData);
        try self.uploadValue(self.texture_table + @as(u64, index) * @sizeOf(types.TextureData), &entry);
    }

    pub fn setLighting(self: *Context, l: *const types.Lighting) Error!void {
        if (l.light_count > types.max_lights) return fail(error.InvalidArgument, "höchstens {d} Lichter", .{types.max_lights});
        self.lighting = l.*;
        // Die Umgebungskarte gehört dem Kontext, nicht dem Aufrufer: seine
        // Felder werden überschrieben, damit ein altes Lighting sie nicht
        // versehentlich löscht.
        self.lighting.env_data = self.env_data;
        self.lighting.env_marginal = self.env_marginal;
        self.lighting.env_cond = self.env_cond;
        self.lighting.env_width = self.env_w;
        self.lighting.env_height = self.env_h;
        // Diagnose: ohne Verteilung fällt die Lichtabtastung der Karte weg,
        // es bleibt reines Abtasten über den Cosinus-Lappen (A/B-Vergleich).
        self.lighting.env_total = if (std.c.getenv("PYRIT_ENV_NOMIS") != null) 0 else self.env_total;
        self.lighting.env_mean = self.env_mean;
        if (self.env_data != 0 and self.lighting.env_intensity == 0) self.lighting.env_intensity = 1;
        try self.uploadValue(self.lighting_dev, &self.lighting);
    }

    /// Umgebungskarte setzen: equirektangulär, `w x h`, 4 Floats je Texel
    /// (RGB + ungenutzt). Pyrit baut daraus die Verteilung für das
    /// Importance-Sampling: je Zeile eine Summenfunktion über die Spalten und
    /// eine über die Zeilen, beide mit sin(theta) gewichtet (sonst wären die
    /// Pole überrepräsentiert).
    pub fn setEnvironment(self: *Context, w: u32, h: u32, pixels: []const f32) Error!void {
        self.freeEnv();
        if (w == 0 or h == 0) {
            try self.setLighting(&self.lighting);
            return;
        }
        const need = @as(usize, w) * h * 4;
        if (pixels.len < need) return fail(error.InvalidArgument, "Umgebungskarte {d}x{d} braucht {d} Floats", .{ w, h, need });

        const cond = try oom(self.gpa.alloc(f32, @as(usize, h) * (w + 1)));
        defer self.gpa.free(cond);
        const marginal = try oom(self.gpa.alloc(f32, @as(usize, h) + 1));
        defer self.gpa.free(marginal);
        // Die Verteilung wird über den *Überschuss* über den Mittelwert
        // gebildet, nicht über die Helligkeit selbst. Grund: den gleichmäßigen
        // Teil des Himmels trifft der Cosinus-Strahl der GI bereits perfekt;
        // richtet man die Lichtabtastung auch darauf, landen die meisten
        // Abtastungen dort und werden von MIS anschließend verworfen – der
        // Sonnenbeitrag käme dann nur in jedem vierten Frame, dafür vierfach,
        // und das flimmert sichtbar. Über den Überschuss zielt die
        // Lichtabtastung auf genau die Spitzen, die dem Cosinus-Strahl
        // entgehen; wo der Überschuss 0 ist, übernimmt ihn die GI ganz
        // (MIS bleibt dabei erwartungstreu).
        var mean: f64 = 0;
        for (0..@as(usize, h) * w) |k| {
            const px = pixels[k * 4 ..][0..3];
            mean += 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        }
        mean /= @floatFromInt(@as(usize, h) * w);

        var total: f64 = 0;
        for (0..h) |y| {
            const sin_t = @sin((@as(f64, @floatFromInt(y)) + 0.5) / @as(f64, @floatFromInt(h)) * std.math.pi);
            var row_sum: f64 = 0;
            const row = cond[@as(usize, y) * (w + 1) ..][0 .. w + 1];
            row[0] = 0;
            for (0..w) |x| {
                const px = pixels[(@as(usize, y) * w + x) * 4 ..][0..3];
                const lu = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
                row_sum += @max(@as(f64, lu) - mean, 0) * sin_t;
                row[x + 1] = @floatCast(row_sum);
            }
            // Zeile auf 1 normieren (0 bleibt 0: dunkle Zeilen werden nie gezogen)
            if (row_sum > 0) {
                for (row) |*v| v.* = @floatCast(@as(f64, v.*) / row_sum);
            }
            marginal[y] = @floatCast(total);
            total += row_sum;
        }
        marginal[h] = @floatCast(total);
        if (total > 0) {
            for (marginal) |*v| v.* = @floatCast(@as(f64, v.*) / total);
        }

        self.env_data = try self.devAlloc(need * 4, "Umgebungskarte");
        try self.upload(self.env_data, std.mem.sliceAsBytes(pixels[0..need]));
        self.env_cond = try self.devAlloc(cond.len * 4, "Umgebungskarte: Verteilung");
        try self.upload(self.env_cond, std.mem.sliceAsBytes(cond));
        self.env_marginal = try self.devAlloc(marginal.len * 4, "Umgebungskarte: Verteilung");
        try self.upload(self.env_marginal, std.mem.sliceAsBytes(marginal));
        self.env_w = w;
        self.env_h = h;
        // Mittelwert der gewichteten Helligkeit je Texel (für die Dichte)
        self.env_total = @floatCast(total);
        self.env_mean = @floatCast(mean);
        try self.setLighting(&self.lighting);
    }

    fn freeEnv(self: *Context) void {
        for ([_]*cuda.CUdeviceptr{ &self.env_data, &self.env_cond, &self.env_marginal }) |b| {
            if (b.* != 0) _ = self.drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        self.env_w = 0;
        self.env_h = 0;
        self.env_total = 0;
        self.env_mean = 0;
    }

    // -----------------------------------------------------------------------
    // Nachbearbeitung: temporal (TAA/Akkumulation), À-trous, Tonemapping
    // -----------------------------------------------------------------------

    fn freePost(drv: *const cuda.Driver, v: *ViewSlot) void {
        for (&v.post_buf) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        for (&v.post_mom) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        for (&v.post_var) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        v.post_w = 0;
        v.post_h = 0;
        v.post_valid = false;
        if (v.lr_buf != 0) _ = drv.cuMemFree_v2(v.lr_buf);
        v.lr_buf = 0;
        if (v.gi_buf != 0) _ = drv.cuMemFree_v2(v.gi_buf);
        v.gi_buf = 0;
        for ([_]*cuda.CUdeviceptr{ &v.fg_depth, &v.fg_mv }) |b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        v.gi_w = 0;
        v.gi_h = 0;
        freeUpscale(drv, v);
        freeFx(drv, v);
    }

    /// Puffer (Verlauf 2x, Normale/Tiefe 2x, Filter 2x) passend zur Auflösung
    fn ensurePost(self: *Context, v: *ViewSlot, w: u32, h: u32) Error!void {
        if (v.post_w == w and v.post_h == h) return;
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        freePost(&self.drv, v);
        // interne Puffer halbgenau: halber Speicher, halbe Bandbreite
        for (&v.post_buf) |*b| b.* = try self.devAlloc(@as(u64, w) * h * 8, "Nachbearbeitung");
        // Momente (2 x f16) und Varianz (f16) für die varianzgeführte Filterung
        for (&v.post_mom) |*b| b.* = try self.devAlloc(@as(u64, w) * h * 4, "Nachbearbeitung");
        for (&v.post_var) |*b| b.* = try self.devAlloc(@as(u64, w) * h * 2, "Nachbearbeitung");
        v.post_w = w;
        v.post_h = h;
    }

    // -----------------------------------------------------------------------
    // Kamera- und Bildeffekte (src/device/postfx.zig)
    // -----------------------------------------------------------------------

    fn freeFx(drv: *const cuda.Driver, v: *ViewSlot) void {
        for (&v.fx_buf) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        for (&v.fx_bloom) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        for ([_]*cuda.CUdeviceptr{ &v.fx_mvd, &v.fx_expose }) |b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        v.fx_w = 0;
        v.fx_h = 0;
        v.fx_levels = 0;
    }

    fn ensureFx(self: *Context, v: *ViewSlot, w: u32, h: u32, levels: u32) Error!void {
        if (v.fx_w == w and v.fx_h == h and v.fx_levels == levels) return;
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        freeFx(&self.drv, v);
        const n = @as(u64, w) * h;
        for (&v.fx_buf) |*b| b.* = try self.devAlloc(n * 16, "Bildeffekte");
        v.fx_mvd = try self.devAlloc(n * 16, "Bildeffekte: Bewegung und Tiefe");
        // Belichtung: 2 x u32 Zähler, 2 x f32 Zustand. Der Zustand überlebt
        // Frames, deshalb hier auf 0 setzen (0 = noch keine Messung).
        v.fx_expose = try self.devAlloc(16, "Bildeffekte: Belichtung");
        try self.check(self.drv.cuMemsetD8Async(v.fx_expose, 0, 16, self.stream), "cuMemsetD8Async");
        var lw = @max(w / 2, 1);
        var lh = @max(h / 2, 1);
        var i: u32 = 0;
        while (i < levels and i < v.fx_bloom.len) : (i += 1) {
            v.fx_bloom[i] = try self.devAlloc(@as(u64, lw) * lh * 8, "Bildeffekte: Bloom");
            v.fx_bloom_w[i] = lw;
            v.fx_bloom_h[i] = lh;
            lw = @max(lw / 2, 1);
            lh = @max(lh / 2, 1);
        }
        v.fx_w = w;
        v.fx_h = h;
        v.fx_levels = levels;
    }

    fn fxLaunch(self: *Context, f: cuda.CUfunction, p: *types.PostFxParams, count: u64) Error!void {
        const b = types.postfx_block;
        const grid: u32 = @intCast((count + b - 1) / b);
        const params = [_]?*anyopaque{@ptrCast(p)};
        try self.launch(f, .{ @max(grid, 1), 1, 1 }, .{ b, 1, 1 }, &params);
    }

    /// Die Effektkette auf einem HDR-Bild in Ausgabeauflösung. `hdr` ist die
    /// Quelle (4 x f32); `mvd` liefert Bewegung und Tiefe, oder 0, dann wird
    /// beides aus der Renderauflösung gepackt.
    fn runFx(self: *Context, v: *ViewSlot, in: *const api.Targets, info: *const api.PostInfo, fx: *const api.PostFx, w: u32, h: u32, hdr: cuda.CUdeviceptr, mvd_in: cuda.CUdeviceptr) Error!void {
        const levels: u32 = if (fx.flags & api.postfx_bloom != 0)
            @min(if (fx.bloom_levels == 0) 5 else fx.bloom_levels, v.fx_bloom.len)
        else
            0;
        try self.ensureFx(v, w, h, levels);
        const n = @as(u64, w) * h;

        var p = std.mem.zeroes(types.PostFxParams);
        p.width = w;
        p.height = h;
        p.dst_width = w;
        p.dst_height = h;
        p.tonemap = info.tonemap;
        p.bgra = @intFromBool(info.flags & api.post_bgra != 0);
        p.exposure = if (info.exposure > 0) info.exposure else 1;

        // Bewegung und Tiefe
        var mvd = mvd_in;
        if (mvd == 0) {
            p.mvd = v.fx_mvd;
            p.mv_src = in.motion;
            p.normal_src = in.normal;
            p.dst_width = v.last.camera.width;
            p.dst_height = v.last.camera.height;
            try self.fxLaunch(self.fn_fx_pack_mvd, &p, n);
            p.dst_width = w;
            p.dst_height = h;
            mvd = v.fx_mvd;
        }
        p.mvd = mvd;

        var src = hdr;
        var other = if (hdr == v.fx_buf[0]) v.fx_buf[1] else v.fx_buf[0];

        if (fx.flags & api.postfx_dof != 0) {
            p.dof_autofocus = @intFromBool(fx.flags & api.postfx_autofocus != 0);
            p.dof_far_only = @intFromBool(fx.flags & api.postfx_dof_far_only != 0);
            p.dof_focus = if (fx.focus_distance > 0) fx.focus_distance else 10;
            p.dof_strength = if (fx.dof_strength > 0) fx.dof_strength else 3;
            p.dof_max_coc = if (fx.dof_max_coc > 0) fx.dof_max_coc else 12;
            p.color = src;
            p.dst = other;
            try self.fxLaunch(self.fn_fx_dof, &p, n);
            const t = src;
            src = other;
            other = t;
        }

        if (fx.flags & api.postfx_motion_blur != 0) {
            p.blur_scale = if (fx.motion_blur_scale > 0) fx.motion_blur_scale else 0.5;
            p.blur_max = if (fx.motion_blur_max > 0) fx.motion_blur_max else 64;
            p.blur_samples = if (fx.motion_blur_samples > 0) fx.motion_blur_samples else 12;
            p.color = src;
            p.dst = other;
            try self.fxLaunch(self.fn_fx_motion, &p, n);
            const t = src;
            src = other;
            other = t;
        }

        if (levels > 0) {
            p.bloom_threshold = if (fx.bloom_threshold > 0) fx.bloom_threshold else 1;
            p.bloom_knee = if (fx.bloom_knee > 0) fx.bloom_knee else 0.5;
            // Abschöpfen und halbieren
            p.color = src;
            p.src_half = 0;
            p.width = w;
            p.height = h;
            p.dst = v.fx_bloom[0];
            p.dst_width = v.fx_bloom_w[0];
            p.dst_height = v.fx_bloom_h[0];
            try self.fxLaunch(self.fn_fx_bloom_pre, &p, @as(u64, p.dst_width) * p.dst_height);
            p.src_half = 1;
            var i: u32 = 1;
            while (i < levels) : (i += 1) {
                p.color = v.fx_bloom[i - 1];
                p.width = v.fx_bloom_w[i - 1];
                p.height = v.fx_bloom_h[i - 1];
                p.dst = v.fx_bloom[i];
                p.dst_width = v.fx_bloom_w[i];
                p.dst_height = v.fx_bloom_h[i];
                try self.fxLaunch(self.fn_fx_bloom_down, &p, @as(u64, p.dst_width) * p.dst_height);
            }
            // und wieder hoch, jede Stufe in ihre größere addiert
            i = levels - 1;
            while (i > 0) : (i -= 1) {
                p.color = v.fx_bloom[i];
                p.width = v.fx_bloom_w[i];
                p.height = v.fx_bloom_h[i];
                p.dst = v.fx_bloom[i - 1];
                p.dst_width = v.fx_bloom_w[i - 1];
                p.dst_height = v.fx_bloom_h[i - 1];
                try self.fxLaunch(self.fn_fx_bloom_up, &p, @as(u64, p.dst_width) * p.dst_height);
            }
            p.bloom = v.fx_bloom[0];
            p.bloom_width = v.fx_bloom_w[0];
            p.bloom_height = v.fx_bloom_h[0];
            p.bloom_strength = if (fx.bloom_strength > 0) fx.bloom_strength else 0.05;
            p.width = w;
            p.height = h;
            p.dst_width = w;
            p.dst_height = h;
            p.src_half = 0;
        }

        if (fx.flags & api.postfx_auto_exposure != 0) {
            p.expose_acc = v.fx_expose;
            p.expose_state = v.fx_expose + 8;
            p.expose_speed = if (fx.exposure_speed > 0) fx.exposure_speed else 0.05;
            p.expose_min = if (fx.exposure_min > 0) fx.exposure_min else 0.03;
            p.expose_max = if (fx.exposure_max > 0) fx.exposure_max else 30;
            p.expose_compensation = fx.exposure_compensation;
            p.color = src;
            const step: u64 = 4;
            const sw = (@as(u64, w) + step - 1) / step;
            const sh = (@as(u64, h) + step - 1) / step;
            try self.fxLaunch(self.fn_fx_expose_scan, &p, sw * sh);
            try self.fxLaunch(self.fn_fx_expose_apply, &p, 1);
        }

        if (fx.flags & api.postfx_grade != 0) {
            p.grade = 1;
            p.temperature = fx.temperature;
            p.tint = fx.tint;
            p.contrast = if (fx.contrast > 0) fx.contrast else 1;
            p.saturation = if (fx.saturation > 0) fx.saturation else 1;
            p.lift = fx.lift;
            p.gamma = .{
                if (fx.gamma[0] > 0) fx.gamma[0] else 1,
                if (fx.gamma[1] > 0) fx.gamma[1] else 1,
                if (fx.gamma[2] > 0) fx.gamma[2] else 1,
            };
            p.gain = .{
                if (fx.gain[0] > 0) fx.gain[0] else 1,
                if (fx.gain[1] > 0) fx.gain[1] else 1,
                if (fx.gain[2] > 0) fx.gain[2] else 1,
            };
        }
        p.lut = fx.lut;
        p.lut_size = fx.lut_size;
        p.color = src;
        p.out_hdr = info.output_hdr;
        p.out_ldr = info.output_ldr;
        try self.fxLaunch(self.fn_fx_resolve, &p, n);
    }

    fn freeDlss(self: *Context, v: *ViewSlot) void {
        if (v.dlss_feature) |f| f.destroy(self);
        v.dlss_feature = null;
        for ([_]*cuda.CUdeviceptr{ &v.depth_buf, &v.rough_buf, &v.spec_buf }) |b| {
            if (b.* != 0) _ = self.drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
    }

    fn freeUpscale(drv: *const cuda.Driver, v: *ViewSlot) void {
        for (&v.up_buf) |*b| {
            if (b.* != 0) _ = drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
        v.up_w = 0;
        v.up_h = 0;
        v.up_valid = false;
        v.up_frames = 0;
    }

    pub fn postprocess(self: *Context, view: usize, in: *const api.Targets, info: *const api.PostInfo) Error!void {
        // Nachbearbeitung auf einem eigenen Stream: sie hängt nur am Rendern
        // *dieses* Frames, nicht am nächsten. DLSS und TAA laufen dann auf den
        // Tensorkernen, während die Shader-Einheiten schon den nächsten Frame
        // rechnen. Ohne PYR_CREATE_ASYNC_POST wartet der nächste Frame
        // trotzdem, weil er sonst in dieselben Ziele schreiben würde.
        if (self.post_stream != null) {
            try self.check(self.drv.cuEventRecord(self.render_done, self.stream), "cuEventRecord");
            try self.check(self.drv.cuStreamWaitEvent(self.post_stream, self.render_done, 0), "cuStreamWaitEvent");
            self.on_post = true;
        }
        defer if (self.on_post) {
            self.on_post = false;
            _ = self.drv.cuEventRecord(self.post_done, self.post_stream);
            self.post_pending = true;
            // Ohne eigene Zielsätze muss der nächste Frame warten: sonst
            // überschreibt er die Ziele, aus denen hier noch gelesen wird.
            if (!self.async_post) _ = self.drv.cuStreamWaitEvent(self.stream, self.post_done, 0);
        };
        const v = try self.viewSlot(view);
        if (!v.has_last) return fail(error.InvalidArgument, "pyr_postprocess vor dem ersten pyr_render dieser Ansicht", .{});
        if (in.color == 0 or in.normal == 0 or in.albedo == 0 or in.motion == 0 or in.hits == 0)
            return fail(error.InvalidArgument, "pyr_postprocess braucht color, normal, albedo, motion und hits aus pyr_render", .{});
        const w = v.last.camera.width;
        const h = v.last.camera.height;
        const out_w = if (info.output_width == 0) w else info.output_width;
        const out_h = if (info.output_height == 0) h else info.output_height;
        const mode: u32 = if (info.upscaler == api.upscaler_auto) api.upscaler_taau else info.upscaler;
        switch (mode) {
            api.upscaler_none => if (out_w != w or out_h != h)
                return fail(error.InvalidArgument, "Hochskalieren ({d}x{d} -> {d}x{d}) braucht einen Upscaler", .{ w, h, out_w, out_h }),
            api.upscaler_taau => if (out_w < w or out_h < h or out_w > 4 * w or out_h > 4 * h)
                return fail(error.InvalidArgument, "Ausgabe {d}x{d} muss zwischen 1x und 4x der Renderauflösung {d}x{d} liegen", .{ out_w, out_h, w, h }),
            api.upscaler_dlss, api.upscaler_dlss_rr => return self.postprocessDlss(v, view, in, info, out_w, out_h, mode == api.upscaler_dlss_rr),
            else => return fail(error.InvalidArgument, "unbekannter Upscaler {d}", .{mode}),
        }

        try self.ensurePost(v, w, h);
        const cur = v.post_parity;
        const prev = cur ^ 1;
        const temporal_ok = v.post_valid and v.post_frame + 1 == v.last_frame and info.flags & (api.post_reset | api.post_no_temporal) == 0;

        var p = std.mem.zeroes(types.PostParams);
        p.width = w;
        p.height = h;
        p.color = in.color;
        p.normal = in.normal;
        p.albedo = in.albedo;
        p.motion = in.motion;
        p.hits = in.hits;
        p.hist_color = v.post_buf[prev];
        p.hist_normal = v.post_buf[2 + @as(usize, prev)];
        p.out_color = v.post_buf[cur];
        p.out_normal = v.post_buf[2 + @as(usize, cur)];
        // Diagnose: PYRIT_POST_NOVAR=1 schaltet auf die alte Schätzung über die
        // Framezahl zurück (A/B-Vergleich des Rauschens).
        if (std.c.getenv("PYRIT_POST_NOVAR") == null) {
            p.hist_moments = v.post_mom[prev];
            p.out_moments = v.post_mom[cur];
            p.out_var = v.post_var[0];
        }
        p.phi_lum = if (info.denoise_phi > 0) info.denoise_phi else 4.0;
        // Diagnose-Schalter; die Vorgaben sind gemessen (siehe post.zig)
        p.normal_reject = if (std.c.getenv("PYRIT_POST_NORMREJ")) |e| (std.fmt.parseFloat(f32, std.mem.span(e)) catch -1) else -1;
        p.edge_frames = if (std.c.getenv("PYRIT_POST_EDGEFRAMES")) |e| (std.fmt.parseFloat(f32, std.mem.span(e)) catch 4) else 4;
        p.alpha_min = if (info.temporal_alpha > 0) info.temporal_alpha else 0.05;
        p.clamp_sigma = info.clamp_sigma;
        p.reset = @intFromBool(!temporal_ok);
        p.exposure = if (info.exposure > 0) info.exposure else 1;
        p.tonemap = info.tonemap;
        p.bgra = @intFromBool(info.flags & api.post_bgra != 0);
        p.out_hdr = info.output_hdr;
        p.out_ldr = info.output_ldr;

        const b = types.post_block;
        const grid = [3]u32{ (w + b - 1) / b, (h + b - 1) / b, 1 };
        const block = [3]u32{ b, b, 1 };
        const params_ptr = [_]?*anyopaque{@ptrCast(&p)};
        try self.launch(self.fn_temporal, grid, block, &params_ptr);

        // Diagnose: PYRIT_POST_STATS=1 meldet, wie viele Frames der Verlauf je
        // Pixel im Mittel hält. Bricht der Wert bei Bewegung ein, scheitert die
        // Reprojektion – dann hilft kein Filter, sondern nur deren Korrektur.
        if (std.c.getenv("PYRIT_POST_STATS") != null) try self.reportHistory(v, w, h, cur);

        // Der ungefilterte Verlauf bleibt für den nächsten Frame; gefiltert wird eine Kopie
        var src = v.post_buf[cur];
        var vsrc = v.post_var[0];
        var it: u32 = 0;
        while (it < @min(info.denoise_iterations, 8)) : (it += 1) {
            p.src = src;
            p.dst = v.post_buf[4 + @as(usize, it & 1)];
            p.var_src = if (p.out_moments != 0) vsrc else 0;
            p.var_dst = if (p.out_moments != 0) v.post_var[1 + @as(usize, it & 1)] else 0;
            p.step = @as(u32, 1) << @intCast(it);
            try self.launch(self.fn_atrous, grid, block, &params_ptr);
            src = p.dst;
            vsrc = p.var_dst;
        }
        p.var_src = 0;
        p.var_dst = 0;
        p.src = src;
        const fx: ?*const api.PostFx = if (info.fx) |f| (if (f.flags != 0) f else null) else null;
        if (mode == api.upscaler_none) {
            if (fx) |f| {
                // Erst HDR in den Effektpuffer, dann die Kette; sie schreibt
                // die endgültige Ausgabe.
                try self.ensureFx(v, w, h, if (f.flags & api.postfx_bloom != 0) @min(if (f.bloom_levels == 0) 5 else f.bloom_levels, 8) else 0);
                p.out_hdr = v.fx_buf[0];
                p.hdr_half = 0;
                p.out_ldr = 0;
                try self.launch(self.fn_resolve, grid, block, &params_ptr);
                try self.runFx(v, in, info, f, w, h, v.fx_buf[0], 0);
            } else {
                try self.launch(self.fn_resolve, grid, block, &params_ptr);
            }
        } else {
            // entrauschte HDR-Farbe in Renderauflösung, dann TAAU in Ausgabeauflösung
            if (v.lr_buf == 0) v.lr_buf = try self.devAlloc(@as(u64, w) * h * 8, "Nachbearbeitung");
            p.out_hdr = v.lr_buf;
            p.hdr_half = 1;
            p.out_ldr = 0;
            try self.launch(self.fn_resolve, grid, block, &params_ptr);
            try self.upscale(v, in, info, out_w, out_h, temporal_ok, fx);
        }

        v.post_parity = prev;
        v.post_valid = true;
        v.post_frame = v.last_frame;
    }

    /// Mittlerer Akkumulationszähler des Verlaufs (Diagnose, langsam)
    fn reportHistory(self: *Context, v: *ViewSlot, w: u32, h: u32, cur: u1) Error!void {
        const n: usize = @as(usize, w) * h;
        const buf = self.gpa.alloc([4]f16, n) catch return;
        defer self.gpa.free(buf);
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        try self.check(self.drv.cuMemcpyDtoH_v2(buf.ptr, v.post_buf[cur], n * 8), "cuMemcpyDtoH");
        var sum: f64 = 0;
        var fresh: u64 = 0;
        for (buf) |c| {
            const k: f64 = @floatCast(c[3]);
            sum += k;
            if (k < 2.5) fresh += 1;
        }
        const pct = 100.0 * @as(f64, @floatFromInt(fresh)) / @as(f64, @floatFromInt(n));
        std.debug.print("Verlauf: im Mittel {d:.1} Frames, {d:.1} % frisch (< 3)\n", .{ sum / @as(f64, @floatFromInt(n)), pct });
    }

    /// DLSS: SR hinter dem eigenen Denoiser, RR ersetzt Denoiser und TAA
    fn postprocessDlss(self: *Context, v: *ViewSlot, view: usize, in: *const api.Targets, info: *const api.PostInfo, out_w: u32, out_h: u32, rr: bool) Error!void {
        if (!dlss.available) return fail(error.NotFound, "DLSS ist in diesem Build nicht verfügbar (mit -Ddlss-sdk bauen)", .{});
        const w = v.last.camera.width;
        const h = v.last.camera.height;
        if (out_w < w or out_h < h) return fail(error.InvalidArgument, "DLSS: Ausgabe kleiner als die Renderauflösung", .{});
        const ngx = self.ngx orelse blk: {
            self.ngx = try dlss.Ngx.init(self.gpa);
            break :blk self.ngx.?;
        };
        const mode: dlss.Mode = if (rr) .ray_reconstruction else .super_resolution;
        var fresh = false;
        if (v.dlss_feature) |f| {
            if (f.mode != mode or f.in_w != w or f.in_h != h or f.out_w != out_w or f.out_h != out_h) {
                try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
                self.freeDlss(v);
            }
        }
        if (v.dlss_feature == null) {
            // SR bekommt die entrauschte Farbe halbgenau, RR die rohe (f32)
            v.dlss_feature = try dlss.Feature.create(ngx, self, mode, w, h, out_w, out_h, !rr);
            fresh = true;
        }
        const f = v.dlss_feature.?;

        // Eingänge: Tiefe als f32, Farbe (RR: verrauscht, SR: eigener Denoiser)
        const n = w * h;
        if (v.depth_buf == 0) {
            v.depth_buf = try self.devAlloc(@as(u64, n) * 4, "DLSS-Tiefe");
            v.rough_buf = try self.devAlloc(@as(u64, n) * 4, "DLSS-Rauheit");
            v.spec_buf = try self.devAlloc(@as(u64, n) * 16, "DLSS-Spiegelalbedo");
        }
        {
            var np = in.normal;
            var ap = in.albedo;
            var mp = in.material;
            var have: u32 = @intFromBool(in.material != 0);
            var dp = v.depth_buf;
            var rp = v.rough_buf;
            var sp = v.spec_buf;
            var cnt = n;
            const params = [_]?*anyopaque{ @ptrCast(&np), @ptrCast(&ap), @ptrCast(&mp), @ptrCast(&have), @ptrCast(&dp), @ptrCast(&rp), @ptrCast(&sp), @ptrCast(&cnt) };
            try self.launch(self.fn_dlss_prepare, .{ (n + types.update_block - 1) / types.update_block, 1, 1 }, .{ types.update_block, 1, 1 }, &params);
        }
        var color = in.color;
        if (!rr) {
            // eigener temporaler Denoiser + À-trous bis zur HDR-Farbe in Renderauflösung
            var mod = info.*;
            mod.upscaler = api.upscaler_none;
            mod.output_width = 0;
            mod.output_height = 0;
            try self.ensurePost(v, w, h);
            if (v.lr_buf == 0) v.lr_buf = try self.devAlloc(@as(u64, n) * 8, "Nachbearbeitung");
            mod.output_hdr = v.lr_buf;
            mod.output_ldr = 0;
            try self.postprocess(view, in, &mod);
            color = v.lr_buf;
        }
        const history_ok = !fresh and v.up_valid and v.up_frame + 1 == v.last_frame and info.flags & api.post_reset == 0;
        var inputs = dlss.Inputs{
            .color = color,
            .albedo = in.albedo,
            .normal = in.normal,
            .depth = v.depth_buf,
            .roughness = v.rough_buf,
            .specular = v.spec_buf,
            .motion = in.motion,
            .jitter = v.last.camera.jitter,
            .reset = !history_ok,
            .world_to_view = undefined,
            .view_to_clip = undefined,
        };
        cameraMatrices(&v.last, &inputs.world_to_view, &inputs.view_to_clip);
        try f.evaluate(ngx, self, &inputs);

        // Ausgabe, Verlauf und MVs für die Frame Generation
        if (v.up_w != out_w or v.up_h != out_h) {
            try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
            freeUpscale(&self.drv, v);
            v.up_buf[0] = try self.devAlloc(@as(u64, out_w) * out_h * 8, "TAAU");
            v.up_buf[1] = try self.devAlloc(@as(u64, out_w) * out_h * 8, "TAAU");
            v.up_buf[2] = try self.devAlloc(@as(u64, out_w) * out_h * 16, "TAAU");
            v.up_w = out_w;
            v.up_h = out_h;
        }
        const cur = v.up_parity;
        var u = std.mem.zeroes(types.UpscaleParams);
        u.in_width = w;
        u.in_height = h;
        u.out_width = out_w;
        u.out_height = out_h;
        u.color = f.output;
        u.normal = in.normal;
        u.motion = in.motion;
        u.jitter = v.last.camera.jitter;
        u.hist_out = v.up_buf[cur];
        u.mvd_out = v.up_buf[2];
        u.exposure = if (info.exposure > 0) info.exposure else 1;
        u.tonemap = info.tonemap;
        u.bgra = @intFromBool(info.flags & api.post_bgra != 0);
        // 2,5 statt der früheren 1,25: gemessen weniger Unruhe an Voxelkanten
        // (stärkste Sprünge 10,3 -> 9,4 Stufen) und bei Bewegung sogar minimal
        // schärfer (8,64 -> 8,80). Enger zu begrenzen kostet also nur.
        u.clamp_sigma = if (std.c.getenv("PYRIT_TAAU_CLAMP")) |e| (std.fmt.parseFloat(f32, std.mem.span(e)) catch 2.5) else 2.5;
        u.out_hdr = info.output_hdr;
        u.out_ldr = info.output_ldr;
        const b = types.upscale_block;
        const params_ptr = [_]?*anyopaque{@ptrCast(&u)};
        try self.launch(self.fn_present, .{ (out_w + b - 1) / b, (out_h + b - 1) / b, 1 }, .{ b, b, 1 }, &params_ptr);
        v.up_parity = cur ^ 1;
        v.up_frames = if (history_ok) v.up_frames + 1 else 1;
        v.up_valid = true;
        v.up_frame = v.last_frame;
        v.up_exposure = u.exposure;
        v.up_tonemap = u.tonemap;
    }

    /// Kamera als 4x4 (Zeilenvektoren, D3D-Konvention wie in den NGX-Beispielen)
    fn cameraMatrices(cd: *const types.CameraData, w2v: *[16]f32, v2c: *[16]f32) void {
        const m = cd.world_to_view;
        // Spaltenvektor-Form M (3x4) -> Zeilenvektor-Form M^T (4x4)
        w2v.* = .{ m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, m[3], m[7], m[11], 1 };
        const cam = &cd.camera;
        const sx = 1.0 / cam.scale[0];
        const sy = 1.0 / cam.scale[1];
        const near = @max(cam.near_plane, 1e-3);
        // rechtshändig, Blick entlang -z, Tiefe umgekehrt unendlich (z_ndc = near / -z)
        v2c.* = .{ sx, 0, 0, 0, 0, sy, 0, 0, cam.shift[0], cam.shift[1], 0, -1, 0, 0, near, 0 };
    }

    fn upscale(self: *Context, v: *ViewSlot, in: *const api.Targets, info: *const api.PostInfo, out_w: u32, out_h: u32, history_ok: bool, fx: ?*const api.PostFx) Error!void {
        const w = v.last.camera.width;
        const h = v.last.camera.height;
        if (v.up_w != out_w or v.up_h != out_h) {
            try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
            freeUpscale(&self.drv, v);
            // Verlauf halbgenau, Bewegung + Tiefe voll genau
            v.up_buf[0] = try self.devAlloc(@as(u64, out_w) * out_h * 8, "TAAU");
            v.up_buf[1] = try self.devAlloc(@as(u64, out_w) * out_h * 8, "TAAU");
            v.up_buf[2] = try self.devAlloc(@as(u64, out_w) * out_h * 16, "TAAU");
            v.up_w = out_w;
            v.up_h = out_h;
        }
        const ok = history_ok and v.up_valid and v.up_frame + 1 == v.last_frame and info.flags & api.post_reset == 0;
        const cur = v.up_parity;
        var u = std.mem.zeroes(types.UpscaleParams);
        u.in_width = w;
        u.in_height = h;
        u.out_width = out_w;
        u.out_height = out_h;
        u.color = v.lr_buf;
        u.color_half = 1;
        u.normal = in.normal;
        u.motion = in.motion;
        u.hits = in.hits;
        u.jitter = v.last.camera.jitter;
        u.hist_in = v.up_buf[cur ^ 1];
        u.hist_out = v.up_buf[cur];
        u.mvd_out = v.up_buf[2];
        u.reset = @intFromBool(!ok);
        u.max_weight = 1.0 / (if (info.temporal_alpha > 0) info.temporal_alpha else 0.1);
        u.exposure = if (info.exposure > 0) info.exposure else 1;
        u.tonemap = info.tonemap;
        u.bgra = @intFromBool(info.flags & api.post_bgra != 0);
        // 2,5 statt der früheren 1,25: gemessen weniger Unruhe an Voxelkanten
        // (stärkste Sprünge 10,3 -> 9,4 Stufen) und bei Bewegung sogar minimal
        // schärfer (8,64 -> 8,80). Enger zu begrenzen kostet also nur.
        u.clamp_sigma = if (std.c.getenv("PYRIT_TAAU_CLAMP")) |e| (std.fmt.parseFloat(f32, std.mem.span(e)) catch 2.5) else 2.5;
        u.out_hdr = info.output_hdr;
        u.out_ldr = info.output_ldr;
        if (fx) |f| {
            // TAAU liefert HDR und (in mvd_out) Bewegung und Tiefe in
            // Ausgabeauflösung – genau, was die Effektkette braucht.
            try self.ensureFx(v, out_w, out_h, if (f.flags & api.postfx_bloom != 0) @min(if (f.bloom_levels == 0) 5 else f.bloom_levels, 8) else 0);
            u.out_hdr = v.fx_buf[0];
            u.out_ldr = 0;
        }
        const b = types.upscale_block;
        const params_ptr = [_]?*anyopaque{@ptrCast(&u)};
        try self.launch(self.fn_taau, .{ (out_w + b - 1) / b, (out_h + b - 1) / b, 1 }, .{ b, b, 1 }, &params_ptr);
        if (fx) |f| try self.runFx(v, in, info, f, out_w, out_h, v.fx_buf[0], v.up_buf[2]);
        v.up_parity = cur ^ 1;
        v.up_frames = if (ok) v.up_frames + 1 else 1;
        v.up_valid = true;
        v.up_frame = v.last_frame;
        v.up_exposure = u.exposure;
        v.up_tonemap = u.tonemap;
        v.up_bgra = u.bgra;
    }

    /// Zwischenbild zwischen den letzten beiden TAAU-Ausgaben
    pub fn frameGenerate(self: *Context, view: usize, info: *const api.FrameGenInfo) Error!void {
        // gehört zur Ausgabekette, läuft also auf demselben Stream wie die
        // Nachbearbeitung und überlappt mit dem nächsten Frame
        if (self.post_stream != null) self.on_post = true;
        defer if (self.on_post) {
            self.on_post = false;
            _ = self.drv.cuEventRecord(self.post_done, self.post_stream);
            self.post_pending = true;
            if (!self.async_post) _ = self.drv.cuStreamWaitEvent(self.stream, self.post_done, 0);
        };
        const v = try self.viewSlot(view);
        if (!v.up_valid or v.up_frames < 2)
            return fail(error.InvalidArgument, "Frame Generation braucht zwei aufeinanderfolgende pyr_postprocess mit TAAU", .{});
        var f = std.mem.zeroes(types.FrameGenParams);
        f.width = v.up_w;
        f.height = v.up_h;
        f.t = if (info.t > 0) @min(info.t, 1) else 0.5;
        f.exposure = v.up_exposure;
        f.tonemap = v.up_tonemap;
        // zuletzt geschrieben: up_buf[parity ^ 1]
        f.cur_color = v.up_buf[v.up_parity ^ 1];
        f.prev_color = v.up_buf[v.up_parity];
        f.motion_depth = v.up_buf[2];
        f.out_hdr = info.output_hdr;
        f.out_ldr = info.output_ldr;
        f.bgra = @intFromBool(info.flags & api.post_bgra != 0);
        f.user = @intFromPtr(info.user);
        if (f.out_hdr == 0 and f.out_ldr == 0) return fail(error.InvalidArgument, "keine Ausgabe für die Frame Generation", .{});
        // Vorwärtsprojektion der Bewegungsvektoren ins Zwischenbild
        const n = @as(u64, f.width) * f.height;
        if (v.fg_depth == 0) {
            v.fg_depth = try self.devAlloc(n * 4, "Frame Generation");
            v.fg_mv = try self.devAlloc(n * 8, "Frame Generation");
        }
        f.depth_mid = v.fg_depth;
        f.mv_mid = v.fg_mv;
        if (info.generate) |gen| {
            gen(info.user, &f, @ptrCast(self.stream));
            return;
        }
        const b = types.upscale_block;
        const grid = [3]u32{ (f.width + b - 1) / b, (f.height + b - 1) / b, 1 };
        const block = [3]u32{ b, b, 1 };
        const params_ptr = [_]?*anyopaque{@ptrCast(&f)};
        try self.check(self.drv.cuMemsetD8Async(v.fg_depth, 0xFF, n * 4, self.stream), "cuMemsetD8Async");
        try self.launch(self.fn_fg_splat, grid, block, &params_ptr);
        try self.launch(self.fn_fg_splat_mv, grid, block, &params_ptr);
        try self.launch(self.fn_framegen, grid, block, &params_ptr);
    }

    pub fn stats(self: *const Context) api.Stats {
        var live_instances: u32 = 0;
        for (self.instances[0..self.instance_high]) |i| live_instances += @intFromBool(i.alive);
        var live_geometries: u32 = 0;
        for (self.geometries[0..self.geometry_high]) |g| live_geometries += @intFromBool(g.alive);
        return .{
            .node_pool_used = self.node_alloc.used * 4,
            .node_pool_capacity = self.node_alloc.capacity * 4,
            .leaf_pool_used = self.leaf_alloc.used * 8,
            .leaf_pool_capacity = self.leaf_alloc.capacity * 8,
            .attribute_pool_used = self.attr_alloc.used * 4,
            .attribute_pool_capacity = self.attr_alloc.capacity * 4,
            .geometries = live_geometries,
            .instances = live_instances,
            .frame = self.frame,
            .features = self.features(),
            .reserved = 0,
        };
    }

    pub fn features(self: *const Context) u32 {
        return if (self.rt != null) api.feature_rt_cores else 0;
    }

    pub fn synchronize(self: *Context) Error!void {
        try self.check(self.drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        if (self.post_pending) {
            try self.check(self.drv.cuStreamSynchronize(self.post_stream), "cuStreamSynchronize");
            self.post_pending = false;
        }
        try self.releaseDeferred(false);
    }
};

test "Handles" {
    const h = encodeHandle(5, 7);
    const d = decodeHandle(h);
    try std.testing.expectEqual(@as(u32, 5), d.index);
    try std.testing.expectEqual(@as(u32, 7), d.generation);
    try std.testing.expect(encodeHandle(0, 0) != 0);
}
