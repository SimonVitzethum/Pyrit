//! RT-Backend: Strahlverfolgung auf den RT-Cores über OptiX.
//!
//! - GAS pro Geometrie: AABBs der belegten Teilbäume (rt_prims.zig), gebaut mit
//!   PREFER_FAST_TRACE und kompaktiert.
//! - IAS pro Frame: auf der GPU aus dem Instanzzustand erzeugt
//!   (pyr_k_build_rt_instances), ohne Umweg über den Host.
//! - Pipeline: OptiX-Programme aus Zig (src/rt_kernels.zig), eine Hitgroup pro
//!   Geometrie im SBT, damit der Intersection-Shader seine Daten mit einem
//!   Zugriff hat.

const std = @import("std");
const types = @import("pyrit_device").types;
const optix = @import("optix.zig");
const cuda = @import("cuda.zig");
const diag = @import("diag.zig");
const dag_builder = @import("dag_builder.zig");
const rt_prims = @import("rt_prims.zig");
const Context = @import("context.zig").Context;

const Error = diag.Error;
const fail = diag.fail;

/// PTX der OptiX-Programme (src/rt_kernels.zig), beim Bauen erzeugt
const rt_ptx: []const u8 = @embedFile("pyrit_rt_ptx");

const hit_record_size = @sizeOf(types.RtHitRecord);

comptime {
    std.debug.assert(@sizeOf(types.RtInstance) == 80);
    std.debug.assert(hit_record_size % optix.sbt_record_alignment == 0);
    std.debug.assert(@sizeOf(rt_prims.Aabb) == @sizeOf(optix.Aabb));
}

pub const GasSlot = struct {
    buffer: cuda.CUdeviceptr = 0,
    /// eigene Primitive (werden mit dem Slot freigegeben; 0 bei Batch-Chunks)
    prims: cuda.CUdeviceptr = 0,
    /// Primitive, auf die der SBT-Eintrag zeigt (auch die einer Batch)
    sbt_prims: cuda.CUdeviceptr = 0,
    /// dauerhafte Hüllen der Primitive (Batch), 0 = nicht aufbewahrt
    sbt_aabbs: cuda.CUdeviceptr = 0,
    handle: optix.TraversableHandle = 0,
    prim_count: u32 = 0,
};

pub const Rt = struct {
    api: optix.Api,
    ctx: optix.DeviceContext = null,
    rtcore_version: u32 = 0,
    rt_log2: u32,

    /// Nur Traversierung und Strahllisten; die Schattierung läuft in CUDA
    /// (src/device/replay.zig), sonst braucht OptiX zum Übersetzen Minuten
    /// und GB.
    module: optix.Module = null,
    groups: [4]optix.ProgramGroup = .{ null, null, null, null },
    pipeline: optix.Pipeline = null,

    sbt_raygen: [2]cuda.CUdeviceptr = .{ 0, 0 },
    /// Strahlplätze der Wiederholungs-Wavefront (je Streifen)
    replay: ReplayBuffers = .{},
    sbt_miss: cuda.CUdeviceptr = 0,
    sbt_hit: cuda.CUdeviceptr = 0,
    hit_header: [optix.sbt_record_header_size]u8 align(16) = undefined,
    params_dev: cuda.CUdeviceptr = 0,

    gas: []GasSlot = &.{},
    gas_handles: cuda.CUdeviceptr = 0,
    ias_instances: cuda.CUdeviceptr = 0,
    ias_temp: cuda.CUdeviceptr = 0,
    ias_temp_size: usize = 0,
    ias_out: cuda.CUdeviceptr = 0,
    ias_out_size: usize = 0,
    ias_handle: optix.TraversableHandle = 0,
    ias_dirty: bool = true,
    /// Struktur geändert (Anzahl, Geometrien, Masken): voller Neubau nötig
    ias_structural: bool = true,
    ias_count: u32 = 0,
    ias_updates: u32 = 0,

    const g_raygen_trace = 0;
    const g_raygen_slots = 1;
    const g_miss = 2;
    const g_hit = 3;

    fn check(self: *Rt, r: optix.Result, what: []const u8) Error!void {
        if (r == optix.success) return;
        return fail(error.Cuda, "OptiX {s}: {s} ({d})", .{ what, self.api.errorString(r), r });
    }

    fn logCallback(level: c_uint, tag: [*:0]const u8, message: [*:0]const u8, data: ?*anyopaque) callconv(.c) void {
        const c: *Context = @ptrCast(@alignCast(data orelse return));
        c.logf(@intCast(level), "OptiX [{s}]: {s}", .{ tag, message });
    }

    /// Richtet OptiX ein. error.NotFound, wenn OptiX oder RT-Cores fehlen.
    pub fn init(c: *Context, rt_log2: u32) Error!*Rt {
        const api = optix.Api.load() catch return fail(error.NotFound, "OptiX (libnvoptix, ABI {d}) nicht verfügbar", .{optix.abi_version});
        const self = c.gpa.create(Rt) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        self.* = .{ .api = api, .rt_log2 = if (rt_log2 == 0) rt_prims.default_log2 else rt_log2 };
        self.setup(c) catch |e| {
            self.deinit(c);
            return e;
        };
        return self;
    }

    fn setup(self: *Rt, c: *Context) Error!void {
        const ft = &self.api.ft;
        const opts = optix.DeviceContextOptions{
            .logCallbackFunction = logCallback,
            .logCallbackData = c,
            .logCallbackLevel = if (c.debug) 4 else 2,
            .validationMode = if (c.debug) optix.validation_mode_all else optix.validation_mode_off,
        };
        try self.check(ft.optixDeviceContextCreate(c.cu_ctx, &opts, &self.ctx), "optixDeviceContextCreate");
        try self.check(ft.optixDeviceContextGetProperty(self.ctx, optix.device_property_rtcore_version, &self.rtcore_version, 4), "RTCORE_VERSION");
        if (self.rtcore_version == 0) return fail(error.NotFound, "GPU ohne RT-Cores", .{});

        try self.createPipeline(c);

        // SBT: zwei Raygen-Einträge, ein Miss-Eintrag, eine Hitgroup pro Geometrie
        var header: [optix.sbt_record_header_size]u8 align(16) = undefined;
        for ([_]usize{ g_raygen_trace, g_raygen_slots }, 0..) |g, i| {
            try self.check(ft.optixSbtRecordPackHeader(self.groups[g], &header), "optixSbtRecordPackHeader");
            self.sbt_raygen[i] = try c.devAlloc(header.len, "SBT");
            try c.upload(self.sbt_raygen[i], &header);
        }
        try self.check(ft.optixSbtRecordPackHeader(self.groups[g_miss], &header), "optixSbtRecordPackHeader");
        self.sbt_miss = try c.devAlloc(header.len, "SBT");
        try c.upload(self.sbt_miss, &header);
        try self.check(ft.optixSbtRecordPackHeader(self.groups[g_hit], &self.hit_header), "optixSbtRecordPackHeader");

        const max_geo = c.geometries.len;
        self.sbt_hit = try c.devAlloc(max_geo * hit_record_size, "SBT");
        const empty = types.RtHitRecord{ .header = self.hit_header, .data = std.mem.zeroes(types.RtGeometry) };
        for (0..max_geo) |i| try c.uploadValue(self.sbt_hit + i * hit_record_size, &empty);

        self.params_dev = try c.devAlloc(@sizeOf(types.RtParams), "RT-Parameter");
        self.gas = c.gpa.alloc(GasSlot, max_geo) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        @memset(self.gas, .{});
        self.gas_handles = try c.devAlloc(max_geo * 8, "GAS-Tabelle");
        try c.check(c.drv.cuMemsetD8Async(self.gas_handles, 0, max_geo * 8, c.stream), "cuMemsetD8Async");

        // IAS-Puffer für die maximale Instanzanzahl
        const max_inst = c.instances.len;
        self.ias_instances = try c.devAlloc(max_inst * @sizeOf(types.RtInstance), "IAS-Instanzen");
        const input = optix.BuildInput.instances(.{ .instances = self.ias_instances, .numInstances = @intCast(max_inst) });
        const bo = optix.AccelBuildOptions{ .buildFlags = ias_build_flags };
        var sizes = optix.AccelBufferSizes{};
        try self.check(ft.optixAccelComputeMemoryUsage(self.ctx, &bo, @ptrCast(&input), 1, &sizes), "optixAccelComputeMemoryUsage(IAS)");
        self.ias_temp_size = @max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes);
        self.ias_out_size = sizes.outputSizeInBytes;
        self.ias_temp = try c.devAlloc(self.ias_temp_size, "IAS");
        self.ias_out = try c.devAlloc(self.ias_out_size, "IAS");
    }

    fn createPipeline(self: *Rt, c: *Context) Error!void {
        const ft = &self.api.ft;
        var log: [4096]u8 = undefined;
        var log_size: usize = log.len;

        var mco = optix.ModuleCompileOptions{
            .debugLevel = if (c.debug) optix.compile_debug_level_minimal else optix.compile_debug_level_none,
        };
        // Diagnose: PYRIT_OPTIX_OPT=0..3 wählt die Optimierungsstufe
        if (std.c.getenv("PYRIT_OPTIX_OPT")) |e| {
            const lvl = std.fmt.parseInt(u32, std.mem.span(e), 10) catch 3;
            mco.optLevel = @as(c_uint, 0x2340) + @min(lvl, 3);
        }
        const pco = optix.PipelineCompileOptions{
            .traversableGraphFlags = optix.traversable_graph_flag_allow_single_level_instancing,
            .numPayloadValues = 4,
            .numAttributeValues = 2,
            .exceptionFlags = optix.exception_flag_none,
            .pipelineLaunchParamsVariableName = "pyr_rt_params",
            .pipelineLaunchParamsSizeInBytes = @sizeOf(types.RtParams),
            .usesPrimitiveTypeFlags = optix.primitive_type_flags_custom,
        };
        var ts0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts0);
        const r = ft.optixModuleCreate(self.ctx, &mco, &pco, rt_ptx.ptr, rt_ptx.len, &log, &log_size, &self.module);
        if (r != optix.success) return fail(error.Compile, "OptiX-Modul: {s}\n{s}", .{ self.api.errorString(r), log[0..@min(log_size, log.len)] });
        var ts1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts1);
        const secs = @as(f64, @floatFromInt(ts1.sec - ts0.sec)) + @as(f64, @floatFromInt(ts1.nsec - ts0.nsec)) / 1e9;
        if (std.c.getenv("PYRIT_OPTIX_TIMING") != null) std.debug.print("OptiX-Modul ({d} KB PTX) übersetzt in {d:.2} s\n", .{ rt_ptx.len >> 10, secs });

        const descs = [4]optix.ProgramGroupDesc{
            .{ .kind = optix.program_group_kind_raygen, .u = .{ .raygen = .{ .module = self.module, .entryFunctionName = "__raygen__trace" } } },
            .{ .kind = optix.program_group_kind_raygen, .u = .{ .raygen = .{ .module = self.module, .entryFunctionName = "__raygen__slots" } } },
            .{ .kind = optix.program_group_kind_miss, .u = .{ .miss = .{ .module = self.module, .entryFunctionName = "__miss__none" } } },
            .{ .kind = optix.program_group_kind_hitgroup, .u = .{ .hitgroup = .{
                .moduleCH = self.module,
                .entryFunctionNameCH = "__closesthit__dag",
                .moduleIS = self.module,
                .entryFunctionNameIS = "__intersection__dag",
            } } },
        };
        const pgo = optix.ProgramGroupOptions{};
        log_size = log.len;
        const rg = ft.optixProgramGroupCreate(self.ctx, &descs, descs.len, &pgo, &log, &log_size, &self.groups);
        if (rg != optix.success) return fail(error.Compile, "OptiX-Programmgruppen: {s}\n{s}", .{ self.api.errorString(rg), log[0..@min(log_size, log.len)] });

        const plo = optix.PipelineLinkOptions{ .maxTraceDepth = 1 };
        log_size = log.len;
        const rp = ft.optixPipelineCreate(self.ctx, &pco, &plo, &self.groups, self.groups.len, &log, &log_size, &self.pipeline);
        if (rp != optix.success) return fail(error.Compile, "OptiX-Pipeline: {s}\n{s}", .{ self.api.errorString(rp), log[0..@min(log_size, log.len)] });

        // Stapelgrößen für Verfolgungstiefe 1, ohne Callables
        var m = optix.StackSizes{};
        for (self.groups) |g| {
            var s = optix.StackSizes{};
            try self.check(ft.optixProgramGroupGetStackSize(g, &s, self.pipeline), "optixProgramGroupGetStackSize");
            inline for (std.meta.fields(optix.StackSizes)) |f| @field(m, f.name) = @max(@field(m, f.name), @field(s, f.name));
        }
        const continuation = m.cssRG + @max(@max(m.cssCH, m.cssMS), m.cssIS + m.cssAH);
        try self.check(ft.optixPipelineSetStackSize(self.pipeline, 0, 0, continuation, 2), "optixPipelineSetStackSize");
    }

    pub fn deinit(self: *Rt, c: *Context) void {
        const ft = &self.api.ft;
        for (self.gas) |g| freeGas(c, g);
        c.gpa.free(self.gas);
        self.replay.free(c);
        for ([_]cuda.CUdeviceptr{ self.sbt_raygen[0], self.sbt_raygen[1], self.sbt_miss, self.sbt_hit, self.params_dev, self.gas_handles, self.ias_instances, self.ias_temp, self.ias_out }) |p| {
            if (p != 0) _ = c.drv.cuMemFree_v2(p);
        }
        if (self.pipeline != null) _ = ft.optixPipelineDestroy(self.pipeline);
        for (self.groups) |g| if (g != null) {
            _ = ft.optixProgramGroupDestroy(g);
        };
        if (self.module != null) _ = ft.optixModuleDestroy(self.module);
        if (self.ctx != null) _ = ft.optixDeviceContextDestroy(self.ctx);
        self.api.lib.close();
        c.gpa.destroy(self);
    }

    fn freeGas(c: *Context, g: GasSlot) void {
        if (g.buffer != 0) _ = c.drv.cuMemFree_v2(g.buffer);
        if (g.prims != 0) _ = c.drv.cuMemFree_v2(g.prims);
    }

    // -----------------------------------------------------------------------
    // Geometrie: AABB-Primitive -> kompaktierte GAS
    // -----------------------------------------------------------------------

    /// GAS aus einer Host-DAG (CPU-Bau): Primitive auf dem Host bestimmen, hochladen.
    pub fn buildGeometry(self: *Rt, c: *Context, index: u32, dag: *const dag_builder.Dag, data: types.GeometryData) Error!void {
        var prims = rt_prims.extract(c.gpa, dag, self.rt_log2) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        defer prims.deinit(c.gpa);
        var prims_dev: cuda.CUdeviceptr = 0;
        var aabbs_dev: cuda.CUdeviceptr = 0;
        defer if (aabbs_dev != 0) {
            _ = c.drv.cuMemFree_v2(aabbs_dev);
        };
        if (prims.prims.len > 0) {
            prims_dev = try c.devAlloc(prims.prims.len * @sizeOf(types.RtPrim), "RT-Primitive");
            errdefer _ = c.drv.cuMemFree_v2(prims_dev);
            try c.upload(prims_dev, std.mem.sliceAsBytes(prims.prims));
            aabbs_dev = try c.devAlloc(prims.aabbs.len * @sizeOf(rt_prims.Aabb), "AABBs");
            try c.upload(aabbs_dev, std.mem.sliceAsBytes(prims.aabbs));
        }
        try self.buildGas(c, index, prims_dev, aabbs_dev, @intCast(prims.prims.len), prims.rt_log2, data);
    }

    /// Baut die GAS aus Primitiven im Gerätespeicher. `prims` geht in den Besitz
    /// der Geometrie über, `aabbs` bleibt beim Aufrufer. Synchronisiert den Stream
    /// (Kompaktierung), daher nur beim Anlegen/Ändern, nie pro Frame.
    pub fn buildGas(self: *Rt, c: *Context, index: u32, prims: cuda.CUdeviceptr, aabbs: cuda.CUdeviceptr, count: u32, rt_log2: u32, data: types.GeometryData) Error!void {
        const ft = &self.api.ft;
        var slot = GasSlot{ .prim_count = count, .prims = prims, .sbt_prims = prims };
        errdefer freeGas(c, slot);
        if (count > 0) {
            const aabb_ptrs = [_]cuda.CUdeviceptr{aabbs};
            const flags = [_]c_uint{optix.geometry_flag_disable_anyhit};
            const input = optix.BuildInput.custom(.{
                .aabbBuffers = &aabb_ptrs,
                .numPrimitives = count,
                .flags = &flags,
                .numSbtRecords = 1,
            });
            const bo = optix.AccelBuildOptions{ .buildFlags = optix.build_flag_prefer_fast_trace | optix.build_flag_allow_compaction };
            var sizes = optix.AccelBufferSizes{};
            try self.check(ft.optixAccelComputeMemoryUsage(self.ctx, &bo, @ptrCast(&input), 1, &sizes), "optixAccelComputeMemoryUsage(GAS)");

            const temp = try c.devAlloc(sizes.tempSizeInBytes, "GAS-Temp");
            defer _ = c.drv.cuMemFree_v2(temp);
            const out_size = std.mem.alignForward(usize, sizes.outputSizeInBytes, 8);
            const out = try c.devAlloc(out_size + 8, "GAS");
            var out_owned = true;
            defer if (out_owned) {
                _ = c.drv.cuMemFree_v2(out);
            };
            const emit = [_]optix.AccelEmitDesc{.{ .result = out + out_size, .type = optix.property_type_compacted_size }};
            var handle: optix.TraversableHandle = 0;
            try self.check(ft.optixAccelBuild(self.ctx, c.stream, &bo, @ptrCast(&input), 1, temp, sizes.tempSizeInBytes, out, sizes.outputSizeInBytes, &handle, &emit, 1), "optixAccelBuild(GAS)");
            try c.check(c.drv.cuStreamSynchronize(c.stream), "cuStreamSynchronize");
            var compacted: u64 = 0;
            try c.check(c.drv.cuMemcpyDtoH_v2(&compacted, out + out_size, 8), "cuMemcpyDtoH");

            if (compacted > 0 and compacted < sizes.outputSizeInBytes) {
                const small = try c.devAlloc(compacted, "GAS");
                errdefer _ = c.drv.cuMemFree_v2(small);
                try self.check(ft.optixAccelCompact(self.ctx, c.stream, handle, small, compacted, &handle), "optixAccelCompact");
                try c.check(c.drv.cuStreamSynchronize(c.stream), "cuStreamSynchronize");
                slot.buffer = small;
            } else {
                slot.buffer = out;
                out_owned = false;
            }
            slot.handle = handle;
            c.logf(4, "RT: Geometrie {d}: {d} Primitive (2^{d}), GAS {d} KiB", .{ index, count, rt_log2, (if (compacted > 0) compacted else sizes.outputSizeInBytes) / 1024 });
        }

        const record = types.RtHitRecord{
            .header = self.hit_header,
            .data = .{
                .nodes = c.node_pool + @as(u64, data.node_offset) * 4,
                .leaves = c.leaf_pool + @as(u64, data.leaf_offset) * 8,
                .attributes = if (data.flags & types.geometry_has_attributes != 0) c.attr_pool + @as(u64, data.attribute_offset) * 4 else 0,
                .prims = slot.prims,
                .rt_log2 = rt_log2,
                .default_attribute = data.default_attribute,
                .reserved = .{ 0, 0 },
            },
        };
        try c.uploadValue(self.sbt_hit + @as(u64, index) * hit_record_size, &record);
        try c.uploadValue(self.gas_handles + @as(u64, index) * 8, &slot.handle);
        self.gas[index] = slot;
        self.markInstancesDirty(true);
    }

    /// SBT-Eintrag einer Geometrie neu schreiben (Attribute verschoben,
    /// Pfadänderung); GAS und Primitive bleiben
    pub fn updateRecord(self: *Rt, c: *Context, index: u32, data: types.GeometryData, rt_log2: u32) Error!void {
        const record = types.RtHitRecord{
            .header = self.hit_header,
            .data = .{
                .nodes = c.node_pool + @as(u64, data.node_offset) * 4,
                .leaves = c.leaf_pool + @as(u64, data.leaf_offset) * 8,
                .attributes = if (data.flags & types.geometry_has_attributes != 0) c.attr_pool + @as(u64, data.attribute_offset) * 4 else 0,
                .prims = self.gas[index].sbt_prims,
                .rt_log2 = rt_log2,
                .default_attribute = data.default_attribute,
                .reserved = .{ 0, 0 },
            },
        };
        try c.uploadValue(self.sbt_hit + @as(u64, index) * hit_record_size, &record);
    }

    pub const GasJob = struct {
        index: u32,
        /// Primitive und AABBs (gehören weiter dem Aufrufer; prims muss leben,
        /// solange die Geometrie existiert)
        prims: cuda.CUdeviceptr,
        aabbs: cuda.CUdeviceptr,
        count: u32,
        data: types.GeometryData,
    };

    /// Viele kleine GAS auf einmal (Chunks), vollständig auf `stream` und ohne
    /// jede Synchronisation: kein Kompaktieren (bei wenigen AABBs je Chunk
    /// spart es kaum Speicher, kostet aber ein Warten auf die GPU), Speicher
    /// stream-geordnet (Freigabe per cuMemFreeAsync, Deferred.async_free).
    pub fn buildGasMany(self: *Rt, c: *Context, jobs: []const GasJob, rt_log2: u32, stream: cuda.CUstream) Error!void {
        if (jobs.len == 0) return;
        const ft = &self.api.ft;
        const flags = [_]c_uint{optix.geometry_flag_disable_anyhit};
        const bo = optix.AccelBuildOptions{ .buildFlags = optix.build_flag_prefer_fast_trace };
        const A = struct {
            fn alloc(cc: *Context, st: cuda.CUstream, bytes: u64) Error!cuda.CUdeviceptr {
                var p: cuda.CUdeviceptr = 0;
                const r = cc.drv.cuMemAllocAsync(&p, @max(bytes, 16), st);
                if (r != cuda.CUDA_SUCCESS) return fail(error.OutOfMemory, "GPU-Speicher für GAS ({d} Bytes): {s}", .{ bytes, cc.drv.errorString(r) });
                return p;
            }
        };
        var temp_max: usize = 0;
        for (jobs) |j| {
            const aabb_ptrs = [_]cuda.CUdeviceptr{j.aabbs};
            const input = optix.BuildInput.custom(.{ .aabbBuffers = &aabb_ptrs, .numPrimitives = j.count, .flags = &flags, .numSbtRecords = 1 });
            var bs = optix.AccelBufferSizes{};
            try self.check(ft.optixAccelComputeMemoryUsage(self.ctx, &bo, @ptrCast(&input), 1, &bs), "optixAccelComputeMemoryUsage(GAS)");
            temp_max = @max(temp_max, bs.tempSizeInBytes);
        }
        const temp = try A.alloc(c, stream, temp_max);
        defer _ = c.drv.cuMemFreeAsync(temp, stream);

        var done: usize = 0;
        errdefer for (jobs[0..done]) |j| {
            _ = c.drv.cuMemFreeAsync(self.gas[j.index].buffer, stream);
            self.gas[j.index] = .{};
        };
        var total: u64 = 0;
        for (jobs, 0..) |j, i| {
            const aabb_ptrs = [_]cuda.CUdeviceptr{j.aabbs};
            const input = optix.BuildInput.custom(.{ .aabbBuffers = &aabb_ptrs, .numPrimitives = j.count, .flags = &flags, .numSbtRecords = 1 });
            var bs = optix.AccelBufferSizes{};
            try self.check(ft.optixAccelComputeMemoryUsage(self.ctx, &bo, @ptrCast(&input), 1, &bs), "optixAccelComputeMemoryUsage(GAS)");
            const out = try A.alloc(c, stream, bs.outputSizeInBytes);
            errdefer _ = c.drv.cuMemFreeAsync(out, stream);
            var h: optix.TraversableHandle = 0;
            try self.check(ft.optixAccelBuild(self.ctx, stream, &bo, @ptrCast(&input), 1, temp, temp_max, out, bs.outputSizeInBytes, &h, null, 0), "optixAccelBuild(GAS)");
            self.gas[j.index] = .{ .buffer = out, .prims = 0, .sbt_prims = j.prims, .sbt_aabbs = j.aabbs, .handle = h, .prim_count = j.count };
            done = i + 1;
            total += bs.outputSizeInBytes;
        }

        for (jobs) |j| {
            const record = types.RtHitRecord{
                .header = self.hit_header,
                .data = .{
                    .nodes = c.node_pool + @as(u64, j.data.node_offset) * 4,
                    .leaves = c.leaf_pool + @as(u64, j.data.leaf_offset) * 8,
                    .attributes = if (j.data.flags & types.geometry_has_attributes != 0) c.attr_pool + @as(u64, j.data.attribute_offset) * 4 else 0,
                    .prims = j.prims,
                    .rt_log2 = rt_log2,
                    .default_attribute = j.data.default_attribute,
                    .reserved = .{ 0, 0 },
                },
            };
            try c.uploadValue(self.sbt_hit + @as(u64, j.index) * hit_record_size, &record);
            try c.uploadValue(self.gas_handles + @as(u64, j.index) * 8, &self.gas[j.index].handle);
        }
        c.logf(4, "RT: {d} Chunk-GAS, zusammen {d} KiB", .{ jobs.len, total / 1024 });
    }

    /// Nimmt die GAS-Puffer einer Geometrie heraus (zur verzögerten Freigabe).
    pub fn takeGas(self: *Rt, index: u32) [2]cuda.CUdeviceptr {
        const g = self.gas[index];
        self.gas[index] = .{};
        return .{ g.buffer, g.prims };
    }

    /// Geometrie entfernt: Tabelleneintrag löschen, IAS neu bauen
    pub fn forgetGeometry(self: *Rt, c: *Context, index: u32) Error!void {
        const zero: u64 = 0;
        try c.uploadValue(self.gas_handles + @as(u64, index) * 8, &zero);
        self.markInstancesDirty(true);
    }

    // -----------------------------------------------------------------------
    // Instanzen: IAS aus dem aktuellen Zustand, vollständig auf der GPU
    // -----------------------------------------------------------------------

    /// Nach so vielen Refits wird die IAS neu gebaut (Qualität der BVH)
    const max_refits = 32;
    const ias_build_flags = optix.build_flag_prefer_fast_trace | optix.build_flag_allow_update;

    /// structural: Instanzen angelegt/gelöscht, Geometrie oder Maske geändert;
    /// sonst nur Transformationen -> Refit statt Neubau
    pub fn markInstancesDirty(self: *Rt, structural: bool) void {
        self.ias_dirty = true;
        if (structural) self.ias_structural = true;
    }

    pub fn buildInstances(self: *Rt, c: *Context, instances: cuda.CUdeviceptr, count: u32) Error!void {
        if (!self.ias_dirty) return;
        self.ias_dirty = false;
        if (count == 0) {
            self.ias_handle = 0;
            return;
        }
        var cur = instances;
        var n = count;
        var handles = self.gas_handles;
        var out = self.ias_instances;
        const params = [_]?*anyopaque{ @ptrCast(&cur), @ptrCast(&n), @ptrCast(&handles), @ptrCast(&out) };
        try c.launch(c.fn_rt_instances, .{ (count + types.update_block - 1) / types.update_block, 1, 1 }, .{ types.update_block, 1, 1 }, &params);

        const input = optix.BuildInput.instances(.{ .instances = self.ias_instances, .numInstances = count });
        const refit = !self.ias_structural and count == self.ias_count and self.ias_handle != 0 and self.ias_updates < max_refits;
        const bo = optix.AccelBuildOptions{
            .buildFlags = ias_build_flags,
            .operation = if (refit) optix.build_operation_update else optix.build_operation_build,
        };
        try self.check(self.api.ft.optixAccelBuild(self.ctx, c.stream, &bo, @ptrCast(&input), 1, self.ias_temp, self.ias_temp_size, self.ias_out, self.ias_out_size, &self.ias_handle, null, 0), "optixAccelBuild(IAS)");
        if (refit) {
            self.ias_updates += 1;
        } else {
            self.ias_updates = 0;
            self.ias_count = count;
        }
        self.ias_structural = false;
    }

    // -----------------------------------------------------------------------
    // Launches
    // -----------------------------------------------------------------------

    fn launch(self: *Rt, c: *Context, raygen: usize, params: *const types.RtParams, w: u32, h: u32) Error!void {
        try c.uploadValue(self.params_dev, params);
        const sbt = optix.ShaderBindingTable{
            .raygenRecord = self.sbt_raygen[raygen],
            .missRecordBase = self.sbt_miss,
            .missRecordStrideInBytes = optix.sbt_record_header_size,
            .missRecordCount = 1,
            .hitgroupRecordBase = self.sbt_hit,
            .hitgroupRecordStrideInBytes = hit_record_size,
            .hitgroupRecordCount = @intCast(c.geometries.len),
        };
        try self.check(self.api.ft.optixLaunch(self.pipeline, c.stream, self.params_dev, @sizeOf(types.RtParams), &sbt, w, h, 1), "optixLaunch");
        if (c.debug) try c.check(c.drv.cuStreamSynchronize(c.stream), "optixLaunch");
    }

    /// Bild über die Wiederholungs-Wavefront (src/device/replay.zig):
    /// Schattierung in CUDA, Strahlen in Listen über die RT-Cores.
    pub fn render(self: *Rt, c: *Context, p: *const types.RenderParams) Error!void {
        try self.replayRun(c, p, false, p.cur.camera.width, p.cur.camera.height);
    }

    /// Zweiter Durchgang: indirekte Beleuchtung in halber Auflösung
    pub fn renderGi(self: *Rt, c: *Context, p: *const types.RenderParams) Error!void {
        try self.replayRun(c, p, true, p.gi_width, p.gi_height);
    }

    /// Höchstens so viele Durchgänge (Tiefe der Abhängigkeiten: Primärstrahl,
    /// Schatten/GI/Spiegelung, deren Schatten, Wasserschichten ...). Pixel, die
    /// fertig sind, überspringt der Schattierungskern.
    /// Gemessen: das Bild braucht 4 Durchgänge (fast alle Strahlen im
    /// ersten), die indirekte Beleuchtung 6.
    pub const replay_passes: u32 = 5;
    pub const replay_passes_gi: u32 = 7;
    /// Einträge der Strahlliste je Pixel. Gemessen fragt der erste Durchgang
    /// 5 je Pixel an; was nicht passt, kommt im nächsten Durchgang dran.
    // Durchgang 1 fragt im Mittel 6,3 Strahlen je Pixel an (gemessen); was
    // nicht in die Liste passt, rutscht in den nächsten Durchgang
    const replay_list_per_pixel: u32 = 8;
    /// Pixel je Streifen: begrenzt den Speicher der Strahlplätze (etwa 60 MB)
    // gemessen (1080p, DLSS 2x): 65536 -> 262144 spart 3 ms, weil die fast leeren
    // letzten Durchgänge nicht mehr achtmal hintereinander laufen
    const replay_tile_pixels: u32 = 262144;

    fn replayRun(self: *Rt, c: *Context, p: *const types.RenderParams, gi: bool, w: u32, h: u32) Error!void {
        if (w == 0 or h == 0) return;
        const tile: u32 = if (std.c.getenv("PYRIT_REPLAY_TILE")) |e| (std.fmt.parseInt(u32, std.mem.span(e), 10) catch replay_tile_pixels) else replay_tile_pixels;
        const rows = @max(@min(h, tile / @max(w, 1)), 1);
        const tile_px = w * rows;
        try self.replay.ensure(c, tile_px);
        var params = std.mem.zeroes(types.RtParams);
        params.handle = self.ias_handle;
        params.scene = c.scene_dev;
        params.flags = p.flags;
        params.render = p.*;
        const fn_pass = if (gi) c.fn_replay_gi else c.fn_replay_render;
        const stats = std.c.getenv("PYRIT_REPLAY_STATS") != null;
        var per_pass = [_]u64{0} ** 16;
        const primary_pre = std.c.getenv("PYRIT_REPLAY_NOPRIMARY") == null;
        const per_pass_timing = std.c.getenv("PYRIT_GPU_TIMING_PASSES") != null;
        const pass_names = [_][]const u8{ "Bild D0", "Bild D1", "Bild D2", "Bild D3", "Bild D4" };
        const rt_names = [_][]const u8{ "RT D0", "RT D1", "RT D2", "RT D3", "RT D4" };
        // Durchgänge ohne Spekulation am Anfang (Diagnose PYRIT_REPLAY_NOSPEC)
        const no_spec: u32 = if (std.c.getenv("PYRIT_REPLAY_NOSPEC")) |e| (std.fmt.parseInt(u32, std.mem.span(e), 10) catch 1) else 1;
        var y0: u32 = 0;
        while (y0 < h) : (y0 += rows) {
            const rb = &self.replay;
            var rp = types.ReplayParams{
                .rays = rb.rays,
                .hits = rb.hits,
                .state = rb.state,
                .done = rb.done,
                .list = rb.list,
                .count = rb.count,
                .capacity = @min(rb.capacity, (w * @min(rows, h - y0)) * replay_list_per_pixel),
                .y0 = y0,
                .rows = @min(rows, h - y0),
                .gi = @intFromBool(gi),
                .speculate = 1,
            };
            const n_px = w * rp.rows;
            try c.check(c.drv.cuMemsetD8Async(rb.state, 0, @as(u64, n_px) * types.replay_slots * 4, c.stream), "cuMemsetD8Async");
            try c.check(c.drv.cuMemsetD8Async(rb.done, 0, @as(u64, n_px) * 4, c.stream), "cuMemsetD8Async");
            c.mark("Replay leeren");
            params.replay = rp;
            const passes = if (gi) replay_passes_gi else replay_passes;
            var pass: u32 = 0;
            while (pass < passes) : (pass += 1) {
                try c.check(c.drv.cuMemsetD8Async(rb.count, 0, 4, c.stream), "cuMemsetD8Async");
                // GI hängt an keinem Primärstrahl: seine Strahlen sind von Anfang an
                // unabhängig und dürfen gemeinsam in einen Durchgang
                rp.speculate = @intFromBool(gi or pass >= no_spec);
                params.replay = rp;
                var pr = p.*;
                var tf = p.flags;
                const args = [_]?*anyopaque{ @ptrCast(&pr), @ptrCast(&rp), @ptrCast(&tf) };
                const b = types.replay_block;
                // Bild: der erste Durchgang trägt nur die Primärstrahlen ein
                const f = if (!gi and pass == 0 and primary_pre) c.fn_replay_primary else fn_pass;
                try c.launch(f, .{ (n_px + b - 1) / b, 1, 1 }, .{ b, 1, 1 }, &args);
                c.mark(if (gi) "GI schattieren" else if (per_pass_timing) pass_names[@min(pass, 4)] else "Bild schattieren");
                // der letzte Durchgang schattiert nur noch fertig, er verfolgt nichts
                if (pass + 1 < passes) try self.launch(c, g_raygen_slots, &params, @min(rb.capacity, n_px * replay_list_per_pixel), 1);
                c.mark(if (gi) "GI RT" else if (per_pass_timing) rt_names[@min(pass, 4)] else "Bild RT");
                // Diagnose: Anfragen je Durchgang über alle Streifen (synchronisiert)
                if (stats) {
                    var n: u32 = 0;
                    try c.check(c.drv.cuStreamSynchronize(c.stream), "cuStreamSynchronize");
                    try c.check(c.drv.cuMemcpyDtoH_v2(&n, rb.count, 4), "cuMemcpyDtoH");
                    per_pass[pass] += n;
                }
            }
        }
        if (stats) {
            std.debug.print("{s}: {d} Pixel, Strahlen je Pixel und Durchgang:", .{ if (gi) "GI" else "Bild", w * h });
            for (per_pass[0..if (gi) replay_passes_gi else replay_passes]) |n| std.debug.print(" {d:.2}", .{@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(w * h))});
            std.debug.print("\n", .{});
        }
    }

    pub fn trace(self: *Rt, c: *Context, p: *const types.TraceParams) Error!void {
        var params = std.mem.zeroes(types.RtParams);
        params.handle = self.ias_handle;
        params.scene = c.scene_dev;
        params.flags = p.flags;
        params.trace = p.*;
        // OptiX begrenzt eine Launch-Dimension; große Mengen in Teilen
        const max_chunk: u32 = 1 << 30;
        var start: u32 = 0;
        while (start < p.count) : (start += max_chunk) {
            const n = @min(p.count - start, max_chunk);
            params.trace.rays = p.rays + @as(u64, start) * @sizeOf(types.Ray);
            const hit_size: u64 = if (p.flags & types.trace_extended != 0) @sizeOf(types.HitEx) else @sizeOf(types.Hit);
            params.trace.hits = p.hits + @as(u64, start) * hit_size;
            params.trace.count = n;
            try self.launch(c, g_raygen_trace, &params, n, 1);
        }
    }
};

/// Speicher der Wiederholungs-Wavefront, einmal je Streifengröße angelegt
const ReplayBuffers = struct {
    rays: cuda.CUdeviceptr = 0,
    hits: cuda.CUdeviceptr = 0,
    state: cuda.CUdeviceptr = 0,
    done: cuda.CUdeviceptr = 0,
    list: cuda.CUdeviceptr = 0,
    count: cuda.CUdeviceptr = 0,
    pixels: u32 = 0,
    capacity: u32 = 0,

    fn ensure(self: *ReplayBuffers, c: *Context, pixels: u32) Error!void {
        if (self.pixels >= pixels) return;
        self.free(c);
        const places = @as(u64, pixels) * types.replay_slots;
        self.rays = try c.devAlloc(places * @sizeOf(types.ReplayRay), "Wavefront-Strahlen");
        self.hits = try c.devAlloc(places * @sizeOf(types.ReplayHit), "Wavefront-Treffer");
        self.state = try c.devAlloc(places * 4, "Wavefront-Zustand");
        self.done = try c.devAlloc(@as(u64, pixels) * 4, "Wavefront-Pixel");
        self.list = try c.devAlloc(places * 4, "Wavefront-Liste");
        self.count = try c.devAlloc(4, "Wavefront-Zähler");
        self.pixels = pixels;
        self.capacity = @intCast(places);
    }

    fn free(self: *ReplayBuffers, c: *Context) void {
        for ([_]cuda.CUdeviceptr{ self.rays, self.hits, self.state, self.done, self.list, self.count }) |b| {
            if (b != 0) _ = c.drv.cuMemFree_v2(b);
        }
        self.* = .{};
    }
};
