//! Minimale Zig-Anbindung an OptiX 9.1 (ABI 118) für die RT-Cores.
//!
//! OptiX ist Teil des NVIDIA-Treibers (libnvoptix). Die Funktionen kommen über
//! optixQueryFunctionTable; Pyrit enthält keine OptiX-Header. Layouts und
//! Konstanten sind aus der öffentlichen OptiX-ABI übernommen und werden mit
//! `zig build optix-abi-test -Doptix-include=<pfad>` gegen die Original-Header
//! geprüft.

const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda.zig");

pub const abi_version: c_int = 118;

pub const Result = c_int;
pub const success: Result = 0;

pub const DeviceContext = ?*opaque {};
pub const Module = ?*opaque {};
pub const ProgramGroup = ?*opaque {};
pub const Pipeline = ?*opaque {};
pub const TraversableHandle = u64;

pub const LogCallback = ?*const fn (level: c_uint, tag: [*:0]const u8, message: [*:0]const u8, data: ?*anyopaque) callconv(.c) void;

pub const DeviceContextOptions = extern struct {
    logCallbackFunction: LogCallback = null,
    logCallbackData: ?*anyopaque = null,
    logCallbackLevel: c_int = 0,
    validationMode: c_uint = 0,
};

pub const validation_mode_off: c_uint = 0;
pub const validation_mode_all: c_uint = 0xFFFF_FFFF;
pub const device_property_rtcore_version: c_uint = 0x2005;

// ---------------------------------------------------------------------------
// Beschleunigungsstrukturen
// ---------------------------------------------------------------------------

pub const Aabb = extern struct {
    minX: f32,
    minY: f32,
    minZ: f32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
};

pub const BuildInputCustomPrimitiveArray = extern struct {
    aabbBuffers: ?[*]const cuda.CUdeviceptr = null,
    numPrimitives: c_uint = 0,
    strideInBytes: c_uint = 0,
    flags: ?[*]const c_uint = null,
    numSbtRecords: c_uint = 0,
    sbtIndexOffsetBuffer: cuda.CUdeviceptr = 0,
    sbtIndexOffsetSizeInBytes: c_uint = 0,
    sbtIndexOffsetStrideInBytes: c_uint = 0,
    primitiveIndexOffset: c_uint = 0,
};

pub const BuildInputInstanceArray = extern struct {
    instances: cuda.CUdeviceptr = 0,
    numInstances: c_uint = 0,
    instanceStride: c_uint = 0,
};

pub const build_input_type_custom_primitives: c_uint = 0x2142;
pub const build_input_type_instances: c_uint = 0x2143;

pub const BuildInput = extern struct {
    type: c_uint,
    data: extern union {
        pad: [1024]u8,
        customPrimitiveArray: BuildInputCustomPrimitiveArray,
        instanceArray: BuildInputInstanceArray,
    },

    pub fn custom(a: BuildInputCustomPrimitiveArray) BuildInput {
        var b = std.mem.zeroes(BuildInput);
        b.type = build_input_type_custom_primitives;
        b.data.customPrimitiveArray = a;
        return b;
    }

    pub fn instances(a: BuildInputInstanceArray) BuildInput {
        var b = std.mem.zeroes(BuildInput);
        b.type = build_input_type_instances;
        b.data.instanceArray = a;
        return b;
    }
};

pub const MotionOptions = extern struct {
    numKeys: c_ushort = 0,
    flags: c_ushort = 0,
    timeBegin: f32 = 0,
    timeEnd: f32 = 0,
};

pub const AccelBuildOptions = extern struct {
    buildFlags: c_uint = 0,
    operation: c_uint = build_operation_build,
    motionOptions: MotionOptions = .{},
};

pub const AccelBufferSizes = extern struct {
    outputSizeInBytes: usize = 0,
    tempSizeInBytes: usize = 0,
    tempUpdateSizeInBytes: usize = 0,
};

pub const AccelEmitDesc = extern struct {
    result: cuda.CUdeviceptr,
    type: c_uint,
};

pub const build_flag_allow_update: c_uint = 1 << 0;
pub const build_flag_allow_compaction: c_uint = 1 << 1;
pub const build_flag_prefer_fast_trace: c_uint = 1 << 2;
pub const build_flag_prefer_fast_build: c_uint = 1 << 3;
pub const build_operation_build: c_uint = 0x2161;
pub const build_operation_update: c_uint = 0x2162;
pub const property_type_compacted_size: c_uint = 0x2181;
pub const geometry_flag_disable_anyhit: c_uint = 1 << 0;
pub const instance_flag_disable_anyhit: c_uint = 1 << 2;

pub const accel_buffer_alignment = 128;
pub const instance_alignment = 16;
pub const aabb_alignment = 8;
pub const sbt_record_alignment = 16;
pub const sbt_record_header_size = 32;

// ---------------------------------------------------------------------------
// Module, Programmgruppen, Pipeline
// ---------------------------------------------------------------------------

pub const compile_optimization_default: c_uint = 0;
pub const compile_debug_level_none: c_uint = 0x2350;
pub const compile_debug_level_minimal: c_uint = 0x2351;

pub const ModuleCompileOptions = extern struct {
    maxRegisterCount: c_int = 0,
    optLevel: c_uint = compile_optimization_default,
    debugLevel: c_uint = compile_debug_level_none,
    boundValues: ?*const anyopaque = null,
    numBoundValues: c_uint = 0,
    numPayloadTypes: c_uint = 0,
    payloadTypes: ?*const anyopaque = null,
    baseModule: Module = null,
};

pub const traversable_graph_flag_allow_single_level_instancing: c_uint = 1 << 1;
pub const primitive_type_flags_custom: c_uint = 1 << 0;
pub const exception_flag_none: c_uint = 0;

pub const PipelineCompileOptions = extern struct {
    usesMotionBlur: c_int = 0,
    traversableGraphFlags: c_uint = 0,
    numPayloadValues: c_int = 0,
    numAttributeValues: c_int = 0,
    exceptionFlags: c_uint = 0,
    pipelineLaunchParamsVariableName: ?[*:0]const u8 = null,
    pipelineLaunchParamsSizeInBytes: usize = 0,
    usesPrimitiveTypeFlags: c_uint = 0,
    allowOpacityMicromaps: c_int = 0,
    allowClusteredGeometry: c_int = 0,
};

pub const PipelineLinkOptions = extern struct {
    maxTraceDepth: c_uint = 0,
    maxContinuationCallableDepth: c_uint = 0,
    maxDirectCallableDepthFromState: c_uint = 0,
    maxDirectCallableDepthFromTraversal: c_uint = 0,
    maxTraversableGraphDepth: c_uint = 0,
};

pub const program_group_kind_raygen: c_uint = 0x2421;
pub const program_group_kind_miss: c_uint = 0x2422;
pub const program_group_kind_exception: c_uint = 0x2423;
pub const program_group_kind_hitgroup: c_uint = 0x2424;

pub const ProgramGroupSingleModule = extern struct {
    module: Module = null,
    entryFunctionName: ?[*:0]const u8 = null,
};

pub const ProgramGroupHitgroup = extern struct {
    moduleCH: Module = null,
    entryFunctionNameCH: ?[*:0]const u8 = null,
    moduleAH: Module = null,
    entryFunctionNameAH: ?[*:0]const u8 = null,
    moduleIS: Module = null,
    entryFunctionNameIS: ?[*:0]const u8 = null,
};

pub const ProgramGroupCallables = extern struct {
    moduleDC: Module = null,
    entryFunctionNameDC: ?[*:0]const u8 = null,
    moduleCC: Module = null,
    entryFunctionNameCC: ?[*:0]const u8 = null,
};

pub const ProgramGroupDesc = extern struct {
    kind: c_uint,
    flags: c_uint = 0,
    u: extern union {
        hitgroup: ProgramGroupHitgroup,
        raygen: ProgramGroupSingleModule,
        miss: ProgramGroupSingleModule,
        exception: ProgramGroupSingleModule,
        callables: ProgramGroupCallables,
    },
};

pub const ProgramGroupOptions = extern struct {
    payloadType: ?*const anyopaque = null,
};

pub const ShaderBindingTable = extern struct {
    raygenRecord: cuda.CUdeviceptr = 0,
    exceptionRecord: cuda.CUdeviceptr = 0,
    missRecordBase: cuda.CUdeviceptr = 0,
    missRecordStrideInBytes: c_uint = 0,
    missRecordCount: c_uint = 0,
    hitgroupRecordBase: cuda.CUdeviceptr = 0,
    hitgroupRecordStrideInBytes: c_uint = 0,
    hitgroupRecordCount: c_uint = 0,
    callablesRecordBase: cuda.CUdeviceptr = 0,
    callablesRecordStrideInBytes: c_uint = 0,
    callablesRecordCount: c_uint = 0,
};

pub const StackSizes = extern struct {
    cssRG: c_uint = 0,
    cssMS: c_uint = 0,
    cssCH: c_uint = 0,
    cssAH: c_uint = 0,
    cssIS: c_uint = 0,
    cssCC: c_uint = 0,
    dssDC: c_uint = 0,
};

// ---------------------------------------------------------------------------
// Funktionstabelle (Reihenfolge = OptiX-ABI 118)
// ---------------------------------------------------------------------------

const Unused = ?*const anyopaque;
const Ctx = DeviceContext;
const Stream = cuda.CUstream;
const Dptr = cuda.CUdeviceptr;

pub const FunctionTable = extern struct {
    optixGetErrorName: *const fn (Result) callconv(.c) [*:0]const u8,
    optixGetErrorString: *const fn (Result) callconv(.c) [*:0]const u8,
    optixDeviceContextCreate: *const fn (cuda.CUcontext, *const DeviceContextOptions, *Ctx) callconv(.c) Result,
    optixDeviceContextDestroy: *const fn (Ctx) callconv(.c) Result,
    optixDeviceContextGetProperty: *const fn (Ctx, c_uint, ?*anyopaque, usize) callconv(.c) Result,
    optixDeviceContextSetLogCallback: Unused,
    optixDeviceContextSetCacheEnabled: *const fn (Ctx, c_int) callconv(.c) Result,
    optixDeviceContextSetCacheLocation: Unused,
    optixDeviceContextSetCacheDatabaseSizes: Unused,
    optixDeviceContextGetCacheEnabled: Unused,
    optixDeviceContextGetCacheLocation: Unused,
    optixDeviceContextGetCacheDatabaseSizes: Unused,
    optixModuleCreate: *const fn (Ctx, *const ModuleCompileOptions, *const PipelineCompileOptions, [*]const u8, usize, ?[*]u8, ?*usize, *Module) callconv(.c) Result,
    optixModuleCreateWithTasks: Unused,
    optixModuleGetCompilationState: Unused,
    optixModuleCancelCreation: Unused,
    optixStub: Unused,
    optixDeviceContextCancelCreations: Unused,
    optixModuleDestroy: *const fn (Module) callconv(.c) Result,
    optixBuiltinISModuleGet: Unused,
    optixTaskExecute: Unused,
    optixTaskGetSerializationKey: Unused,
    optixTaskSerializeOutput: Unused,
    optixTaskDeserializeOutput: Unused,
    optixProgramGroupCreate: *const fn (Ctx, [*]const ProgramGroupDesc, c_uint, *const ProgramGroupOptions, ?[*]u8, ?*usize, [*]ProgramGroup) callconv(.c) Result,
    optixProgramGroupDestroy: *const fn (ProgramGroup) callconv(.c) Result,
    optixProgramGroupGetStackSize: *const fn (ProgramGroup, *StackSizes, Pipeline) callconv(.c) Result,
    optixPipelineCreate: *const fn (Ctx, *const PipelineCompileOptions, *const PipelineLinkOptions, [*]const ProgramGroup, c_uint, ?[*]u8, ?*usize, *Pipeline) callconv(.c) Result,
    optixPipelineDestroy: *const fn (Pipeline) callconv(.c) Result,
    optixPipelineSetStackSizeFromCallDepths: Unused,
    optixPipelineSetStackSize: *const fn (Pipeline, c_uint, c_uint, c_uint, c_uint) callconv(.c) Result,
    optixPipelineSymbolMemcpyAsync: Unused,
    optixAccelComputeMemoryUsage: *const fn (Ctx, *const AccelBuildOptions, [*]const BuildInput, c_uint, *AccelBufferSizes) callconv(.c) Result,
    optixAccelBuild: *const fn (Ctx, Stream, *const AccelBuildOptions, [*]const BuildInput, c_uint, Dptr, usize, Dptr, usize, *TraversableHandle, ?[*]const AccelEmitDesc, c_uint) callconv(.c) Result,
    optixAccelGetRelocationInfo: Unused,
    optixCheckRelocationCompatibility: Unused,
    optixAccelRelocate: Unused,
    optixAccelCompact: *const fn (Ctx, Stream, TraversableHandle, Dptr, usize, *TraversableHandle) callconv(.c) Result,
    optixAccelEmitProperty: Unused,
    optixConvertPointerToTraversableHandle: Unused,
    optixOpacityMicromapArrayComputeMemoryUsage: Unused,
    optixOpacityMicromapArrayBuild: Unused,
    optixOpacityMicromapArrayGetRelocationInfo: Unused,
    optixOpacityMicromapArrayRelocate: Unused,
    stub1: Unused,
    stub2: Unused,
    optixClusterAccelComputeMemoryUsage: Unused,
    optixClusterAccelBuild: Unused,
    optixSbtRecordPackHeader: *const fn (ProgramGroup, ?*anyopaque) callconv(.c) Result,
    optixLaunch: *const fn (Pipeline, Stream, Dptr, usize, *const ShaderBindingTable, c_uint, c_uint, c_uint) callconv(.c) Result,
    optixCoopVecMatrixConvert: Unused,
    optixCoopVecMatrixComputeSize: Unused,
    optixDenoiserCreate: Unused,
    optixDenoiserDestroy: Unused,
    optixDenoiserComputeMemoryResources: Unused,
    optixDenoiserSetup: Unused,
    optixDenoiserInvoke: Unused,
    optixDenoiserComputeIntensity: Unused,
    optixDenoiserComputeAverageColor: Unused,
    optixDenoiserCreateWithUserModel: Unused,
};

const QueryFn = *const fn (abi_id: c_int, num_options: c_uint, option_keys: ?*anyopaque, option_values: ?*anyopaque, table: *anyopaque, table_size: usize) callconv(.c) Result;

pub const Api = struct {
    lib: std.DynLib,
    ft: FunctionTable,

    /// Lädt libnvoptix aus dem Treiber; error.NotFound, wenn nicht vorhanden
    /// oder der Treiber ABI 118 nicht anbietet.
    pub fn load() error{NotFound}!Api {
        const names: []const []const u8 = if (builtin.os.tag == .windows) &.{"nvoptix.dll"} else &.{ "libnvoptix.so.1", "libnvoptix.so" };
        var lib: std.DynLib = for (names) |n| {
            if (std.DynLib.open(n)) |l| break l else |_| {}
        } else return error.NotFound;
        errdefer lib.close();
        const query = lib.lookup(QueryFn, "optixQueryFunctionTable") orelse return error.NotFound;
        var ft: FunctionTable = undefined;
        if (query(abi_version, 0, null, null, &ft, @sizeOf(FunctionTable)) != success) return error.NotFound;
        return .{ .lib = lib, .ft = ft };
    }

    pub fn errorString(self: *const Api, r: Result) []const u8 {
        return std.mem.span(self.ft.optixGetErrorString(r));
    }
};
