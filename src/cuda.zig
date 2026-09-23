//! Dynamisch geladene CUDA-Treiber-API.
//!
//! libcuda wird zur Laufzeit geöffnet: Pyrit baut ohne CUDA-SDK, und das
//! spätere HIP-Backend (libamdhip64) folgt demselben Muster. Die Kernel kommen
//! als PTX aus Zig; der Treiber übersetzt sie beim Laden für die vorhandene GPU.

const std = @import("std");
const builtin = @import("builtin");

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUcontext = ?*opaque {};
pub const CUstream = ?*opaque {};
pub const CUmodule = ?*opaque {};
pub const CUfunction = ?*opaque {};
pub const CUevent = ?*opaque {};
pub const CUmemoryPool = ?*opaque {};
pub const CUarray = ?*opaque {};
pub const CUtexObject = u64;

pub const CU_AD_FORMAT_HALF: c_int = 0x10;
pub const CU_AD_FORMAT_FLOAT: c_int = 0x20;
pub const CUDA_ARRAY3D_SURFACE_LDST: c_uint = 0x02;
pub const CU_RESOURCE_TYPE_ARRAY: c_int = 0;
pub const CU_MEMORYTYPE_DEVICE: c_int = 2;
pub const CU_MEMORYTYPE_ARRAY: c_int = 3;
pub const CU_TR_ADDRESS_MODE_CLAMP: c_int = 1;
pub const CU_TR_FILTER_MODE_POINT: c_int = 0;

pub const CUDA_ARRAY3D_DESCRIPTOR = extern struct {
    Width: usize,
    Height: usize,
    Depth: usize,
    Format: c_int,
    NumChannels: c_uint,
    Flags: c_uint,
};

pub const CUDA_RESOURCE_DESC = extern struct {
    resType: c_int,
    res: extern union {
        array: extern struct { hArray: CUarray },
        reserved: [32]c_int,
    },
    flags: c_uint,
};

pub const CUDA_TEXTURE_DESC = extern struct {
    addressMode: [3]c_int,
    filterMode: c_int,
    flags: c_uint,
    maxAnisotropy: c_uint,
    mipmapFilterMode: c_int,
    mipmapLevelBias: f32,
    minMipmapLevelClamp: f32,
    maxMipmapLevelClamp: f32,
    borderColor: [4]f32,
    reserved: [12]c_int,
};

pub const CUDA_MEMCPY2D = extern struct {
    srcXInBytes: usize = 0,
    srcY: usize = 0,
    srcMemoryType: c_int = 0,
    srcHost: ?*const anyopaque = null,
    srcDevice: CUdeviceptr = 0,
    srcArray: CUarray = null,
    srcPitch: usize = 0,
    dstXInBytes: usize = 0,
    dstY: usize = 0,
    dstMemoryType: c_int = 0,
    dstHost: ?*anyopaque = null,
    dstDevice: CUdeviceptr = 0,
    dstArray: CUarray = null,
    dstPitch: usize = 0,
    WidthInBytes: usize = 0,
    Height: usize = 0,
};

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: c_uint = 4;
pub const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

/// Funktionszeiger der Treiber-API; Feldnamen = exportierte Symbolnamen.
pub const Driver = struct {
    lib: std.DynLib,
    cuInit: *const fn (c_uint) callconv(.c) CUresult,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) CUresult,
    cuGetErrorString: *const fn (CUresult, *?[*:0]const u8) callconv(.c) CUresult,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDeviceGetAttribute: *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult,
    cuDeviceGetName: *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult,
    cuDevicePrimaryCtxRetain: *const fn (*CUcontext, CUdevice) callconv(.c) CUresult,
    cuDevicePrimaryCtxRelease_v2: *const fn (CUdevice) callconv(.c) CUresult,
    cuCtxSetCurrent: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxPushCurrent_v2: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxPopCurrent_v2: *const fn (*CUcontext) callconv(.c) CUresult,
    cuMemAlloc_v2: *const fn (*CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemGetInfo_v2: *const fn (*usize, *usize) callconv(.c) CUresult,
    cuMemcpyDtoD_v2: *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemsetD8_v2: *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,
    cuMemFree_v2: *const fn (CUdeviceptr) callconv(.c) CUresult,
    cuMemHostAlloc: *const fn (*?*anyopaque, usize, c_uint) callconv(.c) CUresult,
    cuMemFreeHost: *const fn (?*anyopaque) callconv(.c) CUresult,
    cuMemcpyHtoD_v2: *const fn (CUdeviceptr, ?*const anyopaque, usize) callconv(.c) CUresult,
    cuMemcpyDtoH_v2: *const fn (?*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemcpyHtoDAsync_v2: *const fn (CUdeviceptr, ?*const anyopaque, usize, CUstream) callconv(.c) CUresult,
    cuMemcpyDtoHAsync_v2: *const fn (?*anyopaque, CUdeviceptr, usize, CUstream) callconv(.c) CUresult,
    cuMemcpyDtoDAsync_v2: *const fn (CUdeviceptr, CUdeviceptr, usize, CUstream) callconv(.c) CUresult,
    cuMemsetD8Async: *const fn (CUdeviceptr, u8, usize, CUstream) callconv(.c) CUresult,
    /// stream-geordneter Speicher (CUDA 11.2+): kein Geräte-Sync beim Freigeben
    cuMemAllocAsync: *const fn (*CUdeviceptr, usize, CUstream) callconv(.c) CUresult,
    cuMemFreeAsync: *const fn (CUdeviceptr, CUstream) callconv(.c) CUresult,
    cuDeviceGetDefaultMemPool: *const fn (*CUmemoryPool, CUdevice) callconv(.c) CUresult,
    cuMemPoolSetAttribute: *const fn (CUmemoryPool, c_int, *anyopaque) callconv(.c) CUresult,
    cuStreamWaitEvent: *const fn (CUstream, CUevent, c_uint) callconv(.c) CUresult,
    cuArray3DCreate_v2: *const fn (*CUarray, *const CUDA_ARRAY3D_DESCRIPTOR) callconv(.c) CUresult,
    cuArrayDestroy: *const fn (CUarray) callconv(.c) CUresult,
    cuTexObjectCreate: *const fn (*CUtexObject, *const CUDA_RESOURCE_DESC, *const CUDA_TEXTURE_DESC, ?*const anyopaque) callconv(.c) CUresult,
    cuTexObjectDestroy: *const fn (CUtexObject) callconv(.c) CUresult,
    cuSurfObjectCreate: *const fn (*u64, *const CUDA_RESOURCE_DESC) callconv(.c) CUresult,
    cuSurfObjectDestroy: *const fn (u64) callconv(.c) CUresult,
    cuMemcpy2DAsync_v2: *const fn (*const CUDA_MEMCPY2D, CUstream) callconv(.c) CUresult,
    cuStreamCreate: *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2: *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,
    cuEventCreate: *const fn (*CUevent, c_uint) callconv(.c) CUresult,
    cuEventDestroy_v2: *const fn (CUevent) callconv(.c) CUresult,
    cuEventRecord: *const fn (CUevent, CUstream) callconv(.c) CUresult,
    cuEventQuery: *const fn (CUevent) callconv(.c) CUresult,
    cuEventSynchronize: *const fn (CUevent) callconv(.c) CUresult,
    cuEventElapsedTime: *const fn (*f32, CUevent, CUevent) callconv(.c) CUresult,
    cuModuleLoadDataEx: *const fn (*CUmodule, ?*const anyopaque, c_uint, ?[*]const c_int, ?[*]?*anyopaque) callconv(.c) CUresult,
    cuModuleUnload: *const fn (CUmodule) callconv(.c) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult,
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) CUresult,

    pub fn load() error{NotFound}!Driver {
        const names: []const []const u8 = if (builtin.os.tag == .windows) &.{"nvcuda.dll"} else &.{ "libcuda.so.1", "libcuda.so" };
        var lib = openFirst(names, null) orelse return error.NotFound;
        errdefer lib.close();
        return resolve(Driver, &lib);
    }

    pub fn errorString(self: *const Driver, r: CUresult) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.cuGetErrorString(r, &s) != CUDA_SUCCESS or s == null) return "unbekannter CUDA-Fehler";
        return std.mem.span(s.?);
    }
};

fn openFirst(names: []const []const u8, root: ?[]const u8) ?std.DynLib {
    var buf: [1024]u8 = undefined;
    for (names) |name| {
        const path = if (root) |r| std.fmt.bufPrint(&buf, "{s}/lib64/{s}", .{ r, name }) catch continue else name;
        if (std.DynLib.open(path)) |lib| return lib else |_| {}
    }
    return null;
}

fn resolve(comptime T: type, lib: *std.DynLib) error{NotFound}!T {
    var out: T = undefined;
    out.lib = lib.*;
    inline for (std.meta.fields(T)) |f| {
        if (comptime std.mem.eql(u8, f.name, "lib")) continue;
        @field(out, f.name) = out.lib.lookup(f.type, f.name) orelse return error.NotFound;
    }
    return out;
}
