//! GPU-Kernel der Demo (nvptx64 -> PTX, eingebettet in die Demo). Pyrit ruft
//! den Generator über PyrWorldInfo.generate auf; die Demo startet dann diesen
//! Kernel auf dem übergebenen Stream.

const std = @import("std");
const builtin = @import("builtin");
const pyr = @import("pyrit_device");
const types = pyr.types;
const terrain = @import("terrain.zig");

pub const panic = std.debug.no_panic;

const kernel: std.builtin.CallingConvention = switch (builtin.cpu.arch) {
    .nvptx, .nvptx64 => .nvptx_kernel,
    .amdgcn => .amdgcn_kernel,
    else => @compileError("Kernel nur für nvptx64 oder amdgcn"),
};

/// Threads je Block (fest, siehe Hinweis zu @workGroupSize in src/gpu_kernels.zig)
pub const block: u32 = 128;

export fn demo_k_generate(g: types.WorldGenParams, p: terrain.Params) callconv(kernel) void {
    const i = @workGroupId(0) * block + @workItemId(0);
    terrain.column(&g, &p, i);
}
