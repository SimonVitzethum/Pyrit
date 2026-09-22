//! Transzendente Funktionen für alle Ziele. Auf NVPTX gibt es in LLVM keine
//! Bibliotheksaufrufe (CUDA nutzt libdevice); dort werden die schnellen
//! Hardware-Näherungen der SFU direkt per PTX verwendet. CPU und AMD nutzen
//! die normalen Zig-Builtins.

const builtin = @import("builtin");
const nvptx = builtin.cpu.arch == .nvptx or builtin.cpu.arch == .nvptx64;

inline fn ptx1(comptime op: []const u8, x: f32) f32 {
    return asm (op ++ " %[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
}

pub inline fn log2(x: f32) f32 {
    return if (nvptx) ptx1("lg2.approx.ftz.f32", x) else @log2(x);
}

pub inline fn exp2(x: f32) f32 {
    return if (nvptx) ptx1("ex2.approx.ftz.f32", x) else @exp2(x);
}

pub inline fn sin(x: f32) f32 {
    return if (nvptx) ptx1("sin.approx.ftz.f32", x) else @sin(x);
}

pub inline fn cos(x: f32) f32 {
    return if (nvptx) ptx1("cos.approx.ftz.f32", x) else @cos(x);
}

pub inline fn tan(x: f32) f32 {
    return sin(x) / cos(x);
}

pub inline fn pow(x: f32, y: f32) f32 {
    return exp2(log2(x) * y);
}

pub inline fn exp(x: f32) f32 {
    return exp2(x * 1.4426950408889634);
}
