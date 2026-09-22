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

/// arcsin über eine Polynomnäherung (NVPTX kennt keine Umkehrfunktionen;
/// der Fehler liegt unter 1e-4 und reicht für Richtungen und Winkel).
pub inline fn asin(x: f32) f32 {
    const a = @abs(@min(@max(x, -1), 1));
    // Näherung nach Abramowitz/Stegun 4.4.45
    const r = @sqrt(@max(1 - a, 0)) *
        (1.5707288 - 0.2121144 * a + 0.0742610 * a * a - 0.0187293 * a * a * a);
    const v = 1.5707963267948966 - r;
    return if (x < 0) -v else v;
}

pub inline fn acos(x: f32) f32 {
    return 1.5707963267948966 - asin(x);
}

/// arctan(y/x) mit richtigem Quadranten, Ergebnis in (-pi, pi]
pub inline fn atan2(y: f32, x: f32) f32 {
    const ax = @abs(x);
    const ay = @abs(y);
    const big = @max(ax, ay);
    if (big <= 0) return 0;
    const small = @min(ax, ay);
    const z = small / big;
    // arctan(z) für z in [0, 1], Polynomnäherung
    const z2 = z * z;
    var r = z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))));
    if (ay > ax) r = 1.5707963267948966 - r;
    if (x < 0) r = 3.1415926535897932 - r;
    return if (y < 0) -r else r;
}
