//! Warp-Zusammenfassung für Zähler, die viele Threads gleichzeitig hochzählen.
//!
//! Hängen alle Threads eines Kernels an *einen* Zähler an (Strahllisten der
//! Wiederholungs-Wavefront, Belichtungsmessung), serialisiert die GPU die
//! Atomics auf dieser einen Adresse. Gemessen: der Kernel, der je Pixel nur
//! einen Primärstrahl einträgt, brauchte dafür 1,1 ms bei 960x540. Hier
//! zählt je Warp nur ein Thread für alle aktiven, die anderen bekommen ihren
//! Platz über einen Shuffle.
//!
//! Auf CPU und AMD fällt das auf ein gewöhnliches Atomic zurück.

const std = @import("std");
const builtin = @import("builtin");

const nvptx = builtin.cpu.arch == .nvptx64 or builtin.cpu.arch == .nvptx;

inline fn activeMask() u32 {
    return asm volatile ("activemask.b32 %[m];"
        : [m] "=r" (-> u32),
    );
}

inline fn laneId() u32 {
    return asm volatile ("mov.u32 %[l], %%laneid;"
        : [l] "=r" (-> u32),
    );
}

inline fn shuffle(mask: u32, v: u32, src: u32) u32 {
    return asm volatile ("shfl.sync.idx.b32 %[r], %[v], %[s], 31, %[m];"
        : [r] "=r" (-> u32),
        : [v] "r" (v),
          [s] "r" (src),
          [m] "r" (mask),
    );
}

/// Wie @atomicRmw(.Add, 1) auf `counter`, liefert den alten Wert – aber nur
/// ein Atomic je Warp. Die vergebenen Plätze sind lückenlos und eindeutig.
pub inline fn increment(counter: *u32) u32 {
    if (comptime !nvptx) return @atomicRmw(u32, counter, .Add, 1, .monotonic);
    const mask = activeMask();
    const lane = laneId();
    const leader: u32 = @ctz(mask);
    const below = mask & ((@as(u32, 1) << @intCast(lane)) -% 1);
    var base: u32 = 0;
    if (lane == leader) base = @atomicRmw(u32, counter, .Add, @popCount(mask), .monotonic);
    base = shuffle(mask, base, leader);
    return base + @popCount(below);
}

/// Summe über einen vollen Warp, dann ein Atomic je Warp. Ist der Warp nicht
/// vollständig aktiv (Bildrand, Verzweigung), zählt jeder Thread selbst –
/// die Schmetterlingssumme gilt nur, wenn alle 32 Bahnen mitmachen.
pub inline fn add(counter: *u32, v: u32) void {
    if (comptime !nvptx) {
        _ = @atomicRmw(u32, counter, .Add, v, .monotonic);
        return;
    }
    if (activeMask() != 0xFFFF_FFFF) {
        _ = @atomicRmw(u32, counter, .Add, v, .monotonic);
        return;
    }
    var sum = v;
    inline for (.{ 16, 8, 4, 2, 1 }) |off| {
        sum += asm volatile ("shfl.sync.bfly.b32 %[r], %[v], " ++ std.fmt.comptimePrint("{d}", .{off}) ++ ", 31, -1;"
            : [r] "=r" (-> u32),
            : [v] "r" (sum),
        );
    }
    if (laneId() == 0) _ = @atomicRmw(u32, counter, .Add, sum, .monotonic);
}
