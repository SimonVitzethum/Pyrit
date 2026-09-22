//! Host-Mathematik für 3x4-Transformationen (zeilenweise), gerechnet in f64.

const std = @import("std");

pub const Mat34 = [12]f32;

/// Inverse einer affinen 3x4-Matrix; null, wenn singulär.
pub fn inverse(m: *const Mat34) ?Mat34 {
    const a = [9]f64{ m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10] };
    const t = [3]f64{ m[3], m[7], m[11] };
    const c00 = a[4] * a[8] - a[5] * a[7];
    const c01 = a[5] * a[6] - a[3] * a[8];
    const c02 = a[3] * a[7] - a[4] * a[6];
    const det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    if (!(@abs(det) > 1e-30) or !std.math.isFinite(det)) return null;
    const id = 1.0 / det;
    const r = [9]f64{
        c00 * id,                           (a[2] * a[7] - a[1] * a[8]) * id, (a[1] * a[5] - a[2] * a[4]) * id,
        c01 * id,                           (a[0] * a[8] - a[2] * a[6]) * id, (a[2] * a[3] - a[0] * a[5]) * id,
        c02 * id,                           (a[1] * a[6] - a[0] * a[7]) * id, (a[0] * a[4] - a[1] * a[3]) * id,
    };
    var out: Mat34 = undefined;
    var row: usize = 0;
    while (row < 3) : (row += 1) {
        const r0 = r[row * 3 + 0];
        const r1 = r[row * 3 + 1];
        const r2 = r[row * 3 + 2];
        out[row * 4 + 0] = @floatCast(r0);
        out[row * 4 + 1] = @floatCast(r1);
        out[row * 4 + 2] = @floatCast(r2);
        out[row * 4 + 3] = @floatCast(-(r0 * t[0] + r1 * t[1] + r2 * t[2]));
    }
    return out;
}

/// Welt-AABB des Würfels [0, size]^3 unter m (Arvo).
pub fn transformedBox(m: *const Mat34, size: f32) struct { min: [3]f32, max: [3]f32 } {
    var lo: [3]f32 = undefined;
    var hi: [3]f32 = undefined;
    var row: usize = 0;
    while (row < 3) : (row += 1) {
        lo[row] = m[row * 4 + 3];
        hi[row] = m[row * 4 + 3];
        var col: usize = 0;
        while (col < 3) : (col += 1) {
            const e = m[row * 4 + col] * size;
            lo[row] += @min(e, 0);
            hi[row] += @max(e, 0);
        }
    }
    // kleiner Rand gegen Rundungsfehler im Strahltest
    for (0..3) |i| {
        const eps = 1e-5 * @max(1.0, @max(@abs(lo[i]), @abs(hi[i])));
        lo[i] -= eps;
        hi[i] += eps;
    }
    return .{ .min = lo, .max = hi };
}

test "inverse" {
    const m: Mat34 = .{ 2, 0, 0, 1, 0, 0, -3, 2, 0, 1, 0, 3 };
    const inv = inverse(&m).?;
    // m * inv = I (auf Punkte angewandt)
    const p = [3]f32{ 0.5, -1.25, 7 };
    const q = apply(&m, apply(&inv, p));
    for (0..3) |i| try std.testing.expectApproxEqAbs(p[i], q[i], 1e-5);
    try std.testing.expectEqual(@as(?Mat34, null), inverse(&.{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }));
}

pub fn apply(m: *const Mat34, p: [3]f32) [3]f32 {
    return .{
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    };
}
