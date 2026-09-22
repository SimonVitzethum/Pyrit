//! Kleine Vektor- und Matrixmathematik für Host und GPU.

pub const Vec3 = @Vector(3, f32);

pub inline fn vec3(x: f32, y: f32, z: f32) Vec3 {
    return .{ x, y, z };
}

pub inline fn splat(s: f32) Vec3 {
    return @splat(s);
}

pub inline fn dot(a: Vec3, b: Vec3) f32 {
    return @reduce(.Add, a * b);
}

pub inline fn length(a: Vec3) f32 {
    return @sqrt(dot(a, a));
}

pub inline fn normalize(a: Vec3) Vec3 {
    return a * splat(1.0 / length(a));
}

/// 3x4-Matrix, zeilenweise: m[0..4] erste Zeile, m[4..8] zweite, m[8..12] dritte.
pub inline fn xformPoint(m: *const [12]f32, p: Vec3) Vec3 {
    return .{
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    };
}

pub inline fn xformVector(m: *const [12]f32, v: Vec3) Vec3 {
    return .{
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
    };
}

/// Normale mit der Inversen transformieren: n' = transpose(inv) * n
pub inline fn xformNormal(inv: *const [12]f32, n: Vec3) Vec3 {
    return .{
        inv[0] * n[0] + inv[4] * n[1] + inv[8] * n[2],
        inv[1] * n[0] + inv[5] * n[1] + inv[9] * n[2],
        inv[2] * n[0] + inv[6] * n[1] + inv[10] * n[2],
    };
}

pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}
