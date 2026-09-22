//! OptiX-Geräteintrinsics als Zig-Inline-Assembler (nur nvptx64).
//!
//! OptiX erkennt Aufrufe der Pseudo-Funktionen `_optix_*` im PTX und ersetzt
//! sie beim Übersetzen des Moduls. Die Namen und Operanden entsprechen der
//! OptiX-ABI (OptiX 9.x); die Header selbst sind nicht Teil von Pyrit.

const vec = @import("vec.zig");
const Vec3 = vec.Vec3;

pub const ray_flag_none: u32 = 0;
pub const ray_flag_disable_anyhit: u32 = 1 << 0;
pub const ray_flag_terminate_on_first_hit: u32 = 1 << 2;
pub const ray_flag_disable_closesthit: u32 = 1 << 3;

inline fn getU32(comptime name: []const u8) u32 {
    return asm ("call (%[r]), " ++ name ++ ", ();"
        : [r] "=r" (-> u32),
    );
}

inline fn getF32(comptime name: []const u8) f32 {
    return asm ("call (%[r]), " ++ name ++ ", ();"
        : [r] "=f" (-> f32),
    );
}

pub inline fn launchIndex() [3]u32 {
    return .{ getU32("_optix_get_launch_index_x"), getU32("_optix_get_launch_index_y"), getU32("_optix_get_launch_index_z") };
}

pub inline fn launchDimensions() [3]u32 {
    return .{ getU32("_optix_get_launch_dimension_x"), getU32("_optix_get_launch_dimension_y"), getU32("_optix_get_launch_dimension_z") };
}

pub inline fn objectRayOrigin() Vec3 {
    return .{ getF32("_optix_get_object_ray_origin_x"), getF32("_optix_get_object_ray_origin_y"), getF32("_optix_get_object_ray_origin_z") };
}

pub inline fn objectRayDirection() Vec3 {
    return .{ getF32("_optix_get_object_ray_direction_x"), getF32("_optix_get_object_ray_direction_y"), getF32("_optix_get_object_ray_direction_z") };
}

pub inline fn rayTmin() f32 {
    return getF32("_optix_get_ray_tmin");
}

pub inline fn rayTmax() f32 {
    return getF32("_optix_get_ray_tmax");
}

pub inline fn primitiveIndex() u32 {
    return getU32("_optix_read_primitive_idx");
}

/// OptixInstance.instanceId (bei Pyrit = Instanz-Index)
pub inline fn instanceId() u32 {
    return getU32("_optix_read_instance_id");
}

pub inline fn hitKind() u32 {
    return getU32("_optix_get_hit_kind");
}

pub inline fn attribute0() u32 {
    return getU32("_optix_get_attribute_0");
}

pub inline fn sbtDataPointer() u64 {
    return asm ("call (%[r]), _optix_get_sbt_data_ptr_64, ();"
        : [r] "=l" (-> u64),
    );
}

pub inline fn getPayload(comptime index: u32) u32 {
    return asm ("call (%[r]), _optix_get_payload, (%[i]);"
        : [r] "=r" (-> u32),
        : [i] "r" (index),
    );
}

pub inline fn setPayload(comptime index: u32, value: u32) void {
    asm volatile ("call _optix_set_payload, (%[i], %[v]);"
        :
        : [i] "r" (index),
          [v] "r" (value),
    );
}

/// Meldet einen Treffer mit t, hitKind (0..127) und einem Attributwert.
pub inline fn reportIntersection1(t: f32, kind: u32, a0: u32) bool {
    const r = asm volatile ("call (%[r]), _optix_report_intersection_1, (%[t], %[k], %[a0]);"
        : [r] "=r" (-> u32),
        : [t] "f" (t),
          [k] "r" (kind),
          [a0] "r" (a0),
    );
    return r != 0;
}

/// optixTrace mit vier Payload-Werten (Ein- und Ausgabe über `p`).
pub inline fn trace4(handle: u64, o: Vec3, d: Vec3, tmin: f32, tmax: f32, mask: u32, flags: u32, sbt_offset: u32, sbt_stride: u32, miss_index: u32, p: *[4]u32) void {
    var r0: u32 = undefined;
    var r1: u32 = undefined;
    var r2: u32 = undefined;
    var r3: u32 = undefined;
    const zero: u32 = 0;
    asm volatile (
        \\{
        \\.reg .b32 q<28>;
        \\call (%[r0], %[r1], %[r2], %[r3], q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, q24, q25, q26, q27), _optix_trace_typed_32, (%[type], %[h], %[ox], %[oy], %[oz], %[dx], %[dy], %[dz], %[tmin], %[tmax], %[time], %[mask], %[flags], %[sbto], %[sbts], %[miss], %[n], %[p0], %[p1], %[p2], %[p3], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z], %[z]);
        \\}
        : [r0] "=r" (r0),
          [r1] "=r" (r1),
          [r2] "=r" (r2),
          [r3] "=r" (r3),
        : [type] "r" (zero),
          [h] "l" (handle),
          [ox] "f" (o[0]),
          [oy] "f" (o[1]),
          [oz] "f" (o[2]),
          [dx] "f" (d[0]),
          [dy] "f" (d[1]),
          [dz] "f" (d[2]),
          [tmin] "f" (tmin),
          [tmax] "f" (tmax),
          [time] "f" (@as(f32, 0)),
          [mask] "r" (mask),
          [flags] "r" (flags),
          [sbto] "r" (sbt_offset),
          [sbts] "r" (sbt_stride),
          [miss] "r" (miss_index),
          [n] "r" (@as(u32, 4)),
          [p0] "r" (p[0]),
          [p1] "r" (p[1]),
          [p2] "r" (p[2]),
          [p3] "r" (p[3]),
          [z] "r" (zero),
    );
    p.* = .{ r0, r1, r2, r3 };
}
