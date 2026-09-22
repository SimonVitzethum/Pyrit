//! Skelettanimation auf der GPU: je Knochen mit Voxel-Teil ein Thread. Er
//! tastet die Clips des Akteurs ab (lineare Interpolation, Quaternion-Nlerp,
//! Mischen zweier Clips), multipliziert die Kette bis zur Wurzel und schreibt
//! Transformation, Inverse und AABB direkt in die Instanz des aktuellen
//! Frames. Der Vorframe bleibt im zweiten Puffer: Motion Vectors sind exakt.

const std = @import("std");
const types = @import("types.zig");
const fm = @import("fmath.zig");

pub const Mat = [12]f32;
pub const identity: Mat = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };

/// a * b (affin, zeilenweise 3x4)
pub fn mul(a: *const Mat, b: *const Mat) Mat {
    var r: Mat = undefined;
    inline for (0..3) |i| {
        inline for (0..4) |j| {
            var v: f32 = a[i * 4 + 0] * b[0 * 4 + j] + a[i * 4 + 1] * b[1 * 4 + j] + a[i * 4 + 2] * b[2 * 4 + j];
            if (j == 3) v += a[i * 4 + 3];
            r[i * 4 + j] = v;
        }
    }
    return r;
}

pub fn inverse(m: *const Mat) Mat {
    const c00 = m[5] * m[10] - m[6] * m[9];
    const c01 = m[6] * m[8] - m[4] * m[10];
    const c02 = m[4] * m[9] - m[5] * m[8];
    const det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    const id = 1.0 / (if (@abs(det) > 1e-30) det else 1e-30);
    const r = [9]f32{
        c00 * id,                             (m[2] * m[9] - m[1] * m[10]) * id, (m[1] * m[6] - m[2] * m[5]) * id,
        c01 * id,                             (m[0] * m[10] - m[2] * m[8]) * id, (m[2] * m[4] - m[0] * m[6]) * id,
        c02 * id,                             (m[1] * m[8] - m[0] * m[9]) * id,  (m[0] * m[5] - m[1] * m[4]) * id,
    };
    var out: Mat = undefined;
    inline for (0..3) |i| {
        out[i * 4 + 0] = r[i * 3 + 0];
        out[i * 4 + 1] = r[i * 3 + 1];
        out[i * 4 + 2] = r[i * 3 + 2];
        out[i * 4 + 3] = -(r[i * 3 + 0] * m[3] + r[i * 3 + 1] * m[7] + r[i * 3 + 2] * m[11]);
    }
    return out;
}

pub const Trs = struct {
    t: [3]f32,
    s: f32,
    q: [4]f32,

    pub fn matrix(x: Trs) Mat {
        const q = x.q;
        const xx = q[0] * q[0];
        const yy = q[1] * q[1];
        const zz = q[2] * q[2];
        const xy = q[0] * q[1];
        const xz = q[0] * q[2];
        const yz = q[1] * q[2];
        const wx = q[3] * q[0];
        const wy = q[3] * q[1];
        const wz = q[3] * q[2];
        const s = x.s;
        return .{
            (1 - 2 * (yy + zz)) * s, 2 * (xy - wz) * s,       2 * (xz + wy) * s,       x.t[0],
            2 * (xy + wz) * s,       (1 - 2 * (xx + zz)) * s, 2 * (yz - wx) * s,       x.t[1],
            2 * (xz - wy) * s,       2 * (yz + wx) * s,       (1 - 2 * (xx + yy)) * s, x.t[2],
        };
    }
};

fn fromKey(k: *const types.Keyframe) Trs {
    return .{ .t = k.translation, .s = k.scale, .q = k.rotation };
}

/// Mischen: linear, Quaternion normalisiert (kürzester Weg)
pub fn blendTrs(a: Trs, b: Trs, w: f32) Trs {
    var r: Trs = undefined;
    inline for (0..3) |i| r.t[i] = a.t[i] + (b.t[i] - a.t[i]) * w;
    r.s = a.s + (b.s - a.s) * w;
    const d = a.q[0] * b.q[0] + a.q[1] * b.q[1] + a.q[2] * b.q[2] + a.q[3] * b.q[3];
    const sign: f32 = if (d < 0) -1 else 1;
    var len: f32 = 0;
    inline for (0..4) |i| {
        r.q[i] = a.q[i] * (1 - w) + b.q[i] * sign * w;
        len += r.q[i] * r.q[i];
    }
    const inv = 1.0 / @sqrt(@max(len, 1e-30));
    inline for (0..4) |i| r.q[i] *= inv;
    return r;
}

/// acos für x in [0, 1] (Abramowitz/Stegun 4.4.45, Fehler < 1e-4)
inline fn acos01(x: f32) f32 {
    return @sqrt(@max(1 - x, 0)) * (1.5707288 + x * (-0.2121144 + x * (0.0742610 - 0.0187293 * x)));
}

/// Keyframes: linear, Rotation mit Slerp (konstante Winkelgeschwindigkeit)
pub fn interpTrs(a: Trs, b: Trs, w: f32) Trs {
    var r: Trs = undefined;
    inline for (0..3) |i| r.t[i] = a.t[i] + (b.t[i] - a.t[i]) * w;
    r.s = a.s + (b.s - a.s) * w;
    var d = a.q[0] * b.q[0] + a.q[1] * b.q[1] + a.q[2] * b.q[2] + a.q[3] * b.q[3];
    const sign: f32 = if (d < 0) -1 else 1;
    d = @min(@abs(d), 1);
    const theta = acos01(d);
    const st = fm.sin(theta);
    if (st < 1e-3) return blendTrs(a, b, w);
    const wa = fm.sin((1 - w) * theta) / st;
    const wb = fm.sin(w * theta) / st * sign;
    var len: f32 = 0;
    inline for (0..4) |i| {
        r.q[i] = a.q[i] * wa + b.q[i] * wb;
        len += r.q[i] * r.q[i];
    }
    const inv = 1.0 / @sqrt(len);
    inline for (0..4) |i| r.q[i] *= inv;
    return r;
}

/// Lokale Lage eines Knochens in einem Clip zur Zeit t; null ohne Keys
fn sampleClip(p: *const types.AnimParams, clip_index: u32, bone: u32, t_raw: f32) ?Trs {
    if (clip_index == types.no_clip) return null;
    const clip = &@as([*]const types.AnimClip, @ptrFromInt(p.clips))[clip_index];
    if (bone >= clip.bone_count) return null;
    const track = @as([*]const [2]u32, @ptrFromInt(p.tracks))[clip.track_offset + bone];
    if (track[1] == 0) return null;
    const keys = @as([*]const types.Keyframe, @ptrFromInt(p.keys)) + track[0];
    var t = t_raw;
    if (clip.flags & types.clip_loop != 0 and clip.duration > 0) {
        t -= @floor(t / clip.duration) * clip.duration;
    }
    if (track[1] == 1 or t <= keys[0].time) return fromKey(&keys[0]);
    if (t >= keys[track[1] - 1].time) return fromKey(&keys[track[1] - 1]);
    // Binärsuche: keys[lo].time <= t < keys[hi].time
    var lo: u32 = 0;
    var hi: u32 = track[1] - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (keys[mid].time <= t) lo = mid else hi = mid;
    }
    const span = keys[hi].time - keys[lo].time;
    const w = if (span > 0) (t - keys[lo].time) / span else 0;
    return interpTrs(fromKey(&keys[lo]), fromKey(&keys[hi]), w);
}

/// Drehung um die z-Achse (Schwanken in x) mit kleinem Winkel
fn swayMatrix(angle: f32) Mat {
    const c = fm.cos(angle);
    const s = fm.sin(angle);
    return .{ c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0 };
}

fn localMatrix(p: *const types.AnimParams, a: *const types.AnimActor, bone: u32, b: *const types.AnimBone) Mat {
    const time: f32 = @floatCast(p.time);
    const ta = (time - a.start_time) * a.speed;
    const tb = (time - a.blend_start_time) * a.blend_speed;
    const sa = sampleClip(p, a.clip, bone, ta);
    const sb = if (a.blend > 0) sampleClip(p, a.blend_clip, bone, tb) else null;
    var m = b.rest;
    if (sa) |x| {
        m = if (sb) |y| blendTrs(x, y, a.blend).matrix() else x.matrix();
    } else if (sb) |y| {
        m = y.matrix();
    }
    // Wind: jeder Knochen biegt sich um seinen Anteil
    if (a.wind_amplitude != 0 and b.sway != 0) {
        const ang = a.wind_amplitude * b.sway * fm.sin(6.2831853 * a.wind_frequency * time + a.wind_phase);
        const w = swayMatrix(ang);
        m = mul(&m, &w);
    }
    return m;
}

/// Weltlage eines Knochens (Wurzel des Akteurs * Kette)
pub fn boneWorld(p: *const types.AnimParams, a: *const types.AnimActor, bone: u32) Mat {
    const bones = @as([*]const types.AnimBone, @ptrFromInt(p.bones)) + a.bone_offset;
    var m = localMatrix(p, a, bone, &bones[bone]);
    var parent = bones[bone].parent;
    var guard: u32 = 0;
    while (parent >= 0 and guard < 256) : (guard += 1) {
        const pi: u32 = @intCast(parent);
        const lm = localMatrix(p, a, pi, &bones[pi]);
        m = mul(&lm, &m);
        parent = bones[pi].parent;
    }
    return mul(&a.root, &m);
}

pub fn run(p: *const types.AnimParams, i: u32) void {
    if (i >= p.job_count) return;
    const job = @as([*]const types.AnimJob, @ptrFromInt(p.jobs))[i];
    const a = &@as([*]const types.AnimActor, @ptrFromInt(p.actors))[job.actor];
    const bone = &(@as([*]const types.AnimBone, @ptrFromInt(p.bones)) + a.bone_offset)[job.bone];
    const world = boneWorld(p, a, job.bone);
    const m = mul(&world, &bone.part);
    const inst = &@as([*]types.InstanceData, @ptrFromInt(p.instances))[job.instance];
    inst.object_to_world = m;
    inst.world_to_object = inverse(&m);
    // AABB des Würfels [0, size]^3
    const size = bone.part_size;
    var lo: [3]f32 = .{ m[3], m[7], m[11] };
    var hi = lo;
    inline for (0..3) |r| {
        inline for (0..3) |c| {
            const v = m[r * 4 + c] * size;
            if (v < 0) lo[r] += v else hi[r] += v;
        }
    }
    inst.bounds_min = lo;
    inst.bounds_max = hi;
}
