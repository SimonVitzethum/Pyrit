//! Gemeinsamer Teil der OptiX-Module (ohne eigene Einsprünge): Launch-
//! Parameter und der Strahlverfolger über optixTrace.

const std = @import("std");
const pyr = @import("pyrit_device");
const types = pyr.types;
const vec = pyr.vec;
const ox = pyr.optix;
const Vec3 = vec.Vec3;

/// Launch-Parameter im Konstantenspeicher. Zig erlaubt dort nur `extern const`;
/// tools/ptx_fixup.zig macht daraus die Definition, die OptiX erwartet.
pub extern const pyr_rt_params: types.RtParams addrspace(.constant);

pub inline fn params() *const types.RtParams {
    return @addrSpaceCast(&pyr_rt_params);
}

/// Größtes tmax, das an die Hardware geht
const tmax_limit: f32 = 1e16;

/// Die *einzige* Stelle mit optixTrace. Eingebettet stand der Aufruf an jeder
/// der 26 Strahlstellen der Schattierung (Primärstrahl, Deckung, Sonne,
/// Himmel, Lampen, GI, Spiegelung, Wasser, Dunst) als eigene Kopie im
/// OptiX-Programm; das Übersetzen brauchte damit gut 2,5 Minuten und bis zu
/// 10 GB Arbeitsspeicher.
pub noinline fn traceOnce(handle: u64, o: Vec3, d: Vec3, tmin: f32, tmax: f32, mask: u32, flags: u32, p: *[4]u32) void {
    ox.trace4(handle, o, d, tmin, tmax, mask, flags, 0, 1, 0, p);
}

pub const RtTracer = struct {
    handle: u64,
    flags: u32,

    pub inline fn trace(self: RtTracer, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?pyr.scene.TraceHit {
        if (self.handle == 0) return null;
        // p[3] trägt hin die Absicht "durchsichtige Voxel überspringen" und
        // zurück die getroffene Fläche
        var p = [4]u32{ @bitCast(types.flt_max), types.no_hit, 0, @intFromBool((self.flags | flags) & types.trace_skip_transparent != 0) };
        var rf = ox.ray_flag_disable_anyhit;
        if ((self.flags | flags) & types.trace_any_hit != 0) rf |= ox.ray_flag_terminate_on_first_hit;
        traceOnce(self.handle, o, d, tmin, @min(tmax, tmax_limit), ray_mask & 0xFF, rf, &p);
        if (p[1] == types.no_hit) return null;
        const t: f32 = @bitCast(p[0]);
        const inst = &pyr.scene.instances(s)[p[1]];
        return .{
            .t = t,
            .instance = p[1],
            .attribute = p[2],
            .face = p[3],
            .voxel = .{ 0, 0, 0 },
            // Ruheposition aus dem Weltpunkt (für den Motion Vector)
            .p_object = vec.xformPoint(&inst.world_to_object, o + d * vec.splat(t)),
        };
    }
};

