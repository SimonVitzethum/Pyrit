//! Pyrit-Geräte-API: dieselben Funktionen laufen auf CPU, NVIDIA (nvptx64) und
//! AMD (amdgcn). Eigene Kernel importieren dieses Modul:
//!
//!     const pyr = @import("pyrit_device");
//!
//!     export fn shadows(scene: *const pyr.Scene, ...) callconv(.nvptx_kernel) void {
//!         const hit = pyr.trace(scene, ray, 0xFFFF_FFFF, pyr.trace_any_hit | pyr.trace_no_attribute);
//!         const lit = hit.instance == pyr.no_hit;
//!     }
//!
//! Den Gerätezeiger auf `Scene` liefert pyr_scene_device(); er bleibt über alle
//! Frames gleich, sein Inhalt wird in Stream-Reihenfolge bei pyr_commit aktualisiert.

pub const types = @import("types.zig");
pub const vec = @import("vec.zig");
pub const dag = @import("dag.zig");
pub const scene = @import("trace.zig");
pub const render = @import("render.zig");
pub const shade = @import("shade.zig");
pub const post = @import("post.zig");
pub const upscale = @import("upscale.zig");
pub const anim = @import("anim.zig");
pub const gbuild = @import("gbuild.zig");
pub const worldedit = @import("worldedit.zig");
pub const replay = @import("replay.zig");
pub const postfx = @import("postfx.zig");
pub const fmath = @import("fmath.zig");
/// RT-Pfad: Teil-DAG-Verfolgung (Host und GPU)
pub const rt = @import("rt.zig");
/// OptiX-Intrinsics (nur nvptx64, nur in OptiX-Programmen verwendbar)
pub const optix = @import("optix.zig");

pub const Ray = types.Ray;
pub const Hit = types.Hit;
pub const Camera = types.Camera;
pub const Scene = types.Scene;
pub const GeometryData = types.GeometryData;
pub const InstanceData = types.InstanceData;
pub const no_hit = types.no_hit;
pub const flt_max = types.flt_max;
pub const trace_any_hit = types.trace_any_hit;
pub const trace_no_attribute = types.trace_no_attribute;
pub const hit_face_mask = types.hit_face_mask;
pub const hit_new = types.hit_new;
pub const hit_no_history = types.hit_no_history;
pub const hit_inside = types.hit_inside;

pub const Vec3 = vec.Vec3;
pub const trace = scene.trace;
pub const traceScene = scene.traceScene;
pub const hitNormal = scene.hitNormal;
