//! Prüft, dass include/pyrit.h (für C-Aufrufer) genau zur Zig-Implementierung
//! passt: Strukturlayouts, Konstanten und Funktionssignaturen.

const std = @import("std");
const c = @import("c");
const pyrit = @import("pyrit");
const types = pyrit.types;
const api = pyrit.api;
const testing = std.testing;

fn expectSameLayout(comptime Z: type, comptime C: type) !void {
    try testing.expectEqual(@sizeOf(Z), @sizeOf(C));
    try testing.expectEqual(@alignOf(Z), @alignOf(C));
    inline for (std.meta.fields(Z)) |f| {
        if (!@hasField(C, f.name)) {
            std.debug.print("Feld {s}.{s} fehlt im C-Header\n", .{ @typeName(Z), f.name });
            return error.FeldFehlt;
        }
        try testing.expectEqual(@offsetOf(Z, f.name), @offsetOf(C, f.name));
        try testing.expectEqual(@sizeOf(f.type), @sizeOf(@FieldType(C, f.name)));
    }
    try testing.expectEqual(std.meta.fields(Z).len, std.meta.fields(C).len);
}

test "Strukturlayouts" {
    try expectSameLayout(types.Ray, c.PyrRay);
    try expectSameLayout(api.WorldEdit, c.PyrWorldEdit);
    try expectSameLayout(api.PostFx, c.PyrPostFx);
    try expectSameLayout(types.TextureData, c.PyrTextureData);
    try expectSameLayout(types.HitEx, c.PyrHitEx);
    try expectSameLayout(types.Hit, c.PyrHit);
    try expectSameLayout(types.Camera, c.PyrCamera);
    try expectSameLayout(types.GeometryData, c.PyrGeometryData);
    try expectSameLayout(types.InstanceData, c.PyrInstanceData);
    try expectSameLayout(types.Scene, c.PyrScene);
    try expectSameLayout(api.CreateInfo, c.PyrCreateInfo);
    try expectSameLayout(api.Voxel, c.PyrVoxel);
    try expectSameLayout(api.DagInfo, c.PyrDagInfo);
    try expectSameLayout(api.FrameInfo, c.PyrFrameInfo);
    try expectSameLayout(api.Targets, c.PyrTargets);
    try expectSameLayout(types.Material, c.PyrMaterial);
    try expectSameLayout(types.Light, c.PyrLight);
    try expectSameLayout(types.Lighting, c.PyrLighting);
    try expectSameLayout(api.PostInfo, c.PyrPostInfo);
    try expectSameLayout(api.Stats, c.PyrStats);
    try expectSameLayout(types.ChunkKey, c.PyrChunkKey);
    try expectSameLayout(types.WorldGenParams, c.PyrWorldGenParams);
    try expectSameLayout(api.WorldInfo, c.PyrWorldInfo);
    try expectSameLayout(api.WorldStats, c.PyrWorldStats);
    try expectSameLayout(types.FrameGenParams, c.PyrFrameGenParams);
    try expectSameLayout(api.FrameGenInfo, c.PyrFrameGenInfo);
    try expectSameLayout(pyrit.anim_mod.Bone, c.PyrBone);
    try expectSameLayout(types.Keyframe, c.PyrKeyframe);
    try expectSameLayout(pyrit.anim_mod.ActorInfo, c.PyrActorInfo);
}

test "Konstanten" {
    const pairs = .{
        .{ types.no_hit, c.PYR_NO_HIT },
        .{ types.face_neg_z, c.PYR_FACE_NEG_Z },
        .{ types.hit_face_mask, c.PYR_HIT_FACE_MASK },
        .{ types.hit_new, c.PYR_HIT_NEW },
        .{ types.hit_no_history, c.PYR_HIT_NO_HISTORY },
        .{ types.hit_inside, c.PYR_HIT_INSIDE },
        .{ types.trace_any_hit, c.PYR_TRACE_ANY_HIT },
        .{ types.trace_no_attribute, c.PYR_TRACE_NO_ATTRIBUTE },
        .{ types.trace_extended, c.PYR_TRACE_EXTENDED },
        .{ types.trace_skip_transparent, c.PYR_TRACE_SKIP_TRANSPARENT },
        .{ types.material_transparent, c.PYR_MATERIAL_TRANSPARENT },
        .{ types.material_waves, c.PYR_MATERIAL_WAVES },
        .{ api.post_bgra, c.PYR_POST_BGRA },
        .{ types.projection_orthographic, c.PYR_PROJECTION_ORTHOGRAPHIC },
        .{ types.geometry_has_attributes, c.PYR_GEOMETRY_HAS_ATTRIBUTES },
        .{ types.instance_active, c.PYR_INSTANCE_ACTIVE },
        .{ types.instance_keep_history, c.PYR_INSTANCE_KEEP_HISTORY },
        .{ api.version, c.PYR_VERSION },
        .{ api.create_debug, c.PYR_CREATE_DEBUG },
        .{ api.world_sync, c.PYR_WORLD_SYNC },
        .{ types.clip_loop, c.PYR_CLIP_LOOP },
        .{ api.upscaler_none, c.PYR_UPSCALER_NONE },
        .{ api.upscaler_taau, c.PYR_UPSCALER_TAAU },
        .{ api.upscaler_dlss, c.PYR_UPSCALER_DLSS },
        .{ api.upscaler_dlss_rr, c.PYR_UPSCALER_DLSS_RR },
        .{ api.create_no_rt, c.PYR_CREATE_NO_RT },
        .{ api.create_force_rt, c.PYR_CREATE_FORCE_RT },
        .{ api.feature_rt_cores, c.PYR_FEATURE_RT_CORES },
        .{ api.dag_no_attributes, c.PYR_DAG_NO_ATTRIBUTES },
        .{ api.ok, c.PYR_OK },
        .{ api.error_invalid_argument, c.PYR_ERROR_INVALID_ARGUMENT },
        .{ api.error_version, c.PYR_ERROR_VERSION },
        .{ types.max_materials, c.PYR_MAX_MATERIALS },
        .{ types.material_voxel_color, c.PYR_MATERIAL_VOXEL_COLOR },
        .{ types.material_refract, c.PYR_MATERIAL_REFRACT },
        .{ types.max_transparent_layers, c.PYR_MAX_TRANSPARENT_LAYERS },
        .{ types.max_lights, c.PYR_MAX_LIGHTS },
        .{ types.lighting_shadows, c.PYR_LIGHTING_SHADOWS },
        .{ types.lighting_gi, c.PYR_LIGHTING_GI },
        .{ types.lighting_ao, c.PYR_LIGHTING_AO },
        .{ types.lighting_sun_disk, c.PYR_LIGHTING_SUN_DISK },
        .{ types.lighting_reflections, c.PYR_LIGHTING_REFLECTIONS },
        .{ types.lighting_gi_half, c.PYR_LIGHTING_GI_HALF },
        .{ api.post_reset, c.PYR_POST_RESET },
        .{ api.build_host_input, c.PYR_BUILD_HOST_INPUT },
        .{ api.build_editable, c.PYR_BUILD_EDITABLE },
        .{ api.post_no_temporal, c.PYR_POST_NO_TEMPORAL },
        .{ types.tonemap_aces, c.PYR_TONEMAP_ACES },
        .{ types.tonemap_reinhard, c.PYR_TONEMAP_REINHARD },
        .{ types.tonemap_none, c.PYR_TONEMAP_NONE },
        .{ types.tonemap_aces_fitted, c.PYR_TONEMAP_ACES_FITTED },
        .{ types.tonemap_neutral, c.PYR_TONEMAP_NEUTRAL },
    };
    inline for (pairs) |p| try testing.expectEqual(@as(i64, p[0]), @as(i64, p[1]));
    try testing.expectEqual(types.flt_max, c.PYR_FLT_MAX);
}

test "Funktionssignaturen" {
    @setEvalBranchQuota(100_000);
    var count: usize = 0;
    inline for (@typeInfo(c).@"struct".decls) |d| {
        if (comptime !std.mem.startsWith(u8, d.name, "pyr_")) continue;
        const CF = @TypeOf(@field(c, d.name));
        if (@typeInfo(CF) != .@"fn") continue;
        if (!@hasDecl(pyrit, d.name)) {
            std.debug.print("{s} ist im Header deklariert, aber nicht implementiert\n", .{d.name});
            return error.FunktionFehlt;
        }
        const cf = @typeInfo(CF).@"fn";
        const zf = @typeInfo(@TypeOf(@field(pyrit, d.name))).@"fn";
        try testing.expectEqual(cf.params.len, zf.params.len);
        inline for (cf.params, zf.params) |a, b| try testing.expectEqual(@sizeOf(a.type.?), @sizeOf(b.type.?));
        try testing.expectEqual(@sizeOf(cf.return_type.?), @sizeOf(zf.return_type.?));
        count += 1;
    }
    try testing.expectEqual(@as(usize, 69), count);
}
