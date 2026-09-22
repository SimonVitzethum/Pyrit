//! Prüft src/optix.zig gegen die Original-OptiX-Header (nicht Teil von Pyrit):
//!
//!   zig build optix-abi-test -Doptix-include=<optix-sdk>/include

const std = @import("std");
const c = @import("optix_c");
const pyrit = @import("pyrit");
const optix = pyrit.optix;
const types = pyrit.types;
const testing = std.testing;

fn expectSameLayout(comptime Z: type, comptime C: type) !void {
    @setEvalBranchQuota(100_000);
    if (@sizeOf(Z) != @sizeOf(C) or @alignOf(Z) != @alignOf(C)) {
        std.debug.print("{s}: Größe {d}/{d}, Ausrichtung {d}/{d}\n", .{ @typeName(Z), @sizeOf(Z), @sizeOf(C), @alignOf(Z), @alignOf(C) });
        return error.LayoutAbweichung;
    }
    inline for (std.meta.fields(Z)) |f| {
        if (@hasField(C, f.name)) {
            if (@offsetOf(Z, f.name) != @offsetOf(C, f.name)) {
                std.debug.print("{s}.{s}: Offset {d}/{d}\n", .{ @typeName(Z), f.name, @offsetOf(Z, f.name), @offsetOf(C, f.name) });
                return error.LayoutAbweichung;
            }
        }
    }
}

test "OptiX-Strukturen" {
    try expectSameLayout(optix.DeviceContextOptions, c.OptixDeviceContextOptions);
    try expectSameLayout(optix.Aabb, c.OptixAabb);
    try expectSameLayout(optix.BuildInputCustomPrimitiveArray, c.OptixBuildInputCustomPrimitiveArray);
    try expectSameLayout(optix.BuildInputInstanceArray, c.OptixBuildInputInstanceArray);
    try expectSameLayout(optix.BuildInput, c.OptixBuildInput);
    try expectSameLayout(optix.MotionOptions, c.OptixMotionOptions);
    try expectSameLayout(optix.AccelBuildOptions, c.OptixAccelBuildOptions);
    try expectSameLayout(optix.AccelBufferSizes, c.OptixAccelBufferSizes);
    try expectSameLayout(optix.AccelEmitDesc, c.OptixAccelEmitDesc);
    try expectSameLayout(optix.ModuleCompileOptions, c.OptixModuleCompileOptions);
    try expectSameLayout(optix.PipelineCompileOptions, c.OptixPipelineCompileOptions);
    try expectSameLayout(optix.PipelineLinkOptions, c.OptixPipelineLinkOptions);
    try expectSameLayout(optix.ProgramGroupSingleModule, c.OptixProgramGroupSingleModule);
    try expectSameLayout(optix.ProgramGroupHitgroup, c.OptixProgramGroupHitgroup);
    try expectSameLayout(optix.ProgramGroupDesc, c.OptixProgramGroupDesc);
    try expectSameLayout(optix.ProgramGroupOptions, c.OptixProgramGroupOptions);
    try expectSameLayout(optix.ShaderBindingTable, c.OptixShaderBindingTable);
    try expectSameLayout(optix.StackSizes, c.OptixStackSizes);

    // OptixInstance (Pyrit: types.RtInstance, eigene Feldnamen)
    try testing.expectEqual(@sizeOf(c.OptixInstance), @sizeOf(types.RtInstance));
    try testing.expectEqual(@offsetOf(c.OptixInstance, "instanceId"), @offsetOf(types.RtInstance, "instance_id"));
    try testing.expectEqual(@offsetOf(c.OptixInstance, "sbtOffset"), @offsetOf(types.RtInstance, "sbt_offset"));
    try testing.expectEqual(@offsetOf(c.OptixInstance, "visibilityMask"), @offsetOf(types.RtInstance, "visibility_mask"));
    try testing.expectEqual(@offsetOf(c.OptixInstance, "flags"), @offsetOf(types.RtInstance, "flags"));
    try testing.expectEqual(@offsetOf(c.OptixInstance, "traversableHandle"), @offsetOf(types.RtInstance, "traversable"));
}

test "OptiX-Funktionstabelle" {
    @setEvalBranchQuota(100_000);
    try testing.expectEqual(@sizeOf(c.OptixFunctionTable), @sizeOf(optix.FunctionTable));
    inline for (std.meta.fields(optix.FunctionTable)) |f| {
        try testing.expectEqual(@offsetOf(c.OptixFunctionTable, f.name), @offsetOf(optix.FunctionTable, f.name));
    }
    try testing.expectEqual(@as(c_int, c.OPTIX_ABI_VERSION), optix.abi_version);
}

test "OptiX-Konstanten" {
    const pairs = .{
        .{ optix.build_input_type_custom_primitives, c.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES },
        .{ optix.build_input_type_instances, c.OPTIX_BUILD_INPUT_TYPE_INSTANCES },
        .{ optix.build_flag_allow_compaction, c.OPTIX_BUILD_FLAG_ALLOW_COMPACTION },
        .{ optix.build_flag_prefer_fast_trace, c.OPTIX_BUILD_FLAG_PREFER_FAST_TRACE },
        .{ optix.build_operation_build, c.OPTIX_BUILD_OPERATION_BUILD },
        .{ optix.property_type_compacted_size, c.OPTIX_PROPERTY_TYPE_COMPACTED_SIZE },
        .{ optix.geometry_flag_disable_anyhit, c.OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT },
        .{ optix.instance_flag_disable_anyhit, c.OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT },
        .{ optix.compile_debug_level_none, c.OPTIX_COMPILE_DEBUG_LEVEL_NONE },
        .{ optix.compile_debug_level_minimal, c.OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL },
        .{ optix.traversable_graph_flag_allow_single_level_instancing, c.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING },
        .{ optix.primitive_type_flags_custom, c.OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM },
        .{ optix.program_group_kind_raygen, c.OPTIX_PROGRAM_GROUP_KIND_RAYGEN },
        .{ optix.program_group_kind_miss, c.OPTIX_PROGRAM_GROUP_KIND_MISS },
        .{ optix.program_group_kind_hitgroup, c.OPTIX_PROGRAM_GROUP_KIND_HITGROUP },
        .{ optix.device_property_rtcore_version, c.OPTIX_DEVICE_PROPERTY_RTCORE_VERSION },
        .{ optix.sbt_record_header_size, c.OPTIX_SBT_RECORD_HEADER_SIZE },
        .{ pyrit.optix_device.ray_flag_disable_anyhit, c.OPTIX_RAY_FLAG_DISABLE_ANYHIT },
        .{ pyrit.optix_device.ray_flag_terminate_on_first_hit, c.OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT },
    };
    inline for (pairs) |p| try testing.expectEqual(@as(i64, p[1]), @as(i64, p[0]));
}
