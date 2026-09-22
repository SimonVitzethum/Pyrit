//! Runtime-Kernel. Wird für nvptx64-cuda (PTX) und amdgcn-amdhsa übersetzt.
//! Die eigentliche Logik liegt im Modul pyrit_device und ist auf der CPU testbar.

const std = @import("std");
const builtin = @import("builtin");
const pyr = @import("pyrit_device");
const types = pyr.types;

pub const panic = std.debug.no_panic;

const kernel: std.builtin.CallingConvention = switch (builtin.cpu.arch) {
    .nvptx, .nvptx64 => .nvptx_kernel,
    .amdgcn => .amdgcn_kernel,
    else => @compileError("Kernel nur für nvptx64 oder amdgcn"),
};

// Hinweis: @workGroupSize wird bewusst nicht verwendet (erzeugt auf amdgcn
// ungültiges Bitcode); die Blockgrößen sind Konstanten aus types.zig.

/// Zustandsaktualisierung: erst neue Werte, dann Copy-Forward der Einträge,
/// die sich im Vorframe geändert haben. Beide Listen sind disjunkt.
export fn pyr_k_update_instances(
    cur: [*]types.InstanceData,
    prev: [*]const types.InstanceData,
    writes: [*]const types.InstanceWrite,
    write_count: u32,
    copies: [*]const u32,
    copy_count: u32,
) callconv(kernel) void {
    const i = @workGroupId(0) * types.update_block + @workItemId(0);
    if (i < write_count) {
        cur[writes[i].index] = writes[i].data;
    } else if (i < write_count + copy_count) {
        const k = copies[i - write_count];
        cur[k] = prev[k];
    }
}

/// OptixInstance-Feld der IAS aus dem aktuellen Instanzzustand (RT-Cores).
/// Inaktive Instanzen bekommen keinen Traversable und Maske 0.
export fn pyr_k_build_rt_instances(
    cur: [*]const types.InstanceData,
    count: u32,
    gas: [*]const u64,
    out: [*]types.RtInstance,
) callconv(kernel) void {
    const i = @workGroupId(0) * types.update_block + @workItemId(0);
    if (i >= count) return;
    const in = &cur[i];
    const active = in.flags & types.instance_active != 0;
    const handle: u64 = if (active) gas[in.geometry] else 0;
    out[i] = .{
        .transform = in.object_to_world,
        .instance_id = i,
        .sbt_offset = if (active) in.geometry else 0,
        .visibility_mask = if (handle != 0) in.mask & 0xFF else 0,
        .flags = 1 << 2, // OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
        .traversable = handle,
        .pad = .{ 0, 0 },
    };
}

export fn pyr_k_render(p: types.RenderParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.render_block_x + @workItemId(0);
    const y = @workGroupId(1) * types.render_block_y + @workItemId(1);
    if (x >= p.cur.camera.width or y >= p.cur.camera.height) return;

    const s: *const types.Scene = @ptrFromInt(p.scene);
    const r = pyr.render.renderPixel(&p, s, x, y);

    pyr.render.writePixel(&p, @as(u64, y) * p.cur.camera.width + x, &r);
}

export fn pyr_k_trace(p: types.TraceParams) callconv(kernel) void {
    const i = @workGroupId(0) * types.trace_block + @workItemId(0);
    if (i >= p.count) return;
    const s: *const types.Scene = @ptrFromInt(p.scene);
    const ray = @as([*]const types.Ray, @ptrFromInt(p.rays))[i];
    if (p.flags & types.trace_extended != 0) {
        const h = pyr.traceScene(s, ray.origin, ray.direction, ray.tmin, ray.tmax, p.ray_mask, p.flags);
        @as([*]types.HitEx, @ptrFromInt(p.hits))[i] = pyr.scene.extendedHit(s, ray.origin, ray.direction, h);
    } else {
        @as([*]types.Hit, @ptrFromInt(p.hits))[i] = pyr.trace(s, ray, p.ray_mask, p.flags);
    }
}

// ---------------------------------------------------------------------------
// Nachbearbeitung (Logik in src/device/post.zig)
// ---------------------------------------------------------------------------

inline fn postCoords(p: *const types.PostParams) ?[2]u32 {
    const x = @workGroupId(0) * types.post_block + @workItemId(0);
    const y = @workGroupId(1) * types.post_block + @workItemId(1);
    if (x >= p.width or y >= p.height) return null;
    return .{ x, y };
}

export fn pyr_k_temporal(p: types.PostParams) callconv(kernel) void {
    const c = postCoords(&p) orelse return;
    pyr.post.temporal(&p, c[0], c[1]);
}

export fn pyr_k_atrous(p: types.PostParams) callconv(kernel) void {
    const c = postCoords(&p) orelse return;
    pyr.post.atrous(&p, c[0], c[1]);
}

export fn pyr_k_resolve(p: types.PostParams) callconv(kernel) void {
    const c = postCoords(&p) orelse return;
    pyr.post.resolve(&p, c[0], c[1]);
}

// ---------------------------------------------------------------------------
// DAG-Bau (Logik in src/device/gbuild.zig)
// ---------------------------------------------------------------------------

export fn pyr_k_build(p: types.BuildParams) callconv(kernel) void {
    const i = @workGroupId(0) * types.build_block + @workItemId(0);
    pyr.gbuild.run(&p, i);
}

// ---------------------------------------------------------------------------
// Welt-Streaming: eingebauter Geländegenerator (Logik in src/device/worldgen.zig)
// ---------------------------------------------------------------------------

export fn pyr_k_gen_terrain(g: types.WorldGenParams, t: types.TerrainParams) callconv(kernel) void {
    const i = @workGroupId(0) * types.gen_block + @workItemId(0);
    pyr.worldgen.terrainColumn(&g, &t, i);
}

// ---------------------------------------------------------------------------
// Hochskalieren und Frame Generation (Logik in src/device/upscale.zig)
// ---------------------------------------------------------------------------

export fn pyr_k_taau(p: types.UpscaleParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.upscale_block + @workItemId(0);
    const y = @workGroupId(1) * types.upscale_block + @workItemId(1);
    if (x >= p.out_width or y >= p.out_height) return;
    pyr.upscale.taau(&p, x, y);
}

export fn pyr_k_framegen(p: types.FrameGenParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.upscale_block + @workItemId(0);
    const y = @workGroupId(1) * types.upscale_block + @workItemId(1);
    if (x >= p.width or y >= p.height) return;
    pyr.upscale.frameGen(&p, x, y);
}

export fn pyr_k_present(p: types.UpscaleParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.upscale_block + @workItemId(0);
    const y = @workGroupId(1) * types.upscale_block + @workItemId(1);
    if (x >= p.out_width or y >= p.out_height) return;
    pyr.upscale.present(&p, x, y);
}

/// Eingaben für DLSS: lineare Tiefe (Himmel 65504), Rauheit und spiegelnde
/// Albedo (F0). Ohne material-Ziel bleiben Rauheit 0,5 und F0 0,04.
export fn pyr_k_dlss_prepare(
    normal: [*]const [4]f32,
    albedo: [*]const [4]f32,
    material: [*]const [2]f32,
    have_material: u32,
    depth: [*]f32,
    rough: [*]f32,
    specular: [*][4]f32,
    count: u32,
) callconv(kernel) void {
    const i = @workGroupId(0) * types.update_block + @workItemId(0);
    if (i >= count) return;
    depth[i] = @min(normal[i][3], 65504.0);
    const r: f32 = if (have_material != 0) material[i][0] else 0.5;
    const metal: f32 = if (have_material != 0) material[i][1] else 0;
    rough[i] = r;
    const alb = albedo[i];
    var f0: [4]f32 = .{ 0.04, 0.04, 0.04, 1 };
    inline for (0..3) |k| f0[k] = 0.04 + (alb[k] - 0.04) * metal;
    specular[i] = f0;
}

// ---------------------------------------------------------------------------
// Skelettanimation (Logik in src/device/anim.zig)
// ---------------------------------------------------------------------------

export fn pyr_k_animate(p: types.AnimParams) callconv(kernel) void {
    const i = @workGroupId(0) * types.anim_block + @workItemId(0);
    pyr.anim.run(&p, i);
}

/// Indirekte Beleuchtung in halber Auflösung (CUDA-Pfad)
export fn pyr_k_gi(p: types.RenderParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.render_block_x + @workItemId(0);
    const y = @workGroupId(1) * types.render_block_y + @workItemId(1);
    if (x >= p.gi_width or y >= p.gi_height) return;
    const s: *const types.Scene = @ptrFromInt(p.scene);
    pyr.render.giPixel(pyr.render.SoftwareTracer{}, &p, s, x, y);
}

/// Hochskalieren der indirekten Beleuchtung und dazurechnen (beide Pfade)
export fn pyr_k_gi_combine(p: types.RenderParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.render_block_x + @workItemId(0);
    const y = @workGroupId(1) * types.render_block_y + @workItemId(1);
    if (x >= p.cur.camera.width or y >= p.cur.camera.height) return;
    pyr.render.combinePixel(&p, x, y);
}

export fn pyr_k_fg_splat(p: types.FrameGenParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.upscale_block + @workItemId(0);
    const y = @workGroupId(1) * types.upscale_block + @workItemId(1);
    if (x >= p.width or y >= p.height) return;
    pyr.upscale.frameGenSplat(&p, x, y);
}

export fn pyr_k_fg_splat_mv(p: types.FrameGenParams) callconv(kernel) void {
    const x = @workGroupId(0) * types.upscale_block + @workItemId(0);
    const y = @workGroupId(1) * types.upscale_block + @workItemId(1);
    if (x >= p.width or y >= p.height) return;
    pyr.upscale.frameGenSplatMv(&p, x, y);
}
