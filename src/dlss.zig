//! NVIDIA DLSS über NGX-CUDA: Super Resolution und Ray Reconstruction (DLSS-RR,
//! ersetzt Denoiser und TAA). Nur mit `zig build -Ddlss-sdk=<Pfad>`; sonst
//! meldet `available` false und Pyrit nutzt das eigene TAAU.
//!
//! NGX liest Eingaben über CUDA-Texturobjekte (formatierte Lesezugriffe).
//! Pyrit kopiert seine linearen Puffer dafür je Frame in CUDA-Arrays (reine
//! Kopien auf dem Stream, keine Synchronisation) und das Ergebnis zurück.

const std = @import("std");
const build_options = @import("build_options");
const cuda = @import("cuda.zig");
const diag = @import("diag.zig");
const Context = @import("context.zig").Context;

const Error = diag.Error;
const fail = diag.fail;

pub const available = build_options.dlss;
const c = if (available) @import("ngx") else struct {};
const Param = if (available) c.NVSDK_NGX_Parameter else anyopaque;
const Handle = if (available) c.NVSDK_NGX_Handle else anyopaque;
const WChar = if (available) c.wchar_t else u32;

pub const Mode = enum { super_resolution, ray_reconstruction };

/// eindeutige Projektkennung von Pyrit für NGX
const project_id = "5b3e1c2a-7d4f-4e8a-9b61-707972697400";

/// NGX-Fehler liegen im Bereich 0xBAD00000
fn ok(r: c_uint) bool {
    return r & 0xFFF0_0000 != 0xBAD0_0000;
}

fn check(r: c_uint, what: []const u8) Error!void {
    if (!ok(r)) return fail(error.NotFound, "DLSS: {s} fehlgeschlagen (NGX 0x{x})", .{ what, r });
}

/// NGX-Laufzeit je Kontext
pub const Ngx = struct {
    params: ?*Param = null,
    sr: bool = false,
    rr: bool = false,

    pub fn init(gpa: std.mem.Allocator) Error!*Ngx {
        if (!available) return fail(error.NotFound, "DLSS ist in diesem Build nicht verfügbar (mit -Ddlss-sdk bauen)", .{});
        const self = gpa.create(Ngx) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        errdefer gpa.destroy(self);
        self.* = .{};

        // Suchpfad für libnvidia-ngx-dlss*.so: SDK-Verzeichnis und Programmverzeichnis
        var wpath: [1024]WChar = undefined;
        const lib_dir: []const u8 = build_options.dlss_lib_dir;
        const dir: []const u8 = if (std.c.getenv("PYRIT_DLSS_PATH")) |e| std.mem.span(e) else lib_dir;
        const n = @min(dir.len, wpath.len - 1);
        for (dir[0..n], 0..) |ch, i| wpath[i] = ch;
        wpath[n] = 0;
        const dot = [_:0]WChar{'.'};
        const paths = [_][*c]const WChar{ &wpath, &dot };
        var fi = std.mem.zeroes(c.NVSDK_NGX_FeatureCommonInfo);
        fi.PathListInfo.Path = &paths;
        fi.PathListInfo.Length = paths.len;
        fi.LoggingInfo.MinimumLoggingLevel = c.NVSDK_NGX_LOGGING_LEVEL_OFF;
        const data = [_:0]WChar{ '/', 't', 'm', 'p' };
        try check(c.NVSDK_NGX_CUDA_Init_with_ProjectID(project_id, c.NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", &data, &fi, c.NVSDK_NGX_Version_API), "NVSDK_NGX_CUDA_Init");
        errdefer _ = c.NVSDK_NGX_CUDA_Shutdown();
        try check(c.NVSDK_NGX_CUDA_GetCapabilityParameters(&self.params), "GetCapabilityParameters");
        var v: c_int = 0;
        if (ok(c.NVSDK_NGX_Parameter_GetI(self.params, c.NVSDK_NGX_Parameter_SuperSampling_Available, &v))) self.sr = v != 0;
        v = 0;
        if (ok(c.NVSDK_NGX_Parameter_GetI(self.params, c.NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &v))) self.rr = v != 0;
        return self;
    }

    pub fn deinit(self: *Ngx, gpa: std.mem.Allocator) void {
        if (!available) return;
        if (self.params) |p| _ = c.NVSDK_NGX_CUDA_DestroyParameters(p);
        _ = c.NVSDK_NGX_CUDA_Shutdown();
        gpa.destroy(self);
    }
};

/// Eingänge (Renderauflösung) und Ausgang (Ausgabeauflösung) als CUDA-Arrays
pub const Slot = enum(u32) { color, albedo, specular, normal, roughness, depth, motion, output };
const slot_count = 8;

pub const Inputs = struct {
    /// [4]f32: verrauschte (RR) bzw. entrauschte (SR) HDR-Farbe
    color: u64,
    /// [4]f32
    albedo: u64,
    /// [4]f32: Normale + Tiefe in w
    normal: u64,
    /// f32 lineare Tiefe
    depth: u64,
    /// f32 Rauheit, [4]f32 spiegelnde Albedo (F0)
    roughness: u64,
    specular: u64,
    /// [2]f32
    motion: u64,
    jitter: [2]f32,
    reset: bool,
    /// Kamera für RR (zeilenweise 4x4)
    world_to_view: [16]f32,
    view_to_clip: [16]f32,
};

pub const Feature = struct {
    mode: Mode,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    handle: ?*Handle = null,
    arrays: [slot_count]cuda.CUarray = .{null} ** slot_count,
    tex: [slot_count]cuda.CUtexObject = .{0} ** slot_count,
    /// Ausgang zusätzlich als Surface (NGX schreibt dorthin)
    out_surf: u64 = 0,
    /// Ergebnis linear (Ausgabeauflösung, [4]f32)
    output: cuda.CUdeviceptr = 0,
    world_to_view: [16]f32 = undefined,
    view_to_clip: [16]f32 = undefined,
    /// Farbeingang halbgenau (Super Resolution hinter dem eigenen Denoiser)
    color_half: bool = false,

    fn format(self: *const Feature, slot: Slot) struct { fmt: c_int, ch: c_uint, bytes: u32 } {
        return switch (slot) {
            .roughness, .depth => .{ .fmt = cuda.CU_AD_FORMAT_FLOAT, .ch = 1, .bytes = 4 },
            .motion => .{ .fmt = cuda.CU_AD_FORMAT_FLOAT, .ch = 2, .bytes = 8 },
            // die Farbe kommt bei SR halbgenau aus der Nachbearbeitung
            .color => if (self.color_half)
                .{ .fmt = cuda.CU_AD_FORMAT_HALF, .ch = 4, .bytes = 8 }
            else
                .{ .fmt = cuda.CU_AD_FORMAT_FLOAT, .ch = 4, .bytes = 16 },
            else => .{ .fmt = cuda.CU_AD_FORMAT_FLOAT, .ch = 4, .bytes = 16 },
        };
    }

    pub fn create(ngx: *Ngx, ctx: *Context, mode: Mode, in_w: u32, in_h: u32, out_w: u32, out_h: u32, color_half: bool) Error!*Feature {
        if (!available) return fail(error.NotFound, "DLSS nicht verfügbar", .{});
        switch (mode) {
            .super_resolution => if (!ngx.sr) return fail(error.NotFound, "DLSS Super Resolution wird von GPU/Treiber nicht angeboten", .{}),
            .ray_reconstruction => if (!ngx.rr) return fail(error.NotFound, "DLSS Ray Reconstruction wird von GPU/Treiber nicht angeboten", .{}),
        }
        const f = ctx.gpa.create(Feature) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        f.* = .{ .mode = mode, .in_w = in_w, .in_h = in_h, .out_w = out_w, .out_h = out_h, .color_half = color_half };
        errdefer f.destroy(ctx);

        const drv = &ctx.drv;
        for (0..slot_count) |i| {
            const slot: Slot = @enumFromInt(i);
            const fm = f.format(slot);
            const is_out = slot == .output;
            const desc = cuda.CUDA_ARRAY3D_DESCRIPTOR{
                .Width = if (is_out) out_w else in_w,
                .Height = if (is_out) out_h else in_h,
                .Depth = 0,
                .Format = fm.fmt,
                .NumChannels = fm.ch,
                .Flags = cuda.CUDA_ARRAY3D_SURFACE_LDST,
            };
            try ctx.check(drv.cuArray3DCreate_v2(&f.arrays[i], &desc), "cuArray3DCreate");
            var rd = std.mem.zeroes(cuda.CUDA_RESOURCE_DESC);
            rd.resType = cuda.CU_RESOURCE_TYPE_ARRAY;
            rd.res.array.hArray = f.arrays[i];
            var td = std.mem.zeroes(cuda.CUDA_TEXTURE_DESC);
            td.addressMode = .{ cuda.CU_TR_ADDRESS_MODE_CLAMP, cuda.CU_TR_ADDRESS_MODE_CLAMP, cuda.CU_TR_ADDRESS_MODE_CLAMP };
            td.filterMode = cuda.CU_TR_FILTER_MODE_POINT;
            try ctx.check(drv.cuTexObjectCreate(&f.tex[i], &rd, &td, null), "cuTexObjectCreate");
            if (is_out) try ctx.check(drv.cuSurfObjectCreate(&f.out_surf, &rd), "cuSurfObjectCreate");
        }
        f.output = try ctx.devAlloc(@as(u64, out_w) * out_h * 16, "DLSS-Ausgabe");

        const p = ngx.params;
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_Input1, ctx.cu_ctx);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_Input2, ctx.stream);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_Width, in_w);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_Height, in_h);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_OutWidth, out_w);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_OutHeight, out_h);
        c.NVSDK_NGX_Parameter_SetI(p, c.NVSDK_NGX_Parameter_PerfQualityValue, perfQuality(in_w, out_w));
        c.NVSDK_NGX_Parameter_SetI(p, c.NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, c.NVSDK_NGX_DLSS_Feature_Flags_IsHDR | c.NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | c.NVSDK_NGX_DLSS_Feature_Flags_AutoExposure);
        c.NVSDK_NGX_Parameter_SetI(p, c.NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
        const feature = switch (mode) {
            .super_resolution => c.NVSDK_NGX_Feature_SuperSampling,
            .ray_reconstruction => blk: {
                c.NVSDK_NGX_Parameter_SetI(p, c.NVSDK_NGX_Parameter_DLSS_Denoise_Mode, c.NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);
                c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_DLSS_Roughness_Mode, c.NVSDK_NGX_DLSS_Roughness_Mode_Unpacked);
                c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_Use_HW_Depth, c.NVSDK_NGX_DLSS_Depth_Type_Linear);
                break :blk c.NVSDK_NGX_Feature_RayReconstruction;
            },
        };
        try check(c.NVSDK_NGX_CUDA_CreateFeature(@intCast(feature), p, &f.handle), "CreateFeature");
        return f;
    }

    fn perfQuality(in_w: u32, out_w: u32) c_int {
        const r = @as(f32, @floatFromInt(in_w)) / @as(f32, @floatFromInt(out_w));
        if (r >= 0.99) return c.NVSDK_NGX_PerfQuality_Value_DLAA;
        if (r >= 0.64) return c.NVSDK_NGX_PerfQuality_Value_MaxQuality;
        if (r >= 0.57) return c.NVSDK_NGX_PerfQuality_Value_Balanced;
        if (r >= 0.49) return c.NVSDK_NGX_PerfQuality_Value_MaxPerf;
        return c.NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    }

    pub fn destroy(self: *Feature, ctx: *Context) void {
        if (available) {
            if (self.handle) |h| _ = c.NVSDK_NGX_CUDA_ReleaseFeature(h);
        }
        for (self.tex) |t| {
            if (t != 0) _ = ctx.drv.cuTexObjectDestroy(t);
        }
        if (self.out_surf != 0) _ = ctx.drv.cuSurfObjectDestroy(self.out_surf);
        for (self.arrays) |a| {
            if (a != null) _ = ctx.drv.cuArrayDestroy(a);
        }
        if (self.output != 0) _ = ctx.drv.cuMemFree_v2(self.output);
        ctx.gpa.destroy(self);
    }

    fn copyHost(self: *Feature, ctx: *Context, slot: Slot, data: []const u8, pitch: u32) Error!void {
        const h = if (slot == .output) self.out_h else self.in_h;
        const cp = cuda.CUDA_MEMCPY2D{
            .srcMemoryType = 1, // CU_MEMORYTYPE_HOST
            .srcHost = data.ptr,
            .srcPitch = pitch,
            .dstMemoryType = cuda.CU_MEMORYTYPE_ARRAY,
            .dstArray = self.arrays[@intFromEnum(slot)],
            .WidthInBytes = pitch,
            .Height = h,
        };
        try ctx.check(ctx.drv.cuMemcpy2DAsync_v2(&cp, ctx.stream), "cuMemcpy2DAsync");
        try ctx.check(ctx.drv.cuStreamSynchronize(ctx.stream), "cuStreamSynchronize");
    }

    fn copyIn(self: *Feature, ctx: *Context, slot: Slot, src: u64) Error!void {
        const fm = self.format(slot);
        const cp = cuda.CUDA_MEMCPY2D{
            .srcMemoryType = cuda.CU_MEMORYTYPE_DEVICE,
            .srcDevice = src,
            .srcPitch = self.in_w * fm.bytes,
            .dstMemoryType = cuda.CU_MEMORYTYPE_ARRAY,
            .dstArray = self.arrays[@intFromEnum(slot)],
            .WidthInBytes = self.in_w * fm.bytes,
            .Height = self.in_h,
        };
        try ctx.check(ctx.drv.cuMemcpy2DAsync_v2(&cp, ctx.stream), "cuMemcpy2DAsync");
    }

    /// Auswerten auf dem Stream des Kontexts; Ergebnis in `output`.
    pub fn evaluate(self: *Feature, ngx: *Ngx, ctx: *Context, in: *const Inputs) Error!void {
        if (!available) return fail(error.NotFound, "DLSS nicht verfügbar", .{});
        try self.copyIn(ctx, .color, in.color);
        try self.copyIn(ctx, .depth, in.depth);
        try self.copyIn(ctx, .motion, in.motion);
        if (self.mode == .ray_reconstruction) {
            try self.copyIn(ctx, .albedo, in.albedo);
            try self.copyIn(ctx, .normal, in.normal);
            try self.copyIn(ctx, .roughness, in.roughness);
            try self.copyIn(ctx, .specular, in.specular);
        }
        const p = ngx.params;
        const T = struct {
            fn ptr(f: *Feature, s: Slot) ?*anyopaque {
                return @ptrCast(&f.tex[@intFromEnum(s)]);
            }
        };
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_Color, T.ptr(self, .color));
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_Output, @ptrCast(&self.out_surf));
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_Depth, T.ptr(self, .depth));
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_MotionVectors, T.ptr(self, .motion));
        // Vorzeichen des Jitters: NVIDIAs Beispiele verwenden den Versatz der
        // Projektion, wir den der Abtastposition. PYRIT_DLSS_JITTER erlaubt den
        // Vergleich (Messung siehe docs/API.md).
        const jsign: f32 = if (std.c.getenv("PYRIT_DLSS_JITTER")) |e| (if (e[0] == '-') -1 else 1) else -1;
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_Jitter_Offset_X, jsign * in.jitter[0]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_Jitter_Offset_Y, jsign * in.jitter[1]);
        c.NVSDK_NGX_Parameter_SetI(p, c.NVSDK_NGX_Parameter_Reset, @intFromBool(in.reset));
        // Vorzeichen der Bewegungsvektoren (Vergleich über PYRIT_DLSS_MV)
        const msign: f32 = if (std.c.getenv("PYRIT_DLSS_MV")) |e| (if (e[0] == '-') -1 else 1) else 1;
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_MV_Scale_X, msign);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_MV_Scale_Y, msign);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, self.in_w);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, self.in_h);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1);
        if (self.mode == .ray_reconstruction) {
            self.world_to_view = in.world_to_view;
            self.view_to_clip = in.view_to_clip;
            c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_DiffuseAlbedo, T.ptr(self, .albedo));
            c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_SpecularAlbedo, T.ptr(self, .specular));
            c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_GBuffer_Normals, T.ptr(self, .normal));
            c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_GBuffer_Roughness, T.ptr(self, .roughness));
            // Matrizen sind laut NVIDIA optional (nur ohne spiegelnde MVs);
            // PYRIT_DLSS_MATRICES=0 schaltet sie zum Vergleich ab
            const with_mat = if (std.c.getenv("PYRIT_DLSS_MATRICES")) |e| e[0] != '0' else true;
            if (with_mat) {
                c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX, @ptrCast(&self.world_to_view));
                c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX, @ptrCast(&self.view_to_clip));
            }
        }
        try check(c.NVSDK_NGX_CUDA_EvaluateFeature_C(self.handle, p, null), "EvaluateFeature");

        const cp = cuda.CUDA_MEMCPY2D{
            .srcMemoryType = cuda.CU_MEMORYTYPE_ARRAY,
            .srcArray = self.arrays[@intFromEnum(Slot.output)],
            .dstMemoryType = cuda.CU_MEMORYTYPE_DEVICE,
            .dstDevice = self.output,
            .dstPitch = self.out_w * 16,
            .WidthInBytes = self.out_w * 16,
            .Height = self.out_h,
        };
        try ctx.check(ctx.drv.cuMemcpy2DAsync_v2(&cp, ctx.stream), "cuMemcpy2DAsync");
    }
};
