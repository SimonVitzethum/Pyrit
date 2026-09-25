//! Szene der Demo: Generator auf der GPU, Blockmaterialien, Himmel und Licht.
//! Gemeinsam für das Fenster (demo/main.zig) und den Testrenderer
//! (tools/pyrit_render.zig), damit beide genau dieselbe Welt zeigen.

const std = @import("std");
const pyrit = @import("pyrit");
const pyr = @import("pyrit_device");
const api = pyrit.api;
const types = pyr.types;
const cuda = pyrit.cuda;
pub const terrain = @import("terrain.zig");
const blocktex = @import("blocktex.zig");
const sky_mod = @import("sky.zig");

const demo_ptx: [:0]const u8 = @embedFile("demo_ptx");

fn req(r: api.Result) void {
    if (r != api.ok) std.debug.panic("{s}: {s}", .{ pyrit.pyr_result_string(r), pyrit.pyr_error_message() });
}

/// Geländegenerator als eigener CUDA-Kernel. Pyrit ruft `generate` auf seinem
/// Hintergrund-Thread auf (der Kontext ist dort aktuell); der Kernel läuft
/// auf dem übergebenen Stream, gewartet wird nicht.
pub const Generator = struct {
    drv: *const cuda.Driver,
    module: cuda.CUmodule = null,
    func: cuda.CUfunction = null,
    params: terrain.Params = .{},

    pub fn init(drv: *const cuda.Driver, params: terrain.Params) !Generator {
        var g = Generator{ .drv = drv, .params = params };
        if (drv.cuModuleLoadDataEx(&g.module, demo_ptx.ptr, 0, null, null) != cuda.CUDA_SUCCESS) return error.ModuleLoad;
        if (drv.cuModuleGetFunction(&g.func, g.module, "demo_k_generate") != cuda.CUDA_SUCCESS) return error.ModuleLoad;
        return g;
    }

    pub fn deinit(self: *Generator) void {
        if (self.module != null) _ = self.drv.cuModuleUnload(self.module);
        self.module = null;
    }

    pub fn generate(user: ?*anyopaque, params: *const types.WorldGenParams, stream: ?*anyopaque) callconv(.c) void {
        const self: *Generator = @ptrCast(@alignCast(user.?));
        var g = params.*;
        var p = self.params;
        const threads = g.count << @intCast(2 * g.chunk_log2);
        const b: u32 = 128; // demo/kernels.zig: block
        var args = [_]?*anyopaque{ @ptrCast(&g), @ptrCast(&p) };
        const r = self.drv.cuLaunchKernel(self.func, (threads + b - 1) / b, 1, 1, b, 1, 1, 0, @ptrCast(stream), &args, null);
        if (r != cuda.CUDA_SUCCESS) std.debug.panic("Demo-Generator: {s}", .{self.drv.errorString(r)});
    }

    /// Geländehöhe (Blöcke) für die Kamera
    pub fn height(self: *const Generator, x: f64, z: f64) f32 {
        return terrain.height(&self.params, @floatCast(x), @floatCast(z));
    }

    /// Höhe, wie die feinste Stufe sie erzeugt (Oktaven bis Wellenlänge 2)
    pub fn heightFinest(self: *const Generator, x: f64, z: f64) f32 {
        const xf: f32 = @floatCast(x);
        const zf: f32 = @floatCast(z);
        return terrain.heightWith(&self.params, terrain.climate(&self.params, xf, zf), xf, zf, 2);
    }

    /// Welt mit diesem Generator: Minecraft-Höhenbereich, 1024 Chunks Sicht
    pub fn worldInfo(self: *Generator) api.WorldInfo {
        var wi = std.mem.zeroes(api.WorldInfo);
        wi.generate = generate;
        wi.user = self;
        wi.y_min = terrain.y_min;
        wi.y_max = terrain.y_max;
        wi.view_distance = view_distance;
        return wi;
    }
};

/// 1024 Minecraft-Chunks (je 16 Blöcke) in jede Richtung
pub const view_distance: f32 = 1024 * 16;

/// Blockmaterialien und ihre Texturen (eine 32x32-Kachel je Block)
pub fn materials(ctx: ?*anyopaque, gpa: std.mem.Allocator) !void {
    const tex = try gpa.alloc(u8, blocktex.size * blocktex.size * 4);
    defer gpa.free(tex);
    const ntex = try gpa.alloc(u8, blocktex.size * blocktex.size * 4);
    defer gpa.free(ntex);
    // rough: Rauheit; bump: Stärke des Reliefs aus der Kachel. Laub hat
    // einen leichten Glanz (wachsige Blätter), Schnee einen feinen.
    const kinds = [_]struct { k: blocktex.Kind, m: u32, rough: f32, bump: f32 }{
        .{ .k = .grass, .m = terrain.mat_grass, .rough = 0.9, .bump = 1.6 },
        .{ .k = .rock, .m = terrain.mat_stone, .rough = 0.8, .bump = 2.2 },
        .{ .k = .sand, .m = terrain.mat_sand, .rough = 0.95, .bump = 1.0 },
        .{ .k = .snow, .m = terrain.mat_snow, .rough = 0.6, .bump = 2.2 },
        .{ .k = .wood, .m = terrain.mat_wood, .rough = 0.85, .bump = 2.4 },
        .{ .k = .leaves, .m = terrain.mat_leaves, .rough = 0.88, .bump = 1.4 },
        .{ .k = .dirt, .m = terrain.mat_dirt, .rough = 0.95, .bump = 1.6 },
        .{ .k = .gravel, .m = terrain.mat_gravel, .rough = 0.85, .bump = 2.4 },
    };
    for (kinds) |e| {
        blocktex.make(e.k, tex);
        var idx: u32 = 0;
        req(pyrit.pyr_texture_create(@ptrCast(ctx), blocktex.size, blocktex.size, tex.ptr, &idx));
        // Oberflächenrelief aus derselben Kachel
        blocktex.normalMap(tex, ntex, e.bump);
        var nidx: u32 = 0;
        req(pyrit.pyr_texture_create(@ptrCast(ctx), blocktex.size, blocktex.size, ntex.ptr, &nidx));
        var m: types.Material = undefined;
        pyrit.pyr_material_default(&m);
        m.flags = types.material_voxel_color;
        m.roughness = e.rough;
        m.texture = idx;
        m.normal_texture = nidx;
        m.normal_strength = 1;
        m.texture_scale = 1; // eine Kachel je Block
        if (e.k == .leaves) {
            // durchbrochen: man sieht durch die Krone, und die Sonne wirft
            // Lichtflecken auf den Boden
            m.flags |= types.material_cutout;
            m.subsurface = 0.5; // Laub leuchtet von hinten durch
            m.subsurface_color = .{ 0.5, 0.9, 0.3 };
        }
        if (e.k == .snow) m.clearcoat = 0.25;
        if (e.k == .grass) {
            // Seiten des Grasblocks: Erde mit Grasrand, Farbe aus der Kachel
            blocktex.grassSide(tex);
            var sidx: u32 = 0;
            req(pyrit.pyr_texture_create(@ptrCast(ctx), blocktex.size, blocktex.size, tex.ptr, &sidx));
            m.side_texture = sidx;
            m.side_color = .{ 1, 1, 1 };
        }
        req(pyrit.pyr_material_set(@ptrCast(ctx), e.m, &m));
    }
    const w = water();
    req(pyrit.pyr_material_set(@ptrCast(ctx), terrain.mat_water, &w));
}

pub fn water() types.Material {
    var m: types.Material = undefined;
    pyrit.pyr_material_default(&m);
    m.flags = types.material_voxel_color | types.material_transparent | types.material_refract | types.material_waves;
    m.roughness = 0.04;
    m.ior = 1.33;
    // Absorption: flaches Wasser zeigt den Grund, tieferes wird schnell
    // blaugrün undurchsichtig (bei 0,05 lag die Treppe des Meeresgrunds bis
    // in große Tiefe kontrastreich sichtbar)
    m.density = 0.08;
    m.opacity = 0.03;
    // Große, ruhige Wellen: feine sprenkeln in der Ferne, weil viele davon
    // auf ein Pixel fallen und die Normale von Pixel zu Pixel springt.
    m.wave_height = 0.5;
    m.wave_length = 48;
    m.wave_speed = 0.18;
    m.clearcoat = 1; // nasse, lackartige Oberfläche
    m.clearcoat_roughness = 0.03;
    return m;
}

/// Medium um die Kamera, solange sie unter Wasser ist: dieselbe Absorption
/// wie das Wasser, dazu blaugrünes Streulicht (Tageslicht, das im Wasser
/// gestreut wird). `top`: Höhe der Oberfläche relativ zum Render-Ursprung.
pub fn underwater(l: *types.Lighting, on: bool, top: f32) void {
    if (!on) {
        l.camera_medium_density = 0;
        return;
    }
    const w = water();
    l.camera_medium_density = w.density;
    // Farbe des Wassers (Attribut in terrain.zig), linear
    l.camera_medium_color = .{ 0.013, 0.40, 0.64 };
    l.camera_medium_scatter = .{ 0.004, 0.035, 0.05 };
    l.camera_medium_top = top;
}

/// Welteinheiten je Wiederholung der Wolkenschatten
pub const cloud_shadow_scale: f32 = 6000;

/// Wolkenschatten an der Welt festhalten, wenn der Render-Ursprung springt
pub fn cloudShadowOffset(l: *types.Lighting, origin: [3]f64) void {
    const sc: f64 = cloud_shadow_scale;
    l.sun_shadow_offset = .{ @floatCast(@mod(origin[0], sc)), @floatCast(@mod(origin[2], sc)) };
}

pub const Sun = struct {
    /// Abstand vom Zenit und Richtung um die Hochachse (Radiant)
    theta: f32 = 0.9,
    phi: f32 = 3.0,
    /// Winkelradius der Scheibe: die echte Sonne (0,27°). Mit 1,4° waren alle
    /// Halbschatten gleich breit und weich, als wären sie verwischt.
    radius: f32 = 0.0047,
};

/// Licht der Demo: Himmel mit Sonne als Umgebungskarte, zwei GI-Bounces, Nebel
pub fn lighting(ctx: ?*anyopaque, gpa: std.mem.Allocator, sun: Sun) !types.Lighting {
    var l: types.Lighting = undefined;
    pyrit.pyr_lighting_default(&l);
    l.gi_distance = 96;
    l.gi_bounces = 2;
    // Dunst: dünn, mit der Höhe abnehmend, nach vorn streuend (Lichtschächte
    // zur Sonne). Er reicht bis zum Horizont und gibt fernen Bergen die
    // Luftperspektive; leicht bläulich wie echte Luft.
    l.fog_density = 0.0011;
    l.fog_height = 64;
    l.fog_falloff = 0.012;
    l.fog_color = .{ 0.82, 0.9, 1.0 };
    l.fog_anisotropy = 0.6;
    // Richtung zur Sonne, daraus Himmel und Sonnenfarbe (Einfachstreuung)
    l.sun_direction = .{ @sin(sun.theta) * @cos(sun.phi), @cos(sun.theta), @sin(sun.theta) * @sin(sun.phi) };
    // 1024 x 512: gröber zeichnet die Karte die Wolkenränder als Treppen
    const sk = try sky_mod.build(gpa, 1024, 512, l.sun_direction, 2.6);
    defer gpa.free(sk.env);
    req(pyrit.pyr_environment_set(@ptrCast(ctx), 1024, 512, &sk.env[0]));
    l.env_intensity = 1;
    // Die Sonne ist ein eigenes Licht, nicht Teil der Karte: so bekommt jeder
    // Punkt jeden Frame einen Schattenstrahl in die Sonnenscheibe, und nur der
    // Halbschatten rauscht. Steckt sie in der Karte, zieht die Abtastung
    // zufällig mal die Sonne, mal den Himmel – dann rauscht auch voll
    // besonnter Boden, und genau dieses Rauschen muss bei Bewegung die
    // zeitliche Mittelung wegschaffen.
    l.sun_angular_radius = sun.radius;
    l.sun_color = sk.sun_color;
    // Wolkenschatten: dieselbe Art Deckung wie die Wolken am Himmel, auf
    // 1500 m Höhe, eine Kachel über 6 km
    const cmap = try sky_mod.cloudShadowMap(gpa, 512);
    defer gpa.free(cmap);
    req(pyrit.pyr_texture_create(@ptrCast(ctx), 512, 512, cmap.ptr, &l.sun_shadow_texture));
    l.sun_shadow_height = 1500;
    l.sun_shadow_scale = cloud_shadow_scale;
    l.sun_shadow_strength = 0.8;
    return l;
}

/// Nachbearbeitung der Demo (ohne Ausgabepuffer)
pub fn post() api.PostInfo {
    var p = std.mem.zeroes(api.PostInfo);
    p.denoise_iterations = 4;
    p.exposure = 1.0;
    p.clamp_sigma = 1.5;
    p.temporal_alpha = 0.05;
    p.tonemap = types.tonemap_neutral;
    return p;
}

/// Bildeffekte der Demo. Ohne Tiefenschärfe und Bewegungsunschärfe: alles
/// bleibt scharf, nah wie fern (zuschaltbar über PyrPostFx).
pub fn fx() api.PostFx {
    var f = std.mem.zeroes(api.PostFx);
    f.flags = api.postfx_bloom | api.postfx_auto_exposure | api.postfx_grade;
    f.bloom_strength = 0.06;
    f.bloom_threshold = 1.2;
    f.dof_strength = 2.5;
    f.motion_blur_scale = 0.4;
    f.contrast = 1.06;
    f.saturation = 0.92;
    f.temperature = 0;
    // Die Automatik legt das geometrische Mittel auf 18 % Grau. Himmel,
    // Wolken und Wasser heben dieses Mittel an; ohne Korrektur lag besonntes
    // Gras dann bei sRGB 96 und das Land wirkte trüb. Die neutrale Kurve
    // rollt die hellen Flächen weich ab, deshalb darf es knapp eine halbe
    // Blende heller sein.
    f.exposure_compensation = 0.4;
    return f;
}

/// Kontext mit genug Platz für eine Welt mit 1024 Chunks Sichtweite
pub fn createInfo(flags: u32) api.CreateInfo {
    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = flags;
    ci.max_geometries = 65536;
    ci.max_instances = 65536;
    ci.node_pool_bytes = 512 << 20;
    ci.leaf_pool_bytes = 512 << 20;
    ci.attribute_pool_bytes = 512 << 20;
    // PYRIT_LOG=<Stufe>: Meldungen bis zu dieser Stufe auf stderr
    if (std.c.getenv("PYRIT_LOG") != null) ci.log = logToStderr;
    return ci;
}

fn logToStderr(_: ?*anyopaque, level: i32, msg: [*:0]const u8) callconv(.c) void {
    const max: i32 = if (std.c.getenv("PYRIT_LOG")) |e| (std.fmt.parseInt(i32, std.mem.span(e), 10) catch 3) else 0;
    if (level <= max) std.debug.print("[pyrit] {s}\n", .{msg});
}
