//! pyrit-render: rendert ein Bild mit der Pyrit-API (GPU) und speichert es als PPM.
//!
//!   zig build render -- [--vox datei.vox] [--out bild.ppm] [--size 1280x720]
//!                       [--frames 64] [--no-rt] [--no-gi]
//!   zig build render -- --world [--scale 2] [--fg] [--dlss|--rr] [--profile]
//!
//! Ohne --vox wird eine Landschaft erzeugt. Die Voxel werden auf dem Host
//! erzeugt bzw. gelesen; Bau, Rendern, Akkumulation und Tonemapping laufen
//! auf der GPU.

const std = @import("std");
const pyrit = @import("pyrit");
const pyr = @import("pyrit_device");
const api = pyrit.api;
const types = pyr.types;
const cuda = pyrit.cuda;
const blocktex = @import("blocktex.zig");

var drv: cuda.Driver = undefined;

fn cu(r: cuda.CUresult) void {
    if (r != cuda.CUDA_SUCCESS) std.debug.panic("CUDA: {s}", .{drv.errorString(r)});
}

fn req(r: api.Result) void {
    if (r != api.ok) std.debug.panic("{s}: {s}", .{ pyrit.pyr_result_string(r), pyrit.pyr_error_message() });
}

fn logCb(_: ?*anyopaque, level: i32, message: [*:0]const u8) callconv(.c) void {
    std.debug.print("[pyrit {d}] {s}\n", .{ level, message });
}

fn devAlloc(bytes: usize) u64 {
    var p: cuda.CUdeviceptr = 0;
    cu(drv.cuMemAlloc_v2(&p, bytes));
    return p;
}

fn voxel(material: u32, r: u32, g: u32, b: u32) u32 {
    return pyrit.pyr_voxel_attribute(material, r, g, b);
}

fn hash(x: i32, z: i32) f32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 374761393 +% @as(u32, @bitCast(z)) *% 668265263;
    h = (h ^ (h >> 13)) *% 1274126177;
    return @as(f32, @floatFromInt(h & 0xFFFF)) / 65535.0;
}

fn noise(x: f32, z: f32) f32 {
    const xi: i32 = @intFromFloat(@floor(x));
    const zi: i32 = @intFromFloat(@floor(z));
    const fx = x - @floor(x);
    const fz = z - @floor(z);
    const u = fx * fx * (3 - 2 * fx);
    const v = fz * fz * (3 - 2 * fz);
    const a = hash(xi, zi);
    const b = hash(xi + 1, zi);
    const c = hash(xi, zi + 1);
    const d = hash(xi + 1, zi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/// Hügellandschaft mit Wasser, Schnee, Bäumen und leuchtenden Kristallen
fn landscape(gpa: std.mem.Allocator, n: u32) ![]api.Voxel {
    var list: std.ArrayList(api.Voxel) = .empty;
    const nf: f32 = @floatFromInt(n);
    const water: i32 = @intFromFloat(nf * 0.18);
    for (0..n) |zi| for (0..n) |xi| {
        const x: f32 = @floatFromInt(xi);
        const z: f32 = @floatFromInt(zi);
        var h = noise(x / 64, z / 64) * 0.6 + noise(x / 24, z / 24) * 0.3 + noise(x / 8, z / 8) * 0.1;
        h = h * h * nf * 0.55;
        const top: i32 = @intFromFloat(h);
        var y: i32 = 0;
        while (y <= @max(top, water)) : (y += 1) {
            const vx: i32 = @intCast(xi);
            const vz: i32 = @intCast(zi);
            const a: u32 = if (y > top)
                voxel(2, 40, 90, 160) // Wasser
            else if (y == top and top > @as(i32, @intFromFloat(nf * 0.42)))
                voxel(0, 240, 240, 245) // Schnee
            else if (y == top and top > water + 1)
                voxel(0, 70, 140, 50) // Gras
            else if (y == top)
                voxel(0, 200, 190, 140) // Sand
            else if (y > top - 4)
                voxel(0, 110, 80, 55) // Erde
            else
                voxel(0, 120, 120, 125); // Stein
            // nur Oberfläche und etwas darunter speichern (Rest ist unsichtbar)
            if (y >= @min(top, water) - 6) try list.append(gpa, .{ .x = vx, .y = y, .z = vz, .attribute = a });
        }
        // Bäume
        if (top > water + 2 and top < @as(i32, @intFromFloat(nf * 0.35)) and hash(@intCast(xi * 7), @intCast(zi * 13)) > 0.992) {
            var t: i32 = 1;
            while (t < 9) : (t += 1) try list.append(gpa, .{ .x = @intCast(xi), .y = top + t, .z = @intCast(zi), .attribute = voxel(0, 90, 60, 35) });
            var dz: i32 = -3;
            while (dz <= 3) : (dz += 1) {
                var dx: i32 = -3;
                while (dx <= 3) : (dx += 1) {
                    var dy: i32 = 6;
                    while (dy <= 11) : (dy += 1) {
                        if (dx * dx + dz * dz + (dy - 8) * (dy - 8) <= 10) {
                            const px = @as(i32, @intCast(xi)) + dx;
                            const pz = @as(i32, @intCast(zi)) + dz;
                            if (px >= 0 and pz >= 0 and px < n and pz < n)
                                try list.append(gpa, .{ .x = px, .y = top + dy, .z = pz, .attribute = voxel(0, 45, 110, 40) });
                        }
                    }
                }
            }
        }
        // Kristalle (emittierend)
        if (top > water + 1 and hash(@intCast(xi * 31), @intCast(zi * 17)) > 0.9985) {
            var t: i32 = 1;
            while (t < 5) : (t += 1) try list.append(gpa, .{ .x = @intCast(xi), .y = top + t, .z = @intCast(zi), .attribute = voxel(1, 255, 160, 60) });
        }
    };
    return list.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var vox_path: ?[]const u8 = null;
    var out_path: []const u8 = "pyrit.ppm";
    var w: u32 = 1280;
    var h: u32 = 720;
    var frames: u32 = 64;
    var flags: u32 = 0;
    var gi = true;
    var world_mode = false;
    var scale: u32 = 1;
    var fg = false;
    var upscaler: u32 = api.upscaler_auto;
    var profile = false;
    var voxel_px: f32 = 0;
    var budget_mib: u32 = 0;
    var denoise: u32 = 2;
    var clamp_sigma: f32 = 1.5;
    var coarse_secondary = false;
    var gi_distance: f32 = 0;
    var half_gi = false;
    // Kameradrehung je Frame in Radiant (wie Mausblick); 0 = starre Blickrichtung
    var turn: f32 = 0;
    var flicker = false;
    var edit_test = false;
    var fx_flags: u32 = 0;
    var materials = false;
    var use_env = false;
    var env_flat = false;
    var firefly: f32 = 0;
    var alpha: f32 = 0;
    var no_shadows = false;
    var no_jitter = false;
    var super: u32 = 1;
    var chunks_per_update: u32 = 0;
    var bounces: u32 = 0;
    var fog: f32 = 0;
    var async_post = false;
    var chunk_capacity: u32 = 0;
    var edit_load = false;
    var edit_stream = false;
    var edit_file: ?[]const u8 = null;
    // Sichtweite in Grundvoxeln (0 = Vorgabe 16384 = 1024 Minecraft-Chunks)
    var view_distance: f32 = 0;
    var sea_level: f32 = 0;
    var rt_leaf: u32 = 0;
    var static_cam = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--vox") and i + 1 < args.len) {
            i += 1;
            vox_path = args[i];
        } else if (std.mem.eql(u8, a, "--out") and i + 1 < args.len) {
            i += 1;
            out_path = args[i];
        } else if (std.mem.eql(u8, a, "--size") and i + 1 < args.len) {
            i += 1;
            var it = std.mem.splitScalar(u8, args[i], 'x');
            w = try std.fmt.parseInt(u32, it.next() orelse "1280", 10);
            h = try std.fmt.parseInt(u32, it.next() orelse "720", 10);
        } else if (std.mem.eql(u8, a, "--frames") and i + 1 < args.len) {
            i += 1;
            frames = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-rt")) {
            flags |= api.create_no_rt;
        } else if (std.mem.eql(u8, a, "--no-gi")) {
            gi = false;
        } else if (std.mem.eql(u8, a, "--world")) {
            world_mode = true;
        } else if (std.mem.eql(u8, a, "--scale") and i + 1 < args.len) {
            i += 1;
            scale = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--fg")) {
            fg = true;
        } else if (std.mem.eql(u8, a, "--dlss")) {
            upscaler = api.upscaler_dlss;
        } else if (std.mem.eql(u8, a, "--rr")) {
            upscaler = api.upscaler_dlss_rr;
        } else if (std.mem.eql(u8, a, "--profile")) {
            profile = true;
        } else if (std.mem.eql(u8, a, "--voxel-px") and i + 1 < args.len) {
            i += 1;
            voxel_px = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--denoise") and i + 1 < args.len) {
            i += 1;
            denoise = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--gi-distance") and i + 1 < args.len) {
            i += 1;
            gi_distance = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--static")) {
            static_cam = true;
        } else if (std.mem.eql(u8, a, "--rt-leaf") and i + 1 < args.len) {
            i += 1;
            rt_leaf = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--sea") and i + 1 < args.len) {
            i += 1;
            sea_level = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--turn") and i + 1 < args.len) {
            i += 1;
            turn = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--view") and i + 1 < args.len) {
            i += 1;
            view_distance = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--chunk-capacity") and i + 1 < args.len) {
            i += 1;
            chunk_capacity = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--bounces") and i + 1 < args.len) {
            i += 1;
            bounces = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--fog") and i + 1 < args.len) {
            i += 1;
            fog = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--async-post")) {
            async_post = true;
        } else if (std.mem.eql(u8, a, "--chunks") and i + 1 < args.len) {
            i += 1;
            chunks_per_update = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--super") and i + 1 < args.len) {
            i += 1;
            super = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-jitter")) {
            no_jitter = true;
        } else if (std.mem.eql(u8, a, "--no-taau")) {
            upscaler = api.upscaler_none;
        } else if (std.mem.eql(u8, a, "--no-shadows")) {
            no_shadows = true;
        } else if (std.mem.eql(u8, a, "--alpha") and i + 1 < args.len) {
            i += 1;
            alpha = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--clamp-fire") and i + 1 < args.len) {
            i += 1;
            firefly = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--env-flat")) {
            use_env = true;
            env_flat = true;
        } else if (std.mem.eql(u8, a, "--env")) {
            use_env = true;
        } else if (std.mem.eql(u8, a, "--materials")) {
            materials = true;
        } else if (std.mem.eql(u8, a, "--fx")) {
            fx_flags |= api.postfx_bloom | api.postfx_auto_exposure | api.postfx_grade;
        } else if (std.mem.eql(u8, a, "--bloom")) {
            fx_flags |= api.postfx_bloom;
        } else if (std.mem.eql(u8, a, "--dof")) {
            fx_flags |= api.postfx_dof | api.postfx_autofocus | api.postfx_dof_far_only;
        } else if (std.mem.eql(u8, a, "--motion-blur")) {
            fx_flags |= api.postfx_motion_blur;
        } else if (std.mem.eql(u8, a, "--auto-exposure")) {
            fx_flags |= api.postfx_auto_exposure;
        } else if (std.mem.eql(u8, a, "--grade")) {
            fx_flags |= api.postfx_grade;
        } else if (std.mem.eql(u8, a, "--edit")) {
            edit_test = true;
        } else if (std.mem.eql(u8, a, "--edit-file") and i + 1 < args.len) {
            i += 1;
            edit_file = args[i];
        } else if (std.mem.eql(u8, a, "--edit-stream")) {
            edit_stream = true;
        } else if (std.mem.eql(u8, a, "--edit-load")) {
            edit_load = true;
        } else if (std.mem.eql(u8, a, "--flicker")) {
            flicker = true;
        } else if (std.mem.eql(u8, a, "--half-gi")) {
            half_gi = true;
        } else if (std.mem.eql(u8, a, "--coarse-gi")) {
            coarse_secondary = true;
        } else if (std.mem.eql(u8, a, "--clamp") and i + 1 < args.len) {
            i += 1;
            clamp_sigma = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--budget") and i + 1 < args.len) {
            i += 1;
            budget_mib = try std.fmt.parseInt(u32, args[i], 10);
        } else {
            std.debug.print("unbekanntes Argument: {s}\n", .{a});
            return error.Usage;
        }
    }

    drv = cuda.Driver.load() catch return error.NoCuda;
    cu(drv.cuInit(0));
    var dev: cuda.CUdevice = 0;
    cu(drv.cuDeviceGet(&dev, 0));
    var cu_ctx: cuda.CUcontext = null;
    cu(drv.cuDevicePrimaryCtxRetain(&cu_ctx, dev));
    cu(drv.cuCtxSetCurrent(cu_ctx));

    var ci = std.mem.zeroes(api.CreateInfo);
    ci.struct_size = @sizeOf(api.CreateInfo);
    ci.version = api.version;
    ci.flags = flags;
    if (async_post) ci.flags |= api.create_async_post;
    if (world_mode) {
        // feines LOD braucht viele Chunks
        if (std.c.getenv("PYRIT_LOG") != null or std.c.getenv("PYRIT_WORLD_PROFILE") != null) ci.log = logCb;
    ci.max_geometries = 65536;
        ci.max_instances = 65536;
        ci.node_pool_bytes = 512 << 20;
        ci.leaf_pool_bytes = 512 << 20;
        ci.attribute_pool_bytes = 512 << 20;
    }
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    defer pyrit.pyr_destroy(@ptrCast(ctx));

    if (world_mode) return renderWorld(init, ctx, w, h, frames, gi, out_path, scale, fg, upscaler, profile, voxel_px, budget_mib, denoise, clamp_sigma, coarse_secondary, gi_distance, half_gi, sea_level, rt_leaf, static_cam, turn, flicker, view_distance, edit_test, edit_load, edit_file, chunk_capacity, edit_stream, fx_flags, materials, use_env, env_flat, firefly, alpha, no_shadows, no_jitter, super, chunks_per_update, bounces, fog, async_post);

    // Szene
    var voxels: []api.Voxel = undefined;
    var log2: u32 = 8;
    if (vox_path) |path| {
        const data = try std.Io.Dir.cwd().readFileAlloc(init.io, path, gpa, .unlimited);
        defer gpa.free(data);
        var count: u32 = 0;
        req(pyrit.pyr_vox_parse(data.ptr, data.len, 0, null, &count, &log2));
        voxels = try gpa.alloc(api.Voxel, count);
        req(pyrit.pyr_vox_parse(data.ptr, data.len, 0, voxels.ptr, &count, &log2));
    } else {
        voxels = try landscape(gpa, 256);
    }
    defer gpa.free(voxels);
    const t0 = std.Io.Timestamp.now(init.io, .awake);
    var geo: api.Handle = null;
    req(pyrit.pyr_geometry_build(@ptrCast(ctx), log2, voxels.ptr, @intCast(voxels.len), api.build_host_input, &geo));
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    std.debug.print("Geometrie: {d} Voxel, 2^{d}, GPU-Bau {d:.1} ms\n", .{ voxels.len, log2, @as(f64, @floatFromInt(t0.untilNow(init.io, .awake).toNanoseconds())) / 1e6 });

    var inst: api.Handle = null;
    req(pyrit.pyr_instance_create(@ptrCast(ctx), geo, &inst));
    const size: f32 = @floatFromInt(@as(u32, 1) << @intCast(log2));
    const m = [12]f32{ 1, 0, 0, -size / 2, 0, 1, 0, 0, 0, 0, 1, -size / 2 };
    req(pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst, &m));

    // Materialien: 0 diffus, 1 Kristall (leuchtend), 2 Wasser (glatt)
    var mat: types.Material = undefined;
    pyrit.pyr_material_default(&mat);
    mat.emission = .{ 6, 3.5, 1.2 };
    mat.roughness = 0.3;
    req(pyrit.pyr_material_set(@ptrCast(ctx), 1, &mat));
    pyrit.pyr_material_default(&mat);
    mat.roughness = 0.08;
    mat.metallic = 0.0;
    req(pyrit.pyr_material_set(@ptrCast(ctx), 2, &mat));

    var light: types.Lighting = undefined;
    pyrit.pyr_lighting_default(&light);
    if (bounces > 0) light.gi_bounces = bounces;
    if (no_shadows) light.flags &= ~types.lighting_shadows;
    if (firefly > 0) light.firefly_clamp = firefly;
    if (fog > 0) {
        light.fog_density = fog;
        light.fog_height = 90;
        light.fog_falloff = 0.02;
        light.fog_color = .{ 1, 0.98, 0.92 };
        light.fog_anisotropy = 0.7;
    }
    if (use_env) {
        // Umgebungskarte: Himmelsverlauf mit kleiner, sehr heller Sonne.
        // Genau der Fall, der ohne Importance-Sampling hoffnungslos rauscht.
        const ew: u32 = 512;
        const eh: u32 = 256;
        const env = try init.gpa.alloc(f32, ew * eh * 4);
        defer init.gpa.free(env);
        const sun_theta: f32 = 0.75;
        const sun_phi: f32 = 1.1;
        for (0..eh) |y| {
            const theta = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(eh)) * std.math.pi;
            for (0..ew) |x| {
                const phi = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, @floatFromInt(ew)) * 2 * std.math.pi;
                const up = @cos(theta);
                // Verlauf: unten Boden, oben Himmel
                var r: f32 = if (up > 0) 0.35 + 0.25 * up else 0.10;
                var g: f32 = if (up > 0) 0.45 + 0.35 * up else 0.09;
                var b: f32 = if (up > 0) 0.65 + 0.35 * up else 0.08;
                // kleine Sonnenscheibe, sehr hell
                const dt = theta - sun_theta;
                var dp = phi - sun_phi;
                if (dp > std.math.pi) dp -= 2 * std.math.pi;
                if (dp < -std.math.pi) dp += 2 * std.math.pi;
                if (!env_flat and dt * dt + dp * dp * @sin(theta) * @sin(theta) < 0.03 * 0.03) {
                    // Strahldichte so gewaehlt, dass L * Raumwinkel etwa der
                    // Beleuchtungsstaerke der analytischen Sonne entspricht
                    // (0.03 rad Scheibe -> 0.00283 sr, 2.6 / 0.00283 ~ 920)
                    r += 920;
                    g += 870;
                    b += 780;
                }
                const o = (y * ew + x) * 4;
                env[o + 0] = r;
                env[o + 1] = g;
                env[o + 2] = b;
                env[o + 3] = 0;
            }
        }
        req(pyrit.pyr_environment_set(@ptrCast(ctx), ew, eh, &env[0]));
        light.env_intensity = 1;
        // Die eigene Sonne aus: sie steckt jetzt in der Karte
        light.sun_color = .{ 0, 0, 0 };
        std.debug.print("Umgebungskarte {d}x{d} mit Sonnenscheibe gesetzt\n", .{ ew, eh });
    }
    light.sun_direction = .{ 0.6, 0.55, 0.3 };
    if (!gi) light.flags = types.lighting_shadows | types.lighting_ao | types.lighting_sun_disk;
    req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));

    var view: api.Handle = null;
    req(pyrit.pyr_view_create(@ptrCast(ctx), &view));
    var cam = std.mem.zeroes(types.Camera);
    const eye = [3]f32{ size * 0.55, size * 0.5, size * 0.75 };
    const target = [3]f32{ 0, size * 0.12, 0 };
    const up = [3]f32{ 0, 1, 0 };
    pyrit.pyr_camera_look_at(&cam, &eye, &target, &up);
    pyrit.pyr_camera_perspective(&cam, 0.8, w, h, 0.1);

    const n: usize = @as(usize, w) * h;
    var tg = std.mem.zeroes(api.Targets);
    tg.hits = devAlloc(n * 16);
    tg.motion = devAlloc(n * 8);
    tg.color = devAlloc(n * 16);
    tg.normal = devAlloc(n * 16);
    tg.albedo = devAlloc(n * 16);
    const ldr = devAlloc(n * 4);
    defer for ([_]u64{ tg.hits, tg.motion, tg.color, tg.normal, tg.albedo, ldr }) |b| {
        _ = drv.cuMemFree_v2(b);
    };
    var post = std.mem.zeroes(api.PostInfo);
    post.output_ldr = ldr;
    post.denoise_iterations = 3;
    post.temporal_alpha = if (alpha > 0) alpha else 1.0 / @as(f32, @floatFromInt(@max(frames, 1)));
    post.exposure = 1.0;

    const t1 = std.Io.Timestamp.now(init.io, .awake);
    for (0..frames) |f| {
        var j: [2]f32 = undefined;
        pyrit.pyr_jitter_halton(@intCast(f), &j);
        cam.jitter = j;
        req(pyrit.pyr_commit(@ptrCast(ctx), null));
        req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tg));
        req(pyrit.pyr_postprocess(@ptrCast(ctx), view, &tg, &post));
    }
    req(pyrit.pyr_synchronize(@ptrCast(ctx)));
    const ms = @as(f64, @floatFromInt(t1.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
    std.debug.print("{d} Frames {d}x{d}: {d:.2} ms pro Frame (RT-Cores: {s})\n", .{ frames, w, h, ms / @as(f64, @floatFromInt(@max(frames, 1))), if (pyrit.pyr_features(@ptrCast(ctx)) & api.feature_rt_cores != 0) "ja" else "nein" });

    try savePpm(init, ldr, w, h, out_path);
    _ = drv.cuDevicePrimaryCtxRelease_v2(dev);
}

/// Box-Mittel von `src` (sw x sh) nach `dst` (dw x dh); sw/dw muss ganzzahlig sein
fn downsample(src: []const [4]u8, sw: u32, sh: u32, dst: [][4]u8, dw: u32, dh: u32) void {
    if (sw == dw and sh == dh) {
        @memcpy(dst, src[0..dst.len]);
        return;
    }
    const fx = sw / dw;
    const fy = sh / dh;
    const n: u32 = fx * fy;
    for (0..dh) |y| {
        for (0..dw) |x| {
            var acc: [4]u32 = .{ 0, 0, 0, 0 };
            for (0..fy) |sy| {
                for (0..fx) |sx| {
                    const p4 = src[(y * fy + sy) * sw + (x * fx + sx)];
                    inline for (0..4) |k| acc[k] += p4[k];
                }
            }
            var o: [4]u8 = undefined;
            inline for (0..4) |k| o[k] = @intCast(acc[k] / n);
            dst[y * dw + x] = o;
        }
    }
}

fn savePpmScaled(init: std.process.Init, ldr: u64, sw: u32, sh: u32, dw: u32, dh: u32, out_path: []const u8) !void {
    const gpa = init.gpa;
    const src = try gpa.alloc([4]u8, @as(usize, sw) * sh);
    defer gpa.free(src);
    cu(drv.cuMemcpyDtoH_v2(src.ptr, ldr, src.len * 4));
    const dst = try gpa.alloc([4]u8, @as(usize, dw) * dh);
    defer gpa.free(dst);
    downsample(src, sw, sh, dst, dw, dh);
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(gpa);
    var hb: [64]u8 = undefined;
    try file.appendSlice(gpa, try std.fmt.bufPrint(&hb, "P6\n{d} {d}\n255\n", .{ dw, dh }));
    for (dst) |q| try file.appendSlice(gpa, q[0..3]);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = file.items });
    std.debug.print("Bild: {s}\n", .{out_path});
}

fn savePpm(init: std.process.Init, ldr: u64, w: u32, h: u32, out_path: []const u8) !void {
    const gpa = init.gpa;
    const n: usize = @as(usize, w) * h;
    const px = try gpa.alloc([4]u8, n);
    defer gpa.free(px);
    cu(drv.cuMemcpyDtoH_v2(px.ptr, ldr, n * 4));
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(gpa);
    var header_buf: [64]u8 = undefined;
    try file.appendSlice(gpa, try std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ w, h }));
    for (px) |p| try file.appendSlice(gpa, p[0..3]);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = file.items });
    std.debug.print("Bild: {s}\n", .{out_path});
}

fn msSince(init: std.process.Init, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.untilNow(init.io, .awake).toNanoseconds())) / 1e6;
}

/// Große Welt: Gelände auf der GPU, LOD-Streaming, Flug über die Landschaft
fn renderWorld(init: std.process.Init, ctx: ?*anyopaque, out_w: u32, out_h: u32, frames: u32, gi: bool, out_path: []const u8, scale: u32, fg: bool, upscaler: u32, profile: bool, voxel_px: f32, budget_mib: u32, denoise: u32, clamp_sigma: f32, coarse_secondary: bool, gi_distance: f32, half_gi: bool, sea_level: f32, rt_leaf: u32, static_cam: bool, turn: f32, flicker: bool, view_distance: f32, edit_test: bool, edit_load: bool, edit_file: ?[]const u8, chunk_capacity: u32, edit_stream: bool, fx_flags: u32, materials: bool, use_env: bool, env_flat: bool, firefly: f32, alpha: f32, no_shadows: bool, no_jitter: bool, super: u32, chunks_per_update: u32, bounces: u32, fog: f32, async_post: bool) !void {
    // Supersampling: alles läuft in super-facher Auflösung, erst ganz am Ende
    // wird gemittelt. Damit entscheidet sich die Deckung einer Voxelkante
    // schon *innerhalb* eines Frames statt über die Zeit.
    const ow = out_w * @max(super, 1);
    const oh = out_h * @max(super, 1);
    const w = ow / scale;
    const h = oh / scale;
    var terrain: api.TerrainInfo = undefined;
    pyrit.pyr_terrain_default(&terrain);
    // Wasser (Material 2, durchsichtig) und Bäume (Material 0)
    if (sea_level > 0) terrain.sea_level = sea_level;
    terrain.attr_water = voxel(2, 40, 90, 140);
    terrain.attr_leaves = voxel(0, 48, 112, 40);
    terrain.attr_wood = voxel(0, 96, 68, 44);

    // Blockarten bekommen eigene Materialien, damit jede ihre eigene 32x32-
    // Kachel tragen kann (eine Kachel je Grundvoxel). Die Farbe steckt weiter
    // im Voxelattribut, die Textur moduliert sie nur.
    {
        const mat_grass: u32 = 1;
        const mat_rock: u32 = 3;
        const mat_sand: u32 = 4;
        const mat_snow: u32 = 5;
        const mat_wood: u32 = 6;
        const mat_leaves: u32 = 7;
        terrain.attr_grass = voxel(mat_grass, 84, 140, 58);
        terrain.attr_dirt = voxel(mat_grass, 122, 92, 62);
        terrain.attr_rock = voxel(mat_rock, 118, 112, 106);
        terrain.attr_sand = voxel(mat_sand, 214, 196, 142);
        terrain.attr_snow = voxel(mat_snow, 236, 240, 245);
        terrain.attr_wood = voxel(mat_wood, 96, 68, 44);
        terrain.attr_leaves = voxel(mat_leaves, 48, 112, 40);

        const tex = try init.gpa.alloc(u8, blocktex.size * blocktex.size * 4);
        defer init.gpa.free(tex);
        const kinds = [_]struct { k: blocktex.Kind, m: u32, rough: f32 }{
            .{ .k = .grass, .m = mat_grass, .rough = 0.95 },
            .{ .k = .rock, .m = mat_rock, .rough = 0.85 },
            .{ .k = .sand, .m = mat_sand, .rough = 0.98 },
            .{ .k = .snow, .m = mat_snow, .rough = 0.75 },
            .{ .k = .wood, .m = mat_wood, .rough = 0.9 },
            .{ .k = .leaves, .m = mat_leaves, .rough = 0.95 },
        };
        for (kinds) |e| {
            blocktex.make(e.k, tex);
            var idx: u32 = 0;
            req(pyrit.pyr_texture_create(@ptrCast(ctx), blocktex.size, blocktex.size, tex.ptr, &idx));
            var m: types.Material = undefined;
            pyrit.pyr_material_default(&m);
            m.flags = types.material_voxel_color;
            m.roughness = e.rough;
            m.texture = idx;
            m.texture_scale = 1; // eine Kachel je Grundvoxel
            if (e.k == .leaves) {
                m.subsurface = 0.35; // Laub leuchtet von hinten durch
                m.subsurface_color = .{ 0.45, 0.85, 0.35 };
            }
            if (e.k == .snow) m.clearcoat = 0.25;
            req(pyrit.pyr_material_set(@ptrCast(ctx), e.m, &m));
        }
    }

    var water: types.Material = undefined;
    pyrit.pyr_material_default(&water);
    water.flags = types.material_voxel_color | types.material_transparent | types.material_refract | types.material_waves;
    water.roughness = 0.04;
    water.ior = 1.33;
    water.density = 0.05;
    water.opacity = 0.03;
    // Große, ruhige Wellen: feine sprenkeln in der Ferne, weil viele davon
    // auf ein Pixel fallen und die Normale von Pixel zu Pixel springt.
    water.wave_height = 0.5;
    water.wave_length = 48;
    water.wave_speed = 0.18;
    req(pyrit.pyr_material_set(@ptrCast(ctx), 2, &water));

    // Materialprobe: Textur auf dem Boden, Detailnormale auf dem Fels,
    // Streuung im Laub, Klarlack auf dem Wasser.
    if (materials) {
        // kleine Kacheltextur: unregelmäßige Flecken, damit man das Filtern sieht
        const tw: u32 = 64;
        const tex = try init.gpa.alloc(u8, tw * tw * 4);
        defer init.gpa.free(tex);
        for (0..tw) |ty| for (0..tw) |tx| {
            const nz = (tx *% 73 +% ty *% 151) ^ ((tx *% 19) >> 2);
            const v: u8 = @intCast(180 + (nz % 76));
            const o = (ty * tw + tx) * 4;
            tex[o + 0] = v;
            tex[o + 1] = @intCast(@min(@as(u32, v) + 12, 255));
            tex[o + 2] = @intCast(@as(u32, v) * 3 / 4);
            tex[o + 3] = 255;
        };
        var tex_index: u32 = 0;
        req(pyrit.pyr_texture_create(@ptrCast(ctx), tw, tw, tex.ptr, &tex_index));

        var ground: types.Material = undefined;
        pyrit.pyr_material_default(&ground);
        ground.flags = types.material_voxel_color;
        ground.texture = tex_index;
        ground.texture_scale = 8; // eine Kachel je 8 Voxel
        ground.normal_strength = 0.35;
        ground.normal_scale = 2;
        ground.subsurface = 0.25; // Laub liegt auf demselben Material
        ground.subsurface_color = .{ 0.4, 0.8, 0.3 };
        req(pyrit.pyr_material_set(@ptrCast(ctx), 0, &ground));

        water.clearcoat = 1;
        water.clearcoat_roughness = 0.03;
        req(pyrit.pyr_material_set(@ptrCast(ctx), 2, &water));
        std.debug.print("Materialprobe: Textur {d} ({d}x{d}), Detailnormale, Streuung, Klarlack\n", .{ tex_index, tw, tw });
    }

    var wi = std.mem.zeroes(api.WorldInfo);
    wi.terrain = &terrain;
    wi.voxel_pixels = voxel_px;
    wi.view_distance = view_distance;
    wi.chunk_capacity = chunk_capacity;
    wi.chunks_per_update = chunks_per_update;
    wi.rt_leaf_log2 = rt_leaf;
    wi.memory_budget = @as(u64, budget_mib) << 20;
    if (coarse_secondary) {
        wi.secondary_mask = 0x2; // grobe Fassung nur für Schatten und GI
    }
    var world: ?*anyopaque = null;
    req(pyrit.pyr_world_create(@ptrCast(ctx), &wi, @ptrCast(&world)));
    defer _ = pyrit.pyr_world_destroy(@ptrCast(ctx), @ptrCast(world));

    var light: types.Lighting = undefined;
    pyrit.pyr_lighting_default(&light);
    if (bounces > 0) light.gi_bounces = bounces;
    if (no_shadows) light.flags &= ~types.lighting_shadows;
    if (firefly > 0) light.firefly_clamp = firefly;
    if (fog > 0) {
        light.fog_density = fog;
        light.fog_height = 90;
        light.fog_falloff = 0.02;
        light.fog_color = .{ 1, 0.98, 0.92 };
        light.fog_anisotropy = 0.7;
    }
    if (use_env) {
        // Umgebungskarte: Himmelsverlauf mit kleiner, sehr heller Sonne.
        // Genau der Fall, der ohne Importance-Sampling hoffnungslos rauscht.
        const ew: u32 = 512;
        const eh: u32 = 256;
        const env = try init.gpa.alloc(f32, ew * eh * 4);
        defer init.gpa.free(env);
        const sun_theta: f32 = 0.75;
        const sun_phi: f32 = 1.1;
        for (0..eh) |y| {
            const theta = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(eh)) * std.math.pi;
            for (0..ew) |x| {
                const phi = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, @floatFromInt(ew)) * 2 * std.math.pi;
                const up = @cos(theta);
                // Verlauf: unten Boden, oben Himmel
                var r: f32 = if (up > 0) 0.35 + 0.25 * up else 0.10;
                var g: f32 = if (up > 0) 0.45 + 0.35 * up else 0.09;
                var b: f32 = if (up > 0) 0.65 + 0.35 * up else 0.08;
                // kleine Sonnenscheibe, sehr hell
                const dt = theta - sun_theta;
                var dp = phi - sun_phi;
                if (dp > std.math.pi) dp -= 2 * std.math.pi;
                if (dp < -std.math.pi) dp += 2 * std.math.pi;
                if (!env_flat and dt * dt + dp * dp * @sin(theta) * @sin(theta) < 0.03 * 0.03) {
                    // Strahldichte so gewaehlt, dass L * Raumwinkel etwa der
                    // Beleuchtungsstaerke der analytischen Sonne entspricht
                    // (0.03 rad Scheibe -> 0.00283 sr, 2.6 / 0.00283 ~ 920)
                    r += 920;
                    g += 870;
                    b += 780;
                }
                const o = (y * ew + x) * 4;
                env[o + 0] = r;
                env[o + 1] = g;
                env[o + 2] = b;
                env[o + 3] = 0;
            }
        }
        req(pyrit.pyr_environment_set(@ptrCast(ctx), ew, eh, &env[0]));
        light.env_intensity = 1;
        // Die eigene Sonne aus: sie steckt jetzt in der Karte
        light.sun_color = .{ 0, 0, 0 };
        std.debug.print("Umgebungskarte {d}x{d} mit Sonnenscheibe gesetzt\n", .{ ew, eh });
    }
    light.sun_direction = .{ 0.5, 0.35, 0.4 };
    if (!gi) light.flags = types.lighting_shadows | types.lighting_ao | types.lighting_sun_disk;
    light.gi_distance = gi_distance;
    if (half_gi) light.flags |= types.lighting_gi_half;
    req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));

    var view: api.Handle = null;
    req(pyrit.pyr_view_create(@ptrCast(ctx), &view));
    const n: usize = @as(usize, w) * h;
    // Mit überlappender Nachbearbeitung braucht es zwei Zielsätze: der nächste
    // Frame rendert schon, während aus dem vorigen noch gelesen wird.
    const sets: usize = if (async_post) 2 else 1;
    var tgs: [2]api.Targets = .{ std.mem.zeroes(api.Targets), std.mem.zeroes(api.Targets) };
    for (0..sets) |si| {
        tgs[si].hits = devAlloc(n * 16);
        tgs[si].motion = devAlloc(n * 8);
        tgs[si].color = devAlloc(n * 16);
        tgs[si].normal = devAlloc(n * 16);
        tgs[si].albedo = devAlloc(n * 16);
        // DLSS Ray Reconstruction bekommt Rauheit und Metall je Pixel
        if (upscaler == api.upscaler_dlss_rr) tgs[si].material = devAlloc(n * 8);
        tgs[si].ray_mask = 0x1;
        if (coarse_secondary) tgs[si].secondary_mask = 0x2;
    }
    var tg = tgs[0];
    const n_out: usize = @as(usize, ow) * oh;
    const ldr = devAlloc(n_out * 4);
    const ldr_fg = devAlloc(n_out * 4);
    defer {
        for (0..sets) |si| {
            for ([_]u64{ tgs[si].hits, tgs[si].motion, tgs[si].color, tgs[si].normal, tgs[si].albedo }) |b| {
                if (b != 0) _ = drv.cuMemFree_v2(b);
            }
        }
        for ([_]u64{ ldr, ldr_fg }) |b| _ = drv.cuMemFree_v2(b);
    }
    var post = std.mem.zeroes(api.PostInfo);
    post.output_ldr = ldr;
    // 0 = Vorgabe der Bibliothek (0,05, also 20 Frames Mittelung)
    post.temporal_alpha = alpha;
    post.denoise_iterations = denoise;
    post.exposure = 1.0;
    post.clamp_sigma = clamp_sigma;
    post.output_width = ow;
    post.upscaler = upscaler;
    post.output_height = oh;
    var fxi = std.mem.zeroes(api.PostFx);
    fxi.flags = fx_flags;
    fxi.saturation = 1.05;
    fxi.contrast = 1.05;
    if (fx_flags != 0) post.fx = &fxi;
    var fgi = std.mem.zeroes(api.FrameGenInfo);
    fgi.output_ldr = ldr_fg;
    var fg_frames: u32 = 0;
    var prof = [3]f64{ 0, 0, 0 };
    // Flimmermaß: mittlerer Unterschied aufeinanderfolgender Ausgabebilder über
    // die letzten Frames. Bei stehender Kamera ist jeder Unterschied Rauschen.
    var prev_px: []([4]u8) = &.{};
    var cur_px: []([4]u8) = &.{};
    defer if (prev_px.len != 0) init.gpa.free(prev_px);
    defer if (cur_px.len != 0) init.gpa.free(cur_px);
    var edit_ms: f64 = 0;
    var edit_calls: u64 = 0;
    var flick_sum: f64 = 0;
    var pump_sum: f64 = 0;
    var loud_sum: f64 = 0;
    var diff_map: []u8 = &.{};
    var src_px: []([4]u8) = &.{};
    defer if (src_px.len != 0) init.gpa.free(src_px);
    defer if (diff_map.len != 0) init.gpa.free(diff_map);
    var p999_sum: f64 = 0;
    var flick_n: u64 = 0;
    var have_prev = false;

    // Kamera fliegt mit 2 Voxeln pro Frame über das Gelände; Ursprung folgt in 1024er-Schritten
    var cam = std.mem.zeroes(types.Camera);
    const start = [3]f64{ 100_000, 0, 100_000 };
    var update_ms: f64 = 0;
    var update_max: f64 = 0;
    var st: api.WorldStats = undefined;
    const total = frames + 1;
    const t_all = std.Io.Timestamp.now(init.io, .awake);
    var f: u32 = 0;
    var warm: u32 = 0;
    var t_flight = t_all;
    while (f < total) : (f += 1) {
        const moved: f64 = if (static_cam) 0 else @floatFromInt(f);
        const fx = start[0] + moved * 2;
        const fz = start[2] + moved * 1;
        const ground = pyrit.pyr_terrain_height(&terrain, fx, fz);
        const pos = [3]f64{ fx, @max(ground, terrain.sea_level) + 60, fz };
        const origin = [3]f64{ @floor(pos[0] / 1024) * 1024, 0, @floor(pos[2] / 1024) * 1024 };
        // Kamera zuerst: die Welt wählt das LOD nach ihrem Bildschirmmaß
        const eye = [3]f32{ @floatCast(pos[0] - origin[0]), @floatCast(pos[1] - origin[1]), @floatCast(pos[2] - origin[2]) };
        // Blickrichtung dreht mit `turn` je Frame (Mausblick)
        const ang = turn * @as(f32, @floatFromInt(f));
        const dx = 200 * @cos(ang) - 100 * @sin(ang);
        const dz = 200 * @sin(ang) + 100 * @cos(ang);
        const target = [3]f32{ eye[0] + dx, eye[1] - 45, eye[2] + dz };
        const up = [3]f32{ 0, 1, 0 };
        pyrit.pyr_camera_look_at(&cam, &eye, &target, &up);
        pyrit.pyr_camera_perspective(&cam, 1.0, w, h, 0.1);
        var j: [2]f32 = undefined;
        pyrit.pyr_jitter_halton(f, &j);
        cam.jitter = if (no_jitter) .{ 0, 0 } else j;
        // Laufende Einzeländerungen von der CPU: je Frame ein Voxel, an
        // wandernder Stelle – der Fall "einzelne Chunks kommen nach".
        if (edit_stream and f > 0) {
            const bx: i64 = @intFromFloat(pos[0] + 40);
            const bz: i64 = @intFromFloat(pos[2] + 20 + @as(f64, @floatFromInt(f % 64)));
            const gy: i64 = @intFromFloat(pyrit.pyr_terrain_height(&terrain, @floatFromInt(bx), @floatFromInt(bz)));
            const one = [1]api.WorldEdit{.{ .x = bx, .y = gy + 2, .z = bz, .attribute = pyrit.pyr_voxel_attribute(0, 240, 240, 40) }};
            const te = std.Io.Timestamp.now(init.io, .awake);
            req(pyrit.pyr_world_edit(@ptrCast(ctx), @ptrCast(world), &one, 1));
            edit_ms += msSince(init, te);
            edit_calls += 1;
        }
        const t0 = std.Io.Timestamp.now(init.io, .awake);
        req(pyrit.pyr_world_update(@ptrCast(ctx), @ptrCast(world), &pos, &origin, &cam));
        const u = msSince(init, t0);
        req(pyrit.pyr_world_stats(@ptrCast(world), &st));
        if (coarse_secondary and st.secondary_bias != light.secondary_bias) {
            light.secondary_bias = st.secondary_bias;
            req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));
        }
        if (f == 0 and edit_load) {
            // Frische Welt: Änderungen aus der Datei übernehmen, bevor irgendein
            // Chunk gebaut wurde – sie müssen schon bei der Erzeugung greifen.
            const data = try std.Io.Dir.cwd().readFileAlloc(init.io, edit_file.?, init.gpa, .limited(1 << 28));
            defer init.gpa.free(data);
            req(pyrit.pyr_world_edits_load(@ptrCast(ctx), @ptrCast(world), data.ptr, data.len));
            std.debug.print("Änderungen geladen: {d} Bytes\n", .{data.len});
        }
        if (f == 0) {
            // Aufwärmen: alles um die Startposition fertig bauen
            const tw = std.Io.Timestamp.now(init.io, .awake);
            while (st.pending_chunks > 0 and warm < 8000) : (warm += 1) {
                req(pyrit.pyr_world_wait(@ptrCast(ctx), @ptrCast(world), &pos));
                req(pyrit.pyr_world_update(@ptrCast(ctx), @ptrCast(world), &pos, &origin, &cam));
                req(pyrit.pyr_world_stats(@ptrCast(world), &st));
            }
            // Probe für pyr_world_edit: einen Turm setzen und eine Grube
            // ausheben, beides vor der Kamera. Danach warten, bis die
            // betroffenen Chunks neu gebaut sind.
            if (edit_test) {
                var list: std.ArrayList(api.WorldEdit) = .empty;
                defer list.deinit(init.gpa);
                const bx: i64 = @intFromFloat(pos[0] + 60);
                const bz: i64 = @intFromFloat(pos[2] + 30);
                const gy: i64 = @intFromFloat(pyrit.pyr_terrain_height(&terrain, @floatFromInt(bx), @floatFromInt(bz)));
                const red = pyrit.pyr_voxel_attribute(0, 220, 40, 40);
                var ex: i64 = 0;
                while (ex < 12) : (ex += 1) {
                    var ez: i64 = 0;
                    while (ez < 12) : (ez += 1) {
                        var ey: i64 = 0;
                        while (ey < 24) : (ey += 1) {
                            try list.append(init.gpa, .{ .x = bx + ex, .y = gy + 1 + ey, .z = bz + ez, .attribute = red });
                        }
                        // Grube: alles unter der Oberfläche entfernen
                        ey = 0;
                        while (ey < 14) : (ey += 1) {
                            try list.append(init.gpa, .{ .x = bx + ex - 16, .y = gy - ey, .z = bz + ez, .attribute = 0 });
                        }
                    }
                }
                req(pyrit.pyr_world_edit(@ptrCast(ctx), @ptrCast(world), list.items.ptr, @intCast(list.items.len)));
                var guard: u32 = 0;
                while (guard < 64) : (guard += 1) {
                    req(pyrit.pyr_world_update(@ptrCast(ctx), @ptrCast(world), &pos, &origin, &cam));
                    req(pyrit.pyr_world_wait(@ptrCast(ctx), @ptrCast(world), &pos));
                }
                const bytes = pyrit.pyr_world_edits_bytes(@ptrCast(world));
                const buf = try init.gpa.alloc(u8, @intCast(bytes));
                defer init.gpa.free(buf);
                req(pyrit.pyr_world_edits_save(@ptrCast(world), buf.ptr, bytes));
                if (edit_file) |path| try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = buf });
                std.debug.print("{d} Änderungen gesetzt, gesichert in {d} Bytes\n", .{ list.items.len, bytes });
            }
            t_flight = std.Io.Timestamp.now(init.io, .awake); // Aufwärmen nicht mitmessen
            std.debug.print("Welt aufgebaut: {d} Updates, {d:.1} ms, {d} Chunks resident, {d} sichtbar, {d:.1} MiB\n", .{ warm + 1, msSince(init, tw) + u, st.resident_chunks, st.visible_chunks, @as(f64, @floatFromInt(st.bytes)) / (1 << 20) });
        } else {
            update_ms += u;
            update_max = @max(update_max, u);

        }
        const fi = api.FrameInfo{ .time = @as(f64, @floatFromInt(f)) / 60.0, .origin = origin };
        const tp0 = std.Io.Timestamp.now(init.io, .awake);
        req(pyrit.pyr_commit(@ptrCast(ctx), &fi));
        if (profile) req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const tp1 = std.Io.Timestamp.now(init.io, .awake);
        tg = tgs[if (sets == 2) f % 2 else 0];
        tg = tgs[if (sets == 2) f % 2 else 0];
        req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tg));
        if (profile) req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const tp2 = std.Io.Timestamp.now(init.io, .awake);
        req(pyrit.pyr_postprocess(@ptrCast(ctx), view, &tg, &post));
        if (profile) req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const tp3 = std.Io.Timestamp.now(init.io, .awake);
        if (fg and f >= 2) {
            req(pyrit.pyr_frame_generate(@ptrCast(ctx), view, &fgi));
            fg_frames += 1;
        }
        if (f > 0) {
            prof[0] += msSince(init, tp0) - msSince(init, tp1);
            prof[1] += msSince(init, tp1) - msSince(init, tp2);
            prof[2] += msSince(init, tp2) - msSince(init, tp3);
        }
        // wie eine Anwendung mit Present: höchstens ein Frame im Voraus
        req(pyrit.pyr_synchronize(@ptrCast(ctx)));

        if (flicker and f + 8 >= total) {
            const np: usize = @as(usize, out_w) * out_h;
            const nsrc: usize = @as(usize, ow) * oh;
            if (cur_px.len == 0) {
                cur_px = try init.gpa.alloc([4]u8, np);
                prev_px = try init.gpa.alloc([4]u8, np);
                src_px = try init.gpa.alloc([4]u8, nsrc);
                diff_map = try init.gpa.alloc(u8, np);
                @memset(diff_map, 0);
            }
            cu(drv.cuMemcpyDtoH_v2(src_px.ptr, ldr, nsrc * 4));
            downsample(src_px, ow, oh, cur_px, out_w, out_h);
            if (have_prev) {
                var sum: f64 = 0;
                var mean_a: f64 = 0;
                var mean_b: f64 = 0;
                // Verteilung der Unterschiede: ein wanderndes Rauschen an
                // Kanten betrifft nur wenige Prozent der Pixel und geht im
                // Mittelwert unter. Deshalb zusätzlich zählen, wie viele
                // Pixel sich deutlich ändern, und wie stark die stärksten.
                var hist = [_]u32{0} ** 256;
                for (cur_px, prev_px) |a, b| {
                    var pmax: u32 = 0;
                    inline for (0..3) |k| {
                        const d = @abs(@as(f64, @floatFromInt(a[k])) - @as(f64, @floatFromInt(b[k])));
                        sum += d;
                        mean_a += @floatFromInt(a[k]);
                        mean_b += @floatFromInt(b[k]);
                        pmax = @max(pmax, @as(u32, @intFromFloat(d)));
                    }
                    hist[@min(pmax, 255)] += 1;
                }
                // Karte der Unterschiede mitschreiben: erst daran sieht man,
                // *wo* sich etwas bewegt.
                if (diff_map.len == np) {
                    for (cur_px, prev_px, 0..) |a, b, k| {
                        var pmax: u32 = 0;
                        inline for (0..3) |q| pmax = @max(pmax, @as(u32, @intFromFloat(@abs(@as(f64, @floatFromInt(a[q])) - @as(f64, @floatFromInt(b[q]))))));
                        diff_map[k] = @max(diff_map[k], @as(u8, @intCast(@min(pmax, 255))));
                    }
                }
                {
                    // Anteil der Pixel mit mehr als 4 Stufen Unterschied und
                    // das 99,9-Perzentil
                    var above: u64 = 0;
                    for (5..256) |k| above += hist[k];
                    loud_sum += @as(f64, @floatFromInt(above)) / @as(f64, @floatFromInt(np)) * 100;
                    var acc: u64 = 0;
                    const want = np - np / 1000;
                    var p999: u32 = 0;
                    for (hist, 0..) |c, k| {
                        acc += c;
                        if (acc >= want) {
                            p999 = @intCast(k);
                            break;
                        }
                    }
                    p999_sum += @floatFromInt(p999);
                }
                const n3: f64 = @floatFromInt(np * 3);
                flick_sum += sum / n3;
                // Getrennt: verschiebt sich die *mittlere* Helligkeit? Das ist
                // kein örtliches Rauschen, sondern ein Pumpen des ganzen
                // Bildes – das Auge sieht es viel eher als der Mittelwert
                // über Einzelpixel.
                pump_sum += @abs(mean_a - mean_b) / n3;
                flick_n += 1;
            }
            @memcpy(prev_px, cur_px);
            have_prev = true;
        }
    }
    if (edit_calls > 0) {
        std.debug.print("Einzeländerungen: {d} Aufrufe, pyr_world_edit im Mittel {d:.4} ms auf dem Hauptthread\n", .{ edit_calls, edit_ms / @as(f64, @floatFromInt(edit_calls)) });
    }
    if (flicker and flick_n > 0) {
        const fn_f: f64 = @floatFromInt(flick_n);
        if (diff_map.len != 0) {
            var f2: std.ArrayList(u8) = .empty;
            defer f2.deinit(init.gpa);
            var hd: [64]u8 = undefined;
            try f2.appendSlice(init.gpa, try std.fmt.bufPrint(&hd, "P5\n{d} {d}\n255\n", .{ out_w, out_h }));
            for (diff_map) |d| try f2.append(init.gpa, @intCast(@min(@as(u32, d) * 6, 255)));
            try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = "unruhe.pgm", .data = f2.items });
            std.debug.print("Karte der Unterschiede: unruhe.pgm\n", .{});
        }
        std.debug.print("Flimmern: Mittel {d:.3}, unruhige Pixel {d:.2} %, staerkste (99,9 %) {d:.1} Stufen, Pumpen {d:.4}\n", .{ flick_sum / fn_f, loud_sum / fn_f, p999_sum / fn_f, pump_sum / fn_f });
    }
    const all = msSince(init, t_flight);
    std.debug.print("{d} Frames {d}x{d} -> {d}x{d} im Flug: {d:.2} ms pro Frame gesamt ({d} mit Zwischenbild), Welt-Update Mittel {d:.2} ms, max {d:.2} ms\n", .{ frames, w, h, out_w, out_h, all / @as(f64, @floatFromInt(@max(frames, 1))), fg_frames, update_ms / @as(f64, @floatFromInt(@max(frames, 1))), update_max });
    const nf: f64 = @floatFromInt(@max(frames, 1));
    if (profile) std.debug.print("Aufteilung: Commit {d:.2} ms, Rendern {d:.2} ms, Nachbearbeitung {d:.2} ms\n", .{ prof[0] / nf, prof[1] / nf, prof[2] / nf });
    std.debug.print("Am Ende: {d} Chunks resident, {d} sichtbar, {d} ausstehend, {d:.1} MiB, Ziel {d:.1} px/Voxel, gröbste Stufe {d}, Überläufe {d}\n", .{ st.resident_chunks, st.visible_chunks, st.pending_chunks, @as(f64, @floatFromInt(st.bytes)) / (1 << 20), st.voxel_pixels, st.top_lod, st.overflow_chunks });
    try savePpmScaled(init, ldr, ow, oh, out_w, out_h, out_path);
    if (fg_frames > 0) {
        var buf: [512]u8 = undefined;
        const fg_path = try std.fmt.bufPrint(&buf, "{s}.fg.ppm", .{out_path});
        try savePpm(init, ldr_fg, out_w, out_h, fg_path);
    }
}
