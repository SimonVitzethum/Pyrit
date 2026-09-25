//! Pyrit-Demo: eine Minecraft-artige Welt, auf der GPU erzeugt (eigener
//! CUDA-Kernel in demo/kernels.zig), 1024 Chunks Sichtweite, Zuschauermodus.
//!
//!   zig build demo -Ddlss-sdk=$PWD/DLSS -- [--mode native|taau|dlaa|dlss|rr]
//!                     [--size 1280x720] [--scale 2] [--seed 1] [--fg [dlss|dlss3|dlss4|dlss6]]
//!                     [--super 2] [--voxel-px 4] [--no-rt] [--no-vsync]
//!   zig build demo -- --record bilder [--still 10] [--move 10] [--warm 40]
//!
//! Steuerung (Zuschauermodus): W/S fliegen in Blickrichtung, A/D seitlich,
//! Leertaste hoch, Umschalt runter, Strg schneller, Maus ziehen dreht,
//! Q/E drehen die Sonne, +/- ändern die Zielgröße der Voxel,
//! 1–5 Bildaufbau: nativ, TAAU, DLAA, DLSS, DLSS Ray Reconstruction,
//! F Zwischenbilder: aus, eigene CUDA-Frame-Generation, DLSS Frame Generation
//! (nur mit TAAU/DLAA/DLSS/RR), B/T/U Bloom/Tiefenschärfe/Bewegungsunschärfe,
//! N Nebel, G Bounces, K Mittelung, J Filter, M/L/V Diagnose, Esc beendet.
//!
//! --record rendert ohne Fenster: erst Einschwingen, dann stehende und
//! danach fliegende Frames, alle hintereinander als PPM.
//!
//! Das Fenster ist reines Wayland (libwayland-client dynamisch geladen, keine
//! Grafik-API): Pyrit rendert in einen CUDA-Puffer, die Demo kopiert ihn
//! direkt in den Shared-Memory-Puffer des Compositors (XRGB8888 = PYR_POST_BGRA).

const std = @import("std");
const pyrit = @import("pyrit");
const pyr = @import("pyrit_device");
const wl = @import("wayland.zig");
const api = pyrit.api;
const types = pyr.types;
const cuda = pyrit.cuda;
const scene = @import("scene.zig");

var drv: cuda.Driver = undefined;

fn cu(r: cuda.CUresult) void {
    if (r != cuda.CUDA_SUCCESS) std.debug.panic("CUDA: {s}", .{drv.errorString(r)});
}

fn req(r: api.Result) void {
    if (r != api.ok) std.debug.panic("{s}: {s}", .{ pyrit.pyr_result_string(r), pyrit.pyr_error_message() });
}

fn devAlloc(bytes: usize) u64 {
    var p: cuda.CUdeviceptr = 0;
    cu(drv.cuMemAlloc_v2(&p, bytes));
    return p;
}

fn nowSeconds(init: std.process.Init) f64 {
    const t = std.Io.Timestamp.now(init.io, .awake);
    return @as(f64, @floatFromInt(t.toNanoseconds())) / 1e9;
}

// ---------------------------------------------------------------------------
// Fensterzustand; die Wayland-Rückrufe schreiben hinein.
// ---------------------------------------------------------------------------

const Win = struct {
    c: wl.Client = undefined,
    display: *wl.Display = undefined,
    compositor: ?*wl.Proxy = null,
    shm: ?*wl.Proxy = null,
    seat: ?*wl.Proxy = null,
    wm_base: ?*wl.Proxy = null,
    surface: ?*wl.Proxy = null,
    xdg_surface: ?*wl.Proxy = null,
    toplevel: ?*wl.Proxy = null,
    pointer: ?*wl.Proxy = null,
    keyboard: ?*wl.Proxy = null,

    // Shared-Memory-Puffer (zwei, abwechselnd)
    pool: ?*wl.Proxy = null,
    bufs: [2]?*wl.Proxy = .{ null, null },
    busy: [2]bool = .{ false, false },
    mem: []align(std.heap.page_size_min) u8 = &.{},
    fd: std.posix.fd_t = -1,
    buf_w: u32 = 0,
    buf_h: u32 = 0,

    frame_cb: ?*wl.Proxy = null,
    frame_done: bool = true,

    width: u32 = 0,
    height: u32 = 0,
    configured: bool = false,
    running: bool = true,

    keys: [256]bool = .{false} ** 256,
    dragging: bool = false,
    mouse: [2]f32 = .{ 0, 0 },
    have_mouse: bool = false,
    dyaw: f32 = 0,
    dpitch: f32 = 0,
    esc: bool = false,
    zoom_in: bool = false,
    zoom_out: bool = false,
    toggle_fg: bool = false,
    /// F3: Debugbildschirm
    debug: bool = false,
    /// Tasten 1–5: Bildaufbau (Mode) wählen
    mode_key: ?u8 = null,
    /// Umschalter für die Bildeffekte (Tasten B, T, U, N, G)
    toggle: [10]bool = .{false} ** 10,

    fn marshal(self: *Win, p: *wl.Proxy, op: u32, iface: ?*const wl.Interface, args: anytype) ?*wl.Proxy {
        const ver = self.c.proxy_get_version(p);
        return @call(.auto, self.c.proxy_marshal_flags, .{ p, op, iface, ver, @as(u32, 0) } ++ args);
    }
};

var W: Win = .{};

fn fixedToFloat(v: i32) f32 {
    return @as(f32, @floatFromInt(v)) / 256.0;
}

// --- Rückrufe --------------------------------------------------------------

const RegistryListener = extern struct {
    global: *const fn (?*anyopaque, *wl.Proxy, u32, [*:0]const u8, u32) callconv(.c) void,
    global_remove: *const fn (?*anyopaque, *wl.Proxy, u32) callconv(.c) void,
};

fn bindGlobal(reg: *wl.Proxy, name: u32, iface: *const wl.Interface, version: u32) ?*wl.Proxy {
    return W.c.proxy_marshal_flags(reg, wl.registry_bind, iface, version, 0, name, iface.name, version, @as(?*anyopaque, null));
}

fn onGlobal(_: ?*anyopaque, reg: *wl.Proxy, name: u32, iface: [*:0]const u8, version: u32) callconv(.c) void {
    const s = std.mem.span(iface);
    if (std.mem.eql(u8, s, "wl_compositor")) {
        W.compositor = bindGlobal(reg, name, W.c.compositor, @min(version, 4));
    } else if (std.mem.eql(u8, s, "wl_shm")) {
        W.shm = bindGlobal(reg, name, W.c.shm, 1);
    } else if (std.mem.eql(u8, s, "wl_seat")) {
        W.seat = bindGlobal(reg, name, W.c.seat, 1);
        if (W.seat) |sp| _ = W.c.proxy_add_listener(sp, @ptrCast(&seat_listener), null);
    } else if (std.mem.eql(u8, s, "xdg_wm_base")) {
        W.wm_base = bindGlobal(reg, name, &wl.xdg_wm_base_interface, 1);
        if (W.wm_base) |bp| _ = W.c.proxy_add_listener(bp, @ptrCast(&wm_base_listener), null);
    }
}

fn onGlobalRemove(_: ?*anyopaque, _: *wl.Proxy, _: u32) callconv(.c) void {}

var registry_listener = RegistryListener{ .global = onGlobal, .global_remove = onGlobalRemove };

const PingListener = extern struct {
    ping: *const fn (?*anyopaque, *wl.Proxy, u32) callconv(.c) void,
};

fn onPing(_: ?*anyopaque, base: *wl.Proxy, serial: u32) callconv(.c) void {
    _ = W.marshal(base, wl.wm_base_pong, null, .{serial});
}

var wm_base_listener = PingListener{ .ping = onPing };

const XdgSurfaceListener = extern struct {
    configure: *const fn (?*anyopaque, *wl.Proxy, u32) callconv(.c) void,
};

fn onSurfaceConfigure(_: ?*anyopaque, xs: *wl.Proxy, serial: u32) callconv(.c) void {
    _ = W.marshal(xs, wl.xdg_surface_ack_configure, null, .{serial});
    W.configured = true;
}

var xdg_surface_listener = XdgSurfaceListener{ .configure = onSurfaceConfigure };

const ToplevelListener = extern struct {
    configure: *const fn (?*anyopaque, *wl.Proxy, i32, i32, *anyopaque) callconv(.c) void,
    close: *const fn (?*anyopaque, *wl.Proxy) callconv(.c) void,
};

fn onToplevelConfigure(_: ?*anyopaque, _: *wl.Proxy, w: i32, h: i32, _: *anyopaque) callconv(.c) void {
    if (std.c.getenv("PYRIT_DEMO_DEBUG") != null) std.debug.print("configure {d}x{d}\n", .{ w, h });
    if (w > 0 and h > 0) {
        W.width = @intCast(w);
        W.height = @intCast(h);
    }
}

fn onToplevelClose(_: ?*anyopaque, _: *wl.Proxy) callconv(.c) void {
    W.running = false;
}

var toplevel_listener = ToplevelListener{ .configure = onToplevelConfigure, .close = onToplevelClose };

const BufferListener = extern struct {
    release: *const fn (?*anyopaque, *wl.Proxy) callconv(.c) void,
};

fn onBufferRelease(_: ?*anyopaque, buf: *wl.Proxy) callconv(.c) void {
    for (W.bufs, 0..) |b, k| if (b == buf) {
        W.busy[k] = false;
    };
}

var buffer_listener = BufferListener{ .release = onBufferRelease };

const CallbackListener = extern struct {
    done: *const fn (?*anyopaque, *wl.Proxy, u32) callconv(.c) void,
};

fn onFrameDone(_: ?*anyopaque, cb: *wl.Proxy, _: u32) callconv(.c) void {
    W.c.proxy_destroy(cb);
    if (W.frame_cb == cb) W.frame_cb = null;
    W.frame_done = true;
}

var callback_listener = CallbackListener{ .done = onFrameDone };

const SeatListener = extern struct {
    capabilities: *const fn (?*anyopaque, *wl.Proxy, u32) callconv(.c) void,
    name: *const fn (?*anyopaque, *wl.Proxy, [*:0]const u8) callconv(.c) void,
};

fn onSeatCaps(_: ?*anyopaque, seat: *wl.Proxy, caps: u32) callconv(.c) void {
    if (caps & 1 != 0 and W.pointer == null) {
        W.pointer = W.marshal(seat, wl.seat_get_pointer, W.c.pointer, .{@as(?*anyopaque, null)});
        if (W.pointer) |p| _ = W.c.proxy_add_listener(p, @ptrCast(&pointer_listener), null);
    }
    if (caps & 2 != 0 and W.keyboard == null) {
        W.keyboard = W.marshal(seat, wl.seat_get_keyboard, W.c.keyboard, .{@as(?*anyopaque, null)});
        if (W.keyboard) |k| _ = W.c.proxy_add_listener(k, @ptrCast(&keyboard_listener), null);
    }
}

fn onSeatName(_: ?*anyopaque, _: *wl.Proxy, _: [*:0]const u8) callconv(.c) void {}

var seat_listener = SeatListener{ .capabilities = onSeatCaps, .name = onSeatName };

const KeyboardListener = extern struct {
    keymap: *const fn (?*anyopaque, *wl.Proxy, u32, i32, u32) callconv(.c) void,
    enter: *const fn (?*anyopaque, *wl.Proxy, u32, *wl.Proxy, *anyopaque) callconv(.c) void,
    leave: *const fn (?*anyopaque, *wl.Proxy, u32, *wl.Proxy) callconv(.c) void,
    key: *const fn (?*anyopaque, *wl.Proxy, u32, u32, u32, u32) callconv(.c) void,
    modifiers: *const fn (?*anyopaque, *wl.Proxy, u32, u32, u32, u32, u32) callconv(.c) void,
};

fn onKeymap(_: ?*anyopaque, _: *wl.Proxy, _: u32, fd: i32, _: u32) callconv(.c) void {
    // Wir werten evdev-Codes direkt aus, die Tabelle brauchen wir nicht.
    if (fd >= 0) _ = std.c.close(fd);
}

fn onKbdEnter(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: *wl.Proxy, _: *anyopaque) callconv(.c) void {}

fn onKbdLeave(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: *wl.Proxy) callconv(.c) void {
    W.keys = .{false} ** 256;
}

fn onKey(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: u32, key: u32, state: u32) callconv(.c) void {
    const down = state == 1;
    if (key < W.keys.len) W.keys[key] = down;
    if (!down) return;
    switch (key) {
        wl.key_esc => W.esc = true,
        wl.key_equal => W.zoom_in = true,
        wl.key_minus => W.zoom_out = true,
        wl.key_f => W.toggle_fg = true,
        wl.key_f3 => W.debug = !W.debug,
        wl.key_b => W.toggle[0] = true,
        wl.key_t => W.toggle[1] = true,
        wl.key_u => W.toggle[2] = true,
        wl.key_n => W.toggle[3] = true,
        wl.key_g => W.toggle[4] = true,
        wl.key_k => W.toggle[5] = true,
        wl.key_j => W.toggle[6] = true,
        // Diagnose bei Bewegung
        wl.key_m => W.toggle[7] = true, // zeitliche Mittelung (TAA) aus/an
        wl.key_l => W.toggle[8] = true, // LOD und Nachladen einfrieren
        wl.key_v => W.toggle[9] = true, // Wellen aus/an
        wl.key_1, wl.key_2, wl.key_3, wl.key_4, wl.key_5 => W.mode_key = @intCast(key - wl.key_1),
        else => {},
    }
}

fn onModifiers(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: u32, _: u32, _: u32, _: u32) callconv(.c) void {}

var keyboard_listener = KeyboardListener{
    .keymap = onKeymap,
    .enter = onKbdEnter,
    .leave = onKbdLeave,
    .key = onKey,
    .modifiers = onModifiers,
};

const PointerListener = extern struct {
    enter: *const fn (?*anyopaque, *wl.Proxy, u32, *wl.Proxy, i32, i32) callconv(.c) void,
    leave: *const fn (?*anyopaque, *wl.Proxy, u32, *wl.Proxy) callconv(.c) void,
    motion: *const fn (?*anyopaque, *wl.Proxy, u32, i32, i32) callconv(.c) void,
    button: *const fn (?*anyopaque, *wl.Proxy, u32, u32, u32, u32) callconv(.c) void,
    axis: *const fn (?*anyopaque, *wl.Proxy, u32, u32, i32) callconv(.c) void,
};

fn pointerAt(x: i32, y: i32) void {
    const px = fixedToFloat(x);
    const py = fixedToFloat(y);
    if (W.dragging and W.have_mouse) {
        W.dyaw += px - W.mouse[0];
        W.dpitch += py - W.mouse[1];
    }
    W.mouse = .{ px, py };
    W.have_mouse = true;
}

fn onPtrEnter(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: *wl.Proxy, x: i32, y: i32) callconv(.c) void {
    W.mouse = .{ fixedToFloat(x), fixedToFloat(y) };
    W.have_mouse = true;
}

fn onPtrLeave(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: *wl.Proxy) callconv(.c) void {
    W.have_mouse = false;
    W.dragging = false;
}

fn onPtrMotion(_: ?*anyopaque, _: *wl.Proxy, _: u32, x: i32, y: i32) callconv(.c) void {
    pointerAt(x, y);
}

fn onPtrButton(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: u32, button: u32, state: u32) callconv(.c) void {
    if (button == wl.btn_left) W.dragging = state == 1;
}

fn onPtrAxis(_: ?*anyopaque, _: *wl.Proxy, _: u32, _: u32, _: i32) callconv(.c) void {}

var pointer_listener = PointerListener{
    .enter = onPtrEnter,
    .leave = onPtrLeave,
    .motion = onPtrMotion,
    .button = onPtrButton,
    .axis = onPtrAxis,
};

// --- Fenster und Puffer ----------------------------------------------------

fn openWindow(title: [*:0]const u8, w: u32, h: u32) !void {
    W.c = try wl.Client.load();
    wl.initXdg(&W.c);
    W.display = W.c.display_connect(null) orelse return error.NoWaylandDisplay;
    W.width = w;
    W.height = h;

    const reg = W.c.proxy_marshal_flags(@ptrCast(W.display), wl.display_get_registry, W.c.registry, 1, 0, @as(?*anyopaque, null)) orelse return error.NoRegistry;
    _ = W.c.proxy_add_listener(reg, @ptrCast(&registry_listener), null);
    _ = W.c.display_roundtrip(W.display); // Globals
    _ = W.c.display_roundtrip(W.display); // Seat-Fähigkeiten

    const comp = W.compositor orelse return error.NoCompositor;
    const base = W.wm_base orelse return error.NoXdgShell;
    _ = W.shm orelse return error.NoShm;

    W.surface = W.marshal(comp, wl.compositor_create_surface, W.c.surface, .{@as(?*anyopaque, null)}) orelse return error.NoSurface;
    W.xdg_surface = W.marshal(base, wl.wm_base_get_xdg_surface, &wl.xdg_surface_interface, .{ @as(?*anyopaque, null), W.surface.? }) orelse return error.NoSurface;
    _ = W.c.proxy_add_listener(W.xdg_surface.?, @ptrCast(&xdg_surface_listener), null);
    W.toplevel = W.marshal(W.xdg_surface.?, wl.xdg_surface_get_toplevel, &wl.xdg_toplevel_interface, .{@as(?*anyopaque, null)}) orelse return error.NoSurface;
    _ = W.c.proxy_add_listener(W.toplevel.?, @ptrCast(&toplevel_listener), null);
    _ = W.marshal(W.toplevel.?, wl.toplevel_set_title, null, .{title});
    _ = W.marshal(W.toplevel.?, wl.toplevel_set_app_id, null, .{@as([*:0]const u8, "dev.velve.pyrit")});
    _ = W.marshal(W.surface.?, wl.surface_commit, null, .{});

    while (!W.configured and W.running) {
        if (W.c.display_dispatch(W.display) < 0) return error.WaylandLost;
    }
}

/// Legt Pool und zwei Puffer in der gewünschten Größe an.
fn resizeBuffers(w: u32, h: u32) !void {
    if (W.buf_w == w and W.buf_h == h) return;
    if (std.c.getenv("PYRIT_DEMO_DEBUG") != null) std.debug.print("Puffer {d}x{d}\n", .{ w, h });
    for (&W.bufs) |*b| if (b.*) |p| {
        W.c.proxy_destroy(p);
        b.* = null;
    };
    if (W.pool) |p| {
        W.c.proxy_destroy(p);
        W.pool = null;
    }
    if (W.mem.len != 0) {
        std.posix.munmap(W.mem);
        W.mem = &.{};
    }
    if (W.fd >= 0) {
        _ = std.c.close(W.fd);
        W.fd = -1;
    }

    const stride = w * 4;
    const size: usize = @as(usize, stride) * h * 2;
    W.fd = try std.posix.memfd_create("pyrit-view", 0);
    if (std.c.ftruncate(W.fd, @intCast(size)) != 0) return error.ShmResize;
    W.mem = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, W.fd, 0);

    W.pool = W.marshal(W.shm.?, wl.shm_create_pool, W.c.shm_pool, .{ @as(?*anyopaque, null), W.fd, @as(i32, @intCast(size)) }) orelse return error.NoShmPool;
    for (&W.bufs, 0..) |*b, k| {
        const off: i32 = @intCast(@as(usize, stride) * h * k);
        b.* = W.marshal(W.pool.?, wl.shm_pool_create_buffer, W.c.buffer, .{
            @as(?*anyopaque, null), off,
            @as(i32, @intCast(w)),  @as(i32, @intCast(h)),
            @as(i32, @intCast(stride)), wl.shm_format_xrgb8888,
        }) orelse return error.NoShmBuffer;
        _ = W.c.proxy_add_listener(b.*.?, @ptrCast(&buffer_listener), null);
        W.busy[k] = false;
    }
    W.buf_w = w;
    W.buf_h = h;
}

/// Liefert den Index eines freien Puffers (wartet notfalls auf release).
fn acquireBuffer() !usize {
    var guard: u32 = 0;
    while (true) {
        for (W.busy, 0..) |b, k| if (!b) return k;
        if (W.c.display_dispatch(W.display) < 0) return error.WaylandLost;
        guard += 1;
        if (guard > 1000) return 0;
    }
}

fn present(index: usize, vsync: bool) void {
    const surf = W.surface.?;
    _ = W.marshal(surf, wl.surface_attach, null, .{ W.bufs[index].?, @as(i32, 0), @as(i32, 0) });
    _ = W.marshal(surf, wl.surface_damage_buffer, null, .{
        @as(i32, 0),                     @as(i32, 0),
        @as(i32, @intCast(W.buf_w)), @as(i32, @intCast(W.buf_h)),
    });
    if (vsync) {
        W.frame_done = false;
        W.frame_cb = W.marshal(surf, wl.surface_frame, W.c.callback, .{@as(?*anyopaque, null)});
        if (W.frame_cb) |cb| _ = W.c.proxy_add_listener(cb, @ptrCast(&callback_listener), null);
    }
    _ = W.marshal(surf, wl.surface_commit, null, .{});
    W.busy[index] = true;
    _ = W.c.display_flush(W.display);
}

const Targets = struct {
    w: u32 = 0,
    h: u32 = 0,
    out_w: u32 = 0,
    out_h: u32 = 0,
    tg: api.Targets = std.mem.zeroes(api.Targets),
    ldr: u64 = 0,
    /// Zwischenbilder (bis zu fünf je Frame: DLSS Multi Frame Generation 6x)
    ldr_fg: [max_gens]u64 = .{0} ** max_gens,
    coverage: u32 = 4,

    fn free(self: *Targets) void {
        for ([_]u64{ self.tg.hits, self.tg.motion, self.tg.color, self.tg.normal, self.tg.albedo, self.tg.material, self.ldr }) |b| {
            if (b != 0) _ = drv.cuMemFree_v2(b);
        }
        for (self.ldr_fg) |b| {
            if (b != 0) _ = drv.cuMemFree_v2(b);
        }
        self.* = .{ .coverage = self.coverage };
    }

    /// Deckungsabtastung nur ohne Jitter (dort übernimmt sie die
    /// Kantenglättung); mit Jitter macht das der Upscaler über die Zeit.
    fn setMode(self: *Targets, m: Mode) void {
        self.coverage = if (m.jitter()) 1 else 2;
        self.tg.coverage = self.coverage;
    }

    fn resize(self: *Targets, w: u32, h: u32, out_w: u32, out_h: u32) void {
        if (self.w == w and self.h == h and self.out_w == out_w and self.out_h == out_h) return;
        self.free();
        const n: usize = @as(usize, w) * h;
        const n_out: usize = @as(usize, out_w) * out_h;
        self.tg = std.mem.zeroes(api.Targets);
        self.tg.hits = devAlloc(n * 16);
        self.tg.motion = devAlloc(n * 8);
        self.tg.color = devAlloc(n * 16);
        self.tg.normal = devAlloc(n * 16);
        self.tg.albedo = devAlloc(n * 16);
        // Rauheit und Metall je Pixel für DLSS Ray Reconstruction
        self.tg.material = devAlloc(n * 8);
        self.tg.ray_mask = 0x1;
        // Deckung an Kanten: 4 Abtastungen. Nur Kantenpixel zahlen dafür;
        // mit 2 blieben ferne Grate gegen den Himmel treppig.
        self.tg.coverage = self.coverage;
        // Texturen für die Ausgabe scharf abtasten, nicht für die Renderauflösung
        self.tg.detail_scale = @as(f32, @floatFromInt(out_h)) / @as(f32, @floatFromInt(h));
        self.ldr = devAlloc(n_out * 4);
        for (&self.ldr_fg) |*b| b.* = devAlloc(n_out * 4);
        self.w = w;
        self.h = h;
        self.out_w = out_w;
        self.out_h = out_h;
    }
};

/// höchstens so viele Zwischenbilder je Frame (6x)
const max_gens = 5;

fn sleepUntil(init: std.process.Init, t: f64) void {
    const d = t - nowSeconds(init);
    if (d <= 0.0002) return;
    const ts = std.c.timespec{ .sec = @intFromFloat(@floor(d)), .nsec = @intFromFloat((d - @floor(d)) * 1e9) };
    _ = std.c.nanosleep(&ts, null);
}

/// Angehefteter Zwischenspeicher eines Frames: die Zwischenbilder
/// (host[0..max_gens]) und das gerenderte Bild (host[max_gens])
const Stage = struct {
    host: [max_gens + 1][]u8 = .{&.{}} ** (max_gens + 1),
    bytes: usize = 0,
    gens: u32 = 0,
    w: u32 = 0,
    h: u32 = 0,
    valid: bool = false,

    fn ensure(self: *Stage, bytes: usize) !void {
        if (self.bytes >= bytes) return;
        self.free();
        for (&self.host) |*hb| {
            var ptr: ?*anyopaque = null;
            cu(drv.cuMemHostAlloc(&ptr, bytes, 0));
            hb.* = @as([*]u8, @ptrCast(ptr.?))[0..bytes];
        }
        self.bytes = bytes;
    }

    fn free(self: *Stage) void {
        for (&self.host) |*hb| if (hb.len != 0) {
            _ = drv.cuMemFreeHost(hb.ptr);
            hb.* = &.{};
        };
        self.* = .{};
    }
};

/// Kamera im Zuschauermodus: fliegt frei, ohne Kollision, wohin man schaut
const Spectator = struct {
    pos: [3]f64,
    yaw: f32 = 0.6,
    pitch: f32 = -0.25,

    fn dir(self: *const Spectator) [3]f64 {
        return .{ @cos(self.pitch) * @sin(self.yaw), @sin(self.pitch), @cos(self.pitch) * @cos(self.yaw) };
    }

    fn right(self: *const Spectator) [3]f64 {
        return .{ @cos(self.yaw), 0, -@sin(self.yaw) };
    }

    /// Ursprung der Szene in 1024er-Schritten: die Kamera bleibt nahe 0,
    /// damit f32 auf der GPU genau genug ist
    fn origin(self: *const Spectator) [3]f64 {
        return .{ @floor(self.pos[0] / 1024) * 1024, 0, @floor(self.pos[2] / 1024) * 1024 };
    }

    fn camera(self: *const Spectator, cam: *types.Camera, w: u32, h: u32, jitter_frame: ?u32) void {
        const o = self.origin();
        const d = self.dir();
        const eye = [3]f32{ @floatCast(self.pos[0] - o[0]), @floatCast(self.pos[1] - o[1]), @floatCast(self.pos[2] - o[2]) };
        const target = [3]f32{
            eye[0] + @as(f32, @floatCast(d[0])) * 100,
            eye[1] + @as(f32, @floatCast(d[1])) * 100,
            eye[2] + @as(f32, @floatCast(d[2])) * 100,
        };
        pyrit.pyr_camera_look_at(cam, &eye, &target, &.{ 0, 1, 0 });
        pyrit.pyr_camera_perspective(cam, 1.1, w, h, 0.05);
        // Kein Jitter: an einer Voxelkante entschied er jeden Frame neu, welche
        // der beiden verschieden beleuchteten Flächen das Pixel sieht – ein
        // echtes Wechselsignal, das kein Filter glätten kann (wandernde
        // Schatten). Die Kantenglättung macht `tg.coverage` im Frame.
        cam.jitter = .{ 0, 0 };
        if (jitter_frame) |f| pyrit.pyr_jitter_halton(f, &cam.jitter);
    }
};

/// Bildaufbau: eigener Denoiser in voller Auflösung oder Hochskalieren.
/// DLSS und TAAU brauchen Subpixel-Jitter, um Details zu rekonstruieren – sie
/// sind dafür gebaut und verteilen ihn sauber über die Zeit. Nur der eigene
/// Denoiser läuft ohne: dort verursachte der Jitter die wandernden Schatten.
/// Zwischenbilder (Taste F) macht Pyrits eigene Frame Generation oder DLSS
/// Frame Generation (NGX über die kopflose Vulkan-Interop-Schicht, src/dlssg.zig).
/// Zwischenbild je Frame
const overlay = @import("overlay.zig");

const Fg = enum {
    off,
    pyrit,
    /// DLSS Frame Generation 2x, 3x, 4x (1–3 Zwischenbilder, Multi Frame Generation)
    dlss,
    dlss3,
    dlss4,
    dlss6,

    fn next(f: Fg) Fg {
        return switch (f) {
            .off => .pyrit,
            .pyrit => .dlss,
            .dlss => .dlss3,
            .dlss3 => .dlss4,
            .dlss4 => .dlss6,
            .dlss6 => .off,
        };
    }
    fn isDlss(f: Fg) bool {
        return f == .dlss or f == .dlss3 or f == .dlss4 or f == .dlss6;
    }
    /// Zwischenbilder je gerendertem Frame
    fn count(f: Fg) u32 {
        return switch (f) {
            .off => 0,
            .pyrit, .dlss => 1,
            .dlss3 => 2,
            .dlss4 => 3,
            .dlss6 => 5,
        };
    }
    fn flags(f: Fg) u32 {
        return if (f.isDlss()) api.framegen_dlss | api.framegenCount(f.count()) else 0;
    }
    fn label(f: Fg) []const u8 {
        return switch (f) {
            .off => "",
            .pyrit => " +Zwischenbild",
            .dlss => " +DLSS-FG 2x",
            .dlss3 => " +DLSS-FG 3x",
            .dlss4 => " +DLSS-FG 4x",
            .dlss6 => " +DLSS-FG 6x",
        };
    }
};

const Mode = enum {
    native,
    taau,
    dlaa,
    dlss,
    rr,

    fn upscaler(m: Mode) u32 {
        return switch (m) {
            .native => api.upscaler_none,
            .taau => api.upscaler_taau,
            .dlaa, .dlss => api.upscaler_dlss,
            .rr => api.upscaler_dlss_rr,
        };
    }

    /// Teiler der Renderauflösung
    fn scale(m: Mode) u32 {
        return switch (m) {
            .native, .dlaa, .rr => 1,
            .taau, .dlss => 2,
        };
    }

    fn jitter(m: Mode) bool {
        return m != .native;
    }

    fn name(m: Mode) []const u8 {
        return switch (m) {
            .native => "nativ",
            .taau => "TAAU 2x",
            .dlaa => "DLAA",
            .dlss => "DLSS 2x",
            .rr => "DLSS Ray Reconstruction",
        };
    }
};

/// Fluggeschwindigkeit wie der Zuschauermodus in Minecraft (Blöcke je Sekunde)
const fly_speed: f64 = 11;
const sprint_factor: f64 = 4;

const Options = struct {
    out_w: u32 = 1280,
    out_h: u32 = 720,
    /// 0 = nach Modus
    scale: u32 = 0,
    mode: Mode = .native,
    flags: u32 = 0,
    /// Zielgröße eines Voxels in Ausgabepixeln, bevor die Welt gröbere Stufen
    /// nimmt (4 ließ nahe Bereiche sichtbar grob; 1 sprengt das Chunk-Budget)
    voxel_px: f32 = 2,
    seed: u32 = 1,
    fg: Fg = .off,
    half_gi: bool = true,
    vsync: bool = true,
    super: u32 = 1,
    check: bool = false,
    max_frames: u32 = 0,
    shot: ?[]const u8 = null,
    start: [2]f64 = .{ 0, 0 },
    yaw: f32 = 0.6,
    pitch: f32 = -0.25,
    /// über dem Gelände (Blöcke)
    altitude: f64 = 24,
    // Aufnahme ohne Fenster
    record: ?[]const u8 = null,
    warm: u32 = 40,
    still: u32 = 10,
    move: u32 = 10,
    /// Drehung je Frame während der Fahrt (Radiant)
    turn: f32 = 0.004,
    /// Je Flugframe eine eingeschwungene Referenz mit so vielen Frames
    /// stehender Kamera (eigene Ansicht); 0 = aus
    reference: u32 = 0,
    // Diagnose
    freeze_lod: bool = false,
    clamp: f32 = -1,
    denoise: i32 = -1,
    fog: f32 = -1,
    fog_steps: u32 = 0,
    /// feste Belichtung statt der Automatik (0 = Automatik)
    exposure: f32 = 0,
    no_sun: bool = false,
    no_waves: bool = false,
    /// Teilbaumgröße der RT-Primitive (2^n Voxel, 0 = Vorgabe)
    rt_leaf: u32 = 0,
    /// Messung am Ende der Aufnahme: so viele Blöcke setzen (0 = aus)
    edit_bench: u32 = 0,
    /// Prüfung der Bewegungsvektoren der Frame Generation (bewegter Würfel)
    mv_test: bool = false,
};

fn parseArgs(args: []const [:0]const u8) !Options {
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const has_val = i + 1 < args.len;
        if (std.mem.eql(u8, a, "--size") and has_val) {
            i += 1;
            var it = std.mem.splitScalar(u8, args[i], 'x');
            o.out_w = try std.fmt.parseInt(u32, it.next() orelse "1280", 10);
            o.out_h = try std.fmt.parseInt(u32, it.next() orelse "720", 10);
        } else if (std.mem.eql(u8, a, "--mode") and has_val) {
            i += 1;
            o.mode = std.meta.stringToEnum(Mode, args[i]) orelse {
                std.debug.print("--mode: native, taau, dlaa, dlss oder rr\n", .{});
                return error.Usage;
            };
        } else if (std.mem.eql(u8, a, "--scale") and has_val) {
            i += 1;
            o.scale = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--voxel-px") and has_val) {
            i += 1;
            o.voxel_px = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--seed") and has_val) {
            i += 1;
            o.seed = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--start") and has_val) {
            i += 1;
            var it = std.mem.splitScalar(u8, args[i], ',');
            o.start[0] = try std.fmt.parseFloat(f64, it.next() orelse "0");
            o.start[1] = try std.fmt.parseFloat(f64, it.next() orelse "0");
        } else if (std.mem.eql(u8, a, "--yaw") and has_val) {
            i += 1;
            o.yaw = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--pitch") and has_val) {
            i += 1;
            o.pitch = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--altitude") and has_val) {
            i += 1;
            o.altitude = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, a, "--frames") and has_val) {
            i += 1;
            o.max_frames = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--shot") and has_val) {
            i += 1;
            o.shot = args[i];
        } else if (std.mem.eql(u8, a, "--record") and has_val) {
            i += 1;
            o.record = args[i];
        } else if (std.mem.eql(u8, a, "--warm") and has_val) {
            i += 1;
            o.warm = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--still") and has_val) {
            i += 1;
            o.still = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--move") and has_val) {
            i += 1;
            o.move = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--reference") and has_val) {
            i += 1;
            o.reference = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--fog") and has_val) {
            i += 1;
            o.fog = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--fog-steps") and has_val) {
            i += 1;
            o.fog_steps = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--exposure") and has_val) {
            i += 1;
            o.exposure = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--no-waves")) {
            o.no_waves = true;
        } else if (std.mem.eql(u8, a, "--no-sun")) {
            o.no_sun = true;
        } else if (std.mem.eql(u8, a, "--freeze-lod")) {
            o.freeze_lod = true;
        } else if (std.mem.eql(u8, a, "--clamp") and has_val) {
            i += 1;
            o.clamp = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--denoise") and has_val) {
            i += 1;
            o.denoise = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--turn") and has_val) {
            i += 1;
            o.turn = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--fg")) {
            o.fg = .pyrit;
            if (has_val) {
                if (std.meta.stringToEnum(Fg, args[i + 1])) |f| {
                    i += 1;
                    o.fg = f;
                }
            }
        } else if (std.mem.eql(u8, a, "--check")) {
            o.check = true;
        } else if (std.mem.eql(u8, a, "--super") and has_val) {
            i += 1;
            o.super = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-vsync")) {
            o.vsync = false;
        } else if (std.mem.eql(u8, a, "--no-half-gi")) {
            o.half_gi = false;
        } else if (std.mem.eql(u8, a, "--mv-test")) {
            o.mv_test = true;
        } else if (std.mem.eql(u8, a, "--no-images")) {
            no_images = true;
        } else if (std.mem.eql(u8, a, "--edit-bench") and has_val) {
            i += 1;
            o.edit_bench = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--rt-leaf") and has_val) {
            i += 1;
            o.rt_leaf = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-rt")) {
            o.flags |= api.create_no_rt;
        } else {
            std.debug.print("unbekanntes Argument: {s}\n", .{a});
            return error.Usage;
        }
    }
    return o;
}

/// Alles, was Fenster und Aufnahme gemeinsam haben
const App = struct {
    ctx: ?*anyopaque = null,
    gen: scene.Generator,
    world: ?*anyopaque = null,
    wi: api.WorldInfo,
    light: types.Lighting,
    water: types.Material,
    view: api.Handle = null,
    stream: cuda.CUstream = null,

    fn init(self: *App, init_: std.process.Init, o: Options) !void {
        drv = cuda.Driver.load() catch return error.NoCuda;
        cu(drv.cuInit(0));
        var dev: cuda.CUdevice = 0;
        cu(drv.cuDeviceGet(&dev, 0));
        var cu_ctx: cuda.CUcontext = null;
        cu(drv.cuDevicePrimaryCtxRetain(&cu_ctx, dev));
        cu(drv.cuCtxSetCurrent(cu_ctx));

        // eigener Stream: die Demo hängt ihre Kopien zum Fenster daran an
        cu(drv.cuStreamCreate(&self.stream, cuda.CU_STREAM_NON_BLOCKING));
        var ci = scene.createInfo(o.flags);
        ci.cuda_stream = self.stream;
        req(pyrit.pyr_create(&ci, @ptrCast(&self.ctx)));
        try scene.materials(self.ctx, init_.gpa);
        self.water = scene.water();
        if (o.no_waves) {
            self.water.wave_height = 0;
            req(pyrit.pyr_material_set(@ptrCast(self.ctx), scene.terrain.mat_water, &self.water));
        }
        self.gen = try scene.Generator.init(&drv, .{ .seed = o.seed });
        // Adresse bleibt fest: Pyrit hält `user` für die Lebensdauer der Welt
        self.wi = self.gen.worldInfo();
        self.wi.voxel_pixels = o.voxel_px;
        self.wi.rt_leaf_log2 = o.rt_leaf;
        req(pyrit.pyr_world_create(@ptrCast(self.ctx), &self.wi, @ptrCast(&self.world)));

        self.light = try scene.lighting(self.ctx, init_.gpa, .{});
        if (o.half_gi) self.light.flags |= types.lighting_gi_half;
        if (o.fog >= 0) self.light.fog_density = o.fog;
        if (o.fog_steps > 0) self.light.fog_steps = o.fog_steps;
        if (o.no_sun) self.light.sun_color = .{ 0, 0, 0 };
        req(pyrit.pyr_set_lighting(@ptrCast(self.ctx), &self.light));
        req(pyrit.pyr_view_create(@ptrCast(self.ctx), &self.view));
    }

    fn deinit(self: *App) void {
        _ = pyrit.pyr_world_destroy(@ptrCast(self.ctx), @ptrCast(self.world));
        self.gen.deinit();
        pyrit.pyr_destroy(@ptrCast(self.ctx));
    }

    fn spectatorAt(self: *const App, o: Options) Spectator {
        const g = self.gen.height(o.start[0], o.start[1]);
        const ground = @max(@as(f64, g), @as(f64, self.gen.params.sea_level));
        return .{ .pos = .{ o.start[0], ground + o.altitude, o.start[1] }, .yaw = o.yaw, .pitch = o.pitch };
    }

    /// Welt um die Kamera vollständig laden (Ladebildschirm)
    fn load(self: *App, sp: *const Spectator, cam: *const types.Camera) void {
        const org = sp.origin();
        var st: api.WorldStats = undefined;
        req(pyrit.pyr_world_update(@ptrCast(self.ctx), @ptrCast(self.world), &sp.pos, &org, cam));
        req(pyrit.pyr_world_stats(@ptrCast(self.world), &st));
        var guard: u32 = 0;
        while (st.pending_chunks > 0 and guard < 8000) : (guard += 1) {
            req(pyrit.pyr_world_wait(@ptrCast(self.ctx), @ptrCast(self.world), &sp.pos));
            req(pyrit.pyr_world_update(@ptrCast(self.ctx), @ptrCast(self.world), &sp.pos, &org, cam));
            req(pyrit.pyr_world_stats(@ptrCast(self.world), &st));
        }
    }

    /// Ein Frame: Welt nachführen, rendern, nachbearbeiten
    /// Welt nachführen (Planer, Laden, Sichtbarkeit). Die Welt wählt ihr LOD
    /// nach der *Ausgabe*auflösung: beim Hochskalieren rendert die Kamera
    /// kleiner, das Bild zeigt die Voxel aber in voller Größe (mit der
    /// Renderkamera waren nahe Voxel doppelt so grob).
    fn updateWorld(self: *App, sp: *const Spectator, cam: *const types.Camera, tgs: *const Targets) void {
        const org = sp.origin();
        var out_cam = cam.*;
        out_cam.width = tgs.out_w;
        out_cam.height = tgs.out_h;
        req(pyrit.pyr_world_update(@ptrCast(self.ctx), @ptrCast(self.world), &sp.pos, &org, &out_cam));
    }

    fn frame(self: *App, sp: *const Spectator, cam: *const types.Camera, time: f64, tgs: *Targets, post: *api.PostInfo, update_world: bool) void {
        self.frameOn(self.view, sp, cam, time, tgs, post, update_world);
    }

    fn frameOn(self: *App, view: api.Handle, sp: *const Spectator, cam: *const types.Camera, time: f64, tgs: *Targets, post: *api.PostInfo, update_world: bool) void {
        const org = sp.origin();
        // Wolkenschatten haften an der Welt, nicht am Render-Ursprung
        const before = self.light.sun_shadow_offset;
        const medium_before = self.light.camera_medium_density;
        scene.cloudShadowOffset(&self.light, org);
        // Unter Wasser: die Welt trägt nur die Oberfläche, das Medium darunter
        // kennt die Anwendung (Kamera unter dem Meeresspiegel, über dem Grund)
        const sea: f64 = self.gen.params.sea_level;
        const under = sp.pos[1] < sea and self.gen.height(sp.pos[0], sp.pos[2]) < sp.pos[1];
        scene.underwater(&self.light, under, @floatCast(sea - org[1]));
        if (before[0] != self.light.sun_shadow_offset[0] or before[1] != self.light.sun_shadow_offset[1] or
            medium_before != self.light.camera_medium_density)
            req(pyrit.pyr_set_lighting(@ptrCast(self.ctx), &self.light));
        if (update_world) self.updateWorld(sp, cam, tgs);
        const fi = api.FrameInfo{ .time = time, .origin = org };
        req(pyrit.pyr_commit(@ptrCast(self.ctx), &fi));
        req(pyrit.pyr_render(@ptrCast(self.ctx), view, cam, &tgs.tg));
        post.output_ldr = tgs.ldr;
        post.output_width = tgs.out_w;
        post.output_height = tgs.out_h;
        req(pyrit.pyr_postprocess(@ptrCast(self.ctx), view, &tgs.tg, post));
    }
};

/// BGRA-Bild (w x h) als PPM schreiben
/// --no-images: Aufnahmen messen nur (keine Dateien – /tmp ist eine RAM-Disk,
/// und Hunderte Bilder zu 6 MB füllten sie)
var no_images = false;

fn writePpm(init: std.process.Init, path: []const u8, bgra: []const u8, w: u32, h: u32) !void {
    if (no_images) return;
    const gpa = init.gpa;
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(gpa);
    var head: [64]u8 = undefined;
    try file.appendSlice(gpa, try std.fmt.bufPrint(&head, "P6\n{d} {d}\n255\n", .{ w, h }));
    try file.ensureUnusedCapacity(gpa, @as(usize, w) * h * 3);
    var k: usize = 0;
    while (k + 4 <= @as(usize, w) * h * 4) : (k += 4) {
        file.appendSliceAssumeCapacity(&.{ bgra[k + 2], bgra[k + 1], bgra[k] });
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = file.items });
}

/// Anteil der Pixel (in %), deren stärkster Farbkanal sich um mehr als
/// `threshold` Stufen unterscheidet
fn loudShare(a: []const u8, b: []const u8, threshold: u8) f64 {
    var n: u64 = 0;
    var k: usize = 0;
    while (k + 4 <= a.len) : (k += 4) {
        var m: u8 = 0;
        inline for (0..3) |q| m = @max(m, if (a[k + q] > b[k + q]) a[k + q] - b[k + q] else b[k + q] - a[k + q]);
        if (m > threshold) n += 1;
    }
    return @as(f64, @floatFromInt(n)) * 100 / @as(f64, @floatFromInt(a.len / 4));
}

/// Mittlerer Unterschied (Stufen) und Anteil über 8 Stufen (%); `map`
/// bekommt den stärksten Kanalunterschied je Pixel, verstärkt
fn compare(a: []const u8, b: []const u8, map: []u8) [2]f64 {
    var sum: u64 = 0;
    var loud: u64 = 0;
    var k: usize = 0;
    while (k + 4 <= a.len) : (k += 4) {
        var m: u32 = 0;
        inline for (0..3) |q| {
            const d: u32 = if (a[k + q] > b[k + q]) a[k + q] - b[k + q] else b[k + q] - a[k + q];
            sum += d;
            m = @max(m, d);
        }
        if (m > 8) loud += 1;
        map[k / 4] = @intCast(@min(m * 6, 255));
    }
    const px: f64 = @floatFromInt(a.len / 4);
    return .{ @as(f64, @floatFromInt(sum)) / (px * 3), @as(f64, @floatFromInt(loud)) * 100 / px };
}

fn writePgm(init: std.process.Init, path: []const u8, gray: []const u8, w: u32, h: u32) !void {
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(init.gpa);
    var head: [64]u8 = undefined;
    try file.appendSlice(init.gpa, try std.fmt.bufPrint(&head, "P5\n{d} {d}\n255\n", .{ w, h }));
    try file.appendSlice(init.gpa, gray);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = file.items });
}

/// Aufnahme ohne Fenster: erst `warm` Frames Einschwingen, dann `still`
/// Frames mit stehender Kamera und `move` Frames im Flug (vorwärts mit
/// Fluggeschwindigkeit, dabei leicht drehend), alle hintereinander.
/// Die Bilder landen als frame_000.ppm … im Verzeichnis `dir`.
fn record(init: std.process.Init, o: Options, dir: []const u8) !void {
    var app: App = undefined;
    try app.init(init, o);
    defer app.deinit();

    var tgs = Targets{};
    defer tgs.free();
    const ss = @max(o.super, 1);
    const post_w = o.out_w * ss;
    const post_h = o.out_h * ss;
    const div = if (o.scale > 0) o.scale else o.mode.scale();
    const rw = @max(post_w / div, 16);
    const rh = @max(post_h / div, 16);
    tgs.resize(rw, rh, post_w, post_h);

    var post = scene.post();
    post.flags = api.post_bgra;
    if (o.clamp >= 0) post.clamp_sigma = o.clamp;
    if (o.denoise >= 0) post.denoise_iterations = @intCast(o.denoise);
    var fxi = scene.fx();
    fxi.supersample = ss;
    if (o.exposure > 0) {
        fxi.flags &= ~api.postfx_auto_exposure;
        post.exposure = o.exposure;
    }
    post.fx = &fxi;

    var sp = app.spectatorAt(o);
    var cam = std.mem.zeroes(types.Camera);
    const jit = o.mode.jitter();
    sp.camera(&cam, rw, rh, null);
    app.load(&sp, &cam);

    std.Io.Dir.cwd().createDirPath(init.io, dir) catch {};
    const bytes: usize = @as(usize, post_w) * post_h * 4;
    post.upscaler = o.mode.upscaler();
    tgs.setMode(o.mode);
    const px = try init.gpa.alloc(u8, bytes);
    defer init.gpa.free(px);

    // Referenz: eigene Ansicht mit eigener Historie
    var ref_view: api.Handle = null;
    if (o.reference > 0) req(pyrit.pyr_view_create(@ptrCast(app.ctx), &ref_view));
    const ref_px = try init.gpa.alloc(u8, if (o.reference > 0) bytes else 0);
    defer init.gpa.free(ref_px);
    const diff = try init.gpa.alloc(u8, if (o.reference > 0) bytes / 4 else 0);
    defer init.gpa.free(diff);
    var err_sum: f64 = 0;
    var loud_sum: f64 = 0;
    var err_n: u32 = 0;
    const Pose = struct { sp: Spectator, cam: types.Camera, t: f64, index: u32 };
    var poses: std.ArrayList(Pose) = .empty;
    defer poses.deinit(init.gpa);
    var moved_px: std.ArrayList(u8) = .empty;
    defer moved_px.deinit(init.gpa);
    // Unruhe bei stehender Kamera: Unterschied aufeinanderfolgender Frames
    const prev_px = try init.gpa.alloc(u8, bytes);
    defer init.gpa.free(prev_px);
    var still_loud: f64 = 0;
    var still_n: u32 = 0;

    const dt: f64 = 1.0 / 60.0;
    var t: f64 = 0;
    var n_out: u32 = 0;
    const total = o.warm + o.still + o.move;
    var f: u32 = 0;
    const t0 = nowSeconds(init);
    var ms_still: f64 = 0;
    var ms_move: f64 = 0;
    var ms_fg: f64 = 0;
    while (f < total) : (f += 1) {
        const moving = f >= o.warm + o.still;
        if (moving) {
            // Fahrt wie mit gedrückter W-Taste, dabei eine langsame Drehung
            const d = sp.dir();
            for (0..3) |k| sp.pos[k] += d[k] * fly_speed * dt;
            sp.yaw += o.turn;
        }
        sp.camera(&cam, rw, rh, if (jit) f else null);
        const tf = nowSeconds(init);
        // Diagnose: im Flug die Welt einfrieren (keine LOD-Wechsel)
        app.frame(&sp, &cam, t, &tgs, &post, !(o.freeze_lod and moving));
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
        const ms = (nowSeconds(init) - tf) * 1000;
        var kf: u32 = 1;
        while (f >= o.warm and o.fg != .off and o.mode != .native and kf <= o.fg.count()) : (kf += 1) {
            // Zwischenbilder vor diesem Frame (mid_N_k liegt zwischen frame_N-1 und frame_N)
            var fgi = std.mem.zeroes(api.FrameGenInfo);
            fgi.output_ldr = tgs.ldr_fg[0];
            fgi.flags = (post.flags & api.post_bgra) | o.fg.flags();
            fgi.t = @as(f32, @floatFromInt(kf)) / @as(f32, @floatFromInt(o.fg.count() + 1));
            const tg = nowSeconds(init);
            req(pyrit.pyr_frame_generate(@ptrCast(app.ctx), app.view, &fgi));
            req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
            ms_fg += (nowSeconds(init) - tg) * 1000;
            cu(drv.cuMemcpyDtoH_v2(px.ptr, tgs.ldr_fg[0], bytes));
            var buf: [512]u8 = undefined;
            const path = try std.fmt.bufPrint(&buf, "{s}/mid_{d:0>3}_{d}.ppm", .{ dir, n_out, kf });
            try writePpm(init, path, px, post_w, post_h);
        }
        if (f >= o.warm) {
            if (moving) ms_move += ms else ms_still += ms;
            cu(drv.cuMemcpyDtoH_v2(px.ptr, tgs.ldr, bytes));
            var buf: [512]u8 = undefined;
            const path = try std.fmt.bufPrint(&buf, "{s}/frame_{d:0>3}.ppm", .{ dir, n_out });
            try writePpm(init, path, px, post_w, post_h);
            if (!moving and f > o.warm) {
                still_loud += loudShare(px, prev_px, 4);
                still_n += 1;
            }
            @memcpy(prev_px, px);
            if (moving and o.reference > 0) {
                // für die Referenz nach der Aufnahme: Pose, Zeit und Bild
                try poses.append(init.gpa, .{ .sp = sp, .cam = cam, .t = t, .index = n_out });
                try moved_px.appendSlice(init.gpa, px);
            }
            n_out += 1;
        }
        t += dt;
    }
    // Referenzen erst nach der Aufnahme: sie teilen sich den Kontext (Belichtung,
    // Framezähler) mit der Aufnahme und dürfen sie nicht beeinflussen.
    for (poses.items, 0..) |q, qi| {
        // Dieselbe Pose, aber eingeschwungen: so sähe der Frame ohne jede
        // Nachwirkung der Bewegung aus. Die Welt bleibt dabei stehen.
        req(pyrit.pyr_view_reset_history(@ptrCast(app.ctx), ref_view));
        var k: u32 = 0;
        while (k < o.reference) : (k += 1) app.frameOn(ref_view, &q.sp, &q.cam, q.t, &tgs, &post, false);
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
        cu(drv.cuMemcpyDtoH_v2(ref_px.ptr, tgs.ldr, bytes));
        const e = compare(moved_px.items[qi * bytes ..][0..bytes], ref_px, diff);
        err_sum += e[0];
        loud_sum += e[1];
        err_n += 1;
        var buf: [512]u8 = undefined;
        const rp = try std.fmt.bufPrint(&buf, "{s}/ref_{d:0>3}.ppm", .{ dir, q.index });
        try writePpm(init, rp, ref_px, post_w, post_h);
        const dp = try std.fmt.bufPrint(&buf, "{s}/diff_{d:0>3}.pgm", .{ dir, q.index });
        try writePgm(init, dp, diff, post_w, post_h);
    }
    if (still_n > 0) std.debug.print("Stehend: {d:.3} % der Pixel ändern sich von Frame zu Frame um mehr als 4 Stufen\n", .{still_loud / @as(f64, @floatFromInt(still_n))});
    if (err_n > 0) std.debug.print("Im Flug gegen eingeschwungene Referenz: mittlerer Fehler {d:.3} Stufen, {d:.3} % der Pixel über 8 Stufen (Karten: diff_*.pgm)\n", .{ err_sum / @as(f64, @floatFromInt(err_n)), loud_sum / @as(f64, @floatFromInt(err_n)) });
    if (ms_fg > 0) std.debug.print("Zwischenbild ({s}): {d:.2} ms je Bild (mid_*.ppm)\n", .{ @tagName(o.fg), ms_fg / @as(f64, @floatFromInt(@max((o.still + o.move) * o.fg.count(), 1))) });
    var st: api.WorldStats = undefined;
    req(pyrit.pyr_world_stats(@ptrCast(app.world), &st));
    std.debug.print("Aufnahme: {d} Bilder in {s} ({d} stehend, {d} im Flug), {d}x{d}, Frame {d:.1} ms stehend / {d:.1} ms im Flug, gesamt {d:.1} s, {d} Chunks, {d:.0} MiB\n", .{
        n_out,                                                   dir,                                               o.still, o.move, post_w, post_h,
        ms_still / @as(f64, @floatFromInt(@max(o.still, 1))), ms_move / @as(f64, @floatFromInt(@max(o.move, 1))), nowSeconds(init) - t0,
        st.resident_chunks,                                      @as(f64, @floatFromInt(st.bytes)) / (1 << 20),
    });
    if (o.mv_test) mvTest(init, &app, &sp, &cam, &tgs, &post, dir);
    if (o.edit_bench > 0) {
        editBench(init, &app, &sp, &cam, &tgs, &post, o.edit_bench, false);
        editBench(init, &app, &sp, &cam, &tgs, &post, o.edit_bench, true);
        digBench(init, &app, &sp, &cam, &tgs);
    }
}

/// Prüfung der Bewegungsvektoren für die Frame Generation: ein Würfel
/// (eigene Instanz) fliegt quer durchs Bild, die Kamera steht. Schreibt die
/// Zwischenbilder als mvtest_<k>.ppm; zwei Läufe mit PYRIT_DLSSG_MV=+/-1
/// vergleichen – sind sie gleich, liest DLSS-G die Vektoren nicht.
fn mvTest(init: std.process.Init, app: *App, sp: *Spectator, cam: *types.Camera, tgs: *Targets, post: *api.PostInfo, dir: []const u8) void {
    // die sechs Prüfbilder auch mit --no-images schreiben
    const saved = no_images;
    no_images = false;
    defer no_images = saved;
    var vox: [512]api.Voxel = undefined;
    for (&vox, 0..) |*v, i| v.* = .{ .x = @intCast(i % 8), .y = @intCast((i / 8) % 8), .z = @intCast(i / 64), .attribute = scene.terrain.Block.snow.attribute() };
    var geo: api.Handle = null;
    req(pyrit.pyr_geometry_build(@ptrCast(app.ctx), 3, &vox, vox.len, api.build_host_input, @ptrCast(&geo)));
    var inst: api.Handle = null;
    req(pyrit.pyr_instance_create(@ptrCast(app.ctx), @ptrCast(geo), @ptrCast(&inst)));
    req(pyrit.pyr_instance_set_mask(@ptrCast(app.ctx), @ptrCast(inst), 0x1));
    const d = sp.dir();
    const org = sp.origin();
    const r = sp.right();
    // 50 Blöcke voraus, 6 über dem Boden; startet links und fliegt nach rechts
    const base = [3]f64{ sp.pos[0] + d[0] * 50, 0, sp.pos[2] + d[2] * 50 };
    const ground = app.gen.height(base[0], base[2]);
    var k: u32 = 0;
    while (k < 16) : (k += 1) {
        const off = (@as(f64, @floatFromInt(k)) - 8) * 1.5;
        const pos = [3]f64{ base[0] + r[0] * off - org[0], ground + 6 - org[1], base[2] + r[2] * off - org[2] };
        const xf = [12]f32{ 1, 0, 0, @floatCast(pos[0]), 0, 1, 0, @floatCast(pos[1]), 0, 0, 1, @floatCast(pos[2]) };
        req(pyrit.pyr_instance_set_transform(@ptrCast(app.ctx), @ptrCast(inst), &xf));
        sp.camera(cam, tgs.w, tgs.h, k);
        app.frame(sp, cam, nowSeconds(init), tgs, post, false);
        if (k >= 4) {
            var fgi = std.mem.zeroes(api.FrameGenInfo);
            fgi.output_ldr = tgs.ldr_fg[0];
            fgi.flags = (post.flags & api.post_bgra) | api.framegen_dlss;
            req(pyrit.pyr_frame_generate(@ptrCast(app.ctx), app.view, &fgi));
        }
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
        if (k >= 10 and k < 13) {
            const bytes = @as(usize, tgs.out_w) * tgs.out_h * 4;
            const px = init.gpa.alloc(u8, bytes) catch return;
            defer init.gpa.free(px);
            var buf: [512]u8 = undefined;
            cu(drv.cuMemcpyDtoH_v2(px.ptr, tgs.ldr_fg[0], bytes));
            writePpm(init, std.fmt.bufPrint(&buf, "{s}/mvtest_mid_{d}.ppm", .{ dir, k }) catch return, px, tgs.out_w, tgs.out_h) catch {};
            cu(drv.cuMemcpyDtoH_v2(px.ptr, tgs.ldr, bytes));
            writePpm(init, std.fmt.bufPrint(&buf, "{s}/mvtest_echt_{d}.ppm", .{ dir, k }) catch return, px, tgs.out_w, tgs.out_h) catch {};
        }
    }
    _ = pyrit.pyr_instance_destroy(@ptrCast(app.ctx), @ptrCast(inst));
}

/// Entfernen prüfen: den Haufen wieder weg und darunter ein 5x5-Loch zwei
/// Blöcke tief; danach Höhe und Attribut eines 64x64-Rasters als Prüfsumme
/// (mit PYRIT_NO_PATH_EDIT=1 vergleichen: der volle Neubau muss gleich sein)
fn digBench(init: std.process.Init, app: *App, sp: *Spectator, cam: *types.Camera, tgs: *Targets) void {
    const d = sp.dir();
    const cx = sp.pos[0] + d[0] * 80;
    const cz = sp.pos[2] + d[2] * 80;
    var edits: [125]api.WorldEdit = undefined;
    var n: usize = 0;
    var i: u32 = 0;
    while (i < 25) : (i += 1) {
        const x = cx + @as(f64, @floatFromInt(i % 5)) - 2;
        const z = cz + @as(f64, @floatFromInt(i / 5)) - 2;
        const g = @floor(app.gen.height(x, z));
        var dy: i32 = -1;
        while (dy <= 2) : (dy += 1) {
            edits[n] = .{ .x = @intFromFloat(@floor(x)), .y = @as(i64, @intFromFloat(g)) + dy, .z = @intFromFloat(@floor(z)), .attribute = 0 };
            n += 1;
        }
    }
    req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
    const t0 = nowSeconds(init);
    req(pyrit.pyr_world_edit(@ptrCast(app.ctx), @ptrCast(app.world), &edits, @intCast(n)));
    var rounds: u32 = 0;
    while (rounds < 1000) : (rounds += 1) {
        app.updateWorld(sp, cam, tgs);
        var ws: api.WorldStats = undefined;
        req(pyrit.pyr_world_stats(@ptrCast(app.world), &ws));
        if (ws.pending_chunks == 0 and rounds > 0 and ws.built_chunks == 0) break;
        req(pyrit.pyr_world_wait(@ptrCast(app.ctx), @ptrCast(app.world), &sp.pos));
    }
    const t1 = nowSeconds(init);
    // Diagnose PYRIT_DIG_SETTLE: danach noch Runden, damit auch vorgemerkte
    // Neubauten fertig sind (Vergleich mit PYRIT_NO_PATH_EDIT)
    if (std.c.getenv("PYRIT_DIG_SETTLE") != null) {
        var k: u32 = 0;
        while (k < 20) : (k += 1) {
            app.updateWorld(sp, cam, tgs);
            req(pyrit.pyr_world_wait(@ptrCast(app.ctx), @ptrCast(app.world), &sp.pos));
        }
    }
    const org = sp.origin();
    const nr = 64 * 64;
    var grid: [nr]types.Ray = undefined;
    for (&grid, 0..) |*r, gi| {
        const gx = cx + @as(f64, @floatFromInt(gi % 64)) - 32;
        const gz = cz + @as(f64, @floatFromInt(gi / 64)) - 32;
        r.* = .{ .origin = .{ @floatCast(@floor(gx) + 0.5 - org[0]), 400, @floatCast(@floor(gz) + 0.5 - org[2]) }, .tmin = 0, .direction = .{ 0, -1, 0 }, .tmax = 1000 };
    }
    const grd = devAlloc(@sizeOf(@TypeOf(grid)));
    const ghd = devAlloc(nr * @sizeOf(types.Hit));
    defer _ = drv.cuMemFree_v2(grd);
    defer _ = drv.cuMemFree_v2(ghd);
    cu(drv.cuMemcpyHtoD_v2(grd, &grid, @sizeOf(@TypeOf(grid))));
    req(pyrit.pyr_trace(@ptrCast(app.ctx), grd, ghd, nr, 0x1, 0));
    req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
    var gh: [nr]types.Hit = undefined;
    cu(drv.cuMemcpyDtoH_v2(&gh, ghd, @sizeOf(@TypeOf(gh))));
    var sum: u64 = 0;
    var deep: u32 = 0;
    for (gh, 0..) |h, gi| {
        sum +%= @as(u64, @as(u32, @bitCast(h.t))) *% 31 +% h.attribute;
        const gx = cx + @as(f64, @floatFromInt(gi % 64)) - 32;
        const gz = cz + @as(f64, @floatFromInt(gi / 64)) - 32;
        if (400 - h.t < @floor(app.gen.height(gx, gz)) - 1.5) deep += 1;
    }
    if (std.c.getenv("PYRIT_DIG_DUMP")) |path| {
        if (std.c.fopen(path, "wb")) |f| {
            _ = std.c.fwrite(@ptrCast(&gh), 1, @sizeOf(@TypeOf(gh)), f);
            _ = std.c.fclose(f);
        }
    }
    std.debug.print("Loch: {d} Blöcke entfernt in {d:.1} ms, {d} Säulen tiefer als der Boden, Prüfsumme {x}\n", .{ n, (t1 - t0) * 1000, deep, sum });
}

/// Messung: n Blöcke setzen und die Zeit, bis alle betroffenen Chunks neu
/// gebaut sind und ein Frame sie zeigt. `spread`: über 60 Blöcke um die
/// Kamera verstreut, sonst als Haufen direkt vor der Kamera.
fn editBench(init: std.process.Init, app: *App, sp: *Spectator, cam: *types.Camera, tgs: *Targets, post: *api.PostInfo, n: u32, spread: bool) void {
    var edits: [4096]api.WorldEdit = undefined;
    const count = @min(n, edits.len);
    var prng = std.Random.DefaultPrng.init(if (spread) 7 else 3);
    const rnd = prng.random();
    const d = sp.dir();
    const cx = sp.pos[0] + d[0] * 80;
    const cz = sp.pos[2] + d[2] * 80;
    for (edits[0..count], 0..) |*e, i| {
        var x: f64 = undefined;
        var z: f64 = undefined;
        if (spread) {
            x = sp.pos[0] + (rnd.float(f64) * 2 - 1) * 60;
            z = sp.pos[2] + (rnd.float(f64) * 2 - 1) * 60;
        } else {
            x = cx + @as(f64, @floatFromInt(i % 5)) - 2;
            z = cz + @as(f64, @floatFromInt((i / 5) % 5)) - 2;
        }
        const ground = app.gen.height(x, z);
        const y = @floor(ground) + 1 + (if (spread) 0 else @as(f64, @floatFromInt(i / 25)));
        e.* = .{ .x = @intFromFloat(@floor(x)), .y = @intFromFloat(y), .z = @intFromFloat(@floor(z)), .attribute = scene.terrain.Block.stone.attribute() };
    }
    // Blick von oben auf die Stelle (Beweisbild vorher/nachher)
    var top = sp.*;
    top.pos = .{ cx, app.gen.height(cx, cz) + 14, cz - 6 };
    top.pitch = -1.1;
    top.yaw = 0;
    const shot = std.c.getenv("PYRIT_EDIT_SHOT");
    if (shot != null and !spread) editShot(init, app, &top, cam, tgs, post, "vorher");
    req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
    const t0 = nowSeconds(init);
    req(pyrit.pyr_world_edit(@ptrCast(app.ctx), @ptrCast(app.world), &edits, count));
    const t1 = nowSeconds(init);
    var built: u64 = 0;
    var rounds: u32 = 0;
    while (rounds < 1000) : (rounds += 1) {
        app.updateWorld(sp, cam, tgs);
        var ws: api.WorldStats = undefined;
        req(pyrit.pyr_world_stats(@ptrCast(app.world), &ws));
        built += ws.built_chunks;
        if (ws.pending_chunks == 0 and rounds > 0 and ws.built_chunks == 0) break;
        req(pyrit.pyr_world_wait(@ptrCast(app.ctx), @ptrCast(app.world), &sp.pos));
        // pyr_world_wait übernimmt den Auftrag selbst: seine Chunks zählen hier
        req(pyrit.pyr_world_stats(@ptrCast(app.world), &ws));
        built += ws.built_chunks;
    }
    const t2 = nowSeconds(init);
    app.frame(sp, cam, t2, tgs, post, false);
    req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
    const t3 = nowSeconds(init);
    if (shot != null and !spread) editShot(init, app, &top, cam, tgs, post, "nachher");
    // Diagnose PYRIT_EDIT_COLUMNS: Höhe jeder Säule des Haufens per senkrechtem Strahl
    if (!spread and std.c.getenv("PYRIT_EDIT_COLUMNS") != null) {
        var rays: [25]types.Ray = undefined;
        const org = sp.origin();
        for (&rays, 0..) |*r, i| {
            const x = cx + @as(f64, @floatFromInt(i % 5)) - 2;
            const z = cz + @as(f64, @floatFromInt(i / 5)) - 2;
            r.* = .{ .origin = .{ @floatCast(@floor(x) + 0.5 - org[0]), 400, @floatCast(@floor(z) + 0.5 - org[2]) }, .tmin = 0, .direction = .{ 0, -1, 0 }, .tmax = 1000 };
        }
        const rd = devAlloc(@sizeOf(@TypeOf(rays)));
        const hd = devAlloc(25 * @sizeOf(types.Hit));
        defer _ = drv.cuMemFree_v2(rd);
        defer _ = drv.cuMemFree_v2(hd);
        cu(drv.cuMemcpyHtoD_v2(rd, &rays, @sizeOf(@TypeOf(rays))));
        req(pyrit.pyr_trace(@ptrCast(app.ctx), rd, hd, 25, 0x1, 0));
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
        var hits: [25]types.Hit = undefined;
        cu(drv.cuMemcpyDtoH_v2(&hits, hd, @sizeOf(@TypeOf(hits))));
        // Raster 64x64 um den Haufen: Treffer und Attribut als Prüfsumme
        {
            const nr = 64 * 64;
            var grid: [nr]types.Ray = undefined;
            for (&grid, 0..) |*r, gi| {
                const gx = cx + @as(f64, @floatFromInt(gi % 64)) - 32;
                const gz = cz + @as(f64, @floatFromInt(gi / 64)) - 32;
                r.* = .{ .origin = .{ @floatCast(@floor(gx) + 0.5 - org[0]), 400, @floatCast(@floor(gz) + 0.5 - org[2]) }, .tmin = 0, .direction = .{ 0, -1, 0 }, .tmax = 1000 };
            }
            const grd = devAlloc(@sizeOf(@TypeOf(grid)));
            const ghd = devAlloc(nr * @sizeOf(types.Hit));
            defer _ = drv.cuMemFree_v2(grd);
            defer _ = drv.cuMemFree_v2(ghd);
            cu(drv.cuMemcpyHtoD_v2(grd, &grid, @sizeOf(@TypeOf(grid))));
            req(pyrit.pyr_trace(@ptrCast(app.ctx), grd, ghd, nr, 0x1, 0));
            req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
            var gh: [nr]types.Hit = undefined;
            cu(drv.cuMemcpyDtoH_v2(&gh, ghd, @sizeOf(@TypeOf(gh))));
            var n_hit: u32 = 0;
            var sum: u64 = 0;
            for (gh) |h| {
                if (h.instance != types.no_hit) n_hit += 1;
                sum +%= @as(u64, @as(u32, @bitCast(h.t))) *% 31 +% h.attribute;
            }
            std.debug.print("Raster: {d} Treffer von {d}, Prüfsumme {x}\n", .{ n_hit, nr, sum });
            if (std.c.getenv("PYRIT_EDIT_GRID_DUMP")) |path| {
                if (std.c.fopen(path, "wb")) |f| {
                    _ = std.c.fwrite(@ptrCast(&gh), 1, @sizeOf(@TypeOf(gh)), f);
                    _ = std.c.fclose(f);
                }
            }
        }
        std.debug.print("Säulen (Oberkante, Boden):", .{});
        for (hits, 0..) |h, i| {
            const x = cx + @as(f64, @floatFromInt(i % 5)) - 2;
            const z = cz + @as(f64, @floatFromInt(i / 5)) - 2;
            std.debug.print(" {d:.0}/{d:.0}", .{ 400 - h.t, @floor(app.gen.height(x, z)) });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("Änderung ({s}): {d} Blöcke, Aufruf {d:.2} ms, Neubau {d:.1} ms ({d} Chunks in {d} Runden), sichtbar nach {d:.1} ms\n", .{
        if (spread) "verstreut" else "Haufen", count, (t1 - t0) * 1000, (t2 - t1) * 1000, built, rounds, (t3 - t0) * 1000,
    });
}

/// Einige Frames aus `view` rendern (Welt nachführen, Verlauf einschwingen)
/// und das letzte als <PYRIT_EDIT_SHOT>_<name>.ppm speichern
fn editShot(init: std.process.Init, app: *App, view: *const Spectator, cam: *types.Camera, tgs: *Targets, post: *api.PostInfo, name: []const u8) void {
    // Beweisbilder auch mit --no-images
    const saved = no_images;
    no_images = false;
    defer no_images = saved;
    var k: u32 = 0;
    while (k < 30) : (k += 1) {
        view.camera(cam, tgs.w, tgs.h, k);
        app.updateWorld(view, cam, tgs);
        req(pyrit.pyr_world_wait(@ptrCast(app.ctx), @ptrCast(app.world), &view.pos));
        app.frame(view, cam, nowSeconds(init), tgs, post, false);
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
    }
    const bytes = @as(usize, tgs.out_w) * tgs.out_h * 4;
    const px = init.gpa.alloc(u8, bytes) catch return;
    defer init.gpa.free(px);
    cu(drv.cuMemcpyDtoH_v2(px.ptr, tgs.ldr, bytes));
    var buf: [512]u8 = undefined;
    const f = std.fmt.bufPrint(&buf, "{s}_{s}.ppm", .{ std.mem.span(std.c.getenv("PYRIT_EDIT_SHOT").?), name }) catch return;
    writePpm(init, f, px, tgs.out_w, tgs.out_h) catch {};
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const o = try parseArgs(args);
    if (o.record) |dir| return record(init, o, dir);

    try openWindow("Pyrit", o.out_w, o.out_h);
    defer W.c.display_disconnect(W.display);

    if (o.check) {
        const n = if (o.max_frames > 0) o.max_frames else 60;
        var k: u32 = 0;
        while (k < n and W.running) : (k += 1) {
            _ = W.c.display_dispatch_pending(W.display);
            try resizeBuffers(@max(W.width, 16), @max(W.height, 16));
            const idx = try acquireBuffer();
            const bytes: usize = @as(usize, W.buf_w) * W.buf_h * 4;
            const px = W.mem[idx * bytes ..][0..bytes];
            for (0..W.buf_h) |y| for (0..W.buf_w) |x| {
                const off = (y * W.buf_w + x) * 4;
                px[off + 0] = @intCast((x + k) & 0xff); // B
                px[off + 1] = @intCast(y & 0xff); // G
                px[off + 2] = @intCast((x ^ y) & 0xff); // R
                px[off + 3] = 255;
            };
            present(idx, o.vsync);
            if (o.vsync) while (!W.frame_done and W.running) {
                if (W.c.display_dispatch(W.display) < 0) break;
            };
        }
        std.debug.print("Wayland in Ordnung: {d} Bilder, Fenster {d}x{d}, Zeiger {s}, Tastatur {s}\n", .{
            k,                                             W.buf_w,
            W.buf_h,                                       if (W.pointer != null) "ja" else "nein",
            if (W.keyboard != null) "ja" else "nein",
        });
        return;
    }

    var app: App = undefined;
    try app.init(init, o);
    defer app.deinit();
    const light = &app.light;
    const fog_default = light.fog_density;
    var water = app.water;
    var fg = o.fg;
    var voxel_px = o.voxel_px;
    var mode = o.mode;
    var taa_off = false;
    var lod_frozen = false;
    var waves_off = false;

    var tgs = Targets{};
    defer tgs.free();

    var sp = app.spectatorAt(o);
    var cam = std.mem.zeroes(types.Camera);
    var post = scene.post();
    post.flags = api.post_bgra;
    var fxi = scene.fx();
    post.fx = &fxi;
    var fgi = std.mem.zeroes(api.FrameGenInfo);
    fgi.flags = api.post_bgra;

    var frame: u32 = 0;
    var t_prev = nowSeconds(init);
    var title_time = t_prev;
    var frame_ms: f64 = 0;
    var win_w = W.width;
    var win_h = W.height;
    var last_shot: []u8 = &.{};
    defer if (last_shot.len != 0) gpa.free(last_shot);
    // Zwischenspeicher (angeheftet) für zwei Frames im Wechsel: in den einen
    // kopiert die GPU, aus dem anderen zeigt die CPU
    var stage = [2]Stage{ .{}, .{} };
    defer for (&stage) |*st| st.free();
    var stage_cur: u1 = 0;
    var shown: u64 = 0;
    var rendered: u64 = 0;
    var rendered_fps: f64 = 0;
    // F3-Bildschirm: Zeilen, alle 0,5 s neu
    var dbg_buf: [10][96]u8 = undefined;
    var dbg_lines: [10][]const u8 = undefined;
    var dbg_n: usize = 0;
    var cycle_s: f64 = 0;
    var slept: f64 = 0;
    var cycle_at = t_prev;
    var shown_at = t_prev;
    var shown_fps: f64 = 0;

    while (W.running) {
        // Ereignisse abholen (nicht blockierend; bei vsync wartet present())
        _ = W.c.display_dispatch_pending(W.display);
        if (W.esc) {
            W.running = false;
            break;
        }
        if (W.toggle_fg) {
            fg = fg.next();
            W.toggle_fg = false;
        }
        if (W.toggle[0]) {
            fxi.flags ^= api.postfx_bloom;
            W.toggle[0] = false;
        }
        if (W.toggle[1]) {
            fxi.flags ^= api.postfx_dof;
            W.toggle[1] = false;
        }
        if (W.toggle[2]) {
            fxi.flags ^= api.postfx_motion_blur;
            W.toggle[2] = false;
        }
        if (W.toggle[3]) {
            light.fog_density = if (light.fog_density > 0) 0 else fog_default;
            req(pyrit.pyr_set_lighting(@ptrCast(app.ctx), light));
            W.toggle[3] = false;
        }
        if (W.toggle[4]) {
            light.gi_bounces = if (light.gi_bounces > 1) 1 else 3;
            req(pyrit.pyr_set_lighting(@ptrCast(app.ctx), light));
            W.toggle[4] = false;
        }
        if (W.toggle[5]) {
            // Länge der zeitlichen Mittelung: 20, 50, 100 Frames
            post.temporal_alpha = switch (@as(u32, @intFromFloat(@round(1 / @max(post.temporal_alpha, 0.05))))) {
                0...20 => 0.02,
                21...50 => 0.01,
                else => 0.05,
            };
            W.toggle[5] = false;
        }
        if (W.toggle[6]) {
            post.denoise_iterations = if (post.denoise_iterations >= 6) 2 else post.denoise_iterations + 2;
            W.toggle[6] = false;
        }
        // --- Diagnose bei Bewegung: die Verdächtigen einzeln abschalten ---
        if (W.toggle[7]) {
            // Ohne Historie gibt es keine Reprojektion: bleibt das Wandern,
            // liegt es nicht am TAA.
            taa_off = !taa_off;
            post.temporal_alpha = if (taa_off) 1.0 else 0.05;
            W.toggle[7] = false;
        }
        if (W.toggle[8]) {
            // Keine Weltaktualisierung mehr: die LOD-Stufen stehen fest.
            lod_frozen = !lod_frozen;
            W.toggle[8] = false;
        }
        if (W.toggle[9]) {
            waves_off = !waves_off;
            water.wave_height = if (waves_off) 0 else 0.5;
            req(pyrit.pyr_material_set(@ptrCast(app.ctx), scene.terrain.mat_water, &water));
            W.toggle[9] = false;
        }
        if (W.zoom_in) {
            voxel_px = @max((if (voxel_px == 0) 4 else voxel_px) * 0.8, 1);
            app.wi.voxel_pixels = voxel_px;
            W.zoom_in = false;
        }
        if (W.zoom_out) {
            voxel_px = @min((if (voxel_px == 0) 4 else voxel_px) * 1.25, 32);
            app.wi.voxel_pixels = voxel_px;
            W.zoom_out = false;
        }
        sp.yaw -= W.dyaw * 0.004;
        sp.pitch = std.math.clamp(sp.pitch - W.dpitch * 0.004, -1.5, 1.5);
        W.dyaw = 0;
        W.dpitch = 0;

        // Zeit und Bewegung im Zuschauermodus: W/S in Blickrichtung, A/D
        // seitlich, Leertaste hoch, Umschalt runter, Strg schneller
        const now = nowSeconds(init);
        const dt: f32 = @floatCast(@min(now - t_prev, 0.1));
        t_prev = now;
        const dir = sp.dir();
        const right = sp.right();
        var speed: f64 = fly_speed;
        if (W.keys[wl.key_leftctrl]) speed *= sprint_factor;
        speed *= dt;
        for (0..3) |k| {
            if (W.keys[wl.key_w]) sp.pos[k] += dir[k] * speed;
            if (W.keys[wl.key_s]) sp.pos[k] -= dir[k] * speed;
            if (W.keys[wl.key_d]) sp.pos[k] += right[k] * speed;
            if (W.keys[wl.key_a]) sp.pos[k] -= right[k] * speed;
        }
        if (W.keys[wl.key_space]) sp.pos[1] += speed;
        if (W.keys[wl.key_leftshift]) sp.pos[1] -= speed;
        sp.pos[1] = std.math.clamp(sp.pos[1], scene.terrain.y_min, scene.terrain.y_max + 256);
        if (W.keys[wl.key_q]) light.sun_direction[0] += dt * 0.5;
        if (W.keys[wl.key_e]) light.sun_direction[0] -= dt * 0.5;
        if (W.keys[wl.key_q] or W.keys[wl.key_e]) req(pyrit.pyr_set_lighting(@ptrCast(app.ctx), light));

        // Puffer an die Fenstergröße anpassen
        win_w = @max(W.width, 16);
        win_h = @max(W.height, 16);
        try resizeBuffers(win_w, win_h);
        const ss = @max(o.super, 1);
        const post_w = win_w * ss;
        const post_h = win_h * ss;
        if (W.mode_key) |mk| {
            mode = @enumFromInt(mk);
            W.mode_key = null;
        }
        post.upscaler = mode.upscaler();
        tgs.setMode(mode);
        const div = if (o.scale > 0) o.scale else mode.scale();
        const rw = @max(post_w / div, 16);
        const rh = @max(post_h / div, 16);
        tgs.resize(rw, rh, post_w, post_h);
        const frame_bytes: usize = @as(usize, win_w) * win_h * 4;
        fxi.supersample = ss;

        // Kamera zuerst: die Welt wählt danach ihr LOD
        sp.camera(&cam, rw, rh, if (mode.jitter()) frame else null);
        // Vor dem ersten Bild die Welt um die Kamera laden (höchstens 3 s):
        // sonst sah man zuerst nur Dunst mit nahen Bäumen, und die Ferne kam
        // nach und nach dazu – das wirkte wie Herauszoomen
        if (frame == 0) {
            const t_load = nowSeconds(init);
            while (nowSeconds(init) - t_load < 3) {
                app.updateWorld(&sp, &cam, &tgs);
                req(pyrit.pyr_world_wait(@ptrCast(app.ctx), @ptrCast(app.world), &sp.pos));
                var ws: api.WorldStats = undefined;
                req(pyrit.pyr_world_stats(@ptrCast(app.world), &ws));
                if (ws.pending_chunks == 0) break;
            }
        }
        // Die Welt wird hinter dem Einreihen des vorigen Frames nachgeführt
        // (unten), während die GPU rechnet
        app.frame(&sp, &cam, now, &tgs, &post, false);

        // Zwischenbilder (brauchen Bewegung und Tiefe in Ausgabeauflösung, die
        // liefern nur TAAU und DLSS) gleich hinter dem Frame einreihen, dann
        // alles asynchron in den Zwischenspeicher kopieren
        const cur = &stage[stage_cur];
        try cur.ensure(frame_bytes);
        cur.gens = 0;
        cur.w = win_w;
        cur.h = win_h;
        var k: u32 = 1;
        while (fg != .off and mode != .native and frame > 2 and k <= fg.count()) : (k += 1) {
            fgi.output_ldr = tgs.ldr_fg[k - 1];
            fgi.flags = api.post_bgra | fg.flags();
            fgi.t = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(fg.count() + 1));
            const r = pyrit.pyr_frame_generate(@ptrCast(app.ctx), app.view, &fgi);
            if (r != api.ok and fg.isDlss()) {
                // ohne DLSS-SDK, Treiber, Vulkan oder Multi Frame Generation:
                // weiter ohne Zwischenbild
                std.debug.print("DLSS Frame Generation: {s}\n", .{pyrit.pyr_error_message()});
                fg = .off;
                break;
            }
            req(r);
            cu(drv.cuMemcpyDtoHAsync_v2(cur.host[k - 1].ptr, tgs.ldr_fg[k - 1], frame_bytes, app.stream));
            cur.gens = k;
        }
        cu(drv.cuMemcpyDtoHAsync_v2(cur.host[max_gens].ptr, tgs.ldr, frame_bytes, app.stream));
        cur.valid = true;

        // Während die GPU diesen Frame rechnet, zeigt die CPU den vorigen:
        // erst seine Zwischenbilder, dann ihn selbst. Die Bilder werden
        // gleichmäßig über die gemessene Dauer eines Frames verteilt (eigenes
        // Frame-Pacing) – nur im Bildtakt des Compositors kamen sie sonst
        // schnell hintereinander, und dann klaffte bis zum nächsten Frame
        // eine Lücke: das Ruckeln bei 4x und 6x.
        var idx: usize = 0;
        const prev_stage = &stage[stage_cur ^ 1];
        if (prev_stage.valid and prev_stage.w == win_w and prev_stage.h == win_h) {
            const n_show = prev_stage.gens + 1;
            // höchstens 0,1 s je Frame verteilen, auch wenn die Messung (noch) Unsinn sagt
            const step = @min(cycle_s, 0.1) / @as(f64, @floatFromInt(n_show));
            const t_first = nowSeconds(init);
            var q: u32 = 0;
            while (q < n_show and W.running) : (q += 1) {
                // Takt: frühestens zum geplanten Zeitpunkt und nie schneller,
                // als der Compositor Bilder abnimmt
                const target = t_first + step * @as(f64, @floatFromInt(q));
                const t_sleep = nowSeconds(init);
                sleepUntil(init, target);
                slept += nowSeconds(init) - t_sleep;
                if (o.vsync) while (!W.frame_done and W.running) {
                    if (W.c.display_dispatch(W.display) < 0) break;
                };
                const src = if (q < prev_stage.gens) prev_stage.host[q] else prev_stage.host[max_gens];
                idx = try acquireBuffer();
                @memcpy(W.mem[idx * frame_bytes ..][0..frame_bytes], src[0..frame_bytes]);
                if (W.debug) overlay.draw(W.mem[idx * frame_bytes ..][0..frame_bytes], win_w, win_h, dbg_lines[0..dbg_n], if (win_h >= 1000) 3 else 2);
                present(idx, o.vsync);
                shown += 1;
            }
        }
        req(pyrit.pyr_synchronize(@ptrCast(app.ctx)));
        cu(drv.cuStreamSynchronize(app.stream));
        // Welt erst nachführen, wenn die GPU mit dem Frame fertig ist: die
        // Übernahme schreibt neue Chunks in Pool-Bereiche, die verdrängte
        // Chunks frei gemacht haben – lief sie parallel zum Rendern, las der
        // laufende Frame halb überschriebene Geometrie (verformte Formen)
        if (!lod_frozen) app.updateWorld(&sp, &cam, &tgs);
        stage_cur ^= 1;
        // Arbeitszeit eines Durchlaufs, geglättet: danach richtet sich der
        // Takt. Die eigenen Pausen zählen nicht mit – sonst schaukelte sich
        // der Takt auf (der erste Durchlauf enthält das Laden der Welt, und
        // jede Pause verlängerte den nächsten Durchlauf um genau sich selbst).
        const t_cycle = nowSeconds(init);
        const work = @max(t_cycle - cycle_at - slept, 0);
        if (frame > 0) cycle_s = if (cycle_s == 0) work else cycle_s * 0.9 + work * 0.1;
        cycle_at = t_cycle;
        slept = 0;

        frame += 1;
        if (o.max_frames > 0 and frame >= o.max_frames) {
            if (o.shot != null) {
                if (last_shot.len != frame_bytes) {
                    if (last_shot.len != 0) gpa.free(last_shot);
                    last_shot = try gpa.alloc(u8, frame_bytes);
                }
                @memcpy(last_shot, W.mem[idx * frame_bytes ..][0..frame_bytes]);
            }
            W.running = false;
        }
        frame_ms = frame_ms * 0.9 + (nowSeconds(init) - now) * 1000 * 0.1;
        rendered += 1;
        if (now - title_time > 0.5) {
            shown_fps = @as(f64, @floatFromInt(shown)) / @max(now - shown_at, 1e-3);
            rendered_fps = @as(f64, @floatFromInt(rendered)) / @max(now - shown_at, 1e-3);
            shown = 0;
            rendered = 0;
            shown_at = now;
            {
                var ws: api.WorldStats = undefined;
                req(pyrit.pyr_world_stats(@ptrCast(app.world), &ws));
                const L = struct {
                    fn line(buf: *[96]u8, comptime fmt: []const u8, a: anytype) []const u8 {
                        return std.fmt.bufPrint(buf, fmt, a) catch buf[0..0];
                    }
                };
                dbg_n = 0;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "PYRIT DEMO (F3)", .{});
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "FPS {d:.0} ANGEZEIGT, {d:.0} GERENDERT", .{ shown_fps, rendered_fps });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "FRAME {d:.1} MS ARBEIT, {d:.1} MS DURCHLAUF", .{ cycle_s * 1000, frame_ms });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "MODUS {s}  FG {s}", .{ mode.name(), if (fg == .off) " AUS" else fg.label() });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "RENDER {d}X{d} -> {d}X{d}", .{ tgs.w, tgs.h, tgs.out_w, tgs.out_h });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "POS {d:.1} {d:.1} {d:.1}", .{ sp.pos[0], sp.pos[1], sp.pos[2] });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "CHUNKS {d} SICHTBAR {d} OFFEN {d}", .{ ws.resident_chunks, ws.visible_chunks, ws.pending_chunks });
                dbg_n += 1;
                dbg_lines[dbg_n] = L.line(&dbg_buf[dbg_n], "SPEICHER {d:.0} MIB  LOD {d:.1} PX JE VOXEL", .{ @as(f64, @floatFromInt(ws.bytes)) / (1 << 20), ws.voxel_pixels });
                dbg_n += 1;
            }
            title_time = now;
            var st: api.WorldStats = undefined;
            req(pyrit.pyr_world_stats(@ptrCast(app.world), &st));
            var buf: [256]u8 = undefined;
            const title = try std.fmt.bufPrintZ(&buf, "Pyrit-Demo – {s} – {d:.1} ms je Frame, {d:.0} Bilder/s{s} · {d}x{d} · {d} Chunks, {d:.0} MiB · x {d:.0} y {d:.0} z {d:.0}", .{
                mode.name(),        frame_ms,
                shown_fps,
                if (fg != .off and mode == .native) " (FG braucht TAAU/DLSS)" else fg.label(), rw,
                rh,                 st.resident_chunks,
                @as(f64, @floatFromInt(st.bytes)) / (1 << 20),
                sp.pos[0],          sp.pos[1],
                sp.pos[2],
            });
            _ = W.marshal(W.toplevel.?, wl.toplevel_set_title, null, .{title.ptr});
        }

        // Den Bildtakt des Compositors wartet das Zeigen oben ab – dann
        // rechnet die GPU schon am nächsten Frame
        if (W.c.display_dispatch_pending(W.display) < 0) W.running = false;
    }

    std.debug.print("Demo: {s}{s}, zuletzt {d:.1} ms je gerendertem Frame, {d:.0} angezeigte Bilder/s\n", .{ mode.name(), fg.label(), frame_ms, shown_fps });
    // Für Prüfläufe: das zuletzt gezeigte Bild speichern (BGRA -> RGB)
    if (o.shot) |path| {
        const src = if (last_shot.len != 0) last_shot else W.mem[0 .. @as(usize, win_w) * win_h * 4];
        try writePpm(init, path, src, win_w, win_h);
        std.debug.print("Bild: {s} ({d} Frames, zuletzt {d:.1} ms)\n", .{ path, frame, frame_ms });
    }
}
