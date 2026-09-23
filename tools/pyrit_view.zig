//! pyrit-view: interaktiver Betrachter für die gestreamte Welt.
//!
//!   zig build view -- [--size 1280x720] [--scale 2] [--fg] [--no-half-gi]
//!                     [--voxel-px 4] [--sea 0] [--no-rt] [--no-vsync]
//!
//! Steuerung: W/A/S/D bewegen, Leertaste hoch, Linke Umschalttaste schneller,
//! Maus ziehen dreht, Q/E rollen die Sonne, +/- ändern die Zielgröße der Voxel,
//! F schaltet Zwischenbilder, Esc beendet.
//!
//! Das Fenster ist reines Wayland (libwayland-client dynamisch geladen, keine
//! Grafik-API): Pyrit rendert in einen CUDA-Puffer, der Betrachter kopiert ihn
//! direkt in den Shared-Memory-Puffer des Compositors (XRGB8888 = PYR_POST_BGRA).

const std = @import("std");
const pyrit = @import("pyrit");
const pyr = @import("pyrit_device");
const wl = @import("wayland.zig");
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
    /// Umschalter für die Bildeffekte (Tasten B, T, U, N, G)
    toggle: [7]bool = .{false} ** 7,

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
        wl.key_b => W.toggle[0] = true,
        wl.key_t => W.toggle[1] = true,
        wl.key_u => W.toggle[2] = true,
        wl.key_n => W.toggle[3] = true,
        wl.key_g => W.toggle[4] = true,
        wl.key_k => W.toggle[5] = true,
        wl.key_j => W.toggle[6] = true,
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
    ldr_fg: u64 = 0,

    fn free(self: *Targets) void {
        for ([_]u64{ self.tg.hits, self.tg.motion, self.tg.color, self.tg.normal, self.tg.albedo, self.ldr, self.ldr_fg }) |b| {
            if (b != 0) _ = drv.cuMemFree_v2(b);
        }
        self.* = .{};
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
        self.tg.ray_mask = 0x1;
        self.tg.coverage = 2;
        self.ldr = devAlloc(n_out * 4);
        self.ldr_fg = devAlloc(n_out * 4);
        self.w = w;
        self.h = h;
        self.out_w = out_w;
        self.out_h = out_h;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out_w: u32 = 1280;
    var out_h: u32 = 720;
    var scale: u32 = 1;
    var flags: u32 = 0;
    var voxel_px: f32 = 0;
    // Meeresspiegel der Demo: hoch genug, dass Buchten und Küste in Sicht sind
    var sea: f32 = 300;
    var fg = false;
    var half_gi = true;
    var vsync = true;
    // Supersampling: gerendert wird in super-facher Fenstergröße
    var super: u32 = 1;
    // Selbstprüfung des Fensterpfads ohne GPU (Testbild statt Rendern)
    var check = false;
    var max_frames: u32 = 0;
    var shot: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--size") and i + 1 < args.len) {
            i += 1;
            var it = std.mem.splitScalar(u8, args[i], 'x');
            out_w = try std.fmt.parseInt(u32, it.next() orelse "1280", 10);
            out_h = try std.fmt.parseInt(u32, it.next() orelse "720", 10);
        } else if (std.mem.eql(u8, a, "--scale") and i + 1 < args.len) {
            i += 1;
            scale = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--voxel-px") and i + 1 < args.len) {
            i += 1;
            voxel_px = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--sea") and i + 1 < args.len) {
            i += 1;
            sea = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--frames") and i + 1 < args.len) {
            i += 1;
            max_frames = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--shot") and i + 1 < args.len) {
            i += 1;
            shot = args[i];
        } else if (std.mem.eql(u8, a, "--fg")) {
            fg = true;
        } else if (std.mem.eql(u8, a, "--check")) {
            check = true;
        } else if (std.mem.eql(u8, a, "--super") and i + 1 < args.len) {
            i += 1;
            super = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-vsync")) {
            vsync = false;
        } else if (std.mem.eql(u8, a, "--no-half-gi")) {
            half_gi = false;
        } else if (std.mem.eql(u8, a, "--no-rt")) {
            flags |= api.create_no_rt;
        } else {
            std.debug.print("unbekanntes Argument: {s}\n", .{a});
            return error.Usage;
        }
    }

    try openWindow("Pyrit", out_w, out_h);
    defer W.c.display_disconnect(W.display);

    if (check) {
        const n = if (max_frames > 0) max_frames else 60;
        var k: u32 = 0;
        while (k < n and W.running) : (k += 1) {
            _ = W.c.display_dispatch_pending(W.display);
            try resizeBuffers(@max(W.width, 16), @max(W.height, 16));
            const idx = try acquireBuffer();
            const bytes: usize = @as(usize, W.buf_w) * W.buf_h * 4;
            const px = W.mem[idx * bytes ..][0..bytes];
            for (0..W.buf_h) |y| for (0..W.buf_w) |x| {
                const o = (y * W.buf_w + x) * 4;
                px[o + 0] = @intCast((x + k) & 0xff); // B
                px[o + 1] = @intCast(y & 0xff); // G
                px[o + 2] = @intCast((x ^ y) & 0xff); // R
                px[o + 3] = 255;
            };
            present(idx, vsync);
            if (vsync) while (!W.frame_done and W.running) {
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
    ci.max_geometries = 65536;
    ci.max_instances = 65536;
    ci.node_pool_bytes = 512 << 20;
    ci.leaf_pool_bytes = 512 << 20;
    ci.attribute_pool_bytes = 512 << 20;
    var ctx: ?*anyopaque = null;
    req(pyrit.pyr_create(&ci, @ptrCast(&ctx)));
    defer pyrit.pyr_destroy(@ptrCast(ctx));

    // Welt mit Wasser und Bäumen
    var terrain: api.TerrainInfo = undefined;
    pyrit.pyr_terrain_default(&terrain);
    if (sea > 0) terrain.sea_level = sea;
    terrain.attr_water = pyrit.pyr_voxel_attribute(2, 40, 90, 140);
    terrain.attr_leaves = pyrit.pyr_voxel_attribute(0, 48, 112, 40);
    terrain.attr_wood = pyrit.pyr_voxel_attribute(0, 96, 68, 44);

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
        terrain.attr_grass = pyrit.pyr_voxel_attribute(mat_grass, 84, 140, 58);
        terrain.attr_dirt = pyrit.pyr_voxel_attribute(mat_grass, 122, 92, 62);
        terrain.attr_rock = pyrit.pyr_voxel_attribute(mat_rock, 118, 112, 106);
        terrain.attr_sand = pyrit.pyr_voxel_attribute(mat_sand, 214, 196, 142);
        terrain.attr_snow = pyrit.pyr_voxel_attribute(mat_snow, 236, 240, 245);
        terrain.attr_wood = pyrit.pyr_voxel_attribute(mat_wood, 96, 68, 44);
        terrain.attr_leaves = pyrit.pyr_voxel_attribute(mat_leaves, 48, 112, 40);

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
    water.wave_height = 0.5;
    water.wave_length = 48;
    water.wave_speed = 0.18;
    water.clearcoat = 1; // nasse, lackartige Oberfläche
    water.clearcoat_roughness = 0.03;
    req(pyrit.pyr_material_set(@ptrCast(ctx), 2, &water));

    // Boden und Laub (Material 0): Textur, Detailnormale, Streuung
    {
        const tw: u32 = 64;
        const tex = try gpa.alloc(u8, tw * tw * 4);
        defer gpa.free(tex);
        for (0..tw) |ty| for (0..tw) |tx| {
            const nz = (tx *% 73 +% ty *% 151) ^ ((tx *% 19) >> 2);
            const val: u8 = @intCast(180 + (nz % 76));
            const o = (ty * tw + tx) * 4;
            tex[o + 0] = val;
            tex[o + 1] = @intCast(@min(@as(u32, val) + 12, 255));
            tex[o + 2] = @intCast(@as(u32, val) * 3 / 4);
            tex[o + 3] = 255;
        };
        var tex_index: u32 = 0;
        req(pyrit.pyr_texture_create(@ptrCast(ctx), tw, tw, tex.ptr, &tex_index));
        var ground: types.Material = undefined;
        pyrit.pyr_material_default(&ground);
        ground.flags = types.material_voxel_color;
        ground.texture = tex_index;
        ground.texture_scale = 8;
        ground.normal_strength = 0.35;
        ground.normal_scale = 2;
        ground.subsurface = 0.25;
        ground.subsurface_color = .{ 0.4, 0.8, 0.3 };
        req(pyrit.pyr_material_set(@ptrCast(ctx), 0, &ground));
    }

    var wi = std.mem.zeroes(api.WorldInfo);
    wi.terrain = &terrain;
    wi.voxel_pixels = voxel_px;
    var world: ?*anyopaque = null;
    req(pyrit.pyr_world_create(@ptrCast(ctx), &wi, @ptrCast(&world)));
    defer _ = pyrit.pyr_world_destroy(@ptrCast(ctx), @ptrCast(world));

    var light: types.Lighting = undefined;
    pyrit.pyr_lighting_default(&light);
    light.sun_direction = .{ 0.5, 0.45, 0.35 };
    if (half_gi) light.flags |= types.lighting_gi_half;
    light.gi_distance = 96;
    light.gi_bounces = 2;
    // Nebel: Höhenabnahme, nach vorn streuend (Lichtschächte zur Sonne)
    light.fog_density = 0.006;
    light.fog_height = 90;
    light.fog_falloff = 0.02;
    light.fog_color = .{ 1, 0.98, 0.92 };
    light.fog_anisotropy = 0.7;
    // Umgebungskarte: Himmelsverlauf mit Sonnenscheibe, nach Helligkeit abgetastet
    {
        const ew: u32 = 512;
        const eh: u32 = 256;
        const env = try gpa.alloc(f32, ew * eh * 4);
        defer gpa.free(env);
        // Scheibenradius und Strahldichte zusammen so gewaehlt, dass
        // L * Raumwinkel der Beleuchtungsstaerke einer echten Sonne
        // entspricht (2.6). Zu helle Scheiben ueberstrahlen nicht nur, sie
        // verstaerken auch das Rauschen im Halbschatten: mit einem
        // Schattenstrahl je Pixel und Frame waechst es proportional zur
        // Beleuchtungsstaerke (gemessen: Flimmern 0,87 bei 7,4 gegen 0,57
        // bei 2,6 Beleuchtungsstaerke).
        const sun_radius: f32 = 0.025;
        const sun_theta: f32 = 0.85;
        const sun_phi: f32 = 1.1;
        for (0..eh) |y| {
            const theta = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(eh)) * std.math.pi;
            for (0..ew) |x| {
                const phi = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, @floatFromInt(ew)) * 2 * std.math.pi;
                const up = @cos(theta);
                var r: f32 = if (up > 0) 0.30 + 0.20 * up else 0.08;
                var g: f32 = if (up > 0) 0.40 + 0.30 * up else 0.07;
                var b: f32 = if (up > 0) 0.62 + 0.32 * up else 0.06;
                const dt = theta - sun_theta;
                var dp = phi - sun_phi;
                if (dp > std.math.pi) dp -= 2 * std.math.pi;
                if (dp < -std.math.pi) dp += 2 * std.math.pi;
                const st = @sin(theta);
                if (dt * dt + dp * dp * st * st < sun_radius * sun_radius) {
                    const peak = 2.6 / (std.math.pi * sun_radius * sun_radius);
                    r += peak;
                    g += peak * 0.94;
                    b += peak * 0.84;
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
        // Richtung der Sonne in der Karte auch für Schatten und Nebel
        light.sun_direction = .{ @sin(sun_theta) * @cos(sun_phi), @cos(sun_theta), @sin(sun_theta) * @sin(sun_phi) };
        light.sun_color = .{ 0, 0, 0 }; // das Licht steckt jetzt in der Karte
    }
    req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));

    var view: api.Handle = null;
    req(pyrit.pyr_view_create(@ptrCast(ctx), &view));

    var tgs = Targets{};
    defer tgs.free();

    // Kamera
    var pos = [3]f64{ 100_000, 0, 100_000 };
    pos[1] = @as(f64, pyrit.pyr_terrain_height(&terrain, pos[0], pos[2])) + 40;
    var yaw: f32 = 0.6;
    var pitch: f32 = -0.25;

    var cam = std.mem.zeroes(types.Camera);
    var post = std.mem.zeroes(api.PostInfo);
    post.denoise_iterations = 4;
    post.exposure = 1.0;
    post.clamp_sigma = 1.5;
    post.temporal_alpha = 0.05;
    post.flags = api.post_bgra;
    // Kamera- und Bildeffekte
    var fxi = std.mem.zeroes(api.PostFx);
    // Ohne Tiefenschärfe und Bewegungsunschärfe: alles bleibt scharf, nah wie
    // fern. Beide lassen sich über PyrPostFx jederzeit zuschalten (Tasten T und U).
    fxi.flags = api.postfx_bloom | api.postfx_auto_exposure | api.postfx_grade;
    fxi.bloom_strength = 0.06;
    fxi.bloom_threshold = 1.2;
    fxi.dof_strength = 2.5;
    fxi.motion_blur_scale = 0.4;
    fxi.contrast = 1.06;
    fxi.saturation = 1.08;
    fxi.temperature = 4;
    fxi.exposure_compensation = 0.2;
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

    while (W.running) {
        // Ereignisse abholen (nicht blockierend; bei vsync wartet present())
        _ = W.c.display_dispatch_pending(W.display);
        if (W.esc) {
            W.running = false;
            break;
        }
        if (W.toggle_fg) {
            fg = !fg;
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
            light.fog_density = if (light.fog_density > 0) 0 else 0.006;
            req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));
            W.toggle[3] = false;
        }
        if (W.toggle[4]) {
            light.gi_bounces = if (light.gi_bounces > 1) 1 else 3;
            req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));
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
        if (W.zoom_in) {
            voxel_px = @max((if (voxel_px == 0) 4 else voxel_px) * 0.8, 1);
            wi.voxel_pixels = voxel_px;
            W.zoom_in = false;
        }
        if (W.zoom_out) {
            voxel_px = @min((if (voxel_px == 0) 4 else voxel_px) * 1.25, 32);
            wi.voxel_pixels = voxel_px;
            W.zoom_out = false;
        }
        yaw -= W.dyaw * 0.004;
        pitch = std.math.clamp(pitch - W.dpitch * 0.004, -1.5, 1.5);
        W.dyaw = 0;
        W.dpitch = 0;

        // Zeit und Bewegung
        const now = nowSeconds(init);
        const dt: f32 = @floatCast(@min(now - t_prev, 0.1));
        t_prev = now;
        const dir = [3]f64{ @cos(pitch) * @sin(yaw), @sin(pitch), @cos(pitch) * @cos(yaw) };
        const right = [3]f64{ @cos(yaw), 0, -@sin(yaw) };
        var speed: f64 = 40;
        if (W.keys[wl.key_leftshift]) speed *= 6;
        speed *= dt;
        if (W.keys[wl.key_w]) for (0..3) |k| {
            pos[k] += dir[k] * speed;
        };
        if (W.keys[wl.key_s]) for (0..3) |k| {
            pos[k] -= dir[k] * speed;
        };
        if (W.keys[wl.key_d]) for (0..3) |k| {
            pos[k] += right[k] * speed;
        };
        if (W.keys[wl.key_a]) for (0..3) |k| {
            pos[k] -= right[k] * speed;
        };
        if (W.keys[wl.key_space]) pos[1] += speed;
        if (W.keys[wl.key_q]) light.sun_direction[0] += dt * 0.5;
        if (W.keys[wl.key_e]) light.sun_direction[0] -= dt * 0.5;
        if (W.keys[wl.key_q] or W.keys[wl.key_e]) req(pyrit.pyr_set_lighting(@ptrCast(ctx), &light));

        // Puffer an die Fenstergröße anpassen
        win_w = @max(W.width, 16);
        win_h = @max(W.height, 16);
        try resizeBuffers(win_w, win_h);
        const ss = @max(super, 1);
        const post_w = win_w * ss;
        const post_h = win_h * ss;
        const rw = @max(post_w / scale, 16);
        const rh = @max(post_h / scale, 16);
        tgs.resize(rw, rh, post_w, post_h);
        const frame_bytes: usize = @as(usize, win_w) * win_h * 4;

        // Kamera setzen (die Welt wählt danach ihr LOD)
        const origin = [3]f64{ @floor(pos[0] / 1024) * 1024, 0, @floor(pos[2] / 1024) * 1024 };
        const eye = [3]f32{ @floatCast(pos[0] - origin[0]), @floatCast(pos[1] - origin[1]), @floatCast(pos[2] - origin[2]) };
        const target = [3]f32{
            eye[0] + @as(f32, @floatCast(dir[0])) * 100,
            eye[1] + @as(f32, @floatCast(dir[1])) * 100,
            eye[2] + @as(f32, @floatCast(dir[2])) * 100,
        };
        pyrit.pyr_camera_look_at(&cam, &eye, &target, &.{ 0, 1, 0 });
        pyrit.pyr_camera_perspective(&cam, 1.1, rw, rh, 0.1);
        // Kein Jitter: er war die Ursache der wandernden Schatten. An einer
        // Voxelkante entschied er jeden Frame neu, welche der beiden
        // verschieden beleuchteten Flächen das Pixel sieht – ein echtes
        // Wechselsignal, das kein Filter glätten kann. Die Kantenglättung
        // macht stattdessen `tg.coverage` innerhalb eines Frames.
        cam.jitter = .{ 0, 0 };

        req(pyrit.pyr_world_update(@ptrCast(ctx), @ptrCast(world), &pos, &origin, &cam));
        const fi = api.FrameInfo{ .time = now, .origin = origin };
        req(pyrit.pyr_commit(@ptrCast(ctx), &fi));
        req(pyrit.pyr_render(@ptrCast(ctx), view, &cam, &tgs.tg));
        post.output_ldr = tgs.ldr;
        post.output_width = post_w;
        post.output_height = post_h;
        fxi.supersample = ss;
        req(pyrit.pyr_postprocess(@ptrCast(ctx), view, &tgs.tg, &post));

        if (fg and frame > 2) {
            fgi.output_ldr = tgs.ldr_fg;
            req(pyrit.pyr_frame_generate(@ptrCast(ctx), view, &fgi));
            req(pyrit.pyr_synchronize(@ptrCast(ctx)));
            const idx = try acquireBuffer();
            cu(drv.cuMemcpyDtoH_v2(W.mem.ptr + idx * frame_bytes, tgs.ldr_fg, frame_bytes));
            present(idx, vsync);
            if (vsync) while (!W.frame_done and W.running) {
                if (W.c.display_dispatch(W.display) < 0) break;
            };
        }

        req(pyrit.pyr_synchronize(@ptrCast(ctx)));
        const idx = try acquireBuffer();
        cu(drv.cuMemcpyDtoH_v2(W.mem.ptr + idx * frame_bytes, tgs.ldr, frame_bytes));
        present(idx, vsync);

        frame += 1;
        if (max_frames > 0 and frame >= max_frames) {
            if (shot != null) {
                if (last_shot.len != frame_bytes) {
                    if (last_shot.len != 0) gpa.free(last_shot);
                    last_shot = try gpa.alloc(u8, frame_bytes);
                }
                @memcpy(last_shot, W.mem[idx * frame_bytes ..][0..frame_bytes]);
            }
            W.running = false;
        }
        frame_ms = frame_ms * 0.9 + (nowSeconds(init) - now) * 1000 * 0.1;
        if (now - title_time > 0.5) {
            title_time = now;
            var st: api.WorldStats = undefined;
            req(pyrit.pyr_world_stats(@ptrCast(world), &st));
            var buf: [256]u8 = undefined;
            const title = try std.fmt.bufPrintZ(&buf, "Pyrit – {d:.1} ms ({d:.0} fps){s} · {d}x{d} · {d} Chunks, {d:.0} MiB · Bloom {s}, Nebel {s}, GI {d}x, Mittelung {d} Frames, Filter {d}, Supersampling {d}x", .{
                frame_ms,                                                     1000 / @max(frame_ms, 0.001),
                if (fg) " +Zwischenbild" else "",                              rw,
                rh,                                                            st.resident_chunks,
                @as(f64, @floatFromInt(st.bytes)) / (1 << 20),
                if (fxi.flags & api.postfx_bloom != 0) "an" else "aus",
                if (light.fog_density > 0) "an" else "aus",
                @max(light.gi_bounces, 1),
                @as(u32, @intFromFloat(@round(1 / @max(post.temporal_alpha, 1e-3)))),
                post.denoise_iterations,
                ss,
            });
            _ = W.marshal(W.toplevel.?, wl.toplevel_set_title, null, .{title.ptr});
        }

        // Bildtakt: auf das nächste Frame-Ereignis des Compositors warten
        if (vsync) while (!W.frame_done and W.running) {
            if (W.c.display_dispatch(W.display) < 0) {
                W.running = false;
                break;
            }
        };
    }

    // Für Prüfläufe: das zuletzt gezeigte Bild speichern (BGRA -> RGB)
    if (shot) |path| {
        const src = if (last_shot.len != 0) last_shot else W.mem[0 .. @as(usize, win_w) * win_h * 4];
        var file: std.ArrayList(u8) = .empty;
        defer file.deinit(gpa);
        var head: [64]u8 = undefined;
        try file.appendSlice(gpa, try std.fmt.bufPrint(&head, "P6\n{d} {d}\n255\n", .{ win_w, win_h }));
        var k: usize = 0;
        while (k + 4 <= src.len) : (k += 4) {
            try file.appendSlice(gpa, &.{ src[k + 2], src[k + 1], src[k] });
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = file.items });
        std.debug.print("Bild: {s} ({d} Frames, zuletzt {d:.1} ms)\n", .{ path, frame, frame_ms });
    }
}
