//! Minimale Wayland-Anbindung für den Betrachter: libwayland-client wird
//! dynamisch geladen, die Kernschnittstellen liefert die Bibliothek als Daten.
//! xdg-shell ist nicht Teil von libwayland-client, seine Schnittstellentabellen
//! stehen daher hier (erzeugt aus xdg-shell.xml, Version 1).
//!
//! Es wird keine Grafik-API benutzt: das Bild landet in einem Shared-Memory-
//! Puffer, den der Compositor anzeigt.

const std = @import("std");

pub const Proxy = opaque {};
pub const Display = opaque {};

pub const Message = extern struct {
    name: [*:0]const u8,
    signature: [*:0]const u8,
    types: ?[*]const ?*const Interface,
};

pub const Interface = extern struct {
    name: [*:0]const u8,
    version: c_int,
    method_count: c_int,
    methods: ?[*]const Message,
    event_count: c_int,
    events: ?[*]const Message,
};

pub const marshal_flag_destroy: u32 = 1;

pub const Client = struct {
    lib: std.DynLib,

    display_connect: *const fn (?[*:0]const u8) callconv(.c) ?*Display,
    display_disconnect: *const fn (*Display) callconv(.c) void,
    display_dispatch: *const fn (*Display) callconv(.c) c_int,
    display_dispatch_pending: *const fn (*Display) callconv(.c) c_int,
    display_roundtrip: *const fn (*Display) callconv(.c) c_int,
    display_flush: *const fn (*Display) callconv(.c) c_int,
    proxy_marshal_flags: *const fn (*Proxy, u32, ?*const Interface, u32, u32, ...) callconv(.c) ?*Proxy,
    proxy_add_listener: *const fn (*Proxy, [*]const ?*const anyopaque, ?*anyopaque) callconv(.c) c_int,
    proxy_destroy: *const fn (*Proxy) callconv(.c) void,
    proxy_get_version: *const fn (*Proxy) callconv(.c) u32,

    registry: *const Interface,
    compositor: *const Interface,
    shm: *const Interface,
    shm_pool: *const Interface,
    surface: *const Interface,
    buffer: *const Interface,
    seat: *const Interface,
    keyboard: *const Interface,
    pointer: *const Interface,
    callback: *const Interface,

    pub fn load() !Client {
        var lib = std.DynLib.open("libwayland-client.so.0") catch return error.NoWayland;
        errdefer lib.close();
        var c: Client = undefined;
        c.lib = lib;
        const fns = .{
            .{ "display_connect", "wl_display_connect" },
            .{ "display_disconnect", "wl_display_disconnect" },
            .{ "display_dispatch", "wl_display_dispatch" },
            .{ "display_dispatch_pending", "wl_display_dispatch_pending" },
            .{ "display_roundtrip", "wl_display_roundtrip" },
            .{ "display_flush", "wl_display_flush" },
            .{ "proxy_marshal_flags", "wl_proxy_marshal_flags" },
            .{ "proxy_add_listener", "wl_proxy_add_listener" },
            .{ "proxy_destroy", "wl_proxy_destroy" },
            .{ "proxy_get_version", "wl_proxy_get_version" },
        };
        inline for (fns) |f| {
            @field(c, f[0]) = c.lib.lookup(@FieldType(Client, f[0]), f[1]) orelse return error.NoWayland;
        }
        const ifaces = .{
            .{ "registry", "wl_registry_interface" },
            .{ "compositor", "wl_compositor_interface" },
            .{ "shm", "wl_shm_interface" },
            .{ "shm_pool", "wl_shm_pool_interface" },
            .{ "surface", "wl_surface_interface" },
            .{ "buffer", "wl_buffer_interface" },
            .{ "seat", "wl_seat_interface" },
            .{ "keyboard", "wl_keyboard_interface" },
            .{ "pointer", "wl_pointer_interface" },
            .{ "callback", "wl_callback_interface" },
        };
        inline for (ifaces) |f| {
            @field(c, f[0]) = c.lib.lookup(*const Interface, f[1]) orelse return error.NoWayland;
        }
        return c;
    }
};

// ---------------------------------------------------------------------------
// xdg-shell (aus xdg-shell.xml, hier auf Version 1 beschränkt)
// ---------------------------------------------------------------------------

var xdg_positioner_types = [_]?*const Interface{null};

pub var xdg_wm_base_interface: Interface = undefined;
pub var xdg_surface_interface: Interface = undefined;
pub var xdg_toplevel_interface: Interface = undefined;

var wm_base_methods: [4]Message = undefined;
var wm_base_events: [1]Message = undefined;
var surface_methods: [5]Message = undefined;
var surface_events: [1]Message = undefined;
var toplevel_methods: [14]Message = undefined;
var toplevel_events: [2]Message = undefined;

var t_wm_get_surface: [2]?*const Interface = undefined;
var t_surface_toplevel: [1]?*const Interface = undefined;
var t_none3: [4]?*const Interface = .{ null, null, null, null };

/// Füllt die xdg-shell-Tabellen; braucht wl_surface/wl_seat aus der Bibliothek.
pub fn initXdg(c: *const Client) void {
    t_wm_get_surface = .{ &xdg_surface_interface, c.surface };
    t_surface_toplevel = .{&xdg_toplevel_interface};

    wm_base_methods = .{
        .{ .name = "destroy", .signature = "", .types = null },
        .{ .name = "create_positioner", .signature = "n", .types = &xdg_positioner_types },
        .{ .name = "get_xdg_surface", .signature = "no", .types = &t_wm_get_surface },
        .{ .name = "pong", .signature = "u", .types = &t_none3 },
    };
    wm_base_events = .{.{ .name = "ping", .signature = "u", .types = &t_none3 }};
    xdg_wm_base_interface = .{
        .name = "xdg_wm_base",
        .version = 1,
        .method_count = wm_base_methods.len,
        .methods = &wm_base_methods,
        .event_count = wm_base_events.len,
        .events = &wm_base_events,
    };

    surface_methods = .{
        .{ .name = "destroy", .signature = "", .types = null },
        .{ .name = "get_toplevel", .signature = "n", .types = &t_surface_toplevel },
        .{ .name = "get_popup", .signature = "n?oo", .types = &t_none3 },
        .{ .name = "set_window_geometry", .signature = "iiii", .types = &t_none3 },
        .{ .name = "ack_configure", .signature = "u", .types = &t_none3 },
    };
    surface_events = .{.{ .name = "configure", .signature = "u", .types = &t_none3 }};
    xdg_surface_interface = .{
        .name = "xdg_surface",
        .version = 1,
        .method_count = surface_methods.len,
        .methods = &surface_methods,
        .event_count = surface_events.len,
        .events = &surface_events,
    };

    toplevel_methods = .{
        .{ .name = "destroy", .signature = "", .types = null },
        .{ .name = "set_parent", .signature = "?o", .types = &t_none3 },
        .{ .name = "set_title", .signature = "s", .types = &t_none3 },
        .{ .name = "set_app_id", .signature = "s", .types = &t_none3 },
        .{ .name = "show_window_menu", .signature = "ouii", .types = &t_none3 },
        .{ .name = "move", .signature = "ou", .types = &t_none3 },
        .{ .name = "resize", .signature = "ouu", .types = &t_none3 },
        .{ .name = "set_max_size", .signature = "ii", .types = &t_none3 },
        .{ .name = "set_min_size", .signature = "ii", .types = &t_none3 },
        .{ .name = "set_maximized", .signature = "", .types = null },
        .{ .name = "unset_maximized", .signature = "", .types = null },
        .{ .name = "set_fullscreen", .signature = "?o", .types = &t_none3 },
        .{ .name = "unset_fullscreen", .signature = "", .types = null },
        .{ .name = "set_minimized", .signature = "", .types = null },
    };
    toplevel_events = .{
        .{ .name = "configure", .signature = "iia", .types = &t_none3 },
        .{ .name = "close", .signature = "", .types = null },
    };
    xdg_toplevel_interface = .{
        .name = "xdg_toplevel",
        .version = 1,
        .method_count = toplevel_methods.len,
        .methods = &toplevel_methods,
        .event_count = toplevel_events.len,
        .events = &toplevel_events,
    };
}

// Opcodes der benutzten Anfragen
pub const display_get_registry = 1;
pub const registry_bind = 0;
pub const compositor_create_surface = 0;
pub const shm_create_pool = 0;
pub const shm_pool_create_buffer = 0;
pub const surface_attach = 1;
pub const surface_frame = 3;
pub const surface_commit = 6;
pub const surface_damage_buffer = 9;
pub const seat_get_pointer = 0;
pub const seat_get_keyboard = 1;
pub const wm_base_get_xdg_surface = 2;
pub const wm_base_pong = 3;
pub const xdg_surface_get_toplevel = 1;
pub const xdg_surface_ack_configure = 4;
pub const toplevel_set_title = 2;
pub const toplevel_set_app_id = 3;

/// wl_shm-Format XRGB8888 (Bytes B, G, R, ungenutzt) – passt zu PYR_POST_BGRA
pub const shm_format_xrgb8888: u32 = 1;

// evdev-Tastencodes (Wayland liefert sie ohne X11-Versatz)
pub const key_esc = 1;
pub const key_w = 17;
pub const key_e = 18;
pub const key_a = 30;
pub const key_s = 31;
pub const key_d = 32;
pub const key_q = 16;
pub const key_f = 33;
pub const key_f3 = 61;
pub const key_b = 48;
pub const key_t = 20;
pub const key_n = 49;
pub const key_g = 34;
pub const key_u = 22;
pub const key_k = 37;
pub const key_j = 36;
pub const key_space = 57;
pub const key_leftshift = 42;
pub const key_leftctrl = 29;
pub const key_minus = 12;
pub const key_equal = 13;
pub const key_l = 38;
pub const key_m = 50;
pub const key_v = 47;
pub const key_1 = 2;
pub const key_2 = 3;
pub const key_3 = 4;
pub const key_4 = 5;
pub const key_5 = 6;
pub const btn_left: u32 = 0x110;
