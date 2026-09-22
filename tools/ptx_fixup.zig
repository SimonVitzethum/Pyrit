//! Build-Werkzeug: bereitet das LLVM-IR der Zig-Kernel für NVPTX/OptiX vor.
//!
//! 1. Zig stellt exportierte Funktionen als Alias dar; NVPTX erlaubt keine
//!    Aliasse auf Kernel. Das Ziel bekommt direkt den Exportnamen und wird
//!    öffentlich.
//! 2. OptiX erwartet die Launch-Parameter als definierte `.const`-Variable.
//!    In Zig sind sie `extern const ... addrspace(.constant)` (Zig erlaubt dort
//!    keine veränderlichen Werte); hier werden sie zu einer extern
//!    initialisierten Definition im NVPTX-Konstantenraum (addrspace 4).
//!
//!   ptx_fixup eingabe.ll ausgabe.ll

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) return error.Usage;
    const cwd = std.Io.Dir.cwd();
    const src = try cwd.readFileAlloc(init.io, args[1], arena, .unlimited);
    const out = try fixup(arena, src);
    try cwd.writeFile(init.io, .{ .sub_path = args[2], .data = out });
}

pub fn fixup(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    var renames: std.ArrayList([2][]const u8) = .empty;
    var kept: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |line| {
        if (parseAlias(line)) |pair| {
            try renames.append(gpa, pair);
            continue;
        }
        if (try constantParam(gpa, line)) |def| {
            try kept.appendSlice(gpa, def);
        } else {
            try kept.appendSlice(gpa, line);
        }
        try kept.append(gpa, '\n');
    }

    var text: []u8 = kept.items;
    for (renames.items) |r| text = try replaceSymbol(gpa, text, r[1], r[0]);
    // Nur die Ziele der Aliasse (die Exporte) öffentlich machen; interne
    // Funktionen bleiben intern, damit LLVM ihre Namen PTX-gültig umbenennt.
    for (renames.items) |r| text = try publish(gpa, text, r[0]);
    // Zigs Konstantenraum (2) auf den NVPTX-Konstantenraum (4) abbilden
    text = try std.mem.replaceOwned(u8, gpa, text, "addrspace(2)", "addrspace(4)");
    return text;
}

/// Entfernt private/internal in der Definitionszeile von @name.
fn publish(gpa: std.mem.Allocator, text: []u8, name: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    const needle = try std.fmt.allocPrint(gpa, "@{s}(", .{name});
    while (lines.next()) |line| {
        if (!first) try out.append(gpa, '\n');
        first = false;
        if (std.mem.startsWith(u8, line, "define ") and std.mem.indexOf(u8, line, needle) != null) {
            var l = try std.mem.replaceOwned(u8, gpa, line, "define private ", "define ");
            l = try std.mem.replaceOwned(u8, gpa, l, "define internal ", "define ");
            try out.appendSlice(gpa, l);
        } else {
            try out.appendSlice(gpa, line);
        }
    }
    return out.items;
}

/// "@kern = alias void (%a.P), ptr @a.kern" -> { "kern", "a.kern" }
fn parseAlias(line: []const u8) ?[2][]const u8 {
    if (!std.mem.startsWith(u8, line, "@")) return null;
    const eq = std.mem.indexOf(u8, line, " = alias ") orelse return null;
    const ptr = std.mem.lastIndexOf(u8, line, ", ptr @") orelse return null;
    return .{ line[1..eq], std.mem.trim(u8, line[ptr + 7 ..], " \r") };
}

/// "@x = external local_unnamed_addr addrspace(2) constant %T, align 8"
///   -> "@x = addrspace(4) externally_initialized global %T zeroinitializer, align 8"
fn constantParam(gpa: std.mem.Allocator, line: []const u8) !?[]u8 {
    if (!std.mem.startsWith(u8, line, "@")) return null;
    const marker = " addrspace(2) constant ";
    const m = std.mem.indexOf(u8, line, marker) orelse return null;
    const eq = std.mem.indexOf(u8, line, " = external ") orelse return null;
    const rest = line[m + marker.len ..];
    const comma = std.mem.indexOfScalar(u8, rest, ',') orelse rest.len;
    return try std.fmt.allocPrint(gpa, "{s} = addrspace(4) externally_initialized global {s} zeroinitializer{s}", .{ line[0..eq], rest[0..comma], rest[comma..] });
}

fn isIdentChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '$' or ch == '-';
}

/// Ersetzt @from (als ganzes Symbol) durch @to.
fn replaceSymbol(gpa: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '@' and std.mem.startsWith(u8, text[i + 1 ..], from)) {
            const end = i + 1 + from.len;
            if (end == text.len or !isIdentChar(text[end])) {
                try out.append(gpa, '@');
                try out.appendSlice(gpa, to);
                i = end;
                continue;
            }
        }
        try out.append(gpa, text[i]);
        i += 1;
    }
    return out.items;
}

test "Alias und Konstantenparameter" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src =
        \\@p = external local_unnamed_addr addrspace(2) constant %rt.Params, align 8
        \\@__raygen__a = alias void (), ptr @rt.__raygen__a
        \\define private ptx_kernel void @rt.__raygen__a() {
        \\}
        \\define internal void @rt.helper() {
        \\  %1 = load i32, ptr addrspace(2) @p, align 8
        \\}
    ;
    const out = try fixup(arena.allocator(), src);
    try std.testing.expect(std.mem.indexOf(u8, out, "alias") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "define ptx_kernel void @__raygen__a()") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "@p = addrspace(4) externally_initialized global %rt.Params zeroinitializer, align 8") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "addrspace(2)") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "define internal void @rt.helper()") != null);
}
