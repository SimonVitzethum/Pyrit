//! Fehlerarten und thread-lokale Fehlermeldungen der C-API.

const std = @import("std");

pub const Error = error{
    InvalidArgument,
    InvalidHandle,
    OutOfMemory,
    Capacity,
    InUse,
    Cuda,
    Compile,
    NotFound,
    Version,
};

threadlocal var message_buf: [4096]u8 = undefined;
threadlocal var message_len: usize = 0;

pub fn clear() void {
    message_len = 0;
}

/// Setzt die Meldung und gibt `err` zurück, damit `return fail(...)` reicht.
pub fn fail(err: Error, comptime fmt: []const u8, args: anytype) Error {
    const s = std.fmt.bufPrint(message_buf[0 .. message_buf.len - 1], fmt, args) catch blk: {
        // zu lang: abschneiden
        break :blk message_buf[0 .. message_buf.len - 1];
    };
    message_len = s.len;
    message_buf[message_len] = 0;
    return err;
}

pub fn message() [*:0]const u8 {
    if (message_len == 0) return "";
    return @ptrCast(&message_buf);
}

/// Monotone Zeit in Millisekunden (Messungen auf dem Host)
pub fn nowMs() f64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}
