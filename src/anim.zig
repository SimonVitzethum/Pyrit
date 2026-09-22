//! Skelettanimation, Host-Seite: Skelette (Knochen mit Voxel-Teilen), Clips
//! (Keyframes), Akteure (Instanz je Teil). Abgespielt wird vollständig auf der
//! GPU (src/device/anim.zig) bei jedem pyr_commit aus der Szenenzeit; der Host
//! lädt nur hoch, was sich geändert hat.

const std = @import("std");
const types = @import("pyrit_device").types;
const api = @import("api.zig");
const cuda = @import("cuda.zig");
const diag = @import("diag.zig");
const ctx_mod = @import("context.zig");
const Context = ctx_mod.Context;

const Error = diag.Error;
const fail = diag.fail;

fn oom(v: anytype) Error!@typeInfo(@TypeOf(v)).error_union.payload {
    return v catch return fail(error.OutOfMemory, "Host-Speicher", .{});
}

pub const Bone = extern struct {
    /// Elternknochen (kleinerer Index) oder -1
    parent: i32,
    reserved: u32,
    /// Voxel-Teil des Knochens oder null
    geometry: api.Handle,
    /// Ruhelage relativ zum Elternknochen (ohne Keyframes)
    rest: [12]f32,
    /// Voxel-Teil relativ zum Knochen
    part: [12]f32,
    /// Anteil an der Windbewegung (0 = starr)
    sway: f32,
    reserved2: u32,
};

pub const ActorInfo = extern struct {
    /// Lage des Akteurs (3x4, relativ zum Render-Ursprung)
    root: [12]f32,
    clip: api.Handle,
    blend_clip: api.Handle,
    /// Clipzeit = (Szenenzeit - start_time) * speed
    start_time: f32,
    speed: f32,
    blend_start_time: f32,
    blend_speed: f32,
    /// Gewicht von blend_clip in [0, 1]
    blend: f32,
    /// Wind: Auslenkung in Radiant, Frequenz in Hz, Phase
    wind_amplitude: f32,
    wind_frequency: f32,
    wind_phase: f32,
    /// Instanzmaske der Teile, 0 = 0xFF
    mask: u32,
    /// PyrInstanceData.user der Teile
    user: u32,
    reserved: u32,
};

const Skeleton = struct {
    generation: u32 = 1,
    alive: bool = false,
    bone_offset: u32 = 0,
    bone_count: u32 = 0,
    geometries: []usize = &.{},
    actors: u32 = 0,
};

const Clip = struct {
    generation: u32 = 1,
    alive: bool = false,
    skeleton: u32 = 0,
};

const Actor = struct {
    generation: u32 = 1,
    alive: bool = false,
    skeleton: u32 = 0,
    instances: []usize = &.{},
};

/// Gerätespiegel eines Host-Arrays (wächst, wird bei Änderung komplett hochgeladen)
fn Mirror(comptime T: type) type {
    return struct {
        items: std.ArrayList(T) = .empty,
        dev: cuda.CUdeviceptr = 0,
        cap: usize = 0,
        dirty: bool = false,

        const Self = @This();

        fn sync(self: *Self, c: *Context) Error!void {
            if (!self.dirty) return;
            self.dirty = false;
            const n = self.items.items.len;
            if (n == 0) return;
            if (n > self.cap) {
                // alter Puffer kann noch von einem laufenden Kernel gelesen werden
                if (self.dev != 0) {
                    try c.check(c.drv.cuStreamSynchronize(c.stream), "cuStreamSynchronize");
                    _ = c.drv.cuMemFree_v2(self.dev);
                }
                self.cap = @max(n * 2, 64);
                self.dev = try c.devAlloc(self.cap * @sizeOf(T), "Animation");
            }
            try c.upload(self.dev, std.mem.sliceAsBytes(self.items.items));
        }

        fn deinit(self: *Self, c: *Context) void {
            self.items.deinit(c.gpa);
            if (self.dev != 0) _ = c.drv.cuMemFree_v2(self.dev);
        }
    };
}

pub const Animation = struct {
    skeletons: std.ArrayList(Skeleton) = .empty,
    clips: std.ArrayList(Clip) = .empty,
    actors: std.ArrayList(Actor) = .empty,
    actor_free: std.ArrayList(u32) = .empty,

    bones: Mirror(types.AnimBone) = .{},
    tracks: Mirror([2]u32) = .{},
    keys: Mirror(types.Keyframe) = .{},
    clip_data: Mirror(types.AnimClip) = .{},
    actor_data: Mirror(types.AnimActor) = .{},
    jobs: Mirror(types.AnimJob) = .{},
    jobs_stale: bool = false,

    pub fn deinit(self: *Animation, c: *Context) void {
        for (self.skeletons.items) |s| c.gpa.free(s.geometries);
        for (self.actors.items) |a| c.gpa.free(a.instances);
        self.skeletons.deinit(c.gpa);
        self.clips.deinit(c.gpa);
        self.actors.deinit(c.gpa);
        self.actor_free.deinit(c.gpa);
        self.bones.deinit(c);
        self.tracks.deinit(c);
        self.keys.deinit(c);
        self.clip_data.deinit(c);
        self.actor_data.deinit(c);
        self.jobs.deinit(c);
    }

    fn lookup(comptime T: type, list: []T, h: usize, what: []const u8) Error!u32 {
        const d = ctx_mod.decodeHandle(h);
        if (h == 0 or d.index >= list.len or !list[d.index].alive or list[d.index].generation != d.generation)
            return fail(error.InvalidHandle, "ungültige(r) {s}", .{what});
        return d.index;
    }

    // -------------------------------------------------------------------
    // Skelette und Clips
    // -------------------------------------------------------------------

    pub fn skeletonCreate(self: *Animation, c: *Context, bones: []const Bone) Error!usize {
        if (bones.len == 0 or bones.len > 256) return fail(error.InvalidArgument, "ein Skelett hat 1 bis 256 Knochen", .{});
        const geos = try oom(c.gpa.alloc(usize, bones.len));
        errdefer c.gpa.free(geos);
        const offset: u32 = @intCast(self.bones.items.items.len);
        try oom(self.bones.items.ensureUnusedCapacity(c.gpa, bones.len));
        for (bones, 0..) |b, i| {
            if (b.parent >= @as(i32, @intCast(i)) or b.parent < -1)
                return fail(error.InvalidArgument, "Knochen {d}: Eltern müssen vor dem Kind stehen (parent {d})", .{ i, b.parent });
            geos[i] = @intFromPtr(b.geometry);
            var size: f32 = 0;
            if (b.geometry != null) size = @floatFromInt(@as(u32, 1) << @intCast(try c.geometryLog2(geos[i])));
            self.bones.items.appendAssumeCapacity(.{ .parent = b.parent, .part_size = size, .sway = b.sway, .reserved = 0, .rest = b.rest, .part = b.part });
        }
        self.bones.dirty = true;
        try oom(self.skeletons.append(c.gpa, .{ .alive = true, .bone_offset = offset, .bone_count = @intCast(bones.len), .geometries = geos }));
        return ctx_mod.encodeHandle(@intCast(self.skeletons.items.len - 1), 1);
    }

    pub fn clipCreate(self: *Animation, c: *Context, skeleton: usize, keys_in: []const types.Keyframe, duration: f32, flags: u32) Error!usize {
        const si = try lookup(Skeleton, self.skeletons.items, skeleton, "Skelett");
        const sk = &self.skeletons.items[si];
        if (!(duration >= 0)) return fail(error.InvalidArgument, "duration < 0", .{});
        // nach (Knochen, Zeit) sortiert ablegen
        const keys = try oom(c.gpa.dupe(types.Keyframe, keys_in));
        defer c.gpa.free(keys);
        for (keys) |k| {
            if (k.bone >= sk.bone_count) return fail(error.InvalidArgument, "Keyframe für Knochen {d}, das Skelett hat {d}", .{ k.bone, sk.bone_count });
        }
        std.mem.sort(types.Keyframe, keys, {}, struct {
            fn less(_: void, a: types.Keyframe, b: types.Keyframe) bool {
                return if (a.bone != b.bone) a.bone < b.bone else a.time < b.time;
            }
        }.less);
        const track_offset: u32 = @intCast(self.tracks.items.items.len);
        const key_offset: u32 = @intCast(self.keys.items.items.len);
        try oom(self.tracks.items.ensureUnusedCapacity(c.gpa, sk.bone_count));
        var k: u32 = 0;
        var b: u32 = 0;
        while (b < sk.bone_count) : (b += 1) {
            const first = k;
            while (k < keys.len and keys[k].bone == b) k += 1;
            self.tracks.items.appendAssumeCapacity(.{ key_offset + first, k - first });
        }
        try oom(self.keys.items.appendSlice(c.gpa, keys));
        try oom(self.clip_data.items.append(c.gpa, .{ .track_offset = track_offset, .bone_count = sk.bone_count, .duration = duration, .flags = flags }));
        try oom(self.clips.append(c.gpa, .{ .alive = true, .skeleton = si }));
        self.tracks.dirty = true;
        self.keys.dirty = true;
        self.clip_data.dirty = true;
        return ctx_mod.encodeHandle(@intCast(self.clips.items.len - 1), 1);
    }

    // -------------------------------------------------------------------
    // Akteure
    // -------------------------------------------------------------------

    fn clipIndex(self: *Animation, h: api.Handle, skeleton: u32) Error!u32 {
        if (h == null) return types.no_clip;
        const ci = try lookup(Clip, self.clips.items, @intFromPtr(h), "Clip");
        if (self.clips.items[ci].skeleton != skeleton) return fail(error.InvalidArgument, "Clip gehört zu einem anderen Skelett", .{});
        return ci;
    }

    fn stateOf(self: *Animation, si: u32, info: *const ActorInfo) Error!types.AnimActor {
        const sk = &self.skeletons.items[si];
        return .{
            .root = info.root,
            .bone_offset = sk.bone_offset,
            .bone_count = sk.bone_count,
            .clip = try self.clipIndex(info.clip, si),
            .blend_clip = try self.clipIndex(info.blend_clip, si),
            .start_time = info.start_time,
            .speed = info.speed,
            .blend_start_time = info.blend_start_time,
            .blend_speed = info.blend_speed,
            .blend = std.math.clamp(info.blend, 0, 1),
            .wind_amplitude = info.wind_amplitude,
            .wind_frequency = info.wind_frequency,
            .wind_phase = info.wind_phase,
        };
    }

    pub fn actorCreate(self: *Animation, c: *Context, skeleton: usize, info: *const ActorInfo) Error!usize {
        const si = try lookup(Skeleton, self.skeletons.items, skeleton, "Skelett");
        const state = try self.stateOf(si, info);
        const sk = &self.skeletons.items[si];
        const insts = try oom(c.gpa.alloc(usize, sk.bone_count));
        @memset(insts, 0);
        errdefer {
            for (insts) |h| {
                if (h != 0) c.instanceDestroy(h) catch {};
            }
            c.gpa.free(insts);
        }
        const idx: u32 = self.actor_free.pop() orelse blk: {
            try oom(self.actors.append(c.gpa, .{}));
            try oom(self.actor_data.items.append(c.gpa, state));
            break :blk @intCast(self.actors.items.len - 1);
        };
        for (sk.geometries, 0..) |g, b| {
            if (g == 0) continue;
            const h = try c.instanceCreate(g);
            insts[b] = h;
            try c.instanceSetMask(h, if (info.mask == 0) 0xFF else info.mask);
            try c.instanceSetUser(h, info.user);
        }
        const a = &self.actors.items[idx];
        a.alive = true;
        a.skeleton = si;
        a.instances = insts;
        self.actor_data.items.items[idx] = state;
        self.actor_data.dirty = true;
        self.jobs_stale = true;
        sk.actors += 1;
        return ctx_mod.encodeHandle(idx, a.generation);
    }

    /// Gibt ein Skelett frei (die Knochen bleiben im Gerätearray liegen, sie
    /// sind winzig; der Platz wird wiederverwendet, sobald er gebraucht wird).
    pub fn skeletonDestroy(self: *Animation, c: *Context, skeleton: usize) Error!void {
        const si = try lookup(Skeleton, self.skeletons.items, skeleton, "Skelett");
        const sk = &self.skeletons.items[si];
        if (sk.actors != 0) return fail(error.InUse, "Skelett wird noch von {d} Akteur(en) verwendet", .{sk.actors});
        c.gpa.free(sk.geometries);
        sk.geometries = &.{};
        sk.alive = false;
        sk.generation +%= 1;
    }

    pub fn clipDestroy(self: *Animation, c: *Context, clip: usize) Error!void {
        const ci = try lookup(Clip, self.clips.items, clip, "Clip");
        // läuft der Clip noch irgendwo?
        for (self.actors.items, 0..) |a, i| {
            if (!a.alive) continue;
            const st = self.actor_data.items.items[i];
            if (st.clip == ci or st.blend_clip == ci)
                return fail(error.InUse, "Clip wird noch von einem Akteur abgespielt", .{});
        }
        _ = c;
        self.clips.items[ci].alive = false;
        self.clips.items[ci].generation +%= 1;
    }

    pub fn actorSet(self: *Animation, c: *Context, actor: usize, info: *const ActorInfo) Error!void {
        const ai = try lookup(Actor, self.actors.items, actor, "Akteur");
        const a = &self.actors.items[ai];
        self.actor_data.items.items[ai] = try self.stateOf(a.skeleton, info);
        self.actor_data.dirty = true;
        for (a.instances) |h| {
            if (h == 0) continue;
            try c.instanceSetMask(h, if (info.mask == 0) 0xFF else info.mask);
            try c.instanceSetUser(h, info.user);
        }
    }

    pub fn actorDestroy(self: *Animation, c: *Context, actor: usize) Error!void {
        const ai = try lookup(Actor, self.actors.items, actor, "Akteur");
        const a = &self.actors.items[ai];
        for (a.instances) |h| {
            if (h != 0) try c.instanceDestroy(h);
        }
        c.gpa.free(a.instances);
        a.instances = &.{};
        a.alive = false;
        a.generation +%= 1;
        self.skeletons.items[a.skeleton].actors -= 1;
        try oom(self.actor_free.append(c.gpa, ai));
        self.jobs_stale = true;
    }

    // -------------------------------------------------------------------
    // Je Frame (pyr_commit): Knochen auf der GPU in die Instanzen schreiben
    // -------------------------------------------------------------------

    /// Liefert true, wenn Instanzen geschrieben wurden (IAS muss nachziehen).
    pub fn apply(self: *Animation, c: *Context, instances: cuda.CUdeviceptr, time: f64) Error!bool {
        if (self.jobs_stale) {
            self.jobs_stale = false;
            self.jobs.items.clearRetainingCapacity();
            for (self.actors.items, 0..) |a, ai| {
                if (!a.alive) continue;
                for (a.instances, 0..) |h, b| {
                    if (h == 0) continue;
                    try oom(self.jobs.items.append(c.gpa, .{ .actor = @intCast(ai), .bone = @intCast(b), .instance = ctx_mod.decodeHandle(h).index, .reserved = 0 }));
                }
            }
            self.jobs.dirty = true;
        }
        const n: u32 = @intCast(self.jobs.items.items.len);
        if (n == 0) return false;
        try self.bones.sync(c);
        try self.tracks.sync(c);
        try self.keys.sync(c);
        try self.clip_data.sync(c);
        try self.actor_data.sync(c);
        try self.jobs.sync(c);
        var p = types.AnimParams{
            .bones = self.bones.dev,
            .tracks = self.tracks.dev,
            .keys = self.keys.dev,
            .clips = self.clip_data.dev,
            .actors = self.actor_data.dev,
            .jobs = self.jobs.dev,
            .instances = instances,
            .job_count = n,
            .reserved = 0,
            .time = time,
        };
        const params = [_]?*anyopaque{@ptrCast(&p)};
        try c.launch(c.fn_animate, .{ (n + types.anim_block - 1) / types.anim_block, 1, 1 }, .{ types.anim_block, 1, 1 }, &params);
        return true;
    }
};
