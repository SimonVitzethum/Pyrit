//! Wiederholungs-Wavefront: die ganze Schattierung läuft in CUDA, die
//! RT-Cores verfolgen nur Strahlen.
//!
//! Warum: steht optixTrace *in* der Schattierung, zerlegt OptiX das Programm
//! an jedem Aufruf in Fortsetzungen und muss den Zustand über den ganzen
//! Aufrufgraphen retten. Das Schattierungsmodul brauchte damit über zehn
//! Minuten und mehrere GB Arbeitsspeicher zum Übersetzen – ohne den Aufruf
//! 1,5 Sekunden. OptiX bekommt deshalb nur noch ein winziges Programm, das
//! eine Liste von Strahlen verfolgt.
//!
//! Wie, ohne die Schattierung ein zweites Mal zu schreiben: ein CUDA-Kernel
//! führt je Pixel die unveränderte Schattierung aus. Sein Strahlverfolger
//! nummeriert die Aufrufe: ist Aufruf i schon beantwortet und der Strahl
//! derselbe, gibt er den Treffer zurück; sonst trägt er den Strahl ein und
//! meldet vorläufig "kein Treffer". OptiX verfolgt dann alle eingetragenen
//! Strahlen auf einmal, und der nächste Durchgang wiederholt die Schattierung.
//! Die Zufallszahlen hängen nur von Pixel und Frame ab – jeder Durchgang
//! wiederholt also genau dieselben Strahlen, bis ein neu bekannter Treffer
//! den Weg ändert. Strahlen, die nicht voneinander abhängen (alle
//! Schattenstrahlen eines Pixels), fallen in denselben Durchgang; die Zahl
//! der Durchgänge ist die Tiefe der Abhängigkeiten, nicht die Zahl der
//! Strahlen. Das Ergebnis ist bitgleich zum Pfad ohne RT-Cores.

const types = @import("types.zig");
const vec = @import("vec.zig");
const tr = @import("trace.zig");
const render = @import("render.zig");
const Vec3 = vec.Vec3;

pub const Tracer = struct {
    rp: *const types.ReplayParams,
    /// Platzbasis dieses Pixels (Pixel im Streifen · slots)
    base: u64,
    flags: u32,
    next: *u32,
    pending: *bool,

    pub fn trace(self: Tracer, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?tr.TraceHit {
        const i = self.next.*;
        self.next.* += 1;
        // mehr Aufrufe als Plätze: dieser Strahl bleibt ohne Treffer (selten)
        if (i >= types.replay_slots) return null;
        // Nach einer offenen Anfrage rechnet die Schattierung mit einem
        // vorläufigen "kein Treffer" weiter. Die Strahlen, die sie dann
        // schießt, gehören meist zu einem Weg, den es gar nicht gibt (am
        // Primärstrahl gemessen: 4 von 5 wurden im nächsten Durchgang anders
        // angefragt). Ohne Spekulation bleibt es bei der ersten Anfrage.
        if (self.pending.* and self.rp.speculate == 0) return null;
        const k = self.base + i;
        const want = types.ReplayRay{
            .o = .{ o[0], o[1], o[2] },
            .tmin = tmin,
            .d = .{ d[0], d[1], d[2] },
            .tmax = tmax,
            .mask = ray_mask,
            .flags = self.flags | flags,
        };
        const rays: [*]types.ReplayRay = @ptrFromInt(self.rp.rays);
        const state: [*]u32 = @ptrFromInt(self.rp.state);
        if (state[k] == 2 and same(rays[k], want)) {
            const h = @as([*]const types.ReplayHit, @ptrFromInt(self.rp.hits))[k];
            if (h.instance == types.no_hit) return null;
            const inst = &tr.instances(s)[h.instance];
            return .{
                .t = h.t,
                .instance = h.instance,
                .attribute = h.attribute,
                .face = h.face,
                .voxel = .{ 0, 0, 0 },
                .p_object = vec.xformPoint(&inst.world_to_object, o + d * vec.splat(h.t)),
            };
        }
        // neu oder anders als beim letzten Mal: anfragen
        request(self.rp, k, want);
        self.pending.* = true;
        return null;
    }
};

fn request(rp: *const types.ReplayParams, k: u64, want: types.ReplayRay) void {
    const rays: [*]types.ReplayRay = @ptrFromInt(rp.rays);
    const state: [*]u32 = @ptrFromInt(rp.state);
    rays[k] = want;
    state[k] = 1;
    const counter: *u32 = @ptrFromInt(rp.count);
    const slot = @atomicRmw(u32, counter, .Add, 1, .monotonic);
    if (slot < rp.capacity) {
        @as([*]u32, @ptrFromInt(rp.list))[slot] = @intCast(k);
    } else {
        state[k] = 0; // Liste voll: im nächsten Durchgang erneut
    }
}

/// Erster Durchgang des Bildes ohne Schattierung: nur den Primärstrahl
/// eintragen (Platz 0, bitgleich zu dem, den renderPixelWith anfragt).
/// Die Schattierung liefe hier sonst ganz durch, nur um ihn anzufragen.
pub fn primaryPass(p: *const types.RenderParams, rp: *const types.ReplayParams, i: u32, trace_flags: u32) void {
    const w = p.cur.camera.width;
    if (i >= w * rp.rows) return;
    const y = rp.y0 + i / w;
    if (y >= p.cur.camera.height) return;
    const pr = render.primaryRay(p, i % w, y);
    request(rp, @as(u64, i) * types.replay_slots, .{
        .o = .{ pr.ray.o[0], pr.ray.o[1], pr.ray.o[2] },
        .tmin = pr.ray.tmin,
        .d = .{ pr.ray.d[0], pr.ray.d[1], pr.ray.d[2] },
        .tmax = pr.ray.tmax,
        .mask = pr.mask,
        .flags = trace_flags | pr.flags,
    });
}

inline fn same(a: types.ReplayRay, b: types.ReplayRay) bool {
    const x: [10]u32 = @bitCast(a);
    const y: [10]u32 = @bitCast(b);
    inline for (0..10) |q| if (x[q] != y[q]) return false;
    return true;
}

/// Ein Durchgang für Pixel `i` des Streifens (Bild)
pub fn renderPass(p: *const types.RenderParams, rp: *const types.ReplayParams, s: *const types.Scene, i: u32, trace_flags: u32) void {
    const w = p.cur.camera.width;
    if (i >= w * rp.rows) return;
    const done: [*]u32 = @ptrFromInt(rp.done);
    if (done[i] != 0) return;
    const x = i % w;
    const y = rp.y0 + i / w;
    if (y >= p.cur.camera.height) return;
    var next: u32 = 0;
    var pending = false;
    const t = Tracer{ .rp = rp, .base = @as(u64, i) * types.replay_slots, .flags = trace_flags, .next = &next, .pending = &pending };
    const r = render.renderPixelWith(t, p, s, x, y);
    render.writePixel(p, @as(u64, y) * w + x, &r);
    if (!pending) done[i] = 1;
}

/// Ein Durchgang für GI-Pixel `i` des Streifens (halbe Auflösung)
pub fn giPass(p: *const types.RenderParams, rp: *const types.ReplayParams, s: *const types.Scene, i: u32, trace_flags: u32) void {
    const w = p.gi_width;
    if (i >= w * rp.rows) return;
    const done: [*]u32 = @ptrFromInt(rp.done);
    if (done[i] != 0) return;
    const x = i % w;
    const y = rp.y0 + i / w;
    if (y >= p.gi_height) return;
    var next: u32 = 0;
    var pending = false;
    const t = Tracer{ .rp = rp, .base = @as(u64, i) * types.replay_slots, .flags = trace_flags, .next = &next, .pending = &pending };
    render.giPixel(t, p, s, x, y);
    if (!pending) done[i] = 1;
}
