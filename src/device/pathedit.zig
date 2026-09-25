//! Pfadänderung: Voxel eines fertigen Chunks setzen, ändern oder entfernen,
//! ohne den Chunk neu zu erzeugen und zu bauen.
//!
//! Eine Änderung berührt nur den Weg von der Wurzel zum Brick: der Brick und
//! die Knoten darüber (bei 32^3 drei) werden als Kopien mit angepasster Maske,
//! Voxelzahl und Verweis neu geschrieben, alles andere bleibt geteilt. Die
//! Kopien liegen in einer Arena am oberen Ende der Pools. Weil sie oberhalb
//! jeder Batch liegt, erreicht jeder Chunk sie mit den gewöhnlichen relativen
//! Verweisen des Formats (dag.zig) – Format und Traversierung bleiben gleich.
//!
//! Attribute liegen in Tiefensuch-Reihenfolge (Rang aus den Voxelzahlen).
//! Ein neuer oder entfernter Voxel verschiebt alle späteren um eins; das
//! erledigen alle Threads des Blocks gemeinsam mit zwei Puffern im Wechsel.
//! Die RT-Primitive des Chunks zeigen auf Teilbäume: ihr Knoten und der Rang
//! ihres ersten Voxels werden nachgezogen. Entsteht ein Teilbaum ganz neu,
//! bräuchte der Chunk ein neues Primitiv (und einen neuen GAS) – dann meldet
//! der Auftrag path_edit_rebuild, und der Host baut ihn wie bisher neu.
//!
//! Ein GPU-Block je Auftrag (Chunk); Thread 0 geht den Baum, alle Threads
//! verschieben die Attribute.

const builtin = @import("builtin");
const types = @import("types.zig");
const dag = @import("dag.zig");

const nvptx = builtin.cpu.arch == .nvptx64 or builtin.cpu.arch == .nvptx;

inline fn barrier() void {
    if (comptime nvptx) asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn bitOf(v: u32, s: u32) u32 {
    return (v >> @intCast(s)) & 1;
}

inline fn childIndex(v: [3]u32, s: u32) u32 {
    return bitOf(v[0], s) | (bitOf(v[1], s) << 1) | (bitOf(v[2], s) << 2);
}

/// Morton-Code in der Kindreihenfolge des Formats (x niedrigstes Bit je Stufe)
fn morton(x: u32, y: u32, z: u32) u64 {
    var m: u64 = 0;
    var b: u6 = 0;
    while (b < 21) : (b += 1) {
        m |= @as(u64, (x >> @intCast(b)) & 1) << (3 * b);
        m |= @as(u64, (y >> @intCast(b)) & 1) << (3 * b + 1);
        m |= @as(u64, (z >> @intCast(b)) & 1) << (3 * b + 2);
    }
    return m;
}

inline fn unpackCell(c: u64) [3]u32 {
    return .{ @intCast(c & 0x1F_FFFF), @intCast((c >> 21) & 0x1F_FFFF), @intCast((c >> 42) & 0x1F_FFFF) };
}

const op_none: u32 = 0;
const op_set: u32 = 1;
const op_insert: u32 = 2;
const op_remove: u32 = 3;

/// Zustand von Thread 0 über die Änderungen eines Auftrags
const Walker = struct {
    nodes: [*]u32, // Basis = Pool + node_offset (relative Verweise)
    leaves: [*]u64, // Basis = Pool + leaf_offset
    node_next: u32, // nächster freier Arena-Platz, relativ zur Basis
    node_end: u32,
    leaf_next: u32,
    leaf_end: u32,
    overflow: bool = false,

    fn allocNode(self: *Walker, words: u32) ?u32 {
        if (self.node_next + words > self.node_end) {
            self.overflow = true;
            return null;
        }
        const r = self.node_next;
        self.node_next += words;
        return r;
    }

    fn allocLeaf(self: *Walker) ?u32 {
        if (self.leaf_next >= self.leaf_end) {
            self.overflow = true;
            return null;
        }
        const r = self.leaf_next;
        self.leaf_next += 1;
        return r;
    }
};

/// Weg von der Wurzel zum Brick, der Voxel v enthält
const Walk = struct {
    path: [dag.max_log2 + 1]u32,
    /// Stufe, auf der der Weg abbricht (Kind fehlt); 0 = bis zum Brick vorhanden
    missing: u32,
    brick: u64,
    /// Voxel vor diesem Brick (Attributrang seines ersten Voxels)
    prefix: u32,
};

fn walk(w: *const Walker, root: u32, log2_size: u32, v: [3]u32) Walk {
    var r = Walk{ .path = undefined, .missing = 0, .brick = 0, .prefix = 0 };
    var node = root;
    var s: u32 = log2_size - 1;
    while (true) {
        r.path[s] = node;
        const mask = w.nodes[node] & 0xFF;
        const ci = childIndex(v, s);
        const before: u32 = @popCount(mask & ((@as(u32, 1) << @intCast(ci)) - 1));
        var k: u32 = 0;
        while (k < before) : (k += 1) {
            const ref = w.nodes[node + 2 + k];
            r.prefix +%= if (s == 2) @popCount(w.leaves[ref]) else w.nodes[ref + 1];
        }
        if (mask & (@as(u32, 1) << @intCast(ci)) == 0) {
            r.missing = s;
            return r;
        }
        const ref = w.nodes[node + 2 + before];
        if (s == 2) {
            r.brick = w.leaves[ref];
            return r;
        }
        node = ref;
        s -= 1;
    }
}

const Rewrite = struct {
    /// neue Wurzel (null: Chunk leer)
    root: ?u32,
    /// neuer Knoten des RT-Teilbaums, der v enthält (null: Teilbaum leer)
    sub: ?u32,
};

/// Den Weg zum Brick von v mit neuem Inhalt `new_brick` neu schreiben; die
/// Voxelzahlen darüber ändern sich um `delta`. Einmal je Brick, egal wie
/// viele Voxel darin geändert wurden.
fn rewrite(w: *Walker, wk: *const Walk, log2_size: u32, sub_level: u32, v: [3]u32, new_brick: u64, delta: i32) ?Rewrite {
    var child: ?u32 = null;
    if (new_brick != 0) {
        const l = w.allocLeaf() orelse return null;
        w.leaves[l] = new_brick;
        child = l;
    }
    var sub: ?u32 = null;
    var s: u32 = 2;
    while (s < log2_size) : (s += 1) {
        const ci = childIndex(v, s);
        const cbit = @as(u32, 1) << @intCast(ci);
        const exists = wk.missing == 0 or s >= wk.missing;
        var mask: u32 = 0;
        var count: u32 = 0;
        var refs: [8]u32 = undefined;
        var n: u32 = 0;
        if (exists) {
            const old = wk.path[s];
            const om = w.nodes[old] & 0xFF;
            count = w.nodes[old + 1] +% @as(u32, @bitCast(delta));
            var k: u32 = 0;
            var slot: u32 = 0;
            while (k < 8) : (k += 1) {
                const kb = @as(u32, 1) << @intCast(k);
                if (k == ci) {
                    if (child) |c| {
                        refs[n] = c;
                        n += 1;
                        mask |= kb;
                    }
                    if (om & kb != 0) slot += 1;
                } else if (om & kb != 0) {
                    refs[n] = w.nodes[old + 2 + slot];
                    n += 1;
                    slot += 1;
                    mask |= kb;
                }
            }
        } else if (child) |c| {
            mask = cbit;
            refs[0] = c;
            n = 1;
            count = @bitCast(delta);
        }
        if (mask == 0) {
            child = null;
        } else {
            const nn = w.allocNode(2 + n) orelse return null;
            w.nodes[nn] = mask;
            w.nodes[nn + 1] = count;
            var k: u32 = 0;
            while (k < n) : (k += 1) w.nodes[nn + 2 + k] = refs[k];
            child = nn;
        }
        if (s + 1 == sub_level) sub = child;
    }
    return .{ .root = child, .sub = sub };
}

/// Enge Hülle des Teilbaums `node` (Knoten der Stufe s, Kante 2^(s+1)) in
/// seinen eigenen Koordinaten: lo = kleinste belegte Koordinate, hi = größte + 1
/// – genau wie der Bau sie bildet (gbuild), damit der GAS gleich ausfällt.
fn subtreeBox(w: *const Walker, node: u32, s_top: u32) struct { lo: [3]u32, hi: [3]u32 } {
    var lo = [3]u32{ 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF };
    var hi = [3]u32{ 0, 0, 0 };
    const Entry = struct { node: u32, s: u32, off: [3]u32 };
    var stack: [64]Entry = undefined;
    var sp: u32 = 1;
    stack[0] = .{ .node = node, .s = s_top, .off = .{ 0, 0, 0 } };
    while (sp > 0) {
        sp -= 1;
        const e = stack[sp];
        const mask = w.nodes[e.node] & 0xFF;
        const size = @as(u32, 1) << @intCast(e.s);
        var slot: u32 = 0;
        var k: u32 = 0;
        while (k < 8) : (k += 1) {
            if (mask & (@as(u32, 1) << @intCast(k)) == 0) continue;
            const ref = w.nodes[e.node + 2 + slot];
            slot += 1;
            const off = [3]u32{ e.off[0] + (k & 1) * size, e.off[1] + ((k >> 1) & 1) * size, e.off[2] + (k >> 2) * size };
            if (e.s == 2) {
                const brick = w.leaves[ref];
                var b: u32 = 0;
                while (b < 64) : (b += 1) {
                    if ((brick >> @intCast(b)) & 1 == 0) continue;
                    const v = [3]u32{ off[0] + (b & 3), off[1] + ((b >> 2) & 3), off[2] + (b >> 4) };
                    inline for (0..3) |a| {
                        lo[a] = @min(lo[a], v[a]);
                        hi[a] = @max(hi[a], v[a] + 1);
                    }
                }
            } else if (sp < stack.len) {
                stack[sp] = .{ .node = ref, .s = e.s - 1, .off = off };
                sp += 1;
            }
        }
    }
    return .{ .lo = lo, .hi = hi };
}

/// Ein Block je Auftrag
pub fn run(p: *const types.PathEditParams, block: u32, tid: u32) void {
    if (block >= p.count) return;
    const job: *types.PathEditJob = &@as([*]types.PathEditJob, @ptrFromInt(p.jobs))[block];
    const vjob: *volatile types.PathEditJob = job;
    const attrs: [*]u32 = @ptrFromInt(p.attributes);
    // Thread 0 überschreibt jeden Eintrag nach der Baumänderung mit
    // (Operation, Rang, Attribut): daraus bilden danach alle Threads die
    // Attribute in einem Durchgang
    const edits: [*][4]u32 = @ptrFromInt(p.edits);
    const threads = types.path_edit_block;
    const first = job.edit_first;
    const n_edits = job.edit_count;

    var w = Walker{
        .nodes = @as([*]u32, @ptrFromInt(p.nodes)) + job.node_offset,
        .leaves = @as([*]u64, @ptrFromInt(p.leaves)) + job.leaf_offset,
        .node_next = job.node_arena - job.node_offset,
        .node_end = job.node_arena - job.node_offset + job.node_cap,
        .leaf_next = job.leaf_arena - job.leaf_offset,
        .leaf_end = job.leaf_arena - job.leaf_offset + job.leaf_cap,
    };
    const sub_level = job.rt_log2; // Teilbaum der Kante 2^rt_log2 = Knoten der Stufe rt_log2 - 1

    // 0. Palette (Thread 0 hält sie in Registern): neue Werte kommen hinten
    //    dazu; mehr als 16 -> Überlauf, der Chunk wird voll neu gebaut
    var pal: [16]u32 = undefined;
    var n_pal: u32 = 0;
    if (tid == 0 and job.palette != 0) {
        var q: u32 = 0;
        while (q < 16) : (q += 1) pal[q] = attrs[job.pal_src + q];
        // aufgefüllt wird mit dem ersten Wert: bis zu seiner Wiederholung
        n_pal = 1;
        while (n_pal < 16 and pal[n_pal] != pal[0]) n_pal += 1;
    }

    // 1. Baum (Thread 0), je Brick einmal: die Änderungen eines Bricks
    //    werden zusammen in seine Maske geschrieben und der Weg darüber nur
    //    einmal neu. Der Rang jedes Voxels folgt aus dem Brick-Präfix und der
    //    aktuellen Maske, in der Reihenfolge, in der die Attribute sie danach
    //    wieder abspielen; innerhalb eines Voxels bleibt die Reihenfolge.
    //    (Gemessen: der Weg kostet ~16 µs je Änderung, reine Latenz.)
    if (tid == 0) {
        var root = job.root;
        var flags: u32 = 0;
        var count = job.attr_count;
        // Der Host hat die Einträge je Brick zusammengelegt (stabil sortiert):
        // ein Brick ist eine zusammenhängende Folge. Die Operationen stehen
        // danach in derselben Reihenfolge da, in der sie verarbeitet wurden –
        // genau so spielen die Attribute sie wieder ab.
        var e: u32 = 0;
        while (e < n_edits) {
            const lead = edits[first + e];
            const v0 = [3]u32{ lead[0], lead[1], lead[2] };
            // Ende der Folge dieses Bricks
            var end = e + 1;
            while (end < n_edits) : (end += 1) {
                const q = edits[first + end];
                if ((q[0] >> 2) != (v0[0] >> 2) or (q[1] >> 2) != (v0[1] >> 2) or (q[2] >> 2) != (v0[2] >> 2)) break;
            }
            if (flags & (types.path_edit_empty | types.path_edit_overflow) != 0) {
                while (e < end) : (e += 1) edits[first + e] = .{ op_none, 0, edits[first + e][3], 0 };
                continue;
            }
            const wk = walk(&w, root, job.log2_size, v0);
            var brick = wk.brick;
            var delta: i32 = 0;
            while (e < end) : (e += 1) {
                const q = edits[first + e];
                const bit = dag.brickBit(q[0] & 3, q[1] & 3, q[2] & 3);
                const bm = @as(u64, 1) << @intCast(bit);
                const rank = wk.prefix +% @as(u32, @popCount(brick & (bm - 1)));
                const has = brick & bm != 0;
                // Palette: Wert -> Index (neu: anhängen). Voll: diese Änderung
                // entfällt hier, der Neubau (Überlauf) bringt sie
                var val = q[3];
                if (job.palette != 0 and q[3] != 0) {
                    var idx: u32 = 0;
                    while (idx < n_pal and pal[idx] != val) idx += 1;
                    if (idx == n_pal) {
                        if (n_pal == 16) {
                            flags |= types.path_edit_overflow;
                            edits[first + e] = .{ op_none, 0, 0, 0 };
                            continue;
                        }
                        pal[n_pal] = val;
                        n_pal += 1;
                    }
                    val = idx;
                }
                var op: u32 = op_none;
                if (q[3] != 0 and has) op = op_set;
                if (q[3] != 0 and !has) {
                    op = op_insert;
                    brick |= bm;
                    delta += 1;
                }
                if (q[3] == 0 and has) {
                    op = op_remove;
                    brick &= ~bm;
                    delta -= 1;
                }
                // die Attribute spielen die Operationen in Listenreihenfolge ab
                edits[first + e] = .{ op, rank, val, 0 };
            }
            if (brick != wk.brick) {
                const r = rewrite(&w, &wk, job.log2_size, sub_level, v0, brick, delta) orelse {
                    flags |= types.path_edit_overflow;
                    continue;
                };
                if (r.root) |nr| root = nr else flags |= types.path_edit_empty;
                count = @bitCast(@as(i32, @bitCast(count)) + delta);
                updatePrims(&w, job, v0, delta, r.sub, &flags);
            }
        }
        if (w.overflow) flags |= types.path_edit_overflow;
        if (job.palette != 0) {
            var q: u32 = 0;
            while (q < 16) : (q += 1) attrs[job.pal_dst + q] = if (q < n_pal) pal[q] else pal[0];
        }
        job.out_root = root;
        job.out_count = count; // Platz reicht immer: attr_cap = Voxel + Änderungen
        job.out_attr_b = 0;
        job.out_flags = flags;
    }
    barrier();

    // 2. Attribute: jedes Ziel einzeln durch die Operationen zurückverfolgen
    //    (rückwärts: jede bildet einen Index nach der Operation auf einen davor ab)
    const n_final = vjob.out_count;
    // (auch bei Überlauf: die bis dahin geschriebenen Änderungen sind in Baum,
    // Primitiven und Attributen dann gleichermaßen enthalten)
    if (vjob.out_flags & types.path_edit_empty == 0 and p.reserved & 1 == 0 and job.palette != 0) {
        // Palettenformat: ein Thread je Ausgabewort (8 Indizes), keine Konflikte
        const words = (n_final + 7) / 8;
        var wd = tid;
        while (wd < words) : (wd += threads) {
            var word: u32 = 0;
            var j: u32 = 0;
            while (j < 8) : (j += 1) {
                const i = wd * 8 + j;
                if (i >= n_final) break;
                const v = sourceOf(edits, first, n_edits, i);
                const idx = if (v.found) v.val else (attrs[job.attr_src + (v.idx >> 3)] >> @intCast((v.idx & 7) * 4)) & 15;
                word |= idx << @intCast(j * 4);
            }
            attrs[job.attr_a + wd] = word;
        }
    } else if (vjob.out_flags & types.path_edit_empty == 0 and p.reserved & 1 == 0) {
        var i = tid;
        while (i < n_final) : (i += threads) {
            var idx = i;
            var val: u32 = 0;
            var found = false;
            var k = n_edits;
            while (k > 0) {
                k -= 1;
                // gewöhnlich gelesen: nach der Schranke sichtbar, und so im
                // Cache (volatile ging jedes Mal in den Speicher)
                const o = edits[first + k];
                const rank = o[1];
                switch (o[0]) {
                    op_insert => {
                        if (idx == rank) {
                            val = o[2];
                            found = true;
                            break;
                        }
                        if (idx > rank) idx -= 1;
                    },
                    op_remove => {
                        if (idx >= rank) idx += 1;
                    },
                    op_set => {
                        if (idx == rank) {
                            val = o[2];
                            found = true;
                            break;
                        }
                    },
                    else => {},
                }
            }
            attrs[job.attr_a + i] = if (found) val else attrs[job.attr_src + idx];
        }
    }

    // 3. Enge Hüllen aller Primitive neu: ein neuer Voxel kann außerhalb der
    //    alten liegen, die RT-Cores übersprängen ihn sonst (gemessen: fehlende
    //    Schatten auf einem Haufen; die CUDA-Traversierung sah ihn)
    if (job.aabbs != 0 and p.reserved & 2 == 0) {
        const prims: [*]const types.RtPrim = @ptrFromInt(job.prims);
        const boxes: [*][6]f32 = @ptrFromInt(job.aabbs);
        const size = @as(u32, 1) << @intCast(job.rt_log2);
        const eps: f32 = 1.0 / 256.0;
        var k = tid;
        var grew = false;
        while (k < job.prim_count) : (k += threads) {
            const cell = unpackCell(prims[k].cell);
            var b = subtreeBox(&w, prims[k].node, job.rt_log2 - 1);
            if (b.lo[0] > b.hi[0]) b = .{ .lo = .{ 0, 0, 0 }, .hi = .{ 0, 0, 0 } }; // leer
            var box: [6]f32 = undefined;
            inline for (0..3) |a| {
                const base: f32 = @floatFromInt(cell[a] * size);
                box[a] = base + @as(f32, @floatFromInt(b.lo[a])) - eps;
                box[3 + a] = base + @as(f32, @floatFromInt(b.hi[a])) + eps;
            }
            if (job.aabbs_keep == 0) {
                boxes[k] = box;
                grew = true;
            } else {
                // Liegt die neue Hülle in der alten, bleibt alles (auch beim
                // Entfernen: zu groß ist nur langsamer, nie falsch). Sonst auf
                // die ganze Zelle: weitere Änderungen darin brauchen dann
                // keinen neuen GAS mehr.
                const old = boxes[k];
                var inside = true;
                inline for (0..3) |a| {
                    if (box[a] < old[a] or box[3 + a] > old[3 + a]) inside = false;
                }
                if (!inside) {
                    inline for (0..3) |a| {
                        const base: f32 = @floatFromInt(cell[a] * size);
                        boxes[k][a] = base - eps;
                        boxes[k][3 + a] = base + @as(f32, @floatFromInt(size)) + eps;
                    }
                    grew = true;
                }
            }
        }
        if (grew) _ = @atomicRmw(u32, &job.out_flags, .Or, types.path_edit_grew, .monotonic);
    }
}

/// Herkunft des Werts an Stelle `i` nach allen Operationen: rückwärts durch
/// die Liste, jede Operation bildet einen Index nach ihr auf einen davor ab
fn sourceOf(edits: [*]const [4]u32, first: u32, n_edits: u32, i: u32) struct { found: bool, val: u32, idx: u32 } {
    var idx = i;
    var k = n_edits;
    while (k > 0) {
        k -= 1;
        const o = edits[first + k];
        const rank = o[1];
        switch (o[0]) {
            op_insert => {
                if (idx == rank) return .{ .found = true, .val = o[2], .idx = 0 };
                if (idx > rank) idx -= 1;
            },
            op_remove => {
                if (idx >= rank) idx += 1;
            },
            op_set => {
                if (idx == rank) return .{ .found = true, .val = o[2], .idx = 0 };
            },
            else => {},
        }
    }
    return .{ .found = false, .val = 0, .idx = idx };
}

/// RT-Primitive nach der Änderung eines Bricks (bei v) nachziehen (Thread 0):
/// der Teilbaum, der v enthält, bekommt seinen neuen Knoten; spätere
/// Teilbäume verschieben ihren Attributrang um die Änderung der Voxelzahl.
fn updatePrims(w: *Walker, job: *const types.PathEditJob, v: [3]u32, delta: i32, sub: ?u32, flags: *u32) void {
    if (job.prims == 0) return;
    const cell = [3]u32{ v[0] >> @intCast(job.rt_log2), v[1] >> @intCast(job.rt_log2), v[2] >> @intCast(job.rt_log2) };
    const mc = morton(cell[0], cell[1], cell[2]);
    const prims: [*]types.RtPrim = @ptrFromInt(job.prims);
    const d: u32 = @bitCast(delta);
    var found = false;
    var k: u32 = 0;
    while (k < job.prim_count) : (k += 1) {
        const pc = unpackCell(prims[k].cell);
        const mp = morton(pc[0], pc[1], pc[2]);
        if (mp == mc) {
            found = true;
            // leerer Teilbaum: auf einen leeren Knoten zeigen
            prims[k].node = sub orelse blk: {
                const en = w.allocNode(2) orelse {
                    flags.* |= types.path_edit_overflow;
                    break :blk prims[k].node;
                };
                w.nodes[en] = 0;
                w.nodes[en + 1] = 0;
                break :blk en;
            };
        } else if (mp > mc) {
            prims[k].attr_base +%= d;
        }
    }
    // ein Teilbaum ohne Primitiv hat jetzt Voxel: dafür braucht es einen neuen GAS
    if (!found and sub != null) flags.* |= types.path_edit_rebuild;
}
