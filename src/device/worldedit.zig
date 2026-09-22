//! Änderungen an einer gestreamten Welt auf die frisch erzeugten Voxel anwenden.
//!
//! Der Host führt die Änderungen als Überlagerung in Grundvoxel-Koordinaten
//! (world.zig). Vor dem DAG-Bau bekommt jeder Chunk des Auftrags die Liste der
//! Änderungen, die in ihm liegen, bereits in seine LOD-Auflösung umgerechnet.
//! Drei Durchgänge, weil die Voxelliste unsortiert und dicht gepackt ist:
//!
//!   1. apply:   je Voxel die Änderungsliste des Chunks durchsehen. Treffer mit
//!               Attribut != 0 überschreibt das Attribut, Attribut 0 markiert
//!               den Voxel als entfernt. Der Treffer vermerkt die Änderung als
//!               verbraucht, damit sie nicht noch einmal angehängt wird.
//!   2. compact: die überlebenden Voxel dicht in den Zielpuffer schreiben; die
//!               Zähler des Ziels entstehen dabei neu, also bleiben die
//!               Präfixsummen für den Bau exakt.
//!   3. append:  unverbrauchte Änderungen mit Attribut != 0 anhängen – das sind
//!               die Voxel, die es im erzeugten Gelände noch nicht gab.
//!
//! Chunk c belegt im Voxelpuffer [c · capacity, c · capacity + counts[c]).

const types = @import("types.zig");

/// markiert einen entfernten Voxel (x kann sonst nie so groß werden)
pub const removed: u32 = 0xFFFF_FFFF;

/// Durchgang 1: Änderungen auf vorhandene Voxel anwenden
pub fn apply(p: *const types.WorldEditParams, i: u64) void {
    const c: u32 = @intCast(i / p.capacity);
    if (c >= p.count) return;
    const counts: [*]const u32 = @ptrFromInt(p.counts);
    const local: u32 = @intCast(i % p.capacity);
    if (local >= counts[c]) return;

    const voxels: [*][4]u32 = @ptrFromInt(p.voxels);
    const v = &voxels[i];
    if (v[0] == removed) return;

    const offsets: [*]const u32 = @ptrFromInt(p.edit_offsets);
    const entries: [*]const [4]u32 = @ptrFromInt(p.edits);
    const used: [*]u32 = @ptrFromInt(p.edit_used);
    var e = offsets[c];
    const end = offsets[c + 1];
    while (e < end) : (e += 1) {
        const q = entries[e];
        if (q[0] == v[0] and q[1] == v[1] and q[2] == v[2]) {
            // Vermerk zuerst: auch das Entfernen verbraucht die Änderung
            @atomicStore(u32, &used[e], 1, .monotonic);
            if (q[3] == 0) v[0] = removed else v[3] = q[3];
        }
    }
}

/// Durchgang 2: überlebende Voxel dicht in den Zielpuffer
pub fn compact(p: *const types.WorldEditParams, i: u64) void {
    const c: u32 = @intCast(i / p.capacity);
    if (c >= p.count) return;
    const counts: [*]const u32 = @ptrFromInt(p.counts);
    const local: u32 = @intCast(i % p.capacity);
    if (local >= counts[c]) return;

    const voxels: [*]const [4]u32 = @ptrFromInt(p.voxels);
    const v = voxels[i];
    if (v[0] == removed) return;
    const out: [*][4]u32 = @ptrFromInt(p.out_voxels);
    const out_counts: [*]u32 = @ptrFromInt(p.out_counts);
    const k = @atomicRmw(u32, &out_counts[c], .Add, 1, .monotonic);
    if (k < p.capacity) out[@as(u64, c) * p.capacity + k] = v;
}

/// Durchgang 3: neue Voxel anhängen, die im Gelände noch nicht vorkamen
pub fn append(p: *const types.WorldEditParams, i: u64) void {
    const offsets: [*]const u32 = @ptrFromInt(p.edit_offsets);
    if (i >= offsets[p.count]) return;
    const used: [*]const u32 = @ptrFromInt(p.edit_used);
    if (used[i] != 0) return;
    const entries: [*]const [4]u32 = @ptrFromInt(p.edits);
    const q = entries[i];
    if (q[3] == 0) return; // Entfernen ohne Ziel: nichts zu tun

    // zugehörigen Chunk suchen (die Listen sind kurz, binäre Suche genügt)
    var lo: u32 = 0;
    var hi: u32 = p.count;
    while (lo + 1 < hi) {
        const mid = (lo + hi) / 2;
        if (offsets[mid] <= i) lo = mid else hi = mid;
    }
    const out: [*][4]u32 = @ptrFromInt(p.out_voxels);
    const out_counts: [*]u32 = @ptrFromInt(p.out_counts);
    const k = @atomicRmw(u32, &out_counts[lo], .Add, 1, .monotonic);
    if (k < p.capacity) out[@as(u64, lo) * p.capacity + k] = q;
}
