//! Minecraft-artiges Gelände der Demo. Läuft als Kernel auf der GPU
//! (demo/kernels.zig) und unverändert auf der CPU (Kamerahöhe, Tests).
//!
//! Ein Block ist ein Grundvoxel. Die Höhe hängt nur von der Weltposition ab,
//! damit alle LOD-Stufen zusammenpassen; jede Stufe tastet in ihrem eigenen
//! Voxelabstand ab und lässt Oktaven weg, die sie nicht mehr darstellen kann.
//!
//! Aufbau wie bei Minecraft seit 1.18, stark vereinfacht:
//!   Kontinentalität  – Ozean, Küste, Binnenland
//!   Erosion          – flache Ebenen gegen zerklüftetes Bergland
//!   Grate            – Gipfel und Täler im Bergland
//!   Temperatur/Feuchte – Biome: Ebene, Wald, Birkenwald, Taiga, Wüste,
//!                        Schneeland, Strand, Ozean, Gebirge
//! Bäume stehen auf einem 8er-Raster, je Zelle höchstens einer und ganz in
//! seiner Zelle. Damit sieht jede Spalte alle Bäume, die sie berühren, ohne
//! Nachbarn abzufragen – auch über Chunkgrenzen hinweg.

const pyr = @import("pyrit_device");
const types = pyr.types;
const fm = pyr.fmath;

// ---------------------------------------------------------------------------
// Blöcke. Attribut = Farbe (RGB) | Material. Die Farbe moduliert die Textur
// des Materials; mehrere Blöcke teilen sich so ein Material.
// ---------------------------------------------------------------------------

pub const mat_water: u32 = 2;
pub const mat_grass: u32 = 1;
pub const mat_stone: u32 = 3;
pub const mat_sand: u32 = 4;
pub const mat_snow: u32 = 5;
pub const mat_wood: u32 = 6;
pub const mat_leaves: u32 = 7;
pub const mat_dirt: u32 = 8;
pub const mat_gravel: u32 = 9;

fn rgb(m: u32, r: u32, g: u32, b: u32) u32 {
    return (r << 24) | (g << 16) | (b << 8) | m;
}

/// Farben als Albedo (sRGB): Pflanzen reflektieren im Sichtbaren wenig –
/// Gras um 0,1 im Grün. Hellere Werte sahen nach Neon aus und ließen die
/// Belichtungsautomatik den Himmel abdunkeln.
pub const Block = enum(u8) {
    grass,
    dry_grass,
    /// Nadelwaldboden: dunkler und kühler als die Wiese
    taiga_grass,
    snowy_grass,
    dirt,
    stone,
    sand,
    sandstone,
    gravel,
    snow,
    water,
    oak_log,
    birch_log,
    spruce_log,
    oak_leaves,
    birch_leaves,
    spruce_leaves,
    cactus,

    pub fn attribute(b: Block) u32 {
        return switch (b) {
            .grass => rgb(mat_grass, 66, 94, 38),
            .dry_grass => rgb(mat_grass, 84, 98, 46),
            .taiga_grass => rgb(mat_grass, 56, 74, 42),
            .snowy_grass => rgb(mat_snow, 184, 190, 202),
            .dirt => rgb(mat_dirt, 122, 90, 62),
            .stone => rgb(mat_stone, 125, 125, 125),
            .sand => rgb(mat_sand, 184, 170, 126),
            .sandstone => rgb(mat_sand, 170, 154, 108),
            .gravel => rgb(mat_gravel, 128, 122, 118),
            .snow => rgb(mat_snow, 188, 194, 206),
            // Wasser: die Farbe ist zugleich der Absorptionston (Farbe hoch
            // Dichte · Strecke). Mit Dichte 0,08 ergibt (30, 170, 210) etwa
            // 0,35 / 0,07 / 0,04 je Block für Rot/Grün/Blau – klares
            // Meerwasser: Rot ist nach wenigen Blöcken fort, Blaugrün reicht
            // gut 20 Blöcke weit.
            .water => rgb(mat_water, 30, 170, 210),
            .oak_log => rgb(mat_wood, 100, 86, 66),
            .birch_log => rgb(mat_wood, 214, 210, 196),
            .spruce_log => rgb(mat_wood, 62, 52, 42),
            .oak_leaves => rgb(mat_leaves, 54, 92, 34),
            .birch_leaves => rgb(mat_leaves, 74, 100, 46),
            .spruce_leaves => rgb(mat_leaves, 34, 60, 36),
            .cactus => rgb(mat_leaves, 60, 104, 44),
        };
    }
};

pub const Params = extern struct {
    seed: u32 = 1,
    /// Meeresspiegel in Blöcken. 64 statt 63 wie in Minecraft: die
    /// Wasseroberfläche muss auf *jeder* LOD-Stufe auf einer Zellgrenze
    /// liegen, sonst springt sie beim Stufenwechsel um bis zu eine Zelle – das
    /// zeigte sich als gerade Kanten und Stufen mitten auf dem Meer.
    sea_level: f32 = 64,
    /// Bäume je Rasterzelle im Wald (0..1); Ebenen haben ein Zehntel davon
    tree_density: f32 = 0.55,
    /// 0 = kein Wasser (Tests: dann ist die Oberfläche genau das Höhenfeld)
    water: u32 = 1,
};

/// Senkrechter Bereich der Welt (Minecraft: -64 .. 320)
pub const y_min: i32 = -64;
pub const y_max: i32 = 320;

// ---------------------------------------------------------------------------
// Rauschen
// ---------------------------------------------------------------------------

inline fn hash(x: i32, z: i32, seed: u32) u32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 0x8da6b343 +% @as(u32, @bitCast(z)) *% 0xd8163841 +% seed *% 0xcb1ab31f;
    h ^= h >> 15;
    h *%= 0x2c1b3c6d;
    h ^= h >> 12;
    h *%= 0x297a2d39;
    h ^= h >> 15;
    return h;
}

inline fn rnd(x: i32, z: i32, seed: u32) f32 {
    return @as(f32, @floatFromInt(hash(x, z, seed) >> 8)) * (1.0 / 16777216.0);
}

/// Wertrauschen in [-1, 1], C2-stetig interpoliert (keine Knicke an den
/// Gitterlinien, die man sonst als Stufenmuster im Gelände sähe)
fn noise(x: f32, z: f32, seed: u32) f32 {
    const fx = @floor(x);
    const fz = @floor(z);
    const ix: i32 = @intFromFloat(fx);
    const iz: i32 = @intFromFloat(fz);
    const tx = x - fx;
    const tz = z - fz;
    const sx = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
    const sz = tz * tz * tz * (tz * (tz * 6 - 15) + 10);
    const a = rnd(ix, iz, seed);
    const b = rnd(ix + 1, iz, seed);
    const c = rnd(ix, iz + 1, seed);
    const d = rnd(ix + 1, iz + 1, seed);
    const top = a + (b - a) * sx;
    const bot = c + (d - c) * sx;
    return (top + (bot - top) * sz) * 2 - 1;
}

/// Fraktales Rauschen in etwa [-1, 1]. Oktaven unter `min_wl` fallen weg und
/// werden durch ihren Erwartungswert (0) ersetzt – die Höhe bleibt dadurch
/// über die Stufen stabil, nur das Detail verschwindet.
fn fbm(x: f32, z: f32, wavelength: f32, octaves: u32, seed: u32, min_wl: f32) f32 {
    var wl = wavelength;
    var amp: f32 = 1;
    var sum: f32 = 0;
    var norm: f32 = 0;
    var o: u32 = 0;
    while (o < octaves) : (o += 1) {
        norm += amp;
        if (wl >= min_wl) {
            // jede Oktave gedreht, sonst fallen alle auf dieselben Gitterachsen
            const ca = fm.cos(0.9 * @as(f32, @floatFromInt(o)));
            const sa = fm.sin(0.9 * @as(f32, @floatFromInt(o)));
            sum += amp * noise((x * ca - z * sa) / wl + @as(f32, @floatFromInt(o)) * 17.3, (x * sa + z * ca) / wl, seed +% o *% 0x9e3779b9);
        }
        amp *= 0.5;
        wl *= 0.5;
    }
    return sum / norm;
}

inline fn smoothstep(a: f32, b: f32, x: f32) f32 {
    const t = @min(@max((x - a) / (b - a), 0), 1);
    return t * t * (3 - 2 * t);
}

inline fn mix(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// Klima und Höhe
// ---------------------------------------------------------------------------

pub const Climate = struct {
    /// Kontinentalität: < 0 Ozean, um 0 Küste, > 0 Binnenland
    cont: f32,
    /// Erosion: hoch = flach, niedrig = zerklüftet
    erosion: f32,
    temperature: f32,
    humidity: f32,
};

pub fn climate(p: *const Params, x: f32, z: f32) Climate {
    // leichtes Verbiegen der Abtastung: Küsten und Biomgrenzen werden krumm
    const wx = x + 180 * noise(x / 900 + 3.1, z / 900 - 1.7, p.seed ^ 0x1234);
    const wz = z + 180 * noise(x / 900 - 7.9, z / 900 + 4.4, p.seed ^ 0x4321);
    return .{
        .cont = fbm(wx, wz, 2600, 4, p.seed ^ 0xc0, 0) * 1.6 + 0.12,
        .erosion = fbm(wx, wz, 1400, 3, p.seed ^ 0xe1, 0) * 1.5,
        .temperature = fbm(wx, wz, 2200, 2, p.seed ^ 0x7e, 0) * 1.7,
        .humidity = fbm(wx, wz, 1700, 2, p.seed ^ 0x4d, 0) * 1.7,
    };
}

/// Geländehöhe in Blöcken (Oberkante des obersten festen Blocks + 1 in etwa).
/// `min_wl`: kürzeste Wellenlänge, die die Stufe noch darstellt.
pub fn heightWith(p: *const Params, cl: Climate, x: f32, z: f32, min_wl: f32) f32 {
    const sea = p.sea_level;
    // Grundform aus der Kontinentalität: Tiefsee, Schelf, Küste, Binnenland
    var base: f32 = undefined;
    if (cl.cont < -0.45) {
        base = mix(sea - 34, sea - 22, smoothstep(-1.2, -0.45, cl.cont));
    } else if (cl.cont < -0.08) {
        base = mix(sea - 22, sea - 4, smoothstep(-0.45, -0.08, cl.cont));
    } else if (cl.cont < 0.08) {
        base = mix(sea - 4, sea + 3, smoothstep(-0.08, 0.08, cl.cont));
    } else {
        base = mix(sea + 3, sea + 22, smoothstep(0.08, 0.9, cl.cont));
    }
    const inland = smoothstep(0.0, 0.35, cl.cont);

    // Hügel: überall im Binnenland, flacher, wo die Erosion hoch ist
    const flat = smoothstep(-0.2, 0.7, cl.erosion);
    const hills = fbm(x, z, 320, 4, p.seed ^ 0x9a, min_wl) * mix(22, 5, flat);

    // Gebirge: nur bei niedriger Erosion. Grate aus |Rauschen|, geschärft,
    // mit einer zweiten, feineren Gratlage für Kämme und Rinnen.
    const mountain = inland * (1 - smoothstep(-0.6, 0.05, cl.erosion));
    var peaks: f32 = 0;
    if (mountain > 0.001) {
        const r1 = 1 - @abs(fbm(x + 311, z - 97, 900, 3, p.seed ^ 0x5b, min_wl));
        const r2 = 1 - @abs(fbm(x - 57, z + 233, 260, 3, p.seed ^ 0x6c, min_wl));
        peaks = (r1 * r1 * r1 * 150 + r2 * r2 * 28) * mountain;
    }

    // Blockdetail: wenige Blöcke, damit die Oberfläche nicht wie gedrechselt
    // wirkt. Die feinste Lage (12 Blöcke) bricht an Hängen das gleichmäßige
    // Treppenmuster auf, das ein glatter Hang in Blöcken sonst ergibt.
    const detail = fbm(x, z, 48, 3, p.seed ^ 0x3f, min_wl) * 3.0 * (0.4 + 0.6 * inland) +
        fbm(x, z, 12, 2, p.seed ^ 0x4e, min_wl) * 1.3 * inland;
    return base + hills * mix(0.35, 1, inland) + peaks + detail;
}

pub fn height(p: *const Params, x: f32, z: f32) f32 {
    return heightWith(p, climate(p, x, z), x, z, 0);
}

pub const Biome = enum { ocean, beach, plains, forest, birch_forest, taiga, snowy, desert, mountains };

pub fn biome(p: *const Params, cl: Climate, h: f32, slope: f32) Biome {
    if (h < p.sea_level - 1) return .ocean;
    if (h < p.sea_level + 2.5 and cl.cont < 0.2 and slope < 0.9) {
        return if (cl.temperature < -0.55) .snowy else .beach;
    }
    if (h > p.sea_level + 95) return .mountains;
    if (cl.temperature > 0.45 and cl.humidity < 0.05) return .desert;
    if (cl.temperature < -0.55) return .snowy;
    if (cl.temperature < -0.15) return .taiga;
    if (cl.humidity > 0.3) return if (cl.temperature > 0.2) .forest else .birch_forest;
    if (cl.humidity > -0.05) return .forest;
    return .plains;
}

/// Oberflächenblock und Block darunter
fn surfaceBlocks(p: *const Params, b: Biome, h: f32, slope: f32, x: f32, z: f32) [2]Block {
    if (slope > 1.25) return .{ .stone, .stone };
    return switch (b) {
        .ocean => if (h < p.sea_level - 12) .{ .gravel, .gravel } else .{ .sand, .sand },
        .beach => .{ .sand, .sandstone },
        .desert => .{ .sand, .sandstone },
        .snowy => .{ .snowy_grass, .dirt },
        .mountains => blk: {
            // Schnee oben, von Rauschen aufgelöst; steile Hänge bleiben Fels
            const line = p.sea_level + 125 + 14 * noise(x / 40, z / 40, p.seed ^ 0x51);
            if (h > line and slope < 0.9) break :blk .{ .snow, .stone };
            if (slope > 0.7 or h > p.sea_level + 110) break :blk .{ .stone, .stone };
            break :blk .{ .grass, .dirt };
        },
        .taiga => if (h > p.sea_level + 70) .{ .snowy_grass, .dirt } else .{ .taiga_grass, .dirt },
        .plains => if (noise(x / 30, z / 30, p.seed ^ 0x77) > 0.55) .{ .dry_grass, .dirt } else .{ .grass, .dirt },
        else => .{ .grass, .dirt },
    };
}

/// Farbe eines Attributs skalieren (Material bleibt). `warm` verschiebt zu
/// Gelb hin (trockene Stellen).
fn tint(a: u32, f: f32, warm: f32) u32 {
    const r0: f32 = @floatFromInt((a >> 24) & 0xFF);
    const g0: f32 = @floatFromInt((a >> 16) & 0xFF);
    const b0: f32 = @floatFromInt((a >> 8) & 0xFF);
    const r: u32 = @intFromFloat(@min(@max(r0 * (f + warm * 0.30), 0), 255));
    const g: u32 = @intFromFloat(@min(@max(g0 * (f + warm * 0.10), 0), 255));
    const b: u32 = @intFromFloat(@min(@max(b0 * (f - warm * 0.20), 0), 255));
    return (r << 24) | (g << 16) | (b << 8) | (a & 0xFF);
}

/// Farbschwankung des Bodens je Spalte: große Flecken (feuchter, trockener)
/// und eine feine Sprenkelung. Ohne sie ist eine Wiese exakt eine Farbe und
/// wirkt aus jeder Entfernung wie ein Teppich.
fn groundTint(p: *const Params, x: f32, z: f32, fine: bool) [2]f32 {
    const coarse = noise(x / 150 + 3.1, z / 150 - 7.4, p.seed ^ 0x3c19);
    const mid = noise(x / 23 - 1.7, z / 23 + 9.2, p.seed ^ 0xb72d);
    const speck: f32 = if (fine) rnd(@intFromFloat(@floor(x)), @intFromFloat(@floor(z)), p.seed ^ 0x5eed) * 2 - 1 else 0;
    const f = 1 + coarse * 0.16 + mid * 0.10 + speck * 0.06;
    return .{ f, @max(coarse - 0.2, 0) * 0.5 };
}

// ---------------------------------------------------------------------------
// Bäume
// ---------------------------------------------------------------------------

pub const tree_cell: i32 = 8;

const TreeKind = enum { none, oak, birch, spruce, cactus };

const Tree = struct {
    kind: TreeKind,
    /// Stamm (Block) und Boden darunter (erster Luftblock)
    x: i32,
    z: i32,
    ground: i32,
    trunk: i32,
    radius: i32,
};

/// Der Baum der Rasterzelle (cx, cz), falls einer steht
fn treeIn(p: *const Params, cx: i32, cz: i32) Tree {
    const none = Tree{ .kind = .none, .x = 0, .z = 0, .ground = 0, .trunk = 0, .radius = 0 };
    const r = rnd(cx, cz, p.seed ^ 0x7ee);
    // Art zuerst aus dem Klima der Zellmitte: sie bestimmt den Kronenradius
    // und damit, wie weit der Stamm in der Zelle wandern darf – die Krone
    // muss ganz in der Zelle bleiben. Laubbäume (Radius 2) haben so vier
    // Positionen je Achse, das bricht das Raster auf.
    const mx: f32 = @floatFromInt(cx * tree_cell + 4);
    const mz: f32 = @floatFromInt(cz * tree_cell + 4);
    const cl = climate(p, mx, mz);
    const coniferous = cl.temperature < -0.15 or heightWith(p, cl, mx, mz, 32) > p.sea_level + 95;
    const span: f32 = if (coniferous) 1.99 else 3.99;
    const base: i32 = if (coniferous) 3 else 2;
    const tx = cx * tree_cell + base + @as(i32, @intFromFloat(rnd(cx, cz, p.seed ^ 0x111) * span));
    const tz = cz * tree_cell + base + @as(i32, @intFromFloat(rnd(cx, cz, p.seed ^ 0x222) * span));
    const fx: f32 = @floatFromInt(tx);
    const fz: f32 = @floatFromInt(tz);
    const h = heightWith(p, cl, fx + 0.5, fz + 0.5, 0);
    if (h < p.sea_level + 1) return none;
    const hx = heightWith(p, cl, fx + 2.5, fz + 0.5, 0);
    const hz = heightWith(p, cl, fx + 0.5, fz + 2.5, 0);
    const slope = @max(@abs(hx - h), @abs(hz - h)) / 2;
    if (slope > 0.9) return none;
    const b = biome(p, cl, h, slope);
    // Wald wächst in Gruppen mit Lichtungen dazwischen, nicht gleichmäßig
    const clump = smoothstep(-0.35, 0.45, fbm(fx, fz, 90, 2, p.seed ^ 0xf0, 0));
    const dens = p.tree_density * (0.25 + 1.1 * clump);
    const want: f32 = switch (b) {
        .forest, .birch_forest, .taiga => dens,
        .plains => dens * 0.08,
        .snowy => dens * 0.12,
        .mountains => if (h < p.sea_level + 115) dens * 0.25 else 0,
        .desert => dens * 0.06,
        else => 0,
    };
    if (r >= want) return none;
    const r2 = rnd(cx, cz, p.seed ^ 0x333);
    const kind: TreeKind = switch (b) {
        .taiga, .snowy, .mountains => if (coniferous) .spruce else .oak,
        .birch_forest => if (r2 < 0.7) .birch else .oak,
        .desert => .cactus,
        else => if (r2 < 0.2) .birch else .oak,
    };
    const ground: i32 = @intFromFloat(@floor(h));
    return switch (kind) {
        .oak => .{ .kind = kind, .x = tx, .z = tz, .ground = ground, .trunk = 4 + @as(i32, @intFromFloat(r2 * 2.99)), .radius = 2 },
        .birch => .{ .kind = kind, .x = tx, .z = tz, .ground = ground, .trunk = 5 + @as(i32, @intFromFloat(r2 * 2.99)), .radius = 2 },
        // Fichten: schlanke junge (Radius 2) und ausladende alte (Radius 3)
        .spruce => .{ .kind = kind, .x = tx, .z = tz, .ground = ground, .trunk = 5 + @as(i32, @intFromFloat(r2 * 5.99)), .radius = if (r2 < 0.35) 2 else 3 },
        .cactus => .{ .kind = kind, .x = tx, .z = tz, .ground = ground, .trunk = 1 + @as(i32, @intFromFloat(r2 * 2.99)), .radius = 0 },
        .none => none,
    };
}

/// Block des Baums an (x, y, z), sonst null. Formen wie in Minecraft.
fn treeBlock(p: *const Params, t: Tree, x: i32, y: i32, z: i32) ?Block {
    const dx = x - t.x;
    const dz = z - t.z;
    const top = t.ground + t.trunk; // oberster Stammblock
    const ax: i32 = @intCast(@abs(dx));
    const az: i32 = @intCast(@abs(dz));
    switch (t.kind) {
        .none => return null,
        .cactus => return if (dx == 0 and dz == 0 and y > t.ground and y <= top) .cactus else null,
        .oak, .birch => {
            if (dx == 0 and dz == 0 and y > t.ground and y <= top) return if (t.kind == .oak) .oak_log else .birch_log;
            const leaves: Block = if (t.kind == .oak) .oak_leaves else .birch_leaves;
            const k = y - top;
            // zwei breite Lagen (5x5, Ecken zufällig), darüber zwei schmale
            // (3x3 / Kreuz). Hohe Bäume bekommen eine dritte breite Lage.
            const deep = t.trunk >= 6;
            if (k == -2 or k == -1 or (deep and k == -3)) {
                if (ax > 2 or az > 2) return null;
                if (ax == 2 and az == 2 and rnd(x * 3 + y, z * 5 - y, p.seed ^ 0x999) < 0.6) return null;
                // Ränder ausdünnen: eine Krone ist kein voller Würfel, durch
                // Lücken fällt Licht und die Silhouette wird unregelmäßig
                if ((ax == 2 or az == 2) and rnd(x * 7 - y, z * 3 + y * 5, p.seed ^ 0x9a1) < 0.22) return null;
                return leaves;
            }
            if (k == 0) return if (ax <= 1 and az <= 1 and !(ax == 1 and az == 1 and rnd(x, z + y, p.seed ^ 0x998) < 0.5)) leaves else null;
            if (k == 1) return if (ax + az <= 1) leaves else null;
            return null;
        },
        .spruce => {
            if (dx == 0 and dz == 0 and y > t.ground and y <= top) return .spruce_log;
            const k = top - y; // Blöcke unter der Spitze
            if (y == top + 1) return if (dx == 0 and dz == 0) .spruce_leaves else null;
            if (k < 0 or y <= t.ground + 2) return null;
            // Lagen abwechselnd breit und schmal, nach unten breiter werdend
            const rad: i32 = @min(if (@mod(k, 2) == 0) @divTrunc(k, 3) + 1 else @divTrunc(k, 3), t.radius);
            if (ax + az > rad + @divTrunc(rad, 2) or ax > rad or az > rad) return null;
            if (rad > 1 and (ax == rad or az == rad) and rnd(x * 5 + y, z * 7 - y * 3, p.seed ^ 0x9a2) < 0.2) return null;
            return .spruce_leaves;
        },
    }
}

// ---------------------------------------------------------------------------
// Generator: ein Thread je Spalte (Chunk c, x, z)
// ---------------------------------------------------------------------------

fn emit(g: *const types.WorldGenParams, c: u32, x: u32, y: i32, z: u32, attr: u32) void {
    const counts: [*]u32 = @ptrFromInt(g.counts);
    const out: [*][4]u32 = @ptrFromInt(g.voxels);
    const k = @atomicRmw(u32, &counts[c], .Add, 1, .monotonic);
    if (k < g.capacity) out[@as(u64, c) * g.capacity + k] = .{ x, @intCast(y), z, attr };
}

pub fn column(g: *const types.WorldGenParams, p: *const Params, i: u32) void {
    const cl: u5 = @intCast(g.chunk_log2);
    const n: u32 = @as(u32, 1) << cl;
    const c = i >> (2 * cl);
    if (c >= g.count) return;
    const col = i & ((n * n) - 1);
    const lx = col & (n - 1);
    const lz = col >> cl;
    const key = @as([*]const types.ChunkKey, @ptrFromInt(g.chunks))[c];
    const step_i: i32 = @as(i32, 1) << @intCast(key.lod);
    const step: f32 = @floatFromInt(step_i);
    const ni: i32 = @intCast(n);

    // Blockkoordinaten der Spalte (Ecke) und ihre Mitte
    const bx = (key.x * ni + @as(i32, @intCast(lx))) * step_i;
    const bz = (key.z * ni + @as(i32, @intCast(lz))) * step_i;
    const wx = @as(f32, @floatFromInt(bx)) + 0.5 * step;
    const wz = @as(f32, @floatFromInt(bz)) + 0.5 * step;
    const min_wl = 2 * step;

    const clim = climate(p, wx, wz);
    const h = heightWith(p, clim, wx, wz, min_wl);
    const hx0 = heightWith(p, clim, wx - step, wz, min_wl);
    const hx1 = heightWith(p, clim, wx + step, wz, min_wl);
    const hz0 = heightWith(p, clim, wx, wz - step, min_wl);
    const hz1 = heightWith(p, clim, wx, wz + step, min_wl);
    const slope = @max(@abs(hx1 - hx0), @abs(hz1 - hz0)) / (2 * step);
    // Haut bis unter den niedrigsten Nachbarn: an Hängen keine Löcher, und an
    // Chunkrändern, wo verschiedene Stufen aneinanderstoßen, keine Nähte
    // (in einer Mulde liegen alle Nachbarn höher: dann wenigstens die oberste Zelle)
    const low = @min(@min(@min(hx0, hx1), @min(hz0, hz1)) - 2.5 * step, h - step);

    const y0 = key.y * ni * step_i; // unterster Block des Chunks
    const y0f: f32 = @floatFromInt(y0);
    // Zelle ly ist fest, wenn ihre Mitte unter der Höhe liegt
    const top_f = (h - y0f) / step - 0.5;
    const bot_f = (low - y0f) / step - 0.5;

    // Wasseroberfläche unabhängig vom Boden: über tiefem Meer liegt in diesem
    // Chunk vielleicht gar kein Boden, das Wasser aber schon.
    var water_y: i32 = -1;
    if (p.water != 0 and h < p.sea_level) {
        const wf = (p.sea_level - y0f) / step - 0.5;
        if (wf >= 0 and wf < @as(f32, @floatFromInt(n))) water_y = @intFromFloat(@floor(wf));
    }
    if (water_y >= 0) emit(g, c, lx, water_y, lz, Block.water.attribute());

    const b = biome(p, clim, h, slope);
    const sb = surfaceBlocks(p, b, h, slope, wx, wz);
    const has_ground = top_f >= 0 and bot_f < @as(f32, @floatFromInt(n));
    var top: i32 = if (has_ground) @min(@as(i32, @intFromFloat(@floor(top_f))), ni - 1) else -1;
    const bot: i32 = if (has_ground) @max(@as(i32, @intFromFloat(@floor(bot_f))), 0) else 0;
    // Die Zelle der Wasseroberfläche gehört dem Wasser
    if (water_y >= 0 and top >= water_y) top = water_y - 1;
    const surf_y: i32 = @intFromFloat(@floor(top_f)); // oberste feste Zelle (lokal, ungekappt)
    if (has_ground and bot <= top) {
        // Sprenkelung nur auf feinen Stufen: auf groben stünde sie für ganze
        // Blockgruppen und flimmerte beim Stufenwechsel
        // Farbschwankung je Säule: rechnet der Shader aus der Weltposition
        // (Material.variation) – im Attribut machte sie fast jedes Voxel
        // einzigartig, und Attribute sind der größte Posten im Weltspeicher
        var y = top;
        while (y >= bot) : (y -= 1) {
            const depth = @as(f32, @floatFromInt(surf_y - y)) * step; // Blöcke unter der Oberfläche
            const blk: Block = if (depth < 1) sb[0] else if (depth < 4) sb[1] else .stone;
            emit(g, c, lx, y, lz, blk.attribute());
        }
    }

    // --- Bäume ---
    if (step >= 8) {
        // Ein Baum ist hier kleiner als eine Zelle. Die Waldfläche bleibt als
        // Laubschicht über dem Boden sichtbar, anteilig zur Kronendeckung;
        // aus der Ferne ist das der richtige Anblick.
        const ly = surf_y + 1;
        if (ly < 0 or ly >= ni or water_y >= 0) return;
        const cover: f32 = switch (b) {
            .forest, .birch_forest, .taiga => p.tree_density * 0.55,
            .plains, .snowy => p.tree_density * 0.06,
            .mountains => if (h < p.sea_level + 115) p.tree_density * 0.15 else 0,
            else => 0,
        };
        if (cover <= 0.01) return;
        if (rnd(bx, bz, p.seed ^ 0xa1) < cover) {
            const leaf: Block = if (b == .taiga or b == .snowy or b == .mountains) .spruce_leaves else if (b == .birch_forest) .birch_leaves else .oak_leaves;
            emit(g, c, lx, ly, lz, leaf.attribute());
        }
        return;
    }

    // Alle Rasterzellen, die diese Spalte berührt (auf gröberen Stufen deckt
    // eine Spalte mehrere Blöcke ab)
    const cx0 = @divFloor(bx, tree_cell);
    const cz0 = @divFloor(bz, tree_cell);
    const cx1 = @divFloor(bx + step_i - 1, tree_cell);
    const cz1 = @divFloor(bz + step_i - 1, tree_cell);
    var tcz = cz0;
    while (tcz <= cz1) : (tcz += 1) {
        var tcx = cx0;
        while (tcx <= cx1) : (tcx += 1) {
            const t = treeIn(p, tcx, tcz);
            if (t.kind == .none) continue;
            // senkrechter Bereich des Baums in lokalen Zellen dieser Stufe
            const yb = @divFloor(t.ground + 1 - y0, step_i);
            const yt = @divFloor(t.ground + t.trunk + 2 - y0, step_i);
            var ly = @max(yb, 0);
            while (ly <= @min(yt, ni - 1)) : (ly += 1) {
                if (ly <= top and has_ground) continue; // Boden liegt hier
                if (ly == water_y) continue;
                // Zelle ist belegt, wenn irgendein Block des Baums darin liegt
                // (auf Stufe 0 genau ein Block). Laub gewinnt gegen Holz: von
                // außen sieht man die Krone, nicht den Stamm darin – sonst
                // tragen grobe Stufen die Stammspitze als hellen Punkt.
                var found: ?Block = null;
                var leafy = false;
                var sy: i32 = 0;
                while (sy < step_i and !leafy) : (sy += 1) {
                    var sz: i32 = 0;
                    while (sz < step_i and !leafy) : (sz += 1) {
                        var sx: i32 = 0;
                        while (sx < step_i and !leafy) : (sx += 1) {
                            if (treeBlock(p, t, bx + sx, y0 + ly * step_i + sy, bz + sz)) |tb| {
                                leafy = tb == .oak_leaves or tb == .birch_leaves or tb == .spruce_leaves;
                                // Ein Stamm ist einen Block dick: ab Stufe 2
                                // füllte er eine Zelle von 4x4 Blöcken und
                                // stünde als dicker Pfahl unter der Krone.
                                if (leafy or step_i <= 2) found = tb;
                            }
                        }
                    }
                }
                if (found) |fb| {
                    // Grünton je Baum: Material.variation (mittlere Schwankung)
                    emit(g, c, lx, ly, lz, fb.attribute());
                }
            }
        }
    }
}
