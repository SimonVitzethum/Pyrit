//! Shading: Materialien, Sonne, Punktlichter, Emission, Himmel, Schatten,
//! eine indirekte Reflexion (GI) oder Umgebungsverdeckung.
//!
//! Die Strahlverfolgung kommt über `tracer` (CUDA-Traversierung oder
//! RT-Cores); derselbe Code läuft auf CPU und GPU.
//!
//! Konvention: Lambert-Anteil = Albedo · E · (n·l). Die Intensitäten in
//! `Lighting` sind also bereits durch π geteilt; Glanz (GGX) wird passend
//! mit π multipliziert.

const std = @import("std");
const types = @import("types.zig");
const vec = @import("vec.zig");
const tr = @import("trace.zig");
const dag = @import("dag.zig");
const fm = @import("fmath.zig");
const Vec3 = vec.Vec3;
const splat = vec.splat;

const pi: f32 = std.math.pi;

// ---------------------------------------------------------------------------
// Zufallszahlen (PCG-Hash), reproduzierbar pro Pixel und Frame
// ---------------------------------------------------------------------------

pub const Rng = struct {
    state: u32,

    pub fn init(x: u32, y: u32, frame: u32, salt: u32) Rng {
        return .{ .state = hash(x *% 1973 +% hash(y *% 9277 +% hash(frame *% 26699 +% salt))) };
    }

    pub fn hash(v: u32) u32 {
        const state = v *% 747796405 +% 2891336453;
        const word = ((state >> @intCast((state >> 28) + 4)) ^ state) *% 277803737;
        return (word >> 22) ^ word;
    }

    pub fn next(self: *Rng) f32 {
        self.state = hash(self.state);
        return @as(f32, @floatFromInt(self.state >> 8)) * (1.0 / 16777216.0);
    }
};

// ---------------------------------------------------------------------------
// Attribute und Materialien
// ---------------------------------------------------------------------------

/// Farbe eines Attributs (0xRRGGBB in Bits 8..31), sRGB -> linear
pub fn attributeColor(attribute: u32) Vec3 {
    const c = Vec3{
        @floatFromInt((attribute >> 24) & 0xFF),
        @floatFromInt((attribute >> 16) & 0xFF),
        @floatFromInt((attribute >> 8) & 0xFF),
    } * splat(1.0 / 255.0);
    return srgbToLinear(c);
}

pub fn srgbToLinear(c: Vec3) Vec3 {
    // genau genug und schnell: c^2.2
    return .{ fm.pow(@max(c[0], 1e-8), 2.2), fm.pow(@max(c[1], 1e-8), 2.2), fm.pow(@max(c[2], 1e-8), 2.2) };
}

pub const Surface = struct {
    albedo: Vec3,
    emission: Vec3,
    roughness: f32,
    metallic: f32,
    clearcoat: f32 = 0,
    clearcoat_roughness: f32 = 0.1,
    /// Licht wickelt sich um die Kante (Haut, Laub, Wachs)
    subsurface: f32 = 0,
    subsurface_color: Vec3 = .{ 1, 1, 1 },
    /// gestörte Normale; {0,0,0} = die geometrische behalten
    normal: Vec3 = .{ 0, 0, 0 },
};

pub fn surface(s: *const types.Scene, attribute: u32) Surface {
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    const m = &mats[attribute & 0xFF];
    var albedo: Vec3 = m.base_color;
    if (m.flags & types.material_voxel_color != 0) albedo *= attributeColor(attribute);
    return .{
        .albedo = albedo,
        .emission = m.emission,
        .roughness = @max(m.roughness, 0.02),
        .metallic = m.metallic,
        .clearcoat = m.clearcoat,
        .clearcoat_roughness = @max(m.clearcoat_roughness, 0.02),
        .subsurface = m.subsurface,
        .subsurface_color = m.subsurface_color,
    };
}

// ---------------------------------------------------------------------------
// Texturen: von Hand bilinear aus einem dicht gepackten RGBA8-Puffer. Keine
// Texturhardware, damit derselbe Code später auch auf AMD läuft.
// ---------------------------------------------------------------------------

/// Lage und Größe einer Verkleinerungsstufe im Puffer
fn levelInfo(t: *const types.TextureData, lod: u32) struct { off: u64, w: u32, h: u32 } {
    var off: u64 = 0;
    var w = t.width;
    var h = t.height;
    var k: u32 = 0;
    while (k < lod) : (k += 1) {
        off += @as(u64, w) * h;
        w = @max(w / 2, 1);
        h = @max(h / 2, 1);
    }
    return .{ .off = off, .w = w, .h = h };
}

fn texel(t: *const types.TextureData, off: u64, w: u32, h: u32, x: i32, y: i32) Vec3 {
    const cx: u32 = @intCast(@mod(x, @as(i32, @intCast(w))));
    const cy: u32 = @intCast(@mod(y, @as(i32, @intCast(h))));
    const px = @as([*]const [4]u8, @ptrFromInt(t.data))[off + @as(u64, cy) * w + cx];
    const inv = 1.0 / 255.0;
    return .{ @as(f32, @floatFromInt(px[0])) * inv, @as(f32, @floatFromInt(px[1])) * inv, @as(f32, @floatFromInt(px[2])) * inv };
}

fn sampleLevel(t: *const types.TextureData, lod: u32, u: f32, v: f32) Vec3 {
    const li = levelInfo(t, lod);
    const fx = u * @as(f32, @floatFromInt(li.w)) - 0.5;
    const fy = v * @as(f32, @floatFromInt(li.h)) - 0.5;
    const x0: i32 = @intFromFloat(@floor(fx));
    const y0: i32 = @intFromFloat(@floor(fy));
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);
    const a = texel(t, li.off, li.w, li.h, x0, y0);
    const b = texel(t, li.off, li.w, li.h, x0 + 1, y0);
    const c = texel(t, li.off, li.w, li.h, x0, y0 + 1);
    const d = texel(t, li.off, li.w, li.h, x0 + 1, y0 + 1);
    const top = a + (b - a) * splat(tx);
    const bot = c + (d - c) * splat(tx);
    return top + (bot - top) * splat(ty);
}

/// `texels_per_pixel`: wie viele Texel der Grundstufe auf ein Bildschirmpixel
/// fallen. Darüber wird die Verkleinerungsstufe gewählt und zwischen zwei
/// Stufen überblendet – ohne das flimmert jede Textur in der Ferne.
fn sampleTexture(s: *const types.Scene, index: u32, u: f32, v: f32, texels_per_pixel: f32) Vec3 {
    if (index == 0 or index >= s.texture_count) return .{ 1, 1, 1 };
    const tt: [*]const types.TextureData = @ptrFromInt(s.textures);
    const t = &tt[index];
    if (t.data == 0) return .{ 1, 1, 1 };
    const levels = @max(t.levels, 1);
    if (levels == 1 or texels_per_pixel <= 1) return sampleLevel(t, 0, u, v);
    const lod_f = @min(fm.log2(texels_per_pixel), @as(f32, @floatFromInt(levels - 1)));
    const lo: u32 = @intFromFloat(@floor(lod_f));
    const frac = lod_f - @floor(lod_f);
    const a = sampleLevel(t, lo, u, v);
    if (lo + 1 >= levels or frac <= 0) return a;
    const b = sampleLevel(t, lo + 1, u, v);
    return a + (b - a) * splat(frac);
}

/// Flächenparameter eines Voxeltreffers: welche zwei Weltachsen die Fläche
/// aufspannen. Voxelflächen sind achsenparallel, deshalb genügt eine Ebene –
/// Triplanar-Mischen wäre hier reine Verschwendung.
fn faceUv(p: Vec3, n: Vec3, scale: f32) [2]f32 {
    const inv = 1.0 / @max(scale, 1e-4);
    const ax = @abs(n[0]);
    const ay = @abs(n[1]);
    const az = @abs(n[2]);
    if (ax >= ay and ax >= az) return .{ p[2] * inv, p[1] * inv };
    if (ay >= az) return .{ p[0] * inv, p[2] * inv };
    return .{ p[0] * inv, p[1] * inv };
}

/// Zwei Tangenten der Fläche, passend zu faceUv
fn faceTangents(n: Vec3) [2]Vec3 {
    const ax = @abs(n[0]);
    const ay = @abs(n[1]);
    const az = @abs(n[2]);
    if (ax >= ay and ax >= az) return .{ .{ 0, 0, 1 }, .{ 0, 1, 0 } };
    if (ay >= az) return .{ .{ 1, 0, 0 }, .{ 0, 0, 1 } };
    return .{ .{ 1, 0, 0 }, .{ 0, 1, 0 } };
}

fn hash2(x: i32, y: i32) f32 {
    var h: u32 = @as(u32, @bitCast(x)) *% 0x8da6b343 +% @as(u32, @bitCast(y)) *% 0xd8163841;
    h ^= h >> 15;
    h *%= 0x2c1b3c6d;
    h ^= h >> 12;
    return @as(f32, @floatFromInt(h >> 8)) * (1.0 / 16777216.0);
}

fn valueNoise2(x: f32, y: f32) f32 {
    const fx = @floor(x);
    const fy = @floor(y);
    const ix: i32 = @intFromFloat(fx);
    const iy: i32 = @intFromFloat(fy);
    const tx = x - fx;
    const ty = y - fy;
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const a = hash2(ix, iy);
    const b = hash2(ix + 1, iy);
    const c = hash2(ix, iy + 1);
    const d = hash2(ix + 1, iy + 1);
    const top = a + (b - a) * sx;
    const bot = c + (d - c) * sx;
    return top + (bot - top) * sy;
}

/// Oberfläche am Treffer: wie `surface`, zusätzlich mit Textur und
/// Detailnormale. `p` ist der Weltpunkt, `n` die geometrische Normale.
pub fn surfaceAt(s: *const types.Scene, attribute: u32, p: Vec3, n: Vec3, footprint: f32) Surface {
    var sf = surface(s, attribute);
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    const m = &mats[attribute & 0xFF];
    if (m.texture == 0 and m.normal_texture == 0 and m.normal_strength == 0) return sf;

    const scale = if (m.texture_scale > 0) m.texture_scale else 1;
    const uv = faceUv(p, n, scale);
    // Wie viele Texel auf ein Pixel fallen: Kantenlänge einer Kachel in
    // Welteinheiten gegen die Größe eines Pixels an dieser Stelle.
    var tpp: f32 = 0;
    if (footprint > 0 and (m.texture != 0 or m.normal_texture != 0)) {
        const tt: [*]const types.TextureData = @ptrFromInt(s.textures);
        const idx = if (m.texture != 0) m.texture else m.normal_texture;
        if (idx < s.texture_count) tpp = footprint / scale * @as(f32, @floatFromInt(tt[idx].width));
    }
    if (m.texture != 0) sf.albedo *= sampleTexture(s, m.texture, uv[0], uv[1], tpp);

    // Detailnormale: aus der Normalentextur oder erzeugt
    var du: f32 = 0;
    var dv: f32 = 0;
    if (m.normal_texture != 0) {
        const t = sampleTexture(s, m.normal_texture, uv[0], uv[1], tpp);
        du = (t[0] * 2 - 1) * @max(m.normal_strength, 1);
        dv = (t[1] * 2 - 1) * @max(m.normal_strength, 1);
    } else if (m.normal_strength != 0) {
        const ns = if (m.normal_scale > 0) m.normal_scale else 1;
        // Detail ausblenden, sobald mehr als eine Rauschperiode auf ein Pixel
        // fällt: sonst flimmert die Fläche in der Ferne bei jeder Bewegung.
        const fade = if (footprint > 0) @min(@max(ns / @max(2 * footprint, 1e-6), 0), 1) else 1;
        if (fade <= 0.01) return sf;
        const nu = p[0] / ns;
        const nv = p[1] / ns;
        const nw = p[2] / ns;
        const e: f32 = 0.5;
        // Steigung des Rauschens in den beiden Flächenrichtungen
        du = (valueNoise2(nu + e, nv + nw) - valueNoise2(nu - e, nv + nw)) * m.normal_strength * fade;
        dv = (valueNoise2(nu, nv + nw + e) - valueNoise2(nu, nv + nw - e)) * m.normal_strength * fade;
    }
    if (du != 0 or dv != 0) {
        const tg = faceTangents(n);
        sf.normal = vec.normalize(n - tg[0] * splat(du) - tg[1] * splat(dv));
    }
    return sf;
}

// ---------------------------------------------------------------------------
// Himmel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Umgebungskarte (equirektangulär) mit Importance-Sampling
//
// Ohne Karte bleibt der analytische Himmel. Mit Karte wird sie zweifach
// abgetastet – über den Cosinus-Lappen (der GI-Strahl) und über die
// Helligkeitsverteilung der Karte – und beides nach der Potenz-Heuristik
// gewichtet (MIS). Sonst rauscht eine kleine, helle Sonne in der Karte
// hoffnungslos, oder sie wird doppelt gezählt.
// ---------------------------------------------------------------------------

pub inline fn hasEnv(l: *const types.Lighting) bool {
    return l.env_data != 0 and l.env_width > 0 and l.env_height > 0;
}

fn envTexel(l: *const types.Lighting, x: u32, y: u32) Vec3 {
    const t = @as([*]const [4]f32, @ptrFromInt(l.env_data))[@as(u64, y) * l.env_width + x];
    return .{ t[0], t[1], t[2] };
}

/// Richtung -> Strahldichte aus der Karte (bilinear, in u wiederholend)
pub fn envRadiance(l: *const types.Lighting, d: Vec3) Vec3 {
    const dir = vec.normalize(d);
    const phi = fm.atan2(dir[2], dir[0]) - l.env_rotation;
    const theta = fm.acos(@min(@max(dir[1], -1), 1));
    var u = phi * (0.5 / pi);
    u -= @floor(u);
    const v = theta * (1.0 / pi);
    const fx = u * @as(f32, @floatFromInt(l.env_width)) - 0.5;
    const fy = @min(@max(v * @as(f32, @floatFromInt(l.env_height)) - 0.5, 0), @as(f32, @floatFromInt(l.env_height - 1)));
    const x0: i32 = @intFromFloat(@floor(fx));
    const y0: u32 = @intFromFloat(@floor(fy));
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);
    const w: i32 = @intCast(l.env_width);
    const xa: u32 = @intCast(@mod(x0, w));
    const xb: u32 = @intCast(@mod(x0 + 1, w));
    const ya = y0;
    const yb = @min(y0 + 1, l.env_height - 1);
    const a = envTexel(l, xa, ya);
    const b = envTexel(l, xb, ya);
    const c = envTexel(l, xa, yb);
    const e = envTexel(l, xb, yb);
    const top = a + (b - a) * splat(tx);
    const bot = c + (e - c) * splat(tx);
    return (top + (bot - top) * splat(ty)) * splat(l.env_intensity);
}

/// Wahrscheinlichkeitsdichte, mit der envSample diese Richtung liefert
/// (bezogen auf den Raumwinkel)
pub fn envPdf(l: *const types.Lighting, d: Vec3) f32 {
    if (l.env_total <= 0) return 0;
    const dir = vec.normalize(d);
    const theta = fm.acos(@min(@max(dir[1], -1), 1));
    const sin_t = fm.sin(theta);
    if (sin_t < 1e-4) return 0;
    var u = (fm.atan2(dir[2], dir[0]) - l.env_rotation) * (0.5 / pi);
    u -= @floor(u);
    const x: u32 = @min(@as(u32, @intFromFloat(u * @as(f32, @floatFromInt(l.env_width)))), l.env_width - 1);
    const y: u32 = @min(@as(u32, @intFromFloat(theta * (1.0 / pi) * @as(f32, @floatFromInt(l.env_height)))), l.env_height - 1);
    const c = envTexel(l, x, y);
    // über den Überschuss, genau wie beim Bau der Verteilung
    const lu = @max(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] - l.env_mean, 0) * sin_t;
    if (lu <= 0) return 0;
    const nw: f32 = @floatFromInt(l.env_width);
    const nh: f32 = @floatFromInt(l.env_height);
    // Dichte über Pixel -> über Raumwinkel
    return lu / l.env_total * (nw * nh) / (2 * pi * pi * sin_t);
}

fn searchCdf(cdf: [*]const f32, n: u32, target: f32) u32 {
    var lo: u32 = 0;
    var hi: u32 = n;
    while (lo + 1 < hi) {
        const mid = (lo + hi) / 2;
        if (cdf[mid] <= target) lo = mid else hi = mid;
    }
    return lo;
}

pub const EnvSample = struct { dir: Vec3, radiance: Vec3, pdf: f32 };

/// Richtung nach der Helligkeit der Karte ziehen
pub fn envSample(l: *const types.Lighting, r1: f32, r2: f32) EnvSample {
    const marginal: [*]const f32 = @ptrFromInt(l.env_marginal);
    const cond: [*]const f32 = @ptrFromInt(l.env_cond);
    const y = searchCdf(marginal, l.env_height + 1, r1);
    const row = cond + @as(u64, y) * (l.env_width + 1);
    const x = searchCdf(row, l.env_width + 1, r2);
    // innerhalb des Texels gleichverteilt, damit keine Streifen entstehen
    const dy = blk: {
        const a = marginal[y];
        const b = marginal[y + 1];
        break :blk if (b > a) (r1 - a) / (b - a) else 0.5;
    };
    const dx = blk: {
        const a = row[x];
        const b = row[x + 1];
        break :blk if (b > a) (r2 - a) / (b - a) else 0.5;
    };
    const u = (@as(f32, @floatFromInt(x)) + dx) / @as(f32, @floatFromInt(l.env_width));
    const v = (@as(f32, @floatFromInt(y)) + dy) / @as(f32, @floatFromInt(l.env_height));
    const theta = v * pi;
    const phi = u * 2 * pi + l.env_rotation;
    const st = fm.sin(theta);
    const dir = Vec3{ st * fm.cos(phi), fm.cos(theta), st * fm.sin(phi) };
    // Strahldichte aus *demselben* Texel, über das auch die Dichte gebildet
    // wird. Nähme man hier den bilinearen Wert, bekäme ein dunkles Texel neben
    // der Sonne deren Helligkeit, aber seine eigene winzige Dichte – das
    // Verhältnis explodiert und ergibt Leuchtpunkte, die jeden Frame
    // woanders sitzen (gemessen: Flimmern 0,898 statt 0,388).
    return .{
        .dir = dir,
        .radiance = envTexel(l, x, y) * splat(l.env_intensity),
        .pdf = envPdf(l, dir),
    };
}

pub fn sky(l: *const types.Lighting, d: Vec3, with_sun: bool) Vec3 {
    if (hasEnv(l)) return envRadiance(l, d);
    return skyAnalytic(l, d, with_sun);
}

fn skyAnalytic(l: *const types.Lighting, d: Vec3, with_sun: bool) Vec3 {
    const zen: Vec3 = l.sky_zenith;
    const hor: Vec3 = l.sky_horizon;
    const gnd: Vec3 = l.ground_color;
    var c: Vec3 = undefined;
    if (d[1] >= 0) {
        const t = @sqrt(@min(d[1], 1.0));
        c = hor + (zen - hor) * splat(t);
    } else {
        const t = @sqrt(@min(-d[1], 1.0));
        c = hor + (gnd - hor) * splat(t);
    }
    c *= splat(l.sky_intensity);
    if (with_sun and l.flags & types.lighting_sun_disk != 0) {
        const sd = vec.normalize(l.sun_direction);
        const cos_r = fm.cos(l.sun_angular_radius);
        if (vec.dot(d, sd) > cos_r) {
            const r = @max(l.sun_angular_radius, 1e-3);
            c += @as(Vec3, l.sun_color) * splat(1.0 / (pi * r * r));
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Hilfen
// ---------------------------------------------------------------------------

fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

/// Orthonormalbasis um n (Duff et al. 2017)
fn basis(n: Vec3) [2]Vec3 {
    const sign: f32 = if (n[2] >= 0) 1 else -1;
    const a = -1.0 / (sign + n[2]);
    const b = n[0] * n[1] * a;
    return .{
        .{ 1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0] },
        .{ b, sign + n[1] * n[1] * a, -n[1] },
    };
}

fn cosineSample(n: Vec3, rng: *Rng) Vec3 {
    const r1 = rng.next();
    const r2 = rng.next();
    const r = @sqrt(r1);
    const phi = 2.0 * pi * r2;
    const t = basis(n);
    return vec.normalize(t[0] * splat(r * fm.cos(phi)) + t[1] * splat(r * fm.sin(phi)) + n * splat(@sqrt(@max(0, 1 - r1))));
}

/// Richtung innerhalb eines Kegels um d (Halbwinkel mit tan = spread)
fn coneSample(d: Vec3, spread: f32, rng: *Rng) Vec3 {
    if (spread <= 0) return d;
    const r = spread * @sqrt(rng.next());
    const phi = 2.0 * pi * rng.next();
    const t = basis(d);
    return vec.normalize(d + t[0] * splat(r * fm.cos(phi)) + t[1] * splat(r * fm.sin(phi)));
}

/// Wellen: zwei gekreuzte Sinuszüge stören die Normale (Wasser). Ableitung
/// der Höhenfunktion, daher exakt für die Spiegelung.
/// `footprint`: Größe eines Bildschirmpixels in Welteinheiten an dieser Stelle.
/// Fallen mehrere Wellen auf ein Pixel, springt die Normale von Pixel zu Pixel
/// und die Spiegelung mit ihr – das sieht als fleckige Wasserfläche aus.
/// Deshalb werden die Wellen ausgeblendet, sobald sie feiner als das Pixel
/// werden.
///
/// Die dabei verlorene Neigung wird aber *nicht* weggeworfen, sondern in
/// Rauheit umgerechnet: eine Fläche mit Wellen unterhalb der Pixelgröße ist
/// nicht glatt, sie ist rau. Wirft man sie weg, bleibt ein spiegelglattes
/// Wasser mit einem einzigen scharfen Sonnenglanz übrig, der bei jeder
/// kleinsten Änderung springt – gemessen die groesste verbleibende Quelle
/// der Bildunruhe. Als Rauheit wird daraus ein breiter, ruhiger Glanz.
pub const Waves = struct {
    normal: Vec3,
    /// zusätzliche Rauheit aus der ausgeblendeten Neigung
    roughness: f32,
};

pub fn waveNormal(s: *const types.Scene, m: *const types.Material, p: Vec3, n: Vec3, footprint: f32) Waves {
    const wl = @max(m.wave_length, 1e-3);
    const k = 6.2831853 / wl;
    const t: f32 = @floatCast(s.time);
    const ph = k * m.wave_speed * wl * t;
    // Höhe h(x, z) = A · (sin(k·x + φ) + sin(0.7·k·(x + z) + 1.3·φ))
    // Deutlich früher ausblenden: schon wenn eine Welle nur noch acht Pixel
    // breit ist, beginnt die Spiegelung zu sprenkeln.
    const fade = if (footprint > 0) @min(@max(wl / (16 * footprint), 0), 1) else 1;
    // Neigungsmaß der vollen Wellen; was `fade` davon wegnimmt, wird Rauheit.
    const slope = m.wave_height * k;
    const lost = @sqrt(@max(slope * slope * (1 - fade * fade), 0));
    const extra = @min(0.5 * lost, 1);
    if (fade <= 0.01) return .{ .normal = n, .roughness = extra };
    const a = m.wave_height * fade;
    const dhdx = a * k * (fm.cos(k * p[0] + ph) + 0.7 * fm.cos(0.7 * k * (p[0] + p[2]) + 1.3 * ph));
    const dhdz = a * k * (0.7 * fm.cos(0.7 * k * (p[0] + p[2]) + 1.3 * ph));
    // Störung senkrecht zur Fläche
    var t1 = Vec3{ 1, 0, 0 };
    if (@abs(n[0]) > 0.9) t1 = .{ 0, 1, 0 };
    const b1 = vec.normalize(vec.cross(n, t1));
    const b2 = vec.cross(n, b1);
    return .{ .normal = vec.normalize(n - b1 * splat(dhdx) - b2 * splat(dhdz)), .roughness = extra };
}

pub fn worldNormal(inst: *const types.InstanceData, face: u32) Vec3 {
    const f = face & types.hit_face_mask;
    const sign: f32 = if (f & 1 != 0) -1 else 1;
    const axis = f >> 1;
    const n = Vec3{ if (axis == 0) sign else 0, if (axis == 1) sign else 0, if (axis == 2) sign else 0 };
    return vec.normalize(vec.xformNormal(&inst.world_to_object, n));
}

/// Weltgröße eines Voxels (für den Versatz von Folgestrahlen)
pub fn voxelSize(inst: *const types.InstanceData) f32 {
    const m = &inst.object_to_world;
    return vec.length(.{ m[0], m[4], m[8] });
}

/// Lambert + GGX, bereits mit π multipliziert (siehe Konvention oben)
fn ggx(ndh: f32, rough: f32) f32 {
    const a = rough * rough;
    const a2 = a * a;
    const dd = ndh * ndh * (a2 - 1) + 1;
    return a2 / (pi * dd * dd);
}

fn brdf(n: Vec3, v: Vec3, l: Vec3, sf: *const Surface) Vec3 {
    const ndl = @max(vec.dot(n, l), 0);
    const ndv = @max(vec.dot(n, v), 1e-4);
    const h = vec.normalize(v + l);
    const ndh = @max(vec.dot(n, h), 0);
    const vdh = @max(vec.dot(v, h), 0);
    const a = sf.roughness * sf.roughness;
    const a2 = a * a;
    const dd = ndh * ndh * (a2 - 1) + 1;
    const d = a2 / (pi * dd * dd);
    const k = a * 0.5;
    const vis = 1.0 / (4.0 * (ndl * (1 - k) + k) * (ndv * (1 - k) + k));
    const f0 = splat(0.04) + (sf.albedo - splat(0.04)) * splat(sf.metallic);
    const q = 1 - vdh;
    const fw = q * q * q * q * q;
    const f = f0 + (splat(1) - f0) * splat(fw);
    const spec = f * splat(@min(d * vis * pi, 64));
    var out = sf.albedo * splat(1 - sf.metallic) + spec;

    // Klarlack: eine zweite, glatte Schicht darüber. Was sie reflektiert,
    // fehlt darunter – sonst würde das Material heller als sein Licht.
    if (sf.clearcoat > 0) {
        const dc = ggx(ndh, sf.clearcoat_roughness);
        const kc = sf.clearcoat_roughness * 0.5;
        const visc = 1.0 / (4.0 * (ndl * (1 - kc) + kc) * (ndv * (1 - kc) + kc));
        const fc = (0.04 + 0.96 * fw) * sf.clearcoat;
        out = out * splat(1 - fc) + splat(@min(dc * visc * pi, 64) * fc);
    }

    return out;
}

/// Licht, das durch das Material hindurch zur Vorderseite kommt
/// (Unterflächenstreuung). Getrennt vom BRDF, damit es nicht doppelt zählt.
fn transmit(sf: *const Surface) Vec3 {
    return sf.albedo * @as(Vec3, sf.subsurface_color) * splat(sf.subsurface * (1 - sf.metallic));
}

fn occluded(tracer: anytype, s: *const types.Scene, o: Vec3, d: Vec3, tmax: f32, mask: u32) bool {
    // durchsichtige Voxel halten kein Licht auf; ihre Tönung rechnet transmission()
    return tracer.trace(s, o, d, 0, tmax, mask, types.trace_any_hit | types.trace_skip_transparent) != null;
}

/// Gibt es überhaupt durchsichtige Instanzen oder Materialien?
pub inline fn anyTransparent(s: *const types.Scene, trans_mask: u32) bool {
    if (trans_mask != 0) return true;
    return (s.transparent_materials[0] | s.transparent_materials[1] |
        s.transparent_materials[2] | s.transparent_materials[3]) != 0;
}

/// Ist dieser Treffer durchsichtig – über die Instanzmaske oder das Material?
pub inline fn hitTransparent(s: *const types.Scene, h: tr.TraceHit, trans_mask: u32) bool {
    if (trans_mask != 0 and tr.instances(s)[h.instance].mask & trans_mask != 0) return true;
    if (s.materials == 0) return false;
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    return mats[h.attribute & 0xFF].flags & types.material_transparent != 0;
}

/// Durchlässigkeit transparenter Körper entlang eines Schattenstrahls:
/// je Grenzfläche (1 − Deckkraft) und Absorption im Medium bis zum Austritt.
/// Damit werfen Wasser und Glas getönte Schatten statt gar keiner.
/// Ergebnis eines Strahls, der durch durchsichtige Körper läuft
pub const ThroughHit = struct {
    /// erster undurchsichtiger Treffer (null = keiner)
    hit: ?tr.TraceHit,
    /// Abschwächung durch die durchquerten Körper
    att: Vec3,
};

/// Ein Durchlauf statt zwei: läuft bis zum ersten undurchsichtigen Treffer und
/// sammelt dabei die Abschwächung der durchsichtigen Körper ein. Ohne
/// Transparenz in der Szene ist das ein gewöhnlicher Strahl.
pub fn traceThrough(tracer: anytype, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, trans_mask: u32) ThroughHit {
    if (!anyTransparent(s, trans_mask)) {
        return .{ .hit = tracer.trace(s, o, d, tmin, tmax, ray_mask, 0), .att = splat(1) };
    }
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    var att = splat(@as(f32, 1));
    var t0 = tmin;
    var layer: u32 = 0;
    while (layer < types.max_transparent_layers + 1) : (layer += 1) {
        const h = tracer.trace(s, o, d, t0, tmax, ray_mask, 0) orelse return .{ .hit = null, .att = att };
        if (!hitTransparent(s, h, trans_mask)) return .{ .hit = h, .att = att };
        const inst = &tr.instances(s)[h.instance];
        const m = &mats[h.attribute & 0xFF];
        const sf = surface(s, h.attribute);
        const eps = 1e-3 * voxelSize(inst);
        att *= splat(1 - m.opacity);
        const own = trans_mask != 0 and inst.mask & trans_mask != 0;
        const rest = tmax - h.t;
        var len = rest;
        if (own) {
            const g = tr.dagOf(s, &tr.geometries(s)[inst.geometry]);
            const pin = o + d * splat(h.t + eps);
            const oo = vec.xformPoint(&inst.world_to_object, pin);
            const od = vec.xformVector(&inst.world_to_object, d);
            if (dag.traceExit(&g, oo, od, 0, @min(rest, 1e6))) |e| len = e.t;
        }
        if (m.density > 0) att *= absorb(sf.albedo, m.density * @min(len, 1e4));
        if (@reduce(.Max, att) < 1e-3) return .{ .hit = h, .att = splat(0) };
        t0 = h.t + (if (own) len else 0) + eps;
        if (!(t0 < tmax)) return .{ .hit = null, .att = att };
    }
    return .{ .hit = null, .att = att };
}

fn transmission(tracer: anytype, s: *const types.Scene, o: Vec3, d: Vec3, tmax: f32, ray_mask: u32, trans_mask: u32) Vec3 {
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    var att = splat(@as(f32, 1));
    var t0: f32 = 0;
    var layer: u32 = 0;
    while (layer < types.max_transparent_layers) : (layer += 1) {
        const h = tracer.trace(s, o, d, t0, tmax, ray_mask, 0) orelse break;
        if (!hitTransparent(s, h, trans_mask)) break;
        const inst = &tr.instances(s)[h.instance];
        const m = &mats[h.attribute & 0xFF];
        const sf = surface(s, h.attribute);
        const eps = 1e-3 * voxelSize(inst);
        att *= splat(1 - m.opacity);
        // Weg durch das Medium bis zum Austritt (oder bis zum Licht)
        const g = tr.dagOf(s, &tr.geometries(s)[inst.geometry]);
        const pin = o + d * splat(h.t + eps);
        const oo = vec.xformPoint(&inst.world_to_object, pin);
        const od = vec.xformVector(&inst.world_to_object, d);
        const rest = tmax - h.t;
        // eigene Instanz: bis zum Austritt aus ihren Voxeln; nur über das
        // Material markiert (Wasser im Bodenchunk): bis zum Ende des Strahls
        const own = trans_mask != 0 and inst.mask & trans_mask != 0;
        const ex = if (own) dag.traceExit(&g, oo, od, 0, @min(rest, 1e6)) else null;
        const len = if (ex) |e| e.t else rest;
        if (m.density > 0) att *= absorb(sf.albedo, m.density * @min(len, 1e4));
        if (@reduce(.Max, att) < 1e-3) return splat(0);
        t0 = h.t + len + eps;
        if (!(t0 < tmax)) break;
    }
    return att;
}

/// Direktes Licht (Sonne + Punktlichter) an Punkt p mit Normale n
fn direct(tracer: anytype, s: *const types.Scene, l: *const types.Lighting, p: Vec3, n: Vec3, v: Vec3, sf: *const Surface, rng: *Rng, mask: u32, trans_mask: u32) Vec3 {
    return directAt(tracer, s, l, p, n, v, sf, rng, mask, trans_mask, true);
}

fn directAt(tracer: anytype, s: *const types.Scene, l: *const types.Lighting, p: Vec3, n: Vec3, v: Vec3, sf: *const Surface, rng: *Rng, mask: u32, trans_mask: u32, want_shadows: bool) Vec3 {
    var c = splat(0);
    const shadows = want_shadows and l.flags & types.lighting_shadows != 0;

    // Schwarze Sonne (ihr Licht steckt in der Umgebungskarte): dann gar nicht
    // erst einen Schattenstrahl werfen. Das Ergebnis wäre null, der Strahl
    // kostet aber voll – und zwar an *jedem* Schattierungspunkt, also auch an
    // jedem GI-Treffer.
    const sun_lit = l.sun_always != 0 or l.sun_color[0] > 0 or l.sun_color[1] > 0 or l.sun_color[2] > 0;
    const sd = vec.normalize(l.sun_direction);
    const ld = if (shadows) coneSample(sd, fm.tan(l.sun_angular_radius), rng) else sd;
    const ndl = vec.dot(n, ld);
    // Mit Unterflächenstreuung zählt auch Licht von hinten (es wandert durch
    // das Material); brdf() liefert dafür den Rückseitenanteil.
    if (sun_lit and (ndl > 0 or (sf.subsurface > 0 and ndl > -1))) {
        var tint = splat(@as(f32, 1));
        var lit = true;
        if (shadows) {
            if (anyTransparent(s, trans_mask)) {
                const r = traceThrough(tracer, s, p, ld, 0, types.flt_max, mask | trans_mask, trans_mask);
                lit = r.hit == null;
                tint = r.att;
            } else lit = !occluded(tracer, s, p, ld, types.flt_max, mask);
        }
        if (lit) {
            if (ndl > 0) c += brdf(n, v, ld, sf) * @as(Vec3, l.sun_color) * splat(ndl) * tint;
            if (sf.subsurface > 0 and ndl < 0)
                c += transmit(sf) * @as(Vec3, l.sun_color) * splat(-ndl) * tint;
        }
    }

    // Umgebungskarte: eine Richtung nach ihrer Helligkeit ziehen und gegen die
    // Cosinus-Abtastung des GI-Strahls gewichten (Potenz-Heuristik). Ohne das
    // rauscht eine kleine helle Sonne in der Karte hoffnungslos.
    if (hasEnv(l) and l.env_total > 0) {
        const es = envSample(l, rng.next(), rng.next());
        const endl = vec.dot(n, es.dir);
        if (endl > 0 and es.pdf > 1e-8) {
            var tint = splat(@as(f32, 1));
            var lit = true;
            if (shadows) {
                if (anyTransparent(s, trans_mask)) {
                    const r = traceThrough(tracer, s, p, es.dir, 0, types.flt_max, mask | trans_mask, trans_mask);
                    lit = r.hit == null;
                    tint = r.att;
                } else lit = !occluded(tracer, s, p, es.dir, types.flt_max, mask);
            }
            if (lit) {
                const pdf_bsdf = endl / pi;
                const w = es.pdf * es.pdf / (es.pdf * es.pdf + pdf_bsdf * pdf_bsdf);
                c += clampContribution(l, brdf(n, v, es.dir, sf) * es.radiance * splat(endl * w / es.pdf) * tint);
            }
        }
    }

    var i: u32 = 0;
    while (i < @min(l.light_count, types.max_lights)) : (i += 1) {
        const light = &l.lights[i];
        // Abtastpunkt auf der Lichtquelle und ihr Öffnungsfaktor
        var target: Vec3 = light.position;
        var area_cos: f32 = 1;
        var area: f32 = 0;
        if (light.kind == types.light_rect) {
            // gleichverteilt auf dem Rechteck; die Dichte rechnet unten über
            // den Raumwinkel um (Fläche · cos / Abstand²)
            const ln = vec.normalize(light.normal);
            const tb = basis(ln);
            const t1 = tb[0];
            const t2 = tb[1];
            const a = (rng.next() * 2 - 1) * light.size[0];
            const b = (rng.next() * 2 - 1) * light.size[1];
            target += t1 * splat(a) + t2 * splat(b);
            area = 4 * light.size[0] * light.size[1];
            area_cos = @max(vec.dot(ln, vec.normalize(p - target)), 0);
            if (area_cos <= 0) continue;
        } else if (light.radius > 0) {
            target += cosineSample(vec.normalize(p - target), rng) * splat(light.radius);
        }
        const to = target - p;
        const dist2 = @max(vec.dot(to, to), 1e-8);
        const dist = @sqrt(dist2);
        if (light.range > 0 and dist > light.range) continue;
        const dir = to * splat(1.0 / dist);
        if (light.kind == types.light_spot) {
            // weicher Rand zwischen innerem und äußerem Winkel
            const ln = vec.normalize(light.normal);
            const cd = vec.dot(ln, -dir);
            const ci = light.size[0];
            const co = light.size[1];
            if (cd <= co) continue;
            const t = if (ci > co) @min(@max((cd - co) / (ci - co), 0), 1) else 1;
            area_cos = t * t;
        }
        const nl = vec.dot(n, dir);
        if (nl <= 0 and sf.subsurface <= 0) continue;
        var tint = splat(@as(f32, 1));
        if (shadows) {
            if (anyTransparent(s, trans_mask)) {
                const r = traceThrough(tracer, s, p, dir, 0, dist * 0.999, mask | trans_mask, trans_mask);
                if (r.hit != null) continue;
                tint = r.att;
            } else if (occluded(tracer, s, p, dir, dist * 0.999, mask)) continue;
        }
        const r2 = @max(light.radius * light.radius, 1e-4);
        // Rechteck: Fläche · cos / Abstand² ist der Raumwinkel; Kugel und
        // Kegel fallen mit 1/Abstand², der Kegel zusätzlich zum Rand hin.
        const falloff = if (light.kind == types.light_rect)
            area * area_cos / @max(dist2, 1e-6)
        else
            area_cos / @max(dist2, r2);
        if (nl > 0) c += brdf(n, v, dir, sf) * @as(Vec3, light.color) * splat(nl * falloff) * tint;
        if (sf.subsurface > 0 and nl < 0)
            c += transmit(sf) * @as(Vec3, light.color) * splat(-nl * falloff) * tint;
    }
    return c;
}

pub const Shaded = struct {
    color: Vec3,
    albedo: Vec3,
    normal: Vec3,
    /// diffuser Anteil (1 − metallic); der indirekte Durchgang multipliziert damit
    diffuse: f32 = 1,
    roughness: f32 = 1,
};

/// Farbe des Treffers h des Strahls o + t d.
/// `mask`: Instanzen, die Sekundärstrahlen (Schatten, GI, Reflexion) sehen –
/// ohne transparente Ebene.
pub fn shadeHit(tracer: anytype, s: *const types.Scene, o: Vec3, d: Vec3, h: tr.TraceHit, rng: *Rng, mask: u32, trans_mask: u32, footprint: f32) Shaded {
    const l: *const types.Lighting = @ptrFromInt(s.lighting);
    const inst = &tr.instances(s)[h.instance];
    const n = worldNormal(inst, h.face);
    const p = o + d * splat(h.t) + n * splat(1e-3 * voxelSize(inst));
    const sf = surfaceAt(s, h.attribute, p, n, footprint);
    // Beleuchtet wird mit der gestörten Normale, versetzt und weiterverfolgt
    // mit der geometrischen – sonst würden Strahlen in die Fläche laufen.
    const ns = if (sf.normal[0] != 0 or sf.normal[1] != 0 or sf.normal[2] != 0) sf.normal else n;
    // Sehen Sekundärstrahlen eine gröbere Fassung, müssen sie über deren
    // Voxel hinaus starten (in Entfernung t etwa secondary_bias · t groß)
    const ps = if (l.secondary_bias > 0) p + n * splat(l.secondary_bias * h.t) else p;
    const v = -d;

    var c = sf.emission + direct(tracer, s, l, ps, ns, v, &sf, rng, mask, trans_mask);

    // Indirekt: in halber Auflösung rechnet ein eigener Durchgang (gi_half),
    // sonst hier. Der diffuse Faktor kommt in beiden Fällen dazu.
    if (l.flags & types.lighting_gi_half == 0)
        c += sf.albedo * splat(1 - sf.metallic) * indirect(tracer, s, l, ps, ns, rng, mask, trans_mask);

    const refl_rough = if (sf.clearcoat > 0) @min(sf.roughness, sf.clearcoat_roughness) else sf.roughness;
    if (l.flags & types.lighting_reflections != 0 and refl_rough < 0.5) {
        const f0 = splat(0.04) + (sf.albedo - splat(0.04)) * splat(sf.metallic);
        const refl = reflection(tracer, s, l, ps, ns, d, refl_rough, rng, mask, trans_mask);
        const fr = fresnel(f0, @max(vec.dot(ns, v), 0)) + splat(0.04 * sf.clearcoat);
        c += fr * refl * splat(1 - 2 * refl_rough);
    }
    // Die Normale im Ziel bleibt die geometrische: Denoiser, TAA und
    // Reprojektion brauchen sie stabil, nicht mit Detail überlagert.
    return .{ .color = c, .albedo = sf.albedo, .normal = n, .diffuse = 1 - sf.metallic, .roughness = sf.roughness };
}

/// Günstige Beleuchtung ohne GI und Reflexionen – für den Untergrund hinter
/// transparenten Körpern, wo der Aufwand nicht sichtbar wäre.
pub fn shadeSimple(tracer: anytype, s: *const types.Scene, o: Vec3, d: Vec3, h: tr.TraceHit, rng: *Rng, mask: u32, trans_mask: u32) Vec3 {
    const l: *const types.Lighting = @ptrFromInt(s.lighting);
    const inst = &tr.instances(s)[h.instance];
    const n = worldNormal(inst, h.face);
    const sf = surface(s, h.attribute);
    const p = o + d * splat(h.t) + n * splat(1e-3 * voxelSize(inst));
    return sf.emission + direct(tracer, s, l, p, n, -d, &sf, rng, mask, trans_mask) +
        sf.albedo * splat(1 - sf.metallic) * sky(l, n, false) * splat(0.5 + 0.5 * n[1]);
}

/// Eintreffende indirekte Strahldichte an (p, n): ein Kosinus-Strahl (GI),
/// Umgebungsverdeckung oder – ohne beides – der Himmel grob nach der Normalen.
/// Das Ergebnis wird mit Albedo · (1 − metallic) multipliziert.
/// Beitrag einer einzelnen Abtastung begrenzen. Das verschiebt den
/// Erwartungswert leicht nach unten, nimmt aber genau die vereinzelten
/// Ausreißer weg, die als Flimmern auffallen.
fn clampContribution(l: *const types.Lighting, c: Vec3) Vec3 {
    if (l.firefly_clamp <= 0) return c;
    const m = @max(@max(c[0], c[1]), c[2]);
    if (m <= l.firefly_clamp) return c;
    return c * splat(l.firefly_clamp / m);
}

pub fn indirect(tracer: anytype, s: *const types.Scene, l: *const types.Lighting, p: Vec3, n: Vec3, rng: *Rng, mask: u32, trans_mask: u32) Vec3 {
    if (l.flags & types.lighting_gi != 0) {
        const gi_max = if (l.gi_distance > 0) l.gi_distance else types.flt_max;
        const bounces = @max(l.gi_bounces, 1);
        // Mehrere Reflexionen: der Lichtweg wird verfolgt, solange noch
        // nennenswert Energie übrig ist. Ab der zweiten entscheidet russisches
        // Roulette – so bleibt der Erwartungswert richtig, ohne jeden Pfad
        // bis zum Ende zu rechnen.
        var acc: Vec3 = splat(0);
        var throughput: Vec3 = splat(1);
        var op = p;
        var on = n;
        var b: u32 = 0;
        while (b < bounces) : (b += 1) {
            const gd = cosineSample(on, rng);
            const r = traceThrough(tracer, s, op, gd, 0, gi_max, mask | trans_mask, trans_mask);
            if (r.hit) |g| {
                const ginst = &tr.instances(s)[g.instance];
                const gn = worldNormal(ginst, g.face);
                const hp = op + gd * splat(g.t);
                // Indirekte Treffer brauchen kein Detail: gröbste Stufe
                const gsf = surfaceAt(s, g.attribute, hp, gn, 1e6);
                const gp = hp + gn * splat(1e-3 * voxelSize(ginst));
                // Schattenstrahlen an indirekten Treffern sind der teuerste
                // Posten der ganzen Beleuchtung. Bis gi_shadow_depth werfen
                // sie welche, darüber nehmen sie das Licht ungeschattet.
                const want_sh = b < l.gi_shadow_depth;
                acc += clampContribution(l, throughput * r.att * (gsf.emission + directAt(tracer, s, l, gp, gn, -gd, &gsf, rng, mask, trans_mask, want_sh)));
                if (b + 1 >= bounces) break;
                // Weiter mit dem diffusen Anteil der getroffenen Fläche
                throughput *= r.att * gsf.albedo * splat(1 - gsf.metallic);
                const q = @max(@max(throughput[0], throughput[1]), throughput[2]);
                if (q < 0.05) break;
                if (q < 1) {
                    if (rng.next() > q) break;
                    throughput *= splat(1 / q);
                }
                op = gp;
                on = gn;
                continue;
            }
            // In den Himmel: dort endet der Pfad.
            if (hasEnv(l) and l.env_total > 0) {
                const pdf_env = envPdf(l, gd);
                const pdf_bsdf = @max(vec.dot(on, gd), 0) / pi;
                const w = if (pdf_bsdf > 0) pdf_bsdf * pdf_bsdf / (pdf_bsdf * pdf_bsdf + pdf_env * pdf_env) else 0;
                acc += throughput * r.att * envRadiance(l, gd) * splat(w);
            } else {
                acc += throughput * r.att * sky(l, gd, false);
            }
            break;
        }
        return acc;
    }
    if (l.flags & types.lighting_ao != 0) {
        const ad = cosineSample(n, rng);
        const vis: f32 = if (occluded(tracer, s, p, ad, @max(l.ao_radius, 1e-3), mask)) 0 else 1;
        return sky(l, ad, false) * splat(vis);
    }
    // ohne Sekundärstrahlen: Himmel grob nach der Normalen
    return sky(l, n, false) * splat(0.5 + 0.5 * n[1]);
}

fn fresnel(f0: Vec3, cos_theta: f32) Vec3 {
    const q = 1 - cos_theta;
    const q5 = q * q * q * q * q;
    return f0 + (splat(1) - f0) * splat(q5);
}

fn reflect(d: Vec3, n: Vec3) Vec3 {
    return d - n * splat(2 * vec.dot(d, n));
}

/// Licht aus der Spiegelrichtung (gestreut nach Rauheit): direktes Licht und
/// Emission am Treffer, sonst Himmel
fn reflection(tracer: anytype, s: *const types.Scene, l: *const types.Lighting, p: Vec3, n: Vec3, d: Vec3, roughness: f32, rng: *Rng, mask: u32, trans_mask: u32) Vec3 {
    var rd = coneSample(reflect(d, n), roughness * roughness, rng);
    if (vec.dot(rd, n) <= 0) rd = reflect(d, n);
    if (tracer.trace(s, p, rd, 0, types.flt_max, mask, 0)) |h| {
        const inst = &tr.instances(s)[h.instance];
        const hn = worldNormal(inst, h.face);
        const hsf = surface(s, h.attribute);
        const hp = p + rd * splat(h.t) + hn * splat(1e-3 * voxelSize(inst));
        return hsf.emission + direct(tracer, s, l, hp, hn, -rd, &hsf, rng, mask, trans_mask) + hsf.albedo * sky(l, hn, false) * splat(0.5 + 0.5 * hn[1]);
    }
    return sky(l, rd, false);
}

/// Transparente Ebene (Wasser, Glas) über dem Untergrund `behind`:
/// Fresnel-Reflexion, Deckkraft der Oberfläche, Absorption auf der Strecke
/// bis zum Untergrund (dist). Brechung wird vernachlässigt (gerader Strahl).

/// Brechung nach Snell (eta = n1 / n2); null bei Totalreflexion
fn refract(d: Vec3, n: Vec3, eta: f32) ?Vec3 {
    const cosi = -vec.dot(n, d);
    const k = 1 - eta * eta * (1 - cosi * cosi);
    if (k < 0) return null;
    return vec.normalize(d * splat(eta) + n * splat(eta * cosi - @sqrt(k)));
}

/// Mehrere transparente Körper vor dem undurchsichtigen Untergrund.
/// Je Körper: Eintrittsfläche (Fresnel-Spiegelung, Deckkraft, Brechung mit
/// material_refract) → Absorption im Medium (density, getönt mit base_color)
/// bis zum echten Austritt (dag.traceExit) → Austrittsfläche (Fresnel,
/// Rückbrechung) → weiter zum nächsten Körper. Reicht ein Medium bis an den
/// Untergrund (Wasser auf Boden), endet es dort.
/// `behind`: fertige Farbe des Untergrunds entlang des ungebrochenen Strahls
/// (bis `t_opaque`), wiederverwendet, solange nichts gebrochen hat.
pub fn transparentLayers(tracer: anytype, s: *const types.Scene, o0: Vec3, d0: Vec3, tmin: f32, t_opaque: f32, behind: Vec3, ray_mask: u32, trans_mask: u32, opaque_mask: u32, secondary_mask: u32, rng: *Rng, footprint: f32) Vec3 {
    const l: *const types.Lighting = @ptrFromInt(s.lighting);
    const mats: [*]const types.Material = @ptrFromInt(s.materials);
    // aktueller Strahl o + t d mit t in [0, t_end); t_end = Untergrund (flt_max: Himmel)
    var o = o0 + d0 * splat(tmin);
    var d = d0;
    var t_end = t_opaque - tmin;
    var bent = false;
    var result: Vec3 = splat(0);
    var throughput: Vec3 = splat(1);

    // Startet die Kamera schon in einem Medium (Blick von unter Wasser), gilt
    // dessen Absorption ab dem ersten Schritt; die Grenzfläche kommt beim
    // Austritt.
    if (tracer.trace(s, o, d, 0, t_end, ray_mask, 0)) |first| {
        if (first.face & types.hit_inside != 0 and hitTransparent(s, first, trans_mask)) {
            const m0 = &mats[first.attribute & 0xFF];
            const sf0 = surface(s, first.attribute);
            const inst0 = &tr.instances(s)[first.instance];
            const own0 = trans_mask != 0 and inst0.mask & trans_mask != 0;
            var len = t_end;
            if (own0) {
                const g0 = tr.dagOf(s, &tr.geometries(s)[inst0.geometry]);
                const oo0 = vec.xformPoint(&inst0.world_to_object, o);
                const od0 = vec.xformVector(&inst0.world_to_object, d);
                if (dag.traceExit(&g0, oo0, od0, 0, @min(t_end, 1e6))) |e0| len = e0.t;
            }
            if (m0.density > 0) throughput *= absorb(sf0.albedo, m0.density * @min(len, 1e4));
            if (len < t_end) {
                // Austritt: Fresnel und Rückbrechung an der Grenzfläche
                const r0 = (m0.ior - 1) / (m0.ior + 1);
                const f0 = splat(r0 * r0);
                const p0 = o + d * splat(len);
                var n0 = worldNormal(inst0, first.face);
                if (vec.dot(n0, d) < 0) n0 = -n0; // zeigt nach außen
                throughput *= splat(1) - fresnel(f0, @max(vec.dot(n0, d), 0));
                if (m0.flags & types.material_refract != 0) {
                    if (refract(d, -n0, m0.ior)) |dr| {
                        d = dr;
                        bent = true;
                    } else return result; // Totalreflexion an der Wasseroberfläche
                }
                o = p0 + d * splat(1e-3 * voxelSize(inst0));
                t_end = if (bent) (if (tracer.trace(s, o, d, 0, types.flt_max, opaque_mask, types.trace_skip_transparent)) |oh| oh.t else types.flt_max) else t_end - len;
            }
        }
    }

    var layer: u32 = 0;
    while (layer < types.max_transparent_layers) : (layer += 1) {
        const th = tracer.trace(s, o, d, 0, t_end, ray_mask, 0) orelse break;
        if (!hitTransparent(s, th, trans_mask)) break;
        if (th.face & types.hit_inside != 0) break; // schon im Medium: oben behandelt
        const inst = &tr.instances(s)[th.instance];
        // Eigene Instanz: der Körper endet, wo seine Voxel enden. Nur über das
        // Material markiert (Wasser im selben Chunk wie der Boden): das Medium
        // reicht bis zum Untergrund.
        const own_instance = trans_mask != 0 and inst.mask & trans_mask != 0;
        const m = &mats[th.attribute & 0xFF];
        var sf = surface(s, th.attribute);
        var n = worldNormal(inst, th.face);
        if (m.flags & types.material_waves != 0) {
            const w = waveNormal(s, m, o + d * splat(th.t), n, footprint * th.t);
            n = w.normal;
            sf.roughness = @min(@sqrt(sf.roughness * sf.roughness + w.roughness * w.roughness), 1);
            // Der scharfe Glanz sitzt auf der Lackschicht, nicht auf dem
            // Grundmaterial: sie muss genauso aufgeweitet werden.
            sf.clearcoat_roughness = @min(@sqrt(sf.clearcoat_roughness * sf.clearcoat_roughness +
                w.roughness * w.roughness), 1);
        }
        if (vec.dot(n, d) > 0) n = -n;
        const eps = 1e-3 * voxelSize(inst);
        const p = o + d * splat(th.t);
        const r0 = (m.ior - 1) / (m.ior + 1);
        const f0 = splat(r0 * r0);
        const f = fresnel(f0, @max(vec.dot(n, -d), 0));

        // Eintrittsfläche: Spiegelung und (nach Deckkraft) diffuse Oberfläche
        const pv = p + n * splat(eps);
        var refl = sky(l, reflect(d, n), true);
        if (l.flags & types.lighting_reflections != 0) refl = reflection(tracer, s, l, pv, n, d, sf.roughness, rng, secondary_mask, trans_mask);
        const surface_col = sf.emission + direct(tracer, s, l, pv, n, -d, &sf, rng, secondary_mask, trans_mask) + sf.albedo * sky(l, n, false) * splat(0.5 + 0.5 * n[1]);
        result += throughput * (refl * f + surface_col * splat(m.opacity) * (splat(1) - f));
        throughput *= (splat(1) - f) * splat(1 - m.opacity);
        if (@reduce(.Max, throughput) < 1e-3) return result;

        const refracts = m.flags & types.material_refract != 0 and m.ior > 0;
        var din = d;
        if (refracts) din = refract(d, n, 1.0 / m.ior) orelse return result; // Totalreflexion
        const pin = p - n * splat(eps);
        const turned = @reduce(.Or, din != d);
        var remaining = t_end - th.t; // bis zum Untergrund (nur gültig ohne Knick)
        if (turned) {
            bent = true;
            remaining = if (tracer.trace(s, pin, din, 0, types.flt_max, opaque_mask, types.trace_skip_transparent)) |oh| oh.t else types.flt_max;
        }

        // Austritt aus dem Medium (im Objektraum; t bleibt gleich)
        const g = tr.dagOf(s, &tr.geometries(s)[inst.geometry]);
        const oo = vec.xformPoint(&inst.world_to_object, pin);
        const od = vec.xformVector(&inst.world_to_object, din);
        // Austritt erst kurz vor dem Untergrund zählt als "Medium liegt auf"
        const ex_raw = if (own_instance) dag.traceExit(&g, oo, od, 0, @min(remaining, 1e6)) else null;
        const ex = if (ex_raw) |x| (if (x.t < remaining - 2 * eps) x else null) else null;
        const inside_len = if (ex) |e| e.t else remaining;
        if (m.density > 0) throughput *= absorb(sf.albedo, m.density * @min(inside_len, 1e4));
        const e = ex orelse {
            // Medium reicht bis an den Untergrund
            o = pin;
            d = din;
            t_end = remaining;
            break;
        };

        // Austrittsfläche: Transmission und Rückbrechung
        const n_out = worldNormal(inst, e.face); // zeigt in Laufrichtung
        const pout = pin + din * splat(e.t);
        throughput *= splat(1) - fresnel(f0, @max(vec.dot(n_out, din), 0));
        var dout = din;
        if (refracts) dout = refract(din, -n_out, m.ior) orelse din; // innere Totalreflexion: gerade weiter (Näherung)
        o = pout + n_out * splat(eps);
        // auch eine parallel versetzte Richtung (Scheibe) macht `behind` ungültig
        bent = bent or @reduce(.Or, dout != din);
        if (bent) {
            d = dout;
            t_end = if (tracer.trace(s, o, d, 0, types.flt_max, opaque_mask, types.trace_skip_transparent)) |oh| oh.t else types.flt_max;
        } else {
            t_end = remaining - e.t - eps;
        }
    }

    // Untergrund
    var bg = behind;
    if (bent) {
        if (tracer.trace(s, o, d, 0, types.flt_max, opaque_mask, types.trace_skip_transparent)) |h| {
            bg = shadeSimple(tracer, s, o, d, h, rng, secondary_mask, trans_mask);
        } else bg = sky(l, d, true);
    }
    return result + throughput * bg;
}

inline fn absorb(color: Vec3, k: f32) Vec3 {
    return .{ fm.pow(@max(color[0], 1e-4), k), fm.pow(@max(color[1], 1e-4), k), fm.pow(@max(color[2], 1e-4), k) };
}

// ---------------------------------------------------------------------------
// Teilnehmendes Medium: Nebel und Lichtschächte
//
// Entlang des Primärstrahls wird in festen Schritten marschiert. An jedem
// Schritt kommt Licht von der Sonne dazu, sofern der Punkt sie sieht – das
// sind die Lichtschächte. Was dahinter liegt, wird um die Durchlässigkeit
// gedämpft. Die Schrittlage wird je Pixel verschoben (sonst entstehen
// Bänder), das Rauschen daraus nimmt der zeitliche Filter weg.
// ---------------------------------------------------------------------------

/// Dichte auf Höhe y: unterhalb fog_height voll, darüber exponentiell weniger
inline fn fogDensity(l: *const types.Lighting, y: f32) f32 {
    if (l.fog_falloff <= 0) return l.fog_density;
    const dy = y - l.fog_height;
    if (dy <= 0) return l.fog_density;
    return l.fog_density * fm.exp(-dy * l.fog_falloff);
}

/// Henyey-Greenstein: wie stark das Medium nach vorn streut
inline fn phaseHG(g: f32, cos_t: f32) f32 {
    if (g == 0) return 1.0 / (4 * pi);
    const g2 = g * g;
    const d = 1 + g2 - 2 * g * cos_t;
    return (1 - g2) / (4 * pi * d * @sqrt(@max(d, 1e-6)));
}

pub inline fn hasFog(l: *const types.Lighting) bool {
    return l.fog_density > 0;
}

/// Farbe hinter dem Medium dämpfen und das eingestreute Licht dazurechnen.
/// `dist` ist die Länge des Sichtstrahls (flt_max für den Himmel).
pub fn applyFog(tracer: anytype, s: *const types.Scene, l: *const types.Lighting, o: Vec3, d: Vec3, dist: f32, color: Vec3, rng: *Rng, mask: u32) Vec3 {
    const max_dist = @min(dist, if (l.gi_distance > 0) l.gi_distance * 4 else 4096);
    if (!(max_dist > 0)) return color;
    const steps: u32 = if (l.fog_steps == 0) 12 else @min(l.fog_steps, 64);
    const dt = max_dist / @as(f32, @floatFromInt(steps));
    const sd = vec.normalize(l.sun_direction);
    const phase = phaseHG(l.fog_anisotropy, vec.dot(d, sd));
    const shadows = l.flags & types.lighting_shadows != 0;

    var transmittance: f32 = 1;
    var inscatter: Vec3 = splat(0);
    const jitter = rng.next();
    var i: u32 = 0;
    while (i < steps) : (i += 1) {
        const t = (@as(f32, @floatFromInt(i)) + jitter) * dt;
        const pos = o + d * splat(t);
        const dens = fogDensity(l, pos[1]);
        if (dens <= 0) continue;
        const sigma = dens * dt;
        // Sonne sichtbar? Das ergibt die Schächte.
        var vis: f32 = 1;
        if (shadows and occluded(tracer, s, pos, sd, types.flt_max, mask)) vis = 0;
        if (vis > 0) {
            const li = @as(Vec3, l.sun_color) * splat(phase * vis);
            inscatter += li * @as(Vec3, l.fog_color) * splat(sigma * transmittance);
        }
        // Umgebungslicht im Medium (grob: der Himmel von oben)
        inscatter += sky(l, .{ 0, 1, 0 }, false) * @as(Vec3, l.fog_color) * splat(sigma * transmittance * (1.0 / (4 * pi)));
        transmittance *= fm.exp(-sigma);
        if (transmittance < 0.01) break;
    }
    return color * splat(transmittance) + inscatter;
}
