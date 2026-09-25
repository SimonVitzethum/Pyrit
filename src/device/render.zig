//! Primärstrahlen, Tiefe und Motion Vectors pro Pixel.
//!
//! Motion Vector = Position im Vorframe − Position jetzt, in Pixeln, ohne Jitter.
//! Er wird aus dem exakten Trefferpunkt im Objektraum berechnet: dieselbe
//! Ruheposition wird mit der Vorframe-Transformation und -Kamera projiziert.

const types = @import("types.zig");
const vec = @import("vec.zig");
const tr = @import("trace.zig");
const shade = @import("shade.zig");
const Vec3 = vec.Vec3;

pub const CameraRay = struct { o: Vec3, d: Vec3, tmin: f32, tmax: f32 };

/// Strahl durch die Bildposition (px, py) in Pixeln; t misst Weltdistanz.
pub fn cameraRay(c: *const types.Camera, px: f32, py: f32) CameraRay {
    const nx = 2.0 * px / @as(f32, @floatFromInt(c.width)) - 1.0;
    const ny = 1.0 - 2.0 * py / @as(f32, @floatFromInt(c.height));
    const vx = (nx + c.shift[0]) * c.scale[0];
    const vy = (ny + c.shift[1]) * c.scale[1];
    const ortho = c.projection == types.projection_orthographic;
    const ov: Vec3 = if (ortho) .{ vx, vy, 0 } else .{ 0, 0, 0 };
    const dv: Vec3 = if (ortho) .{ 0, 0, -1 } else .{ vx, vy, -1 };
    const dw = vec.xformVector(&c.view_to_world, dv);
    const len = vec.length(dw);
    return .{
        .o = vec.xformPoint(&c.view_to_world, ov),
        .d = dw * vec.splat(1.0 / len),
        .tmin = c.near_plane * len,
        .tmax = if (c.far_plane > 0) c.far_plane * len else types.flt_max,
    };
}

pub const Projected = struct { x: f32, y: f32, depth: f32 };

inline fn toPixels(c: *const types.CameraData, x: f32, y: f32) [2]f32 {
    const nx = x / c.camera.scale[0] - c.camera.shift[0];
    const ny = y / c.camera.scale[1] - c.camera.shift[1];
    return .{
        (nx + 1.0) * 0.5 * @as(f32, @floatFromInt(c.camera.width)),
        (1.0 - ny) * 0.5 * @as(f32, @floatFromInt(c.camera.height)),
    };
}

/// Weltpunkt -> Bildposition in Pixeln; null hinter der Kamera.
pub fn projectPoint(c: *const types.CameraData, pw: Vec3) ?Projected {
    const v = vec.xformPoint(&c.world_to_view, pw);
    const depth = -v[2];
    var x = v[0];
    var y = v[1];
    if (c.camera.projection != types.projection_orthographic) {
        if (!(depth > 0)) return null;
        x /= depth;
        y /= depth;
    }
    const p = toPixels(c, x, y);
    return .{ .x = p[0], .y = p[1], .depth = depth };
}

/// Richtung (Punkt im Unendlichen) -> Bildposition; nur perspektivisch.
pub fn projectDirection(c: *const types.CameraData, dw: Vec3) ?[2]f32 {
    if (c.camera.projection == types.projection_orthographic) return null;
    const v = vec.xformVector(&c.world_to_view, dw);
    const depth = -v[2];
    if (!(depth > 0)) return null;
    return toPixels(c, v[0] / depth, v[1] / depth);
}

pub const PixelResult = struct {
    hit: types.Hit,
    depth: f32,
    motion: [2]f32,
    /// nur mit Shading (p.color != 0)
    color: [4]f32 = .{ 0, 0, 0, 0 },
    normal: [4]f32 = .{ 0, 0, 0, types.flt_max },
    albedo: [4]f32 = .{ 1, 1, 1, 0 },
    /// Rauheit und Metallanteil (für DLSS Ray Reconstruction)
    material: [2]f32 = .{ 1, 0 },
};

/// Schreibt die angeforderten Ausgaben eines Pixels.
pub fn writePixel(p: *const types.RenderParams, i: u64, r: *const PixelResult) void {
    if (p.hits != 0) @as([*]types.Hit, @ptrFromInt(p.hits))[i] = r.hit;
    if (p.depth != 0) @as([*]f32, @ptrFromInt(p.depth))[i] = r.depth;
    if (p.motion != 0) @as([*][2]f32, @ptrFromInt(p.motion))[i] = r.motion;
    if (p.color != 0) @as([*][4]f32, @ptrFromInt(p.color))[i] = r.color;
    if (p.normal != 0) @as([*][4]f32, @ptrFromInt(p.normal))[i] = r.normal;
    if (p.albedo != 0) @as([*][4]f32, @ptrFromInt(p.albedo))[i] = r.albedo;
    if (p.material != 0) @as([*][2]f32, @ptrFromInt(p.material))[i] = r.material;
}

/// Software-Traversierung (CUDA-Kerne); Alternative: RT-Cores in rt_kernels.zig.
pub const SoftwareTracer = struct {
    pub inline fn trace(_: SoftwareTracer, s: *const types.Scene, o: Vec3, d: Vec3, tmin: f32, tmax: f32, ray_mask: u32, flags: u32) ?tr.TraceHit {
        return tr.traceScene(s, o, d, tmin, tmax, ray_mask, flags);
    }
};

pub fn renderPixel(p: *const types.RenderParams, s: *const types.Scene, x: u32, y: u32) PixelResult {
    return renderPixelWith(SoftwareTracer{}, p, s, x, y);
}

/// Primärstrahl, Tiefe und Motion Vector eines Pixels. `tracer.trace` liefert
/// den nächsten Treffer samt exakter Ruheposition (p_object).
/// Berührt o + t d für t in [t0, t1] die Box [lo, hi]? Großzügig (Rand +1 %),
/// denn ein falsches Nein ließe Wasser verschwinden.
fn segmentTouches(o: Vec3, d: Vec3, t0: f32, t1: f32, lo: [3]f32, hi: [3]f32) bool {
    var a = t0;
    var b = t1;
    inline for (0..3) |k| {
        if (lo[k] > hi[k]) return false; // leere Hülle
        const pad = (hi[k] - lo[k]) * 0.01 + 1e-2;
        const l = lo[k] - pad;
        const h = hi[k] + pad;
        if (@abs(d[k]) < 1e-12) {
            if (o[k] < l or o[k] > h) return false;
        } else {
            const inv = 1.0 / d[k];
            const ta = (l - o[k]) * inv;
            const tb = (h - o[k]) * inv;
            a = @max(a, @min(ta, tb));
            b = @min(b, @max(ta, tb));
            if (a > b) return false;
        }
    }
    return true;
}

/// Primärstrahl eines Pixels: Maske und Flags, wie renderPixelWith ihn
/// verfolgt (die Wiederholungs-Wavefront trägt ihn damit vorab bitgleich ein)
pub const Primary = struct { ray: CameraRay, mask: u32, flags: u32 };

pub inline fn primaryRay(p: *const types.RenderParams, x: u32, y: u32) Primary {
    const cam = &p.cur.camera;
    const px = @as(f32, @floatFromInt(x)) + 0.5 + cam.jitter[0];
    const py = @as(f32, @floatFromInt(y)) + 0.5 + cam.jitter[1];
    // Ohne Überspringen durchsichtiger Voxel: trifft er etwas Undurchsichtiges
    // (oder nichts), liegt davor sicher kein Wasser – dann entfallen beide
    // Strahlen der transparenten Schicht (siehe renderPixelWith)
    return .{ .ray = cameraRay(cam, px, py), .mask = p.ray_mask & ~p.transparent_mask, .flags = p.flags };
}

pub fn renderPixelWith(tracer: anytype, p: *const types.RenderParams, s: *const types.Scene, x: u32, y: u32) PixelResult {
    const cam = &p.cur.camera;
    const px = @as(f32, @floatFromInt(x)) + 0.5 + cam.jitter[0];
    const py = @as(f32, @floatFromInt(y)) + 0.5 + cam.jitter[1];
    const prim = primaryRay(p, x, y);
    const ray = prim.ray;
    const opaque_mask = p.ray_mask & ~p.transparent_mask;
    // Schatten, GI und Reflexionen dürfen eine andere (gröbere) Auswahl sehen
    const secondary_mask = if (p.secondary_mask != 0) p.secondary_mask else opaque_mask;
    // durchsichtige Voxel überspringt die Traversierung selbst
    const trans_mask = p.ray_mask & p.transparent_mask;
    // Erst ohne Überspringen: ist der erste Treffer durchsichtig (Wasser,
    // Glas), braucht es den Untergrund dahinter noch einmal mit Überspringen.
    // Sonst ist er selbst der Untergrund, und davor liegt nichts Durchsichtiges.
    const first = tracer.trace(s, ray.o, ray.d, ray.tmin, ray.tmax, prim.mask, prim.flags);
    const trans_mask_early = p.ray_mask & p.transparent_mask;
    const water_front = if (first) |f| shade.anyTransparent(s, trans_mask_early) and shade.hitTransparent(s, f, trans_mask_early) else false;
    const found = if (water_front) tracer.trace(s, ray.o, ray.d, ray.tmin, ray.tmax, prim.mask, prim.flags | types.trace_skip_transparent) else first;

    var r = PixelResult{ .hit = tr.toHit(found), .depth = types.flt_max, .motion = .{ 0, 0 } };
    r.hit.meta |= @as(u32, 0xFF) << types.hit_fog_shift; // klar, bis der Dunst etwas anderes sagt
    const history = p.history_valid != 0;

    if (found) |h| {
        const pw = ray.o + ray.d * vec.splat(h.t);
        r.depth = -vec.xformPoint(&p.cur.world_to_view, pw)[2];

        const cur = &tr.instances(s)[h.instance];
        const prev = &tr.instancesPrev(s)[h.instance];
        const fresh = prev.flags & types.instance_active == 0 or prev.history != cur.history;
        if (fresh and cur.flags & types.instance_keep_history == 0) r.hit.meta |= types.hit_new;

        if (history) {
            // Ruheposition mit dem Vorframe-Zustand abbilden; ohne Vorgeschichte nur Kamerabewegung
            const pw_prev = if (fresh) pw else vec.xformPoint(&prev.object_to_world, h.p_object);
            if (projectPoint(&p.prev, pw_prev)) |q| r.motion = .{ q.x - px, q.y - py };
        }
    } else if (history) {
        if (projectDirection(&p.prev, ray.d)) |q| r.motion = .{ q[0] - px, q[1] - py };
    }
    if (!history) r.hit.meta |= types.hit_no_history;

    if (p.color != 0 or p.normal != 0 or p.albedo != 0) {
        if (found) |h| {
            var rng = shade.Rng.init(x, y, p.frame_index, 0);
            // Deckungsabtastung gegen wandernde Kanten
            //
            // Die Ursache ist nicht das Rauschen, sondern die Deckung: an
            // einer Voxelkante entscheidet der Jitter jeden Frame neu, welche
            // der beiden Flächen das Pixel sieht – und die sind sehr
            // verschieden beleuchtet. Ein Filter kann das nicht heilen, weil
            // das Signal echt wechselt.
            //
            // Hier wird die Deckung stattdessen *innerhalb* eines Frames
            // aufgelöst: zusätzliche Primärstrahlen prüfen, welche Fläche das
            // Pixel wirklich zu welchem Anteil sieht. Schattiert wird nur, was
            // sich unterscheidet – im Bildinneren treffen alle Abtastungen
            // dieselbe Fläche und es bleibt bei einer Schattierung. Nur an
            // Kanten, also wenigen Prozent der Pixel, kommt eine zweite dazu.
            // Größe eines Bildschirmpixels in Welteinheiten am Treffer.
            // Daraus wählt das Shading die Verkleinerungsstufe der Texturen
            // und blendet die Detailnormale aus, bevor sie flimmern kann.
            const cm = &p.cur.camera;
            // in *Ausgabe*pixeln: beim Hochskalieren sind sie um detail_scale kleiner
            const px_per_unit = 2 * cm.scale[1] / (@as(f32, @floatFromInt(@max(cm.height, 1))) * @max(p.detail_scale, 1));
            const footprint = if (cm.projection == types.projection_orthographic) px_per_unit else h.t * px_per_unit;
            const sh = shade.shadeHit(tracer, s, ray.o, ray.d, h, &rng, secondary_mask, trans_mask, footprint, water_front);
            var col = vec.Vec3{ sh.color[0], sh.color[1], sh.color[2] };
            // Albedo wird genauso gemittelt wie die Farbe. Die Nachbearbeitung
            // filtert Farbe / Albedo; stammte die Farbe aus mehreren Flächen
            // (oder zur Hälfte aus dem Himmel), das Albedo aber nur aus der
            // Mitte, entstanden an Silhouetten Werte weit außerhalb jeder
            // echten Beleuchtung – Himmelblau durch dunkles Laubgrün ist
            // Violett. Der Filter verschmierte sie, und wieder mit dem Albedo
            // der Nachbarn multipliziert zeigten sie sich als violette Säume.
            var alb = vec.Vec3{ sh.albedo[0], sh.albedo[1], sh.albedo[2] };

            var cov = @min(@max(p.coverage, 1), 4);
            // Die zusätzlichen Strahlen lohnen nur an einer Voxelkante. Wo
            // der Treffer weit genug von der Kante entfernt in seiner Fläche
            // liegt, landen alle Abtastungen ohnehin im selben Voxel – das
            // steht hier schon fest, ohne einen einzigen weiteren Strahl.
            // Das betrifft die große Mehrheit der Pixel, und für sie kostet
            // die Deckung damit gar nichts.
            if (cov > 1) {
                const inst = &tr.instances(s)[h.instance];
                // Spaltenlänge der Welt->Objekt-Matrix: Weltmaß -> Objektmaß.
                var w2o: f32 = 0;
                inline for (0..3) |a| w2o += inst.world_to_object[a] * inst.world_to_object[a];
                const fp_obj = footprint * @sqrt(w2o);
                const axis = (h.face & types.hit_face_mask) >> 1;
                // Laub mit Lochmuster: die Kanten der Lochzellen zählen wie
                // Voxelkanten, sonst flimmerten die Löcher bei Bewegung
                const mats: [*]const types.Material = @ptrFromInt(s.materials);
                const cells: f32 = if (mats[h.attribute & 0xFF].flags & types.material_cutout != 0) @floatFromInt(@import("dag.zig").cutout_cells) else 1;
                var near_edge = false;
                inline for (0..3) |a| {
                    if (a != axis) {
                        const q = h.p_object[a] * cells;
                        const f = q - @floor(q);
                        if (@min(f, 1 - f) < 0.75 * fp_obj * cells) near_edge = true;
                    }
                }
                if (!near_edge) cov = 1;
            }
            if (cov > 1) {
                // Feste Versätze auf einem gedrehten Gitter: sie liegen
                // gleichmäßig im Pixel und sind über die Frames konstant, der
                // Jitter verschiebt sie gemeinsam.
                const offs = [3][2]f32{ .{ 0.3, -0.1 }, .{ -0.1, 0.3 }, .{ -0.3, -0.3 } };
                var wsum: f32 = 1;
                var k: u32 = 0;
                while (k + 1 < cov) : (k += 1) {
                    const o2 = offs[k];
                    const r2 = cameraRay(cam, px + o2[0], py + o2[1]);
                    const f2 = tracer.trace(s, r2.o, r2.d, r2.tmin, r2.tmax, opaque_mask, p.flags | types.trace_skip_transparent);
                    if (f2) |h2| {
                        // Dieselbe Fläche in derselben Tiefe heißt: dieselbe
                        // Beleuchtung. Unterscheidet sich nur das Attribut,
                        // also die Farbe des Nachbarvoxels, genügt es, sie
                        // auszutauschen – dafür braucht es keinen einzigen
                        // weiteren Strahl. Nur an echten Geometriekanten
                        // (andere Fläche oder andere Tiefe) wird voll neu
                        // schattiert, und das sind wenige Pixel.
                        const same_face = h2.instance == h.instance and
                            (h2.face & types.hit_face_mask) == (h.face & types.hit_face_mask) and
                            @abs(h2.t - h.t) < 0.002 * h.t + 1e-3;
                        if (same_face and h2.attribute == h.attribute) {
                            col += vec.Vec3{ sh.color[0], sh.color[1], sh.color[2] };
                            alb += vec.Vec3{ sh.albedo[0], sh.albedo[1], sh.albedo[2] };
                        } else if (same_face) {
                            const hp2 = r2.o + r2.d * vec.splat(h2.t);
                            const nn = vec.Vec3{ sh.normal[0], sh.normal[1], sh.normal[2] };
                            const s2 = shade.surfaceAt(s, h2.attribute, hp2, nn, footprint);
                            var scaled: vec.Vec3 = undefined;
                            inline for (0..3) |q| {
                                scaled[q] = sh.color[q] * (s2.albedo[q] / @max(sh.albedo[q], 1e-4));
                            }
                            col += scaled;
                            alb += s2.albedo;
                        } else {
                            var rng2 = shade.Rng.init(x, y, p.frame_index, 8 + k);
                            const fp2 = if (cm.projection == types.projection_orthographic) px_per_unit else h2.t * px_per_unit;
                            const s2 = shade.shadeHit(tracer, s, r2.o, r2.d, h2, &rng2, secondary_mask, trans_mask, fp2, water_front);
                            col += vec.Vec3{ s2.color[0], s2.color[1], s2.color[2] };
                            alb += vec.Vec3{ s2.albedo[0], s2.albedo[1], s2.albedo[2] };
                        }
                    } else {
                        // Himmel wird nicht demoduliert: er zählt mit Albedo 1
                        col += shade.sky(@ptrFromInt(s.lighting), r2.d, true);
                        alb += vec.splat(1);
                    }
                    wsum += 1;
                }
                col *= vec.splat(1.0 / wsum);
                alb *= vec.splat(1.0 / wsum);
            }

            r.color = .{ col[0], col[1], col[2], 1 };
            r.normal = .{ sh.normal[0], sh.normal[1], sh.normal[2], r.depth };
            // w trägt den diffusen Anteil für den indirekten Durchgang
            r.albedo = .{ alb[0], alb[1], alb[2], sh.diffuse };
            r.material = .{ sh.roughness, 1 - sh.diffuse };
        } else {
            const c = shade.sky(@ptrFromInt(s.lighting), ray.d, true);
            r.color = .{ c[0], c[1], c[2], 0 };
        }
        // Transparente Flächen (mehrere, mit Brechung) vor dem Untergrund
        if (p.color != 0) {
            const t_behind = if (found) |h| h.t else ray.tmax;
            var rng = shade.Rng.init(x, y, p.frame_index, 1);
            const behind = vec.Vec3{ r.color[0], r.color[1], r.color[2] };
            const cm2 = &p.cur.camera;
            const fp_scale = 2 * cm2.scale[1] / (@as(f32, @floatFromInt(@max(cm2.height, 1))) * @max(p.detail_scale, 1));
            // Die transparente Schicht sucht mit zwei Strahlen nach Wasser und
            // Glas vor dem Untergrund. Nötig nur, wenn der erste Strahl (ohne
            // Überspringen) etwas Durchsichtiges traf – oder bei Instanzen der
            // transparenten Ebene, die er wegen der Maske nicht sieht und deren
            // Hülle der Sichtstrahl berührt. Sonst käme `behind` ohnehin
            // bitgleich zurück; gespart: zwei Strahlen je Land- und Himmelspixel.
            const need_layers = water_front or
                (trans_mask != 0 and segmentTouches(ray.o, ray.d, ray.tmin, t_behind, p.trans_lo, p.trans_hi));
            var c = if (need_layers)
                shade.transparentLayers(tracer, s, ray.o, ray.d, ray.tmin, t_behind, behind, p.ray_mask, trans_mask, opaque_mask, secondary_mask, &rng, fp_scale)
            else
                behind;
            // Ohne durchsichtige Schicht kommt `behind` bitgleich zurück.
            // Hinter Wasser (water_front) immer markieren: danach richtet sich
            // auch, wo das Himmelslicht gerechnet wird (shadeHit / giPixel).
            if (water_front or @reduce(.Or, c != behind)) r.hit.meta |= types.hit_through_transparent;
            // Nebel ganz zum Schluss: er dämpft alles dahinter, auch die
            // transparenten Schichten, und steuert die Lichtschächte bei.
            const lg: *const types.Lighting = @ptrFromInt(s.lighting);
            if (lg.camera_medium_density > 0) {
                // Kamera im Medium (unter Wasser): bis zum Treffer oder bis
                // zur Oberfläche darüber
                var dist = @min(t_behind, 1e4);
                if (ray.d[1] > 1e-4) dist = @min(dist, @max((lg.camera_medium_top - ray.o[1]) / ray.d[1], 0));
                const tm = shade.absorbPublic(lg.camera_medium_color, lg.camera_medium_density * dist);
                const sc: vec.Vec3 = lg.camera_medium_scatter;
                c = c * tm + sc * (vec.splat(1) - tm);
                // der verschleierte Anteil zählt mit Albedo 1 (wie beim Dunst)
                if (r.color[3] > 0.5) {
                    const tv = @min(@max((tm[0] + tm[1] + tm[2]) / 3, 0), 1);
                    if (lg.flags & types.lighting_gi_half != 0) {
                        r.hit.meta = (r.hit.meta & ~(@as(u32, 0xFF) << types.hit_fog_shift)) |
                            (@as(u32, @intFromFloat(tv * 255 + 0.5)) << types.hit_fog_shift);
                    } else {
                        inline for (0..3) |k| r.albedo[k] = r.albedo[k] * tv + (1 - tv);
                    }
                }
            } else if (shade.hasFog(lg)) {
                var frng = shade.Rng.init(x, y, p.frame_index, 3);
                const cv = vec.Vec3{ c[0], c[1], c[2] };
                const fogged = shade.applyFog(tracer, s, lg, ray.o, ray.d, t_behind, cv, &frng, secondary_mask);
                c = fogged.color;
                // Der Dunst liegt vor der Fläche und hat mit ihrem Albedo
                // nichts zu tun. Die Nachbearbeitung filtert aber Farbe /
                // Albedo: bläulicher Dunst durch das Grün von Laub geteilt
                // ergab Magenta, das der Filter verschmierte und auf die
                // Nachbarn (Stämme) übertrug. Der verschleierte Anteil zählt
                // deshalb wie der Himmel mit Albedo 1.
                if (r.color[3] > 0.5) {
                    const tf = @min(@max(fogged.transmittance, 0), 1);
                    if (lg.flags & types.lighting_gi_half != 0) {
                        // Das indirekte Licht kommt erst im Kombinieren dazu:
                        // dort wird es gedämpft und das Albedo angeglichen.
                        r.hit.meta = (r.hit.meta & ~(@as(u32, 0xFF) << types.hit_fog_shift)) |
                            (@as(u32, @intFromFloat(tf * 255 + 0.5)) << types.hit_fog_shift);
                    } else {
                        inline for (0..3) |k| r.albedo[k] = r.albedo[k] * tf + (1 - tf);
                    }
                }
            }
            r.color = .{ c[0], c[1], c[2], r.color[3] };
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Indirekte Beleuchtung in halber Auflösung (lighting_gi_half)
//
// 1. Hauptdurchgang: alles außer der indirekten Beleuchtung; albedo.w trägt
//    den diffusen Anteil (1 − metallic).
// 2. giPixel: je 2x2-Block ein Strahl, Abtastpunkt wandert mit dem Frame.
// 3. combinePixel: kantenbewusst hochskalieren (Normale, Tiefe) und dazurechnen.
// ---------------------------------------------------------------------------

const H4 = [4]f16;

inline fn giStore(ptr: u64, i: u64, v: [4]f32) void {
    var h: H4 = undefined;
    inline for (0..4) |k| h[k] = @floatCast(@min(@max(v[k], -65504.0), 65504.0));
    @as([*]H4, @ptrFromInt(ptr))[i] = h;
}

inline fn giLoad(ptr: u64, i: u64) [4]f32 {
    const h = @as([*]const H4, @ptrFromInt(ptr))[i];
    return .{ h[0], h[1], h[2], h[3] };
}

/// Abtastpixel eines 2x2-Blocks. Der Versatz ist *fest*.
///
/// Wanderte er mit dem Frame, sprang die indirekte Beleuchtung eines Blocks
/// an Voxelkanten jeden Frame auf eine andere Fläche – dieselbe Ursache wie
/// beim Jitter: ein echter Signalwechsel, den kein Filter glätten kann.
/// Die zeitliche Abdeckung, die der wandernde Versatz brachte, ist ohne
/// Jitter ohnehin nicht mehr vorhanden.
pub inline fn giSample(p: *const types.RenderParams, hx: u32, hy: u32) [2]u32 {
    const ox: u32 = 0;
    const oy: u32 = 0;
    return .{
        @min(hx * 2 + ox, p.cur.camera.width - 1),
        @min(hy * 2 + oy, p.cur.camera.height - 1),
    };
}

/// Ein Strahl je 2x2-Block: Treffer und Normale kommen aus dem Hauptdurchgang.
pub fn giPixel(tracer: anytype, p: *const types.RenderParams, s: *const types.Scene, hx: u32, hy: u32) void {
    const gi_index = @as(u64, hy) * p.gi_width + hx;
    const c = giSample(p, hx, hy);
    const i = @as(u64, c[1]) * p.cur.camera.width + c[0];
    const hit = @as([*]const types.Hit, @ptrFromInt(p.hits))[i];
    if (hit.instance == types.no_hit) {
        giStore(p.gi, gi_index, .{ 0, 0, 0, 0 });
        return;
    }
    const nw = @as([*]const [4]f32, @ptrFromInt(p.normal))[i];
    const n = Vec3{ nw[0], nw[1], nw[2] };
    const cam = &p.cur.camera;
    const ray = cameraRay(cam, @as(f32, @floatFromInt(c[0])) + 0.5 + cam.jitter[0], @as(f32, @floatFromInt(c[1])) + 0.5 + cam.jitter[1]);
    const l: *const types.Lighting = @ptrFromInt(s.lighting);
    const inst = &tr.instances(s)[hit.instance];
    const pw = ray.o + ray.d * vec.splat(hit.t) + n * vec.splat(1e-3 * shade.voxelSize(inst));
    const ps = if (l.secondary_bias > 0) pw + n * vec.splat(l.secondary_bias * hit.t) else pw;
    const opaque_mask = p.ray_mask & ~p.transparent_mask;
    const mask = if (p.secondary_mask != 0) p.secondary_mask else opaque_mask;
    var rng = shade.Rng.init(c[0], c[1], p.frame_index, 2);
    var li = shade.indirect(tracer, s, l, ps, n, &rng, mask, p.ray_mask & p.transparent_mask);
    // Himmelslicht (nächstes Ereignis) ebenfalls hier, in halber Auflösung –
    // die volle Auflösung lässt es dafür weg (shadeHit)
    // (nicht hinter Wasser: dort rechnet es shadeHit, damit die Schicht es dämpft)
    if (l.flags & types.lighting_gi != 0 and hit.meta & types.hit_through_transparent == 0)
        li += shade.envNee(tracer, s, l, ps, n, &rng, mask, p.ray_mask & p.transparent_mask);
    giStore(p.gi, gi_index, .{ li[0], li[1], li[2], 1 });
}

/// Hochskalieren und dazurechnen: Gewichte aus Normale und Tiefe, damit die
/// indirekte Beleuchtung nicht über Kanten läuft.
pub fn combinePixel(p: *const types.RenderParams, x: u32, y: u32) void {
    const w = p.cur.camera.width;
    const i = @as(u64, y) * w + x;
    const nw = @as([*]const [4]f32, @ptrFromInt(p.normal))[i];
    if (!(nw[3] < types.flt_max)) return; // Himmel
    const n = Vec3{ nw[0], nw[1], nw[2] };
    const depth = nw[3];

    const fx = (@as(f32, @floatFromInt(x)) - 0.5) * 0.5;
    const fy = (@as(f32, @floatFromInt(y)) - 0.5) * 0.5;
    const bx: i32 = @intFromFloat(@floor(fx));
    const by: i32 = @intFromFloat(@floor(fy));
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);

    var sum: Vec3 = @splat(0);
    var wsum: f32 = 0;
    var best: Vec3 = @splat(0);
    var best_w: f32 = -1;
    inline for (0..4) |k| {
        const hx = bx + @as(i32, k & 1);
        const hy = by + @as(i32, k >> 1);
        if (hx >= 0 and hy >= 0 and hx < p.gi_width and hy < p.gi_height) {
            const g = giLoad(p.gi, @as(u64, @intCast(hy)) * p.gi_width + @as(u32, @intCast(hx)));
            if (g[3] > 0) {
                const c = giSample(p, @intCast(hx), @intCast(hy));
                const sn = @as([*]const [4]f32, @ptrFromInt(p.normal))[@as(u64, c[1]) * w + c[0]];
                const nd = @max(n[0] * sn[0] + n[1] * sn[1] + n[2] * sn[2], 0);
                const n4 = nd * nd * nd * nd;
                const wd = 1.0 / (1.0 + 20.0 * @abs(sn[3] - depth) / @max(depth, 1e-3));
                const bw = (if (k & 1 != 0) tx else 1 - tx) * (if (k >> 1 != 0) ty else 1 - ty);
                const wgt = bw * n4 * n4 * wd;
                const li = Vec3{ g[0], g[1], g[2] };
                sum += li * vec.splat(wgt);
                wsum += wgt;
                if (n4 * wd > best_w) {
                    best_w = n4 * wd;
                    best = li;
                }
            }
        }
    }
    if (best_w < 0) return; // kein gültiger Nachbar
    const li = if (wsum > 1e-4) sum * vec.splat(1.0 / wsum) else best;
    const albp = @as([*][4]f32, @ptrFromInt(p.albedo));
    const alb = albp[i];
    // Dunst vor der Fläche: dämpft auch das indirekte Licht, und der
    // verschleierte Anteil zählt für die Nachbearbeitung mit Albedo 1
    // (siehe renderPixelWith)
    const meta = @as([*]const types.Hit, @ptrFromInt(p.hits))[i].meta;
    const tf = @as(f32, @floatFromInt((meta >> types.hit_fog_shift) & 0xFF)) / 255.0;
    const add = Vec3{ alb[0], alb[1], alb[2] } * vec.splat(alb[3] * tf) * li;
    const col = @as([*][4]f32, @ptrFromInt(p.color));
    col[i] = .{ col[i][0] + add[0], col[i][1] + add[1], col[i][2] + add[2], col[i][3] };
    if (tf < 1) albp[i] = .{ alb[0] * tf + (1 - tf), alb[1] * tf + (1 - tf), alb[2] * tf + (1 - tf), alb[3] };
}
