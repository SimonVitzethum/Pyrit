//! Gemeinsame Datentypen von Host und GPU.
//!
//! Diese `extern struct`s sind die ABI zwischen Host-Bibliothek, Runtime-Kerneln
//! und eigenen Kerneln. include/pyrit.h beschreibt dieselben Strukturen für
//! C-Aufrufer; ein Test prüft Größen und Offsets beider Seiten.

const std = @import("std");

pub const flt_max: f32 = std.math.floatMax(f32);

// ---------------------------------------------------------------------------
// Strahlen und Treffer
// ---------------------------------------------------------------------------

pub const Ray = extern struct {
    origin: [3]f32,
    tmin: f32,
    /// muss nicht normiert sein; t wird in Vielfachen davon gemessen
    direction: [3]f32,
    tmax: f32,
};

pub const no_hit: u32 = 0xFFFF_FFFF;

/// Fläche des getroffenen Voxels im Objektraum
pub const face_pos_x: u32 = 0;
pub const face_neg_x: u32 = 1;
pub const face_pos_y: u32 = 2;
pub const face_neg_y: u32 = 3;
pub const face_pos_z: u32 = 4;
pub const face_neg_z: u32 = 5;

pub const hit_face_mask: u32 = 0x7;
/// Instanz ohne gültige Vorgeschichte: Motion Vector nur aus der Kamerabewegung
pub const hit_new: u32 = 0x8;
/// Ansicht ohne Vorframe: Motion Vector = 0
pub const hit_no_history: u32 = 0x10;
/// Strahl beginnt in einem gefüllten Voxel: t = tmin, Fläche undefiniert
pub const hit_inside: u32 = 0x20;
/// Das Pixel sieht durch eine durchsichtige Fläche (Wasser, Glas): Spiegelung
/// und Brechung bewegen sich anders als der Untergrund, dessen Motion Vector
/// das Pixel trägt. Die zeitliche Mittelung bleibt dort kurz.
pub const hit_through_transparent: u32 = 0x40;
/// Bits 24..31: Durchlässigkeit des Dunsts bis zum Treffer (255 = klar).
/// Intern für die indirekte Beleuchtung in halber Auflösung, die erst nach
/// dem Dunst dazukommt und von ihm gedämpft werden muss.
pub const hit_fog_shift: u5 = 24;

pub const Hit = extern struct {
    /// Strahlparameter; flt_max bei Fehlschuss
    t: f32,
    /// Instanz-Index oder no_hit
    instance: u32,
    /// Voxelattribut (frei belegbar, z. B. Material oder Farbe)
    attribute: u32,
    /// Fläche (hit_face_mask) | hit_*
    meta: u32,
};

// ---------------------------------------------------------------------------
// Kamera
//
// Kameraraum: rechtshändig, Blick entlang -Z, +Y oben, +X rechts.
// Pixel (0,0) ist oben links; Pixelmitten liegen bei +0.5.
// ---------------------------------------------------------------------------

pub const projection_perspective: u32 = 0;
pub const projection_orthographic: u32 = 1;

pub const Camera = extern struct {
    /// Kamerapose, 3x4 zeilenweise (Spalte 3 = Position)
    view_to_world: [12]f32,
    projection: u32,
    width: u32,
    height: u32,
    /// perspektivisch: tan(fov_x/2), tan(fov_y/2); orthografisch: halbe Breite/Höhe
    scale: [2]f32,
    /// Lens-Shift in NDC-Einheiten
    shift: [2]f32,
    /// Subpixel-Versatz in Pixeln; wirkt nur auf die Strahlerzeugung
    jitter: [2]f32,
    /// Strahlstart entlang der Blickachse
    near_plane: f32,
    /// Strahlende entlang der Blickachse; 0 = unendlich
    far_plane: f32,
};

// ---------------------------------------------------------------------------
// Szene auf der GPU
// ---------------------------------------------------------------------------

pub const geometry_has_attributes: u32 = 0x1;

pub const GeometryData = extern struct {
    /// in 32-Bit-Worten im Knotenpool
    node_offset: u32,
    /// in 64-Bit-Bricks im Blattpool
    leaf_offset: u32,
    /// in 32-Bit-Werten im Attributpool
    attribute_offset: u32,
    /// Wurzel, relativ zu node_offset
    root: u32,
    /// Kantenlänge = 1 << log2_size Voxel
    log2_size: u32,
    flags: u32,
    /// Attribut, wenn keine Attribute gespeichert sind
    default_attribute: u32,
    reserved: u32,
};

pub const instance_active: u32 = 0x1;
/// Neue Instanz ohne eigene Vorgeschichte: statt PYR_HIT_NEW gilt der Verlauf
/// der Umgebung weiter (statische Geometrie, z. B. Welt-Chunks beim LOD-Wechsel)
pub const instance_keep_history: u32 = 0x2;

pub const InstanceData = extern struct {
    /// 3x4 zeilenweise, Welt relativ zum Render-Ursprung
    object_to_world: [12]f32,
    world_to_object: [12]f32,
    /// Welt-AABB
    bounds_min: [3]f32,
    geometry: u32,
    bounds_max: [3]f32,
    /// Sichtbarkeitsmaske, UND-verknüpft mit der Strahlmaske
    mask: u32,
    user: u32,
    /// ändert sich bei Neuanlage/Reset: dann keine Vorgeschichte
    history: u32,
    flags: u32,
    reserved: u32,
};

/// Adressen sind Gerätezeiger als u64 (stabile ABI für alle Sprachen).
pub const Scene = extern struct {
    nodes: u64,
    leaves: u64,
    attributes: u64,
    geometries: u64,
    /// aktueller Frame
    instances: u64,
    /// Vorframe
    instances_prev: u64,
    /// Obergrenze belegter Indizes
    instance_count: u32,
    geometry_count: u32,
    frame: u64,
    time: f64,
    time_prev: f64,
    /// Render-Ursprung dieses Frames in Weltkoordinaten
    origin: [3]f64,
    origin_prev: [3]f64,
    /// const Material[max_materials]
    materials: u64,
    /// const Lighting*
    lighting: u64,
    /// Bit je Material: gesetzt = material_transparent (für trace_skip_transparent)
    transparent_materials: [4]u64,
    /// Bit je Material: gesetzt = material_cutout
    cutout_materials: [4]u64,
    /// const TextureData[texture_count], 1-basiert angesprochen
    textures: u64,
    texture_count: u32,
    reserved_tex: u32,
};

// ---------------------------------------------------------------------------
// Materialien und Licht
//
// Voxelattribut: Bits 0..7 = Materialindex, Bits 8..31 = Farbe 0xRRGGBB.
// Materialien mit material_voxel_color multiplizieren ihre Grundfarbe mit
// dieser Voxelfarbe. Attribut 0 bedeutet beim DAG-Bau "leer".
// ---------------------------------------------------------------------------

pub const max_materials: u32 = 256;

pub const material_voxel_color: u32 = 0x1;
/// transparente Ebene: Strahl wird nach Snell gebrochen (ior); sonst gerade
/// hindurch (dünne Scheiben, Laub)
pub const material_refract: u32 = 0x2;
/// Voxel dieses Materials sind durchsichtig: Primärstrahlen laufen hindurch,
/// die Transparenzschleife sammelt sie. So liegt Wasser in derselben Geometrie
/// wie der Boden, ohne eigene Instanz.
pub const material_transparent: u32 = 0x4;
/// Wellen: die Normale wird zeitabhängig gestört (Wasser). Die Geometrie
/// bleibt stehen, damit Treffer, Tiefe und Motion Vectors exakt bleiben.
pub const material_waves: u32 = 0x8;
/// Durchbrochen (Laub): jede Voxelfläche trägt ein festes 4x4-Muster mit
/// etwa 30 % Löchern. Strahlen, die ein Loch treffen, laufen durch den Voxel
/// hindurch – auch Schattenstrahlen, das ergibt Lichtflecken am Boden. Das
/// Muster hängt nur von Voxel, Fläche und Attribut ab: es rauscht nicht.
pub const material_cutout: u32 = 0x10;
/// höchstens so viele transparente Grenzflächen je Pixel
pub const max_transparent_layers: u32 = 4;

pub const Material = extern struct {
    /// linear
    base_color: [3]f32,
    roughness: f32,
    /// Strahldichte (linear, beliebig hell)
    emission: [3]f32,
    metallic: f32,
    /// material_*
    flags: u32,
    /// nur transparente Ebene: Deckkraft der Oberfläche (0 = klar)
    opacity: f32,
    /// Brechungsindex (Fresnel-Reflexion)
    ior: f32,
    /// nur transparente Ebene: Absorption pro Welteinheit, getönt mit base_color
    density: f32,
    /// material_waves: Höhe und Wellenlänge der Störung (Welteinheiten),
    /// Geschwindigkeit in Wellenlängen je Sekunde
    wave_height: f32,
    wave_length: f32,
    wave_speed: f32,
    /// Klarlack: zweite, glatte Schicht über dem Grundmaterial (Lack, Nässe)
    clearcoat: f32,
    clearcoat_roughness: f32,
    /// Lichtstreuung unter der Oberfläche: das Licht wickelt sich um die Kante
    /// (Haut, Laub, Wachs). 0 = aus.
    subsurface: f32,
    subsurface_color: [3]f32,
    /// Texturen (1-basiert, 0 = keine) und Kantenlänge einer Kachel in
    /// Welteinheiten. Auf Voxelflächen wird achsenparallel projiziert, also
    /// genau eine Ebene je Fläche – kein Triplanar-Mischen nötig.
    texture: u32,
    normal_texture: u32,
    texture_scale: f32,
    /// Stärke und Wellenlänge einer erzeugten Detailnormale (ohne Textur)
    normal_strength: f32,
    normal_scale: f32,
    /// Seitenflächen (Normale waagerecht) nehmen diese Textur statt `texture`
    /// – wie der Grasblock, dessen Seiten Erde mit Grasrand zeigen. 0 = wie
    /// oben. Die Normalentextur gilt weiter für alle Flächen.
    side_texture: u32,
    /// Farbe (linear) der Seitenflächen statt der Voxelfarbe; {0,0,0} = die
    /// Voxelfarbe behalten. Mit {1,1,1} trägt die Seitentextur die Farbe selbst.
    side_color: [3]f32,
};

/// Eine Textur: dicht gepackte RGBA8-Zeilen. Gefiltert wird von Hand
/// (bilinear, wiederholend) – keine Texturhardware, damit derselbe Code
/// später auch auf AMD läuft.
pub const TextureData = extern struct {
    data: u64,
    width: u32,
    height: u32,
    /// Anzahl der Verkleinerungsstufen (1 = nur die Grundstufe). Sie liegen
    /// hintereinander im selben Puffer, jede halb so groß wie die vorige.
    levels: u32,
    reserved_tex_data: u32,
};

pub const max_lights: u32 = 64;

/// Kugel (Punktlicht mit Radius)
pub const light_sphere: u32 = 0;
/// Rechteck: strahlt in Richtung `normal`, Kantenlängen 2·size
pub const light_rect: u32 = 1;
/// Kegel um `normal` mit weichem Rand zwischen den beiden Winkeln
pub const light_spot: u32 = 2;

pub const Light = extern struct {
    position: [3]f32,
    /// Radius der Kugellichtquelle (weiche Schatten)
    radius: f32,
    /// Intensität (linear); Beleuchtungsstärke = color / Abstand²
    color: [3]f32,
    /// Reichweite; 0 = unbegrenzt
    range: f32,
    /// light_*
    kind: u32,
    /// Rechteck und Kegel: Richtung (wird normiert)
    normal: [3]f32,
    /// Rechteck: halbe Kantenlängen; Kegel: cos(innen), cos(außen)
    size: [2]f32,
    reserved_light: [2]u32,
};

pub const lighting_shadows: u32 = 0x1;
/// eine indirekte Diffus-Reflexion (Global Illumination)
pub const lighting_gi: u32 = 0x2;
/// Umgebungsverdeckung statt GI (günstiger)
pub const lighting_ao: u32 = 0x4;
/// Sonnenscheibe im sichtbaren Himmel
pub const lighting_sun_disk: u32 = 0x8;
/// Reflexionsstrahlen für glatte Oberflächen (Rauheit < 0.5) und transparente Ebenen
pub const lighting_reflections: u32 = 0x10;
/// Indirekte Beleuchtung in halber Auflösung berechnen und kantenbewusst
/// hochskalieren (ein Viertel der Strahlen). Braucht color, normal, albedo
/// und hits als Ziele.
pub const lighting_gi_half: u32 = 0x20;

pub const Lighting = extern struct {
    /// Richtung zur Sonne (wird normiert)
    sun_direction: [3]f32,
    /// Winkelradius der Sonne in Radiant (weiche Schatten), z. B. 0.00465
    sun_angular_radius: f32,
    /// Beleuchtungsstärke der Sonne (linear)
    sun_color: [3]f32,
    /// lighting_*
    flags: u32,
    sky_zenith: [3]f32,
    sky_intensity: f32,
    sky_horizon: [3]f32,
    /// Reichweite der Umgebungsverdeckung in Welteinheiten
    ao_radius: f32,
    ground_color: [3]f32,
    light_count: u32,
    /// Versatz der Sekundärstrahlen entlang der Normale, als Anteil der
    /// Trefferentfernung. Nötig, wenn Sekundärstrahlen eine gröbere Fassung
    /// sehen (secondary_mask): deren Voxel sind in Entfernung t etwa
    /// secondary_bias · t groß und würden sonst die Fläche selbst verschatten.
    secondary_bias: f32,
    /// Reichweite der indirekten Beleuchtung in Welteinheiten; darüber zählt
    /// der Himmel. 0 = unbegrenzt. Kürzere Strahlen sind deutlich billiger.
    gi_distance: f32,
    /// Umgebungskarte (equirektangulär, 4 x f32 je Texel) und ihre Verteilung
    /// für das Importance-Sampling: `env_marginal` hat height+1 Werte,
    /// `env_cond` je Zeile width+1. 0 = keine Karte, dann der analytische
    /// Himmel aus sky_*.
    env_data: u64,
    env_marginal: u64,
    env_cond: u64,
    env_width: u32,
    env_height: u32,
    env_intensity: f32,
    /// Drehung der Karte um die Y-Achse in Radiant
    env_rotation: f32,
    /// Summe des Helligkeitsüberschusses (für die Wahrscheinlichkeitsdichte)
    env_total: f32,
    /// mittlere Helligkeit der Karte; die Lichtabtastung zielt nur auf das,
    /// was darüber liegt (den Rest trifft der Cosinus-Strahl ohnehin)
    env_mean: f32,
    /// Indirekte Reflexionen: 1 = eine (Vorgabe), mehr für tiefere Lichtwege.
    /// Ab der zweiten wird russisches Roulette angewandt.
    gi_bounces: u32,
    /// Bis zu welcher Tiefe indirekte Treffer eigene Schattenstrahlen werfen.
    /// 0 = nur der Primärtreffer schattet, die Bounces nehmen das Licht
    /// ungeschattet. Das ist die mit Abstand teuerste Stelle: jeder Bounce
    /// kostet sonst einen weiteren Schattenstrahl je Pixel.
    gi_shadow_depth: u32,
    /// Diagnose: Sonnenstrahl auch bei schwarzer Sonne werfen (A/B-Vergleich)
    sun_always: u32,
    /// Teilnehmendes Medium (Nebel, Lichtschächte): Dichte je Welteinheit auf
    /// Höhe `fog_height`, darüber exponentiell abnehmend mit `fog_falloff`.
    fog_density: f32,
    fog_color: [3]f32,
    fog_height: f32,
    fog_falloff: f32,
    /// Streurichtung nach Henyey-Greenstein: 0 = gleichmäßig, >0 nach vorn
    fog_anisotropy: f32,
    /// Schritte der Strahlmarschierung; 0 = 12
    fog_steps: u32,
    /// Obergrenze für den Beitrag einer einzelnen Abtastung (0 = aus).
    /// Sehr helle, sehr kleine Lichter liefern sonst vereinzelte Ausreißer,
    /// die jeden Frame woanders sitzen und sichtbar flimmern.
    firefly_clamp: f32,
    /// Schatten von Wolken (oder anderem sehr Fernem): eine Textur, entlang
    /// der Sonnenrichtung auf eine Ebene in Höhe `sun_shadow_height`
    /// projiziert. Ihr Rotanteil dämpft das Sonnenlicht um bis zu
    /// `sun_shadow_strength`. `sun_shadow_scale` Welteinheiten je
    /// Texturwiederholung; `sun_shadow_offset` verschiebt das Muster in x/z,
    /// damit es beim Nachführen des Render-Ursprungs an der Welt haften
    /// bleibt (Ursprung modulo scale eintragen). 0 = keine.
    sun_shadow_texture: u32,
    sun_shadow_height: f32,
    sun_shadow_scale: f32,
    sun_shadow_strength: f32,
    sun_shadow_offset: [2]f32,
    reserved: [1]u32,
    lights: [max_lights]Light,
};

// ---------------------------------------------------------------------------
// Parameter der Runtime-Kernel
// ---------------------------------------------------------------------------

pub const CameraData = extern struct {
    camera: Camera,
    world_to_view: [12]f32,
};

pub const trace_any_hit: u32 = 0x1;
pub const trace_no_attribute: u32 = 0x2;
/// pyr_trace schreibt HitEx statt Hit (Voxel, Position, Normale; z. B. Picking)
pub const trace_extended: u32 = 0x4;
/// Voxel mit material_transparent überspringen (Primärstrahlen, Schatten):
/// die Traversierung läuft weiter bis zum nächsten undurchsichtigen Voxel
pub const trace_skip_transparent: u32 = 0x8;

pub const HitEx = extern struct {
    hit: Hit,
    /// Voxelkoordinate im Objektraum der Geometrie
    voxel: [3]i32,
    reserved0: u32,
    /// Trefferpunkt in Weltkoordinaten (relativ zum Render-Ursprung)
    position: [3]f32,
    reserved1: u32,
    /// Weltnormale
    normal: [3]f32,
    reserved2: u32,
};

pub const RenderParams = extern struct {
    /// Abtastungen je Pixel für die *Deckung* (1..4, 1 = aus). Zusätzliche
    /// Primärstrahlen sind billig; geschattet wird nur, was sich unterscheidet.
    coverage: u32,
    scene: u64,
    cur: CameraData,
    prev: CameraData,
    history_valid: u32,
    ray_mask: u32,
    flags: u32,
    reserved: u32,
    /// Hit[width * height] oder 0
    hits: u64,
    /// f32[width * height], lineare Tiefe entlang der Blickachse
    depth: u64,
    /// f32[2 * width * height], Pixel, vorher − jetzt, ohne Jitter
    motion: u64,
    /// [4]f32 pro Pixel: lineare HDR-Farbe (a = 1 Treffer, 0 Himmel); 0 = kein Shading
    color: u64,
    /// [4]f32 pro Pixel: Weltnormale, w = lineare Tiefe
    normal: u64,
    /// [4]f32 pro Pixel: Albedo (für Denoiser/Upscaler)
    albedo: u64,
    /// [2]f32 pro Pixel: Rauheit und Metallanteil (für DLSS Ray Reconstruction)
    material: u64,
    /// Frame-Zähler für Zufallszahlen
    frame_index: u32,
    /// Instanzen mit (mask & transparent_mask) != 0 bilden die transparente Ebene
    transparent_mask: u32,
    /// Maske für Schatten-, GI- und Reflexionsstrahlen; 0 = wie ray_mask.
    /// Damit sehen Sekundärstrahlen z. B. nur die gröbere Fassung der Welt.
    secondary_mask: u32,
    /// halbe Auflösung für die indirekte Beleuchtung (lighting_gi_half):
    /// [4]f16 je Halbpixel, Strahldichte + Gültigkeit
    gi: u64,
    gi_width: u32,
    gi_height: u32,
};

pub const TraceParams = extern struct {
    scene: u64,
    rays: u64,
    hits: u64,
    count: u32,
    ray_mask: u32,
    flags: u32,
    reserved: u32,
};

pub const InstanceWrite = extern struct {
    index: u32,
    reserved: [3]u32,
    data: InstanceData,
};

/// Blockgrößen der Runtime-Kernel (zur Übersetzungszeit fest, auch für AMD nötig)
pub const render_block_x: u32 = 8;
pub const render_block_y: u32 = 8;
pub const trace_block: u32 = 128;
pub const update_block: u32 = 256;

// ---------------------------------------------------------------------------
// RT-Cores (OptiX): Datentypen des RT-Pfads
// ---------------------------------------------------------------------------

/// Layout von OptixInstance; wird auf der GPU aus dem Instanzzustand erzeugt.
pub const RtInstance = extern struct {
    transform: [12]f32,
    instance_id: u32,
    sbt_offset: u32,
    visibility_mask: u32,
    flags: u32,
    traversable: u64,
    pad: [2]u32,
};

/// Ein belegter Teilbaum der Kantenlänge 2^rt_log2 = ein AABB-Primitiv der
/// Hardware-BVH. Der Intersection-Shader verfolgt nur diesen Teil-DAG.
pub const RtPrim = extern struct {
    /// Wurzel des Teil-DAG (Wortoffset relativ zur Geometrie)
    node: u32,
    /// Attributrang des ersten Voxels des Teilbaums
    attr_base: u32,
    /// Zellkoordinate in Einheiten von 2^rt_log2: x | y << 21 | z << 42
    cell: u64,
};

/// Daten eines Hitgroup-SBT-Eintrags (eine Geometrie); Zeiger sind absolut.
pub const RtGeometry = extern struct {
    nodes: u64,
    leaves: u64,
    /// 0 = keine Attribute
    attributes: u64,
    prims: u64,
    rt_log2: u32,
    default_attribute: u32,
    reserved: [2]u32,
};

pub const rt_sbt_header_size = 32;

pub const RtHitRecord = extern struct {
    header: [rt_sbt_header_size]u8 align(16),
    data: RtGeometry,
};

/// Launch-Parameter (OptiX-Konstantenspeicher, Name "pyr_rt_params")
pub const RtParams = extern struct {
    /// IAS aller Instanzen; 0 = leere Szene
    handle: u64,
    scene: u64,
    /// trace_* dieses Launches
    flags: u32,
    reserved: u32,
    render: RenderParams,
    trace: TraceParams,
};

// ---------------------------------------------------------------------------
// Nachbearbeitung (temporale Akkumulation/TAA, Denoiser, Tonemapping)
// ---------------------------------------------------------------------------

pub const post_block: u32 = 8;

pub const PostParams = extern struct {
    width: u32,
    height: u32,
    /// Eingaben aus pyr_render
    color: u64,
    normal: u64,
    albedo: u64,
    motion: u64,
    hits: u64,
    /// Verlauf des Vorframes (Bestrahlungsstärke + Anzahl) und seine Normale/Tiefe
    hist_color: u64,
    hist_normal: u64,
    /// Ausgaben
    out_color: u64,
    out_normal: u64,
    /// Filterpuffer
    src: u64,
    dst: u64,
    step: u32,
    /// kleinstes Mischgewicht des aktuellen Frames (1 / maximale Anzahl)
    alpha_min: f32,
    /// Varianzbegrenzung des Verlaufs in Standardabweichungen; 0 = aus
    clamp_sigma: f32,
    /// 1 = Verlauf verwerfen
    reset: u32,
    exposure: f32,
    tonemap: u32,
    /// Ausgaben des letzten Schritts
    out_hdr: u64,
    out_ldr: u64,
    /// out_hdr halbgenau ablegen (interner TAAU-Eingang)
    hdr_half: u32,
    /// out_ldr als BGRA statt RGBA (Wayland/X11, Windows-Fenster)
    bgra: u32,
    /// Helligkeitsmomente (Mittel, Quadratmittel) des Verlaufs; 0 = aus
    hist_moments: u64,
    out_moments: u64,
    /// Varianz der Helligkeit: Ausgabe der Akkumulation, Ein-/Ausgabe des Filters
    out_var: u64,
    var_src: u64,
    var_dst: u64,
    /// Stärke der Varianzführung im À-trous-Filter (SVGF: 4)
    phi_lum: f32,
    /// Unterhalb dieser Normalenübereinstimmung wird der Verlauf ganz
    /// verworfen; darüber geht er gewichtet ein (0,9 = altes, hartes Verhalten)
    normal_reject: f32,
    /// Höchstzahl gemittelter Frames auf Voxelkanten (dort mittelt der Jitter
    /// zwei Flächen; lange Mittelung würde nachziehen)
    edge_frames: f32,
};

pub const tonemap_aces: u32 = 0;
pub const tonemap_reinhard: u32 = 1;
pub const tonemap_none: u32 = 2;
/// ACES RRT+ODT als Anpassung (Hill): wirkt auf die Farbe als Ganzes,
/// entsättigt helle Farben zum Weiß hin und hat eine weiche Schulter
pub const tonemap_aces_fitted: u32 = 3;
/// Khronos PBR Neutral: bis 0,76 linear (kein Fuß, Schatten bleiben so hell,
/// wie Albedo und Licht es sagen), darüber weich komprimiert und zum Weiß
/// hin entsättigt
pub const tonemap_neutral: u32 = 4;

// ---------------------------------------------------------------------------
// DAG-Bau auf der GPU (src/device/gbuild.zig)
// ---------------------------------------------------------------------------

pub const build_block: u32 = 256;

pub const BuildParams = extern struct {
    op: u32,
    log2_size: u32,
    rt_log2: u32,
    /// Kindgröße der aktuellen Ebene als log2 (Bricks: 2)
    level: u32,
    pass: u32,
    chunk: u32,
    chunks: u32,
    n_old: u32,
    n_new: u32,
    n_total: u32,
    /// Länge des Scans: Zählerindex oder, wenn 0xFFFFFFFF, scan_len
    scan_len_slot: u32,
    scan_len: u32,
    scan_result_slot: u32,
    hash_cap: u32,
    /// Zähler (u32[]), siehe gbuild.c_*
    counts: u64,
    edits: u64,
    keys_in: u64,
    vals_in: u64,
    keys_out: u64,
    vals_out: u64,
    hist: u64,
    scan_data: u64,
    scan_totals: u64,
    flags: u64,
    ukey: u64,
    uval: u64,
    masks: u64,
    ch_key: u64,
    ch_ref: u64,
    ch_count: u64,
    ch_lo: u64,
    ch_hi: u64,
    par_key: u64,
    par_ref: u64,
    par_count: u64,
    par_lo: u64,
    par_hi: u64,
    node_tmp: u64,
    node_len: u64,
    hash_keys: u64,
    hash_owner: u64,
    slot: u64,
    nodes_out: u64,
    leaves_out: u64,
    prims_out: u64,
    aabbs_out: u64,
    /// Verkleinern: Quellvoxel (Schlüssel, Attribute) und Faktor 2^shift
    src_keys: u64,
    src_vals: u64,
    shift: u32,
    /// Chunk-Batch: Anzahl Chunks und Bits der Chunknummer im Schlüssel (0 = ein Chunk)
    chunk_bits: u32,
    chunk_count: u32,
    /// Kapazität je Chunk im Eingabepuffer (Voxel); Anfang der belegten
    /// Einträge je Chunk in der dichten Zählung u32[K + 1] (exklusive Präfixsumme)
    chunk_capacity: u32,
    chunk_offsets: u64,
    /// Ausgaben je Chunk: erstes Voxel (Attributrang), Wurzel, erstes Primitiv
    chunk_first: u64,
    roots_out: u64,
    prim_start: u64,
};

// ---------------------------------------------------------------------------
// Welt-Streaming: Generatoren
// ---------------------------------------------------------------------------

/// Chunk (lod, x, y, z): Kantenlänge 2^chunk_log2 Voxel zu je 2^lod Grundvoxeln,
/// Ursprung (x, y, z) * 2^(chunk_log2 + lod) in Grundvoxeln.
pub const ChunkKey = extern struct {
    x: i32,
    y: i32,
    z: i32,
    lod: u32,
};

/// Auftrag an einen Generator: für jeden Chunk c Voxel [x, y, z, attribut] in
/// Chunk-lokalen Koordinaten [0, 2^chunk_log2) nach
/// voxels[c * capacity + atomicAdd(&counts[c], 1)] schreiben (nur wenn der
/// Index < capacity ist). counts ist beim Aufruf 0.
pub const WorldGenParams = extern struct {
    chunks: u64,
    voxels: u64,
    counts: u64,
    count: u32,
    capacity: u32,
    chunk_log2: u32,
    reserved: u32,
    user: u64,
};

/// Änderungen an einer gestreamten Welt (src/device/worldedit.zig).
/// Die Einträge liegen je Chunk gruppiert, `edit_offsets` hat count+1 Werte;
/// ein Eintrag ist [x, y, z, attribut] in Chunk-lokalen Koordinaten der Stufe,
/// Attribut 0 bedeutet entfernen.
pub const WorldEditParams = extern struct {
    voxels: u64,
    counts: u64,
    out_voxels: u64,
    out_counts: u64,
    edits: u64,
    edit_offsets: u64,
    edit_used: u64,
    count: u32,
    capacity: u32,
};

pub const edit_block: u32 = 128;

// ---------------------------------------------------------------------------
// Hochskalieren (TAAU) und Frame Generation
// ---------------------------------------------------------------------------

/// Kamera- und Bildeffekte (src/device/postfx.zig). Ein Aufruf je Stufe; die
/// Stufen teilen sich die Struktur, nicht benutzte Felder bleiben 0.
pub const PostFxParams = extern struct {
    /// Auflösung der Quelle
    width: u32,
    height: u32,
    /// Auflösung des Ziels (Bloom-Stufen halbieren)
    dst_width: u32,
    dst_height: u32,
    /// HDR-Quelle (4 x f32, außer in der Bloom-Pyramide)
    color: u64,
    /// Ziel der laufenden Stufe
    dst: u64,
    /// Quelle liegt halbgenau vor
    src_half: u32,
    reserved0: u32,
    /// Bewegung (xy, Ausgabepixel) und Tiefe, in Ausgabeauflösung
    mvd: u64,
    /// Quellen für packMvd: Bewegung (2 x f32) und Normale+Tiefe (4 x f32) in
    /// Renderauflösung (dst_width x dst_height)
    mv_src: u64,
    normal_src: u64,

    /// Tiefenschärfe
    dof_focus: f32,
    dof_strength: f32,
    dof_max_coc: f32,
    /// Schärfeebene aus der Tiefe in der Bildmitte nehmen
    dof_autofocus: u32,
    /// nur hinter der Schärfeebene unscharf zeichnen (Vordergrund bleibt scharf)
    dof_far_only: u32,
    /// Bewegungsunschärfe
    blur_scale: f32,
    blur_max: f32,
    blur_samples: u32,

    /// Bloom
    bloom: u64,
    bloom_width: u32,
    bloom_height: u32,
    bloom_strength: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
    reserved1: u32,

    /// Belichtungsautomatik: Zähler (2 x u32) und Zustand (Ziel, aktuell)
    expose_acc: u64,
    expose_state: u64,
    expose_speed: f32,
    expose_min: f32,
    expose_max: f32,
    expose_compensation: f32,

    /// Farbkorrektur
    grade: u32,
    temperature: f32,
    tint: f32,
    contrast: f32,
    saturation: f32,
    lift: [3]f32,
    gamma: [3]f32,
    gain: [3]f32,
    lut: u64,
    lut_size: u32,

    /// Ausgabe
    exposure: f32,
    tonemap: u32,
    bgra: u32,
    out_hdr: u64,
    out_ldr: u64,
    /// Supersampling: die ganze Kette läuft in `supersample`-facher Auflösung,
    /// der letzte Schritt mittelt je supersample² Pixel zusammen. 0/1 = aus.
    supersample: u32,
    out_width: u32,
    out_height: u32,
};

pub const postfx_block: u32 = 256;

pub const UpscaleParams = extern struct {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    /// Renderauflösung: HDR-Farbe (entrauscht, halbgenau wenn color_half),
    /// Normale + Tiefe, MV, Treffer
    color: u64,
    color_half: u32,
    reserved: u32,
    normal: u64,
    motion: u64,
    hits: u64,
    /// Jitter des aktuellen Frames in Renderpixeln
    jitter: [2]f32,
    /// Ausgabeauflösung: Verlauf (rgb + akkumuliertes Gewicht)
    hist_in: u64,
    hist_out: u64,
    /// Ausgabeauflösung: MV (Ausgabepixel) + Tiefe für die Frame Generation
    mvd_out: u64,
    reset: u32,
    /// größtes Verlaufsgewicht (= 1 / kleinstes Mischgewicht)
    max_weight: f32,
    exposure: f32,
    tonemap: u32,
    out_hdr: u64,
    out_ldr: u64,
    /// out_ldr als BGRA statt RGBA (Fenstersysteme)
    bgra: u32,
    /// Varianzbegrenzung des Verlaufs in Standardabweichungen (YCoCg)
    clamp_sigma: f32,
    /// Schärfe des Rekonstruktionskerns: groß = schmal (nur das nächste
    /// Sample zählt), klein = breit (mittelt über Nachbarn)
    kernel_sharp: f32,
};

pub const upscale_block: u32 = 8;

/// LDR-Ausgabe als BGRA (Fenstersysteme) statt RGBA
pub const post_bgra: u32 = 0x4;

/// Frame Generation: Zwischenbild zur Zeit t zwischen Vorframe (0) und
/// aktuellem Frame (1). Eigene Kernel bekommen dieselbe Struktur.
pub const FrameGenParams = extern struct {
    width: u32,
    height: u32,
    t: f32,
    exposure: f32,
    /// [4]f32 HDR linear, Ausgabeauflösung
    prev_color: u64,
    cur_color: u64,
    /// [4]f32: MV aktuell -> Vorframe in Pixeln (xy), Tiefe (z)
    motion_depth: u64,
    tonemap: u32,
    reserved: u32,
    /// Ausgaben (0 = nicht schreiben): [4]f32 HDR, [4]u8 sRGB
    out_hdr: u64,
    out_ldr: u64,
    user: u64,
    /// Zwischenspeicher für die Vorwärtsprojektion: kleinste Tiefe je
    /// Zwischenbildpixel (u32, Bitmuster von f32) und der zugehörige MV
    depth_mid: u64,
    mv_mid: u64,
    /// out_ldr als BGRA statt RGBA (Fenstersysteme)
    bgra: u32,
    reserved_bgra: u32,
};

// ---------------------------------------------------------------------------
// Animation: Skelette aus Voxel-Teilen, Keyframe-Clips, auf der GPU abgespielt
// ---------------------------------------------------------------------------

pub const AnimBone = extern struct {
    /// Elternknochen (Index im Skelett, < eigener Index) oder -1
    parent: i32,
    /// Kantenlänge des Voxel-Teils in Objekteinheiten (für die AABB); 0 = keiner
    part_size: f32,
    /// Anteil an der Windbewegung (0 = starr, 1 = volle Auslenkung). Jeder
    /// Knochen der Kette biegt sich ein Stück, zusammen ergibt das ein weiches
    /// Schwanken – exakt affin, also mit exakten Motion Vectors.
    sway: f32,
    reserved: u32,
    /// Ruhelage relativ zum Elternknochen (3x4), gilt ohne Keyframes
    rest: [12]f32,
    /// Voxel-Teil relativ zum Knochen (3x4), z. B. Verschiebung zum Drehpunkt
    part: [12]f32,
};

pub const Keyframe = extern struct {
    bone: u32,
    time: f32,
    translation: [3]f32,
    scale: f32,
    /// Quaternion (x, y, z, w)
    rotation: [4]f32,
};

pub const clip_loop: u32 = 0x1;

pub const AnimClip = extern struct {
    /// je Knochen [erster Key, Anzahl] ab track_offset
    track_offset: u32,
    bone_count: u32,
    duration: f32,
    flags: u32,
};

pub const no_clip: u32 = 0xFFFF_FFFF;

pub const AnimActor = extern struct {
    root: [12]f32,
    bone_offset: u32,
    bone_count: u32,
    clip: u32,
    blend_clip: u32,
    start_time: f32,
    speed: f32,
    blend_start_time: f32,
    blend_speed: f32,
    /// Gewicht von blend_clip (0 = nur clip)
    blend: f32,
    /// Wind: Auslenkung in Radiant, Frequenz in Hz, Phase (z. B. aus der Lage)
    wind_amplitude: f32,
    wind_frequency: f32,
    wind_phase: f32,
};

/// ein Knochen mit Voxel-Teil eines Akteurs -> eine Instanz
pub const AnimJob = extern struct {
    actor: u32,
    bone: u32,
    instance: u32,
    reserved: u32,
};

pub const anim_block: u32 = 128;

pub const AnimParams = extern struct {
    bones: u64,
    tracks: u64,
    keys: u64,
    clips: u64,
    actors: u64,
    jobs: u64,
    /// InstanceData[] des aktuellen Frames
    instances: u64,
    job_count: u32,
    reserved: u32,
    /// Szenenzeit (PyrFrameInfo.time)
    time: f64,
};
