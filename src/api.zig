//! Typen und Konstanten der Host-API (C-ABI, siehe include/pyrit.h).

const types = @import("pyrit_device").types;

pub const version_major: u32 = 0;
pub const version_minor: u32 = 1;
pub const version: u32 = (version_major << 16) | version_minor;

pub const Result = i32;
pub const ok: Result = 0;
pub const error_invalid_argument: Result = -1;
pub const error_invalid_handle: Result = -2;
pub const error_out_of_memory: Result = -3;
pub const error_capacity: Result = -4;
pub const error_in_use: Result = -5;
pub const error_cuda: Result = -6;
pub const error_compile: Result = -7;
pub const error_not_found: Result = -8;
pub const error_version: Result = -9;

/// Undurchsichtige Handles (Index + Generation im Zeigerwert)
pub const Handle = ?*opaque {};

pub const LogFn = ?*const fn (user: ?*anyopaque, level: i32, message: [*:0]const u8) callconv(.c) void;

/// zusätzliche Prüfungen, synchrone Fehlerberichte
pub const create_debug: u32 = 0x1;
/// Nachbearbeitung überlappt mit dem nächsten Frame: DLSS und TAA laufen auf
/// den Tensorkernen, während die Shader-Einheiten schon weiterrechnen. Dafür
/// muss die Anwendung zwei Sätze von PyrTargets abwechselnd benutzen, sonst
/// überschreibt der nächste Frame, woraus die Nachbearbeitung noch liest.
pub const create_async_post: u32 = 0x40;
/// RT-Cores nicht verwenden, immer CUDA-Traversierung
pub const create_no_rt: u32 = 0x2;
/// RT-Cores immer verwenden (sonst erst ab einigen Instanzen, wo sie schneller sind)
pub const create_force_rt: u32 = 0x4;

/// Rückgabe von pyr_features
pub const feature_rt_cores: u32 = 0x1;

pub const CreateInfo = extern struct {
    struct_size: u32,
    version: u32,
    /// CUDA-Geräteindex
    device: i32,
    flags: u32,
    /// vorhandener CUcontext oder null (primärer Kontext)
    cuda_context: ?*anyopaque,
    /// vorhandener CUstream oder null (eigener Stream)
    cuda_stream: ?*anyopaque,
    /// 0 = 65536
    max_instances: u32,
    /// 0 = 4096
    max_geometries: u32,
    /// 0 = 16
    max_views: u32,
    /// Texturplätze, 0 = 256
    max_textures: u32,
    /// Kantenlänge der AABB-Primitive der RT-Cores als log2 (3..), 0 = 7 (128^3)
    rt_leaf_log2: u32,
    /// 0 = 256 MiB
    node_pool_bytes: u64,
    leaf_pool_bytes: u64,
    attribute_pool_bytes: u64,
    /// gepinnter Upload-Puffer, 0 = 64 MiB
    staging_bytes: u64,
    log: LogFn,
    log_user: ?*anyopaque,
};

/// nur Geometrie speichern; Treffer liefern das Attribut 1
pub const dag_no_attributes: u32 = 0x1;

pub const Voxel = extern struct {
    x: i32,
    y: i32,
    z: i32,
    /// != 0
    attribute: u32,
};

/// Attribut des Voxels (x, y, z); 0 = leer
pub const VoxelFn = ?*const fn (user: ?*anyopaque, x: i32, y: i32, z: i32) callconv(.c) u32;
/// Optional: Ist der Würfel [x, x+size)^3 sicher leer? Rückgabe != 0 überspringt ihn.
pub const RegionEmptyFn = ?*const fn (user: ?*anyopaque, x: i32, y: i32, z: i32, size: u32) callconv(.c) i32;

pub const DagInfo = extern struct {
    log2_size: u32,
    root: u32,
    voxel_count: u64,
    nodes: ?[*]const u32,
    node_words: u64,
    leaves: ?[*]const u64,
    leaf_count: u64,
    /// null bei dag_no_attributes
    attributes: ?[*]const u32,
    attribute_count: u64,
};

pub const FrameInfo = extern struct {
    /// Sekunden, monoton
    time: f64,
    /// Render-Ursprung in Weltkoordinaten
    origin: [3]f64,
};

pub const Targets = extern struct {
    /// Hit[width * height] oder 0
    hits: u64,
    /// f32[width * height] oder 0
    depth: u64,
    /// f32[2 * width * height] oder 0
    motion: u64,
    /// [4]f32 pro Pixel: lineare HDR-Farbe (a = 1 Treffer, 0 Himmel); 0 = kein Shading
    color: u64,
    /// [4]f32 pro Pixel: Weltnormale, w = lineare Tiefe
    normal: u64,
    /// [4]f32 pro Pixel: Albedo
    albedo: u64,
    /// [2]f32 pro Pixel: Rauheit, Metallanteil (nur für DLSS Ray Reconstruction)
    material: u64,
    /// 0 = alle Instanzen
    ray_mask: u32,
    /// trace_no_attribute
    flags: u32,
    /// Instanzmaske der transparenten Ebene (Wasser, Glas); 0 = keine
    transparent_mask: u32,
    /// Maske für Schatten-, GI- und Reflexionsstrahlen; 0 = wie ray_mask
    secondary_mask: u32,
};

// ---------------------------------------------------------------------------
// Nachbearbeitung
// ---------------------------------------------------------------------------

/// Verlauf verwerfen (Schnitt, Teleport)
pub const post_reset: u32 = 0x1;
/// keine temporale Akkumulation
pub const post_no_temporal: u32 = 0x2;
/// output_ldr als BGRA statt RGBA (Fenstersysteme wie X11)
pub const post_bgra: u32 = 0x4;

pub const PostInfo = extern struct {
    /// [4]f32 pro Pixel oder 0
    output_hdr: u64,
    /// [4]u8 pro Pixel (sRGB, RGBA) oder 0
    output_ldr: u64,
    /// 0 = 1
    exposure: f32,
    /// tonemap_* (aces, reinhard, none)
    tonemap: u32,
    /// Schritte des À-trous-Filters, 0 = kein räumlicher Filter
    denoise_iterations: u32,
    /// kleinstes Gewicht des aktuellen Frames, 0 = 0.05
    temporal_alpha: f32,
    /// Varianzbegrenzung (TAA) in Standardabweichungen, 0 = aus
    clamp_sigma: f32,
    /// post_*
    flags: u32,
    /// Ausgabeauflösung (output_hdr/ldr); 0 = Renderauflösung
    output_width: u32,
    output_height: u32,
    /// upscaler_*
    upscaler: u32,
    /// Stärke der Varianzführung im Denoiser, 0 = 4 (kleiner = glatter)
    denoise_phi: f32,
    /// Kamera- und Bildeffekte; NULL = keine
    fx: ?*const PostFx,
};

// Kamera- und Bildeffekte (PyrPostFx.flags)
pub const postfx_bloom: u32 = 0x1;
pub const postfx_dof: u32 = 0x2;
pub const postfx_motion_blur: u32 = 0x4;
pub const postfx_auto_exposure: u32 = 0x8;
pub const postfx_grade: u32 = 0x10;
/// Schärfeebene aus der Tiefe in der Bildmitte nachführen
pub const postfx_autofocus: u32 = 0x20;
/// nur hinter der Schärfeebene unscharf zeichnen, Vordergrund bleibt scharf
pub const postfx_dof_far_only: u32 = 0x40;

/// Kamera- und Bildeffekte auf dem fertigen Bild. Alle 0 = aus; die Werte
/// unten sind die Vorgaben, wenn ein Feld 0 bleibt.
pub const PostFx = extern struct {
    /// postfx_*
    flags: u32,
    /// Bloom
    bloom_strength: f32 = 0,
    bloom_threshold: f32 = 0,
    bloom_knee: f32 = 0,
    bloom_levels: u32 = 0,
    /// Tiefenschärfe: Entfernung der Schärfeebene (mit postfx_autofocus egal),
    /// Zerstreuungskreis in Pixeln bei unendlich (0 = 3) und dessen Obergrenze
    focus_distance: f32 = 0,
    dof_strength: f32 = 0,
    dof_max_coc: f32 = 0,
    /// Bewegungsunschärfe: Verschlussanteil, größte Strecke, Abtastungen
    motion_blur_scale: f32 = 0,
    motion_blur_max: f32 = 0,
    motion_blur_samples: u32 = 0,
    /// Belichtungsautomatik
    exposure_speed: f32 = 0,
    exposure_min: f32 = 0,
    exposure_max: f32 = 0,
    /// Korrektur in Blendenstufen
    exposure_compensation: f32 = 0,
    /// Farbkorrektur (nur mit postfx_grade)
    temperature: f32 = 0,
    tint: f32 = 0,
    contrast: f32 = 0,
    saturation: f32 = 0,
    lift: [3]f32 = .{ 0, 0, 0 },
    gamma: [3]f32 = .{ 1, 1, 1 },
    gain: [3]f32 = .{ 1, 1, 1 },
    /// 3D-LUT im Anzeigeraum: RGBA8, Kantenlänge lut_size, Reihenfolge x+y·n+z·n²
    lut: u64 = 0,
    lut_size: u32 = 0,
    reserved_fx: u32 = 0,
};

/// TAAU (auch ohne Skalierung: TAA mit Subpixel-Rekonstruktion)
pub const upscaler_auto: u32 = 0;
/// direkt in Renderauflösung auflösen (ohne TAAU)
pub const upscaler_none: u32 = 1;
pub const upscaler_taau: u32 = 2;
/// NVIDIA DLSS Super Resolution (NGX über CUDA)
pub const upscaler_dlss: u32 = 3;
/// NVIDIA DLSS Ray Reconstruction: ersetzt Denoiser und TAAU
pub const upscaler_dlss_rr: u32 = 4;

pub const FrameGenParams = types.FrameGenParams;
/// Eigene Frame Generation: Kernel auf `stream` starten, der params.out_*
/// schreibt (Eingaben siehe PyrFrameGenParams).
pub const FrameGenFn = ?*const fn (user: ?*anyopaque, params: *const FrameGenParams, stream: ?*anyopaque) callconv(.c) void;

pub const FrameGenInfo = extern struct {
    /// Ausgaben in Ausgabeauflösung (wie PostInfo)
    output_hdr: u64,
    output_ldr: u64,
    /// Zeitpunkt zwischen Vorframe (0) und letztem Frame (1); 0 = 0.5
    t: f32,
    flags: u32,
    /// null = eingebaute Frame Generation
    generate: FrameGenFn,
    user: ?*anyopaque,
};

// ---------------------------------------------------------------------------
// Statistik
// ---------------------------------------------------------------------------

pub const Stats = extern struct {
    node_pool_used: u64,
    node_pool_capacity: u64,
    leaf_pool_used: u64,
    leaf_pool_capacity: u64,
    attribute_pool_used: u64,
    attribute_pool_capacity: u64,
    geometries: u32,
    instances: u32,
    frame: u64,
    features: u32,
    reserved: u32,
};

// ---------------------------------------------------------------------------
// DAG-Bau auf der GPU
// ---------------------------------------------------------------------------

/// voxels zeigt auf Host-Speicher (wird hochgeladen); sonst Gerätezeiger
pub const build_host_input: u32 = 0x1;
/// Voxelliste behalten, damit pyr_geometry_edit möglich ist
pub const build_editable: u32 = 0x2;

// ---------------------------------------------------------------------------
// Große Welten (Streaming mit LOD, Erzeugung auf der GPU)
// ---------------------------------------------------------------------------

pub const ChunkKey = types.ChunkKey;
pub const WorldGenParams = types.WorldGenParams;
pub const TerrainInfo = types.TerrainParams;

/// Eigener Generator: startet Kernel auf `stream`, die params.voxels/counts
/// füllen (siehe PyrWorldGenParams). Läuft ganz auf der GPU; kein Warten nötig.
pub const WorldGenFn = ?*const fn (user: ?*anyopaque, params: *const WorldGenParams, stream: ?*anyopaque) callconv(.c) void;

/// Erzeugen und Bauen im Aufruf von pyr_world_update statt im Hintergrund
/// (deterministisch, für Tests und Werkzeuge)
pub const world_sync: u32 = 0x1;

pub const WorldInfo = extern struct {
    /// Chunkkante als log2 (3..8), 0 = 5 (32^3 Voxel je Chunk und Stufe)
    chunk_log2: u32,
    /// Obergrenze der Baumtiefe (Sicherheitsnetz), 0 = 20
    max_lod: u32,
    /// Sichtradius der gröbsten Stufe in Chunks, 0 = 2
    view_chunks: u32,
    /// Zielgröße eines Voxels auf dem Bildschirm in Pixeln; darüber wird
    /// verfeinert. 0 = 4. Kleiner = feiner und mehr Speicher.
    voxel_pixels: f32,
    /// Sichtweite in Grundvoxeln; 0 = 16384
    view_distance: f32,
    reserved0: u32,
    /// Obergrenze für den Chunk-Speicher in Bytes; darüber wird die Welt
    /// gleitend gröber (die Zielgröße der Voxel wächst). 0 = 256 MiB.
    memory_budget: u64,
    /// senkrechter Bereich in Grundvoxeln [y_min, y_max); beide 0 = aus dem Gelände
    y_min: i32,
    y_max: i32,
    /// Chunks je pyr_world_update, 0 = 64
    chunks_per_update: u32,
    /// Instanzmaske der Welt, 0 = 0x1
    mask: u32,
    /// Zusätzliche Maske für eine gröbere Fassung der Welt, die nur
    /// Sekundärstrahlen (Schatten, GI, Reflexionen) sehen; 0 = aus.
    /// In PyrTargets.secondary_mask eintragen.
    secondary_mask: u32,
    /// Zielgröße eines Voxels in dieser groben Fassung, 0 = 4 · voxel_pixels
    secondary_pixels: f32,
    /// Voxel je Chunk im Generatorpuffer, 0 = 8 · chunk² (Oberflächenhaut)
    chunk_capacity: u32,
    /// Kantenlänge der RT-Primitive als log2, 0 = chunk_log2 - 2 (mindestens 3)
    rt_leaf_log2: u32,
    /// Frames, die ein nicht mehr gebrauchter Chunk bleibt, 0 = 8
    keep_frames: u32,
    flags: u32,
    /// null = eingebautes Gelände (terrain)
    generate: WorldGenFn,
    user: ?*anyopaque,
    /// null = Standardgelände
    terrain: ?*const TerrainInfo,
};

/// Eine Änderung an der Welt: ein Grundvoxel setzen oder entfernen.
/// Koordinaten in Grundvoxeln (volle Auflösung), unabhängig von der Stufe,
/// in der der Chunk gerade vorliegt.
pub const WorldEdit = extern struct {
    x: i64,
    y: i64,
    z: i64,
    /// Voxelattribut; 0 entfernt den Voxel
    attribute: u32,
    reserved: u32 = 0,
};

pub const WorldStats = extern struct {
    visible_chunks: u32,
    resident_chunks: u32,
    pending_chunks: u32,
    built_chunks: u32,
    /// Voxel im letzten Update erzeugt
    built_voxels: u32,
    /// Chunks, deren Generatorpuffer übergelaufen ist (Voxel fehlen)
    overflow_chunks: u32,
    /// aktuelle Zielgröße eines Voxels in Pixeln (mit Budget angepasst)
    voxel_pixels: f32,
    /// gröbste benutzte Stufe
    top_lod: u32,
    /// empfohlener PyrLighting.secondary_bias für secondary_mask
    secondary_bias: f32,
    reserved1: u32,
    /// GPU-Bytes der Welt (Knoten, Blätter, Attribute)
    bytes: u64,
};
