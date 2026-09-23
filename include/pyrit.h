/*
 * pyrit.h – Host-API der Pyrit-Runtime (Version 0.1, reines CUDA)
 *
 * Die Bibliothek ist in Zig geschrieben; dieser Header beschreibt ihre C-ABI für
 * Aufrufer in C, C++, Rust, Java (FFM) usw. Zig-Programme verwenden die
 * Zig-Module direkt. Eigene GPU-Kernel schreibt man in Zig mit dem Modul
 * pyrit_device (pyr.trace, pyr.Scene, ...).
 *
 * Geometrie besteht ausschließlich aus Sparse Voxel DAGs. Die Runtime liefert
 * Treffer, Tiefe und exakte Motion Vectors pro Pixel in CUDA-Puffer der
 * Anwendung und stellt Strahlverfolgung per pyr_trace bereit.
 *
 * Ablauf:
 *
 *   PyrContext* ctx;   pyr_create(&info, &ctx);
 *   PyrDag* dag;       pyr_dag_build_fn(8, voxel_fn, NULL, user, 0, &dag);   // CPU, ohne GPU
 *   PyrGeometry geo;   pyr_geometry_create(ctx, dag, &geo);  pyr_dag_destroy(dag);
 *   PyrInstance inst;  pyr_instance_create(ctx, geo, &inst);
 *   PyrView view;      pyr_view_create(ctx, &view);
 *
 *   pro Frame:
 *     pyr_instance_set_transform(ctx, inst, m);      // beliebig viele Änderungen
 *     pyr_commit(ctx, &frame);                       // neuer Frame, Änderungen werden aktiv
 *     pyr_render(ctx, view, &camera, &targets);      // Treffer, Tiefe, Motion Vectors
 *     pyr_trace(ctx, rays, hits, n, mask, flags);    // beliebige weitere Strahlen
 *
 * Alle GPU-Arbeit läuft asynchron auf dem Stream des Kontexts (pyr_cuda_stream).
 * Funktionen eines Kontexts sind nicht threadsicher; pyr_dag_* sind es.
 */
#ifndef PYRIT_H
#define PYRIT_H

#include <stddef.h>
#if defined(__cplusplus)
#  include <cstdint>
#else
#  include <stdint.h>
#endif

#define PYR_FLT_MAX 3.402823466e+38f

/* ---------------------------------------------------------------------------
 * Strahlen und Treffer
 * ------------------------------------------------------------------------- */

typedef struct PyrRay {
    float origin[3];
    float tmin;
    float direction[3];   /* muss nicht normiert sein; t wird in Vielfachen davon gemessen */
    float tmax;
} PyrRay;

#define PYR_NO_HIT 0xFFFFFFFFu

/* Fläche des getroffenen Voxels im Objektraum */
#define PYR_FACE_POS_X 0u
#define PYR_FACE_NEG_X 1u
#define PYR_FACE_POS_Y 2u
#define PYR_FACE_NEG_Y 3u
#define PYR_FACE_POS_Z 4u
#define PYR_FACE_NEG_Z 5u

#define PYR_HIT_FACE_MASK    0x7u
#define PYR_HIT_NEW          0x8u   /* Instanz ohne gültige Vorgeschichte: MV nur aus Kamera */
#define PYR_HIT_NO_HISTORY   0x10u  /* Ansicht ohne Vorframe: MV = 0 */
#define PYR_HIT_INSIDE       0x20u  /* Strahl beginnt in einem gefüllten Voxel: t = tmin, Fläche undefiniert */

typedef struct PyrHit {
    float    t;          /* Strahlparameter; PYR_FLT_MAX bei Fehlschuss */
    uint32_t instance;   /* Instanz-Index oder PYR_NO_HIT */
    uint32_t attribute;  /* Voxelattribut */
    uint32_t meta;       /* Fläche (PYR_HIT_FACE_MASK) | PYR_HIT_* */
} PyrHit;

#define PYR_TRACE_ANY_HIT      0x1u  /* erster gefundener Treffer genügt (Schatten, Sichtbarkeit) */
#define PYR_TRACE_NO_ATTRIBUTE 0x2u  /* Attribut nicht bestimmen */
#define PYR_TRACE_SKIP_TRANSPARENT 0x8u /* durchsichtige Voxel überspringen */
#define PYR_TRACE_EXTENDED     0x4u  /* pyr_trace schreibt PyrHitEx (Voxel, Position, Normale), z. B. für Picking */

typedef struct PyrHitEx {
    PyrHit   hit;
    int32_t  voxel[3];       /* Voxelkoordinate im Objektraum */
    uint32_t reserved0;
    float    position[3];    /* Trefferpunkt (Welt, relativ zum Render-Ursprung) */
    uint32_t reserved1;
    float    normal[3];      /* Weltnormale */
    uint32_t reserved2;
} PyrHitEx;

/* ---------------------------------------------------------------------------
 * Kamera: Blick entlang -Z, +Y oben, +X rechts; Pixel (0,0) oben links.
 * ------------------------------------------------------------------------- */

#define PYR_PROJECTION_PERSPECTIVE  0u
#define PYR_PROJECTION_ORTHOGRAPHIC 1u

typedef struct PyrCamera {
    float    view_to_world[12]; /* Kamerapose, 3x4 zeilenweise (Spalte 3 = Position) */
    uint32_t projection;        /* PYR_PROJECTION_* */
    uint32_t width, height;
    float    scale[2];          /* perspektivisch: tan(fov_x/2), tan(fov_y/2); orthografisch: halbe Breite/Höhe */
    float    shift[2];          /* Lens-Shift in NDC-Einheiten */
    float    jitter[2];         /* Subpixel-Versatz in Pixeln; nur Strahlerzeugung */
    float    near_plane;
    float    far_plane;         /* 0 = unendlich */
} PyrCamera;

/* ---------------------------------------------------------------------------
 * Szene auf der GPU (Inhalt von pyr_scene_device)
 * ------------------------------------------------------------------------- */

#define PYR_GEOMETRY_HAS_ATTRIBUTES 0x1u

typedef struct PyrGeometryData {
    uint32_t node_offset, leaf_offset, attribute_offset, root;
    uint32_t log2_size, flags, default_attribute, reserved;
} PyrGeometryData;

#define PYR_INSTANCE_ACTIVE 0x1u
#define PYR_INSTANCE_KEEP_HISTORY 0x2u /* neue statische Instanz: Verlauf der Umgebung behalten */

typedef struct PyrInstanceData {
    float    object_to_world[12];
    float    world_to_object[12];
    float    bounds_min[3];
    uint32_t geometry;
    float    bounds_max[3];
    uint32_t mask;
    uint32_t user;
    uint32_t history;
    uint32_t flags;
    uint32_t reserved;
} PyrInstanceData;

/* Eine Textur auf dem Gerät: dicht gepackte RGBA8-Zeilen */
typedef struct PyrTextureData {
    uint64_t data;
    uint32_t width;
    uint32_t height;
    uint32_t levels;   /* Verkleinerungsstufen, hintereinander im Puffer */
    uint32_t reserved_tex_data;
} PyrTextureData;

typedef struct PyrScene {
    uint64_t nodes, leaves, attributes, geometries;   /* Gerätezeiger */
    uint64_t instances, instances_prev;               /* aktueller Frame, Vorframe */
    uint32_t instance_count, geometry_count;
    uint64_t frame;
    double   time, time_prev;
    double   origin[3], origin_prev[3];
    uint64_t materials;                               /* const PyrMaterial[PYR_MAX_MATERIALS] */
    uint64_t lighting;                                /* const PyrLighting* */
    uint64_t transparent_materials[4];                /* Bit je Material: PYR_MATERIAL_TRANSPARENT */
    uint64_t textures;                                /* const PyrTextureData[texture_count], 1-basiert */
    uint32_t texture_count;
    uint32_t reserved_tex;
} PyrScene;

/* ---------------------------------------------------------------------------
 * Materialien und Licht
 *
 * Voxelattribut: Bits 0..7 = Materialindex, Bits 8..31 = Farbe 0xRRGGBB (sRGB).
 * Materialien mit PYR_MATERIAL_VOXEL_COLOR multiplizieren ihre Grundfarbe mit
 * der Voxelfarbe. Attribut 0 bedeutet beim DAG-Bau "leer".
 * ------------------------------------------------------------------------- */

#define PYR_MAX_MATERIALS 256u
#define PYR_MATERIAL_VOXEL_COLOR 0x1u
/* transparente Ebene: Brechung nach Snell mit ior (sonst gerade hindurch, z. B.
 * dünne Scheiben). Bis zu PYR_MAX_TRANSPARENT_LAYERS Körper je Pixel, jeder mit
 * Fresnel an Ein- und Austritt und Absorption (density) im Inneren. */
#define PYR_MATERIAL_REFRACT 0x2u
/* Voxel dieses Materials sind durchsichtig: Primärstrahlen laufen hindurch,
 * die Transparenzschleife sammelt sie. So liegt Wasser in derselben Geometrie
 * wie der Boden, ohne eigene Instanz. */
#define PYR_MATERIAL_TRANSPARENT 0x4u
/* Wellen: zeitabhängig gestörte Normale (Wasser). Die Geometrie bleibt stehen,
 * Treffer, Tiefe und Motion Vectors bleiben exakt. */
#define PYR_MATERIAL_WAVES 0x8u
#define PYR_MAX_TRANSPARENT_LAYERS 4u
#define PYR_VOXEL(material, r, g, b) \
    ((((uint32_t)(r) & 0xFFu) << 24) | (((uint32_t)(g) & 0xFFu) << 16) | (((uint32_t)(b) & 0xFFu) << 8) | ((uint32_t)(material) & 0xFFu))

typedef struct PyrMaterial {
    float    base_color[3];   /* linear */
    float    roughness;
    float    emission[3];     /* linear, beliebig hell */
    float    metallic;
    uint32_t flags;           /* PYR_MATERIAL_* */
    float    opacity;         /* transparente Ebene: Deckkraft der Oberfläche (0 = klar) */
    float    ior;             /* Brechungsindex für die Fresnel-Reflexion */
    float    density;
    /* PYR_MATERIAL_WAVES: Höhe und Wellenlänge der Normalenstörung
     * (Welteinheiten), Geschwindigkeit in Wellenlängen je Sekunde */
    float    wave_height, wave_length, wave_speed;
    /* Klarlack: zweite, glatte Schicht darüber (Lack, Nässe) */
    float    clearcoat;
    float    clearcoat_roughness;
    /* Streuung unter der Oberfläche: Licht wickelt sich um die Kante
     * (Haut, Laub, Wachs). 0 = aus. */
    float    subsurface;
    float    subsurface_color[3];
    /* Texturen (1-basiert, 0 = keine) und Kantenlänge einer Kachel in
     * Welteinheiten. Voxelflächen sind achsenparallel, es wird genau eine
     * Ebene projiziert. */
    uint32_t texture;
    uint32_t normal_texture;
    float    texture_scale;
    /* Ohne Normalentextur: Stärke und Wellenlänge einer erzeugten Detailnormale */
    float    normal_strength;
    float    normal_scale;
    uint32_t reserved;
} PyrMaterial;


#define PYR_MAX_LIGHTS 64u

/* PyrLight.kind */
#define PYR_LIGHT_SPHERE 0u
#define PYR_LIGHT_RECT   1u
#define PYR_LIGHT_SPOT   2u

typedef struct PyrLight {
    float position[3];
    float radius;             /* Kugellicht, weiche Schatten */
    float color[3];           /* Beleuchtungsstärke = color / Abstand² */
    float range;              /* 0 = unbegrenzt */
    uint32_t kind;            /* PYR_LIGHT_* */
    float normal[3];          /* Rechteck und Kegel: Richtung */
    float size[2];            /* Rechteck: halbe Kanten; Kegel: cos(innen), cos(aussen) */
    uint32_t reserved_light[2];
} PyrLight;

#define PYR_LIGHTING_SHADOWS  0x1u
#define PYR_LIGHTING_GI       0x2u  /* eine indirekte Reflexion */
#define PYR_LIGHTING_AO       0x4u  /* Umgebungsverdeckung statt GI */
#define PYR_LIGHTING_SUN_DISK 0x8u
#define PYR_LIGHTING_REFLECTIONS 0x10u  /* Reflexionsstrahlen (glatte Oberflächen, transparente Ebene) */
/* Indirekte Beleuchtung (GI oder AO) in halber Auflösung: ein Viertel der
 * Strahlen, kantenbewusst hochskaliert. Braucht color, normal, albedo und
 * hits als Ziele in PyrTargets. */
#define PYR_LIGHTING_GI_HALF 0x20u

typedef struct PyrLighting {
    float    sun_direction[3];  /* zur Sonne */
    float    sun_angular_radius;
    float    sun_color[3];
    uint32_t flags;             /* PYR_LIGHTING_* */
    float    sky_zenith[3];
    float    sky_intensity;
    float    sky_horizon[3];
    float    ao_radius;
    float    ground_color[3];
    uint32_t light_count;
    /* Versatz der Sekundärstrahlen entlang der Normale als Anteil der
     * Trefferentfernung; nötig bei secondary_mask (siehe PyrWorldStats). */
    float    secondary_bias;
    float    gi_distance;       /* Reichweite der indirekten Beleuchtung, 0 = unbegrenzt */
    /* Umgebungskarte: setzt pyr_environment_set, nicht der Aufrufer */
    uint64_t env_data;
    uint64_t env_marginal;
    uint64_t env_cond;
    uint32_t env_width;
    uint32_t env_height;
    float    env_intensity;     /* 0 = 1 */
    float    env_rotation;      /* Drehung um Y in Radiant */
    float    env_total;
    float    env_mean;
    uint32_t gi_bounces;        /* indirekte Reflexionen, 0/1 = eine */
    /* Teilnehmendes Medium: Nebel und Lichtschaechte */
    float    fog_density;       /* je Welteinheit auf Hoehe fog_height, 0 = aus */
    float    fog_color[3];
    float    fog_height;
    float    fog_falloff;       /* exponentielle Abnahme darueber, 0 = gleichmaessig */
    float    fog_anisotropy;    /* Henyey-Greenstein g, >0 streut nach vorn */
    uint32_t fog_steps;         /* 0 = 12 */
    float    firefly_clamp;     /* Obergrenze je Abtastung, 0 = aus */
    uint32_t reserved[1];
    PyrLight lights[PYR_MAX_LIGHTS];
} PyrLighting;

#ifdef __cplusplus
extern "C" {
#endif

#define PYR_VERSION_MAJOR 0
#define PYR_VERSION_MINOR 1
#define PYR_VERSION ((uint32_t)((PYR_VERSION_MAJOR << 16) | PYR_VERSION_MINOR))

#if defined(_WIN32)
#  define PYR_API __declspec(dllimport)
#else
#  define PYR_API __attribute__((visibility("default")))
#endif

#define PYR_DEFINE_HANDLE(name) typedef struct name##_T* name

typedef struct PyrContext PyrContext;
typedef struct PyrDag PyrDag;
PYR_DEFINE_HANDLE(PyrGeometry);
PYR_DEFINE_HANDLE(PyrInstance);
PYR_DEFINE_HANDLE(PyrView);

/* ---------------------------------------------------------------------------
 * Ergebnisse
 * ------------------------------------------------------------------------- */

typedef int32_t PyrResult;
#define PYR_OK                      0
#define PYR_ERROR_INVALID_ARGUMENT -1
#define PYR_ERROR_INVALID_HANDLE   -2
#define PYR_ERROR_OUT_OF_MEMORY    -3  /* Host, Gerät oder ein Pool ist voll */
#define PYR_ERROR_CAPACITY         -4  /* max_instances / max_geometries / max_views erreicht */
#define PYR_ERROR_IN_USE           -5  /* Geometrie wird noch von Instanzen verwendet */
#define PYR_ERROR_CUDA             -6  /* Fehler der CUDA-Treiber-API */
#define PYR_ERROR_COMPILE          -7  /* Kernel konnten nicht geladen werden (PTX-JIT) */
#define PYR_ERROR_NOT_FOUND        -8  /* libcuda nicht gefunden */
#define PYR_ERROR_VERSION          -9

PYR_API const char* pyr_result_string(PyrResult result);

/* Ausführliche Meldung zum letzten Fehler dieses Threads (z. B. JIT-Log). */
PYR_API const char* pyr_error_message(void);

/* ---------------------------------------------------------------------------
 * Kontext
 * ------------------------------------------------------------------------- */

/* Nachbearbeitung überlappt mit dem nächsten Frame: DLSS und TAA laufen auf
 * den Tensorkernen, während die Shader-Einheiten schon weiterrechnen. Dafür
 * muss die Anwendung zwei Saetze von PyrTargets abwechselnd benutzen, sonst
 * ueberschreibt der naechste Frame, woraus die Nachbearbeitung noch liest. */
#define PYR_CREATE_ASYNC_POST 0x40u
#define PYR_CREATE_DEBUG 0x1u  /* zusätzliche Prüfungen, synchrone Fehlerberichte */
#define PYR_CREATE_NO_RT    0x2u  /* RT-Cores nicht verwenden, immer CUDA-Traversierung */
#define PYR_CREATE_FORCE_RT 0x4u  /* RT-Cores immer verwenden (sonst erst ab einigen Instanzen) */

#define PYR_FEATURE_RT_CORES 0x1u  /* RT-Cores verfügbar; ab einigen Instanzen nutzen pyr_render/pyr_trace sie automatisch */

typedef void (*PyrLogFn)(void* user, int32_t level, const char* message);

typedef struct PyrCreateInfo {
    uint32_t struct_size;          /* sizeof(PyrCreateInfo) */
    uint32_t version;              /* PYR_VERSION */
    int32_t  device;               /* CUDA-Geräteindex */
    uint32_t flags;                /* PYR_CREATE_* */
    void*    cuda_context;         /* vorhandener CUcontext oder NULL (primärer Kontext) */
    void*    cuda_stream;          /* vorhandener CUstream oder NULL (eigener Stream) */
    uint32_t max_instances;        /* 0 = 65536 */
    uint32_t max_geometries;       /* 0 = 4096 */
    uint32_t max_views;            /* 0 = 16 */
    uint32_t max_textures;         /* 0 = 256 */
    uint32_t rt_leaf_log2;         /* Kantenlänge der RT-AABB-Primitive als log2, 0 = 7 (128^3) */
    uint64_t node_pool_bytes;      /* 0 = 256 MiB */
    uint64_t leaf_pool_bytes;      /* 0 = 256 MiB */
    uint64_t attribute_pool_bytes; /* 0 = 256 MiB */
    uint64_t staging_bytes;        /* gepinnter Upload-Ring, 0 = 64 MiB */
    PyrLogFn log;
    void*    log_user;
} PyrCreateInfo;

PYR_API PyrResult pyr_create(const PyrCreateInfo* info, PyrContext** out);
PYR_API void      pyr_destroy(PyrContext* ctx);   /* wartet auf die GPU */

PYR_API void*     pyr_cuda_stream(PyrContext* ctx);   /* CUstream */
PYR_API uint32_t  pyr_features(PyrContext* ctx);      /* PYR_FEATURE_* */
PYR_API PyrResult pyr_synchronize(PyrContext* ctx);

/* ---------------------------------------------------------------------------
 * DAG-Bau auf der CPU (ohne Kontext, ohne GPU)
 *
 * Kantenlänge 2^log2_size mit log2_size in [3, 20]. Attributwert 0 bedeutet
 * „leer“; jeder andere Wert ist ein belegtes Voxel mit diesem Attribut.
 * ------------------------------------------------------------------------- */

#define PYR_DAG_NO_ATTRIBUTES 0x1u  /* nur Geometrie speichern; Treffer liefern default_attribute = 1 */

typedef struct PyrVoxel {
    int32_t  x, y, z;
    uint32_t attribute;            /* != 0 */
} PyrVoxel;

/* Attribut des Voxels (x, y, z); 0 = leer. */
typedef uint32_t (*PyrVoxelFn)(void* user, int32_t x, int32_t y, int32_t z);

/* Optional: Ist der Würfel [x, x+size)^3 sicher leer? Rückgabe != 0 überspringt ihn. */
typedef int32_t (*PyrRegionEmptyFn)(void* user, int32_t x, int32_t y, int32_t z, uint32_t size);

typedef struct PyrDagInfo {
    uint32_t        log2_size;
    uint32_t        root;
    uint64_t        voxel_count;
    const uint32_t* nodes;        uint64_t node_words;
    const uint64_t* leaves;       uint64_t leaf_count;
    const uint32_t* attributes;   uint64_t attribute_count;   /* NULL bei PYR_DAG_NO_ATTRIBUTES */
} PyrDagInfo;

/* voxels[x + N * (y + N * z)], N = 2^log2_size */
PYR_API PyrResult pyr_dag_build_dense(uint32_t log2_size, const uint32_t* voxels, uint32_t flags, PyrDag** out);
PYR_API PyrResult pyr_dag_build_points(uint32_t log2_size, const PyrVoxel* voxels, size_t count, uint32_t flags,
                                       PyrDag** out);
PYR_API PyrResult pyr_dag_build_fn(uint32_t log2_size, PyrVoxelFn voxel, PyrRegionEmptyFn region_empty, void* user,
                                   uint32_t flags, PyrDag** out);
PYR_API void      pyr_dag_get_info(const PyrDag* dag, PyrDagInfo* out);
PYR_API void      pyr_dag_destroy(PyrDag* dag);

/* Binärformat für fertige DAGs. buffer == NULL: nur *size setzen. */
PYR_API PyrResult pyr_dag_save(const PyrDag* dag, void* buffer, size_t* size);
PYR_API PyrResult pyr_dag_load(const void* data, size_t size, PyrDag** out);

/* MagicaVoxel (.vox): Modell `model` als Voxelliste (y oben, Farbe aus der
 * Palette, Material 0). out == NULL: nur *count und *log2_size setzen;
 * sonst muss *count die Kapazität von out angeben. Danach z. B.
 * pyr_geometry_build(ctx, log2_size, voxels, count, PYR_BUILD_HOST_INPUT, &geo). */
PYR_API PyrResult pyr_vox_parse(const void* data, size_t size, uint32_t model, PyrVoxel* out, uint32_t* count,
                                uint32_t* log2_size);

/* ---------------------------------------------------------------------------
 * Geometrie auf der GPU
 * ------------------------------------------------------------------------- */

/* Lädt die DAG hoch (asynchron); dag darf danach sofort zerstört werden. */
PYR_API PyrResult pyr_geometry_create(PyrContext* ctx, const PyrDag* dag, PyrGeometry* out);
/* Baut die DAG auf der GPU aus einer Voxelliste (PyrVoxel[count], Reihenfolge
 * beliebig, doppelte Koordinaten: der letzte gewinnt, Attribut 0 = leer).
 * voxels ist ein Gerätezeiger, mit PYR_BUILD_HOST_INPUT ein Host-Zeiger. */
#define PYR_BUILD_HOST_INPUT 0x1u
#define PYR_BUILD_EDITABLE   0x2u  /* Voxelliste auf der GPU behalten: pyr_geometry_edit möglich */
PYR_API PyrResult pyr_geometry_build(PyrContext* ctx, uint32_t log2_size, const void* voxels, uint32_t count, uint32_t flags,
                                     PyrGeometry* out);

/* Setzt oder entfernt Voxel (Attribut 0 = entfernen) einer änderbaren Geometrie.
 * Neubau auf der GPU; wirkt sofort in Stream-Reihenfolge, Instanzen bleiben. */
PYR_API PyrResult pyr_geometry_edit(PyrContext* ctx, PyrGeometry geometry, const void* voxels, uint32_t count, uint32_t flags);

/* Gröbere Fassung (LOD) einer änderbaren Geometrie, auf der GPU gebaut:
 * Kantenlänge 2^(log2_size - shift), je grober Zelle ein Voxel (das oberste).
 * Für dieselbe Weltgröße die Instanz um 2^shift skalieren. flags: PYR_BUILD_EDITABLE. */
PYR_API PyrResult pyr_geometry_downsample(PyrContext* ctx, PyrGeometry geometry, uint32_t shift, uint32_t flags, PyrGeometry* out);

/* Kopiert eine Geometrie von der GPU in eine Host-DAG (z. B. für pyr_dag_save). */
PYR_API PyrResult pyr_geometry_download(PyrContext* ctx, PyrGeometry geometry, PyrDag** out);

/* PYR_ERROR_IN_USE, solange Instanzen die Geometrie verwenden. */
PYR_API PyrResult pyr_geometry_destroy(PyrContext* ctx, PyrGeometry geometry);
PYR_API uint32_t  pyr_geometry_index(PyrGeometry geometry);

/* ---------------------------------------------------------------------------
 * Instanzen: Geometrie + Transformation mit stabilem Index und Vorgeschichte
 *
 * Änderungen werden mit dem nächsten pyr_commit sichtbar. Transformationen
 * sind Objekt -> Welt, 3x4 zeilenweise, relativ zum Render-Ursprung. Der
 * Objektraum einer Geometrie ist [0, 2^log2_size)^3 mit Voxelgröße 1.
 * ------------------------------------------------------------------------- */

PYR_API PyrResult pyr_instance_create(PyrContext* ctx, PyrGeometry geometry, PyrInstance* out);
PYR_API PyrResult pyr_instance_destroy(PyrContext* ctx, PyrInstance instance);
PYR_API PyrResult pyr_instance_set_transform(PyrContext* ctx, PyrInstance instance, const float object_to_world[12]);
PYR_API PyrResult pyr_instance_set_geometry(PyrContext* ctx, PyrInstance instance, PyrGeometry geometry);
PYR_API PyrResult pyr_instance_set_mask(PyrContext* ctx, PyrInstance instance, uint32_t mask);   /* Standard 0xFF */
PYR_API PyrResult pyr_instance_set_user(PyrContext* ctx, PyrInstance instance, uint32_t user);
/* Nächster Frame ohne Vorgeschichte (Teleport): Treffer tragen PYR_HIT_NEW. */
PYR_API PyrResult pyr_instance_reset_history(PyrContext* ctx, PyrInstance instance);
PYR_API uint32_t  pyr_instance_index(PyrInstance instance);   /* = PyrHit.instance */

/* ---------------------------------------------------------------------------
 * Frames
 * ------------------------------------------------------------------------- */

typedef struct PyrFrameInfo {
    double time;        /* Sekunden, monoton */
    double origin[3];   /* Render-Ursprung in Weltkoordinaten */
} PyrFrameInfo;

/* Beginnt einen neuen Frame: der bisherige Zustand wird Vorframe, alle
 * ausstehenden Änderungen werden aktiv. info darf NULL sein. */
PYR_API PyrResult pyr_commit(PyrContext* ctx, const PyrFrameInfo* info);

/* Gerätezeiger auf PyrScene für eigene Kernel; gleich über alle Frames. */
PYR_API uint64_t  pyr_scene_device(PyrContext* ctx);

/* ---------------------------------------------------------------------------
 * Ansichten und Rendern
 *
 * Eine Ansicht merkt sich ihre Kamera aus dem Vorframe. Wurde sie im
 * vorigen Frame nicht gerendert oder hat sich die Auflösung geändert, gibt
 * es keine Vorgeschichte (PYR_HIT_NO_HISTORY, Motion Vectors = 0).
 * ------------------------------------------------------------------------- */

typedef struct PyrTargets {
    uint64_t hits;     /* PyrHit[width * height] oder 0 */
    uint64_t depth;    /* float[width * height]: lineare Tiefe entlang der Blickachse, PYR_FLT_MAX bei Fehlschuss */
    uint64_t motion;   /* float[2 * width * height]: Pixel, vorher − jetzt, ohne Jitter */
    uint64_t color;    /* float[4 * width * height]: lineare HDR-Farbe (a = 1 Treffer); 0 = kein Shading */
    uint64_t normal;   /* float[4 * width * height]: Weltnormale, w = lineare Tiefe */
    uint64_t albedo;   /* float[4 * width * height]: Albedo */
    uint64_t material; /* float[2 * width * height]: Rauheit, Metall (für DLSS-RR) */
    uint32_t ray_mask; /* 0 = alle Instanzen */
    uint32_t flags;    /* PYR_TRACE_NO_ATTRIBUTE */
    uint32_t transparent_mask; /* Instanzmaske der transparenten Ebene (Wasser, Glas), 0 = keine */
    /* Maske für Schatten-, GI- und Reflexionsstrahlen; 0 = wie ray_mask. So
     * sehen Sekundärstrahlen z. B. nur die gröbere Fassung der Welt. */
    uint32_t secondary_mask;
} PyrTargets;

PYR_API PyrResult pyr_view_create(PyrContext* ctx, PyrView* out);
PYR_API PyrResult pyr_view_destroy(PyrContext* ctx, PyrView view);
PYR_API PyrResult pyr_view_reset_history(PyrContext* ctx, PyrView view);
PYR_API PyrResult pyr_render(PyrContext* ctx, PyrView view, const PyrCamera* camera, const PyrTargets* targets);

/* Beliebige Strahlen: rays und hits sind Gerätezeiger (PyrRay[count], PyrHit[count]). */
PYR_API PyrResult pyr_trace(PyrContext* ctx, uint64_t rays, uint64_t hits, uint32_t count, uint32_t ray_mask,
                            uint32_t flags);

/* ---------------------------------------------------------------------------
 * Materialien, Licht
 * ------------------------------------------------------------------------- */

PYR_API void      pyr_material_default(PyrMaterial* material);
PYR_API PyrResult pyr_material_set(PyrContext* ctx, uint32_t index, const PyrMaterial* material);
/* Textur: dicht gepackte RGBA8-Zeilen. Gefiltert wird von Hand (bilinear,
 * wiederholend) – keine Texturhardware, damit derselbe Code später auch auf
 * AMD läuft. Der zurückgegebene Index geht 1-basiert ins Material. */
PYR_API PyrResult pyr_texture_create(PyrContext* ctx, uint32_t width, uint32_t height,
                                     const void* rgba8, uint32_t* out_index);
PYR_API PyrResult pyr_texture_destroy(PyrContext* ctx, uint32_t index);

/* Umgebungskarte (equirektangulär, 4 Floats je Texel: RGB + ungenutzt).
 * Pyrit baut daraus die Verteilung fuer das Importance-Sampling, damit auch
 * eine kleine helle Sonne in der Karte rauschfrei beleuchtet. width = 0
 * schaltet sie ab (dann gilt der analytische Himmel aus PyrLighting).
 * Helligkeit und Drehung stehen in PyrLighting. */
PYR_API PyrResult pyr_environment_set(PyrContext* ctx, uint32_t width, uint32_t height, const float* rgba);
PYR_API void      pyr_lighting_default(PyrLighting* lighting);
PYR_API PyrResult pyr_set_lighting(PyrContext* ctx, const PyrLighting* lighting);
PYR_API uint32_t  pyr_voxel_attribute(uint32_t material, uint32_t r, uint32_t g, uint32_t b);

/* ---------------------------------------------------------------------------
 * Nachbearbeitung (GPU): temporale Akkumulation / TAA über die Motion
 * Vectors, kantenerhaltender Denoiser, Tonemapping. Eingaben sind die
 * Targets des letzten pyr_render dieser Ansicht.
 * ------------------------------------------------------------------------- */

#define PYR_POST_RESET       0x1u  /* Verlauf verwerfen */
#define PYR_POST_NO_TEMPORAL 0x2u
#define PYR_POST_BGRA        0x4u  /* output_ldr als BGRA (X11 und andere Fenster) */

/* Hochskalieren: gerendert wird in der Auflösung der Kamera, ausgegeben in
 * output_width x output_height. Jitter (pyr_jitter_halton) ist für TAAU und
 * DLSS nötig. */
#define PYR_UPSCALER_AUTO     0u  /* TAAU (bei Faktor 1: TAA) */
#define PYR_UPSCALER_NONE     1u  /* nur Renderauflösung, ohne TAAU */
#define PYR_UPSCALER_TAAU     2u
#define PYR_UPSCALER_DLSS     3u  /* DLSS Super Resolution (falls verfügbar) */
#define PYR_UPSCALER_DLSS_RR  4u  /* DLSS Ray Reconstruction (ersetzt Denoiser) */

#define PYR_TONEMAP_ACES     0u
#define PYR_TONEMAP_REINHARD 1u
#define PYR_TONEMAP_NONE     2u

/* Kamera- und Bildeffekte (PyrPostFx.flags) */
#define PYR_POSTFX_BLOOM          0x1u
#define PYR_POSTFX_DOF            0x2u
#define PYR_POSTFX_MOTION_BLUR    0x4u
#define PYR_POSTFX_AUTO_EXPOSURE  0x8u
#define PYR_POSTFX_GRADE          0x10u
#define PYR_POSTFX_AUTOFOCUS      0x20u
#define PYR_POSTFX_DOF_FAR_ONLY   0x40u

/* Kamera- und Bildeffekte auf dem fertigen Bild in Ausgabeauflösung.
 * Felder auf 0 nehmen die jeweilige Vorgabe. */
typedef struct PyrPostFx {
    uint32_t flags;                 /* PYR_POSTFX_* */
    float    bloom_strength;        /* 0 = 0.05 */
    float    bloom_threshold;       /* 0 = 1.0 */
    float    bloom_knee;            /* 0 = 0.5 */
    uint32_t bloom_levels;          /* 0 = 5 */
    float    focus_distance;        /* 0 = 10; mit AUTOFOCUS egal */
    float    dof_strength;          /* Zerstreuungskreis in Pixeln bei unendlich, 0 = 3 */
    float    dof_max_coc;           /* 0 = 12 Pixel */
    float    motion_blur_scale;     /* 0 = 0.5 (Verschlusswinkel 180 Grad) */
    float    motion_blur_max;       /* 0 = 64 Pixel */
    uint32_t motion_blur_samples;   /* 0 = 12 */
    float    exposure_speed;        /* 0 = 0.05 je Frame */
    float    exposure_min;          /* 0 = 0.03 */
    float    exposure_max;          /* 0 = 30 */
    float    exposure_compensation; /* Blendenstufen */
    float    temperature;           /* warm > 0, kalt < 0 */
    float    tint;                  /* gruen > 0, magenta < 0 */
    float    contrast;              /* 0 = 1 */
    float    saturation;            /* 0 = 1 */
    float    lift[3];
    float    gamma[3];              /* 0 = 1 */
    float    gain[3];               /* 0 = 1 */
    uint64_t lut;                   /* 3D-LUT (RGBA8) im Anzeigeraum, 0 = keine */
    uint32_t lut_size;              /* Kantenlaenge */
    /* Supersampling: gerendert wird in diesem Faktor hoeherer Aufloesung,
     * der letzte Schritt mittelt zusammen. 0/1 = aus. Kostet den Faktor im
     * Quadrat an Strahlen, ist aber das einzige wirksame Mittel gegen
     * wandernde Kanten bei Voxelgeometrie. */
    uint32_t supersample;
} PyrPostFx;

typedef struct PyrPostInfo {
    uint64_t output_hdr;          /* float[4 * w * h] oder 0 */
    uint64_t output_ldr;          /* uint8[4 * w * h] (sRGB RGBA) oder 0 */
    float    exposure;            /* 0 = 1 */
    uint32_t tonemap;             /* PYR_TONEMAP_* */
    uint32_t denoise_iterations;  /* 0 = kein räumlicher Filter, typisch 4 */
    float    temporal_alpha;      /* kleinstes Gewicht des neuen Frames, 0 = 0.05 */
    float    clamp_sigma;         /* Varianzbegrenzung (TAA), 0 = aus, typisch 1.5 */
    uint32_t flags;               /* PYR_POST_* */
    uint32_t output_width;        /* Ausgabeauflösung, 0 = Renderauflösung */
    uint32_t output_height;
    uint32_t upscaler;            /* PYR_UPSCALER_* */
    float    denoise_phi;         /* Varianzführung des Denoisers, 0 = 4 (kleiner = glatter) */
    const PyrPostFx* fx;          /* Kamera- und Bildeffekte, NULL = keine */
} PyrPostInfo;

PYR_API PyrResult pyr_postprocess(PyrContext* ctx, PyrView view, const PyrTargets* input, const PyrPostInfo* info);

/* ---------------------------------------------------------------------------
 * Frame Generation (reines CUDA): Zwischenbild zwischen den letzten beiden
 * Ausgaben von pyr_postprocess mit exakten Motion Vectors und Tiefe. Aufruf
 * nach pyr_postprocess von Frame N; das Ergebnis vor Frame N anzeigen. Für
 * mehrere Zwischenbilder mehrfach mit t = 1/3, 2/3, ... aufrufen.
 * Eigene Verfahren: generate setzen; Pyrit liefert alle Eingaben.
 * ------------------------------------------------------------------------- */

typedef struct PyrFrameGenParams {
    uint32_t width, height;
    float    t, exposure;
    uint64_t prev_color;      /* float[4] HDR linear, Frame N-1 */
    uint64_t cur_color;       /* float[4] HDR linear, Frame N */
    uint64_t motion_depth;    /* float[4]: MV N -> N-1 in Pixeln (xy), Tiefe (z) */
    uint32_t tonemap, reserved;
    uint64_t out_hdr, out_ldr;
    uint64_t user;
    /* Zwischenspeicher der Vorwärtsprojektion (Pyrit stellt sie bereit):
     * kleinste Tiefe je Zwischenbildpixel und der zugehörige Bewegungsvektor */
    uint64_t depth_mid, mv_mid;
    uint32_t bgra, reserved_bgra;   /* out_ldr als BGRA (PYR_POST_BGRA) */
} PyrFrameGenParams;

typedef void (*PyrFrameGenFn)(void* user, const PyrFrameGenParams* params, void* stream);

typedef struct PyrFrameGenInfo {
    uint64_t output_hdr;      /* float[4] oder 0 (Ausgabeauflösung) */
    uint64_t output_ldr;      /* uint8[4] sRGB oder 0 */
    float    t;               /* 0 = 0.5 */
    uint32_t flags;
    PyrFrameGenFn generate;   /* NULL = eingebaut */
    void*    user;
} PyrFrameGenInfo;

PYR_API PyrResult pyr_frame_generate(PyrContext* ctx, PyrView view, const PyrFrameGenInfo* info);

/* Subpixel-Versatz für TAA (Halton 2,3), Bereich [-0.5, 0.5) */
PYR_API void pyr_jitter_halton(uint32_t frame, float out[2]);

/* ---------------------------------------------------------------------------
 * Statistik
 * ------------------------------------------------------------------------- */

typedef struct PyrStats {
    uint64_t node_pool_used, node_pool_capacity;
    uint64_t leaf_pool_used, leaf_pool_capacity;
    uint64_t attribute_pool_used, attribute_pool_capacity;
    uint32_t geometries, instances;
    uint64_t frame;
    uint32_t features;
    uint32_t reserved;
} PyrStats;

PYR_API PyrResult pyr_get_stats(PyrContext* ctx, PyrStats* out);

/* ---------------------------------------------------------------------------
 * Skelettanimation (auf der GPU)
 *
 * Ein Skelett besteht aus Knochen; jeder kann einen Voxel-Teil (Geometrie)
 * tragen. Clips sind Keyframes je Knochen. Ein Akteur ist eine Instanz des
 * Skeletts (eine PyrInstance je Teil) und spielt Clips aus der Szenenzeit
 * (PyrFrameInfo.time) ab: bei jedem pyr_commit rechnet ein Kernel alle Knochen
 * und schreibt die Instanzen direkt – kein Host-Aufruf pro Frame, exakte
 * Motion Vectors. Skelette und Clips leben bis pyr_destroy.
 * ------------------------------------------------------------------------- */

PYR_DEFINE_HANDLE(PyrSkeleton);
PYR_DEFINE_HANDLE(PyrClip);
PYR_DEFINE_HANDLE(PyrActor);

typedef struct PyrBone {
    int32_t     parent;        /* Index eines früheren Knochens oder -1 */
    uint32_t    reserved;
    PyrGeometry geometry;      /* Voxel-Teil oder NULL */
    float       rest[12];      /* Ruhelage relativ zum Elternknochen (3x4) */
    float       part[12];      /* Voxel-Teil relativ zum Knochen (3x4) */
    float       sway;          /* Anteil an der Windbewegung, 0 = starr */
    uint32_t    reserved2;
} PyrBone;

typedef struct PyrKeyframe {
    uint32_t bone;
    float    time;             /* Sekunden im Clip */
    float    translation[3];
    float    scale;            /* gleichmäßig */
    float    rotation[4];      /* Quaternion x, y, z, w */
} PyrKeyframe;

#define PYR_CLIP_LOOP 0x1u

typedef struct PyrActorInfo {
    float    root[12];         /* Lage des Akteurs (3x4) */
    PyrClip  clip;             /* NULL = Ruhelage */
    PyrClip  blend_clip;       /* zweiter Clip zum Überblenden oder NULL */
    float    start_time, speed;             /* Clipzeit = (Zeit - start) * speed */
    float    blend_start_time, blend_speed;
    float    blend;            /* Gewicht von blend_clip, 0..1 */
    /* Wind: Auslenkung in Radiant, Frequenz in Hz, Phase (z. B. aus der Lage).
     * Jeder Knochen biegt sich um seinen sway-Anteil – weiches Schwanken mit
     * exakten Motion Vectors, ohne eigene Keyframes. */
    float    wind_amplitude, wind_frequency, wind_phase;
    uint32_t mask;             /* Instanzmaske der Teile, 0 = 0xFF */
    uint32_t user;             /* PyrInstanceData.user der Teile */
    uint32_t reserved;
} PyrActorInfo;

PYR_API PyrResult pyr_skeleton_create(PyrContext* ctx, const PyrBone* bones, uint32_t count, PyrSkeleton* out);
/* Keyframes in beliebiger Reihenfolge; je Knochen nach Zeit interpoliert
 * (linear, Rotation normalisiert), außerhalb gehalten bzw. mit PYR_CLIP_LOOP
 * wiederholt. Knochen ohne Keyframes behalten ihre Ruhelage. */
PYR_API PyrResult pyr_clip_create(PyrContext* ctx, PyrSkeleton skeleton, const PyrKeyframe* keys, uint32_t count,
                                  float duration, uint32_t flags, PyrClip* out);
PYR_API PyrResult pyr_actor_create(PyrContext* ctx, PyrSkeleton skeleton, const PyrActorInfo* info, PyrActor* out);
/* Clip wechseln, überblenden, versetzen: gilt ab dem nächsten pyr_commit */
PYR_API PyrResult pyr_actor_set(PyrContext* ctx, PyrActor actor, const PyrActorInfo* info);
PYR_API PyrResult pyr_actor_destroy(PyrContext* ctx, PyrActor actor);
/* Erst freigeben, wenn keine Akteure mehr darauf laufen (sonst PYR_ERROR_IN_USE) */
PYR_API PyrResult pyr_skeleton_destroy(PyrContext* ctx, PyrSkeleton skeleton);
PYR_API PyrResult pyr_clip_destroy(PyrContext* ctx, PyrClip clip);

/* ---------------------------------------------------------------------------
 * Große Welten
 *
 * Eine Welt streamt Chunks um die Kamera: nahe fein, fern grob (LOD-Octree).
 * Jeder Chunk hat 2^chunk_log2 Voxel je Kante; auf Stufe l ist ein Voxel
 * 2^l Grundvoxel groß. Feste Stufen gibt es nicht: verfeinert wird, solange
 * ein Voxel auf dem Bildschirm größer als voxel_pixels wäre – der Übergang
 * folgt also aus Auflösung und Blickwinkel, die Tiefe des Baums aus
 * view_distance, und memory_budget macht die Welt gleitend gröber.
 * Chunks werden auf der GPU erzeugt (eingebautes
 * Gelände oder eigener Generator), auf der GPU zu DAGs gebaut und als
 * Instanzen eingetragen. Grobe Chunks bleiben sichtbar, bis alle feineren
 * fertig sind; nicht mehr gebrauchte werden freigegeben.
 *
 * Erzeugen und Bauen laufen auf einem eigenen Thread und CUDA-Stream;
 * pyr_world_update wartet nie auf die GPU. Die Welt vor dem Kontext zerstören.
 *
 * Je Frame:
 *     pyr_world_update(ctx, world, kamera_welt, origin, &camera);
 *     pyr_commit(ctx, &(PyrFrameInfo){ .time = t, .origin = origin });
 *     pyr_render(...);
 * ------------------------------------------------------------------------- */

typedef struct PyrWorld PyrWorld;

typedef struct PyrChunkKey {
    int32_t  x, y, z;   /* Ursprung = (x, y, z) * 2^(chunk_log2 + lod) Grundvoxel */
    uint32_t lod;
} PyrChunkKey;

/* Auftrag an einen Generator. Für jeden Chunk c Voxel {x, y, z, attribut} in
 * Chunk-Koordinaten [0, 2^chunk_log2) schreiben:
 *     uint32_t k = atomicAdd(&counts[c], 1);
 *     if (k < capacity) voxels[c * capacity + k] = v;
 * Nur die sichtbare Haut erzeugen (kein volles Inneres), in der Auflösung
 * der Stufe (ein Voxel = 2^lod Grundvoxel). counts ist beim Aufruf 0. */
typedef struct PyrWorldGenParams {
    uint64_t chunks;      /* Gerät: PyrChunkKey[count] */
    uint64_t voxels;      /* Gerät: PyrVoxel[count * capacity] */
    uint64_t counts;      /* Gerät: uint32_t[count] */
    uint32_t count, capacity, chunk_log2, reserved;
    uint64_t user;        /* PyrWorldInfo.user */
} PyrWorldGenParams;

/* Startet eigene Kernel auf stream (CUstream); nicht synchronisieren. */
typedef void (*PyrWorldGenFn)(void* user, const PyrWorldGenParams* params, void* stream);

/* Eingebautes Gelände (Höhenfeld, fBm). Attribute 0 = eingebaute Farben. */
typedef struct PyrTerrainInfo {
    uint32_t seed, octaves;
    float    base_height, amplitude;   /* Grundvoxel */
    float    wavelength;               /* gröbste Oktave, Grundvoxel */
    float    sea_level, snow_height;
    float    rock_slope;               /* Steigung, ab der Fels entsteht */
    uint32_t attr_grass, attr_dirt, attr_rock, attr_snow, attr_sand;
    uint32_t attr_water;        /* Wasser bis sea_level; 0 = keines. Material mit
                                   PYR_MATERIAL_TRANSPARENT anlegen. */
    uint32_t attr_leaves;       /* Bäume; 0 = keine Vegetation */
    uint32_t attr_wood;
    float    tree_density;      /* Anteil der Spalten mit Baum, z. B. 0,004 */
} PyrTerrainInfo;

/* Erzeugen und Bauen im Aufruf von pyr_world_update statt auf dem
 * Hintergrund-Thread (deterministisch; für Tests und Werkzeuge) */
#define PYR_WORLD_SYNC 0x1u

typedef struct PyrWorldInfo {
    uint32_t chunk_log2;         /* 3..8, 0 = 5 (32^3) */
    uint32_t max_lod;            /* Obergrenze der Baumtiefe, 0 = 20 */
    uint32_t view_chunks;        /* Radius der gröbsten Stufe in Chunks, 0 = 2 */
    float    voxel_pixels;       /* Zielgröße eines Voxels in Pixeln, 0 = 4 */
    float    view_distance;      /* Grundvoxel, 0 = 16384 */
    uint32_t reserved0;
    uint64_t memory_budget;      /* Bytes für Chunks; darüber wird die Welt gröber. 0 = 256 MiB */
    int32_t  y_min, y_max;       /* Grundvoxel; beide 0 = aus dem Gelände */
    uint32_t chunks_per_update;  /* 0 = 256; bestimmt, wie schnell eine frisch
                                    betretene Welt volle Schaerfe erreicht */
    uint32_t mask;               /* Instanzmaske, 0 = 0x1 */
    /* Zusätzliche Maske für eine gröbere Fassung der Welt, die nur Schatten-,
     * GI- und Reflexionsstrahlen sehen (PyrTargets.secondary_mask); 0 = aus.
     * Die Chunks liegen ohnehin im Speicher, es kostet also nur Auswahl. */
    uint32_t secondary_mask;
    float    secondary_pixels;   /* Zielgröße dort, 0 = 4 · voxel_pixels */
    uint32_t chunk_capacity;     /* Voxel je Chunk im Generatorpuffer, 0 = 8 * chunk^2 */
    uint32_t rt_leaf_log2;       /* Kantenlänge der RT-AABBs, 0 = chunk_log2 - 2 (min 3) */
    uint32_t keep_frames;        /* ungenutzte Chunks so lange behalten, 0 = 8 */
    uint32_t flags;              /* PYR_WORLD_* */
    PyrWorldGenFn generate;      /* NULL = eingebautes Gelände; läuft auf dem
                                    Hintergrund-Thread (Kontext ist aktuell) */
    void*    user;
    const PyrTerrainInfo* terrain; /* NULL = Standard */
} PyrWorldInfo;

typedef struct PyrWorldStats {
    uint32_t visible_chunks, resident_chunks, pending_chunks;
    uint32_t built_chunks, built_voxels;   /* im letzten Update */
    uint32_t overflow_chunks;              /* Generatorpuffer übergelaufen (Summe) */
    float    voxel_pixels;                 /* aktuelles Ziel (mit Budget angepasst) */
    uint32_t top_lod;                      /* gröbste benutzte Stufe */
    float    secondary_bias;               /* empfohlener PyrLighting.secondary_bias */
    uint32_t reserved1;
    uint64_t bytes;                        /* GPU-Bytes der Chunk-DAGs */
} PyrWorldStats;

PYR_API PyrResult pyr_world_create(PyrContext* ctx, const PyrWorldInfo* info, PyrWorld** out);
PYR_API PyrResult pyr_world_destroy(PyrContext* ctx, PyrWorld* world);
/* position: Kamera in Weltkoordinaten (Grundvoxel); origin: Render-Ursprung wie
 * in PyrFrameInfo (NULL = 0), Instanzen liegen relativ dazu; camera: Kamera des
 * nächsten Frames für das Bildschirmmaß der LOD-Wahl (NULL = 1080p-Annahme). */
PYR_API PyrResult pyr_world_update(PyrContext* ctx, PyrWorld* world, const double position[3], const double origin[3],
                                   const PyrCamera* camera);
/* Wartet auf den Chunk-Batch, der gerade im Hintergrund entsteht, und
 * übernimmt ihn (Ladebildschirm, Teleport). */
PYR_API PyrResult pyr_world_wait(PyrContext* ctx, PyrWorld* world, const double camera[3]);
/* Eine Änderung an der Welt: ein Grundvoxel setzen oder entfernen.
 * Koordinaten in Grundvoxeln (volle Auflösung), unabhängig davon, in welcher
 * LOD-Stufe der Chunk gerade vorliegt. */
typedef struct PyrWorldEdit {
    int64_t  x, y, z;
    uint32_t attribute;   /* 0 entfernt den Voxel */
    uint32_t reserved;
} PyrWorldEdit;

/* Grundvoxel setzen oder entfernen. Die Änderungen liegen als Überlagerung
 * über dem Generator: betroffene Chunks werden sofort neu gebaut, und jede
 * spätere Neuerzeugung (Verdrängung, LOD-Wechsel) trägt sie wieder auf.
 * Auf gröberen Stufen gilt: Hinzufügen füllt die Zelle immer, Entfernen wirkt
 * erst, wenn alle Grundvoxel der Zelle entfernt sind. */
PYR_API PyrResult pyr_world_edit(PyrContext* ctx, PyrWorld* world, const PyrWorldEdit* edits, uint32_t count);

/* Änderungen sichern und zurückladen. Pyrit fasst keine Dateien an: der
 * Aufrufer legt den Puffer ab, wo er will. Das Gelände wird nicht gespeichert,
 * das erzeugt der Generator jederzeit wieder. */
PYR_API uint64_t  pyr_world_edits_bytes(PyrWorld* world);
PYR_API PyrResult pyr_world_edits_save(PyrWorld* world, void* dst, uint64_t size);
PYR_API PyrResult pyr_world_edits_load(PyrContext* ctx, PyrWorld* world, const void* src, uint64_t size);

PYR_API PyrResult pyr_world_stats(PyrWorld* world, PyrWorldStats* out);
/* Standardgelände und seine Höhe an (x, z), z. B. für die Kamera */
PYR_API void      pyr_terrain_default(PyrTerrainInfo* out);
PYR_API float     pyr_terrain_height(const PyrTerrainInfo* terrain, double x, double z);

/* ---------------------------------------------------------------------------
 * Kamera-Helfer (reine Host-Funktionen)
 * ------------------------------------------------------------------------- */

PYR_API void pyr_camera_look_at(PyrCamera* camera, const float eye[3], const float target[3], const float up[3]);
PYR_API void pyr_camera_perspective(PyrCamera* camera, float fov_y_radians, uint32_t width, uint32_t height,
                                    float near_plane);
PYR_API void pyr_camera_orthographic(PyrCamera* camera, float half_height, uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif /* PYRIT_H */
