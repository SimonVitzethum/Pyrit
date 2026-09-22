/*
 * pyrit.h – Host-API der Pyrit-Runtime (Entwurf, ABI 0.1)
 *
 * Reine C-ABI, nutzbar aus C, C++, Zig, Rust, Java (FFM) usw.
 * In der Implementierung erzeugt der Zig-Build diesen Header aus den
 * `extern`-Deklarationen der Runtime; dieser Entwurf ist die Vorlage dafür.
 *
 * Ablauf pro Frame:
 *
 *   PyrSegment seg;
 *   pyr_acquire(rt, timeout, &seg);          // wartet nur, wenn die GPU >= N Frames zurückliegt
 *   PyrWriter w; pyr_writer_init(&w, seg.base, seg.capacity);
 *   ... Befehle schreiben (pyrit_ring.h) ...
 *   pyr_submit(rt, w.records, w.payload_top, 0);   // startet den CUDA-Graphen
 *
 * Grundsätze:
 *   - Pro Frame genau ein Graph-Launch. Keine synchronen Rückleseoperationen.
 *   - Alle Handles vergibt der Client (siehe PyrRef in pyrit_ring.h).
 *   - Rückmeldungen der GPU laufen asynchron über den Statusblock (pyr_get_status).
 *   - Funktionen sind nicht threadsicher, außer wo vermerkt. pyr_hook_register und
 *     pyr_pipeline_* dürfen aus einem anderen Thread als der Frame-Schleife kommen.
 */
#ifndef PYRIT_H
#define PYRIT_H

#include <stddef.h>
#include <stdint.h>
#include "pyrit_ring.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYR_API_VERSION_MAJOR 0
#define PYR_API_VERSION_MINOR 1
#define PYR_API_VERSION ((uint32_t)((PYR_API_VERSION_MAJOR << 16) | PYR_API_VERSION_MINOR))

#if defined(_WIN32)
#  define PYR_API __declspec(dllimport)
#else
#  define PYR_API __attribute__((visibility("default")))
#endif

typedef struct PyrRuntime PyrRuntime;

/* ---------------------------------------------------------------------------
 * Ergebnisse
 * ------------------------------------------------------------------------- */

typedef int32_t PyrResult;
enum {
    PYR_OK                   = 0,
    PYR_NOT_READY            = 1,   /* Pipeline noch im Bau o. Ä. */
    PYR_TIMEOUT              = 2,   /* pyr_acquire: Segment noch in Benutzung */
    PYR_ERR_INVALID_ARGUMENT = -1,
    PYR_ERR_VERSION_MISMATCH = -2,
    PYR_ERR_OUT_OF_MEMORY    = -3,
    PYR_ERR_UNSUPPORTED      = -4,  /* GPU, Treiber oder Interop fehlt */
    PYR_ERR_COMPILE          = -5,  /* Hook- oder Pipeline-Übersetzung fehlgeschlagen */
    PYR_ERR_CONTRACT         = -6,  /* Hook verletzt die statische PTX-Prüfung */
    PYR_ERR_DEVICE_LOST      = -7,  /* CUDA-Kontext unbrauchbar, pyr_recover() nötig */
    PYR_ERR_DEVICE_HUNG      = -8,  /* Frame nicht innerhalb von hang_timeout_ms fertig */
    PYR_ERR_INTERNAL         = -9
};

PYR_API const char* pyr_result_string(PyrResult r);

/* ---------------------------------------------------------------------------
 * Erzeugen, Größe ändern, Zerstören
 * ------------------------------------------------------------------------- */

typedef int32_t PyrUpscaler;
enum { PYR_UPSCALER_NONE = 0, PYR_UPSCALER_DLAA = 1, PYR_UPSCALER_DLSS_SR = 2 };

#define PYR_CONFIG_DEBUG_VALIDATOR 0x1u  /* MV-Validator + Vertragsprüfungen */
#define PYR_CONFIG_DEBUG_RING      0x2u  /* Ringbefehle auf Eindeutigkeit/Gültigkeit prüfen */
#define PYR_CONFIG_BREADCRUMBS     0x4u  /* Graph-Knoten schreiben Start/Ende in gemappten Speicher */

typedef struct PyrCapacities {
    uint32_t max_slots;            /* bewegte Objekte; Zustand doppelt gepuffert */
    uint32_t max_geometries;
    uint32_t max_materials;
    uint32_t max_textures;
    uint32_t max_grids;
    uint32_t max_blobs;
    uint32_t slot_data_bytes;      /* Obergrenze für eigene Felder pro Slot */
    uint32_t param_arena_bytes;    /* dynamische Hook-Parameter, doppelt gepuffert */
    uint64_t dag_pool_bytes;
    uint64_t brick_pool_bytes;     /* Anker-Bricks */
    uint64_t lod_cache_bytes;
    uint64_t staging_queue_bytes;  /* GPU-Warteschlange für Uploads und budgetierte Bauarbeit */
} PyrCapacities;

typedef struct PyrBudgets {        /* Arbeit pro Frame; der Rest wartet in der Staging-Queue */
    uint32_t chunk_builds;
    uint32_t geometry_builds;
    uint32_t gc_buckets;           /* HashDAG-GC: Buckets pro Frame (inkrementell) */
    uint32_t reserved;
} PyrBudgets;

typedef void (*PyrLogFn)(void* user, int32_t level, const char* message);

typedef struct PyrConfig {
    uint32_t      struct_size;     /* sizeof(PyrConfig) */
    uint32_t      api_version;     /* PYR_API_VERSION */
    int32_t       cuda_device;     /* -1 = automatisch */
    uint32_t      flags;           /* PYR_CONFIG_* */
    uint32_t      render_width, render_height;
    uint32_t      output_width, output_height;
    PyrUpscaler   upscaler;
    uint32_t      ring_segments;   /* >= 2, Standard 3 */
    uint32_t      ring_segment_bytes;
    uint32_t      hang_timeout_ms; /* 0 = Standard (2000) */
    PyrCapacities caps;
    PyrBudgets    budgets;
    PyrLogFn      log;
    void*         log_user;
} PyrConfig;

PYR_API uint32_t  pyr_api_version(void);
PYR_API PyrResult pyr_create(const PyrConfig* config, PyrRuntime** out);
PYR_API void      pyr_destroy(PyrRuntime* rt);   /* wartet auf die GPU */

/* Wartet auf die GPU, legt Bilder und DLSS neu an und erhöht PyrExports.generation.
 * Der Client muss danach neu importieren. Weltinhalt bleibt erhalten. */
PYR_API PyrResult pyr_resize(PyrRuntime* rt, uint32_t render_width, uint32_t render_height,
                             uint32_t output_width, uint32_t output_height);

/* ---------------------------------------------------------------------------
 * Ausgabe-Bilder und Synchronisation
 *
 * Die Runtime besitzt ein eigenes Vulkan-Device und exportiert Bilder und
 * binäre Semaphoren. Der Client importiert sie in GL (GL_EXT_memory_object,
 * GL_EXT_semaphore), Vulkan (VK_KHR_external_memory/_semaphore) oder D3D12.
 * Exportierte Handles gehören nach dem Aufruf dem Client.
 * ------------------------------------------------------------------------- */

typedef int32_t PyrImage;
enum {
    PYR_IMAGE_COLOR_OUT = 0,  /* Ausgabeauflösung, nach Upscaling, RGBA16F */
    PYR_IMAGE_DEPTH_OUT,      /* Ausgabeauflösung, R32F, für Compositing danach */
    PYR_IMAGE_COLOR,          /* Renderauflösung, vor Upscaling (Split-Modus) */
    PYR_IMAGE_DEPTH,
    PYR_IMAGE_MOTION,         /* RG16F, Pixel, vorher − jetzt, ohne Jitter */
    PYR_IMAGE_REACTIVE,       /* R8 */
    PYR_IMAGE_DEBUG,          /* Validator-Heatmap und Debug-Ansichten */
    PYR_IMAGE_COUNT
};

typedef int32_t PyrHandleKind;
enum { PYR_HANDLE_NONE = 0, PYR_HANDLE_OPAQUE_FD = 1, PYR_HANDLE_OPAQUE_WIN32 = 2 };

typedef struct PyrExternalHandle {
    PyrHandleKind kind;
    int32_t       reserved;
    int64_t       value;      /* fd bzw. HANDLE */
} PyrExternalHandle;

typedef struct PyrExportedImage {
    PyrExternalHandle memory;
    uint64_t allocation_size;
    uint64_t offset;
    uint32_t vk_format;       /* VkFormat */
    uint32_t width, height;
    uint32_t dedicated;       /* 1 = dedizierte Allokation */
} PyrExportedImage;

typedef struct PyrExports {
    uint32_t          struct_size;
    uint32_t          generation;       /* steigt bei pyr_resize/pyr_recover */
    uint8_t           device_uuid[16];  /* Client muss dasselbe physische Gerät verwenden */
    PyrExportedImage  images[PYR_IMAGE_COUNT];
    PyrExternalHandle sem_render_done;  /* Runtime signalisiert: Bilder fertig */
    PyrExternalHandle sem_client_done;  /* Client signalisiert: Bilder gelesen */
    PyrExternalHandle sem_composite_done; /* nur Split-Modus: Client hat vor dem Upscaling gezeichnet */
} PyrExports;

PYR_API PyrResult pyr_get_exports(PyrRuntime* rt, PyrExports* out);

/* ---------------------------------------------------------------------------
 * Frames
 * ------------------------------------------------------------------------- */

typedef struct PyrSegment {
    void*    base;       /* write-combined: nur schreiben */
    uint32_t capacity;
    uint32_t reserved;
    uint64_t frame;      /* Nummer des Frames, den dieses Segment speist */
} PyrSegment;

/* Wartet, bis das nächste Segment frei ist (GPU hat Frame frame - ring_segments
 * beendet). Liefert PYR_TIMEOUT, PYR_ERR_DEVICE_LOST oder PYR_ERR_DEVICE_HUNG. */
PYR_API PyrResult pyr_acquire(PyrRuntime* rt, uint64_t timeout_ns, PyrSegment* out);

#define PYR_SUBMIT_SPLIT_UPSCALE 0x1u  /* Graph endet vor dem Upscaling, siehe pyr_upscale */

/* Startet den Frame-Graphen. records und payload_top stammen aus PyrWriter.
 * Eine fertige neue Pipeline (pyr_pipeline_commit) wird hier eingewechselt. */
PYR_API PyrResult pyr_submit(PyrRuntime* rt, uint32_t records, uint32_t payload_top, uint32_t flags);

/* Nur nach PYR_SUBMIT_SPLIT_UPSCALE: Der Client hat eigene Rastergeometrie in
 * COLOR/DEPTH/MOTION/REACTIVE gezeichnet und sem_composite_done signalisiert.
 * Startet den zweiten Graphen (Upscaling + Ausgabe). */
PYR_API PyrResult pyr_upscale(PyrRuntime* rt);

/* ---------------------------------------------------------------------------
 * Statusblock: asynchroner Rückkanal
 *
 * Die GPU schreibt am Ende jedes Frames in gemappten Host-Speicher.
 * pyr_get_status kopiert den letzten vollständigen Stand; blockiert nie.
 * ------------------------------------------------------------------------- */

#define PYR_STATUS_BACKPRESSURE     0x01u  /* Staging-Queue fast voll: Uploads drosseln */
#define PYR_STATUS_POOL_EXHAUSTED   0x02u  /* ein Pool ist voll; Arbeit wurde verworfen */
#define PYR_STATUS_INVALID_COMMAND  0x04u  /* siehe bad_command_* */
#define PYR_STATUS_VALIDATOR_FAIL   0x08u
#define PYR_STATUS_HOOK_DISABLED    0x10u  /* nach pyr_recover: suspect_hooks wurden deaktiviert */
#define PYR_STATUS_LOD_EVICTING     0x20u

typedef struct PyrPoolUsage {
    uint64_t used;
    uint64_t capacity;
} PyrPoolUsage;

typedef int32_t PyrPass;
enum {
    PYR_PASS_DELTAS = 0, PYR_PASS_BUILD, PYR_PASS_GC, PYR_PASS_BRICKS, PYR_PASS_BVH,
    PYR_PASS_BEAM, PYR_PASS_PRIMARY, PYR_PASS_SHADING, PYR_PASS_TRANSLUCENCY,
    PYR_PASS_SECONDARY, PYR_PASS_CUSTOM, PYR_PASS_VALIDATOR, PYR_PASS_OUTPUT,
    PYR_PASS_COUNT
};

typedef struct PyrStatus {
    uint32_t     struct_size;
    uint32_t     flags;              /* PYR_STATUS_* */
    uint64_t     frame_completed;
    PyrPoolUsage dag_pool, brick_pool, lod_cache, staging_queue, param_arena;
    uint32_t     pending_chunk_builds;
    uint32_t     pending_geometry_builds;
    uint64_t     bad_command_frame;
    uint32_t     bad_command_index;  /* Record-Index im Segment */
    uint32_t     bad_command_reason;
    uint32_t     suspect_hooks[8];   /* 0-terminiert */
    uint32_t     validator_bad_pixels;
    float        validator_max_error;  /* in Voxeln */
    float        gpu_ms[PYR_PASS_COUNT];
} PyrStatus;

PYR_API void pyr_get_status(PyrRuntime* rt, PyrStatus* out);

/* Nach PYR_ERR_DEVICE_LOST: legt den CUDA-Kontext neu an, deaktiviert die Hooks
 * des abgestürzten Graph-Knotens (Breadcrumbs) und exportiert neue Bilder.
 * Erhalten bleiben Konfiguration, Zustandsschema und registrierte Hooks.
 * Alle GPU-Inhalte (Geometrie, Texturen, Chunks, Slots) muss der Client neu senden. */
PYR_API PyrResult pyr_recover(PyrRuntime* rt);

/* ---------------------------------------------------------------------------
 * Zustandsschema: eigene, automatisch doppelt gepufferte Felder pro Slot.
 * Nur vor dem ersten pyr_pipeline_commit bzw. zwischen Commits; ein geändertes
 * Schema setzt alle Slot-Daten auf 0.
 * ------------------------------------------------------------------------- */

PYR_API PyrResult pyr_state_declare_field(PyrRuntime* rt, const char* name, uint32_t size,
                                          uint32_t align, uint32_t* out_offset);

/* ---------------------------------------------------------------------------
 * Hooks
 * ------------------------------------------------------------------------- */

typedef int32_t PyrHookKind;
enum {
    PYR_HOOK_MATERIAL     = 1,
    PYR_HOOK_UV_FLOW      = 2,
    PYR_HOOK_WARP         = 3,
    PYR_HOOK_VOXEL_ANIM   = 4,
    PYR_HOOK_GEOMETRY_SDF = 5,
    PYR_HOOK_PASS         = 6
};

typedef int32_t PyrSourceLang;
enum {
    PYR_SRC_CUDA_CPP = 1,  /* wird in die generierten Kernel eingebettet und geinlinet */
    PYR_SRC_PTX      = 2   /* z. B. aus Zig; wird als echter Aufruf gelinkt (langsamer) */
};

#define PYR_ACCESS_READ  0x1u
#define PYR_ACCESS_WRITE 0x2u

typedef struct PyrPassIO {
    const char* buffer;    /* "color", "depth", "motion", "reactive", "hit", "surface"
                              oder ein Name aus pyr_buffer_declare */
    uint32_t    access;    /* PYR_ACCESS_* */
    uint32_t    reserved;
} PyrPassIO;

typedef struct PyrHookDesc {
    uint32_t         struct_size;
    PyrHookKind      kind;
    const char*      name;          /* eindeutig, z. B. "example:sway" */
    const char*      entry;         /* Symbolname der Funktion */
    PyrSourceLang    lang;
    const char*      source;
    size_t           source_size;
    uint32_t         params_size;   /* erwartete Größe eines Parameterblocks in der Arena */
    uint32_t         max_iterations;/* Obergrenze der Runtime-Schleifen um den Hook, 0 = Standard */
    /* Verträge */
    float            max_displacement; /* WARP, VOXEL_ANIM */
    float            lipschitz;        /* WARP: 0 <= L < 1 */
    float            uv_period[2];     /* UV_FLOW: Periode der UV in Texeln (z. B. Texturgröße) */
    float            bounds_min[3];    /* GEOMETRY_SDF, VOXEL_ANIM: Objektraum */
    float            bounds_max[3];
    /* nur PYR_HOOK_PASS */
    const PyrPassIO* pass_io;
    uint32_t         pass_io_count;
    PyrPass          pass_after;
} PyrHookDesc;

/* Übersetzt den Hook allein (NVRTC) und prüft das PTX statisch: keine globalen
 * Variablen, keine Zeitregister (%clock, %globaltimer), keine atomaren Operationen,
 * keine globalen Stores (außer Pass-Hooks auf deklarierte Ausgaben).
 * *out_log bleibt bis zum nächsten Aufruf gültig. Hook-IDs sind >= 1 und stabil. */
PYR_API PyrResult pyr_hook_register(PyrRuntime* rt, const PyrHookDesc* desc,
                                    uint32_t* out_hook, const char** out_log);
PYR_API PyrResult pyr_hook_set_enabled(PyrRuntime* rt, uint32_t hook, int32_t enabled);

/* Zusätzliche Puffer für Pass-Hooks: pro Pixel (bytes_per_pixel) oder fest (fixed_bytes). */
PYR_API PyrResult pyr_buffer_declare(PyrRuntime* rt, const char* name,
                                     uint32_t bytes_per_pixel, uint64_t fixed_bytes);

/* ---------------------------------------------------------------------------
 * Pipeline: erzeugt aus allen aktiven Hooks die CUDA-C++-Kernel (Traversierung,
 * Shading, Anker-Bricks, Voxelisierung), übersetzt sie mit NVRTC, linkt mit
 * nvJitLink und instanziiert den Graphen neu – im Hintergrund. Die fertige
 * Pipeline wird beim nächsten pyr_submit eingewechselt. Ringbefehle dürfen einen
 * Hook erst referenzieren, wenn eine Pipeline mit diesem Hook aktiv ist.
 * ------------------------------------------------------------------------- */

PYR_API PyrResult pyr_pipeline_commit(PyrRuntime* rt, uint64_t* out_ticket);
PYR_API PyrResult pyr_pipeline_poll(PyrRuntime* rt, uint64_t ticket, const char** out_log);

#ifdef __cplusplus
}
#endif

#endif /* PYRIT_H */
