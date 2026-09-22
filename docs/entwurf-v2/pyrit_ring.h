/*
 * pyrit_ring.h – Befehlsformat des Pyrit-Ringpuffers (Entwurf, Format 0.1)
 *
 * Der Ringpuffer ist der eigentliche Datenvertrag zwischen Client und Runtime.
 * In der Implementierung erzeugt der Zig-Build diesen Header aus den
 * `extern struct`s der Runtime; dieser Entwurf ist die Vorlage dafür.
 *
 * Aufbau eines Segments (Kapazität C Bytes):
 *
 *   0                                                                     C
 *   | PyrCmd[0] | PyrCmd[1] | ... | PyrCmd[n-1] |  frei  | Payloads ...   |
 *   Records wachsen von vorn (je 64 Byte), Payloads von hinten (16-Byte-aligned).
 *
 * Regeln:
 *   - Records haben feste Größe, damit die GPU sie parallel dekodieren kann
 *     (ein Thread pro Record, danach Prefix-Scan über Element-Anzahlen).
 *   - Die Reihenfolge der Anwendung richtet sich nach der Befehlsklasse
 *     (siehe PYR_CMD_CLASS_*), nicht nach der Position im Segment.
 *   - Pro Frame höchstens ein Befehl je (Befehlsart, Ziel). Ausnahme: Batch-Befehle,
 *     deren Elemente ebenfalls eindeutige Ziele haben müssen. Der Debug-Build prüft das.
 *   - Das Segment liegt in write-combined, gepinntem Speicher: nur schreiben, nie lesen.
 *   - Alle Handles (PyrRef) vergibt der Client selbst innerhalb der Kapazitäten aus
 *     PyrConfig. Die Runtime liefert deshalb nie IDs zurück.
 *   - Alle reservierten Felder müssen 0 sein.
 */
#ifndef PYRIT_RING_H
#define PYRIT_RING_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYR_RING_FORMAT_MAJOR 0
#define PYR_RING_FORMAT_MINOR 1

/* Handle: Index in einer clientverwalteten Tabelle plus Generation.
 * Wird ein Index wiederverwendet, muss die Generation steigen. */
typedef struct PyrRef {
    uint32_t index;
    uint32_t generation;
} PyrRef;

#define PYR_NONE_INDEX 0xFFFFFFFFu
#define PYR_NO_HOOK    0u     /* Hook-IDs beginnen bei 1 */
#define PYR_NO_BLOB    0u     /* Blob-IDs beginnen bei 1; 0 = Daten liegen im Payload */

/* ---------------------------------------------------------------------------
 * Record
 * ------------------------------------------------------------------------- */

typedef struct PyrCmd {
    uint16_t type;            /* PYR_CMD_* */
    uint16_t flags;           /* reserviert, 0 */
    uint32_t payload_size;    /* Bytes, ohne Padding */
    uint32_t payload_offset;  /* ab Segmentanfang, 16-Byte-aligned */
    uint32_t reserved;
    uint8_t  args[48];        /* typspezifisch, siehe PyrArgs* */
} PyrCmd;

/* ---------------------------------------------------------------------------
 * Befehlsarten. Das obere Nibble ist die Befehlsklasse; Klassen werden
 * pro Frame in aufsteigender Reihenfolge angewendet.
 * ------------------------------------------------------------------------- */

enum {
    PYR_CMD_CLASS_FRAME     = 0x0,  /* Zeit, Kamera, Ursprung */
    PYR_CMD_CLASS_UPLOAD    = 0x1,  /* Blobs */
    PYR_CMD_CLASS_SLOT_LIFE = 0x2,  /* Slot-Belegung */
    PYR_CMD_CLASS_RESOURCE  = 0x3,  /* Texturen, Geometrie, Materialien */
    PYR_CMD_CLASS_STATE     = 0x4,  /* Transformationen, Slot-Daten, Parameter */
    PYR_CMD_CLASS_INSTANCE  = 0x5,
    PYR_CMD_CLASS_GRID      = 0x6
};

enum {
    PYR_CMD_NOP               = 0x0000,
    PYR_CMD_FRAME             = 0x0001, /* genau einmal pro Frame, Payload: PyrFrameArgs */

    PYR_CMD_BLOB_BEGIN        = 0x1000, /* args: PyrArgsBlobBegin */
    PYR_CMD_BLOB_DATA         = 0x1001, /* args: PyrArgsBlobData, Payload: Daten */

    PYR_CMD_SLOT_ALLOC        = 0x2000, /* Payload: PyrRef[] (neue Generation) */
    PYR_CMD_SLOT_FREE         = 0x2001, /* Payload: PyrRef[] */

    PYR_CMD_TEXTURE_CREATE    = 0x3000, /* args: PyrArgsTextureCreate */
    PYR_CMD_TEXTURE_UPLOAD    = 0x3001, /* args: PyrArgsTextureUpload */
    PYR_CMD_TEXTURE_FREE      = 0x3002, /* args: PyrRef */
    PYR_CMD_GEOMETRY_BUILD    = 0x3010, /* args: PyrArgsGeometryBuild */
    PYR_CMD_GEOMETRY_FREE     = 0x3011, /* args: PyrRef */
    PYR_CMD_MATERIAL_SET      = 0x3020, /* args: PyrArgsMaterialSet */

    PYR_CMD_SLOT_TRANSFORMS   = 0x4000, /* Payload: PyrSlotTransform[] */
    PYR_CMD_SLOT_DATA         = 0x4001, /* args: PyrArgsSlotData, Payload: Bytes */
    PYR_CMD_PARAM_WRITE       = 0x4002, /* args: PyrArgsParamWrite, Payload: Bytes */

    PYR_CMD_INSTANCE_SET      = 0x5000, /* args: PyrArgsInstanceSet */
    PYR_CMD_INSTANCE_CLEAR    = 0x5001, /* args: PyrRef (Slot) */

    PYR_CMD_GRID_CONFIG       = 0x6000, /* args: PyrArgsGridConfig */
    PYR_CMD_GRID_RECENTER     = 0x6001, /* args: PyrArgsGridChunk (Mittelpunkt) */
    PYR_CMD_GRID_CELLTYPE_SET = 0x6002, /* args: PyrArgsGridCellType */
    PYR_CMD_GRID_CHUNK_LOAD   = 0x6003, /* args: PyrArgsGridChunkData, Daten: PyrChunkCells */
    PYR_CMD_GRID_CHUNK_UNLOAD = 0x6004, /* args: PyrArgsGridChunk */
    PYR_CMD_GRID_CELL_EDITS   = 0x6005, /* args: PyrArgsGridChunk (nur grid), Payload: PyrCellEdit[] */
    PYR_CMD_GRID_CHUNK_AUX    = 0x6006  /* args: PyrArgsGridChunkData, Daten: Aux-Kanal */
};

#define PYR_CMD_CLASS(type) ((uint16_t)(type) >> 12)

/* ---------------------------------------------------------------------------
 * Frame
 * ------------------------------------------------------------------------- */

typedef struct PyrCamera {
    float view[12];      /* Welt (relativ zu origin) -> Kamera, 3x4 zeilenweise */
    float proj[16];      /* Kamera -> Clip, ohne Jitter */
    float jitter[2];     /* Subpixel-Jitter in Pixeln; MVs werden ohne Jitter geschrieben */
    float reserved[2];
} PyrCamera;

typedef struct PyrFrameArgs {
    double    time;       /* Sekunden, monoton; auf der GPU nur über pyr_time_phase() nutzen */
    float     dt;
    uint32_t  reserved0;
    double    origin[3];  /* Render-Ursprung in Weltkoordinaten; alle float-Positionen relativ dazu */
    uint32_t  reserved1[2];
    PyrCamera camera;
} PyrFrameArgs;

/* ---------------------------------------------------------------------------
 * Uploads über mehrere Frames
 *
 * Große Daten (Geometrie, Texturen, Chunks) werden als Blob in Teilen über
 * beliebig viele Frames übertragen. Die GPU kopiert jeden Teil sofort in die
 * Staging-Warteschlange (PyrCapacities.staging_queue_bytes), sodass das
 * Ring-Segment danach wieder frei ist. Ein Blob wird genau einmal von einem
 * Befehl verbraucht und danach freigegeben. Ist die Warteschlange voll, setzt
 * die Runtime PYR_STATUS_BACKPRESSURE; der Client muss dann drosseln.
 * ------------------------------------------------------------------------- */

typedef struct PyrArgsBlobBegin {
    uint32_t blob;
    uint32_t reserved;
    uint64_t total_size;
} PyrArgsBlobBegin;

typedef struct PyrArgsBlobData {
    uint32_t blob;
    uint32_t reserved;
    uint64_t offset;
} PyrArgsBlobData;

/* ---------------------------------------------------------------------------
 * Ressourcen
 * ------------------------------------------------------------------------- */

enum {
    PYR_TEX_RGBA8_SRGB = 1,
    PYR_TEX_RGBA8      = 2,
    PYR_TEX_RGBA16F    = 3,
    PYR_TEX_R8         = 4,
    PYR_TEX_BC7_SRGB   = 5
};

typedef struct PyrArgsTextureCreate {
    PyrRef   texture;
    uint32_t format;     /* PYR_TEX_* */
    uint32_t width, height, layers, mips;
    uint32_t flags;      /* PYR_TEX_FLAG_* */
} PyrArgsTextureCreate;

#define PYR_TEX_FLAG_NEAREST 0x1u   /* Pixel-Art: kein bilineares Filtern */
#define PYR_TEX_FLAG_WRAP    0x2u

typedef struct PyrArgsTextureUpload {
    PyrRef   texture;
    uint32_t mip, layer;
    uint32_t blob;       /* PYR_NO_BLOB: Daten im Payload */
} PyrArgsTextureUpload;

enum {
    PYR_GEOM_VOXEL_BRICKS = 1, /* dünn besetzte 4x4x4-Bricks, optional mit Attributen; GPU baut die DAG */
    PYR_GEOM_ATTRIB_DAG   = 2, /* offline gebaute DAG mit komprimierten Attributen (Dado et al. 2016) */
    PYR_GEOM_SDF          = 3  /* GPU voxelisiert einen Geometrie-Hook */
};

typedef struct PyrArgsGeometryBuild {
    PyrRef   geometry;
    uint32_t kind;          /* PYR_GEOM_* */
    uint32_t source;        /* Blob-ID bzw. bei PYR_GEOM_SDF die Hook-ID */
    uint32_t param_offset;  /* Param-Arena, nur PYR_GEOM_SDF */
    uint32_t res_log2;      /* Voxel pro Kante = 1 << res_log2 */
    float    bounds_min[3]; /* Objektraum */
    float    bounds_max[3];
} PyrArgsGeometryBuild;

enum {
    PYR_MV_GEOMETRY         = 0, /* Standard */
    PYR_MV_GEOMETRY_UV_FLOW = 1, /* zusätzlich Texturfluss, verlangt uv_flow_hook */
    PYR_MV_REACTIVE         = 2  /* keine stetige Bewegung, in Reactive-Maske markieren */
};

#define PYR_MAT_TRANSLUCENT 0x1u
#define PYR_MAT_EMISSIVE    0x2u

typedef struct PyrArgsMaterialSet {
    PyrRef   material;
    uint32_t material_hook;
    uint32_t uv_flow_hook;       /* PYR_NO_HOOK außer bei PYR_MV_GEOMETRY_UV_FLOW */
    uint32_t mv_source;          /* PYR_MV_* */
    uint32_t flags;              /* PYR_MAT_* */
    uint32_t param_offset;       /* Param-Arena des Material-Hooks */
    uint32_t flow_param_offset;  /* Param-Arena des UV-Fluss-Hooks */
} PyrArgsMaterialSet;

/* ---------------------------------------------------------------------------
 * Zustand
 * ------------------------------------------------------------------------- */

typedef struct PyrSlotTransform {
    PyrRef slot;          /* Generation muss zur aktuellen Belegung passen */
    float  xform[12];     /* Objekt -> Welt (relativ zu origin), 3x4 zeilenweise */
} PyrSlotTransform;

typedef struct PyrArgsSlotData {
    PyrRef   slot;
    uint32_t offset;      /* aus pyr_state_declare_field() */
    uint32_t reserved;
} PyrArgsSlotData;

typedef struct PyrArgsParamWrite {
    uint32_t offset;      /* clientverwaltete Position in der Param-Arena, 16-Byte-aligned */
    uint32_t reserved;
} PyrArgsParamWrite;

/* ---------------------------------------------------------------------------
 * Instanzen: eine Instanz belegt genau einen Slot. Die Bewegungsklasse folgt
 * aus den Angaben: nur Transformation = Rigid, warp_hook = Warped,
 * anim_hook = Anker-Bricks.
 * ------------------------------------------------------------------------- */

#define PYR_INST_CAST_SHADOW 0x1u
#define PYR_INST_VISIBLE     0x2u

typedef struct PyrArgsInstanceSet {
    PyrRef   slot;
    PyrRef   geometry;
    PyrRef   material;     /* PYR_NONE_INDEX = Attribute der Geometrie verwenden */
    uint32_t warp_hook, warp_param;
    uint32_t anim_hook, anim_param;
    uint32_t flags;        /* PYR_INST_* */
    uint32_t user_id;      /* frei für den Client, erscheint in PyrHookCtx */
} PyrArgsInstanceSet;

/* ---------------------------------------------------------------------------
 * Gitter-Welten: Ringpuffer-Gitter aus Chunks um die Kamera. Jede Zelle
 * verweist auf einen Zelltyp (Geometrie + Material + optionaler Warp).
 * Zelltyp 0 ist immer leer.
 * ------------------------------------------------------------------------- */

typedef struct PyrArgsGridConfig {
    uint32_t grid;
    uint32_t chunk_log2;     /* Zellen pro Chunk-Kante = 1 << chunk_log2 */
    uint32_t cell_res_log2;  /* Voxel pro Zellkante (Modell-DAG-Auflösung) */
    uint32_t extent[3];      /* Ringgröße in Chunks */
    float    cell_size;      /* Weltgröße einer Zelle */
    uint32_t flags;
} PyrArgsGridConfig;

#define PYR_CELL_FULL_OPAQUE 0x1u  /* vollständig gefüllt und opak: Abkürzung in der DDA */

typedef struct PyrArgsGridCellType {
    uint32_t grid;
    uint32_t cell_type;
    PyrRef   geometry;
    PyrRef   material;
    uint32_t warp_hook, warp_param;
    uint32_t flags;          /* PYR_CELL_* */
    uint32_t user_id;
} PyrArgsGridCellType;

typedef struct PyrArgsGridChunk {
    uint32_t grid;
    int32_t  chunk[3];
} PyrArgsGridChunk;

typedef struct PyrArgsGridChunkData {
    uint32_t grid;
    int32_t  chunk[3];
    uint32_t blob;           /* PYR_NO_BLOB: Daten im Payload */
    uint32_t channel;        /* nur CHUNK_AUX: 0..7 */
    uint32_t format;         /* nur CHUNK_AUX: Bytes pro Zelle (1, 2, 4, 8) */
    uint32_t reserved;
} PyrArgsGridChunkData;

/* Zelldaten eines Chunks: Palette + bitgepackte Indizes, x läuft am schnellsten,
 * dann z, dann y; LSB zuerst. Direkt danach im Speicher:
 *   uint32_t palette[palette_count];
 *   uint32_t packed[ceil(cells * bits / 32)];                                   */
typedef struct PyrChunkCells {
    uint32_t palette_count;
    uint32_t bits;           /* 0 = der ganze Chunk ist palette[0] */
} PyrChunkCells;

typedef struct PyrCellEdit {
    int32_t  cell[3];        /* Weltzellkoordinate */
    uint32_t cell_type;
} PyrCellEdit;

/* ---------------------------------------------------------------------------
 * Layout-Prüfungen
 * ------------------------------------------------------------------------- */

#if defined(__cplusplus)
#  define PYR_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#  define PYR_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif

PYR_STATIC_ASSERT(sizeof(PyrCmd) == 64, "PyrCmd muss 64 Byte groß sein");
PYR_STATIC_ASSERT(sizeof(PyrArgsBlobBegin)     <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsBlobData)      <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsTextureCreate) <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsTextureUpload) <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsGeometryBuild) <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsMaterialSet)   <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsSlotData)      <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsParamWrite)    <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsInstanceSet)   <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsGridConfig)    <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsGridCellType)  <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsGridChunk)     <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrArgsGridChunkData) <= 48, "args zu groß");
PYR_STATIC_ASSERT(sizeof(PyrSlotTransform) == 56, "PyrSlotTransform-Layout");
PYR_STATIC_ASSERT(sizeof(PyrFrameArgs) == 176, "PyrFrameArgs-Layout");

/* ---------------------------------------------------------------------------
 * Schreibhilfe (nur Client-seitig, rein inline)
 * ------------------------------------------------------------------------- */

typedef struct PyrWriter {
    uint8_t* base;
    uint32_t capacity;
    uint32_t records;
    uint32_t payload_top;    /* Payloads liegen in [payload_top, capacity) */
} PyrWriter;

static inline void pyr_writer_init(PyrWriter* w, void* base, uint32_t capacity) {
    w->base = (uint8_t*)base;
    w->capacity = capacity;
    w->records = 0;
    w->payload_top = capacity & ~15u;
}

/* Legt einen Record an. Liefert NULL, wenn das Segment voll ist; der Client
 * verschiebt den Rest dann in den nächsten Frame. Args werden über einen lokalen
 * Puffer gesetzt und mit memcpy geschrieben (write-combined Speicher). */
static inline PyrCmd* pyr_writer_push(PyrWriter* w, uint16_t type, const void* args, uint32_t args_size,
                                      uint32_t payload_size, void** payload) {
    uint32_t aligned = (payload_size + 15u) & ~15u;
    uint32_t records_end = (w->records + 1u) * (uint32_t)sizeof(PyrCmd);
    if (args_size > 48u || aligned > w->payload_top || w->payload_top - aligned < records_end)
        return (PyrCmd*)0;
    w->payload_top -= aligned;

    PyrCmd cmd;
    memset(&cmd, 0, sizeof cmd);
    cmd.type = type;
    cmd.payload_size = payload_size;
    cmd.payload_offset = payload_size ? w->payload_top : 0u;
    if (args_size) memcpy(cmd.args, args, args_size);

    PyrCmd* dst = (PyrCmd*)(w->base + (uint64_t)w->records * sizeof(PyrCmd));
    memcpy(dst, &cmd, sizeof cmd);
    if (payload) *payload = w->base + w->payload_top;
    w->records++;
    return dst;
}

#ifdef __cplusplus
}
#endif

#endif /* PYRIT_RING_H */
