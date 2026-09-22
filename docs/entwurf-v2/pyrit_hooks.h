/*
 * pyrit_hooks.h – Geräte-seitige Hook-ABI (Entwurf, Hook-ABI 0.1)
 *
 * Wird von NVRTC beim Übersetzen jedes Hooks und der generierten Kernel
 * eingebunden. Die Strukturlayouts erzeugt in der Implementierung der Zig-Build
 * aus denselben `extern struct`s, die die Zig-Kernel verwenden.
 *
 * Verträge für alle Hooks außer PYR_HOOK_PASS:
 *   1. Ergebnis hängt nur von den Argumenten ab (Position, Kontext, Zustand, Parameter).
 *      Keine globalen Variablen, keine Zeitregister, keine atomaren Operationen,
 *      keine Schreibzugriffe außer auf Ausgabeparameter. Die Runtime prüft das
 *      statisch im PTX (pyr_hook_register) und dynamisch im MV-Validator.
 *   2. Zufall nur über pyr_rand(ctx, ...), also aus stabiler ID + Zustand.
 *   3. Zeit nur über pyr_time_phase() bzw. s->dt, nie über (float)s->time.
 *   4. Derselbe Hook wird pro Pixel zweimal aufgerufen: mit dem aktuellen und dem
 *      Vorframe-Zustand. `params` zeigt jeweils in die passende Arena-Hälfte.
 *
 * Signaturen (Funktionsname frei, `entry` in PyrHookDesc):
 *
 *   MATERIAL:      extern "C" __device__ PyrSurface f(const PyrHitInfo* hit, const PyrHookCtx* ctx,
 *                                                     const PyrFrameState* s, const void* params);
 *   UV_FLOW:       extern "C" __device__ float2 f(float3 p_rest, int32_t face, const PyrHookCtx* ctx,
 *                                                 const PyrFrameState* s, const void* params);
 *                  Liefert UV in Texeln, nur modulo PyrHookDesc.uv_period definiert: Die Runtime
 *                  rechnet Differenzen zwischen Frames modulo Periode (kürzester Weg) und macht
 *                  das Wrapping in den Atlas selbst. Innerhalb einer Periode muss die Funktion stetig sein.
 *   WARP:          extern "C" __device__ float3 f(float3 p_rest, const PyrHookCtx* ctx,
 *                                                 const PyrFrameState* s, const void* params);
 *                  Liefert die verformte Position p + d(p); |d| <= max_displacement,
 *                  Lipschitz-Konstante von d < lipschitz < 1.
 *   VOXEL_ANIM:    extern "C" __device__ bool f(int3 voxel, const PyrHookCtx* ctx, const PyrFrameState* s,
 *                                               const void* params, float3* pos_out);
 *                  false = Voxel in diesem Frame unsichtbar.
 *   GEOMETRY_SDF:  extern "C" __device__ float f(float3 p, const PyrHookCtx* ctx, const void* params);
 *                  Kein Zustand: Geometrie ist statisch; bewegt wird sie über Instanzen.
 *   PASS:          extern "C" __global__ void f(const PyrPassArgs* args);
 */
#ifndef PYRIT_HOOKS_H
#define PYRIT_HOOKS_H

#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
#  error "pyrit_hooks.h ist nur für Gerätecode (NVRTC/nvcc) gedacht"
#endif

#ifndef __CUDACC_RTC__
#  include <stdint.h>
#endif

#define PYR_HOOK_ABI_MAJOR 0
#define PYR_HOOK_ABI_MINOR 1

#define PYR_NO_SLOT 0xFFFFFFFFu

/* ---------------------------------------------------------------------------
 * Zustand (eine Hälfte des Doppelpuffers)
 * ------------------------------------------------------------------------- */

struct PyrCameraState {
    float view[12];          /* Welt (relativ zu origin) -> Kamera */
    float proj[16];          /* ohne Jitter */
    float jitter[2];
    float viewport[2];       /* Renderauflösung in Pixeln */
};

struct PyrSlot {
    float    xform[12];      /* Objekt -> Welt */
    float    inv_xform[12];
    uint32_t generation;
    uint32_t flags;
    uint32_t reserved[2];
};

struct PyrFrameState {
    uint64_t              frame;
    double                time;
    float                 dt;
    uint32_t              reserved0;
    double                origin[3];
    uint32_t              reserved1[2];
    PyrCameraState        camera;
    const PyrSlot*        slots;
    const uint8_t*        slot_data;         /* eigene Felder, Stride slot_data_stride */
    uint32_t              slot_data_stride;
    uint32_t              reserved2;
    const uint8_t*        params;            /* Param-Arena dieser Hälfte */
    const unsigned long long* textures;      /* cudaTextureObject_t pro Textur-Index */
};

/* ---------------------------------------------------------------------------
 * Kontext: Identität des getroffenen Punkts. Für aktuellen und Vorframe gleich.
 * ------------------------------------------------------------------------- */

struct PyrHookCtx {
    uint32_t slot;           /* PYR_NO_SLOT bei statischen Gitterzellen */
    uint32_t slot_generation;
    uint32_t geometry;
    uint32_t material;
    uint32_t grid;           /* 0xFFFFFFFF, wenn kein Gitter */
    uint32_t cell_type;
    int32_t  cell[3];        /* Weltzellkoordinate, sonst 0 */
    uint32_t user_id;        /* aus INSTANCE_SET bzw. GRID_CELLTYPE_SET */
    uint32_t seed;           /* Hash aus (slot, generation) bzw. (grid, cell) */
    uint32_t param_offset;   /* Offset des Parameterblocks dieses Hooks */
    uint32_t reserved[2];
};

struct PyrHitInfo {
    float3   p_rest;         /* Ruheposition im Objektraum (bzw. Zellraum 0..1) */
    float3   p_world;        /* relativ zu origin */
    float3   dir;            /* Strahlrichtung */
    float    t;
    int32_t  face;           /* 0..5: +x, -x, +y, -y, +z, -z */
    int32_t  element;        /* von der Geometrie gelieferte Element-ID oder -1 */
    float2   uv;             /* in Texeln; bei UV_FLOW aus dem Hook (modulo uv_period) */
    uint32_t flags;          /* PYR_HIT_* */
    uint32_t reserved;
};

#define PYR_HIT_NEW         0x1u  /* keine Vorgeschichte -> Reactive-Maske */
#define PYR_HIT_TRANSLUCENT 0x2u

/* Ausgabe des Material-Hooks: Oberflächenbeschreibung statt fertiger Farbe,
 * damit Schatten, GI und Reflexionen dieselben Daten nutzen. */
struct PyrSurface {
    float3   albedo;
    float    roughness;
    float3   normal;         /* Welt; Nullvektor = geometrische Normale verwenden */
    float    metallic;
    float3   emission;
    float    opacity;        /* < 1 nur bei PYR_MAT_TRANSLUCENT */
    uint32_t flags;
    uint32_t reserved[3];
};

struct PyrPassBuffer {
    void*    data;
    uint64_t size;
    uint32_t pitch;          /* Bytes pro Zeile bei Pixelpuffern, sonst 0 */
    uint32_t access;
};

struct PyrPassArgs {
    const PyrFrameState* cur;
    const PyrFrameState* prev;
    uint32_t             width, height;
    uint32_t             buffer_count;
    uint32_t             reserved;
    PyrPassBuffer        buffers[16];   /* Reihenfolge wie PyrHookDesc.pass_io */
};

/* ---------------------------------------------------------------------------
 * Hilfsfunktionen
 * ------------------------------------------------------------------------- */

/* Phase in [0, 1) einer Schwingung mit hz Hertz. Rechnet in double und bleibt
 * auch nach Tagen Laufzeit exakt; Ergebnis in sinf(6.2831853f * phase) verwenden. */
static __device__ __forceinline__ float pyr_time_phase(const PyrFrameState* s, double hz) {
    double x = s->time * hz;
    return (float)(x - floor(x));
}

static __device__ __forceinline__ uint32_t pyr_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* Deterministischer Zufall in [0, 1) aus stabiler ID und salt. */
static __device__ __forceinline__ float pyr_rand(const PyrHookCtx* ctx, uint32_t salt) {
    return (float)(pyr_hash(ctx->seed ^ pyr_hash(salt)) >> 8) * (1.0f / 16777216.0f);
}

/* Eigenes Slot-Feld (Offset aus pyr_state_declare_field). */
template <typename T>
static __device__ __forceinline__ const T* pyr_slot_field(const PyrFrameState* s, const PyrHookCtx* ctx,
                                                          uint32_t offset) {
    return (const T*)(s->slot_data + (uint64_t)ctx->slot * s->slot_data_stride + offset);
}

static __device__ __forceinline__ float4 pyr_tex2d(const PyrFrameState* s, uint32_t texture, float2 uv) {
    return tex2D<float4>((cudaTextureObject_t)s->textures[texture], uv.x, uv.y);
}

static __device__ __forceinline__ float4 pyr_tex2d_layer(const PyrFrameState* s, uint32_t texture,
                                                          float2 uv, int32_t layer) {
    return tex2DLayered<float4>((cudaTextureObject_t)s->textures[texture], uv.x, uv.y, layer);
}

/* Aux-Kanal eines Gitter-Chunks (PYR_CMD_GRID_CHUNK_AUX), 0 wenn nicht vorhanden.
 * Implementiert von der Runtime im generierten Code. */
extern "C" __device__ uint64_t pyr_chunk_aux(const PyrHookCtx* ctx, uint32_t channel);

#endif /* PYRIT_HOOKS_H */
