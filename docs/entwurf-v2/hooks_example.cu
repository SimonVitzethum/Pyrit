// Beispiel-Hooks für Pyrit (Hook-ABI 0.1). Übersetzung zur Laufzeit per NVRTC;
// hier nur zum Syntaxtest mit nvcc: nvcc -rdc=true -c -Iinclude examples/hooks_example.cu
#include "pyrit_hooks.h"

struct SwayParams { float amplitude; float hz; float spatial; float reserved; };

// Warp: wehendes Laub. |d| <= amplitude, Lipschitz ~ amplitude * (2*pi*spatial + 1) -> bei Registrierung angeben.
extern "C" __device__ float3 example_sway(float3 p, const PyrHookCtx* ctx, const PyrFrameState* s, const void* params) {
    const SwayParams* sp = (const SwayParams*)params;
    float phase = pyr_time_phase(s, sp->hz) + pyr_rand(ctx, 1) + p.x * sp->spatial;
    float a = sp->amplitude * sinf(6.2831853f * phase) * p.y;       // p.y in [0,1]: Fuß fest, Spitze bewegt
    return make_float3(p.x + a, p.y, p.z);
}

struct ConveyorParams { float texels_per_second; uint32_t texture; float reserved[2]; };

// Texturfluss: UV in Texeln modulo uv_period = {16, 16} (bei der Registrierung angegeben).
// Die Verschiebung wird in double gewickelt, damit sie auch nach Tagen exakt bleibt.
extern "C" __device__ float2 example_conveyor_uv(float3 p, int32_t face, const PyrHookCtx* ctx,
                                                 const PyrFrameState* s, const void* params) {
    const ConveyorParams* cp = (const ConveyorParams*)params;
    double shift = s->time * (double)cp->texels_per_second;
    float wrapped = (float)(shift - 16.0 * floor(shift / 16.0));   // Sprung um 16 ist modulo Periode unsichtbar
    return make_float2(p.x * 16.0f + wrapped, p.z * 16.0f);
}

// Material: Textur + Emission, liefert Oberflächenbeschreibung statt Farbe.
extern "C" __device__ PyrSurface example_material(const PyrHitInfo* hit, const PyrHookCtx* ctx,
                                                  const PyrFrameState* s, const void* params) {
    const ConveyorParams* cp = (const ConveyorParams*)params;
    float4 c = pyr_tex2d(s, cp->texture, make_float2(hit->uv.x / 16.0f, hit->uv.y / 16.0f));
    PyrSurface out = {};
    out.albedo = make_float3(c.x, c.y, c.z);
    out.roughness = 0.8f;
    out.opacity = 1.0f;
    return out;
}
