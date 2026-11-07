#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_GATED_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_GATED_CUH
/* ══ nn's gated residual's DOORS, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════
 * nn's `gated_rank` and `gated_mean` (`nn/gpu/kernels/kernels.cuh`), Flash-Next's gated residual between and after its
 * two maps: the rank's activation and its rotation by the family's own butterfly, a warp a block of the ladder, and the
 * streams' gated mean a lane an output, as nn's body — each value rounded to a half through the family's `to_half` (nn's
 * conversion's bits and fault word, ▶ kernels.cuh) where nn's bodies take its branches. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */
#include "hadamard.cuh"     /* a warp's block of the rotation, the one `hadamard_blocks` runs */

/* ⭐ THE RANK, A WARP A BLOCK OF THE LADDER — `gated_rank`: `act` nn's `gated_act` (`vector_scale` and then `swiglu_combine`'s
 *   arithmetic), and `rot` that rotated in the pack's ladder — blocks of 512, then a 256, then a 128 — each block
 *   `hadamard_blocks`' own with the signs from the vector's start. nn's body sums each rotated output directly from its
 *   block's activations, a lane an output, and so answers as the four doors only where the family's `hadamard_blocks` is
 *   that body; this family's is a butterfly (▶ hadamard.cuh), so here the rotation is that butterfly's: a warp takes a block,
 *   a lane writes `act` at its elements `l, l + 32, …` and `rotation_at` rotates them, each lane reading exactly the
 *   elements it has just written — no staging and no wait, so no bound on the rank. `act` is nn's to the bit and `rot` is
 *   `hadamard_blocks`' on this family over that `act` (`REASONED` from the shared code; to be checked by the door bench).
 *   amd_rocm6_wave64 stages the whole rank in one block's shared memory and waits once; there a block is 256 lanes of a
 *   wave of 64, and its rotation reads a contiguous run a lane, which this family's does not. */
static __device__ __forceinline__ uint16_t nvidia_cuda12__zzprivate_gated_act(const uint16_t* a, uint64_t i, float scale, unsigned int* over) {
    const float s = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(nn__silicon__half_to_float(a[i]) * scale, over));
    const float clamped = (s < -88.0f) ? -88.0f : ((s > 88.0f) ? 88.0f : s);
    return nvidia_cuda12__zzpackage_to_half(clamped / (1.0f + nn__silicon__expf(-clamped)) * 1.0f, over);
}
#define NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS 8u
#define NVIDIA_CUDA12__ZZPRIVATE_GATED_BOUNDS __launch_bounds__(256)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GATED_BOUNDS void nvidia_cuda12__zzprivate_gated_rank(uint16_t* rot, uint16_t* act, const uint16_t* a,
                                                                                   const uint16_t* sign, uint64_t rank, float scale,
                                                                                   unsigned int* over) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t whole = rank / 512u, rest = rank - whole * 512u;
    const uint64_t units = whole + (rest >= 256u ? 1u : 0u) + (rest % 256u != 0u ? 1u : 0u);
    for (uint64_t u = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS
                    + (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         u < units; u += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS) {
        const uint32_t E = u < whole ? 16u : (rest >= 256u && u == whole) ? 8u : 4u;
        const uint64_t at = u < whole ? u * 512u : whole * 512u + (E == 4u && rest >= 256u ? 256u : 0u);
        for (uint32_t e = 0u; e < E; ++e) {
            const uint64_t i = at + lane + NVIDIA_CUDA12__ZZPACKAGE_WARP * e;
            act[i] = nvidia_cuda12__zzprivate_gated_act(a, i, scale, over);
        }
        if (E == 16u)     nvidia_cuda12__zzpackage_rotation_at(rot, act, at, sign, 0u, over, 16u);
        else if (E == 8u) nvidia_cuda12__zzpackage_rotation_at(rot, act, at, sign, 0u, over, 8u);
        else              nvidia_cuda12__zzpackage_rotation_at(rot, act, at, sign, 0u, over, 4u);
    }
}
/* Whole blocks of 128 and the three vectors apart; anything else is nn's. */
static void nvidia_cuda12__override_gated_rank(uint16_t* rot, uint16_t* act, const uint16_t* a, const uint16_t* sign,
                                               uint64_t rank, float scale, unsigned int* over) {
    const bool apart = (rot + rank <= a || a + rank <= rot) && (act + rank <= a || a + rank <= act)
                    && (rot + rank <= act || act + rank <= rot);
    if (rank == 0u || rank % 128u != 0u || !apart) {
        nn__gated__zzabi_launch_rank(rot, act, a, sign, rank, scale, over);
        return;
    }
    const uint64_t units = rank / 512u + (rank % 512u >= 256u ? 1u : 0u) + (rank % 256u != 0u ? 1u : 0u);
    const uint64_t want = (units + NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &rot, &act, &a, &sign, &rank, &scale, &over };
    const size_t sizes[] = { sizeof rot, sizeof act, sizeof a, sizeof sign, sizeof rank, sizeof scale, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gated_rank, "nvidia_cuda12__zzprivate_gated_rank", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_GATED_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 7u);
}

/* ⭐ THE GATED MEAN — nn's `gated_mean`: a lane an output, its streams' σ(gate) times the normed stream summed in their
 *   order, each step rounded as nn's body rounds it, the mean, and the first `streams` lanes the injection's weights
 *   2·σ(mean·ij) — nn's arithmetic in its order, so nn's bits; what is this family's own is only the conversion. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GATED_BOUNDS void nvidia_cuda12__zzprivate_gated_mean(uint16_t* into, uint16_t* inj, const uint16_t* gt,
                                                                                   const uint16_t* an, const uint16_t* ij, uint64_t hidden,
                                                                                   uint64_t streams, float mean, unsigned int* over) {
    const uint64_t stride = (uint64_t)nn__silicon__blocks() * nn__silicon__lanes();
    for (uint64_t j = (uint64_t)nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); j < hidden; j += stride) {
        float m = 0.0f;
        for (uint64_t k = 0ull; k < streams; ++k) {
            const uint64_t i = k * hidden + j;
            const float gs = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(
                                 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(gt[i]))), over));
            const float gn = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(gs * nn__silicon__half_to_float(an[i]), over));
            m = k == 0ull ? gn : nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(m + gn, over));
        }
        into[j] = nvidia_cuda12__zzpackage_to_half(m * mean, over);
        if (inj != 0 && ij != 0 && j < streams) {
            const float s = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(nn__silicon__half_to_float(ij[j]) * mean, over));
            const float g = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(1.0f / (1.0f + nn__silicon__expf(-s)), over));
            inj[j] = nvidia_cuda12__zzpackage_to_half(g * 2.0f, over);
        }
    }
}
/* nn's launch's lanes — a lane an output, blocks of 256 — and the same arguments: every call is this family's. */
static void nvidia_cuda12__override_gated_mean(uint16_t* into, uint16_t* inj, const uint16_t* gt, const uint16_t* an, const uint16_t* ij,
                                               uint64_t hidden, uint64_t streams, float mean, unsigned int* over) {
    const uint64_t want = (hidden + 255u) / 256u;
    const uint32_t blocks = want == 0u ? 1u : want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &into, &inj, &gt, &an, &ij, &hidden, &streams, &mean, &over };
    const size_t sizes[] = { sizeof into, sizeof inj, sizeof gt, sizeof an, sizeof ij, sizeof hidden, sizeof streams, sizeof mean, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gated_mean, "nvidia_cuda12__zzprivate_gated_mean", blocks, 256u,
                        args, sizes, 9u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_GATED_CUH */
