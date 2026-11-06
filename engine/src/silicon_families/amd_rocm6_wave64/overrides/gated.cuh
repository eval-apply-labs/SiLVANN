#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_GATED_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_GATED_CUH
/* ══ nn's gated residual's DOORS, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════
 * nn's `gated_rank` and `gated_mean` (`nn/gpu/kernels/kernels.cuh`), Flash-Next's gated residual between and after its
 * two maps: the rank's activation and its rotation in one block, and the streams' gated mean a lane an output, as nn's
 * body — each value either rounds to a half through the family's `to_half` (nn's conversion's bits and fault word,
 * ▶ kernels.cuh) where nn's bodies take its branches. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */
#include "hadamard.cuh"     /* a wave's block of the rotation, the one `hadamard_blocks` runs */

/* ⭐ THE RANK IN ONE BLOCK — `gated_rank` where the rank is whole blocks of 128 and at most AMD_ROCM6_WAVE64__ZZPRIVATE_GATED_RANK:
 *   every lane's activations staged — nn's `gated_act`, `vector_scale` and `swiglu_combine`'s arithmetic, written out as
 *   `act` too — waited for once, then the ladder's blocks a wave each through `hadamard_at`, the wave the family's
 *   `hadamard_blocks` gives a block, so each block is that door's to the bit. Before it a site ran `vector_scale`,
 *   `swiglu_combine` and two `hadamard_blocks`, four launches: `MEASURED` (rocprof over a decode,
 *   measurements/2026-11-01_stream_norms_node03.md) ~3 µs each but the 256-block's 6, 97 sites a token. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_GATED_RANK 2048u
/* nn's `gated_act` — the value scaled, rounded, clamped, its silu rounded — through the family's conversion. */
static __device__ inline uint16_t amd_rocm6_wave64__zzprivate_gated_act(const uint16_t* a, uint64_t i, float scale, unsigned int* over) {
    const float s = nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(nn__silicon__half_to_float(a[i]) * scale, over));
    const float clamped = (s < -88.0f) ? -88.0f : ((s > 88.0f) ? 88.0f : s);
    return amd_rocm6_wave64__zzpackage_to_half(clamped / (1.0f + nn__silicon__expf(-clamped)) * 1.0f, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_gated_rank(uint16_t* rot, uint16_t* act, const uint16_t* a,
                                                                                                const uint16_t* sign, uint64_t rank, float scale,
                                                                                                unsigned int* over) {
    __shared__ uint16_t staged[AMD_ROCM6_WAVE64__ZZPRIVATE_GATED_RANK];
    const uint32_t threads = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    for (uint64_t i = nn__silicon__lane(); i < rank; i += threads) {
        const uint16_t v = amd_rocm6_wave64__zzprivate_gated_act(a, i, scale, over);
        staged[i] = v;
        act[i] = v;
    }
    __syncthreads();
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t whole = rank / 512u * 512u, rest = rank - whole;
    const uint64_t blocks = whole / 512u + (rest >= 256u ? 1u : 0u) + (rest % 256u != 0u ? 1u : 0u);
    for (uint64_t k = wave; k < blocks; k += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        if (k < whole / 512u) {
            amd_rocm6_wave64__zzpackage_hadamard_at<8u>(rot, staged, k * 512u, sign, 0u, over);
        } else if (rest >= 256u && k == whole / 512u) {
            amd_rocm6_wave64__zzpackage_hadamard_at<4u>(rot, staged, whole, sign, 0u, over);
        } else {
            amd_rocm6_wave64__zzpackage_hadamard_at<2u>(rot, staged, whole + (rest >= 256u ? 256u : 0u), sign, 0u, over);
        }
    }
}
/* Whole blocks of 128 up to the staged room, and the three vectors apart; anything else is nn's. */
static void amd_rocm6_wave64__override_gated_rank(uint16_t* rot, uint16_t* act, const uint16_t* a, const uint16_t* sign,
                                                  uint64_t rank, float scale, unsigned int* over) {
    const bool apart = (rot + rank <= a || a + rank <= rot) && (act + rank <= a || a + rank <= act)
                    && (rot + rank <= act || act + rank <= rot);
    if (rank == 0u || rank % 128u != 0u || rank > AMD_ROCM6_WAVE64__ZZPRIVATE_GATED_RANK || !apart) {
        nn__gated__zzabi_launch_rank(rot, act, a, sign, rank, scale, over);
        return;
    }
    void* args[] = { &rot, &act, &a, &sign, &rank, &scale, &over };
    const size_t sizes[] = { sizeof rot, sizeof act, sizeof a, sizeof sign, sizeof rank, sizeof scale, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_gated_rank, "amd_rocm6_wave64__zzprivate_gated_rank", 1u,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}

/* ⭐ THE GATED MEAN — nn's `gated_mean`: a lane an output, its streams' σ(gate) times the normed stream summed in their
 *   order, each step rounded as nn's body rounds it, the mean, and the first `streams` lanes the injection's weights
 *   2·σ(mean·ij) — nn's arithmetic in its order, fourteen conversions a lane at four streams through `to_half`. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_gated_mean(uint16_t* into, uint16_t* inj, const uint16_t* gt,
                                                                                                const uint16_t* an, const uint16_t* ij,
                                                                                                uint64_t hidden, uint64_t streams, float mean,
                                                                                                unsigned int* over) {
    const uint64_t stride = (uint64_t)nn__silicon__blocks() * nn__silicon__lanes();
    for (uint64_t j = (uint64_t)nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); j < hidden; j += stride) {
        float m = 0.0f;
        for (uint64_t k = 0ull; k < streams; ++k) {
            const uint64_t i = k * hidden + j;
            const float gs = nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(
                                 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(gt[i]))), over));
            const float gn = nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(gs * nn__silicon__half_to_float(an[i]), over));
            m = k == 0ull ? gn : nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(m + gn, over));
        }
        into[j] = amd_rocm6_wave64__zzpackage_to_half(m * mean, over);
        if (inj != 0 && ij != 0 && j < streams) {
            const float s = nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(nn__silicon__half_to_float(ij[j]) * mean, over));
            const float g = nn__silicon__half_to_float(amd_rocm6_wave64__zzpackage_to_half(1.0f / (1.0f + nn__silicon__expf(-s)), over));
            inj[j] = amd_rocm6_wave64__zzpackage_to_half(g * 2.0f, over);
        }
    }
}
/* nn's launch's lanes — a lane an output, blocks of 256 — and the same arguments: every call is this family's. */
static void amd_rocm6_wave64__override_gated_mean(uint16_t* into, uint16_t* inj, const uint16_t* gt, const uint16_t* an, const uint16_t* ij,
                                                  uint64_t hidden, uint64_t streams, float mean, unsigned int* over) {
    const uint32_t threads = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t want = (hidden + threads - 1u) / threads;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &into, &inj, &gt, &an, &ij, &hidden, &streams, &mean, &over };
    const size_t sizes[] = { sizeof into, sizeof inj, sizeof gt, sizeof an, sizeof ij, sizeof hidden, sizeof streams, sizeof mean, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_gated_mean, "amd_rocm6_wave64__zzprivate_gated_mean", blocks, threads,
                        args, sizes, 9u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_GATED_CUH */
