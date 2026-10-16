#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_CONV_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_CONV_CUH
/* ══ conv's DOORS — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════════════════════
 * nn's convolution words (`nn/gpu/kernels/kernels.cuh`), where this family covers them: a convolution's rows. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

/* ⭐ A CONVOLUTION'S ROWS, EIGHT CHANNELS A LANE — nn's `unfold_rows`: output pixel r's kh x kw window, (dy, dx, c), with
 *   the stride, the padding and a nearest upsampling folded in. nn's body writes a half a lane and finds it with six 64-bit
 *   divisions, which a card does in software. Here a lane copies eight channels of one cell of a window — sixteen bytes
 *   from the picture, or sixteen zero bytes past its edge — and finds them with 32-bit arithmetic: the same bytes as nn's.
 *   Channels a multiple of 8, `x` and `o` on sixteen bytes, every index under 2^32; anything else is nn's. */
static __global__ void nvidia_cuda12__zzprivate_unfold(uint4* o, const uint4* x, uint32_t W, uint32_t C8, uint32_t kw, uint32_t cells,
                                                       uint32_t stride, uint32_t pad_top, uint32_t pad_left, uint32_t up, uint32_t Hu,
                                                       uint32_t Wu, uint32_t Wo, uint32_t r0, uint32_t total) {
    const uint32_t step = (uint32_t)nn__silicon__blocks() * (uint32_t)nn__silicon__lanes();
    for (uint32_t i = (uint32_t)nn__silicon__block() * (uint32_t)nn__silicon__lanes() + (uint32_t)nn__silicon__lane(); i < total; i += step) {
        const uint32_t v = i % C8, rest = i / C8, cell = rest % cells, r = rest / cells;
        const uint32_t dy = cell / kw, dx = cell - dy * kw, p = r0 + r, oy = p / Wo, ox = p - oy * Wo;
        const uint32_t iy = oy * stride + dy, ix = ox * stride + dx;                  /* in the padded picture */
        const bool inside = iy >= pad_top && ix >= pad_left && iy - pad_top < Hu && ix - pad_left < Wu;
        o[i] = inside ? x[(((iy - pad_top) / up) * W + (ix - pad_left) / up) * C8 + v] : make_uint4(0u, 0u, 0u, 0u);
    }
}
static void nvidia_cuda12__override_unfold_rows(uint16_t* o, const uint16_t* x, uint64_t H, uint64_t W, uint64_t C, uint64_t kh,
                                                uint64_t kw, uint64_t stride, uint64_t pad_top, uint64_t pad_left, uint64_t up,
                                                uint64_t Wo, uint64_t r0, uint64_t rows) {
    const uint64_t most = 0xFFFFFFFFull, K = kh * kw * C;
    if (rows == 0u || C == 0u || C % 8u != 0u || kh == 0u || kw == 0u || stride == 0u || up == 0u || Wo == 0u
     || ((uint64_t)(uintptr_t)x) % 16u != 0u || ((uint64_t)(uintptr_t)o) % 16u != 0u
     || rows * K / 8u > most || H * W * C / 8u > most || (r0 + rows) > most || H * up + kh > most || W * up + kw > most
     || ((r0 + rows) / Wo) * stride + kh > most || Wo * stride + kw > most) {
        nn__conv__zzabi_launch_unfold_rows(o, x, H, W, C, kh, kw, stride, pad_top, pad_left, up, Wo, r0, rows);
        return;
    }
    uint32_t w = (uint32_t)W, c8 = (uint32_t)(C / 8u), kw32 = (uint32_t)kw, cells = (uint32_t)(kh * kw), st = (uint32_t)stride;
    uint32_t pt = (uint32_t)pad_top, pl = (uint32_t)pad_left, up32 = (uint32_t)up, hu = (uint32_t)(H * up), wu = (uint32_t)(W * up);
    uint32_t wo = (uint32_t)Wo, first = (uint32_t)r0, total = (uint32_t)(rows * K / 8u);
    const uint64_t want = (total + 255u) / 256u;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    uint4* o4 = (uint4*)(void*)o;
    const uint4* x4 = (const uint4*)(const void*)x;
    void* args[] = { &o4, &x4, &w, &c8, &kw32, &cells, &st, &pt, &pl, &up32, &hu, &wu, &wo, &first, &total };
    const size_t sizes[] = { sizeof o4, sizeof x4, sizeof w, sizeof c8, sizeof kw32, sizeof cells, sizeof st, sizeof pt, sizeof pl,
                             sizeof up32, sizeof hu, sizeof wu, sizeof wo, sizeof first, sizeof total };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_unfold, "nvidia_cuda12__zzprivate_unfold", blocks, 256u, args, sizes, 15u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_CONV_CUH */
