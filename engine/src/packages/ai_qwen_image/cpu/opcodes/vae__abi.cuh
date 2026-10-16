#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/objects/vae.cuh"      /* the VAE's tables */
#include "../table.cuh"                          /* reading them */
#include "../rows.cuh"                           /* rows through a matrix */

/* ⭐ THE VAE ON A WORKER — a picture a pixel a row, its convolutions as nn's `unfold_rows` and an ordinary product of the
 * rows (a kernel a row of its matrix, at 16 bits), and its middle's attention as products and a row softmax (▶
 * `contracts/objects/vae.cuh` for the tables and the forms). The arithmetic is diffusers' `AutoencoderKLQwenImage21` on
 * one frame, as the host's `python/silvann_runtime/qwen_image_vae.py` writes it out. */

/* `(ai_qwen_image__conv x y table H W)` -> `y`. */
static sys__heap_node ai_qwen_image__conv__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, y = 0, y_room = 0;
    if (!sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[1], &y, &y_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_IMAGE__VC__TABLE], room[AI_QWEN_IMAGE__VC__TABLE], v[AI_QWEN_IMAGE__VC__TABLE];
    if (!ai_qwen_image__table__zzpackage_read(argv[2].args[0], AI_QWEN_IMAGE__VC__C, AI_QWEN_IMAGE__VC__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    const uint64_t C = v[AI_QWEN_IMAGE__VC__C], O = v[AI_QWEN_IMAGE__VC__O], kh = v[AI_QWEN_IMAGE__VC__KH], kw = v[AI_QWEN_IMAGE__VC__KW];
    const uint64_t st = v[AI_QWEN_IMAGE__VC__STRIDE], pt = v[AI_QWEN_IMAGE__VC__PAD_TOP], pl = v[AI_QWEN_IMAGE__VC__PAD_LEFT];
    const uint64_t pb = v[AI_QWEN_IMAGE__VC__PAD_BOTTOM], pr = v[AI_QWEN_IMAGE__VC__PAD_RIGHT], up = v[AI_QWEN_IMAGE__VC__UP];
    const uint64_t norm = v[AI_QWEN_IMAGE__VC__NORM], f = v[AI_QWEN_IMAGE__VC__SHUFFLE], add = v[AI_QWEN_IMAGE__VC__ADD];
    const uint64_t chunk = v[AI_QWEN_IMAGE__VC__CHUNK], H = argv[3].args[0], W = argv[4].args[0];
    if (C == 0u || O == 0u || kh == 0u || kw == 0u || st == 0u || (up != 1u && up != 2u) || norm > 1u || add > 1u || f == 0u
     || O % (f * f) != 0u || chunk == 0u || chunk > (1ull << 20) || C > (1ull << 16) || O > (1ull << 16) || kh > 16u || kw > 16u
     || pt >= kh || pl >= kw || pb >= kh || pr >= kw || H == 0u || W == 0u || H > (1ull << 15) || W > (1ull << 15)
     || H * up + pt + pb < kh || W * up + pl + pr < kw)
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    const uint64_t K = kh * kw * C, Ho = (H * up + pt + pb - kh) / st + 1u, Wo = (W * up + pl + pr - kw) / st + 1u;
    const uint64_t px = Ho * Wo, out = px * O;           /* the convolution's halves, before any shuffle */
    if (2u * O * K > room[AI_QWEN_IMAGE__VC__WEIGHT] || 2u * O > room[AI_QWEN_IMAGE__VC__BIAS]
     || (norm != 0u && 2u * C > room[AI_QWEN_IMAGE__VC__GAMMA]))
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    /* scratch: the pairs, a chunk's unfolded rows and product, the normed input (NORM), the output before and after its
       shuffle (SHUFFLE) */
    const uint64_t S = at[AI_QWEN_IMAGE__VC__SCRATCH], pairs = S, U = S + ai_qwen_image__rows__zzpackage_pairs_bytes(chunk);
    const uint64_t T = U + 2u * chunk * K, N = T + 2u * chunk * O, S1 = N + (norm != 0u ? 2u * H * W * C : 0u);
    const uint64_t S2 = S1 + (f > 1u ? 2u * out : 0u), end = S2 + (f > 1u ? 2u * out : 0u);
    if (!nn__primitives__fits(H * W * C, x_room) || !nn__primitives__fits(out, y_room)
     || room[AI_QWEN_IMAGE__VC__SCRATCH] < end - S)
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_image__rows__zzpackage_pairs(ctx, pairs, chunk)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    uint64_t src = x;
    if (norm != 0u) {                                     /* the input normed and through SiLU, once, whole */
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)N, (const uint16_t*)(uintptr_t)x,
                            (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__VC__GAMMA], C, H * W, C, C, AI_QWEN_IMAGE__VAE__EPS, over);
        doors->vector_silu((uint16_t*)(uintptr_t)N, (const uint16_t*)(uintptr_t)N, H * W * C, over);
        src = N;
    }
    /* the output a chunk of pixels at a time: written straight into `y` where nothing follows, else into the chunk's own */
    const uint64_t dst = f > 1u ? S1 : y;
    const bool direct = f == 1u && add == 0u;
    for (uint64_t r0 = 0u; r0 < px; r0 += chunk) {
        const uint64_t n = px - r0 < chunk ? px - r0 : chunk, here = direct ? dst + 2u * r0 * O : T;
        doors->unfold_rows((uint16_t*)(uintptr_t)U, (const uint16_t*)(uintptr_t)src, H, W, C, kh, kw, st, pt, pl, up, Wo, r0, n);
        doors->expert_rows((uint16_t*)(uintptr_t)here, (const uint8_t*)(uintptr_t)at[AI_QWEN_IMAGE__VC__WEIGHT],
                           room[AI_QWEN_IMAGE__VC__WEIGHT], (const uint8_t*)(uintptr_t)at[AI_QWEN_IMAGE__VC__BIAS],
                           room[AI_QWEN_IMAGE__VC__BIAS], (const uint16_t*)(uintptr_t)U, (const uint32_t*)(uintptr_t)pairs, n,
                           16u, O, K, over);
        doors->vector_bias_rows((uint16_t*)(uintptr_t)here, (const uint16_t*)(uintptr_t)here,
                                (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__VC__BIAS], O, n, over);
        if (direct) continue;
        if (f == 1u)                                      /* added into y */
            doors->vector_add((uint16_t*)(uintptr_t)(y + 2u * r0 * O), (const uint16_t*)(uintptr_t)(y + 2u * r0 * O),
                              (const uint16_t*)(uintptr_t)T, n * O, over);
        else
            doors->vector_copy((uint16_t*)(uintptr_t)(dst + 2u * r0 * O), (const uint16_t*)(uintptr_t)T, n * O);
    }
    if (f > 1u) {                                         /* the channels to space, then written or added */
        doors->depth_to_space_rows((uint16_t*)(uintptr_t)(add != 0u ? S2 : y), (const uint16_t*)(uintptr_t)S1, Ho, Wo,
                                   O / (f * f), f);
        if (add != 0u)
            doors->vector_add((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)S2, out, over);
    }
    return nn__doors_answer(&argv[1]);
}

/* `(ai_qwen_image__vae_attention x table H W)` -> `x`. */
static sys__heap_node ai_qwen_image__vae_attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_IMAGE__VA__TABLE], room[AI_QWEN_IMAGE__VA__TABLE], v[AI_QWEN_IMAGE__VA__TABLE];
    if (!ai_qwen_image__table__zzpackage_read(argv[1].args[0], AI_QWEN_IMAGE__VA__C, AI_QWEN_IMAGE__VA__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    const uint64_t C = v[AI_QWEN_IMAGE__VA__C], N = argv[2].args[0] * argv[3].args[0];
    if (C == 0u || C > (1ull << 16) || N == 0u || N > (1ull << 16) || 2u * C > room[AI_QWEN_IMAGE__VA__GAMMA])
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    for (unsigned m = AI_QWEN_IMAGE__VA__Q; m <= AI_QWEN_IMAGE__VA__PROJ; m += 2u)
        if (2u * C * C > room[m] || 2u * C > room[m + 1u]) return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_TABLE);
    /* scratch: the pairs, then n · q · k · v · vᵀ · a (N x C each), the scores (N x N) */
    const uint64_t S = at[AI_QWEN_IMAGE__VA__SCRATCH], pairs = S, nb = S + ai_qwen_image__rows__zzpackage_pairs_bytes(N);
    const uint64_t q = nb + 2u * N * C, k = q + 2u * N * C, vv = k + 2u * N * C, vt = vv + 2u * N * C, a = vt + 2u * N * C;
    const uint64_t sc = a + 2u * N * C, end = sc + 2u * N * N;
    if (!nn__primitives__fits(N * C, x_room) || room[AI_QWEN_IMAGE__VA__SCRATCH] < end - S)
        return sys__engine__abi__error(AI_QWEN_IMAGE__VAE__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_image__rows__zzpackage_pairs(ctx, pairs, N)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)nb, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__VA__GAMMA],
                        C, N, C, C, AI_QWEN_IMAGE__VAE__EPS, over);
    const uint64_t outs[3] = {q, k, vv};
    for (unsigned i = 0u; i < 3u; ++i) {                  /* q, k and v: 1 x 1 convolutions, a matrix of halves each */
        const unsigned m = AI_QWEN_IMAGE__VA__Q + 2u * i;
        ai_qwen_image__rows__zzpackage_product(doors, outs[i], at, room, m, 16u, nb, pairs, N, C, C, over);
        doors->vector_bias_rows((uint16_t*)(uintptr_t)outs[i], (const uint16_t*)(uintptr_t)outs[i],
                                (const uint16_t*)(uintptr_t)at[m + 1u], C, N, over);
    }
    doors->vector_scale((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, N * C, 1.0f / nn__silicon__sqrtf((float)(uint32_t)C), over);
    /* the scores: k's rows as a matrix's, every query a row of N; each row's softmax; v's columns as a matrix's rows */
    doors->expert_rows((uint16_t*)(uintptr_t)sc, (const uint8_t*)(uintptr_t)k, 2u * N * C, (const uint8_t*)(uintptr_t)at[AI_QWEN_IMAGE__VA__K_BIAS],
                       room[AI_QWEN_IMAGE__VA__K_BIAS], (const uint16_t*)(uintptr_t)q, (const uint32_t*)(uintptr_t)pairs, N, 16u, N, C, over);
    for (uint64_t r = 0u; r < N; ++r)
        doors->softmax((uint16_t*)(uintptr_t)(sc + 2u * r * N), (const uint16_t*)(uintptr_t)(sc + 2u * r * N), N, over);
    doors->matrix_transpose((uint16_t*)(uintptr_t)vt, (const uint16_t*)(uintptr_t)vv, N, C);
    doors->expert_rows((uint16_t*)(uintptr_t)a, (const uint8_t*)(uintptr_t)vt, 2u * C * N, (const uint8_t*)(uintptr_t)at[AI_QWEN_IMAGE__VA__V_BIAS],
                       room[AI_QWEN_IMAGE__VA__V_BIAS], (const uint16_t*)(uintptr_t)sc, (const uint32_t*)(uintptr_t)pairs, N, 16u, C, N, over);
    /* the projection, then onto x */
    ai_qwen_image__rows__zzpackage_product(doors, q, at, room, AI_QWEN_IMAGE__VA__PROJ, 16u, a, pairs, N, C, C, over);
    doors->vector_bias_rows((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__VA__PROJ_BIAS],
                            C, N, over);
    doors->vector_add((uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)q, N * C, over);
    return nn__doors_answer(&argv[0]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__conv__zzabi_adapter, ai_qwen_image__conv__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__vae_attention__zzabi_adapter, ai_qwen_image__vae_attention__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI_CUH */
