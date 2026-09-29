#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI_CUH
/* ══ THE DeltaNet STATE VERBS, ON THE PROGRAM'S SIDE ═════════════════════════════════════════════════
 * The checks are their bodies' in `deltanet__impl.cuh`, sharing its two shape helpers so the bounds
 * cannot differ between the two roads; the arithmetic is the door's. ▶ `vector__abi.cuh`. */

#include "../doors.cuh"

/* ── ① + ④ — `S[i][j] <- g*S[i][j] + k[i]*d[j]` ─────────────────────────────────────────────────── */
/* `(nn__deltanet__rank_1_update S k d g k_dim v_dim)` — `S <- e^g * S + k dᵀ`, the state in fp32. The
 * decay is taken to its exponential here, on the host, so the door receives the factor itself. Nothing
 * in it rounds to a half, so it has no fault word. */
static sys__heap_node nn__deltanet__zzabi_apply_rank_1_update(const sys__heap_node* argv, unsigned argc,
                                                       sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t s_at = 0, k_at = 0, d_at = 0, s_room = 0, k_room = 0, d_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_FLOAT
     || !nn__primitives__room(&argv[0], &s_at, &s_room)
     || !nn__primitives__room(&argv[1], &k_at, &k_room)
     || !nn__primitives__room(&argv[2], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t k_dim = 0, v_dim = 0, cells = 0;
    if (!nn__deltanet__zzpackage_shape(argv[4], argv[5], &k_dim, &v_dim, &cells)
     || !nn__deltanet__zzpackage_fits32(cells, s_room)
     || !nn__primitives__fits(k_dim, k_room)
     || !nn__primitives__fits(v_dim, d_room)) return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const float g = nn__silicon__expf((float)sys__heap_node__real(argv[3].args[0]));
    doors->deltanet_rank_1_update((float*)(uintptr_t)s_at, (const uint16_t*)(uintptr_t)k_at, (const uint16_t*)(uintptr_t)d_at, g,
                                  k_dim, v_dim);
    return nn__doors_answer(&argv[0]);
}

/* ── ② / ⑤ — `out[j] = Σ_i S[i][j] * x[i]` ──────────────────────────────────────────────────────── */
/* `(nn__deltanet__readout S x out k_dim v_dim)` — `out = Sᵀx`; x runs along the key axis. */
static sys__heap_node nn__deltanet__zzabi_apply_readout(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t s_at = 0, x_at = 0, o_at = 0, s_room = 0, x_room = 0, o_room = 0;
    if (!nn__primitives__room(&argv[0], &s_at, &s_room)
     || !nn__primitives__room(&argv[1], &x_at, &x_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t k_dim = 0, v_dim = 0, cells = 0;
    if (!nn__deltanet__zzpackage_shape(argv[3], argv[4], &k_dim, &v_dim, &cells)
     || !nn__deltanet__zzpackage_fits32(cells, s_room)
     || !nn__primitives__fits(k_dim, x_room)
     || !nn__primitives__fits(v_dim, o_room)) return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->deltanet_readout((uint16_t*)(uintptr_t)o_at, (const float*)(uintptr_t)s_at, (const uint16_t*)(uintptr_t)x_at, k_dim, v_dim,
                            ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ── THE CAUSAL CONV — one token, every channel, and the window rolled forward ─────────────────────── */
/* `(nn__deltanet__conv_step w s x out ch)` — one step of the causal convolution: the window `s` shifts in
 * `x`, and `out` is the tap-weighted sum. */
static sys__heap_node nn__deltanet__zzabi_apply_conv_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, s_at = 0, x_at = 0, o_at = 0, w_room = 0, s_room = 0, x_room = 0, o_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &s_at, &s_room)
     || !nn__primitives__room(&argv[2], &x_at, &x_room)
     || !nn__primitives__room(&argv[3], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t ch = argv[4].args[0];
    if (ch == 0ull || ch > 0xFFFFFFFFFFFFFFFFull / NN__DELTANET__CONV_TAPS)
        return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);
    if (!nn__primitives__fits(ch * NN__DELTANET__CONV_TAPS, w_room)
     || !nn__primitives__fits(ch * NN__DELTANET__CONV_WINDOW, s_room)
     || !nn__primitives__fits(ch, x_room)
     || !nn__primitives__fits(ch, o_room)) return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->deltanet_conv_step((uint16_t*)(uintptr_t)o_at, (uint16_t*)(uintptr_t)s_at, (const uint16_t*)(uintptr_t)w_at,
                              (const uint16_t*)(uintptr_t)x_at, ch, ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* ── ⭐⭐ ONE STEP, EVERY HEAD — the head loop in one launch ────────────────────────────────────────── */
/* `(nn__deltanet__step S conved z beta g w out k_heads v_heads head_dim)` — every value head's recurrence at
 * one token: `S` (fp32, `v_heads` of `head_dim²`) read, decayed by `e^g` and updated in place, and each head's
 * gated, normed readout into `out`. `conved` is the conv's output, `q`s then `k`s (`k_heads` each) then `v`s;
 * `z`, `beta` and `g` are the gate, the write strength and the log decay, one a head for the last two; `w` is
 * the gated norm's weight. ▶ `kernels.cuh` for the arithmetic. Refused: key heads that do not divide the
 * value heads. */
static sys__heap_node nn__deltanet__zzabi_apply_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 10u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[7] = {0, 0, 0, 0, 0, 0, 0}, room[7] = {0, 0, 0, 0, 0, 0, 0};
    for (unsigned a = 0u; a < 7u; ++a)
        if (!nn__primitives__room(&argv[a], &at[a], &room[a])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[7].dtype != SYS__KIND__VALUE_INT || argv[8].dtype != SYS__KIND__VALUE_INT
     || argv[9].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t k_heads = argv[7].args[0], v_heads = argv[8].args[0], d = argv[9].args[0];
    if (k_heads == 0ull || v_heads == 0ull || d == 0ull || v_heads % k_heads != 0ull
     || v_heads > 0xFFFFFFFFull || k_heads > v_heads) return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);
    uint64_t k_dim = 0, v_dim = 0, cells = 0;
    if (!nn__deltanet__zzpackage_shape(argv[9], argv[9], &k_dim, &v_dim, &cells)
     || !nn__deltanet__zzpackage_fits32(cells * v_heads, room[0])
     || !nn__primitives__fits((2ull * k_heads + v_heads) * d, room[1])
     || !nn__primitives__fits(v_heads * d, room[2])
     || !nn__primitives__fits(v_heads, room[3]) || !nn__primitives__fits(v_heads, room[4])
     || !nn__primitives__fits(d, room[5])
     || !nn__primitives__fits(v_heads * d, room[6])) return sys__engine__abi__error(NN__DELTANET__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->deltanet_step((float*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1], (const uint16_t*)(uintptr_t)at[2],
                         (const uint16_t*)(uintptr_t)at[3], (const uint16_t*)(uintptr_t)at[4], (const uint16_t*)(uintptr_t)at[5],
                         (uint16_t*)(uintptr_t)at[6], k_heads, v_heads, d, ctx->fault_word);
    return nn__doors_answer(&argv[6]);
}

SYS__ENGINE__ABI__BRIDGE(nn__deltanet__zzabi_adapter_rank_1_update, nn__deltanet__zzabi_apply_rank_1_update)
SYS__ENGINE__ABI__BRIDGE(nn__deltanet__zzabi_adapter_step,          nn__deltanet__zzabi_apply_step)
SYS__ENGINE__ABI__BRIDGE(nn__deltanet__zzabi_adapter_readout,       nn__deltanet__zzabi_apply_readout)
SYS__ENGINE__ABI__BRIDGE(nn__deltanet__zzabi_adapter_conv_step,     nn__deltanet__zzabi_apply_conv_step)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI_CUH */
