#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI_CUH
/* ══ THE HYPER-CONNECTION, ON THE PROGRAM'S SIDE ═════════════════════════════════════════════════════
 * A residual of `mult` streams, collapsed into a sublayer's input and mixed again after it — GLM 5.3 Flash's and DeepSeek
 * V4's. The arithmetic is the door's (`gpu/kernels/kernels.cuh`); here the operands are checked. */

#include "../doors.cuh"

/* `(nn__hyper__pre streams fn base scale out mix hidden mult iters norm_eps eps)` — the sublayer's input into `out`
 * (hidden halves), and `pre`, `post` and `comb` into `mix` for `nn__hyper__post` — twice (2 + mult)·mult floats, the map's logits
 * passing through its second half between the door's two launches. `fn` is the map,
 * (2 + mult)·mult rows of mult·hidden halves; `base` its (2 + mult)·mult biases; `scale` its three scales. */
static sys__heap_node nn__hyper__zzabi_apply_pre(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 11u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[6] = {0}, room[6] = {0};
    for (unsigned i = 0u; i < 6u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[6].dtype != SYS__KIND__VALUE_INT || argv[7].dtype != SYS__KIND__VALUE_INT || argv[8].dtype != SYS__KIND__VALUE_INT
     || argv[9].dtype != SYS__KIND__VALUE_FLOAT || argv[10].dtype != SYS__KIND__VALUE_FLOAT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t hidden = argv[6].args[0], mult = argv[7].args[0], iters = argv[8].args[0];
    if (hidden == 0ull || hidden > (1ull << 24) || mult == 0ull || mult > NN__HYPER__MULT_MAX || iters == 0ull || iters > 256ull)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t mix = NN__HYPER__MIX(mult), n = mult * hidden;
    if (!nn__primitives__fits(n, room[0]) || !nn__primitives__fits(mix * n, room[1]) || !nn__primitives__fits(mix, room[2])
     || !nn__primitives__fits(3ull, room[3]) || !nn__primitives__fits(hidden, room[4]) || 2ull * mix * 4ull > room[5])
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->hyper_logits((float*)(uintptr_t)at[5] + mix, (const uint16_t*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1], hidden, mult,
                        (float)sys__heap_node__real(argv[9].args[0]));
    doors->hyper_pre((uint16_t*)(uintptr_t)at[4], (float*)(uintptr_t)at[5], (const uint16_t*)(uintptr_t)at[0],
                     (const uint16_t*)(uintptr_t)at[2], (const uint16_t*)(uintptr_t)at[3], hidden, mult, iters,
                     (float)sys__heap_node__real(argv[10].args[0]), ctx->fault_word);
    return nn__doors_answer(&argv[4]);
}

/* `(nn__hyper__post streams y mix out hidden mult)` — stream j becomes `post_j · y + Σ_i comb_ij · stream_i`, into `out`,
 * which may be `streams`. */
static sys__heap_node nn__hyper__zzabi_apply_post(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[4] = {0}, room[4] = {0};
    for (unsigned i = 0u; i < 4u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t hidden = argv[4].args[0], mult = argv[5].args[0];
    if (hidden == 0ull || hidden > (1ull << 24) || mult == 0ull || mult > NN__HYPER__MULT_MAX) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__primitives__fits(mult * hidden, room[0]) || !nn__primitives__fits(hidden, room[1])
     || NN__HYPER__MIX(mult) * 4ull > room[2] || !nn__primitives__fits(mult * hidden, room[3]))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->hyper_post((uint16_t*)(uintptr_t)at[3], (const uint16_t*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1],
                      (const float*)(uintptr_t)at[2], hidden, mult, ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* `(nn__kda__step state conved fb dt_bias a_log gate w out heads head_dim lower eps)` — one step of Kimi Delta Attention,
 * every head: `conved` is q · k · v after their conv, `fb` the forget gate's projection (heads·head_dim) then the β logits
 * (heads) — the two gemvs' outputs side by side, which keeps the verb inside the bridge's twelve arguments — `dt_bias` and
 * `a_log` the gate's parameters, `gate` the output gate's projection, `w` the gated norm's weight (head_dim). The state
 * (heads · head_dim² floats) is read and written in place. ▶ the door in `gpu/kernels/kernels.cuh`. */
static sys__heap_node nn__kda__zzabi_apply_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 12u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[8] = {0}, room[8] = {0};
    for (unsigned i = 0u; i < 8u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[8].dtype != SYS__KIND__VALUE_INT || argv[9].dtype != SYS__KIND__VALUE_INT
     || argv[10].dtype != SYS__KIND__VALUE_FLOAT || argv[11].dtype != SYS__KIND__VALUE_FLOAT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t heads = argv[8].args[0], d = argv[9].args[0];
    if (heads == 0ull || heads > (1ull << 16) || d == 0ull || d > 1024ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = heads * d;
    if (heads * d * d * 4ull > room[0] || !nn__primitives__fits(3ull * n, room[1]) || !nn__primitives__fits(n + heads, room[2])
     || !nn__primitives__fits(n, room[3]) || !nn__primitives__fits(heads, room[4]) || !nn__primitives__fits(n, room[5])
     || !nn__primitives__fits(d, room[6]) || !nn__primitives__fits(n, room[7]))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->kda_step((float*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1], (const uint16_t*)(uintptr_t)at[2],
                    (const uint16_t*)(uintptr_t)(at[2] + 2ull * n), (const uint16_t*)(uintptr_t)at[3], (const uint16_t*)(uintptr_t)at[4],
                    (const uint16_t*)(uintptr_t)at[5], (const uint16_t*)(uintptr_t)at[6], (uint16_t*)(uintptr_t)at[7], heads, d,
                    (float)sys__heap_node__real(argv[10].args[0]), (float)sys__heap_node__real(argv[11].args[0]), ctx->fault_word);
    return nn__doors_answer(&argv[7]);
}

SYS__ENGINE__ABI__BRIDGE(nn__kda__zzabi_adapter_step,   nn__kda__zzabi_apply_step)
SYS__ENGINE__ABI__BRIDGE(nn__hyper__zzabi_adapter_pre,  nn__hyper__zzabi_apply_pre)
SYS__ENGINE__ABI__BRIDGE(nn__hyper__zzabi_adapter_post, nn__hyper__zzabi_apply_post)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI_CUH */
