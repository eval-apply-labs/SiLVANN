#ifndef SILVANN__PACKAGES_NN_GPU_DOORS_CUH
#define SILVANN__PACKAGES_NN_GPU_DOORS_CUH
/* ══ ⭐⭐⭐ nn's DOORS — WHERE A VERB STOPS BEING CODE AND BECOMES A LAUNCH ════════════════════════════
 *
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Where these doors came from, and what the move to a host evaluator costs.
 *   · The bodies behind these doors are the serial loops that ran inlined into `k_eval`.
 *   · The first generator wrote all 24 wrappers identically and was wrong on the five that return a
 *     value; they were found by extracting each body's RETURN TYPE rather than trusting the shape.
 *   · The host-TU vendor-freedom rule below was recorded in commit `8f9b0abb`.
 *   · The 1.654 µs / 28.816 µs figures: `measurements/2026-09-23_S107_how_an_overflow_gets_home.md`.
 *   · `vector__at` was left as a kernel so the wrapping commit stayed mechanical; it is flagged below
 *     rather than special-cased.
 *   · the 24 kernels and 24 doors were written out by hand (the generator that first
 *     wrote them is not in the tree) and the doors took `void*`; both became the list below in one
 *     rearrangement, with every kernel's instructions compared before and after.
 * ⛳ RETIREMENT: when the host evaluator is the only evaluator and these doors are its only callers.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⚖ *"do the rearrangement, sys and the evaluator go host."*
 *
 * `kernels/kernels.cuh` holds the compute bodies with the language taken out of them. This file is the only
 * thing that knows they run on a card: for every row of nn's door list (`contracts/abi/gpu.cuh`) it makes
 * one `__global__` kernel that runs the row's body, and one `extern "C"` door that launches it.
 *
 * ⭐⭐ WRITTEN ONCE, FOR EVERY SILICON FAMILY. ⚖ *"i wanted more like the wrapper over hiplaunchkernelGGL and
 * let the general code be in nn, just using the wrapped silicon specific translation path."* A door starts
 * its kernel through `nn__silicon__launch` — the kernel's address, the blocks and threads, and a pointer to
 * each argument — which each family answers with its runtime's own launch-by-pointer (`hipLaunchKernel`,
 * `cudaLaunchKernel`). `__global__` is spelled the same in both, and a body asks which lane and block it
 * is through `nn__silicon__lane` / `block`, so nothing here names a vendor, and a family's folder holds
 * only what differs.
 *
 * ⭐ THE SIGNATURE IS THE CONTRACT ROW'S, AND ONLY THE CONTRACT ROW'S. The kernel and the door both take
 * the row's parameters, so a launch by pointer — which copies each argument at the kernel's own width —
 * cannot be handed a wrong one: the compiler checks the program's call against the same list.
 *
 * ── WHAT EACH DOOR RUNS, AND HOW WIDE ─────────────────────────────────────────────────────────────────
 * One line per door, named after it. A door in the contract with no line here does not compile.
 *
 *   the form     WIDE       the body divides its work by lane and block and every lane runs it; a body that
 *                           answers (the two TurboQuant refusals, decided by the verb on the CPU before
 *                           launching) is discarded
 *                PLAIN      the body runs on one lane of one block
 *                AS_INT     the body answers a value, on every lane it is started with, and the first lane
 *                AS_FLOAT   writes it into `out` as the call's `stamp` in the top half and the value's 32
 *                           bits in the bottom — one store, so the program never sees half an answer.
 *                           ▶ `nn/contracts/objects/result.cuh`.
 *   the body     in `kernels/kernels.cuh`
 *   the width    blocks, then threads per block — what `nn__silicon__launch` starts. Threads per block
 *                may not pass `NN__SILICON__LANES_MAX` (256), and the compiler refuses a door that does.
 *                ⛳ THE BLOCKS MAY BE AN EXPRESSION OF THE DOOR'S OWN ARGUMENTS — the columns are pasted
 *                into the door's body, where they are in scope — so a matvec starts one block per row
 *   the names    the row's parameter names, in order: the door's arguments, and for AS_INT / AS_FLOAT
 *                `out` and `stamp` first, which the body does not take
 *
 * ⛔ EVERY SERIAL BODY'S WIDTH IS `1u, 1u` AND THAT IS DELIBERATE. A serial body is a loop, and every lane
 * but one would either leave at once or repeat the same work, so a wider launch changes nothing but the cost. Widening is a
 * change to the BODY — each lane taking its own share of the loop, and a combine step for the reductions
 * — and a body is widened on its own, against the host. ⇒ ★ MOVE IT, THEN MAKE IT PARALLEL — never both
 * in one step.
 * ⭐ WHERE `out` LIVES DECIDES WHAT A VALUE COSTS. `MEASURED` on gfx906
 * (`measurements/2026-09-25_S108_readback_split.md`): launch and read the value back with a copy, 27 µs;
 * launch with `out` in MAPPED PINNED RAM and let the CPU watch it land, 4.7 µs. A bare 8-byte copy with
 * the card idle is already 7.8 — so `vector__at` stays a kernel: it is cheaper than a read-back AND
 * ordered behind whatever wrote the buffer.
 */

#include "kernels/kernels.cuh"

/* One block per item of `n`, and one even when there are none: a launch of no blocks is refused. */
#define NN__GPU__ZZPRIVATE_ONE_PER(n)  ((n) == 0ull ? 1u : (uint32_t)(n))
/* One block per item of `n`, up to `cap`: past it, each block walks several items. */
/* A pointwise door's blocks: enough for one item a lane, up to the most a wide door starts. */
#define NN__GPU__ZZPRIVATE_EACH(n)  NN__GPU__ZZPRIVATE_UP_TO(((n) + (uint64_t)NN__SILICON__LANES_MAX - 1ull) / (uint64_t)NN__SILICON__LANES_MAX, NN__KERNELS__BLOCKS_MAX)
#define NN__GPU__ZZPRIVATE_UP_TO(n, cap)  ((n) == 0ull ? 1u : (n) < (uint64_t)(cap) ? (uint32_t)(n) : (uint32_t)(cap))

/*      door                                              form                         body                                      width   names */
#define nn__vector__zzabi_launch_zero__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_zero,              NN__GPU__ZZPRIVATE_EACH(n), 256u, (v, n)
#define nn__vector__zzabi_launch_exp__RUNS                NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_exp,               NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, x, n, over)
#define nn__vector__zzabi_launch_softplus__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_softplus,          NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, x, n, over)
#define nn__vector__zzabi_launch_scale__RUNS              NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_scale,             NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, x, n, factor, over)
#define nn__vector__zzabi_launch_add__RUNS                NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_add,               NN__GPU__ZZPRIVATE_EACH(n), 256u, (z, x, y, n, over)
#define nn__vector__zzabi_launch_pointwise_mul__RUNS      NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_pointwise_mul,     NN__GPU__ZZPRIVATE_EACH(n), 256u, (z, x, y, n, over)
#define nn__rmsnorm__zzabi_launch__RUNS                   NN__GPU__ZZPRIVATE_WIDE,     nn__rmsnorm__zzabi_body,                  1u, 256u, (o, x, w, n, over)
#define nn__vector__zzabi_launch_l2norm__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_l2norm,            1u, 256u, (o, x, n, over)
#define nn__softmax__zzabi_launch__RUNS                   NN__GPU__ZZPRIVATE_WIDE,     nn__softmax__zzabi_body,                  1u, 256u, (o, x, n, over)
#define nn__sigmoid__zzabi_launch__RUNS                   NN__GPU__ZZPRIVATE_WIDE,     nn__sigmoid__zzabi_body,                  NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, x, n, over)
#define nn__vector__zzabi_launch_scale_at__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_scale_at,          NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, x, a, which, n, over)
#define nn__matrix__zzabi_launch_transpose__RUNS          NN__GPU__ZZPRIVATE_PLAIN,    nn__matrix__zzabi_body_transpose,         1u, 1u, (d, s, rows, cols)
#define nn__matrix__zzabi_launch_matvec_transposed__RUNS  NN__GPU__ZZPRIVATE_WIDE,     nn__matrix__zzabi_body_matvec_transposed, NN__GPU__ZZPRIVATE_EACH(cols), 256u, (out, W, x, rows, cols, over)
#define nn__argmax__zzabi_launch__RUNS                    NN__GPU__ZZPRIVATE_AS_INT,   nn__argmax__zzabi_body,                   1u, 256u, (out, stamp, x, n)
#define nn__vector__zzabi_launch_dot_product__RUNS        NN__GPU__ZZPRIVATE_AS_FLOAT, nn__vector__zzabi_body_dot_product,       1u, 256u, (out, stamp, a, b, n)
#define nn__vector__zzabi_launch_at__RUNS                 NN__GPU__ZZPRIVATE_AS_FLOAT, nn__vector__zzabi_body_at,                1u, 1u, (out, stamp, x, i)
#define nn__swiglu__zzabi_launch_combine__RUNS            NN__GPU__ZZPRIVATE_WIDE,     nn__swiglu__zzabi_body_combine,           NN__GPU__ZZPRIVATE_EACH(n), 256u, (o, g, u, n, over)
#define nn__deltanet__zzabi_launch_rank_1_update__RUNS    NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_rank_1_update,   NN__GPU__ZZPRIVATE_EACH(k_dim * v_dim), 256u, (S, k, d, g, k_dim, v_dim)
#define nn__deltanet__zzabi_launch_readout__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_readout,         NN__GPU__ZZPRIVATE_EACH(v_dim), 256u, (o, S, x, k_dim, v_dim, over)
#define nn__deltanet__zzabi_launch_conv_step__RUNS        NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_conv_step,       NN__GPU__ZZPRIVATE_EACH(ch), 256u, (o, s, w, x, ch, over)
#define nn__expert__zzabi_launch_multiply_fp16__RUNS      NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_multiply_fp16,     NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, W, x, rows, cols, over)
#define nn__turboquant__zzabi_launch_decode__RUNS         NN__GPU__ZZPRIVATE_WIDE,     nn__turboquant__zzabi_body_decode,        NN__GPU__ZZPRIVATE_EACH(rows * cols), 256u, (out, weights, w_room, luts, l_room, d, rows, cols, over)
#define nn__turboquant__zzabi_launch_gemv__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__turboquant__zzabi_body_gemv,          NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, weights, w_room, luts, l_room, x, d, rows, cols, over)
#define nn__turboquant__zzabi_launch_gemv_int8__RUNS      NN__GPU__ZZPRIVATE_WIDE,     nn__turboquant__zzabi_body_gemv_int8,     NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, weights, w_room, luts, l_room, x, d, rows, cols, over)
#define nn__hadamard__zzabi_launch_rotate__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__hadamard__zzabi_body_rotate,          NN__GPU__ZZPRIVATE_EACH(n), 256u, (out, in, sign, n, over)
#define nn__rope__zzabi_launch__RUNS                      NN__GPU__ZZPRIVATE_WIDE,     nn__rope__zzabi_body,                     NN__GPU__ZZPRIVATE_EACH(heads * (head_dim / 2ull)), 256u, (o, x, cs, heads, head_dim, over)
#define nn__attention__zzabi_launch_scores__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_scores,         NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (p, q, k, q_heads, kv_heads, head_dim, length)
#define nn__attention__zzabi_launch_mix__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_mix,            NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (o, p, v, q_heads, kv_heads, head_dim, length, over)
#define nn__rope__zzabi_launch_angles__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__rope__zzabi_body_angles,              NN__GPU__ZZPRIVATE_EACH(head_dim / 2ull), 256u, (cs, position, theta, head_dim)
#define nn__attention__zzabi_launch_weights__RUNS         NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_weights,        NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (p, res, q, k, q_heads, kv_heads, head_dim, length)
#define nn__attention__zzabi_launch_residual_mix__RUNS    NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_residual_mix,   NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (res, p, v, q_heads, kv_heads, head_dim, length)
#define nn__attention__zzabi_launch_merge__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_merge,          NN__GPU__ZZPRIVATE_EACH(q_heads), 256u, (into, a, b, q_heads, head_dim)
#define nn__attention__zzabi_launch_finish__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_finish,         NN__GPU__ZZPRIVATE_EACH(q_heads * head_dim), 256u, (o, res, q_heads, head_dim, over)
#define nn__vector__zzabi_launch_top_k__RUNS              NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_top_k,             1u, 256u, (values, stride, weights, x, n, k)
#define nn__turboquant__zzabi_launch_gemv_groups__RUNS    NN__GPU__ZZPRIVATE_WIDE,     nn__turboquant__zzabi_body_gemv_groups,   NN__KERNELS__BLOCKS_MAX, 256u, (out, g, x, cols, over)
#define nn__turboquant__zzabi_launch_gemv_groups_sum__RUNS NN__GPU__ZZPRIVATE_WIDE,   nn__turboquant__zzabi_body_gemv_groups_sum, NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, g, x, w, residual, rows, cols, over)
#define nn__turboquant__zzabi_launch_gemv_groups_int8__RUNS NN__GPU__ZZPRIVATE_WIDE,  nn__turboquant__zzabi_body_gemv_groups_int8, NN__KERNELS__BLOCKS_MAX, 256u, (out, g, x, cols, over)
#define nn__turboquant__zzabi_launch_gemv_groups_sum_int8__RUNS NN__GPU__ZZPRIVATE_WIDE, nn__turboquant__zzabi_body_gemv_groups_sum_int8, NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, g, x, w, residual, rows, cols, over)
#define nn__swiglu__zzabi_launch_pairs__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__swiglu__zzabi_body_pairs,             NN__GPU__ZZPRIVATE_EACH(pairs * n), 256u, (o, gu, pairs, n, over)
#define nn__deltanet__zzabi_launch_step__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_step,            NN__GPU__ZZPRIVATE_UP_TO(v_heads, NN__KERNELS__BLOCKS_MAX), 256u, (S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over)
#define nn__expert__zzabi_launch_rows__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_rows,              NN__GPU__ZZPRIVATE_UP_TO(out_rows, NN__KERNELS__BLOCKS_MAX), 256u, (out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over)
#define nn__expert__zzabi_launch_rows_sum__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_rows_sum,          NN__GPU__ZZPRIVATE_UP_TO(out_rows, NN__KERNELS__BLOCKS_MAX), 256u, (acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols)
#define nn__expert__zzabi_launch_groups__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_groups,            NN__KERNELS__BLOCKS_MAX, 256u, (out, g, x, rows, cols, over)
#define nn__expert__zzabi_launch_rows_finish__RUNS        NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_rows_finish,       NN__GPU__ZZPRIVATE_EACH(n), 256u, (out, acc, residual, n, over)
#define nn__rmsnorm__zzabi_launch_rows__RUNS              NN__GPU__ZZPRIVATE_WIDE,     nn__rmsnorm__zzabi_body_rows,             NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (o, x, w, n, rows, x_stride, o_stride, over)
#define nn__rope__zzabi_launch_rows__RUNS                 NN__GPU__ZZPRIVATE_WIDE,     nn__rope__zzabi_body_rows,                NN__GPU__ZZPRIVATE_EACH(rows * heads * (rot / 2ull)), 256u, (o, x, cs, rows, heads, head_stride, row_stride, rot, over)
#define nn__rope__zzabi_launch_angles_rows__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__rope__zzabi_body_angles_rows,         NN__GPU__ZZPRIVATE_EACH(rows * (head_dim / 2ull)), 256u, (cs, first, theta, head_dim, rows)
#define nn__attention__zzabi_launch_causal_scores__RUNS   NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_causal_scores,  NN__GPU__ZZPRIVATE_UP_TO(rows * q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (p, q, k, q_heads, kv_heads, head_dim, first, rows)
#define nn__attention__zzabi_launch_causal_mix__RUNS      NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_causal_mix,     NN__GPU__ZZPRIVATE_UP_TO(rows * q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (o, p, v, q_heads, kv_heads, head_dim, first, rows, over)
#define nn__attention__zzabi_launch_gate_rows__RUNS       NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_gate_rows,      NN__GPU__ZZPRIVATE_EACH(heads * head_dim), 256u, (o, att, qg, heads, head_dim, over)
#define nn__deltanet__zzabi_launch_gates_rows__RUNS       NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_gates_rows,      NN__GPU__ZZPRIVATE_EACH(heads * rows), 256u, (beta, g, a, b, a_log, dt_bias, heads, rows, over)
#define nn__deltanet__zzabi_launch_conv_steps__RUNS       NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_conv_steps,      NN__GPU__ZZPRIVATE_EACH(ch), 256u, (o, s, w, x, ch, rows, over)
#define nn__deltanet__zzabi_launch_steps__RUNS            NN__GPU__ZZPRIVATE_WIDE,     nn__deltanet__zzabi_body_steps,           NN__GPU__ZZPRIVATE_UP_TO(v_heads, NN__KERNELS__BLOCKS_MAX), 256u, (S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over)
#define nn__hadamard__zzabi_launch_blocks__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__hadamard__zzabi_body_blocks,          NN__GPU__ZZPRIVATE_EACH(n), 256u, (out, in, sign, n, block, inverse, over)
#define nn__turboquant__zzabi_launch_encode_rows__RUNS    NN__GPU__ZZPRIVATE_WIDE,     nn__turboquant__zzabi_body_encode_rows,   NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (codes, scales, x, rows, cols, d, over)
#define nn__attention__zzabi_launch_weights_tq__RUNS      NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_weights_tq,     NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d)
#define nn__attention__zzabi_launch_residual_mix_tq__RUNS NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_residual_mix_tq, NN__GPU__ZZPRIVATE_UP_TO(q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (res, p, codes, scales, q_heads, kv_heads, head_dim, length, d)
#define nn__hyper__zzabi_launch_logits__RUNS              NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_logits,             NN__GPU__ZZPRIVATE_UP_TO(NN__HYPER__MIX(mult), NN__KERNELS__BLOCKS_MAX), 256u, (lg, streams, fn, hidden, mult, norm_eps)
#define nn__hyper__zzabi_launch_pre__RUNS                 NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_pre,                NN__GPU__ZZPRIVATE_EACH(hidden), 256u, (out, mix, streams, base, scale, hidden, mult, iters, eps, over)
#define nn__hyper__zzabi_launch_post__RUNS                NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_post,               NN__GPU__ZZPRIVATE_EACH(hidden), 256u, (out, streams, y, mix, hidden, mult, over)
#define nn__vector__zzabi_launch_top_k_biased__RUNS       NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_top_k_biased,      1u, 256u, (values, stride, weights, x, bias, n, k, scale)
#define nn__swiglu__zzabi_launch_clamped__RUNS            NN__GPU__ZZPRIVATE_WIDE,     nn__swiglu__zzabi_body_clamped,           NN__GPU__ZZPRIVATE_EACH(pairs * n), 256u, (act, gu, pairs, n, limit, over)
#define nn__kda__zzabi_launch_step__RUNS                  NN__GPU__ZZPRIVATE_WIDE,     nn__kda__zzabi_body_step,                 NN__GPU__ZZPRIVATE_UP_TO(heads, NN__KERNELS__BLOCKS_MAX), 256u, (S, conved, f, b, dt, a_log, gate, w, out, heads, head_dim, lower, eps, over)
#define nn__attention__zzabi_launch_absorb__RUNS         NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_absorb,         NN__GPU__ZZPRIVATE_UP_TO(heads, NN__KERNELS__BLOCKS_MAX), 256u, (out, q, W, heads, nope, latent, stride, scale, over)
#define nn__attention__zzabi_launch_expand__RUNS         NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_expand,         NN__GPU__ZZPRIVATE_UP_TO(heads * width, NN__KERNELS__BLOCKS_MAX), 256u, (out, o, W, heads, width, latent, stride, offset, over)
#define nn__index__zzabi_launch_layernorm__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__index__zzabi_body_layernorm,          1u, 256u, (o, x, w, b, n, eps, over)
#define nn__index__zzabi_launch_pool__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__index__zzabi_body_pool,               NN__GPU__ZZPRIVATE_EACH(pools * dim), 256u, (pooled, keys, gates, ape, first, pools, dim, kpool, over)
#define nn__index__zzabi_launch_scores__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__index__zzabi_body_scores,             NN__GPU__ZZPRIVATE_UP_TO(pools, NN__KERNELS__BLOCKS_MAX), 64u, (scores, q, w, pooled, heads, dim, pools, scale)
#define nn__index__zzabi_launch_select__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__index__zzabi_body_select,             1u, 256u, (index, count, scores, pools, k, kpool, tail_first, tail)
#define nn__index__zzabi_launch_gather__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__index__zzabi_body_gather,             NN__GPU__ZZPRIVATE_EACH(n * row), 256u, (out, src, index, n, row)
#define nn__vector__zzabi_launch_copy__RUNS              NN__GPU__ZZPRIVATE_WIDE,     nn__vector__zzabi_body_copy,              NN__GPU__ZZPRIVATE_EACH(n), 256u, (out, in, n)
#define nn__hyper__zzabi_launch_logits_rows__RUNS        NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_logits_rows,        NN__GPU__ZZPRIVATE_UP_TO(rows * NN__HYPER__MIX(mult), NN__KERNELS__BLOCKS_MAX), 256u, (lg, streams, fn, hidden, mult, norm_eps, rows, lg_stride)
#define nn__hyper__zzabi_launch_pre_rows__RUNS           NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_pre_rows,           NN__GPU__ZZPRIVATE_EACH(rows * hidden), 256u, (out, mix, streams, base, scale, hidden, mult, iters, eps, rows, mix_stride, over)
#define nn__hyper__zzabi_launch_post_rows__RUNS          NN__GPU__ZZPRIVATE_WIDE,     nn__hyper__zzabi_body_post_rows,          NN__GPU__ZZPRIVATE_EACH(rows * hidden), 256u, (out, streams, y, mix, hidden, mult, rows, mix_stride, over)
#define nn__expert__zzabi_launch_groups_int8__RUNS       NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_groups_int8,       NN__KERNELS__BLOCKS_MAX, 256u, (out, g, x, rows, cols, over)
#define nn__expert__zzabi_launch_rows_sum_int8__RUNS     NN__GPU__ZZPRIVATE_WIDE,     nn__expert__zzabi_body_rows_sum_int8,     NN__GPU__ZZPRIVATE_UP_TO(out_rows, NN__KERNELS__BLOCKS_MAX), 256u, (acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols)
#define nn__kda__zzabi_launch_steps__RUNS                NN__GPU__ZZPRIVATE_WIDE,     nn__kda__zzabi_body_steps,                NN__GPU__ZZPRIVATE_UP_TO(heads, NN__KERNELS__BLOCKS_MAX), 256u, (S, conved, f, b, dt, a_log, gate, w, out, heads, head_dim, lower, eps, rows, over)
#define nn__rank1__zzabi_launch_dots__RUNS               NN__GPU__ZZPRIVATE_WIDE,     nn__rank1__zzabi_body_dots,               NN__GPU__ZZPRIVATE_UP_TO(n, NN__KERNELS__BLOCKS_MAX), 256u, (s, v, which, x, w, w_at, n, cols, x_stride, w_stride)
#define nn__rank1__zzabi_launch_add__RUNS                NN__GPU__ZZPRIVATE_WIDE,     nn__rank1__zzabi_body_add,                NN__GPU__ZZPRIVATE_EACH(n * rows), 256u, (out, r, s, n, per, rows, out_stride, over)
#define nn__rank1__zzabi_launch_spread__RUNS             NN__GPU__ZZPRIVATE_WIDE,     nn__rank1__zzabi_body_spread,             NN__GPU__ZZPRIVATE_EACH(rows), 256u, (acc, r, s, row_of, n, rows, row_stride)
#define nn__attention__zzabi_launch_scores_grouped__RUNS NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_scores_grouped, NN__GPU__ZZPRIVATE_EACH(q_heads * ((length + NN__ATTENTION__GROUP_SPAN - 1u) / NN__ATTENTION__GROUP_SPAN)), 256u, (p, q, k, q_heads, kv_heads, head_dim, length)
#define nn__attention__zzabi_launch_softmax_rows__RUNS   NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_softmax_rows,   NN__GPU__ZZPRIVATE_UP_TO(rows, NN__KERNELS__BLOCKS_MAX), 256u, (p, rows, length)
#define nn__attention__zzabi_launch_causal_scores_grouped__RUNS NN__GPU__ZZPRIVATE_WIDE, nn__attention__zzabi_body_causal_scores_grouped, NN__GPU__ZZPRIVATE_EACH(rows * q_heads * ((first + rows + NN__ATTENTION__GROUP_SPAN - 1u) / NN__ATTENTION__GROUP_SPAN)), 256u, (p, q, k, q_heads, kv_heads, head_dim, first, rows)
#define nn__attention__zzabi_launch_causal_softmax__RUNS NN__GPU__ZZPRIVATE_WIDE,     nn__attention__zzabi_body_causal_softmax, NN__GPU__ZZPRIVATE_UP_TO(rows * q_heads, NN__KERNELS__BLOCKS_MAX), 256u, (p, q_heads, first, rows)

/* ── EVERY ROW BECOMES A KERNEL AND A DOOR ─────────────────────────────────────────────────────────────── */
/* ⛳ A SERIAL BODY RUNS ON ONE LANE OF ONE BLOCK, so PLAIN sends every other lane home first; a WIDE body
 *   divides its work by lane and block itself (`contracts/abi/gpu.cuh`) and every lane runs it — which is
 *   also what its combine step needs, since that waits for every lane.
 * ⛳ THE TWO VALUE FORMS RUN THE BODY ON EVERY LANE AND LET THE FIRST ONE WRITE, so a value body may be
 *   wide and end in the combine step, which hands every lane the same answer. A serial value body is
 *   started one lane wide, so for it the two are the same thing. */
#define NN__GPU__ZZPRIVATE_ONE_LANE    if (nn__silicon__lane() != 0u || nn__silicon__block() != 0u) return;
#define NN__GPU__ZZPRIVATE_WIDE(BODY, NAMES)      (void)BODY NAMES;
#define NN__GPU__ZZPRIVATE_PLAIN(BODY, NAMES)     NN__GPU__ZZPRIVATE_ONE_LANE (void)BODY NAMES;
#define NN__GPU__ZZPRIVATE_FIRST_LANE  (nn__silicon__lane() == 0u && nn__silicon__block() == 0u)
#define NN__GPU__ZZPRIVATE_AS_INT(BODY, NAMES)                                                              \
    const uint64_t answer = (uint64_t)(uint32_t)BODY NN__GPU__ZZPRIVATE_AFTER_STAMP NAMES;                 \
    if (NN__GPU__ZZPRIVATE_FIRST_LANE)                                                                      \
        NN__GPU__ZZPRIVATE_OUT NAMES = NN__GPU__ZZPRIVATE_STAMP NAMES | answer;
#define NN__GPU__ZZPRIVATE_AS_FLOAT(BODY, NAMES)                                                            \
    const uint64_t answer = (uint64_t)nn__result__zzpackage_float_bits(                                     \
                                (float)BODY NN__GPU__ZZPRIVATE_AFTER_STAMP NAMES);                          \
    if (NN__GPU__ZZPRIVATE_FIRST_LANE)                                                                      \
        NN__GPU__ZZPRIVATE_OUT NAMES = NN__GPU__ZZPRIVATE_STAMP NAMES | answer;
#define NN__GPU__ZZPRIVATE_OUT(out, stamp, ...)          *out
#define NN__GPU__ZZPRIVATE_STAMP(out, stamp, ...)        stamp
#define NN__GPU__ZZPRIVATE_AFTER_STAMP(out, stamp, ...)  (__VA_ARGS__)

/* `(a, b, c)` -> `&a, &b, &c`: the pointer to each argument a launch by pointer takes; `SIZES` gives each
 * one's width and `COUNT` how many, for a launch that sets arguments one at a time. Twelve is the widest
 * a row may be — a verb's own operand limit. */
#define NN__GPU__ZZPRIVATE_NTH(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...)  N
#define NN__GPU__ZZPRIVATE_ADDRESSES(...)                                                                   \
    NN__GPU__ZZPRIVATE_NTH(__VA_ARGS__, NN__GPU__ZZPRIVATE_A16, NN__GPU__ZZPRIVATE_A15, NN__GPU__ZZPRIVATE_A14, NN__GPU__ZZPRIVATE_A13, NN__GPU__ZZPRIVATE_A12, NN__GPU__ZZPRIVATE_A11, NN__GPU__ZZPRIVATE_A10, NN__GPU__ZZPRIVATE_A9, NN__GPU__ZZPRIVATE_A8, \
                           NN__GPU__ZZPRIVATE_A7, NN__GPU__ZZPRIVATE_A6, NN__GPU__ZZPRIVATE_A5,             \
                           NN__GPU__ZZPRIVATE_A4, NN__GPU__ZZPRIVATE_A3, NN__GPU__ZZPRIVATE_A2,             \
                           NN__GPU__ZZPRIVATE_A1)(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A1(a)       &a
#define NN__GPU__ZZPRIVATE_A2(a, ...)  &a, NN__GPU__ZZPRIVATE_A1(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A3(a, ...)  &a, NN__GPU__ZZPRIVATE_A2(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A4(a, ...)  &a, NN__GPU__ZZPRIVATE_A3(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A5(a, ...)  &a, NN__GPU__ZZPRIVATE_A4(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A6(a, ...)  &a, NN__GPU__ZZPRIVATE_A5(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A7(a, ...)  &a, NN__GPU__ZZPRIVATE_A6(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A8(a, ...)  &a, NN__GPU__ZZPRIVATE_A7(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A9(a, ...)  &a, NN__GPU__ZZPRIVATE_A8(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A10(a, ...) &a, NN__GPU__ZZPRIVATE_A9(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A11(a, ...) &a, NN__GPU__ZZPRIVATE_A10(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A12(a, ...) &a, NN__GPU__ZZPRIVATE_A11(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A13(a, ...) &a, NN__GPU__ZZPRIVATE_A12(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A14(a, ...) &a, NN__GPU__ZZPRIVATE_A13(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A15(a, ...) &a, NN__GPU__ZZPRIVATE_A14(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_A16(a, ...) &a, NN__GPU__ZZPRIVATE_A15(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_SIZES(...)                                                                       \
    NN__GPU__ZZPRIVATE_NTH(__VA_ARGS__, NN__GPU__ZZPRIVATE_S16, NN__GPU__ZZPRIVATE_S15, NN__GPU__ZZPRIVATE_S14, NN__GPU__ZZPRIVATE_S13, NN__GPU__ZZPRIVATE_S12, NN__GPU__ZZPRIVATE_S11, NN__GPU__ZZPRIVATE_S10, NN__GPU__ZZPRIVATE_S9, NN__GPU__ZZPRIVATE_S8, \
                           NN__GPU__ZZPRIVATE_S7, NN__GPU__ZZPRIVATE_S6, NN__GPU__ZZPRIVATE_S5,             \
                           NN__GPU__ZZPRIVATE_S4, NN__GPU__ZZPRIVATE_S3, NN__GPU__ZZPRIVATE_S2,             \
                           NN__GPU__ZZPRIVATE_S1)(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S1(a)       sizeof(a)
#define NN__GPU__ZZPRIVATE_S2(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S1(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S3(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S2(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S4(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S3(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S5(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S4(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S6(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S5(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S7(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S6(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S8(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S7(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S9(a, ...)  sizeof(a), NN__GPU__ZZPRIVATE_S8(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S10(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S9(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S11(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S10(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S12(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S11(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S13(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S12(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S14(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S13(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S15(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S14(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_S16(a, ...) sizeof(a), NN__GPU__ZZPRIVATE_S15(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_COUNT(...)                                                                       \
    NN__GPU__ZZPRIVATE_NTH(__VA_ARGS__, 16u, 15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u)

/* The row's `FN##__RUNS` line is pasted and expanded into its five columns before the last step reads them.
 * Every door is expanded inside one `extern "C"` block below: the doors are nn's C interface, and the block
 * is how the C-subset gate knows which definitions escape on purpose. */
#define NN__GPU__ZZPRIVATE_ROW(PKG, RET, NAME, FN, PARAMS)   NN__GPU__ZZPRIVATE_SPREAD(FN, PARAMS, FN##__RUNS)
#define NN__GPU__ZZPRIVATE_SPREAD(FN, PARAMS, ...)          NN__GPU__ZZPRIVATE_EMIT(FN, PARAMS, __VA_ARGS__)
/* ⛳ A FAMILY THAT FINDS KERNELS BY NAME (`SYS__SILICON_FAMILY__KERNELS_BY_NAME`, OpenCL) compiles its
 * kernels from these same rows in its own language, so none is emitted here and the door passes no address. */
#ifndef SYS__SILICON_FAMILY__KERNELS_BY_NAME
/* ⛔⛔ EVERY KERNEL DECLARES THE MOST LANES IT CAN BE STARTED WITH, AND IT IS NOT A HINT. A kernel that
 *   declares nothing is compiled for 1,024 lanes a block, and a card that must hold 1,024 lanes' registers
 *   gives each one a quarter of what 256 would get. `MEASURED` on gfx906: the wide fp16 matvec
 *   compiled to exactly 64 VGPRs — the undeclared ceiling — with 16 spilled to scratch. No door here
 *   starts more than `NN__SILICON__LANES_MAX`, and the compiler refuses one that tries, so the
 *   declaration is the truth and not a promise. */
#define NN__GPU__ZZPRIVATE_BOUNDS  __launch_bounds__(NN__SILICON__LANES_MAX)
#define NN__GPU__ZZPRIVATE_KERNEL(FN, PARAMS, FORM, BODY, NAMES)                                            \
    static __global__ NN__GPU__ZZPRIVATE_BOUNDS void FN##__kernel PARAMS {                                  \
        FORM(BODY, NAMES)                                                                                   \
    }
#define NN__GPU__ZZPRIVATE_ADDRESS(FN)  ((const void*)FN##__kernel)
#else
#define NN__GPU__ZZPRIVATE_KERNEL(FN, PARAMS, FORM, BODY, NAMES)
#define NN__GPU__ZZPRIVATE_ADDRESS(FN)  ((const void*)0)
#endif
#ifndef SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS
#define NN__GPU__ZZPRIVATE_EMIT(FN, PARAMS, FORM, BODY, BLOCKS, THREADS, NAMES)                            \
    static_assert((THREADS) >= 1u && (THREADS) <= NN__SILICON__LANES_MAX,                                   \
                  #FN " starts more lanes in a block than every family can run");                          \
    NN__GPU__ZZPRIVATE_KERNEL(FN, PARAMS, FORM, BODY, NAMES)                                                \
    void FN PARAMS {                                                                                        \
        void* args[] = { NN__GPU__ZZPRIVATE_ADDRESSES NAMES };                                              \
        const size_t sizes[] = { NN__GPU__ZZPRIVATE_SIZES NAMES };                                          \
        nn__silicon__launch(NN__GPU__ZZPRIVATE_ADDRESS(FN), #FN "__kernel", BLOCKS, THREADS, args, sizes,   \
                            NN__GPU__ZZPRIVATE_COUNT NAMES);                                                \
    }
#else
/* ⭐ A FAMILY WHOSE BLOCKS ARE THREADS OF THIS PROCESS (`SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS`, the CPU)
 *   has no kernel to launch by pointer, so each door's arguments travel as one struct: the door fills it,
 *   and the block function — what the family's launch runs once for every block — takes each argument
 *   back out under its own name and runs the row's form exactly as a kernel would. The launch is handed
 *   the block function as the kernel and the struct as its one argument. */
#define NN__GPU__ZZPRIVATE_F1(p)       p;
#define NN__GPU__ZZPRIVATE_F2(p, ...)  p; NN__GPU__ZZPRIVATE_F1(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F3(p, ...)  p; NN__GPU__ZZPRIVATE_F2(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F4(p, ...)  p; NN__GPU__ZZPRIVATE_F3(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F5(p, ...)  p; NN__GPU__ZZPRIVATE_F4(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F6(p, ...)  p; NN__GPU__ZZPRIVATE_F5(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F7(p, ...)  p; NN__GPU__ZZPRIVATE_F6(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F8(p, ...)  p; NN__GPU__ZZPRIVATE_F7(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F9(p, ...)  p; NN__GPU__ZZPRIVATE_F8(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F10(p, ...) p; NN__GPU__ZZPRIVATE_F9(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F11(p, ...) p; NN__GPU__ZZPRIVATE_F10(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F12(p, ...) p; NN__GPU__ZZPRIVATE_F11(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F13(p, ...) p; NN__GPU__ZZPRIVATE_F12(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F14(p, ...) p; NN__GPU__ZZPRIVATE_F13(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F15(p, ...) p; NN__GPU__ZZPRIVATE_F14(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_F16(p, ...) p; NN__GPU__ZZPRIVATE_F15(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_FIELDS(...)                                                                      \
    NN__GPU__ZZPRIVATE_NTH(__VA_ARGS__, NN__GPU__ZZPRIVATE_F16, NN__GPU__ZZPRIVATE_F15, NN__GPU__ZZPRIVATE_F14, NN__GPU__ZZPRIVATE_F13, NN__GPU__ZZPRIVATE_F12, NN__GPU__ZZPRIVATE_F11, NN__GPU__ZZPRIVATE_F10, NN__GPU__ZZPRIVATE_F9, NN__GPU__ZZPRIVATE_F8, \
                           NN__GPU__ZZPRIVATE_F7, NN__GPU__ZZPRIVATE_F6, NN__GPU__ZZPRIVATE_F5,             \
                           NN__GPU__ZZPRIVATE_F4, NN__GPU__ZZPRIVATE_F3, NN__GPU__ZZPRIVATE_F2,             \
                           NN__GPU__ZZPRIVATE_F1)(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L1(a)       __typeof__(zz->a) a = zz->a;
#define NN__GPU__ZZPRIVATE_L2(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L1(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L3(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L2(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L4(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L3(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L5(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L4(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L6(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L5(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L7(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L6(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L8(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L7(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L9(a, ...)  NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L8(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L10(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L9(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L11(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L10(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L12(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L11(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L13(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L12(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L14(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L13(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L15(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L14(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_L16(a, ...) NN__GPU__ZZPRIVATE_L1(a) NN__GPU__ZZPRIVATE_L15(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_LOCALS(...)                                                                      \
    NN__GPU__ZZPRIVATE_NTH(__VA_ARGS__, NN__GPU__ZZPRIVATE_L16, NN__GPU__ZZPRIVATE_L15, NN__GPU__ZZPRIVATE_L14, NN__GPU__ZZPRIVATE_L13, NN__GPU__ZZPRIVATE_L12, NN__GPU__ZZPRIVATE_L11, NN__GPU__ZZPRIVATE_L10, NN__GPU__ZZPRIVATE_L9, NN__GPU__ZZPRIVATE_L8, \
                           NN__GPU__ZZPRIVATE_L7, NN__GPU__ZZPRIVATE_L6, NN__GPU__ZZPRIVATE_L5,             \
                           NN__GPU__ZZPRIVATE_L4, NN__GPU__ZZPRIVATE_L3, NN__GPU__ZZPRIVATE_L2,             \
                           NN__GPU__ZZPRIVATE_L1)(__VA_ARGS__)
#define NN__GPU__ZZPRIVATE_UNPAREN(...)  __VA_ARGS__
#define NN__GPU__ZZPRIVATE_EMIT(FN, PARAMS, FORM, BODY, BLOCKS, THREADS, NAMES)                            \
    static_assert((THREADS) >= 1u && (THREADS) <= NN__SILICON__LANES_MAX,                                   \
                  #FN " starts more lanes in a block than every family can run");                          \
    struct FN##__args { NN__GPU__ZZPRIVATE_FIELDS PARAMS };                                                 \
    static void FN##__block(void* zzargs) {                                                                 \
        const struct FN##__args* zz = (const struct FN##__args*)zzargs;                                     \
        NN__GPU__ZZPRIVATE_LOCALS NAMES                                                                     \
        FORM(BODY, NAMES)                                                                                   \
    }                                                                                                       \
    void FN PARAMS {                                                                                        \
        struct FN##__args zzs = { NN__GPU__ZZPRIVATE_UNPAREN NAMES };                                       \
        void* args[] = { &zzs };                                                                            \
        const size_t sizes[] = { sizeof zzs };                                                              \
        nn__silicon__launch((const void*)FN##__block, #FN "__block", BLOCKS, THREADS, args, sizes, 1u);     \
    }
#endif
extern "C" {
nn__CONTRACT__DOORS(NN__GPU__ZZPRIVATE_ROW, nn)
}

#endif /* SILVANN__PACKAGES_NN_GPU_DOORS_CUH */
