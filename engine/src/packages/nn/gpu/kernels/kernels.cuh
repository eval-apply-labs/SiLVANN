#ifndef SILVANN__PACKAGES_NN_GPU_KERNELS_KERNELS_CUH
#define SILVANN__PACKAGES_NN_GPU_KERNELS_KERNELS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/abi/gpu.cuh"   /* the arithmetic it asks a family for */

/* ══ ⭐⭐⭐ THE COMPUTE, WITHOUT THE LANGUAGE — WHAT BECOMES A KERNEL ══════════════════════════════════
 *
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * How these bodies came out of the verbs, and what the extraction did and did not change.
 *   · THE MOVE IS ATOMIC, which is why every body was extracted before anything flipped: `nn`'s verb
 *     bodies call `sys`, so `sys` cannot go host while a body is still device code, and a device body
 *     cannot launch a kernel. Half a port does not run.
 *   · At extraction each body stayed `__device__ inline`, called in place, so the extracting commit
 *     changed no behaviour and the 1630 host and 445 device checks were its oracle. The `__global__`
 *     wrappers and `extern "C"` doors in `../doors.cuh` are the second, riskier step.
 *   · Every body is the serial loop that sat in its verb, moved as it was — not parallelised. The
 *     DeltaNet three were flagged as the port's risk from the start.
 *   · `over` was a `bool` in the verbs; it is an `unsigned int` here because that is what `atomicOr`
 *     takes, and the verb reads `!= 0` exactly as it read the bool.
 *   · TurboQuant's two bodies were the last to split: their row loop called `sys__opcodes__fails` mid-
 *     compute, which is why they answer a bool now.
 *   · The census that counted 24 buffer verbs keyed on a direct `nn__primitives__room()` call and missed
 *     `rotate`, which was already factored into pointer-only blocks. The count is 25.
 *   · HELPERS, moved here. The first extraction left behind every function the loops call —
 *     pure compute sitting in `*__impl.cuh`. The first draft of the helpers banner said five; the closure
 *     needed three more, then two more. Nothing noticed because `kernels.cuh` shares a translation unit
 *     with the verbs, so the split existed in the FILING and not in the BUILD. Compiling the device TU by
 *     itself and reading the LINKER found them; `-fsyntax-only` on the same TU reported ZERO errors
 *     (0 against 5 on identical input) — a declaration is all a syntax pass needs. The device TU now
 *     compiles and links with no `sys` in it, which is what proves "② reaches for nothing in `sys`".
 *   · The three private TurboQuant helpers were first appended after `zzprivate_weight`, which calls
 *     them, and the device TU failed on exactly those three names.
 *   · `hadamard__header.cuh`, `turboquant__header.cuh` carry forward declarations for this file.
 * ⛳ RETIREMENT: when the host evaluator is the only evaluator.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⚖ *"do the rearrangement, sys and the evaluator go host."*
 *
 * Every `nn` verb that touches a buffer has exactly the same three parts, in this order:
 * ```
 *     ① the arity check, `nn__primitives__room()`, the bounds          ← sys. HOST.
 *     ② the loop, on raw pointers and nothing else                     ← THIS FILE. the card.
 *     ③ `fails()` / `becomes()`                                        ← sys. HOST.
 * ```
 * ⛳ **THE SEAM IS WHERE THE RAW POINTERS APPEAR**, and it is the same line in every one of them. ②
 * reaches for nothing in `sys` at all, and asks a family only for the arithmetic `nn` declares beside its
 * doors (`contracts/abi/gpu.cuh`) — `MEASURED`: the device TU compiles and links with no `sys` in it. That is what makes the split mechanical rather than a
 * redesign.
 *
 * ⛳ EACH BODY IS `__device__ inline`, so a verb can call it in place and `../doors.cuh` can wrap
 * it in a `__global__` and an `extern "C"` door. ▶ that file for the doors and what a launch costs.
 *
 * ⛔ THE OVERFLOW FLAG IS AN `unsigned int`, NOT A BOOL, AND THAT IS ON PURPOSE. A kernel has many
 * threads and needs something they can all write; `unsigned int` is what an `atomicOr` takes. The verb
 * reads it as `!= 0`.
 * ⭐ AND WHAT IS WRITTEN IS THE FAULT'S OWN WORD, NOT A 1 — `NN__PRIMITIVES__FAULT_OVERFLOW`, "NPOV". A door's
 * flag is the worker's fault word, and the engine raises whatever it finds there into the machine's
 * channel, so a program learns WHAT happened and not merely that something did. Every thread writing
 * the same word, OR-ed or stored, leaves that word.
 *
 * ⛳ A BODY IS SERIAL UNTIL IT IS WIDENED, AND EACH ONE IS WIDENED ON ITS OWN, AGAINST THE HOST —
 * `gpu/doors.cuh` says which are WIDE. The sections below say where the argument for widening is hard.
 */
#define NN__KERNELS__OVERFLOWED  ((unsigned int)NN__PRIMITIVES__FAULT_OVERFLOW)   /* ▶ the note above */

/* A float's 32 bits, for the value verbs that hand a float back inside a stamped word. */
static __device__ inline uint32_t nn__result__zzpackage_float_bits(float f) {
    union { float f; uint32_t u; } cast; cast.f = f; return cast.u;
}

/* ⭐ THE SPREAD — a pointwise body's item `i` goes to lane `i` of all the lanes of all the blocks, and each
 * lane then steps by all of them. Neighbouring lanes take neighbouring items, so a wave reads and writes
 * in order; on the host, one lane of one block, it is the serial loop. */
static __device__ inline uint64_t nn__kernels__zzprivate_first(void) {
    return (uint64_t)nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane();
}
static __device__ inline uint64_t nn__kernels__zzprivate_stride(void) {
    return (uint64_t)nn__silicon__blocks() * nn__silicon__lanes();
}

/* ── ONE OPERAND IN, ONE OUT ────────────────────────────────────────────────────────────────────── */

static __device__ inline void nn__vector__zzabi_body_zero(uint16_t* v, uint64_t n) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) v[i] = 0u;
}

static __device__ inline void nn__vector__zzabi_body_exp(uint16_t* o, const uint16_t* x,
                                                          uint64_t n, unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__expf(nn__silicon__half_to_float(x[i])), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⛳ `softplus` IS WRITTEN THE STABLE WAY AND STAYS THAT WAY: `max(v,0) + log1p(exp(-|v|))` never
 * exponentiates a large positive number, which the naive `log(1+exp(v))` does. */
static __device__ inline void nn__vector__zzabi_body_softplus(uint16_t* o, const uint16_t* x,
                                                               uint64_t n, unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const float v   = nn__silicon__half_to_float(x[i]);
        const float mag = (v < 0.0f) ? -v : v;
        const float hi  = (v > 0.0f) ? v : 0.0f;
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   hi + nn__silicon__logf(1.0f + nn__silicon__expf(-mag)), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__vector__zzabi_body_scale(uint16_t* o, const uint16_t* x,
                                                            uint64_t n, float factor,
                                                            unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__half_to_float(x[i]) * factor, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── TWO OPERANDS IN, ONE OUT ───────────────────────────────────────────────────────────────────── */

/* ⛳ THE SUM IS TAKEN IN `float` THOUGH EVERY OPERAND IS A HALF — one add at full width, one rounding
 * at the end, which is what the reference does and what the oracle compares against. */
static __device__ inline void nn__vector__zzabi_body_add(uint16_t* z, const uint16_t* x,
                                                          const uint16_t* y, uint64_t n,
                                                          unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        bool hit = false;
        z[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__half_to_float(x[i])
                 + nn__silicon__half_to_float(y[i]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__vector__zzabi_body_pointwise_mul(uint16_t* z, const uint16_t* x,
                                                                    const uint16_t* y, uint64_t n,
                                                                    unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        bool hit = false;
        z[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__half_to_float(x[i])
                 * nn__silicon__half_to_float(y[i]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* A float infinity, from the compiler itself: every family's compiler has it, and no header is needed. */
#define NN__KERNELS__INFINITY __builtin_inff()

/* ── REDUCTIONS THAT BROADCAST BACK ──────────────────────────────────────────────────────────────
 * ⭐ WIDE OVER ONE BLOCK. Each lane takes items `lane`, `lane + lanes`, … of the reduction, the block's
 * combine step hands every lane the whole answer, and then each lane writes its own items of the output.
 * One block because the answer is needed by every item: two blocks would each have to hold the whole sum,
 * which is the whole reduction again. On the host, one lane of one block, each is the serial loop.
 * ⚠ The sum's order is the combine step's, so the output can differ from the serial loop's in its last
 * bits — the same on every card of a family, run after run. */

static __device__ inline void nn__rmsnorm__zzabi_body(uint16_t* o, const uint16_t* x,
                                                       const uint16_t* w, uint64_t n,
                                                       unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float part = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes) {
        const float v = nn__silicon__half_to_float(x[i]);
        part += v * v;
    }
    const float mean  = nn__silicon__lanes_sum(part) / (float)n;
    const float scale = 1.0f / nn__silicon__sqrtf(mean + 1e-6f);
    for (uint64_t i = lane; i < n; i += lanes) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   (nn__silicon__half_to_float(x[i]) * scale)
                 * nn__silicon__half_to_float(w[i]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__vector__zzabi_body_l2norm(uint16_t* o, const uint16_t* x,
                                                             uint64_t n, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float part = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes) {
        const float v = nn__silicon__half_to_float(x[i]);
        part += v * v;
    }
    const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(part) + 1e-6f);
    for (uint64_t i = lane; i < n; i += lanes) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__half_to_float(x[i]) * scale, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⛳ THE MAX IS SUBTRACTED BEFORE EVERY `exp` AND THAT IS NOT AN OPTIMISATION — it is what stops
 * `exp(large)` becoming an infinity that turns the whole row into NaN. Three passes, all kept, and the
 * first two each end in the combine step. */
static __device__ inline void nn__softmax__zzabi_body(uint16_t* o, const uint16_t* x,
                                                       uint64_t n, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float part = -NN__KERNELS__INFINITY;
    for (uint64_t i = lane; i < n; i += lanes) {
        const float v = nn__silicon__half_to_float(x[i]);
        if (v > part) part = v;
    }
    const float top = nn__silicon__lanes_max(part);
    part = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes)
        part += nn__silicon__expf(nn__silicon__half_to_float(x[i]) - top);
    const float inv = 1.0f / nn__silicon__lanes_sum(part);
    for (uint64_t i = lane; i < n; i += lanes) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__expf(nn__silicon__half_to_float(x[i]) - top) * inv, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__sigmoid__zzabi_body(uint16_t* o, const uint16_t* x,
                                                       uint64_t n, unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const float v = nn__silicon__half_to_float(x[i]);
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(1.0f / (1.0f + nn__silicon__expf(-v)), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⛳ THE FACTOR IS READ OUT OF A BUFFER, NOT OUT OF THE FORM — which is the whole difference from
 * `scale`, and the reason a program can scale by something it just computed.
 * ⛳ WIDE, UNLESS THE OUTPUT COVERS THE FACTOR. A program may scale a buffer by one of its own elements
 * (`o` and `a` the same), and spread over lanes one of them would overwrite `a[which]` before another had
 * read it — so when `a[which]` lies inside `o[0..n)`, one lane runs the loop, reading the factor first.
 * Otherwise every lane reads the factor, which nothing in this launch writes, and takes its own elements.
 * `MEASURED`: serial, this was 0.82 ms at 2048 — 295 of a 35B token's 405 ms, nine a layer. */
static __device__ inline void nn__vector__zzabi_body_scale_at(uint16_t* o, const uint16_t* x,
                                                               const uint16_t* a, uint64_t which,
                                                               uint64_t n, unsigned int* over) {
    const float factor = nn__silicon__half_to_float(a[which]);
    const bool covers = a + which >= (const uint16_t*)o && a + which < (const uint16_t*)o + n;
    if (covers && nn__kernels__zzprivate_first() != 0ull) return;
    const uint64_t first = covers ? 0ull : nn__kernels__zzprivate_first();
    const uint64_t stride = covers ? 1ull : nn__kernels__zzprivate_stride();
    for (uint64_t i = first; i < n; i += stride) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   nn__silicon__half_to_float(x[i]) * factor, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── THE TWO NESTED-LOOP ONES IN THIS FILE ──────────────────────────────────────────────────────── */

static __device__ inline void nn__matrix__zzabi_body_transpose(uint16_t* d, const uint16_t* s,
                                                                uint64_t rows, uint64_t cols) {
    for (uint64_t c = 0ull; c < cols; ++c)
        for (uint64_t r = 0ull; r < rows; ++r)
            d[c * rows + r] = s[r * cols + c];
}

/* ⭐ WIDE, AND IT NEEDS NO COMBINE STEP: a lane owns whole output columns, and for each row neighbouring
 * lanes read neighbouring weights while all of them read the same `x[r]`. */
static __device__ inline void nn__matrix__zzabi_body_matvec_transposed(uint16_t* out,
                                                                        const uint16_t* W,
                                                                        const uint16_t* x,
                                                                        uint64_t rows, uint64_t cols,
                                                                        unsigned int* over) {
    for (uint64_t c = nn__kernels__zzprivate_first(); c < cols; c += nn__kernels__zzprivate_stride()) {
        float sum = 0.0f;
        for (uint64_t r = 0ull; r < rows; ++r)
            sum += nn__silicon__half_to_float(W[r * cols + c])
                 * nn__silicon__half_to_float(x[r]);
        bool hit = false;
        out[c] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── THE THREE THAT ANSWER A NUMBER RATHER THAN FILLING A BUFFER ─────────────────────────────────
 * ⛳ THEY RETURN, WHERE THE OTHERS WRITE — and on the card that becomes a one-element output the host
 * reads back. The verb's own answer node is built above this line and stays there, so what crosses is
 * a float or an index and never a `sys__heap_node`. */

/* ⭐ WIDE OVER ONE BLOCK, AND THE FIRST INDEX OF THE LARGEST STILL WINS. Each lane walks its items upward
 *   keeping its own first-largest; the combine step finds the largest of those, and then the smallest
 *   index holding it — a largest of negated indices, exact while an index fits a float's 24 bits. Past
 *   that the first lane walks the whole vector alone. A NaN is never chosen; the serial loop chose index
 *   0 when the NaN was item 0, and nothing else. */
#define NN__KERNELS__INDEX_EXACT (1ull << 24)
static __device__ inline uint64_t nn__argmax__zzabi_body(const uint16_t* x, uint64_t n) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    if (n > NN__KERNELS__INDEX_EXACT) {
        uint64_t best_i = 0ull;
        float    best_v = nn__silicon__half_to_float(x[0]);
        if (lane == 0ull)
            for (uint64_t i = 1ull; i < n; ++i) {
                const float v = nn__silicon__half_to_float(x[i]);
                if (v > best_v) { best_v = v; best_i = i; }
            }
        return best_i;
    }
    uint64_t best_i = n;
    float    best_v = -NN__KERNELS__INFINITY;
    for (uint64_t i = lane; i < n; i += lanes) {
        const float v = nn__silicon__half_to_float(x[i]);
        if (best_i == n ? v == v : v > best_v) { best_v = v; best_i = i; }
    }
    const float top = nn__silicon__lanes_max(best_v);
    const float first = nn__silicon__lanes_max(best_i != n && best_v == top ? -(float)best_i : -NN__KERNELS__INFINITY);
    return first == -NN__KERNELS__INFINITY ? 0ull : (uint64_t)(-first);
}

/* ⭐⭐ THE `k` LARGEST, AND THEIR SOFTMAX — the router's choice. ⚖ *"we need to have a list of some sort,
 *   which can give us the top k"*. One block: `k` rounds, each the largest value not yet taken, found the
 *   way `argmax` finds it (the first index of the largest; a NaN never chosen). The weights are the
 *   softmax over the `k` values alone, which is exactly HF's router: its softmax over every expert,
 *   top `k`, divided by their sum — the other experts' terms cancel in that division.
 * ⛳ THE INDICES GO WHERE `values` POINTS, `stride` words apart — into the value words of nodes a program
 *   made, eight words to a node — and the weights into `weights` as halves, for `scale_at` to read. */
#define NN__VECTOR__TOP_K_MAX 16u
static __device__ inline void nn__vector__zzabi_body_top_k(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                            const uint16_t* x, uint64_t n, uint64_t k) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    float    chosen_v[NN__VECTOR__TOP_K_MAX];
    for (uint64_t r = 0ull; r < k; ++r) {
        uint64_t best_i = n;
        float    best_v = -NN__KERNELS__INFINITY;
        for (uint64_t i = lane; i < n; i += lanes) {
            bool taken = false;
            for (uint64_t q = 0ull; q < r; ++q) taken = taken || chosen[q] == i;
            const float v = nn__silicon__half_to_float(x[i]);
            if (taken || v != v) continue;
            if (best_i == n || v > best_v) { best_v = v; best_i = i; }
        }
        const float top = nn__silicon__lanes_max(best_v);
        const float first = nn__silicon__lanes_max(best_i != n && best_v == top ? -(float)(uint32_t)best_i
                                                                                : -NN__KERNELS__INFINITY);
        chosen[r]   = first == -NN__KERNELS__INFINITY ? n : (uint64_t)(-first);
        chosen_v[r] = top;
    }
    if (lane != 0ull || nn__silicon__block() != 0u) return;
    float sum = 0.0f;
    for (uint64_t r = 0ull; r < k; ++r) sum += nn__silicon__expf(chosen_v[r] - chosen_v[0]);
    for (uint64_t r = 0ull; r < k; ++r) {
        values[r * stride] = chosen[r];
        bool hit = false;
        weights[r] = nn__primitives__zzpackage_float_to_half(nn__silicon__expf(chosen_v[r] - chosen_v[0]) / sum, &hit);
    }
}

static __device__ inline float nn__vector__zzabi_body_dot_product(const uint16_t* a,
                                                                   const uint16_t* b, uint64_t n) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float part = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes)
        part += nn__silicon__half_to_float(a[i])
              * nn__silicon__half_to_float(b[i]);
    return nn__silicon__lanes_sum(part);
}

static __device__ inline float nn__vector__zzabi_body_at(const uint16_t* x, uint64_t i) {
    return nn__silicon__half_to_float(x[i]);
}

/* ── SWIGLU ─────────────────────────────────────────────────────────────────────────────────────── */
/* ⛳ THE CLAMP AT ±88 IS NOT TIDINESS: `expf(-x)` overflows a float below about -88, so an unclamped
 * gate turns one activation into an infinity and the row into NaN. It is part of the arithmetic. */
static __device__ inline void nn__swiglu__zzabi_body_combine(uint16_t* o, const uint16_t* g,
                                                              const uint16_t* u, uint64_t n,
                                                              unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const float raw     = nn__silicon__half_to_float(g[i]);
        const float clamped = (raw < -88.0f) ? -88.0f : ((raw > 88.0f) ? 88.0f : raw);
        const float silu    = clamped / (1.0f + nn__silicon__expf(-clamped));
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(
                   silu * nn__silicon__half_to_float(u[i]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⭐ SEVERAL SWIGLUS IN ONE LAUNCH, each pair's gate and up side by side as a MoE's gate_up rows leave them:
 *   `o[p·n + j] = silu(gu[p·2n + j]) · gu[p·2n + n + j]` for `pairs` of `n` — the combine above, `pairs` times. */
static __device__ inline void nn__swiglu__zzabi_body_pairs(uint16_t* o, const uint16_t* gu, uint64_t pairs, uint64_t n,
                                                           unsigned int* over) {
    const uint64_t all = pairs * n;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < all; i += nn__kernels__zzprivate_stride()) {
        const uint64_t p = i / n, j = i - p * n;
        const float raw     = nn__silicon__half_to_float(gu[p * 2ull * n + j]);
        const float clamped = (raw < -88.0f) ? -88.0f : ((raw > 88.0f) ? 88.0f : raw);
        const float silu    = clamped / (1.0f + nn__silicon__expf(-clamped));
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(silu * nn__silicon__half_to_float(gu[p * 2ull * n + n + j]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── THE DELTANET THREE — WIDE, EACH WITH ITS OWN ARGUMENT ─────────────────────────────────────────
 * ⛳ `rank_1_update` and `conv_step` both CARRY STATE FORWARD, so a lane per index is right only where no
 * index reads what another writes. Each of the three says why that holds for it, beside its body. */

/* ⛳ A LANE PER ELEMENT OF `S`: `S[i][j] <- g·S[i][j] + k[i]·d[j]` reads and writes that one element and
 *   nothing else of `S`, and `k` and `d` are only read. */
static __device__ inline void nn__deltanet__zzabi_body_rank_1_update(float* S, const uint16_t* k,
                                                                      const uint16_t* d, float g,
                                                                      uint64_t k_dim, uint64_t v_dim) {
    const uint64_t n = k_dim * v_dim;
    for (uint64_t e = nn__kernels__zzprivate_first(); e < n; e += nn__kernels__zzprivate_stride()) {
        const uint64_t i = e / v_dim, j = e - i * v_dim;
        S[e] = g * S[e] + nn__silicon__half_to_float(k[i]) * nn__silicon__half_to_float(d[j]);
    }
}

/* ⛳ A LANE PER OUTPUT COLUMN `j`, summing down the key axis: neighbouring lanes read neighbouring floats
 *   of each row of `S`. `o` must not overlap `x` — a lane writing `o[j]` would change an `x` another lane
 *   has yet to read — and no caller passes one for the other. */
static __device__ inline void nn__deltanet__zzabi_body_readout(uint16_t* o, const float* S,
                                                                const uint16_t* x, uint64_t k_dim,
                                                                uint64_t v_dim, unsigned int* over) {
    for (uint64_t j = nn__kernels__zzprivate_first(); j < v_dim; j += nn__kernels__zzprivate_stride()) {
        float sum = 0.0f;
        for (uint64_t i = 0ull; i < k_dim; ++i)
            sum += S[i * v_dim + j] * nn__silicon__half_to_float(x[i]);
        bool hit = false;
        o[j] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⛳ A LANE PER CHANNEL: every channel owns its own window, so the read of the window and its slide — the
 *   write to the state this step just read — are ordered inside one lane and no other lane touches them. */
static __device__ inline void nn__deltanet__zzabi_body_conv_step(uint16_t* o, uint16_t* s,
                                                                  const uint16_t* w, const uint16_t* x,
                                                                  uint64_t ch, unsigned int* over) {
    for (uint64_t c = nn__kernels__zzprivate_first(); c < ch; c += nn__kernels__zzprivate_stride()) {
        const uint16_t* wc = w + c * NN__DELTANET__CONV_TAPS;
        uint16_t*       sc = s + c * NN__DELTANET__CONV_WINDOW;
        const float xc = nn__silicon__half_to_float(x[c]);
        float acc = 0.0f;
        for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t)
            acc += nn__silicon__half_to_float(wc[t])
                 * nn__silicon__half_to_float(sc[t]);
        acc += nn__silicon__half_to_float(wc[NN__DELTANET__CONV_WINDOW]) * xc;
        bool hit = false;
        o[c] = nn__primitives__zzpackage_float_to_half(acc / (1.0f + nn__silicon__expf(-acc)), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
        for (unsigned t = 0u; t + 1u < NN__DELTANET__CONV_WINDOW; ++t) sc[t] = sc[t + 1u];
        sc[NN__DELTANET__CONV_WINDOW - 1u] = x[c];
    }
}

/* One value head's step: its state `Sh`, its `q`, `k`, `v` and `zh`, its gate and decay. A block runs it with every
 * lane, so each lane reaches both folds. ⛳ Shared by the one-position door and the T-position one, so the
 * two do the same arithmetic. */
static __device__ inline void nn__deltanet__zzprivate_step_head(float* Sh, const uint16_t* q, const uint16_t* k,
                                                                const uint16_t* v, const uint16_t* zh, float beta_h,
                                                                float g_h, const uint16_t* w, uint16_t* oh,
                                                                uint64_t d, bool* hit_out) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float qq = 0.0f, kk = 0.0f;
    for (uint64_t i = lane; i < d; i += lanes) {
        const float qi = nn__silicon__half_to_float(q[i]), ki = nn__silicon__half_to_float(k[i]);
        qq += qi * qi;
        kk += ki * ki;
    }
    const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)d);
    const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
    const float decay = nn__silicon__expf(g_h);
    const float b = beta_h;
    float oo = 0.0f;
    bool hit = false;
    for (uint64_t j = lane; j < d; j += lanes) {
        float kv = 0.0f;
        for (uint64_t i = 0ull; i < d; ++i) kv += Sh[i * d + j] * nn__silicon__half_to_float(k[i]);
        const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
        float o = 0.0f;
        for (uint64_t i = 0ull; i < d; ++i) {
            const float s = decay * Sh[i * d + j] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
            Sh[i * d + j] = s;
            o += s * nn__silicon__half_to_float(q[i]);
        }
        oh[j] = nn__primitives__zzpackage_float_to_half(o * inv_q, &hit);
        const float oj = nn__silicon__half_to_float(oh[j]);
        oo += oj * oj;
    }
    const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)d + 1e-6f);
    for (uint64_t j = lane; j < d; j += lanes) {
        const float zr = nn__silicon__half_to_float(zh[j]);
        const float zc = (zr < -88.0f) ? -88.0f : ((zr > 88.0f) ? 88.0f : zr);
        oh[j] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(oh[j]) * scale
                    * nn__silicon__half_to_float(w[j]) * (zc / (1.0f + nn__silicon__expf(-zc))), &hit);
    }
    if (hit) *hit_out = true;
}

/* ⭐⭐ ONE DELTANET STEP, EVERY HEAD, ONE LAUNCH — ⚖ *"i thought we would do a single opcode for the deltanet to
 *   reduce fragmentation"*. What the head loop did in eleven verbs a head, as one body: for value head `h`
 *   (its key head `h / (v_heads / k_heads)`), with `q`, `k`, `v` cut from the conv's output —
 *     kn = k / |k| · qs = q / |q| / √d        (l2norm, eps 1e-6 under the root, as `l2norm`)
 *     δ  = (v − e^g · Sᵀkn) · β               (the state read before its decay, then decayed: HF's order)
 *     S  ← e^g · S + kn δᵀ
 *     o  = rmsnorm(Sᵀqs) · w · silu(z)         (eps 1e-6 in the mean, silu clamped at ±88, as the verbs)
 *   ⛳ A BLOCK A HEAD AND A LANE A COLUMN `j` OF ITS STATE (`j = lane, lane + lanes, …`): the lane reads and
 *   writes its columns alone, so the update races with nothing, and `kn[i]`, `qs[i]` are formed from `k`, `q`
 *   as each is needed — their norms come from the fold, which hands every lane the sum — so nothing is
 *   staged between lanes and no barrier is needed outside the fold. The readout is written to `out` before
 *   the norm that needs all of it, and read back by the same lane to be normed and gated: a half between the
 *   two, as the readout verb rounded it; everything else stays fp32. */
static __device__ inline void nn__deltanet__zzabi_body_step(float* S, const uint16_t* conved, const uint16_t* z,
                                                             const uint16_t* beta, const uint16_t* g, const uint16_t* w,
                                                             uint16_t* out, uint64_t k_heads, uint64_t v_heads,
                                                             uint64_t head_dim, unsigned int* over) {
    const uint64_t d = head_dim, rep = v_heads / k_heads;
    bool hit = false;
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks())
        nn__deltanet__zzprivate_step_head(S + h * d * d, conved + (h / rep) * d, conved + k_heads * d + (h / rep) * d,
                                          conved + 2ull * k_heads * d + h * d, z + h * d, nn__silicon__half_to_float(beta[h]),
                                          nn__silicon__half_to_float(g[h]), w, out + h * d, d, &hit);
    if (hit) *over = NN__KERNELS__OVERFLOWED;
}

/* ── THE EXPERT MATVEC ───────────────────────────────────────────────────────────────────────────── */
/* ⭐⭐ WIDE: ONE BLOCK PER ROW, LANES ALONG THE ROW — ⚖ *"maintain the source locality and
 * scope the sizing against the holdable origin activation, so only the weights move"*. Neighbouring lanes
 * read neighbouring weights, so a row is read once and in order; `x`, which every block reads, stays where
 * it is. A row's answer is its lanes' parts added, and lane 0 writes it.
 * ⛳ THE ROW LOOP IS STRIDED BY BLOCKS, so fewer blocks than rows still covers every row, and the host's one
 * lane of one block is the serial loop this body was. Every lane of a block runs the same number of rows,
 * which is what lets the combine step wait for all of them. */
/* ⛳ HOW MUCH OF `x` A LANE HOLDS IN REGISTERS: its columns are the same for every row, so a lane reads its
 *   slice once and every row after that moves only weights. A row wider than `lanes` times this reads `x`
 *   from memory as it goes. */
#ifndef NN__KERNELS__X_HELD
#define NN__KERNELS__X_HELD 16u
#endif
/* ⛳ AND THE MOST BLOCKS A WIDE DOOR STARTS: few enough that each of the matvec's blocks serves many rows
 *   with its slice of `x`, enough to keep every compute unit busy. `MEASURED` on gfx906 at the 7B's gate/up
 *   shape, 60..3840 blocks: rising to a plateau at 960-1920. */
#ifndef NN__KERNELS__BLOCKS_MAX
#define NN__KERNELS__BLOCKS_MAX 960u
#endif

/* ⛳ EIGHT HALVES AT A TIME: 16 bytes in two 8-byte loads, so a lane reads a run of the row rather than
 *   one weight per load. Needs the run to start on a 16-byte boundary, which a row does when the matrix
 *   does and its width is a multiple of eight. */
#define NN__KERNELS__RUN 8u
static __device__ inline void nn__kernels__zzprivate_run(const uint16_t* at, float* into) {
    const uint64_t* words = (const uint64_t*)at;
    const uint64_t lo = words[0], hi = words[1];
    for (unsigned e = 0u; e < 4u; ++e) {
        into[e]      = nn__silicon__half_to_float((uint16_t)(lo >> (16u * e)));
        into[e + 4u] = nn__silicon__half_to_float((uint16_t)(hi >> (16u * e)));
    }
}

static __device__ inline void nn__expert__zzabi_body_multiply_fp16(uint16_t* out, const uint16_t* W,
                                                                    const uint16_t* x, uint64_t rows,
                                                                    uint64_t cols, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    /* ⛳ A LANE OWNS RUNS OF EIGHT COLUMNS — run `lane`, `lane + lanes`, … — the same runs on every row. */
    const bool runs = cols % NN__KERNELS__RUN == 0ull && ((uint64_t)(uintptr_t)W) % 16ull == 0ull
                   && ((uint64_t)(uintptr_t)x) % 16ull == 0ull;
    const uint64_t count = cols / NN__KERNELS__RUN, held_runs = NN__KERNELS__X_HELD / NN__KERNELS__RUN;
    const bool held = runs && count <= lanes * held_runs;
    float xs[NN__KERNELS__X_HELD];
    for (uint64_t k = 0ull; k < held_runs; ++k) {
        const uint64_t j = lane + k * lanes;
        if (held && j < count) nn__kernels__zzprivate_run(x + j * NN__KERNELS__RUN, xs + k * NN__KERNELS__RUN);
        else for (unsigned e = 0u; e < NN__KERNELS__RUN; ++e) xs[k * NN__KERNELS__RUN + e] = 0.0f;
    }
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        const uint16_t* row = W + r * cols;
        float part = 0.0f, w[NN__KERNELS__RUN], v[NN__KERNELS__RUN];
        if (held) {
            for (uint64_t k = 0ull; k < held_runs; ++k) {
                const uint64_t j = lane + k * lanes;
                if (j >= count) break;
                nn__kernels__zzprivate_run(row + j * NN__KERNELS__RUN, w);
                for (unsigned e = 0u; e < NN__KERNELS__RUN; ++e) part += w[e] * xs[k * NN__KERNELS__RUN + e];
            }
        } else if (runs) {
            for (uint64_t j = lane; j < count; j += lanes) {
                nn__kernels__zzprivate_run(row + j * NN__KERNELS__RUN, w);
                nn__kernels__zzprivate_run(x + j * NN__KERNELS__RUN, v);
                for (unsigned e = 0u; e < NN__KERNELS__RUN; ++e) part += w[e] * v[e];
            }
        } else {
            for (uint64_t c = lane; c < cols; c += lanes)
                part += nn__silicon__half_to_float(row[c])
                      * nn__silicon__half_to_float(x[c]);
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ── ROPE AND ATTENTION, ONE POSITION AT A TIME ─────────────────────────────────────────────────── */

/* ⭐ THE ROTATION OF EVERY HEAD AT ONE POSITION, WIDE: item `i` is one pair — head `i / (d/2)`, and its
 *   element `j` with its partner `j + d/2` — so each item reads its two and writes its two, and `o` may be
 *   `x`. `cs` is the position's `d/2` cosines then its `d/2` sines, in fp32. It is HF's `rotate_half`:
 *   `x·cos + rotate_half(x)·sin`, with the same cosine for both halves of a head. */
static __device__ inline void nn__rope__zzabi_body(uint16_t* o, const uint16_t* x, const float* cs,
                                                    uint64_t heads, uint64_t head_dim, unsigned int* over) {
    const uint64_t pairs = head_dim / 2ull, n = heads * pairs;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t at = (i / pairs) * head_dim + i % pairs;
        const float a = nn__silicon__half_to_float(x[at]), b = nn__silicon__half_to_float(x[at + pairs]);
        const float c = cs[i % pairs], s = cs[pairs + i % pairs];
        bool hit = false;
        o[at] = nn__primitives__zzpackage_float_to_half(a * c - b * s, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
        hit = false;
        o[at + pairs] = nn__primitives__zzpackage_float_to_half(b * c + a * s, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⭐ THE ANGLES OF ONE POSITION, WIDE: item `i` is pair `i` of a head, and `cs` gets its cosine at `i`
 *   and its sine at `d/2 + i`. HF's rotary in fp32: the frequency `theta^(-2i/d)`, the angle
 *   `position * frequency`.
 * ⛳ THE FREQUENCY IS `r^i` WITH `r = theta^(-2/d)`, MULTIPLIED UP, and not `exp(-2i/d · log theta)`. The
 *   angle is the position times the frequency, so every ulp the frequency is off is multiplied by the
 *   position: `log theta` is about 13.8 for Qwen's base, its rounding times `2i/d` near 1 is ~7 ulps, and
 *   at position 100,000 that was 0.07 rad on the fastest pairs (`MEASURED` against the angle in
 *   double). `r`'s exponent is small, so `r` is good to about an ulp; each multiply adds half an ulp, but
 *   the angle shrinks as `r^i` while that error grows as `i`, and the product peaks near `i = 5` at about
 *   HF's own fp32 error. */
static __device__ inline void nn__rope__zzabi_body_angles(float* cs, uint64_t position, float theta,
                                                           uint64_t head_dim) {
    const uint64_t pairs = head_dim / 2ull;
    /* ⛔ THE WIDTH IS CONVERTED FROM 32 BITS, AND ON PURPOSE: the MI50's OpenCL compiler stops with
     *   "unsupported libcall legalization" on a 64-bit integer made a float in this expression (`MEASURED`
     * the program does not build). The width is at most 2^20, so 32 bits hold it exactly. */
    const float r = nn__silicon__expf(-2.0f / (float)(uint32_t)head_dim * nn__silicon__logf(theta));
    for (uint64_t i = nn__kernels__zzprivate_first(); i < pairs; i += nn__kernels__zzprivate_stride()) {
        float frequency = 1.0f;
        for (uint64_t k = 0ull; k < i; ++k) frequency *= r;
        const float angle = (float)position * frequency;
        cs[i] = nn__silicon__cosf(angle);
        cs[pairs + i] = nn__silicon__sinf(angle);
    }
}

/* ⭐⭐ ATTENTION FOR ONE QUERY OVER `length` POSITIONS, IN TWO BODIES, AND THE LAUNCH BETWEEN THEM IS THE
 *   POINT. The weights of every position are needed by every output element, and a body has no way to
 *   hand one lane's memory writes to another; two launches in order do. So the first body writes each
 *   head's weights into `p`, and the second reads them.
 * ⛳ THE CACHE IS `[position][kv_head][head_dim]`, and query head `h` reads kv head `h / (q_heads/kv_heads)`
 *   — the grouped-query sharing. One block per query head; the head loop is the block's, so every lane of
 *   a block calls the combine step together. */
/* One query head's weights over `length` positions, into `ph`: its scores, their largest, and the softmax. Every
 * lane of the block runs it, and reaches its two folds. */
static __device__ inline void nn__attention__zzprivate_scores_head(float* ph, const uint16_t* qh, const uint16_t* kh,
                                                                   uint64_t row, uint64_t head_dim, uint64_t length,
                                                                   float scale) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    /* each lane scores its own positions, and keeps them in `p`, where only it reads them again */
    float top = -NN__KERNELS__INFINITY;
    for (uint64_t t = lane; t < length; t += lanes) {
        float dot = 0.0f;
        for (uint64_t e = 0ull; e < head_dim; ++e)
            dot += nn__silicon__half_to_float(qh[e]) * nn__silicon__half_to_float(kh[t * row + e]);
        ph[t] = dot * scale;
        if (ph[t] > top) top = ph[t];
    }
    top = nn__silicon__lanes_max(top);
    float part = 0.0f;
    for (uint64_t t = lane; t < length; t += lanes) {
        ph[t] = nn__silicon__expf(ph[t] - top);
        part += ph[t];
    }
    const float inv = 1.0f / nn__silicon__lanes_sum(part);
    for (uint64_t t = lane; t < length; t += lanes) ph[t] *= inv;
}

static __device__ inline void nn__attention__zzabi_body_scores(float* p, const uint16_t* q, const uint16_t* k,
                                                                uint64_t q_heads, uint64_t kv_heads,
                                                                uint64_t head_dim, uint64_t length) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks())
        nn__attention__zzprivate_scores_head(p + h * length, q + h * head_dim, k + (h / group) * head_dim, row, head_dim,
                                             length, scale);
}

/* One query head's answer: each of its elements the weighted sum of the values, a lane an element. */
static __device__ inline void nn__attention__zzprivate_mix_head(uint16_t* oh, const float* ph, const uint16_t* vh,
                                                                uint64_t row, uint64_t head_dim, uint64_t length,
                                                                unsigned int* over) {
    for (uint64_t e = nn__silicon__lane(); e < head_dim; e += nn__silicon__lanes()) {
        float sum = 0.0f;
        for (uint64_t t = 0ull; t < length; ++t) sum += ph[t] * nn__silicon__half_to_float(vh[t * row + e]);
        bool hit = false;
        oh[e] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__attention__zzabi_body_mix(uint16_t* o, const float* p, const uint16_t* v,
                                                             uint64_t q_heads, uint64_t kv_heads,
                                                             uint64_t head_dim, uint64_t length,
                                                             unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks())
        nn__attention__zzprivate_mix_head(o + h * head_dim, p + h * length, v + (h / group) * head_dim, row, head_dim,
                                          length, over);
}

/* ⭐⭐ RESIDUALS — ATTENTION OVER SEVERAL RANGES, ONE SOFTMAX. ⚖ *"in the future we can add residuals"*.
 *   A range's residual is, per query head, `m` its largest score, `l` the sum of `e^(score - m)` and `acc`
 *   the sum of `e^(score - m) · v` — nothing divided yet. Two residuals merge exactly: scaled by
 *   `e^(m_a - m)` and `e^(m_b - m)` to their common largest `m`, their `l` and `acc` add. The answer is
 *   `acc / l` once every range is in, and it is the same softmax as one call over all of them.
 * ⛳ THE LAYOUT, PER HEAD `h`: `res[h·(d+2)]` is `m`, the next `l`, then `d` floats of `acc`. */
#define NN__ATTENTION__RESIDUAL_FLOATS(head_dim)  ((head_dim) + 2ull)

/* The first pass of a residual: each head's scores, `m` and `l` into `res`, and the unnormalised weights
 * `e^(score - m)` into `p` for the second. One block per query head, as for the one-call form. */
static __device__ inline void nn__attention__zzabi_body_weights(float* p, float* res, const uint16_t* q,
                                                                 const uint16_t* k, uint64_t q_heads,
                                                                 uint64_t kv_heads, uint64_t head_dim,
                                                                 uint64_t length) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    const float scale = 1.0f / nn__silicon__sqrtf((float)(uint32_t)head_dim);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint16_t* qh = q + h * head_dim;
        const uint16_t* kh = k + (h / group) * head_dim;
        float* ph = p + h * length;
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t t = lane; t < length; t += lanes) {
            float dot = 0.0f;
            for (uint64_t e = 0ull; e < head_dim; ++e)
                dot += nn__silicon__half_to_float(qh[e]) * nn__silicon__half_to_float(kh[t * row + e]);
            ph[t] = dot * scale;
            if (ph[t] > top) top = ph[t];
        }
        top = nn__silicon__lanes_max(top);
        float part = 0.0f;
        for (uint64_t t = lane; t < length; t += lanes) {
            ph[t] = nn__silicon__expf(ph[t] - top);
            part += ph[t];
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (lane == 0ull) {
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 1ull] = sum;
        }
    }
}

/* The second pass: `acc`, the weights times `v`, into `res`. A lane owns one element of a head. */
static __device__ inline void nn__attention__zzabi_body_residual_mix(float* res, const float* p, const uint16_t* v,
                                                                      uint64_t q_heads, uint64_t kv_heads,
                                                                      uint64_t head_dim, uint64_t length) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        const uint16_t* vh = v + (h / group) * head_dim;
        float* acc = res + h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 2ull;
        for (uint64_t e = nn__silicon__lane(); e < head_dim; e += nn__silicon__lanes()) {
            float sum = 0.0f;
            for (uint64_t t = 0ull; t < length; ++t) sum += ph[t] * nn__silicon__half_to_float(vh[t * row + e]);
            acc[e] = sum;
        }
    }
}

/* Two residuals into one. ⛳ AN ITEM IS A WHOLE HEAD, so `into` may be `a` or `b`: an item reads its
 * head's two `m` and `l` before it writes anything, and nothing else touches that head. The work is a
 * head's `d + 2` floats, which is too little to be worth dividing further. */
static __device__ inline void nn__attention__zzabi_body_merge(float* into, const float* a, const float* b,
                                                               uint64_t q_heads, uint64_t head_dim) {
    const uint64_t w = NN__ATTENTION__RESIDUAL_FLOATS(head_dim);
    for (uint64_t h = nn__kernels__zzprivate_first(); h < q_heads; h += nn__kernels__zzprivate_stride()) {
        const float ma = a[h * w], la = a[h * w + 1ull], mb = b[h * w], lb = b[h * w + 1ull];
        const float m = ma > mb ? ma : mb;
        const float fa = nn__silicon__expf(ma - m), fb = nn__silicon__expf(mb - m);
        for (uint64_t e = 0ull; e < head_dim; ++e)
            into[h * w + 2ull + e] = a[h * w + 2ull + e] * fa + b[h * w + 2ull + e] * fb;
        into[h * w] = m;
        into[h * w + 1ull] = la * fa + lb * fb;
    }
}

/* The answer: `acc / l`, each head's element, as a half. */
static __device__ inline void nn__attention__zzabi_body_finish(uint16_t* o, const float* res, uint64_t q_heads,
                                                                uint64_t head_dim, unsigned int* over) {
    const uint64_t w = NN__ATTENTION__RESIDUAL_FLOATS(head_dim), n = q_heads * head_dim;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t h = i / head_dim;
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half(res[h * w + 2ull + i % head_dim] / res[h * w + 1ull], &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── TURBOQUANT — AND THESE TWO ARE THE ONLY ONES THAT DO NOT SPLIT CLEANLY ───────────────────────
 * ⛔⛔ THEIR ROW LOOP CALLS `sys__opcodes__fails(form, …)` IN THE MIDDLE OF THE COMPUTE: a row whose
 * codebook does not resolve refuses on the spot. That is `sys` INSIDE the seam, and a kernel has no
 * form to fail into.
 * ⇒ ⛳ SO THE BODY ANSWERS `false` INSTEAD OF REFUSING, and the verb turns that into `fails()` with
 *   the width's code. The arithmetic is all here; the one thing the body does not do is REPORT.
 * ⇒ ★ A SEAM IS ONLY CLEAN WHERE THE ERROR PATH IS TOO — and the error path is the last thing anybody
 *   looks at. */

/* ⭐ WIDE, POINTWISE: item `i` is weight `(i / cols, i % cols)`, decoded where it lies. A row whose room
 *   does not hold it is skipped — the verb has already refused a matrix with such a row, on the CPU,
 *   before launching, so this is the second guard and not the first. */
static __device__ inline bool nn__turboquant__zzabi_body_decode(uint16_t* out, const uint8_t* weights,
                                                                 uint64_t w_room, const uint8_t* luts,
                                                                 uint64_t l_room, uint64_t d,
                                                                 uint64_t rows, uint64_t cols,
                                                                 unsigned int* over) {
    const uint64_t n = rows * cols;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t r = i / cols, c = i - r * cols;
        const uint8_t *row_base = 0, *lut_base = 0;
        if (!nn__turboquant__zzpackage_row(weights, w_room, luts, l_room, d, r, cols, &row_base, &lut_base))
            continue;
        bool hit = false;
        out[i] = nn__primitives__zzpackage_float_to_half(
                     nn__turboquant__zzpackage_weight(row_base, d, lut_base, c), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
    return true;
}

/* ⭐⭐ WIDE, THE SHAPE OF THE fp16 MATVEC: one block per row, blocks striding the rows, lanes along the row,
 *   the block's combine step summing the lanes' parts and lane 0 writing the row. The matrix is never
 *   expanded: a code is decoded into a register and multiplied by its `x` in the same expression.
 * ⛳ A LANE TAKES THE ROW EIGHT BYTES AT A TIME — four containers, run `lane`, `lane + lanes`, … — when a
 *   row starts on an eight-byte boundary, which it does when the matrix does and a row is a multiple of
 *   eight bytes; otherwise two bytes at a time. And it holds the `x` of its first runs in registers,
 *   because its columns are the same on every row: after the first row only weights move.
 * ⛳ A ROW THAT FAILS ITS BOUND IS SKIPPED BY THE WHOLE BLOCK, before the combine step, so every lane
 *   still reaches it together; the verb refused such a matrix on the CPU before launching. */
/* ⛳ THE ROW LOOP, CALLED WITH ITS WIDTH AS A LITERAL, SO THE WIDTH IS A CONSTANT: every inner loop unrolls,
 *   every shift is fixed, and every index into the held `x` is known when the kernel is compiled.
 *   `MEASURED` on the 1024 x 2048 expert at 4 bits: with the width a variable the eight-byte
 *   loop was SLOWER than the two-byte one, 53 -> 131 us — a division a code, and an array indexed by a
 *   variable, which the compiler moves to memory.
 * ⛳ A RUN IS EIGHT BYTES — four containers — AND A LANE TAKES RUN `lane`, `lane + lanes`, … of every row.
 * ⛳ ITS `x` IS HELD AS HALVES, TWO TO A REGISTER: sixteen registers hold 32 codes' worth, the first
 *   `32 / codes-a-run` runs, which covers a whole row at the 35B's hidden width at every width it uses.
 *   Its columns are the same on every row, so after the first row only weights move; a run past the
 *   held ones reads its `x` from memory. Registers are the price: `MEASURED`, the kernel at 85 VGPRs
 *   (two float arrays held) ran 551 us at 18944 x 3584 x 4 bits, at 54 (one) 389 us. */
#define NN__TURBOQUANT__HELD_WORDS 16u
/* ⭐⭐ WIDE, THE SHAPE OF THE fp16 MATVEC: one block per row, blocks striding the rows, lanes along the row,
 *   the block's combine step summing the lanes' parts and lane 0 writing the row. The matrix is never
 *   expanded: a code is decoded into a register and multiplied by its `x` in the same expression.
 * ⛳ A LANE TAKES THE ROW EIGHT BYTES AT A TIME — four containers, run `lane`, `lane + lanes`, … — when a
 *   row starts on an eight-byte boundary, which it does when the matrix does and a row is a multiple of
 *   eight bytes; otherwise two bytes at a time. And it holds the `x` of its first runs in registers,
 *   because its columns are the same on every row: after the first row only weights move.
 * ⛳ A ROW THAT FAILS ITS BOUND IS SKIPPED BY THE WHOLE BLOCK, before the combine step, so every lane
 *   still reaches it together; the verb refused such a matrix on the CPU before launching. */
/* ⛳ THE ROW LOOP, CALLED WITH ITS WIDTH AS A LITERAL, SO THE WIDTH IS A CONSTANT: every inner loop unrolls,
 *   every shift is fixed, and every index into the held `x` is known when the kernel is compiled.
 *   `MEASURED` on the 1024 x 2048 expert at 4 bits: with the width a variable the eight-byte
 *   loop was SLOWER than the two-byte one, 53 -> 131 us — a division a code, and an array indexed by a
 *   variable, which the compiler moves to memory.
 * ⛳ A RUN IS EIGHT BYTES — four containers — AND A LANE TAKES RUN `lane`, `lane + lanes`, … of every row.
 * ⛳ ITS `x` IS HELD AS HALVES, TWO TO A REGISTER: sixteen registers hold 32 codes' worth, the first
 *   `32 / codes-a-run` runs, which covers a whole row at the 35B's hidden width at every width it uses.
 *   Its columns are the same on every row, so after the first row only weights move; a run past the
 *   held ones reads its `x` from memory. Registers are the price: `MEASURED`, the kernel at 85 VGPRs
 *   (two float arrays held) ran 551 us at 18944 x 3584 x 4 bits, at 54 (one) 389 us. */
#define NN__TURBOQUANT__HELD_WORDS 16u
static __device__ inline void nn__turboquant__zzprivate_rows(const uint32_t D, uint16_t* out, const uint8_t* weights,
        const uint8_t* luts, const uint16_t* x, uint64_t rows, uint64_t cols, uint64_t rows_held,
        uint64_t scales_held, unsigned int* over) {
    const uint32_t per = 16u / D, run_codes = 4u * per, held = 2u * NN__TURBOQUANT__HELD_WORDS / run_codes;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t row_bytes = ((cols + per - 1ull) / per) * 2ull, runs = row_bytes / 8ull;
    uint32_t xh[NN__TURBOQUANT__HELD_WORDS];
    for (uint32_t w = 0u; w < NN__TURBOQUANT__HELD_WORDS; ++w) {
        uint32_t pair = 0u;
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t i = 2u * w + h, k = i / run_codes, e = i % run_codes;
            const uint64_t c = (lane + k * lanes) * run_codes + e;
            if (k < held && c < cols) pair |= (uint32_t)x[c] << (16u * h);
        }
        xh[w] = pair;
    }
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        if (r >= rows_held || (D != 16u && r >= scales_held)) continue;
        const uint64_t* words = (const uint64_t*)(weights + r * row_bytes);
        float part = 0.0f;
        for (uint32_t k = 0u; k < held; ++k) {
            const uint64_t j = lane + k * lanes;
            if (j >= runs) break;
            const uint64_t word = words[j];
            for (uint32_t e = 0u; e < run_codes; ++e) {
                const uint32_t i = k * run_codes + e;
                const uint32_t code = (uint32_t)(word >> (16u * (e / per) + D * (e % per))) & ((1u << D) - 1u);
                part += nn__turboquant__zzpackage_level(D, code)
                      * nn__silicon__half_to_float((uint16_t)(xh[i / 2u] >> (16u * (i % 2u))));
            }
        }
        for (uint64_t j = lane + (uint64_t)held * lanes; j < runs; j += lanes) {
            const uint64_t word = words[j];
            for (uint32_t e = 0u; e < run_codes; ++e) {
                const uint64_t c = j * run_codes + e;
                const uint32_t code = (uint32_t)(word >> (16u * (e / per) + D * (e % per))) & ((1u << D) - 1u);
                if (c < cols) part += nn__turboquant__zzpackage_level(D, code) * nn__silicon__half_to_float(x[c]);
            }
        }
        const float scale = D == 16u ? 1.0f : nn__silicon__half_to_float(
                                (uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
        const float sum = nn__silicon__lanes_sum(part) * scale;
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ⭐⭐ WIDE, THE SHAPE OF THE fp16 MATVEC: one block per row, blocks striding the rows, lanes along the row,
 *   the block's combine step summing the lanes' parts and lane 0 writing the row. The matrix is never
 *   expanded: a code is decoded into a register and multiplied by its `x` in the same expression.
 * ⛳ AT FOUR BITS AND UP, WITH EIGHT-BYTE ROWS, THE WIDTH'S OWN LOOP ABOVE; otherwise two bytes at a time.
 * ⛳ THE ROW BOUND IS SETTLED ONCE, NOT ONCE A ROW: how many rows the room holds is one division, and a
 *   row is then a multiply — a 64-bit division is a routine of hundreds of instructions on this card.
 * ⛳ A ROW PAST THE BOUND IS SKIPPED BY THE WHOLE BLOCK, before the combine step, so every lane still
 *   reaches it together; the verb refused such a matrix on the CPU before launching. */
static __device__ inline bool nn__turboquant__zzabi_body_gemv(uint16_t* out, const uint8_t* weights,
                                                               uint64_t w_room, const uint8_t* luts,
                                                               uint64_t l_room, const uint16_t* x,
                                                               uint64_t d, uint64_t rows,
                                                               uint64_t cols, unsigned int* over) {
    const uint64_t per = nn__turboquant__values_per_container(d), row_bytes = nn__turboquant__row_bytes(d, cols);
    const uint64_t rows_held = row_bytes == 0ull ? 0ull : w_room / row_bytes, scales_held = l_room / 2ull;
    const bool eights = row_bytes % 8ull == 0ull && ((uint64_t)(uintptr_t)weights) % 8ull == 0ull;
    if (eights && (d == 4ull || d == 5ull || d == 8ull || d == 16ull)) {
        /* each call names its width as a literal, and the compiler folds it into its own copy of the loop */
        if (d == 4ull)       nn__turboquant__zzprivate_rows(4u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
        else if (d == 5ull)  nn__turboquant__zzprivate_rows(5u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
        else if (d == 8ull)  nn__turboquant__zzprivate_rows(8u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
        else                 nn__turboquant__zzprivate_rows(16u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
        return true;
    }
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), units = (cols + per - 1ull) / per;
    const uint32_t mask = (uint32_t)((1ull << d) - 1ull);
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        if (r >= rows_held || (d != 16ull && r >= scales_held)) continue;
        const uint8_t* row_base = weights + r * row_bytes;
        float part = 0.0f;
        for (uint64_t u = lane; u < units; u += lanes) {
            const uint32_t unit = (uint32_t)row_base[u * 2ull] | ((uint32_t)row_base[u * 2ull + 1ull] << 8);
            const uint64_t c0 = u * per;
            for (uint64_t s = 0ull; s < per && c0 + s < cols; ++s)
                part += nn__turboquant__zzpackage_level(d, (unit >> (uint32_t)(d * s)) & mask)
                      * nn__silicon__half_to_float(x[c0 + s]);
        }
        const float scale = d == 16ull ? 1.0f : nn__silicon__half_to_float(
                                (uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
        const float sum = nn__silicon__lanes_sum(part) * scale;
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
    return true;
}

/* ⭐⭐ THE GROUPED GEMVS — several matrices in ONE launch, which is what a model's MoE block needs: eight
 * experts and a shared one read the same `x` (their gate_ups), or each its own `x` into one weighted sum (their
 * downs). A launch costs this card ~3.5 us however little it does (`MEASURED`), so eleven gemvs
 * as one saves ten of those and lets the small matrices share the card instead of taking turns.
 * ⛳ THE SHARED-`x` ONE IS THE GEMV ONCE A GROUP, in the same launch: the groups write disjoint outputs and
 *   only read `x`, so no block waits on another between them, and every block makes the same calls in the
 *   same order — which is what the gemv's fold asks of the lanes of a block. */
static __device__ inline void nn__turboquant__zzabi_body_gemv_groups(uint16_t* out, nn__turboquant__groups g,
                                                                     const uint16_t* x, uint64_t cols, unsigned int* over) {
    for (uint64_t i = 0ull; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
        const uint64_t row_bytes = nn__turboquant__row_bytes(g.d[i], cols);
        (void)nn__turboquant__zzabi_body_gemv(out + g.out_at[i], (const uint8_t*)(uintptr_t)g.codes[i], g.rows[i] * row_bytes,
                                              (const uint8_t*)(uintptr_t)g.luts[i], g.rows[i] * 2ull, x, g.d[i], g.rows[i],
                                              cols, over);
    }
}

/* ⭐ AND THE WEIGHTED SUM: `out[r] = residual[r] + Σ_i w[out_at[i]] · (W_i[r] · x_i)`, `x_i` at `x + i·cols` — a block a
 *   row, every group's part for that row added before the one fold, so the sum needs no second pass and no
 *   atomics. `residual` may be `out` itself: a row reads its own element before writing it, and no other row
 *   touches it. Every group has the given `rows`; each its own width. */
static __device__ inline void nn__turboquant__zzabi_body_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g,
                                                                         const uint16_t* x, const uint16_t* w,
                                                                         const uint16_t* residual, uint64_t rows,
                                                                         uint64_t cols, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        float part = 0.0f;
        for (uint64_t i = 0ull; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
            const uint64_t d = g.d[i], per = nn__turboquant__values_per_container(d);
            const uint64_t units = (cols + per - 1ull) / per;
            const uint32_t mask = (uint32_t)((1ull << d) - 1ull);
            const uint8_t* row_base = (const uint8_t*)(uintptr_t)g.codes[i] + r * nn__turboquant__row_bytes(d, cols);
            const uint16_t* xi = x + i * cols;
            float mine = 0.0f;
            for (uint64_t u = lane; u < units; u += lanes) {
                const uint32_t unit = (uint32_t)row_base[u * 2ull] | ((uint32_t)row_base[u * 2ull + 1ull] << 8);
                const uint64_t c0 = u * per;
                for (uint64_t s = 0ull; s < per && c0 + s < cols; ++s)
                    mine += nn__turboquant__zzpackage_level(d, (unit >> (uint32_t)(d * s)) & mask)
                          * nn__silicon__half_to_float(xi[c0 + s]);
            }
            const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
            const float scale = d == 16ull ? 1.0f
                              : nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8)));
            part += mine * scale * nn__silicon__half_to_float(w[g.out_at[i]]);
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(residual[r]) + sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ══ ⭐⭐⭐ THE EXPERT-MAJOR GEMM — AN EXPERT APPLIED TO A SET OF ROWS ══════════════════════════════════
 *
 * ⚖ *"we need expert major verbs"*. Every matrix is an expert — a slot of a collection, dense or routed — and
 * the unit of work is one applied to several rows of `x` at once, so each weight is read once for all of
 * them: a dense projection over a batch is an expert whose rows are all of it, a routed expert's are the
 * rows that picked it.
 * ⭐ `rows` IS A LIST OF PAIRS, `count` of them: pair `j` is `rows[2j]`, the row of `x` it reads (`cols` wide),
 *   and `rows[2j + 1]`, the row of `out` it writes (`out_rows` wide). The two differ where a batch's rows
 *   fan out — a row's eight picks each get their own row of gate_up — and fan back in: a pick's down result
 *   lands on the row that made it.
 * ⭐⭐ TILED BY THE SILICON'S OWN NUMBERS (`nn__silicon__gemm_tile`): a block takes `w_rows` rows of the matrix
 *   and walks the pairs `x_rows` at a time; each lane decodes one code of each of its rows and multiplies it
 *   into all `x_rows` rows of `x` before decoding the next. The lanes' parts are then folded, one fold per
 *   (row, pair) of the tile, which every lane of the block reaches together because the tile loops are the
 *   block's, not the lane's — the tile's sums folded together, `lanes_sums`, so the block waits once a tile.
 * ⛳ A ROW PAST WHAT `w_room` AND `l_room` HOLD IS COMPUTED AND NOT WRITTEN, so every lane still reaches every
 *   fold; the caller refuses such a matrix before launching, as it does for the gemv.
 * ⛳ WITH ONE LANE, EACH SUM RUNS OVER THE COLUMNS IN ORDER, as the gemv's does, so on the host the two agree
 *   to the bit. */
static __device__ inline const uint16_t* nn__turboquant__zzabi_levels(uint64_t d);   /* a width's levels — below */
static __device__ inline void nn__expert__zzprivate_rows(uint16_t* out, float* acc, const uint8_t* weights,
                                                         uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                                         const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                         uint64_t count, uint64_t d, uint64_t out_rows,
                                                         uint64_t cols, unsigned int* over) {
    const nn__gemm__tile tile = nn__silicon__gemm_tile();
    const uint32_t WR = tile.w_rows, XR = tile.x_rows;
    const uint64_t per = nn__turboquant__values_per_container(d), row_bytes = nn__turboquant__row_bytes(d, cols);
    const uint64_t units = (cols + per - 1ull) / per, rows_held = row_bytes == 0ull ? 0ull : w_room / row_bytes;
    const uint64_t held = d == 16ull || rows_held < l_room / 2ull ? rows_held : l_room / 2ull;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint32_t mask = (uint32_t)((1ull << d) - 1ull);
    /* ⭐ IN PAIRS where a container holds an even number of codes and a row of `x` an even number of columns — 4 and
     *   8 bits, every matrix of the models shipped — so a pair never straddles a container or a row's end, and `x`'s
     *   pair is one aligned word. `nn__silicon__dot2`; the codes' levels are the table's halves, as they are. */
    const bool pairs = d != 16ull && per % 2ull == 0ull && cols % 2ull == 0ull;
    const uint16_t* levels = d == 16ull ? 0 : nn__turboquant__zzabi_levels(d);
    for (uint64_t r0 = (uint64_t)nn__silicon__block() * WR; r0 < out_rows; r0 += (uint64_t)nn__silicon__blocks() * WR) {
        const uint8_t* base[NN__GEMM__W_ROWS_MAX];
        for (uint32_t a = 0u; a < WR; ++a)
            base[a] = r0 + a < held && r0 + a < out_rows ? weights + (r0 + a) * row_bytes : 0;
        for (uint64_t j0 = 0ull; j0 < count; j0 += XR) {
            const uint16_t* xr[NN__GEMM__X_ROWS_MAX];
            for (uint32_t b = 0u; b < XR; ++b)
                xr[b] = j0 + b < count ? x + (uint64_t)rows[2ull * (j0 + b)] * cols : 0;
            float part[NN__GEMM__W_ROWS_MAX][NN__GEMM__X_ROWS_MAX];
            for (uint32_t a = 0u; a < WR; ++a)
                for (uint32_t b = 0u; b < XR; ++b) part[a][b] = 0.0f;
            for (uint64_t u = lane; u < units; u += lanes) {
                uint32_t unit[NN__GEMM__W_ROWS_MAX];
                for (uint32_t a = 0u; a < WR; ++a)
                    unit[a] = base[a] ? (uint32_t)base[a][u * 2ull] | ((uint32_t)base[a][u * 2ull + 1ull] << 8) : 0u;
                const uint64_t c0 = u * per;
                if (pairs) {
                    /* two codes at a time: their levels as two halves, `x`'s two columns as one word, one dot */
                    for (uint64_t s = 0ull; s < per && c0 + s < cols; s += 2ull) {
                        uint32_t xv2[NN__GEMM__X_ROWS_MAX];
                        for (uint32_t b = 0u; b < XR; ++b)
                            xv2[b] = xr[b] ? *(const uint32_t*)(const void*)(xr[b] + c0 + s) : 0u;
                        for (uint32_t a = 0u; a < WR; ++a) {
                            const uint32_t w2 = (uint32_t)levels[(unit[a] >> (uint32_t)(d * s)) & mask]
                                              | ((uint32_t)levels[(unit[a] >> (uint32_t)(d * (s + 1ull))) & mask] << 16);
                            for (uint32_t b = 0u; b < XR; ++b) part[a][b] = nn__silicon__dot2(w2, xv2[b], part[a][b]);
                        }
                    }
                    continue;
                }
                for (uint64_t s = 0ull; s < per && c0 + s < cols; ++s) {
                    float xv[NN__GEMM__X_ROWS_MAX];
                    for (uint32_t b = 0u; b < XR; ++b)
                        xv[b] = xr[b] ? nn__silicon__half_to_float(xr[b][c0 + s]) : 0.0f;
                    for (uint32_t a = 0u; a < WR; ++a) {
                        const float level = nn__turboquant__zzpackage_level(d, (unit[a] >> (uint32_t)(d * s)) & mask);
                        for (uint32_t b = 0u; b < XR; ++b) part[a][b] += level * xv[b];
                    }
                }
            }
            float sums[NN__GEMM__W_ROWS_MAX * NN__GEMM__X_ROWS_MAX];
            for (uint32_t a = 0u; a < WR; ++a)
                for (uint32_t b = 0u; b < XR; ++b) sums[a * XR + b] = part[a][b];
            nn__silicon__lanes_sums(sums, WR * XR);
            for (uint32_t a = 0u; a < WR; ++a)
                for (uint32_t b = 0u; b < XR; ++b) {
                    const float sum = sums[a * XR + b];
                    const uint64_t r = r0 + a, j = j0 + b;
                    if (lane != 0ull || !base[a] || j >= count) continue;
                    const float scale = d == 16ull ? 1.0f : nn__silicon__half_to_float(
                                            (uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
                    const uint64_t at = (uint64_t)rows[2ull * j + 1ull] * out_rows + r;
                    if (out) {
                        bool hit = false;
                        out[at] = nn__primitives__zzpackage_float_to_half(sum * scale, &hit);
                        if (hit) *over = NN__KERNELS__OVERFLOWED;
                    } else {
                        acc[at] += sum * scale * nn__silicon__half_to_float(w[rows[2ull * j]]);
                    }
                }
        }
    }
}

/* `out[pair's out row][r] = W[r] · x[pair's x row]`, rounded to a half, for every pair. */
static __device__ inline void nn__expert__zzabi_body_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room,
                                                          const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                          const uint32_t* rows, uint64_t count, uint64_t d,
                                                          uint64_t out_rows, uint64_t cols, unsigned int* over) {
    nn__expert__zzprivate_rows(out, 0, weights, w_room, luts, l_room, x, rows, 0, count, d, out_rows, cols, over);
}

/* ⭐ AND WEIGHTED ONTO A RUNNING SUM: `acc[pair's out row][r] += w[pair's x row] · (W[r] · x[pair's x row])`, in
 *   fp32 — the MoE's down, one expert a launch, each pick's weight found at the pick's own row. The sum stays
 *   fp32 across every expert of the batch and is rounded once, by `rows_finish`, as the one-row sum door
 *   rounds once. ⛳ ONE BLOCK OWNS A MATRIX ROW, and its first lane writes every pair's element of it in turn,
 *   so two pairs naming the same out row do not race. */
static __device__ inline void nn__expert__zzabi_body_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room,
                                                              const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                              const uint32_t* rows, const uint16_t* w, uint64_t count,
                                                              uint64_t d, uint64_t out_rows, uint64_t cols) {
    nn__expert__zzprivate_rows(0, acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, 0);
}

/* ⭐ SEVERAL EXPERTS, EACH OVER ITS OWN PAIRS, IN ONE LAUNCH — the batch's gate_ups, or a layer's q, k and v.
 *   Each group is the body above in turn; the groups write disjoint outputs and only read `x`, so every
 *   block makes the same calls in the same order and no block waits on another between them. */
static __device__ inline void nn__expert__zzabi_body_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x,
                                                            const uint32_t* rows, uint64_t cols, unsigned int* over) {
    for (uint64_t i = 0ull; i < g.count && i < NN__EXPERT__GROUPS_MAX; ++i) {
        const uint64_t row_bytes = nn__turboquant__row_bytes(g.d[i], cols);
        nn__expert__zzprivate_rows(out + g.out_at[i], 0, (const uint8_t*)(uintptr_t)g.codes[i], g.out_rows[i] * row_bytes,
                                   (const uint8_t*)(uintptr_t)g.luts[i], g.out_rows[i] * 2ull, x, rows + 2ull * g.pairs_at[i],
                                   0, g.pairs[i], g.d[i], g.out_rows[i], cols, over);
    }
}

/* ⭐ AND THE ROUNDING: `out[i] = residual[i] + acc[i]` as a half, and `acc[i]` back to zero for the next batch's
 *   sum. `residual` may be `out`: each element reads its own before writing it. */
static __device__ inline void nn__expert__zzabi_body_rows_finish(uint16_t* out, float* acc, const uint16_t* residual,
                                                                 uint64_t n, unsigned int* over) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        bool hit = false;
        out[i] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(residual[i]) + acc[i], &hit);
        acc[i] = 0.0f;
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ══ ⭐⭐ A LAYER'S OTHER STEPS OVER `rows` POSITIONS AT ONCE — WHAT A PROMPT NEEDS BESIDE THE GEMM ══════════
 *
 * A prompt of `rows` positions goes through a layer as matrices, a row a position, and the steps that are not
 * a projection become these: each the one-position body run once a row inside one launch, so a row's answer
 * is what the one-position door gives for that position, bit for bit, and the check is that door called once
 * a row. Rows are `x_stride` / `o_stride` elements apart where a caller's layout needs it. */

/* Each row normed: a block a row, as the one-row door is one block. */
static __device__ inline void nn__rmsnorm__zzabi_body_rows(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n,
                                                           uint64_t rows, uint64_t x_stride, uint64_t o_stride,
                                                           unsigned int* over) {
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks())
        nn__rmsnorm__zzabi_body(o + r * o_stride, x + r * x_stride, w, n, over);
}

/* Each row's heads rotated by that row's own angles: row `r`'s angles are `cs + r·rot` (the `rot/2` cosines,
 * then the sines), and each of its `heads` heads, `head_stride` apart, has its first `rot` elements rotated —
 * the partial RoPE of one position, every position at once. `o` may be `x`. */
static __device__ inline void nn__rope__zzabi_body_rows(uint16_t* o, const uint16_t* x, const float* cs, uint64_t rows,
                                                        uint64_t heads, uint64_t head_stride, uint64_t row_stride,
                                                        uint64_t rot, unsigned int* over) {
    const uint64_t pairs = rot / 2ull, per_row = heads * pairs, n = rows * per_row;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t r = i / per_row, rest = i - r * per_row, e = rest / pairs, k = rest - e * pairs;
        const uint64_t at = r * row_stride + e * head_stride + k;
        const float a = nn__silicon__half_to_float(x[at]), b = nn__silicon__half_to_float(x[at + pairs]);
        const float c = cs[r * rot + k], s = cs[r * rot + pairs + k];
        bool hit = false;
        o[at] = nn__primitives__zzpackage_float_to_half(a * c - b * s, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
        hit = false;
        o[at + pairs] = nn__primitives__zzpackage_float_to_half(b * c + a * s, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* The angles of positions `first .. first + rows`, row `r` at `cs + r·head_dim`: the one-position body's
 * arithmetic, item for item. */
static __device__ inline void nn__rope__zzabi_body_angles_rows(float* cs, uint64_t first, float theta, uint64_t head_dim,
                                                               uint64_t rows) {
    const uint64_t pairs = head_dim / 2ull, n = rows * pairs;
    const float r = nn__silicon__expf(-2.0f / (float)(uint32_t)head_dim * nn__silicon__logf(theta));
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t row = i / pairs, k = i - row * pairs;
        float frequency = 1.0f;
        for (uint64_t m = 0ull; m < k; ++m) frequency *= r;
        const float angle = (float)(first + row) * frequency;
        cs[row * head_dim + k] = nn__silicon__cosf(angle);
        cs[row * head_dim + pairs + k] = nn__silicon__sinf(angle);
    }
}

/* ⭐ CAUSAL ATTENTION, `rows` QUERIES: row `r` is position `first + r` and attends the cache's rows
 *   `0 .. first + r`, so its keys and values must already be in the cache. A block a (row, query head); the
 *   weights go to `p + (r·q_heads + h)·(first + rows)`, and the second launch mixes them, as for one query. */
static __device__ inline void nn__attention__zzabi_body_causal_scores(float* p, const uint16_t* q, const uint16_t* k,
                                                                       uint64_t q_heads, uint64_t kv_heads,
                                                                       uint64_t head_dim, uint64_t first, uint64_t rows) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim, span = first + rows;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    for (uint64_t i = nn__silicon__block(); i < rows * q_heads; i += nn__silicon__blocks()) {
        const uint64_t r = i / q_heads, h = i - r * q_heads;
        nn__attention__zzprivate_scores_head(p + i * span, q + i * head_dim, k + (h / group) * head_dim, row, head_dim,
                                             first + r + 1ull, scale);
    }
}

static __device__ inline void nn__attention__zzabi_body_causal_mix(uint16_t* o, const float* p, const uint16_t* v,
                                                                    uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                    uint64_t first, uint64_t rows, unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim, span = first + rows;
    for (uint64_t i = nn__silicon__block(); i < rows * q_heads; i += nn__silicon__blocks()) {
        const uint64_t r = i / q_heads, h = i - r * q_heads;
        nn__attention__zzprivate_mix_head(o + i * head_dim, p + i * span, v + (h / group) * head_dim, row, head_dim,
                                          first + r + 1ull, over);
    }
}

/* ⭐ THE OUTPUT GATE OF GATED ATTENTION: `o = att · σ(gate)`, the gate the second half of each head's pair in `qg`
 *   (`[head][query · gate]`, as q_proj leaves them), over `heads` heads of every row. The sigmoid is rounded to a
 *   half before the multiply, as the one-position verbs round it. */
static __device__ inline void nn__attention__zzabi_body_gate_rows(uint16_t* o, const uint16_t* att, const uint16_t* qg,
                                                                  uint64_t heads, uint64_t head_dim, unsigned int* over) {
    const uint64_t n = heads * head_dim;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t h = i / head_dim, e = i - h * head_dim;
        const float v = nn__silicon__half_to_float(qg[h * 2ull * head_dim + head_dim + e]);
        bool hit = false;
        const uint16_t gate = nn__primitives__zzpackage_float_to_half(1.0f / (1.0f + nn__silicon__expf(-v)), &hit);
        o[i] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(att[i]) * nn__silicon__half_to_float(gate), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⭐ THE DELTANET'S GATES FOR EVERY ROW: `β = σ(b)` and `g = −e^{A_log} · softplus(a + dt_bias)`, head `h` of each
 *   row taking `A_log[h]` and `dt_bias[h]`. Each intermediate is rounded to a half where the one-position verbs
 *   round it, so a row's gates are theirs to the bit. */
static __device__ inline void nn__deltanet__zzabi_body_gates_rows(uint16_t* beta, uint16_t* g, const uint16_t* a,
                                                                  const uint16_t* b, const uint16_t* a_log,
                                                                  const uint16_t* dt_bias, uint64_t heads, uint64_t rows,
                                                                  unsigned int* over) {
    const uint64_t n = heads * rows;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t h = i % heads;
        bool hit = false;
        beta[i] = nn__primitives__zzpackage_float_to_half(1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[i]))), &hit);
        const float sum = nn__silicon__half_to_float(nn__primitives__zzpackage_float_to_half(
                              nn__silicon__half_to_float(a[i]) + nn__silicon__half_to_float(dt_bias[h]), &hit));
        const float mag = (sum < 0.0f) ? -sum : sum, top = (sum > 0.0f) ? sum : 0.0f;
        const float sp = nn__silicon__half_to_float(nn__primitives__zzpackage_float_to_half(
                             top + nn__silicon__logf(1.0f + nn__silicon__expf(-mag)), &hit));
        const float ea = nn__silicon__half_to_float(nn__primitives__zzpackage_float_to_half(
                             nn__silicon__expf(nn__silicon__half_to_float(a_log[h])), &hit));
        const float prod = nn__silicon__half_to_float(nn__primitives__zzpackage_float_to_half(sp * ea, &hit));
        g[i] = nn__primitives__zzpackage_float_to_half(prod * -1.0f, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ⭐ THE CONV OVER `rows` POSITIONS IN ORDER: a lane a channel, as the one-position body, walking the positions
 *   itself — its window is its own, so position `t + 1` reads what `t` slid into it and no other lane is
 *   involved. `x` and `o` are `rows` rows of `ch`. */
static __device__ inline void nn__deltanet__zzabi_body_conv_steps(uint16_t* o, uint16_t* s, const uint16_t* w,
                                                                  const uint16_t* x, uint64_t ch, uint64_t rows,
                                                                  unsigned int* over) {
    for (uint64_t c = nn__kernels__zzprivate_first(); c < ch; c += nn__kernels__zzprivate_stride()) {
        const uint16_t* wc = w + c * NN__DELTANET__CONV_TAPS;
        uint16_t*       sc = s + c * NN__DELTANET__CONV_WINDOW;
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint16_t xt = x[t * ch + c];
            float acc = 0.0f;
            for (unsigned k = 0u; k < NN__DELTANET__CONV_WINDOW; ++k)
                acc += nn__silicon__half_to_float(wc[k]) * nn__silicon__half_to_float(sc[k]);
            acc += nn__silicon__half_to_float(wc[NN__DELTANET__CONV_WINDOW]) * nn__silicon__half_to_float(xt);
            bool hit = false;
            o[t * ch + c] = nn__primitives__zzpackage_float_to_half(acc / (1.0f + nn__silicon__expf(-acc)), &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
            for (unsigned k = 0u; k + 1u < NN__DELTANET__CONV_WINDOW; ++k) sc[k] = sc[k + 1u];
            sc[NN__DELTANET__CONV_WINDOW - 1u] = xt;
        }
    }
}

/* ⭐⭐ THE DELTANET OVER `rows` POSITIONS IN ONE LAUNCH — ⚖ *"a for the deltanet"*: the recurrence is sequential,
 *   so each head's block walks the positions in order, its state carried from one to the next in place — the
 *   one-position step, `rows` times, with no launch between. Row `t` reads `conved`, `z`, `beta` and `g` at
 *   its own row and writes `out`'s. */
static __device__ inline void nn__deltanet__zzabi_body_steps(float* S, const uint16_t* conved, const uint16_t* z,
                                                             const uint16_t* beta, const uint16_t* g, const uint16_t* w,
                                                             uint16_t* out, uint64_t k_heads, uint64_t v_heads,
                                                             uint64_t head_dim, uint64_t rows, unsigned int* over) {
    const uint64_t d = head_dim, rep = v_heads / k_heads, ch = (2ull * k_heads + v_heads) * d, vd = v_heads * d;
    bool hit = false;
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks())
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint16_t* c = conved + t * ch;
            nn__deltanet__zzprivate_step_head(S + h * d * d, c + (h / rep) * d, c + k_heads * d + (h / rep) * d,
                                              c + 2ull * k_heads * d + h * d, z + t * vd + h * d,
                                              nn__silicon__half_to_float(beta[t * v_heads + h]),
                                              nn__silicon__half_to_float(g[t * v_heads + h]), w, out + t * vd + h * d, d, &hit);
        }
    if (hit) *over = NN__KERNELS__OVERFLOWED;
}

/* ── THE HADAMARD ROTATION ──────────────────────────────────────────────────────────────────────── */
/* ⭐ WIDE: A LANE PER OUTPUT, AND NO BUTTERFLY. The butterfly's stages each need every lane's previous
 *   stage, which is a barrier per stage, and a card barrier belongs only in a fold. So each output is its
 *   own row of the Hadamard matrix applied directly — `out[r] = (1/√b) · Σ_c (−1)^popcount(r & c) · s[c] ·
 *   in[c]` over its block of at most 512 — summed in fp32 and rounded to a half once, where the butterfly
 *   rounded after every stage. The block ladder (512s, then a 256, then a 128) and the sign vector
 *   starting afresh at each block are the butterfly's.
 * ⛔ WHEN `out` OVERLAPS `x` — rotating in place, which the verb allows — a lane would read an input another
 *   lane had already overwritten, so one lane runs the butterfly instead, which reaches `out` first and
 *   works there. Correct either way; wide only when the two are apart. */
/* One term of an output's direct sum: `(−1)^popcount(r & c) · s[c] · in[c]`, for a block of at most 512. */
static __device__ inline float nn__hadamard__zzprivate_term(const uint16_t* in, const uint16_t* sign, uint64_t r, uint64_t c) {
    uint64_t p = r & c;
    p ^= p >> 8; p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
    const float v = nn__silicon__half_to_float(in[c]) * nn__silicon__half_to_float(sign[c]);
    return (p & 1ull) ? -v : v;
}
static __device__ inline void nn__hadamard__zzabi_body_rotate(uint16_t* out, const uint16_t* in,
                                                               const uint16_t* sign, uint64_t n,
                                                               unsigned int* over) {
    const uint64_t whole = n < NN__HADAMARD__WHOLE_BELOW ? 0ull : (n / NN__HADAMARD__WHOLE_BELOW) * NN__HADAMARD__WHOLE_BELOW;
    const uint64_t rest = n - whole;
    if (out < in + n && in < out + n) {
        if (nn__kernels__zzprivate_first() != 0ull) return;
        bool hit = false;
        if (whole == 0ull) nn__hadamard__zzpackage_one_block(out, in, sign, n, &hit);
        for (uint64_t a = 0ull; a < whole; a += NN__HADAMARD__WHOLE_BELOW)
            nn__hadamard__zzpackage_one_block(out + a, in + a, sign, NN__HADAMARD__WHOLE_BELOW, &hit);
        uint64_t a = whole;
        if (whole != 0ull && rest >= 256ull) { nn__hadamard__zzpackage_one_block(out + a, in + a, sign, 256ull, &hit); a += 256ull; }
        if (whole != 0ull && n - a >= 128ull) nn__hadamard__zzpackage_one_block(out + a, in + a, sign, 128ull, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
        return;
    }
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        uint64_t a = 0ull, b = n;
        if (i < whole) { a = i - i % NN__HADAMARD__WHOLE_BELOW; b = NN__HADAMARD__WHOLE_BELOW; }
        else if (whole != 0ull) {
            if (rest >= 256ull && i < whole + 256ull) { a = whole; b = 256ull; }
            else { a = whole + (rest >= 256ull ? 256ull : 0ull); b = 128ull; }
        }
        const uint64_t r = i - a;
        /* ⛳ FOUR SUMS, NOT ONE: a single chain waits on each term's loads before the next can add, and a
         *   lane walks up to 512 of them — four independent chains keep four terms' loads in flight. */
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint64_t c = 0ull;
        for (; c + 4ull <= b; c += 4ull) {
            s0 += nn__hadamard__zzprivate_term(in + a, sign, r, c);
            s1 += nn__hadamard__zzprivate_term(in + a, sign, r, c + 1ull);
            s2 += nn__hadamard__zzprivate_term(in + a, sign, r, c + 2ull);
            s3 += nn__hadamard__zzprivate_term(in + a, sign, r, c + 3ull);
        }
        for (; c < b; ++c) s0 += nn__hadamard__zzprivate_term(in + a, sign, r, c);
        const float sum = (s0 + s1) + (s2 + s3);
        bool hit = false;
        out[i] = nn__primitives__zzpackage_float_to_half(sum / nn__silicon__sqrtf((float)b), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}


/* ══ ⭐⭐⭐ THE HELPERS THE LOOPS CALL ═════════════════════════════════════════════════════════════════
 *
 * ⛔⛔ EVERY FUNCTION A BODY ABOVE CALLS LIVES HERE TOO, NOT IN AN `*__impl.cuh` BESIDE THE VERBS. Each
 * is pure compute — `MEASURED`, zero `sys` references among them; `nn__silicon__sqrtf`, which is
 * the seam — so it belongs on the card side of the split. ⛳ NO COUNT HERE: everything below this banner.
 *
 * ⇒ ★★ AND ONLY A BUILD CAN CHECK THAT, NOT A READING. `kernels.cuh` is included into the same
 * translation unit as the verbs, so a call from a loop to a helper two files away resolves exactly as
 * well as a call to one next door: **a split that exists in the FILING and not in the BUILD** is
 * invisible to a normal compile.
 * ⛳ THE INSTRUMENT is compiling the device TU BY ITSELF and reading the LINKER.
 * ⛔⛔ `-fsyntax-only` ON THAT SAME TU CANNOT SEE THIS DEFECT: a declaration is all a syntax pass needs,
 * and it is the cheaper check, so it is the one that gets reached for. ⇒ ★★ **A GREEN FROM A PASS THAT
 * CANNOT SEE THE DEFECT CLASS IS NOT EVIDENCE** — `MEASURED`, the two passes disagreed 0 against 5 on
 * identical input.
 */

/* ⛔ THESE THREE COME FIRST AND THE ORDER IS LOAD-BEARING. Unlike the ones below, none of them is
 * DECLARED in a package header — they are private to TurboQuant, so in this file a definition is the
 * only declaration there is and it has to precede its callers (`zzprivate_weight` calls all three).
 * ⇒ ★ A PRIVATE HELPER'S DECLARATION IS WHEREVER IT IS DEFINED, and a file with no header to lean on
 * orders by dependency or not at all. */

/* ══ GAUSSIAN TABLE — GENERATED, BEGIN ══════════════════════════════════════════════════ */
/* ⛔⛔ GENERATED BY `backstage/scripts/nn_gaussian_tables.py` — DO NOT HAND-EDIT.
 * Regenerate with `--emit`; `--check` fails the build if this block and the script
 * disagree. ▶ that script for the construction, and for what it validates.
 * 318 fp16 entries: 2@D1 + 4@D2 + 8@D3 + 16@D4 + 32@D5 + 256@D8.
 * ⛳ AT D=4 THESE ARE THE VERBATIM bitsandbytes NF4 LEVELS, checked to fp16's own resolution. */
static __device__ const uint16_t nn__turboquant__zzprivate_gaussian[318] = {
    /* D=1 — 2 levels, -1.0000 … +1.0000 */
    0xBC00, 0x3C00,
    /* D=2 — 4 levels, -1.0000 … +1.0000 */
    0xBC00, 0xB4CC, 0x34CC, 0x3C00,
    /* D=3 — 8 levels, -1.0000 … +1.0000 */
    0xBC00, 0xB8FF, 0xB59F, 0xAF4A, 0x2F4A, 0x359F, 0x38FF, 0x3C00,
    /* D=4 — 16 levels, -1.0000 … +1.0000 */
    0xBC00, 0xBA0F, 0xB8BD, 0xB75B, 0xB585, 0xB3B1, 0xB08B, 0xAA04,
    0x2A04, 0x308B, 0x33B1, 0x3585, 0x375B, 0x38BD, 0x3A0F, 0x3C00,
    /* D=5 — 32 levels, -1.0000 … +1.0000 */
    0xBC00, 0xBA9A, 0xB9B0, 0xB8FA, 0xB863, 0xB7BC, 0xB6CD, 0xB5F2,
    0xB525, 0xB464, 0xB355, 0xB1F0, 0xB094, 0xAE81, 0xABC6, 0xA52C,
    0x252C, 0x2BC6, 0x2E81, 0x3094, 0x31F0, 0x3355, 0x3464, 0x3525,
    0x35F2, 0x36CD, 0x37BC, 0x3863, 0x38FA, 0x39B0, 0x3A9A, 0x3C00,
    /* D=8 — 256 levels, -1.0000 … +1.0000 */
    0xBC00, 0xBB47, 0xBAD2, 0xBA7C, 0xBA37, 0xB9FC, 0xB9C9, 0xB99C,
    0xB974, 0xB94F, 0xB92C, 0xB90D, 0xB8EF, 0xB8D3, 0xB8B9, 0xB89F,
    0xB888, 0xB871, 0xB85B, 0xB846, 0xB832, 0xB81E, 0xB80B, 0xB7F2,
    0xB7CF, 0xB7AD, 0xB78B, 0xB76B, 0xB74B, 0xB72C, 0xB70E, 0xB6F0,
    0xB6D3, 0xB6B7, 0xB69B, 0xB680, 0xB666, 0xB64B, 0xB632, 0xB618,
    0xB5FF, 0xB5E7, 0xB5CF, 0xB5B7, 0xB5A0, 0xB589, 0xB572, 0xB55B,
    0xB545, 0xB52F, 0xB51A, 0xB504, 0xB4EF, 0xB4DA, 0xB4C5, 0xB4B1,
    0xB49D, 0xB489, 0xB475, 0xB461, 0xB44E, 0xB43A, 0xB427, 0xB414,
    0xB401, 0xB3DD, 0xB3B8, 0xB394, 0xB36F, 0xB34B, 0xB327, 0xB303,
    0xB2E0, 0xB2BC, 0xB299, 0xB276, 0xB254, 0xB231, 0xB20F, 0xB1ED,
    0xB1CB, 0xB1AA, 0xB188, 0xB167, 0xB146, 0xB125, 0xB104, 0xB0E3,
    0xB0C3, 0xB0A2, 0xB082, 0xB062, 0xB042, 0xB022, 0xB002, 0xAFC4,
    0xAF85, 0xAF46, 0xAF07, 0xAEC8, 0xAE8A, 0xAE4C, 0xAE0D, 0xADCF,
    0xAD92, 0xAD54, 0xAD16, 0xACD9, 0xAC9C, 0xAC5F, 0xAC21, 0xABC9,
    0xAB4F, 0xAAD6, 0xAA5C, 0xA9E3, 0xA96A, 0xA8F1, 0xA878, 0xA7FF,
    0xA70E, 0xA61C, 0xA52B, 0xA43B, 0xA294, 0xA0B3, 0x9DA3, 0x9784,
    0x1784, 0x1DA3, 0x20B3, 0x2294, 0x243B, 0x252B, 0x261C, 0x270E,
    0x27FF, 0x2878, 0x28F1, 0x296A, 0x29E3, 0x2A5C, 0x2AD6, 0x2B4F,
    0x2BC9, 0x2C21, 0x2C5F, 0x2C9C, 0x2CD9, 0x2D16, 0x2D54, 0x2D92,
    0x2DCF, 0x2E0D, 0x2E4C, 0x2E8A, 0x2EC8, 0x2F07, 0x2F46, 0x2F85,
    0x2FC4, 0x3002, 0x3022, 0x3042, 0x3062, 0x3082, 0x30A2, 0x30C3,
    0x30E3, 0x3104, 0x3125, 0x3146, 0x3167, 0x3188, 0x31AA, 0x31CB,
    0x31ED, 0x320F, 0x3231, 0x3254, 0x3276, 0x3299, 0x32BC, 0x32E0,
    0x3303, 0x3327, 0x334B, 0x336F, 0x3394, 0x33B8, 0x33DD, 0x3401,
    0x3414, 0x3427, 0x343A, 0x344E, 0x3461, 0x3475, 0x3489, 0x349D,
    0x34B1, 0x34C5, 0x34DA, 0x34EF, 0x3504, 0x351A, 0x352F, 0x3545,
    0x355B, 0x3572, 0x3589, 0x35A0, 0x35B7, 0x35CF, 0x35E7, 0x35FF,
    0x3618, 0x3632, 0x364B, 0x3666, 0x3680, 0x369B, 0x36B7, 0x36D3,
    0x36F0, 0x370E, 0x372C, 0x374B, 0x376B, 0x378B, 0x37AD, 0x37CF,
    0x37F2, 0x380B, 0x381E, 0x3832, 0x3846, 0x385B, 0x3871, 0x3888,
    0x389F, 0x38B9, 0x38D3, 0x38EF, 0x390D, 0x392C, 0x394F, 0x3974,
    0x399C, 0x39C9, 0x39FC, 0x3A37, 0x3A7C, 0x3AD2, 0x3B47, 0x3C00,
};

/* ⭐ THE SAME LEVELS AS int8 — round(127 x level), half away from zero — for the int8 gemv, which
 * multiplies them against x quantised to int8. One table, generated here, so every family's int8
 * body reads the same numbers. */
static __device__ const int8_t nn__turboquant__zzprivate_gaussian_i8[318] = {
    -127, 127,
    -127, -38, 38, 127,
    -127, -79, -45, -14, 14, 45, 79, 127,
    -127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127,
    -127, -105, -90, -79, -70, -61, -54, -47, -41, -35, -29, -24, -18, -13, -8, -3,
    3, 8, 13, 18, 24, 29, 35, 41, 47, 54, 61, 70, 79, 90, 105, 127,
    -127, -116, -108, -103, -99, -95, -92, -89, -87, -84, -82, -80, -78, -77, -75, -73,
    -72, -71, -69, -68, -67, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55,
    -54, -53, -52, -52, -51, -50, -49, -48, -48, -47, -46, -45, -45, -44, -43, -43,
    -42, -41, -40, -40, -39, -39, -38, -37, -37, -36, -35, -35, -34, -34, -33, -32,
    -32, -31, -31, -30, -30, -29, -28, -28, -27, -27, -26, -26, -25, -25, -24, -24,
    -23, -22, -22, -21, -21, -20, -20, -19, -19, -18, -18, -17, -17, -16, -16, -15,
    -15, -14, -14, -13, -13, -12, -12, -12, -11, -11, -10, -10, -9, -9, -8, -8,
    -7, -7, -6, -6, -5, -5, -4, -4, -3, -3, -3, -2, -2, -1, -1, 0,
    0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15,
    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23,
    24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 30, 30, 31, 31, 32,
    32, 33, 34, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 40, 41, 42,
    43, 43, 44, 45, 45, 46, 47, 48, 48, 49, 50, 51, 52, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72,
    73, 75, 77, 78, 80, 82, 84, 87, 89, 92, 95, 99, 103, 108, 116, 127,
};

/* Where width `d`'s block starts. ⛔ A width with no table is a REFUSAL, not a zero:
 * every caller checks `nn__turboquant__rate_is_known` first, and this answers the widths that
 * check admits and nothing else. */
static __device__ inline uint64_t nn__turboquant__zzprivate_gaussian_at(uint64_t d) {
    if (d == 1ull) return 0ull;
    if (d == 2ull) return 2ull;
    if (d == 3ull) return 6ull;
    if (d == 4ull) return 14ull;
    if (d == 5ull) return 30ull;
    if (d == 8ull) return 62ull;
    return 318ull;                                   /* unreachable past `rate_is_known` */
}
/* ══ GAUSSIAN TABLE — GENERATED, END ════════════════════════════════════════════════════ */

/* ⭐⭐ THE int8 GEMV — THE SAME MATRIX, MULTIPLIED IN INTEGERS. ⚖ *"it should be two different instructions
 *   so at boot time the lisp can decide how to write its defun"*: this is the second one, faster where
 *   integer dot products are cheap and ~1% of the rows' rms away from the exact gemv (`MEASURED` on the CPU,
 *   `measurements/2026-09-27_avx2_expert_gemv.md`). What every family's body must agree on:
 *     x         quantised in blocks of 32 columns from column 0: scale = max|x| / 127 over the block, each
 *               value rounded to the nearest int8, half away from zero
 *     levels    `nn__turboquant__zzprivate_gaussian_i8`, the generated table: round(127 x level)
 *     products  int8 x int8 summed in int32 over a block — exact; then the block's float scale, and the
 *               row's scale / 127 once the lanes are summed
 * ⛳ A LANE OWNS BLOCKS OF 32 COLUMNS — `lane`, `lane + lanes`, … — and quantises its own blocks of `x` once a
 *   kernel, holding the first two in registers (64 int8, 16 words) with their scales; the rest are
 *   quantised as they are met. Nothing crosses between lanes but the row's sum, so there is no scratch
 *   buffer and no second launch. */
#define NN__TURBOQUANT__INT8_BLOCK 32u
static __device__ inline float nn__turboquant__zzprivate_quantise(const uint16_t* x, uint64_t c0, uint32_t* q) {
    float mx = 0.0f;
    for (uint32_t i = 0u; i < NN__TURBOQUANT__INT8_BLOCK; ++i) {
        const float v = nn__silicon__half_to_float(x[c0 + i]);
        const float a = v < 0.0f ? -v : v;
        if (a > mx) mx = a;
    }
    const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
    for (uint32_t w = 0u; w < NN__TURBOQUANT__INT8_BLOCK / 4u; ++w) {
        uint32_t packed = 0u;
        for (uint32_t b = 0u; b < 4u; ++b) {
            const float t = nn__silicon__half_to_float(x[c0 + 4u * w + b]) * inv;
            const int32_t v = t >= 0.0f ? (int32_t)(t + 0.5f) : -(int32_t)(-t + 0.5f);
            packed |= ((uint32_t)(uint8_t)(int8_t)v) << (8u * b);
        }
        q[w] = packed;
    }
    return mx / 127.0f;
}
static __device__ inline int32_t nn__turboquant__zzprivate_block_i8(const uint32_t D, const uint8_t* row, uint64_t c0,
                                                                    const uint32_t* q) {
    const uint32_t per = 16u / D;
    const uint64_t table = nn__turboquant__zzprivate_gaussian_at(D);
    int32_t acc = 0;
    for (uint32_t i = 0u; i < NN__TURBOQUANT__INT8_BLOCK; ++i) {
        const uint64_t c = c0 + i, u = c / per;
        const uint32_t unit = (uint32_t)row[2ull * u] | ((uint32_t)row[2ull * u + 1ull] << 8);
        const uint32_t code = (unit >> (D * (uint32_t)(c % per))) & ((1u << D) - 1u);
        const int32_t xv = (int32_t)(int8_t)(uint8_t)(q[i / 4u] >> (8u * (i % 4u)));
        acc += xv * (int32_t)nn__turboquant__zzprivate_gaussian_i8[table + code];
    }
    return acc;
}
static __device__ inline void nn__turboquant__zzprivate_rows_i8(const uint32_t D, uint16_t* out, const uint8_t* weights,
                                                                 const uint8_t* luts, const uint16_t* x, uint64_t rows,
                                                                 uint64_t cols, uint64_t rows_held, uint64_t scales_held,
                                                                 unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t row_bytes = nn__turboquant__row_bytes(D, cols), blocks = cols / NN__TURBOQUANT__INT8_BLOCK;
    uint32_t q0[NN__TURBOQUANT__INT8_BLOCK / 4u], q1[NN__TURBOQUANT__INT8_BLOCK / 4u];
    const float s0 = lane < blocks ? nn__turboquant__zzprivate_quantise(x, lane * NN__TURBOQUANT__INT8_BLOCK, q0) : 0.0f;
    const float s1 = lane + lanes < blocks
                   ? nn__turboquant__zzprivate_quantise(x, (lane + lanes) * NN__TURBOQUANT__INT8_BLOCK, q1) : 0.0f;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        if (r >= rows_held || r >= scales_held) continue;
        const uint8_t* row = weights + r * row_bytes;
        float part = 0.0f;
        if (lane < blocks)
            part += (float)nn__turboquant__zzprivate_block_i8(D, row, lane * NN__TURBOQUANT__INT8_BLOCK, q0) * s0;
        if (lane + lanes < blocks)
            part += (float)nn__turboquant__zzprivate_block_i8(D, row, (lane + lanes) * NN__TURBOQUANT__INT8_BLOCK, q1) * s1;
        for (uint64_t b = lane + 2ull * lanes; b < blocks; b += lanes) {
            uint32_t q[NN__TURBOQUANT__INT8_BLOCK / 4u];
            const float s = nn__turboquant__zzprivate_quantise(x, b * NN__TURBOQUANT__INT8_BLOCK, q);
            part += (float)nn__turboquant__zzprivate_block_i8(D, row, b * NN__TURBOQUANT__INT8_BLOCK, q) * s;
        }
        const float scale = nn__silicon__half_to_float(
                                (uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
        const float sum = nn__silicon__lanes_sum(part) * scale / 127.0f;
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}
/* ── ⭐ WHAT A FAMILY'S OVERRIDE OF A TURBOQUANT DOOR MAY USE ─────────────────────────────────────────
 * ⚖ *"a for the family bodies"*: a family may run its own body for a door (`silicon_families/<family>/
 * overrides/`), and it must compute what nn's does. These are nn's definitions, published for exactly that:
 * a width's levels, as fp16 and as int8, `x` quantised the way the int8 gemv says, and a result written the
 * way every body writes one — so an override reuses the definition rather than restating it. */
static __device__ inline const uint16_t* nn__turboquant__zzabi_levels(uint64_t d) {
    return nn__turboquant__zzprivate_gaussian + nn__turboquant__zzprivate_gaussian_at(d);
}
static __device__ inline const int8_t* nn__turboquant__zzabi_levels_i8(uint64_t d) {
    return nn__turboquant__zzprivate_gaussian_i8 + nn__turboquant__zzprivate_gaussian_at(d);
}
static __device__ inline float nn__turboquant__zzabi_quantise(const uint16_t* x, uint64_t c0, uint32_t* q) {
    return nn__turboquant__zzprivate_quantise(x, c0, q);
}
/* A float as the half an output holds, the door's fault word raised if it overflowed. */
static __device__ inline uint16_t nn__kernels__zzabi_to_half(float f, unsigned int* over) {
    bool hit = false;
    const uint16_t h = nn__primitives__zzpackage_float_to_half(f, &hit);
    if (hit) *over = NN__KERNELS__OVERFLOWED;
    return h;
}

/* ⛳ EACH CALL NAMES ITS WIDTH AS A LITERAL, as the exact gemv's does. The verb has refused a width with no
 *   table and columns not a multiple of 32, so every width reaching here has a block of the int8 table. */
static __device__ inline bool nn__turboquant__zzabi_body_gemv_int8(uint16_t* out, const uint8_t* weights,
                                                                    uint64_t w_room, const uint8_t* luts,
                                                                    uint64_t l_room, const uint16_t* x,
                                                                    uint64_t d, uint64_t rows,
                                                                    uint64_t cols, unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const uint64_t rows_held = row_bytes == 0ull ? 0ull : w_room / row_bytes, scales_held = l_room / 2ull;
    if (d == 1ull)      nn__turboquant__zzprivate_rows_i8(1u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    else if (d == 2ull) nn__turboquant__zzprivate_rows_i8(2u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    else if (d == 3ull) nn__turboquant__zzprivate_rows_i8(3u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    else if (d == 4ull) nn__turboquant__zzprivate_rows_i8(4u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    else if (d == 5ull) nn__turboquant__zzprivate_rows_i8(5u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    else if (d == 8ull) nn__turboquant__zzprivate_rows_i8(8u, out, weights, luts, x, rows, cols, rows_held, scales_held, over);
    return true;
}


/* ⭐ THE GROUPED GEMVS IN INTEGERS — the grouped gemvs' shapes with the int8 gemv's arithmetic, for a caller that chose
 *   it at boot (⚖ *"two different instructions"*). Every width must have an int8 table and `cols` be whole blocks of 32. */
static __device__ inline void nn__turboquant__zzabi_body_gemv_groups_int8(uint16_t* out, nn__turboquant__groups g,
                                                                          const uint16_t* x, uint64_t cols, unsigned int* over) {
    for (uint64_t i = 0ull; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
        const uint64_t row_bytes = nn__turboquant__row_bytes(g.d[i], cols);
        (void)nn__turboquant__zzabi_body_gemv_int8(out + g.out_at[i], (const uint8_t*)(uintptr_t)g.codes[i], g.rows[i] * row_bytes,
                                                   (const uint8_t*)(uintptr_t)g.luts[i], g.rows[i] * 2ull, x, g.d[i], g.rows[i],
                                                   cols, over);
    }
}

/* `out[r] = residual[r] + Σ_i w[out_at[i]] · (W_i[r] · x_i)` in integers — `x_i` at `x + i·cols`, each quantised in
 *   blocks of 32 as the int8 gemv says, a block a lane; the weighted sum as the grouped gemv's. */
static __device__ inline void nn__turboquant__zzabi_body_gemv_groups_sum_int8(uint16_t* out, nn__turboquant__groups g,
                                                                              const uint16_t* x, const uint16_t* w,
                                                                              const uint16_t* residual, uint64_t rows,
                                                                              uint64_t cols, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), blocks = cols / NN__TURBOQUANT__INT8_BLOCK;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        float part = 0.0f;
        for (uint64_t i = 0ull; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
            const uint32_t D = (uint32_t)g.d[i];
            const uint8_t* row = (const uint8_t*)(uintptr_t)g.codes[i] + r * nn__turboquant__row_bytes(D, cols);
            float mine = 0.0f;
            for (uint64_t b = lane; b < blocks; b += lanes) {
                uint32_t q[NN__TURBOQUANT__INT8_BLOCK / 4u];
                const float s = nn__turboquant__zzprivate_quantise(x + i * cols, b * NN__TURBOQUANT__INT8_BLOCK, q);
                mine += (float)nn__turboquant__zzprivate_block_i8(D, row, b * NN__TURBOQUANT__INT8_BLOCK, q) * s;
            }
            const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8)));
            part += mine * scale / 127.0f * nn__silicon__half_to_float(w[g.out_at[i]]);
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (lane == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(residual[r]) + sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN INTEGERS — `expert_groups_int8`, `expert_rows_sum_int8` ═════════════════════════════
 * The same shapes as the two exact doors above, in the int8 gemv's arithmetic with a wider block: each row of `x` cut
 * into blocks of `NN__EXPERT__INT8_BLOCK`, a block quantised to int8 by its largest magnitude (127 to it, rounded half away
 * from zero), the weights as the width's int8 levels, a block's products summed in integers and scaled once, the row's
 * scale and 1/127 last. ⛳ WHY 256 AND NOT THE GEMV'S 32 — `MEASURED` on the R730's Xeons (the family's override, 28
 * threads, 7 and 28 rows an expert): a block of 32 needs a float step every 32 products and ran 68-300 GMAC/s, a block of
 * 256 ran 300-545, the exact fp32 GEMM 100-132. A block of 256 of a rotated row still spans its values' spread.
 * Widths with an int8 table only (4, 5, 8 bits), `cols` whole blocks. */
#define NN__EXPERT__INT8_BLOCK 256u

/* Block `c0` of a row of `x` quantised into `q`; its scale answered. */
static __device__ inline float nn__expert__zzprivate_quantise(const uint16_t* x, uint64_t c0, int8_t* q) {
    float mx = 0.0f;
    for (uint32_t i = 0u; i < NN__EXPERT__INT8_BLOCK; ++i) {
        const float v = nn__silicon__half_to_float(x[c0 + i]);
        const float a = v < 0.0f ? -v : v;
        if (a > mx) mx = a;
    }
    const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
    for (uint32_t i = 0u; i < NN__EXPERT__INT8_BLOCK; ++i) {
        const float t = nn__silicon__half_to_float(x[c0 + i]) * inv;
        q[i] = (int8_t)(t >= 0.0f ? (int32_t)(t + 0.5f) : -(int32_t)(-t + 0.5f));
    }
    return mx / 127.0f;
}
static __device__ inline void nn__expert__zzprivate_rows_int8(uint16_t* out, float* acc, const uint8_t* weights,
                                                              uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                                              const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                              uint64_t count, uint64_t d, uint64_t out_rows,
                                                              uint64_t cols, unsigned int* over) {
    const uint32_t D = (uint32_t)d, per = 16u / D, mask = (1u << D) - 1u;
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols), blocks = cols / NN__EXPERT__INT8_BLOCK;
    const uint64_t rows_held = row_bytes == 0ull ? 0ull : w_room / row_bytes, held = rows_held < l_room / 2ull ? rows_held : l_room / 2ull;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), table = nn__turboquant__zzprivate_gaussian_at(D);
    for (uint64_t r = nn__silicon__block(); r < out_rows; r += nn__silicon__blocks()) {
        const uint8_t* row = r < held ? weights + r * row_bytes : 0;
        for (uint64_t j = 0ull; j < count; ++j) {
            const uint16_t* xr = x + (uint64_t)rows[2ull * j] * cols;
            float part = 0.0f;
            for (uint64_t b = lane; row != 0 && b < blocks; b += lanes) {
                int8_t q[NN__EXPERT__INT8_BLOCK];
                const float s = nn__expert__zzprivate_quantise(xr, b * NN__EXPERT__INT8_BLOCK, q);
                int32_t dot = 0;
                for (uint32_t i = 0u; i < NN__EXPERT__INT8_BLOCK; ++i) {
                    const uint64_t c = b * NN__EXPERT__INT8_BLOCK + i, u = c / per;
                    const uint32_t unit = (uint32_t)row[2ull * u] | ((uint32_t)row[2ull * u + 1ull] << 8);
                    dot += (int32_t)q[i] * (int32_t)nn__turboquant__zzprivate_gaussian_i8[table + ((unit >> (D * (uint32_t)(c % per))) & mask)];
                }
                part += (float)dot * s;
            }
            const float sum = nn__silicon__lanes_sum(part);
            if (lane != 0ull || row == 0) continue;
            const float v = sum * nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)))
                          / 127.0f;
            const uint64_t at = (uint64_t)rows[2ull * j + 1ull] * out_rows + r;
            if (out) {
                bool hit = false;
                out[at] = nn__primitives__zzpackage_float_to_half(v, &hit);
                if (hit) *over = NN__KERNELS__OVERFLOWED;
            } else {
                acc[at] += v * nn__silicon__half_to_float(w[rows[2ull * j]]);
            }
        }
    }
}
static __device__ inline void nn__expert__zzabi_body_groups_int8(uint16_t* out, nn__expert__groups g, const uint16_t* x,
                                                                 const uint32_t* rows, uint64_t cols, unsigned int* over) {
    for (uint64_t i = 0ull; i < g.count && i < NN__EXPERT__GROUPS_MAX; ++i) {
        const uint64_t row_bytes = nn__turboquant__row_bytes(g.d[i], cols);
        nn__expert__zzprivate_rows_int8(out + g.out_at[i], 0, (const uint8_t*)(uintptr_t)g.codes[i], g.out_rows[i] * row_bytes,
                                        (const uint8_t*)(uintptr_t)g.luts[i], g.out_rows[i] * 2ull, x, rows + 2ull * g.pairs_at[i],
                                        0, g.pairs[i], g.d[i], g.out_rows[i], cols, over);
    }
}
static __device__ inline void nn__expert__zzabi_body_rows_sum_int8(float* acc, const uint8_t* weights, uint64_t w_room,
                                                                   const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                                   const uint32_t* rows, const uint16_t* w, uint64_t count,
                                                                   uint64_t d, uint64_t out_rows, uint64_t cols) {
    nn__expert__zzprivate_rows_int8(0, acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, 0);
}


/* Little-endian, a byte at a time. ⛳ NOT A `uint16_t` LOAD THROUGH A CAST: a row begins at a stride
 * the producer aligns to two, but this verb is not the one that gets to assume it — an unaligned load
 * is undefined where this is merely slower, and the decode is not the hot path.
 * ⛳ IT WAS A 32-BIT WORD. ▶ `row_bytes` for why the container narrowed, and for the
 * measurement that says a denser packing would be faster on paper and slower on the card. */
static __device__ inline uint32_t nn__turboquant__zzprivate_unit(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}


/* ⭐ ONE BLOCK: the sign flip, the butterfly, the normalisation — over `out[0 .. b)` reading `sign[0 .. b)`.
 *
 * ⛔⛔ THE SIGN VECTOR IS READ FROM ITS **PREFIX** FOR EVERY BLOCK, NOT FROM AN OFFSET INTO IT. ⚖ The
 * ruled plane is 4096 signs for the WHOLE MODEL, so a vector of 18,944 has nowhere further along to read
 * — and nothing needs the patterns to differ between blocks, because the flip's job is to break alignment
 * with the Hadamard basis WITHIN one. ⇒ ★ SO THE SIGN BUFFER ONLY EVER NEEDS TO BE ONE BLOCK WIDE, which
 * is what makes a 512-byte plane enough for any width. The caller's bound is checked against the widest
 * BLOCK and not against `n`. */
static __device__ inline void nn__hadamard__zzpackage_one_block(uint16_t* out, const uint16_t* in,
                                                                const uint16_t* sign, uint64_t b,
                                                                bool* over) {
    for (uint64_t i = 0ull; i < b; ++i) {
        out[i] = nn__primitives__zzpackage_float_to_half(
                     nn__silicon__half_to_float(in[i])
                   * nn__silicon__half_to_float(sign[i]), over);
    }
    for (uint64_t h = 1ull; h < b; h <<= 1) {
        for (uint64_t i = 0ull; i < b; i += (h << 1)) {
            for (uint64_t j = i; j < i + h; ++j) {
                const float a = nn__silicon__half_to_float(out[j]);
                const float c = nn__silicon__half_to_float(out[j + h]);
                out[j]     = nn__primitives__zzpackage_float_to_half(a + c, over);
                out[j + h] = nn__primitives__zzpackage_float_to_half(a - c, over);
            }
        }
    }
    const float inv = 1.0f / nn__silicon__sqrtf((float)b);
    for (uint64_t i = 0ull; i < b; ++i) {
        out[i] = nn__primitives__zzpackage_float_to_half(
                     nn__silicon__half_to_float(out[i]) * inv, over);
    }
}

/* ⭐⭐ THE TWO FAMILIES DIFFER IN ONE EXPRESSION, AND THAT IS THE WHOLE REASON THIS WAS CHEAP. The bit
 * extraction, the word shape, the sign-in-bit-0 convention and the parameter-block addressing are
 * IDENTICAL; only what the magnitude index MEANS changes — an entry of a table, or a count of steps.
 * ⇒ ★ A SECOND FAMILY THAT REUSES THE BIT LAYOUT COSTS A BRANCH, NOT A CODEC. ⛳ And the branch is on
 * a value that is constant for the whole row, so it hoists out of the loop the way every other
 * per-row decision in this file already does. */
static __device__ inline float nn__turboquant__zzpackage_level(uint64_t d, uint32_t code) {
    if (d == 16ull) return nn__silicon__half_to_float((uint16_t)code);
    return nn__silicon__half_to_float(nn__turboquant__zzprivate_gaussian[nn__turboquant__zzprivate_gaussian_at(d) + code]);
}

static __device__ inline float nn__turboquant__zzpackage_weight(const uint8_t* row_base, uint64_t d,
                                                          const uint8_t* lut_base, uint64_t c) {
    const uint64_t per   = nn__turboquant__values_per_container(d);
    const uint64_t unit_i = c / per;
    const uint64_t slot   = c - unit_i * per;
    const uint32_t unit   = nn__turboquant__zzprivate_unit(row_base + unit_i * 2ull);
    const uint32_t code   = (unit >> (uint32_t)(d * slot)) & (uint32_t)((1ull << d) - 1ull);
    /* ⛳ `D == 16` IS ONE VALUE A CONTAINER, so the arithmetic above already located it and the only
     * thing that differs is what the bits MEAN: the code IS the fp16, not an index into anything.
     * ⇒ ★ A SHAPE BRANCH AND AN INTERPRETATION BRANCH ARE DIFFERENT THINGS, AND ONLY THE SECOND IS
     * NEEDED HERE. `row_bytes` locates every width with one expression; this branch cannot go, because
     * key-as-value is a different claim about the bits and not a different way of finding them. */
    if (d == 16ull) return nn__silicon__half_to_float((uint16_t)code);
    /* ⭐⭐ THE WHOLE CODE INDEXES A SIGNED TABLE — there is no sign bit, and that recovered bit is the
     * family's entire reason for existing: `2^D` levels out of `D` bits where the retired codebook
     * family spent bit 0 on a sign and indexed `2^(D-1)` magnitudes.
     * ⛳ The table is a build constant — no VRAM read, and the same value for every lane of a wave —
     * and the row's parameter block is ONE fp16, its scale. ▶ `turboquant__header.cuh` for the measurement
     * that says dropping the per-row codebook is where the format's bits actually came from. */
    const uint16_t entry = nn__turboquant__zzprivate_gaussian[nn__turboquant__zzprivate_gaussian_at(d) + code];
    const float scale = nn__silicon__half_to_float(
                            (uint16_t)((uint32_t)lut_base[0] | ((uint32_t)lut_base[1] << 8)));
    return nn__silicon__half_to_float(entry) * scale;
}

/* ── THE PER-ROW LOCATION AND ITS BOUND ──────────────────────────────────────────────────────────────
 * ⛔⛔ THIS IS THE GUARD THAT STOPS A ROW READING SOMEBODY ELSE'S SPAN AND ANSWERING HAPPILY. Written
 * twice it would be checked twice and could be WEAKER in one of them — so the decode and the gemv
 * share it, and a bound that holds in the more permissive of two places is a bound that holds.
 *
 * ⭐⭐ IT TAKES NO PLANES ANY MORE, AND THAT IS THE WHOLE OF THE COLLAPSE. With ONE width
 * for the tensor, three of the five planes were pure arithmetic and existed only so a row could
 * differ from its neighbour:
 * ```
 *     vbr_offsets[r]     == r * row_bytes          a stride
 *     vbr_lut_offsets[r] == r                      one fp16 a row, in order
 *     bitrates[r]        == (family << 5) | d      one constant, now an argument
 * ```
 * `MEASURED`: they cost 9 bytes a row — **0.1406 b/w at 512 columns, 0.0352 at 2048** — which is MORE
 * than the per-row width ladder they existed to support could ever have earned (<= 0.058 b/w over 16
 * real 35B tensors, and 0.000 on most). ⇒ ★★ THE KNOB COST MORE IN BOOKKEEPING THAN IT EARNED IN
 * CODES, and the gap is widest on the narrow rows where the ladder was weakest.
 * ⛳ THE ARITHMETIC REPLACING THEM IS ALSO A BOUND THAT CANNOT BE FOOLED BY DATA: an offset plane is a
 * number a producer wrote and this verb had to distrust; `r * row_bytes` is a number this verb
 * computed. ⇒ ★ A FIELD YOU DERIVE NEEDS NO VALIDATION, WHICH IS A SECOND SAVING NOBODY PRICED. */
static __device__ inline bool nn__turboquant__zzpackage_row(const uint8_t* weights, uint64_t w_room,
                                                      const uint8_t* luts, uint64_t l_room,
                                                      uint64_t d, uint64_t r, uint64_t cols,
                                                      const uint8_t** row_base,
                                                      const uint8_t** lut_base) {
    const uint64_t need = nn__turboquant__row_bytes(d, cols);
    /* ⛳ DIVISION, NOT MULTIPLICATION, SO NOTHING CAN WRAP INTO THE ROOM: `(r+1)*need <= w_room` is
     * the test, and `r+1 <= w_room/need` is the same test with no product to overflow. */
    if (need == 0ull || need > w_room || r + 1ull > w_room / need) return false;
    *row_base = weights + r * need;
    /* ⛔ A `D == 16` ROW HAS NO PARAMETER BLOCK AT ALL — key-as-value carries its own magnitude, so
     * there is no scale to point at and asking for one would bound-check a plane the producer had no
     * reason to write. */
    if (d == 16ull) { *lut_base = luts; return true; }
    if (l_room < 2ull || r + 1ull > l_room / 2ull) return false;
    *lut_base = luts + r * 2ull;
    return true;
}

/* ── ⭐⭐ THE TWO CONVERSIONS, AND THEY DO NOT SHARE AN ARGUMENT ──────────────────────────────────────
 * ⚖ A BODY READS A HALF THROUGH THE SEAM, `nn__silicon__half_to_float`, AND THE ONE BELOW IS
 * WHAT THE HOST'S OWN CODE READS WITH. The exactness argument below holds — every correct conversion gives
 * the same bits, a signalling NaN aside — and it is exactly why crossing costs nothing: the families differ in speed and in
 * nothing else, and at the fp16 matvec the speed was 188 against 570 GB/s. ▶ `contracts/abi/gpu.cuh`.
 * ⛔⛔ THE REASON `half_to_float` WAS NOT A SEAM VERB DOES **NOT** TRANSFER TO ITS INVERSE, AND A READER
 * FOLLOWING IT WOULD REACH THE WRONG CONCLUSION. The stated argument was *"every fp16 is exactly
 * representable in fp32, so this rounds nothing … an exact conversion has no backend"*. Going the
 * other way DISCARDS 13 mantissa bits and must decide what to do with them.
 * ⇒ ★★ A TRUE CONCLUSION BY A FALSE ROUTE — the conclusion still holds, for a different reason:
 * round-to-nearest-even is fixed by IEEE 754, so the answer is determined rather than backend-chosen,
 * and a hand-written version is the SAME answer everywhere while `__float2half_rn` is one more name
 * each backend must publish. ⛳ The seam's rule is "what crosses is what DIFFERS", and nothing differs.
 * ⛔ WHAT WOULD FALSIFY THAT: a backend whose float->half is not RNE. Nothing in HIP or CUDA is. */
static __device__ inline float nn__primitives__zzpackage_half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;                                            /* +0 / -0 */
        } else {
            /* A half subnormal is `mant * 2^-24`. Find its top set bit, and that exponent and the
             * remainder below it are a normal float — no loop of decrements, no off-by-one to make. */
            uint32_t top = 0u, walk = mant;
            while ((walk >> 1) != 0u) { walk >>= 1; top += 1u; }
            bits = sign | ((uint32_t)((int)top - 24 + 127) << 23)
                        | ((mant - (1u << top)) << (23u - top));
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);                   /* inf, and NaN with its payload */
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);          /* 127 - 15 = 112 */
    }
    union { uint32_t u; float f; } cast;                            /* the one legal pun in C++ here */
    cast.u = bits;
    return cast.f;
}

/* ⛔ THE SUBNORMAL ARM IS NOT DEAD WEIGHT HERE EITHER, AND IT IS THE ARM A TEST IS LEAST LIKELY TO
 * REACH: a quantised weight is `level * scale`, and a small level times a small scale lands under
 * 2^-14 routinely. Flushing those to zero would zero the SMALLEST weights, which is the direction an
 * error hides best — the pooled metrics this codec is judged by would not move. */
static __device__ inline uint16_t nn__primitives__zzpackage_float_to_half(float f, bool* over) {
    union { float f; uint32_t u; } cast;
    cast.f = f;
    const uint32_t bits = cast.u;
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t e    = (bits >> 23) & 0xFFu;
    uint32_t       m    = bits & 0x7FFFFFu;

    if (e == 0xFFu) {
        /* ⛳ AN INPUT THAT IS ALREADY inf OR NaN IS NOT AN OVERFLOW OF THIS CONVERSION — it arrived
         *   broken, and flagging it here would report the wrong verb. Passed through unchanged. */
        return (uint16_t)(sign | 0x7C00u | (m != 0u ? ((m >> 13) | 0x200u) : 0u));
    }
    int32_t he = (int32_t)e - 127 + 15;
    if (he >= 0x1F) {                                  /* larger than a half can hold */
        if (over) *over = true;
        return (uint16_t)(sign | 0x7C00u);
    }
    if (he > 0) {
        uint32_t mh  = m >> 13;
        const uint32_t rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (mh & 1u) != 0u)) {
            mh += 1u;
            if (mh == 0x400u) {                         /* the round carried out of the mantissa */
                mh = 0u;
                he += 1;
                /* ⛳ AND THE CARRY PATH FLAGS TOO. A value just under the ceiling that ROUNDS over it
                 *   is an overflow by exactly the same argument, and it is the arm a test is least
                 *   likely to reach. */
                if (he >= 0x1F) { if (over) *over = true; return (uint16_t)(sign | 0x7C00u); }
            }
        }
        return (uint16_t)(sign | ((uint32_t)he << 10) | mh);
    }
    if (he < -10) return (uint16_t)sign;                /* below half of the smallest subnormal */
    m |= 0x800000u;                                     /* the implicit one, now explicit */
    const uint32_t shift = (uint32_t)(14 - he);
    uint32_t mh = m >> shift;
    const uint32_t rem  = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (mh & 1u) != 0u)) mh += 1u;
    /* ⛳ `mh` REACHING 0x400 HERE IS CORRECT AND NEEDS NO CARRY: it lands in the exponent field as 1,
     * which is exactly the smallest NORMAL half, and that is the right answer for a subnormal that
     * rounded up past the top of its range. */
    return (uint16_t)(sign | mh);
}

static __device__ inline uint64_t nn__turboquant__values_per_container(uint64_t d) {
    return 16ull / d;                                   /* past `rate_is_known`, never zero */
}

static __device__ inline uint64_t nn__turboquant__row_bytes(uint64_t d, uint64_t cols) {
    const uint64_t per = nn__turboquant__values_per_container(d);
    return ((cols + per - 1ull) / per) * 2ull;
}

/* ══ ⭐⭐ THE KV CACHE IN TIERS — rotated from the start, then TurboQuant 8 and 4 as positions age ══════════════
 * ⚖ *"rotated from the start, tq8 for the warm tier … and tq4 for the cold (the remaining part)"* · *"the most recent
 * quarter, excluding the first 256 tokens and the most recent 4k tokens"*. A head's key and value are kept as R·k and
 * R·v, R the rotation over one head's width, and `q·k = (R·q)·(R·k)`: the query is rotated once a step and every key
 * read as it lies, and the weighted sum of rotated values is un-rotated once a step. Three things are new to nn: the
 * rotation a head at a time, the codec's quantisation of a row that is already rotated, and a residual whose keys and
 * values are TurboQuant rows. */

/* The rotation in blocks of `block` (at most 512, a power of two), each on its own; `inverse` applies Rᵀ = S ⊙ H·u
 * rather than R = H·(S ⊙ u). `out` and `in` must not overlap: a lane reads the whole block it writes into. */
static __device__ inline void nn__hadamard__zzabi_body_blocks(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                               uint64_t n, uint64_t block, uint64_t inverse,
                                                               unsigned int* over) {
    const float norm = 1.0f / nn__silicon__sqrtf((float)(uint32_t)block);
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) {
        const uint64_t a = i - i % block, r = i - a;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint64_t c = 0ull; c < block; c += 2ull) {
            uint64_t p0 = r & c, p1 = r & (c + 1ull);
            p0 ^= p0 >> 8; p0 ^= p0 >> 4; p0 ^= p0 >> 2; p0 ^= p0 >> 1;
            p1 ^= p1 >> 8; p1 ^= p1 >> 4; p1 ^= p1 >> 2; p1 ^= p1 >> 1;
            float v0 = nn__silicon__half_to_float(in[a + c]), v1 = nn__silicon__half_to_float(in[a + c + 1ull]);
            if (inverse == 0ull) {
                v0 *= nn__silicon__half_to_float(sign[c]);
                v1 *= nn__silicon__half_to_float(sign[c + 1ull]);
            }
            s0 += (p0 & 1ull) ? -v0 : v0;
            s1 += (p1 & 1ull) ? -v1 : v1;
        }
        float sum = (s0 + s1) * norm;
        if (inverse != 0ull) sum *= nn__silicon__half_to_float(sign[r]);
        bool hit = false;
        out[i] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* The codec's own clip for a width: a row's scale is `fp16(clip · rms)`. Only the cache's two widths. */
#define NN__TURBOQUANT__CLIP_4  2.7325895709952577f
#define NN__TURBOQUANT__CLIP_8  4.6036453045214065f
#define NN__TURBOQUANT__FP16_MIN_NORMAL 6.103515625e-05f

/* A code: how many of the level table's midpoints lie below `v` — the codec's `searchsorted(mid, v)`. */
static __device__ inline uint32_t nn__turboquant__zzprivate_nearest(uint64_t d, float v) {
    uint32_t lo = 0u, hi = (1u << (uint32_t)d) - 1u;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const float m = 0.5f * (nn__turboquant__zzpackage_level(d, mid) + nn__turboquant__zzpackage_level(d, mid + 1u));
        if (m < v) lo = mid + 1u; else hi = mid;
    }
    return lo;
}

/* `rows` rows of `cols` halves, already rotated, as TurboQuant rows of width `d` (4 or 8): a block a row, its scale
 * from the row's rms, a lane a container. A row whose scale is below a normal half is stored as zeros, as the codec
 * stores one. */
static __device__ inline void nn__turboquant__zzabi_body_encode_rows(uint8_t* codes, uint16_t* scales, const uint16_t* x,
                                                                     uint64_t rows, uint64_t cols, uint64_t d,
                                                                     unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t per = nn__turboquant__values_per_container(d), row_bytes = nn__turboquant__row_bytes(d, cols);
    const uint64_t units = row_bytes / 2ull;
    const float clip = d == 4ull ? NN__TURBOQUANT__CLIP_4 : NN__TURBOQUANT__CLIP_8;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        const uint16_t* xr = x + r * cols;
        float part = 0.0f;
        for (uint64_t c = lane; c < cols; c += lanes) {
            const float v = nn__silicon__half_to_float(xr[c]);
            part += v * v;
        }
        const float rms = nn__silicon__sqrtf(nn__silicon__lanes_sum(part) / (float)cols);
        bool hit = false;
        uint16_t sh = nn__primitives__zzpackage_float_to_half(clip * rms, &hit);
        float scale = nn__silicon__half_to_float(sh);
        const bool dead = scale < NN__TURBOQUANT__FP16_MIN_NORMAL;
        if (dead) { sh = 0u; scale = 0.0f; }
        if (lane == 0ull) scales[r] = sh;
        uint8_t* cr = codes + r * row_bytes;
        for (uint64_t u = lane; u < units; u += lanes) {
            uint32_t unit = 0u;
            for (uint64_t s = 0ull; s < per; ++s) {
                const uint64_t c = u * per + s;
                if (c >= cols || dead) continue;
                unit |= nn__turboquant__zzprivate_nearest(d, nn__silicon__half_to_float(xr[c]) / scale) << (uint32_t)(d * s);
            }
            cr[u * 2ull] = (uint8_t)(unit & 0xffu);
            cr[u * 2ull + 1ull] = (uint8_t)((unit >> 8) & 0xffu);
        }
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* One element of a TurboQuant row: its level times the row's scale. */
static __device__ inline float nn__turboquant__zzprivate_element(const uint8_t* row, float scale, uint64_t d, uint64_t c) {
    const uint64_t per = nn__turboquant__values_per_container(d);
    const uint32_t unit = (uint32_t)row[(c / per) * 2ull] | ((uint32_t)row[(c / per) * 2ull + 1ull] << 8);
    return scale * nn__turboquant__zzpackage_level(d, (unit >> (uint32_t)(d * (c % per))) & ((1u << (uint32_t)d) - 1u));
}

/* A residual's first pass over TurboQuant keys: `attention_weights`, a key row `t · kv_heads + g`. `q` is rotated. */
static __device__ inline void nn__attention__zzabi_body_weights_tq(float* p, float* res, const uint16_t* q,
                                                                    const uint8_t* codes, const uint16_t* scales,
                                                                    uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                    uint64_t length, uint64_t d) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t group = q_heads / kv_heads, row_bytes = nn__turboquant__row_bytes(d, head_dim);
    const float inv = 1.0f / nn__silicon__sqrtf((float)(uint32_t)head_dim);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint16_t* qh = q + h * head_dim;
        const uint64_t g = h / group;
        float* ph = p + h * length;
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t t = lane; t < length; t += lanes) {
            const uint64_t row = t * kv_heads + g;
            const uint8_t* kr = codes + row * row_bytes;
            const float scale = nn__silicon__half_to_float(scales[row]);
            float dot = 0.0f;
            for (uint64_t e = 0ull; e < head_dim; ++e)
                dot += nn__silicon__half_to_float(qh[e]) * nn__turboquant__zzprivate_element(kr, scale, d, e);
            ph[t] = dot * inv;
            if (ph[t] > top) top = ph[t];
        }
        top = nn__silicon__lanes_max(top);
        float part = 0.0f;
        for (uint64_t t = lane; t < length; t += lanes) {
            ph[t] = nn__silicon__expf(ph[t] - top);
            part += ph[t];
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (lane == 0ull) {
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 1ull] = sum;
        }
    }
}

/* Its second pass: `acc`, the weights times the TurboQuant values. */
static __device__ inline void nn__attention__zzabi_body_residual_mix_tq(float* res, const float* p, const uint8_t* codes,
                                                                         const uint16_t* scales, uint64_t q_heads,
                                                                         uint64_t kv_heads, uint64_t head_dim,
                                                                         uint64_t length, uint64_t d) {
    const uint64_t group = q_heads / kv_heads, row_bytes = nn__turboquant__row_bytes(d, head_dim);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        const uint64_t g = h / group;
        float* acc = res + h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 2ull;
        for (uint64_t e = nn__silicon__lane(); e < head_dim; e += nn__silicon__lanes()) {
            float sum = 0.0f;
            for (uint64_t t = 0ull; t < length; ++t) {
                const uint64_t row = t * kv_heads + g;
                sum += ph[t] * nn__turboquant__zzprivate_element(codes + row * row_bytes,
                                                                  nn__silicon__half_to_float(scales[row]), d, e);
            }
            acc[e] = sum;
        }
    }
}

/* ══ ⭐⭐ THE HYPER-CONNECTION — a residual of several streams, collapsed before a sublayer and mixed after ══════════
 * Manifold-constrained hyper-connections (GLM 5.3 Flash, DeepSeek V4): the residual is `mult` streams of `hidden`.
 * Before a sublayer, one small map of the streams — rms-normalised, flattened — gives three things: `pre` (a weight a
 * stream, the sublayer's input their weighted sum), `post` (where its output goes, a weight a stream) and `comb` (how the
 * streams mix), made doubly stochastic by Sinkhorn's alternating normalisation. After it, stream j becomes
 * `post_j · y + Σ_i comb_ij · stream_i`. ⛳ Every number but the map is a handful, so one block does it all. */
#define NN__HYPER__MULT_MAX 4u
#define NN__HYPER__MIX(m)   ((2u + (m)) * (m))

/* One block: the rms of the flattened streams, the map's rows against them, the three weights, the collapse into `out`.
 * `mix` receives `pre` (m) · `post` (m) · `comb` (m·m, row i the stream mixed from), as floats. `fn`, `base` and `scale`
 * are halves: the map's `mix × m·hidden` weights, its `mix` biases and its three scales. */
/* ⭐ THE MAP'S LOGITS, A BLOCK A ROW: `lg[r] = (fn_r · streams) / rms(streams)` — the four streams flattened. Each block
 * takes the rms itself, which is one more read of the streams, rather than waiting on another launch for it. */
static __device__ inline void nn__hyper__zzabi_body_logits(float* lg, const uint16_t* streams, const uint16_t* fn,
                                                            uint64_t hidden, uint64_t mult, float norm_eps) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), n = mult * hidden, rows = NN__HYPER__MIX(mult);
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        float sq = 0.0f, dot = 0.0f;
        for (uint64_t c = lane; c < n; c += lanes) {
            const float v = nn__silicon__half_to_float(streams[c]);
            sq += v * v;
            dot += nn__silicon__half_to_float(fn[r * n + c]) * v;
        }
        const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + norm_eps);
        const float got = nn__silicon__lanes_sum(dot) * inv;
        if (lane == 0ull) lg[r] = got;
    }
}

/* The weights from the logits `lg` (at `mix + (2 + mult)·mult`): `pre = σ(·scale₀ + base) + eps`, `post = 2σ(·scale₁ + base)`,
 * `comb` a row-softmax then `iters` Sinkhorn steps — written into `mix` by the first index alone — and the collapse
 * `out = Σ pre_i · stream_i`, wide: every index takes `pre` for itself, four sigmoids. */
/* `pre`, a weight a stream, from the logits: every index of a collapse takes it for itself. */
static __device__ inline void nn__hyper__zzprivate_pre(float* pre, const float* lg, const uint16_t* base, float s0,
                                                       uint64_t mult, float eps) {
    for (uint64_t i = 0ull; i < mult; ++i)
        pre[i] = 1.0f / (1.0f + nn__silicon__expf(-(lg[i] * s0 + nn__silicon__half_to_float(base[i])))) + eps;
}

/* `pre`, `post` and `comb` into `mix`, by one index of a collapse. */
static __device__ inline void nn__hyper__zzprivate_weights(float* mix, const float* lg, const float* pre, const uint16_t* base,
                                                           float s1, float s2, uint64_t mult, uint64_t iters, float eps) {
    float post[NN__HYPER__MULT_MAX], comb[NN__HYPER__MULT_MAX * NN__HYPER__MULT_MAX];
    for (uint64_t i = 0ull; i < mult; ++i)
        post[i] = 2.0f / (1.0f + nn__silicon__expf(-(lg[mult + i] * s1 + nn__silicon__half_to_float(base[mult + i]))));
    for (uint64_t i = 0ull; i < mult; ++i) {                /* each row a softmax over its columns, plus eps */
        float top = -NN__KERNELS__INFINITY, sum = 0.0f;
        for (uint64_t j = 0ull; j < mult; ++j) {
            const uint64_t k = 2ull * mult + i * mult + j;
            comb[i * mult + j] = lg[k] * s2 + nn__silicon__half_to_float(base[k]);
            if (comb[i * mult + j] > top) top = comb[i * mult + j];
        }
        for (uint64_t j = 0ull; j < mult; ++j) { comb[i * mult + j] = nn__silicon__expf(comb[i * mult + j] - top); sum += comb[i * mult + j]; }
        for (uint64_t j = 0ull; j < mult; ++j) comb[i * mult + j] = comb[i * mult + j] / sum + eps;
    }
    for (uint64_t it = 0ull; it < iters; ++it) {            /* the columns, then — past the first — rows and columns */
        if (it != 0ull)
            for (uint64_t i = 0ull; i < mult; ++i) {
                float sum = 0.0f;
                for (uint64_t j = 0ull; j < mult; ++j) sum += comb[i * mult + j];
                for (uint64_t j = 0ull; j < mult; ++j) comb[i * mult + j] /= sum + eps;
            }
        for (uint64_t j = 0ull; j < mult; ++j) {
            float sum = 0.0f;
            for (uint64_t i = 0ull; i < mult; ++i) sum += comb[i * mult + j];
            for (uint64_t i = 0ull; i < mult; ++i) comb[i * mult + j] /= sum + eps;
        }
    }
    for (uint64_t i = 0ull; i < mult; ++i) { mix[i] = pre[i]; mix[mult + i] = post[i]; }
    for (uint64_t k = 0ull; k < mult * mult; ++k) mix[2ull * mult + k] = comb[k];
}

static __device__ inline void nn__hyper__zzabi_body_pre(uint16_t* out, float* mix, const uint16_t* streams,
                                                         const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t mult,
                                                         uint64_t iters, float eps, unsigned int* over) {
    const float* lg = mix + NN__HYPER__MIX(mult);
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    float pre[NN__HYPER__MULT_MAX];
    nn__hyper__zzprivate_pre(pre, lg, base, s0, mult, eps);
    const uint64_t first = nn__kernels__zzprivate_first();
    if (first == 0ull) nn__hyper__zzprivate_weights(mix, lg, pre, base, s1, s2, mult, iters, eps);
    for (uint64_t e = first; e < hidden; e += nn__kernels__zzprivate_stride()) {
        float sum = 0.0f;
        for (uint64_t i = 0ull; i < mult; ++i) sum += pre[i] * nn__silicon__half_to_float(streams[i * hidden + e]);
        bool hit = false;
        out[e] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* After the sublayer: stream j becomes `post_j · y + Σ_i comb_ij · stream_i`, an element of every stream a lane — so `out`
 * may be `streams` itself. */
static __device__ inline void nn__hyper__zzabi_body_post(uint16_t* out, const uint16_t* streams, const uint16_t* y, const float* mix,
                                                          uint64_t hidden, uint64_t mult, unsigned int* over) {
    for (uint64_t e = nn__kernels__zzprivate_first(); e < hidden; e += nn__kernels__zzprivate_stride()) {
        float s[NN__HYPER__MULT_MAX];
        for (uint64_t i = 0ull; i < mult; ++i) s[i] = nn__silicon__half_to_float(streams[i * hidden + e]);
        const float ye = nn__silicon__half_to_float(y[e]);
        for (uint64_t j = 0ull; j < mult; ++j) {
            float sum = mix[mult + j] * ye;
            for (uint64_t i = 0ull; i < mult; ++i) sum += mix[2ull * mult + i * mult + j] * s[i];
            bool hit = false;
            out[j * hidden + e] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ⭐ THE SAME THREE OVER `rows` ROWS — a row a position, its streams `mult · hidden` halves, its output `hidden`, and its
 * weights a record of `mix_stride` floats: `pre · post · comb`, then the logits the map gives. */
static __device__ inline void nn__hyper__zzabi_body_logits_rows(float* lg, const uint16_t* streams, const uint16_t* fn,
                                                                 uint64_t hidden, uint64_t mult, float norm_eps, uint64_t rows,
                                                                 uint64_t lg_stride) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), n = mult * hidden, mix = NN__HYPER__MIX(mult);
    for (uint64_t tr = nn__silicon__block(); tr < rows * mix; tr += nn__silicon__blocks()) {
        const uint64_t t = tr / mix, r = tr - t * mix;
        const uint16_t* st = streams + t * n;
        float sq = 0.0f, dot = 0.0f;
        for (uint64_t c = lane; c < n; c += lanes) {
            const float v = nn__silicon__half_to_float(st[c]);
            sq += v * v;
            dot += nn__silicon__half_to_float(fn[r * n + c]) * v;
        }
        const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + norm_eps);
        const float got = nn__silicon__lanes_sum(dot) * inv;
        if (lane == 0ull) lg[t * lg_stride + r] = got;
    }
}

static __device__ inline void nn__hyper__zzabi_body_pre_rows(uint16_t* out, float* mix, const uint16_t* streams,
                                                              const uint16_t* base, const uint16_t* scale, uint64_t hidden,
                                                              uint64_t mult, uint64_t iters, float eps, uint64_t rows,
                                                              uint64_t mix_stride, unsigned int* over) {
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    for (uint64_t i = nn__kernels__zzprivate_first(); i < rows * hidden; i += nn__kernels__zzprivate_stride()) {
        const uint64_t t = i / hidden, e = i - t * hidden;
        float* mt = mix + t * mix_stride;
        const float* lg = mt + NN__HYPER__MIX(mult);
        float pre[NN__HYPER__MULT_MAX];
        nn__hyper__zzprivate_pre(pre, lg, base, s0, mult, eps);
        if (e == 0ull) nn__hyper__zzprivate_weights(mt, lg, pre, base, s1, s2, mult, iters, eps);
        const uint16_t* st = streams + t * mult * hidden;
        float sum = 0.0f;
        for (uint64_t k = 0ull; k < mult; ++k) sum += pre[k] * nn__silicon__half_to_float(st[k * hidden + e]);
        bool hit = false;
        out[i] = nn__primitives__zzpackage_float_to_half(sum, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

static __device__ inline void nn__hyper__zzabi_body_post_rows(uint16_t* out, const uint16_t* streams, const uint16_t* y,
                                                               const float* mix, uint64_t hidden, uint64_t mult, uint64_t rows,
                                                               uint64_t mix_stride, unsigned int* over) {
    for (uint64_t ie = nn__kernels__zzprivate_first(); ie < rows * hidden; ie += nn__kernels__zzprivate_stride()) {
        const uint64_t t = ie / hidden, e = ie - t * hidden;
        const float* mt = mix + t * mix_stride;
        const uint16_t* st = streams + t * mult * hidden;
        uint16_t* ot = out + t * mult * hidden;
        float s[NN__HYPER__MULT_MAX];
        for (uint64_t i = 0ull; i < mult; ++i) s[i] = nn__silicon__half_to_float(st[i * hidden + e]);
        const float ye = nn__silicon__half_to_float(y[ie]);
        for (uint64_t j = 0ull; j < mult; ++j) {
            float sum = mt[mult + j] * ye;
            for (uint64_t i = 0ull; i < mult; ++i) sum += mt[2ull * mult + i * mult + j] * s[i];
            bool hit = false;
            ot[j * hidden + e] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ══ ⭐⭐ KDA — KIMI DELTA ATTENTION, ONE STEP, EVERY HEAD (GLM 5.3 Flash's linear attention) ════════════════════════
 * DeltaNet's recurrence with the decay a VECTOR per head, one number per key channel. For head `h` of `d`, with `q`, `k`,
 * `v` cut from the conv's output (the conv and its SiLU already applied):
 *   g_i = lower · σ(e^{A_log[h]} · (f_i + dt_bias_i))   the forget gate, bounded below by `lower`
 *   β   = σ(b[h])
 *   kn  = k / |k| · qs = q / |q| / √d                  (eps 1e-6 under the root)
 *   S   ← diag(e^g) · S · δ = (v − Sᵀkn) · β · S ← S + kn δᵀ          (the model's order: decay first)
 *   o   = rmsnorm(Sᵀqs) · w · σ(gate)                   (eps `eps` in the mean — the gated norm's sigmoid, not SiLU)
 * A block a head and a lane a column `j` of its state, as `deltanet_step`: a lane reads and writes its columns alone. */
static __device__ inline void nn__kda__zzprivate_step_head(float* Sh, const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                                           const uint16_t* f, const uint16_t* dt, float a, float beta,
                                                           const uint16_t* gate, const uint16_t* w, uint16_t* oh, uint64_t d,
                                                           float lower, float eps, bool* hit_out) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float qq = 0.0f, kk = 0.0f;
    for (uint64_t i = lane; i < d; i += lanes) {
        const float qi = nn__silicon__half_to_float(q[i]), ki = nn__silicon__half_to_float(k[i]);
        qq += qi * qi;
        kk += ki * ki;
    }
    const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)d);
    const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
    float oo = 0.0f;
    bool hit = false;
    for (uint64_t j = lane; j < d; j += lanes) {
        float kv = 0.0f;
        for (uint64_t i = 0ull; i < d; ++i) {
            const float gi = lower / (1.0f + nn__silicon__expf(-a * (nn__silicon__half_to_float(f[i]) + nn__silicon__half_to_float(dt[i]))));
            Sh[i * d + j] *= nn__silicon__expf(gi);
            kv += Sh[i * d + j] * nn__silicon__half_to_float(k[i]) * inv_k;
        }
        const float delta = (nn__silicon__half_to_float(v[j]) - kv) * beta;
        float o = 0.0f;
        for (uint64_t i = 0ull; i < d; ++i) {
            const float s = Sh[i * d + j] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
            Sh[i * d + j] = s;
            o += s * nn__silicon__half_to_float(q[i]);
        }
        oh[j] = nn__primitives__zzpackage_float_to_half(o * inv_q, &hit);
        const float oj = nn__silicon__half_to_float(oh[j]);
        oo += oj * oj;
    }
    const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)d + eps);
    for (uint64_t j = lane; j < d; j += lanes) {
        const float gz = nn__silicon__half_to_float(gate[j]);
        oh[j] = nn__primitives__zzpackage_float_to_half(nn__silicon__half_to_float(oh[j]) * scale * nn__silicon__half_to_float(w[j])
                                                        / (1.0f + nn__silicon__expf(-gz)), &hit);
    }
    if (hit) *hit_out = true;
}

static __device__ inline void nn__kda__zzabi_body_step(float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b,
                                                        const uint16_t* dt, const uint16_t* a_log, const uint16_t* gate,
                                                        const uint16_t* w, uint16_t* out, uint64_t heads, uint64_t head_dim,
                                                        float lower, float eps, unsigned int* over) {
    const uint64_t d = head_dim;
    bool hit = false;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[h])));
        nn__kda__zzprivate_step_head(S + h * d * d, conved + h * d, conved + heads * d + h * d, conved + 2ull * heads * d + h * d,
                                     f + h * d, dt + h * d, nn__silicon__expf(nn__silicon__half_to_float(a_log[h])), beta,
                                     gate + h * d, w, out + h * d, d, lower, eps, &hit);
    }
    if (hit) *over = NN__KERNELS__OVERFLOWED;
}

/* ⭐ THE STEP OVER `rows` POSITIONS IN ORDER, ONE LAUNCH: a block a head walks them, its state its own. `conved` is the
 * rows' `q`, then their `k`, then their `v`; row `t` of each, of `f`, `gate` and `out` is `heads · head_dim` wide, of `b`
 * `heads`. */
static __device__ inline void nn__kda__zzabi_body_steps(float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b,
                                                         const uint16_t* dt,
                                                         const uint16_t* a_log, const uint16_t* gate, const uint16_t* w,
                                                         uint16_t* out, uint64_t heads, uint64_t head_dim, float lower, float eps,
                                                         uint64_t rows, unsigned int* over) {
    const uint64_t d = head_dim, width = heads * d;
    const uint16_t* q = conved;
    const uint16_t* k = conved + rows * width;
    const uint16_t* v = conved + 2ull * rows * width;
    bool hit = false;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        const float a = nn__silicon__expf(nn__silicon__half_to_float(a_log[h]));
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint64_t at = t * width + h * d;
            const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[t * heads + h])));
            nn__kda__zzprivate_step_head(S + h * d * d, q + at, k + at, v + at, f + at, dt + h * d, a, beta, gate + at, w,
                                         out + at, d, lower, eps, &hit);
        }
    }
    if (hit) *over = NN__KERNELS__OVERFLOWED;
}

/* ══ ⭐ THE BIASED SIGMOID ROUTER AND THE CLAMPED SWIGLU — GLM 5.3 Flash's MoE ══════════════════════════════════════
 * The router chooses the `k` experts whose σ(logit) + bias is largest — the bias steers the choice and nothing else —
 * and weighs them by σ(logit) alone, normalised to one and scaled. One block, the tie rule `top_k`'s: the lowest index. */
static __device__ inline void nn__vector__zzabi_body_top_k_biased(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                                   const uint16_t* x, const uint16_t* bias, uint64_t n,
                                                                   uint64_t k, float scale) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    for (uint64_t r = 0ull; r < k; ++r) {
        uint64_t best_i = n;
        float    best_v = -NN__KERNELS__INFINITY;
        for (uint64_t i = lane; i < n; i += lanes) {
            bool taken = false;
            for (uint64_t q = 0ull; q < r; ++q) taken = taken || chosen[q] == i;
            const float v = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(x[i]))) + nn__silicon__half_to_float(bias[i]);
            if (taken || v != v) continue;
            if (best_i == n || v > best_v) { best_v = v; best_i = i; }
        }
        const float top = nn__silicon__lanes_max(best_v);
        const float first = nn__silicon__lanes_max(best_i != n && best_v == top ? -(float)(uint32_t)best_i
                                                                                : -NN__KERNELS__INFINITY);
        chosen[r] = first == -NN__KERNELS__INFINITY ? n : (uint64_t)(-first);
    }
    if (lane != 0ull || nn__silicon__block() != 0u) return;
    float s[NN__VECTOR__TOP_K_MAX], sum = 0.0f;
    for (uint64_t r = 0ull; r < k; ++r) {
        s[r] = chosen[r] < n ? 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(x[chosen[r]]))) : 0.0f;
        sum += s[r];
    }
    for (uint64_t r = 0ull; r < k; ++r) {
        values[r * stride] = chosen[r];
        bool hit = false;
        weights[r] = nn__primitives__zzpackage_float_to_half(s[r] / (sum + 1e-20f) * scale, &hit);
    }
}

/* `act = silu(min(g, limit)) · clamp(u, −limit, limit)` — GLM's clamped swiglu — for `pairs` pairs, each its gate then its
 * up, `n` each, as the grouped gate-and-up launch leaves them. */
static __device__ inline void nn__swiglu__zzabi_body_clamped(uint16_t* act, const uint16_t* gu, uint64_t pairs, uint64_t n,
                                                              float limit, unsigned int* over) {
    const uint64_t all = pairs * n;
    for (uint64_t i = nn__kernels__zzprivate_first(); i < all; i += nn__kernels__zzprivate_stride()) {
        const uint64_t p = i / n, j = i - p * n;
        float g = nn__silicon__half_to_float(gu[p * 2ull * n + j]), u = nn__silicon__half_to_float(gu[p * 2ull * n + n + j]);
        g = g > limit ? limit : g;
        u = u > limit ? limit : (u < -limit ? -limit : u);
        bool hit = false;
        act[i] = nn__primitives__zzpackage_float_to_half(g / (1.0f + nn__silicon__expf(-g)) * u, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* ── MULTI-HEAD LATENT ATTENTION — THE KEYS AND VALUES OF EVERY HEAD FROM ONE CACHED LATENT ────────────────
 * `W` is the latent's expansion, `heads · stride` rows of `latent`: a head's `stride` rows are its key rows, then its
 * value rows. A query is ABSORBED into the latent — `Wₖᵀ q` a head — so its scores are dot products with the latent
 * itself, and the latent is the values too; the answer, a latent a head, is EXPANDED by the head's value rows. */

/* `out[h·latent + c] = scale · Σᵢ W[(h·stride + i)·latent + c] · q[h·nope + i]`, a block a head, a lane a column. */
static __device__ inline void nn__attention__zzabi_body_absorb(uint16_t* out, const uint16_t* q, const uint16_t* W,
                                                                uint64_t heads, uint64_t nope, uint64_t latent,
                                                                uint64_t stride, float scale, unsigned int* over) {
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        const uint16_t* qh = q + h * nope;
        for (uint64_t c = nn__silicon__lane(); c < latent; c += nn__silicon__lanes()) {
            const uint16_t* w = W + h * stride * latent + c;
            float sum = 0.0f;
            for (uint64_t i = 0ull; i < nope; ++i)
                sum += nn__silicon__half_to_float(w[i * latent]) * nn__silicon__half_to_float(qh[i]);
            bool hit = false;
            out[h * latent + c] = nn__primitives__zzpackage_float_to_half(sum * scale, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* `out[h·width + j] = Σ_c W[(h·stride + offset + j)·latent + c] · o[h·latent + c]`, a block a row, its lanes summing. */
static __device__ inline void nn__attention__zzabi_body_expand(uint16_t* out, const uint16_t* o, const uint16_t* W,
                                                                uint64_t heads, uint64_t width, uint64_t latent,
                                                                uint64_t stride, uint64_t offset, unsigned int* over) {
    for (uint64_t r = nn__silicon__block(); r < heads * width; r += nn__silicon__blocks()) {
        const uint64_t h = r / width;
        const uint16_t* row = W + (h * stride + offset + r % width) * latent;
        const uint16_t* x = o + h * latent;
        float part = 0.0f;
        for (uint64_t c = nn__silicon__lane(); c < latent; c += nn__silicon__lanes())
            part += nn__silicon__half_to_float(row[c]) * nn__silicon__half_to_float(x[c]);
        const float sum = nn__silicon__lanes_sum(part);
        if (nn__silicon__lane() == 0ull) {
            bool hit = false;
            out[r] = nn__primitives__zzpackage_float_to_half(sum, &hit);
            if (hit) *over = NN__KERNELS__OVERFLOWED;
        }
    }
}

/* ── THE SPARSE ATTENTION'S INDEXER (DeepSeek's DSA, pooled as GLM 5.3 Flash pools it) ─────────────────────────────
 * A query attends only the positions an indexer picks: every position keeps a small key and a gate, a pool of `kpool`
 * positions is scored as one — its key the positions' keys weighed by a softmax of their gates — and the top pools'
 * positions, and the positions of the pool still filling, are the ones attended. These are its pieces. */

/* `o = (x - mean) / √(var + eps) · w + b` — a LayerNorm, over one row of `n`, a block. */
static __device__ inline void nn__index__zzabi_body_layernorm(uint16_t* o, const uint16_t* x, const uint16_t* w, const uint16_t* b,
                                                              uint64_t n, float eps, unsigned int* over) {
    if (nn__silicon__block() != 0ull) return;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float sum = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes) sum += nn__silicon__half_to_float(x[i]);
    const float mean = nn__silicon__lanes_sum(sum) / (float)n;
    float sq = 0.0f;
    for (uint64_t i = lane; i < n; i += lanes) {
        const float d = nn__silicon__half_to_float(x[i]) - mean;
        sq += d * d;
    }
    const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + eps);
    for (uint64_t i = lane; i < n; i += lanes) {
        bool hit = false;
        o[i] = nn__primitives__zzpackage_float_to_half((nn__silicon__half_to_float(x[i]) - mean) * inv * nn__silicon__half_to_float(w[i])
                                                       + nn__silicon__half_to_float(b[i]), &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* Pools `first .. first + pools`: each channel `c` of pool `p` the softmax over its `kpool` positions of gate + ape,
 * weighing their keys — `keys` and `gates` a row of `dim` a position, `ape` a row a place in the pool. */
static __device__ inline void nn__index__zzabi_body_pool(uint16_t* pooled, const uint16_t* keys, const uint16_t* gates,
                                                         const uint16_t* ape, uint64_t first, uint64_t pools, uint64_t dim,
                                                         uint64_t kpool, unsigned int* over) {
    for (uint64_t e = nn__kernels__zzprivate_first(); e < pools * dim; e += nn__kernels__zzprivate_stride()) {
        const uint64_t p = first + e / dim, c = e % dim;
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t i = 0ull; i < kpool; ++i) {
            const float l = nn__silicon__half_to_float(gates[(p * kpool + i) * dim + c]) + nn__silicon__half_to_float(ape[i * dim + c]);
            if (l > top) top = l;
        }
        float den = 0.0f, num = 0.0f;
        for (uint64_t i = 0ull; i < kpool; ++i) {
            const float wgt = nn__silicon__expf(nn__silicon__half_to_float(gates[(p * kpool + i) * dim + c])
                                                + nn__silicon__half_to_float(ape[i * dim + c]) - top);
            den += wgt;
            num += wgt * nn__silicon__half_to_float(keys[(p * kpool + i) * dim + c]);
        }
        bool hit = false;
        pooled[p * dim + c] = nn__primitives__zzpackage_float_to_half(num / den, &hit);
        if (hit) *over = NN__KERNELS__OVERFLOWED;
    }
}

/* A pool's score, a block a pool: `Σ_h w_h / √heads · relu(q_h · K_p · scale)` — `q` a row of `dim` a head. */
static __device__ inline void nn__index__zzabi_body_scores(float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled,
                                                           uint64_t heads, uint64_t dim, uint64_t pools, float scale) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const float per_head = 1.0f / nn__silicon__sqrtf((float)heads);
    for (uint64_t p = nn__silicon__block(); p < pools; p += nn__silicon__blocks()) {
        float part = 0.0f;
        for (uint64_t h = lane; h < heads; h += lanes) {
            float dot = 0.0f;
            for (uint64_t c = 0ull; c < dim; ++c)
                dot += nn__silicon__half_to_float(q[h * dim + c]) * nn__silicon__half_to_float(pooled[p * dim + c]);
            dot *= scale;
            part += (dot > 0.0f ? dot : 0.0f) * nn__silicon__half_to_float(w[h]) * per_head;
        }
        const float got = nn__silicon__lanes_sum(part);
        if (lane == 0ull) scores[p] = got;
    }
}

/* A score as a key that orders as the score does: larger floats, larger keys. */
static __device__ inline uint32_t nn__index__zzprivate_key(float f) {
    union { float f; uint32_t u; } cast;
    cast.f = f;
    return (cast.u & 0x80000000u) != 0u ? ~cast.u : (cast.u | 0x80000000u);
}

/* ⭐ THE TOP `k` OF `pools` SCORES, AS POSITIONS — one block. The `k`-th largest key is found a bit at a time from the top
 * (a count across the lanes a bit), then the pools above it, and those equal to it by lowest index until `k` are
 * chosen, in index order; each chosen pool's `kpool` positions are written to `index`, then the `tail` positions from
 * `tail_first`. Answers the count in `count[0]`. With `pools <= k` every pool is chosen. */
static __device__ inline void nn__index__zzabi_body_select(uint32_t* index, uint32_t* count, const float* scores, uint64_t pools,
                                                           uint64_t k, uint64_t kpool, uint64_t tail_first, uint64_t tail) {
    if (nn__silicon__block() != 0ull) return;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    uint32_t cut = 0u;
    if (pools > k) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t want = cut | (1u << (uint32_t)bit);
            float have = 0.0f;
            for (uint64_t p = lane; p < pools; p += lanes) have += nn__index__zzprivate_key(scores[p]) >= want ? 1.0f : 0.0f;
            if (nn__silicon__lanes_sum(have) >= (float)k) cut = want;
        }
    }
    if (lane != 0ull) return;
    uint64_t above = 0ull, n = 0ull;
    if (pools > k)
        for (uint64_t p = 0ull; p < pools; ++p) above += nn__index__zzprivate_key(scores[p]) > cut ? 1ull : 0ull;
    uint64_t ties = pools > k ? k - above : 0ull;
    for (uint64_t p = 0ull; p < pools; ++p) {
        bool take = pools <= k;
        if (!take) {
            const uint32_t key = nn__index__zzprivate_key(scores[p]);
            if (key > cut) take = true;
            else if (key == cut && ties != 0ull) { take = true; --ties; }
        }
        if (take)
            for (uint64_t i = 0ull; i < kpool; ++i) index[n++] = (uint32_t)(p * kpool + i);
    }
    for (uint64_t i = 0ull; i < tail; ++i) index[n++] = (uint32_t)(tail_first + i);
    count[0] = (uint32_t)n;
}

/* Rows `index[0 .. n)` of `src`, `row` halves each, into `out` one after another. */
static __device__ inline void nn__index__zzabi_body_gather(uint16_t* out, const uint16_t* src, const uint32_t* index, uint64_t n,
                                                           uint64_t row) {
    for (uint64_t e = nn__kernels__zzprivate_first(); e < n * row; e += nn__kernels__zzprivate_stride())
        out[e] = src[(uint64_t)index[e / row] * row + e % row];
}

/* `out[i] = in[i]` for `n` halves — a gather's rows moved into place. */
static __device__ inline void nn__vector__zzabi_body_copy(uint16_t* out, const uint16_t* in, uint64_t n) {
    for (uint64_t i = nn__kernels__zzprivate_first(); i < n; i += nn__kernels__zzprivate_stride()) out[i] = in[i];
}

#endif /* SILVANN__PACKAGES_NN_GPU_KERNELS_KERNELS_CUH */
