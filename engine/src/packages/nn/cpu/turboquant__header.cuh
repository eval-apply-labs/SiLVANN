#ifndef SILVANN__PACKAGES_NN_CPU_TURBOQUANT__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_TURBOQUANT__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/heap_node__header.cuh" /* the node a form's arguments arrive in */
#include "../contracts/objects/turboquant.cuh" /* its constants and fault words */
/* ══ nn — THE GAUSSIAN CODEC: A ROTATED ROW, ONE WIDTH A TENSOR, ONE SCALE A ROW ═════════════════════
 *
 * ⚖ ARCHITECT, 2026-09-20: *"build that quant format with the global nf table, fwht row scaler and no
 * block scaler."* · 2026-09-21: *"the old vbr is dead, lets move to the new format fully."* This is the
 * whole format (▶ `src/docs/design/quant_families.md` ①–④, the codec as built):
 * ```
 *   per tensor:  D = the record's `d`         the BIT WIDTH, D ∈ {1,2,3,4,5,8,16}, one for every row
 *   per row r:   N = 16 / D                   codes to a uint16 container
 *                column c lives in container (c/N) at bit position D*(c%N)
 *                code = (container >> D*slot) & (2^D - 1)
 *                value = scale[r] × level_D[code]   the code indexes the SIGNED Gaussian table whole —
 *                                                   2^D levels, no sign bit; scale[r] the row's one fp16,
 *                                                   the `vbr_lut` plane
 *   D == 16:     the code is the weight's own fp16 — no scale, no table, nothing rotated
 * ```
 * The table is a build constant, one block a width (▶ `nn__turboquant__zzpackage_level`), and every row
 * of a quantised tensor is stored rotated, `R(w)` (▶ `hadamard__header.cuh`).
 * ── ⭐⭐ WHAT A "ROW" IS, BECAUSE THE WORD IS THE WHOLE POINT OF THE LAYOUT ───────────────────────────
 * ⚖ ARCHITECT: *"i do have an activation that i mentally see as a row … i want to do the
 * compression on each column so it is contained within one dot product. is that what i am doing?"*
 * **Yes.** `MEASURED` three ways, and they agree:
 * ```
 *   nn_compressor.py:530 `rows, cols = W.shape` on HF's [out_features, in_features]
 *                        ⇒ a ROW is an OUTPUT feature and `cols` is the activation's length
 *   vbr_planar.cuh:906   `row_id < output_dim`, and ONE WARP OWNS ONE ROW
 *   the gemv below       out[r] = Σ_c W[r][c]·x[c], with x exactly `cols` long
 * ```
 * ⇒ ⭐ **ONE COMPRESSION UNIT = ONE OUTPUT ELEMENT = ONE COMPLETE DOT PRODUCT.** The scale and the
 * contiguous byte run are per row; the width is the tensor's.
 * ⛳ THE "COLUMN" READING IS THE SAME THING: HF stores the transpose, so column j of a K×N matrix is
 * row j of the N×K one on disk. Same mechanism, different name — and nothing in this file changes if
 * you say column, so long as you mean *the thing that consumes the whole activation to produce one
 * number*.
 * ⇒ ★★ AND THAT IS WHY A KERNEL FOR ONE DOT PRODUCT CAN BE LEAN: it needs ONE `D`, ONE scale and ONE
 * contiguous run, and the table is a build constant — no memory read, the same for every lane. **Every
 * branch on bit width sits at the row boundary, never inside the loop** — which is the register pressure
 * a per-term opcode is trying to get out from under.
 *
 * ⛔ **THIS FORMAT HAS ONE SCALE A ROW AND NO ZERO-POINT.** The table is normalised to ±1 and shared by
 * every row, so a row carries one fp16: `k_D · rms(row)`, the rms taken after the rotation and `k_D` the
 * table's outermost level in units of σ (▶ `quant_families.md` ③). The table has no zero level, and a row
 * that must be zero takes a scale of 0. There is no block scale.
 *
 * ── ⭐ THE fp16 READ IS ARITHMETIC, NOT A SEAM VERB, AND THAT IS A RULING NOT A SHORTCUT ─────────────
 * Every fp16 is exactly representable in fp32, so the conversion is a BIT REARRANGEMENT with no
 * rounding and no approximation anywhere in it. The seam's rule is that what crosses is what DIFFERS
 * between backends — `expf` differs, a shift does not. ⇒ ★ AN EXACT CONVERSION HAS NO BACKEND, so
 * putting it in the seam would add a verb every backend must keep identical to produce the one answer
 * arithmetic already forces. ⛳ A hardware `__half2float` is marginally faster and bit-identical; that
 * is a PERFORMANCE question and it is not this one.
 *
 * ── ⏳ WHAT IS DELIBERATELY NOT DONE, AND THE ARCHITECT NAMED BOTH ───────────────────────────────────
 * ⚖ *"let's do the fp16 as its own method, we did fold it in with the others for uniformity because we
 * did not want to break the noinline in a monolithic kernel that was computing different things in
 * different blocks, since now each block is autonomous it makes sense to re-separate it. keep it as is
 * for now tho, this is a performance improvement for after the oracle is minted."*
 * ⇒ ⭐⭐ **THE REASON FOR THE FOLD IS GONE, AND THE FOLD STAYS.** `D == 16` rides the same dispatch as the
 * table path, a fold made when ONE kernel ran over blocks doing different work, so a separate function was
 * a register-union cost paid by every block. Here a verb is a verb. The re-separation is a PERFORMANCE
 * change with the oracle as its gate — `test/src_gaussian_oracle.py`.
 * ⚖ AND THE LARGER SHAPE IT SERVES: *"ideally experts should declare their compression type in their id
 * card so we could mix and match and/or have multiple strategies without costing us performance …
 * maybe even per row."* ⛳ ⚖ SETTLED IT PER TENSOR: one width per tensor, one family (the Gaussian), and
 * the family named by each record's `family` scalar — ▶ `src/docs/design/quant_families.md` §⑤ (the width)
 * and §① (the family). ⚖ *"i keep my version for non tensor core cards and the turboquant one for those
 * who have the tensor cores."*
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

static __device__ inline uint64_t nn__turboquant__width(uint64_t rate);
static __device__ inline uint64_t nn__turboquant__family(uint64_t rate);

/* ⛔ THE WIDTHS THE FORMAT DEFINES, AND NOTHING ELSE: D ∈ {1,2,3,4,5,8,16}. ⚖ RULED — q6
 * left (2 a halfword is q8's cost for 15.7x its error) and q1 arrived (16 a halfword, zero waste, and
 * a meme). ⛳ A D the packer cannot emit is a config that disagrees with the file it is reading, so it
 * REFUSES rather than computing a container shape nobody packs — `16 / 7` is 2 with two bits left
 * over, and the leftover is silent.
 *
 * ⭐⭐⭐ AND `D == 16` IS THE WIDTH EVERY UNQUANTISED TENSOR TAKES — ⚖ ARCHITECT: *"turn
 * that escape hatch into the reality, we have abandoned the rest of the old vbr code so let's promote
 * the d==16 as the real user."* At this width the code IS the weight's own fp16 bit pattern: one value
 * a container, `row_bytes == cols * 2`, no scale in the parameter block, no table read, and nothing
 * rotated. ⇒ ★★ IT IS THE ONE WIDTH WHOSE GEOMETRY NEEDS NO BRANCH ANYWHERE — every formula in this
 * file gives the right answer for it — which is what makes it a width rather than a special case.
 * ⛳ SO A NORM, A BIAS AND A 3584-COLUMN PROJECTION ALL TRAVEL AS ORDINARY RECORDS. The three reasons a
 * tensor is not quantised — a policy decision, no contraction axis, or a contraction axis that is not
 * a power of two — are ALL facts about the ROTATION, and there is no rotation here.
 * ⛔ THE ROTATION MUST NOT BE APPLIED AT THIS WIDTH, ON EITHER SIDE. `nn__turboquant__zzpackage_weight`'s
 * `d == 16` arm reads the halfword straight back as a value and does not invert anything, so a producer
 * that rotated would hand over a record decoding to `R(w)` while every consumer believed it held `w`.
 * ⇒ ★ THE TWO SIDES AGREE BY BOTH DOING NOTHING, and the hazard is that one of them does something
 * reasonable. */
static __device__ inline bool nn__turboquant__rate_is_known(uint64_t d);
static __device__ inline uint64_t nn__turboquant__values_per_container(uint64_t d);

/* How many bytes one row of `cols` weights occupies at width D — the producer's arithmetic restated:
 * a row is uint16-aligned, `ceil(cols / (16/D)) * 2`.
 * ⛳ A ROW'S PARAMETER BLOCK IS ONE fp16 — its scale — at every width but `D == 16`, which carries
 * none. So there is nothing to ask about a block's size: `row_bytes` and the row's index are the
 * whole of a row's geometry. */
static __device__ inline uint64_t nn__turboquant__row_bytes(uint64_t d, uint64_t cols);

/* `(nn__turboquant__decode weights lut out rows cols d)` — the TWO planes of a packed matrix, its codes and its
 * scales, to a dense fp16 `[rows*cols]`, answering OUT. ⭐ Two planes because with one width for the tensor
 * every other per-row field is arithmetic. ▶ the impl, which carries what that was worth.
 * ⛳ IT IS THE WHOLE MATRIX AND NOT ONE ROW. A per-row verb would be the better primitive the day a caller
 * wants one row; nothing wants one yet. */

/* `(nn__turboquant__gemv weights lut x out rows cols d)` — `out[r] = Σ W[r][c]·x[c]`
 * with the matrix never expanded. ⭐ THIS IS WHAT THE COMPRESSION IS FOR: `decode` costs `rows × cols`
 * floats of scratch to prove the layout, and a 2048×4096 expert is 32 MiB dense against ~4 MiB packed.
 * ⇒ ★ A CODEC THAT MUST EXPAND BEFORE IT CAN MULTIPLY HAS SPENT THE SAVING IT EXISTS TO MAKE.
 * ⛔ THE REDUCTION IS fp32, AND ITS ORDER IS NOT A PROMISE — `MEASURED`: the hardware reassociates it
 * under plain `-O3`. ▶ the impl, which carries the measurement. */

/* ⭐ DECLARED HERE BECAUSE `kernels.cuh` CALLS THEM, AND IT IS INCLUDED BEFORE THIS PACKAGE'S IMPLS.
 * ⛳ It is the same reason every verb in this tree is declared apart from its definition: the compute
 * bodies are gathered into one file so they can become kernels, and a gathered body still has to reach
 * the row walker that decides where a quantised row starts. ▶ `kernels.cuh` for what that file is for. */
/* What code `code` of a `d`-bit container means before the row's scale: a level of the Gaussian table,
 * or at `d == 16` the half the code is. For a loop that has already taken the code out of its container. */
static __device__ inline float nn__turboquant__zzpackage_level(uint64_t d, uint32_t code);
static __device__ inline float nn__turboquant__zzpackage_weight(const uint8_t* row_base, uint64_t d,
                                                          const uint8_t* lut_base, uint64_t c);
static __device__ inline bool nn__turboquant__zzpackage_row(const uint8_t* weights, uint64_t w_room,
                                                      const uint8_t* luts, uint64_t l_room,
                                                      uint64_t d, uint64_t r, uint64_t cols,
                                                      const uint8_t** row_base,
                                                      const uint8_t** lut_base);

#endif /* SILVANN__PACKAGES_NN_CPU_TURBOQUANT__HEADER_CUH */
