#ifndef SILVANN__PACKAGES_NN_CPU_TURBOQUANT__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_TURBOQUANT__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/heap_node__header.cuh" /* the node a form's arguments arrive in */
#include "../contracts/objects/turboquant.cuh" /* its constants and fault words */
/* ══ nn — THE VBR CODEC, IMPORTED AS IT STANDS ═══════════════════════════════════════════════════════
 *
 * ⚖ ARCHITECT: *"do the vbr codec it should be pretty straightforward … let's import it as
 * is for now."* It is, and this is the whole format:
 * ```
 *   per row r:  D = bitrates[r]            the row's BIT WIDTH — and rows of different D sit in one
 *                                          matrix, which is why an expert has no single quant level
 *               N = 32 / D                 codes to a 32-bit word
 *               column c lives in word (c/N) at bit position D*(c%N)
 *               code = (word >> D*slot) & (2^D - 1)
 *               bit 0 is the SIGN; the high D-1 bits index the row's own codebook
 *               value = ±vbr_lut[vbr_lut_offsets[r] + (code >> 1)]
 *   D == 16:    raw fp16, key-as-value, NO codebook at all
 * ```
 * ── ⭐⭐ WHAT A "ROW" IS, BECAUSE THE WORD IS THE WHOLE POINT OF THE LAYOUT ───────────────────────────
 * ⚖ ARCHITECT: *"i do have an activation that i mentally see as a row … i want to do the
 * compression on each column so it is contained within one dot product. is that what i am doing?"*
 * **Yes.** `MEASURED` three ways, and they agree:
 * ```
 *   compressor.py:122    `rows, cols = weight_tensor.shape` on HF's [out_features, in_features]
 *                        ⇒ a ROW is an OUTPUT feature and `cols` is the activation's length
 *   vbr_planar.cuh:906   `row_id < output_dim`, and ONE WARP OWNS ONE ROW
 *   the gemv below       out[r] = Σ_c W[r][c]·x[c], with x exactly `cols` long
 * ```
 * ⇒ ⭐ **ONE COMPRESSION UNIT = ONE OUTPUT ELEMENT = ONE COMPLETE DOT PRODUCT.** The bit width, the
 * codebook and the contiguous byte run are all per-unit and self-describing.
 * ⛳ THE "COLUMN" READING IS THE SAME THING: HF stores the transpose, so column j of a K×N matrix is
 * row j of the N×K one on disk. Same mechanism, different name — and nothing in this file changes if
 * you say column, so long as you mean *the thing that consumes the whole activation to produce one
 * number*.
 * ⇒ ★★ AND THAT IS WHY A KERNEL FOR ONE DOT PRODUCT CAN BE LEAN: it needs ONE `D`, ONE codebook
 * (512 B of smem, loaded once) and ONE contiguous run. **Every branch on bit width sits at the row
 * boundary, never inside the loop** — which is the register pressure a per-term opcode is trying to
 * get out from under.
 *
 * ⛔ **THERE IS NO SCALE AND NO ZERO-POINT.** `MEASURED` from the producer (`compressor.py:173`): the
 * per-row scale is MULTIPLIED INTO the codebook at pack time — `actual_lut = win_centroids * win_scales`
 * — so the LUT holds real magnitudes and the decode is one indexed read. ⇒ ★ A SEPARATE SCALE IS A
 * SECOND NUMBER OBLIGED TO AGREE WITH A FIRST; MULTIPLYING IT IN DELETES IT.
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
 * ⇒ ⭐⭐ **THE REASON THE FOLD EXISTED HAS ALREADY GONE, AND THE FOLD HAS NOT.** `D == 16` rides the same
 * dispatch as the codebook path because the old tree ran ONE kernel over blocks doing different work, so
 * a separate function was a register-union cost paid by every block. Here a verb is a verb. The
 * re-separation is a PERFORMANCE change with the oracle as its gate — ▶ `nn_port.md` §⑭.
 * ⚖ AND THE LARGER SHAPE IT SERVES: *"ideally experts should declare their compression type in their id
 * card so we could mix and match and/or have multiple strategies without costing us performance …
 * maybe even per row."* ⛳ ⚖ SETTLED IT PER TENSOR: one width per tensor, one family (the
 * Gaussian), and the family named by the bundle's per-tensor tag — ▶ `pending/quant_families.md` §④. ⚖ *"i keep my version for non tensor core cards and the turboquant
 * one for those who have the tensor cores."*
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

/* `(nn__turboquant__decode weights lut out rows cols d)` — the TWO planes of a VBR matrix to a dense fp16
 * `[rows*cols]`, answering OUT. ⭐ It was five planes; with one width for the tensor
 * the other three are arithmetic. ▶ the impl, which carries what that was worth.
 * ⛳ IT IS THE WHOLE MATRIX AND NOT ONE ROW, because that is the shape the old tree's standalone decoder
 * has and ⚖ *"import it as is"*. A per-row verb would be the better primitive the day a caller wants one
 * row; nothing wants one yet. */

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
