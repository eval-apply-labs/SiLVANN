#ifndef SILVANN__PACKAGES_NN_CPU_SWIGLU__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_SWIGLU__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/heap_node__header.cuh" /* the node a form's arguments arrive in */
#include "../contracts/objects/swiglu.cuh" /* its constants and fault words */
/* ══ nn — SwiGLU's COMBINE, AND IT IS THIS TREE'S FIRST ARITHMETIC ═══════════════════════════════════
 *
 * ⚖ ARCHITECT: *"let's test it with swiglu combine."* Chosen over `MOE_ROUTELOG` because it
 * is real floating-point arithmetic — `silu(gate) · up` — and so **cannot pass by accident**: an integer
 * oracle checks numbers the host staged itself through plumbing already proven, while this one has a
 * transcendental in it that a reference implementation must independently agree with.
 *
 * ── ⭐⭐ WHAT THE PORT DELETED, WHICH IS MOST OF IT ───────────────────────────────────────────────────
 * The old handler (`MODEL_CMD_SWIGLU_COMBINE.cuh`, 47 lines) is the port thesis in one file:
 * ```
 *   resolve_signature_buffers(...)     GONE — the buffers are ARGUMENTS now
 *   gate = mem_ctx->alt.live           GONE — the gate came from an implicit REGISTER because a
 *                                      handler takes no arguments. ⇒ ★ THE OLD OP'S FIRST ACT WAS TO
 *                                      GO AND FIND AN OPERAND NOBODY HAD PASSED IT.
 *   threadIdx/blockIdx strided loop    GONE — the door runs the body as one block of one thread
 *                                      (`gpu/doors.cuh`), so this is a serial loop
 *   on_device_grid_sync(d_state)       GONE — one thread has nobody to rendezvous with
 *   flip_buffer(...)                   GONE — there is no live/scratch pair to swap; `out` is named
 *   trap() on a null pointer           GONE — a refusal is a value here, not a halt
 * ```
 * ⇒ ★★ **FIVE OF THE SEVEN THINGS THAT HANDLER DID WERE CONSEQUENCES OF HOW IT WAS CALLED, NOT OF WHAT
 * IT COMPUTED.** What survives is the arithmetic and the bounds, which is what the row was always for.
 * ⚠ AND THE SERIAL LOOP IS A PROPERTY OF THE DOOR'S LAUNCH WIDTH, NOT A DESIGN: the day this runs wider
 * it needs a stride and a rendezvous back. It is correct and it is not fast, and those are different claims.
 *
 * ── ⛔ THE CLAMP IS NOT DECORATION ───────────────────────────────────────────────────────────────────
 * `gate` is clamped to [-88, 88] BEFORE the exponential, and the bound is the float's, not the model's:
 * `expf(89)` overflows fp32. Without it a large positive gate gives `g / (1 + 0) = g` by luck and a
 * large negative one gives `g / inf = -0`, so the failure is silent in one direction and wrong in the
 * other. ⛳ The old tree clamps to the same figures and this keeps them.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* `(nn__swiglu__combine <gate> <up> <out> <count>)` — three `nn__buffer`s and a length, and it answers
 * the OUT buffer so a program can chain it. ⛳ IT ANSWERS THE BUFFER IT WAS GIVEN rather than a fresh
 * one: the caller already holds it, allocation is the pool's business, and a verb that quietly allocated
 * would make every chain a leak the caller could not see. */

#endif /* SILVANN__PACKAGES_NN_CPU_SWIGLU__HEADER_CUH */
