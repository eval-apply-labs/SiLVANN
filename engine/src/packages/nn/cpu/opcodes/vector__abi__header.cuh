#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI__HEADER_CUH
/* ══ THE ADAPTER — HOW A PROGRAM REACHES A NEW-ABI VERB THROUGH THE EVALUATOR ════════════════════════
 */

/* The ruled ABI is
 *     sys__heap_node verb(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
 * and the evaluator's dispatch calls `APPLY(base, form)` — a generated switch in
 * `sys__heap_object__apply` in `sys/cpu/heap_object__impl.cuh`. This file is the bridge between the two shapes.
 *
 * ⛳ THE ADAPTER IS SCAFFOLDING AND SHOULD NOT SURVIVE. It exists so ONE verb can be moved at a time
 * while the other 64 keep working — if the evaluator's dispatch were changed to the new shape wholesale,
 * every verb would have to move in one commit and a failure would have 65 candidate causes. When the
 * last verb has moved, the dispatch calls the new shape directly and this file is deleted.
 */

/* ⛳ `sys__engine__ctx` AND `sys__engine__ctx_current` ARE NOT DECLARED HERE — they live in `sys/verb_abi.cuh`, which
 * is where the verb BOUNDARY lives. ⇒ ★ the boundary belongs to the language, not to the package that
 * happened to need it first; a second copy here is "typedef redefinition with different types". */

#define nn__vector__zzabi_error sys__engine__abi__error   /* sys/verb_abi.cuh owns it */

/* The verb itself, in the ruled shape: the same bounds checks and `room()` calls as every buffer verb. */
static sys__heap_node nn__vector__zzabi_apply_add(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_zero(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_exp(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_softplus(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_l2norm(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_pointwise_mul(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_scale(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_scale_at(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);

/* ⛳ THE ADAPTER IS DECLARED IN THE HEADER PHASE AND DEFINED IN THE BODY PHASE, like every verb here.
 * ⛔ IT HAS TO BE: the dispatch switch is expanded inside `sys/cpu/heap_object__impl.cuh`, which the
 * registry includes BEFORE nn's bodies — so a definition alone is "use of undeclared identifier" at a
 * line in somebody else's package. */
static __device__ __noinline__ void nn__vector__zzabi_adapter_add(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_zero(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_exp(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_softplus(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_l2norm(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_pointwise_mul(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_scale(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_scale_at(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__vector__zzabi_apply_top_k(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__vector__zzabi_adapter_top_k(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__vector__zzabi_apply_top_k_biased(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__vector__zzabi_adapter_top_k_biased(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__vector__zzabi_apply_penalize(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__vector__zzabi_adapter_penalize(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI__HEADER_CUH */
