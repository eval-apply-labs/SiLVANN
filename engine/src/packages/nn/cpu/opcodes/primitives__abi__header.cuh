#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI__HEADER_CUH
/* The rest of the primitives in the ruled shape — normalisations and the two matrix verbs. ▶
 * `vector__abi__header.cuh` for the adapter they are reached through and why it is scaffolding. */

static sys__heap_node nn__rmsnorm__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__softmax__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__sigmoid__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__matrix__zzabi_apply_transpose(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__matrix__zzabi_apply_matvec_transposed(const sys__heap_node* argv, unsigned argc,
                                                         sys__engine__ctx* ctx);

static __device__ __noinline__ void nn__rmsnorm__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__softmax__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__sigmoid__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__matrix__zzabi_adapter_transpose(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__matrix__zzabi_adapter_matvec_transposed(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI__HEADER_CUH */
