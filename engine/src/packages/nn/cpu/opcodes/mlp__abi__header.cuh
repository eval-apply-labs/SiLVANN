#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI__HEADER_CUH
/* A dense MLP in the ruled shape. ▶ `mlp__abi.cuh`. */
static sys__heap_node nn__mlp__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__mlp__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__mlp__zzabi_apply_rows(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__mlp__zzabi_adapter_rows(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI__HEADER_CUH */
