#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI__HEADER_CUH
/* The hyper-connection in the ruled shape. ▶ `hyper__abi.cuh`. */
static sys__heap_node nn__hyper__zzabi_apply_pre(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__hyper__zzabi_adapter_pre(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__hyper__zzabi_apply_post(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__hyper__zzabi_adapter_post(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__kda__zzabi_apply_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__kda__zzabi_adapter_step(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_HYPER__ABI__HEADER_CUH */
