#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI__HEADER_CUH
/* RoPE and attention in the ruled shape. ▶ `vector__abi__header.cuh`. */
static sys__heap_node nn__rope__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__attention__zzabi_apply_decode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__rope__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__attention__zzabi_adapter_decode(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__rope__zzabi_apply_angles(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__rope__zzabi_adapter_angles(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__attention__zzabi_apply_residual(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__attention__zzabi_apply_merge(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__attention__zzabi_apply_finish(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__attention__zzabi_adapter_residual(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__attention__zzabi_adapter_merge(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__attention__zzabi_adapter_finish(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__attention__zzabi_apply_residual_tq(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__attention__zzabi_adapter_residual_tq(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__attention__zzabi_apply_absorb(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__attention__zzabi_adapter_absorb(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__attention__zzabi_apply_expand(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__attention__zzabi_adapter_expand(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI__HEADER_CUH */
