#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI__HEADER_CUH
/* TurboQuant's decode and gemv in the ruled shape. ▶ `vector__abi__header.cuh`. */
static sys__heap_node nn__turboquant__zzabi_apply_decode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__turboquant__zzabi_apply_gemv(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__turboquant__zzabi_adapter_decode(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__turboquant__zzabi_adapter_gemv(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__turboquant__zzabi_apply_gemv_int8(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__turboquant__zzabi_adapter_gemv_int8(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__turboquant__zzabi_apply_encode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__turboquant__zzabi_adapter_encode(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI__HEADER_CUH */
