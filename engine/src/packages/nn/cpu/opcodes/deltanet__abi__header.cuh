#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI__HEADER_CUH
/* The DeltaNet state verbs in the ruled shape. ▶ `vector__abi__header.cuh`. */
static sys__heap_node nn__deltanet__zzabi_apply_rank_1_update(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__deltanet__zzabi_apply_readout(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__deltanet__zzabi_apply_conv_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__deltanet__zzabi_apply_step(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__deltanet__zzabi_adapter_rank_1_update(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__deltanet__zzabi_adapter_readout(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__deltanet__zzabi_adapter_conv_step(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__deltanet__zzabi_adapter_step(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_DELTANET__ABI__HEADER_CUH */
