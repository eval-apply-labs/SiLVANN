#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI__HEADER_CUH
/* The Hadamard rotation in the ruled shape. ▶ `vector__abi__header.cuh`. */
static sys__heap_node nn__hadamard__zzabi_apply_rotate(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__hadamard__zzabi_adapter_rotate(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__hadamard__zzabi_apply_blocks(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__hadamard__zzabi_adapter_blocks(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI__HEADER_CUH */
