#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI__HEADER_CUH
/* The three verbs that answer a value, and the one that reads it, in the ruled shape. ▶
 * `vector__abi__header.cuh`, and `contracts/objects/result.cuh` for the word they share. */
static sys__heap_node nn__argmax__zzabi_apply_find(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_dot_product(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__vector__zzabi_apply_at(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__buffer__zzabi_apply_read(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__argmax__zzabi_adapter_find(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_dot_product(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__vector__zzabi_adapter_at(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__buffer__zzabi_adapter_read(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__vector__zzabi_apply_draw(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__vector__zzabi_adapter_draw(sys__heap_node* base, uint64_t form);
static sys__heap_node nn__buffer__zzabi_apply_copy(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__buffer__zzabi_adapter_copy(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI__HEADER_CUH */
