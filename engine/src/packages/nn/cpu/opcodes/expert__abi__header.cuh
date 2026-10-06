#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI__HEADER_CUH
/* The full-precision expert matvec in the ruled shape. ▶ `vector__abi__header.cuh`. */
static sys__heap_node nn__expert__zzabi_apply_multiply_fp16(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert__zzabi_adapter_multiply_fp16(sys__heap_node* base, uint64_t form);
/* The LRU's miss count. ▶ `expert__abi.cuh`. */
static sys__heap_node nn__expert__zzabi_apply_misses(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert__zzabi_adapter_misses(sys__heap_node* base, uint64_t form);
/* The LRU's words. ▶ `expert__abi.cuh`. */
static sys__heap_node nn__expert__zzabi_apply_request(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__expert__zzabi_apply_prefetch(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node nn__expert__zzabi_apply_settle(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert__zzabi_adapter_request(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__expert__zzabi_adapter_prefetch(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__expert__zzabi_adapter_settle(sys__heap_node* base, uint64_t form);
/* A card's tier given its share of a prompt chunk. ▶ `expert__abi.cuh`. */
static sys__heap_node nn__expert_tier__zzabi_apply_note(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert_tier__zzabi_adapter_note(sys__heap_node* base, uint64_t form);
/* The CPUs done: the writes held back go. ▶ `expert__abi.cuh`. */
static sys__heap_node nn__expert_tier__zzabi_apply_written(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert_tier__zzabi_adapter_written(sys__heap_node* base, uint64_t form);
/* Whether the link is full duplex, timed. ▶ `expert__abi.cuh`. */
static sys__heap_node nn__expert_tier__zzabi_apply_duplex(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void nn__expert_tier__zzabi_adapter_duplex(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI__HEADER_CUH */
