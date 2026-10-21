#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI__HEADER_CUH
/* A tensor-parallel group's partial sums, added over the devices' own memory. ▶ `partials__abi.cuh`. */
static sys__heap_node nn__partials__zzabi_apply_join(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void nn__partials__zzabi_adapter_join(sys__heap_node* base, uint64_t form);
static inline uint64_t nn__partials__reduce_onto(sys__engine__ctx* ctx, const struct nn__doors* doors, uint64_t group, uint64_t part,
                                                 uint64_t out, uint64_t n, uint64_t residual);
static sys__heap_node nn__partials__zzabi_apply_reduce(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void nn__partials__zzabi_adapter_reduce(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI__HEADER_CUH */
