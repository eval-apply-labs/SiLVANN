#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI__HEADER_CUH
/* An n-gram memory's rows, hashed and read inside the program. ▶ `contracts/objects/ngram.cuh`. */
static sys__heap_node nn__ngram__zzabi_apply_rows(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void nn__ngram__zzabi_adapter_rows(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI__HEADER_CUH */
