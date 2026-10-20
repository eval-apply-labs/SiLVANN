#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
/* The layer's verbs. ▶ `layer__abi.cuh`. */
static sys__heap_node ai_deepseek_4__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__attention__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_deepseek_4__route__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__route__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_deepseek_4__shared__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__shared__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_deepseek_4__close__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__close__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_deepseek_4__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__experts__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_deepseek_4__engram__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_deepseek_4__engram__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH */
