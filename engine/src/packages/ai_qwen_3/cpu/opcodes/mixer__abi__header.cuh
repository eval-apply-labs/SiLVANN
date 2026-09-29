#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI__HEADER_CUH
/* The mixers. ▶ `mixer__abi.cuh`. */
static sys__heap_node ai_qwen_3__deltanet__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__deltanet__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__attention__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__attention_tiered__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__attention_tiered__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__mlp__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__mlp__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI__HEADER_CUH */
