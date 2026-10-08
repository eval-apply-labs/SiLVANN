#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI__HEADER_CUH
/* A prompt's rows through the layer's three words. ▶ `prefill__abi.cuh`. */
static sys__heap_node ai_qwen_3__deltanet_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__attention_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__attention_tiered_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__moe_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__deltanet_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__attention_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__attention_tiered_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__moe_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__stream_counts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__stream_counts__zzabi_adapter(sys__heap_node* base, uint64_t form);
static void ai_qwen_3__stream__zzpackage_close(void);

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI__HEADER_CUH */
