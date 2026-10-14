#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI__HEADER_CUH
/* A picture through the vision tower: the patch embedding, a block, the merger. ▶ `vision__abi.cuh`. */
static sys__heap_node ai_qwen_3__vision_embed__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__vision_block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__vision_merger__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__vision_embed__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__vision_block__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__vision_merger__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI__HEADER_CUH */
