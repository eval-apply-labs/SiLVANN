#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
/* A picture through the vision tower: the patch embedding, a block, the tail. ▶ `vision__abi.cuh`. */
static sys__heap_node ai_deepseek_4__vision_embed__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_deepseek_4__vision_block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_deepseek_4__vision_tail__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_deepseek_4__vision_embed__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ SYS__NOINLINE void ai_deepseek_4__vision_block__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ SYS__NOINLINE void ai_deepseek_4__vision_tail__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI__HEADER_CUH */
