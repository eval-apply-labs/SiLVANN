#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI__HEADER_CUH
/* The MoE verb and its three parts. ▶ `moe__abi.cuh`. */
static sys__heap_node ai_qwen_3__moe__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__moe__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__pre_expert__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__post_expert__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__pre_expert__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__experts__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__post_expert__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__prefetch__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__prefetch__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__experts_up__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__experts_down__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__experts_up__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__experts_down__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__pre_expert_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static sys__heap_node ai_qwen_3__post_expert_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__pre_expert_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__experts_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_3__experts_note__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_3__experts_note__zzabi_adapter(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void ai_qwen_3__post_expert_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI__HEADER_CUH */
