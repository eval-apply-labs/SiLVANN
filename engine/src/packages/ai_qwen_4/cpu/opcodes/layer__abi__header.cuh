#ifndef SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH
/* Flash-Next's layer sites on a worker: the gated residual's before-step, the attention, the DeltaNet, the shared
 * expert. ▶ `layer__abi.cuh`. */
static sys__heap_node ai_qwen_4__gated_pre__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__gated_pre__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__attention__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__deltanet__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__deltanet__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__shared__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__shared__zzabi_adapter(sys__heap_node* base, uint64_t form);
/* …and a prompt's chunk, its rows at once */
static sys__heap_node ai_qwen_4__gated_pre_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__gated_pre_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__inject_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__inject_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__deltanet_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__deltanet_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__attention_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__attention_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__shared_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__shared_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_4__route_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__route_rows__zzabi_adapter(sys__heap_node* base, uint64_t form);
/* …and the router's site at a position */
static sys__heap_node ai_qwen_4__route__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_4__route__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI__HEADER_CUH */
