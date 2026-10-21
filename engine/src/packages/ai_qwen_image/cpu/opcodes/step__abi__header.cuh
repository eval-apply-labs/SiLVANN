#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI__HEADER_CUH
/* What a step does around the blocks: the timestep's modulation, the prompt's and a picture's rows in, the step's end.
 * ▶ `step__abi.cuh`. */
static sys__heap_node ai_qwen_image__time__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_image__time__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_image__text__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_image__text__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_image__latents__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_image__latents__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_image__out__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ SYS__NOINLINE void ai_qwen_image__out__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI__HEADER_CUH */
