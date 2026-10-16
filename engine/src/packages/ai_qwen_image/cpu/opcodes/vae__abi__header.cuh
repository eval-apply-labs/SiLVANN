#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI__HEADER_CUH
/* The VAE on a worker: a convolution, and the middle's attention. ▶ `vae__abi.cuh`. */
static sys__heap_node ai_qwen_image__conv__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_image__conv__zzabi_adapter(sys__heap_node* base, uint64_t form);
static sys__heap_node ai_qwen_image__vae_attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
static __device__ __noinline__ void ai_qwen_image__vae_attention__zzabi_adapter(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_VAE__ABI__HEADER_CUH */
