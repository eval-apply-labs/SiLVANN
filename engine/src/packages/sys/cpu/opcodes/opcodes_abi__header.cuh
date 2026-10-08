#ifndef SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI__HEADER_CUH
#include "../heap_node__header.cuh"   /* the node every value in this language is made of */
/* Declarations for the converted `sys` verbs' adapters. ▶ `opcodes_abi.cuh` for the bodies.
 * ⛔ THE SPLIT IS REQUIRED, NOT TIDY: the dispatch switch is expanded inside `heap_object__impl.cuh`,
 * which the registry includes before these bodies, so a definition alone is "use of undeclared
 * identifier" at a line in another file. Every verb in this tree is declared for the same reason. */
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_eq(sys__heap_node* base, uint64_t form);

/* The twelve arithmetic adapters. `sys__int__add`'s contract row reaches its adapter through a macro,
 * `SYS__OPCODES__ZZABI_INT_ADD_APPLY`, so the `SILVANN_ONE_READ` experiment can put its own body there. */
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_int_add(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_float_add(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_int_sub(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_float_sub(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_int_less(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_float_less(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_add(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_sub(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_less(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_int_mul(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_float_mul(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_mul(sys__heap_node* base, uint64_t form);

/* The object verbs and the register's two doors. */
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_clone(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_type(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_create(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_register_get(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_register_set(sys__heap_node* base, uint64_t form);

/* A scope placed by hand, and the branch. */
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_add_bindings(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_if(sys__heap_node* base, uint64_t form);

/* The last nine. */
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_viewonly(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_bindings_set(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_compute(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_completed(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_result(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_dict_put(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_dict_get(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_array_get(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_array_set(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_file_open(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_file_close(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_worker(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_unquote(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_package_init(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void sys__opcodes__zzabi_adapter_defun(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI__HEADER_CUH */
