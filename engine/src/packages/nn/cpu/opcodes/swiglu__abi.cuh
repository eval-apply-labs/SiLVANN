#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_SWIGLU__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_SWIGLU__ABI_CUH
/* ══ THE SwiGLU COMBINE, ON THE PROGRAM'S SIDE ═══════════════════════════════════════════════════════
 * The checks are its body's in `swiglu__impl.cuh`, with its own fault word; the arithmetic is the door's.
 * ▶ `vector__abi.cuh` for the shape and the refusals. */

#include "../doors.cuh"

/* `(nn__swiglu__combine gate up out n)` — `out = silu(gate) * up`. */
static sys__heap_node nn__swiglu__zzabi_apply_combine(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t g_at = 0, u_at = 0, o_at = 0, g_room = 0, u_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &g_at, &g_room)
     || !nn__primitives__room(&argv[1], &u_at, &u_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    if (!nn__primitives__fits(n, g_room) || !nn__primitives__fits(n, u_room)
     || !nn__primitives__fits(n, o_room)) return sys__engine__abi__error(NN__SWIGLU__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->swiglu_combine((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)g_at, (const uint16_t*)(uintptr_t)u_at, n,
                          ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__swiglu__clamped gu out n limit)` — GLM's swiglu: `silu(min(g, limit)) · clamp(u, −limit, limit)`, `gu` the gate
 * then the up, `n` halves each, into `out`. */
static sys__heap_node nn__swiglu__zzabi_apply_clamped(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t g_at = 0, g_room = 0, o_at = 0, o_room = 0;
    if (!nn__primitives__room(&argv[0], &g_at, &g_room) || !nn__primitives__room(&argv[1], &o_at, &o_room)
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_FLOAT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[2].args[0];
    if (n == 0ull || !nn__primitives__fits(2ull * n, g_room) || !nn__primitives__fits(n, o_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)g_at, 1ull, n, (float)sys__heap_node__real(argv[3].args[0]),
                          ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

SYS__ENGINE__ABI__BRIDGE(nn__swiglu__zzabi_adapter_combine,  nn__swiglu__zzabi_apply_combine)

SYS__ENGINE__ABI__BRIDGE(nn__swiglu__zzabi_adapter_clamped, nn__swiglu__zzabi_apply_clamped)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_SWIGLU__ABI_CUH */
