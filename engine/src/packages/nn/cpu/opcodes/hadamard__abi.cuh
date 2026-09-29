#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI_CUH
/* ══ THE HADAMARD ROTATION, ON THE PROGRAM'S SIDE ════════════════════════════════════════════════════
 * The checks are its body's in `hadamard__impl.cuh`; the arithmetic is the door's. ▶ `vector__abi.cuh`
 * for the shape and the refusals. */

#include "../doors.cuh"

/* `(nn__hadamard__rotate x sign out n)` — the sign vector need only be as wide as the widest block, and
 * a length the block ladder cannot place is refused with the rotation's own word. */
static sys__heap_node nn__hadamard__zzabi_apply_rotate(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, s_at = 0, o_at = 0, x_room = 0, s_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &s_at, &s_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    const uint64_t widest = (n < NN__HADAMARD__WHOLE_BELOW) ? n : NN__HADAMARD__WHOLE_BELOW;
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(widest, s_room)
     || !nn__primitives__fits(n, o_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (!nn__hadamard__zzpackage_ladder_places(n)) return sys__engine__abi__error(NN__HADAMARD__FAULT_LENGTH);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const uint16_t*)(uintptr_t)s_at, n,
                           ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__hadamard__blocks x sign out n block inverse)` — the rotation in blocks of `block` (a power of two, at most 512),
 * each on its own: a head at a time, for the KV cache. `inverse` 1 un-rotates. `x` and `out` must not overlap. */
static sys__heap_node nn__hadamard__zzabi_apply_blocks(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, s_at = 0, o_at = 0, x_room = 0, s_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &s_at, &s_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0], block = argv[4].args[0], inverse = argv[5].args[0];
    if (block < 2ull || block > NN__HADAMARD__WHOLE_BELOW || (block & (block - 1ull)) != 0ull || n % block != 0ull || inverse > 1ull)
        return sys__engine__abi__error(NN__HADAMARD__FAULT_LENGTH);
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(block, s_room) || !nn__primitives__fits(n, o_room)
     || (x_at < o_at + 2ull * n && o_at < x_at + 2ull * n)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const uint16_t*)(uintptr_t)s_at, n, block,
                           inverse, ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__hadamard__zzabi_adapter_rotate, nn__hadamard__zzabi_apply_rotate)
SYS__ENGINE__ABI__BRIDGE(nn__hadamard__zzabi_adapter_blocks, nn__hadamard__zzabi_apply_blocks)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_HADAMARD__ABI_CUH */
