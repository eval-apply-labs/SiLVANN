#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__IMPL_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "package__header.cuh"

/* The engine asks every package for its room before boot and gives it back at shutdown; this one takes none. */
static inline bool ai_qwen_3__package__host_init(const char* config, uint64_t length, sys__package_room* room) {
    (void)config; (void)length;
    if (room == 0) return false;
    room->at = 0; room->bytes = 0ull;
    room->ram_at = 0; room->ram_card_at = 0; room->ram_bytes = 0ull;
    return true;
}
/* ...and gives back the side queue a streamed prompt opened, if one did. */
static inline void ai_qwen_3__package__host_teardown(sys__package_room* room) { (void)room; ai_qwen_3__stream__zzpackage_close(); }

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__IMPL_CUH */
