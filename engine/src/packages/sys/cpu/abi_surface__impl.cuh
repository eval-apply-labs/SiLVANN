#ifndef SILVANN__PACKAGES_SYS_CPU_ABI_SURFACE__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_ABI_SURFACE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/abi/cpu.cuh"   /* the calls, the shapes, and the table they are described by */

/* ══ sys's CALLS, AS DATA ═══════════════════════════════════════════════════════════════════════════════
 * One row per call in `contracts/abi/cpu.cuh`, for a host language to bind (▶ that file for the shapes).
 * Three calls answer in a shape of their own — a block number narrower than the shape's, and tallies in
 * named out-parameters — so each has a small adapter of the shape's signature here, and the C interface
 * a C caller uses is left as it is. */
#ifndef SYS__SILICON__HARNESS   /* the calls exist only in a real build, not in a harness */

static unsigned long long sys__abi_surface__zzprivate_base_address(unsigned long long block) {
    return block > 0xFFFFFFFFull ? 0ull : sys_abi_base_address((unsigned int)block);
}

/* The allocator's tallies: claims, refusals, and deallocations — which is ABSENT, not zero, when its counter
 * is compiled out. */
static int sys__abi_surface__zzprivate_counters(unsigned long long* words, int* present) {
    unsigned int claims = 0u, refusals = 0u, frees = 0u;
    if (!sys_abi_counters(&claims, &refusals, &frees)) return 0;
    words[0] = claims;  present[0] = 1;
    words[1] = refusals; present[1] = 1;
    words[2] = frees;   present[2] = sys_abi_counts_deallocations();
    return 1;
}

static int sys__abi_surface__zzprivate_fault(unsigned long long* words, int* present) {
    unsigned int raised = 0u; unsigned long long pc = 0ull, op = 0ull;
    if (!sys_abi_fault(&raised, &pc, &op)) return 0;
    words[0] = raised; words[1] = pc; words[2] = op;
    present[0] = present[1] = present[2] = 1;
    return 1;
}

static const sys__abi_entry sys__abi_surface__zzprivate_entries[] = {
    { "peek", SYS__ABI_SHAPE_ADDRESS_TO_NODE, (sys__abi_call)sys_abi_peek,
      { "address", 0 }, { 0 }, "that is not a heap address, or nothing is up",
      "The heap node at an address, while the machine works or not. -> (kind, op_code, [args])." },
    /* ⚖ *"if the result is an offset a full_address method that returns offset plus base is the solution."*
     * Anything that answers a reference answers an OFFSET, and this turns one into the place `peek` can be
     * pointed at. */
    { "full_address", SYS__ABI_SHAPE_NUMBER_TO_NUMBER, (sys__abi_call)sys_abi_full_address,
      { "offset", 0 }, { 0 }, 0,
      "A heap offset as an address. 0 when it names nothing, or nothing is up." },
    { "base_address", SYS__ABI_SHAPE_NUMBER_TO_NUMBER, (sys__abi_call)sys__abi_surface__zzprivate_base_address,
      { "block", 0 }, { 0 }, 0,
      "Where a block's computing base is. 0 for a block the machine does not have." },
    /* ⛔ `deallocations` is None when its counter is compiled out, rather than 0. A switched-off tally and a
     *   tally of nothing are the same number and completely different facts. */
    { "counters", SYS__ABI_SHAPE_WORDS, (sys__abi_call)sys__abi_surface__zzprivate_counters,
      { 0 }, { "chunk_claims", "chunk_refusals", "deallocations", 0 }, "nothing is up",
      "The allocator's own tallies. `deallocations` is None when that counter is compiled out." },
    /* Whether anything refused, and with what word — a verb that declines answers its caller with nothing,
     * so without this a legitimate nothing and a refusal look the same. */
    { "fault", SYS__ABI_SHAPE_WORDS, (sys__abi_call)sys__abi_surface__zzprivate_fault,
      { 0 }, { "raised", "pc", "op", 0 }, "nothing is up",
      "What the package last refused with. `raised` is 0 when nothing has." },
    { "fault_clear", SYS__ABI_SHAPE_NOTHING, (sys__abi_call)sys_abi_fault_clear,
      { 0 }, { 0 }, "nothing is up",
      "Put the fault channel back to nothing-went-wrong." },
};

static const sys__abi_surface sys__abi_surface__zzprivate_surface = {
    "sys: looking inside the heap, the allocator and the fault channel.",
    (uint32_t)(sizeof sys__abi_surface__zzprivate_entries / sizeof sys__abi_surface__zzprivate_entries[0]),
    sys__abi_surface__zzprivate_entries,
};

extern "C" const sys__abi_surface* sys_abi_surface(void) { return &sys__abi_surface__zzprivate_surface; }

#endif

#endif /* SILVANN__PACKAGES_SYS_CPU_ABI_SURFACE__IMPL_CUH */
