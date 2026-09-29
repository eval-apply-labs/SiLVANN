#ifndef SILVANN__PACKAGES_NN_CPU_ABI_SURFACE__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_ABI_SURFACE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/abi/cpu.cuh"   /* the calls, and — from sys — the table they are described by */

/* ══ nn's CALLS, AS DATA ════════════════════════════════════════════════════════════════════════════════
 * One row per call in `contracts/abi/cpu.cuh`, for a host language to bind; the shapes are sys's
 * (`sys/contracts/abi/cpu.cuh`). Both calls already have their shape's signature, so no adapter is needed. */
#ifndef SYS__SILICON__HARNESS   /* the calls exist only in a real build, not in a harness */

static const sys__abi_entry nn__abi_surface__zzprivate_entries[] = {
    /* ⭐ THE ORACLE'S WRITE HALF. Takes `bytes`, so numpy hands it `arr.tobytes()`. */
    { "write", SYS__ABI_SHAPE_WRITE_BYTES, (sys__abi_call)nn_abi_write,
      { "address", "payload" }, { 0 }, "those bytes did not go",
      "Copy bytes into a buffer on the worker's card." },
    /* ⭐⭐ THE ORACLE'S READ HALF. Answers `bytes`, so numpy reads it with `frombuffer` and a dtype. */
    { "read", SYS__ABI_SHAPE_READ_BYTES, (sys__abi_call)nn_abi_read,
      { "address", "bytes" }, { 0 }, "those bytes did not come back",
      "Copy `bytes` out of a buffer on the worker's card." },
};

static const sys__abi_surface nn__abi_surface__zzprivate_surface = {
    "nn: bytes in and out of a buffer on the worker's card.",
    (uint32_t)(sizeof nn__abi_surface__zzprivate_entries / sizeof nn__abi_surface__zzprivate_entries[0]),
    nn__abi_surface__zzprivate_entries,
};

extern "C" const sys__abi_surface* nn_abi_surface(void) { return &nn__abi_surface__zzprivate_surface; }

#endif

#endif /* SILVANN__PACKAGES_NN_CPU_ABI_SURFACE__IMPL_CUH */
