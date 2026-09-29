#ifndef SILVANN__PACKAGES_SYS_MANIFEST_CUH
#define SILVANN__PACKAGES_SYS_MANIFEST_CUH
/* ══ sys — WHAT THE PACKAGE DEFINES ══════════════════════════════════════════════════════════════════
 *
 * The other half of `manifest__header.cuh`, which is where the reasoning for both lives. This one names
 * the implementations, which nothing includes and which would therefore never be compiled at all.
 * ⭐ IT IS A LIST AND NOT AN ORDER for the same reason: every file names what it needs.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "cpu/heap_object__impl.cuh"
#include "cpu/bindings__impl.cuh"
#include "cpu/carousel__impl.cuh"
#include "cpu/computing_base__impl.cuh"
#include "cpu/fault__impl.cuh"
#include "cpu/heap__impl.cuh"
#include "cpu/list__impl.cuh"
#include "cpu/node_array__impl.cuh"
#include "cpu/package__impl.cuh"
#include "cpu/silicon/silicon__impl.cuh"
#include "cpu/stack__impl.cuh"
#include "cpu/dictionary__impl.cuh"
#include "cpu/system_register__impl.cuh"
#include "cpu/opcodes/opcodes_abi.cuh"
#include "cpu/abi_surface__impl.cuh"    /* its C interface, as data for a host language to bind */

/* The second half of the roll call. A package that declared everything and defined nothing compiles
 * clean and links against nothing, so the registry asks separately. */
#define PACKAGE_sys_PRESENT 1

#endif /* SILVANN__PACKAGES_SYS_MANIFEST_CUH */
