#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_MANIFEST_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_MANIFEST_CUH

/* ══ khronos_opencl2_base — THE OPENCL 2.0 FAMILIES' ANSWERS TO EVERY PACKAGE THEY SUPPORT ══════════════════════════════
 * ⚖ *"only through a manifest so i dont have to keep track of them"*. One `<package>.cuh` per package this
 * family answers, and nothing else: whoever compiles the family includes this file and never names what
 * is in it, so a package the family comes to support is added here and nowhere else.
 * ⭐ ONE SOURCE, TWO FAMILIES (⚖ 2026-11-09 *"opencl should have opencl12 and opencl2 as their base, and then have
 *   subclasses so old_nvidia does its own thing and does not stomp against the intel old igpu"*): not a family of the
 *   roster — `khronos_opencl2` (any GPU whose OpenCL is 2.0) and `intel_opencl2` (Intel's GPUs from Gen9: the integrated
 *   ones, Arc and Max) are each built from it by their own `unit.cuh`, which names the family, its cards and its name. */
#include "sys.cuh"   /* sys's doors */
#include "nn.cuh"    /* nn's requests */

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_MANIFEST_CUH */
