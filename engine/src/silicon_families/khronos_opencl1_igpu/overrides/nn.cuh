#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "doors.cuh"        /* the doors, launching kernels.cuh's kernels by name, and into nn's table */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * Each door has its door's signature and takes its place in nn's table (`khronos_opencl1_igpu__override_doors`, called from
 * `../entry.cuh` once the table is built). Their kernels (`kernels.cuh`) are not in this binary as code: the build writes
 * them into the family's OpenCL C 1.2 program beside nn's (`scripts/opencl1_kernel_source.py`). ⛳ EVERY ONE FALLS BACK TO
 * nn's GENERIC DOOR for what it does not cover, so an override can only ever be faster, never different in what it
 * accepts, and deleting this folder loses nothing but speed. */

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_NN_CUH */
