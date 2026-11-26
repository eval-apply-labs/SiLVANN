#ifndef SILVANN__SILICON_FAMILIES_X86_BASE_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_X86_BASE_UNIT_CUH

/* ══ THE WHOLE OF x86_base, AS ONE TRANSLATION UNIT ══════════════════════════════════════════════════════════════════
 * The processors' family where x86_avx2 cannot run (▶ includes.cuh): x86_avx2's own primitives — its memory, its pool of
 * threads, its file reads — with none of its AVX2 overrides, so every door is nn's generic body, and a half widened in C
 * where F16C would widen it. `scripts/build_silicon_families.sh` compiles this with `g++ -march=x86-64-v2` into
 * `libsilvann_x86_base.so`; its doors keep x86_avx2's names (the source's), and it IS x86_base, its number and its name
 * the program sees. A CPU worker on it computes slowly and in full: where there is an integrated GPU, that computes
 * instead (NN-216). */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS 1
#define X86_AVX2__NEEDS_AVX2 0                    /* a device on any x86-64-v2 processor */
#define X86_AVX2__F16C 0                          /* a half widened in C */
#include "../x86_avx2/manifest.cuh"               /* the primitives, x86_avx2's */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#define SYS__SILICON_FAMILY__THIS x86_avx2        /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   x86_base        /* the family it is: its number and its name */
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_X86_BASE_UNIT_CUH */
