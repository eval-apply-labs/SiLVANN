#ifndef SILVANN__PACKAGES_NN_MANIFEST_CUH
#define SILVANN__PACKAGES_NN_MANIFEST_CUH
/* ══ nn — WHAT THE PACKAGE DEFINES ═══════════════════════════════════════════════════════════
 *
 * The other half of the file beside this one, which is where the reasoning for both lives. This one
 * names implementations, which nothing includes and which would therefore never be compiled.
 *
 * One: standing the package up. It is here rather than beside its declaration because its body names a
 * verb id, and those are generated from every package's rows at once in a phase after every header has
 * been read — so a definition that names one cannot sit in a header at all.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⭐ THE COMPUTE COMES FIRST, BECAUSE EVERY VERB BELOW CALLS INTO IT. ▶ `kernels.cuh` for why the
 * loops live apart from the verbs that own them: they are what becomes a kernel when `sys` goes host. */
#include "gpu/kernels/kernels.cuh"
#include "cpu/buffer__impl.cuh"
#include "cpu/expert__impl.cuh"
#include "cpu/loader__impl.cuh"
#include "cpu/routed__impl.cuh"
#include "cpu/cartridge__impl.cuh"
#include "cpu/primitives__impl.cuh"
#include "cpu/deltanet__impl.cuh"
#include "cpu/swiglu__impl.cuh"
#include "cpu/turboquant__impl.cuh"
#include "cpu/hadamard__impl.cuh"
#include "cpu/package__impl.cuh"
/* ⛳ LAST: the evaluator's bridge calls into the impls above, so it comes after them. */
#include "cpu/opcodes/vector__abi.cuh"
#include "cpu/opcodes/primitives__abi.cuh"
#include "cpu/opcodes/swiglu__abi.cuh"
#include "cpu/opcodes/hadamard__abi.cuh"
#include "cpu/opcodes/deltanet__abi.cuh"
#include "cpu/opcodes/expert__abi.cuh"
#include "cpu/opcodes/turboquant__abi.cuh"
#include "cpu/opcodes/result__abi.cuh"
#include "cpu/opcodes/attention__abi.cuh"
#include "cpu/opcodes/hyper__abi.cuh"
#include "cpu/opcodes/mlp__abi.cuh"
#include "cpu/abi_surface__impl.cuh"    /* its C interface, as data for a host language to bind */

/* The second half of the roll call. */
#define PACKAGE_nn_PRESENT 1

#endif /* SILVANN__PACKAGES_NN_MANIFEST_CUH */
