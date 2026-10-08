#ifndef SILVANN__PACKAGES_NN_MANIFEST__HEADER_CUH
#define SILVANN__PACKAGES_NN_MANIFEST__HEADER_CUH
/* ══ nn — WHAT THE PACKAGE DECLARES ══════════════════════════════════════════════════════════
 *
 * The library a model is built with: the arithmetic its layers are made of, the residency policy that
 * keeps its weights within reach of the card, and the words a model program is written in. The language
 * is next door and this package is written against it, never the other way about.
 *
 * ⭐ THIS IS A LIST AND NOT AN ORDER. Every file names what it needs at its own top, so the compiler
 * works the order out and a line here cannot be in the wrong place. What the list is FOR is that an
 * implementation is included by nobody — a caller needs a declaration and never a definition — so
 * something has to name one or it is never compiled at all. Declarations here, definitions beside this.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "cpu/buffer__header.cuh"
#include "cpu/expert__header.cuh"
#include "cpu/routed__header.cuh"
#include "cpu/cartridge__header.cuh"
#include "cpu/primitives__header.cuh"
#include "cpu/deltanet__header.cuh"
#include "cpu/swiglu__header.cuh"
#include "cpu/turboquant__header.cuh"
#include "cpu/hadamard__header.cuh"
#include "language_contract.cuh"
#include "cpu/package__header.cuh"
#include "cpu/opcodes/vector__abi__header.cuh"
#include "cpu/opcodes/primitives__abi__header.cuh"
#include "cpu/opcodes/swiglu__abi__header.cuh"
#include "cpu/opcodes/hadamard__abi__header.cuh"
#include "cpu/opcodes/deltanet__abi__header.cuh"
#include "cpu/opcodes/expert__abi__header.cuh"
#include "cpu/opcodes/turboquant__abi__header.cuh"
#include "cpu/opcodes/result__abi__header.cuh"
#include "cpu/opcodes/attention__abi__header.cuh"
#include "cpu/opcodes/hyper__abi__header.cuh"
#include "cpu/opcodes/mlp__abi__header.cuh"

/* The package answering the roll call the registry takes after this phase. A row with no include says so
 * at the registry, naming this package, instead of somewhere further down. */
#define PACKAGE_nn_HEADER_PRESENT 1

#endif /* SILVANN__PACKAGES_NN_MANIFEST__HEADER_CUH */
