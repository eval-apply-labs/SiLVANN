#ifndef SILVANN__PACKAGES_NN_CONTRACTS_DEFAULTS_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_DEFAULTS_CUH

/* Nothing in this package has to come first: this file names none of it. */
/* ══ nn's dials, and what they are when nobody turns them ══════════════════════════════════════════════
 *
 * Values that are a CHOICE rather than a consequence, kept apart from the code that reads them — the same
 * file sys keeps (`sys/contracts/defaults.cuh`), for the same reason.
 * ⛳ EACH IS `#ifndef`-GUARDED so a build overrides it without editing anything, and each is guarded
 * again where it is read with the same value, so commenting one out here leaves a working program. */

/* ⚖ *"yes build it, default exact"* — whether a card family computes nn's `expf` and `logf` with the
 * precise functions (0) or with the card's fast ones (1: `__expf` / `__logf` on HIP and CUDA,
 * `native_exp` / `native_log` in OpenCL). `sqrtf` stays precise either way: on these cards the two cost
 * about the same, so there is nothing to turn. The host computes with libm under both.
 * ⛳ EXACT BY DEFAULT because an oracle is only worth running against a reference somebody else computed,
 * and the precise functions are what numpy and PyTorch use. The fast ones differ by far less than an fp16
 * activation's own spacing (`REASONED`, not yet measured); the measurement that would flip this default
 * is every oracle built both ways, after the bodies are wide enough for the difference to cost anything.
 * ▶ `contracts/abi/gpu.cuh` for why each function is the one it is. Override: `SILVANN_NN_ARITHMETIC_FAST=1`
 * in the environment of `scripts/build_silicon_families.sh`. */
#ifndef NN__ARITHMETIC__FAST
#define NN__ARITHMETIC__FAST 0
#endif

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_DEFAULTS_CUH */
