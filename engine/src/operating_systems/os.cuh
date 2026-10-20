#ifndef SILVANN__OPERATING_SYSTEMS_OS_CUH
#define SILVANN__OPERATING_SYSTEMS_OS_CUH

/* ══ WHICH SYSTEM ANSWERS `sys__os__*` — THE ONE THE TREE IS BUILT FOR ═════════════════════════════════════════════
 * Every caller includes this file, never a system's own; the compiler's target picks the folder.
 * ▶ the contract: `packages/sys/contracts/abi/os.cuh`. */
#if defined(_WIN32)
#include "windows/os.cuh"
#else
#include "linux/os.cuh"
#endif

#endif /* SILVANN__OPERATING_SYSTEMS_OS_CUH */
