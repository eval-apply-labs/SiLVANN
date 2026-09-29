#ifndef SILVANN__PACKAGES_NN_CONTRACTS_ABI_CPU_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_ABI_CPU_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../sys/contracts/abi/cpu.cuh"   /* the shape of a package's calls as data */

/* ══ ⭐ nn's C INTERFACE — THE ORACLE'S TWO ENDS ════════════════════════════════════════════════════════
 *
 * ⚖ RULED, *"cant we read the value from DMA?"*, and *"heap-only"* for `sys`: bytes
 * in a buffer are nn's, so the door that carries them in and out of a program is nn's. The host puts known
 * numbers into a buffer, a program transforms them, and the host reads them back to compare against a
 * reference — only the transform is a verb. Python binds these as `nn.read` / `nn.write`.
 * ⛳ ON THE WORKER'S CARD, through its family, after the work already queued there has completed.
 * ⛳ AN ADDRESS INSIDE THE HEAP IS REFUSED: the heap is `sys`'s, and is looked at with `sys.peek`.
 * ⛳ ANSWERS 0 FOR A REFUSAL — no worker, a heap address, a zero length — and 1 when the bytes moved. */
extern "C" {
int nn_abi_read(unsigned long long address, unsigned long long bytes, void* into);
int nn_abi_write(unsigned long long address, unsigned long long bytes, const void* from);
/* ⭐ AND A FILE AS A BUFFER'S BYTES: `bytes` of the file at `path`, from `offset`, mapped over the buffer at `address`
 *   read-only — the disk read as each page is first touched, the pages kept as the system keeps a file's. A routed
 *   expert collection larger than the machine's memory is this: its slots written once as a file in their own
 *   layout, and mapped. Page-aligned `address` and `offset`; a card, which cannot, answers 0. */
int nn_abi_map_file(unsigned long long address, unsigned long long bytes, const char* path, unsigned long long offset);
}

extern "C" {
/* nn's calls, as data — the shapes and the table are sys's (`sys/contracts/abi/cpu.cuh`). */
const sys__abi_surface* nn_abi_surface(void);
}

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_ABI_CPU_CUH */
