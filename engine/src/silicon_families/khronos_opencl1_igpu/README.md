# khronos_opencl1_igpu — OpenCL 1.2, for old integrated GPUs

**The target is an old integrated GPU:** one whose OpenCL lacks what `khronos_opencl2` needs (shared virtual memory, the
generic address space), on a machine with no CUDA or ROCm and a CPU too old for `x86_avx2` (no AVX2) or too slow to beat
its own GPU. `ASSUMED` — none has run this family yet. What would falsify it: a driver that will not build the program,
or a verb that fails on one.

**Which OpenCL driver a GPU has decides whether it is reachable.** The graphics driver is a separate thing: a desktop
can draw with Mesa and have no OpenCL at all. Intel's Ivy Bridge and Haswell GPUs (2012-2013, the Iris Pro among them)
and the Bay Trail Atoms stop at OpenCL 1.2.
- **Beignet**, Intel's own Linux OpenCL for them, is not in Ubuntu 22.04, the base of the Docker image. `MEASURED`:
  `apt-cache policy beignet-opencl-icd` on precision has no candidate.
- **Intel's compute runtime** (`intel-opencl-icd` 22.14 there) starts at Broadwell.
- **Mesa's Clover** (`mesa-opencl-icd` 23.2) is what 22.04 has for them. `MEASURED` on the package (`apt-get download`,
  its file list and the strings of `libMesaOpenCL.so`): it ships `pipe_crocus.so`, Mesa's driver for Intel's GPUs up to
  Haswell, and Clover names `crocus` and reads `CLOVER_DEVICE_VERSION_OVERRIDE` and
  `CLOVER_DEVICE_CLC_VERSION_OVERRIDE`. Clover states OpenCL 1.1 unless those say 1.2, and this family asks for OpenCL C
  1.2. `ASSUMED`, not run: whether Clover on crocus builds this family's program and passes nn's verbs on a Haswell.
- **Mesa's rusticl**, the newer one, is not in 22.04's package (only Clover's library is). A rusticl build carries only
  the drivers it was built with. `MEASURED` by the architect (2026-10-13) on a Linux Mint laptop with a Haswell Iris Pro:
  `clinfo -l` lists only the `rusticl` platform, and `RUSTICL_ENABLE=crocus clinfo -l` answers `crocus: driver missing`.
  That laptop runs on its CPU (`x86_avx2`), and the Docker image is not changed for it.

**Every GPU that rusticl drives belongs to this family.** Rusticl states at most OpenCL C 1.2 for any device
(`clc_version = min(Cl1_2, …)` in its `core/device.rs`), and `khronos_opencl2` builds with `-cl-std=CL2.0` and asks for
fine-grained shared virtual memory. Those GPUs are Intel's from Broadwell (`iris`), AMD's (`radeonsi`) and ARM's
(`panfrost`). Rusticl advertises none of them until `RUSTICL_ENABLE` names its driver, and by default it enables only
`asahi`, `freedreno`, `panfrost` and `radeonsi`. `REASONED` from Mesa's source (main, 2026-10-13), not run. An Intel laptop
from Broadwell on, through `RUSTICL_ENABLE=iris`, is the nearest real iGPU to test this family on.

**A discrete card is not the target.** Every other card family ranks above this one (75 in `../roster.cuh`), so a card
with its own vendor's family goes there. Through this family a discrete card is slow. `MEASURED` on the only device it
has run on, an RTX 5060 Ti through NVIDIA's OpenCL, the 27B D4 decodes 1.61 tokens/s. The same card does 21 through
`nvidia_cuda12`, and the CPU alone (`x86_avx2`) does 2.12. A token takes 627 ms:

- **363 ms are copies.** OpenCL 1.2 has no memory that the host and the card both see, so host RAM a kernel reads is
  kept twice, once in the program's memory and once on the card. nn's RAM span holds the embedding table (1.27 GB) and
  a 64-byte result written every token, and that one write sends the whole span over PCIe in both directions.
- **258 ms are the kernels:** nn's generic bodies, with no overrides. As OpenCL C 1.2 they run about 1.9× slower than
  CUDA's compile of the same bodies, which takes 135 ms for the whole token.

▶ `backstage/measurements/2026-10-13_khronos_opencl1.md` (written under the family's first name, `khronos_opencl1`).

**On an integrated GPU the two copies should be one,** because the card's memory is the host's RAM
(`CL_DEVICE_HOST_UNIFIED_MEMORY`). That leaves the kernels to compete with the machine's own CPU. `REASONED`, not built:
this is NN-73 in the backlog.

## How it works

- **The kernels:** `scripts/opencl1_kernel_source.py` turns nn's card side into one OpenCL C 1.2 program. It parses the
  code with pycparser (`scripts/third_party/`) and works out every pointer's address space starting from the kernels,
  so nn's source names none. The driver compiles the program for the device it finds.
- **The addresses:** with no shared virtual memory, card memory is spans inside up to 32 buffers (arenas) a card. An
  address carries its arena and offset, and every kernel is handed the arenas after its own arguments (`sys.cuh`).
- **Mapped host RAM** is two copies. The family writes the host copy to the card before a launch that names it, and
  reads it back when the card is waited for (`khronos_opencl1_igpu__zzpackage_launching`, `compute_completed`).

## Taking a device

`KHRONOS_OPENCL1_IGPU__TARGETS` in `includes.cuh` names the devices the family takes: today only `sm_120`, the test
card, which `nvidia_cuda12` takes first. A device is added once nn's verbs pass on it through this family
(`SILVANN_SKIP_BUILD=1 SILVANN_VERBS_FAMILY=khronos_opencl1_igpu bash scripts/src_abi_verbs_on_device.sh`) and a model
answers against HF (`test/src_runtime_qwen3_5.py`). On a box where another family takes the card, run it with
`SILVANN_SILICON_FAMILY_DIR=<a folder holding only this family's library and x86_avx2's>
RUNTIME_SILICON=khronos_opencl1_igpu`.
