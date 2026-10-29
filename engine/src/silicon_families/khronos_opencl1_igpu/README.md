# khronos_opencl1_igpu — OpenCL 1.2, for old integrated GPUs

**The target is an old integrated GPU:** one whose OpenCL lacks what `khronos_opencl2` needs (shared virtual memory, the
generic address space), on a machine with no CUDA or ROCm and a CPU too old for `x86_avx2` (no AVX2) or too slow to beat
its own GPU. `ASSUMED` — none has run this family yet. What would falsify it: a driver that will not build the program,
or a verb that fails on one.

**Which OpenCL driver a GPU has decides whether it is reachable.** The graphics driver is a separate thing: a desktop
can draw with Mesa and have no OpenCL at all. Intel's Ivy Bridge and Haswell GPUs (2012-2013, the Iris Pro among them)
and the Bay Trail Atoms stop at OpenCL 1.2.
- **Beignet**, Intel's own Linux OpenCL for them, is not in Ubuntu 22.04, the system SiLVANN is tested on. `MEASURED`:
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
  That laptop runs on its CPU (`x86_avx2`).

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
  kept twice, once in the program's memory and once on the card. nn's RAM span holds the embedding table (1.27 GB), and
  the one kernel a token that reads a row of it sends the whole span over PCIe in both directions: a launch cannot tell
  a read from a write, so the card's copy is taken as the newer and read back whole when the card is waited for.
- **258 ms are the kernels:** nn's generic bodies, with no overrides. As OpenCL C 1.2 they run about 1.9× slower than
  CUDA's compile of the same bodies, which takes 135 ms for the whole token.

▶ `measurements/2026-10-13_khronos_opencl1.md` (written under the family's first name, `khronos_opencl1`).

**On an integrated GPU the two copies are one (NN-73, 2026-10-20),** because the card's memory is the host's RAM: where
the device says `CL_DEVICE_HOST_UNIFIED_MEMORY`, a span of mapped RAM is a buffer of its own over the program's copy
(`CL_MEM_USE_HOST_PTR`), handed to the card by unmapping it and back by mapping it, which moves nothing. A laptop's GPU, an
APU's. Kept only where mapping answers the program's own address; anything else is the two copies, and
`SILVANN_OPENCL1_UNIFIED=0` asks for them. `MEASURED` on PoCL (a CPU's OpenCL, unified, on node-01's Xeon, through
`SILVANN_OPENCL1_TAKE=pthread` and `SILVANN_SILICON_CARDS=sm_120`): nn's verbs 335/0 either way, PoCL's log showing the
host-pointer buffers only with it; the 35B-A3B against HF 6/0 either way, the same logits, a token 2852 ms against
3088 — there the kernels on the CPU are most of a token; on a GPU's faster kernels the copies are the larger share.
`ASSUMED` on Intel's and AMD's integrated GPUs, never run on one. ⚠ A runtime can answer the program's address when
mapped and still not be one memory: Apple's (an M4, 2026-10-20, before macOS was dropped) faulted at the first heap
object with it and booted without it.

**The embedding on the CPU (NN-96, 2026-10-26).** With the experts on the CPU, the Qwen models keep the embedding table
in the CPU worker's RAM on this family and the CPU decodes a token's row, which is copied to the card (`embed_on`,
`docs/config.md`): the table takes no room on the card and nothing of it crosses. `MEASURED`
on the RTX 5060 Ti, the 35B D8E4 against HF: a generated token 83 ms against 231 with the table on the card, the same
checks and continuation (▶ `measurements/2026-10-26_nn96_embedding_cpu_precision.md`).

## How it works

- **The kernels:** `scripts/opencl1_kernel_source.py` turns nn's card side into one OpenCL C 1.2 program. It parses the
  code with pycparser (`scripts/third_party/`) and works out every pointer's address space starting from the kernels,
  so nn's source names none. The driver compiles the program for the device it finds.
- **The addresses:** with no shared virtual memory, card memory is spans inside up to 32 buffers (arenas) a card. An
  address carries its arena and offset, and every kernel is handed the arenas after its own arguments (`sys.cuh`).
- **Mapped host RAM** is two copies, except on a device with unified memory (above). The family writes the host copy to the card before a launch that names it, and
  reads it back when the card is waited for (`khronos_opencl1_igpu__zzpackage_launching`, `compute_completed`).

## Its overrides (NN-124)

Most doors run nn's own body; a few run the family's own kernels (`overrides/`), each nn's card body to the bit and
falling back to nn's for what it does not cover. Their kernels are written in nn's card dialect (`overrides/kernels.cuh`)
and the build writes them into the same OpenCL C 1.2 program, from the file's `KHRONOS_OPENCL1_IGPU__OVERRIDE_ROWS`; the
doors (`overrides/doors.cuh`) launch them by name. They are KDA's steps over a prompt's rows (a lane's column of the state
kept in its private memory for every row), MLA's absorb and expand (a weight loaded once for several rows), the indexer's
pool scores (several pools a block) and the mix over rows of their own (a value loaded once for several heads). What
OpenCL C 1.2 lacks — shuffles, and block-shared memory but the folds' — keeps the latent's scores in tiles and the
indexer's top `k` by ballots on nn's bodies here; the CUDA and AMD families have both.

**The GEMVs and the rotation (NN-96, 2026-10-26).** nn's generic GEMV makes a block one row and folds its 256 lanes for
it; here a block is sixteen runs of 16 lanes, a run a row, each lane reading sixteen bytes of codes a step, the levels as
floats from the card's memory rather than the program's constants (which a card serves one lane at a time when the
lanes ask for different ones), the runs folded at once (`khronos_opencl1_igpu__zzpackage_lanes_sum_each`, a fold the generator writes for the
overrides alone). The weighted sum onto the residual (the MoE's downs) is the same shape; the two-input GEMV runs the
one-input kernel an input at a time. nn's Hadamard rotation applies each output's row of the matrix directly, a lane an
output and 512 terms a lane; here it is the butterfly, a block of the ladder a block of 256 lanes, its stages through the
fold's scratch (`khronos_opencl1_igpu__zzpackage_lanes_hadamard`). These outputs are within the bench's bound of nn's, not its bytes, as
CUDA's and AMD's are. `MEASURED` on the RTX 5060 Ti: the 35B's head 1.32 ms (385 GB/s), its MoE down onto the residual
310 µs → 29, a rotation 27-58 µs → 4.1 at every width; the 35B D8E4 against HF 6/0 at 37.7 ms a token, against 83.0 with
nn's bodies (▶ `measurements/2026-10-26_nn96_opencl1_gemv_precision.md`).

**GLM's small doors and the sampler's top k (NN-137, 2026-10-28),** each nn's card body to the bit. A site's `pre` at four
streams (halves and floats), its mix's Sinkhorn steps in the first lane's registers where nn's body sizes its arrays by
the streams it is handed; the map's logits, a lane's sixteen reads in flight; KDA's step a row, a lane's column in its
private memory between the step's two passes. The sampler's top `k` over a vocabulary: each lane's largest, a bar that
`k` lanes' largest reach (sixteen folds bisecting the halves), each lane's values at or above it, then nn's rounds over
those alone, where nn's body makes `k` passes over every logit on one block. MLA's one-query scores (GLM's and DeepSeek's
latent): a tile of one, two or four positions an item where nn's is eight, as many items as fill the card, at a decode's
short context. MLA's expand a run of 32 lanes an output (nn's fold's first three steps in a lane, its last five the
run's), and the norm of a row of halves a lane's sixteen reads in flight (▶ `measurements/2026-10-28_opencl1_expand_norm_precision.md`:
the expand at decode 133 → 75 us, the norm 13.5 → 7.6 us in GLM, nn's bytes). `MEASURED` on the RTX 5060 Ti, the bench:
`hyper_pre` 39.6 → 11.8 us, `hyper_pre_fp32` 41.2 → 11.7, `hyper_logits` 18.9 → 12.0, `kda_step` 39.2 → 29.7, `top_k`
over 248,320 logits 7.2 ms → 143 us (top 64: 86 ms → 197 us), MLA's scores at 37 positions 108 → 36 us; every output the same bytes as nn's bodies
(▶ `measurements/2026-10-28_opencl1_glm_doors_precision.md`).

**The GEMV two or four rows a run (2026-10-28):** a lane reads `x`'s halves once for its run's rows' codes, each row's
sum the one-row kernel's bytes; four from 2,048 rows, two from 768. `MEASURED` on the RTX 5060 Ti: GLM's attention
output 150 → 87 us, the 27B D4 14.6 → 17.3 tokens/s (▶ `measurements/2026-10-28_opencl1_gemv_rows_precision.md`).

**Two shapes, an integrated GPU's and a discrete card's.** An integrated GPU shares the host's memory and its bandwidth,
so there each load serves twice as many outputs (sixteen rows, pools or heads against eight). The doors take the
integrated shape where the device says `CL_DEVICE_HOST_UNIFIED_MEMORY`; `SILVANN_OPENCL1_SHAPE=igpu` or `=discrete`
asks for either. `REASONED` from that premise. Both shapes are checked on the RTX 5060 Ti through NVIDIA's OpenCL,
each forced, against nn's bodies byte for byte (▶ `measurements/2026-10-25_nn124_precision.md`); neither has run on an
integrated GPU.

## Taking a device

`KHRONOS_OPENCL1_IGPU__TARGETS` in `includes.cuh` names the devices the family has run on: today `sm_120`, the test
card, which `nvidia_cuda12` takes first. It also takes every card `opencl_versions.cuh` judges to be OpenCL 1.2 only — a
best guess from the card's name, a developer preview. A device is added once nn's verbs pass on it through this family
(`SILVANN_SKIP_BUILD=1 SILVANN_VERBS_FAMILY=khronos_opencl1_igpu bash scripts/src_abi_verbs_on_device.sh`) and a model
answers against HF (`test/src_runtime_qwen3_5.py`). On a box where another family takes the card, run it with
`SILVANN_SILICON_FAMILY_DIR=<a folder holding only this family's library and x86_avx2's>
RUNTIME_SILICON=khronos_opencl1_igpu`.
