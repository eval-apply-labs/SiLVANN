# Adding a silicon family

A **silicon family** is a runtime plus the traits a build is compiled for — ROCm 6 on 64-wide AMD
cards, CUDA 12 on NVIDIA cards from Turing on. A card is named by its instruction set (`gfx906`,
`sm_86`), and it goes to the highest-numbered family that includes it and works on this machine.

The packages **ask** — each lists in its `contracts/abi/gpu.cuh` what it needs from silicon — and a
family **answers**. A family's folder holds only what differs between silicon: the vendor's calls
for memory, copies and waiting, the one call that starts a kernel, and the arithmetic whose results
differ by vendor. The kernels, the doors that launch them and everything else belong to the packages,
and a family compiles them against its answers.

## What a family is made of

```
silicon_families/<family>/
  includes.cuh   which cards it takes — a TARGETS row, and <family>__includes(card)
  sys.cuh        sys's doors: memory, copies, waiting, the side channel, and the devices
  nn.cuh         nn's requests: nn__silicon__launch, nn__silicon__expf / sqrtf / logf / cosf / sinf,
                 the lanes and blocks, the combine step and half_to_float
  <package>.cuh  one per package the family supports: that package's requests
                 (packages/<package>/contracts/abi/gpu.cuh), answered
  manifest.cuh   includes each <package>.cuh above, and nothing else — no one outside the family names them
  unit.cuh       its whole translation unit, which the build compiles into libsilvann_<family>.so
```

⚖ *"a folder for primitives and one for overrides so it is clear what is expected to run and what are
performance upgrades"*. A family may split itself in two, as `x86_avx2/` does:

```
  primitives/    the answers above — sys.cuh, nn.cuh, and whatever they share. Required: the family does
                 not build without every one (the conformance rule reads primitives/ as the family's own)
  overrides/     a door's faster body, put into the package's table in place of the generic one by the
                 hook `SYS__SILICON_FAMILY__OVERRIDE_DOORS(tables)` (entry.cuh). Optional: an override
                 falls back to the package's generic door for anything it does not cover, so removing
                 the folder loses nothing but speed
```
An override computes what the generic body computes, and a package publishes what an override may reuse
of its definitions as `zzabi_` functions (nn's: `nn__turboquant__zzabi_levels`, `_levels_i8`, `_quantise`,
`nn__kernels__zzabi_to_half`), so an override never restates a definition or reaches into a private name.
`amd_rocm6_wave64/` has an `overrides/` (the TurboQuant gemvs at 4 bits through `v_dot4_i32_i8` and
`v_dot2_f32_f16`, the exact one at 8 bits too, and the rotation as a butterfly through wave shuffles) while its
primitives still sit at the family's root.

⛳ A FAMILY WHOSE BLOCKS ARE THREADS OF THIS PROCESS — the CPU — defines
`SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS` in its `unit.cuh`. nn then compiles no kernel: each door packs
its arguments into a struct and hands the launch a block function, which the family's pool runs once for
every block. ▶ `x86_avx2/primitives/pool.cuh` for a pool sized to the machine's free cores.

`sys.cuh` and `nn.cuh` are two instances of the same rule: **a family supports a package by answering
what that package asks.** A new package that runs on a card lists its requests in its own
`contracts/abi/gpu.cuh`, and each family that supports it gains a `<package>.cuh` answering them.

## Steps

### 1. Name it and give it its number

The name is `<vendor>_<runtime><major>_<trait>` — `amd_rocm7_wave64`, `nvidia_cuda13`. It is the
folder's name, the prefix of every function the family defines, the name of its binary and the name
a boot asks for it by.

Its number places it in the roster, `SILICON_FAMILIES__LIST` in `roster.cuh`: the host is 0,
`khronos_opencl2` 100, `amd_rocm6_wave64` 200, `nvidia_cuda12` 300. Numbers leave gaps on purpose, so a family can be placed
between two others; they must rise down the list, and the build refuses one that does not, or one past
`SYS__SILICON_FAMILY__SLOTS`. Numbers never move and are never reused, because a machine records the
number of the silicon it stood up on.

⛔ **The number is also the preference,** which is why the roster is tended by hand and not enrolled
like a package is. A card that several families include is offered to the highest number first. So a
new family must never be a worse choice than an older one that takes the same cards. If it would be,
list fewer cards in it; renumbering is not an option.

### 2. `includes.cuh` — the cards it takes

```c
#define AMD_ROCM7_WAVE64__TARGETS(X)  X(gfx942) X(gfx950)

static inline bool amd_rocm7_wave64__includes(const char* card) { … }
```

Copy `amd_rocm6_wave64/includes.cuh` and change the row. The row is the one list of what the family
is: the build script reads its targets from it, so a family cannot take a card its binary has no
code for. List only cards the vendor's support matrix says the runtime supports — AMD's ROCm
compatibility matrix, NVIDIA's CUDA release notes. A card not named here is not taken, even a newer
one of the same line.

### 3. `sys.cuh` — sys's doors

Every door in `packages/sys/contracts/abi/gpu.cuh`, each a plain host function named
`<family>__<door>`, taking the door's parameters. `amd_rocm6_wave64/sys.cuh` is the model: each is one
or two calls into the vendor's runtime.

Three of them are the devices, and they count **only the cards this family includes**, in the
vendor's order. The vendor numbers every card its driver shows; a family numbers its own:

```
device_count()     how many cards the runtime shows that <family>__includes() takes
bind_device(i)     make the i-th of those the calling thread's
bound_device()     which of those the calling thread is on, or 0xFFFFFFFF
```

A machine with no card of the family — or no driver — answers zero rather than failing.

### 4. `nn.cuh` — nn's requests

```c
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks,
                                       uint32_t threads, void** args, const size_t* sizes,
                                       uint32_t count);         /* start one kernel */
static __device__ inline float nn__silicon__expf(float x);
static __device__ inline float nn__silicon__sqrtf(float x);
static __device__ inline float nn__silicon__logf(float x);
static __device__ inline uint32_t nn__silicon__lane(void);   /* which lane of its block a body runs on */
static __device__ inline uint32_t nn__silicon__lanes(void);  /* how many lanes the block has */
static __device__ inline uint32_t nn__silicon__block(void);  /* which block, of how many */
static __device__ inline uint32_t nn__silicon__blocks(void);
static __device__ inline float nn__silicon__lanes_sum(float part);  /* every lane's part, added; waits for all */
static __device__ inline float nn__silicon__lanes_max(float part);
static __device__ inline void nn__silicon__lanes_sums(float* parts, uint32_t n);  /* n sums at once, each as lanes_sum */
static __device__ inline float nn__silicon__half_to_float(uint16_t h);  /* exact: the card's own instruction */
```

`nn__silicon__launch` starts a kernel on `blocks` of `threads`, with `args` pointing at each of its
`count` arguments and `sizes` giving each one's width. A runtime that launches by pointer —
`hipLaunchKernel`, `cudaLaunchKernel` — takes `kernel` and ignores the rest. One that finds kernels by
name in a program it compiled takes `name`, `sizes` and `count`: `khronos_opencl2` is that case, and
defines `SYS__SILICON_FAMILY__KERNELS_BY_NAME` so nn compiles no kernel for it, and its kernels come
from `scripts/opencl_kernel_source.py` instead. nn's doors (`packages/nn/gpu/doors.cuh`) are written
once against it, for every family.
The arithmetic is the precise function, never a fast approximation: `packages/nn/contracts/abi/gpu.cuh`
says why for each. The lanes are the runtime's own thread index and block width (`threadIdx.x`,
`blockDim.x`; `get_local_id(0)`, `get_local_size(0)`), and a family that runs no kernel answers 0 and 1.
A door starts at most `NN__SILICON__LANES_MAX` lanes in a block — 256, the MI50's OpenCL limit — so a
family must run a block of 256.

### 5. `unit.cuh` — the translation unit

```c
#include <the vendor's runtime header>
#include "manifest.cuh"
#include "../../packages/manifest__gpu.cuh"
#define SYS__SILICON_FAMILY__THIS <family>
#include "../entry.cuh"
```

Nothing else. The family's own `manifest.cuh` names its answers, one `<package>.cuh` each, so supporting a
new package is a file and a line there. Which headers each package's card side needs is that package's
business (`packages/<pkg>/manifest__gpu.cuh`), so a family names no package header.

### 6. The roster

In `roster.cuh`, include the family's `includes.cuh` beside the others and add its row:

```c
X(3u, AMD_ROCM7_WAVE64, amd_rocm7_wave64, amd)
```

The last column is the vendor. **Two families of one vendor are never opened in one process.** A
second ROCm's HIP would be bound to the first one's HSA runtime, which the loader matches by name.
Cards that need two families of one vendor need a process each.

### 7. The build

In `scripts/build_silicon_families.sh`, add a case for the family: its compiler, reading its targets
from its TARGETS row, and building `unit.cuh` into `build/silicon_families/libsilvann_<family>.so`
with hidden visibility. Copy the nearest existing case. The script builds every family on the roster
but the host, so a family with no case is reported as having no recipe.
The build checks that the binary exports `silvann_silicon_family_get` and nothing else of ours.

### 8. A new vendor only: naming its cards

`cards_present.cuh` finds the cards on the machine **before any family is loaded**, from the kernel
driver or the vendor's driver library — never from a vendor runtime. A new vendor needs a way to
name its cards there, in the same form its TARGETS row uses.

## Proving it

```
python3 scripts/src_claim_gate.py            silicon_seam_conformance checks every door and request
                                             the family defines, by signature and scope
python3 scripts/src_c_subset_gate.py         names stay in their tiers
bash scripts/src_silicon_family_choice.sh    the order and the rules of the choice
bash scripts/build_silicon_families.sh       it builds, and exports one symbol
```

And on a machine with one of its cards:

```
bash scripts/src_silicon_family_check.sh     it loads, counts its devices, and a worker runs on each
bash scripts/src_abi_verbs_on_device.sh      every nn verb on the card against the host
python3 test/src_wheel_silicon_families.py   the wheel gives the card to it
```

`e.cards()` shows every card, the family that took it, and every family it was offered to, with the
reason a family was passed over.

## A family that does not answer everything

A family can be incomplete while it is being written, or when a package adds a request that no one
has answered for it yet. Before compiling a family, the build asks which requests it does not answer
(`scripts/silicon_family_unanswered.py <family>`, using the same reading as the claim gate). If there
are any:

- the family is **skipped**, not compiled, and the build carries on with the others and exits cleanly:
  `⚠ amd_rocm7_wave64 skipped — not built: it does not answer nn__silicon__expf`;
- the reason is left beside where the binary would be, as `libsilvann_<family>.skipped`, and the wheel
  reads it. `e.cards()` shows it against the card, and the card goes to the next family down that
  includes it, or to none;
- the claim gate notes the skip and stays green, but still fails on anything the family defines with
  the wrong signature.

The host family is compiled into the program itself, so a request it does not answer is a program that
does not build, and the gate fails on it.

## What does not go in a family

- **A kernel, or a door.** Those are the package's, in `packages/<pkg>/gpu/`, written once against
  `nn__silicon__launch`.
- **Anything a package did not ask for.** If a package needs something new from silicon, the package
  declares it in its `contracts/abi/gpu.cuh`. The conformance rule then requires it of every family,
  and each family answers it.
- **A package's headers.** `unit.cuh` includes `packages/manifest__gpu.cuh` and nothing else of the
  packages.
