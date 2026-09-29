# `src` — the engine

This tree is the Eval-Apply GPU Interpreter and the neural network library written in it — the two
halves of SiLVANN:

```
the Eval-Apply GPU Interpreter   a small lisp: its evaluator (engine/) and its language (packages/sys)
the Neural Network library       what a model is built with: buffers, experts, and the arithmetic of a
                                 layer, published as verbs of that lisp (packages/nn)
```

A model is a **program** in this language: a list of forms that name `nn`'s verbs and its own package's, run
by the evaluator, with the heavy arithmetic launched on a graphics card or on the CPU's cores. Two model
packages ship: `packages/ai_qwen_3` (Qwen 3.5, 3.6 and 3.8, dense and mixture of experts) and
`packages/ai_glm_5_3` (GLM 5.3 Flash). The Python side that boots a model folder onto the machine is
`silvann_runtime`.

## How it runs

- **The evaluator runs on the host CPU.** A machine is a set of runner threads — one per *block* — over
  one heap in host memory. Everything the evaluator touches is a 64-byte node on that heap. Why it is
  shaped this way: ▶ `docs/design/evaluator.md`.
- **Graphics cards are reached through *silicon families*.** A family is a runtime plus the traits a build
  is compiled for — `nvidia_cuda12`, `amd_rocm6_wave64` (ROCm 6, the 64-wide AMD cards), and
  `khronos_opencl2`, the last one a card is offered to before it is refused — and each is its own binary,
  `libsilvann_<family>.so`. The program never names a vendor: each package lists what it needs
  from a card, and each family answers it. ▶ `silicon_families/add_a_family.md`.
- **The Python wheel is the host.** `silvann_engine_cpu` finds the cards on the machine, gives each to the
  highest family that includes it and works, loads only those families' binaries, and exposes the engine
  to Python.

## Layout

```
manifest.cu          the one translation unit of the program
pybind.cuh           the Python module: the engine's calls, each package's calls under its name, and
                     the loading of silicon families
engine/              the evaluator
  eval.cuh             the scan: find the next form, hand it to its verb, pick up what it left
  launch.cuh           how runners are started — one, many, or resident until told to stop
  boot.cuh             standing a machine up and taking it down
  opcodes.cuh          what a parked block does between programs
  abi.cuh + abi/       the C interface a host calls: the machine, building programs, running them,
                       residency, silicon families, and the vocabulary
packages/            the language and the library, each a folder that owns its own names
  sys/                 the language — ▶ packages/sys/README.md
  nn/                  the neural network library
  manifest__header.cuh the package list, and what a package owes
  manifest.cuh,        GENERATED from the list: the packages as the program compiles them, and as a
  manifest__gpu.cuh    silicon family's binary does
silicon_families/    what answers the packages on real silicon: the roster, how a card is given to a
                     family, and one folder per family
docs/                design notes — docs/design/ for what is built, docs/design/pending/ for what is not
```

A package is `contracts/` (definitions only: objects, the lists it publishes, its interfaces), `cpu/`
(what runs), and — for `nn` — `gpu/` (its kernels and the doors that launch them). A new package is a new
folder: the next build enrols it and gives it the next id.

## Building

From `backstage/`:

```
bash scripts/rebuild_engine.sh            the wheel (silvann_engine_cpu), then every silicon family
                                          into build/silicon_families/
bash scripts/build_silicon_families.sh    the families alone
```

A family whose compiler is missing, or that does not answer everything the packages ask, is skipped with
the reason printed, and the rest still build.

## Using it

The language is written as text through a small composer, `python/silvann_runtime/composer.py`:

```python
import silvann_engine_cpu as e
from src_composer import Composer            # backstage/test/

e.boot()
c = Composer(e, e.bindings_create(8))
c.run("(+ 1 2)")                             # (kind, 3) — kind is sys__value_int in e.kinds()
c.run("(< 2 1)")                             # sys__value_false
e.shutdown()
```

Underneath, the module's calls are the engine's C interface, `engine/abi/`: boot and shut down, build
lists, cells, strings and names, freeze a program into a picture and thaw one back, run it, and keep a
machine resident and schedule work onto it. Each package's own calls sit under its name — `e.sys`
(looking inside the heap, the allocator, the fault channel) and `e.nn` (bytes in and out of a buffer on
the card). `e.kinds()`, `e.verbs()` and `e.packages()` list the vocabulary; `e.cards()` and
`e.silicon_families()` say what silicon was found and loaded.

## The code is C-shaped, and that is checked

The program is written in a C subset — no classes, no templates, no overloading — so that it can be
translated to C mechanically. A name carries its package, its type and its visibility
(`sys__stack__push`, `sys__stack__zzprivate_set_current_chunk`); ▶ `packages/sys/README.md` for the rules.

```
python3 scripts/src_c_subset_gate.py      the code stays C-shaped, and names stay within their tiers
python3 scripts/src_claim_gate.py         every claim a comment makes about the code still holds
```

## Testing

From `backstage/`:

```
bash scripts/src_selftest.sh                   the language, compiled for the host
python3 test/src_device_selftest.py            the whole engine, through the wheel
python3 test/src_program_selftest.py           programs that use nn's verbs
python3 test/src_composer_selftest.py          programs written as text
bash scripts/src_cpu_evaluator_check.sh        the evaluator on the host, three ways
bash scripts/src_abi_verbs_on_device.sh        every nn verb on a card, against the host
bash scripts/src_silicon_family_check.sh       the families load, and a worker runs on each device
bash scripts/src_silicon_family_choice.sh      which family a card goes to
python3 test/src_wheel_silicon_families.py     the wheel's choice, on this machine's cards
```
