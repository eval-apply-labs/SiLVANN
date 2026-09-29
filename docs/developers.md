# Working on SiLVANN

## The shape of it

```
engine/src/engine/                the Eval-Apply GPU Interpreter's evaluator
engine/src/packages/sys/          the language: lists, symbols, bindings, if, while, defun, compute and result
engine/src/packages/nn/           the Neural Network library: buffers, the expert collections, the arithmetic
engine/src/packages/ai_qwen_3/    the Qwen 3.5 family's layers as verbs
engine/src/packages/ai_glm_5_3/   GLM 5.3 Flash's layers as verbs
engine/src/silicon_families/      one folder a kind of silicon: what it answers for the packages
runtime/silvann_runtime/          Python: a model folder booted onto the machine, its tables and programs
```

A **model is a program**. The runtime writes, for each layer, a short program naming the model package's verbs
over tables of that layer's weights; the evaluator runs it on the host, and each verb launches its arithmetic on
the silicon that holds the data — a card, or the CPU's cores. The machine is a set of **workers**, one a device:
the card is worker 0, a CPU socket a worker of the `x86_avx2` family, and a program hands work to another worker
with `sys__compute` and collects it with `sys__result`.

Every package has a README; read `engine/src/README.md` first, then `sys`'s and `nn`'s.

## Where the programs are

A model's programs are written by its runtime class, as text, when the model boots — they depend on the model's
shape and on the settings it boots with, so they are generated rather than kept in files:

| runtime | where its programs are written |
|---|---|
| `runtime/silvann_runtime/qwen3_5.py` | `_procedures` (a layer at a position, a layer over a chunk of rows, the embedding, the head), `_tables` (what each verb reads), `_cpu_experts` (the pictures a CPU worker runs) |
| `runtime/silvann_runtime/glm5.py` | `_procedures` and `_fused_tables` (a token, the hand-off to the CPU sockets), `_rows_procedures` (a prompt as rows) |

**Every boot writes them out**: `models/<name>/programs.lisp` holds every procedure and picture the model defined,
in order, one form a line. The published models' programs, at their default settings, are in
[`docs/programs/`](programs/) and in each model's Hugging Face folder. A layer of the 35B reads:

```lisp
(defun (layer0 pos) (begin
    (ai_qwen_3__deltanet x mx0 h1)
    (ai_qwen_3__moe h1 mr0 x)
    (type x)))
```

and a token is the one program the runtime composes a step — `(begin (embed 0 0) (nn__rope__angles cs POS …)
(layers POS) (head))` — in `step`. A table (`mx0`, `mr0`) is a node array of the weights' planes and the shape's
integers; the cells each verb reads are named in its package's `contracts/objects/`.

## Adding a model

1. **Pack it.** The weights become a TurboQuant bundle: every matrix a record, rotated along its input and
   quantised to 4 or 8 bits, with a per-row scale. The packer's command line is not in this preview — the
   runtime carries only the decoder it needs (`runtime/nn_unpack_model.py`, `nn_compressor.py`) — so for now a
   new model means the published bundles' format, read from `runtime/nn_load_bundle.py`.
2. **Write its package** (`engine/src/packages/ai_<name>/`) if its layers are not ones an existing package has:
   the tables' cells in `contracts/objects/`, the verbs in `cpu/opcodes/`, each a site of a layer written
   against `nn`'s doors. `engine/scripts/src_enroll.py` enrols a new package and `engine/scripts/src_assemble.sh`
   regenerates the manifests.
3. **Write its runtime class** (`runtime/silvann_runtime/<name>.py`): the collections and the boot's section,
   the load, the buffers, the tables, the programs, `step` and `prefill`. `qwen3_5.py` and `glm5.py` are the
   two to copy from; `silvann_runtime/__init__.py` maps an architecture name to its class.

## Adding a silicon

`engine/src/silicon_families/add_a_family.md`. A family answers a small contract: memory and copies, a launch,
and a handful of arithmetic primitives (the lanes' sums, a half to a float, the tile it likes its GEMM in). The
library's kernels are written once against those answers; a family may also override a door with its own code,
as `x86_avx2` and `amd_rocm6_wave64` do for the hottest ones. The `nvidia_cuda12` and `khronos_opencl2` families
build and pass the family checks, and have not been run on a real card.

## When a card faults

A kernel that reads a bad address does not come back with a stack trace. On AMD the runtime prints one line —
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` — and aborts, and nothing after it runs. Find it with the platform's
GPU debugger, which stops the wave at the fault before the runtime aborts:

| | AMD | NVIDIA |
|---|---|---|
| debugger | `rocgdb` (stock `gdb` cannot) | `cuda-gdb` |
| memory checker | none ships with ROCm 6 | `compute-sanitizer` — reach for it first |
| exact instruction | `set amdgpu precise-memory on` | `set cuda memcheck on` |

The precise-memory switches are off by default because they serialise every memory access; without them the
reported instruction may sit a few past the faulting one — enough to name the kernel and the region, and the
surrounding instructions usually pin the line. When you think an address belongs to a known structure, check a
field of it you can predict independently before believing it.

## Checking a change

The engine's own suites and the model checks live in the development repository, not in this release. What
every change was held to there: the engine's self-tests, the claim and C-subset gates, and for a model, its
answer against Hugging Face's reference implementation and its prompt read as rows against a position at a time.
