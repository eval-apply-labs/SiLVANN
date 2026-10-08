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

A model's program is Lisp, written by hand, and it travels with the model: a model folder carries it in `lisp/`, and
its pack on Hugging Face with it. The boot reads the folder's files and hands every form to the evaluator, as a REPL
would; a procedure is then a binding like any other, which a program can call or define again. The files are one a
family — `qwen3_5.lisp`, `glm5.lisp`, `mistral3.lisp`, with `qwen3_5_cpu.lisp` and `glm5_cpu.lisp` for what a CPU worker
runs, and `sampling.lisp` for the sampler every model shares — written in the release's `lisp/`, and a model folder
holds its family's copy. A program names the words of the engine it was written for: one an engine
does not have is an error when the program reaches it, and a `defun` of that name in the program — the old word
written as the new one — answers it. A token's layers are a loop:

```lisp
(defun (layers pos)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i pos) (set! i (+ i 1))))
    (type x)))
```

and a layer reads its tables out of arrays indexed by its place, so one procedure serves every layer of its kind:

```lisp
(defun (layer i pos)                                   ; the 35B: the mixer, then the MoE as one word
  (begin (mixer i pos h1) (ai_qwen_3__moe h1 (nth MOE i) x) (type x)))
```

The settings that differ between boots — a pack that keeps its residual rotated, the attention cache in tiers, the
experts on the CPU, a card's tier of them — are names the boot binds true or false, and the file's `if`s choose the
procedures by them when it loads. `MEASURED` on the 35B: the loop costs nothing a run resolves against the layers
written out one by one (99.5-102.5 tok/s against 99.8-102.9), the host's work hidden behind the card's.

The family's runtime class (`qwen3_5.py` …) makes what the program reads — the collections, the load, the buffers,
the tables (a node array of a matrix's planes and the shape's integers; the cells each verb reads are named in its
package's `contracts/objects/`) — and binds the names the file is written against: `_bindings`. Three forms are the
machine's rather than the language's: `(worker N …)` defines its forms as worker N, `(picture name form)` freezes a form
for `sys__compute` to hand another block, `(view name)` snapshots the bindings a computed block sees — and a top-level
`if` choosing between `defun`s is defined as it stands (▶ `Machine.load_source`).

**A token** is one program the runtime writes a step — `(begin (position TOKEN POS) (head))` — the token's embedding read
by the card from a table in RAM, the layers, the head. **An answer** is `(generate TOKEN POS N …)` on the card, the next
token each step's input, greedy or drawn by the sampler, until an end token; the session asks for eight a call.
**The sampler** is `head_sample`: the repetition penalty, the temperature, the top k, the top-p cut and a draw seeded
by the conversation's position, all on the card (`nn__vector__penalize`, `nn__vector__top_k`, `nn__vector__draw`).

`models/<name>/programs.lisp` holds what a boot defined, with the key of what it was defined from — the pack, the
settings, the runtime's code and the folder's `lisp/` — for the boots that follow to read rather than define again.

**The card's tier of experts** is the `nn` package's: its expert records keep each expert's last two calls, and
`nn__expert__qualifies` moves one to the card when the older of them is newer than the older of the one it would
replace (and freely while the layer has room). The tier is a binding, `nn__expert_tier`: an entry for every worker, each
the tier its card holds (the card slot's layout, and where the CPUs keep every expert's parts, a layer at a time) or 0. A
program hands it to the verbs that use it and each reads its own worker's entry, so two cards hold two tiers under one
name. nn does the rest — `nn__expert_tier__visit` runs a token's picks through it, the `nn__expert_tier__note` word a
prompt chunk's, and `nn__expert_tier__stitch` puts an expert's parts back together in a card slot.

**The routed experts are run by nn** — `nn__routed__position` for a token's picks, `nn__routed__rows` for a prompt
chunk's (`engine/src/packages/nn/cpu/routed__*.cuh`): the picks in memory computed while the rest are read, the grouped
gate-and-up, the activation, the rotation, the weighted down, the sums in pick order. A model package's experts verb
reads its table into an `nn__routed` and keeps what is its own: the router, the hand's layout, the activation (Qwen's
SwiGLU, GLM's clamped one), a mask.

**A dense MLP is nn's word** — `nn__mlp__apply` for a position, `nn__mlp__rows` for a prompt chunk
(`engine/src/packages/nn/cpu/opcodes/mlp__abi.cuh`, its table's cells in `contracts/objects/mlp.cuh`): the RMS norm, the
gate and up in one grouped launch, the SwiGLU, the rotation, the down with the residual added in the same launch. It is
the MLP site of Qwen's dense models and of Ministral.

**The experts' sums do not depend on which expert was ready first.** A verb computes the experts that are in memory while
the rest arrive, but each pick's output is kept apart and the picks are added in their own order — a token's downs in
one launch, a prompt chunk's in a slot each, added by `expert_rows_reduce` — so the same prompt gives the same answer
bit for bit, whatever the system had paged in — except with a card's tier of experts, whose share of a token's picks
follows its promotions' timing: the card's sum and the CPUs' then split differently run to run, and round
differently. Over a prompt, an expert fewer positions pick than `nn__expert_major__vram_promotion_threshold_picks`
stays on the CPU: a variable bound at boot (`runtime/silvann_runtime/expert_tier.py`) by timing an expert's copy to the
card against the CPU computing one — 15 for GLM and 4 for the 35B on the test machine — and read by the prompt's
program.

## Adding a model

1. **Pack it.** The weights become a TurboQuant bundle: every matrix a record, rotated along its input and
   quantised to 4 or 8 bits, with a per-row scale. The packer's command line is not in this preview — the
   runtime carries only the decoder it needs (`runtime/nn_unpack_model.py`, `nn_compressor.py`) — so for now a
   new model means the published bundles' format, read from `runtime/nn_load_bundle.py`.
2. **Write its package** (`engine/src/packages/ai_<name>/`) if its layers are not ones nn or an existing package has:
   the tables' cells in `contracts/objects/`, the verbs in `cpu/opcodes/`, each a site of a layer written
   against `nn`'s doors. `engine/scripts/src_enroll.py` enrols a new package and `engine/scripts/src_assemble.sh`
   regenerates the manifests.
3. **Write its program and its runtime class**: the program in `lisp/<name>.lisp`, copied into the model folder's
   `lisp/`, and the class (`runtime/silvann_runtime/<name>.py`) with the collections and the boot's section, the load,
   the buffers, the tables, the names the program reads, `step` and `prefill`. `mistral3` is the smallest to copy from
   (nn's words only), `qwen3_5` the fullest; `silvann_runtime/__init__.py` maps an architecture name to its class.

## Adding a silicon

`engine/src/silicon_families/add_a_family.md`. A family answers a small contract: memory and copies, a launch,
and a handful of arithmetic primitives (the lanes' sums, a half to a float, the tile it likes its GEMM in). The
library's kernels are written once against those answers; a family may also override a door with its own code,
as `x86_avx2` and `amd_rocm6_wave64` do for the hottest ones. `khronos_opencl2` runs on the MI50 through ROCm's OpenCL
(nn's verbs, the 27B and the 35B against HF); `nvidia_cuda12` builds and passes the family checks, and has not been run on
a real card.

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
