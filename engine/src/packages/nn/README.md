# `nn` — the Neural Network library

`nn` is what a model is built with: the arithmetic its layers are made of, the structures that keep its
weights, activations and KV cache within reach of a card, and the verbs a model program is written in.
It is the second half of SiLVANN. The first half is the Eval-Apply GPU Interpreter — the language
(`sys`, next door) and the evaluator that walks a program (`engine/`). `nn` is written against `sys` and
never the other way about: it adds kinds, objects and verbs to the language through the same three lists
`sys` uses, and a program calls `nn__vector__add` exactly as it calls `add`.

The evaluator runs on the host CPU, one runner thread per block. An `nn` verb is a call on one of those
runners; when it has arithmetic to do on a card, it hands the work to the card through a **door**, and
the card runs a **body** — a plain loop over raw pointers that knows nothing of the language.

## The ideas, in the order you meet them

**Storage is fp16; arithmetic is fp32.** A buffer never knows its element type: it answers an address and
a byte count, and each verb casts for itself. Every element is two bytes (`NN__PRIMITIVES__ELEMENT_BYTES`);
every body loads to `float`, computes in `float`, and rounds once on the way out. The one exception is the
DeltaNet recurrent state, which is fp32 because its decay multiplies into the same array every token.
▶ `contracts/objects/primitives.cuh`.

**An overflow is reported, not thrown.** A value too large for a half is stored as `inf`, as IEEE says,
and the body writes `NN__PRIMITIVES__FAULT_OVERFLOW` ("NPOV") into the worker's fault word, which reaches
the machine's fault channel at the next apply boundary. The value does not become an error object: that
would cost a synchronisation on every verb call (`MEASURED` on gfx906: 1.654 µs to launch and keep going,
28.816 µs to launch, wait and read the word back). ▶ `contracts/objects/primitives.cuh`.

**Kinds and verbs are rows.** `nn`'s kinds, objects and verbs are the same X-macro rows as `sys`'s, tagged
with this package's id so no two packages collide. From Python, `e.kinds()` and `e.verbs()` list them
alongside the language's. Count them rather than trusting a number here (today 3 kinds and 45 verbs):

```bash
cd backstage/src/packages/nn
grep -c 'X(PKG' contracts/macros/language_contract__kinds.cuh
grep -c 'X(PKG' contracts/macros/language_contract__verbs.cuh
```

**The kinds.** All three are objects — reference-counted, with a row in
`contracts/macros/language_contract__objects.cuh` — and none of them keeps its bytes in the heap. The
64-byte reference lives in the heap; the bytes it names live in `nn`'s own span or in a page.

| | what it is | what its release does |
|---|---|---|
| `nn__buffer` | an activation: an address and a class, lent from the pool | files it back in its class's free list |
| `nn__kv_ref` | a place in the conversation's KV cache, and the format its bytes are in | nothing — the cartridge owns the bytes |
| `nn__weights` | weights resident in a slot, or a view into a buffer: an address and a length | gives back its hold on the buffer it views, if any; a slot's bytes are the page's |

None of them can be built with `create` or cloned: a program asks a verb for one, and the verb decides
which slot it gets.

**One span, carved once.** At boot the host sums the sizes the config names and `nn` takes one allocation,
carved by arithmetic into its tenants: the activation buffers (five classes, one per role — `resid`,
`main`, `attn_out`, `logits`, `partials`), the
expert L1 arena, and the conversation cartridge (the KV tiers `sink`, `tq4`, `tq6`, `fp32`, the text and
the DeltaNet state). A sixth buffer class, `result`, is carved from a separate span of pinned RAM the
card can write. A config naming no `nn` key takes no span, and the package stands up without one.
The package's own device-wide state sits in its rows of the system register, `nn__HATCH_ROW_LIST` in
`contracts/objects/package.cuh`.

## The verbs, by what they are for

```
standing up         nn__package__init
buffers             nn__buffer__getnew  nn__buffer__getnew_ram  nn__vector__range  nn__vector__zero
vector arithmetic   nn__vector__add  nn__vector__pointwise_mul  nn__vector__scale  nn__vector__scale_at
                    nn__vector__exp  nn__vector__softplus  nn__vector__dot_product
matrix arithmetic   nn__expert__multiply_fp16  nn__matrix__matvec_transposed  nn__matrix__transpose
norms               nn__rmsnorm__apply  nn__vector__l2norm
activations         nn__softmax__apply  nn__sigmoid__apply  nn__swiglu__combine
attention           nn__rope__angles  nn__rope__apply  nn__attention__decode
                    nn__attention__residual  nn__attention__merge  nn__attention__finish
deltanet            nn__deltanet__rank_1_update  nn__deltanet__readout  nn__deltanet__conv_step
turboquant codec    nn__turboquant__decode  nn__turboquant__gemv  nn__turboquant__gemv_int8
hadamard            nn__hadamard__rotate
experts, weights    nn__expert__layer  nn__expert__reserve  nn__expert__assign  nn__expert__revoke
                    nn__expert__handed  nn__expert__plane
routing             nn__vector__top_k
reading values      nn__argmax__find  nn__vector__at  nn__vector__get_values_at  nn__buffer__read
the cartridge       nn__kv__reference
```

A few things worth knowing before reading them:

- **An arithmetic verb takes its output as an operand and answers it**, so calls chain:
  `(nn__swiglu__combine gate up out n)` answers `out`. Passing `nn__vector__add` the same buffer as an
  input and as the output is how a residual accumulates in place.
- **`nn__vector__l2norm` is a sum of squares, not a mean.** `nn__rmsnorm__apply` divides by `n`; this does
  not.
- **Two ways to scale.** `nn__vector__scale` takes a factor the program holds, as a `sys__value_float`;
  `nn__vector__scale_at` reads one out of a buffer something upstream computed.
- **A verb whose answer is a number writes it into a buffer.** `argmax`, `dot_product` and `at` store one
  stamped 64-bit word — kind, a ticket naming the call, and the value — into their `out` buffer and
  answer the buffer; `nn__buffer__read` turns it into a `sys__value_int` or `sys__value_float` when the
  program needs the number. Put `out` in a `nn__buffer__getnew_ram` buffer and the read watches the word
  land in pinned RAM instead of waiting on the card (`MEASURED` on gfx906: 4.7 µs against 27 µs for a
  launch and a copy back). ▶ `contracts/objects/result.cuh`.
- **The router's choice lands in nodes the program made.** `(nn__vector__top_k x n k picks weights)`
  writes the `k` largest indices into the value words of `picks`, a node array the program created, and
  their softmax — HF's renormalised router scores — into `weights` as halves; the program reads expert `j`
  with `(sys__node_array__get picks j)`. The boot registers the heap with the worker's card so the card
  writes the nodes itself; a family that cannot (OpenCL on the MI50) has the verb copy them in.
- **A weight is loaded once and read where it lies.** The host loads a matrix with
  `reserve` → `nn.write` → `assign` (`python/nn_load_bundle.py`); `nn__expert__plane` then mints an
  `nn__weights` for a span of that slot, bounds-checked against the slot, and every arithmetic verb takes
  it as it takes a buffer. `nn__vector__range` mints the same kind as a view into a buffer, which is how
  a program cuts one wide projection into heads. ▶ `src/docs/design/weight_load.md`.
- **The codec's verbs are rows of their own.** `nn__turboquant__gemv` multiplies by a quantised matrix
  without expanding it; `nn__turboquant__decode` expands it to fp16. `nn__hadamard__rotate` is the
  rotation the Gaussian family depends on, with the sign flip fused in so half a transform cannot be
  applied. ▶ `src/docs/design/pending/quant_families.md`.
- **Two gemvs, and the program picks.** `nn__turboquant__gemv` is exact in fp32; `nn__turboquant__gemv_int8`
  quantises `x` to int8 in blocks of 32 and multiplies against the levels in int8 (a generated table beside the
  fp16 one), ~0.5–0.7% of the rows' rms away from it. They are two verbs so a boot program writes its layer
  `defun` with the one it wants. The generic body defines the arithmetic; a family's faster body for a door
  goes in its `overrides/`.
- **DeltaNet at one token is one verb, every head in one launch.** `nn__deltanet__step` reads and updates each
  head's fp32 state in place and writes its gated, normed readout. The three verbs it was built from
  (`rank_1_update`, `readout`, `conv_step`) remain, and `conv_step` still runs before it. ▶ `cpu/deltanet__header.cuh`.
- **Attention is one query at a time.** `nn__rope__angles` writes a position's cos and sin (fp32) on the
  card, once for every layer, and `nn__rope__apply` rotates every head by them; `nn__attention__decode` is one query's
  attention over positions `from` to `to` of a `[position][kv_head][head_dim]` cache, grouped-query — so a
  sentence is a range of the one cache and needs no pages of its own. Several ranges — a conversation
  with a sentence evicted from its middle — are one `nn__attention__residual` each (per head the largest
  score, the sum of weights and the unnormalised weighted `v`), folded by `nn__attention__merge` and
  turned into halves by `nn__attention__finish`: the same softmax as one call over all of them.
  There is no cache-writer verb: a program points `v_proj`'s output, and RoPE's output for `k`, at the
  cache's row for the position. These shapes are proposals awaiting a ruling. ▶ `cpu/opcodes/attention__abi.cuh`.

## How a verb reaches a card

```
a program             (nn__vector__add x y out n)
the evaluator         finds the verb's row and calls it on the runner that holds the form
the verb              checks arity, kinds and bounds (nn__primitives__room, nn__primitives__fits)
                      finds the calling worker's doors (cpu/doors.cuh) — none: refuses with "NPND"
the door              nn__vector__zzabi_launch_add: packs a pointer to each argument and calls
                      nn__silicon__launch on the family the worker's card belongs to
the card              runs the body, nn__vector__zzabi_body_add, and returns; nothing waits for it
```

The verbs that touch a card's bytes are written in the evaluator's verb shape and reach it through
`SYS__ENGINE__ABI__BRIDGE`; the buffer pool, the expert index and the KV resolver work on the heap and
`nn`'s own tables and launch nothing.

**The doors are one list.** `nn__CONTRACT__DOORS` in `contracts/abi/gpu.cuh` has one row per door: its
name, its function and its parameters, type for type with its kernel's. From that list come the table a
family fills (`nn__doors`), the program's calls through it, and — in `gpu/doors.cuh` — one `__global__`
kernel and one `extern "C"` door per row. A row is only ever appended; the table carries its size, and a
program never reads past the size it was built with.

**Each door has one `__RUNS` line** in `gpu/doors.cuh` saying what it runs and how wide:

```
the form     WIDE       every lane runs the body, which divides the work by lane and block itself
             PLAIN      the body runs on one lane of one block; every other lane returns at once
             AS_INT     one lane, and the body's answer is stored as the call's stamp | a 32-bit int
             AS_FLOAT   the same, with a float's 32 bits
the body     in gpu/kernels/kernels.cuh
the width    blocks, then lanes per block. The blocks may be an expression of the door's own
             arguments — one block per row for a matvec, ceil(n / 256) for a pointwise body
the names    the row's parameter names, in order
```

⛔ **No door starts more than `NN__SILICON__LANES_MAX` lanes a block — 256, the narrowest family's limit**
(OpenCL on the MI50). The compiler refuses a door that tries, and every kernel declares it with
`__launch_bounds__`, because a kernel that declares nothing is compiled for 1,024 lanes and gets a quarter
of the registers.

**The host compiles the same bodies.** The host family answers `nn`'s arithmetic from libm and is one lane
of one block, so every body compiles into the program as a plain function and runs as the serial loop it
is. That is what the on-device check measures a card against (`scripts/src_abi_verbs_on_device.sh`). The
host family carries no `nn` door table, so an `nn` verb on a worker with no card refuses with
`NN__PRIMITIVES__FAULT_NO_DEVICE`.

## What `nn` asks of a silicon family

A family answers these, beside its `sys` doors, in `silicon_families/<family>/nn.cuh`:

```
nn__silicon__launch           start a kernel: blocks, lanes, a pointer to each argument and its width
nn__silicon__expf             the precise functions by default — the ones numpy and PyTorch use
nn__silicon__sqrtf
nn__silicon__logf
nn__silicon__cosf / sinf      RoPE's angles — always the precise ones: an angle can be 100,000 radians
nn__silicon__lane / lanes     which lane of its block a body is on, and how many lanes
nn__silicon__block / blocks   which block, of how many
nn__silicon__lanes_sum        every lane's part added (or the largest); every lane of the block must
nn__silicon__lanes_max        call it at the same point, because it waits for all of them
nn__silicon__lanes_sums       several parts added at once, each as lanes_sum adds it — a GEMM tile's sums
nn__silicon__half_to_float    a half as the float it is — exact on every family; only the cost differs
```

Only what differs between silicon crosses: a clamp or a multiply is written once in C. The float-to-half
direction stays `nn`'s own, because it rounds and it is where an overflow is noticed.

`NN__ARITHMETIC__FAST` in `contracts/defaults.cuh` turns `expf` and `logf` to the card's fast functions.
It is **0, exact, by default**, so an oracle compares against a reference somebody else computed; override
it with `SILVANN_NN_ARITHMETIC_FAST=1` in the environment of `scripts/build_silicon_families.sh`. A card's
`expf` is not libm's either way, so host-against-card comparisons of softmax or rmsnorm need a tolerance.

## Which bodies are wide

```
WIDE      vector_zero  vector_exp  vector_softplus  vector_scale  vector_add  vector_pointwise_mul
          sigmoid  swiglu_combine        item i on lane i of all the blocks, stepping by all of them
          matrix_matvec_transposed       a lane owns whole output columns; no combine step
          expert_multiply_fp16           one block per row, lanes along the row, a lane sum per row
          rmsnorm  vector_l2norm  softmax  vector_dot_product  argmax
                                         one block over the vector, the combine step between passes;
                                         argmax keeps the first index of the largest
          rope                           item i is one pair of one head
          turboquant_gemv                one block per row, lanes along it eight bytes at a time, the
                                         width's own loop, x held as halves; codes decoded in registers
          turboquant_decode              item i is one weight
          attention_scores  attention_mix
                                         one block per query head; two launches, so the weights one
                                         writes are there for the other to read
serial    vector_scale_at                a program may scale a buffer by one of its own elements, and
                                         spread over lanes one would overwrite the factor first
          deltanet_rank_1_update  deltanet_readout  deltanet_conv_step
                                         they carry state forward; thread-per-index would be wrong
          vector_at                      a value form, one lane
          matrix_transpose  hadamard_rotate
```

A serial door's width is `1u, 1u` on purpose: widening is a change to the body, made on its own and
checked against the host. ⭐ **Move it, then make it parallel — never both in one step.**

`MEASURED` on gfx906 (MI50), at a 7B's MLP shapes, weights' bytes over time:

```
nn__expert__multiply_fp16       18944 x 3584   237 µs   573 GB/s
                                3584 x 18944   217 µs   626 GB/s
nn__matrix__matvec_transposed   18944 x 3584   6.9 ms    20 GB/s   14 blocks, one per 256 columns
                                3584 x 18944   1.2 ms   112 GB/s
nn__rmsnorm__apply              3584            40 µs             serial it was 1.8 ms
nn__softmax__apply              152064         740 µs             serial it was 96 ms
nn__argmax__find                152064         295 µs             serial it was 20 ms
```

⭐ **And the whole of Qwen2.5-7B, fp16 on one MI50**, a position at a time through all 28 layers and each
position one program: **41 ms a position** after the first, every token Hugging Face's greedy token in
fp32 on the CPU, its top five logits within 0.1. ▶ `backstage/test/src_qwen25_7b_decode.py`.

The matvec got there with `x` held in registers, 960 blocks striding the rows (`NN__KERNELS__BLOCKS_MAX`,
swept to a plateau at 960–1920), eight halves a load, and `nn__silicon__half_to_float` — without that last
step the software conversion held it at 188 GB/s. OpenCL runs `multiply_fp16` at 615–628 GB/s, with one
context per card: a context over several cards puts a coarse-grained allocation in host RAM, and every
weight is then read at PCIe speed. ▶ `silicon_families/khronos_opencl2/sys.cuh`.
Re-derive from the commits: `git log --oneline -8 -- ':(top)backstage/src/packages/nn'`, then read the
messages.

## What is built and what is not

Built: the activation pool, the cartridge's addressing (the four tiers, the resolver behind
`nn__kv__reference`, the layer table read from the config key `nn__model__layer_types`), the expert index and the load
protocol, the byte doors, and the verbs above. Composed programs are checked against Hugging Face and
numpy by the oracles in `backstage/test/` — `src_weights_oracle.py`, `src_mlp_oracle.py`,
`src_deltanet_oracle.py`, `src_deltanet_layer_oracle.py`, `src_attention_program.py`,
`src_gaussian_oracle.py`, and the whole of Qwen2.5-7B by `src_qwen25_7b_decode.py`.

Not built:

- **the router**: top-k and the expert dispatch
- **the cartridge's KV tiers in use**: the 7B's cache is two plain fp16 planes a layer; admitting a token
  into the hot tier, sinking it through the quantised tiers and evicting the cold tail are not built
- **a many-query attention** for a prompt: a prompt is fed a position at a time
- the expert **eviction policy** and the L2 tier behind the L1 arena
- the `tq8` tier ruled to replace `tq6`
- the conversion from a model's `config.json` to a silvann config

⚠ The evaluator's runners are real threads, so the pool's list lock and the expert index's single writer
can meet real concurrency; whether any test drives two runners into one free list at once is unchecked.
The board, and why each open row is open: `src/docs/design/pending/nn_port.md`.

## Layout

```
language_contract.cuh    this package's three lists, gathered: kinds, verbs, objects
manifest__header.cuh     what the package DECLARES
manifest.cuh             what the package DEFINES
manifest__gpu.cuh        what a silicon family's binary takes of it: the requests, the doors, the bodies
                         — and nothing of the language (scripts/src_compute_stands_alone.sh holds it to that)

contracts/               definitions only — nothing in here runs
  macros/                the three lists — language_contract__{kinds,verbs,objects}.cuh
  objects/               one file per subject: constants, layouts and four-letter fault words
                         buffer · cartridge · expert · deltanet · hadamard · turboquant · swiglu ·
                         primitives · result · package
  abi/
    cpu.cuh                the C interface: nn_abi_read, nn_abi_write — bytes in and out of a buffer
    gpu.cuh                what nn asks of every family, and the door list
  defaults.cuh           the dials, and what they are when nobody turns them

cpu/                     everything the host evaluator runs
  <object>__header.cuh   what you can use: the description, then the contract
  <object>__impl.cuh     how it works
  doors.cuh              finding the calling worker's door table
  opcodes/               the verbs that reach a card, a header and a body per subject
  abi_surface__impl.cuh  the C interface as data, for a host language to bind

gpu/
  doors.cuh              one kernel and one door per row of the door list, and each door's __RUNS line
  kernels/kernels.cuh    the bodies: loops over raw pointers, nothing of sys in them
```

## Using it from outside

Python binds the C interface as `e.nn`, beside `e.sys`. It moves bytes between the host and a buffer on
the worker's card; an address inside the heap is refused, because the heap is `sys`'s.

```python
e.nn.write(address, arr.tobytes())
np.frombuffer(e.nn.read(address, n * 2), dtype="<f2")
```

A buffer's address comes from its heap node: `e.sys.peek(e.sys.full_address(ref))[2][0]`.

## Conventions

`nn` follows `sys`'s — names as package, type and rest with a visibility marker after the type
(`zzpackage`, `zzabi`, `zzprivate`), three kinds of comment, headers as description then contract, and
`AI TEMPORARY COMMENT` blocks at the top of a file. ▶ `../sys/README.md`. One difference: this package's
prose is gated. `scripts/src_rot_scan.py` fails on any comment here that names a file or a symbol the tree
does not have, or tells the story of how the code got this way.

## Checks

```
bash scripts/src_selftest.sh               the language and nn's structures, compiled for the host
bash scripts/src_abi_verbs_on_device.sh    every nn verb on a card against the same body on the host
bash scripts/src_compute_stands_alone.sh   the bodies compile and link with no sys in them
python3 scripts/src_rot_scan.py            nn's prose describes the tree it is in
python3 scripts/src_claim_gate.py          every claim this tree writes down holds
```

## Where to read next

- `src/docs/design/pending/nn_port.md` — what is built, what is open, and why
- `src/docs/design/activation_buffer_pool.md` — the buffer pool
- `src/docs/design/nn_span_layout.md` — the span, and the config keys that size it
- `src/docs/design/expert_lru_node_array.md` and `weight_load.md` — the expert index and how weights get in
- `src/docs/design/pack_container.md` and `pending/quant_families.md` — the packed bundle and the codec
- `src/silicon_families/add_a_family.md` — the seam from the family's side
