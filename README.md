# SiLVANN

> **Developer preview, v0.6.0.** It runs the models below well on the hardware it was tested on, and it is shared
> for people who can find their way around a build and a stack trace. NVIDIA cards are tested on one, a GeForce
> RTX 5060 Ti; Apple silicon is not tested yet.

**Silicon Linked Virtualized Architecture for Neural Networks** — a from-scratch inference engine for large language
models, and the two things it is made of:

- **the Eval-Apply GPU Interpreter** — a small Lisp whose evaluator runs on the CPU and hands arithmetic to whichever
  silicon holds the data: a graphics card, or the CPU's own cores;
- **the Neural Network library** (`nn`) — the arithmetic, the memory structures and the words a model is written in.

A model is a program in that language (`engine/src/packages/ai_qwen_3`, `engine/src/packages/ai_glm_5_3`), and the
weights are packed with TurboQuant: rotated rows at 4 or 8 bits, so a 35-billion-parameter model fits in 19 GB.

## Models, and what they run on

One model a kind of machine — the one that suits it best, measured:

| Your machine | Model | Answer |
|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4) (`D4E4`) on the CPU alone, its experts read from the disk | 11.7 tokens/s |
| An 8 GB card | [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D4) (`D4`) on the card, at a 4,096-token context | 42 tokens/s ◦ |
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) on the card | 30.5 tokens/s |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D8) (`D8`) on the card, at an 8,192-token context | 30 tokens/s ◦ |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) on the card | 87 tokens/s |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) (`D4E4`) — the dense part on the card, the experts on the CPU | 4.7 tokens/s ◦ |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 19.9 tokens/s |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4), its experts in RAM and 6 GB of them on the card | 7.5 tokens/s |
| A 32 GB card, 256 GB of RAM | [Qwen 3.5 397B-A17B](https://huggingface.co/eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4), its experts in RAM and 24 GB of them on the card | 9.0 tokens/s |

On AMD Instinct MI50 cards in a Dell R730, each process held to the memory its row names — except the rows with experts
on the card, which ran on the machine's 256 GB and name what the model needs. ◦ Ran on a 32 GB card: the model fits the
smaller one, which has not been measured. The rows with a model entirely on the card are v0.4.0's; the others were
measured with v0.3, and where a card does part of the work they are a floor. **Every model and pack, the other
machines (down to a 4 GB card reading its experts from the disk), and how long a conversation fits:
[`docs/models.md`](docs/models.md)**; every measurement: [`docs/results.md`](docs/results.md).

On an NVIDIA card — a GeForce RTX 5060 Ti 16 GB in a desktop with a four-core Xeon W-2123 and two DDR4-2400 sticks
(62 GB), about what a local-model enthusiast's machine has — measured with v0.5.0 the same way, a prompt of about 2,000
positions and then the answer:

| Model | How it runs | Prompt | Answer |
|---|---|---|---|
| [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D4) (`D4`) | on the card | 958 positions/s | 39 tokens/s (46 on a short prompt) |
| [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D8) (`D8`) | on the card | 873 positions/s | 25 tokens/s (27 on a short prompt) |
| [Qwen 3.8 27B](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) | on the card | 549 positions/s | 23.6 tokens/s (24.4 on a short prompt) |
| [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) (`D8E4`) | the dense part on the card, the experts on the CPU, and 8 GB of the card the tier of experts used most | 117 positions/s | 52 tokens/s |
| the same | without the tier | 55 positions/s | 28 tokens/s |
| the same | on the CPU alone | 21 positions/s | 8.6 tokens/s |

**New in v0.6.0:** a family for old integrated GPUs, `khronos_opencl1_igpu`: the library's kernels as OpenCL C 1.2, for
a GPU whose OpenCL has neither shared virtual memory nor the generic address space. The build works out every pointer's
address space itself, so the library's source names none. It has run on one device so far, the RTX 5060 Ti through
NVIDIA's OpenCL: the library's checks pass, and the 27B answers as Hugging Face does, at 1.6 tokens a second. On a card
with memory of its own, the RAM the card reads crosses the bus every token; an integrated GPU, whose memory is the
machine's, is what it is for. It takes a device only once the checks have passed on it, and so far that is the 5060 Ti
alone, which the CUDA family takes first: no integrated GPU runs it yet
([`engine/src/silicon_families/khronos_opencl1_igpu/README.md`](engine/src/silicon_families/khronos_opencl1_igpu/README.md)).
And a model holds less of the machine's RAM: it loads its output head and its embedding table before its layers and
keeps no copy of its own once the engine has them. Ministral 3 14B on the CPU alone holds 7.4 GB, from 10.0, and needs
7.4 GB while it loads, from 10.6; with a card, the 27B holds 0.3 GB of the machine's RAM, from 5.9 (beside the pack's
pages the system caches, which it gives back when it needs them).

**New in v0.5.1:** the library's own attention, which runs wherever a family has no kernel of its own for it — a
prompt's on AMD cards and through OpenCL — reads a prompt's scores four rows at a time, each key once for the four: on
one MI50 the 27B's scores over a 256-row chunk take 6.2 ms, from 19, and the 27B reads a prompt at 101 positions a
second, from 96 (80 from 76 over 6,033 positions). Past the 4,352 positions the cache keeps at full precision, its
scores over the 8- and 4-bit tiers read eight bytes of a key at once: through OpenCL the 27B then answers at 6.6 tokens
a second, from 6.1, and reads that prompt at 28.7 positions a second, from 26.9. On the CPU the matrix products of a
prompt keep the four sums a token's products keep, one pass each: the 35B on the four-core Xeon alone reads a prompt at
27.6 positions a second, from 20.8 (with AVX2 alone 18.1, from 11.7), the 27B at 3.6, from 2.1 — and with AVX2 a
prompt's products now add in the order a token's do. GLM boots on any card's family, an NVIDIA card's too. One
configuration gives one answer to the bit, run after run; a prompt read as rows is checked against the same prompt a
position at a time with a bound ([`docs/developers.md`](docs/developers.md)). Fixes: on AVX-512 an empty group of
experts read past its rows; the CPU's pool started no threads where `/sys` gives no topology; a launch over no blocks is
no longer made on NVIDIA and AMD cards, and a launch a card refuses is named. GLM with its experts read from an NVMe
disk, on an RTX 5060 Ti with 32 and 60 GB of RAM: [`docs/results.md`](docs/results.md).

**New in v0.5.0:** NVIDIA cards. The CUDA family ran on a card for the first time, an RTX 5060 Ti, and runs the matrix
products of a prompt, its attention and the experts on the card's tensor cores, and a token's matrix products through
them too: the 27B reads a prompt at 549 positions a second, from 19 with the library's generic kernels, and answers at
24 tokens a second, from 7.8 — past the 4,352 positions the cache keeps at full precision a chunk of the prompt attends
in a launch, not a row at a time (the 27B over 6,033 positions, 370 to 472 a second); every pack but GLM's is checked
against Hugging Face on it. Ministral reads a prompt as rows on every card, through the library's own attention over
many rows (958 positions a second on the 5060 Ti, from 34 a position at a time). On the CPU, a processor with AVX-512
runs the matrix products with it, and the CPU family has its own attention and DeltaNet convolution: beside the 5060 Ti
the 35B with its experts on the CPU answers at 28 tokens a second, from 15-17, and on the four-core Xeon alone at 9.0
on a short prompt, from 1.7, reading a prompt at 21 positions a second. And a fix: the CPU's pool of threads left out
the CPU the caller ran on rather than its core, so in about half of all runs the caller shared a core with a worker
(7.6 tokens a second against 9.0, the 35B on the CPU alone).

And on AMD cards, past the 4,352 positions the cache keeps at full precision, the attention over its 8-bit and 4-bit
tiers runs the family's own kernels: on one MI50 the 35B answers after a 6,033-position prompt at 59-63 tokens a
second, from 26, and reads that prompt at 144 positions a second, from 94. GLM and Ministral now make their answers on
the card as Qwen does, eight tokens a call (`lisp/generate.lisp`, which every family shares). The line between the
experts the card copies and those the CPUs compute over a prompt is set from the medians of five readings rather than
one, which once sent every expert a prompt picked to the card. A model folder downloaded before a file of its program
existed is offered `lisp/` again.

**New in v0.4.0:** a model's program travels with the model. A model folder carries it in `lisp/` — the Lisp the
evaluator runs the model with, which the boot reads from there: read it, change it, and the next boot runs what you
wrote. The packs on Hugging Face carry it, and for a model downloaded before, `silvann_core.py download-model <name>`
fetches `lisp/` alone. Sampling runs on the card, so a sampled answer comes about as fast as a greedy one (the 35B on
one card, answering a chat: from 75 to 101 tokens/s), and Qwen's answers are made on the card eight tokens at a time.
And the card's kernels are faster: on one MI50, at about 2,000 positions, the 35B answers at 87 tokens/s (from 38), the
27B at 30.5 (from 20.3), and Ministral 3 14B at 42 (`D4`) and 30 (`D8`) — measured now at a long prompt as the other
rows are; on a short prompt, 56 and 37 tokens/s, where v0.3 measured 52 and 26.

**Known issue:** a kernel update can leave the AMD driver unbuilt for the new kernel, and the machine then boots on the
kernel's own `amdgpu`, where ROCm runs but card-to-host copies corrupt memory (`free(): invalid pointer`, at random).
After a kernel update, `modinfo -n amdgpu` should name a file under `updates/dkms/`; if it does not, boot the previous
kernel, or install an `amdgpu-dkms` that builds for the new one.

**New in v0.3.3:** a fix — on the CPU, a small buffer's header was written where its release did not read it, and
releasing it could unmap part of the process's memory; it showed as `free(): invalid pointer` at the end of a run, or
not at all. And the 27B and the 35B on one card, measured again with the test machine's cooling working (it had been
throttling the card): 20.3 and 38 tokens/s, from 17.4 and 37.

**New in v0.3.2:** a fix — on a machine busy when the engine starts, the CPU's pool of threads could be sized to none,
and the experts on the CPU were then skipped without an error, so the answers were wrong. Upgrade if your experts run
on the CPU. And GLM 5.3 Flash on a 32 GB card, measured again with 24 GB of tier: 10.7 tokens/s.

**New in v0.3.1:** Mistral AI's Ministral 3 14B (Apache 2.0) — the first model of a third family, in two packs; two more
packs, the 397B and GLM 5.3 Flash with its dense part at 8 bits; how close each pack comes to the original weights, as
perplexity and bits per byte ([`docs/results.md`](docs/results.md#how-good-each-pack-is-perplexity)); GLM's experts on
the CPU added in the same order every run, so a prompt gives the same answer each time (with a card's tier, which experts
the card holds depends on when its copies land, and that can still change the last bits); Qwen's experts on the CPU
multiplied in integers (`experts_int8`), as GLM's already were; and for GLM an exclusive tier (`card_exclusive`), where
what the card holds is not also kept in RAM ([`docs/models.md`](docs/models.md#the-cards-tier-of-experts)).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs) |
| **Untested** | Windows itself — on a Windows PC, boot Linux from a USB SSD; WSL2 may work ([`docs/users.md`](docs/users.md#on-windows-linux-on-a-usb-ssd)); other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not yet** | Apple silicon |

Images are not accepted yet: the server answers an image with 501.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash   # into ./silvann, and a `silvann` command
silvann download Qwen3.6-35B-A3B_silvann_tq_D8E4
silvann serve Qwen3.6-35B-A3B_silvann_tq_D8E4                       # on port 8765
silvann ui                                                          # a chat in the browser, at http://localhost:8090
```

Or by hand, from this folder:

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
SILVANN_PYTHON=venv/bin/python ./build.sh                        # needs g++; ROCm's hipcc for AMD cards, CUDA's nvcc for NVIDIA
venv/bin/python silvann_core.py download-model Qwen3.6-35B-A3B_silvann_tq_D8E4
venv/bin/python start_silvann_server.py --model models/Qwen3.6-35B-A3B_silvann_tq_D8E4   # on port 8765
venv/bin/python silvann_ui.py --port 8090                          # a chat in the browser, on the server above
```

A model that is not downloaded is refused with the command that fetches it. The settings a model boots with, the
experts on the disk, and a model split over machines: [`docs/users.md`](docs/users.md). Every measurement:
[`docs/results.md`](docs/results.md). How it is built and how a model or a silicon is added:
[`docs/developers.md`](docs/developers.md). The ideas behind the design, the Lisp lineage and the model as a collapse of meaning:
[`docs/concepts.md`](docs/concepts.md); how the weights are packed: [`docs/compression.md`](docs/compression.md).
Where the design could go next — a
model whose output is a program the engine runs, sizing the experts after they answer, choosing an adapter at the
end of a sentence, layers replaced by simpler programs, multi-token prediction as a program: [`docs/directions.md`](docs/directions.md).

## What is here

```
install.sh          fetches, builds and installs the `silvann` command — the one-line install
build.sh            builds the engine into lib/
engine/src/         the interpreter (engine, sys), the library (nn), the models (ai_qwen_3, ai_glm_5_3), and
                    silicon_families/ — one folder a kind of silicon, what it answers for the library
runtime/            the Python side: a model folder booted onto the machine, conversations, the chat framing
lisp/               each model family's program, in Lisp — a model folder carries its family's in its own lisp/
silvann_core.py     download-model · list-models
start_silvann_server.py, silvann_ui.py
models/             where downloaded models go
docs/               users.md — running it; models.md — every model and what it needs; results.md — every measurement;
                    developers.md — how it is built, adding a model or a silicon;
                    concepts.md — the ideas behind it; compression.md — how the weights are packed;
                    directions.md — where it could go
```

## Contact

Questions, ideas and results from your own machine go in this repository's **Discussions** tab; bugs go in
**Issues**. For anything you would rather not post in public, write to **admin@evalapply.co.uk**.

## Contributing

Reports from hardware we have not tried, bug reports and fixes are all welcome — see `CONTRIBUTING.md`. Pull
requests are made under the Contributor License Agreement in `CLA.md`, which a bot asks you to accept once.

## Licence

The engine is AGPL-3.0 (`LICENSE`). Third-party notices are in `ATTRIBUTION.md`. The model weights keep their
authors' licences.
