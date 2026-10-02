# SiLVANN

> **Developer preview, v0.2.1.** It runs the models below well on the hardware it was tested on, and it is shared
> for people who can find their way around a build and a stack trace. The NVIDIA and Apple paths, and tensor
> cores, are not tested yet.

**Silicon Linked Virtualized Architecture for Neural Networks** — a from-scratch inference engine for large language
models, and the two things it is made of:

- **the Eval-Apply GPU Interpreter** — a small Lisp whose evaluator runs on the CPU and hands arithmetic to whichever
  silicon holds the data: a graphics card, or the CPU's own cores;
- **the Neural Network library** (`nn`) — the arithmetic, the memory structures and the words a model is written in.

A model is a program in that language (`engine/src/packages/ai_qwen_3`, `engine/src/packages/ai_glm_5_3`), and the
weights are packed with TurboQuant: rotated rows at 4 or 8 bits, so a 35-billion-parameter model fits in 19 GB.

## Models

| Model | Weights | Download |
|---|---|---|
| Qwen 3.6 35B-A3B (mixture of experts, 3B active) | 19 GB — experts 4 bits, the rest 8 | [eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) |
| Qwen 3.6 35B-A3B, for the CPU alone | 19 GB — 4 bits | [eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4) |
| Qwen 3.5 122B-A10B (mixture of experts, 10B active), for 12 GB cards | 67 GB — experts 4 bits, the rest 8 | [eval-apply/Qwen3.5-122B-A10B_silvann_tq_D8E4](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D8E4) |
| Qwen 3.5 122B-A10B, for 8 GB cards | 65 GB — 4 bits | [eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) |
| Qwen 3.8 27B (dense) | 15 GB — 4 bits | [eval-apply/Qwen3.8-27B_silvann_tq_D4](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) |
| GLM 5.3 Flash (288 experts, 8 active) | 151 GB — 4 bits | [eval-apply/GLM-5.3-Flash_silvann_tq_D4E4](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4) |

`D8E4` names the bits: dense matrices at 8, experts at 4. Each model keeps its own licence (Qwen: Apache-2.0,
GLM: MIT), which is in its folder once downloaded.

## What it runs on

From a machine with no graphics card to a model near the frontier on salvaged server parts — one model a step, each the
one that suits that machine best:

| Your machine | Model, and how it runs | Answer |
|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk (2.5 GB/s) | Qwen 3.6 35B-A3B (`D4E4`) on the CPU alone, its experts read from the disk, 8 GB of them kept in memory | 11.7 tokens/s |
| A 16 GB card | Qwen 3.8 27B entirely on the card, at a 32,000-token context (14 GB) | 17.4 tokens/s (18.5 on a short prompt) |
| A 24 GB card | Qwen 3.6 35B-A3B entirely on the card (17 GB at a 32,000-token context) | 37 tokens/s (40 on a short prompt) |
| A 4 GB card, 8 GB of RAM, an NVMe disk | Qwen 3.6 35B-A3B — the dense part on the card, the experts on the CPU, read from the disk as they are needed | 11.9 tokens/s ◦ |
| An 8 GB card, 32 GB of RAM, an NVMe disk | Qwen 3.5 122B-A10B (`D4E4`) — the same split | 4.7 tokens/s ◦ |
| A 16 GB card, 64 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | about 2.5 tokens/s, projected |
| A 16 GB card, 128 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | 5.7 tokens/s |
| A 16 GB card, 150 GB of RAM | GLM 5.3 Flash with all of its experts (142 GB) in RAM, nothing read from the disk | 6.7 tokens/s |

Measured on a Dell R730 (two Xeon E5-2680 v4, AMD Instinct MI50 cards; disk speed 5 GB/s, RAM speed 126 GB/s), each
row with its process held to the memory it names — leave room beside it for the operating system — and from a cold
start where the experts come from the disk. ◦ The dense part ran on a 32 GB MI50: it fits the smaller card, but a card that size has not been
measured. The projected row is worked out from GLM's measured rows at 64 GB and 128 GB.

Every measurement, the other packs and memory sizes, and what limits a model reading its experts from the disk:
[`docs/results.md`](docs/results.md).

The conversation's cache keeps its first 256 positions and the latest 4,096 at full precision and the rest at 8 bits.
A model boots with a 32,000-token context; past the first 4,352, each 100 MB of the card's memory holds about 10,000
more tokens of the 35B's conversation, 3,200 of the 27B's, and 8,800 of GLM 5.3 Flash's — set `max_context` in the
model's `configs/default.json` to what your card has room for.

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; the CPU alone with AVX2 (Xeon E5-2680 v4); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second, its prompt still slow (about 4 positions a second) |
| **Untested** | Windows; NVIDIA cards (the CUDA family builds, but has not been run); other AMD generations; OpenCL on other devices (an Intel iGPU among them) |
| **Not yet** | Apple silicon; AVX-512 CPU paths |

Images are not accepted yet: the server answers an image with 501.

## Quick start

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
SILVANN_PYTHON=venv/bin/python ./build.sh                        # needs g++, and ROCm's hipcc for AMD cards
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
build.sh            builds the engine into lib/
engine/src/         the interpreter (engine, sys), the library (nn), the models (ai_qwen_3, ai_glm_5_3), and
                    silicon_families/ — one folder a kind of silicon, what it answers for the library
runtime/            the Python side: a model folder booted onto the machine, conversations, the chat framing
silvann_core.py     download-model · list-models
start_silvann_server.py, silvann_ui.py
models/             where downloaded models go
docs/               users.md — running it; results.md — every measurement;
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
