# SiLVANN

> **Developer preview, v0.1.1.** It runs the models below well on the hardware it was tested on, and it is shared
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
| Qwen 3.8 27B (dense) | 15 GB — 4 bits | [eval-apply/Qwen3.8-27B_silvann_tq_D4](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) |
| GLM 5.3 Flash (288 experts, 8 active) | 151 GB — 4 bits | [eval-apply/GLM-5.3-Flash_silvann_tq_D4E4](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4) |

`D8E4` names the bits: dense matrices at 8, experts at 4. Each model keeps its own licence (Qwen: Apache-2.0,
GLM: MIT), which is in its folder once downloaded.

## Which model for which machine

| Your machine | Suggested model |
|---|---|
| A 2-4 GB card, 8-16 GB of RAM and an NVMe disk | Qwen 3.6 35B-A3B, its experts on the CPU and read from the disk as they are needed (`experts_on: "cpu"`, `experts_from: "disk"`) |
| No graphics card | the same, the whole model on the CPU — *not yet in this release* |
| An 8-16 GB card and 16-32 GB of RAM | Qwen 3.6 122B-A10B — *not yet in this release* |
| An 8 GB card, 32-64 GB of RAM and an NVMe disk | GLM 5.3 Flash — its dense part (4.4 GB) on the card, the experts on the CPU, read from the disk as they are needed; about 200,000 positions of context — *not yet measured on an 8 GB card* |
| A 16 GB card and 32 GB of RAM | Qwen 3.6 35B-A3B, the dense part on the card and its experts (16 GB) on the CPU in RAM (`experts_on: "cpu"`): about 16 tokens a second |
| A 16 GB card | Qwen 3.8 27B (14 GB at a 32,000-token context), with a smaller context window than larger cards allow |
| A 24-32 GB card | Qwen 3.6 35B-A3B entirely on the card (17 GB at a 32,000-token context): about 19-31 tokens a second |
| A 16-32 GB card and 32-128 GB of RAM | GLM 5.3 Flash — the fixed matrices on the card, the experts on the CPU and the disk; the more RAM, the fewer reads: about 1 token a second with 32 GB, 3 with 128 GB |
| A 16 GB card and 192 GB of RAM or more | GLM 5.3 Flash with all of its experts (142 GB) in RAM, nothing read from the disk: about 5 tokens a second |

The conversation's cache keeps its first 256 positions and the latest 4,096 at full precision and the rest at 8 bits.
A model boots with a 32,000-token context; past the first 4,352, each 100 MB of the card's memory holds about 10,000
more tokens of the 35B's conversation, 3,200 of the 27B's, and 8,800 of GLM 5.3 Flash's — set `max_context` in the
model's `configs/default.json` to what your card has room for.

## Performance

Measured on a Dell R730: two Xeon E5-2680 v4, 256 GB of DDR4-2133, two AMD Instinct MI50, the experts read from an
NVMe disk where they are not in memory. On the card: a prompt of about 2,000 positions, then 32-64 tokens. With the
experts on the CPU: about 1,000 positions, then 128 tokens (the 35B) or 32 (GLM) — "all in RAM" with every expert
already in memory, the others from a cold start under the memory limit shown. The answer's rate is measured after
the prompt.

| Model | Where it runs | Prompt | Answer |
|---|---|---|---|
| Qwen 3.6 35B-A3B | one MI50 | 81 positions/s | 19 tokens/s (31 on a short prompt) |
| Qwen 3.8 27B | one MI50 | 32 positions/s | 5.8 tokens/s (7.6 on a short prompt) |
| Qwen 3.6 35B-A3B | one MI50 for the dense part, the experts on the CPU, all in RAM | 18 positions/s* | 16 tokens/s |
| Qwen 3.6 35B-A3B | the same, the experts read from the disk, 8 GB of RAM | 4.9 positions/s*† | 8.1 tokens/s — 92% of expert reads from memory |
| Qwen 3.6 35B-A3B | the same, 6 GB of RAM | 3.4 positions/s*† | 5.5 tokens/s — 81% from memory |
| GLM 5.3 Flash | one MI50 for the dense part, the experts on the CPU, all in RAM | 6.7 positions/s | 4.9 tokens/s |
| GLM 5.3 Flash | the same, the experts read from the disk, 128 GB of RAM | 5.8 positions/s† | 2.9 tokens/s — 93% from memory |
| GLM 5.3 Flash | the same, 64 GB of RAM | 5.6 positions/s† | 1.4 tokens/s — 71% from memory |
| GLM 5.3 Flash | the same, 32 GB of RAM | 5.6 positions/s† | 1.0 tokens/s — 58% from memory |

\* With its experts on the CPU, the 35B still reads a prompt a position at a time, so its prompt is no faster than
its answer. GLM reads a prompt 1,024 positions at a time through each layer, so each expert is read about once a
chunk.
† Measured from a cold start, so reading the prompt is also when the experts are first read from the disk into
memory; the answer that follows runs on what the prompt left in memory.

⚠ **Reading a prompt is not fully optimized yet.** It runs a chunk of positions through each layer at once, through
a tiled matrix kernel (tiles staged in shared memory); the 35B's mixture-of-experts sites, where each expert gets
few rows of a chunk, and GLM's card sites, still read a position at a time inside a layer, remain well below what
the card allows.

The 27B also runs split over machines (pipeline parallel): as three stages it answers the same tokens at 6.8
tokens/s.

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6 |
| **Untested** | Windows; NVIDIA cards (the CUDA family builds, but has not been run); other AMD generations |
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
experts on the disk, and a model split over machines: [`docs/users.md`](docs/users.md). How it is built and how a
model or a silicon is added: [`docs/developers.md`](docs/developers.md). The ideas behind the design, the Lisp lineage and the model as a collapse of meaning:
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
docs/               users.md — running it; developers.md — how it is built, adding a model or a silicon;
                    concepts.md — the ideas behind it; compression.md — how the weights are packed;
                    directions.md — where it could go
```

## Contact

Questions, ideas and results from your own machine go in this repository's **Discussions** tab; bugs go in
**Issues**. For anything you would rather not post in public, write to **admin@evalapply.co.uk**.

## Licence

The engine is AGPL-3.0 (`LICENSE`). Third-party notices are in `ATTRIBUTION.md`. The model weights keep their
authors' licences.
