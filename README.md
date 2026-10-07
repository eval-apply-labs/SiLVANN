# SiLVANN

> **Developer preview, v0.3.3.** It runs the models below well on the hardware it was tested on, and it is shared
> for people who can find their way around a build and a stack trace. The NVIDIA and Apple paths, and tensor
> cores, are not tested yet.

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
| An 8 GB card | [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D4) (`D4`) on the card, at a 4,096-token context | 52 tokens/s ◦ |
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) on the card | 20.3 tokens/s |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D8) (`D8`) on the card, at an 8,192-token context | 26 tokens/s ◦ |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) on the card | 38 tokens/s |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) (`D4E4`) — the dense part on the card, the experts on the CPU | 4.7 tokens/s ◦ |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 19.9 tokens/s |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4), its experts in RAM and 6 GB of them on the card | 7.5 tokens/s |
| A 32 GB card, 256 GB of RAM | [Qwen 3.5 397B-A17B](https://huggingface.co/eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4), its experts in RAM and 24 GB of them on the card | 9.0 tokens/s |

On AMD Instinct MI50 cards in a Dell R730, each process held to the memory its row names — except the rows with experts
on the card, which ran on the machine's 256 GB and name what the model needs. ◦ Ran on a 32 GB card: the model fits the
smaller one, which has not been measured. **Every model and pack, the other machines (down to a 4 GB card
reading its experts from the disk), and how long a conversation fits: [`docs/models.md`](docs/models.md)**; every
measurement: [`docs/results.md`](docs/results.md).

**New in v0.3.3:** a fix — on the CPU, a small buffer's header was written where its release did not read it, and
releasing it could unmap part of the process's memory; it showed as `free(): invalid pointer` at the end of a run, or
not at all. And the 27B and the 35B on one card, measured again with the test machine's cooling working (it had been
throttling the card): 20.3 and 38 tokens/s, from 17.4 and 37.

**Known issue:** a kernel update can leave the AMD driver unbuilt for the new kernel, and the machine then boots on the
kernel's own `amdgpu`, where ROCm runs but card-to-host copies corrupt memory (`free(): invalid pointer`, at random).
After a kernel update, `modinfo -n amdgpu` should name a file under `updates/dkms/`; if it does not, boot the previous
kernel, or install an `amdgpu-dkms` that builds for the new one.

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
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; the CPU alone with AVX2 (Xeon E5-2680 v4); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without) |
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
