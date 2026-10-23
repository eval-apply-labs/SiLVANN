# SiLVANN

> **Developer preview, v0.18.0.** It runs the models below well on the hardware it was tested on, and it is shared
> for people who can find their way around a build and a stack trace. NVIDIA cards are tested on two, a GeForce
> RTX 5060 Ti and an RTX 5070 Ti; Windows natively on one PC; macOS is not supported.

**Silicon Linked Virtualized Architecture for Neural Networks** — a from-scratch inference engine for large language
models, and the two things it is made of:

- **the Eval-Apply GPU Interpreter** — a small Lisp whose evaluator runs on the CPU and hands arithmetic to whichever
  silicon holds the data: a graphics card, or the CPU's own cores;
- **the Neural Network library** (`nn`) — the arithmetic, the memory structures and the words a model is written in.

A model is a program in that language (`engine/src/packages/ai_qwen_3`, `engine/src/packages/ai_z__glm_5_3`), and the
weights are packed with TurboQuant: rotated rows at 4 or 8 bits, so a 35-billion-parameter model fits in 19 GB.

## Models, and what they run on

One model a kind of machine — the one that suits it best — and the machine it was measured on:

| Your machine | Model | Answer | Measured on |
|---|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`†) on the CPU alone, its experts read from the disk | 11.7 tokens/s | the R730's two Xeons, v0.3 |
| An 8 GB card | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) (`D4`) on the card, at a 4,096-token context | 47 tokens/s | an RTX 5060 Ti with 7.8 GB of it free, in the chat, v0.11.1 (an MI50 after a prompt: 42) |
| An 8 GB card, 32 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 3.5 GB of the card the tier of experts used most | 56 tokens/s | an RTX 5060 Ti with 7.8 GB of it free, in the chat, v0.11.1 |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`†) — the dense part on the card, the experts on the CPU | 8.6 tokens/s | an RTX 5060 Ti with 7.3 GB of it free, in the chat, v0.11.1 |
| A 12 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 7 GB of the card the tier | 65 tokens/s | an RTX 5060 Ti with 11.6 GB of it free and 62 GB of RAM, in the chat, v0.11.1 |
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) on the card | 26 tokens/s | an RTX 5060 Ti, in the chat, v0.11.1 (an MI50 after a prompt: 31.3) |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) (`D8`) on the card, at an 8,192-token context | 28 tokens/s | an RTX 5060 Ti, in the chat, v0.11.1 (an MI50 after a prompt: 30.6) |
| A 16 GB card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 11 GB of the card the tier of experts used most | 68 tokens/s | an RTX 5060 Ti, in the chat, v0.11.1 (an RTX 5070 Ti with 32 GB after a prompt: 68.3) |
| A 16 GB card, 28 GB of RAM, an NVMe disk | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 2.3 tokens/s | an RTX 5060 Ti, in the chat, v0.11.1 |
| A 16 GB card, 62 GB of RAM | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`†) — the dense part on the card, the experts on the CPU in RAM, and 9.5 GB of the card the tier of experts used most | 20 tokens/s | an RTX 5060 Ti, in the chat, v0.11.1 |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`†), its experts in RAM, a 300,000-token context and 6 GB of the card the tier | 8.0 tokens/s | an MI50, 15.0 GB of it, v0.8.0 |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) on the card | 87 tokens/s | an MI50, v0.8.0 |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 24.5 tokens/s | an MI50, v0.8.0 |

The MI50s are in a Dell R730 with two Xeon E5-2680 v4; the RTX 5060 Ti in a desktop with a four-core Xeon W-2123 and
two DDR4-2400 sticks; the RTX 5070 Ti beside a Ryzen 7 5800X. Each process was held to the memory its row names, and a
smaller card's room was a larger card with the rest of it held. † Measured with the model's `D4E4` pack, retired in
v0.18.0: the `D8E4` linked holds the same 4-bit experts and its dense part at 8 bits — the 122B's layer matrices 4.66 GB
on the card against 2.37, the 35B's 1.45 against 0.75 — and its rate on that machine is to be measured again. "In the chat" is the server with the model's own config, as the chat's meter reads a short
conversation, measured with v0.11.1; every other row takes the most likely token after a prompt
([`docs/results.md`](docs/results.md)). A row of v0.3 was measured with its card kernels, and where a card does part of the work it is
a floor. A model whose config names no setup — the default's — chooses its own when the server starts, by trying it on
the card: where its experts go, how much of the card holds the ones used most, the context the card has room for
([`docs/config.md`](docs/config.md)). **Every model and pack, the other machines (down to a 4 GB card reading its
experts from the disk), and how long a conversation fits: [`docs/models.md`](docs/models.md)**; every measurement, the
machines, and each row's version: [`docs/results.md`](docs/results.md).

On NVIDIA cards — a GeForce RTX 5060 Ti 16 GB in the desktop above, about what a local-model enthusiast's machine has,
and an RTX 5070 Ti 16 GB — in the chat, and after a prompt of about 2,000 positions:

| Model | How it runs | RTX 5060 Ti: in the chat | RTX 5060 Ti: prompt · answer | RTX 5070 Ti: prompt · answer |
|---|---|---|---|---|
| [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) (`D4`) | on the card | 47 tokens/s | 953 positions/s · 39 tokens/s | 1,682 positions/s · 59 tokens/s |
| [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) (`D8`) | on the card | 28 tokens/s | 869 positions/s · 25 tokens/s | — |
| [Qwen 3.8 27B](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) | on the card | 26 tokens/s | 545 positions/s · 23.5 tokens/s | — |
| [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) | the dense part on the card, the experts on the CPU, and the tier of experts the setup chooses at boot | 67 tokens/s | 115 positions/s · 57 tokens/s | 155 positions/s · 68 tokens/s |
| the same | without the tier | 29 tokens/s | 53 positions/s · 29 tokens/s | — |
| the same | on the CPU alone | 10.5 tokens/s | 27 positions/s · 8.7 tokens/s | — |
| [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`†) | the dense part on the card, the experts on the CPU in RAM (54 GB of the desktop's 62), and the tier the setup chooses at boot | 20 tokens/s | 48 positions/s · 18 tokens/s | — |

"In the chat" is the server with the model's own config — its sampler, the setup it chooses — as the chat's meter reads
a short conversation (from 54 positions; Ministral's from 583, its own system prompt), measured with v0.11.1
(2026-10-18). The prompt and the answer after it take the most likely token; the RTX 5060 Ti's measured again with
v0.11.1, the RTX 5070 Ti's with v0.11.0. v0.11.0's server drew each token with nn's generic top k on an NVIDIA
card, through OpenCL and on the CPU alone: the 35B answered in the chat at 32-35 tokens a second on the RTX 5060 Ti,
and answers at 67 now ([`docs/results.md`](docs/results.md)).

**New in v0.18.0:** a Qwen MoE model reads an expert from the disk in one read, the server answers while a model
generates, and the `D4E4` packs are retired.

**One read an expert.** Qwen 3.6 35B-A3B's and Qwen 3.5 122B-A10B's packs keep each routed expert as a record of its
own, laid out as the engine holds it, so an expert the CPU reads from the disk is one read where it was four. The 122B
with its experts read from an NVMe array, held to 32 GB of RAM, answers at 9.3-9.4 tokens a second where the pack before
gave 8.0-8.3. The new packs need v0.18.0; a pack downloaded before keeps working as it did — download it again for the
new layout ([`docs/users.md`](docs/users.md#5-the-experts-on-the-disk)).

**The server answers while a model generates.** A model's turn no longer holds the server's other requests: `/health`,
`/busy` and the rest answer in a few milliseconds through an answer, where they waited seconds.

**AMD's prompt mixes as its tokens do.** On an AMD card the attention's mix of a prompt read as rows is now the same, to
the bit, as the same positions read a token at a time, and the prompt reads 2% faster on the 27B.

**The `D4E4` packs are retired, and the 397B is not published.** The 35B's, the 122B's and GLM 5.3 Flash's `D4E4` are
no longer offered: their `D8E4` hold the same 4-bit experts, and the dense part at 8 bits is the one that pays
([`docs/compression.md`](docs/compression.md)). Qwen 3.5 397B-A17B still runs, but its pack is no longer published: GLM
5.3 Flash and DeepSeek-V4.1-Flash do more at its size.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6 — a model on one card, and the 27B's layers over two and four (tensor parallelism); DeepSeek-V4.1-Flash on one card with its experts on the CPU, checked against DeepSeek's own code; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs); Windows with an NVIDIA card (an RTX 5070 Ti, a Ryzen 7 5800X), natively: built from source with CUDA 13.4, Ministral 3 14B `D4` on the card ([`docs/users.md`](docs/users.md#on-windows)) |
| **Untested** | on Windows: the prebuilt libraries on a PC (GitHub's runners, which build them, have no card), the CPU alone, a model's experts on the CPU or read from the disk, AMD's and Intel's GPUs through OpenCL; Linux booted from a USB SSD; tensor parallelism on NVIDIA cards; other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell; an older card — Kepler, Maxwell, Pascal, Volta — goes to OpenCL 1.2, the GTX 1080 first); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them (on one, the model's memory is read where it is, not copied to the GPU: checked only through PoCL, a CPU's OpenCL) — which card goes to OpenCL 1.2 and which to 2 is a best guess from the vendors' published versions (OpenCL 1.2: every NVIDIA card CUDA does not take, Intel's iGPUs before Gen9, AMD's TeraScale and GCN 1; OpenCL 2: the rest), and a card on the wrong one is an issue to open, with its `clinfo`; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not supported** | macOS |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0: [`release_logs/v0.md`](release_logs/v0.md#v070)), pictures by DeepSeek-V4.1-Flash (v0.17.0: [`docs/users.md`](docs/users.md#3-serve-it-and-chat)); Ministral 3 reads neither.
Every model calls the tool, a Lisp REPL on the machine, in its own trained format (v0.9.0: [`docs/users.md`](docs/users.md#the-models-tool)).
Pictures are made from a prompt by Qwen-Image 2.1, a server's model of its own (v0.10.0: [`docs/users.md`](docs/users.md#a-picture-model-qwen-image-21)).
A model's tool draws with the picture model, opens subagents and keeps skills of its own; a dense Qwen model's layers split over AMD cards (v0.11.0: [`docs/tools.md`](docs/tools.md), [`docs/config.md`](docs/config.md)).

## Quick start

### On Linux: the one-liner install script

One line in a terminal fetches SiLVANN into `./silvann` and builds it for this machine — the CPU, and the card it finds:
an NVIDIA card with CUDA 12's `nvcc`, an AMD card with ROCm 6's `hipcc`, each for that card alone (where the compiler is
missing it says where to get it, and the CPU runs the models meanwhile). It installs nothing system-wide and asks for no
password, and it leaves a `silvann` command (`silvann help` lists what it does). Run it again — or `silvann update` —
to update: the models, the conversations and their settings stay. `silvann models <folder>` keeps the models, the
conversations and the rest it keeps for you in another folder from then on — SiLVANN on a USB drive, say, and its models
on the machine's own disk ([`docs/users.md`](docs/users.md)).

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash   # into ./silvann, and a `silvann` command
silvann download silvann_tq_D8E4_qwen3.6_35b_a3b
silvann serve silvann_tq_D8E4_qwen3.6_35b_a3b                       # on port 8765
silvann ui                                                          # a chat in the browser, at http://localhost:8090
```

Or by hand, from this folder:

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
SILVANN_PYTHON=venv/bin/python ./build.sh                        # needs g++; ROCm's hipcc for AMD cards, CUDA's nvcc for NVIDIA
venv/bin/python silvann_core.py download-model silvann_tq_D8E4_qwen3.6_35b_a3b
venv/bin/python start_silvann_server.py --model models/silvann_tq_D8E4_qwen3.6_35b_a3b   # on port 8765
venv/bin/python silvann_ui.py --port 8090                          # a chat in the browser, on the server above (docs/ui.md)
```

### On Windows: `install.ps1`

Python 3.10 to 3.14, Git and the card's own driver (`winget install -e --id Python.Python.3.12`, `winget install -e --id
Git.Git`). In PowerShell, in the folder SiLVANN should go under:

```powershell
curl.exe -fsSLO https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1      # into .\silvann, and a `silvann` command
```

Then, in a new PowerShell — the one the installer ran in does not have `silvann` on its PATH yet:

```powershell
silvann models E:\silvann_data                 # if you like: the models and the conversations on another drive
silvann download silvann_tq_D4_ministral_3_14b
silvann serve silvann_tq_D4_ministral_3_14b     # on port 8765
silvann ui                                     # a chat in the browser, at http://localhost:8090
```

The installer takes the libraries GitHub built for the release when they were built from the folder's engine, and builds
them otherwise. `build.ps1`, in the folder, builds them on the PC: it needs Visual Studio's Build Tools with their C++
Clang tools, and CUDA's toolkit (12 or 13) for an NVIDIA card. Run `install.ps1` again — or `silvann update` — to
update: the models, the conversations and their settings stay ([`docs/users.md`](docs/users.md#on-windows)).

### More info

A model that is not downloaded is offered for download when the server starts on a terminal, and otherwise refused
with the command that fetches it. Every setting a model boots with, and its
options: [`docs/config.md`](docs/config.md). The model's tool, and adding your own commands to it in Python:
[`docs/tools.md`](docs/tools.md). The experts on the disk, and a model split over machines:
[`docs/users.md`](docs/users.md). Every measurement:
[`docs/results.md`](docs/results.md). How it is built and how a model or a silicon is added:
[`docs/developers.md`](docs/developers.md). The ideas behind the design, the Lisp lineage and the model as a collapse of meaning:
[`docs/concepts.md`](docs/concepts.md); how the weights are packed: [`docs/compression.md`](docs/compression.md).
Where the design could go next — a
model whose output is a program the engine runs, sizing the experts after they answer, choosing an adapter at the
end of a sentence, layers replaced by simpler programs, multi-token prediction as a program: [`docs/directions.md`](docs/directions.md).
Why a model on your own machine matters — for the work that cannot leave the room, and against the bad actors who could
run one too: [`docs/local_ai.md`](docs/local_ai.md).

## What is here

```
install.sh          fetches, builds and installs the `silvann` command — the one-line install on Linux
install.ps1         the same on Windows, with the libraries the release was built with; build.ps1 builds them on the PC
build.sh, build.py  build the engine into lib/ — on Linux, on Windows
docker/             a Dockerfile, for a container of your own — no image is published
engine/src/         the interpreter (engine, sys), the library (nn), the models (ai_qwen_3, ai_z__glm_5_3, ai_qwen_image,
                    ai_deepseek_4), silicon_families/ — one folder a kind of silicon, what it answers for the library —
                    and operating_systems/ — one folder a system, what it answers for the engine
runtime/            the Python side: a model folder booted onto the machine, conversations, the chat framing
lisp/               each model family's program, in Lisp — a model folder carries its family's in its own lisp/
silvann_core.py     download-model · list-models · hf-token
start_silvann_server.py, silvann_ui.py, silvann_cli.py     the server, the chat in a browser, the chat in a terminal
models/             where downloaded models go
skills/             commands for the model's tool, a folder each — docs/tools.md says how to add yours
release_logs/       v0.md — what each 0.x release brought, newest first
docs/               users.md — running it; ui.md — the chat in the browser; config.md — a model's settings; tools.md — the model's tool and your own
                    commands; models.md — every model and what it needs; results.md — every measurement;
                    developers.md — how it is built, adding a model or a silicon;
                    concepts.md — the ideas behind it; compression.md — how the weights are packed;
                    directions.md — where it could go; local_ai.md — why AI on your own machine
```

## Contact

Questions, ideas and results from your own machine go in this repository's **Discussions** tab; bugs go in
**Issues**. For anything you would rather not post in public, write to **admin@evalapply.co.uk**.

## Contributing

Reports from hardware we have not tried, bug reports and fixes are all welcome — see `CONTRIBUTING.md`. Pull
requests are made under the Contributor License Agreement in `CLA.md`, which a bot asks you to accept once.

## Licence

The engine is AGPL-3.0 (`LICENSE`), with an additional permission to convey it linked with NVIDIA's CUDA runtime
(`ATTRIBUTION.md` §1). Third-party notices are in `ATTRIBUTION.md`, an image built from `docker/`'s (§6) and the prebuilt Windows
libraries' (§7) among them. The model weights
keep their authors' licences.
