# SiLVANN

> **Developer preview, v0.21.1.** It runs the models below well on the hardware it was tested on, and it is shared
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

A row a kind of machine: the model that suits it, how it runs there, its answer in the chat and its prompt and answer
after a prompt, and the machine it was measured on:

| Your machine | Model, and how it runs | In the chat | After a prompt: prompt · answer | Measured on |
|---|---|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) on the CPU alone, its experts read from the disk | — | 45 positions/s · 12.2 tokens/s | the R730's two Xeons, v0.20.0 |
| No graphics card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) on the CPU alone, in RAM | 9.2 tokens/s | 28 positions/s · 8.7 tokens/s | the desktop's four-core Xeon, v0.20.0 (the R730's 28 cores: 65 · 16.1) |
| An 8 GB card | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) (`D4`) on the card, at a 4,096-token context | 45 tokens/s | 948 positions/s · 40 tokens/s | an RTX 5060 Ti with 7.8 GB of it free, v0.20.0 (an RTX 5070 Ti: 1,682 · 59, v0.11.0; an MI50: 162 · 42, v0.7.0) |
| An 8 GB card, 32 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 4 GB of the card the tier of experts used most | 62 tokens/s | 164 positions/s · 52.2 tokens/s | an RTX 5060 Ti with 7.8 GB of it free, v0.21.0 |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 8.3 tokens/s | 19.0 positions/s · 7.8 tokens/s | an RTX 5060 Ti with 7.5 GB of it free, v0.19.0 |
| A 12 GB card, 62 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the same split, and 7.5 GB of the card the tier | 70 tokens/s | — | an RTX 5060 Ti with 11.6 GB of it free, v0.21.0 |
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) on the card | 24.5 tokens/s | 562 positions/s · 24 tokens/s | an RTX 5060 Ti, v0.20.0 (an MI50: 206 · 31.8) |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) (`D8`) on the card, at an 8,192-token context | 28 tokens/s | 868 positions/s · 26 tokens/s | an RTX 5060 Ti, v0.20.0 (an MI50: 158 · 30.6, v0.7.0) |
| A 16 GB card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 11.75 GB of the card the tier of experts used most | 72 tokens/s | 174 positions/s · 59 tokens/s | an RTX 5060 Ti, v0.21.0 (with 62 GB: 194 · 59; an RTX 5070 Ti with 32 GB: 155 · 68, v0.11.0) |
| the same | the same, without the tier | 30 tokens/s | 61 positions/s · 29 tokens/s | an RTX 5060 Ti, v0.20.0 |
| A 16 GB card, 28 GB of RAM, an NVMe disk | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 2.4 tokens/s | 11.1 positions/s · 2.2 tokens/s | an RTX 5060 Ti, v0.20.0 |
| A 16 GB card, 62 GB of RAM | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM (its config's `"experts_from": "ram"`‖), and 7.75 GB of the card the tier of experts used most | 17.5 tokens/s | 56 positions/s · 15.9 tokens/s | an RTX 5060 Ti, v0.21.0 |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`), its experts in RAM, a 300,000-token context and 1.5 GB of the card the tier, lent 3.1 GB more of the cache's unreached rows | — | 50 positions/s · 9.2 tokens/s | an MI50 with 16 GiB of it held, v0.20.0 |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) on the card | — | 379 positions/s · 88 tokens/s | an MI50, v0.21.0 (after 32,000 positions: 132 · 55) |
| A 32 GB card, 80 GB of RAM | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) (`D8E4`), its experts in RAM and 24 GB of them on the card | — | 92 positions/s · 25.0 tokens/s | an MI50, v0.20.0 |
| Two 32 GB cards, 251 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`) — every layer split over both cards (its config's `"num_cards": 2`), the experts on the CPU in RAM, 14 GB of each card the tier of experts used most, the model's whole 1,048,576-token context — the setup chosen at boot | 12.7-14.5 tokens/s | 55-56 positions/s · 11.2-11.3 tokens/s | two MI50s, v0.21.0 |
| Two 32 GB cards, 128 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`) — the same, the tier exclusive (its config's `"card_experts_gb": 14`, `"card_exclusive": true`‡): an expert on a card is not also in RAM, so 114 GB of the 142 are | 11.4-11.8 tokens/s | 36 positions/s · 10.1 tokens/s | two MI50s, the process held to 120 GiB, v0.21.0 (with 251 GB: 11.3-11.9; 41 · 10.1-10.3) |
| Two 32 GB cards, 251 GB of RAM | [DeepSeek-V4.1-Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017) (`D8E4`) — every layer split over both cards, the experts on the CPU, all in RAM, and 22.5 GB of each card an exclusive tier of them (its config's `"num_cards": 2`, `"card_experts_gb": 22.5`, `"card_exclusive": true`‡), the model's whole 1,048,576-token context | 10.5-13.7 tokens/s (6.8-9.6 the first answer after a boot) | 21.8-23.7 positions/s · 6.5-7.9 tokens/s | two MI50s, v0.21.0 |
| An older card or an integrated GPU, through OpenCL 1.2 | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, no tier | — | 37 positions/s · 23 tokens/s | the RTX 5060 Ti through OpenCL 1.2, v0.20.0 |

**In the chat** is the server with the model's own config — its sampler, the setup it chooses, or the setup its row
names — as the chat's meter reads a short conversation (from 54 positions; Ministral's from 583, its own system prompt). **After a prompt** is the
most likely token after a prompt of about 2,000 positions (2,568 for Ministral; about 1,000 where the experts are read
from the disk), the answer from its 10th token. A row's version is the release it was last measured with. The MI50s are
in a Dell R730 with two Xeon E5-2680 v4; the RTX 5060 Ti in a desktop with a four-core Xeon W-2123 and two DDR4-2400
sticks; the RTX 5070 Ti beside a Ryzen 7 5800X. Each process was held to the memory its row names, and a smaller card's
room was a larger card with the rest of it held. ‡ An exclusive tier keeps each expert on a card or in RAM, never both, so the
cards' tiers are RAM given back: GLM's needs 28 GB less of it, and answers a little slower than with its tier inclusive. DeepSeek's routed experts are
272 GB: on one MI50 34 of its 40 layers' fit in RAM and the rest are read from the disk; with 22.5 GB of each of two
cards exclusive its experts all fit beside the R730's 251 GB, and it answers about twice as fast. The server does not
choose an exclusive tier by itself: the config names it ([`docs/config.md`](docs/config.md)). ‖ With 62 GB the server reads the 122B's experts from the disk unless
its config says `"experts_from": "ram"` (they and the server beside them miss what the system has free by under a
gigabyte): 9.7 tokens/s. A model whose config names no setup — the default's — chooses its own when the server starts,
by trying it on the card: where its experts go, how much of the card holds the ones used most, the context the card has
room for ([`docs/config.md`](docs/config.md)). **Every model and pack, the other machines (down to a 4 GB card reading
its experts from the disk), and how long a conversation fits: [`docs/models.md`](docs/models.md)**; every measurement,
the machines, and each row's version: [`docs/results.md`](docs/results.md).

**New in v0.21.1:** SiLVANN in a container sets itself up in one line, the container's docs are split into one page for getting
started and one for the rest, and in a container the model's commands can no longer write the Python the server
imports as it starts.

**The container in one line.** `docker/setup.sh` finds Podman or Docker and the card — AMD, NVIDIA or none — says
what the machine is missing and the command that fixes it, asks for a Brave Search API key for the model's web search,
and starts SiLVANN at `http://silvann.localhost:8080`, or on the first free port after 8080; `update`, `stop` and
`start` after. `docker/install_nonfree.sh` adds PyAV, for MP4 and other video files, which the image does not
redistribute. `docker/README.md` is the getting started; `docker/advanced.md` the rest — SearXNG, a separate container
beside SiLVANN's, names under localhost, several cards and machines, building an image. The README's quick start
leads with the one line, and ends with a prompt to paste into an AI assistant when something does not work
([`docker/README.md`](docker/README.md)).

**The server's own Python, out of the model's reach.** The image puts the data folder's `python/` — where
`install_nonfree.sh` puts PyAV — on the server's import path. In a container the model's commands could write there,
and a module written there would have run in the server's own process at its next start, outside the fence that keeps
the tools file's keys from the model. It is now read-only to them, as the skill folders are; `setup.sh update` brings
the image that holds it ([`docs/tools.md`](docs/tools.md)).

**One table of models.** The README's models and what they run on are one table, with DeepSeek-V4.1-Flash and GLM 5.3
Flash over two MI50s with every expert in RAM among its rows: DeepSeek at 10.5-13.7 tokens a second in the chat, GLM
at 12.7-14.5 ([`docs/results.md`](docs/results.md)).

v0.21.0 brought the servers' registry for the model's subagents, long conversations on an MI50 two to three times as
fast, and `experts_register` pinning the experts' RAM only where a config asks.
v0.20.0 brought SiLVANN container-first, the model's tools set up in one file a container keeps from the model, the
web searched, tools of the model's own and from repositories of them, and a prompt's experts streamed onto the card.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6 — a model on one card, and the 27B's layers over two and four (tensor parallelism); DeepSeek-V4.1-Flash on one card with its experts on the CPU, checked against DeepSeek's own code, and over both MI50s with an exclusive tier; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs); Windows with an NVIDIA card (an RTX 5070 Ti, a Ryzen 7 5800X), natively: built from source with CUDA 13.4, Ministral 3 14B `D4` on the card ([`docs/users.md`](docs/users.md#on-windows)) |
| **Untested** | on Windows: the prebuilt libraries on a PC (GitHub's runners, which build them, have no card), the CPU alone, a model's experts on the CPU or read from the disk, AMD's and Intel's GPUs through OpenCL; Linux booted from a USB SSD; tensor parallelism on NVIDIA cards; DeepSeek-V4.1-Flash on an NVIDIA card; other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell; an older card — Kepler, Maxwell, Pascal, Volta — goes to OpenCL 1.2, the GTX 1080 first); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them (on one, the model's memory is read where it is, not copied to the GPU: checked only through PoCL, a CPU's OpenCL) — which card goes to OpenCL 1.2 and which to 2 is a best guess from the vendors' published versions (OpenCL 1.2: every NVIDIA card CUDA does not take, Intel's iGPUs before Gen9, AMD's TeraScale and GCN 1; OpenCL 2: the rest), and a card on the wrong one is an issue to open, with its `clinfo`; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not supported** | macOS |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0: [`release_logs/v0.md`](release_logs/v0.md#v070)), pictures by DeepSeek-V4.1-Flash (v0.17.0: [`docs/users.md`](docs/users.md#3-serve-it-and-chat)); Ministral 3 reads neither.
Every model calls the tool, a Lisp REPL on the machine, in its own trained format (v0.9.0: [`docs/users.md`](docs/users.md#the-models-tool)).
Pictures are made from a prompt by Qwen-Image 2.1, a server's model of its own (v0.10.0: [`docs/users.md`](docs/users.md#a-picture-model-qwen-image-21)).
A model's tool draws with the picture model, opens subagents and keeps skills of its own; a dense Qwen model's layers split over AMD cards (v0.11.0: [`docs/tools.md`](docs/tools.md), [`docs/config.md`](docs/config.md)).

## Quick start

Three ways in; each ends with the chat in your browser.

### Linux: a container (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash
```

It sets SiLVANN up in a rootless container for your card — AMD, NVIDIA, or none — says what the machine is missing if
anything is, asks for a Brave Search API key so the model can search the web (Enter skips it), and starts it. Then open
the address it prints — http://silvann.localhost:8080, or the next free port where 8080 is taken: in **Options**,
**Download models** fetches a model and **start** runs it ([`docs/ui.md`](docs/ui.md)). To read MP4 and other video
files, run [`docker/install_nonfree.sh`](docker/README.md#set-it-up) once after it. In the container the model's tool reaches one folder of yours, `~/SiLVANN` — the models,
the conversations, and the workspace where you put what it should read — and nothing else of the machine
([why](docs/local_ai.md#container-first)). Updating, stopping, and the commands by hand:
[`docker/README.md`](docker/README.md).

### Windows: the installer

You need Python 3.10 to 3.14, Git and your card's driver; if Python or Git is missing, `winget install -e --id
Python.Python.3.12` and `winget install -e --id Git.Git`, then open a new PowerShell. In PowerShell, in the folder
SiLVANN should go under:

```powershell
curl.exe -fsSLO https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Then open a new PowerShell (the `silvann` command is on the path from there) and start a model:

```powershell
silvann download silvann_tq_D4_ministral_3_14b
silvann serve silvann_tq_D4_ministral_3_14b      # keeps running: open another PowerShell for the chat
```

and in another PowerShell, `silvann ui`: the chat opens at http://localhost:8090. To update, `silvann update`; the
models and the conversations stay. On Windows the model's shell starts switched off
([`docs/tools.md`](docs/tools.md#where-the-models-commands-reach)); the rest: [`docs/users.md`](docs/users.md#on-windows).

### Linux, from source

One line builds SiLVANN in `./silvann` for this machine's CPU and card, installs nothing system-wide, asks for no
password, and leaves a `silvann` command:

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
silvann download silvann_tq_D8E4_qwen3.6_35b_a3b
silvann serve silvann_tq_D8E4_qwen3.6_35b_a3b    # keeps running: open another terminal for the chat
```

and in another terminal, `silvann ui`: the chat opens at http://localhost:8090. `silvann help` lists the rest;
`silvann update` updates, and the models and the conversations stay. The compilers a card needs, building by hand,
and keeping the models on another drive: [`docs/users.md`](docs/users.md).

### If you get stuck

An AI assistant can walk you through it. Paste this into one:

```text
Help me install SiLVANN, a program that runs large language models on my own machine. Read its instructions first:
https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/README.md and, on Linux,
https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/README.md. Then ask me to run these commands one
at a time and paste what they print, and from what they show tell me what to do next, using only the commands in those
instructions and saying what each one does before I run it:
uname -a; cat /etc/os-release; podman --version; docker --version; nvidia-smi -L; ls /dev/kfd; id -nG; free -g
```

### Where to read on

- Every setting a model boots with, and each model's options: [`docs/config.md`](docs/config.md).
- The model's tool, and adding your own commands to it: [`docs/tools.md`](docs/tools.md).
- The experts read from the disk, and a model split over several machines: [`docs/users.md`](docs/users.md).
- Every measurement: [`docs/results.md`](docs/results.md).
- How it is built, and how a model or a silicon is added: [`docs/developers.md`](docs/developers.md).
- The ideas behind the design, the Lisp lineage and the model as a collapse of meaning:
  [`docs/concepts.md`](docs/concepts.md); how the weights are packed: [`docs/compression.md`](docs/compression.md).
- Where the design could go next: [`docs/directions.md`](docs/directions.md).
- Why a model on your own machine matters: [`docs/local_ai.md`](docs/local_ai.md).

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
