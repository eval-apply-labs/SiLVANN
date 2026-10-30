# SiLVANN

> **Developer preview, v0.23.0.** It runs the models below well on the hardware it was tested on, and it is shared
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

The model for each kind of machine, from no card to two server cards — the card, the RAM, the CPU and the disk it was
measured on, how fast it reads a prompt and answers after one:

<!-- rates:readme -->
| GPU | RAM | CPU | Disk | Model | Setup | Prefill | Decode | Max context | Or |
|---|---|---|---|---|---|---|---|---|---|
| [none](docs/results.md#no-graphics-card) | 16 GB | 4 cores | — | [Qwen 3.5 4B](docs/models.md#qwen-35-4b) `D4` | the CPU alone | 18.9 positions/s | 7.95 tokens/s after 8k | 8,320 | — |
| [8 GB](docs/results.md#an-8-gb-card) | 32 GB | 4 cores | — | [Qwen 3.6 35B-A3B](docs/models.md#qwen-36-35b-a3b) `D8E4` | the experts in RAM, a tier on the card | 94.9 positions/s | 14.34 tokens/s after 100k | 262,144 | the 4B on the card |
| [16 GB](docs/results.md#a-16-gb-card) | 32 GB | 4 cores | NVMe, 5.7 GB/s | [Qwen 3.5 122B-A10B](docs/models.md#qwen-35-122b-a10b) `D8E4` | the experts from the disk, ~30 GB of them in RAM | 17.6 positions/s | 5.65 tokens/s after 100k | 110,000 | GLM 5.3 Flash from the disk |
| [16 GB](docs/results.md#a-16-gb-card) | 14 GB | 4 cores | NVMe, 5.7 GB/s | [GLM 5.3 Flash](docs/models.md#glm-53-flash) `D8E4` | the experts from the disk | 9.1 positions/s | 2.21 tokens/s after 100k | 110,000 | — |
| [16 GB](docs/results.md#a-16-gb-card) | 16 GB | 4 cores | — | [Qwen 3.8 27B](docs/models.md#qwen-38-27b) `D4` | on the card | 221.3 positions/s | 14.32 tokens/s after 35k | 35,840 | — |
| [2× 16 GB](docs/results.md#two-cards-or-more) | 62 GB | 28 cores | — | [Qwen 3.6 35B-A3B](docs/models.md#qwen-36-35b-a3b) `D8E4` | every layer over both cards, a tier on each | 85.3 positions/s | 30.5 tokens/s after 100k | 262,144 | the 35B on one card, faster after a short prompt |
| [32 GB](docs/results.md#a-24-or-32-gb-card) | 160 GB | 28 cores | — | [GLM 5.3 Flash](docs/models.md#glm-53-flash) `D8E4` | every expert in RAM, a tier on the card | 72.1 positions/s | 9.86 tokens/s after 100k | 1,048,576 | DeepSeek-V4.1-Flash over two cards |
| [2× 32 GB](docs/results.md#two-cards-or-more) | 251 GB | 28 cores | — | [DeepSeek-V4.1-Flash](docs/models.md#deepseek-v41-flash) `D8E4` | every expert in RAM, exclusive tiers | 31.3 positions/s | 9.72 tokens/s after 100k | 1,048,576 | GLM 5.3 Flash on one card |
<!-- /rates -->

Every model measured on each kind of machine, and how: [`docs/results.md`](docs/results.md#which-model-for-your-machine);
every machine one model was measured on, its packs and its notes: [`docs/models.md`](docs/models.md). A model whose config
names no setup chooses its own when the server starts, by trying it on the card: where its experts go, how much of the
card holds the ones used most, the context the card has room for ([`docs/config.md`](docs/config.md)).

**New in v0.23.0**, and in every release before it: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6 — a model on one card, and the 27B's layers over two and four (tensor parallelism); DeepSeek-V4.1-Flash on one card with its experts on the CPU, checked against DeepSeek's own code, over both MI50s with an exclusive tier, and on the CPU alone from the disk; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs); Windows with an NVIDIA card (an RTX 5070 Ti, a Ryzen 7 5800X), natively: built from source with CUDA 13.4, Ministral 3 14B `D4` on the card ([`docs/users.md`](docs/users.md#on-windows)) |
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

Where there is no `curl` (Debian's netinstall has only `wget`): `wget -qO- https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash`.

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

Then, in a new PowerShell (the `silvann` command is on the path from there), `silvann ui`: the chat opens at
http://localhost:8090, and in **Options**, **Download models** fetches a model and **start** runs it. To update,
`silvann update`; the models and the conversations stay.

Where Windows 11's Smart App Control is on, it refuses the libraries GitHub built, which are not signed yet; a proper
code signature is underway. Libraries built on the PC are subject to Smart App Control's lottery — some PCs let them
load, others do not — and anyone who wants to play it meanwhile can: `.\install.ps1 -FromSource` builds them on the
PC, with Visual Studio's Build Tools and their Clang tools, and CUDA's toolkit for an NVIDIA card
([`docs/users.md`](docs/users.md#on-windows), which has the line that installs them).
On Windows the model's shell starts switched off ([`docs/tools.md`](docs/tools.md#where-the-models-commands-reach)).

### Linux, from source

One line builds SiLVANN in `./silvann` for this machine's CPU and card, installs nothing system-wide, asks for no
password, and leaves a `silvann` command; `silvann ui` then opens the chat at http://localhost:8090, where **Options**
downloads a model and starts it:

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
silvann ui
```

`silvann help` lists the rest; `silvann update` updates, and the models and the conversations stay. The compilers a
card needs, building by hand, and keeping the models on another drive: [`docs/users.md`](docs/users.md).

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
