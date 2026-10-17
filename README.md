# SiLVANN

> **Developer preview, v0.11.0.** It runs the models below well on the hardware it was tested on, and it is shared
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
| No graphics card, 12 GB of RAM, an NVMe disk | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D4E4_qwen3.6_35b_a3b) (`D4E4`) on the CPU alone, its experts read from the disk | 11.7 tokens/s |
| An 8 GB card | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) (`D4`) on the card, at a 4,096-token context | 42 tokens/s ◦ |
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) on the card | 31.5 tokens/s |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) (`D8`) on the card, at an 8,192-token context | 30.6 tokens/s ◦ |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) on the card | 90 tokens/s |
| An 8 GB card, 32 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 2 GB of the card the tier of experts used most | 42 tokens/s § |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/silvann_tq_D4E4_qwen3.5_122b_a10b) (`D4E4`) — the dense part on the card, the experts on the CPU | 4.7 tokens/s ◦ |
| A 16 GB card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 8 GB of the card the tier of experts used most | 55.9 tokens/s ‡ |
| A 16 GB card, 28 GB of RAM, an NVMe disk | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 2.2 tokens/s ‡ |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 19.9 tokens/s |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/silvann_tq_D4E4_glm_5.3_flash), its experts in RAM and 6 GB of them on the card | 7.5 tokens/s |
| A 32 GB card, 256 GB of RAM | [Qwen 3.5 397B-A17B](https://huggingface.co/eval-apply/silvann_tq_D4E4_qwen3.5_397b_a17b), its experts in RAM and 24 GB of them on the card | 9.0 tokens/s |

On AMD Instinct MI50 cards in a Dell R730, each process held to the memory its row names — except the rows with experts
on the card, which ran on the machine's 256 GB and name what the model needs. ◦ Ran on a 32 GB card: the model fits the
smaller one, which has not been measured. ‡ On the RTX 5060 Ti desktop below, with v0.7.0. § The same, with v0.8.0 and
7.5 GB of the card held: the room an 8 GB card with nothing else on it leaves, the setup the one v0.8.0 chooses there.
The rows with a model entirely on the card were measured with v0.7.0 before its pictures were added (its answers since
within 1% of them); the others were measured with v0.3, and where a card does part of the work they are a floor. A model
whose config names no setup — the default's — chooses its own when the server starts, by trying it on the card: where
its experts go, how much of the card holds the ones used most, the context the card has room for
([`docs/config.md`](docs/config.md)). **Every model and pack, the other machines (down to a 4 GB
card reading its experts from the disk), and how long a conversation fits: [`docs/models.md`](docs/models.md)**; every
measurement: [`docs/results.md`](docs/results.md).

On an NVIDIA card — a GeForce RTX 5060 Ti 16 GB in a desktop with a four-core Xeon W-2123 and two DDR4-2400 sticks
(62 GB), about what a local-model enthusiast's machine has — measured with v0.5.0 the same way, a prompt of about 2,000
positions and then the answer:

| Model | How it runs | Prompt | Answer |
|---|---|---|---|
| [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) (`D4`) | on the card | 958 positions/s | 39 tokens/s (46 on a short prompt) |
| [Ministral 3 14B](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) (`D8`) | on the card | 873 positions/s | 25 tokens/s (27 on a short prompt) |
| [Qwen 3.8 27B](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) | on the card | 549 positions/s | 23.6 tokens/s (24.4 on a short prompt) |
| [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) (`D8E4`) | the dense part on the card, the experts on the CPU, and 8 GB of the card the tier of experts used most | 117 positions/s | 52 tokens/s |
| the same | without the tier | 55 positions/s | 28 tokens/s |
| the same | on the CPU alone | 21 positions/s | 8.6 tokens/s |

**New in v0.11.0:** the model's tool grows into a workbench — it draws pictures, hands work to other models and
keeps commands of its own; a Qwen MoE model's context and its card's tier of experts share the card with nothing to
set; and a dense model's layers split over several AMD cards.

**Pictures from the tool.** A text model's REPL draws with the picture model: `(image "a lighthouse at dusk --size
768x1024 --seed 7")`. On the text model's own card the model is put down, the picture drawn and the model booted again
with its conversations where they were — back in 10 to 11 s on an MI50 and on the RTX 5060 Ti 16 GB — or the picture is
drawn on another card (`image_card`) and nothing is put down. The config's `image_model` says which picture model:
`"auto"`, a downloaded one, by default; without one the tool says so at once, and the model tells you. The picture is
shown in the conversation and kept with it, and it is the picture model's own: the same text and seed through the 27B's
tool and through Qwen-Image's server gave the same picture, every pixel. `--count 4` draws four, seeds s to s + 3, one a
card at once — on four MI50s in 120 s against 261 s on one card, the same pictures — in a picture model's messages too.

**Subagents.** `(agent 'model "config" "task")` opens a conversation with another model, or the same one under a LoRA,
and the model is then inside it as in a REPL: each call is a message, `(answer 'once)` or `(answer 'refuse)` answers the
subagent's ask — only what the caller could do itself without asking, anything more asked of you — and `(exit)` ends it.
It runs on the same server (`'self`), on a machine in the config's `workers`, or in place, the card lent to the other
model for each exchange; its REPL reads and lists only, unless its config allows more. `(whoami)` says which model this
is, its cards and context, its LoRAs with what each is for, and what `(agent …)` can reach: given a task, the 27B and the
35B chose the LoRA whose description fit it.

**The REPL.** `(reset)` starts it afresh — its names, its folder, a waiting ask — and keeps the files, the folders you
allowed, its mode and the library: definitions kept across conversations with `(keep 'name "what it does")`,
`(library)`, `(import 'name)` and `(forget 'name)`, in `tools_library.json` beside `tools.json`. The names the models
reached for and the REPL lacked now answer: `read-file`, `slurp-file`, `read-all`, `cat` and `slurp` are `read`;
`system`, `shell` and `sh` are `bash` — 18 of the 25 such calls in 159.

**Skills** — commands that live in folders: a `SKILL.md` names each command, what it touches (so it is asked as a
built-in command would be) and the Python function or the program that runs it, in the data folder's `skills/`. The
release ships one, `(find "folder" "*.txt")`. `(skills)` lists them, `(help)` names their commands, and `(guide)` gives
the model the tool's guide, `docs/tools.md`, so it can write a skill of its own and keep it: the 35B and the 27B each
did, from the guide alone.

**Tensor parallelism, opt-in, on AMD cards.** `"options": {"tensor_parallel": 2}` splits each layer of a dense Qwen
model over that many cards of one memory size: the 27B from 29.9 to 23.1 ms a token on two MI50s, a 2,000-position
prompt 1.8 times faster on two cards and 2.2 on four; refused for the mixture-of-experts models, over cards of unequal
memory, and on one card. Untested on NVIDIA cards. It needs `lisp/qwen3_5_tp.lisp` in the model's folder, on the Qwen
packs on Hugging Face: `silvann download` of a model already here fetches its pack's `lisp/` whole.

**No context to set, no tier to size.** Every model's `max_context` is `"auto"` by default now: as many positions as
the card holds beside the model, up to the model's own length, found when the server starts. With a mixture-of-experts
model's experts on the CPU, the card's tier of experts also holds experts in the cache's rows no conversation has
reached, giving them back 8,192 positions at a time before a conversation writes there, and taking them again for a
shorter one. On the RTX 5060 Ti 16 GB the 35B-A3B at 262,144 positions lends its tier 2.4 GB that way, and answers as
fast as a boot sized for the conversation: 54.0 tokens/s after 2,000 positions, 40.8 after 16,000 and 30.8 after
32,000, against 54.1, 40.7 and 30.8. GLM 5.3 Flash's tier is lent its latent cache's rows the same way: 11 GB on a 32 GB
MI50 at its whole 1,048,576 positions. Where the card cannot hold the whole length beside the model, the tier keeps the
least it takes, a slot a layer, and the context the rest: GLM on 16 GB of an MI50 790,528 positions and a 0.5 GB tier,
lent the rows a conversation has not reached. With `"auto"` the experts stay on a card that holds them with 32,768
positions or more. A config written before keeps the `max_context` it names: set it to `"auto"` in Options
([`docs/config.md`](docs/config.md#the-context-and-the-tier-share-the-card)).

A model booted a second time in one process — the text model back after a picture, or after lending its card to a
subagent — no longer takes the first boot's expert copies still in flight into its tier: the 35B with a tier answered
another conversation wrongly after an `(image …)`, and now answers as if it had never been put down. A GLM conversation
past 2,048 positions, put away while another had the card and brought back, chose the positions it attends from the
other's indexer: a conversation keeps its indexer's caches with it now, and its next positions are those it would have
had never put away, to the bit.

**The docs and the chat.** [`docs/config.md`](docs/config.md) — every key of a model's config, every family's options —
and [`docs/tools.md`](docs/tools.md) — the REPL, the library and skills, step by step — are both in the UI's Help, and
the UI's New config writes the keys the server reads. The chat's top bar shows how full the conversation is: its tokens
of the model's context. An emoji split over two tokens is shown whole (it was "�"), and `kv` refuses a value it does not
know (any other ran the fp16 cache). The Help view lists every document the release ships, in four groups, first a
guide to the UI itself ([`docs/ui.md`](docs/ui.md)), and a link from one document to another opens it there.

**The command and the setup.** `silvann models <folder>` keeps the models, the conversations and the rest SiLVANN keeps
for you in another folder from then on: SiLVANN on a USB drive, its models on the machine's own disk. `silvann help`
lists the commands, and a word it does not know says so. A setup kept in `autotune.json` is chosen again after an
update, which may change what a boot needs; and a tier's size named alone in a Qwen config puts the experts on the CPU
and chooses the rest, where it stopped the boot. `vram_margin_mb` (50 by default) is how much of each card the
setup chosen at boot leaves free for a desktop or a browser — `0` on a card nothing else uses; it was a fixed 256.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6 — a model on one card, and the 27B's layers over two and four (tensor parallelism); Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs); Windows 11 and Windows 10 with an NVIDIA card, through the Docker image ([`docker/README.md`](docker/README.md)) — where the driver gives a model too large for the card the PC's RAM as shared memory, the placement is named in the model's config ([`docs/users.md`](docs/users.md#on-windows-the-docker-image)) |
| **Untested** | Linux booted from a USB SSD; tensor parallelism on NVIDIA cards; other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not yet** | Apple silicon |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0: [`release_logs/v0.md`](release_logs/v0.md#v070)); Ministral 3 does not read them.
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

### Anywhere else: using Docker

The way on Windows, and a second way on Linux: the image `ghcr.io/eval-apply-labs/silvann:main`, the nightly, is
SiLVANN built for NVIDIA cards and the CPU, with the chat as the app in the browser; the models and the conversations
live in a Docker volume, which an update keeps.

On Windows, with WSL 2 (`wsl --install` in a PowerShell run as administrator, then restart), Docker Desktop and NVIDIA's
ordinary driver: [download SiLVANN](https://github.com/eval-apply-labs/SiLVANN/archive/refs/heads/main.zip), unzip it,
and in its `docker\windows` folder double-click

1. `silvann_install.cmd` — it checks the PC, pulls the image and puts **SiLVANN** and **SiLVANN - stop** on the desktop;
2. **SiLVANN** — the chat opens at http://localhost:8080, and its Options download a model and start it;
3. `silvann_install_nonfree.cmd`, once, to read video files other than animated pictures (PyAV, which the image leaves
   out).

**SiLVANN data**, also on the desktop, opens the models and the conversations in Explorer: a model downloaded on another
machine is pasted into its `models` folder.

On Linux, with Docker Engine (or Podman) and NVIDIA's container toolkit for the card:

```bash
curl -fsSLO https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/silvann_docker.sh
bash silvann_docker.sh install            # the card checked, the image pulled, the volume made
bash silvann_docker.sh start              # the chat at http://localhost:8080 — its Options download a model and start it
bash silvann_docker.sh install_nonfree    # once, for video files (PyAV)
```

Updating, stopping, sharing on the LAN, exporting to another machine: [`docker/README.md`](docker/README.md).

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

## What is here

```
install.sh          fetches, builds and installs the `silvann` command — the one-line install
build.sh            builds the engine into lib/
docker/             the Docker image, and the scripts that run it as an app on Windows and Linux
engine/src/         the interpreter (engine, sys), the library (nn), the models (ai_qwen_3, ai_glm_5_3, ai_qwen_image), and
                    silicon_families/ — one folder a kind of silicon, what it answers for the library
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
                    directions.md — where it could go
```

## Contact

Questions, ideas and results from your own machine go in this repository's **Discussions** tab; bugs go in
**Issues**. For anything you would rather not post in public, write to **admin@evalapply.co.uk**.

## Contributing

Reports from hardware we have not tried, bug reports and fixes are all welcome — see `CONTRIBUTING.md`. Pull
requests are made under the Contributor License Agreement in `CLA.md`, which a bot asks you to accept once.

## Licence

The engine is AGPL-3.0 (`LICENSE`), with an additional permission to convey it linked with NVIDIA's CUDA runtime
(`ATTRIBUTION.md` §1). Third-party notices are in `ATTRIBUTION.md`, the Docker image's among them (§6). The model weights
keep their authors' licences.
