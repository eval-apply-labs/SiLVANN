# SiLVANN

> **Developer preview, v0.8.0.** It runs the models below well on the hardware it was tested on, and it is shared
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
| A 16 GB card | [Qwen 3.8 27B](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) on the card | 31.5 tokens/s |
| A 16 GB card — alt | [Ministral 3 14B](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D8) (`D8`) on the card, at an 8,192-token context | 30.6 tokens/s ◦ |
| A 24 GB card | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) on the card | 90 tokens/s |
| An 8 GB card, 32 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 2 GB of the card the tier of experts used most | 42 tokens/s § |
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) (`D4E4`) — the dense part on the card, the experts on the CPU | 4.7 tokens/s ◦ |
| A 16 GB card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 8 GB of the card the tier of experts used most | 55.9 tokens/s ‡ |
| A 16 GB card, 28 GB of RAM, an NVMe disk | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D8E4) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 2.2 tokens/s ‡ |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 19.9 tokens/s |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4), its experts in RAM and 6 GB of them on the card | 7.5 tokens/s |
| A 32 GB card, 256 GB of RAM | [Qwen 3.5 397B-A17B](https://huggingface.co/eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4), its experts in RAM and 24 GB of them on the card | 9.0 tokens/s |

On AMD Instinct MI50 cards in a Dell R730, each process held to the memory its row names — except the rows with experts
on the card, which ran on the machine's 256 GB and name what the model needs. ◦ Ran on a 32 GB card: the model fits the
smaller one, which has not been measured. ‡ On the RTX 5060 Ti desktop below, with v0.7.0. § The same, with v0.8.0 and
7.5 GB of the card held: the room an 8 GB card with nothing else on it leaves, the setup the one v0.8.0 chooses there.
The rows with a model entirely on the card were measured with v0.7.0 before its pictures were added (its answers since
within 1% of them); the others were measured with v0.3, and where a card does part of the work they are a floor. A model
whose config names no setup — the default's — chooses its own when the server starts, by trying it on the card: where
its experts go, how much of the card holds the ones used most, the context the card has room for
([`docs/users.md`](docs/users.md#4-the-models-settings)). **Every model and pack, the other machines (down to a 4 GB
card reading its experts from the disk), and how long a conversation fits: [`docs/models.md`](docs/models.md)**; every
measurement: [`docs/results.md`](docs/results.md).

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

**New in v0.8.0:** pipeline parallelism, and a model's setup chosen for the machine it starts on.

A ring of machines: a model's layers split over several PCs, each holding a run of them, each server started from its
own config — `pipeline_parallelism` "yes", its `pipeline_parallelism_layers` and `pipeline_parallelism_next`, the
machine after it; the one whose layers begin at 0 answers the UI, the one with the last layers runs the head
([`docs/users.md`](docs/users.md)). A model larger than one machine's RAM or card runs over several: GLM 5.3 Flash over
two stages, its experts in their RAM, gives the whole model's tokens, 33 of 33, at 6.95 tokens a second on two MI50s;
the 27B over two stages the same, 33 of 33. A config's `card` picks a machine's card, or `"cpu"` its CPU and RAM — so a
card and the CPU beside it are two machines of one ring: on the RTX 5060 Ti, the 35B-A3B half on the card and half in
RAM answers as Hugging Face's does (7 of 8 tokens, then its second at its own near-tie) at 13.8 tokens a second. What
crosses the network is a row of numbers a position. A Qwen model's pictures are read on one machine for now.

A config that names no placement — where the experts sit, how much of the card holds the ones used most, how long a
conversation the card has room for — gets one when the server starts: the setups are tried on the card fastest first
(one the card cannot hold is refused in about 2 seconds), the RAM is measured — a container's or a service's memory
limit counts, and 500 MB is left spare beside what the server holds — and the choice is kept in the model's folder,
`autotune.json`, so the next start takes it without trying ([`docs/users.md`](docs/users.md#4-the-models-settings)).
`max_context` "auto" fills the card, up to the model's own length; `card_experts` "yes" or "no" keeps or leaves out the
card's tier, sized in 256 MB steps; a placement the config names is started as it is. On an MI50 it gives the 122B with
80 GB of RAM a 25 GB tier — 24.6 tokens a second, as the 24 GB given by hand — and GLM with 150 GB a 21 GB tier, 9.2-9.4
against 7.9-8.1; on an RTX 5060 Ti with 28 GB of RAM, the 35B-A3B an 11 GB tier, 57 tokens a second, and the 122B and
GLM their experts from the disk, as their rows of the list. A server that cannot start says why, in one line in the UI's
Options, with the setting to change. When RAM cannot hold two conversations at the full context, the one put aside is
saved to its folder and read back when it returns.

A video is read to its budget: the frames the processor samples, each fitted as it is read — a minute of 1080p held 0.39
GB at the peak, from 10.48 — and `max_tokens_per_video` (8,192 by default) sets how many tokens one becomes. With the
experts on the CPU the server no longer grows as it answers (40 rounds had added half a gigabyte; now flat), and its
answers are the same to the bit.

The Docker image keeps what it holds across an update: a test updates a running app from one image to a newer one of the
same tag, over the same volume, and checks that the model answers, that the conversations — a picture's and a video's
among them — come back with their history and go on, and that the video reader and the token are still there; it passed
from v0.7.0's image to this one, and runs before every release. Its data is a folder: the desktop's **SiLVANN data**
(`silvann_open_data`) opens it in Explorer, and a model downloaded on another machine is pasted into its `models`; the
image gives everything in it to the app's user each time it starts. `silvann_reclaim_space` gives Windows back the room
a deleted model leaves in Docker Desktop's disk file. The Windows scripts make their folder whole at each run — an
action or a desktop shortcut an older install lacks. `SILVANN_NAME` and `SILVANN_VOLUME` keep a second SiLVANN beside
the first.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs) |
| **Untested** | Windows — the Docker image ([`docker/README.md`](docker/README.md)), run so far on Linux only, or Linux booted from a USB SSD ([`docs/users.md`](docs/users.md#on-windows-the-docker-image)); other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not yet** | Apple silicon |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0: [`release_logs/v0.md`](release_logs/v0.md#v070)); Ministral 3 does not read them.

## Quick start

### Linux: the install script

One line in a terminal fetches SiLVANN into `./silvann` and builds it for this machine — the CPU, and the card it finds:
an NVIDIA card with CUDA 12's `nvcc`, an AMD card with ROCm 6's `hipcc`, each for that card alone (where the compiler is
missing it says where to get it, and the CPU runs the models meanwhile). It installs nothing system-wide and asks for no
password, and it leaves a `silvann` command. Run it again — or `silvann update` — to update: the models, the
conversations and their settings stay.

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

### Docker

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
docker/             the Docker image, and the scripts that run it as an app on Windows and Linux
engine/src/         the interpreter (engine, sys), the library (nn), the models (ai_qwen_3, ai_glm_5_3), and
                    silicon_families/ — one folder a kind of silicon, what it answers for the library
runtime/            the Python side: a model folder booted onto the machine, conversations, the chat framing
lisp/               each model family's program, in Lisp — a model folder carries its family's in its own lisp/
silvann_core.py     download-model · list-models
start_silvann_server.py, silvann_ui.py
models/             where downloaded models go
release_logs/       v0.md — what each 0.x release brought, newest first
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

The engine is AGPL-3.0 (`LICENSE`), with an additional permission to convey it linked with NVIDIA's CUDA runtime
(`ATTRIBUTION.md` §1). Third-party notices are in `ATTRIBUTION.md`, the Docker image's among them (§6). The model weights
keep their authors' licences.
