# SiLVANN

> **Developer preview, v0.7.0.** It runs the models below well on the hardware it was tested on, and it is shared
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
| An 8 GB card, 32 GB of RAM, an NVMe disk | [Qwen 3.5 122B-A10B](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) (`D4E4`) — the dense part on the card, the experts on the CPU | 4.7 tokens/s ◦ |
| A 16 GB card, 28 GB of RAM | [Qwen 3.6 35B-A3B](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 8 GB of the card the tier of experts used most | 55.9 tokens/s ‡ |
| A 16 GB card, 28 GB of RAM, an NVMe disk | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D8E4) (`D8E4`) — the dense part on the card, the experts on the CPU, read from the disk | 2.2 tokens/s ‡ |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B, its experts in RAM and 24 GB of them on the card | 19.9 tokens/s |
| A 16 GB card, 150 GB of RAM | [GLM 5.3 Flash](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4), its experts in RAM and 6 GB of them on the card | 7.5 tokens/s |
| A 32 GB card, 256 GB of RAM | [Qwen 3.5 397B-A17B](https://huggingface.co/eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4), its experts in RAM and 24 GB of them on the card | 9.0 tokens/s |

On AMD Instinct MI50 cards in a Dell R730, each process held to the memory its row names — except the rows with experts
on the card, which ran on the machine's 256 GB and name what the model needs. ◦ Ran on a 32 GB card: the model fits the
smaller one, which has not been measured. ‡ On the RTX 5060 Ti desktop below, with v0.7.0. The rows with a model
entirely on the card were measured on this release before its pictures were added (its answers since within 1% of them);
the others were measured with v0.3, and where a card does part of the work they are a floor. **Every model and pack, the
other machines (down to a 4 GB card reading its experts from the disk), and how long a conversation fits:
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

**New in v0.7.0:** pictures. The Qwen models read one — the 27B, the 35B-A3B and the 122B, whose packs carry the vision
tower: in the UI the 📎 button attaches it to a message, and over the server's API `/cartridges/<id>/images` reads it
and a message names it. The picture is prepared as Hugging Face's processor prepares it and read by the model's tower;
its tokens take the model's three-part positions (time, row and column: M-RoPE), which the library computes in a word of
its own. The tower runs on the card, in the library's words — LayerNorm, GELU, attention over a set of patches, their
two-axis rotary — over the pack's own codes, with the CUDA family's set attention on the tensor cores and the AMD
family's own: pictures up to 4 megapixels (2048 x 2048, 4,096 tokens), from a quarter of a second for a small one to
about 15 s at the largest on an RTX 5060 Ti, and from 0.13 s to about 8.5 s on an MI50. Its weights (0.6 GB) stay on the card when the experts are on the CPU and in RAM otherwise, and its working
memory on the card is sized at boot for the largest picture — the 27B at its default context still fits a 16 GB card
(`picture_pixels`, `picture_weights`: [`docs/users.md`](docs/users.md)). Checked against Hugging Face on the same
weights: the tower's output within 0.5% (Hugging Face's own tower in bfloat16 is 2.5% from its float32), and over a
picture's prompt the first attention layer within 6e-4. A conversation keeps its pictures, and is saved and brought back
with them. GLM 5.3 Flash reads pictures too, through the tower its pack carries (24 blocks of its own design, in the
library's words on the card: 0.25 s for 448 x 448, 12 s for 2044 x 2044 on an MI50; within 0.92% of Hugging Face's
float32, whose own bfloat16 is 7.6% away). Ministral 3 reads none: its pack carries no tower.

Videos, for the Qwen models and GLM: the same button attaches one — an animated GIF, WebP or PNG as it is, any other
video file through PyAV, which `install.sh` installs from `requirements-nonfree.txt` and the Docker image leaves out
(its scripts' `install_nonfree` adds it: [`docker/README.md`](docker/README.md)). Its frames are taken at 2 a second, two frames read
together as one picture by the tower, each pair's time written into the message as the model's processor writes it; a
video is made at most half the conversation's context in tokens. Checked against Hugging Face's processor and tower on
the same weights — the frames, their times, the prompt to the token, the Qwen models' positions, the tower's rows
within 6e-4 — and a 6-second video's digits named in order by the 27B, the 35B-A3B and GLM.

Files, for every model: a Word document (`.docx`), a PDF (through pypdf, which the install brings) or a text file
(Markdown, CSV, JSON, code), attached with the same button, is read when it is attached, and its text goes into the
message before what you type, cut with a note to what the conversation's context still holds.

A model's program travels with it: the pictures' part is a file of its own, `lisp/qwen3_5_pictures.lisp`, which a folder fetched
before this release lacks — it reads text as before and refuses pictures until `silvann download <model>` fetches the
file.

A Docker image, for Windows above all ([`docker/README.md`](docker/README.md)). It is the release, its environment and
its build for an NVIDIA card or the CPU, with the chat as the app in the browser; scripts install, update, start, stop,
export and import it — the models and the conversations included, so a copy on one drive starts again on another. It is
not published yet: `docker build -f docker/Dockerfile .` builds it. It has run on Linux, where the 27B and the 35B
answer in it as Hugging Face does, at the speed they have outside it; not yet on Windows. The UI downloads a model too:
its Options list the models this release fetches, download one with its progress and start it; a Hugging Face token
saved there, or with `silvann token --set`, makes the downloads faster. `SILVANN_DATA` puts the models and the
conversations in a folder of their own. And on a machine of two processors the runtime keeps itself out of the kernel's
NUMA balancing, whose sampling cost a card's run a few percent: on an MI50 in a two-socket server the 35B answers at 90
tokens a second, from 87.5, and reads a prompt at 199 positions a second, from 194.5; the card rows above are this
release's.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs) |
| **Untested** | Windows — the Docker image ([`docker/README.md`](docker/README.md)), run so far on Linux only, or Linux booted from a USB SSD ([`docs/users.md`](docs/users.md#on-windows-the-docker-image)); other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not yet** | Apple silicon |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0, above); Ministral 3 does not read them.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash   # into ./silvann, and a `silvann` command
silvann download Qwen3.6-35B-A3B_silvann_tq_D8E4
silvann serve Qwen3.6-35B-A3B_silvann_tq_D8E4                       # on port 8765
silvann ui                                                          # a chat in the browser, at http://localhost:8090
```

With Docker — the way on Windows, and a second way on Linux: the image `ghcr.io/eval-apply-labs/silvann:main`, the
nightly, is SiLVANN built for NVIDIA cards and the CPU, with the chat as the app in the browser; the models and the
conversations live in a Docker volume, which an update keeps.

On Windows, with WSL 2 (`wsl --install` in a PowerShell run as administrator, then restart), Docker Desktop and NVIDIA's
ordinary driver: [download SiLVANN](https://github.com/eval-apply-labs/SiLVANN/archive/refs/heads/main.zip), unzip it,
and in its `docker\windows` folder double-click

1. `silvann_install.cmd` — it checks the PC, pulls the image and puts **SiLVANN** and **SiLVANN - stop** on the desktop;
2. **SiLVANN** — the chat opens at http://localhost:8080, and its Options download a model and start it;
3. `silvann_install_nonfree.cmd`, once, to read video files other than animated pictures (PyAV, which the image leaves
   out).

On Linux, with Docker Engine (or Podman) and NVIDIA's container toolkit for the card:

```bash
curl -fsSLO https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/silvann_docker.sh
bash silvann_docker.sh install            # the card checked, the image pulled, the volume made
bash silvann_docker.sh start              # the chat at http://localhost:8080 — its Options download a model and start it
bash silvann_docker.sh install_nonfree    # once, for video files (PyAV)
```

Updating, stopping, sharing on the LAN, exporting to another machine: [`docker/README.md`](docker/README.md).

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
