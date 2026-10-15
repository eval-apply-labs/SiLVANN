# SiLVANN

> **Developer preview, v0.9.0.** It runs the models below well on the hardware it was tested on, and it is shared
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
([`docs/users.md`](docs/users.md#4-the-models-settings)). **Every model and pack, the other machines (down to a 4 GB
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

**New in v0.9.0:** tool calling — a model opens a Lisp REPL on the machine it runs on, in its answer.

The model calls one tool, `lisp`, in the format it was trained on — Qwen's and GLM's `<tool_call>`, Ministral's
`[TOOL_CALLS]` — with a form to evaluate: the answer stops where the call ends, the form runs in a REPL beside the
server, and its value is read back in as the model's chat template frames a tool's answer, the answer going on from
there. `(help)` lists the commands: `ls`, `read`, `write`, `cd` and `pwd` for the folders; `bash`, a shell command in
the working folder (stopped, with what it started, after 60 seconds — up to 600 when the form asks — or after 1 MB of
output); `fetch`, a web page's words without its markup; and Lisp to put them together — `define`, `let`, `lambda`,
`map`, `filter`, text and lists. The UI shows a run of calls as one `<!tools>` block ([`docs/users.md`](docs/users.md#the-models-tool)).

What a command may do without asking you is the conversation's, chosen beside its **tools** switch: ask first — reads in
the workspace and in folders you allowed; reads unasked — every read, the default (a config's `tools`, `"chill"`); full
auto — everything. Anything else stops the answer with a question under it — allow the folder, allow once, or refuse —
and a refusal is said to the model, which answers without it. The switch works as the thinking switch does: off, the
model's calls are answered that the tool is off; a conversation's first message declares the tool, so one begun without
it does not take it on later. `"tools": "no"` in a config turns it off for every conversation on that server. The REPL
starts in `workspace/` beside the conversations (`tools_folder` names another) and runs as the server's user — in the
Docker image, the container's.

In a terminal, `silvann_cli` (`silvann_cli.py`, in the release from this one) has the conversation the UI has — over
ssh, say: the answer written as it comes, its thinking dimmed, the tool's calls and their values shown, and an ask
answered at a prompt — `y` once, `f` the folder (a read), `n` or Enter refuses. It takes the server's defaults unless
`--notools` or `--nothink` says otherwise, and `/mode` chooses what the tool may do
([`docs/users.md`](docs/users.md#3-serve-it-and-chat)).

Checked through the server's API on two MI50s: the 27B, the 35B-A3B, the 122B (`D4E4`), GLM 5.3 Flash and Ministral 3
14B (`D8`) each found a word in a note in its workspace through the tool; the 27B, the 35B-A3B, the 122B and GLM asked
before reading a file outside it and went on once allowed; the 27B told the user of a write refused, and the 122B's
`(bash "wc -l list.md")` ran once allowed. A call and its answer are framed as each model's chat template frames them,
to the token. The terminal's chat, on the 35B-A3B: a read outside the workspace unasked in "chill", a `bash` asked
and allowed with `y`, a read asked and refused with `n`, an answer stopped with Ctrl-C, and a conversation put away
after two idle minutes answering again. On an RTX 5060 Ti the same through the server for the 27B, the 35B-A3B, the
122B and GLM, and for the 27B and the 35B-A3B a `fetch` and a write asked too; the terminal's chat the same on the
35B-A3B; the UI's switch, its selector and the three answers to an ask clicked in a browser; and inside the Docker
image, the tool on the volume's `workspace/` and `docker exec -it silvann silvann_cli`.

⚠ The models have new names — `silvann_tq_<bits>_<model>`, the model's name in lowercase:
`silvann_tq_D8E4_qwen3.6_35b_a3b`, `silvann_tq_D4_qwen3.8_27b`, `silvann_tq_D4E4_glm_5.3_flash`,
`silvann_tq_D8_ministral_3_14b` and the rest (`silvann list`). A model downloaded with an earlier release keeps its
folder and its old name, and runs as before, its conversations with it; but `silvann list` and a download know only the
new names, and a later release's new program files reach a model under its new name alone — download it again under
that name when you can (the conversations stay with the old folder). The repositories on Hugging Face were renamed, and
their old names lead to the new ones, so an earlier release still downloads.

Also: two servers started on one machine boot one after the other, each trying the card as the other leaves it. A setup
kept in `autotune.json` is chosen again when the card holds 1 GB less than when it was chosen, and on an AMD card a kept
setup is taken at the next start again — the card's memory in use, which changes from one boot to the next, had kept it
from matching; one the card refuses is chosen again around the config's own settings. An integrated GPU
(`khronos_opencl1_igpu`) reads pictures on the CPU unless the config's `picture_pixels` gives them room in its pool. The
UI starts again the model it last started only when that start booted. An install's `.gitignore` leaves out the Hugging
Face token, the UI's last start and the tool's files.

Every release before this one, and this one's notes again: [`release_logs/v0.md`](release_logs/v0.md).

## Platforms

| | |
|---|---|
| **Tested** | Linux with AMD Instinct MI50 (gfx906), ROCm 6; Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595 — every pack but GLM's checked against Hugging Face, the 35B's tier of experts included; the CPU alone with AVX2 (Xeon E5-2680 v4) and with AVX-512 (Xeon W-2123); OpenCL 2.0 with sub-groups, on the MI50 — the 27B and the 35B checked against Hugging Face, the 27B answering at 9-11 tokens a second and reading a prompt at 49 positions a second; the 122B's tier of experts, 11.7 tokens a second with 18 GB of the card (9.7 without); OpenCL 1.2 through NVIDIA's OpenCL on the RTX 5060 Ti — the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs) |
| **Untested** | Windows — the Docker image ([`docker/README.md`](docker/README.md)), run so far on Linux only, or Linux booted from a USB SSD ([`docs/users.md`](docs/users.md#on-windows-the-docker-image)); other NVIDIA generations (the CUDA family is built for sm_75 to sm_120: Turing to Blackwell); GLM 5.3 Flash on an NVIDIA card; other AMD generations; OpenCL on other devices, the integrated GPUs `khronos_opencl1_igpu` is for among them; AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons) |
| **Not yet** | Apple silicon |

Pictures and videos are read by the Qwen models and GLM 5.3 Flash (v0.7.0: [`release_logs/v0.md`](release_logs/v0.md#v070)); Ministral 3 does not read them.
Every model calls the tool, a Lisp REPL on the machine, in its own trained format (v0.9.0: [`docs/users.md`](docs/users.md#the-models-tool)).

## Quick start

### Linux: the install script

One line in a terminal fetches SiLVANN into `./silvann` and builds it for this machine — the CPU, and the card it finds:
an NVIDIA card with CUDA 12's `nvcc`, an AMD card with ROCm 6's `hipcc`, each for that card alone (where the compiler is
missing it says where to get it, and the CPU runs the models meanwhile). It installs nothing system-wide and asks for no
password, and it leaves a `silvann` command. Run it again — or `silvann update` — to update: the models, the
conversations and their settings stay.

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
start_silvann_server.py, silvann_ui.py, silvann_cli.py     the server, the chat in a browser, the chat in a terminal
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
