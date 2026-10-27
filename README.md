# SiLVANN

> **Developer preview, v0.21.0.** It runs the models below well on the hardware it was tested on, and it is shared
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

**New in v0.21.0:** SiLVANN servers find each other through a registry for the model's subagents, a long conversation
on an MI50 reads and answers two to three times as fast, `experts_register` no longer pins the experts' RAM by itself,
and the container docs give each app a name under localhost.

**Servers that find each other.** `silvann registry` is a small registry that the SiLVANN servers of a network offer
themselves to — each its address, its model, the model's LoRAs and context — and that a server's `(whoami)` and
`(agent …)` read: the model sees each other server's model as the call that starts it, and its subagent runs there. A
server renews its offer while it runs and withdraws it as it stops; one that stops without withdrawing leaves the
listing 90 seconds later. The tools file's `agent` entry names the registry and its key, which the model never sees
([`docs/tools.md`](docs/tools.md#servers-found-through-a-registry),
[`docker/README.md`](docker/README.md#servers-that-find-each-other)).

**A long conversation on an MI50.** Past the 4,352 positions a Qwen model's cache keeps at full precision, an MI50 now
reads the rest a slice at a time across the whole card: Qwen 3.8 27B reads a 32,098-position prompt at 69 positions a
second where it read 25, and answers after it at 21 tokens a second where it answered at 10; after 100,081 positions, at
35 and 13. Qwen 3.6 35B-A3B, on the card whole, reads a 32,033-position prompt at 132 where it read 42, and answers after
it at 55 where it answered at 18 ([`docs/results.md`](docs/results.md)).

**A tier's prompt, faster on NVIDIA.** With a tier of experts on the card fed from the experts in RAM, `"auto"`
registered that RAM with an NVIDIA card so the card copied each expert from where it lies. Measured again on an RTX
5060 Ti it made the prompt slower — Qwen 3.5 122B-A10B's by 25%, Qwen 3.6 35B-A3B's by 9%, the answers the same — and on
AMD's cards it bought nothing. `"auto"` now registers nowhere, and `"experts_register": true` still does
([`docs/config.md`](docs/config.md)). On the 5060 Ti the 35B with an 11.75 GB tier reads a 2,035-position prompt at
194-203 positions a second where it read 181; as v0.21.0 installs, 194, and the answer after it 59.4 tokens a second
against 60.9 — in the chat the same 72.

**Each app a name.** The apps in a pod — the chat, SearXNG, any other — can each have a name under localhost behind one
small proxy on one port: `silvann.localhost:8000` the chat, `searxng.localhost:8000` the search, `localhost:8000` the
list of them. Their ports stop colliding, the address says what it opens, and the browser keeps each app's cookies and
stored data apart ([`docker/README.md`](docker/README.md#each-app-a-name-under-localhost)).

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

### On Linux: a container (recommended)

The images hold the release built for a card — AMD's (gfx906: the Instinct MI50 and MI60) or NVIDIA's — and the CPU's
family, run rootless under Podman: the model's tool reaches the container and the folder you give it, and nothing else
of the machine ([`docs/local_ai.md`](docs/local_ai.md#container-first)). That folder, `~/SiLVANN` below, is yours on
both sides: the models, the conversations, and the model's workspace, where you put what it is to read and find what it
made. The chat is at http://localhost:8080, and its Options download a model and start it:

```bash
mkdir -p ~/SiLVANN ~/.config/containers
python3 -c "import json, sys; p = json.load(open('/usr/share/containers/seccomp.json')); p['syscalls'].append({'names': \
  ['landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self'], 'action': 'SCMP_ACT_ALLOW'}); \
  json.dump(p, open(sys.argv[1], 'w'))" ~/.config/containers/silvann-seccomp.json   # Podman's, with Landlock's calls
podman run -d --name silvann --userns=keep-id --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd      # an AMD card
podman run -d --name silvann --userns=keep-id --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --device nvidia.com/gpu=all \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-nvidia   # an NVIDIA card
```

The profile lets the model's commands be closed off from the machine's tools file — its keys — inside the container
([`docs/tools.md`](docs/tools.md#where-the-models-commands-reach)); without it, Ubuntu 22.04's Podman refuses the calls
that do it, and the model has no shell. For the model's web search, a SearXNG in a container beside it:
[`docker/README.md`](docker/README.md#searching-the-web-a-searxng-beside-it).

What the host needs (the card's driver, Podman with `crun`, your groups), Docker instead of Podman, and updating:
[`docker/README.md`](docker/README.md). A container under Docker Desktop on Windows runs the models well below what the
PC gives: on Windows, install it.

### On Windows: `install.ps1`, with the libraries GitHub built

Python 3.10 to 3.14, Git and the card's own driver (`winget install -e --id Python.Python.3.12`, `winget install -e --id
Git.Git`). In PowerShell, in the folder SiLVANN should go under:

```powershell
curl.exe -fsSLO https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1      # into .\silvann, and a `silvann` command
```

Then, in a new PowerShell — the one the installer ran in does not have `silvann` on its PATH yet:

```powershell
silvann data E:\silvann_data                   # if you like: the models and the conversations on another drive
silvann download silvann_tq_D4_ministral_3_14b
silvann serve silvann_tq_D4_ministral_3_14b     # on port 8765
silvann ui                                     # a chat in the browser, at http://localhost:8090
```

The installer takes the libraries GitHub built for the release when they were built from the folder's engine, and builds
them otherwise. `build.ps1`, in the folder, builds them on the PC: it needs Visual Studio's Build Tools with their C++
Clang tools, and CUDA's toolkit (12 or 13) for an NVIDIA card. Run `install.ps1` again — or `silvann update` — to
update: the models, the conversations and their settings stay ([`docs/users.md`](docs/users.md#on-windows)).

On Windows the model's shell — `(powershell …)`, and Git's `(bash …)` — and its adding tools of its own start switched
off: nothing holds them there, so they are yours to switch on, in the data folder's `tools\default.json`
([`docs/tools.md`](docs/tools.md#where-the-models-commands-reach)).

### From source: the installers

On Linux one line in a terminal fetches SiLVANN into `./silvann` and builds it for this machine — the CPU, and the card
it finds: an NVIDIA card with CUDA 12's `nvcc`, an AMD card with ROCm 6's `hipcc`, each for that card alone (where the
compiler is missing it says where to get it, and the CPU runs the models meanwhile). On Windows, `build.ps1` builds the
libraries on the PC (above). `install.sh` installs nothing system-wide and asks for no password, and it leaves a `silvann` command (`silvann help` lists what it does). Run it again — or `silvann update` —
to update: the models, the conversations and their settings stay. `silvann data <folder>` keeps the models, the
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
