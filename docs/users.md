# Running SiLVANN

This is a **developer preview (v0.11.0)**. It is tested on Linux with AMD Instinct MI50 cards (gfx906, ROCm 6), with an
NVIDIA GeForce RTX 5060 Ti (sm_120, CUDA 12.9), and on the CPU alone with AVX2 and with AVX-512 — on Ubuntu 22.04. On a
Windows PC, boot Linux from an SSD (below); the Docker image runs as fast as the install on Linux, but well below Linux
on the same PC under Windows, so there it is for evaluating SiLVANN. Windows itself and other NVIDIA generations are
untested, Apple silicon is not supported yet. If something does not build or does not run on your
machine, you are expected to be able to read the error and the source.

## 1. Build

**The quickest way** — in a terminal, in the folder you want SiLVANN under:

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
```

It fetches the release into `./silvann`, makes its Python environment, finds the card and its compiler, builds the
engine and the families this machine needs (the whole build's output in `silvann/build.log`), and puts a `silvann`
command in `~/.local/bin`: `silvann list`, `silvann download <model>`, `silvann serve <model>`, `silvann ui`, and
`silvann update` for the next release (stop the server first); `silvann help` lists the rest.

`silvann models <folder>` keeps what SiLVANN keeps for you in that folder from then on: SiLVANN on a USB drive, say, with
its models on the machine's own disk. That is the models (`models/`), the conversations (`user_sessions/`), and beside
them your skills (`skills/`), the tool's workspace, the folders it was allowed and the definitions it kept
(`workspace/`, `tools.json`, `tools_library.json`), the Hugging Face token (`.huggingface_token`) and the UI's last start
(`ui_last_start.json`). What the old folder holds stays there; the command names what to move across. The `silvann`
command's `serve`, `ui` and `download` read the folder; the server and the UI started by hand read `SILVANN_DATA`, which
also comes before it. It installs nothing system-wide and asks for no
password: a tool it needs and does not find — Python's `venv`, `g++`, `git` — it names, with the `apt` line that installs
it. A card whose compiler is missing (CUDA's `nvcc`, ROCm's `hipcc`) is named too; install it, then `silvann build`.

**By hand:** you need Python 3.10 or newer, `g++`, for AMD cards ROCm's `hipcc` (`/opt/rocm/bin/hipcc`), and for NVIDIA cards CUDA
12's `nvcc` — found on the `PATH`, or named by `SILVANN_NVCC=/usr/local/cuda-12.9/bin/nvcc`.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
SILVANN_PYTHON=venv/bin/python ./build.sh
```

`build.sh` compiles the evaluator and one library per silicon family whose toolchain it finds, into `lib/`. A
family whose toolchain is missing is skipped and named. `SILVANN_HIP_ARCHES=gfx906` builds for one AMD card
generation only, which is much faster than building for every one, and `SILVANN_CUDA_ARCHES=120` the same for NVIDIA
(the number is the card's compute capability: 120 for the RTX 50 series, 89 for the 40, 86 for the 30). A card is given
to the family that includes it when the server starts; a model's config only picks which card (`card`, in [`config.md`](config.md)).

### On Windows: the Docker image

⚠ **On Windows the Docker image is for evaluating SiLVANN: it runs well below what the same PC gives under Linux** (on
Linux the image runs as fast as the install, below). On one PC (an RTX 5070 Ti, a Ryzen 7 5800X, 32 GB of RAM), Docker Desktop answered at 37 tokens a second
with Ministral 3 14B `D4`, where Linux gives 46, and at 22 with the Qwen 3.6 35B-A3B `D8E4`, its experts in RAM and a
tier of them on the card, where Linux gives 40 — the model that goes between the card and the CPU at every layer loses
the most. To use SiLVANN, run Linux on an SSD, [below](#on-windows-linux-on-a-usb-ssd).

SiLVANN runs on Linux, and on Windows it runs in Linux through WSL 2: the Docker image holds the release, its
environment and its build for an NVIDIA card or the CPU, and Docker Desktop runs it, with the chat in the browser. The
models are downloaded from the UI into the image's volume. [`docker/README.md`](../docker/README.md) has the steps and
the scripts: `silvann_install`, `silvann_start`, `silvann_stop`, and `silvann_export` / `silvann_import` to carry it,
models and all, to another drive or PC. It has been run on Linux (Docker, an RTX 5060 Ti: the 27B and the 35B against
Hugging Face as outside it, at the same speed) and on two Windows PCs with NVIDIA cards: on Windows 11 well; on
Windows 10 the card's shared memory misled the setup's choice of where the experts go — name it, as below.

**A model larger than the card, on Windows.** NVIDIA's Windows driver lets a program use more memory than the card has,
the rest in the PC's RAM ("shared GPU memory" in Task Manager), so a model too large for the card starts on it all the
same and then reads that rest over PCIe at every token. On one PC (an RTX 4060 Ti 16 GB) the Qwen 3.6 35B-A3B `D8E4`
started with all its experts on the card, 4 GB of them in shared memory, and answered at 6 tokens a second. The CUDA family now
refuses an allocation past the card's free memory, which has not been checked on Windows yet; until it is, name the placement in the model's config (the Options view): the experts on the CPU, and a tier
of them on the card that leaves Task Manager's shared GPU memory near 0 —
`"options": {"kv": "tiered", "tiers": [256, 4096, "rest"], "experts_on": "cpu", "experts_from": "ram",
"card_experts_gb": 6}` answered at 16 tokens a second there. The driver's "CUDA - Sysmem Fallback Policy" set to
"Prefer No Sysmem Fallback" left CUDA in WSL 2 without the card on that PC: keep it at its default.

### On Windows: Linux on a USB SSD

SiLVANN runs on Linux: it loads its libraries, pins its threads and reads the experts from the disk through Linux's own
calls, so it does not run as a Windows program or in PowerShell. **The path we support is Linux installed on a USB SSD**
— a real installation, not a live session, which forgets the card's driver and the build at every shutdown — booted
from the PC's boot menu, the Windows disk untouched:

1. **A drive:** an NVMe SSD in a USB-C enclosure (a USB stick is too slow to load models from), or one in the PC beside
   Windows', large enough for the models you want — 7 GB to 200 GB each — and about 40 GB for the system.
2. **Ubuntu 22.04 LTS**, the system SiLVANN is tested on: write its installer to a second USB stick (balenaEtcher or
   Rufus), boot from it, and install onto the SSD — choosing the SSD as the installation disk, and its boot loader.
3. **The card's driver:** NVIDIA's from Ubuntu's *Additional Drivers* (570 or newer for an RTX 50 card), then CUDA 12's
   toolkit from developer.nvidia.com/cuda-downloads; for an MI50, ROCm 6.1.3 with `amdgpu-dkms` (and the README's note
   on kernel updates). The CPU alone needs neither.
4. **SiLVANN:** the one line above.

**WSL2 — possible, and untested.** WSL2, the Linux that runs inside Windows 10 (21H2 or later) and 11, should run it on
an NVIDIA card or the CPU alone: `wsl --install` in PowerShell as administrator; NVIDIA's Windows driver (no Linux
driver inside WSL) and CUDA's toolkit from NVIDIA's WSL-Ubuntu page; the installer, run in WSL's own files (`~`, not
`/mnt/c`, which is many times slower to read models from); `.wslconfig`'s `memory=` raised past the default half of the
machine's RAM for a model with its experts in RAM; the UI at `http://localhost:8090` in a Windows browser. Nobody has
tried it yet: tell us in Discussions if you do. ROCm inside WSL supports recent Radeon cards only, not the MI50.

## 2. Get a model

```bash
venv/bin/python silvann_core.py list-models
venv/bin/python silvann_core.py download-model silvann_tq_D8E4_qwen3.6_35b_a3b
```

A model lands in `models/<name>/` with a `configs/default.json` and its program in `lisp/` — the Lisp the evaluator
runs it with, yours to read and change. `silvann download <model>` on a model that is there fetches its pack's `lisp/`
afresh, whole — the program its pack carries now, over any change of yours. A server started on a model that is not there asks, on a terminal, whether to
download it; started without one (a service, the UI), it refuses and names this command.

The UI's Options download a model too, with its progress. Hugging Face serves a client that is not signed in more slowly:
a token with read access (huggingface.co/settings/tokens) saved with `venv/bin/python silvann_core.py hf-token --set`
(`silvann token --set`), or in the UI's Options, is used by every download after; `HF_TOKEN` in the environment comes
first.

## 3. Serve it, and chat

```bash
venv/bin/python start_silvann_server.py --model models/silvann_tq_D8E4_qwen3.6_35b_a3b [--port 8765]
venv/bin/python silvann_ui.py --port 8090
venv/bin/python silvann_cli.py [--notools] [--nothink]     # or the chat in this terminal
```

**In a terminal**, `silvann_cli` (`silvann_cli.py`) talks to the server as the UI does — over ssh, say: the answer
written as it comes, its thinking dimmed, the model's tool calls and their values shown. A new conversation takes the
server's defaults, its thinking and its tool in the config's mode, unless `--notools` or `--nothink` says otherwise; an
ask stops the answer at a prompt — `y` allows it once, `f` the folder (a read), `n` or Enter refuses. In the
conversation: `/new`, `/sys <prompt>` (a new conversation with it), `/mode ask|chill|full-auto`, `/quit`; Ctrl-C stops
an answer. `--endpoint` names another server (`http://localhost:8765` by default).

**In the browser** ([`ui.md`](ui.md) goes through the UI, view by view; it is the first document in its Help), a model
with no conversation saved opens one as its server starts, and a message typed brings the
open conversation online, as **Activate** does. While the model thinks the answer says "thinking…"; **show thinking**
shows the thought itself, in a section of its own. A run of the model's tool calls is a section, "tool · 3 calls", closed
until you open it. The top bar shows how full the conversation is: its tokens of the most the model takes (its
`max_context`), as "4,210 / 32,768" beside a bar that turns amber at three quarters and red at nine tenths.

**Pictures** (the Qwen models): the UI's 📎 button attaches one to the next message — it is read when it is attached,
and the message carries it; the server reads one at `POST /cartridges/<id>/images` and a message names it in `images`.
The model's vision tower runs on the card: a picture is made at most 4 megapixels (`picture_pixels`; its tokens: a
1024 x 1024 picture is 1,024, a 2048 x 2048 one 4,096), and takes from a tenth of a second (a small one) to about 6.5 s (the
largest) on the test machine's RTX 5060 Ti, and from 0.13 s to about 8.5 s on an MI50 — every patch of a picture attends
every other, so its time grows with the
square of its size. A folder fetched before
v0.7.0 lacks `lisp/qwen3_5_pictures.lisp` and refuses pictures until `silvann download <model>` fetches it.

**GLM 5.3 Flash reads pictures and videos** the same way: its own tower (24 blocks) runs on
the card, a picture at most 4 megapixels by default (`picture_pixels`; 21,396 patches of 14 x 14) — 0.25 s for 448 x 448
and 12 s for 2044 x 2044 on an MI50.

**Videos**, for the Qwen models and GLM, are attached with the same button: an animated GIF, WebP or PNG is read as it
is, any other video file (MP4 and the like) through PyAV, which `install.sh` installs from `requirements-nonfree.txt`
(the Docker image leaves it out, and its scripts' `install_nonfree` adds it). Its frames are taken
at 2 a second, two frames read together as one picture, each pair's time written into the message as the model's own
processor writes it (Qwen's before the pair, the middle of its two frames: "<1.2 seconds>"; GLM's after it, its first
frame's: "1.0 seconds"); a video is made at most half the conversation's context in tokens, the Qwen models also keep
it within 25 megapixels over all its frames, as their processor does — and every video within the config's
`max_tokens_per_video` (8,192 by default, below).

**Files**, for every model: a Word document (`.docx`), a PDF (its text, through pypdf, which `install.sh` installs) or
a text file (Markdown, CSV, JSON, code) attached with the same button is read when it is attached, and its text goes
into the message before what you type, as `<file name="…">…</file>` — cut, with a note, to what the conversation's
context still holds. The server takes them at the same `POST /cartridges/<id>/images` (it answers `kind: "document"`),
and `/health` says `files`.

The UI is a web page on port 8090 (`silvann ui`; `silvann_ui.py` alone takes 8080 unless `--port` says) that talks to the server on port 8765; it can also start a server itself (Options). Conversations are saved under `user_sessions/` and
can be exported and imported. One conversation runs at a time; a second request while one is running is
answered `503` with `Retry-After: 5`, and the UI retries. The card holds one conversation's cache: when another takes
the card, the one on it is kept in RAM — or, when RAM has no room for two conversations at the full context beside 500
MB, put away to its folder, as one is when the UI or client that opened it has been gone for two minutes, and
**Activate** brings it back.

On a machine with no card the model runs on the CPU's own cores (the `x86_avx2` family, which needs AVX2, FMA and F16C:
Intel from Haswell, 2013, AMD from Excavator, 2015; an older processor has no family to run on). The pack made for
that is `silvann_tq_D4E4_qwen3.6_35b_a3b`: its dense part at 4 bits, about 15 tokens a second on the test machine's
two Xeons. On a processor with AVX-512 (F, BW, VL and DQ) the family's matrix products use it: on a four-core Xeon
W-2123 with two DDR4-2400 sticks the 35B D8E4 alone ran at 9.0 tokens a second against 1.7 with AVX2, and with its
experts on the CPU beside a card at 27-29 against 15-17. `SILVANN_CPU_AVX512=0` keeps the AVX2 code.

### The model's tool

A conversation has the model's tool unless its **tools** switch is off, or the config says `"tools": "no"`: the model
can open a Lisp REPL on the server's machine in its answer: it sends one form at a time — `(help)` lists the commands —
and reads each value back before it writes on. The commands list folders, read files and write them, run a shell
command and read a web page: `(ls "notes")`, `(read "notes/a.md" 140 20)`, `(write "out.txt" "…")`,
`(bash "git log -3")` (in the working folder, stopped after 60 seconds or `(bash "…" 300)`), `(fetch "https://…")` (the
page's words, without its markup), `(cd "…")`, `(pwd)`; and Lisp to put them together — `define`, `let`, `lambda`,
`map`, `filter`, text and lists. The names other Lisps and Python give these are the same commands, asked the same
way: `read-file`, `slurp-file`, `read-all`, `cat` and `slurp` are `read`; `system`, `shell` and `sh` are `bash`. The UI shows a run of calls as one section, which opens to:
```
<!tools>
> (ls "notes")
("secret.txt")
> (read "notes/secret.txt")
"The secret word is PERIWINKLE.\n"
</tools>
```
The REPL starts in the workspace folder (`tools_folder`). What its commands may do without asking you is chosen beside
the tools switch, for that conversation — a new one starts with the config's `tools`. The folders you allowed are
every conversation's, kept in `tools.json` beside the conversations.

Each conversation's REPL keeps what the model defines — its names, its functions, its working folder — from one call to
the next and through the conversation, until `(reset)` (or `(exit)`) starts it afresh; a server's restart starts every
conversation's REPL afresh too. What is outside the REPL stays through a reset: the files, the folders allowed, the
conversation's mode, and the library — `(keep 'name "what it does")` puts a definition the model made in
`tools_library.json` beside the conversations, `(library)` lists what is kept, and `(import 'name)` brings one into any
conversation's REPL, the kept names it uses first; `(forget 'name)` takes one out. It is plain JSON, a definition's
source as the model wrote it: yours to read, edit or share.

| the UI's setting (`tools`) | reads | writes, commands, web pages |
|---|---|---|
| ask first (`"ask"`) | in the workspace, and in folders you allowed: unasked; elsewhere asked — allow the folder, allow once, or refuse | asked — allow once, or refuse |
| reads unasked (`"chill"`, the default) | unasked, anywhere the server's user can read | asked |
| full auto (`"full-auto"`) | unasked | unasked |

An ask stops the answer: its buttons are under it, and the answer goes on once you choose — a refusal is said to the
model, which answers without it. The switch works as the thinking switch does, between messages: turned off, the
model's calls are answered that the tool is off; turned on, only in a conversation that began with it, as the tool is
declared in a conversation's first message. ⚠ The REPL runs as the server's user, with its files and its network: in a
Docker container, the container's. A command allowed once runs as written — read it before you allow it; "full auto"
runs every command the model writes.

### A picture from the model's tool

With a picture model downloaded (Qwen-Image 2.1, below), a text model's REPL has `(image "a red apple on a table")`.
Ask the model for a picture and it calls it. The picture appears in the conversation, and the model says what it
made.

On one card, the text model is put down while the picture is drawn and brought back after, its conversation where it
was. A free card can draw instead, with `image_card` in the config. How, and what it costs:
[`config.md`](config.md#a-picture-from-the-text-model).

### Skills

A skill gives the model's REPL commands of yours, written in Python or as a program: a folder with a `SKILL.md` that
declares them, in `skills/` beside the conversations (the Docker image's `/data/skills`). The release ships `files`:
`(find "folder" "pattern")`. `(skills)` lists them. **How to write one, step by step, and how values cross between Lisp
and Python: [`tools.md`](tools.md).**

### A picture model: Qwen-Image 2.1

`silvann_tq_D8_qwen_image_2.1` makes pictures. Started as a server's model, each message is a prompt and its answer a
picture:

```bash
venv/bin/python silvann_core.py download-model silvann_tq_D8_qwen_image_2.1
venv/bin/python start_silvann_server.py --model models/silvann_tq_D8_qwen_image_2.1
```

In the UI a message typed is a prompt: beside the box the picture's progress — the prompt read, each step, the decode —
and then the picture in the conversation, kept as a PNG in the conversation's `pictures/` folder. A message sets its own
picture's size, steps and seed — `A lighthouse at dusk, oil painting --size 768x1024 --steps 20 --seed 7` — and
otherwise takes the config's `image` settings: 1,024 x 1,024, 40 steps, a new seed each picture, which the answer says
under it. Sides are made multiples of 32, from 256 to 2,048; steps from 1 to 100, fewer faster and coarser. `card` in
the config picks which of the machine's cards it runs on (`"auto"`: the first). The model has no thinking and no tool,
and the UI hides those switches for it; reference pictures (the 📎 button) come with a later release.
`silvann_cli` shows the progress on one line and saves the picture in its working folder, saying where. Through the API
the answer's text names the picture — `![picture](picture:NAME)` — served at `GET /cartridges/<id>/pictures/NAME` (a
PNG; `?format=jpeg` for a JPEG), and the message's stream tells its stages as `progress` events.

The card holds one of the model's parts at a time — the text encoder to read the prompt, the transformer for the
steps, then the picture decoder. A 1,024 x 1,024 picture at 40 steps took 146 s on the test machine's RTX 5060 Ti
16 GB (3 s a step) and 572 s on an MI50; a 512 x 512 one at 12 steps, 35 s in the UI. Checked against diffusers'
pipeline from the same prompt and seed: the pictures are nearer its float32 picture than its own bfloat16 pipeline's.
⚠ Qwen-Image 2.1 is under the Qwen Research License Agreement, in the model's folder: research and other
non-commercial use only.

## 4. The model's settings

`models/<name>/configs/default.json` is read when the server starts; another file can be named as the server's
first argument. The UI's Options edit it, and their **New** starts a file with every key the server reads.

**Every key, every model's options, what the config leaves out and how it is chosen for the machine, and tensor
parallelism: [`config.md`](config.md).**

### The card's tier of experts

With the experts on the CPU, the card's tier (`card_experts`, on by default; `card_experts_gb` a size of your own) gives
the card's memory the context leaves to the experts used most, and
the card computes those while the CPU computes the rest. It starts empty and fills as the model is used — the first
prompt fills most of it. An expert moves to the card when it has been used three times more recently than the one
it would replace; the CPU copies it over while the card computes, so a token never waits on one. In a prompt, the
experts too few positions pick stay on the CPU; how many is worked out at boot from the card and the CPU's speeds.
Give it the card's memory the model's dense part and its cache leave: on our MI50 GLM 5.3 Flash went from 6.8 to 7.5
tokens a second with a 300,000-position cache and a 6 GB tier in 16 GB, and to 8.1 with 18 GB on a 32 GB card; the
122B went from 10.9 to 19.9 with 24 GB. [`results.md`](results.md) has every run.

**The context and the tier share the card** (the Qwen MoE models and GLM 5.3 Flash): with `max_context` `"auto"`, every
model's default, the cache is sized for the model's whole length where the card holds it, and the tier holds experts in the cache's
rows no conversation has reached yet — on an RTX 5060 Ti 16 GB the 35B-A3B lends its tier 2.4 GB that way at 262,144
positions, and on a 32 GB MI50 GLM 11 GB at 1,048,576. As a conversation
grows the tier gives them back 8,192 positions at a time, before the conversation writes there, and a shorter
conversation or a new one has them lent again: nothing to size, and no context to choose
([`config.md`](config.md#the-context-and-the-tier-share-the-card)). A config written before v0.11.0 keeps the
`max_context` it names; set it to `"auto"` in the Options view to have this.

With `card_exclusive: true` (GLM 5.3 Flash, experts in RAM), an expert is on the card or in RAM, never both: the card
starts with its share of every layer, and each expert it takes in sends one back, written into the RAM the newcomer
left. The memory the tier holds is RAM given back — about 18 GB for an 18 GB tier — so a box short of RAM by about the
tier's size runs the model without the disk. It costs speed where a card cannot copy both ways at once: on our MI50 a
token went from 107 to 138 ms. The engine measures that at boot and, on such a card, sends an expert back only after
the one coming in has arrived.

## 5. The experts on the disk

With `experts_from: "disk"`, the first start writes the model's experts to `models/<name>/experts/` in the layout
the CPU reads them in — about 150 GB for GLM 5.3 Flash, 55 GB for the 122B, 16 GB for the 35B — and later starts
map that file. The
disk should be an NVMe drive: RAM keeps the experts that are used, and every other one is read from the disk
when a token picks it, so the disk's speed is the answer's speed. On our test machine GLM 5.3 Flash answered at
5.7 tokens a second with 128 GB of RAM from a disk that reads 5 GB/s. Keep the disk cool: a hot NVMe drive slows itself down. Every measurement: [`results.md`](results.md). The engine keeps as many experts
as the memory there is allows — the process's own limit when it has one — less a quarter of it for everything else,
at least 3 GB and at most 8 GB.

## 6. A model over several machines

A model can run over several machines in a ring, each holding a run of its layers — two gaming PCs with 96 GB of RAM
each, say, holding half of GLM's experts apiece (`silvann_runtime/pipeline.py`). Every machine has the model and starts
its server the same way; its config (`configs/default.json`) says its part:

| setting | |
|---|---|
| `pipeline_parallelism` | `"no"` (the default) or `"yes"` |
| `pipeline_parallelism_layers` | this machine's run of the layers, first-last — GLM has 45 (0-44), the 122B 48, the 35B 40, the 27B 64 |
| `pipeline_parallelism_next` | the machine after this one, `host:port`; the last machine names the first |
| `pipeline_parallelism_port` | where this machine listens for the one before it, 8766 by default |
| `max_context` | the same number on every machine (not `"auto"`) |
| `card` | which of the machine's cards to run on (`0`, `1`, …), `"auto"` the first, `"cpu"` none — its CPU and RAM. Two cards in one PC are two machines of the ring, on ports of their own; so are a card and the CPU beside it. Two on one card boot one after the other, and the first takes what its setup chooses of it: give each its share, `card_experts_gb` in its options |

The machine whose layers begin at 0 is the first: it answers the UI and holds the conversations. The one whose layers end
at the model's last runs its head, draws the next token there, and hands it back to the first. Each machine chooses its
own setup for its layers (▶ section 4) — where its experts sit, its card's tier. For GLM over two machines:
```
  PC A (192.0.2.2):  "pipeline_parallelism": "yes", "pipeline_parallelism_layers": "0-22",  "pipeline_parallelism_next": "192.0.2.3:8766"
  PC B (192.0.2.3):  "pipeline_parallelism": "yes", "pipeline_parallelism_layers": "23-44", "pipeline_parallelism_next": "192.0.2.2:8766"
```
Start both servers (in either order: each waits for the other); the UI talks to PC A. What crosses the network is one row
of numbers a position (GLM's 32 KB) — a gigabit link is enough. Text, files and GLM's pictures and videos go round the
ring; a Qwen model's pictures are read on one machine only, for now. A conversation's caches are spread over the
machines, so one taken up again after another ran reads its tokens again.
