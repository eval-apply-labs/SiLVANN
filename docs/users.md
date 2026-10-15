# Running SiLVANN

This is a **developer preview (v0.7.0)**. It is tested on Linux with AMD Instinct MI50 cards (gfx906, ROCm 6), with an
NVIDIA GeForce RTX 5060 Ti (sm_120, CUDA 12.9), and on the CPU alone with AVX2 and with AVX-512 — on Ubuntu 22.04. On a
Windows PC, run the Docker image or boot Linux from a USB SSD (below); Windows itself and other NVIDIA generations are
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
`silvann update` for the next release (stop the server first). It installs nothing system-wide and asks for no
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
to the family that includes it when the server starts — nothing in a model's config names the card.

### On Windows: the Docker image

SiLVANN runs on Linux, and on Windows it runs in Linux through WSL 2: the Docker image holds the release, its
environment and its build for an NVIDIA card or the CPU, and Docker Desktop runs it, with the chat in the browser. The
models are downloaded from the UI into the image's volume. [`docker/README.md`](../docker/README.md) has the steps and
the scripts: `silvann_install`, `silvann_start`, `silvann_stop`, and `silvann_export` / `silvann_import` to carry it,
models and all, to another drive or PC. It has been run on Linux (Docker, an RTX 5060 Ti: the 27B and the 35B against
Hugging Face as outside it, at the same speed) and not yet on Windows.

### On Windows: Linux on a USB SSD

SiLVANN runs on Linux: it loads its libraries, pins its threads and reads the experts from the disk through Linux's own
calls, so it does not run as a Windows program or in PowerShell. **The path we support is Linux installed on a USB SSD**
— a real installation, not a live session, which forgets the card's driver and the build at every shutdown — booted
from the PC's boot menu, the Windows disk untouched:

1. **A drive:** an NVMe SSD in a USB-C enclosure (a USB stick is too slow to load models from), large enough for the
   models you want — 7 GB to 200 GB each — and about 40 GB for the system.
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
runs it with, yours to read and change. The server refuses a model that is not there and tells you this command.

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

**Pictures** (the Qwen models): the UI's 📎 button attaches one to the next message — it is read when it is attached,
and the message carries it; the server reads one at `POST /cartridges/<id>/images` and a message names it in `images`.
The model's vision tower runs on the card: a picture is made at most 4 megapixels (`picture_pixels`; its tokens: a
1024 x 1024 picture is 1,024, a 2048 x 2048 one 4,096), and takes from a quarter of a second (a small one) to about 15 s (the
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

The UI is a web page on port 8090 that talks to the server on port 8765; it can also start a server itself (Options). Conversations are saved under `user_sessions/` and
can be exported and imported. One conversation runs at a time; a second request while one is running is
answered `503` with `Retry-After: 5`, and the UI retries. The card holds one conversation's cache: when another takes
the card, the one on it is kept in RAM — or, when RAM has no room for two conversations at the full context beside 500
MB, put away to its folder, as one left idle for two minutes is, and **Activate** brings it back.

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
`map`, `filter`, text and lists. The UI shows a run of calls as one block:
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

## 4. The model's settings

`models/<name>/configs/default.json` is read when the server starts; another file can be named as the server's
first argument.

```json
{
  "_note": "… every 100 MB of the card's memory holds about 10000 positions of it …",
  "max_context": 32768,
  "lora": null,
  "thinking": true,
  "sampler": "generation",
  "options": { ... }
}
```

| key | what it sets |
|---|---|
| `max_context` | the longest conversation, in positions, 32,768 by default (the 27B's config: `"auto"`); the cache is sized for it at boot, and the `_note` beside it says how many positions 100 MB of the card holds for this model — lower it if the model does not fit on your card, raise it if there is room. `"auto"` (or `"max"`): as many as the card holds beside the model, up to the model's own length (262,144 for the Qwen models and Ministral, 1,048,576 for GLM) — the card's tier of experts then gets only what is left once the context is that whole length |
| `max_tokens_per_video` | the most tokens a video attached to a message becomes, 8,192 by default: its frames share them, so a longer video is seen as smaller pictures of each frame; more sees it in more detail and takes as many positions of the conversation. Reading a video holds 57-74 KB of RAM a token, depending on the model — its patches twice and its row (0.45 GB at 8,192 for GLM, 0.58 for the 27B) — and the server counts it when it chooses whether the experts fit in RAM |
| `thinking` | whether the model reasons before it answers, by default |
| `tools` | `"chill"` (default), `"ask"` or `"full-auto"`: what the model's tool, a Lisp REPL on the server's machine, may do without asking you in a new conversation — the UI chooses a conversation's own (section 3, the model's tool); `"no"`: no conversation has the tool |
| `tools_folder` | where the model's REPL starts and reads unasked; by default `workspace/` beside the conversations |
| `lora` | an adapter from the model's folder to load at boot, or `null` |
| `sampler` | how the next token is chosen: `"generation"` (the model's own `generation_config.json`), `null` (the most likely, greedy), or `{"temperature", "top_k", "top_p", "repetition_penalty"}`. It is drawn on the card — only the token comes back — when `temperature` is above 0 and `top_k` 1 to 64, and on the host over the logits otherwise |
| `options` | handed to the model's runtime, below |

**What the config leaves out is chosen for the machine** (`runtime/silvann_runtime/autotune.py`). A model whose
`options` name no placement — `experts_on`, `experts_from`, `card_experts` — has it tried on the card when the server
starts, fastest first: the experts on the card, if it holds them; else on the CPU — in RAM when they fit in what the
server may hold with about 3 GB to spare — the server's own 2 GB, what reading a video holds (`max_tokens_per_video`,
below: 0.45-0.58 GB at its 8,192) and 500 MB more (a container's or a capped service's memory
limit counts), else read from the disk — and, with them in RAM, the largest tier of the card it boots with beside the
context's cache, in steps of 256 MB, one step left free. The cache for the whole context comes first: a tier a little
smaller costs a little speed, a cache too small refuses a conversation. If nothing boots, the context is halved, down
to 4,096. A trial the card cannot hold is refused at once, so the trials take seconds. The
server's log says what was chosen and why; the choice is kept in the model's folder, `autotune.json`, and a start on
the same machine with the same config takes it without trying (delete the file to choose again); a card that holds
1 GB less than when the choice was made — another program, or another server, held it then — has it chosen again. Two
servers started on one machine boot one after the other, each trying the card as the other leaves it. What the config names
of the placement is kept and the rest is chosen around it — `"auto"` names nothing, so `{"experts_on": "cpu",
"experts_from": "auto"}` has where from and the tier chosen, and `"card_experts": "no"` asks for no tier — and its other
options are kept in every trial. A machine with no card of its own (the CPU alone, an integrated GPU) is not tried.

The Qwen models' `options`:

| option | values | |
|---|---|---|
| `kv` | `"tiered"` (default), `"fp16"` | the conversation's cache: tiered keeps the first 256 and the latest positions in halves and the rest at 8 bits |
| `tiers` | `[sink, hot, warm]` | the tiers' sizes; `warm` may be `"rest"`. `[256, 4096, "rest"]` is the default; `[256, 4096, N]` puts what is older than `N` at 4 bits, for small cards |
| `chunk` | positions, 1 to 256, default 256 | how many positions of a prompt go through a layer at once |
| `experts_on` | `"card"`, `"cpu"` | the routed experts on the card, or on the CPU — the setup for a card too small to hold them; chosen at boot when the config does not say (above) |
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | with the experts on the CPU: copied into RAM, or read from a file on the disk as they are needed — `disk` keeps the ones used in the engine's own memory, as much of it as there is (`experts_ram_gb` to set it), `mapped` leaves that to the system's page cache and is slower; `auto` chooses by the memory there is |
| `card_experts` | `"yes"` (default), `"no"` | with the experts on the CPU from RAM: the card's own tier of them (the L1), sized at boot to what the card holds beside the context — or none. `card_experts_gb` pins a size instead, in gigabytes, below |
| `experts_int8` | `false` (default), `true` | with the experts on the CPU: a generated token's experts multiplied in integers, faster on the CPU and slightly less exact — ▶ the quantisation table |
| `picture_pixels` | pixels, default `4194304` (2048 x 2048) | the largest picture the boot makes room for on the card, about 23 KB of the card's memory a 16 x 16 patch (0.37 GB at the default); a larger picture is made smaller. `0` keeps none and reads pictures on the CPU instead (1.8 GB of RAM, at most a megapixel) — the default on an integrated GPU (OpenCL 1.2), whose pool is refused the 4 MP room |
| `picture_weights` | `"auto"` (default), `"card"`, `"ram"` | the vision tower's weights, 0.6 GB: on the card, or in RAM and written to the card a block at a time as a picture is read (about 0.1 s more a picture); `auto` puts them on the card when the experts are on the CPU and in RAM otherwise |

GLM 5.3 Flash's `options`:

| option | values | |
|---|---|---|
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | its routed experts (142 GB) in RAM, or read from a file on the disk as they are needed — `disk` keeps the ones used in the engine's own memory (`experts_ram_gb` to set how much), `mapped` leaves that to the system's page cache; `auto` puts them in RAM when the available memory holds them with about 3 GB to spare (above) — about 145 GB available |
| `chunk` | positions, default 256 | how many positions of a prompt go through a layer at once; with the experts on the disk, 1024 reads a prompt faster |
| `sockets` | `1`, `2` | the CPU sockets the experts are split over; by default, all of them |
| `card_experts` | `"yes"` (default), `"no"` | with the experts in RAM: the card's own tier of experts (the L1), sized at boot — or none; `card_experts_gb` pins a size, in gigabytes, below |
| `card_exclusive` | `false` (default), `true` | with `experts_from: "ram"`: what the card's tier holds is not also kept in RAM, below |

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
as the memory there is allows — the process's own limit when it has one — less a quarter of it for everything else.

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
