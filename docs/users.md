# Running SiLVANN

This is a **developer preview (v0.5.0)**. It is tested on Linux with AMD Instinct MI50 cards (gfx906, ROCm 6), with an
NVIDIA GeForce RTX 5060 Ti (sm_120, CUDA 12.9), and on the CPU alone with AVX2 and with AVX-512 — on Ubuntu 22.04. On a
Windows PC, boot Linux from a USB SSD (below); WSL2 and other NVIDIA generations are untested, Apple silicon is not
supported yet. If something does not build or does not run on your
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
venv/bin/python silvann_core.py download-model Qwen3.6-35B-A3B_silvann_tq_D8E4
```

A model lands in `models/<name>/` with a `configs/default.json` and its program in `lisp/` — the Lisp the evaluator
runs it with, yours to read and change. The server refuses a model that is not there and tells you this command.

## 3. Serve it, and chat

```bash
venv/bin/python start_silvann_server.py --model models/Qwen3.6-35B-A3B_silvann_tq_D8E4 [--port 8765]
venv/bin/python silvann_ui.py --port 8090
```

The UI is a web page on port 8090 that talks to the server on port 8765; it can also start a server itself (Options). Conversations are saved under `user_sessions/` and
can be exported and imported. One conversation runs at a time; a second request while one is running is
answered `503` with `Retry-After: 5`, and the UI retries.

On a machine with no card the model runs on the CPU's own cores (the `x86_avx2` family). The pack made for
that is `Qwen3.6-35B-A3B_silvann_tq_D4E4`: its dense part at 4 bits, about 15 tokens a second on the test machine's
two Xeons. On a processor with AVX-512 (F, BW, VL and DQ) the family's matrix products use it: on a four-core Xeon
W-2123 with two DDR4-2400 sticks the 35B D8E4 alone ran at 9.0 tokens a second against 1.7 with AVX2, and with its
experts on the CPU beside a card at 27-29 against 15-17. `SILVANN_CPU_AVX512=0` keeps the AVX2 code.

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
| `max_context` | the longest conversation, in positions, 32,768 by default; the cache is sized for it at boot, and the `_note` beside it says how many positions 100 MB of the card holds for this model — lower it if the model does not fit on your card, raise it if there is room |
| `thinking` | whether the model reasons before it answers, by default |
| `lora` | an adapter from the model's folder to load at boot, or `null` |
| `sampler` | how the next token is chosen: `"generation"` (the model's own `generation_config.json`), `null` (the most likely, greedy), or `{"temperature", "top_k", "top_p", "repetition_penalty"}`. It is drawn on the card — only the token comes back — when `temperature` is above 0 and `top_k` 1 to 64, and on the host over the logits otherwise |
| `options` | handed to the model's runtime, below |

The Qwen models' `options`:

| option | values | |
|---|---|---|
| `kv` | `"tiered"` (default), `"fp16"` | the conversation's cache: tiered keeps the first 256 and the latest positions in halves and the rest at 8 bits |
| `tiers` | `[sink, hot, warm]` | the tiers' sizes; `warm` may be `"rest"`. `[256, 4096, "rest"]` is the default; `[256, 4096, N]` puts what is older than `N` at 4 bits, for small cards |
| `chunk` | positions, 1 to 256, default 256 | how many positions of a prompt go through a layer at once |
| `experts_on` | `"card"` (default), `"cpu"` | the routed experts on the card, or on the CPU — the setup for a card too small to hold them; the 122B's `default.json` starts with `"cpu"` |
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | with the experts on the CPU: copied into RAM, or read from a file on the disk as they are needed — `disk` keeps the ones used in the engine's own memory, as much of it as there is (`experts_ram_gb` to set it), `mapped` leaves that to the system's page cache and is slower; `auto` chooses by the memory there is |
| `card_experts_gb` | gigabytes, default `0` | with the experts on the CPU: the card's own tier of them, below |
| `experts_int8` | `false` (default), `true` | with the experts on the CPU: a generated token's experts multiplied in integers, faster on the CPU and slightly less exact — ▶ the quantisation table |

GLM 5.3 Flash's `options`:

| option | values | |
|---|---|---|
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | its routed experts (142 GB) in RAM, or read from a file on the disk as they are needed — `disk` keeps the ones used in the engine's own memory (`experts_ram_gb` to set how much), `mapped` leaves that to the system's page cache; `auto` puts them in RAM when they leave a fifth of the available memory free — about 180 GB available |
| `chunk` | positions, default 256 | how many positions of a prompt go through a layer at once; with the experts on the disk, 1024 reads a prompt faster |
| `sockets` | `1`, `2` | the CPU sockets the experts are split over; by default, all of them |
| `card_experts_gb` | gigabytes, default `0` | the card's own tier of experts, below |
| `card_exclusive` | `false` (default), `true` | with `experts_from: "ram"`: what the card's tier holds is not also kept in RAM, below |

### The card's tier of experts

With the experts on the CPU, `card_experts_gb` gives that much of the card's memory to the experts used most, and
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

A model can be split by layers over machines, each holding a run of them (`silvann_runtime/pipeline.py`). This
is tested — the 27B as three stages answers exactly as it does in one process — but has no launcher yet; the
module's docstring shows how a stage is started.
