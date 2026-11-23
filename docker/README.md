# SiLVANN in a container

On Linux, SiLVANN runs in a container. The model has a tool that reads files, runs commands and fetches web pages; in a
rootless container it reaches the container and one folder of yours, and nothing else of the machine
([`docs/local_ai.md`](../docs/local_ai.md#container-first)). It runs the models as fast as an install does.

## Set it up

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash
```

Where there is no `curl` (Debian's netinstall has only `wget`): `wget -qO- https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash`.

The script finds Podman (or Docker) and your card — AMD, NVIDIA, or none — and checks what the machine needs; where
something is missing, it says which command fixes it, and stops. It makes the folder `~/SiLVANN`, asks for a Brave
Search API key so the model can search the web (Enter skips it), and starts SiLVANN on port 8080 — or on the first
free port after it, where another program has 8080 — and prints the address.

Then open http://silvann.localhost:8080 (or http://localhost:8080). In **Options**, **Download models** fetches a
model and **start** runs it ([`docs/ui.md`](../docs/ui.md)). Put files for the model in `~/SiLVANN/workspace`; what it
makes appears there too.

To read MP4 and other video files, once — PyAV, which the image leaves out
([below](#what-the-image-leaves-out)):

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_nonfree.sh | bash
```

Later:

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash -s -- update
podman stop -t 300 silvann       # stop: it saves the conversations first
podman start silvann             # start again: the model it ran last starts with it
```

`update` fetches the newest version; the models, the conversations and the settings in `~/SiLVANN` stay. `bash -s --
--help` lists the script's options (another folder, another port, `--card cpu`, `--dry-run` to see what it would run).

**The name `silvann.localhost`.** A browser sends every name ending in `.localhost` to this machine, so the address
works with nothing to set up, and the browser keeps SiLVANN's cookies and stored data apart from other apps on
`localhost`. The name with no port at all — `http://silvann.localhost` — and other apps beside it under names of their
own: [`advanced.md`](advanced.md#each-app-a-name-under-localhost).

## What the machine needs

The script checks each of these and says what to run where one is missing.

- **Podman**, run as you. On Ubuntu and Debian: `sudo apt install podman crun uidmap slirp4netns fuse-overlayfs`.
  Docker works too.
- **An AMD card**, and you in the `video` and `render` groups: `sudo usermod -aG video,render $USER`, then log out and
  in again. The image holds ROCm 7's runtime; the card's driver is the machine's — the `amdgpu` driver your system
  has, or AMD's `amdgpu-dkms`. It runs the Instinct MI50 and MI60 and the Radeon VII (gfx906), the Instinct MI100 and
  later, and the Radeon RX 9060 and 9070 (RDNA4), and every such card of the machine is the container's. An AMD card
  the image runs comes before an NVIDIA card in the same machine, as it does where SiLVANN is installed: `--card
  nvidia` takes the NVIDIA card. A card the image is not built for (an older Radeon, a processor's own graphics) is
  hidden beside one that is, comes after an NVIDIA card, and alone it runs no image: `--card cpu`, or an image of your
  own ([`advanced.md`](advanced.md#building-an-image)).
- **An NVIDIA card:** its driver and NVIDIA's container toolkit, then
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` — once, and again after every driver update. Podman needs
  to be 4.1 or later for this (Ubuntu 24.04, Debian 12 and later); on Ubuntu 22.04 use Docker. A card older than the CUDA
  family takes (compute capability below 7.5: a GTX 10x0) runs through OpenCL 1.2 in the same image, the toolkit handing
  in the driver's OpenCL — not yet checked on such a card.
- **No card:** nothing more; the CPU runs the models.

## Web search

The model's `(search …)` needs a search service. The simplest is a **Brave Search API key** (a free plan at
https://brave.com/search/api/): the script asks for it, or put it in `~/SiLVANN/tools/default.json` later —

```json
{"search": {"enabled": "auto", "config": {"brave_key": "your key"}}}
```

— then **Rebuild the tools** in Options. The key stays with the server; the model never reads it. To run a search
engine of your own instead, a SearXNG beside SiLVANN: the companions, below.

## Companions: a browser and a search engine of the model's own

Containers beside SiLVANN's that the model's tools use, each set up by its own script — all of them in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_companions.sh | bash
```

| companion | its script | what the model gets |
|---|---|---|
| a Firefox, `silvann-firefox` | `install_companion_firefox.sh` | `(browser …)`: pages opened, looked at, clicked, typed into, Violentmonkey's userscripts ([`docs/tools.md`](../docs/tools.md#the-browser)) |
| SearXNG, `searxng` | `install_companion_searxng.sh` | `(search …)` without a key: a search engine of your own, asking the others |

They run on a network of their own, `silvann-companions`, which `setup.sh` puts SiLVANN on — a container made before
it, by an older `setup.sh`, is put on it by running `setup.sh` once with `update` (the folder stays). Then restart
SiLVANN, or press **Rebuild the tools** in Options.

⛔ **The browser never runs in SiLVANN's container.** A page that goes wrong — a download, an exploit — would find every
conversation there. The companion's firewall lets it reach the public internet and nothing private: not SiLVANN's
container, not this machine, not your network (a private address is refused at once). Its root is read-only, its
profile in memory and fresh for each session SiLVANN opens, Firefox runs without privileges, and nothing is kept.
Any script with `remove` takes its companion away; where SiLVANN runs without a container (`install.sh`), the
companions go on this machine's loopback instead, `127.0.0.1:9222` and `127.0.0.1:8888`. By hand, with Compose:
`docker compose -f companions/compose.yaml up -d --build` — the browser is always built here, from Debian's Firefox ESR
and Violentmonkey, never pulled ready-made: no published Firefox image carries its firewall.

## By hand

What the script runs, for doing it yourself. A folder first — `mkdir -p ~/SiLVANN` — then one command.

An AMD card:

```bash
podman run -d --name silvann --userns=keep-id --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:1-amd
```

(Where the machine holds AMD cards the image is not built for, hide them: `-e ROCR_VISIBLE_DEVICES=<the numbers of
the image's cards>` — [`advanced.md`](advanced.md#a-card-for-each-container).)

An NVIDIA card:

```bash
podman run -d --name silvann --userns=keep-id --device nvidia.com/gpu=all \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:1-nvidia
```

No card: either command without its `--device` and `--group-add` flags. To update: `podman pull` the image,
`podman rm silvann`, and the same `podman run`.

⚗ **An integrated GPU beside the card — experimental.** On a laptop whose Intel GPU sits beside its NVIDIA card, the Intel
GPU can do what the processors would — a larger model's experts in RAM, an older card's embedding — and leave them free:
add `--device /dev/dri --group-add keep-groups -e SILVANN_IGPU_CPU=1` to the NVIDIA command (Docker: `--device /dev/dri
-e SILVANN_IGPU_CPU=1`), which has Linux name the Intel GPU. The image holds Intel's OpenCL for it; `igpu_cpu` in
[config.md](../docs/config.md) says what goes there and when. It has run on one laptop's Intel GPU — reported, not measured here: the 35B-A3B at 12 tokens a
second with its experts there, 5 with them on the processors; the processors' load beside the CPU's own is the report it
still needs. A Linux machine's — Docker Desktop on Windows reaches a GPU through WSL 2's `/dev/dxg`, which the image's
Intel OpenCL is not built for (`ASSUMED`); there the native install sees the integrated GPU itself (the same variable).

- `--userns=keep-id` runs SiLVANN as you, so the files in `~/SiLVANN` stay yours (`setup.sh` runs it another way
  under Ubuntu 22.04's Podman: below).
- `--group-add keep-groups` keeps your `video` and `render` groups, which own the AMD card.
- `-p 127.0.0.1:8080:8080` keeps the chat to this machine: its first page asks for the machine's account (or
  `podman run --rm -it … silvann users add <name>` before it starts), and a single user's machine is one account's at a
  time. Made multi user (the UI's Options, or `silvann users multi` while no model runs;
  [docs/users.md](../docs/users.md#several-people-on-one-server-a-user-each)) the accounts are signed in side by side, and
  `-p 8080:8080` lets other machines on your network reach it; `setup.sh update` does so itself, unless `--private`.
- `-v ~/SiLVANN:/data` is the folder SiLVANN keeps everything in. With SELinux (Fedora and the like):
  `-v ~/SiLVANN:/data:Z`.

**Ubuntu 22.04** has an older Podman (3.4), which needs two things once:

```bash
mkdir -p ~/.config/containers
python3 -c "import json, sys; p = json.load(open('/usr/share/containers/seccomp.json')); p['syscalls'].append({'names': \
  ['landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self'], 'action': 'SCMP_ACT_ALLOW'}); \
  json.dump(p, open(sys.argv[1], 'w'))" ~/.config/containers/silvann-seccomp.json
test -e ~/.config/containers/containers.conf || printf '[engine]\nruntime = "crun"\n' > ~/.config/containers/containers.conf
```

then `--security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json` in the `podman run` (if you already had a
`containers.conf`, put `runtime = "crun"` under its `[engine]`). Why: SiLVANN fences the model's commands off from the
server's secrets with the kernel's Landlock, and 22.04's security profile refuses Landlock's calls — without the
profile above the model has no shell, and its `(help)` says why. Handing in the AMD card needs the `crun` runtime.

That Podman also opens no port for a container that runs as you on a network of the containers' own, and `setup.sh`
puts SiLVANN on one, the companions': the chat answers inside the container and nothing listens outside it
(`MEASURED`). So under Podman 3 `setup.sh` leaves `--userns=keep-id` out and gives `--cap-drop=all --read-only --tmpfs
/tmp` in its place: SiLVANN runs as the container's root — whom rootless Podman maps to you, so the files in
`~/SiLVANN` stay yours — with none of root's capabilities, and the image's own files read-only. The command above, by
hand, keeps its port as it is: it is on Podman's own network, where the companions do not find it.

**Docker** instead of Podman: leave out `--userns=keep-id` and `--group-add keep-groups`; an NVIDIA card is
`--gpus all`, an AMD card `--device /dev/kfd --device /dev/dri --group-add video --group-add render`. Under rootless
Docker, add `--read-only --tmpfs /tmp`. Docker needs no security profile.

## What the image leaves out

PyAV, through which SiLVANN reads MP4 and other video files (it reads an animated GIF, WebP or PNG without): its wheels
carry FFmpeg's libraries under the LGPL and the GPL, which the image does not redistribute. `install_nonfree.sh`
(above) fetches them from PyPI into `~/SiLVANN/python`, where they stay through an update, and restarts SiLVANN; run it
again after an update that changes the image's Python. The model's commands read that folder and never write it: the
server imports what is there as it starts.

## The images

| image | for |
|---|---|
| `ghcr.io/eval-apply-labs/silvann:1-amd` | AMD's Instinct MI50 and MI60 and Radeon VII (gfx906), Instinct MI100 and later (gfx908, gfx90a, gfx942, gfx950) and Radeon RX 9060 and 9070 (gfx1200, gfx1201), with ROCm 7.2.4's runtime; and the CPU |
| `ghcr.io/eval-apply-labs/silvann:1-nvidia` | NVIDIA cards CUDA 12.9 builds for (the GeForce RTX 20 to 50 series among them); and the CPU |

`:1-amd` and `:1-nvidia` are the newest release of major version 1; `:1.5.0-amd` and the like stay that release.
1.4.4 had two AMD images, `:1-amd-rocm6` and `:1-amd-rocm7`: there is one AMD image from 1.5.0 and those two names
ended with 1.6.0 — [`setup.sh`](setup.sh)'s `update` moves a container made with either to `:1-amd`, and a command of
your own names `:1-amd` in their place. The image with ROCm 6's runtime ended with 1.4.4 (`:1.4.4-amd-rocm6`): the
MI50 and MI60 run on ROCm 7's from 1.5.0.
`:0-amd` and `:0-nvidia` stay 0.29.1, the last of major version 0.
`/opt/silvann/IMAGE_NOTICES.md` in each lists what it holds, each part's licence, and the commit it was built from.
On Windows, install SiLVANN instead (the README's quick start): under Docker Desktop the models run well below what
the PC gives.

**Tested:** `setup.sh` as it is on node03 (Ubuntu 22.04, rootless Podman 3.4, two MI50s), with the 1.4.3 image: a
first run made the companions' network and started the container with the chat answering at `silvann.localhost`, both
cards in it and the folder's files the user's; run again it said so; it was stopped, started and updated, the port
kept; a `--port` named busy was refused, and with 8080 taken it would take 8081; `install_nonfree.sh` put PyAV 17.1 in
the folder, restarted SiLVANN, and the container imported it. In a container with the same flags the Qwen 3.6 35B-A3B
`D8E4` (its experts in RAM) and the Qwen 3.5 2B `D4` passed against Hugging Face on an MI50, at the rates of a
container run as the user (the 2B 212.8 tokens a second against 211.5), and a model's commands stayed fenced, 95
checks of 95 (`MEASURED`, measurements/2026-11-23_amd_rocm7_image_node03.md). With the script of 0.21.0 there: a search
key kept in the tools file (which the server's own reading of the file keeps) through a second run and an update, and
PyAV still imported after an update (`MEASURED` then). The search companion beside a container run this way, set up
by `install_companion_searxng.sh`, answered a search from inside SiLVANN's container (`MEASURED`,
measurements/2026-11-23_v144_candidate.md); the browser companion so: not yet run. On the Dell
Precision (Ubuntu 22.04, an RTX 5060 Ti) under rootless Docker: the NVIDIA card found and handed in, a search key kept,
the Qwen 3.8 27B started from Options and answering at 24.3 tokens a second, `update`, `stop` and `start` with the model
started again by itself, the model's shell on under Docker's own profile (`MEASURED`,
measurements/2026-10-27_nn134_setup_precision.md). Podman 4.1 or later with an NVIDIA card, and a distribution whose
own profile lets Landlock through: `ASSUMED`, not yet run. In the container, on an MI50, the Qwen 3.6 35B-A3B `D8E4`
read a prompt at 378.7-379.0 positions a second and answered at 86.5-88.6 tokens a second, where the same release
installed on the host gave 378.7-379.3 and 87.6-88.4, with the same words (`MEASURED`, v0.20.0,
measurements/2026-10-26_nn119_v0200_image_node03.md). Ubuntu 22.04 refusing Landlock without the profile: `MEASURED`
on node03; the newer distributions letting it through: `REASONED` from containers/common's profile, which has let it
through since v0.49. Under rootless Docker 29.9 the model's commands were fenced off from the tools file
(`MEASURED`, measurements/2026-10-26_nn119_nvidia_image_precision.md). Under Docker Desktop on Windows, Ministral 3 14B
`D4` answered at 37 tokens a second on an RTX 5070 Ti, where the same version installed on Linux gave 46.

**The AMD image, on ROCm 7.** Built on node03 from `Dockerfile.amd` (ROCm 7.2.4, 730 MB), its family holding code for
gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1200 and gfx1201, and run on that machine's two MI50s: `setup.sh` started it
with the chat answering and both cards in it; in a container with its flags the Qwen 3.5 2B and 4B `D4`, Ministral 3
14B `D4` and the Qwen 3.6 35B-A3B `D8E4` (its experts in RAM) passed against Hugging Face, the 4B over both cards too,
and a model's commands stayed fenced, 95 checks of 95. Against 1.4.4's image for the MI50, which held ROCm 6.1.3, on
the same card: the same tokens, a prompt read as fast or 3 to 4 % faster, the answer 2 to 5 % slower — the 2B 167
tokens a second where it gave 176 (`MEASURED`, measurements/2026-11-23_nn206_gfx906_rocm7_node03.md). An Instinct
MI100 or later, a Radeon RX 9060 or 9070, an MI50 beside one of them in the container, and an MI50 under a kernel
newer than 5.15: `ASSUMED`, not yet run — the family has run on an MI355X, an RX 9070 XT and an RX 9060 XT outside a
container ([`docs/platforms.md`](../docs/platforms.md)).

## More

SearXNG, each app a name under localhost, a card for each container, a model over several containers or machines,
servers that find each other, videos, and building an image: [`advanced.md`](advanced.md).
