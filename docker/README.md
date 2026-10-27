# SiLVANN in a container

On Linux, SiLVANN runs in a container. The model has a tool that reads files, runs commands and fetches web pages; in a
rootless container it reaches the container and one folder of yours, and nothing else of the machine
([`docs/local_ai.md`](../docs/local_ai.md#container-first)). It runs the models as fast as an install does.

## Set it up

```bash
curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash
```

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
- **An AMD card** of the gfx906 generation (Instinct MI50, MI60): its driver (ROCm 6's `amdgpu-dkms`), and you in the
  `video` and `render` groups: `sudo usermod -aG video,render $USER`, then log out and in again.
- **An NVIDIA card:** its driver and NVIDIA's container toolkit, then once
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Podman needs to be 4.1 or later for this (Ubuntu 24.04,
  Debian 12 and later); on Ubuntu 22.04 use Docker.
- **No card:** nothing more; the CPU runs the models.

## Web search

The model's `(search …)` needs a search service. The simplest is a **Brave Search API key** (a free plan at
https://brave.com/search/api/): the script asks for it, or put it in `~/SiLVANN/tools/default.json` later —

```json
{"search": {"enabled": "auto", "config": {"brave_key": "your key"}}}
```

— then **Rebuild the tools** in Options. The key stays with the server; the model never reads it. To run a search
engine of your own instead, a SearXNG beside SiLVANN: [`advanced.md`](advanced.md#searxng-a-search-engine-of-your-own).

## By hand

What the script runs, for doing it yourself. A folder first — `mkdir -p ~/SiLVANN` — then one command.

An AMD card:

```bash
podman run -d --name silvann --userns=keep-id --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
```

An NVIDIA card:

```bash
podman run -d --name silvann --userns=keep-id --device nvidia.com/gpu=all \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-nvidia
```

No card: either command without its `--device` and `--group-add` flags. To update: `podman pull` the image,
`podman rm silvann`, and the same `podman run`.

- `--userns=keep-id` runs SiLVANN as you, so the files in `~/SiLVANN` stay yours.
- `--group-add keep-groups` keeps your `video` and `render` groups, which own the AMD card.
- `-p 127.0.0.1:8080:8080` keeps the chat to this machine: it has no login.
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
| `ghcr.io/eval-apply-labs/silvann:0-amd` | AMD cards of the gfx906 generation (Instinct MI50 and MI60), with ROCm 6.1.3's runtime; and the CPU |
| `ghcr.io/eval-apply-labs/silvann:0-nvidia` | NVIDIA cards CUDA 12.9 builds for (the GeForce RTX 20 to 50 series among them); and the CPU |

`:0-amd` and `:0-nvidia` are the newest release of major version 0; `:0.21.0-amd` and the like stay that release.
`/opt/silvann/IMAGE_NOTICES.md` in each lists what it holds, each part's licence, and the commit it was built from.
On Windows, install SiLVANN instead (the README's quick start): under Docker Desktop the models run well below what
the PC gives.

**Tested:** `setup.sh` on node03 (Ubuntu 22.04, Podman 3.4, an MI50): it found the AMD card, wrote the 22.04 profile,
started the container with the chat answering, kept a search key in the tools file (which the server's own reading
of the file keeps), and was run again, stopped, started and updated, the key kept; with 8080 taken it took 8081, a
`--port` named busy was refused, and the chat answered at `silvann.localhost` on both; `install_nonfree.sh` put PyAV
17.1 in the folder, restarted SiLVANN, and the container imported it, after an update too (`MEASURED`). Its NVIDIA and Docker
paths, and the newer distributions: `ASSUMED`, not yet run. In the container, on an MI50, the Qwen 3.6 35B-A3B `D8E4`
read a prompt at 378.7-379.0 positions a second and answered at 86.5-88.6 tokens a second, where the same release
installed on the host gave 378.7-379.3 and 87.6-88.4, with the same words (`MEASURED`, v0.20.0,
measurements/2026-10-26_nn119_v0200_image_node03.md). Ubuntu 22.04 refusing Landlock without the profile: `MEASURED`
on node03; the newer distributions letting it through: `REASONED` from containers/common's profile, which has let it
through since v0.49. Under rootless Docker 29.9 the model's commands were fenced off from the tools file
(`MEASURED`, measurements/2026-10-26_nn119_nvidia_image_precision.md). Under Docker Desktop on Windows, Ministral 3 14B
`D4` answered at 37 tokens a second on an RTX 5070 Ti, where the same version installed on Linux gave 46.

## More

SearXNG, each app a name under localhost, a card for each container, a model over several containers or machines,
servers that find each other, videos, and building an image: [`advanced.md`](advanced.md).
