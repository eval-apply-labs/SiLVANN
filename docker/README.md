# SiLVANN in a container

From v0.20.0 a container is the way SiLVANN recommends running it on Linux: rootless under Podman, so that the model's
tool — a Lisp REPL that reads files, runs commands and fetches from the network when it is allowed to — reaches the
container and the folder it is given, and nothing else of the machine ([`docs/local_ai.md`](../docs/local_ai.md#container-first)).
On Linux the container runs the models as fast as the install does: on an MI50 the Qwen 3.6 35B-A3B `D8E4` read a
prompt at 378.7-379.0 positions a second and answered at 86.5-88.6 tokens a second rootless, where the same release on
the host gave 378.7-379.3 and 87.6-88.4, the same words (v0.20.0,
`measurements/2026-10-26_nn119_v0200_image_node03.md`). The installers (`install.sh`, `install.ps1`) stay for a build of
your own and for Windows ([`README.md`](../README.md)).

## The images

| image | holds |
|---|---|
| `ghcr.io/eval-apply-labs/silvann:0-amd` | AMD's family for gfx906 (Instinct MI50 and MI60) with ROCm 6.1.3's HIP runtime, and the CPU's |
| `ghcr.io/eval-apply-labs/silvann:0-nvidia` | NVIDIA's family for every card CUDA 12.9 builds for (compute capability 7.5 to 12.0, the GeForce RTX 20 to 50 series among them), and the CPU's |

`:0-…` is major version 0's newest release, updated in place; `:0.20.0-amd` and the like stay that release. Each holds
the release installed by `install.sh` on a plain Ubuntu 22.04 with no compiler, the models and the conversations in a
folder of yours at `/data`, and the chat as the app on port 8080: its Options download a model and start it.
`/opt/silvann/IMAGE_NOTICES.md` in each lists what it holds, each part's licence, and the commit it was built from.

## Running it rootless with Podman

**Once:** the folder the container keeps everything in, and a seccomp profile — Podman's own, with Landlock's three
calls let through (below):

```bash
mkdir -p ~/SiLVANN ~/.config/containers
python3 -c "import json, sys; p = json.load(open('/usr/share/containers/seccomp.json')); p['syscalls'].append({'names': \
  ['landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self'], 'action': 'SCMP_ACT_ALLOW'}); \
  json.dump(p, open(sys.argv[1], 'w'))" ~/.config/containers/silvann-seccomp.json
```

**An AMD card.** The card's kernel driver is the host's (`amdgpu`, from ROCm 6's `amdgpu-dkms`); the container is handed
its device nodes and keeps your groups, which own them:

```bash
podman run -d --name silvann --userns=keep-id --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
```

**An NVIDIA card.** The host's driver and NVIDIA's container toolkit, which describes the card to Podman (CDI) — once,
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`; CDI needs Podman 4.1 or later:

```bash
podman run -d --name silvann --userns=keep-id --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --device nvidia.com/gpu=all -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-nvidia
```

**No card:** either image without the device flags; the CPU runs the models.

The chat is at http://localhost:8080. What each flag does:

- `--userns=keep-id` runs the app as you: the files in `~/SiLVANN` are yours inside and outside the container.
- `--security-opt seccomp=…` lets the model's commands be closed off from what is not theirs: the machine's tools file,
  with its keys, and the server's own process ([`docs/tools.md`](../docs/tools.md#where-the-models-commands-reach)).
  The kernel's Landlock does it, and Ubuntu 22.04's Podman (3.4) refuses its calls (`MEASURED` on node03): without the
  profile the model has no shell, and its `(help)` says why. The profile is Podman's own with those three calls added,
  and it stays outside `~/SiLVANN`, which the model reaches.
- `--group-add keep-groups` keeps your groups in the container — for an AMD card `video` and `render`, which own
  `/dev/kfd` and `/dev/dri`. It needs Podman's `crun` runtime (`runtime = "crun"` under `[engine]` in
  `~/.config/containers/containers.conf`); you must be in both groups on the host (`sudo usermod -aG video,render $USER`,
  then log in again).
- `-p 127.0.0.1:8080:8080` keeps the chat to this machine: the UI has no login, and a conversation's password is its
  only lock.
- `-v ~/SiLVANN:/data` is the models, the conversations, the tools file (`~/SiLVANN/tools/default.json`, its keys and
  switches — [`docs/tools.md`](../docs/tools.md#the-tools-file)) and the model's workspace (`~/SiLVANN/workspace`):
  put a file there and the model reads it; what it makes there is yours to open. A new image run over the same folder
  keeps them all. On a machine with SELinux (Fedora and the like), `-v ~/SiLVANN:/data:Z`.

Rootless Podman on Ubuntu 22.04: `sudo apt install podman crun uidmap slirp4netns fuse-overlayfs`. Its Podman, 3.4, takes
an AMD card; for an NVIDIA card's CDI, Ubuntu 24.04's Podman or a newer build.

**Stopping and updating.** `podman stop -t 300 silvann` gives the UI time to save every conversation and stop the
server. A new release: `podman pull` the image, `podman rm silvann`, and the same `podman run` over the same folder.
The model started last starts again when the container does.

**With Docker.** The same images run under Docker: `--gpus all` for an NVIDIA card (NVIDIA's container toolkit), or
`--device /dev/kfd --device /dev/dri --group-add video --group-add render` for an AMD card, without `--userns` and
`keep-groups`, and `-v ~/SiLVANN:/data` the same. Under Docker run as root, the image gives what the folder holds to its own user,
uid 1000 — yours, where you are the machine's first user — and runs the app as it. Under rootless Docker the container's
root is you, and its uid 1000 is not: the app runs as the container's root and leaves the folder as it is, yours — add
`--read-only --tmpfs /tmp`, so that the app, and the model's commands, cannot change the image's own system:

```bash
docker run -d --name silvann --read-only --tmpfs /tmp --gpus all -p 127.0.0.1:8080:8080 -v ~/SiLVANN:/data \
       ghcr.io/eval-apply-labs/silvann:0-nvidia
```

Docker's own seccomp profile lets Landlock's calls through: under rootless Docker 29.9 on the Dell Precision the
server's log said `the REPL's (bash …): the machine but its tools file and the server's process (Landlock ABI 1)`, and a
command so confined was refused the tools file (`MEASURED`, measurements/2026-10-26_nn119_nvidia_image_precision.md). Under
Docker Desktop on Windows the models run well below what the PC gives — 37 tokens a second with Ministral 3 14B `D4` on
an RTX 5070 Ti, where the same version installed on Linux gave 46 — so on Windows install SiLVANN instead
(`install.ps1`, [`README.md`](../README.md)).

## Searching the web: a SearXNG beside it

The model's `(search …)` asks a SearXNG, a search engine of your own that asks the others
([`docs/tools.md`](../docs/tools.md#searching-the-web)). Under Podman, run it in a pod with SiLVANN: the two share an
address, SearXNG listens on 8888, and SiLVANN finds it at `127.0.0.1:8888` by itself as it starts.

```bash
mkdir -p ~/.config/silvann/searxng
printf 'use_default_settings: true\nserver:\n  secret_key: "%s"\n  limiter: false\n  image_proxy: false\nsearch:\n  formats: [html, json]\n' \
       "$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > ~/.config/silvann/searxng/settings.yml
podman pod create --name silvann --userns=keep-id -p 127.0.0.1:8080:8080
podman run -d --pod silvann --name searxng -e SEARXNG_PORT=8888 -v ~/.config/silvann/searxng:/etc/searxng \
       docker.io/searxng/searxng
podman run -d --pod silvann --name silvann --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --group-add keep-groups --device /dev/kfd --device /dev/dri -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
```

In a pod, `--userns=keep-id` and the chat's port go on the pod, and each container keeps its own flags (an NVIDIA card's
`--device nvidia.com/gpu=all` in place of AMD's). SiLVANN's log says at the start `the search skill: ready`.
`podman pod stop -t 300 silvann` and `podman pod start silvann` stop and start both. SearXNG's settings stay outside
`~/SiLVANN`. A network of their own, the other way containers find each other, loses the chat's port under Ubuntu
22.04's Podman (3.4) when the app runs as you (`--userns=keep-id`: `MEASURED` on node03); the pod keeps both.

Under Docker: `docker network create silvann`, SearXNG started with `--name searxng --network silvann`, and SiLVANN's
`docker run` with `--network silvann` — SiLVANN finds it at `http://searxng:8080` (`ASSUMED`: tried under Podman alone).

A Brave Search API key is the other way, with no SearXNG: the `search` entry's `brave_key` in
`~/SiLVANN/tools/default.json`, then Rebuild the tools.

## Each app a name under localhost

A browser sends every name ending in `.localhost` to this machine by itself, with nothing to set up, so the apps in a
pod can each have a name instead of a port. One small proxy in the pod holds the pod's only published port and passes
each name to its app. The apps publish nothing and keep their own ports inside the pod, so they never collide with each
other's or the machine's:

```bash
mkdir -p ~/.config/silvann/caddy
cat > ~/.config/silvann/caddy/Caddyfile <<'END'
{
	auto_https off
}
http://silvann.localhost:8000 {
	reverse_proxy 127.0.0.1:8080
}
http://searxng.localhost:8000 {
	reverse_proxy 127.0.0.1:8888
}
http://localhost:8000 {
	respond <<LIST
	silvann.localhost:8000    SiLVANN's chat
	searxng.localhost:8000    SearXNG, the search SiLVANN's model asks
	LIST 200
}
http://:8000 {
	respond "no app is named so here: localhost:8000 lists them" 404
}
END
podman pod create --name silvann --userns=keep-id -p 127.0.0.1:8000:8000
podman run -d --pod silvann --name searxng -e SEARXNG_PORT=8888 -v ~/.config/silvann/searxng:/etc/searxng \
       docker.io/searxng/searxng
podman run -d --pod silvann --name silvann --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --group-add keep-groups --device /dev/kfd --device /dev/dri -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
podman run -d --pod silvann --name caddy -v ~/.config/silvann/caddy/Caddyfile:/etc/caddy/Caddyfile:ro \
       docker.io/library/caddy:2.8.4
```

with SearXNG's settings as [above](#searching-the-web-a-searxng-beside-it). The chat is then at
`http://silvann.localhost:8000`, SearXNG at `http://searxng.localhost:8000`, and `http://localhost:8000` lists what the
pod serves. Another app joins as a container in the pod and three lines in the file, then `podman restart caddy`. On
node03, under Ubuntu 22.04's Podman 3.4 (`MEASURED` with curl, which sends `*.localhost` to the loopback as the browsers
do): the chat's page and its websocket through `silvann.localhost`, SearXNG's search through `searxng.localhost`, the
list at `localhost`, a 404 for any other name, and the host holding port 8000 alone
(measurements/2026-10-27_localhost_names_node03_artifacts/pod.log).

**A name is also a wall.** A browser keeps cookies and stored data by name, not by port, so two apps at
`localhost:8080` and `localhost:8090` share them; at `silvann.localhost` and `searxng.localhost` each keeps its own, and
one app's page cannot read another's session.

**Without the port in the address**, the pod publishes port 80. Rootless Podman binds no port below 1024 unless the
system allows it: `sudo sysctl net.ipv4.ip_unprivileged_port_start=80` (a line in `/etc/sysctl.d/` keeps it), then
`-p 127.0.0.1:80:8000` on the pod, and the chat is at `http://silvann.localhost` (`ASSUMED`: not tried here). Under
Docker, which has no pods, the containers share one network with `--network container:caddy` (`ASSUMED`).

## One card a container, and a model across containers

**A card of its own.** Each container can be given its own card, so two models — or two stages of one — never share
one:

- **AMD:** hand in the devices whole and name the card: `--device /dev/kfd --device /dev/dri -e ROCR_VISIBLE_DEVICES=1`
  — the container's HIP runtime sees that card alone, as its card 0. A card's render node alone
  (`--device /dev/dri/renderD129`) is not enough: ROCm 6.1's runtime then sees no card at all (`MEASURED` on node03).
  `ROCR_VISIBLE_DEVICES` takes the card's index on the host, as `rocm-smi` numbers them.
- **NVIDIA:** `--device nvidia.com/gpu=1` hands in that card alone (`ASSUMED` for a second card; `gpu=0` gave the
  one card of the Dell Precision's, `MEASURED`).
- **No card:** no device flags, and `"card": "cpu"` in the model's config — the CPU and its RAM run it.

**A model over several containers** ([`docs/users.md` §6](../docs/users.md#6-a-model-over-several-machines)): each
container is a machine of the ring, its config naming its layers, its port and the next machine's. In one pod they
reach each other at `127.0.0.1`; on several machines, publish each one's ring port. The first machine (layer 0) serves
the conversations' API; the others serve the ring. Each runs the server itself:

```bash
podman pod create --name ring --userns=keep-id -p 127.0.0.1:8765:8765
podman run -d --pod ring --name ring-cpu -v ~/SiLVANN-cpu:/data --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       ghcr.io/eval-apply-labs/silvann:0-amd /opt/silvann/venv/bin/python /opt/silvann/start_silvann_server.py \
       --model /data/models/<model> /data/ring.json --host 127.0.0.1 --port 8081
podman run -d --pod ring --name ring-card -v ~/SiLVANN:/data --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json \
       --group-add keep-groups --device /dev/kfd --device /dev/dri -e ROCR_VISIBLE_DEVICES=0 \
       ghcr.io/eval-apply-labs/silvann:0-amd /opt/silvann/venv/bin/python /opt/silvann/start_silvann_server.py \
       --model /data/models/<model> /data/ring.json --host 0.0.0.0 --port 8765
```

with `~/SiLVANN/ring.json` `{"max_context": 8192, "card": 0, "pipeline_parallelism": "yes",
"pipeline_parallelism_layers": "0-19", "pipeline_parallelism_next": "127.0.0.1:8767", "pipeline_parallelism_port":
8766}` and `~/SiLVANN-cpu/ring.json` the same with `"card": "cpu"`, `"0-19"` → `"20-39"`, the next `127.0.0.1:8766` and
the port `8767`. The chat reaches the ring as another machine's server: Options → Send conversations to → remote,
`http://localhost:8765` — from a UI on the host, or the image's own command in the pod with the chat's port published
too (`ASSUMED`: not yet tried). On node03 the Qwen 3.6 35B-A3B ran so, its first 20 layers on one MI50 and the rest on the CPU in a
container with no card, at 21.7 tokens a second (`MEASURED`, measurements/2026-10-26_nn119_v0200_image_node03.md). An
NVIDIA card's stage is the NVIDIA image in the same pod: the ring is the model's, whatever runs each run of layers
(`ASSUMED`: an AMD and an NVIDIA image in one ring not yet tried). `"card": 0` is the one card the container sees.

## Servers that find each other

The registry of servers, through which the servers of a network find each other for the subagents
([`docs/tools.md`](../docs/tools.md#servers-found-through-a-registry)), is in the same image under another command. On
the machine that keeps it:

```bash
mkdir -p ~/SiLVANN-registry
podman pod create --name silvann-registry --userns=keep-id -p 8770:8770
podman run -d --pod silvann-registry --name registry -v ~/SiLVANN-registry:/data \
       ghcr.io/eval-apply-labs/silvann:0-amd silvann registry
```

It writes its key into `~/SiLVANN-registry/registry/key` at its first start. A server offered to it is one that other
machines reach: started on its own with `--host 0.0.0.0` and its port published, the chat's UI pointed at it (Options
→ Send conversations to → remote), as a ring's first machine is above. The UI's own server, which the image starts by
default, listens inside its container alone. On each machine, with that machine's address:

```bash
podman pod create --name silvann-server --userns=keep-id -p 8765:8765
podman run -d --pod silvann-server --name server -v ~/SiLVANN:/data \
       --security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json --group-add keep-groups \
       --device /dev/kfd --device /dev/dri \
       -e SILVANN_TOOL_AGENT_REGISTRY=http://192.0.2.10:8770 -e SILVANN_TOOL_AGENT_KEY="$(cat key)" \
       -e SILVANN_TOOL_AGENT_ADDRESS=http://192.0.2.20:8765 \
       ghcr.io/eval-apply-labs/silvann:0-amd silvann serve <model> --host 0.0.0.0
```

where `key` is a copy of the registry's key file. Inside a container the server's own address is the container's,
which other machines do not reach, so `SILVANN_TOOL_AGENT_ADDRESS` names the machine's address and the published port.
The server's log says `offered to the registry … as …` as it starts, and the registry's `offered: …`; stopped, the
server withdraws its offer. A pod keeps the published port where Ubuntu 22.04's Podman (3.4) loses it for a container
run as you on a network of its own (`MEASURED` on node03, as above).

On node03 the registry and two servers each ran so in a pod of their own, from the v0.21.0 image — Qwen 3.8 27B on one
MI50, Qwen 3.6 35B-A3B on the other, each reaching the others only at the machine's address and published port, as
another machine would: both offered themselves, the 35B ran the 27B as a subagent from `(whoami)`'s listing alone, and
the 27B's pod stopped left the listing (`MEASURED`, measurements/2026-10-27_v0210_two_servers_node03.md).

## What the image leaves out

`requirements-nonfree.txt` lists what SiLVANN reads more with and runs without, whose wheels carry binaries under the
LGPL and the GPL, which the image does not redistribute: today PyAV, through which a video file other than an animated
picture (MP4 and the like) is read; without it an animated GIF, WebP or PNG is still read, and such a file is refused
with its reason. The image's environment looks for it in the folder, under `/data/python/<its Python version>`, so it
stays through a new image:

```bash
podman run --rm --userns=keep-id -v ~/SiLVANN:/data --entrypoint /opt/silvann/venv/bin/python \
       ghcr.io/eval-apply-labs/silvann:0-amd -c "import subprocess, sys; \
  sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--target', '/data/python/%d.%d' % sys.version_info[:2], \
  '-r', '/opt/silvann/requirements-nonfree.txt']))"
```

then restart the container. `install.sh`, outside a container, installs it with the rest (`SILVANN_NONFREE=0` leaves it
out).

## Building one

From the release's folder. A build takes `install.sh` and the image's two scripts from the folder, and the release
from a repository's URL, as `install.sh` fetches it — GitHub's by default:

```bash
podman build -f docker/Dockerfile.amd -t silvann:amd .                                  # AMD, gfx906
podman build -f docker/Dockerfile.amd --build-arg AMD_ARCH=gfx90a -t silvann:amd-gfx90a .   # another AMD architecture
podman build -f docker/Dockerfile -t silvann:nvidia .                                   # NVIDIA
podman build -f docker/Dockerfile --build-arg SILVANN_SOURCE=<a repository> --build-arg SILVANN_REF=<branch> \
             -t silvann:nvidia .                                                        # another repository or branch
```

A checkout with changes of its own goes in by a repository it is pushed to: the build mounts nothing of the machine's,
which Ubuntu 22.04's Podman (3.4) does not take. The images published are built by `.github/workflows/images.yml` on a
release's tag.
