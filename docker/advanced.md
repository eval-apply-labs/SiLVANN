# SiLVANN in a container: more setups

[`README.md`](README.md) sets up one chat on one machine, which is what most people want. This page is for the rest:
a search engine of your own, each app a name under localhost, a card for each container, a model over several
containers or machines, servers that find each other, videos by hand, and building an image.

The commands here are the by-hand ones. On Ubuntu 22.04, add the security profile to each SiLVANN container
(`--security-opt seccomp=$HOME/.config/containers/silvann-seccomp.json`, made as in
[`README.md`](README.md#by-hand)).

## SearXNG: a search engine of your own

SearXNG is a separate program — a search engine that asks the others and answers in JSON — in a container of its own,
from SearXNG's image; SiLVANN's image does not include it. Run it beside SiLVANN, in a *pod*: containers in one pod
share an address, so SiLVANN finds SearXNG by itself as it starts. It is the alternative to a Brave Search API key
([`README.md`](README.md#web-search)); with both, SiLVANN asks SearXNG first.

Once, SearXNG's settings, outside `~/SiLVANN`:

```bash
mkdir -p ~/.config/silvann/searxng
printf 'use_default_settings: true\nserver:\n  secret_key: "%s"\n  limiter: false\n  image_proxy: false\nsearch:\n  formats: [html, json]\n' \
       "$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > ~/.config/silvann/searxng/settings.yml
```

Then the pod, in place of SiLVANN's own container (`podman stop -t 300 silvann; podman rm silvann` first):

```bash
podman pod create --name silvann --userns=keep-id -p 127.0.0.1:8080:8080
podman run -d --pod silvann --name searxng -e SEARXNG_PORT=8888 -v ~/.config/silvann/searxng:/etc/searxng \
       docker.io/searxng/searxng
podman run -d --pod silvann --name silvann --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
```

In a pod, `--userns=keep-id` and the port go on the pod; an NVIDIA card's container takes `--device nvidia.com/gpu=all`
in place of AMD's flags. SiLVANN's log says `the search skill: ready` as it starts. `podman pod stop -t 300 silvann`
and `podman pod start silvann` stop and start both.

Under Docker, which has no pods: `docker network create silvann`, SearXNG started with `--name searxng --network
silvann`, and SiLVANN with `--network silvann` — it finds SearXNG at `http://searxng:8080` (`ASSUMED`: tried under
Podman alone).

**Tested** on node03 under Ubuntu 22.04's Podman 3.4: a network of their own, the other way containers find each other,
lost the chat's port there when SiLVANN runs as you (`--userns=keep-id`, `MEASURED`); the pod keeps both.

## Each app a name under localhost

A browser sends every name ending in `.localhost` to this machine by itself, so the apps in a pod can each have a name
instead of a port. One small proxy in the pod holds its only published port and passes each name to its app; the apps
keep their own ports inside the pod, so they never collide with each other's or the machine's:

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
podman run -d --pod silvann --name silvann --group-add keep-groups --device /dev/kfd --device /dev/dri \
       -v ~/SiLVANN:/data ghcr.io/eval-apply-labs/silvann:0-amd
podman run -d --pod silvann --name caddy -v ~/.config/silvann/caddy/Caddyfile:/etc/caddy/Caddyfile:ro \
       docker.io/library/caddy:2.8.4
```

with SearXNG's settings as [above](#searxng-a-search-engine-of-your-own). The chat is then at
`http://silvann.localhost:8000`, SearXNG at `http://searxng.localhost:8000`, and `http://localhost:8000` lists them.
Another app joins as a container in the pod and three lines in the file, then `podman restart caddy`.

**A name is also a wall.** A browser keeps cookies and stored data by name, not by port: two apps at `localhost:8080`
and `localhost:8090` share them, while `silvann.localhost` and `searxng.localhost` each keep their own, and one app's
page cannot read another's session.

**Without the port in the address**, the pod publishes port 80. Rootless Podman binds no port below 1024 unless the
system allows it: `sudo sysctl net.ipv4.ip_unprivileged_port_start=80` (a line in `/etc/sysctl.d/` keeps it), then
`-p 127.0.0.1:80:8000` on the pod, and the chat is at `http://silvann.localhost` (`ASSUMED`: not tried). Under Docker,
which has no pods, the containers share one network with `--network container:caddy` (`ASSUMED`).

**Tested** on node03 under Ubuntu 22.04's Podman 3.4, with curl, which sends `*.localhost` to the loopback as the
browsers do: the chat's page and its websocket through `silvann.localhost`, SearXNG's search through
`searxng.localhost`, the list at `localhost`, a 404 for any other name, and the host holding port 8000 alone
(`MEASURED`, measurements/2026-10-27_localhost_names_node03_artifacts/pod.log).

## A card for each container

Each container can be given its own card, so two models — or two parts of one — never share one:

- **AMD:** hand in the devices whole and name the card by its number on the host, as `rocm-smi` lists them:
  `--device /dev/kfd --device /dev/dri -e ROCR_VISIBLE_DEVICES=1`. The container sees that card alone, as its card 0.
  A card's render node alone (`--device /dev/dri/renderD129`) is not enough: ROCm 6.1 then sees no card at all
  (`MEASURED` on node03).
- **NVIDIA:** `--device nvidia.com/gpu=1` hands in that card alone (`ASSUMED` for a second card; `gpu=0` gave the
  Dell Precision's one card, `MEASURED`).
- **No card:** no device flags, and `"card": "cpu"` in the model's config — the CPU and its RAM run it.

A model split over several cards of one container (`"num_cards": 2`): [`docs/config.md`](../docs/config.md).

## A model over several containers

A model can run as a ring of machines, each holding a run of its layers
([`docs/users.md` §6](../docs/users.md#6-a-model-over-several-machines)); each container is one machine of the ring.
In one pod they reach each other at `127.0.0.1`; on several machines, publish each one's ring port. The first (layer 0)
serves the conversations; each runs the server itself:

```bash
podman pod create --name ring --userns=keep-id -p 127.0.0.1:8765:8765
podman run -d --pod ring --name ring-cpu -v ~/SiLVANN-cpu:/data \
       ghcr.io/eval-apply-labs/silvann:0-amd /opt/silvann/venv/bin/python /opt/silvann/start_silvann_server.py \
       --model /data/models/<model> /data/ring.json --host 127.0.0.1 --port 8081
podman run -d --pod ring --name ring-card -v ~/SiLVANN:/data --group-add keep-groups \
       --device /dev/kfd --device /dev/dri -e ROCR_VISIBLE_DEVICES=0 \
       ghcr.io/eval-apply-labs/silvann:0-amd /opt/silvann/venv/bin/python /opt/silvann/start_silvann_server.py \
       --model /data/models/<model> /data/ring.json --host 0.0.0.0 --port 8765
```

with `~/SiLVANN/ring.json`:

```json
{"max_context": 8192, "card": 0, "pipeline_parallelism": "yes", "pipeline_parallelism_layers": "0-19",
 "pipeline_parallelism_next": "127.0.0.1:8767", "pipeline_parallelism_port": 8766}
```

and `~/SiLVANN-cpu/ring.json` the same with `"card": "cpu"`, the layers `"20-39"`, the next `127.0.0.1:8766` and the
port `8767`. The chat reaches the ring as another machine's server: Options → Send conversations to → remote,
`http://localhost:8765`, from a UI on the host. `"card": 0` is the one card the container sees.

**Tested** on node03: the Qwen 3.6 35B-A3B so, its first 20 layers on one MI50 and the rest on the CPU in a container
with no card, at 21.7 tokens a second (`MEASURED`, measurements/2026-10-26_nn119_v0200_image_node03.md). The image's own
UI in the pod, and an AMD and an NVIDIA image in one ring: `ASSUMED`, not yet tried.

## Servers that find each other

The registry of servers lets the SiLVANN servers of a network find each other for the model's subagents
([`docs/tools.md`](../docs/tools.md#servers-found-through-a-registry)). It is in the same image, under another command.
On the machine that keeps it:

```bash
mkdir -p ~/SiLVANN-registry
podman pod create --name silvann-registry --userns=keep-id -p 8770:8770
podman run -d --pod silvann-registry --name registry -v ~/SiLVANN-registry:/data \
       ghcr.io/eval-apply-labs/silvann:0-amd silvann registry
```

It writes its key into `~/SiLVANN-registry/registry/key` when it first starts; copy that file to each machine as `key`.
A server offered to the registry must be reachable by other machines, so it runs on its own, published on the
network — not the chat's own server, which stays inside its container. On each machine, with that machine's address:

```bash
podman pod create --name silvann-server --userns=keep-id -p 8765:8765
podman run -d --pod silvann-server --name server -v ~/SiLVANN:/data --group-add keep-groups \
       --device /dev/kfd --device /dev/dri \
       -e SILVANN_TOOL_AGENT_REGISTRY=http://192.0.2.10:8770 -e SILVANN_TOOL_AGENT_KEY="$(cat key)" \
       -e SILVANN_TOOL_AGENT_ADDRESS=http://192.0.2.20:8765 \
       ghcr.io/eval-apply-labs/silvann:0-amd silvann serve <model> --host 0.0.0.0
```

`SILVANN_TOOL_AGENT_ADDRESS` is the machine's address and published port: inside a container the server's own address
is the container's, which other machines cannot reach. Point the chat at this server with Options → Send conversations
to → remote. The server's log says `offered to the registry … as …` as it starts; stopped, it withdraws its offer.

**Tested** on node03: the registry and two servers each in a pod of their own, from the v0.21.0 image — Qwen 3.8 27B
on one MI50, Qwen 3.6 35B-A3B on the other, each reaching the others only at the machine's address, as another machine
would: both offered themselves, the 35B ran the 27B as a subagent from `(whoami)`'s listing alone, and the 27B's pod
stopped left the listing (`MEASURED`, measurements/2026-10-27_v0210_two_servers_node03.md).

## Videos, by hand

What `install_nonfree.sh` ([`README.md`](README.md#what-the-image-leaves-out)) runs, for a container set up by hand:
PyAV into the folder, under `python/`, where it stays through a new image —

```bash
podman run --rm --userns=keep-id -v ~/SiLVANN:/data --entrypoint /opt/silvann/venv/bin/python \
       ghcr.io/eval-apply-labs/silvann:0-amd -c "import subprocess, sys; \
  sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--target', '/data/python/%d.%d' % sys.version_info[:2], \
  '-r', '/opt/silvann/requirements-nonfree.txt']))"
```

then stop and start the container. `install.sh`, outside a container, installs it with the rest (`SILVANN_NONFREE=0`
leaves it out).

## Building an image

From the release's folder. A build takes `install.sh` and the image's two scripts from the folder, and the release
from a repository, as `install.sh` fetches it — GitHub's by default:

```bash
podman build -f docker/Dockerfile.amd -t silvann:amd .                                  # AMD, gfx906
podman build -f docker/Dockerfile.amd --build-arg AMD_ARCH=gfx90a -t silvann:amd-gfx90a .   # another AMD architecture
podman build -f docker/Dockerfile -t silvann:nvidia .                                   # NVIDIA
podman build -f docker/Dockerfile --build-arg SILVANN_SOURCE=<a repository> --build-arg SILVANN_REF=<branch> \
             -t silvann:nvidia .                                                        # another repository or branch
```

A checkout with changes of its own goes in through a repository it is pushed to: the build mounts nothing from the
machine, which Ubuntu 22.04's Podman (3.4) does not support. The published images are built by
`.github/workflows/images.yml` on a release's tag. `setup.sh --image silvann:amd` sets up a container of an image of
your own, as it does the published one.
