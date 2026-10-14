# SiLVANN in a container

On Linux, the one-line installer is the way we recommend (`README.md`). The Docker image is the supported second way,
and the way on Windows: SiLVANN, its environment and its build in one image, run by Docker Desktop (or Podman) on
WSL 2, with the chat UI in the browser.

- **The image:** `ghcr.io/eval-apply-labs/silvann:main`, the nightly, built from this repository's `main`; until v1.0
  the only one. ⚠ Not published yet: until it is, build it (below) and the scripts use the image built on the machine. From v1.0 each major version has its own tag (`:1`, then `:2`), which its minor and patch releases
  update in place: within a major an update keeps the models and the conversations working, and a change that would
  break them is the next major.
- **What it runs:** an NVIDIA card, through `nvidia_cuda12`, built for every compute capability that family lists (7.5,
  8.0, 8.6, 8.9, 9.0 and 12.0: the GeForce RTX 20 to 50 series among them; only the RTX 5060 Ti has run it), or the
  CPU alone, through `x86_avx2` (AVX2, FMA and F16C: Intel from 2013, AMD from 2015). AMD's cards are not in this
  image.
- **The data:** the models and the conversations live in a volume, `silvann-data`, mounted at `/data`. A new image runs
  over the same volume, so an update keeps them.
- **The UI** is the app: Options lists the models this release can fetch, downloads one from Hugging Face with its
  progress, and starts it. The model started last starts again when the container does.

## On Windows

1. **WSL 2:** in a PowerShell run as administrator, `wsl --install`, then restart.
2. **Docker Desktop** with its WSL 2 backend (docs.docker.com/desktop/setup/install/windows-install/). Docker Desktop is
   free for personal use and small companies; its licence page says which. To keep the models on an NVMe drive, set
   *Settings → Resources → Advanced → Disk image location* to it before the first download.
3. **NVIDIA's Windows driver**, the ordinary one. Nothing is installed inside Linux for the card.
4. **The scripts:** this folder's `windows/`. Double-click `silvann_install.cmd`. It checks the above, offers WSL 2
   more of the PC's memory (WSL 2 takes half by default, and a model with its experts in RAM needs more), pulls the
   image, checks that a container sees the card, and puts two shortcuts on the desktop: **SiLVANN** starts it and opens
   the browser, **SiLVANN - stop** stops it.

| script | what it does |
|---|---|
| `silvann_install` | the checks, the image, the volume, the shortcuts |
| `silvann_update` | the newest image of the same tag and its scripts, over the same volume; a running app is restarted on it |
| `silvann_start` | the app at http://localhost:8080, for this PC only |
| `silvann_start_shared` | the same on every network interface: other machines on the LAN reach the UI |
| `silvann_stop` | stops it; the UI saves every conversation first |
| `silvann_delete` | removes the container and the image, and the volume only after a second yes |
| `silvann_export` | stops it and writes the image and the volume, the models included, into a folder: a portable copy |
| `silvann_import` | loads such a folder on this PC or another, which then starts as it was — refused where SiLVANN is installed already: export that one first, or delete it |
| `silvann_install_nonfree` | fetches what the image leaves out, `requirements-nonfree.txt` (below), from PyPI into the volume: a container started for it, then stopped; a running app is restarted |

⚠ Shared means every machine that reaches this PC on port 8080 reaches the UI, and the UI has no login: a
conversation's password is its only lock.

A Hugging Face token makes the downloads faster: Hugging Face serves a client that is not signed in more slowly. Save
one in the UI's Options (it is kept in the volume, so an export carries it: **forget** it before giving an export to
someone else), or set `HF_TOKEN` before `silvann_start`, which hands it to the container.

## On Linux

Docker Engine or Podman, and for the card NVIDIA's container toolkit
(docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then the same actions:

```bash
bash docker/silvann_docker.sh install          # also: update, start, start_shared, stop, delete, export <dir>, import <dir>,
                                               # install_nonfree
```

Or by hand:

```bash
docker run -d --name silvann --gpus all -p 127.0.0.1:8080:8080 -v silvann-data:/data --stop-timeout 300 \
    ghcr.io/eval-apply-labs/silvann:main
```

Podman hands in the card as `--device nvidia.com/gpu=all` (CDI, `nvidia-ctk cdi generate`). Without a card, leave the
card's flag out and the CPU runs the models.

## What the image leaves out

`requirements-nonfree.txt` lists what SiLVANN reads more with and runs without, whose wheels carry binaries under the
LGPL and the GPL, which the image does not redistribute: today PyAV, through which a video file other than an animated
picture (MP4 and the like) is read; without it an animated GIF, WebP or PNG is still read, and such a file is refused
with its reason. `install_nonfree` fetches it from PyPI into the volume, under `/data/python/<the image's Python
version>`, where the image's environment looks when it starts: it stays through updates, and an update to an image on
another Python needs `install_nonfree` again. An export carries it, since it is in the volume. `install.sh`, outside
Docker, installs it with the rest (`SILVANN_NONFREE=0` leaves it out).

## Building it

From this repository's folder:

```bash
docker build -f docker/Dockerfile -t ghcr.io/eval-apply-labs/silvann:main .                                   # main
docker build -f docker/Dockerfile --build-arg SILVANN_REF=<branch> -t ghcr.io/eval-apply-labs/silvann:main .  # a branch
```

Tagged with the published image's name, it is the one the scripts use; under another name, set `SILVANN_IMAGE` to it.

The build runs `install.sh`, as a Linux machine does, with the compilers in a first stage: the evaluator, CUDA 12.9's
family for every card it lists, and the CPU's. The image that comes out holds no compiler. `IMAGE_NOTICES.md` in it
(`/opt/silvann/IMAGE_NOTICES.md`) lists what it holds and each part's licence, and the commit it was built from.
