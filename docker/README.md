# SiLVANN in a container — the Dockerfile

From v0.12.0 SiLVANN publishes no image and keeps no scripts to run one: this folder holds the Dockerfile, for whoever
wants SiLVANN in a container of their own. The ways to run SiLVANN are its installers ([`README.md`](../README.md)):
`install.sh` on Linux, `install.ps1` on Windows.

What it builds: the release installed by `install.sh`, with the families for every NVIDIA card `nvidia_cuda12` lists
(compute capability 7.5 to 12.0, the GeForce RTX 20 to 50 series among them) and for the CPU (`x86_avx2`), copied onto a
plain Ubuntu 22.04 that holds no compiler; the models and the conversations in a volume at `/data`; the chat as the app
on port 8080, its Options downloading a model and starting it. AMD's cards are not in it. On Linux the image ran as fast
as the install — on an RTX 5060 Ti the Qwen 3.6 35B-A3B `D8E4` decoded at 55.2 and 55.5 tokens a second where the
install gave 54.6 to 55.7 — and under Docker Desktop on Windows well below what the PC gives: 37 tokens a second with
Ministral 3 14B `D4` on an RTX 5070 Ti, where the same version installed on Linux gave 46.

## Building it

From the release's folder. The build takes only `install.sh` and the image's two scripts from the folder:
`install.sh` fetches the release from GitHub, its default branch or `SILVANN_REF`'s. A checkout on this machine, with
its own changes, goes in as a build context of its own:

```bash
docker build -f docker/Dockerfile -t silvann .                                   # GitHub's main: the newest release
docker build -f docker/Dockerfile --build-arg SILVANN_REF=<branch> -t silvann .  # a branch
docker build -f docker/Dockerfile --build-context source=<a git checkout> --build-arg SILVANN_SOURCE=/source \
             --build-arg SILVANN_REF=<its branch> -t silvann .                    # a checkout here
```

The build runs `install.sh`, as a Linux machine does, with the compilers in a first stage: the evaluator, CUDA 12.9's
family for every card it lists, and the CPU's. `IMAGE_NOTICES.md` in the image (`/opt/silvann/IMAGE_NOTICES.md`) lists
what it holds and each part's licence, and the commit it was built from.

## Running it

```bash
docker run -d --name silvann --gpus all -p 127.0.0.1:8080:8080 -v silvann-data:/data --stop-timeout 300 silvann
```

The chat is at http://localhost:8080. NVIDIA's container toolkit hands in the card (`--gpus all`); Podman takes
`--device nvidia.com/gpu=all` (CDI, `nvidia-ctk cdi generate`). Without a card, leave the card's flag out and the CPU
runs the models. A new image run over the same volume keeps the models and the conversations; the UI saves every
conversation when the container stops, which `--stop-timeout` gives it time to. The model started last starts again
when the container does. `-p 127.0.0.1:8080:8080` keeps the UI to this machine: the UI has no login, and a
conversation's password is its only lock.

## What the image leaves out

`requirements-nonfree.txt` lists what SiLVANN reads more with and runs without, whose wheels carry binaries under the
LGPL and the GPL, which the image does not redistribute: today PyAV, through which a video file other than an animated
picture (MP4 and the like) is read; without it an animated GIF, WebP or PNG is still read, and such a file is refused
with its reason. The image's environment looks for it in the volume, under `/data/python/<its Python version>`, so it
stays through a new image:

```bash
docker run --rm -v silvann-data:/data --entrypoint /opt/silvann/venv/bin/python silvann -c "import subprocess, sys; \
  sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--target', '/data/python/%d.%d' % sys.version_info[:2], \
  '-r', '/opt/silvann/requirements-nonfree.txt']))"
```

then restart the container. `install.sh`, outside Docker, installs it with the rest (`SILVANN_NONFREE=0` leaves it out).
