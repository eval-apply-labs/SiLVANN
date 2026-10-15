# syntax=docker/dockerfile:1.7
# SiLVANN's image: the release installed by install.sh — the one line a Linux machine runs — with the families for
# every NVIDIA card nvidia_cuda12 lists and for the CPU (x86_avx2), then copied onto a plain Ubuntu 22.04 that holds no
# compiler. The models and the conversations live in a volume at /data; the UI is the app, on port 8080.
#
#   from the release's folder:
#   docker build -f docker/Dockerfile -t silvann .                                   GitHub's default branch
#   docker build -f docker/Dockerfile --build-arg SILVANN_REF=<branch> -t silvann .  a branch of it
#   docker build -f docker/Dockerfile --build-context source=<a git checkout of a release> \
#                --build-arg SILVANN_SOURCE=/source --build-arg SILVANN_REF=<branch> -t silvann .
#                                                                                    a checkout on this machine
#   docker run --gpus all -p 127.0.0.1:8080:8080 -v silvann-data:/data silvann     ▶ docker/README.md, its scripts
#
# CUDA's runtime is linked into the family (`-cudart static`) and the container is handed the host's driver
# (`--gpus all`, or CDI), so the final image carries no CUDA toolkit. What it holds, and each part's licence:
# /opt/silvann/IMAGE_NOTICES.md.

ARG UBUNTU=ubuntu:22.04

# A checkout to install from instead of a URL (`--build-context source=…`); empty unless one is given.
FROM scratch AS source

# ── the build: install.sh with the compilers ─────────────────────────────────────────────────────────────────────────
FROM ${UBUNTU} AS build
ARG DEBIAN_FRONTEND=noninteractive
# CUDA 12.9's compiler and runtime headers from NVIDIA's repository (not its multi-GB devel image)
ARG CUDA_VERSION=12-9
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git g++ python3 python3-venv python3-dev \
 && curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-${CUDA_VERSION} cuda-cudart-dev-${CUDA_VERSION} \
        cuda-documentation-${CUDA_VERSION} \
 && rm -rf /var/lib/apt/lists/*

ARG SILVANN_SOURCE=https://github.com/eval-apply-labs/SiLVANN
ARG SILVANN_REF=
# the commit SILVANN_REF must be at, when the caller knows it (a workflow does): the build fails on any other
ARG SILVANN_COMMIT=
COPY install.sh /tmp/install.sh
RUN --mount=type=bind,from=source,target=/source \
    SILVANN_DIR=/opt/silvann SILVANN_BIN=/usr/local/bin \
    SILVANN_SOURCE=${SILVANN_SOURCE} SILVANN_REF=${SILVANN_REF} \
    SILVANN_SILICON_FAMILIES="x86_avx2 nvidia_cuda12" SILVANN_NONFREE=0 \
    SILVANN_NVCC=/usr/local/cuda/bin/nvcc \
    bash /tmp/install.sh \
 && cd /opt/silvann \
 && test -f lib/silicon_families/libsilvann_nvidia_cuda12.so \
 && test -f lib/silicon_families/libsilvann_x86_avx2.so \
 && commit=$(git rev-parse HEAD) \
 && { [ -z "${SILVANN_COMMIT}" ] || [ "$commit" = "${SILVANN_COMMIT}" ] \
      || { echo "SILVANN_REF is at $commit, not SILVANN_COMMIT ${SILVANN_COMMIT}"; exit 1; }; } \
 && printf '%s\n%s\n' "$commit" "${SILVANN_SOURCE}" > SOURCE_COMMIT \
 && mkdir -p notices \
 && cp /usr/local/cuda/EULA.txt notices/CUDA_EULA.txt \
 && /usr/local/cuda/bin/nvcc --version | tail -2 > notices/CUDA_VERSION.txt \
 && rm -rf .git engine/build build.log \
 && find . -path ./venv -prune -o -name __pycache__ -prune -exec rm -rf {} +

# ── the image: Python, the release, no compiler ──────────────────────────────────────────────────────────────────────
FROM ${UBUNTU}
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --create-home --shell /bin/bash silvann \
 && mkdir -p /data && chown silvann:silvann /data
COPY --from=build /opt/silvann /opt/silvann
COPY docker/image_notices.py /opt/silvann/docker/image_notices.py
COPY docker/entrypoint.sh /opt/silvann/docker/entrypoint.sh
# requirements-nonfree.txt is not in the image: the scripts' `install_nonfree` installs it from PyPI into the volume, under
# /data/python/<the image's Python version>, which silvann-data.pth puts on the environment's path when it exists
RUN ln -s /opt/silvann/silvann /usr/local/bin/silvann \
 && ln -s /opt/silvann/silvann_cli /usr/local/bin/silvann_cli \
 && chmod 755 /opt/silvann/docker/entrypoint.sh \
 && /opt/silvann/venv/bin/python /opt/silvann/docker/image_notices.py > /opt/silvann/IMAGE_NOTICES.md \
 && /opt/silvann/venv/bin/python -c "import site, sys; open(site.getsitepackages()[0] + '/silvann-data.pth', 'w').write('/data/python/%d.%d\n' % sys.version_info[:2])"

# the container toolkit hands in the driver's compute libraries and nvidia-smi
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SILVANN_DATA=/data \
    NICEGUI_STORAGE_PATH=/data/.nicegui \
    PYTHONUNBUFFERED=1 \
    HOME=/home/silvann
# no USER: the entrypoint starts as root, gives what the volume holds to silvann (a model pasted in through Windows'
# Explorer comes in as root's), and runs the app as silvann
WORKDIR /opt/silvann
VOLUME /data
EXPOSE 8080
LABEL org.opencontainers.image.title="SiLVANN" \
      org.opencontainers.image.description="The Eval-Apply GPU Interpreter and its Neural Network library, with the chat UI" \
      org.opencontainers.image.source="https://github.com/eval-apply-labs/SiLVANN" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
# tini passes `docker stop`'s SIGTERM to the UI, which saves the conversations and stops the server before it exits
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/silvann/docker/entrypoint.sh"]
CMD ["/opt/silvann/venv/bin/python", "/opt/silvann/silvann_ui.py", "--local", "--host", "0.0.0.0", "--port", "8080", \
     "--no-browser"]
