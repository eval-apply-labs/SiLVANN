# Attribution and third-party notices

> Not legal advice. Every identification below comes with the command that produced it (§5), so it can be
> re-derived instead of re-researched.

## 1. SiLVANN

SiLVANN — the Eval-Apply GPU Interpreter and the Neural Network library — is licensed under the **GNU Affero
General Public License, version 3** ([`LICENSE`](LICENSE)). Running a modified version as a network service
obliges you to offer its source to that service's users. A commercial licence is available on request.

### Additional permission under GNU AGPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it with the NVIDIA CUDA runtime or the other
redistributable components of the NVIDIA CUDA Toolkit (or modified versions of them), containing parts covered by the
terms of the NVIDIA CUDA Toolkit End User License Agreement, the licensors of this Program grant you additional
permission to convey the resulting work.

The licensor is Eval-Apply, the Owner under `CLA.md`, through which every contribution is licensed to it. The permission
is what lets a binary of the `nvidia_cuda12` family — which links the CUDA runtime statically — be conveyed with the
rest of SiLVANN under the AGPL, as the Docker image (§6) conveys one. The CUDA runtime itself stays under NVIDIA's terms.

## 2. Third-party code in the source tree

One parser, carried unchanged in `engine/scripts/third_party/` (its README says why). The engine, the runtime and
the other build scripts are original work. The release is source: `install.sh` builds the engine on the user's own
machine, so it redistributes no vendor binary. The Docker image is the exception, and §6 says what it carries.

| component | version | licence | its notice | used by |
|---|---|---|---|---|
| pycparser (Eli Bendersky) | 2.22 | BSD-3-Clause | `engine/scripts/third_party/pycparser/LICENSE` | `engine/scripts/opencl1_kernel_source.py`, which writes nn's kernels as OpenCL C 1.2 for the `khronos_opencl1_igpu` family |
| PLY (David M. Beazley), the copy pycparser carries | 3.9 | BSD-3-Clause | the header of each file in `engine/scripts/third_party/pycparser/ply/` | pycparser |

## 3. What the build compiles against, and what the runtime imports

Headers and libraries the builder installs themselves:

| component | licence | used by |
|---|---|---|
| pybind11 3.0 | BSD-3-Clause | the evaluator's Python module (`engine/src/pybind.cuh`) |
| AMD ROCm / HIP runtime | MIT (HIP headers and runtime) | the `amd_rocm6_wave64` family |
| NVIDIA CUDA toolkit | NVIDIA CUDA EULA | the `nvidia_cuda12` family, built only where `nvcc` is installed; it links the CUDA runtime statically, so a binary of it may be redistributed only under that EULA's terms |
| Khronos OpenCL headers | Apache-2.0 | the `khronos_opencl2` and `khronos_opencl1_igpu` families |

Python packages the runtime imports (`requirements.txt`), installed by the user:

| package | licence |
|---|---|
| numpy | BSD-3-Clause |
| tokenizers | Apache-2.0 |
| huggingface_hub | Apache-2.0 |
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic | MIT |
| httpx | BSD-3-Clause |
| nicegui | MIT |
| Pillow | MIT-CMU (HPND) — a picture decoded and resized for the vision tower |
| pypdf | BSD-3-Clause — a PDF's text, for a file attached to a message |
| PyAV (`av`), in `requirements-nonfree.txt` — `install.sh` installs it; the Docker image does not carry it, and its scripts' `install_nonfree` fetches it from PyPI into the user's volume | BSD-3-Clause — a video file's frames. Its wheels carry FFmpeg's libraries (LGPL-3.0-or-later, as FFmpeg reports itself) and the codec libraries they link, x264 and x265 among them (GPL-2.0-or-later) |

## 4. Models

The model weights are not part of this repository and are not covered by its licence. Each published model
is a re-quantized repacking of its upstream checkpoint, with no further training, and keeps its authors'
licence, which ships in its folder:

| model | upstream authors | licence |
|---|---|---|
| Qwen 3.6 35B-A3B, Qwen 3.8 27B | the Qwen team, Alibaba Cloud | Apache-2.0 |
| GLM 5.3 Flash | Z.AI | MIT |

The model folders also carry the upstream tokenizer and configuration files unchanged; their licence is the
model's.

## 5. How to re-derive this file

```bash
# third-party source in the tree (expect: nothing in engine/src and runtime; pycparser and its PLY in engine/scripts)
grep -rlE "Copyright|SPDX-License" engine/src runtime engine/scripts
grep -h "__version__ =" engine/scripts/third_party/pycparser/__init__.py engine/scripts/third_party/pycparser/ply/__init__.py
# the headers the build includes
grep -rhoE "#include <(CL/[a-z_.]+|hip/[a-z_.]+|cuda[a-z_.]*|pybind11/[a-z_.]+)>" engine/src | sort -u
# the Python packages' licences, from their installed metadata
python3 -c "import importlib.metadata as md; [print(n, md.metadata(n).get('License-Expression') or md.metadata(n).get('License')) for n in ('numpy','pybind11','tokenizers','huggingface_hub','fastapi','uvicorn','pydantic','httpx','nicegui')]"
```

## 6. The Docker image

`ghcr.io/eval-apply-labs/silvann` (`docker/`) is the release built: what it holds is listed, each part with its licence,
in `/opt/silvann/IMAGE_NOTICES.md`, which `docker/image_notices.py` writes from what is installed when the image is
built.

| part | licence | its notice in the image |
|---|---|---|
| SiLVANN, its source at `/opt/silvann` and built there: the evaluator, `nvidia_cuda12`, `x86_avx2` | AGPL-3.0 with the §1 additional permission | `LICENSE`, `ATTRIBUTION.md`, and `SOURCE_COMMIT`: the commit of the public repository it was built from, which is its corresponding source |
| NVIDIA's CUDA runtime, linked statically into `libsilvann_nvidia_cuda12.so` | NVIDIA CUDA Toolkit EULA (a redistributable component) | `notices/CUDA_EULA.txt`, `notices/CUDA_VERSION.txt` |
| Ubuntu 22.04 and the packages installed on it (Python 3.10, tini, ca-certificates, and what they require) | each package's own; their source is Ubuntu's archive at the versions listed | `/usr/share/doc/<package>/copyright` |
| the Python environment `/opt/silvann/venv`: `requirements.txt` (§3) and what it requires | each package's own (§3) | each package's `.dist-info` |

Not in it: PyAV (`requirements-nonfree.txt`, §3), which the scripts' `install_nonfree` fetches from PyPI into the user's
volume; NVIDIA's driver, which the container toolkit hands in from the host when the container starts; the models.
