# Attribution and third-party notices

> Not legal advice. Every identification below comes with the command that produced it (§5), so it can be
> re-derived instead of re-researched.

## 1. SiLVANN

SiLVANN — the Eval-Apply GPU Interpreter and the Neural Network library — is licensed under the **GNU Affero
General Public License, version 3** ([`LICENSE`](LICENSE)). Running a modified version as a network service
obliges you to offer its source to that service's users. A commercial licence is available on request.

## 2. Third-party code in the source tree

One parser, carried unchanged in `engine/scripts/third_party/` (its README says why). The engine, the runtime and
the other build scripts are original work. The release is source only: each user builds the engine on their own
machine, so no vendor binary is redistributed.

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
