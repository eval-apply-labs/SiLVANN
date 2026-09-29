# Attribution and third-party notices

> Not legal advice. Every identification below comes with the command that produced it (§5), so it can be
> re-derived instead of re-researched.

## 1. SiLVANN

SiLVANN — the Eval-Apply GPU Interpreter and the Neural Network library — is licensed under the **GNU Affero
General Public License, version 3** ([`LICENSE`](LICENSE)). Running a modified version as a network service
obliges you to offer its source to that service's users. A commercial licence is available on request.

## 2. Third-party code in the source tree

**None.** The engine, the runtime and the build scripts are original work; no third-party source file is
vendored. The release is source only: each user builds the engine on their own machine, so no vendor binary
is redistributed.

## 3. What the build compiles against, and what the runtime imports

Headers and libraries the builder installs themselves:

| component | licence | used by |
|---|---|---|
| pybind11 3.0 | BSD-3-Clause | the evaluator's Python module (`engine/src/pybind.cuh`) |
| AMD ROCm / HIP runtime | MIT (HIP headers and runtime) | the `amd_rocm6_wave64` family |
| NVIDIA CUDA toolkit | NVIDIA CUDA EULA | the `nvidia_cuda12` family, built only where `nvcc` is installed; it links the CUDA runtime statically, so a binary of it may be redistributed only under that EULA's terms |
| Khronos OpenCL headers | Apache-2.0 | the `khronos_opencl2` family |

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
# third-party source in the tree (expect: nothing)
grep -rlE "Copyright|SPDX-License" engine/src runtime
# the headers the build includes
grep -rhoE "#include <(CL/[a-z_.]+|hip/[a-z_.]+|cuda[a-z_.]*|pybind11/[a-z_.]+)>" engine/src | sort -u
# the Python packages' licences, from their installed metadata
python3 -c "import importlib.metadata as md; [print(n, md.metadata(n).get('License-Expression') or md.metadata(n).get('License')) for n in ('numpy','pybind11','tokenizers','huggingface_hub','fastapi','uvicorn','pydantic','httpx','nicegui')]"
```
