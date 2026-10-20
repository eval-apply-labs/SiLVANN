# Third-party code carried in the tree

| what | version | licence | used by |
|---|---|---|---|
| `pycparser/` — a C parser in Python, with its own copy of PLY | 2.22, unchanged from its source release (`pip download pycparser==2.22 --no-binary :all:`) | BSD-3-Clause, `pycparser/LICENSE`; PLY 3.9 in `pycparser/ply/`, BSD-3-Clause, the notice in each file's header | `scripts/opencl1_kernel_source.py`, which writes nn's kernels as OpenCL C 1.2 for `khronos_opencl1_igpu` |
| `OpenCL-Headers/CL/` — `cl.h`, `cl_platform.h`, `cl_version.h`, the Khronos Group's OpenCL C API headers | v2022.01.04, unchanged (Ubuntu 22.04's `opencl-c-headers` 3.0~2022.01.04-1, which packages Khronos' tag of that name) | Apache-2.0, `OpenCL-Headers/LICENSE`, the notice in each file's header | `release/build.py`, which builds the OpenCL families on Windows: no Windows toolchain carries them, and an OpenCL driver installs `OpenCL.dll` without them. On Linux the system's own are used |

They are carried rather than installed because a family is built on the machine it runs on (`release/install.sh`,
`release/install.ps1`), and the build should need nothing the machine does not already have.
