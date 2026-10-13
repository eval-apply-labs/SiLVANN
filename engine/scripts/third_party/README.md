# Third-party code carried in the tree

| what | version | licence | used by |
|---|---|---|---|
| `pycparser/` — a C parser in Python, with its own copy of PLY | 2.22, unchanged from its source release (`pip download pycparser==2.22 --no-binary :all:`) | BSD-3-Clause, `pycparser/LICENSE`; PLY 3.9 in `pycparser/ply/`, BSD-3-Clause, the notice in each file's header | `scripts/opencl1_kernel_source.py`, which writes nn's kernels as OpenCL C 1.2 for `khronos_opencl1_igpu` |

It is carried rather than installed because the family is built on the machine it runs on (`release/install.sh`), and the
build should need nothing the machine does not already have.
