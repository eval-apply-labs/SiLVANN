#!/usr/bin/env python3
"""opencl_kernel_source.py — every package's kernels as one OpenCL C program, for `khronos_opencl2`.

    python3 scripts/opencl_kernel_source.py <out.c>

The other families compile the kernels from `<pkg>/gpu/doors.cuh` as their own language (HIP, CUDA). An
OpenCL runtime compiles OpenCL C, at run time, from source — so this writes that source, from the same
rows, and writes it as a C string the family's binary is linked with:

  1. FLATTEN what a family's binary takes of each package with doors — its `manifest__gpu.cuh`, with the
     doors swapped for the compute bodies (`gpu/kernels/kernels.cuh`) — through the C preprocessor, as
     OpenCL C 2.0 (`__OPENCL_C_VERSION__` defined, which hides the host's door table and launch).
  2. WRITE one `__kernel` per row of the package's door list, running the body its `__RUNS` line names.
     A pointer argument arrives as its address — a 64-bit number, the same on both sides under shared
     virtual memory — and is turned back into a `__global` pointer, so the launch sets every argument by
     its bytes and never needs to know which were pointers.
  3. CHECK it with ROCm's OpenCL compiler for gfx906 when one is installed, so a body OpenCL C cannot take
     fails the build here rather than the first launch on somebody's card.

The spellings OpenCL C lacks are mapped in the prelude: `constexpr` -> `const`, `static_assert`,
`alignas`/`alignof` to their C11 forms, and the `<stdint.h>` names to OpenCL's own types.
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(os.path.dirname(HERE), 'src')
PK   = os.path.join(SRC, 'packages')
CLANG = '/opt/rocm/llvm/bin/clang'

FAKE = {
    'stdint.h': ('#pragma once\n'
                 'typedef ulong uint64_t; typedef uint uint32_t; typedef ushort uint16_t; typedef uchar uint8_t;\n'
                 'typedef long int64_t; typedef int int32_t; typedef short int16_t; typedef char int8_t;\n'
                 'typedef ulong uintptr_t; typedef long intptr_t;\n'
                 '#define UINT64_MAX 0xFFFFFFFFFFFFFFFFul\n#define UINT32_MAX 0xFFFFFFFFu\n'),
    'stddef.h': '#pragma once\n', 'stdbool.h': '#pragma once\n', 'math.h': '#pragma once\n',
    'string.h': '#pragma once\n', 'stdlib.h': '#pragma once\n',
}
SPELLINGS = ('#define constexpr const\n#define static_assert _Static_assert\n'
             '#define alignas(n) __attribute__((aligned(n)))\n#define alignof(t) __alignof__(t)\n'
             '#define __device__\n#define __noinline__ __attribute__((noinline))\n')
# ⛳ nn's arithmetic dial is read here the way a family reads it: its defaults file first, then one
#   definition per function with the choice inside it. `SILVANN_NN_ARITHMETIC_FAST` overrides it, as it
#   does for the other families' compilers.
ARITHMETIC = ('#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n'
              '#include "%s"\n' % os.path.join(PK, 'nn', 'contracts', 'defaults.cuh') +
              'static inline float nn__silicon__expf(float x) {\n'
              '#if NN__ARITHMETIC__FAST\n    return native_exp(x);\n#else\n    return exp(x);\n#endif\n}\n'
              'static inline float nn__silicon__sqrtf(float x) { return sqrt(x); }\n'
              'static inline float nn__silicon__logf(float x) {\n'
              '#if NN__ARITHMETIC__FAST\n    return native_log(x);\n#else\n    return log(x);\n#endif\n}\n'
              'static inline float nn__silicon__cosf(float x) { return cos(x); }\n'
              'static inline float nn__silicon__sinf(float x) { return sin(x); }\n'
              'static inline uint nn__silicon__lane(void)  { return (uint)get_local_id(0); }\n'
              'static inline uint nn__silicon__lanes(void) { return (uint)get_local_size(0); }\n'
              'static inline uint nn__silicon__block(void)  { return (uint)get_group_id(0); }\n'
              'static inline uint nn__silicon__blocks(void) { return (uint)get_num_groups(0); }\n'
              'static inline float nn__silicon__lanes_sum(float part) { return work_group_reduce_add(part); }\n'
              'static inline float nn__silicon__lanes_max(float part) { return work_group_reduce_max(part); }\n'
              'static inline void nn__silicon__lanes_sums(float* parts, uint n) {\n'
              '    for (uint k = 0u; k < n; ++k) parts[k] = work_group_reduce_add(parts[k]);\n}\n'
              'static inline float nn__silicon__half_to_float(ushort h) { return vload_half(0, (const half*)&h); }\n'
              'static inline float nn__silicon__dot2(uint a, uint b, float c) {\n'
              '    return (c + nn__silicon__half_to_float((ushort)(a & 0xffffu)) * nn__silicon__half_to_float((ushort)(b & 0xffffu)))\n'
              '         + nn__silicon__half_to_float((ushort)(a >> 16)) * nn__silicon__half_to_float((ushort)(b >> 16));\n}\n'
              # ⛳ the tile names nn's own type, so the contract that defines it comes first
              '#include "%s"\n' % os.path.join(PK, 'nn', 'contracts', 'abi', 'gpu.cuh') +
              'static inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {1u, 4u}; return t; }\n')


def packages():
    """Package names in roster order, from the generated card-side manifest."""
    text = open(os.path.join(PK, 'manifest__gpu.cuh'), encoding='utf-8').read()
    return re.findall(r'^#include "([a-z0-9_]+)/manifest__gpu\.cuh"', text, re.M)


def flatten(with_doors):
    # ⛳ PHASE ZERO FIRST, IN THE CARD-SIDE MANIFEST'S OWN ORDER: the package list, each package's contract
    #   lists, and the kinds generated from them — what every header here may name.
    top = open(os.path.join(PK, 'manifest__gpu.cuh'), encoding='utf-8').read()
    zero = re.findall(r'^#include "([^"]+)"', top[:top.index('/manifest__gpu.cuh"')], re.M)
    lines = [SPELLINGS, ARITHMETIC] + ['#include "%s"' % os.path.join(PK, f) for f in zero if not f.endswith('/manifest__gpu.cuh')]
    for p in with_doors:
        own = open(os.path.join(PK, p, 'manifest__gpu.cuh'), encoding='utf-8').read()
        for inc in re.findall(r'^#include "([^"]+)"', own, re.M):
            inc = 'gpu/kernels/kernels.cuh' if inc == 'gpu/doors.cuh' else inc
            lines.append('#include "%s"' % os.path.join(PK, p, inc))
    tmp = tempfile.mkdtemp()
    for name, body in FAKE.items():
        open(os.path.join(tmp, name), 'w').write(body)
    tu = os.path.join(tmp, 'tu.c')
    open(tu, 'w').write('\n'.join(lines) + '\n')
    fast = os.environ.get('SILVANN_NN_ARITHMETIC_FAST')
    dial = ['-DNN__ARITHMETIC__FAST=%s' % fast] if fast else []
    r = subprocess.run(['gcc', '-E', '-P', '-x', 'c', '-nostdinc', '-undef', '-D__OPENCL_C_VERSION__=200'] + dial +
                       ['-I', tmp, tu], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit('⛔ the card side would not preprocess as OpenCL C:\n' + r.stderr[:2000])
    return r.stdout


def rows(p):
    text = open(os.path.join(PK, p, 'contracts', 'abi', 'gpu.cuh'), encoding='utf-8').read()
    return re.findall(r'X\(PKG,\s*void,\s*\w+,\s*(\w+),\s*\(([^)]*)\)\)', text)


def runs(p):
    text = open(os.path.join(PK, p, 'gpu', 'doors.cuh'), encoding='utf-8').read()
    out = {}
    for fn, form, body, blocks, threads, names in re.findall(
            r'^#define (\w+)__RUNS\s+(\w+),\s+(\w+),\s+(.+?),\s+(\w+),\s+\(([^)]*)\)\s*$', text, re.M):
        out[fn] = (form, body, [n.strip() for n in names.split(',')])
    return out


FIRST = 'nn__silicon__lane() == 0u && nn__silicon__block() == 0u'


def kernel(fn, params, run):
    form, body, names = run
    kparams, unpack = [], []
    for prm in [x.strip() for x in params.split(',')]:
        name = re.findall(r'(\w+)\s*$', prm)[0]
        ctype = prm[:len(prm) - len(name)].strip()
        if ctype.endswith('*'):
            kparams.append('ulong %s__address' % name)
            unpack.append('    __global %s %s = (__global %s)%s__address;' % (ctype, name, ctype, name))
        else:
            kparams.append('%s %s' % (ctype, name))
    rest = ', '.join(names[2:])
    call = {'NN__GPU__ZZPRIVATE_WIDE':     '(void)%s(%s);' % (body, ', '.join(names)),
            'NN__GPU__ZZPRIVATE_PLAIN':    '(void)%s(%s);' % (body, ', '.join(names)),
            'NN__GPU__ZZPRIVATE_AS_INT':   'const uint64_t answer = (uint64_t)(uint32_t)%s(%s);\n'
                                           '    if (%s) *%s = %s | answer;' % (body, rest, FIRST, names[0], names[1]),
            'NN__GPU__ZZPRIVATE_AS_FLOAT': 'const uint64_t answer = (uint64_t)nn__result__zzpackage_float_bits((float)%s(%s));\n'
                                           '    if (%s) *%s = %s | answer;' % (body, rest, FIRST, names[0], names[1])}.get(form)
    if call is None:
        sys.exit('⛔ %s: its __RUNS form %s is not one this generator knows' % (fn, form))
    # ⛳ A SERIAL BODY RUNS ON ONE LANE OF ONE BLOCK; a WIDE one on every lane, and a value body on every
    #   lane with the first one writing, as the doors say.
    guard = '    if (!(%s)) return;\n' % FIRST if form == 'NN__GPU__ZZPRIVATE_PLAIN' else ''
    return ('__kernel void %s__kernel(%s) {\n%s\n%s    %s\n}\n'
            % (fn, ', '.join(kparams), '\n'.join(unpack), guard, call))


def main():
    if len(sys.argv) != 2: sys.exit(__doc__)
    with_doors = [p for p in packages() if os.path.isfile(os.path.join(PK, p, 'gpu', 'doors.cuh'))]
    source = flatten(with_doors)
    made = 0
    for p in with_doors:
        r = runs(p)
        for fn, params in rows(p):
            if fn not in r: sys.exit('⛔ %s has a door row and no __RUNS line' % fn)
            source += kernel(fn, params, r[fn]); made += 1
    if os.path.exists(CLANG):
        tmp = tempfile.mkdtemp(); cl = os.path.join(tmp, 'k.cl'); open(cl, 'w').write(source)
        c = subprocess.run([CLANG, '-x', 'cl', '-cl-std=CL2.0', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx906',
                            '-fsyntax-only', '-w', cl], capture_output=True, text=True)
        if c.returncode != 0:
            sys.exit('⛔ the kernels are not valid OpenCL C:\n' + c.stderr[:3000])
    lit = ''.join('    "%s\\n"\n' % l.replace('\\', '\\\\').replace('"', '\\"') for l in source.split('\n'))
    with open(sys.argv[1], 'w') as f:
        f.write('/* GENERATED by scripts/opencl_kernel_source.py — do not edit, and do not check in. */\n'
                'extern "C" const char khronos_opencl2__kernel_source[];\n'
                'extern "C" const char khronos_opencl2__kernel_source[] =\n%s;\n' % lit)
    print('✅ %d kernels from %s, %d lines of OpenCL C%s' % (made, ', '.join(with_doors), source.count('\n'),
          ', checked for gfx906' if os.path.exists(CLANG) else ''))


if __name__ == '__main__':
    main()
