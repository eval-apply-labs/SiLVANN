#!/usr/bin/env python3
"""opencl1_kernel_source.py — every package's kernels as one OpenCL C 1.2 program, for `khronos_opencl1_igpu`.

    python3 scripts/opencl1_kernel_source.py <out.c>

OpenCL C 1.2 has no generic address space: every pointer says where what it points at lives — `__global`
(the card's memory), `__constant`, `__local` (a block's shared memory) or `__private` (a lane's own, which
an unqualified pointer means). nn's bodies are written once for every family, with plain pointers, and
OpenCL C 2.0 takes them as generic ones. So this writes the 1.2 program from the same flattened source as
`opencl_kernel_source.py` and works out each pointer's space, rather than nn's source naming one:

  1. PARSE the flattened card side (pycparser, `scripts/third_party/`), with one plain C function per door
     row standing in for its kernel.
  2. INFER, from the kernels down: a kernel's pointer arguments are the card's memory; an array declared in
     a body is the lane's own; a table at the program's top is `__constant`. Each pointer the program
     declares — a parameter, a local, a return, a cast — is unified with every pointer that flows into it
     (an initialiser, an assignment, an argument, a return). Two spaces meeting in one pointer stop the
     build with the function's name: one function would need a copy for each.
  3. AN ADDRESS IS AN ARENA AND AN OFFSET. A 1.2 card has no shared virtual memory, so the family hands
     out addresses inside a few large buffers (`khronos_opencl1_igpu/sys.cuh`) and every kernel is given those
     buffers. Wherever the program turns a number into a pointer — a kernel's argument, an address a table
     carries — it becomes the buffer's pointer plus the offset.
  4. THE FOLDS through the block's shared memory: `nn__silicon__lanes_sum` and its kin take a `__local`
     scratch the kernel declares, handed down to every function that reaches one.
  5. WRITE the program, and check it with an OpenCL C 1.2 compiler when one is on the machine (ROCm's clang, as the
     2.0 generator does).

`OPENCL1_PROGRAM=<file>` also writes the program as plain OpenCL C, to read or to hand a compiler.
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'third_party'))
sys.path.insert(0, HERE)
from pycparser import c_ast, c_generator, c_parser   # noqa: E402
import opencl_kernel_source as two                      # noqa: E402  the 2.0 generator: the flatten and the door rows

FAMILY = 'khronos_opencl1_igpu'
ARENAS = 32            # buffers a kernel is handed; the family's `KHRONOS_OPENCL1_IGPU__ARENAS`
LANES_MAX = 256        # nn's `NN__SILICON__LANES_MAX`: the fold's scratch, one float a lane
AT = FAMILY + '__zzprivate_at'
G, C, P = '__global', '__constant', '__private'

# the names a parser needs that OpenCL C has built in; parsed, never written
PARSE_ONLY = ('typedef unsigned long ulong; typedef unsigned int uint; typedef unsigned short ushort;\n'
              'typedef unsigned char uchar; typedef _Bool bool; typedef unsigned short half;\n')
# ⛳ `__noinline__` RIDES THROUGH THE PARSER AS `_Noreturn`, the one function specifier it keeps that nn's card side never
#   uses, and is written back as the attribute
SPELLINGS = ('#define constexpr const\n#define static_assert _Static_assert\n'
             '#define alignas(n) __attribute__((aligned(n)))\n#define alignof(t) _Alignof(t)\n'
             '#define __device__\n#define __noinline__ _Noreturn\n')
# what the bodies call of the family, as the parser sees it; the definitions are written in their place
SILICON = {
    'nn__silicon__expf':  'float nn__silicon__expf(float x);',
    'nn__silicon__sqrtf': 'float nn__silicon__sqrtf(float x);',
    'nn__silicon__logf':  'float nn__silicon__logf(float x);',
    'nn__silicon__cosf':  'float nn__silicon__cosf(float x);',
    'nn__silicon__sinf':  'float nn__silicon__sinf(float x);',
    'nn__silicon__lane':  'uint nn__silicon__lane(void);',
    'nn__silicon__lanes': 'uint nn__silicon__lanes(void);',
    'nn__silicon__block':  'uint nn__silicon__block(void);',
    'nn__silicon__blocks': 'uint nn__silicon__blocks(void);',
    'nn__silicon__lanes_sum':  'float nn__silicon__lanes_sum(float part);',
    'nn__silicon__lanes_max':  'float nn__silicon__lanes_max(float part);',
    'nn__silicon__lanes_sums': 'void nn__silicon__lanes_sums(float* parts, uint n);',
    'nn__silicon__half_to_float': 'float nn__silicon__half_to_float(ushort h);',
    'nn__silicon__dot2': 'float nn__silicon__dot2(uint a, uint b, float c);',
}
FOLDS = ('nn__silicon__lanes_sum', 'nn__silicon__lanes_max', 'nn__silicon__lanes_sums')
# ⛳ THE FOLDS WAIT FOR THE WHOLE BLOCK, as `work_group_reduce_add` does in the 2.0 family: every lane of the block reaches
#   each one, which nn's bodies already promise. The order of the sum is a tree over the lanes, the same every run.
DEFINITIONS = r'''
static inline float nn__silicon__expf(float x) {
#if NN__ARITHMETIC__FAST
    return native_exp(x);
#else
    return exp(x);
#endif
}
static inline float nn__silicon__sqrtf(float x) { return sqrt(x); }
static inline float nn__silicon__logf(float x) {
#if NN__ARITHMETIC__FAST
    return native_log(x);
#else
    return log(x);
#endif
}
static inline float nn__silicon__cosf(float x) { return cos(x); }
static inline float nn__silicon__sinf(float x) { return sin(x); }
static inline uint nn__silicon__lane(void)  { return (uint)get_local_id(0); }
static inline uint nn__silicon__lanes(void) { return (uint)get_local_size(0); }
static inline uint nn__silicon__block(void)  { return (uint)get_group_id(0); }
static inline uint nn__silicon__blocks(void) { return (uint)get_num_groups(0); }
static inline float nn__silicon__lanes_sum(float part, __local float* zz_lanes) {
    const uint i = (uint)get_local_id(0), n = (uint)get_local_size(0);
    uint span = 1u;
    while (span < n) span <<= 1;
    zz_lanes[i] = part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (span >>= 1; span > 0u; span >>= 1) {
        if (i < span && i + span < n) zz_lanes[i] = zz_lanes[i] + zz_lanes[i + span];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float all = zz_lanes[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return all;
}
static inline float nn__silicon__lanes_max(float part, __local float* zz_lanes) {
    const uint i = (uint)get_local_id(0), n = (uint)get_local_size(0);
    uint span = 1u;
    while (span < n) span <<= 1;
    zz_lanes[i] = part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (span >>= 1; span > 0u; span >>= 1) {
        if (i < span && i + span < n) zz_lanes[i] = fmax(zz_lanes[i], zz_lanes[i + span]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float all = zz_lanes[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return all;
}
static inline void nn__silicon__lanes_sums(float* parts, uint n, __local float* zz_lanes) {
    for (uint k = 0u; k < n; ++k) parts[k] = nn__silicon__lanes_sum(parts[k], zz_lanes);
}
static inline float nn__silicon__half_to_float(ushort h) { return vload_half(0, (const half*)&h); }
static inline float nn__silicon__dot2(uint a, uint b, float c) {
    return (c + nn__silicon__half_to_float((ushort)(a & 0xffffu)) * nn__silicon__half_to_float((ushort)(b & 0xffffu)))
         + nn__silicon__half_to_float((ushort)(a >> 16)) * nn__silicon__half_to_float((ushort)(b >> 16));
}
'''
# an address the family handed out: its arena in the bits above the offset, as `khronos_opencl1_igpu/sys.cuh` writes it
AT_DEFINITION = ('static inline __global uchar* %s(__global uchar* const* zz_arena, ulong a) {\n'
                 '    return a == 0ul ? (__global uchar*)0 : zz_arena[(a >> 40) & %dul] + (a & 0xFFFFFFFFFFul);\n}\n'
                 % (AT, ARENAS - 1))


def preprocess(text, tmp, dial):
    tu = os.path.join(tmp, 'tu.c')
    open(tu, 'w').write(text)
    r = subprocess.run([two.CPP, '-E', '-P', '-x', 'c', '-nostdinc', '-undef', '-D__OPENCL_C_VERSION__=120'] + dial +
                       ['-I', tmp, tu], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit('⛔ the card side would not preprocess as OpenCL C 1.2:\n' + r.stderr[:2000])
    return r.stdout


def flatten(with_doors, dial):
    """The card side as one C text: the same includes as the 2.0 generator, the family's functions as prototypes."""
    top = open(os.path.join(two.PK, 'manifest__gpu.cuh'), encoding='utf-8').read()
    zero = re.findall(r'^#include "([^"]+)"', top[:top.index('/manifest__gpu.cuh"')], re.M)
    first = [s for n, s in SILICON.items()]
    lines = [SPELLINGS, '#include "%s"' % os.path.join(two.PK, 'nn', 'contracts', 'defaults.cuh')] + first + [
        '#include "%s"' % os.path.join(two.PK, 'nn', 'contracts', 'abi', 'gpu.cuh'),
        'nn__gemm__tile nn__silicon__gemm_tile(void);']
    lines += ['#include "%s"' % os.path.join(two.PK, f) for f in zero if not f.endswith('/manifest__gpu.cuh')]
    for p in with_doors:
        own = open(os.path.join(two.PK, p, 'manifest__gpu.cuh'), encoding='utf-8').read()
        for inc in re.findall(r'^#include "([^"]+)"', own, re.M):
            inc = 'gpu/kernels/kernels.cuh' if inc == 'gpu/doors.cuh' else inc
            lines.append('#include "%s"' % os.path.join(two.PK, p, inc))
    tmp = tempfile.mkdtemp()
    for name, body in two.FAKE.items():
        open(os.path.join(tmp, name), 'w').write(body)
    return preprocess('\n'.join(lines) + '\n', tmp, dial), tmp


def kernel(fn, params, run):
    """A door row's kernel as a plain C function `<fn>__kernel`: its addresses arrive as numbers and become pointers."""
    form, body, names = run
    kparams, unpack = [], []
    for prm in [x.strip() for x in params.split(',')]:
        name = re.findall(r'(\w+)\s*$', prm)[0]
        ctype = prm[:len(prm) - len(name)].strip()
        if ctype.endswith('*'):
            kparams.append('ulong %s__address' % name)
            unpack.append('    %s %s = (%s)%s__address;' % (ctype, name, ctype, name))
        else:
            kparams.append('%s %s' % (ctype, name))
    rest = ', '.join(names[2:])
    call = {'NN__GPU__ZZPRIVATE_WIDE':     '(void)%s(%s);' % (body, ', '.join(names)),
            'NN__GPU__ZZPRIVATE_PLAIN':    '(void)%s(%s);' % (body, ', '.join(names)),
            'NN__GPU__ZZPRIVATE_AS_INT':   'const uint64_t answer = (uint64_t)(uint32_t)%s(%s);\n'
                                           '    if (%s) *%s = %s | answer;' % (body, rest, two.FIRST, names[0], names[1]),
            'NN__GPU__ZZPRIVATE_AS_FLOAT': 'const uint64_t answer = (uint64_t)nn__result__zzpackage_float_bits((float)%s(%s));\n'
                                           '    if (%s) *%s = %s | answer;' % (body, rest, two.FIRST, names[0], names[1])}.get(form)
    if call is None:
        sys.exit('⛔ %s: its __RUNS form %s is not one this generator knows' % (fn, form))
    guard = '    if (!(%s)) return;\n' % two.FIRST if form == 'NN__GPU__ZZPRIVATE_PLAIN' else ''
    return 'void %s__kernel(%s) {\n%s\n%s    %s\n}\n' % (fn, ', '.join(kparams), '\n'.join(unpack), guard, call)


# ── THE SPACES ───────────────────────────────────────────────────────────────────────────────────────────────
class Space:
    """One pointer level's space, unknown until something fixes it; spaces that must agree are merged."""
    __slots__ = ('up', 'fixed', 'why')

    def __init__(self, fixed=None, why=''):
        self.up, self.fixed, self.why = self, fixed, why

    def root(self):
        s = self
        while s.up is not s:
            s.up = s.up.up
            s = s.up
        return s


class Clash(Exception):
    pass


def unify(a, b, where):
    if a is None or b is None:
        return
    for x, y in zip(a, b):
        rx, ry = x.root(), y.root()
        if rx is ry:
            continue
        if rx.fixed and ry.fixed and rx.fixed != ry.fixed:
            raise Clash('%s: a pointer is both %s (%s) and %s (%s)' % (where, rx.fixed, rx.why, ry.fixed, ry.why))
        if rx.fixed is None:
            rx, ry = ry, rx
        ry.up = rx


def depth(t):
    """How many pointer or array levels a declared type has."""
    n = 0
    while isinstance(t, (c_ast.PtrDecl, c_ast.ArrayDecl)):
        n, t = n + 1, t.type
    return n


def fresh(n, fixed=None, why=''):
    return [Space(fixed, why) for _ in range(n)]


def is_null(e):
    return isinstance(e, c_ast.Constant) and re.fullmatch(r'0[uUlL]*', e.value) is not None or \
        isinstance(e, c_ast.Cast) and depth(e.to_type.type) > 0 and is_null(e.expr)


class Program:
    def __init__(self, ast, structs):
        self.ast, self.structs = ast, structs
        self.funcs = {e.decl.name: e for e in ast.ext if isinstance(e, c_ast.FuncDef)}
        self.globals = {e.name: e for e in ast.ext if isinstance(e, c_ast.Decl) and e.name
                        and not isinstance(e.type, c_ast.FuncDecl)}
        self.chain = {}        # id(Decl or Cast) -> [Space]
        self.ret = {}          # function -> [Space] of what it returns
        self.casts = {}        # function -> [Cast] of a number into a pointer
        self.calls = {}        # function -> {callee}
        self.folds = set()     # functions calling a fold directly

    # ── declarations ──
    def declare(self, d, local):
        n = depth(d.type)
        if n == 0:
            return None
        if isinstance(d.type, c_ast.ArrayDecl):
            # an array is where it is declared: a body's is the lane's own, the program's top `__constant`; an array of
            # pointers keeps its elements' spaces open
            levels, t = [], d.type
            while isinstance(t, c_ast.ArrayDecl):
                levels.append(Space(P if local else C, 'the array %s' % d.name))
                t = t.type
            levels += fresh(depth(t))
            c = levels
        else:
            c = fresh(n)
        self.chain[id(d)] = c
        return c

    # ── expressions ──
    def storage(self, e, scope):
        """The space an lvalue lives in."""
        if isinstance(e, c_ast.ID):
            d = self.lookup(e.name, scope)
            if d is not None and d.name in self.globals and self.globals[d.name] is d:
                return Space(C, 'the table %s' % d.name)
            return Space(P, 'the variable %s' % e.name)
        if isinstance(e, c_ast.ArrayRef):
            c = self.type_of(e.name, scope)
            return c[0] if c else Space(P, 'an element')
        if isinstance(e, c_ast.UnaryOp) and e.op == '*':
            c = self.type_of(e.expr, scope)
            return c[0] if c else Space(P, 'a dereference')
        if isinstance(e, c_ast.StructRef):
            if e.type == '->':
                c = self.type_of(e.name, scope)
                return c[0] if c else Space(P, 'a member')
            return self.storage(e.name, scope)
        return Space(P, 'a value')

    def member(self, e, scope):
        """A struct member's own levels, when it is an array."""
        name = e.field.name
        for fields in self.structs.values():
            if name in fields:
                n = fields[name]
                if n == 0:
                    return None
                return [self.storage(e, scope)] + fresh(n - 1)
        return None

    def lookup(self, name, scope):
        for frame in reversed(scope):
            if name in frame:
                return frame[name]
        return self.globals.get(name)

    def type_of(self, e, scope):
        fn = scope[0]['__function__']
        if e is None:
            return None
        if isinstance(e, c_ast.ID):
            d = self.lookup(e.name, scope)
            if d is None:
                return None
            if id(d) not in self.chain and d.name in self.globals and self.globals[d.name] is d:
                self.declare(d, False)
            return self.chain.get(id(d))
        if isinstance(e, c_ast.Constant):
            return None
        if isinstance(e, c_ast.UnaryOp):
            if e.op == '&':
                inner = e.expr
                if isinstance(inner, c_ast.ArrayRef):
                    self.type_of(inner.subscript, scope)
                    return self.type_of(inner.name, scope)
                if isinstance(inner, c_ast.UnaryOp) and inner.op == '*':
                    return self.type_of(inner.expr, scope)
                below = self.type_of(inner, scope) or []
                return [self.storage(inner, scope)] + below
            c = self.type_of(e.expr, scope)
            if e.op == '*':
                return c[1:] if c and len(c) > 1 else None
            if e.op in ('p++', 'p--', '++', '--'):
                return c
            return None
        if isinstance(e, c_ast.BinaryOp):
            l, r = self.type_of(e.left, scope), self.type_of(e.right, scope)
            if e.op in ('+', '-'):
                if l and not r:
                    return l
                if r and not l:
                    return r
            return None
        if isinstance(e, c_ast.Assignment):
            l, r = self.type_of(e.lvalue, scope), self.type_of(e.rvalue, scope)
            if e.op == '=' and l and r:
                unify(l, r, fn)
            return l
        if isinstance(e, c_ast.TernaryOp):
            self.type_of(e.cond, scope)
            a, b = self.type_of(e.iftrue, scope), self.type_of(e.iffalse, scope)
            if a and b:
                unify(a, b, fn)
            return a or b
        if isinstance(e, c_ast.Cast):
            inner = self.type_of(e.expr, scope)
            n = depth(e.to_type.type)
            if n == 0:
                return None
            if inner is None and not is_null(e.expr):
                # ⭐ A NUMBER BECOMES A POINTER ONLY AS AN ADDRESS THE FAMILY HANDED OUT: the card's memory, an arena's
                c = [Space(G, 'an address')] + fresh(n - 1)
                self.casts.setdefault(fn, []).append(e)
            elif inner is None:
                c = fresh(n)
            else:
                c = inner[:n] + fresh(max(0, n - len(inner)))
            self.chain[id(e)] = c
            return c
        if isinstance(e, c_ast.FuncCall):
            name = e.name.name if isinstance(e.name, c_ast.ID) else None
            args = e.args.exprs if e.args else []
            got = [self.type_of(a, scope) for a in args]
            if name in self.funcs:
                self.calls.setdefault(fn, set()).add(name)
                params = self.params(name)
                for i, (pc, ac) in enumerate(zip(params, got)):
                    if pc and ac:
                        unify(pc, ac, '%s, calling %s (argument %d)' % (fn, name, i + 1))
                return self.ret.get(name)
            if name in FOLDS:
                self.folds.add(fn)
                if name == 'nn__silicon__lanes_sums' and got and got[0]:
                    unify(got[0], [Space(P, 'the folds\' parts')], '%s, calling %s' % (fn, name))
            return None
        if isinstance(e, c_ast.ArrayRef):
            self.type_of(e.subscript, scope)
            c = self.type_of(e.name, scope)
            return c[1:] if c and len(c) > 1 else None
        if isinstance(e, c_ast.StructRef):
            self.type_of(e.name, scope)
            return self.member(e, scope)
        if isinstance(e, c_ast.ExprList):
            c = None
            for x in e.exprs:
                c = self.type_of(x, scope)
            return c
        if isinstance(e, c_ast.InitList):
            for x in e.exprs:
                self.type_of(x, scope)
            return None
        if isinstance(e, c_ast.CompoundLiteral):
            self.type_of(e.init, scope)
            return None
        if isinstance(e, (c_ast.Typename, c_ast.TypeDecl)):
            return None
        for _, child in e.children():
            self.type_of(child, scope)
        return None

    # ── statements ──
    def params(self, name):
        f = self.funcs[name]
        if name not in self.ret:
            self.ret[name] = fresh(depth(f.decl.type.type))
            for p in self.param_decls(f):
                self.declare(p, True)
        return [self.chain.get(id(p)) for p in self.param_decls(f)]

    @staticmethod
    def param_decls(f):
        ps = f.decl.type.args.params if f.decl.type.args else []
        return [p for p in ps if isinstance(p, c_ast.Decl)]

    def walk(self, s, scope):
        fn = scope[0]['__function__']
        if s is None:
            return
        if isinstance(s, c_ast.Compound):
            scope.append({})
            for item in s.block_items or []:
                self.walk(item, scope)
            scope.pop()
        elif isinstance(s, c_ast.Decl):
            if isinstance(s.type, c_ast.FuncDecl):
                return
            c = self.declare(s, True)
            scope[-1][s.name] = s
            if s.init is not None:
                if isinstance(s.init, c_ast.InitList) and c and isinstance(s.type, c_ast.ArrayDecl):
                    for x in s.init.exprs:
                        xc = self.type_of(x, scope)
                        if xc and len(c) > 1:
                            unify(c[1:], xc, fn)
                else:
                    ic = self.type_of(s.init, scope)
                    if c and ic:
                        unify(c, ic, '%s, %s' % (fn, s.name))
        elif isinstance(s, c_ast.DeclList):
            for d in s.decls:
                self.walk(d, scope)
        elif isinstance(s, c_ast.For):
            scope.append({})
            if isinstance(s.init, c_ast.DeclList):
                self.walk(s.init, scope)
            else:
                self.type_of(s.init, scope)
            self.type_of(s.cond, scope)
            self.type_of(s.next, scope)
            self.walk(s.stmt, scope)
            scope.pop()
        elif isinstance(s, c_ast.Return):
            c = self.type_of(s.expr, scope)
            if c:
                unify(self.ret[fn], c, '%s, its return' % fn)
        elif isinstance(s, (c_ast.If, c_ast.While, c_ast.DoWhile, c_ast.Switch)):
            self.type_of(s.cond, scope)
            for attr in ('iftrue', 'iffalse', 'stmt'):
                if hasattr(s, attr):
                    self.walk(getattr(s, attr), scope)
        elif isinstance(s, (c_ast.Case, c_ast.Default)):
            if isinstance(s, c_ast.Case):
                self.type_of(s.expr, scope)
            for x in s.stmts or []:
                self.walk(x, scope)
        elif isinstance(s, c_ast.Label):
            self.walk(s.stmt, scope)
        elif isinstance(s, (c_ast.Break, c_ast.Continue, c_ast.EmptyStatement, c_ast.Goto, c_ast.Pragma,
                            c_ast.StaticAssert)):
            return
        else:
            self.type_of(s, scope)

    def reach(self, roots):
        """The functions the kernels reach, through a plain read of their calls."""
        seen, todo = set(), list(roots)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)

            class V(c_ast.NodeVisitor):
                def visit_FuncCall(self, node):
                    if isinstance(node.name, c_ast.ID) and node.name.name in prog.funcs:
                        todo.append(node.name.name)
                    self.generic_visit(node)
            prog = self
            V().visit(self.funcs[n].body)
        return seen

    def infer(self, roots):
        self.reached = self.reach(roots)
        # ⛳ A FUNCTION IS READ AFTER EVERYTHING IT CALLS, so a call's answer has its spaces already
        order, done = [], set()

        def visit(n):
            if n in done:
                return
            done.add(n)
            for c in sorted(self.callees(n)):
                visit(c)
            order.append(n)
        for r in sorted(roots):
            visit(r)
        for n in order:
            self.params(n)
            scope = [{'__function__': n}]
            for p in self.param_decls(self.funcs[n]):
                scope[0][p.name] = p
            self.walk(self.funcs[n].body, scope)
        # the kernels' addresses and the folds, carried up to every caller
        self.needs_arena = {n for n in self.reached if self.casts.get(n)}
        self.needs_lanes = set(self.folds)
        grew = True
        while grew:
            grew = False
            for n in self.reached:
                cs = self.calls.get(n, set())
                if n not in self.needs_arena and cs & self.needs_arena:
                    self.needs_arena.add(n); grew = True
                if n not in self.needs_lanes and cs & self.needs_lanes:
                    self.needs_lanes.add(n); grew = True

    def callees(self, n):
        out = set()

        class V(c_ast.NodeVisitor):
            def visit_FuncCall(self, node):
                if isinstance(node.name, c_ast.ID) and node.name.name in funcs:
                    out.add(node.name.name)
                self.generic_visit(node)
        funcs = self.funcs
        V().visit(self.funcs[n].body)
        out.discard(n)
        return out


# ── WRITING IT ───────────────────────────────────────────────────────────────────────────────────────────────
def qualify(t, chain):
    """Put each level's space onto the type that level points at; a lane's own is what no qualifier means."""
    level = 0
    while isinstance(t, (c_ast.PtrDecl, c_ast.ArrayDecl)) and level < len(chain):
        s = chain[level].root().fixed
        inner = t.type
        if isinstance(t, c_ast.PtrDecl) and s in (G, C) and hasattr(inner, 'quals') and s not in inner.quals:
            inner.quals.append(s)
        t, level = inner, level + 1


def constant_top(d):
    """A variable at the program's top lives in `__constant`, the only space 1.2 lets it have."""
    t = d.type
    while isinstance(t, (c_ast.PtrDecl, c_ast.ArrayDecl)):
        if isinstance(t, c_ast.PtrDecl):
            sys.exit('⛔ %s: a pointer at the program\'s top — OpenCL C 1.2 has no space for it' % d.name)
        t = t.type
    if C not in t.quals:
        t.quals.append(C)


def hidden_param(text):
    return c_parser.CParser().parse(PARSE_ONLY + 'void f(%s);' % text).ext[-1].type.args.params[0]


def rewrite(prog, roots):
    lanes_param = hidden_param('float* zz_lanes')
    lanes_param.type.type.quals.append('__local')
    arena_param = hidden_param('uchar* const* zz_arena')
    arena_param.type.type.type.quals.append(G)
    # the spaces, onto every declaration and cast in reach
    for n in prog.reached:
        f = prog.funcs[n]
        qualify(f.decl.type.type, prog.ret.get(n, []))

        class V(c_ast.NodeVisitor):
            def visit_Decl(self, node):
                if id(node) in prog.chain:
                    qualify(node.type, prog.chain[id(node)])
                self.generic_visit(node)

            def visit_Cast(self, node):
                if id(node) in prog.chain:
                    qualify(node.to_type.type, prog.chain[id(node)])
                self.generic_visit(node)

            def visit_FuncCall(self, node):
                self.generic_visit(node)
                name = node.name.name if isinstance(node.name, c_ast.ID) else None
                extra = []
                if name in prog.needs_arena:
                    extra.append(c_ast.ID('zz_arena'))
                if name in prog.needs_lanes or name in FOLDS:
                    extra.append(c_ast.ID('zz_lanes'))
                if extra:
                    if node.args is None:
                        node.args = c_ast.ExprList([])
                    node.args.exprs.extend(extra)
        V().visit(f)
        for p in prog.param_decls(f):
            if id(p) in prog.chain:
                qualify(p.type, prog.chain[id(p)])
        # ⭐ A NUMBER INTO A POINTER: the arena's pointer plus the offset
        for cast in prog.casts.get(n, []):
            cast.expr = c_ast.FuncCall(c_ast.ID(AT), c_ast.ExprList([c_ast.ID('zz_arena'), cast.expr]))
        if n not in roots:
            args = f.decl.type.args
            if args is None or (len(args.params) == 1 and isinstance(args.params[0], c_ast.Typename)):
                f.decl.type.args = args = c_ast.ParamList([])
            if n in prog.needs_arena:
                args.params.append(arena_param)
            if n in prog.needs_lanes:
                args.params.append(lanes_param)


def mentions(e, names):
    found = []

    class V(c_ast.NodeVisitor):
        def visit_ID(self, node):
            if node.name in names:
                found.append(node.name)
    V().visit(e)
    return bool(found)


def write(prog, roots, ext_skip, raw_typedefs, definitions):
    gen = c_generator.CGenerator()
    out, written = [], set()
    for i, e in enumerate(prog.ast.ext):
        if i < ext_skip:
            continue
        if isinstance(e, c_ast.Decl) and isinstance(e.type, c_ast.FuncDecl):
            if e.name in SILICON or e.name == 'nn__silicon__gemm_tile':
                if e.name not in written:
                    written.add(e.name)
                    if e.name == 'nn__silicon__expf':
                        out.append(definitions + AT_DEFINITION)
                    if e.name == 'nn__silicon__gemm_tile':
                        out.append('static inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {1u, 4u}; return t; }\n')
                continue
            # ⛳ A PROTOTYPE IS WRITTEN FROM ITS DEFINITION, which carries the spaces and the hidden parameters
            if e.name in prog.funcs and e.name in prog.reached:
                out.append(gen.visit(prog.funcs[e.name].decl) + ';\n')
            continue
        if isinstance(e, c_ast.StaticAssert) and mentions(e.cond, prog.globals):
            # ⛳ A CHECK ON A TABLE'S VALUE IS THE HOST'S: there the table is a constant expression, here a `__constant`
            #   variable, which C does not fold
            continue
        if isinstance(e, c_ast.FuncDef):
            n = e.decl.name
            if n not in prog.reached:
                continue
            text = gen.visit(e)
            if n in roots:
                params = ', '.join('__global uchar* zz_a%d' % k for k in range(ARENAS))
                text = re.sub(r'^void %s\((.*?)\)\n\{' % re.escape(n),
                              lambda m: '__kernel void %s(%s%s)\n{\n%s%s' % (
                                  n, m.group(1) + ', ' if m.group(1) not in ('', 'void') else '', params,
                                  '  __global uchar* zz_arena[%d] = {%s};\n' % (ARENAS, ', '.join('zz_a%d' % k for k in range(ARENAS)))
                                  if n in prog.needs_arena else '',
                                  '  __local float zz_lanes[%d];\n' % LANES_MAX if n in prog.needs_lanes else ''),
                              text, count=1, flags=re.S)
            out.append(text)
            continue
        if isinstance(e, c_ast.Typedef) and e.name in raw_typedefs:
            out.append(raw_typedefs[e.name])
            continue
        if isinstance(e, c_ast.Decl) and e.name and e.name in prog.globals:
            constant_top(e)
        out.append(gen.visit(e) + (';\n' if not isinstance(e, (c_ast.FuncDef, c_ast.Pragma)) else '\n'))
    text = ''.join(out)
    return text.replace('_Noreturn ', '__attribute__((noinline)) ')


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with_doors = [p for p in two.packages() if os.path.isfile(os.path.join(two.PK, p, 'gpu', 'doors.cuh'))]
    fast = os.environ.get('SILVANN_NN_ARITHMETIC_FAST')
    dial = ['-DNN__ARITHMETIC__FAST=%s' % fast] if fast else []
    source, tmp = flatten(with_doors, dial)
    roots = []
    for p in with_doors:
        r = two.runs(p)
        for fn, params in two.rows(p):
            if fn not in r:
                sys.exit('⛔ %s has a door row and no __RUNS line' % fn)
            if r[fn][0] == 'NN__GPU__ZZPRIVATE_BY_ROW':
                continue
            source += kernel(fn, params, r[fn])
            roots.append(fn + '__kernel')
    # ⛳ A TYPE WITH AN ALIGNMENT IS WRITTEN AS IT WAS: the parser takes no attributes, so it reads one without and the
    #   writer puts back the original
    raw = {}

    def strip(m):
        raw[m.group(2)] = m.group(0) + '\n'
        return m.group(0).replace(m.group(1), '')
    source = re.sub(r'typedef struct (__attribute__\(\(aligned\(\d+\)\)\)) \{[^{}]*\} (\w+);', strip, source)
    if '__attribute__' in source:
        sys.exit('⛔ an attribute the 1.2 generator does not carry: ' +
                 source[source.index('__attribute__') - 60:source.index('__attribute__') + 60])
    parse_text = PARSE_ONLY + source
    try:
        ast = c_parser.CParser().parse(parse_text, '<card side>')
    except Exception as err:   # noqa: BLE001 — the parser's own message names the line
        m = re.search(r':(\d+):', str(err))
        line = int(m.group(1)) if m else 0
        sys.exit('⛔ the card side would not parse: %s\n%s' % (err, '\n'.join(parse_text.split('\n')[max(0, line - 3):line + 2])))
    skip = len(c_parser.CParser().parse(PARSE_ONLY).ext)
    structs = {}
    for e in ast.ext:
        t = e.type if isinstance(e, (c_ast.Typedef, c_ast.Decl)) else None
        while t is not None and not isinstance(t, c_ast.Struct):
            t = getattr(t, 'type', None)
        if isinstance(t, c_ast.Struct) and t.decls:
            structs[t.name or e.name] = {d.name: depth(d.type) for d in t.decls}
            for d in t.decls:
                if isinstance(d.type, c_ast.PtrDecl):
                    sys.exit('⛔ the struct %s holds a pointer (%s): a 1.2 card cannot be handed one' % (t.name or e.name, d.name))
    prog = Program(ast, structs)
    try:
        prog.infer(roots)
    except Clash as clash:
        sys.exit('⛔ %s — this function would need a copy for each space' % clash)
    rewrite(prog, set(roots))
    definitions = preprocess('#include "%s"\n%s' % (os.path.join(two.PK, 'nn', 'contracts', 'defaults.cuh'), DEFINITIONS), tmp, dial)
    text = write(prog, set(roots), skip, raw, definitions)
    if os.path.exists(two.CLANG):
        cl = os.path.join(tmp, 'k.cl'); open(cl, 'w').write(text)
        c = subprocess.run([two.CLANG, '-x', 'cl', '-cl-std=CL1.2', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx906',
                            '-fsyntax-only', '-w', cl], capture_output=True, text=True)
        if c.returncode != 0:
            sys.exit('⛔ the kernels are not valid OpenCL C 1.2:\n' + c.stderr[:3000])
    lit = ''.join('    "%s\\n"\n' % l.replace('\\', '\\\\').replace('"', '\\"') for l in text.split('\n'))
    with open(sys.argv[1], 'w') as f:
        f.write('/* GENERATED by scripts/opencl1_kernel_source.py — do not edit, and do not check in. */\n'
                'extern "C" const char %s__kernel_source[];\n'
                'extern "C" const char %s__kernel_source[] =\n%s;\n'
                '/* how many kernels it holds: the family sizes its table of made kernels from this */\n'
                'extern "C" const unsigned %s__kernel_count;\n'
                'extern "C" const unsigned %s__kernel_count = %du;\n' % (FAMILY, FAMILY, lit, FAMILY, FAMILY, len(roots)))
    print('✅ %d kernels from %s, %d functions in reach, %d lines of OpenCL C 1.2; %d into an arena, %d through the folds%s' % (
        len(roots), ', '.join(with_doors), len(prog.reached), text.count('\n'), len(prog.needs_arena), len(prog.needs_lanes),
        ', checked for gfx906' if os.path.exists(two.CLANG) else ''))
    if os.environ.get('OPENCL1_PROGRAM'):
        open(os.environ['OPENCL1_PROGRAM'], 'w').write(text)


if __name__ == '__main__':
    main()
