"""THE MACHINE — the engine, booted once, with one environment and the conveniences every model needs.

The engine is `silvann_engine_cpu`: the evaluator runs on the host, and the arithmetic reaches a card through a
silicon family the wheel loads. `Machine` chooses the family, boots with the text a model gives it, and then offers
what the tests all wrote for themselves: run a program, make a buffer and bind it, fill a table, define a procedure,
read and write a card's memory.
"""
import importlib
import os
import sys

from .composer import Composer


class EngineError(Exception):
    pass


def _indent(text):
    """A program's text laid out a form a line: a newline before each `(` that opens a form inside `begin`."""
    out, depth = [], 0
    for i, ch in enumerate(text):
        if ch == "(" and depth >= 2 and i and text[i - 1] == " ":
            out.append("\n" + "  " * depth)
        out.append(ch)
        depth += (ch == "(") - (ch == ")")
    return "".join(out).replace(" \n", "\n")


class Machine:
    def __init__(self, silicon=None, engine="cpu"):
        self.programs = []             # every procedure and picture defined, in order, as text — ▶ `write_programs`
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # where the built engine sits: `lib/` in a release, the tree's root in development
        for at in (os.path.join(root, "lib"), root):
            if os.path.isdir(at) and at not in sys.path:
                sys.path.insert(0, at)
        self.e = importlib.import_module("silvann_engine_" + engine)
        # a card if any family drives one — its vendor's own first, then OpenCL — else the CPU's own cores. x86_avx2
        # counts no devices on a CPU without AVX2, which leaves the portable host family
        for family in ("amd_rocm6_wave64", "nvidia_cuda12", "khronos_opencl2", "x86_avx2"):
            if silicon is None and self.e.devices(family) > 0:
                silicon = family
        if silicon is not None:
            self.e.choose_silicon(silicon)
        self.silicon = silicon or "host"
        self.booted = False
        self._abi_write = None

    def boot(self, device_params, alloc_mb=256, workers=None, worker_params=None):
        """Boot with one worker (or the list given), each with the device parameters given, or its own section from
        `worker_params` — {worker: text}."""
        e = self.e
        text = "boot_commands{\nsys__alloc_size_mb:%d\n}\ndevice_params{\n%s}\n" % (alloc_mb, device_params)
        for w, params in sorted((worker_params or {}).items()):
            text += "worker%d_params{\n%s}\n" % (w, params)
        if workers:
            e.choose_workers(workers)
            e.boot(config=text, blocks=len(workers))
        else:
            e.boot(config=text)
        self.booted = True
        self.K, self.V = e.kinds(), e.verbs()
        self.env = e.bindings_create(512)
        self.comp = Composer(e, self.env)

    def act_as(self, worker):
        """What runs from here runs as `worker`: its buffers, its device, its tables. -1 is the default again."""
        self.e.act_as(worker)

    # ── running ──────────────────────────────────────────────────────────────────────────────────────
    def run(self, text):
        """A program's answer as (kind, value)."""
        return self.comp.run(text)

    def must(self, text, what, want="sys__value_int"):
        kind, value = self.comp.run(text)
        fault = self.e.sys.fault()
        if kind != self.K[want] or fault["raised"] != 0:
            self.e.sys.fault_clear()
            raise EngineError("%s did not run: kind %s value %s fault %s" % (what, kind, value, fault))
        return value

    def grid(self, text):
        """A program run with every block standing, so it can hand work to the others (`sys__compute`) and collect it
        (`sys__result`). Block 0 runs it, as the host's worker."""
        cell = self.comp.compose(text)
        try:
            pic = self.e.freeze(cell)
            self.e.act_as(-1)
            try:
                kind, value, ran = self.e.run_grid(self.env, pic[2])
            finally:
                self.e.release(pic[2])
        finally:
            self.e.release(cell[2])
        fault = self.e.sys.fault()
        if not ran or fault["raised"] != 0:
            self.e.sys.fault_clear()
            raise EngineError("the grid program did not run: kind %s value %s fault %s" % (kind, value, fault))
        return kind, value

    def bind(self, name, cell):
        """An object — a picture, a view — bound by name; the binding keeps its own hold and this one goes back."""
        e, K, V = self.e, self.K, self.V
        lst = e.list_create()
        for c in ((K["sys__standard"], V["sys__add_bindings"], 0), (K["sys__value_int"], 0, self.env),
                  self.comp._cell_quoted_name(name), cell):
            if not e.list_append(lst, *c):
                raise EngineError("an append was refused")
        e.release(cell[2])
        pic = e.freeze((K["sys__object_reference"], 0, lst))
        kind, _ = e.execute((K["sys__object_reference"], 0, self.env), pic)
        e.release(pic[2])
        e.release(lst)
        if kind != K["sys__value_true"]:
            raise EngineError("binding %r was refused (kind %s)" % (name, kind))

    def picture(self, name, text):
        """A program frozen and bound by name — what `sys__compute` hands another block."""
        cell = self.comp.compose(text)
        pic = self.e.freeze(cell)
        self.e.release(cell[2])
        self.bind(name, pic)
        self.programs.append(";; picture %s — run by another worker when a program hands it over\n%s" % (name, text))

    def view(self, name):
        """A read-only snapshot of the bindings as they stand, bound by name — the names a computed block sees."""
        e, K, V = self.e, self.K, self.V
        self.bind(name, e.create_executable([(K["sys__standard"], V["sys__bindings__viewonly"], 0),
                                             (K["sys__object_reference"], 0, self.env)]))

    def defun(self, text):
        self.must(text, "the procedure %r" % text[:60], "sys__value_true")
        self.programs.append(text)

    def write_programs(self, path, heading):
        """The procedures and pictures defined so far, as the Lisp they were written in, one after another — what a
        model IS in this engine, readable. ⛳ Written from the text each was defined with; nothing is reconstructed."""
        with open(path, "w") as f:
            f.write(";; %s\n;; %d programs, in the order they were defined\n\n" % (heading, len(self.programs)))
            for text in self.programs:
                f.write(_indent(text) + "\n\n")

    def loader(self):
        """`nn_load_bundle.Loader` over this machine: the verbs it runs are built as cells, as a C caller would."""
        from .model_folder import NL
        e, K, V = self.e, self.K, self.V
        counted = {K["sys__object_reference"], K["sys__quoted_list"], K["sys__procedure_reference"]}

        def cells(*cs):
            lst = e.list_create()
            seen = set()
            for k, verb, val in cs:
                if not e.list_append(lst, k, verb, val):
                    raise EngineError("an append was refused")
                if k in counted and val not in seen:
                    seen.add(val)
                    e.release(val)
            return lst

        def run(program):
            pic = e.freeze((K["sys__object_reference"], 0, program))
            got = e.eval(self.env, pic[2])
            e.release(pic[2])
            e.release(program)
            return got
        return NL.Loader(e, run, lambda n: (K["sys__standard"], V[n], 0), lambda n: (K["sys__value_int"], 0, n),
                         lambda l: (K["sys__object_reference"], 0, l), cells, K)

    # ── memory ───────────────────────────────────────────────────────────────────────────────────────
    def buffer(self, nbytes, name, payload=None, ram=False):
        """A buffer of this worker, bound by name, filled if given bytes. Answers its address.
        ⛳ The program answers the buffer as well as binding it: the answer's hold is this caller's and goes back
        once the address is read, and the binding keeps its own."""
        verb = "nn__buffer__getnew_ram" if ram else "nn__buffer__getnew"
        kind, ref = self.comp.run("(let ((b (%s %d))) (sys__add_bindings %d '%s b) b)" % (verb, nbytes, self.env, name))
        if kind != self.K["sys__object_reference"]:
            raise EngineError("a %d-byte buffer for %r was refused (fault %s)" % (nbytes, name, self.e.sys.fault()))
        at = self.e.sys.peek(self.e.sys.full_address(ref))[2][0]
        self.e.release(ref)
        if at < (1 << 32):
            raise EngineError("a %d-byte buffer for %r answered a fault word — the pool is exhausted" % (nbytes, name))
        if payload is not None:
            self.write(at, payload)
        return at

    def table(self, name, cells):
        """A node array of the cells, bound by name — a verb's plane table."""
        text = ("(let ((t (sys__create %d %d))) " % (self.K["sys__node_array"], len(cells))
                + " ".join("(sys__node_array__set t %d %s)" % (i, c) for i, c in enumerate(cells))
                + " (sys__add_bindings %d '%s t))" % (self.env, name))
        self.must(text, "the table %r" % name, "sys__value_true")

    def node_array(self, name, n):
        self.must("(let ((t (sys__create %d %d))) (sys__add_bindings %d '%s t))"
                  % (self.K["sys__node_array"], n, self.env, name), "the array %r" % name, "sys__value_true")

    def read(self, at, nbytes):
        return self.e.nn.read(at, nbytes)

    def write(self, at, data):
        self.e.nn.write(at, data)

    def write_from(self, at, pointer, nbytes):
        """Bytes already in host memory — a mapped file's pages, say — copied to the card through nn's C door, with no
        Python copy on the way: `MEASURED` 13.8 GB/s for a 44 MB matrix from a mapped bundle, against 1.26 through
        `write` from Python bytes."""
        if self._abi_write is None:
            import ctypes
            f = ctypes.CDLL(self.e.__file__).nn_abi_write
            f.argtypes, f.restype = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_void_p], ctypes.c_int
            self._abi_write = f
        if self._abi_write(at, nbytes, pointer) != 1:
            raise EngineError("the write of %d bytes to %#x was refused" % (nbytes, at))

    def map_file(self, at, nbytes, path, offset=0):
        """A file's bytes as this worker's memory at `at`, read-only — read from the disk as each page is first touched and
        kept as the system keeps a file's pages (nn's `nn_abi_map_file`). Only a CPU worker can; a card refuses."""
        import ctypes
        f = ctypes.CDLL(self.e.__file__).nn_abi_map_file
        f.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_char_p, ctypes.c_ulonglong]
        f.restype = ctypes.c_int
        if f(at, nbytes, os.fsencode(path), offset) != 1:
            raise EngineError("%s could not be mapped at %#x (%d bytes)" % (path, at, nbytes))

    def shutdown(self):
        if self.booted:
            self.e.shutdown()
            self.booted = False
