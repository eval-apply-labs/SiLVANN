"""THE MACHINE — the engine, booted once, with one environment and the conveniences every model needs.

The engine is `silvann_engine_cpu`: the evaluator runs on the host, and the arithmetic reaches a card through a
silicon family the wheel loads. `Machine` chooses the family, boots with the text a model gives it, and then offers
what the tests all wrote for themselves: run a program, make a buffer and bind it, fill a table, define a procedure,
read and write a card's memory.
"""
import contextlib
import ctypes
import importlib
import os
import platform
import re
import sys
import threading

from .composer import Composer, read_all, to_text

# ⭐ THE PROCESS'S MEMORY POLICY IS MPOL_LOCAL, SET WHEN THIS MODULE IS FIRST IMPORTED: a page is placed on the node of
#   the thread that first touches it, as the default places it, but without the default's migrate-on-fault flag, so on
#   a machine of several NUMA nodes the kernel's NUMA balancing never samples this process. Its sampling faults on the
#   host's hot pages — the evaluator's heap, the buffers a card reads — and moved almost nothing here. A thread takes
#   its policy from the thread that starts it, so the call comes before the runtime starts any; the CPU family's own
#   large memory carries its own policy (`mbind`, x86_avx2's sys.cuh) and is not changed by it. A machine of one node,
#   or a container whose seccomp profile refuses the call, keeps the default.
#   `MEASURED` (node03: two Xeon E5-2680 v4, an MI50; test/src_runtime_bench.py, interleaved against the default;
#   ▶ measurements/2026-10-13_v060_on_node03.md, NN-76): the 35B D8E4 reads a 2,035-position prompt at 199 positions/s
#   against 194.5 and answers at 90 tokens/s against 87.5 (105 against 100 after a short prompt), its ~46,000 faults a
#   prompt down to under 100; the 27B, Ministral and the 122B with its experts on the CPU 1-3% faster, the 35B with its
#   experts on the CPU 5% in its prompt; OpenCL and the CPU alone the same.
_SET_MEMPOLICY = {"x86_64": 238, "aarch64": 237}      # the system call's number on each processor
_MPOL_LOCAL = 4

# ⭐ THE SLICE EVERY MACHINE BOOTS WITH (⚖ 2026-11-07 *"set the slice to 0.05s by default"*): a computation that has run
#   50 ms gives its block up to what waits there, at the end of a verb — preemptive, not protected (▶ `boot`)
SLICE_US = 50000


def _memory_policy_local():
    number = _SET_MEMPOLICY.get(platform.machine())
    if not sys.platform.startswith("linux") or number is None:
        return
    try:
        ctypes.CDLL(None, use_errno=True).syscall(number, _MPOL_LOCAL, None, ctypes.c_ulong(0))
    except (OSError, AttributeError):
        pass


_memory_policy_local()


class EngineError(Exception):
    pass


# nn's fault when a card of a group cannot read and write another's memory directly — "NPDM", NN__PARTIALS__FAULT_NO_DMA
#   (▶ src/packages/nn/contracts/objects/partials.cuh)
NO_DMA = 0x4E50444D


# The evaluator's stops, by their word (▶ src/engine/eval.cuh, "A STOP IS A PANIC").
STOPS = {"EVAV": "a form it could not run — a head that is not a verb, or a call it could not make",
         "EVNR": "a form it had no room to make — a frozen form to thaw, or a call's body to copy",
         "EVAD": "a descent it had no room to remember"}


class Panic(EngineError):
    """⛔ THE EVALUATOR STOPPED, AND THE MACHINE WITH IT. An evaluation abandoned where it stood leaves the heap holding
    what it was in the middle of, so the engine refuses every door until it is booted again; the `Machine` it happened
    on is shut down before this is raised. Booting again is the caller's choice (⚖ 2026-11-03: *"a panicked response
    should simply output to the repl/log that it failed and not reinstate the heap ... the reboot is an user's explicit
    choice"*). `word` is the stop's, as text: what is said first."""

    def __init__(self, word):
        self.word = word
        super().__init__("%s — the evaluator stopped on %s; the machine is shut down, and answers nothing until it is "
                         "booted again" % (word, STOPS.get(word, "a stop of its own")))


def _word(number):
    return number.to_bytes(4, "big").decode("ascii", "replace")


def _indent(text):
    """A program's text laid out a form a line: a newline before each `(` that opens a form inside `begin`."""
    out, depth = [], 0
    for i, ch in enumerate(text):
        if ch == "(" and depth >= 2 and i and text[i - 1] == " ":
            out.append("\n" + "  " * depth)
        out.append(ch)
        depth += (ch == "(") - (ch == ")")
    return "".join(out).replace(" \n", "\n")


def _unindent(text):
    """`_indent` undone: each line break and the indent after it back to the one space it replaced."""
    return re.sub(r"\n +", " ", text.strip())


# A program's line in `programs.lisp`, before its text: what it is, its name, and the worker it was defined as.
_ENTRY = re.compile(r"^;; (defun|picture|view)(?: (\S+))?, as worker (-?\d+)$")


OPENCL_FAMILIES = ("intel_opencl2", "khronos_opencl2", "nvidia_opencl1", "intel_opencl1_igpu")
# the OpenCL 1.2 families (`khronos_opencl1_base`): mapped RAM is two copies there, the card's and the program's, unless
#   the device's memory is the host's
OPENCL1_FAMILIES = ("nvidia_opencl1", "intel_opencl1_igpu")
# the processors' own families, the one present the CPU worker's (`Machine.cpu`): x86-64's, and Arm's (`arm_base`) — the
#   DGX Spark's GB10 before any Armv8.2
CPU_FAMILIES = ("x86_avx2", "arm_gb10", "arm_neon")


def opencl_own_memory():
    """{a GPU's name: whether its memory is its own} for every GPU the system's OpenCL loader lists — an NVIDIA card named by
    its compute capability (`sm_61`), any other by its device name (`gfx906:sramecc+:xnack-`), as the OpenCL families name
    theirs — its own where it says `CL_DEVICE_HOST_UNIFIED_MEMORY` false: a card, where an integrated GPU's is the PC's
    RAM. Empty where the system has no OpenCL loader. Asked of OpenCL here rather than of a family, which has no door for
    it, so it reads the same devices the families would."""
    lib = None
    for name in (("OpenCL.dll",) if os.name == "nt" else ("libOpenCL.so.1", "libOpenCL.so")):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return {}
    out, n = {}, ctypes.c_uint(0)
    if lib.clGetPlatformIDs(0, None, ctypes.byref(n)) != 0 or n.value == 0:
        return {}
    platforms = (ctypes.c_void_p * n.value)()
    lib.clGetPlatformIDs(n.value, platforms, None)
    info = lambda d, what, value: lib.clGetDeviceInfo(ctypes.c_void_p(d), ctypes.c_uint(what), ctypes.sizeof(value),
                                                      ctypes.byref(value), None) == 0
    for p in platforms:
        k = ctypes.c_uint(0)
        if lib.clGetDeviceIDs(ctypes.c_void_p(p), ctypes.c_ulonglong(1 << 2), 0, None, ctypes.byref(k)) != 0 or k.value == 0:
            continue
        devices = (ctypes.c_void_p * k.value)()
        lib.clGetDeviceIDs(ctypes.c_void_p(p), ctypes.c_ulonglong(1 << 2), k.value, devices, None)
        for d in devices:
            major, minor, unified = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(1)
            if info(d, 0x4000, major) and info(d, 0x4001, minor):      # cl_nv_device_attribute_query's
                name = "sm_%d%d" % (major.value, minor.value)
            else:
                text = ctypes.create_string_buffer(256)
                if lib.clGetDeviceInfo(ctypes.c_void_p(d), ctypes.c_uint(0x102B), 255, text, None) != 0:
                    continue
                name = text.value.decode(errors="replace")
            out[name] = info(d, 0x1035, unified) and unified.value == 0   # CL_DEVICE_HOST_UNIFIED_MEMORY
    return out


def own_memory(machine):
    """Whether the cards an OpenCL family drives for `machine` have memory of their own — every one of them, by the names
    the roster gave them (a device's name starts with its card's), or with SILVANN_OPENCL1_TAKE the devices it names. False
    for any other silicon, and where OpenCL cannot say."""
    if machine.silicon not in OPENCL_FAMILIES:
        return False
    names = [c["card"] for c in machine.e.cards() if c.get("silicon_family") == machine.silicon]
    take = os.environ.get("SILVANN_OPENCL1_TAKE") if machine.silicon in OPENCL1_FAMILIES else None
    devices = opencl_own_memory()
    mine = [own for name, own in devices.items() if any(name.startswith(n) for n in names) or (take and name.startswith(take))]
    return bool(mine) and all(mine)


def _family_integrated(e, family, own):
    """Whether every device OpenCL `family` drives is an integrated GPU — its memory the host's — by `own`
    (`opencl_own_memory`) and the names the roster gave its cards; False where OpenCL cannot say."""
    names = [c["card"] for c in e.cards() if c.get("silicon_family") == family]
    mine = [o for seen, o in own.items() if any(seen.startswith(n) for n in names)]
    return bool(mine) and not any(mine)


class Machine:
    # ⛔ What a panic tells besides raising: a callable handed the `Panic`, once the machine is shut down — the server's
    #   way of stopping (▶ start_silvann_server.py's `_stopped`). None: the panic is raised and nothing else.
    on_panic = None

    def __init__(self, silicon=None, engine="cpu"):
        # every procedure, picture and view defined, in order: (what, worker, name, text) — ▶ `write_programs`
        self.programs = []
        self._thread = threading.local()   # the worker `act_as` last chose, a thread's own (▶ `worker`)
        self.worker_moved = {}         # a `(worker N ...)` form's N -> the worker it is here (▶ `_source_forms`)
        self.recording = False         # True: the definitions are recorded and not made — ▶ `composed_programs`
        self.read_from = None          # the file the programs were read from, when they were
        self.programs_file = None      # (path, key): where a boot looks for its programs before composing them
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # where the built engine sits: `lib/` in a release, the tree's root in development
        for at in (os.path.join(root, "lib"), root):
            if os.path.isdir(at) and at not in sys.path:
                sys.path.insert(0, at)
        # ⭐ HIP's kernel arguments in the card's memory rather than the host's: a decode is ~800-1,100 small launches a
        #   token, and each reads its arguments. `MEASURED` (node03, one MI50, test/src_runtime_bench.py, `=0` and `=1`
        #   interleaved): the 35B D8E4 at 71 positions 79.4-82.1 tok/s against 84.2-86.7 (four each), at 2,035 72.2-72.4
        #   against 73.3-76.6 (three each) — every run with it above every run without, ~3-5%, and the runs themselves
        #   swing ~5% between two levels either way; the 27B D4 0-2%. Set before the engine loads, as HIP reads it when it
        #   starts; a value the user set stays.
        os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
        self.e = importlib.import_module("silvann_engine_" + engine)
        # a card if any family drives one — its vendor's own first (AMD's: ROCm 7's family for the MI100 on, then ROCm 6's
        #   for the MI50), then OpenCL 2.0 (Intel's own first), then 1.2 — else the CPU's own
        # cores. x86_avx2 counts no devices on a CPU without AVX2 (nor an Arm family on one without its features); the
        # host family then runs the language but answers none of nn's doors, so a model's first verb refuses with "NPND"
        present = [f for f in ("amd_rocm10", "amd_rocm6_wave64", "nvidia_cuda12", "intel_opencl2", "khronos_opencl2",
                               "nvidia_opencl1", "intel_opencl1_igpu") if self.e.devices(f) > 0]
        # ⛳ AN INTEGRATED GPU IS THE CARD ONLY WHERE NO CARD OF ITS OWN MEMORY IS PRESENT (NN-142): two OpenCL families open
        #   at once — a GTX 1060 through OpenCL 1.2 and the HD 630 beside it through Intel's 2.0, the laptop
        #   `SILVANN_IGPU_CPU` asks for — and the card is the 1060, the integrated GPU a worker beside it (`integrated`)
        if len(present) > 1 and any(f in OPENCL_FAMILIES for f in present):
            own = opencl_own_memory()
            present.sort(key=lambda f: f in OPENCL_FAMILIES and _family_integrated(self.e, f, own))
        for family in present + list(CPU_FAMILIES):
            if silicon is None and self.e.devices(family) > 0:
                silicon = family
        if silicon is not None:
            self.e.choose_silicon(silicon)
        self.silicon = silicon or "host"
        # the processors' family — the CPU worker's beside a card, its devices the machine and each NUMA node; x86_avx2
        #   where none is present, as before there was another
        self.cpu = next((f for f in CPU_FAMILIES if self.e.devices(f) > 0), "x86_avx2")
        self.booted = False
        self.held_files = []                 # descriptors held as long as the machine runs (▶ `shutdown`)
        self._abi_write = None

    def boot(self, device_params, alloc_mb=256, workers=None, worker_params=None, bindings=512, slice_us=SLICE_US,
             slice_cycles=0):
        """Boot with one worker (or the list given), each with the device parameters given, or its own section from
        `worker_params` — {worker: text}. `bindings`: how many names the program's environment holds. `slice_us`:
        preemption — a computation that has run that many microseconds gives its block up to what waits there, at the
        end of a verb (▶ sys's `scheduler__impl.cuh`, "THE SLICE"); SLICE_US by default, 0 for none. `slice_cycles`, if
        given, the slice in the processor's clock's cycles instead. A model's steps that share a worker's buffers
        between verbs run inside `compute_inline`, which no slice gives up."""
        e = self.e
        slice_line = ("sys__scheduler__slice_cycles:%d\n" % slice_cycles if slice_cycles else
                      "sys__scheduler__slice_us:%d\n" % slice_us if slice_us else "")
        text = "boot_commands{\nsys__alloc_size_mb:%d\n%s}\ndevice_params{\n%s}\n" % (alloc_mb, slice_line, device_params)
        for w, params in sorted((worker_params or {}).items()):
            text += "worker%d_params{\n%s}\n" % (w, params)
        if workers:
            e.choose_workers(workers)
            e.boot(config=text, blocks=len(workers))
        else:
            e.boot(config=text)
        self.booted = True
        self.K, self.V = e.kinds(), e.verbs()
        self.env = e.bindings_create(bindings)
        self.comp = Composer(e, self.env)

    @property
    def worker(self):
        """The worker `act_as` last chose on this thread, -1 for none: a thread driving one worker acts as it while another
        thread drives another (the resident machine, ▶ `run_in`)."""
        return getattr(self._thread, "worker", -1)

    @worker.setter
    def worker(self, worker):
        self._thread.worker = worker

    def act_as(self, worker):
        """What runs from here runs as `worker`: its buffers, its device, its tables — on this thread. -1 is the default
        again."""
        self.worker = worker
        if not self.recording:
            self.e.act_as(worker)

    # ── a panic ─────────────────────────────────────────────────────────────────────────────────────
    def panicked(self):
        """The word the evaluator stopped with, as text, or None (▶ `Panic`)."""
        if not self.booted or not hasattr(self.e, "panic"):
            return None
        word = self.e.panic()
        return _word(word) if word else None

    def _stop_if_panicked(self, cause=None):
        word = self.panicked()
        if word is None:
            return
        self.shutdown()
        panic = Panic(word)
        try:
            if Machine.on_panic is not None:
                Machine.on_panic(panic)
        finally:
            raise panic from cause                    # whatever the telling did, the panic is what is raised

    @contextlib.contextmanager
    def _evaluating(self):
        """Around every door that evaluates: what it answered or raised — unless the machine panicked under it, and then
        the machine is shut down and the `Panic` raised in its place. ⛳ ASKED ON A FAILURE TOO: after a stop every door
        refuses, so what a caller sees first may be a refusal the panic caused."""
        try:
            yield
        except Panic:
            raise
        except Exception as ex:
            self._stop_if_panicked(ex)
            raise
        self._stop_if_panicked()

    # ── running ──────────────────────────────────────────────────────────────────────────────────────
    def run(self, text):
        """A program's answer as (kind, value)."""
        with self._evaluating():
            got = self.comp.run(text)
        return got

    def run_in(self, worker, text, view=None):
        """A program's answer as (kind, value), run on worker `worker`'s block in the bindings the view bound as `view`
        names — the program's own where it is None. ⭐ A THREAD THAT DRIVES ONE WORKER (⚖ 2026-11-04 the resident machine):
        the program runs on that worker's block beside every other block's, and nothing waits for them to be idle, as a
        `grid` does — what it hands out it collects. The thread acts as `worker` for the call and as it did after."""
        with self._evaluating():
            env = self._env_of(view)
            cell = self.comp.compose(text)
            try:
                pic = self.e.freeze(cell)
                was = self.worker
                self.e.act_as(worker)
                try:
                    kind, value = self.e.execute(env, pic)
                finally:
                    self.e.act_as(was)
                    self.e.release(pic[2])
            finally:
                self.e.release(cell[2])
        # ⛳ A FAULT FAILS IT, AS A `grid`'s DOES: the channel is the machine's, so one raised by another thread's program
        #   meanwhile fails this one too — the run that reads it first says so
        fault = self.e.sys.fault()
        if fault["raised"] != 0:
            self.e.sys.fault_clear()
            raise EngineError("the program on worker %d did not run: kind %s value %s fault %s" % (worker, kind, value, fault))
        return kind, value

    def _env_of(self, view):
        """The bindings a program runs in: the program's own, or the view bound by that name — read once and held."""
        if view is None:
            return (self.K["sys__object_reference"], 0, self.env)
        views = self.__dict__.setdefault("_views", {})
        if view not in views:
            kind, at = self.run("(begin %s)" % view)
            if kind not in (self.K["sys__object_reference"], self.K["sys__quoted_array"]) or not at:
                raise EngineError("%r names no view (kind %s)" % (view, kind))
            views[view] = (kind, 0, at)
        return views[view]

    def must(self, text, what, want="sys__value_int"):
        with self._evaluating():
            kind, value = self.comp.run(text)
        fault = self.e.sys.fault()
        if kind != self.K[want] or fault["raised"] != 0:
            self.e.sys.fault_clear()
            ex = EngineError("%s did not run: kind %s value %s fault %s" % (what, kind, value, fault))
            ex.fault = fault
            raise ex
        return value

    def join_group(self, group, ranks, fan=None):
        """The cards 0 … ranks-1, each on its own worker and card 0 first, joined to nn's group `group` (▶ nn's
        `nn__partials__join`), the worker left at the last card's. A model split over cards needs each pair of them to
        read the other's memory directly (⚖ "tensor parallelism requires DMA"); a card that cannot is refused, saying so
        and what to run instead. A runtime joins once at the start of its load, before a weight is read, so the refusal
        costs no load — `MEASURED` 2026-11-06 debian01 (two MI50s): GLM's two cards were refused with the fault's number
        after its 142 GB of experts had loaded — and again where its group is used: rank 0's join starts the group afresh.
        ⛳ WHAT REFUSES IS THE DRIVER, NOT WHERE THE CARDS SIT: node03 has a card on each socket as debian01 has, and its
        cards reach each other — ROCm's amdgpu-dkms links them (KFD's p2p_links between the two) — where debian01's
        kernel's own amdgpu (Debian 13's 6.12, built without CONFIG_PCI_P2PDMA, so without CONFIG_HSA_AMD_P2P) links a
        card to the CPUs alone (`MEASURED` 2026-11-06, both machines' /sys/class/kfd/kfd/topology and /boot/config)."""
        fan = fan or min(4, ranks)
        for k in range(ranks):
            self.act_as(k)
            try:
                self.must("(nn__partials__join %d %d %d %d)" % (group, k, ranks, fan), "card %d joining the group" % k,
                          "sys__value_true")
            except EngineError as ex:
                if (getattr(ex, "fault", None) or {}).get("op") != NO_DMA:
                    raise
                from .model_folder import Refused
                fix = (" — on Linux, ROCm's amdgpu-dkms gives AMD's cards the path; a kernel's own amdgpu gives it only "
                       "built with CONFIG_HSA_AMD_P2P, which Debian's is not" if getattr(self, "silicon", None) in ("amd_rocm6_wave64", "amd_rocm10")
                       else "")
                raise Refused("card %d cannot read the memory of card%s %s directly, nor they its, and a model split "
                              "over %d cards (tensor_parallel %d: the server's num_cards %d with multi_user \"no\") "
                              "needs each pair of its cards to: this machine's driver gives them no such path%s. Run the "
                              "model on one card (num_cards 1), or a conversation on each card (multi_user \"yes\") "
                              "where the model has it"
                              % (k, "" if k == 1 else "s", ", ".join(map(str, range(k))), ranks, ranks, ranks, fix)) from None

    def grid(self, text):
        """A program run with every block standing, so it can hand work to the others (`sys__scheduler__add`) and collect it
        (`sys__wait_for_result`). Block 0 runs it, as the host's worker."""
        with self._evaluating():
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
        with self._evaluating():
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
        """A program frozen and bound by name — what `sys__scheduler__add` hands another block."""
        self.programs.append(("picture", self.worker, name, text))
        if self.recording:
            return
        cell = self.comp.compose(text)
        pic = self.e.freeze(cell)
        self.e.release(cell[2])
        self.bind(name, pic)

    # ── a process for the host's work (NN-160, ▶ sys's "A PROCESS FOR THE HOST") ──────────────────────────────────────────
    def new_process(self):
        """A new process: a number from the counter every process is made from, so no two are one."""
        return int(self.e.process_new())

    @contextlib.contextmanager
    def as_process(self, pid):
        """What this thread asks of the machine inside — every program it runs, and what they ask and go on as — made as
        process `pid`: `kill` tears it down, `stop` asks it to end at the token it is on. The thread's own after."""
        was = self.e.process_as(int(pid))
        try:
            yield pid
        finally:
            self.e.process_as(was)

    def kill(self, pid):
        """Process `pid` killed: its requests off every queue and its steps ended at their next move or slice, their
        answers SCKL; nothing it did undone. Answers how many requests it took off."""
        return int(self.e.process_kill(int(pid)))

    def stop(self, pid):
        """Process `pid` asked to stop at the token it is on: its program reads `(sys__scheduler__stopped)` once a token
        (▶ lisp/generate.lisp) and ends there."""
        return bool(self.e.process_stop(int(pid)))

    def view(self, name):
        """A read-only snapshot of the bindings as they stand, bound by name — the names a computed block sees."""
        self.programs.append(("view", self.worker, name, ""))
        if self.recording:
            return
        e, K, V = self.e, self.K, self.V
        self.bind(name, e.create_executable([(K["sys__standard"], V["sys__bindings__viewonly"], 0),
                                             (K["sys__object_reference"], 0, self.env)]))

    def defun(self, text):
        self.programs.append(("defun", self.worker, None, text))
        if not self.recording:
            self.must(text, "the procedure %r" % text[:60], "sys__value_true")

    # ── ⭐ A PROGRAM WRITTEN AS A FILE ──────────────────────────────────────────────────────────────────────
    def load_source(self, path):
        """A model's Lisp file — `lisp/<name>.lisp` in its folder (▶ `ModelFolder.program`) — every top-level form in
        order, handed to the evaluator as a REPL hands it a line. A `defun` — or an `if` choosing between defuns by a
        setting the boot bound — is defined as it is; three forms are the machine's rather than the language's:
            (worker N form ...)     the forms inside defined as worker N (▶ `act_as`) — or as the worker the model
                                    moved N to in `worker_moved` (Qwen's CPU, worker 1, after a group's cards)
            (picture name form)     the form frozen and bound by name, for `sys__scheduler__add` to hand another block
            (view name)             the bindings as they stand, bound by name — what a computed block sees"""
        with open(path) as f:
            self._source_forms(read_all(f.read()), os.path.basename(path))

    def _source_forms(self, forms, name):
        for form in forms:
            head = form[0] if isinstance(form, list) and form else None
            if head == "worker":
                before = self.worker
                self.act_as(self.worker_moved.get(form[1], form[1]))
                self._source_forms(form[2:], name)
                self.act_as(before)
            elif head == "picture":
                self.picture(form[1], to_text(form[2]))
            elif head == "view":
                self.view(form[1])
            elif head in ("defun", "if"):
                self.defun(to_text(form))
            else:
                raise EngineError("%s: a top-level form is a defun, an if, worker, picture or view — not %r"
                                  % (name, to_text(form)[:60]))

    def bind_values(self, values):
        """Names bound to values, `{name: value}`: an int, a float, True or False, or a form's text (a plane, say),
        evaluated once. What a model's source reads as its shape and its settings."""
        def text(v):
            if isinstance(v, bool):
                return "(< 0 1)" if v else "(< 1 0)"
            return repr(float(v)) if isinstance(v, float) else str(v)
        items = sorted(values.items())
        for at in range(0, len(items), 32):
            self.must("(begin %s (< 0 1))" % " ".join("(sys__add_bindings %d '%s %s)" % (self.env, k, text(v))
                                                       for k, v in items[at:at + 32]), "the values", "sys__value_true")

    def array(self, name, items):
        """A node array of `items` — names, ints, True or False — bound by name, filled 32 at a time."""
        self.node_array(name, len(items))
        cell = lambda v: ("(< 0 1)" if v else "(< 1 0)") if isinstance(v, bool) else str(v)
        for at in range(0, len(items), 32):
            self.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set %s %d %s)" % (name, i, cell(v))
                                                       for i, v in enumerate(items[at:at + 32], at)),
                      "the array %r" % name, "sys__value_true")

    # ── ⭐ THE PROGRAMS AS A FILE: written once, read at every boot after ────────────────────────────────
    # A model composes its procedures in Python once; the file holds them with the key of what they were composed
    # from — the pack, the settings, and the runtime's own source — and a boot whose key matches defines what the
    # file holds instead of composing them again. Each program carries the worker it was defined as.
    def write_programs(self, path, heading, key):
        """The programs defined so far, as the Lisp they were written in, one after another — what a model IS in this
        engine, readable. ⛳ Written from the text each was defined with; nothing is reconstructed."""
        with open(path, "w") as f:
            f.write(";; %s\n;; key %s\n;; %d programs, in the order they were defined\n\n" % (heading, key, len(self.programs)))
            for what, worker, name, text in self.programs:
                f.write(";; %s%s, as worker %d\n" % (what, "" if name is None else " " + name, worker))
                f.write((_indent(text) + "\n\n") if text else "\n")

    @staticmethod
    def programs_in(path, key):
        """The programs a file holds, [(what, worker, name, text)], if it was written for `key`; else None."""
        try:
            with open(path) as f:
                lines = f.read().split("\n")
        except OSError:
            return None
        if ";; key %s" % key not in lines[:4]:
            return None
        out, body = [], None
        for line in lines[3:] + [";; end"]:
            m = _ENTRY.match(line)
            if m or line == ";; end":
                if body is not None:
                    out[-1] = out[-1][:3] + (_unindent("\n".join(body)),)
                if line == ";; end":
                    break
                out.append((m.group(1), int(m.group(3)), m.group(2), ""))
                body = []
            elif body is not None:
                body.append(line)
        return out

    def read_programs(self, path, key):
        """Every program the file holds defined as it was, if the file was written for `key`. Answers whether it was."""
        held = self.programs_in(path, key)
        if held is None:
            return False
        before = self.worker
        for what, worker, name, text in held:
            self.act_as(worker)
            {"defun": lambda: self.defun(text), "picture": lambda: self.picture(name, text),
             "view": lambda: self.view(name)}[what]()
        self.act_as(before)
        self.read_from = path
        return True

    def define_programs(self, compose):
        """A model's programs: read from `programs_file` when it was written for this key, else `compose()`d."""
        if self.programs_file is None or not self.read_programs(*self.programs_file):
            compose()

    def composed_programs(self, compose):
        """What `compose()` would define, recorded and not made — to set beside what a file held."""
        programs, worker = self.programs, self.worker
        self.programs, self.recording = [], True
        try:
            compose()
            return self.programs
        finally:
            self.programs, self.recording = programs, False
            self.act_as(worker)

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
            with self._evaluating():
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
        with self._evaluating():
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

    def integrated(self):
        """The integrated GPU beside this machine's cards, as a worker — (its OpenCL family, its device) — where an OpenCL
        family drives a GPU whose memory is the host's (`opencl_own_memory`: `CL_DEVICE_HOST_UNIFIED_MEMORY`) and the cards
        are another family's — ROCm's, CUDA's, or another OpenCL family's (a GTX 1060 through `nvidia_opencl1`); None where
        there is none (NN-142)."""
        if self.silicon == "host" or self.silicon in CPU_FAMILIES:
            return None
        own = opencl_own_memory()
        for family in OPENCL_FAMILIES:
            if family == self.silicon:
                continue
            names = [c["card"] for c in self.e.cards() if c.get("silicon_family") == family]
            for device, name in enumerate(names):
                if any(seen.startswith(name) and not mine for seen, mine in own.items()):
                    return (family, device)
        return None

    _ZEROS = None                           # one host buffer of zeros, shared, that `zero` writes again and again

    def zero(self, at, nbytes, piece=64 << 20):
        """`nbytes` of zeros at `at`, a `piece` at a time from one buffer of zeros kept for it — so clearing a
        conversation's state never holds the state's size in host memory (Flash-Next's n-gram values 3.75 GiB at 196,608
        positions: `MEASURED` node03 2026-11-09, the load's peak 63.59 GiB of anonymous memory against its plan's 59.1)."""
        if Machine._ZEROS is None or len(Machine._ZEROS) < piece:
            import numpy as np
            Machine._ZEROS = np.zeros(piece, dtype=np.uint8)
        done = 0
        while done < nbytes:
            n = min(piece, nbytes - done)
            self.write_from(at + done, Machine._ZEROS.ctypes.data, n)
            done += n

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
        while self.held_files:               # a model folder held while it was read (▶ `experts_split.hold`)
            os.close(self.held_files.pop())
