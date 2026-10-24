"""A MODEL'S ROUTED EXPERTS AS A FILE MAPPED OVER A CPU WORKER'S SLOTS — the RAM holds the ones in use, the disk the rest.

A CPU worker's experts live in one collection of equal slots. With memory for all of them they are copied in at the load;
without it, each worker's share is written once as a file in its slots' own layout — from the bundle, which stays as it was
published — and that file is mapped over the slots (nn's `nn_abi_map_file`). An expert is then read from the disk the first
time a token picks it and stays for as long as the system keeps the file's pages: the page cache is the cache, and the
file is what it caches.
```
  <dir>/partNofP.bin        every expert of every layer the model routes, part N of P, at its slot's offset
  <dir>/partNofP.bin.json   what it was written for — the bundle, the split, the slot's shape, the offsets — written
                            last, so a file cut short is written again
```
A run over a subset of the layers (a test's) gets files of its own, named by the layers, so it never overwrites the
model's.
"""
import hashlib
import json
import os

import numpy as np

from .model_folder import Refused
from .system import BINARY, WINDOWS, lock, lockable, memory, spans, write_at


def available():
    """The memory the system says it can give without swapping, in bytes."""
    return memory()[1]


def limit():
    """What this process may hold: its cgroup's memory limit when it has one (a `systemd-run -p MemoryMax=…` scope, a
    container), else what the system says is available."""
    try:
        with open("/proc/self/cgroup") as f:
            path = next(l.strip().split(":", 2)[2] for l in f if l.startswith("0::"))
        with open(os.path.join("/sys/fs/cgroup", path.lstrip("/"), "memory.max")) as f:
            said = f.read().strip()
        if said != "max":
            return min(int(said), available())
    except (OSError, StopIteration, ValueError):
        pass
    return available()


def free():
    """What this process may still take: its cgroup's limit less what the cgroup holds and cannot give back — its
    anonymous memory, with no swap to give it to — else what the system says is available."""
    try:
        with open("/proc/self/cgroup") as f:
            path = next(l.strip().split(":", 2)[2] for l in f if l.startswith("0::"))
        base = os.path.join("/sys/fs/cgroup", path.lstrip("/"))
        with open(os.path.join(base, "memory.max")) as f:
            said = f.read().strip()
        if said != "max":
            with open(os.path.join(base, "memory.stat")) as f:
                anon = next(int(l.split()[1]) for l in f if l.startswith("anon "))
            return min(int(said) - anon, available())
    except (OSError, StopIteration, ValueError):
        pass
    return available()


def cache_bytes(total, ram_gb=None):
    """How much of `total` bytes of experts a cache in memory holds: `ram_gb` when a config says so, else what the process
    may hold less a quarter of it, at least 3 GB and at most 8, for everything else. `MEASURED` on node03, the 35B under
    an 8 GB cap: with 2 GB kept the cgroup killed the process (5.7 GB of its own memory and 1.5 GB of mapped files at the
    kill) — the card's pinned host memory cannot be given back, and the cap counts it."""
    if ram_gb is not None:
        return min(total, int(float(ram_gb) * (1 << 30)))
    held = limit()
    return max(0, min(total, held - min(8 << 30, max(3 << 30, held // 4))))


# what the experts in RAM leave beside them: the process's own memory and the system's spare
OWN = 2 << 30            # MEASURED on node03 (measurements/2026-10-14_v080rc1_on_node03.md): beside its experts the
                         # server holds 1.2-1.7 GiB booted (the 122B; the bench's process 1.0 the 35B, 1.3 the 122B, 1.5
                         # GLM), 2.0 with a Word file, a GIF and an MP4 answered — and, while a video is read, what that
                         # holds (`vision.video_ram`: the config's `max_tokens_per_video` of the model's tokens)
SPARE = 500 << 20        # what is left to the system — ⚖ 2026-10-14, "a flat 500mb of spare ram"


def in_ram(need, held=None, video=0):
    """Whether `need` bytes of experts go in RAM: they, the server's own memory beside them (`OWN`, and `video`, what
    reading a video holds — `vision.video_ram`) and the system's spare fit in what this process may hold (`held`, by
    default `limit`: a container's or a capped scope's cgroup limit, else what the system has available)."""
    return need + OWN + video + SPARE <= (limit() if held is None else held)


def choose(experts_from, need, video=0):
    """`"ram"`, `"disk"` or `"mapped"` for `experts_from` as a config says it: `"auto"` is RAM where the experts fit in it
    (`in_ram`), else the disk. `"disk"` is a cache of experts in memory, read from the file as they are wanted;
    `"mapped"` maps the file and lets the system's page cache be the cache."""
    if experts_from not in ("auto", "ram", "disk", "mapped"):
        raise Refused("experts_from is 'auto', 'ram', 'disk' or 'mapped', not %r" % (experts_from,))
    if experts_from == "auto":
        return "ram" if in_ram(need, video=video) else "disk"
    if experts_from == "mapped" and WINDOWS:
        raise Refused("experts_from 'mapped' maps the file over memory the program holds, which the engine does not do on "
                      "Windows: 'disk' reads the experts from it as they are wanted")
    return experts_from


def map_experts(m, bundle, workers, plan, layers, experts, directory, pieces, all_layers, expert_type=0,
                progress=None):
    """Every layer's experts on each CPU worker, from its file, mapped over its slots.

    workers      [(worker, part, parts)] — each maps its own file
    plan         the slot's shape (an `NL.SlotPlan`)
    layers       the routed layers this run holds; `all_layers` the model's, which name the file when they are the same
    pieces       pieces(view, layer, expert, part) -> [(offset in the slot, a uint8 array)], `view` the layer's blob
    """
    paths = ensure(bundle, workers, plan, layers, experts, directory, pieces, all_layers, progress)
    for w, part, parts in workers:
        m.act_as(w)
        loader, at = m.loader(), {}
        for l in layers:
            loader.layer_width(l, expert_type, experts)
            for x in range(experts):
                at[(l, x)] = loader.reserve(expert_type)
                loader.assign(l, expert_type, x, at[(l, x)])
        if loader.handed() != 0:
            raise Refused("%d expert slots were reserved and never published" % loader.handed())
        lo = min(at.values())
        if any(at[(l, x)] - lo != off for l, x, off in layout(layers, experts, plan.slot_bytes)):
            raise Refused("the slots were not handed out one after another, as the file is laid out")
        m.map_file(lo, plan.slot_bytes * experts * len(layers), paths[part], 0)


def layout(layers, experts, slot_bytes):
    """Where each expert is in a file: layer by layer, expert by expert, a slot each — [(layer, expert, offset)]."""
    return [(l, x, (i * experts + x) * slot_bytes) for i, l in enumerate(layers) for x in range(experts)]


def ensure(bundle, workers, plan, layers, experts, directory, pieces, all_layers, progress=None):
    """Each worker's file, written if it is missing or was written for something else: {part: path}."""
    tag = "" if list(layers) == list(all_layers) else ".layers-" + hashlib.sha256(
        json.dumps(list(layers)).encode()).hexdigest()[:10]
    size = plan.slot_bytes * experts * len(layers)
    places = [[l, x, off] for l, x, off in layout(layers, experts, plan.slot_bytes)]
    todo, paths = [], {}
    for _, part, parts in workers:
        ident = hashlib.sha256(json.dumps(dict(pack=bundle.manifest, parts=parts, part=part, size=size,
                                               slot=[plan.up_data, plan.up_lut, plan.down_data, plan.down_lut],
                                               layout=places), sort_keys=True, default=str).encode()).hexdigest()
        path = paths[part] = os.path.join(directory, "part%dof%d%s.bin" % (part + 1, parts, tag))
        try:
            fresh = json.load(open(path + ".json"))["ident"] == ident and os.path.getsize(path) == size
        except (OSError, ValueError, KeyError):
            fresh = False
        if not fresh:
            todo.append((path, ident, size, part, parts, {(l, x): off for l, x, off in places}))
    if todo:
        _write(bundle, todo, plan, layers, experts, directory, pieces, progress)
    return paths


def cache_experts(m, bundle, workers, plan, layers, experts, directory, pieces, all_layers, expert_type=0,
                  progress=None):
    """Every layer's experts on each CPU worker as a CACHE in its own memory, read as they are wanted (nn's loader,
    straight from the drive) — from the pack itself where its experts lie as the slots take them (`_in_place`, NN-97:
    nothing written beside it), else from the worker's file, written first if it is not there. Answers each worker's
    backing tables, {(worker, layer): name}, for the experts tables to name. ▶ nn's `nn__expert__backing`."""
    lie = _in_place(bundle, workers, layers, experts, pieces, plan.slot_bytes)
    paths = None if lie is not None else ensure(bundle, workers, plan, layers, experts, directory, pieces, all_layers,
                                                progress)
    names = {}
    for w, part, parts in workers:
        m.act_as(w)
        loader = m.loader()
        for l in layers:
            loader.layer_width(l, expert_type, experts)
        handles = {}
        for i, l in enumerate(layers):
            if lie is not None:
                path, slices = lie[part][l]
                if path not in handles:
                    handles[path] = _open(m, path, "the pack's")
                row = [handles[path]] + [v for s in slices for v in s]
            else:
                if part not in handles:
                    handles[part] = _open(m, paths[part], "the experts' file")
                row = [handles[part], i * experts * plan.slot_bytes, plan.slot_bytes, 0, plan.slot_bytes]
            names[(w, l)] = "bk%d_%d" % (l, w)
            m.table(names[(w, l)], row + [0] * (CELLS - len(row)))
    return names


SLICES = 8      # an expert's byte runs a backing names (nn's `NN__EXPERT__SLICES`)
PAGE = 4096     # what a read straight from the drive takes whole: its offset, its length and its memory
GAP = 64 << 10  # the most of a slot's padding one run reads over (▶ `_runs`)
CELLS = 1 + 4 * SLICES      # a backing as a program holds it: the file, then each slice's start, bytes, place, stride
_UNSPELLED = set(" \t\r\n()'\"")    # what ends a name in a program: a path holding one cannot be handed as a name


def _in_place(bundle, workers, layers, experts, pieces, slot_bytes):
    """Where a worker's experts lie in the pack itself — {part: {layer: (its file, [(expert 0's slice there, its bytes,
    its place in the slot, the step to the next expert's), ...])}} — when every piece `pieces` cuts, of every expert, is
    a run of its layer's mapped file and each slice steps by one stride; else None, and the file is written (NN-97).
    A Qwen layer's four slices, its experts stacked by tensor, and GLM's six on one socket lie so; a socket's share of
    GLM's down rows (a column cut of each) and a LoRA's experts (another file) do not. The slices are read as few runs
    as they lie in (`_runs`): a pack with one record an expert, in the slot's order, is one run an expert."""
    found = {}
    for _, part, _parts in workers:
        found[part] = {}
        for l in layers:
            view = np.frombuffer(bundle.blob(l), dtype=np.uint8)
            base, end = view.ctypes.data, view.ctypes.data + view.nbytes

            def runs(x):
                out = []
                for into, piece in pieces(view, l, x, part):
                    a = piece.ctypes.data
                    if not (piece.flags.c_contiguous and base <= a and a + piece.nbytes <= end):
                        return None
                    out.append((a - base, piece.nbytes, into))
                return out
            first = runs(0)
            if first is None or not first or len(first) > SLICES:
                return None
            later = runs(1) if experts > 1 else first
            if later is None or len(later) != len(first):
                return None
            step = [b[0] - a[0] for a, b in zip(first, later)]
            for x in range(2, experts):
                got = runs(x)
                if got is None or len(got) != len(first) or any(
                        g != (f[0] + x * s, f[1], f[2]) for g, f, s in zip(got, first, step)):
                    return None
            path = os.path.join(bundle.layer_dir(l), bundle.manifest["layout"]["weights_bin"])
            found[part][l] = (path, _runs([(f[0], f[1], f[2], s) for f, s in zip(first, step)], slot_bytes,
                                          os.path.getsize(path), experts))
    return found


def in_slots(bundle, layer, experts, pieces, slot_bytes):
    """Where a layer's experts lie in the pack exactly as its slots do — (its file, the first expert's offset) — when an
    expert is one run of the slot's whole size from a page, the next at the next slot (a pack one record an expert, its
    gate and up one: NN-109): the pack then maps over the slots as a written file would, nothing written beside it —
    DeepSeek's mapped layers, 6.3 GB a layer written before (`MEASURED` node03 2026-10-23,
    measurements/2026-10-23_nn109_one_run_glm_deepseek_node03.md). Else None."""
    lie = _in_place(bundle, [(0, 0, 1)], [layer], experts, pieces, slot_bytes)
    if lie is None:
        return None
    path, runs = lie[0][layer]
    if len(runs) != 1:
        return None
    at, n, into, step = runs[0]
    if into != 0 or n != slot_bytes or step != slot_bytes or at % PAGE or at + experts * slot_bytes > os.path.getsize(path):
        return None
    return path, at


def _runs(slices, slot_bytes, file_bytes, experts):
    """An expert's slices as few reads as the pack allows: two that step alike and lie apart in the file as in the slot,
    the gap between them the slot's padding (at most `GAP`), are one run; and a run that starts on a page and ends
    inside one is taken on to that page's end where the slot has room before the next run and the file has the bytes —
    so the drive is asked once an expert, straight into the slot. ⛳ FOUR READS TO FOUR PLACES TOOK ~1.3x ONE READ OF THE
    SAME BYTES: the 122B's experts from node03's NVMe array, 8.0-8.3 tokens a second from the pack stacked by tensor, 9.2
    from the file in the slots' order, 9.3-9.4 from the pack one record an expert — `MEASURED` 2026-10-23,
    measurements/2026-10-23_nn97_one_run_node03.md."""
    out = []
    for f, n, into, step in sorted(slices, key=lambda v: v[2]):
        if out:
            pf, pn, pinto, pstep = out[-1]
            gap = into - (pinto + pn)
            if step == pstep and 0 <= gap <= GAP and f - pf == into - pinto:
                out[-1] = (pf, into + n - pinto, pinto, pstep)
                continue
        out.append((f, n, into, step))
    for i, (f, n, into, step) in enumerate(out):
        whole = (n + PAGE - 1) // PAGE * PAGE
        room = out[i + 1][2] if i + 1 < len(out) else slot_bytes
        if (whole != n and f % PAGE == 0 and into % PAGE == 0 and step % PAGE == 0 and into + whole <= room
                and f + (experts - 1) * step + whole <= file_bytes):
            out[i] = (f, whole, into, step)
    return out


def _open(m, path, what):
    """`path` opened by the worker's family (`sys__file__open`), its handle."""
    if _UNSPELLED & set(path):
        raise Refused("%s %r holds a space or a bracket, which a program cannot name: move the models to a folder "
                      "without one (`silvann data <folder>`)" % (what, path))
    kind, handle = m.run("(sys__file__open '%s)" % path)
    if kind != m.K["sys__value_int"] or handle == 0:
        raise Refused("%s %s could not be opened" % (what, path))
    return handle


def _write(bundle, todo, plan, layers, experts, directory, pieces, progress):
    """Each file: every expert's part in its slot's layout, at its slot's offset — a layer's blob read once for all."""
    os.makedirs(directory, exist_ok=True)
    fds = []
    for path, ident, size, part, parts, off in todo:
        if os.path.exists(path + ".json"):
            os.remove(path + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | BINARY, 0o644)
        os.ftruncate(fd, size)
        fds.append(fd)
    slot = np.zeros(plan.slot_bytes, np.uint8)
    for i, l in enumerate(layers):
        view = np.frombuffer(bundle.blob(l), dtype=np.uint8)
        for fd, (_, _, _, part, _, off) in zip(fds, todo):
            for x in range(experts):
                for into, piece in pieces(view, l, x, part):
                    slot[into:into + piece.nbytes] = piece
                write_at(fd, slot, off[(l, x)])
        if progress:
            progress(i + 1, len(layers))
    for fd, (path, ident, size, part, parts, _) in zip(fds, todo):
        os.fsync(fd)
        os.close(fd)
        with open(path + ".json", "w") as f:
            json.dump(dict(ident=ident, bytes=size, part=part, parts=parts), f)


# ⭐ THE EXPERTS LOCKED INTO RAM (`experts_lock`, node-01's NN-99 ④ item): opt-in, for a machine whose `vm.swappiness`
#   still lets the kernel swap experts out to keep files cached — Linux's mlock, which needs `ulimit -l` past the experts.
def lock_check(on, where, total):
    """`experts_lock` refused at once where it cannot hold: the experts not copied into RAM (`where` not "ram"), or more of
    them (`total` bytes) than this process may lock (`ulimit -l`) — before a long load."""
    if not on:
        return
    if where != "ram":
        said = {"card": "on the card", "disk": "read from the disk", "mapped": "mapped from the pack or their files"}.get(where, repr(where))
        raise Refused("experts_lock locks the experts the CPU holds in RAM (experts_from 'ram'), and here they are %s" % said)
    most = lockable()
    if most == 0:
        raise Refused("experts_lock is Linux's (mlock): on this system the experts stay in RAM as the system keeps them")
    if most is not None and total > most:
        raise Refused("experts_lock: %.1f GB of experts, and this process may lock %.1f GB (ulimit -l) — raise memlock for "
                      "this user (/etc/security/limits.conf, then log in again) or leave experts_lock off"
                      % (total / 2 ** 30, most / 2 ** 30))


def lock_experts(on, addresses, size):
    """`experts_lock`'s lock once the experts are loaded: each at `addresses`, `size` bytes. The GB locked, 0 when off."""
    if not on:
        return 0.0
    try:
        return lock(spans(addresses, size)) / 2 ** 30
    except OSError as ex:
        raise Refused("experts_lock: the system would not lock the experts — %s" % ex.strerror)
