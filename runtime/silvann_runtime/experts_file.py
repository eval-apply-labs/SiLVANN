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


def available():
    """The memory the system says it can give without swapping, in bytes."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def choose(experts_from, need):
    """`"ram"` or `"disk"` for `experts_from` as a config says it: `"auto"` is the disk when the experts would leave less
    than a fifth of what is available — `REASONED`: the rest is the system's and the page cache's."""
    if experts_from not in ("auto", "ram", "disk"):
        raise Refused("experts_from is 'auto', 'ram' or 'disk', not %r" % (experts_from,))
    if experts_from == "auto":
        return "disk" if need > 0.8 * available() else "ram"
    return experts_from


def map_experts(m, bundle, workers, plan, layers, experts, directory, pieces, all_layers, expert_type=0,
                progress=None):
    """Every layer's experts on each CPU worker, from its file, mapped over its slots.

    workers      [(worker, part, parts)] — each maps its own file
    plan         the slot's shape (an `NL.SlotPlan`)
    layers       the routed layers this run holds; `all_layers` the model's, which name the file when they are the same
    pieces       pieces(view, layer, expert, part) -> [(offset in the slot, a uint8 array)], `view` the layer's blob
    """
    tag = "" if list(layers) == list(all_layers) else ".layers-" + hashlib.sha256(
        json.dumps(list(layers)).encode()).hexdigest()[:10]
    todo, mapped = [], []
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
        size = max(at.values()) - lo + plan.slot_bytes
        layout = [[l, x, at[(l, x)] - lo] for l in layers for x in range(experts)]
        ident = hashlib.sha256(json.dumps(dict(pack=bundle.manifest, parts=parts, part=part, size=size,
                                               slot=[plan.up_data, plan.up_lut, plan.down_data, plan.down_lut],
                                               layout=layout), sort_keys=True, default=str).encode()).hexdigest()
        path = os.path.join(directory, "part%dof%d%s.bin" % (part + 1, parts, tag))
        try:
            fresh = json.load(open(path + ".json"))["ident"] == ident and os.path.getsize(path) == size
        except (OSError, ValueError, KeyError):
            fresh = False
        if not fresh:
            todo.append((path, ident, size, part, parts, {(l, x): off for l, x, off in layout}))
        mapped.append((w, lo, size, path))
    if todo:
        _write(bundle, todo, plan, layers, experts, directory, pieces, progress)
    for w, lo, size, path in mapped:
        m.act_as(w)
        m.map_file(lo, size, path, 0)


def _write(bundle, todo, plan, layers, experts, directory, pieces, progress):
    """Each file: every expert's part in its slot's layout, at its slot's offset — a layer's blob read once for all."""
    os.makedirs(directory, exist_ok=True)
    fds = []
    for path, ident, size, part, parts, off in todo:
        if os.path.exists(path + ".json"):
            os.remove(path + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        fds.append(fd)
    slot = np.zeros(plan.slot_bytes, np.uint8)
    for i, l in enumerate(layers):
        view = np.frombuffer(bundle.blob(l), dtype=np.uint8)
        for fd, (_, _, _, part, _, off) in zip(fds, todo):
            for x in range(experts):
                for into, piece in pieces(view, l, x, part):
                    slot[into:into + piece.nbytes] = piece
                os.pwrite(fd, slot, off[(l, x)])
        if progress:
            progress(i + 1, len(layers))
    for fd, (path, ident, size, part, parts, _) in zip(fds, todo):
        os.fsync(fd)
        os.close(fd)
        with open(path + ".json", "w") as f:
            json.dump(dict(ident=ident, bytes=size, part=part, parts=parts), f)
