"""A MODEL'S ROUTED EXPERTS SPLIT A FILE A SOCKET, AND THE PACK PUT BACK TOGETHER FROM THEM (NN-148 ③).

⚖ 2026-11-03 *"we reinstate the per socket file split … via a command like silvann <model> --split or --stitch to put it
back together, and then we leave it parked using the optane as the nvme"*: a two-socket machine keeps each socket's share
of every routed expert on that socket's own memory — a DL380's PMem in App Direct, a mount a socket — read into RAM as the
disk is (▶ `experts_file`), and a boot over two sockets with `experts_dir` [DIR0, DIR1] reads each share from its own.

  usage:  silvann split MODEL [--layers A-B]
          silvann stitch MODEL [--layers A-B]
          silvann split MODEL DIR0 DIR1 [--layers A-B] [--rest DIR] [--skip SUBDIR ...]
          silvann stitch DIR0 DIR1 OUT
          (main/silvann_core.py split-experts …, stitch-experts … — MODEL a model's name under models/, or its folder)

  split MODEL    the folder split in place, as a boot over two sockets splits it (▶ `settle`): each socket's share in
                 the folder's `experts/`, each layer's experts then freed from its file — a layer at a time
  stitch MODEL   the pack put back in place, as a boot over one socket does: each layer's file the pack's sha256 again

  split   each socket's share of every routed expert, in its slots' layout, to its own DIR — the very files a two-socket
          boot with `experts_dir` [DIR0, DIR1] reads (`experts_file`'s partNofP.bin, its identity the pack's), so that boot
          writes nothing — and everything else of the pack to DIR0/rest (or `--rest`): each layer's bytes no expert covers,
          every other file whole (the pack's own `experts/` left out: a boot's files, made from the pack). The pack is
          never changed. `--layers` a test's subset: those layers' files alone, the boot's of that subset. `--skip` a folder
          of the pack left out of the rest (DeepSeek's 190 GB `engram/` on a test's disk): named in the split, and the
          stitch then makes the pack but for it.
  stitch  the pack whole again in OUT from the split, byte for byte — every file checked against the md5 the split took.

The models whose experts a boot splits over sockets: GLM 5.3 (`glm5`) and DeepSeek-V4.1 (`deepseek4`); each answers its
shares (`socket_shares`): the slot's plan a part, the layers, each part's pieces out of a layer and their places in it.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

from . import experts_file
from .model_folder import ModelFolder, Refused
from .system import BINARY, read_at, write_at

SPLIT = "split.json"
CHUNK = 64 << 20


def shares_of(folder, sockets, layers):
    """The folder's family's socket shares (▶ `glm5.socket_shares`, `deepseek4.socket_shares`)."""
    from . import deepseek4, glm5
    for mod in (glm5, deepseek4):
        if folder.architecture in mod.ARCHITECTURES:
            return mod.socket_shares(folder, sockets, layers)
    sys.exit("%s is a %r model: only GLM 5.3's and DeepSeek-V4.1's experts are split over sockets" % (folder.path,
                                                                                                    folder.architecture))


def layer_files(folder, layers):
    """{the path of a layer's weights inside the folder: layer} for every layer of `layers`, by the bundle's layout."""
    b = folder.bundle
    out = {}
    for l in layers:
        rel = os.path.relpath(os.path.join(b.layer_dir(l), b.manifest["layout"]["weights_bin"]), folder.path)
        out[rel] = l
    return out


def covered(sh, l, parts, size):
    """The runs of layer `l`'s blob some part of some expert takes, merged: an [n, 2] array of (start, end)."""
    starts, ends = [], []
    for x in range(sh["experts"]):
        for s in range(parts):
            for _into, src, n, count, step in sh["places"](l, x, s):
                a = src + np.arange(count, dtype=np.int64) * step
                starts.append(a)
                ends.append(a + n)
    a, e = np.concatenate(starts), np.concatenate(ends)
    order = np.argsort(a, kind="stable")
    a, e = a[order], e[order]
    if e.max() > size or a.min() < 0:
        raise SystemExit("layer %d: an expert's places run past its file (%d bytes)" % (l, size))
    reach = np.maximum.accumulate(e)
    new = np.ones(a.size, bool)
    new[1:] = a[1:] > reach[:-1]
    first = np.flatnonzero(new)
    last = np.append(first[1:], a.size) - 1
    return np.stack([a[first], reach[last]], axis=1)


def gaps(runs, size):
    """What `runs` (merged, sorted) leave of [0, size), as (start, end) runs."""
    out, at = [], 0
    for a, e in runs:
        if a > at:
            out.append((at, int(a)))
        at = max(at, int(e))
    if at < size:
        out.append((at, size))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ⭐ THE FOLDER SPLIT IN PLACE (⚖ 2026-11-09 *"the setup should be automated to see if the model state matches the cpu
#   config, if not to automatically split/stitch. this way copying a split model to a single core would make the
#   stitching automatic"* · *"make the move one layer at a time, so even space constrained disks can make it happen"*):
#   a two-socket boot keeps the model once — each socket's share in `experts/partNofP.bin`, the very files it reads, and
#   each moved layer's file with its experts' bytes freed (holes: its size, its dense bytes and their places unchanged) —
#   and a one-socket boot puts the pack back. A layer at a time either way, so the disk needs one layer's experts spare:
#   its shares written and read back, then recorded, then its bytes in the pack freed; put back, its file checked against
#   the sha256 recorded (the pack's own, as Hugging Face lists it), then recorded, then its shares' bytes freed. Cut short
#   anywhere, the next boot carries on from the record. A folder, or a layer's file, reached through a symbolic link is
#   another folder's and never moved (as `fetch` leaves a linked file as it is).
STATE = "moved.json"            # <folder>/experts/moved.json: {"sockets", "architecture", "moved": {file: {layer, bytes, sha256}}}
_LOCK = ".folder.lock"          # <folder>/experts/.folder.lock: a boot holds it shared, a move alone


def state_of(path):
    """The folder's split in place — its record — or None where every layer's file holds its experts."""
    try:
        with open(os.path.join(path, "experts", STATE)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def hold(path, alone=False):
    """The folder held for this process: a lock on `experts/.folder.lock`, shared by every boot of it and taken alone by
    a move — so no layer's bytes are freed under another process that reads them. Its descriptor, which the caller keeps
    open as long as it reads the folder (▶ `Machine.shutdown`) — None where the folder cannot be written; Refused where
    another process holds it the other way."""
    import fcntl
    d = os.path.join(path, "experts")
    try:
        os.makedirs(d, exist_ok=True)
        fd = os.open(os.path.join(d, _LOCK), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as ex:                   # a folder this process cannot write: none of it is moved, so nothing to hold
        if alone:
            raise Refused("%s cannot be written (%s): its experts stay where they are" % (path, ex.strerror))
        return None
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if alone else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Refused("%s is %s by another process, which %s: stop it first" % (
            path, "open" if alone else "being split or stitched",
            "reads the experts this boot would move" if alone else "moves the experts this boot would read"))
    return fd


def _punch(fd, start, length):
    """The file's bytes [start, start + length) freed — a hole, read as zeros — its size kept (Linux's fallocate)."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    if length > 0 and libc.fallocate(fd, 0x02 | 0x01, start, length) != 0:      # FALLOC_FL_PUNCH_HOLE | KEEP_SIZE
        err = ctypes.get_errno()
        raise OSError(err, "fallocate(PUNCH_HOLE) at %d, %d bytes: %s" % (start, length, os.strerror(err)))


def frees_ranges(directory):
    """Whether files in `directory` can have a range freed in place (Linux, and a filesystem that punches holes: ext4,
    xfs, btrfs, tmpfs …) — tried on a file of its own there, which is then removed."""
    if not sys.platform.startswith("linux"):
        return False
    probe = os.path.join(directory, ".punch_probe")
    try:
        with open(probe, "wb") as f:
            f.write(b"\1" * (1 << 20))
            f.flush()
            os.fsync(f.fileno())
            before = os.fstat(f.fileno()).st_blocks
            _punch(f.fileno(), 0, 1 << 19)
            return os.fstat(f.fileno()).st_blocks < before
    except OSError:
        return False
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def _linked(path, rel):
    """Whether the folder, or anything on the way to its file `rel`, is a symbolic link."""
    parts = rel.split(os.sep)
    return os.path.islink(path.rstrip(os.sep)) or any(os.path.islink(os.path.join(path, *parts[:i + 1]))
                                                       for i in range(len(parts)))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(path, record):
    """The record written whole (a file beside it, then renamed over it); no record and no shares once nothing is moved."""
    d = os.path.join(path, "experts")
    target = os.path.join(d, STATE)
    if not record["moved"]:
        for name in [STATE] + ["part%dof%d.bin%s" % (s + 1, record["sockets"], e) for s in range(record["sockets"])
                               for e in ("", ".json")]:
            try:
                os.remove(os.path.join(d, name))
            except FileNotFoundError:
                pass
        return
    with open(target + ".new", "w") as f:
        json.dump(record, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(target + ".new", target)
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class Shares:
    """A split folder's shares: each socket's file mapped, a layer's slot read where it lies (the layout `experts_file`
    writes: the model's routed layers in order, a layer's experts a slot each)."""

    def __init__(self, path, sh, sockets, moved=()):
        self.sh, self.sockets, self.moved = sh, sockets, set(moved)      # `moved`: the layers a boot reads from them
        self.paths = [os.path.join(path, "experts", "part%dof%d.bin" % (s + 1, sockets)) for s in range(sockets)]
        self.maps = [np.memmap(p, dtype=np.uint8, mode="r") for p in self.paths]

    def at(self, s, l, x=0):
        """Where part `s` of layer `l`'s expert `x` starts in its file."""
        return (self.sh["all_layers"].index(l) * self.sh["experts"] + x) * experts_file.of_part(self.sh["plan"], s).slot_bytes

    def slot(self, s, l, x):
        n = experts_file.of_part(self.sh["plan"], s).slot_bytes
        at = self.at(s, l, x)
        return self.maps[s][at:at + n]

    def fill(self, blob, l, xs):
        """`blob` — a writable view of layer `l`'s file (or a private map of it: ▶ `whole_view`) — with experts `xs` put
        back from every share, for what reads a whole expert out of the pack."""
        for x in xs:
            for s in range(self.sockets):
                slot = self.slot(s, l, x)
                for into, src, n, count, step in self.sh["places"](l, x, s):
                    if count == 1:
                        blob[src:src + n] = slot[into:into + n]
                    else:
                        dst = np.lib.stride_tricks.as_strided(blob[src:], shape=(count, n), strides=(step, 1))
                        dst[...] = slot[into:into + count * n].reshape(count, n)


def whole_view(folder, l, shares, xs):
    """Layer `l`'s file as the pack has it for experts `xs`: where the layer is moved, the file mapped privately (nothing
    written to it) with those experts put back from the shares; else the bundle's own map of it."""
    rel = list(layer_files(folder, [l]))[0]
    rec = state_of(folder.path)
    if shares is None or rec is None or rel not in rec["moved"]:
        return np.frombuffer(folder.bundle.blob(l), dtype=np.uint8)
    blob = np.memmap(os.path.join(folder.path, rel), dtype=np.uint8, mode="c")
    shares.fill(blob, l, xs)
    return blob


def settle(folder, sockets, layers, elsewhere=False, progress=None, say=lambda s: print(s, flush=True)):
    """The folder as this boot reads it, its routed layers `layers` moved a layer at a time where they are not: split
    where the boot runs two sockets and reads their shares from the folder's own `experts/` (`experts_dir` not named
    elsewhere), the folder its own (no symbolic link on the way to a layer's file) and its filesystem one that frees a
    file's ranges; else whole. Two sockets with `experts_dir` named elsewhere move nothing (copies there). Its answer: the
    shares to read (a `Shares`, its `moved` the layers they hold; None where none is moved), and the folder's lock, held
    shared (▶ `hold`) — the caller keeps it while it reads the folder."""
    rec = state_of(folder.path)
    sh = shares_of(folder, 2, None) if (sockets == 2 or rec) else None
    files = layer_files(folder, [l for l in layers if sh is None or l in sh["all_layers"]])
    moved = set(rec["moved"]) if rec else set()
    linked = {rel for rel in files if _linked(folder.path, rel)}
    if linked & moved:
        raise Refused("%s's layer %d is split in place and reached through a symbolic link: open the folder it is in"
                      % (folder.path, files[sorted(linked & moved)[0]]))
    # two sockets reading their files elsewhere (`experts_dir` named): the folder left as it is — its shares, where it is
    #   split, what a load into RAM reads; the copies elsewhere never cut from its freed bytes (▶ `experts_file.ensure`)
    if sockets == 2 and elsewhere:
        here = {l for rel, l in files.items() if rel in moved}
        return (Shares(folder.path, sh, 2, here) if here else None), hold(folder.path)
    # split only where every layer this boot reads is the folder's own: a boot reads its layers all one way
    want = (sockets == 2 and bool(files) and not linked
            and frees_ranges(os.path.dirname(os.path.join(folder.path, next(iter(files))))))
    todo = [rel for rel in files if (rel not in moved) == want]
    if todo:
        fd = hold(folder.path, alone=True)
        try:
            record = state_of(folder.path) or dict(sockets=2, architecture=folder.architecture, moved={})
            if record["sockets"] != 2:
                raise Refused("%s is split over %d sockets, which this release does not read" % (folder.path, record["sockets"]))
            (_split_layers if want else _stitch_layers)(folder, sh, record, sorted(files[rel] for rel in todo), progress, say)
        finally:
            os.close(fd)
    return (Shares(folder.path, sh, 2, set(files.values())) if want else None), hold(folder.path)


def _split_layers(folder, sh, record, layers, progress, say):
    t0 = time.time()
    d = os.path.join(folder.path, "experts")
    plans = [experts_file.of_part(sh["plan"], s) for s in range(2)]
    paths = [os.path.join(d, "part%dof2.bin" % (s + 1)) for s in range(2)]
    fds = []
    for s, p in enumerate(paths):
        size = plans[s].slot_bytes * sh["experts"] * len(sh["all_layers"])
        if os.path.exists(p + ".json"):              # a file a boot wrote whole: the record says what is moved from now
            os.remove(p + ".json")
        fd = os.open(p, os.O_RDWR | os.O_CREAT | BINARY, 0o644)
        if os.fstat(fd).st_size != size:
            if record["moved"]:
                raise Refused("%s is %d bytes, not the %d its split was written with" % (p, os.fstat(fd).st_size, size))
            os.ftruncate(fd, size)
        fds.append(fd)
    rels = {l: rel for rel, l in layer_files(folder, layers).items()}
    try:
        for i, l in enumerate(layers):
            src, idx = os.path.join(folder.path, rels[l]), sh["all_layers"].index(l)
            size, digest = os.path.getsize(src), _sha256(src)
            view = np.memmap(src, dtype=np.uint8, mode="r")
            for s, fd in enumerate(fds):
                slot = np.zeros(plans[s].slot_bytes, np.uint8)
                for x in range(sh["experts"]):
                    for into, piece in sh["pieces"](view, l, x, s):
                        slot[into:into + piece.nbytes] = piece
                    at = (idx * sh["experts"] + x) * plans[s].slot_bytes
                    write_at(fd, slot, at)
                    if read_at(fd, slot.nbytes, at) != slot.tobytes():
                        raise Refused("layer %d's share %d read back otherwise than it was written" % (l, s))
                os.fsync(fd)
            runs = covered(sh, l, 2, size)
            del view
            record["moved"][rels[l]] = dict(layer=l, bytes=size, sha256=digest)
            _save(folder.path, record)
            fd = os.open(src, os.O_RDWR | BINARY)
            try:
                for a, e in runs:
                    _punch(fd, int(a), int(e - a))
                os.fsync(fd)
            finally:
                os.close(fd)
            say("  the experts split over two sockets: layer %d (%d of %d, %.0f s)" % (l, i + 1, len(layers), time.time() - t0))
            if progress:
                progress(i + 1, len(layers))
    finally:
        for fd in fds:
            os.close(fd)


def _stitch_layers(folder, sh, record, layers, progress, say):
    t0 = time.time()
    shares = Shares(folder.path, sh, 2)
    rels = {l: rel for rel, l in layer_files(folder, layers).items()}
    for i, l in enumerate(layers):
        rel = rels[l]
        src, ent = os.path.join(folder.path, rel), record["moved"][rel]
        blob = np.memmap(src, dtype=np.uint8, mode="r+")
        shares.fill(blob, l, range(sh["experts"]))
        blob.flush()
        del blob
        fd = os.open(src, os.O_RDONLY | BINARY)
        os.fsync(fd)
        os.close(fd)
        if os.path.getsize(src) != ent["bytes"] or _sha256(src) != ent["sha256"]:
            raise Refused("layer %d put back from its shares is not the pack's (its sha256 differs): its shares are kept, "
                          "%s as it is — download the pack again (silvann download)" % (l, rel))
        del record["moved"][rel]
        _save(folder.path, record)
        for s in range(2):
            if record["moved"]:
                fd = os.open(shares.paths[s], os.O_RDWR | BINARY)
                try:
                    _punch(fd, shares.at(s, l), experts_file.of_part(sh["plan"], s).slot_bytes * sh["experts"])
                finally:
                    os.close(fd)
        say("  the pack put back together: layer %d (%d of %d, %.0f s)" % (l, i + 1, len(layers), time.time() - t0))
        if progress:
            progress(i + 1, len(layers))


def forget(path, rels):
    """The moved files of `rels` taken out of the record — an update replacing them with the pack's own — and their
    layers' shares freed."""
    rec = state_of(path)
    if not rec or not set(rels) & set(rec["moved"]):
        return
    fd = hold(path, alone=True)
    try:
        sh = shares_of(ModelFolder(path), 2, None)
        shares = Shares(path, sh, 2)
        for rel in sorted(set(rels) & set(rec["moved"])):
            l = rec["moved"].pop(rel)["layer"]
            _save(path, rec)
            if rec["moved"]:
                for s in range(2):
                    wfd = os.open(shares.paths[s], os.O_RDWR | BINARY)
                    try:
                        _punch(wfd, shares.at(s, l), experts_file.of_part(sh["plan"], s).slot_bytes * sh["experts"])
                    finally:
                        os.close(wfd)
    finally:
        os.close(fd)


def held_as(path, rel, sha256):
    """Whether the folder holds the pack's file `rel` (its sha256 `sha256`) as a moved layer of its split."""
    rec = state_of(path)
    return bool(rec) and rec["moved"].get(rel, {}).get("sha256") == sha256


def split(args):
    folder = ModelFolder(args.model)
    dirs = [os.path.abspath(d) for d in args.dirs]
    parts = len(dirs)
    layers = None
    if args.layers:
        a, _, b = args.layers.partition("-")
        layers = list(range(int(a), int(b or a) + 1))
    sh = shares_of(folder, parts, layers)
    t0 = time.time()
    # ① each socket's file, as a boot writes it: the same identity, so a boot finds it and writes nothing
    paths = experts_file.ensure(folder.bundle, [(None, s, parts) for s in range(parts)], sh["plan"], sh["layers"],
                                sh["experts"], dirs, sh["pieces"], sh["all_layers"],
                                progress=lambda i, n: print("  the experts: layer %d of %d (%.0f s)" % (i, n, time.time() - t0),
                                                            flush=True))
    # ② the rest: each routed layer's bytes no expert covers, and every other file whole
    rest = os.path.abspath(args.rest or os.path.join(dirs[0], "rest"))
    routed = layer_files(folder, sh["layers"])
    every = layer_files(folder, range(folder.bundle.n_layers + 1)) if layers is not None else {}
    files = {}
    for top, subdirs, names in os.walk(folder.path, followlinks=True):
        rel_top = os.path.relpath(top, folder.path)
        if rel_top == ".":
            subdirs[:] = [d for d in subdirs if d != "experts" and d not in (args.skip or [])]
        for name in names:
            rel = os.path.normpath(os.path.join(rel_top, name))
            src = os.path.join(folder.path, rel)
            if rel in every and rel not in routed:
                continue                                          # a layer the subset leaves out
            os.makedirs(os.path.dirname(os.path.join(rest, rel)), exist_ok=True)
            if rel in routed:
                l, size = routed[rel], os.path.getsize(src)
                left = gaps(covered(sh, l, parts, size), size)
                view = np.frombuffer(folder.bundle.blob(l), dtype=np.uint8)
                with open(os.path.join(rest, rel + ".rest"), "wb") as f:
                    for a, e in left:
                        f.write(view[a:e].tobytes())
                np.save(os.path.join(rest, rel + ".gaps.npy"), left)
                view = None
                folder.bundle.release(l)
                files[rel] = dict(kind="layer", layer=l, bytes=size, md5=md5_of(src), rest=int((left[:, 1] - left[:, 0]).sum()))
                print("  layer %d: %d of its %d bytes the experts', the rest %d (%.0f s)"
                      % (l, size - files[rel]["rest"], size, files[rel]["rest"], time.time() - t0), flush=True)
            else:
                shutil.copyfile(src, os.path.join(rest, rel))
                files[rel] = dict(kind="whole", bytes=os.path.getsize(src), md5=md5_of(src))
    record = dict(model=os.path.basename(folder.path), architecture=folder.architecture, sockets=parts, layers=layers,
                  dirs=dirs, rest=rest, parts={str(s): paths[s] for s in range(parts)}, files=files, skipped=args.skip or [])
    for d in dirs:
        with open(os.path.join(d, SPLIT), "w") as f:
            json.dump(record, f, indent=1)
    print("split %s over %d sockets in %.0f s: %s; the rest in %s" % (folder.path, parts, time.time() - t0,
                                                                     ", ".join(paths[s] for s in range(parts)), rest))


def stitch(args):
    record = json.load(open(os.path.join(args.dirs[0], SPLIT)))
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit("%s is not empty: the pack is stitched into a folder of its own" % out)
    parts, rest, t0 = record["sockets"], record["rest"], time.time()
    paths = {int(s): os.path.join(os.path.abspath(d), os.path.basename(record["parts"][str(s)]))
             for s, d in enumerate(args.dirs)}
    # ① every whole file, then the folder read as a pack — its records' places are in its index, copied whole
    for rel, f in record["files"].items():
        if f["kind"] == "whole":
            os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
            shutil.copyfile(os.path.join(rest, rel), os.path.join(out, rel))
    folder = ModelFolder(out)
    sh = shares_of(folder, parts, record["layers"])
    plans = [experts_file.of_part(sh["plan"], s) for s in range(parts)]
    where = {s: {(l, x): off for l, x, off in experts_file.layout(sh["layers"], sh["experts"], plans[s].slot_bytes)}
             for s in range(parts)}
    maps = {s: np.memmap(paths[s], dtype=np.uint8, mode="r") for s in range(parts)}
    # ② each routed layer: its rest into the gaps, each part's slots back where the pack keeps them
    for rel, f in record["files"].items():
        if f["kind"] != "layer":
            continue
        l, size = f["layer"], f["bytes"]
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        blob = np.memmap(os.path.join(out, rel), dtype=np.uint8, mode="w+", shape=(size,))
        left = np.load(os.path.join(rest, rel + ".gaps.npy"))
        kept = np.fromfile(os.path.join(rest, rel + ".rest"), dtype=np.uint8)
        at = 0
        for a, e in left:
            blob[a:e] = kept[at:at + e - a]
            at += e - a
        for s in range(parts):
            slot_bytes = plans[s].slot_bytes
            for x in range(sh["experts"]):
                off = where[s][(l, x)]
                slot = maps[s][off:off + slot_bytes]
                for into, src, n, count, step in sh["places"](l, x, s):
                    if count == 1:
                        blob[src:src + n] = slot[into:into + n]
                    else:
                        dst = np.lib.stride_tricks.as_strided(blob[src:], shape=(count, n), strides=(step, 1))
                        dst[...] = slot[into:into + count * n].reshape(count, n)
        blob.flush()
        del blob
        print("  layer %d stitched (%.0f s)" % (l, time.time() - t0), flush=True)
    # ③ every file against the md5 the split took
    bad = [rel for rel, f in record["files"].items() if md5_of(os.path.join(out, rel)) != f["md5"]]
    for rel in bad:
        print("⛔ %s differs from the pack's" % rel, flush=True)
    print("%s %d files stitched into %s in %.0f s, %d differ from the pack's%s" % (
        "✅" if not bad else "⛔", len(record["files"]), out, time.time() - t0, len(bad),
        "; left out by the split: %s" % ", ".join(record.get("skipped", [])) if record.get("skipped") else ""), flush=True)
    sys.exit(1 if bad else 0)


def in_place(model, sockets, layers=None):
    """The folder split over two sockets (`sockets` 2) or put back whole (1) in place, as a boot over that many does
    (▶ `settle`) — every routed layer, or `layers`."""
    folder = ModelFolder(model)
    every = shares_of(folder, 2, None)["all_layers"]
    run = [l for l in every if layers is None or l in layers]
    files = layer_files(folder, run)
    if sockets == 2:
        linked = [l for rel, l in files.items() if _linked(folder.path, rel)]
        if linked:
            sys.exit("%s's layer %d is reached through a symbolic link: its file is another folder's, never moved (--layers "
                     "the folder's own)" % (folder.path, linked[0]))
        if not frees_ranges(os.path.dirname(os.path.join(folder.path, next(iter(files))))):
            sys.exit("%s's filesystem does not free a file's ranges (Linux's ext4, xfs, btrfs do): split it to a folder a "
                     "socket instead — silvann split MODEL DIR0 DIR1" % folder.path)
    t0 = time.time()
    _, fd = settle(folder, sockets, run)
    if fd is not None:
        os.close(fd)
    rec = state_of(folder.path)
    print("%s %s in %.0f s: %d of its %d routed layers split over two sockets" % (
        "✅", folder.path, time.time() - t0, len(rec["moved"]) if rec else 0, len(every)), flush=True)


def main(argv, models=None):
    """`argv` [split|stitch, …] (▶ the usage above); a MODEL that is not a folder is a name under `models`."""
    ap = argparse.ArgumentParser(prog="silvann", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("split")
    s.add_argument("model")
    s.add_argument("dirs", nargs="*")
    s.add_argument("--layers")
    s.add_argument("--rest")
    s.add_argument("--skip", nargs="*")
    t = sub.add_parser("stitch")
    t.add_argument("dirs", nargs="+")
    t.add_argument("--layers")
    args = ap.parse_args(argv)
    named = lambda m: m if os.path.isdir(m) or models is None else os.path.join(str(models), m)   # noqa: E731
    layers = None
    if args.layers:
        a, _, b = args.layers.partition("-")
        layers = list(range(int(a), int(b or a) + 1))
    if args.what == "split":
        args.model = named(args.model)
        if not args.dirs:
            return in_place(args.model, 2, layers)
        if len(args.dirs) < 2:
            sys.exit("a split to folders is a folder a socket: two or more (or none, the folder split in place)")
        if state_of(os.path.abspath(args.model)):
            sys.exit("%s is split in place: put it back first (silvann stitch %s)" % (args.model, args.model))
        return split(args)
    if len(args.dirs) == 1:
        return in_place(named(args.dirs[0]), 1, layers)
    if len(args.dirs) < 3:
        sys.exit("silvann stitch MODEL (a folder split in place), or silvann stitch DIR0 DIR1 OUT")
    args.dirs, args.out = args.dirs[:-1], args.dirs[-1]
    stitch(args)
