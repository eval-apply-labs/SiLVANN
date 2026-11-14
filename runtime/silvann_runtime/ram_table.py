"""AN N-GRAM TABLE COPIED INTO RAM — Flash-Next's (`ngram_qwen4`) and DeepSeek's (`engram_ds4`) rows read from memory, a
token's rows never from the disk (NN-185).

A model with an n-gram memory reads a position's rows at random places in a table of tens of gigabytes (Flash-Next's 51
GB, ~16 rows a position; DeepSeek's two layers' 203 GB). Read from the table's file they come through the system's file
cache, which the disk fills as each row is first asked for: on a slow disk that sets the decode's pace (⚖ 2026-11-11, the
architect's Windows VM — 7 tokens/s, 4 by 3,000 tokens, the disk 80% busy — *"the engram in ram is a good thing to
add"*). `RamTable` copies a span of the file into memory this process holds, at boot, in long reads as the experts' read
ahead does (▶ `nn_load_bundle.warm`), its progress said; a row is then read from there — `read(size, offset)` as
`system.Positional` answers it, `rows` many at once — and the engine's verb is handed its address (▶ nn's
`nn__ngram__rows`, A TABLE IN RAM). The bytes are the file's, so the rows, and the tokens, are the same either way.

⭐ OR THE SYSTEM'S FILE CACHE (`"cache"`): the file read through once at boot in the same long runs, nothing kept but the
cache's pages, and every row read from the file after as `"disk"` reads it — from the cache, a read of microseconds,
wherever the system still holds the page, from the disk where it gave the page to something else. Nothing of it is this
process's own memory, so the system takes it back as it needs the room, and a read that misses costs what `"disk"`'s
does. Windows charges a copy its whole size against the commit when it is allocated, as it does every allocation, and
the file cache nothing (▶ `place`).

`place` chooses where a table is read from: RAM where it fits beside what else the process holds in its plan — the
experts' memory, its own — and the system's share; else the disk. On Windows the file cache where it fits beside them,
else the disk.
"""
import os
import threading
import time

import numpy as np

from . import experts_file
from .machine import CPU_FAMILIES, OPENCL1_FAMILIES, own_memory
from .model_folder import Refused
from .system import BINARY, WINDOWS

RUN = 16 << 20                # a read's bytes, as the experts' read-ahead's (`nn_load_bundle.WARM_RUN`)
READERS = 4                   # the threads a copy reads with, each every READERS-th run (▶ `RamTable._copy`)
WHERE = ("auto", "ram", "cache", "disk")
STEPS = 100                   # a copy's progress, said as a load's `progress(done, total, what)` is: this many steps


def steps(progress, total, what="engrams"):
    """A copy's `tick(bytes done)` for a load's `progress(done, total, what)`, called a step at a time of STEPS — not a
    read at a time — so a caller that prints each call prints a hundred lines, not thousands; None where there is no
    `progress`. `what` is what the load says is loading: the tables copied here are an n-gram memory's, the engrams."""
    if progress is None:
        return None
    said = [0]

    def tick(done):
        step = done * STEPS // max(1, total)
        if step > said[0]:
            said[0] = step
            progress(step, STEPS, what)
    return tick


def card_in_ram(machine):
    """Whether the machine's card is the CPU's RAM — the CPU alone, or an OpenCL 1.2 GPU without memory of its own — so
    the dense model's pool is held beside a table in RAM."""
    if machine is None:
        return False
    return machine.silicon in CPU_FAMILIES + ("host",) or (machine.silicon in OPENCL1_FAMILIES and not own_memory(machine))


def place(said, option, need, beside, own=None):
    """`"ram"`, `"cache"` or `"disk"` for a table's `option` as a config says it. `"auto"`: RAM where the table's `need`
    bytes, what the process holds beside it in its plan (`beside`: the experts in RAM, or their cache), its own memory
    (`own`, a family's `HOST_OWN` where it holds more than `experts_file.OWN`) and the system's share —
    `experts_file.FILES_SHARE` of what this process may hold, and the spare — fit in what this process may hold
    (`experts_file.in_ram`'s rule); else the disk. A table is chosen after the experts: "auto" never takes RAM from them.

    ⭐ ON WINDOWS "auto" IS THE FILE CACHE where the table, what is beside it, the process's own and the spare fit in what
    the system has available — the share left out, since the system takes the cache's pages back as it needs them — else
    the disk. ⛔ NOT A COPY THERE: Windows charges every allocation its whole size against the commit limit (the RAM and
    the page file) as it is made, and an NVIDIA card's under its default driver model too, by the forum reports
    measurements/2026-11-11_nn185_table_cache_node03.md ① cites. `REASONED` there from the architect's VM (an RTX PRO
    6000, 0.28.0, Flash-Next's experts in RAM and a 45 GB tier: 133 GB of 137 committed) against the same setup's own
    memory on node03, ~63 GiB (`MEASURED`): a 51 GB copy would have taken the commit past its limit, or grown the page
    file by as much, and left the rest of the system nothing."""
    if said not in WHERE:
        raise Refused("%s is 'auto', 'ram', 'cache' or 'disk', not %r" % (option, said))
    if said != "auto":
        return said
    held = experts_file.limit()
    if WINDOWS:
        return "cache" if experts_file.in_ram(need + beside, held, own_bytes=own) else "disk"
    return "ram" if experts_file.in_ram(need + beside + int(experts_file.FILES_SHARE * held), held, own_bytes=own) else "disk"


def said_where(where, need, beside):
    """The line a setup says of a table's place: its size, and what it is beside — in the GB (2³⁰ bytes) the rest of the
    setup's line says."""
    gb = lambda n: n / 2**30                                        # noqa: E731
    return (WHERE_SAID[where] + " (%.0f GB, beside %.0f GB of experts, %.0f GB this process may hold)") % (
        gb(need), gb(beside), gb(experts_file.limit()))


WHERE_SAID = {"ram": "in RAM", "cache": "read into the system's file cache", "disk": "read from the disk"}


def read_through(path, offset, size, what, tick=None, say=None, into=None):
    """⭐ THE FILE READ IN LONG RUNS — straight into `into` (a copy's memory, ▶ `RamTable`), or, `into` None, through the
    system's file cache alone (`"cache"`, ▶ `warm`) — RUN bytes a read, READERS threads each with a descriptor of its own
    taking every READERS-th run, so the runs go in the file's order, a few in flight at once. Answers the seconds it took.
    `MEASURED` node03 2026-11-11, the first 16 GB of Flash-Next's table cold (copy_rate_nvme.log, copy_rate_hdd.log in
    measurements/2026-11-11_nn185_ngram_in_ram_node03_artifacts/): from the NVMe RAID one thread 0.98 GB/s, four 2.99,
    eight 3.06; from the HDD RAID (6 GB) one 0.29, two 0.37, four 0.29 — a spinning disk no slower for it.
    Into a copy, each run's pages are let go of once it is read (`posix_fadvise` DONTNEED, where the system has it), so the table
    is held once, not again in the system's cache — which, filling, would push the experts in RAM out to swap
    (NN-99 ④). Windows has no such call: there the file is opened for a sequential scan (`O_SEQUENTIAL`), a hint that
    its cache may let the pages behind go first — `ASSUMED`, not measured on Windows. `tick(bytes done)` as the runs
    come in (▶ `steps`); without one, said a tenth at a time.
    ⭐ INTO THE CACHE ALONE, each reader reads into a buffer of its own it reuses, and the pages stay in the system's cache,
    where a row's read then finds them: none let go of, and the file not opened for a sequential scan — the hint the copy
    gives that the cache may let them go first."""
    say = say or (lambda s: print(s, flush=True))
    cached = into is None
    advise = None if cached else getattr(os, "posix_fadvise", None)
    flags = os.O_RDONLY | BINARY | (0 if cached else getattr(os, "O_SEQUENTIAL", 0))
    where = "into the system's file cache" if cached else "into RAM"
    t0 = time.time()
    lock, done, told, failed = threading.Lock(), [0], [0], []

    def came(n):
        with lock:
            done[0] += n
            if tick is not None:
                tick(done[0])
            elif done[0] * 10 // size > told[0]:
                told[0] = done[0] * 10 // size
                say("  %s %s: %d%% (%.1f of %.1f GB, %.0f s)" % (what, where, told[0] * 10, done[0] / 1e9, size / 1e9,
                                                                 time.time() - t0))

    def reader(k):
        try:
            fd = os.open(path, flags)
            buf = memoryview(bytearray(RUN)) if cached else None
            with open(fd, "rb", buffering=0) as f:
                for at in range(k * RUN, size, READERS * RUN):
                    n, got = min(RUN, size - at), 0
                    f.seek(offset + at)
                    while got < n:
                        more = f.readinto(buf[got:n] if cached else into[at + got:at + n])
                        if not more:
                            raise Refused("%s ends before %s's %d bytes" % (path, what, size))
                        got += more
                    if advise is not None:
                        advise(f.fileno(), offset + at, n, os.POSIX_FADV_DONTNEED)
                    came(n)
                    if failed:
                        return
        except BaseException as ex:                # noqa: BLE001 — handed to the caller's thread below
            failed.append(ex)

    threads = [threading.Thread(target=reader, args=(k,), daemon=True) for k in range(READERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise failed[0]
    took = time.time() - t0
    say("%s %s: %.1f GB of %s in %.0f s (%.2f GB/s)" % (
        what[:1].upper() + what[1:], "read into the system's file cache" if cached else "in RAM", size / 1e9, path, took,
        size / 1e9 / max(took, 1e-9)))
    return took


def warm(path, offset, size, what, tick=None, say=None):
    """⭐ A TABLE INTO THE SYSTEM'S FILE CACHE (`"cache"`): `size` bytes of `path` from `offset` read through once, nothing
    kept but the cache's pages; its rows are read from the file after as `"disk"` reads them. Answers the seconds."""
    return read_through(path, int(offset), int(size), what, tick, say)


class RamTable:
    """`size` bytes of the file at `path` from `offset`, copied into this process's memory: `read(size, offset)` and
    `rows(ids, width)` answer from there in the file's own offsets, `address` is where its first byte lies."""

    def __init__(self, path, offset=0, size=None, what="a table", tick=None, say=None):
        size = os.path.getsize(path) - offset if size is None else int(size)
        self.path, self.offset, self.size = path, int(offset), size
        try:
            self.bytes = np.empty(size, dtype=np.uint8)
        except MemoryError:
            raise Refused("%s (%.1f GB of %s) could not be allocated — on Windows, more than the system can commit: "
                          "'cache' reads it into the system's file cache instead, 'disk' reads it from the disk"
                          % (what, size / 1e9, path))
        self.seconds = read_through(path, self.offset, size, what, tick, say, memoryview(self.bytes))
        self.address = self.bytes.ctypes.data

    def read(self, size, offset):
        """`size` bytes at the file's `offset`, from RAM — what `system.Positional.read` answers from the file."""
        at = offset - self.offset
        if at < 0 or at + size > self.size:
            raise ValueError("bytes %d .. %d are not in the table's %d .. %d" % (offset, offset + size, self.offset,
                                                                                  self.offset + self.size))
        return self.bytes[at:at + size].tobytes()

    def rows(self, ids, width):
        """The rows `ids` names, `width` bytes each, the table's first row at its first byte — `[len(ids), width]`."""
        return self.bytes[:self.size // width * width].reshape(-1, width)[np.asarray(ids, dtype=np.int64)]

    def at(self, offset):
        """The address of the file's byte `offset`, in this process's memory."""
        if not self.offset <= offset < self.offset + self.size:
            raise ValueError("byte %d is not in the table's %d .. %d" % (offset, self.offset, self.offset + self.size))
        return self.address + offset - self.offset
