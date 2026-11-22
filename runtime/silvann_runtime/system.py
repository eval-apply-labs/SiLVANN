"""What the runtime asks of the operating system, answered for the one it runs on — the Python side of the engine's OS
layer (`operating_systems/`, ▶ `docs/design/windows_native.md`): Linux reads `/proc`, Windows asks Win32 through
`ctypes`. The Windows answers serve a model natively: `REASONED` from the architect's Windows run (the server's own
setup, so its boot read the RAM here — measurements/2026-10-28_nn94_windows_5070ti.md)."""
import ctypes
import os
import sys
import threading

WINDOWS = sys.platform == "win32"
LIBRARY_SUFFIX = ".dll" if WINDOWS else ".so"     # a silicon family's file is libsilvann_<family><suffix>
# ⛳ `os.open` OPENS IN TEXT MODE ON WINDOWS: a write turns each 0x0A into 0x0D 0x0A and a read stops at 0x1A, so a
#   descriptor that carries weights asks for binary (Linux has no such mode, and the flag is 0 there)
BINARY = getattr(os, "O_BINARY", 0)
_SEEK = threading.Lock()


def write_at(fd, data, offset):
    """`data` into the file at `offset`: `os.pwrite` where the system has it; on Windows, which has none, a seek and
    writes until it is all there — so the caller is the descriptor's one writer."""
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    view = memoryview(data).cast("B")
    os.lseek(fd, offset, os.SEEK_SET)
    done = 0
    while done < len(view):
        done += os.write(fd, view[done:])
    return done


def read_at(fd, size, offset):
    """`size` bytes of the file at `offset`: `os.pread` where the system has it; on Windows a seek and a read under one
    lock, so threads that share the descriptor do not move each other's position."""
    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)
    with _SEEK:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, size)


class Positional:
    """A file read at offsets by many threads at once: `read(size, offset)`. One descriptor and `os.pread` where the system
    has it; on Windows, which has none, a descriptor a thread, each with its own position, so no thread moves another's
    and none waits on a lock while the reads are in flight together. Opened binary (▶ `BINARY`)."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | BINARY)
        self._own = threading.local()
        self._opened, self._lock = [self.fd], threading.Lock()

    def read(self, size, offset):
        if hasattr(os, "pread"):
            return os.pread(self.fd, size, offset)
        fd = getattr(self._own, "fd", None)
        if fd is None:
            fd = os.open(self.path, os.O_RDONLY | BINARY)
            with self._lock:
                self._opened.append(fd)
            self._own.fd = fd
        os.lseek(fd, offset, os.SEEK_SET)
        got = os.read(fd, size)
        while len(got) < size:                       # a short read short of the file's end: the rest
            more = os.read(fd, size - len(got))
            if not more:
                break
            got += more
        return got

    def close(self):
        with self._lock:
            opened, self._opened = self._opened, []
        for fd in opened:
            try:
                os.close(fd)
            except OSError:
                pass


def resident():
    """This process's resident memory, in bytes — what it holds now; 0 where the system does not say (Windows here)."""
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) << 10 for l in f if l.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        return 0


def _status():
    """Windows' own account of its memory (`GlobalMemoryStatusEx`), or None where it does not answer."""
    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = Status()
    s.dwLength = ctypes.sizeof(Status)
    return s if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) else None


def memory():
    """(total, available) in bytes: the machine's RAM, and what the system says it can give without swapping."""
    if WINDOWS:
        s = _status()
        return (s.ullTotalPhys, s.ullAvailPhys) if s else (0, 0)
    total = available = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    return total, available


def commit():
    """(limit, free) in bytes: what Windows lets its programs commit together — the RAM and the page files as they are
    now — and what of it is not committed yet; None on another system, which has no such limit."""
    s = _status() if WINDOWS else None
    return (s.ullTotalPageFile, s.ullAvailPageFile) if s else None


def paging_files(entries):
    """What Windows' list of page files allows (`PagingFiles` in its registry) — (the bytes they may reach together,
    whether Windows sizes one itself). An entry is a file's path, then its first and its largest size in MB
    (`C:\\pagefile.sys 16384 32768`); a path alone, or one with both sizes 0, is a file Windows sizes itself, growing
    it as the commit nears its limit."""
    most, managed = 0, False
    for entry in entries or ():
        words = entry.split()
        if not words:
            continue
        try:
            first, last = int(words[-2]), int(words[-1])
        except (IndexError, ValueError):
            managed = True
            continue
        if first == 0 and last == 0:
            managed = True
        else:
            most += max(first, last) << 20
    return most, managed


def page_file():
    """Windows' page files as `paging_files` reads them — (0, False) on a PC with none; None on another system, or
    where the registry does not say."""
    if not WINDOWS:
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management") as key:
            entries, _ = winreg.QueryValueEx(key, "PagingFiles")
    except OSError:
        return None
    return paging_files(entries if isinstance(entries, (list, tuple)) else [str(entries)])


def _counter(path):
    """A Windows performance counter, an instance at a time — {instance: value} over the instances `path`'s `*`
    matches, {} where it matches none; None where the system has no such counter. The counter is named in English,
    whatever language the system speaks (`PdhAddEnglishCounter`)."""
    pdh = ctypes.WinDLL("pdh")
    handle, dword = ctypes.c_void_p, ctypes.c_ulong

    class Value(ctypes.Structure):
        _fields_ = [("status", dword), ("large", ctypes.c_longlong)]

    class Item(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar_p), ("value", Value)]

    pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(handle)]
    pdh.PdhAddEnglishCounterW.argtypes = [handle, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(handle)]
    pdh.PdhCollectQueryData.argtypes = [handle]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [handle, dword, ctypes.POINTER(dword), ctypes.POINTER(dword), ctypes.c_void_p]
    pdh.PdhCloseQuery.argtypes = [handle]
    large, more_data, no_data = 0x400, 0x800007D2, 0x800007D5            # PDH_FMT_LARGE, PDH_MORE_DATA, PDH_NO_DATA
    query, counter = handle(), handle()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) & 0xFFFFFFFF:
        return None
    try:
        if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) & 0xFFFFFFFF:
            return None
        said = pdh.PdhCollectQueryData(query) & 0xFFFFFFFF
        if said == no_data:
            return {}
        if said:
            return None
        size, count = dword(0), dword(0)
        said = pdh.PdhGetFormattedCounterArrayW(counter, large, ctypes.byref(size), ctypes.byref(count), None) & 0xFFFFFFFF
        if said != more_data:
            return {} if said in (0, no_data) else None
        room = ctypes.create_string_buffer(size.value)
        if pdh.PdhGetFormattedCounterArrayW(counter, large, ctypes.byref(size), ctypes.byref(count), room) & 0xFFFFFFFF:
            return None
        items = ctypes.cast(room, ctypes.POINTER(Item))
        out = {}
        for i in range(count.value):
            out[items[i].name] = out.get(items[i].name, 0) + int(items[i].value.large)
        return out
    finally:
        pdh.PdhCloseQuery(query)


def adapters_held(process, adapters, pid):
    """{adapter: (dedicated, shared, in use)} from Windows' counters of GPU memory: process `pid`'s own bytes of each
    adapter's memory — `dedicated` the adapter's own, `shared` the PC's RAM it uses as its memory — and the adapter's
    own memory `in use` by every program. `process` is (dedicated, shared) as `GPU Process Memory` says them, an
    instance `pid_<pid>_luid_…`, and `adapters` the dedicated usage as `GPU Adapter Memory` says it, an instance
    `luid_…`."""
    mine = "pid_%d_" % pid
    out = {}
    for which, values in enumerate(process):
        for name, value in values.items():
            if name.startswith(mine):
                held = out.setdefault(name[len(mine):], [0, 0, 0])
                held[which] += value
    for name, held in out.items():
        held[2] = adapters.get(name, 0)
    return {name: tuple(held) for name, held in out.items()}


def cards_held(pid=None):
    """What a process (this one, where `pid` is None) holds of each GPU's memory, as Windows counts it — `adapters_held`
    of its counters, the ones Task Manager's GPU memory columns show; None on another system, or on a Windows without
    them."""
    if not WINDOWS:
        return None
    try:
        process = [_counter("\\GPU Process Memory(*)\\%s Usage" % kind) for kind in ("Dedicated", "Shared")]
        adapters = _counter("\\GPU Adapter Memory(*)\\Dedicated Usage")
    except (OSError, AttributeError):
        return None
    if adapters is None or None in process:
        return None
    return adapters_held(process, adapters, os.getpid() if pid is None else pid)


def card_budgets():
    """{adapter: (its own memory, Windows' budget of it for this process)} in bytes, the adapters named as `cards_held`
    names them — the budget what the system lets this process keep on the card: past it, Windows moves the process's
    memory there to RAM when another program wants the card, and brings it back at each use (DXGI's
    `QueryVideoMemoryInfo`, the local segment). None on another system, or where DXGI does not answer."""
    if not WINDOWS:
        return None
    import uuid
    result, at, u32 = ctypes.c_long, ctypes.c_void_p, ctypes.c_uint

    class Guid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    def guid(text):
        g = Guid()
        ctypes.memmove(ctypes.byref(g), uuid.UUID(text).bytes_le, 16)
        return g

    class Description(ctypes.Structure):                       # DXGI_ADAPTER_DESC1
        _fields_ = [("name", ctypes.c_wchar * 128), ("vendor", u32), ("device", u32), ("subsystem", u32), ("revision", u32),
                    ("own", ctypes.c_size_t), ("own_system", ctypes.c_size_t), ("shared", ctypes.c_size_t),
                    ("luid_low", ctypes.c_ulong), ("luid_high", ctypes.c_long), ("flags", u32)]

    class Memory(ctypes.Structure):                            # DXGI_QUERY_VIDEO_MEMORY_INFO
        _fields_ = [("budget", ctypes.c_ulonglong), ("usage", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("reservation", ctypes.c_ulonglong)]

    def method(thing, index, *takes):                          # a COM object's method, by its place in the interface
        table = ctypes.cast(thing, ctypes.POINTER(ctypes.POINTER(at))).contents
        return ctypes.WINFUNCTYPE(result, at, *takes)(table[index])

    try:
        dxgi = ctypes.WinDLL("dxgi")
        dxgi.CreateDXGIFactory1.argtypes, dxgi.CreateDXGIFactory1.restype = [ctypes.POINTER(Guid), ctypes.POINTER(at)], result
        factory = at()
        if dxgi.CreateDXGIFactory1(ctypes.byref(guid("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(factory)) != 0:
            return None
        out, i = {}, 0
        while True:
            adapter = at()
            if method(factory, 12, u32, ctypes.POINTER(at))(factory, i, ctypes.byref(adapter)) != 0:          # EnumAdapters1
                break
            i += 1
            said, third, memory = Description(), at(), Memory()
            if method(adapter, 10, ctypes.POINTER(Description))(adapter, ctypes.byref(said)) == 0 \
                    and method(adapter, 0, ctypes.POINTER(Guid), ctypes.POINTER(at))(                         # IDXGIAdapter3
                        adapter, ctypes.byref(guid("645967a4-1392-4310-a798-8053ce3e93fd")), ctypes.byref(third)) == 0:
                if method(third, 14, u32, u32, ctypes.POINTER(Memory))(third, 0, 0, ctypes.byref(memory)) == 0:  # local
                    out["luid_0x%08X_0x%08X_phys_0" % (said.luid_high & 0xFFFFFFFF, said.luid_low)] = (int(said.own), int(memory.budget))
                method(third, 2)(third)
            method(adapter, 2)(adapter)
        method(factory, 2)(factory)
        return out
    except (OSError, AttributeError):
        return None


def past_the_card(held, budgets, pools):
    """Why a process holds more than its cards keep, as Windows counts it, or None where it holds no more. `held` is
    `cards_held`'s answer, `budgets` `card_budgets`', `pools` the bytes asked of each card. Windows refuses a card's
    memory only past the card's whole size: short of that it takes the request and keeps what does not fit in RAM,
    brought to the card and back at every use.
      · a card's memory in use past Windows' budget for the process: the system moves it to RAM as others want the card;
      · less of the cards' memory in use than was asked of them: part of it is in RAM already.
    `MEASURED` on a GTX 1080 Ti through NVIDIA's OpenCL (Windows 10 19045, driver 582.28, the budget 10,338 MB of its
    11,106): the 35B with the whole context and a 3.5 GB tier, 10,570 MB asked and all of it on the card, 11.6
    tokens/s; with a 4.0 GB tier, 10,074 MB of the ~11,070 asked on the card, 0.36 tokens/s — and none of it shown as
    shared GPU memory."""
    for adapter, (dedicated, _, _) in sorted((held or {}).items()):
        budget = (budgets or {}).get(adapter, (0, 0))[1]
        if budget and dedicated > budget:
            return ("%.0f MB of a card's memory in use, past the %.0f MB Windows budgets this process there"
                    % (dedicated / 2**20, budget / 2**20))
    on_cards = sorted((dedicated for dedicated, _, _ in (held or {}).values()), reverse=True)
    for k, asked in enumerate(sorted(pools or (), reverse=True)):
        there = on_cards[k] if k < len(on_cards) else 0
        if there < asked:
            return ("%.0f MB asked of a card and %.0f MB of it there: the rest is in the PC's RAM"
                    % (asked / 2**20, there / 2**20))
    return None


# An adapter with this much of its own memory in use is a card: an integrated GPU has next to none of its own, and all
#   it holds is the PC's RAM.
CARD_IN_USE = 512 << 20


def spilled(held):
    """The bytes of a process's GPU memory (`held`, as `cards_held` says it) that are the PC's RAM standing in for a
    card's own: its shared memory on every adapter whose own memory is in use (`CARD_IN_USE`). Windows' drivers give a
    program RAM once a card's own memory is full, and every use of it then crosses to the card — a model placed past
    the card's memory boots, and answers many times slower. What an integrated GPU holds is not counted: its memory is
    the RAM by design."""
    return sum(shared for _, shared, in_use in (held or {}).values() if in_use >= CARD_IN_USE)


SYSMEM_FALLBACK = 0x10ECECC9      # "CUDA - Sysmem Fallback Policy" in NVIDIA's driver profiles


def nvidia_shared_memory():
    """Whether NVIDIA's Windows driver gives a program the PC's RAM once a card's own memory is full — its setting
    "CUDA - Sysmem Fallback Policy" in the driver's global profile: True where it does (the driver's default, or
    "Prefer Sysmem Fallback"), False where it is told not to ("Prefer No Sysmem Fallback"); None where there is no
    NVIDIA driver to ask, it does not answer, or it has no such setting — a driver that takes none of the RAM (`MEASURED`
    Windows 10 19045, a GTX 1080 Ti, driver 582.28: the setting refused as one it does not have, and CUDA and OpenCL
    both refused the first GiB past the card). The driver's own library is asked (NVAPI, by its functions' numbers:
    `nvapi_interface.h`). The setting is not among those the driver names: its number and its values — 0 the default,
    1 no fallback, 2 fallback — are the ones NVIDIA's profile tools use (`SYSMEM_FALLBACK`). `ASSUMED`, not yet read
    on a driver that has it: that such a driver lists the setting's values (`NvAPI_DRS_EnumAvailableSettingValues`)."""
    if not WINDOWS:
        return None
    try:
        query = ctypes.CDLL("nvapi64.dll").nvapi_QueryInterface
    except (OSError, AttributeError):
        return None
    query.restype, query.argtypes = ctypes.c_void_p, [ctypes.c_uint]
    u32, handle = ctypes.c_uint, ctypes.c_void_p

    class Setting(ctypes.Structure):
        _fields_ = [("version", u32), ("name", ctypes.c_uint16 * 2048), ("id", u32), ("type", u32), ("location", u32),
                    ("current_predefined", u32), ("predefined_valid", u32),
                    ("predefined", ctypes.c_uint8 * 4100), ("current", ctypes.c_uint8 * 4100)]

    def door(number, *takes):
        at = query(number)
        return ctypes.CFUNCTYPE(ctypes.c_int, *takes)(at) if at else None

    class Values(ctypes.Structure):
        _fields_ = [("version", u32), ("count", u32), ("type", u32), ("default", ctypes.c_uint8 * 4100),
                    ("values", (ctypes.c_uint8 * 4100) * 100)]

    start, create = door(0x0150E828), door(0x0694D52E, ctypes.POINTER(handle))
    load, end = door(0x375DBD6B, handle), door(0xDAD9CFF8, handle)
    profile_of = door(0x617BFF9F, handle, ctypes.POINTER(handle))
    get = door(0x73BF8338, handle, handle, u32, ctypes.POINTER(Setting))
    listed = door(0x2EC39F90, u32, ctypes.POINTER(u32), ctypes.POINTER(Values))
    if None in (start, create, load, end, profile_of, get, listed) or start() != 0:
        return None
    values, most = Values(), u32(100)
    values.version = ctypes.sizeof(Values) | (1 << 16)
    if listed(SYSMEM_FALLBACK, ctypes.byref(most), ctypes.byref(values)) != 0:
        return None                                        # a driver with no such setting
    session, profile = handle(), handle()
    if create(ctypes.byref(session)) != 0:
        return None
    try:
        if load(session) != 0 or profile_of(session, ctypes.byref(profile)) != 0:
            return None
        setting = Setting()
        setting.version = ctypes.sizeof(Setting) | (1 << 16)
        said = get(session, profile, SYSMEM_FALLBACK, ctypes.byref(setting))
        if said == -160:                                   # NVAPI_SETTING_NOT_FOUND: not set, the driver's default
            return True
        if said != 0:
            return None
        return int.from_bytes(bytes(setting.current[:4]), "little") != 1
    finally:
        end(session)


def _nvidia_card():
    """The largest NVIDIA card's memory in bytes, as `nvidia-smi` says it; 0 where it says none."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout    # no console of its own
        return max([int(row.strip()) << 20 for row in out.splitlines() if row.strip()] or [0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


NVIDIA_SHARED_MEMORY = "https://nvidia.custhelp.com/app/answers/detail/a_id/5490"   # NVIDIA's own page on the setting


def notices(shared=None, files=None, card=None):
    """What a Windows PC's settings cost a model on its card, each a notice for whoever runs it — {"id", "title",
    "text", "link"}, `link` a page that shows the setting or None; [] on another system, and where nothing is amiss.
    ⚖ 2026-11-22 *"we should have a popup that says to stop the shared memory from the nvidia driver (and ideally give
    you a link to that setting) if silvann detects that shared memory is on, and then we need to add the check for the
    pagefile"*. `shared`, `files` and `card` stand in for `nvidia_shared_memory`, `page_file` and the card's bytes."""
    if not WINDOWS and shared is None and files is None:
        return []
    out = []
    shared = nvidia_shared_memory() if shared is None else shared
    if shared:
        out.append({"id": "nvidia_shared_memory", "title": "Turn off the NVIDIA driver's shared memory",
                    "text": "The NVIDIA driver is set to give a program the PC's RAM once the card's own memory is "
                            "full (\"shared GPU memory\"). A model then starts on memory the card does not have: it "
                            "answers many times slower, or does not start at all.\n\n"
                            "Turn it off: NVIDIA Control Panel → Manage 3D settings → Global Settings → "
                            "CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback → Apply. Then start the model "
                            "again.",
                    "link": NVIDIA_SHARED_MEMORY})
    files = page_file() if files is None else files
    if files is not None and not files[1]:
        most = files[0]
        if most and card is None:
            card = _nvidia_card()
        if most == 0 or most < (card or 0):
            has = ("This PC has no page file" if most == 0 else
                   "This PC's page file may reach %.0f GB, and its card has %.0f GB" % (most / 2**30, card / 2**30))
            out.append({"id": "page_file", "title": "Give Windows a page file as big as the card",
                        "text": "Remember that on Windows you need a page file as big as the GPU to fully use your "
                                "RAM. %s. Windows counts what a program puts on the card against the RAM and the page "
                                "file together, so a model can be refused memory the PC still has free.\n\n"
                                "Set it: Settings → System → About → Advanced system settings → Performance: Settings "
                                "→ Advanced → Virtual memory: Change → System managed size, or a custom size at least "
                                "the card's memory. Then restart the PC." % has,
                        "link": None})
    return out


def spans(addresses, size):
    """`(address, bytes)` runs of the blocks of `size` bytes at `addresses` (zeros, none there, left out), neighbours
    joined — what `lock` takes."""
    out = []
    for a in sorted(set(x for x in addresses if x)):
        if out and out[-1][0] + out[-1][1] == a:
            out[-1] = (out[-1][0], out[-1][1] + size)
        else:
            out.append((a, size))
    return out


def lockable():
    """The bytes this process may lock into RAM (`ulimit -l`, RLIMIT_MEMLOCK): None where it may lock any, 0 where the
    system has no lock the runtime asks for (only Linux's `mlock` is)."""
    if not sys.platform.startswith("linux"):
        return 0
    import resource
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return None if soft == resource.RLIM_INFINITY else soft


def lock(runs):
    """⭐ MEMORY THE SYSTEM MAY NOT SWAP OUT (node-01's NN-99 ④: at `vm.swappiness` 60 the kernel swapped GLM's experts
    out to keep files cached, and the sockets read them back as they decoded): each `(address, bytes)` run of this
    process's memory locked into RAM with Linux's `mlock`. Answers the bytes locked; raises OSError with the system's
    reason where a run is refused — the runs before it stay locked until the process ends."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes, libc.mlock.restype = [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int
    done = 0
    for at, n in runs:
        if libc.mlock(at, n) != 0:
            e = ctypes.get_errno()
            raise OSError(e, "mlock of %d bytes at %#x: %s" % (n, at, os.strerror(e)))
        done += n
    return done
