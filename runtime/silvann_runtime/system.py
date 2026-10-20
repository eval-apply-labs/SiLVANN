"""What the runtime asks of the operating system, answered for the one it runs on — the Python side of the engine's OS
layer (`operating_systems/`, ▶ `docs/design/pending/windows_native.md`): Linux reads `/proc`, Windows asks Win32 through
`ctypes`. The Windows answers are `ASSUMED` until a Windows PC has run them (NN-94)."""
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


def memory():
    """(total, available) in bytes: the machine's RAM, and what the system says it can give without swapping."""
    if WINDOWS:
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return 0, 0
        return s.ullTotalPhys, s.ullAvailPhys
    total = available = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    return total, available
