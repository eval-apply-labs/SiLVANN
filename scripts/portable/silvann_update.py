#!/usr/bin/env python3
"""SiLVANN's updater, the portable app's (NN-203): `update\\windows.exe` on Windows, `update/linux.sh` on Linux.

    SiLVANN/update/     windows.exe and linux.sh, and what they run: this file, and for Windows a Python of its own
    SiLVANN/windows/    a system's part each: what an update replaces
    SiLVANN/linux/
    SiLVANN/data/       the models, the conversations, the accounts: never touched here

⚖ 2026-11-22 *"ideally we can have a update folder alongside windows, data and linux. this way we can click
update/windows.exe and it does the update for you, and we can also make it update itself if needed... this is only for
the portable version"* · *"triggering an update would download the new ones as windows_updated and linux_updated, then
the update itself would either move the current folder to their version num if you ask to keep it or delete them if you
dont want them, and then finally rename the _updated folder into the main ones"*.

A program cannot rename the folder it runs from, so the app does not update itself: this does, from a folder of its
own, with SiLVANN closed. Opened by a person, it
  ① asks before it reaches the network, then asks GitHub for the latest release: the one site it contacts;
  ② fetches the release's zip and unpacks each of its folders beside the one in use, as `windows_updated`,
     `linux_updated` and `update_updated`, each made whole under another name first;
  ③ asks *"Remove the current version x.y.z?"*;
  ④ renames `windows` and `linux` to their version (`windows-1.4.2`) and `windows_updated` and `linux_updated` to
     `windows` and `linux`, then removes the folders put aside where the person said so.
`update_updated`, the new version's updater, takes `update`'s place when the app next starts (▶ the UI's
`updates.settle`): on Windows this program runs from `update\\python`, which it cannot rename.

It imports nothing but Python's own library. On Windows `update\\windows.exe` is the app's launcher, the same bytes as
`windows\\SiLVANN.exe` — it starts `python\\pythonw.exe` beside itself — and `update\\python` is Python's embeddable build
with no package (▶ pack_portable.py). On Linux `update/linux.sh` runs it with the Linux part's Python, which renaming
its folder does not stop: everything is imported before a folder is renamed, and Linux's part is renamed last.
"""
import ctypes
import json
import os
import re
import shutil
import ssl
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

LATEST = "https://api.github.com/repos/eval-apply-labs/SiLVANN/releases/latest"
ASSET = "silvann_portable_x64.zip"                    # both systems' parts and the updater: what an update is made from
ROOM = 2 << 30                                        # what an update wants free on the app's drive: the zip and its folders
PARTS = ("windows", "linux")                          # Linux's last: on Linux this program's Python runs from it
MEMBER = re.compile(r"^SiLVANN/(windows|linux|update)/(.+)$")     # a file of a folder an update brings
VERSION = Path("runtime") / "silvann_runtime" / "version.py"
WINDOWS = os.name == "nt"
STARTS = "windows\\SiLVANN.exe" if WINDOWS else "linux/SiLVANN.sh"
AGAIN = "update\\windows.exe" if WINDOWS else "update/linux.sh"


class Stopped(Exception):
    """The person closed the window that showed the work."""


def numbers(version: str) -> tuple:
    """A version as numbers, to set two in order: "1.4.2" and the tag "rel1.4.2" are (1, 4, 2)."""
    return tuple(int(n) for n in re.findall(r"\d+", version))


def newer(a: str, than: str) -> bool:
    return numbers(a) > numbers(than)


def release_of(said: dict):
    """What a release says of itself, from GitHub's answer: its version, its page, its portable zip and the zip's
    size; None where the answer names no version."""
    tag = str(said.get("tag_name") or "")
    if not numbers(tag):
        return None
    asset = next((a for a in said.get("assets") or [] if a.get("name") == ASSET), {})
    return {"version": ".".join(str(n) for n in numbers(tag)), "page": said.get("html_url") or "",
            "zip": asset.get("browser_download_url") or "", "bytes": int(asset.get("size") or 0)}


def version_in(folder: Path):
    """The version a part's folder holds, its runtime's `RELEASE`, or None where it is no part."""
    said = folder / VERSION
    if not said.is_file():
        return None
    m = re.search(r'^RELEASE\s*=\s*"([^"]+)"', said.read_text(encoding="utf-8", errors="replace"), re.M)
    return m.group(1) if m else None


class App:
    """A portable app's folder: `windows`, `linux`, `data` and `update`."""

    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve()

    def version(self):
        """The version in use: this system's part's, the other's where this one is not here."""
        for part in (("windows" if WINDOWS else "linux"),) + PARTS:
            version = version_in(self.folder / part)
            if version:
                return version
        return None

    def whole(self) -> bool:
        return (self.folder / "data").is_dir() and self.version() is not None

    def in_use(self) -> bool:
        """Whether SiLVANN runs from this folder now. Windows refuses to open a running program's file for writing,
        and the launcher runs for as long as the app; on Linux every process says the file it runs."""
        if WINDOWS:
            try:
                with open(self.folder / "windows" / "SiLVANN.exe", "r+b"):
                    return False
            except FileNotFoundError:
                return False
            except OSError:
                return True
        part = str(self.folder / "linux") + os.sep
        for pid in os.listdir("/proc"):
            if pid.isdigit() and int(pid) != os.getpid():
                try:
                    if os.readlink("/proc/%s/exe" % pid).startswith(part):
                        return True
                except OSError:
                    continue
        return False

    def trusted(self) -> ssl.SSLContext:
        """The authorities a download is checked against: the system's, and those the app's own packages carry
        (certifi's), which a Python that came with the app does not find by itself."""
        context = ssl.create_default_context()
        for carried in ("windows/python/Lib/site-packages/certifi/cacert.pem",
                        *sorted(str(p.relative_to(self.folder)) for p in
                                self.folder.glob("linux/python/lib/python3*/site-packages/certifi/cacert.pem"))):
            if (self.folder / carried).is_file():
                context.load_verify_locations(cafile=str(self.folder / carried))
        return context

    def latest(self):
        """GitHub's latest release of SiLVANN (▶ `release_of`), asked now. An error where GitHub does not answer."""
        request = urllib.request.Request(LATEST, headers={"Accept": "application/vnd.github+json", "User-Agent": "SiLVANN"})
        with urllib.request.urlopen(request, timeout=20, context=self.trusted()) as r:
            release = release_of(json.loads(r.read().decode("utf-8")))
        if release is None:
            raise RuntimeError("GitHub's answer names no release")
        return release

    def fetched(self):
        """The version fetched and not yet in place: both parts' `_updated`, whole and of one version; else None."""
        versions = {version_in(self.folder / ("%s_updated" % part)) for part in PARTS}
        return versions.pop() if len(versions) == 1 and None not in versions else None

    def size(self, names) -> int:
        return sum(f.stat().st_size for name in names for f in (self.folder / name).rglob("*") if f.is_file())

    def fetch(self, release: dict, report=lambda text: None) -> list:
        """The release's zip fetched and unpacked beside the folders in use (▶ `unpack`), which stay as they are.
        Answers the folders made; refuses in words where it cannot."""
        if not release.get("zip"):
            raise RuntimeError("the release names no %s" % ASSET)
        if shutil.disk_usage(self.folder).free < ROOM:
            raise RuntimeError("the drive has less than %d GB free: an update wants that much while it unpacks" % (ROOM >> 30))
        got = self.folder / (ASSET + ".part")
        try:
            done = 0
            request = urllib.request.Request(release["zip"], headers={"User-Agent": "SiLVANN"})
            with urllib.request.urlopen(request, timeout=120, context=self.trusted()) as r, open(got, "wb") as f:
                total = int(r.headers.get("Content-Length") or release.get("bytes") or 0)
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    report("downloading SiLVANN %s: %d of %d MB" % (release["version"], done >> 20, total >> 20))
            if release.get("bytes") and done != release["bytes"]:
                raise RuntimeError("the download ended at %d of %d bytes" % (done, release["bytes"]))
            return self.unpack(got, report)
        finally:
            if got.exists():
                got.unlink()

    def unpack(self, archive: Path, report=lambda text: None) -> list:
        """A release's zip unpacked: `windows`, `linux` and `update` each into `<name>_updated`, made whole under
        another name first, so a folder cut short is never put in place. A name that would leave its folder is
        refused; what was fetched before and is not yet in place is replaced. `data` is not unpacked."""
        made = {}
        for name in ("windows", "linux", "update"):               # one cut short before
            shutil.rmtree(self.folder / ("%s_updated.unpacking" % name), ignore_errors=True)
        try:
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if MEMBER.match(i.filename) and not i.is_dir()]
                for n, info in enumerate(members):
                    m = MEMBER.match(info.filename)
                    root = made.setdefault(m.group(1), self.folder / ("%s_updated.unpacking" % m.group(1)))
                    target = (root / m.group(2)).resolve()
                    if root.resolve() not in target.parents:
                        raise RuntimeError("%s would be written outside its folder" % info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    mode = (info.external_attr >> 16) & 0o777      # Linux's part keeps what may be run
                    if mode and not WINDOWS:
                        os.chmod(target, mode)
                    if n % 200 == 0:
                        report("unpacking: %d of %d files" % (n, len(members)))
            if not {"windows", "linux"} <= set(made):
                raise RuntimeError("the release's zip does not hold both systems' parts")
            out = []
            for name, root in sorted(made.items()):
                final = self.folder / ("%s_updated" % name)
                if final.exists():
                    shutil.rmtree(final)
                os.rename(root, final)
                out.append(final.name)
            return out
        finally:
            for root in made.values():                             # a folder cut short
                shutil.rmtree(root, ignore_errors=True)

    def aside(self, part: str) -> str:
        """The name `part`'s folder in use takes when its update is put in place: `<part>-<its version>`, with a
        number after it where a folder of that name is here already."""
        name = base = "%s-%s" % (part, version_in(self.folder / part) or "before")
        n = 1
        while (self.folder / name).exists():
            n += 1
            name = "%s_%d" % (base, n)
        return name

    def swap(self) -> list:
        """The update put in place: each part in use renamed to its version, its `_updated` renamed to it. A rename
        refused — SiLVANN open, on Windows — undoes those made, so the folder is as it was, and is the error. Answers
        the folders put aside."""
        made, aside = [], []
        try:
            for part in PARTS:
                new, old = self.folder / ("%s_updated" % part), self.folder / part
                if not new.is_dir():
                    continue
                if old.exists():
                    put = self.folder / self.aside(part)
                    os.rename(old, put)
                    made.append((old, put))
                    aside.append(put.name)
                os.rename(new, old)
                made.append((new, old))
        except OSError:
            for was, now in reversed(made):
                os.rename(now, was)
            raise
        return aside

    def remove(self, names) -> list:
        """The folders put aside removed, whole: a part under its version and nothing else. Answers those that
        could not be removed."""
        left = []
        for name in names:
            folder = self.folder / name
            if re.match(r"^(windows|linux)-[^/\\]+$", name) and version_in(folder):
                shutil.rmtree(folder, ignore_errors=True)
                if folder.exists():
                    left.append(name)
        return left


class Terminal:
    """The updater's words in a terminal: Linux's, and a program's that drives it through its pipes."""

    def say(self, text: str) -> None:
        print("\n" + text, flush=True)

    wrong = say

    def ask(self, question: str) -> bool:
        print("\n" + question, flush=True)
        try:
            return input("[y/n] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def work(self, title: str, job):
        print("\n" + title, flush=True)
        width = [0]

        def report(text):
            sys.stdout.write("\r" + text.ljust(width[0]))
            sys.stdout.flush()
            width[0] = max(width[0], len(text))
        try:
            return job(report)
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()


class Boxes:
    """The updater's words on Windows: a message box a question, and a small window while it fetches."""
    OK, YES_NO, WRONG, QUESTION, SAID = 0x0, 0x4, 0x10, 0x20, 0x40
    FRONT = 0x10000 | 0x40000                         # MB_SETFOREGROUND, MB_TOPMOST
    YES = 6

    def __init__(self):
        from ctypes import wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
        self.user32.MessageBoxW.restype = ctypes.c_int
        self.procedures = []                          # every window's procedure, held for as long as the updater runs

    def box(self, text: str, flags: int) -> int:
        return self.user32.MessageBoxW(None, text, "SiLVANN update", flags | self.FRONT)

    def say(self, text: str) -> None:
        self.box(text, self.OK | self.SAID)

    def wrong(self, text: str) -> None:
        self.box(text, self.OK | self.WRONG)

    def ask(self, question: str) -> bool:
        return self.box(question, self.YES_NO | self.QUESTION) == self.YES

    def work(self, title: str, job):
        """`job(report)` run beside a window that shows what it reports. The window's X stops the job at its next
        report. Where the window cannot be made the job runs unseen."""
        state = {"text": "", "stop": False, "result": None, "error": None}
        done = threading.Event()

        def report(text):
            if state["stop"]:
                raise Stopped()
            state["text"] = text

        def run():
            try:
                state["result"] = job(report)
            except BaseException as ex:               # noqa: BLE001 — raised again where the window was
                state["error"] = ex
            finally:
                done.set()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.window(title, state, done)
        except Exception:                             # noqa: BLE001 — no window: the work goes on
            pass
        done.wait()
        if state["error"] is not None:
            raise state["error"]
        return state["result"]

    def window(self, title: str, state: dict, done: threading.Event) -> None:
        """A window with a title and a line under it, the line set from `state["text"]` on a timer, closed when `done`
        is set; its X sets `state["stop"]` and it stays until the work has stopped."""
        from ctypes import wintypes
        user32, gdi32, kernel32 = self.user32, ctypes.WinDLL("gdi32"), ctypes.WinDLL("kernel32")
        LRESULT = wintypes.LPARAM
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                        ("hIcon", wintypes.HANDLE), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HANDLE)]

        # every call's types declared: a handle or an LPARAM is 64 bits, and ctypes passes an undeclared int as 32
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
        user32.RegisterClassExW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HANDLE,
                                           wintypes.HINSTANCE, wintypes.LPVOID]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
        user32.LoadCursorW.restype = wintypes.HANDLE
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = LRESULT
        user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UpdateWindow.argtypes = [wintypes.HWND]
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.restype = LRESULT
        gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
        gdi32.CreateFontW.restype = wintypes.HANDLE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE

        WM_DESTROY, WM_CLOSE, WM_SETFONT, WM_TIMER = 0x0002, 0x0010, 0x0030, 0x0113
        WS_CAPTION, WS_SYSMENU, WS_CHILD, WS_VISIBLE = 0x00C00000, 0x00080000, 0x40000000, 0x10000000
        WS_EX_TOPMOST, WS_EX_APPWINDOW = 0x00000008, 0x00040000
        shown = {"note": None, "text": None}

        def procedure(hwnd, msg, wparam, lparam):
            if msg == WM_TIMER:
                if done.is_set():
                    user32.DestroyWindow(hwnd)
                else:
                    text = "Stopping." if state["stop"] else state["text"]
                    if text != shown["text"]:
                        shown["text"] = text
                        user32.SetWindowTextW(shown["note"], text)
                return 0
            if msg == WM_CLOSE:                       # the X: the work stops at its next report, and the window with it
                state["stop"] = True
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        callback = WNDPROC(procedure)
        self.procedures.append(callback)              # Windows calls it by its address: it is never let go
        instance = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = callback
        wc.hInstance = instance
        wc.hCursor = user32.LoadCursorW(None, 32512)  # IDC_ARROW
        wc.hbrBackground = 16                         # COLOR_BTNFACE + 1: the text controls' own background
        wc.lpszClassName = "SiLVANNUpdate%d" % len(self.procedures)    # a class a window: each has its own procedure
        if not user32.RegisterClassExW(ctypes.byref(wc)):
            raise ctypes.WinError(ctypes.get_last_error())
        width, height = 480, 150
        hwnd = user32.CreateWindowExW(WS_EX_TOPMOST | WS_EX_APPWINDOW, wc.lpszClassName, "SiLVANN update", WS_CAPTION | WS_SYSMENU,
                                      (user32.GetSystemMetrics(0) - width) // 2, (user32.GetSystemMetrics(1) - height) // 2,
                                      width, height, None, None, instance, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        head = user32.CreateWindowExW(0, "STATIC", title, WS_CHILD | WS_VISIBLE, 24, 20, 420, 32, hwnd, None, instance, None)
        note = user32.CreateWindowExW(0, "STATIC", "", WS_CHILD | WS_VISIBLE, 24, 60, 430, 40, hwnd, None, instance, None)
        shown["note"] = note
        for control, size in ((head, -22), (note, -14)):
            font = gdi32.CreateFontW(size, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")   # DEFAULT_CHARSET, ClearType
            if font and control:
                user32.SendMessageW(control, WM_SETFONT, font, 1)
        user32.SetTimer(hwnd, 1, 300, None)
        user32.ShowWindow(hwnd, 1)                    # SW_SHOWNORMAL
        user32.UpdateWindow(hwnd)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def update(app: App, words, latest=None) -> int:
    """The update, a question at a time. 0 where the folder ends at the latest release or as the person chose to
    leave it, 1 where something was refused."""
    mine = app.version()
    if not app.whole():
        words.wrong("This is not SiLVANN's portable folder: %s has no data folder, or neither windows nor linux.\n\n"
                    "The updater belongs in the folder silvann_portable_x64.zip unzips to." % app.folder)
        return 1
    if app.in_use():
        words.wrong("SiLVANN is open.\n\nClose it, then open %s again: a version in use cannot be replaced." % AGAIN)
        return 1
    fetched = app.fetched()
    if fetched and newer(fetched, mine):
        if not words.ask("SiLVANN %s was downloaded before and is not in place yet. This folder runs %s.\n\n"
                         "Put %s in place now?" % (fetched, mine, fetched)):
            return 0
    else:
        if not words.ask("This folder runs SiLVANN %s.\n\nAsk GitHub for the latest release? "
                         "Nothing else is contacted, and nothing is sent but the question." % mine):
            return 0
        try:
            release = (latest or app.latest)()
        except (OSError, ValueError, RuntimeError) as ex:
            words.wrong("GitHub was not reached: whether a newer release is out is not known.\n\n(%s)" % ex)
            return 1
        if not newer(release["version"], mine):
            words.say("SiLVANN %s is the latest release. Nothing to do." % mine)
            return 0
        if not words.ask("SiLVANN %s is out (%d MB to download).\n\nDownload it and put it in this version's place? "
                         "The models, the conversations and the accounts are in data, which is not touched."
                         % (release["version"], release["bytes"] >> 20)):
            return 0
        try:
            words.work("Fetching SiLVANN %s" % release["version"], lambda report: app.fetch(release, report))
        except Stopped:
            words.say("Stopped. Nothing was changed.")
            return 0
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as ex:
            words.wrong("SiLVANN %s was not downloaded: %s\n\nNothing was changed." % (release["version"], ex))
            return 1
        fetched = app.fetched() or release["version"]
    current = [part for part in PARTS if (app.folder / part).is_dir()]
    kept = " and ".join(app.aside(part) for part in current)
    remove = words.ask("Remove the current version %s?\n\nYes: its folders are deleted (%d MB).\n"
                       "No: they stay beside the new ones, as %s." % (mine, app.size(current) >> 20, kept))
    put = {"aside": [], "left": []}

    def place(report):
        """The renames, then the removal, which takes minutes where the disk is slow: said while it lasts."""
        put["aside"] = app.swap()
        if remove:
            report("removing version %s: %s" % (mine, " and ".join(put["aside"])))
            put["left"] = app.remove(put["aside"])
    try:
        words.work("Putting SiLVANN %s in place" % fetched, place)
    except Stopped:                                   # the window closed before the removal began: the version stays
        remove = False
    except OSError as ex:
        words.wrong("SiLVANN %s could not be put in place: %s\n\nNothing was changed and the download is kept. "
                    "Close SiLVANN if it is open, then open %s again." % (fetched, ex, AGAIN))
        return 1
    aside, left = put["aside"], put["left"]
    if not remove:
        after = "Version %s stays in %s: delete those folders when you no longer want it." % (mine, " and ".join(aside))
    elif left:
        after = "Version %s could not be removed whole: delete %s yourself." % (mine, " and ".join(left))
    else:
        after = "Version %s is removed." % mine
    words.say("SiLVANN %s is in place. Open %s.\n\n%s" % (fetched, STARTS, after))
    return 0


def main(argv) -> int:
    words = Boxes() if WINDOWS else Terminal()
    try:
        code = update(App(Path(__file__).resolve().parents[3]), words)
    except KeyboardInterrupt:
        words.say("Stopped.")
        code = 1
    except Exception as ex:                           # noqa: BLE001 — said to the person: under pythonw nothing else is
        words.wrong("The updater stopped on an error it did not expect: %s: %s" % (type(ex).__name__, ex))
        code = 1
    if "--wait" in argv:                              # a terminal opened for this alone closes with it
        try:
            input("\nPress Enter to close.")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
