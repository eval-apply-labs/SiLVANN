"""THE PORTABLE APP (NN-162) — what SiLVANN.exe starts: the UI, in a window of its own, and nothing left when it closes.

⚖ 2026-11-07 *"ideally i would also like a silvann portable app on windows, so instead of calling it via command line you
open it up as an ui by just doing a double click and when you close it it's over"*. SiLVANN.exe runs this with the folder's
own Python (`python\\pythonw.exe`, no console) inside a job object that ends every process it holds when the launcher goes
(▶ SiLVANN.c), from `windows\\` — Linux's part is `linux\\` beside it, and `data\\` theirs together. Here:

  1. the data — the models, the conversations, the logs — in `data\\` beside SiLVANN.exe, or in Documents\\SiLVANN where
     the folder cannot be written to (a read-only place, Program Files), handed on as `SILVANN_DATA`
  2. the UI (`silvann_ui.py --local`: the app over this machine's models, the model started last started again, its
     servers stopped with it) on a free port of 127.0.0.1, its output in `data\\logs\\ui.log`
  3. Microsoft Edge in app mode — a window with no browser around it — on its own profile in `data\\.edge`, so it is its
     own process and not a tab of an Edge already open. Edge comes with Windows 10 and 11; where it is not found, the
     default browser opens the UI and a message box stands for the window: its OK ends the app.
  4. the app's end, when that window closes: the UI asked to stop as `docker stop` asks it — the conversations saved,
     the model's server stopped — through a route only this app's token opens (Windows has no signal to ask with: a
     terminate there is TerminateProcess, which runs nothing), while a small window says "Shutting down" with its dots
     cycling. ⚖ 2026-11-10 *"the close does say shutting down with the dots cycling and pressing x again would be a hard
     close"*: that window's X ends the UI at once, and the launcher's job ends whatever is left.

ON LINUX (NN-194, ⚖ 2026-11-16 *"i'd like to have a single portable app, and when unzipped it should have windows, linux
and data folders ... one usb key and being able to move it across systems"*): SiLVANN.sh, beside SiLVANN.exe at the top
of the folder, runs this with `linux/python/bin/python3`, a standalone Python, from `linux/` — the release's tree, its
evaluator and families in `linux/lib/` — and the data is the same `data/` Windows' part uses. The same four steps, with
Linux's own means:

  1. Documents/SiLVANN in the home folder where the folder cannot be written to (or ~/SiLVANN where there is no Documents)
  2. the UI in a process group of its own, which its model's servers join; it is sent SIGTERM — the `docker stop` it
     already answers — should this launcher die first (PR_SET_PDEATHSIG), and a hard stop ends the whole group
  3. the window: the Chromium family's app mode (Chrome, Chromium, Edge, Brave) on its own profile in `data/.browser`.
     A browser that hands the window to one already running and returns at once (a snap's, a flatpak's), or none of
     them: the default browser opens the UI and a message box — zenity's, kdialog's or Tk's — stands for the window
  4. the end: the same quit route, a "Shutting down" window drawn by Tk; with no display at all (a terminal over ssh),
     the app runs until Ctrl-C or SIGTERM, and stops the same way
"""
import ctypes
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

WINDOWS = os.name == "nt"
ROOT = Path(__file__).resolve().parents[2]          # scripts\portable\ → this system's part: windows\ or linux/
APP = ROOT.parent                                   # the folder SiLVANN.exe and SiLVANN.sh are in, data\ beside them
CREATE_NO_WINDOW = 0x08000000 if WINDOWS else 0
STOP_SECONDS = 660                                  # the UI's stop without a window to cut it short: the server's save
                                                    # (its 600 s) and its end
HANDED_OVER = 5                                     # seconds: a browser back sooner gave its window to one already running
# Linux's app-mode browsers, in the order looked for: the Chromium family's `--app` and `--user-data-dir`, as Edge's
CHROMIUMS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
             "microsoft-edge-stable", "brave-browser")
TITLE = "SiLVANN is shutting down"
NOTE = "Saving the conversations and stopping the model.\nClose this window to stop at once, without saving."


def say(text, title="SiLVANN", flags=0):
    """A message box (MB_OK, or `flags` on Windows), waited on: the app has no console to write to. On Linux zenity's or
    kdialog's where one is installed, else Tk's; with no display, the line on stderr and no wait (False)."""
    if WINDOWS:
        return ctypes.windll.user32.MessageBoxW(None, text, title, flags)
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        for line in (["zenity", "--info", "--no-wrap", "--title", title, "--text", text],
                     ["kdialog", "--title", title, "--msgbox", text]):
            if shutil.which(line[0]):
                subprocess.call(line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        try:
            import tkinter
            from tkinter import messagebox
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title, text, parent=root)
            root.destroy()
            return True
        except Exception:                             # no Tk, or no display it can open
            pass
    print("%s: %s" % (title, text), file=sys.stderr, flush=True)
    return False


def data_folder():
    """`data\\` beside SiLVANN.exe and SiLVANN.sh, both systems' — where it can be written to, else Documents\\SiLVANN
    (on Linux the home folder's Documents/SiLVANN, or ~/SiLVANN where it has no Documents)."""
    here = APP / "data"
    try:
        here.mkdir(exist_ok=True)
        probe = here / ".writable"
        probe.write_text("ok")
        probe.unlink()
        return here
    except OSError:
        home = Path(os.environ.get("USERPROFILE", str(Path.home()))) if WINDOWS else Path.home()
        docs = (home / "Documents" / "SiLVANN") if WINDOWS or (home / "Documents").is_dir() else (home / "SiLVANN")
        docs.mkdir(parents=True, exist_ok=True)
        return docs


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def answers(url, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def edge():
    """Microsoft Edge's msedge.exe: where Windows installs it, or where the registry's App Paths say."""
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            p = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if p.exists():
                return p
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe") as k:
                    p = Path(winreg.QueryValue(k, None))
                    if p.exists():
                        return p
            except OSError:
                continue
    except ImportError:
        pass
    return None


def app_browser():
    """The browser that opens the UI as a window of its own: Edge on Windows; on Linux the first of CHROMIUMS on the
    PATH, where a display is."""
    if WINDOWS:
        return edge()
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    for name in CHROMIUMS:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _die_with_launcher():
    """Linux, in the UI's process before it runs: SIGTERM to it when this launcher's thread ends (PR_SET_PDEATHSIG), so
    a launcher killed or crashed stops the UI as `docker stop` does — what the job object does on Windows."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:
        pass


def _shutting_down_tk(stopped):
    """Linux's "shutting down" window, by Tk: the same words, the dots cycling, its X a stop at once (False)."""
    import tkinter
    root = tkinter.Tk()
    root.title("SiLVANN")
    root.resizable(False, False)
    title = tkinter.Label(root, text=TITLE, font=("TkDefaultFont", 15), anchor="w", width=28)
    title.pack(padx=24, pady=(20, 6), anchor="w")
    tkinter.Label(root, text=NOTE, justify="left", anchor="w").pack(padx=24, pady=(0, 20), anchor="w")
    root.update_idletasks()
    root.geometry("+%d+%d" % ((root.winfo_screenwidth() - root.winfo_width()) // 2,
                              (root.winfo_screenheight() - root.winfo_height()) // 2))
    root.attributes("-topmost", True)
    window = {"dots": 0, "x": False}

    def tick():
        if stopped.is_set():
            root.destroy()
            return
        window["dots"] = (window["dots"] + 1) % 4
        title.config(text=TITLE + "." * window["dots"])
        root.after(400, tick)

    def close():                                      # the X: stop at once
        window["x"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(400, tick)
    root.mainloop()
    return not window["x"]


def shutting_down(stopped):
    """A small window while the app stops — "SiLVANN is shutting down", its dots cycling — closed when `stopped` is set.
    Its X (or Alt+F4) closes it first: False then, True when the stop finished."""
    if stopped.wait(0.3):                             # nothing to wait for: no window
        return True
    if not WINDOWS:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("no display")         # waited on unseen (▶ stop)
        return _shutting_down_tk(stopped)
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32")
    kernel32 = ctypes.WinDLL("kernel32")
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
    TITLE = "SiLVANN is shutting down"
    window = {"title": None, "dots": 0, "x": False}

    def procedure(hwnd, msg, wparam, lparam):
        if msg == WM_TIMER:
            if stopped.is_set():
                user32.DestroyWindow(hwnd)
            else:
                window["dots"] = (window["dots"] + 1) % 4
                user32.SetWindowTextW(window["title"], TITLE + "." * window["dots"])
            return 0
        if msg == WM_CLOSE:                           # the X: stop at once
            window["x"] = True
            user32.DestroyWindow(hwnd)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    callback = WNDPROC(procedure)                     # held here while the window lives
    instance = kernel32.GetModuleHandleW(None)
    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = callback
    wc.hInstance = instance
    wc.hCursor = user32.LoadCursorW(None, 32512)      # IDC_ARROW
    wc.hbrBackground = 16                             # COLOR_BTNFACE + 1: the text controls' own background
    wc.lpszClassName = "SiLVANNShuttingDown"
    if not user32.RegisterClassExW(ctypes.byref(wc)):
        raise ctypes.WinError(ctypes.get_last_error())
    width, height = 440, 150
    hwnd = user32.CreateWindowExW(WS_EX_TOPMOST | WS_EX_APPWINDOW, wc.lpszClassName, "SiLVANN", WS_CAPTION | WS_SYSMENU,
                                  (user32.GetSystemMetrics(0) - width) // 2, (user32.GetSystemMetrics(1) - height) // 2,
                                  width, height, None, None, instance, None)
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    title = user32.CreateWindowExW(0, "STATIC", TITLE, WS_CHILD | WS_VISIBLE, 24, 20, 380, 32, hwnd, None, instance, None)
    note = user32.CreateWindowExW(0, "STATIC", "Saving the conversations and stopping the model.\n"
                                  "Close this window to stop at once, without saving.",
                                  WS_CHILD | WS_VISIBLE, 24, 60, 390, 40, hwnd, None, instance, None)
    window["title"] = title
    for control, size in ((title, -22), (note, -14)):
        font = gdi32.CreateFontW(size, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")   # DEFAULT_CHARSET, ClearType
        if font and control:
            user32.SendMessageW(control, WM_SETFONT, font, 1)
    user32.SetTimer(hwnd, 1, 400, None)
    user32.ShowWindow(hwnd, 1)                        # SW_SHOWNORMAL
    user32.UpdateWindow(hwnd)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    return not window["x"]


def stop(ui, url, token):
    """The app's end: the UI asked to stop through its quit route — it saves the conversations, stops the model's server
    and exits — while `shutting_down` shows it; that window's X, a UI that does not answer, or no window and
    STOP_SECONDS gone, and the UI is ended at once (the launcher's job then ends what it started; on Linux, the UI's
    process group here)."""
    if ui.poll() is not None:
        return
    stopped = threading.Event()

    def ask():
        try:
            request = urllib.request.Request(url + "/_silvann/quit", data=b"", method="POST",
                                             headers={"X-SiLVANN-Quit": token})
            urllib.request.urlopen(request, timeout=10).close()
            ui.wait()
        except Exception:                             # the UI not answering: nothing to stop through it
            pass
        finally:
            stopped.set()

    threading.Thread(target=ask, daemon=True).start()
    try:
        shutting_down(stopped)
    except Exception:                                 # no window (not Windows, or its calls refused): wait unseen
        stopped.wait(STOP_SECONDS)
    if ui.poll() is None:
        ui.kill()
        try:
            ui.wait(5)
        except subprocess.TimeoutExpired:
            pass
    if not WINDOWS:                                   # what the UI started and left — the job object's end, by its group
        try:
            os.killpg(ui.pid, signal.SIGKILL)
        except OSError:                               # the group gone with the UI: nothing left
            pass


def main():
    if not WINDOWS:                                   # a session's end or a `kill` ends the app as its window does
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    data = data_folder()
    (data / "logs").mkdir(exist_ok=True)
    token = secrets.token_hex(16)                     # opens the UI's quit route: this app's alone
    env = dict(os.environ, SILVANN_DATA=str(data), PYTHONUTF8="1", SILVANN_QUIT_TOKEN=token)
    port = free_port()
    url = "http://127.0.0.1:%d" % port
    log = open(data / "logs" / "ui.log", "a", encoding="utf-8")
    own = (dict(creationflags=CREATE_NO_WINDOW) if WINDOWS else
           dict(start_new_session=True, preexec_fn=_die_with_launcher))
    ui = subprocess.Popen([sys.executable, str(ROOT / "silvann_ui.py"), "--port", str(port), "--host", "127.0.0.1",
                           "--no-browser", "--local"], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, **own)
    window = None
    try:
        if not answers(url, 120):
            say("SiLVANN's UI did not start. What it said is in %s." % (data / "logs" / "ui.log"))
            return 1
        browser = app_browser()
        handed_over = False
        if browser is not None:
            # Edge's profile in data\.edge; a Chromium's on Linux in the home folder's cache, which a Linux filesystem
            #   holds — its lock is a symbolic link, which a drive formatted for both systems (exFAT) cannot make
            profile = (data / ".edge") if WINDOWS else (
                Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "silvann_portable" / "browser")
            started = time.time()
            window = subprocess.Popen([str(browser), "--app=" + url, "--user-data-dir=" + str(profile),
                                       "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"],
                                      **(dict(creationflags=CREATE_NO_WINDOW) if WINDOWS else
                                         dict(stdout=log, stderr=subprocess.STDOUT)))
            code = window.wait()                      # the window closed: the app is over
            if WINDOWS or time.time() - started >= HANDED_OVER:
                return 0
            handed_over = code == 0                   # back at once: its window given to a browser already running
        if WINDOWS:
            os.startfile(url)
        elif not handed_over and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            webbrowser.open(url)                      # with no display it could start a text browser in the terminal
        if not say("SiLVANN is open in your browser at %s.\n\nPress OK to close SiLVANN." % url) and not WINDOWS:
            print("SiLVANN runs at %s until Ctrl-C." % url, file=sys.stderr, flush=True)
            signal.pause()                            # no display: Ctrl-C or SIGTERM
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if window is not None and window.poll() is None:
            window.terminate()
        stop(ui, url, token)
        log.close()                                   # and the launcher's job ends whatever is left


if __name__ == "__main__":
    sys.exit(main())
