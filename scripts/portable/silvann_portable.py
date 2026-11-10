"""THE PORTABLE APP (NN-162) — what SiLVANN.exe starts: the UI, in a window of its own, and nothing left when it closes.

⚖ 2026-11-07 *"ideally i would also like a silvann portable app on windows, so instead of calling it via command line you
open it up as an ui by just doing a double click and when you close it it's over"*. SiLVANN.exe runs this with the folder's
own Python (`python\\pythonw.exe`, no console) inside a job object that ends every process it holds when the launcher goes
(▶ SiLVANN.c). Here:

  1. the data — the models, the conversations, the logs — in `data\\` beside SiLVANN.exe, or in Documents\\SiLVANN where
     the folder cannot be written to (a read-only place, Program Files), handed on as `SILVANN_DATA`
  2. the UI (`silvann_ui.py --local`: the app over this machine's models, the model started last started again, its
     servers stopped with it) on a free port of 127.0.0.1, its output in `data\\logs\\ui.log`
  3. Microsoft Edge in app mode — a window with no browser around it — on its own profile in `data\\.edge`, so it is its
     own process and not a tab of an Edge already open; closing that window ends this script, which ends the UI, and the
     launcher's job ends whatever is left. Edge comes with Windows 10 and 11; where it is not found, the default browser
     opens the UI and a message box stands for the window: its OK ends the app.
"""
import ctypes
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # scripts\portable\ → the folder SiLVANN.exe is in
CREATE_NO_WINDOW = 0x08000000


def say(text, title="SiLVANN", flags=0):
    """A message box (MB_OK, or `flags`): the app has no console to write to."""
    return ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def data_folder():
    """`data\\` beside SiLVANN.exe where it can be written to, else Documents\\SiLVANN."""
    here = ROOT / "data"
    try:
        here.mkdir(exist_ok=True)
        probe = here / ".writable"
        probe.write_text("ok")
        probe.unlink()
        return here
    except OSError:
        docs = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents" / "SiLVANN"
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


def main():
    data = data_folder()
    (data / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, SILVANN_DATA=str(data), PYTHONUTF8="1")
    port = free_port()
    url = "http://127.0.0.1:%d" % port
    log = open(data / "logs" / "ui.log", "a", encoding="utf-8")
    ui = subprocess.Popen([sys.executable, str(ROOT / "silvann_ui.py"), "--port", str(port), "--host", "127.0.0.1",
                           "--no-browser", "--local"], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                          creationflags=CREATE_NO_WINDOW)
    try:
        if not answers(url, 120):
            say("SiLVANN's UI did not start. What it said is in %s." % (data / "logs" / "ui.log"))
            return 1
        browser = edge()
        if browser is not None:
            window = subprocess.Popen([str(browser), "--app=" + url, "--user-data-dir=" + str(data / ".edge"),
                                       "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"],
                                      creationflags=CREATE_NO_WINDOW)
            window.wait()                             # the window closed: the app is over
        else:
            os.startfile(url)
            say("SiLVANN is open in your browser at %s.\n\nPress OK to close SiLVANN." % url)
        return 0
    finally:
        ui.terminate()                                # the UI's --local stops the servers it started
        try:
            ui.wait(15)
        except subprocess.TimeoutExpired:
            ui.kill()
        log.close()                                   # and the launcher's job ends whatever is left


if __name__ == "__main__":
    sys.exit(main())
