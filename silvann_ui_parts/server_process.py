"""The server this UI starts and stops on its own machine, and its log."""
import json
import subprocess
import sys
import threading
from collections import deque
from typing import Optional

import httpx

from .local import LAST_START, REPO_ROOT, SERVER_SCRIPT


PROGRESS = "[silvann_server] loading "       # the server's progress lines, each the whole of it so far each time it grows


class ServerProcess:
    """Single subprocess + a shared line buffer that log panes poll."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: Optional[list] = None
        self.lines: deque[str] = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.rewrites = 0                      # lines replaced in place (a load's progress): a pane repaints on a change
        self.setup: Optional[str] = None       # what the server said it chose for this machine (its "the setup:" line)
        self.failure: Optional[str] = None     # why the last server it started ended on its own, if it did
        self.started = False                   # whether it had booted when it ended (else it never started)
        self.stopping = False                  # `stop` asked it to end: not a failure
        self.remember: Optional[dict] = None   # the start written to LAST_START once the server has booted
        self.args: Optional[tuple] = None      # the last start's model folder, config and port

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status_text(self) -> str:
        if not self.is_running():
            return "stopped"
        return f"running (pid {self.proc.pid}, cmd: {' '.join(self.cmd or [])})"

    def start(self, model_dir: str, config_name: str, port: int, remember: Optional[dict] = None):
        if self.is_running():
            raise RuntimeError("already running — stop first")
        if not SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"server script not found at {SERVER_SCRIPT}")
        self.lines.clear()
        self.setup = self.failure = None
        self.started = self.stopping = False
        self.remember = remember
        self.args = (str(model_dir), config_name, int(port))   # what a switch starts again (▶ start_again_later)
        # The packed-models launch contract: --model <folder> + a config that overrides default_config.
        self.cmd = [sys.executable, str(SERVER_SCRIPT), "--model", str(model_dir),
                    config_name, "--port", str(int(port))]
        self.port = int(port)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, cwd=str(REPO_ROOT),
        )
        threading.Thread(target=self._drain, daemon=True).start()
        self._log(f"[ui] launched: {' '.join(self.cmd)}")

    def stop(self):
        if not self.is_running():
            return
        self.stopping = True
        # Persist conversations BEFORE killing. On Windows proc.terminate() is a hard kill
        # (TerminateProcess), so the server's graceful-shutdown save never runs — ask it to save
        # every bound conversation's KV first (blocking; big windows take a while). Best-effort:
        # if it fails we still stop.
        port = getattr(self, "port", None)
        if port:
            try:
                self._log("[ui] saving conversations before stop…")
                r = httpx.post(f"http://localhost:{port}/admin/save-all", timeout=600.0)
                self._log(f"[ui] saved: {r.json().get('saved', [])}")
            except Exception as e:
                self._log(f"[ui] save-all failed (stopping anyway): {e}")
        self._log(f"[ui] terminating pid {self.proc.pid}")
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._log("[ui] still alive after 15s — sending SIGKILL")
                self.proc.kill()
                self.proc.wait(timeout=5)
        finally:
            self._log("[ui] stopped.")
            self.proc = None
            self.cmd = None

    def _drain(self):
        proc, said = self.proc, None
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                line = line.rstrip()
                self._log(line, progress=PROGRESS in line)
                if "[silvann_server] the setup: " in line:
                    self.setup = line.split("[silvann_server] ", 1)[1]
                if "⛔ not started:" in line:
                    said = line.split("⛔ not started: ", 1)[1]
                if "⛔ stopped:" in line:            # the evaluator stopped, and the server with it: start it again here
                    said = line.split("⛔ stopped: ", 1)[1]
                if "[silvann_server]" in line and " booted in " in line:
                    self.started = True
                    if self.remember:             # a start that never booted is not the one to start again
                        try:
                            LAST_START.write_text(json.dumps(self.remember))
                        except OSError as ex:
                            self._log(f"[ui] the last start not kept: {ex}")
        finally:
            self._log("[ui] subprocess output stream closed.")
        # a server that ended on its own — not one `stop` ended (it clears `proc` first): why, in a line
        rc = proc.wait()
        if not self.stopping and rc != 0:
            last = next((l for l in reversed(self.snapshot()) if l.strip() and not l.startswith("[ui]")), "")
            self.failure = said or ("killed (exit %d) — the machine may have run out of memory" % rc if rc < 0
                                    else "ended (exit %d): %s" % (rc, last[-300:]))
            self.setup = None

    def _log(self, line: str, progress: bool = False):
        """A line of the server's output kept for the panes — a load's progress line in place of the one before it of
        the same thing loading, so the pane shows a line growing for each (`loading weights: ...10...20..`)."""
        with self.lock:
            if progress and self.lines and PROGRESS in self.lines[-1] and \
                    self.lines[-1].partition(":")[0] == line.partition(":")[0]:
                self.lines[-1] = line
                self.rewrites += 1
            else:
                self.lines.append(line)
        # the UI's own output too, so a container's log (`docker logs`) carries the server's lines, its setup among them
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass

    def snapshot(self) -> list:
        with self.lock:
            return list(self.lines)


SERVER = ServerProcess()
