"""The models fetched from Hugging Face, one at a time, and their updates."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from .local import CORE_SCRIPT, MODELS_DIR, REPO_ROOT, TOKEN_FILE, model_present


# ---------- Models fetched from Hugging Face (process-wide singleton) ----------


def folder_bytes(path: Path) -> int:
    """The bytes under a folder, the files a download is still writing included (`.cache/huggingface/download/`)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def human_bytes(b: int) -> str:
    """Bytes as a person reads them: GB, MB or KB."""
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"


def approx_bytes(size: str) -> int:
    """`"20 GB"`, the size the release lists for a model, in bytes; 0 when it does not read as one."""
    m = re.match(r"\s*([\d.]+)\s*([KMGT])B", size or "")
    return int(float(m.group(1)) * 1000 ** "KMGT".index(m.group(2)) * 1000) if m else 0


def saved_token() -> Optional[str]:
    """The Hugging Face token the downloads use: HF_TOKEN in the environment, else the one saved in TOKEN_FILE."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


def save_token(token: str):
    """`token` into TOKEN_FILE, readable by its owner alone (as silvann_core.py's hf-token writes it); "" removes it."""
    if not token:
        TOKEN_FILE.unlink(missing_ok=True)
        return
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token.strip() + "\n")
    os.chmod(TOKEN_FILE, 0o600)


def token_owner(token: str) -> Optional[str]:
    """The account a token signs in as; ValueError when Hugging Face refuses it, None when it could not be asked."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    try:
        return HfApi().whoami(token=token).get("name") or "?"
    except HfHubHTTPError as ex:
        if ex.response is not None and ex.response.status_code in (401, 403):
            raise ValueError(f"Hugging Face refused the token ({ex.response.status_code})")
        return None
    except Exception:
        return None


def repo_bytes(repo: str) -> int:
    """The bytes of every file in a Hugging Face repository; 0 when it cannot be asked."""
    try:
        from huggingface_hub import HfApi
        return sum(s.size or 0 for s in HfApi().model_info(repo, files_metadata=True, token=saved_token()).siblings)
    except Exception:
        return 0


class ModelDownloads:
    """The models this release can fetch, and one at a time fetched into MODELS_DIR by `silvann_core.py download-model`
    in a subprocess, which also writes the folder's `configs/default.json`. Progress is the bytes the folder held when
    the download started and those received since (the subprocess counts them into a file, `SILVANN_DOWNLOAD_PROGRESS`),
    against the repository's: Hugging Face's client writes a file in blocks of a few hundred MB, so the bytes on the
    disk alone move in steps. A download stopped and started again keeps the files it finished. One is refused when
    the disk under MODELS_DIR has less room than the model still needs, and 1 GB.

    A model that is here says **update** or **done** by its pack's version against its own (`silvann_core.py
    check-versions`, ▶ `fetch.state`), read when the panel opens and after each download. An update is the same
    `download-model`, which for a folder that is there fetches the files that differ alone (▶ `fetch.differences`), in
    two runs of the bar: **checking** while it reads the files it has no record of (a folder copied here) against the
    pack's hashes, the bytes read against theirs, then **downloading**, the bytes received against those it fetches."""

    def __init__(self):
        self.known: Optional[list] = None   # [{name, repo, size, base}], read once from silvann_core.py
        self.proc: Optional[subprocess.Popen] = None
        self.name: Optional[str] = None
        self.total = 0
        self.base = 0                        # the bytes the folder held when the download started
        self.report = os.path.join(tempfile.gettempdir(), "silvann_download_%d.json" % os.getpid())
        self.message = ""
        self.lines: deque[str] = deque(maxlen=200)
        self.updating = False                # the one running is an update of a model that is here
        self.states: Optional[dict] = None   # name → `fetch.state` of each model here; None until read (again)
        self.reading = False

    def catalogue(self) -> list:
        if self.known is None:
            r = subprocess.run([sys.executable, str(CORE_SCRIPT), "list-models", "--json"], capture_output=True,
                               text=True, cwd=str(REPO_ROOT), timeout=300)
            if r.returncode != 0:
                raise RuntimeError("silvann_core.py list-models failed: " + (r.stderr or r.stdout)[-500:])
            self.known = [{k: m.get(k) for k in ("name", "repo", "size", "base")}
                          for m in json.loads(r.stdout.strip().splitlines()[-1])]
        return [dict(m, here=model_present(m["name"])) for m in self.known]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _core(self, *args) -> dict:
        r = subprocess.run([sys.executable, str(CORE_SCRIPT), *args, "--json"], capture_output=True, text=True,
                           cwd=str(REPO_ROOT), env=dict(os.environ))
        if r.returncode != 0:
            raise RuntimeError(((r.stderr or r.stdout).strip().splitlines() or ["exit %d" % r.returncode])[-1])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def read_states(self):
        """Each model here against its pack's version (`fetch.state`), into `states`; one the Hub could not be asked
        about says so."""
        try:
            self.states = self._core("check-versions")
        except Exception as ex:
            self.states = {m["name"]: {"state": "unknown", "why": "not checked: %s" % str(ex)[:200]}
                           for m in self.catalogue() if m["here"]}

    def start(self, name: str):
        if self.running():
            raise RuntimeError(f"{self.name} is downloading — one at a time")
        model = next((m for m in self.catalogue() if m["name"] == name), None)
        if model is None:
            raise RuntimeError(f"{name} is not a model this release knows")
        updating = model_present(name)
        if updating:                         # what differs, and the room for it, the subprocess finds and says
            total, base = 0, 0
        else:
            total = repo_bytes(model["repo"]) or approx_bytes(model["size"])
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            base = folder_bytes(MODELS_DIR / name)
            need = max(0, total - base)
            free = shutil.disk_usage(MODELS_DIR).free
            if need + (1 << 30) > free:
                raise RuntimeError(f"{name} needs {need / 1e9:.1f} GB more, and {MODELS_DIR} has {free / 1e9:.1f} GB free")
        self.name, self.total, self.base, self.message, self.updating = name, total, base, "", updating
        self.lines.clear()
        try:
            os.remove(self.report)
        except OSError:
            pass
        self.proc = subprocess.Popen([sys.executable, str(CORE_SCRIPT), "download-model", name],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     cwd=str(REPO_ROOT), env=dict(os.environ, SILVANN_DOWNLOAD_PROGRESS=self.report))
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        proc = self.proc
        for line in iter(proc.stdout.readline, ""):     # text mode turns a progress bar's \r into lines too
            if line.strip():
                self.lines.append(line.rstrip())
        rc = proc.wait()
        self.message = ("done" if rc == 0 else
                        "stopped" if rc < 0 else "failed: " + (self.lines[-1] if self.lines else f"exit {rc}"))
        self.states = None                             # what is here now, read again

    def progress(self) -> tuple:
        """(phase, bytes so far, bytes in all) of the model being or last downloaded (▶ `fetch._Report`): "checking",
        the bytes read against those to read; "downloading", for an update the bytes received against those it fetches,
        for a download those on the disk, or those it held at the start and the bytes received since when that is
        more."""
        if not self.name:
            return "downloading", 0, self.total
        try:
            report = json.load(open(self.report))
        except (OSError, ValueError):
            report = {}
        if report.get("phase") == "checking" and self.running():
            done, total = (report.get("checking") or [0, 0])[:2]
            return "checking", done, total
        received = (report.get("huggingface_hub.snapshot_download.transfer")
                    or report.get("huggingface_hub.snapshot_download") or [0])[0]
        if self.updating:
            total = int(report.get("total") or 0)
            return "downloading", (total if self.message == "done" else min(received, total)), total
        return "downloading", max(folder_bytes(MODELS_DIR / self.name), self.base + received), self.total

    def cancel(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


DOWNLOADS = ModelDownloads()
