"""A PICTURE MODEL BEHIND THE SERVER — Qwen-Image as a conversation's model (v0.10.0): a message is a prompt, its
answer a picture.

⚖ 2026-10-15: *"its own model first"* — the server booted with an image pack (▶ `qwen_image.image_pack`) holds an
`ImageModel` where it holds a text model, and an `ImageRuntime` where it holds `session.Runtime`, in the shapes its
endpoints read: a conversation is the same folder with the same history, its state an empty cartridge (a picture
keeps nothing on the card between messages); a message's picture is made by the picture worker, a process of its own
(▶ `image_worker`: a step is one engine call of seconds, which in the server's process would hold every request up),
which tells the stream its progress — the encoder's boot, the transformer's steps, the decode — and saves the picture in
the conversation's folder (`pictures/`), the answer's text naming it. The card holds one model at a time, a boot each.
"""
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
from multiprocessing.connection import Connection

from .model_folder import Refused
from .qwen_image import Stopped
from .session import Cartridge

DEFAULTS = {"width": 1024, "height": 1024, "steps": 40, "seed": None}
# a message's settings, written in it and taken out of the prompt: `--steps 20`, `--size 768x1024`, `--seed 7`
SETTING = re.compile(r"(?:^|\s)--(steps|size|seed)\s+(\d+(?:x\d+)?)(?=\s|$)")
PICTURE = re.compile(r"!\[[^\]]*\]\(picture:([A-Za-z0-9_.-]+)\)")      # an answer's picture, as its text names it


class ImageFolder:
    """What the server reads of a model's folder, for an image pack."""
    architecture = "qwen_image"

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.generation = {}
        self.lora = None
        h = hashlib.sha256()
        for part in ("transformer", "text_encoder"):
            man = json.load(open(os.path.join(self.path, part, "pack.json")))
            h.update(json.dumps(man.get("source", {}), sort_keys=True).encode())
            h.update(json.dumps(man.get("policy", {}), sort_keys=True).encode())
        self._ident = {"model": h.hexdigest()[:16], "architecture": self.architecture, "lora": None}

    def loras(self):
        return []

    def identity(self):
        return dict(self._ident)


class ImageModel:
    """An image pack where the server holds a model: no context, no state on the card, no LoRA."""
    PARKS = False

    def __init__(self, path, silicon=None, settings=None):
        self.folder = ImageFolder(path)
        self.max_context = 0
        self.active_lora = None
        self.silicon = silicon
        self.settings = dict(DEFAULTS, **(settings or {}))

    def state_regions(self, length):
        return []

    def lora_identity(self):
        return None


class PictureJob:
    """A message's picture in the making: its thread's progress, and a stop the server's `/stop` asks for."""

    def __init__(self):
        self.done = threading.Event()
        self.cancelled = False
        self.error = None


class ImageRuntime:
    """`session.Runtime`'s part the server's endpoints read, for an image model, and the picture a message makes."""

    def __init__(self, model):
        self.model = model
        self.identity = model.folder.identity()
        self.active = None
        self.cartridges = {}
        self.stats = dict(switches=0, lora_swaps=0, lora_seconds=0.0, park_seconds=0.0)
        self._job = None
        self._lock = threading.Lock()
        self._proc = self._conn = None
        self._starting = threading.Lock()

    def new_cartridge(self, lora=None):
        if lora is not None:
            raise Refused("a picture model takes no LoRA")
        c = Cartridge(self.identity, None)
        self.cartridges[c.id] = c
        return c

    def adopt(self, c):
        if not c.matches(self.identity):
            raise Refused("the conversation %s belongs to %s, and this machine runs %s" % (c.id, c.identity, self.identity))
        self.cartridges[c.id] = c
        return c

    def save(self, c, path, with_state=True):
        c.save(path, False)

    def busy(self):
        return self._job is not None and not self._job.done.is_set()

    def close(self):
        if self._job is not None:
            self._job.cancelled = True
            self._job.done.wait()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._conn.send(("quit",))
                self._proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()

    def start(self):
        """The picture worker started before the first picture asks for it, so the table it makes when it starts is
        made while nothing waits on it."""
        self._worker()

    def _worker(self):
        """The picture worker, started the first time it is wanted and again if it has ended: its end of a socket pair
        handed to it by number, this process's path its own."""
        with self._starting:
            if self._proc is not None and self._proc.poll() is None:
                return self._conn
            return self._spawn()

    def _spawn(self):
        mine, theirs = socket.socketpair()
        code = ("import sys; sys.path[:0] = %r; from silvann_runtime.image_worker import main; main()"
                % [p for p in sys.path if p])
        self._proc = subprocess.Popen([sys.executable, "-c", code, self.model.folder.path, str(theirs.fileno()),
                                       self.model.silicon or ""], pass_fds=(theirs.fileno(),))
        theirs.close()
        self._conn = Connection(mine.detach())
        if self._conn.recv()[0] != "ready":
            raise Refused("the picture worker did not start")
        return self._conn

    # ── a message's picture ──────────────────────────────────────────────────────────────────────────────────────
    def settings(self, text):
        """The message's prompt and its settings: the model's (the config's `image`), what the message writes in it
        (`--steps N`, `--size WxH`, `--seed N`) over them. Sides are made multiples of 32, as the pipeline makes them."""
        s = dict(self.model.settings)
        for key, value in SETTING.findall(text):
            if key == "size":
                w, _, h = value.partition("x")
                s["width"], s["height"] = int(w), int(h or w)
            else:
                s[key] = int(value)
        prompt = SETTING.sub(" ", text).strip()
        s["width"], s["height"] = (max(256, min(2048, int(v) // 32 * 32)) for v in (s["width"], s["height"]))
        s["steps"] = max(1, min(100, int(s["steps"])))
        if s["seed"] is None:
            s["seed"] = int.from_bytes(os.urandom(4), "little")
        return prompt, s

    def picture(self, prompt, s, folder, progress):
        """The picture for `prompt` with the settings `s`, made by the picture worker and saved as a PNG in `folder`:
        its file's name, and the seconds it took. `progress(stage, done, total)` as the worker tells it; the job's stop
        is handed on to the worker, which stops between steps (`Stopped`)."""
        if not self._lock.acquire(blocking=False):
            raise Refused("a picture is being made")
        job = self._job = PictureJob()
        try:
            conn = self._worker()
            conn.send(("picture", prompt, s, folder))
            asked = False
            while True:
                if job.cancelled and not asked:
                    conn.send(("stop",))
                    asked = True
                if not conn.poll(0.5):
                    if self._proc.poll() is not None:
                        raise Refused("the picture worker ended (%s)" % self._proc.returncode)
                    continue
                msg = conn.recv()
                if msg[0] == "progress":
                    progress(*msg[1:])
                elif msg[0] == "done":
                    return msg[1], msg[2]
                elif msg[0] == "stopped":
                    job.cancelled = True
                    raise Stopped(msg[1])
                elif msg[0] == "error":
                    raise Refused(msg[1])
        except EOFError:
            raise Refused("the picture worker ended (%s)" % self._proc.poll())
        finally:
            job.done.set()
            self._lock.release()


def caption(name, s, seconds):
    """An answer's text for a picture: the picture, then what made it."""
    return "![picture](picture:%s)\n\n_%d×%d · %d steps · seed %d · %d s_" % (name, s["width"], s["height"], s["steps"],
                                                                           s["seed"], round(seconds))
