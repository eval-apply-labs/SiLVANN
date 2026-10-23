"""SiLVANN chat UI — NiceGUI client that points at a start_silvann_server.py HTTP endpoint.

Layout: a left drawer holds the navigation (Chat / Options / Help), conversations sub-tree, and
endpoint settings. The main area swaps content based on the selected menu. Drawer is collapsible.

Install (anywhere — laptop, Mac, dev box):
    pip install nicegui httpx
Run:
    python3 silvann_ui.py                            # serves UI on http://localhost:8080
    python3 silvann_ui.py --port 8090 --no-browser   # different port, no auto-open

For local mode (Options view) to work, run from the silvann_v4 repo root so ./configs/ and
./start_silvann_server.py are discoverable.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
from nicegui import ui, app


# an answer's think block: closed, or still open while the model thinks (the server sends the `<think>` its prompt opened)
THINK_PART_RE = re.compile(r"<think>(.*?)(</think>|\Z)", re.DOTALL)
# the model's tool (NN-88): the server writes every model's calls the same way in the answer's text
TOOL_RE = re.compile(r"<tool_(call|response|ask|answer)>(.*?)</tool_\1>", re.DOTALL)
ASK_LABELS = {"folder": "allow the folder", "once": "allow once", "refuse": "refuse"}
# a picture model's answer names its picture (v0.10.0): `![picture](picture:NAME)`, drawn from the server's copy
PICTURE_RE = re.compile(r"!\[([^\]]*)\]\(picture:([A-Za-z0-9_.-]+)\)")


def _tool_parts(text: str) -> list:
    """Text outside a think block as ("text", markdown) and ("tools", lines) parts: a run of calls to the model's tool is
    one part — a form after `>`, its value under it, an ask and the user's answer as comments."""
    parts, block, pos = [], [], 0

    def flush():
        if block:
            parts.append(("tools", list(block)))
            block.clear()
    for m in TOOL_RE.finditer(text):
        gap = text[pos:m.start()]
        if gap.strip():
            flush()
            parts.append(("text", gap))
        kind, body = m.group(1), m.group(2)
        block.append({"call": "> " + body, "response": body, "ask": ";; asked: " + body,
                      "answer": ";; you answered: " + ASK_LABELS.get(body, body)}[kind])
        pos = m.end()
    flush()
    if text[pos:].strip():
        parts.append(("text", text[pos:]))
    return parts


def answer_parts(content: str) -> list:
    """An answer's text as the parts it is drawn in, in order: ("think", text, closed) a think block — `closed` False while
    the model is still in it —, ("tools", lines) a run of calls to the model's tool, ("text", markdown) the rest. A
    `</think>` with no `<think>` before it closes a block the answer began in (a conversation saved without the tag)."""
    parts, pos = [], 0
    first_open, first_close = content.find("<think>"), content.find("</think>")
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        parts.append(("think", content[:first_close], True))
        pos = first_close + len("</think>")
    for m in THINK_PART_RE.finditer(content, pos):
        parts += _tool_parts(content[pos:m.start()])
        parts.append(("think", m.group(1), m.group(2) == "</think>"))
        pos = m.end()
    parts += _tool_parts(content[pos:])
    return parts


def tools_block(lines: list) -> str:
    """A run of tool calls as the markdown a tools section shows: one `<!tools> … </tools>` block."""
    return "```lisp\n<!tools>\n%s\n</tools>\n```" % "\n".join(lines)


# ---------- HTTP client ----------

class Client:
    """Wraps the SiLVANN HTTP API. One instance per browser session.

    A `passwords` dict maps cart_id -> plaintext password (kept in app.storage.user). Every
    cartridge-touching method auto-injects the X-Cart-Password header when a password is known.
    `web_user_id` is the per-browser-tab UUID; it's attached as X-Web-User-Id on every request and
    is what server-side ownership/heartbeat is keyed by."""

    def __init__(self, endpoint: str, passwords: Optional[dict] = None,
                 web_user_id: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
        self.passwords: dict = passwords if passwords is not None else {}
        self.web_user_id = web_user_id or ""

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def set_password(self, cart_id: str, password: Optional[str]):
        if password is None or password == "":
            self.passwords.pop(cart_id, None)
        else:
            self.passwords[cart_id] = password

    def _hdr(self, cart_id: Optional[str]) -> dict:
        h: dict = {}
        if self.web_user_id:
            h["X-Web-User-Id"] = self.web_user_id
        if cart_id and cart_id in self.passwords:
            h["X-Cart-Password"] = self.passwords[cart_id]
        return h

    async def health(self) -> dict:
        r = await self.http.get(f"{self.endpoint}/health")
        r.raise_for_status()
        return r.json()

    async def picture(self, cart_id: str, name: str) -> bytes:
        """A picture a conversation's answer made, as a JPEG to show."""
        r = await self.http.get(f"{self.endpoint}/cartridges/{cart_id}/pictures/{name}", params={"format": "jpeg"},
                                headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.content

    async def list_cartridges(self) -> list:
        r = await self.http.get(f"{self.endpoint}/cartridges")
        r.raise_for_status()
        return r.json().get("cartridges", [])

    async def create(self, name: str, system: Optional[str] = None,
                     cooling: Optional[dict] = None,
                     kv_cache_size: Optional[int] = None) -> dict:
        body: dict = {"name": name, "system": system}
        if cooling is not None:
            body["cooling"] = cooling
        if kv_cache_size is not None:
            body["kv_cache_size"] = kv_cache_size
        r = await self.http.post(f"{self.endpoint}/cartridges", json=body)
        r.raise_for_status()
        return r.json()

    async def estimate_vram(self, cooling: Optional[dict] = None,
                            kv_cache_size: Optional[int] = None) -> dict:
        """Ask the server how much VRAM a new conversation would cost under these cache
        settings (no body = the server's configured defaults). Used by the new-chat dialog."""
        body: dict = {}
        if cooling is not None:
            body["cooling"] = cooling
        if kv_cache_size is not None:
            body["kv_cache_size"] = kv_cache_size
        r = await self.http.post(f"{self.endpoint}/cartridges/estimate", json=body)
        r.raise_for_status()
        return r.json()

    async def activate(self, cart_id: str) -> dict:
        """Metadata-only fetch (lazy). Returns name/history/next_pos/enable_thinking/protected
        plus bound/bound_by_self/bound_by_other so the UI knows whether to show 'Activate'."""
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/activate",
                                 headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def bind(self, cart_id: str) -> dict:
        """Load cart into VRAM + claim ownership. 423 if another session owns it, 507 if VRAM
        is exhausted."""
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/bind",
                                 headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def unbind(self, cart_id: str) -> dict:
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/unbind",
                                 headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def ping(self) -> dict:
        r = await self.http.post(f"{self.endpoint}/sessions/ping", headers=self._hdr(None))
        r.raise_for_status()
        return r.json()

    async def busy(self) -> dict:
        """{"busy": bool, "cart": <generating cart id or None>}. The engine serializes turns,
        so this tells a waiting client whether another conversation is mid-turn."""
        r = await self.http.get(f"{self.endpoint}/busy")
        r.raise_for_status()
        return r.json()

    async def rename(self, cart_id: str, name: str):
        r = await self.http.patch(f"{self.endpoint}/cartridges/{cart_id}",
                                  json={"name": name}, headers=self._hdr(cart_id))
        r.raise_for_status()

    async def set_thinking(self, cart_id: str, enable: bool):
        r = await self.http.patch(f"{self.endpoint}/cartridges/{cart_id}",
                                  json={"enable_thinking": bool(enable)},
                                  headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def delete(self, cart_id: str):
        r = await self.http.delete(f"{self.endpoint}/cartridges/{cart_id}",
                                   headers=self._hdr(cart_id))
        r.raise_for_status()

    def export_ticket(self, cart_id: str) -> str:
        """A one-time path the browser downloads a conversation's archive from — ▶ `_export_download`."""
        token = secrets.token_urlsafe(16)
        _EXPORTS[token] = (self.endpoint, cart_id, self._hdr(cart_id), time.time())
        return "/silvann-export/%s" % token

    async def export(self, cart_id: str) -> bytes:
        r = await self.http.get(f"{self.endpoint}/cartridges/{cart_id}/export",
                                headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.content

    async def import_bytes(self, blob: bytes, filename: str = "cart.tar") -> dict:
        files = {"file": (filename, blob, "application/x-tar")}
        r = await self.http.post(f"{self.endpoint}/cartridges/import", files=files)
        r.raise_for_status()
        return r.json()

    async def stop_generation(self, cart_id: str) -> dict:
        """Ask the server to interrupt an in-flight generation. No-op if nothing's running."""
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/stop",
                                 headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def save(self, cart_id: str, password: Optional[str] = None) -> dict:
        """Force-persist + optionally set/clear the password. password=None leaves it unchanged;
        password="" clears it; password="something" sets it. The X-Cart-Password header must
        match the *current* password (so you can't reset someone else's protected cart)."""
        body: dict = {}
        if password is not None:
            body["password"] = password
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/save",
                                 json=body, headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def upload_image(self, cart_id: str, blob: bytes,
                            filename: str = "image.png",
                            content_type: str = "application/octet-stream") -> dict:
        """POST an image to a bound cartridge. Server runs the ViT once and caches the
        encoded rows under image_id (one-shot; the next /messages POST that lists the
        id consumes it). Returns {image_id, merged_thw, n_placeholders}."""
        files = {"file": (filename, blob, content_type)}
        r = await self.http.post(f"{self.endpoint}/cartridges/{cart_id}/images",
                                 files=files, headers=self._hdr(cart_id),
                                 timeout=httpx.Timeout(120.0))
        r.raise_for_status()
        return r.json()

    async def set_tools(self, cart_id: str, enable: bool):
        r = await self.http.patch(f"{self.endpoint}/cartridges/{cart_id}", json={"tools": bool(enable)},
                                  headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def set_tools_mode(self, cart_id: str, mode: str):
        """What the conversation's tool may do unasked: "ask", "chill" or "full-auto"."""
        r = await self.http.patch(f"{self.endpoint}/cartridges/{cart_id}", json={"tools_mode": mode},
                                  headers=self._hdr(cart_id))
        r.raise_for_status()
        return r.json()

    async def tools_settings(self):
        """The server config's tools — a new conversation's mode, or "no" — and the folders (▶ /tools)."""
        r = await self.http.get(f"{self.endpoint}/tools")
        r.raise_for_status()
        return r.json()

    async def stream_answer(self, cart_id: str, choice: str, max_new_tokens: int):
        """The user's answer to the model's ask, then the rest of the answer — as `stream_message` yields it."""
        async for kind, payload in self._stream(f"{self.endpoint}/cartridges/{cart_id}/tools/answer",
                                                {"choice": choice, "max_new_tokens": max_new_tokens}, cart_id):
            yield kind, payload

    async def stream_message(self, cart_id: str, text: str, max_new_tokens: int,
                              images: Optional[list] = None):
        """Yield (event_kind, payload_dict) tuples from the server's SSE stream until done|error.

        `images` is an optional list of image_ids previously uploaded via
        `upload_image`; the server splices their cached embeddings into the prefill
        stream and pops them after a successful turn."""
        url = f"{self.endpoint}/cartridges/{cart_id}/messages"
        body = {"text": text, "max_new_tokens": max_new_tokens}
        if images:
            body["images"] = list(images)
        async for kind, payload in self._stream(url, body, cart_id):
            yield kind, payload

    async def _stream(self, url: str, body: dict, cart_id: str):
        # ⚖ A turn already on the card is answered 503 with Retry-After: 5, and the caller asks again — this is the
        # caller, so it does: a ("waiting", {}) event each time, then another ask 5 s later.
        while True:
            async with self.http.stream("POST", url, json=body,
                                        headers=self._hdr(cart_id),
                                        timeout=httpx.Timeout(None)) as resp:
                if resp.status_code == 503:
                    await resp.aread()
                    yield "waiting", {}
                    await asyncio.sleep(float(resp.headers.get("Retry-After", 5)))
                    continue
                if resp.status_code >= 400:
                    await resp.aread()                # the server's reason, which a caller reads off the error
                resp.raise_for_status()
                event = None
                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    if raw.startswith("event:"):
                        event = raw.split(":", 1)[1].strip()
                    elif raw.startswith("data:") and event is not None:
                        try:
                            payload = json.loads(raw.split(":", 1)[1].strip())
                        except Exception:
                            payload = {}
                        yield event, payload
                        if event in ("done", "error"):
                            return
                        event = None
                return


# ---------- Exports: an archive streamed from the server to the browser over HTTP ----------
# token -> (endpoint, conversation, headers, when) — one use each, and gone after ten minutes unused
_EXPORTS: dict = {}


@app.get("/silvann-export/{token}")
async def _export_download(token: str):
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    for t, (_, _, _, at) in list(_EXPORTS.items()):
        if time.time() - at > 600:
            _EXPORTS.pop(t, None)
    ticket = _EXPORTS.pop(token, None)
    if ticket is None:
        raise HTTPException(404, "this export link was used already, or has expired")
    endpoint, cart_id, headers, _ = ticket
    http = httpx.AsyncClient(timeout=None)
    req = http.build_request("GET", f"{endpoint}/cartridges/{cart_id}/export", headers=headers)
    resp = await http.send(req, stream=True)
    if resp.status_code != 200:
        body = (await resp.aread()).decode(errors="replace")[:300]
        await resp.aclose(); await http.aclose()
        raise HTTPException(resp.status_code, body)

    async def body():
        try:
            async for chunk in resp.aiter_bytes(1 << 20):
                yield chunk
        finally:
            await resp.aclose(); await http.aclose()
    return StreamingResponse(body(), media_type="application/x-tar",
                             headers={"Content-Disposition": f'attachment; filename="{cart_id}.silvann.tar"'})


# ---------- Local server subprocess (process-wide singleton) ----------

REPO_ROOT = Path(__file__).resolve().parent          # main/ — the runtime root
SERVER_SCRIPT = REPO_ROOT / "start_silvann_server.py"
CORE_SCRIPT = REPO_ROOT / "silvann_core.py"          # lists and downloads the models this release knows
# what the machine keeps — the models and the conversations: beside the UI, or where SILVANN_DATA says (a container's
# volume), as the server and silvann_core.py read it
DATA_DIR = Path(os.environ.get("SILVANN_DATA") or REPO_ROOT)
MODELS_DIR = DATA_DIR / "models"                     # models/<name>/ — self-contained model folders
LAST_START = DATA_DIR / "ui_last_start.json"         # the model a --local UI starts again when it starts
TOKEN_FILE = DATA_DIR / ".huggingface_token"         # the Hugging Face token, as silvann_core.py's hf-token saves it
# the documents: docs/ beside the UI in a release, release/ in the development tree
RELEASE_DIR = REPO_ROOT if (REPO_ROOT / "docs" / "users.md").exists() else REPO_ROOT.parent / "release"
DOCS_DIR = RELEASE_DIR / "docs"
# the engine's source and the licence: in a release beside the docs, in the development tree src and its root
ENGINE_SRC = RELEASE_DIR / "engine" / "src" if (RELEASE_DIR / "engine" / "src").is_dir() else REPO_ROOT.parent / "src"
LICENCE_DIR = RELEASE_DIR if (RELEASE_DIR / "LICENSE").exists() else REPO_ROOT.parent


def list_models():
    """The model folders under MODELS_DIR that are there whole — a download still running is not one yet."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir() and model_present(d.name))


def model_configs_dir(model: str) -> Path:
    return MODELS_DIR / model / "configs"


def list_model_configs(model: str):
    """The run configs (*.json) inside a model folder's configs/ dir."""
    cdir = model_configs_dir(model)
    return sorted(p.name for p in cdir.glob("*.json")) if cdir.is_dir() else []


def model_default_config(model: str) -> Optional[str]:
    """The config a model folder declares as its default (pack.json `default_config`).
    Drives the Options-view config selector so it lands on the model's intended setup (e.g. the
    200k cooled config) rather than alphabetical-first. None if pack.json is absent/unreadable."""
    try:
        pack = json.loads((MODELS_DIR / model / "pack.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    dflt = pack.get("default_config")
    return dflt if dflt in list_model_configs(model) else None


# ⭐ A NEW CONFIG: every key the server reads, at its default — settings only. Help → "A model's config" (docs/config.md)
#   says every key and every model's option, and the Options view's **config guide** opens it. Options are left out: a
#   placement named in `options` is kept as it is, where one left out is chosen for the machine at each boot.
CONFIG_GUIDE_DOC = "Using SiLVANN · A model's config"


def new_config(model: str) -> dict:
    """A new config for `model`: every key the server reads at its default."""
    folder = MODELS_DIR / model
    if (folder / "model_index.json").exists():                  # a picture model
        return {"card": "auto",
                "image": {"width": 1024, "height": 1024, "steps": 40, "seed": None}}
    try:
        c = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        layers = (c.get("text_config") or c).get("num_hidden_layers")
    except (OSError, ValueError):
        layers = None
    return {
            # what the card holds beside the model, up to its length, a tier sharing the card with the cache (NN-93)
            "max_context": "auto",
            "max_tokens_per_video": 8192, "thinking": True, "sampler": "generation",
            "tools": "chill", "tools_folder": None, "lora": None, "num_cards": 1, "card": "auto", "vram_margin_mb": 50,
            "image_model": "auto", "image_card": None, "workers": [],
            "options": {},
            "pipeline_parallelism": "no",
            "pipeline_parallelism_layers": "0-%d" % (layers - 1) if layers else "",
            "pipeline_parallelism_next": "192.0.2.2:8766", "pipeline_parallelism_port": 8766}


# local mode needs the server beside the UI; a machine with no model yet downloads one from the same panel
LOCAL_AVAILABLE = SERVER_SCRIPT.exists()
LOCAL_APP = False                                    # --local: the UI as the app over this machine's models

# The Help view's documents, a group each: (group, label, path) — every document the release ships, the UI's guide first
HELP_DOCS = [
    ("Using SiLVANN", "The UI", DOCS_DIR / "ui.md"),
    ("Using SiLVANN", "Overview", RELEASE_DIR / "README.md"),
    ("Using SiLVANN", "Running SiLVANN", DOCS_DIR / "users.md"),
    ("Using SiLVANN", "A model's config", DOCS_DIR / "config.md"),
    ("Using SiLVANN", "The model's tool and your own commands", DOCS_DIR / "tools.md"),
    ("Using SiLVANN", "Skills", RELEASE_DIR / "skills" / "README.md"),
    ("Using SiLVANN", "A container of your own", RELEASE_DIR / "docker" / "README.md"),
    ("Using SiLVANN", "Every release's notes", RELEASE_DIR / "release_logs" / "v0.md"),
    ("Using SiLVANN", "Why AI on your own machine", DOCS_DIR / "local_ai.md"),
    ("The models", "The models", DOCS_DIR / "models.md"),
    ("The models", "Their results", DOCS_DIR / "results.md"),
    ("The models", "A model's folder", RELEASE_DIR / "models" / "README.md"),
    ("The models", "How the weights are packed", DOCS_DIR / "compression.md"),
    ("Developing", "Working on SiLVANN", DOCS_DIR / "developers.md"),
    ("Developing", "The ideas behind it", DOCS_DIR / "concepts.md"),
    ("Developing", "Where it could go", DOCS_DIR / "directions.md"),
    ("Developing", "The engine's source", ENGINE_SRC / "README.md"),
    ("Developing", "Package sys", ENGINE_SRC / "packages" / "sys" / "README.md"),
    ("Developing", "Package nn", ENGINE_SRC / "packages" / "nn" / "README.md"),
    ("Developing", "Package ai_qwen_3", ENGINE_SRC / "packages" / "ai_qwen_3" / "README.md"),
    ("Developing", "Package ai_z__glm_5_3", ENGINE_SRC / "packages" / "ai_z__glm_5_3" / "README.md"),
    ("Developing", "Package ai_qwen_image", ENGINE_SRC / "packages" / "ai_qwen_image" / "README.md"),
    ("Developing", "Package ai_deepseek_4", ENGINE_SRC / "packages" / "ai_deepseek_4" / "README.md"),
    ("Developing", "Adding a silicon family", ENGINE_SRC / "silicon_families" / "add_a_family.md"),
    ("Developing", "OpenCL 1.2 for integrated GPUs", ENGINE_SRC / "silicon_families" / "khronos_opencl1_igpu" / "README.md"),
    ("Developing", "Contributing", LICENCE_DIR / "CONTRIBUTING.md"),
    ("Licences", "The licence", LICENCE_DIR / "LICENSE"),
    ("Licences", "Attribution", LICENCE_DIR / "ATTRIBUTION.md"),
    ("Licences", "The contributor licence agreement", LICENCE_DIR / "CLA.md"),
]
HELP_FIRST = "Using SiLVANN · The UI"


def help_doc(path: Path) -> str:
    """A document as the Help view shows it: Markdown as it is, any other text in a block of its own."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"_`{path.name}` could not be read: {e}_"
    return text if path.suffix == ".md" else "```text\n" + text + "\n```"


HELP_HREF = re.compile(r"\]\(<?([^)\s>]+)")             # a Markdown link's target


def help_target(current: Path, href: str) -> Optional[Path]:
    """The file a relative link in `current` names, without its anchor — None where it names no file."""
    target = href.split("#", 1)[0]
    if not target or re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):
        return None
    path = (current.parent / target).resolve()
    return path if path.is_file() else None


def help_shipped() -> set:
    """The files the Help view shows: the picker's documents, and the files their links name, document to document —
    the release's own text. The browser says which link was clicked, and it can say any path: the Help view shows no
    file outside this set, or anyone who reaches the UI could read any file it can (a token, a home folder's keys)."""
    seen, todo = set(), [path.resolve() for _, _, path in HELP_DOCS if path.is_file()]
    while todo:
        doc = todo.pop()
        if doc in seen:
            continue
        seen.add(doc)
        if doc.suffix == ".md":
            try:
                text = doc.read_text(encoding="utf-8")
            except OSError:
                continue
            todo += [p for p in (help_target(doc, href) for href in HELP_HREF.findall(text)) if p is not None]
    return seen


HELP_SHIPPED = help_shipped()


def help_link(current: Path, href: str) -> Optional[Path]:
    """The document a link in `current` names, without its anchor — None where it names none the release ships."""
    path = help_target(current, href)
    return path if path in HELP_SHIPPED else None


PROGRESS = "[silvann_server] loading: "      # the server's progress line, the whole of it so far each time it grows


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
        """A line of the server's output kept for the panes — a load's progress line in place of the one before it, so
        the pane shows one line growing (`loading: ...10...20..`)."""
        with self.lock:
            if progress and self.lines and PROGRESS in self.lines[-1]:
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


# ---------- Models fetched from Hugging Face (process-wide singleton) ----------

def model_present(name: str) -> bool:
    """A model folder that is there, as silvann_runtime.fetch.present says: a text model's pack and config, or a picture
    model's — diffusers' model_index.json and its transformer packed."""
    d = MODELS_DIR / name
    return ((d / "pack.json").is_file() and (d / "config.json").is_file()
            or (d / "model_index.json").is_file() and (d / "transformer" / "pack.json").is_file())


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
    the disk under MODELS_DIR has less room than the model still needs, and 1 GB."""

    def __init__(self):
        self.known: Optional[list] = None   # [{name, repo, size}], read once from silvann_core.py
        self.proc: Optional[subprocess.Popen] = None
        self.name: Optional[str] = None
        self.total = 0
        self.base = 0                        # the bytes the folder held when the download started
        self.report = os.path.join(tempfile.gettempdir(), "silvann_download_%d.json" % os.getpid())
        self.message = ""
        self.lines: deque[str] = deque(maxlen=200)

    def catalogue(self) -> list:
        if self.known is None:
            r = subprocess.run([sys.executable, str(CORE_SCRIPT), "list-models", "--json"], capture_output=True,
                               text=True, cwd=str(REPO_ROOT), timeout=300)
            if r.returncode != 0:
                raise RuntimeError("silvann_core.py list-models failed: " + (r.stderr or r.stdout)[-500:])
            self.known = [{k: m[k] for k in ("name", "repo", "size")}
                          for m in json.loads(r.stdout.strip().splitlines()[-1])]
        return [dict(m, here=model_present(m["name"])) for m in self.known]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, name: str):
        if self.running():
            raise RuntimeError(f"{self.name} is downloading — one at a time")
        model = next((m for m in self.catalogue() if m["name"] == name), None)
        if model is None:
            raise RuntimeError(f"{name} is not a model this release knows")
        total = repo_bytes(model["repo"]) or approx_bytes(model["size"])
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        base = folder_bytes(MODELS_DIR / name)
        need = max(0, total - base)
        free = shutil.disk_usage(MODELS_DIR).free
        if need + (1 << 30) > free:
            raise RuntimeError(f"{name} needs {need / 1e9:.1f} GB more, and {MODELS_DIR} has {free / 1e9:.1f} GB free")
        self.name, self.total, self.base, self.message = name, total, base, ""
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

    def progress(self) -> tuple:
        """(bytes there, bytes in all) of the model being or last downloaded: those on the disk, or those it held at
        the start and the bytes received since when that is more."""
        if not self.name:
            return 0, self.total
        received = 0
        try:
            bars = json.load(open(self.report))
            received = (bars.get("huggingface_hub.snapshot_download.transfer")
                        or bars.get("huggingface_hub.snapshot_download") or [0])[0]
        except (OSError, ValueError):
            pass
        return max(folder_bytes(MODELS_DIR / self.name), self.base + received), self.total

    def cancel(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


DOWNLOADS = ModelDownloads()


# ---------- Per-browser session state ----------

def _session_default():
    return {
        "endpoint": os.environ.get("SILVANN_DEFAULT_ENDPOINT", "http://localhost:8765"),
        "active_id": None,
        "active_name": None,
        "active_model": None,      # the model the open conversation is with; another serving empties the chat
        # Per-cart passwords this browser session has unlocked. Kept in app.storage.user so they
        # survive page reloads for the same user (one browser identity). They are NOT shared with
        # other users' sessions.
        "passwords": {},
        "protected_active": False,
        # Whether the active cart is currently bound to VRAM. False after a lazy /activate
        # (metadata only) — flips True after the user clicks the Activate banner.
        "active_bound": False,
        "history": [],     # [{"role": "user"|"assistant", "content": str}]
        "streaming": False,
        # Which conversation's inline-actions drawer is currently open (one at a time).
        "expanded_cart_id": None,
        # Display toggle: render <think>...</think> blocks in assistant messages or strip them.
        "show_thinking": False,
        # The page light or dark — this browser's choice.
        "dark": False,
        # Engine toggle for the active conversation: whether the model is asked to think at all.
        # Synced from the server on activate (the canonical value lives in the cartridge sidecar).
        "active_thinking_mode": True,
        # Nav state: which main view is selected and whether the chat sub-tree is expanded.
        # The drawer itself defaults to mini-with-hover-expand; that's a fixed UX, not stored.
        "view": "chat",            # chat | options | help
        "convs_expanded": True,
        # Options-view prefs (was the Server tab)
        "server_mode": "local",    # local (this machine's server) | remote (another's, at `endpoint`)
        "selected_model": "",      # a folder name under main/models/ (drives the config selector)
        "selected_config": "",
        "server_port": 8765,
        # Index of the last log line shown in the subprocess log pane
        "log_cursor": 0,
        # Per-cart pending image attachments awaiting the next /messages send.
        # Shape: {cart_id: [{"image_id": str, "filename": str, "n_pads": int,
        # "merged_thw": [T,H,W]}]}. Cleared after each successful turn (server
        # also pops the cache server-side — one-shot semantics).
        "pending_images": {},
    }


# The page's head: its layout's styles, the chat's looks (NN-102: the thinking and the tool's sections wrapped, the
# dark colours, the copy button on code) and the button's script.
PAGE_HEAD = """
<style>
    /* Keep Quasar's natural q-page-container left-padding so the (mini) drawer pushes the
       content rather than overlapping it. Only zero top-padding (no header here). */
    .q-page-container { padding-top: 0 !important; }
    .q-page {
        padding: 0 !important;
        /* dvh = dynamic viewport height — tracks the visible area on mobile as the address
           bar shows/hides. Falls back to vh on older browsers via the duplicate decl. */
        min-height: 100vh;
        min-height: 100dvh;
        /* ⛳ AND EXACTLY THAT TALL: a page only AT LEAST the viewport leaves its flex children an indefinite
           height, and the chat's scroll area — which needs a definite one — then sizes its track to one
           height and its content to another (the scrollbar spanning the view, the messages only the top). */
        height: 100vh;
        height: 100dvh;
        display: flex;
        flex-direction: column;
        overflow-x: hidden;          /* never scroll the page sideways — overflow goes vertical */
    }
    .nicegui-content {
        padding: 0 !important;
        flex: 1 1 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow-x: hidden;          /* clamp horizontal; the views stack + scroll vertically */
        overflow-y: auto;
    }
    /* The main views: full width, never wider than the area; vertical scroll inside. min-width:0
       lets flex content shrink instead of forcing a sideways scrollbar. */
    .silvann-main {
        flex: 1 1 0; min-height: 0; min-width: 0; max-width: 100%;
        display: flex; flex-direction: column;
        overflow-x: hidden;
    }
    /* The conversation is as wide as its view: a line longer than a bubble — a tool's value, a model's code —
       scrolls sideways inside its own code block. A bubble's text is a flex item, which would otherwise grow to
       its longest line and carry the whole conversation, the user's messages with it, off to the right. */
    .silvann-chat .q-scrollarea__content { width: 100%; }
    .silvann-chat .q-message-container > div { min-width: 0; }
    .silvann-chat .q-message-text pre { overflow-x: auto; }

    /* Menu rows in the drawer. In mini mode (60px wide drawer) only the leading icon shows;
       when the drawer expands on hover (overflow-x: visible), labels/chevrons reappear. */
    .silvann-menu-item {
        cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 0.25rem;
        display: flex; align-items: center; gap: 0.5rem;
        user-select: none; white-space: nowrap;
    }
    .silvann-menu-item:hover { background: rgba(0,0,0,0.05); }
    .silvann-menu-active { background: rgba(59, 130, 246, 0.12); color: rgb(37, 99, 235); font-weight: 600; }

    /* The collapsed mini-drawer is 60px wide. Hide everything past the leading icon so
       labels don't smear into the next item — using `display: none` (not opacity) so the
       hidden elements don't reserve width and the icon really centers. */
    .q-drawer--mini .silvann-menu-item > :not(.q-icon:first-child) {
        display: none;
    }
    .q-drawer--mini .silvann-menu-item { justify-content: center; padding: 0.5rem 0; }
    .q-drawer--mini .silvann-conv-item,
    .q-drawer--mini .silvann-brand-label,
    .q-drawer--mini .silvann-convs-col { display: none; }
    .q-drawer { transition: width 0.18s ease; }
    /* The drawer is a fixed-height (~100vh) flex column. Without flex-wrap:nowrap its content
       WRAPS into side-by-side columns (logo header | Chat/Options/Help | endpoint footer) once
       it's taller than the drawer — that's the "3 columns / horizontal scroll" bug. Force a
       single non-wrapping vertical stack and let it scroll vertically instead. */
    .nicegui-drawer { flex-wrap: nowrap !important; }
    .nicegui-drawer .q-drawer__content {
        flex-wrap: nowrap !important;
        overflow-y: auto;
        overflow-x: hidden !important;
    }

    .silvann-conv-item {
        cursor: pointer; padding: 0.35rem 0.5rem; border-radius: 0.25rem;
        display: flex; align-items: center; gap: 0.4rem;
        user-select: none;
    }
    .silvann-conv-item:hover { background: rgba(0,0,0,0.05); }
    .silvann-conv-active { background: rgba(59, 130, 246, 0.15); }

    /* Logo: plain bold "SiLVANN" wordmark (no styling on the 'i'). */
    .silvann-logo {
        font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: -0.02em;
        display: inline-flex;
        align-items: baseline;
    }

    /* ⭐ THE THINKING AND THE TOOL'S SECTIONS WRAP: their code — the tool's calls and answers, a thought the markdown
       reads as code — goes on to the next line inside its section, where a block that kept its lines widened the
       section past its bubble and was cut off; an answer's own code keeps its lines and scrolls sideways. Prose wraps
       already (Quasar's bubble breaks long words). */
    .silvann-chat .silvann-part pre { white-space: pre-wrap; overflow-wrap: anywhere; overflow-x: visible; }

    /* A code block's copy button (the script below), in the block's top corner, faint until the block is pointed at. */
    .silvann-copy-host { position: relative; }
    .silvann-copy {
        position: absolute; top: 4px; right: 4px; z-index: 1;
        font: 11px ui-sans-serif, system-ui, sans-serif; padding: 1px 6px; border-radius: 4px; cursor: pointer;
        border: 1px solid rgba(0,0,0,0.2); background: rgba(255,255,255,0.85); color: #333;
        opacity: 0.3; transition: opacity 0.15s;
    }
    .silvann-copy-host:hover .silvann-copy, .silvann-copy:focus { opacity: 1; }

    /* ⭐ DARK: Quasar darkens its own components; what this page colours itself follows here — and the chat's bubbles,
       light by Quasar's default, turn dark under the dark colours NiceGUI gives code. */
    .body--dark .silvann-menu-item:hover, .body--dark .silvann-conv-item:hover { background: rgba(255,255,255,0.08); }
    .body--dark .silvann-menu-active { background: rgba(96,165,250,0.18); color: rgb(147,197,253); }
    .body--dark .silvann-conv-active { background: rgba(96,165,250,0.2); }
    .body--dark .q-message-text--received { color: #24302a; }
    .body--dark .q-message-text--sent { color: #30353f; }
    .body--dark .q-message-text-content--received, .body--dark .q-message-text-content--sent { color: #e8e8e8; }
    .body--dark .bg-amber-50 { background: rgba(245,158,11,0.14) !important; }
    .body--dark .bg-slate-100 { background: #1e293b !important; }
    .body--dark .text-slate-800 { color: #e2e8f0 !important; }
    .body--dark .text-amber-700 { color: #fbbf24 !important; }
    .body--dark .text-green-700 { color: #4ade80 !important; }
    .body--dark .text-red-700 { color: #f87171 !important; }
    .body--dark .silvann-copy { background: rgba(30,30,30,0.85); color: #ddd; border-color: rgba(255,255,255,0.25); }
</style>
<script>
/* ⭐ A COPY BUTTON ON EVERY CODE BLOCK OF THE CONVERSATION — put on each block as it is drawn, and drawn again as a
   streaming answer's are; it copies the block's text. The clipboard's own call needs a secure page (https or localhost),
   so a UI reached over the LAN by http copies through a hidden text area. */
(() => {
  const copy = (text) => {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(area);
    area.select();
    try { return document.execCommand('copy') ? Promise.resolve() : Promise.reject(); } finally { area.remove(); }
  };
  const equip = (pre) => {
    const host = pre.parentElement && pre.parentElement.classList.contains('codehilite') ? pre.parentElement : pre;
    if (host.querySelector(':scope > .silvann-copy')) return;
    host.classList.add('silvann-copy-host');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'silvann-copy';
    button.textContent = 'copy';
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      const code = pre.querySelector('code');
      copy((code || pre).textContent).then(() => { button.textContent = 'copied'; }, () => { button.textContent = 'not copied'; });
      setTimeout(() => { button.textContent = 'copy'; }, 1500);
    });
    host.appendChild(button);
  };
  new MutationObserver(() => document.querySelectorAll('.silvann-chat pre').forEach(equip))
    .observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


# ---------- UI ----------

@ui.page("/")
async def page():
    state = app.storage.user.setdefault("silvann", _session_default())
    for k, v in _session_default().items():
        state.setdefault(k, v)
    # Reset transient flags that must NOT survive a page reload. Without this, a session that
    # crashed mid-stream leaves "streaming": True in localStorage forever, which makes the send
    # button forever route to stop().
    state["streaming"] = False
    state["active_bound"] = False
    if LOCAL_APP:
        # the app over this machine's models: its local panel, and the options view until a server is up
        state["server_mode"] = "local"
        if not SERVER.is_running():
            state["view"] = "options"
    # Per-browser-tab UUID. app.storage.tab requires the WS handshake to be done first —
    # await client.connected() before touching it. Survives reloads within the tab, dies on close.
    import uuid as _uuid
    from nicegui import context as _ng_context
    # ⛳ a page whose websocket never connects (a browser reconnecting after a dropped connection) is left quietly —
    #   the browser loads it again once it is back — rather than raising on the tab storage it cannot have
    try:
        await _ng_context.client.connected(timeout=30)
        tab_store = app.storage.tab
    except (RuntimeError, TimeoutError, asyncio.TimeoutError):
        return
    if "web_user_id" not in tab_store:
        tab_store["web_user_id"] = _uuid.uuid4().hex
    web_user_id = tab_store["web_user_id"]
    # Share the in-state passwords dict by reference so client.set_password() and the UI's
    # storage stay in sync — no separate copy to manage.
    client = Client(state["endpoint"],
                    passwords=state.setdefault("passwords", {}),
                    web_user_id=web_user_id)

    # Capture the page-level NiceGUI Client now (while the slot stack is set up) so event-handler
    # tasks — which run with an empty slot stack — can still trigger downloads/notifications by
    # going through this reference instead of through context.client.
    from nicegui import context as _ng_context
    page_client = _ng_context.client

    def safe_notify(msg: str, type: str = "info"):
        """ui.notify needs a slot context that event-handler tasks don't always inherit in
        NiceGUI 3.x. Fall back to a log line if the notify can't render."""
        try:
            ui.notify(msg, type=type)
        except RuntimeError:
            print(f"[ui notify suppressed] {msg}", flush=True)

    # Layout: no top header. Left drawer defaults to mini (icons only); hovering expands it to
    # 280px as an overlay so it doesn't push the main area around. Main area is a flex column
    # holding a thin hamburger bar and three view columns (only one visible at a time).
    ui.add_head_html(PAGE_HEAD)

    # ============================================================
    # LEFT DRAWER (always-on, mini by default, hover-expands to full width)
    # ============================================================
    drawer = ui.left_drawer(value=True
                            ).props("bordered mini mini-to-overlay mini-width=60 width=280 "
                                    "behavior=desktop"
                                    ).classes("flex flex-col")
    # Hover-expand with a 333ms collapse debounce. Brief mouse-outs (e.g. drifting onto a
    # popup q-menu from the 3-dot button) won't flash the drawer shut.
    _collapse = {"task": None}

    async def _collapse_after(delay: float):
        try:
            await asyncio.sleep(delay)
            drawer.props(add="mini")
        except asyncio.CancelledError:
            pass

    def _cancel_pending_collapse():
        t = _collapse["task"]
        if t is not None and not t.done():
            t.cancel()
        _collapse["task"] = None

    def _on_drawer_enter():
        _cancel_pending_collapse()
        drawer.props(remove="mini")

    def _on_drawer_leave():
        _cancel_pending_collapse()
        _collapse["task"] = asyncio.create_task(_collapse_after(0.1))

    drawer.on("mouseenter", _on_drawer_enter)
    drawer.on("mouseleave", _on_drawer_leave)

    with drawer:
        # --- Brand row: plain "SiLVANN" wordmark ---
        with ui.row().classes("items-center w-full px-3 py-3 gap-2"):
            # Mini mode shows just "Si"; the rest ("LVANN") is hidden via .silvann-brand-label.
            ui.html('<span class="silvann-logo">Si<span class="silvann-brand-label">LVANN</span></span>'
                    ).classes("flex-grow")
            status_dot = ui.icon("circle").classes("text-gray-400").style("font-size: 12px")
            with status_dot:
                status_tip = ui.tooltip("connecting…")
        ui.separator()

        # --- Nav rows (Chat / Options / Help) ---
        with ui.column().classes("w-full p-1 gap-0"):
            chat_row = ui.row().classes("silvann-menu-item w-full")
            with chat_row:
                ui.icon("chat")
                ui.label("Chat").classes("flex-grow")
                chat_chevron = ui.icon("expand_more")

            # Conversations sub-tree — populated by render_convs() once the page connects via WS.
            # Show a placeholder synchronously so the drawer isn't empty during the initial paint.
            convs_col = ui.column().classes("pl-3 pr-1 w-full gap-0 silvann-convs-col")
            with convs_col:
                ui.label("(loading…)").classes("opacity-50 text-xs px-2 py-1")

            opts_row = ui.row().classes("silvann-menu-item w-full")
            with opts_row:
                ui.icon("settings")
                ui.label("Options")

            help_row = ui.row().classes("silvann-menu-item w-full")
            with help_row:
                ui.icon("help_outline")
                ui.label("Help")

            theme_row = ui.row().classes("silvann-menu-item w-full")
            with theme_row:
                theme_icon = ui.icon("dark_mode")
                theme_label = ui.label("Dark mode")

        ui.space()
        ui.separator()


    # ============================================================
    # MAIN AREA — three views (only one visible at a time)
    # ============================================================

    # --- Chat view ---
    chat_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-hidden")
    with chat_view:
        # Header row: conversation name on the left, speed counter + two toggles + save button.
        with ui.row().classes("w-full items-center gap-3"):
            chat_header = ui.label("(no conversation active)").classes("text-sm opacity-60 flex-grow")
            # Live tok/s during streaming; final "<n> tok in <s>s · <r> tok/s" after each turn.
            speed_label = ui.label("—").classes("text-xs font-mono opacity-70")
            # the conversation's tokens of the most the model takes: how full it is
            with ui.row().classes("items-center gap-1 no-wrap") as context_box:
                context_bar = ui.linear_progress(value=0, show_value=False, size="6px").props(
                    "rounded color=primary").classes("w-16")
                context_label = ui.label("").classes("text-xs font-mono opacity-70")
            context_box.tooltip("The conversation's tokens, of the most the model takes (its max_context)")
            context_box.set_visibility(False)
            thinking_mode_toggle = ui.switch("thinking mode",
                                              value=state.get("active_thinking_mode", True)
                                              ).props("dense").classes("text-xs")
            show_think_toggle = ui.switch("show thinking",
                                           value=state.get("show_thinking", False)
                                           ).props("dense").classes("text-xs")
            # the model's tool (NN-88): a Lisp REPL on the server's machine, switched as thinking is — declared in a
            # conversation's first turn, so a conversation begun without it cannot turn it on — and what its commands
            # may do unasked, the conversation's (a new one's the server config's tools, whose "no" turns it off)
            tools_toggle = ui.switch("tools", value=False).props("dense").classes("text-xs").tooltip(
                "The model may use a Lisp REPL on the server's machine — list folders, read and write files, run "
                "commands, read web pages. Off: its calls are refused.")
            tools_mode = ui.select({"ask": "ask first", "chill": "reads unasked", "full-auto": "full auto"},
                                   value=None).props("dense options-dense borderless").classes("text-xs").tooltip(
                "What this conversation's tool may do without asking you: 'ask first' reads in its workspace and "
                "asks for anything else; 'reads unasked' reads anywhere and asks to write, run or fetch; 'full auto' "
                "asks nothing.")
            # Lock icon is the password control: click to set / change / remove the password for
            # the active conversation. lock_open = unprotected, lock = protected. Since saves
            # happen automatically after each turn, there's no separate "save" button.
            protected_icon = ui.button(icon="lock_open"
                                       ).props("flat dense round size=sm"
                                               ).classes("opacity-70"
                                                         ).tooltip("Set / change / remove password")

            async def on_thinking_mode(e):
                state["active_thinking_mode"] = bool(e.value)
                if state["active_id"]:
                    try:
                        await client.set_thinking(state["active_id"], bool(e.value))
                    except Exception as ex:
                        safe_notify(f"couldn't update thinking mode: {ex}", type="negative")
            thinking_mode_toggle.on_value_change(on_thinking_mode)

            def on_show_thinking(e):
                state["show_thinking"] = bool(e.value)
                render_chat()
            show_think_toggle.on_value_change(on_show_thinking)

            async def on_tools(e):
                if not state["active_id"] or bool(e.value) == state.get("active_tools", False):
                    return
                try:
                    await client.set_tools(state["active_id"], bool(e.value))
                    state["active_tools"] = bool(e.value)
                except httpx.HTTPStatusError as ex:
                    tools_toggle.set_value(state.get("active_tools", False))
                    safe_notify(ex.response.json().get("detail", str(ex)) if ex.response.status_code == 409
                                else f"couldn't set the tool: {ex}", type="warning")
                except Exception as ex:
                    safe_notify(f"couldn't set the tool: {ex}", type="negative")
            tools_toggle.on_value_change(on_tools)

            async def on_tools_mode(e):
                if not e.value or not state["active_id"] or e.value == state.get("tools_mode"):
                    return
                try:
                    await client.set_tools_mode(state["active_id"], e.value)
                    state["tools_mode"] = e.value
                except Exception as ex:
                    tools_mode.set_value(state.get("tools_mode"))
                    safe_notify(f"couldn't set what the tool may do: {ex}", type="negative")
            tools_mode.on_value_change(on_tools_mode)
        # Banner shown when the active cart is loaded into UI memory but NOT bound to VRAM.
        # User must click Activate to /bind. Hidden once bound; reappears after server eviction.
        with ui.row().classes("w-full items-center gap-2 p-3 rounded bg-amber-50 border border-amber-300"
                              ).bind_visibility_from(state, "active_bound", lambda b: not b
                                                     ) as bind_banner:
            ui.icon("info", size="sm").classes("text-amber-700")
            bind_banner_label = ui.label("Conversation loaded. Type a message, or click Activate, to bring the session online."
                                          ).classes("text-sm flex-grow")
            bind_btn = ui.button("Activate", icon="play_arrow"
                                 ).props("color=primary unelevated")
        chat_scroll = ui.scroll_area().classes("silvann-chat w-full flex-grow min-h-0 border rounded")
        with chat_scroll:
            chat_history_col = ui.column().classes("w-full p-4 gap-3")
            chat_tail_col = ui.column().classes("w-full px-4 pb-4 gap-3")   # below the messages: "…", the model's ask
        # Attached-image chip row. Rendered into by render_pending_images() whenever
        # an image is uploaded; empty (and visually absent) otherwise.
        pending_images_row = ui.row().classes("w-full items-center gap-2 flex-wrap")
        # Shown while an attached image is being uploaded + run through the vision encoder
        # (server-side ViT can take many seconds with no other feedback). Hidden otherwise.
        image_busy_row = ui.row().classes("w-full items-center gap-2")
        with image_busy_row:
            ui.label("Encoding image…").classes("text-sm opacity-80 whitespace-nowrap")
            ui.linear_progress(show_value=False).props(
                "indeterminate rounded color=primary").classes("flex-grow")
        image_busy_row.set_visibility(False)
        with ui.row().classes("w-full items-end gap-2"):
            # The paperclip uses the same `ui.upload` widget the cartridge import uses, but
            # accepting pictures, videos and documents (each read by the server at the same endpoint)
            # and bound to the active cart's /images endpoint.
            # auto_upload fires on file selection so the user doesn't have to click upload twice.
            attach_upload = ui.upload(auto_upload=True, max_files=1,
                                       max_file_size=100 * 1024 * 1024
                                       ).props('label="" accept="image/*,video/*,.pdf,.docx,.txt,.md,.csv,.json,text/*" '
                                               'flat hide-upload-btn color=primary')
            attach_upload.classes("hidden")
            attach_btn = ui.button(icon="attach_file"
                                   ).props("color=primary"
                                           ).tooltip("Attach a document, or a picture or video for a model that reads them, to the next message")
            attach_btn.set_visibility(False)       # shown when the server's model reads pictures or files (its /health)
            input_box = ui.textarea(placeholder="Type a message — Enter to send, Shift+Enter for newline"
                                    ).classes("flex-grow").props("autogrow outlined")
            max_in = ui.number(label="max tokens", value=2048, min=1, max=16384).classes("w-32")
            send_btn = ui.button(icon="send")

        def _cart_pending() -> list:
            """Mutable list of pending image attachments for the active cart."""
            cart_id = state.get("active_id")
            if not cart_id:
                return []
            return state["pending_images"].setdefault(cart_id, [])

        def render_pending_images():
            pending_images_row.clear()
            entries = _cart_pending()
            if not entries:
                pending_images_row.set_visibility(False)
                return
            pending_images_row.set_visibility(True)
            with pending_images_row:
                ui.icon("image", size="sm").classes("opacity-70")
                for idx, ent in enumerate(entries):
                    chip = ui.chip(
                        f"{ent['filename']} · {ent['n_pads']} pads",
                        icon="image", removable=True,
                    ).props("dense color=blue-2 text-color=blue-9")
                    def _on_remove(idx=idx):
                        cur = _cart_pending()
                        if 0 <= idx < len(cur):
                            cur.pop(idx)
                            render_pending_images()
                    chip.on("remove", _on_remove)

        async def on_attach_upload(e):
            cart_id = state.get("active_id")
            if not cart_id or not state.get("active_bound"):
                safe_notify("Activate a conversation first", type="warning")
                return
            # Give the user feedback for the whole round-trip — the upload itself is quick but
            # the server then runs the vision encoder, which has no other progress signal.
            attach_btn.disable()
            image_busy_row.set_visibility(True)
            try:
                # NiceGUI 3.x upload event: e.file is a FileUpload with async read() +
                # name + content_type. (Older NiceGUI used e.content/e.name/e.type.)
                blob = await e.file.read()
                filename = e.file.name or "image.png"
                content_type = e.file.content_type or "application/octet-stream"
                resp = await client.upload_image(cart_id, blob, filename=filename,
                                                  content_type=content_type)
                _cart_pending().append({
                    "image_id": resp["image_id"],
                    "filename": filename,
                    "n_pads": int(resp.get("n_placeholders", 0)),
                    "merged_thw": resp.get("merged_thw", []),
                })
                render_pending_images()
                safe_notify(f"attached {filename} ({resp.get('n_placeholders')} placeholders)",
                            type="positive")
            except httpx.HTTPStatusError as ex:
                safe_notify(f"image upload refused: {ex.response.status_code} "
                            f"{ex.response.text}", type="negative")
            except Exception as ex:
                safe_notify(f"image upload failed: {ex}", type="negative")
            finally:
                image_busy_row.set_visibility(False)
                attach_btn.enable()
                attach_upload.reset()

        attach_upload.on_upload(on_attach_upload)
        attach_btn.on("click", lambda _: attach_upload.run_method("pickFiles"))
        # Hide chips initially (no attachments).
        pending_images_row.set_visibility(False)

    # --- Options view (was "Server" tab) ---
    opts_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-auto")
    with opts_view:
        ui.label("Server control").classes("text-lg font-bold")

        # this machine's models and its server; which server the UI talks to is chosen at the bottom
        with ui.column().classes("w-full gap-2"):
            if not LOCAL_AVAILABLE:
                ui.label(
                    f"Local mode unavailable: this UI process can't find start_silvann_server.py "
                    f"beside it. Run the UI from main/ (the runtime root) to enable local control."
                ).classes("text-orange-600 max-w-2xl")
            else:
                # MODELS — the ones this release can fetch, which are here, and the one downloading
                # collapsed once a model is here: downloading is the first visit's work, starting a server every visit's
                with ui.expansion("Download models", icon="download", value=not list_models()).classes("w-full"):
                    ui.label(f"Downloaded from Hugging Face into {MODELS_DIR}. Help → Running SiLVANN says what "
                             f"each needs; one larger than the card runs its experts on the CPU."
                             ).classes("text-xs opacity-70 max-w-2xl whitespace-normal")

                    # the Hugging Face token: faster downloads than a client that is not signed in, and the repositories
                    #   whose terms its account accepted
                    def token_state():
                        if os.environ.get("HF_TOKEN"):
                            return "Hugging Face token: HF_TOKEN, from the environment the UI was started in"
                        token = saved_token()
                        if token:
                            return f"Hugging Face token: saved ({token[:3]}…{token[-4:]})"
                        return ("Hugging Face token: none — downloads go as a client that is not signed in, which Hugging "
                                "Face serves more slowly. A token with read access: huggingface.co/settings/tokens")

                    with ui.row().classes("items-end gap-2"):
                        token_in = ui.input("Hugging Face token", password=True, password_toggle_button=True
                                            ).props("dense").classes("w-80")
                        token_save = ui.button("save").props("dense outline")
                        token_forget = ui.button("forget").props("dense flat")
                    token_lbl = ui.label(token_state()).classes("text-xs opacity-70 max-w-2xl whitespace-normal")

                    async def save_hf_token():
                        token = (token_in.value or "").strip()
                        if not token:
                            safe_notify("paste a token first", type="warning")
                            return
                        try:
                            who = await asyncio.to_thread(token_owner, token)
                        except ValueError as e:
                            safe_notify(f"not saved: {e}", type="negative")
                            return
                        save_token(token)
                        token_in.value = ""
                        token_lbl.set_text(token_state() + (f" — signed in as {who}" if who else
                                                            " — not checked: Hugging Face did not answer"))
                        safe_notify("token saved", type="positive")

                    def forget_hf_token():
                        save_token("")
                        token_lbl.set_text(token_state())
                        safe_notify("token removed", type="info")
                    token_save.on_click(save_hf_token)
                    token_forget.on_click(forget_hf_token)

                    def _free_bytes():
                        d = MODELS_DIR
                        while not d.exists() and d != d.parent:
                            d = d.parent
                        return shutil.disk_usage(d).free

                    @ui.refreshable
                    def models_table():
                        if DOWNLOADS.known is None:
                            ui.label("reading the list of models…").classes("text-xs opacity-70")
                            return
                        ui.label(f"{_free_bytes() / 1e9:.0f} GB free on this disk").classes("text-xs opacity-70")
                        with ui.grid(columns="auto auto auto").classes("items-center gap-x-6 gap-y-1"):
                            for m in DOWNLOADS.catalogue():
                                ui.label(m["name"]).classes("font-mono text-sm")
                                ui.label(m["size"]).classes("text-sm")
                                if m["here"]:
                                    ui.label("here").classes("text-sm text-green-700")
                                elif DOWNLOADS.running() and DOWNLOADS.name == m["name"]:
                                    ui.label("downloading…").classes("text-sm")
                                else:
                                    ui.button("download", on_click=lambda _, n=m["name"]: start_download(n)
                                              ).props("dense outline")
                    models_table()

                    async def read_catalogue():
                        try:
                            await asyncio.to_thread(DOWNLOADS.catalogue)
                        except Exception as e:
                            safe_notify(f"the list of models could not be read: {e}", type="negative")
                        models_table.refresh()
                    if DOWNLOADS.known is None:
                        asyncio.create_task(read_catalogue())

                    async def start_download(name):
                        try:
                            await asyncio.to_thread(DOWNLOADS.start, name)
                            safe_notify(f"downloading {name}", type="info")
                        except Exception as e:
                            safe_notify(f"not downloaded: {e}", type="negative")
                        models_table.refresh()

                    with ui.row().classes("items-center gap-3 w-full") as dl_row:
                        dl_bar = ui.linear_progress(value=0, show_value=False).classes("w-80")
                        dl_label = ui.label("").classes("text-xs font-mono")
                        dl_stop = ui.button("stop the download", on_click=lambda: asyncio.to_thread(DOWNLOADS.cancel)
                                            ).props("dense flat color=negative")
                    dl_row.set_visibility(False)
                    dl_seen = {"running": DOWNLOADS.running()}

                    def refresh_model_select():
                        opts = list_models()
                        keep = (model_select.value if model_select.value in opts
                                else DOWNLOADS.name if DOWNLOADS.name in opts else (opts[0] if opts else None))
                        model_select.set_options(opts, value=keep)

                    def pump_download():
                        running = DOWNLOADS.running()
                        if DOWNLOADS.name and (running or DOWNLOADS.message):
                            have, total = DOWNLOADS.progress()
                            dl_row.set_visibility(True)
                            dl_bar.set_value(min(1.0, have / total) if total else 0)
                            dl_label.set_text(f"{DOWNLOADS.name}: {have / 1e9:.1f} of {total / 1e9:.1f} GB"
                                              + ("" if running else f" — {DOWNLOADS.message}"))
                            dl_stop.set_visibility(running)
                        if running != dl_seen["running"]:
                            dl_seen["running"] = running
                            models_table.refresh()
                            if not running:
                                refresh_model_select()
                    ui.timer(1.0, pump_download)

                    ui.label(f"Models dir: {MODELS_DIR}").classes("text-xs opacity-70 mt-2")

                def _config_desc(model, name):
                    # Surface the config's "_comment" so users see what each setup does (VRAM,
                    # topology, offload) without opening the JSON.
                    import json as _json
                    if not (model and name):
                        return ""
                    try:
                        return _json.loads((model_configs_dir(model) / name).read_text(encoding="utf-8")).get("_comment", "")
                    except Exception:
                        return ""

                # MODEL selector — the self-contained folders under main/models/ (drives the config path).
                model_options = list_models()
                _init_model = (state["selected_model"] if state["selected_model"] in model_options
                               else (model_options[0] if model_options else None))
                state["selected_model"] = _init_model or ""
                model_select = ui.select(
                    model_options, value=_init_model, label="Model",
                ).classes("w-80")

                # CONFIG selector — scoped to the selected model's configs/ folder.
                _init_cfgs = list_model_configs(_init_model) if _init_model else []
                _init_cfg = (state["selected_config"] if state["selected_config"] in _init_cfgs
                             else ((_init_model and model_default_config(_init_model))
                                   or (_init_cfgs[0] if _init_cfgs else None)))
                state["selected_config"] = _init_cfg or ""
                # ⭐ A CONFIG IS AN OVERRIDE: the boot chooses every setting a config leaves out for this machine, so the
                #   downloaded default.json — every setting "auto" — is the one to run; this section, collapsed, is
                #   for naming one
                cfg_box = ui.expansion("Config: %s" % (_init_cfg or "none"), icon="tune", value=False).classes("w-full")
                with cfg_box:
                    ui.label("Every setting a config leaves out is chosen for this machine at each boot: a config names "
                             "only what it overrides.").classes("text-xs opacity-70 max-w-2xl whitespace-normal")
                    config_select = ui.select(
                        _init_cfgs, value=_init_cfg, label="Config",
                    ).classes("w-80")
                    config_desc_label = ui.label(_config_desc(_init_model, _init_cfg)).classes(
                        "text-xs opacity-70 max-w-2xl whitespace-normal")
                    with ui.row().classes("gap-2 items-end"):
                        ui.button("edit selected",
                                  on_click=lambda: open_config_editor(config_select.value)
                                  ).props("dense outline")
                        ui.button("new config",
                                  on_click=lambda: open_config_editor(None)
                                  ).props("dense outline")
                        ui.button("autotune", icon="tune",
                                  on_click=lambda: forget_autotune()
                                  ).props("dense outline").tooltip("Test the setups again at the next start")
                        ui.button("delete selected",
                                  on_click=lambda: delete_config(config_select.value)
                                  ).props("dense outline color=negative")
                        ui.button("config guide", icon="help_outline",
                                  on_click=lambda: open_config_guide()
                                  ).props("dense flat").tooltip("What every key and every model's option does")

                def _on_config_change(e):
                    state["selected_config"] = e.value
                    config_desc_label.set_text(_config_desc(state["selected_model"], e.value))
                    cfg_box.set_text("Config: %s" % (e.value or "none"))
                config_select.on_value_change(_on_config_change)

                def _on_model_change(e):
                    # Selecting a model RESETS the config selector to that model's configs/ folder.
                    model = e.value
                    state["selected_model"] = model
                    cfgs = list_model_configs(model)
                    new_cfg = model_default_config(model) or (cfgs[0] if cfgs else None)
                    config_select.set_options(cfgs, value=new_cfg)
                    state["selected_config"] = new_cfg or ""
                    config_desc_label.set_text(_config_desc(model, new_cfg))
                    cfg_box.set_text("Config: %s" % (new_cfg or "none"))
                model_select.on_value_change(_on_model_change)
                with ui.row().classes("gap-2 items-end"):
                    port_in = ui.number(label="Port", value=state["server_port"],
                                        min=1, max=65535).classes("w-32")
                status_lbl = ui.label("status: ...").classes("font-mono text-sm")
                # what the server chose for this machine, or why it did not start — said here, not only in the log
                setup_lbl = ui.label("").classes("text-sm whitespace-pre-wrap")
                with ui.row().classes("gap-2"):
                    start_btn = ui.button("start", icon="play_arrow"
                                          ).props("color=positive")
                    stop_btn = ui.button("stop", icon="stop"
                                         ).props("color=negative")
                ui.label("Subprocess log").classes("font-bold mt-2 text-sm")
                # Smaller + softer than a full-height terminal — the page no longer feels half-black.
                log_pane = ui.log(max_lines=500).classes(
                    "w-full h-48 bg-slate-100 text-slate-800 font-mono text-xs p-2 border rounded"
                )

        # ⭐ THE SERVER THE UI TALKS TO: this machine's, at the port above, or another's at an address — a server
        #   elsewhere is started and stopped on its own machine
        ui.separator().classes("mt-4")
        with ui.row().classes("items-center gap-4"):
            ui.label("Send conversations to:").classes("font-bold")
            server_sel = ui.toggle({"local": "this machine", "remote": "remote"}, value=state["server_mode"])
        with ui.row().classes("items-end gap-2") as remote_row:
            endpoint_in = ui.input(label="Address", value=state["endpoint"]).props("dense outlined").classes("w-96")
            connect_btn = ui.button("connect").props("dense outline color=primary")
        remote_row.bind_visibility_from(server_sel, "value", lambda v: v == "remote")
        if not LOCAL_AVAILABLE:                           # no server beside this UI to start: another's only
            server_sel.set_value("remote")
            server_sel.disable()
        server_sel.on_value_change(lambda e: asyncio.create_task(on_server_mode(e.value)))

    # --- Help view: a document picker over the release's documents, the UI's guide first ---
    help_view = ui.column().classes("silvann-main w-full p-6 overflow-auto")
    with help_view:
        with ui.column().classes("max-w-4xl mx-auto gap-3 w-full"):
            help_paths = {"%s · %s" % (g, label): path for g, label, path in HELP_DOCS}
            help_shown = {"path": help_paths[HELP_FIRST]}
            help_doc_select = ui.select(list(help_paths), value=HELP_FIRST, label="Document").classes("w-96")
            help_doc_md = ui.markdown(help_doc(help_paths[HELP_FIRST]))

            def show_help(path: Path):
                help_shown["path"] = path
                help_doc_md.set_content(help_doc(path))

            help_doc_select.on_value_change(lambda e: show_help(help_paths[e.value]))

            def open_config_guide():
                """Help, at the guide to a config's keys and every model's options."""
                help_doc_select.set_value(CONFIG_GUIDE_DOC)
                show_help(help_paths[CONFIG_GUIDE_DOC])
                state["view"] = "help"
                update_view()

            def on_help_link(e):
                """A link between the documents: the one it names, in the picker where it is listed."""
                path = help_link(help_shown["path"], str(e.args))
                if path is None:
                    safe_notify("%s is not in the release" % e.args, type="warning")
                    return
                listed = next((k for k, v in help_paths.items() if v.resolve() == path), None)
                if listed is not None:
                    help_doc_select.set_value(listed)
                else:
                    show_help(path)
            # ⛳ a relative link would take the browser to a page the UI does not serve: it is followed here instead, and
            #   a web page opens in a tab of its own
            help_doc_md.on("click", on_help_link, js_handler="""(e) => {
                const a = e.target.closest('a');
                if (!a) return;
                const href = a.getAttribute('href') || '';
                if (href.startsWith('#')) return;
                e.preventDefault();
                if (/^[a-z][a-z0-9+.-]*:/i.test(href)) { window.open(href, '_blank', 'noopener'); return; }
                emit(href);
            }""")

    # ============================================================
    # Rendering / view-switching helpers (close over everything above)
    # ============================================================
    def update_view():
        v = state["view"]
        expanded = bool(state.get("convs_expanded", True))
        chat_view.set_visibility(v == "chat")
        opts_view.set_visibility(v == "options")
        help_view.set_visibility(v == "help")
        # Sub-tree is shown only when Chat is the active view AND the user hasn't collapsed it.
        convs_col.set_visibility(v == "chat" and expanded)
        chat_chevron.props(
            f"name={'expand_less' if (v == 'chat' and expanded) else 'expand_more'}")
        # Highlight active menu row.
        base = "silvann-menu-item w-full"
        chat_row.classes(replace=base + (" silvann-menu-active" if v == "chat" else ""))
        opts_row.classes(replace=base + (" silvann-menu-active" if v == "options" else ""))
        help_row.classes(replace=base + (" silvann-menu-active" if v == "help" else ""))

    async def on_chat_click():
        # Clicking Chat: if it's already the active view, toggle the sub-tree. Otherwise switch
        # to chat AND ensure the sub-tree is expanded (auto-show on view switch).
        if state["view"] == "chat":
            state["convs_expanded"] = not bool(state.get("convs_expanded", True))
            update_view()
        else:
            state["view"] = "chat"
            state["convs_expanded"] = True
            update_view()
            await render_convs()

    async def select_other_view(v: str):
        state["view"] = v
        update_view()

    chat_row.on("click", lambda: asyncio.create_task(on_chat_click()))
    opts_row.on("click", lambda: asyncio.create_task(select_other_view("options")))
    help_row.on("click", lambda: asyncio.create_task(select_other_view("help")))

    # light or dark, as this browser last chose — the row offers the other
    dark = ui.dark_mode(bool(state.get("dark", False)))

    def show_theme():
        on = bool(state.get("dark", False))
        theme_icon.name = "light_mode" if on else "dark_mode"
        theme_label.text = "Light mode" if on else "Dark mode"

    def on_theme_click():
        state["dark"] = not bool(state.get("dark", False))
        dark.set_value(state["dark"])
        show_theme()

    show_theme()
    theme_row.on("click", on_theme_click)

    async def ping():
        try:
            h = await client.health()
            status_dot.classes(replace="text-green-500").style("font-size: 12px")
            status_tip.text = (f"OK — {h['model_id']} · layers {h['pp_slice']} · "
                               f"gpus {h['num_gpus']} · tenants {h['tenants_loaded']}"
                               + (f" · active {h['active'][:8]}" if h['active'] else ""))
            attach_btn.set_visibility(bool(h.get("images") or h.get("files")))
            # a picture model makes a picture of each message: it has no thinking and no tool to switch
            for switch in (thinking_mode_toggle, show_think_toggle, tools_toggle, tools_mode):
                switch.set_visibility(not h.get("pictures"))
            show_context(context=0 if h.get("pictures") else h.get("max_context", 0))
        except Exception as e:
            status_dot.classes(replace="text-red-500").style("font-size: 12px")
            status_tip.text = f"down: {type(e).__name__}: {e}"
            attach_btn.set_visibility(False)
            return
        if await model_changed(h.get("model_id")):
            close_chat()
            with page_client:                              # the timer's task has no slot of its own to notify in
                safe_notify(f"{h.get('model_id')} is serving now: its conversations are in the menu")
            await render_convs()
        await first_conversation(h.get("model_id"))

    async def model_changed(model: Optional[str]) -> bool:
        """Whether the server serves another model than the open conversation's. Not while an answer runs, here or on
        the server: a subagent lent the card shows its own model until the answer that called it is done."""
        if not state.get("active_id") or state["streaming"]:
            return False
        try:
            if state.get("active_model"):
                if model == state["active_model"]:
                    return False
            elif state["active_id"] in [c["id"] for c in await client.list_cartridges()]:
                state["active_model"] = model              # a conversation opened before its model was kept
                return False
            b = await client.busy()
            if b.get("busy") or b.get("cart"):
                return False
            if state.get("active_model"):
                return (await client.health()).get("model_id") != state["active_model"]
            return state["active_id"] not in [c["id"] for c in await client.list_cartridges()]
        except Exception:
            return False

    def close_chat():
        """The chat emptied of a conversation another model made: the menu lists the serving model's."""
        state["active_id"] = state["active_name"] = state["active_model"] = None
        state["history"] = []
        state["active_bound"] = False
        state["context_used"] = 0
        speed_label.text = "—"                             # the last model's rate
        render_chat()

    async def connect_clicked():
        state["endpoint"] = endpoint_in.value
        client.set_endpoint(endpoint_in.value)
        await ping()
        await render_convs()

    connect_btn.on("click", connect_clicked)

    async def on_server_mode(mode):
        """The server chosen at the bottom of Options: this machine's, at its port, or the address given."""
        state["server_mode"] = mode
        if mode == "local":
            endpoint_in.value = f"http://localhost:{int(state['server_port'])}"
        await connect_clicked()

    # ----- Chat helpers -----
    def show_context(used=None, context=None):
        """The top bar's meter: the open conversation's tokens of the context, from the server's numbers — a
        conversation's when it opens, each token's as the answer streams, the answer's end."""
        if context is not None:
            state["context"] = int(context or 0)
        if used is not None:
            state["context_used"] = int(used)
        total, used = state.get("context") or 0, state.get("context_used") or 0
        context_box.set_visibility(bool(total and state.get("active_id")))
        if not total:
            return
        share = min(1.0, used / total)
        context_bar.value = share
        context_bar.props("color=" + ("negative" if share >= 0.9 else "warning" if share >= 0.75 else "primary"))
        context_label.text = f"{used:,} / {total:,}"

    pictures: dict = {}                                # (conversation, picture) -> its data URI, or None while it loads

    def picture_markdown(alt: str, name: str) -> str:
        """A picture in an answer: drawn once the server's copy is here, fetched the first time it is wanted."""
        key = (state["active_id"], name)
        if key not in pictures:
            pictures[key] = None
            asyncio.create_task(load_picture(*key))
        uri = pictures[key]
        return "![%s](%s)" % (alt, uri) if uri else "_(the picture is loading…)_"

    async def load_picture(cid: str, name: str):
        try:
            data = await client.picture(cid, name)
            pictures[(cid, name)] = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        except Exception as ex:                         # noqa: BLE001 — said where the picture would be
            pictures[(cid, name)] = None
            safe_notify(f"the picture {name} could not be read: {ex}", type="warning")
            return
        if state["active_id"] != cid:
            return
        last = len(state["history"]) - 1
        for index, m in enumerate(state["history"]):     # only the answers that name it drawn again
            if index in answers and m["role"] == "assistant" and "(picture:%s)" % name in (m["content"] or ""):
                refresh_answer(index, bool(state["streaming"]) and index == last)

    # ⭐ AN ANSWER IS DRAWN AS ITS PARTS (`answer_parts`): its text; each run of tool calls a section that opens and closes,
    #   closed at first; its thinking — with "show thinking" on a section of its own, open at first, and off nothing once
    #   the block is closed and "thinking…" while the model is still in it. While an answer streams only its own message
    #   is brought up to date, a part at a time and at most ten times a second: drawing the whole conversation again for
    #   every token is what made the chat flicker.
    opened: dict = {}                                  # (message, part) -> whether its section is open
    answers: dict = {}                                 # an answer's index -> its message and its drawn parts
    live_answer: dict = {}                             # the streaming answer's index, and whether its text has grown

    def shown_parts(content: str, live: bool) -> list:
        """What an answer shows, part by part: ("text", markdown), ("tools", title, markdown), ("thought", title,
        markdown), ("thinking", label)."""
        show_think = bool(state.get("show_thinking", False))
        out = []
        for part in answer_parts(content):
            if part[0] == "text":
                out.append(("text", PICTURE_RE.sub(lambda p: picture_markdown(p.group(1), p.group(2)), part[1])))
            elif part[0] == "tools":
                if out and out[-1][0] == "tools":       # runs apart only by thinking not shown: one section
                    out[-1] = ("tools", out[-1][1] + part[1])
                else:
                    out.append(("tools", list(part[1])))
            elif show_think:
                out.append(("thought", "thinking" if part[2] else "thinking…", part[1].strip() or "…"))
            elif not part[2]:
                out.append(("thinking", "thinking…" if live else "(stopped while thinking)"))
        if live and not out:
            out.append(("thinking", "…"))
        for i, part in enumerate(out):
            if part[0] == "tools":
                n = sum(1 for line in part[1] if line.startswith("> "))
                out[i] = ("tools", "tool · %d call%s" % (n, "" if n == 1 else "s"), tools_block(part[1]))
        return out

    def draw_part(part, key) -> dict:
        """One part's element in the current slot, and what a later update of it changes."""
        kind = part[0]
        if kind == "text":
            md = ui.markdown(part[1])
            return {"kind": kind, "el": md, "md": md}
        if kind == "thinking":
            label = ui.label(part[1]).classes("text-sm italic opacity-60 animate-pulse")
            return {"kind": kind, "el": label, "md": label}
        section = ui.expansion(part[1], icon="build" if kind == "tools" else "psychology",
                               value=opened.get(key, kind == "thought"),
                               on_value_change=lambda e, k=key: opened.__setitem__(k, bool(e.value))
                               ).props("dense").classes("w-full silvann-part")
        with section:
            md = ui.markdown(part[2]).classes("text-sm" + (" opacity-70" if kind == "thought" else ""))
        return {"kind": kind, "el": section, "md": md}

    def update_part(drawn: dict, part):
        if drawn["kind"] == "thinking":
            drawn["md"].text = part[1]
            return
        if drawn["kind"] != "text":
            drawn["el"].text = part[1]
        content = part[1] if drawn["kind"] == "text" else part[2]
        if drawn["md"].content != content:
            drawn["md"].set_content(content)

    def draw_answer(content: str, index: int, live: bool) -> list:
        drawn = [draw_part(p, (index, i)) for i, p in enumerate(shown_parts(content, live))]
        if not drawn:
            drawn.append({"kind": "empty", "el": ui.markdown("_(empty)_"), "md": None})
        return drawn

    def refresh_answer(index: int, live: bool):
        """An answer's message brought up to its text: each part's content changed in place, new parts added after it;
        from the first part whose kind changed (a think block closing), the rest drawn again."""
        msg, drawn = answers[index]["slot"], answers[index]["drawn"]
        parts = shown_parts(state["history"][index]["content"] or "", live)
        keep = 0
        while keep < len(drawn) and keep < len(parts) and drawn[keep]["kind"] == parts[keep][0]:
            update_part(drawn[keep], parts[keep])
            keep += 1
        for d in drawn[keep:]:
            d["el"].delete()
        del drawn[keep:]
        with msg:
            for i in range(keep, len(parts)):
                drawn.append(draw_part(parts[i], (index, i)))
            if not drawn and not live:
                drawn.append({"kind": "empty", "el": ui.markdown("_(empty)_"), "md": None})
        chat_scroll.scroll_to(percent=1.0)

    def refresh_live(live: bool = True):
        """The streaming answer brought up to its text; `live` False settles it as the answer it ended as."""
        live_answer["dirty"] = False
        index = live_answer.get("index")
        if index is None or index not in answers or index != len(state["history"]) - 1:
            render_chat()
            return
        refresh_answer(index, live)

    def mark_live():
        """The streaming answer's text has grown: drawn by the next tick of the live timer."""
        live_answer["dirty"] = True

    def finish_live():
        """A turn over: its answer settled in place and the tail drawn — nothing above it drawn again."""
        refresh_live(live=False)
        live_answer.clear()
        render_tail()

    def draw_message(index: int, m: dict):
        """One message of the conversation into the current slot; a streaming answer's is kept for its updates."""
        _sent = (m["role"] == "user")
        # sent=user gives the bubble its style; self-end/self-start does the actual L/R
        # alignment (the column's align-items:flex-start otherwise keeps both on the left),
        # capped at 85% width so the bubbles don't span the whole row.
        with ui.chat_message(name=m["role"], sent=_sent
                             ).classes("self-end" if _sent else "self-start"
                             ).style("max-width: 85%") as msg:
            if _sent:
                ui.markdown(m["content"] or "_(empty)_")
            else:
                live = bool(state["streaming"]) and index == len(state["history"]) - 1
                answers[index] = {"slot": msg, "drawn": draw_answer(m["content"] or "", index, live)}
                if live:
                    live_answer["index"] = index

    def render_tail():
        """Below the messages: "…" while an answer streams, else the model's ask waiting on the user's answer."""
        chat_tail_col.clear()
        with chat_tail_col:
            if state["streaming"]:
                ui.label("…").classes("opacity-50 self-center animate-pulse")
            elif state.get("ask"):                       # the model's command waits on the user
                with ui.card().classes("self-start bg-amber-50 border border-amber-300"):
                    ui.label("The model asks to %s" % state["ask"]["question"]).classes("text-sm font-mono")
                    with ui.row().classes("gap-2"):
                        for choice in state["ask"]["choices"]:
                            ui.button(ASK_LABELS.get(choice, choice),
                                      on_click=lambda _, c=choice: answer_ask(c)
                                      ).props("dense " + ("flat color=negative" if choice == "refuse" else "color=primary"))
        chat_scroll.scroll_to(percent=1.0)

    def add_messages(start: int):
        """The conversation's messages from `start` on added below the others — a turn's, as it is sent."""
        if empty_note.get("el") is not None:
            empty_note.pop("el").delete()
        with chat_history_col:
            for index in range(start, len(state["history"])):
                draw_message(index, state["history"][index])
        render_tail()

    empty_note: dict = {}                              # the "(empty — say hi)" line while the conversation has none

    def render_chat():
        chat_history_col.clear()
        live_answer.clear()
        answers.clear()
        empty_note.clear()
        if state["active_id"]:
            chat_header.text = state.get("active_name") or state["active_id"]
            chat_header.classes(replace="text-lg font-semibold flex-grow")
        else:
            chat_header.text = "(no conversation active — pick one from the menu or click '+ new')"
            chat_header.classes(replace="text-sm opacity-60 flex-grow")
        show_context()
        with chat_history_col:
            if not state["history"] and state["active_id"]:
                empty_note["el"] = ui.label("(empty — say hi)").classes("opacity-50 text-sm self-center")
            for index, m in enumerate(state["history"]):
                draw_message(index, m)
        render_tail()

    async def activate_cart(cart_id: str):
        """Lazy activate — fetches metadata from the server (which is cheap, just reads the
        sidecar JSON) and renders the chat history. Does NOT load VRAM. The user explicitly
        clicks the 'Activate' banner to /bind."""
        is_protected = False
        try:
            for c in (state.get("_last_carts") or []):
                if c["id"] == cart_id:
                    is_protected = bool(c.get("protected"))
                    break
        except Exception:
            pass
        if is_protected:
            pw = await prompt_password(cart_id)
            if pw is None:
                return
            client.set_password(cart_id, pw)
        # Unbind any previously-bound cart for this session — the server only counts ownership
        # on bind, but we keep the UI's `active_bound` consistent.
        prev = state.get("active_id")
        if prev and prev != cart_id and state.get("active_bound"):
            try:
                await client.unbind(prev)
            except Exception:
                pass
            state["active_bound"] = False
        try:
            try:
                resp = await client.activate(cart_id)
            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 401:
                    pw = await prompt_password(cart_id)
                    if pw is None:
                        return
                    client.set_password(cart_id, pw)
                    resp = await client.activate(cart_id)
                else:
                    raise
            state["active_id"] = cart_id
            state["active_name"] = resp.get("name") or cart_id[:8]
            state["history"] = resp.get("history", [])
            show_context(used=resp.get("used", resp.get("next_pos", 0)), context=resp.get("context"))
            new_thinking = bool(resp.get("enable_thinking", True))
            state["active_thinking_mode"] = new_thinking
            thinking_mode_toggle.set_value(new_thinking)
            state["active_tools"] = bool(resp.get("tools", False))
            tools_toggle.set_value(state["active_tools"])
            state["ask"] = resp.get("ask")
            state["tools_mode"] = resp.get("tools_mode")
            tools_mode.set_value(state["tools_mode"])
            try:
                off = (await client.tools_settings())["tools"] == "no"    # the server's config turns it off
                tools_toggle.set_enabled(not off)
                tools_mode.set_enabled(not off)
            except Exception:
                pass                                       # a server before the tool
            state["protected_active"] = bool(resp.get("protected", False))
            protected_icon.props(f"icon={'lock' if state['protected_active'] else 'lock_open'}")
            # bound_by_self happens if the cart is already bound to OUR session (e.g., we
            # navigated away and back within the heartbeat window). Treat that as already-bound.
            state["active_bound"] = bool(resp.get("bound_by_self", False))
            if resp.get("bound_by_other"):
                bind_banner_label.text = "Conversation is held by another session. Click Activate to take it over (will fail if they're still pinging)."
            else:
                bind_banner_label.text = "Conversation loaded. Type a message, or click Activate, to bring the session online (~3s, allocates VRAM)."
            render_chat()
            await render_convs()
            # the model that made it, as the server's list says — the chat is emptied when another serves (ping)
            state["active_model"] = next((c.get("model_id") for c in state.get("_last_carts") or []
                                          if c["id"] == cart_id), None)
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 401:
                safe_notify("wrong password", type="negative")
                client.set_password(cart_id, None)
            else:
                safe_notify(f"activate failed: {ex}", type="negative")
        except Exception as e:
            safe_notify(f"activate failed: {e}", type="negative")

    starting: dict = {"busy": False, "server": None}   # a first conversation in the making; the server it was looked for on

    async def start_conversation(carts: Optional[list] = None) -> bool:
        """A conversation made and opened without the dialog, named as '+ new' names it — the model's first when it has
        none saved, or the one a message typed with none open goes to. True once it is the active one."""
        if starting["busy"]:
            return False
        starting["busy"] = True
        try:
            carts = await client.list_cartridges() if carts is None else carts
            resp = await client.create(name=f"chat {len(carts) + 1}")
            await activate_cart(resp["id"])
            return state["active_id"] == resp["id"]
        except Exception as ex:
            safe_notify(f"couldn't start a conversation: {ex}", type="negative")
            return False
        finally:
            starting["busy"] = False

    async def first_conversation(model: Optional[str]):
        """⭐ A MODEL WITH NO CONVERSATION SAVED GETS ONE, OPEN, so a user's first instinct — to type — works: looked for
        once a server answers (a model started here, the container's, one already up), again when the model changes."""
        here = (state["endpoint"], model)
        if state["active_id"] or state["streaming"] or starting["busy"] or starting["server"] == here:
            return
        try:
            carts = await client.list_cartridges()
        except Exception:
            return
        starting["server"] = here
        if not carts:
            await start_conversation(carts)

    async def bind_active():
        """Bring the active cartridge online — load it into VRAM, claim ownership. The
        'Activate' banner button calls this."""
        cart_id = state.get("active_id")
        if not cart_id:
            safe_notify("no conversation selected", type="warning")
            return
        bind_btn.disable()
        try:
            resp = await client.bind(cart_id)
            state["active_bound"] = True
            state["active_name"] = resp.get("name") or state["active_name"]
            safe_notify("session online", type="positive")
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 423:
                safe_notify("In use by another session — wait or try another conversation",
                            type="negative")
            elif ex.response.status_code == 507:
                safe_notify("Out of VRAM — close another conversation first (or wait 2 min for idle eviction)",
                            type="negative")
            elif ex.response.status_code == 401:
                safe_notify("wrong password — re-open the conversation to re-prompt",
                            type="negative")
            else:
                safe_notify(f"activate failed: {ex.response.status_code} {ex.response.text}",
                            type="negative")
        except Exception as e:
            safe_notify(f"activate failed: {e}", type="negative")
        finally:
            bind_btn.enable()

    bind_btn.on("click", bind_active)

    def toggle_conv_menu(cart_id: str):
        # Independent of activation. Toggle which conv's inline drawer is open (one at a time).
        if state.get("expanded_cart_id") == cart_id:
            state["expanded_cart_id"] = None
        else:
            state["expanded_cart_id"] = cart_id
        asyncio.create_task(render_convs())

    async def export_chat_history(cart_id: str, name: str):
        """Build a markdown transcript from the live state (if this is the active conv) or fetch
        history via activate. Downloads as <name>.md."""
        history = state["history"] if state["active_id"] == cart_id else []
        if not history:
            try:
                resp = await client.activate(cart_id)
                state["active_id"] = cart_id
                state["active_name"] = resp.get("name") or cart_id[:8]
                state["history"] = resp.get("history", [])
                history = state["history"]
                render_chat()
                await render_convs()
            except Exception as e:
                safe_notify(f"couldn't load chat: {e}", type="negative")
                return
        lines = [f"# {name}\n"]
        for m in history:
            lines.append(f"## {m['role']}\n")
            lines.append(m["content"] or "")
            lines.append("")
        text = "\n".join(lines).encode("utf-8")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name or cart_id)
        # Bypass context.client (which needs a slot stack that event-handler tasks lack) — call
        # the captured page_client directly.
        page_client.download(text, f"{safe}.md", "text/markdown")

    async def create_cart():
        # Open a dialog with the conversation's name, the model's context window and what the conversation takes in RAM
        # once put away — the server reports both via /estimate.
        try:
            existing = await client.list_cartridges()
        except Exception as e:
            safe_notify(f"couldn't reach server: {e}", type="negative")
            return
        try:
            est0 = await client.estimate_vram()
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code in (404, 405):
                # Older server without the estimate endpoint — degrade to a plain create so a
                # new UI still works against a server that predates the cache-size dialog.
                try:
                    resp = await client.create(name=f"chat {len(existing) + 1}")
                    await activate_cart(resp["id"])
                except Exception as e:
                    safe_notify(f"create failed: {e}", type="negative")
                return
            safe_notify(f"estimate failed: {ex.response.text}", type="negative")
            return
        except Exception as e:
            safe_notify(f"couldn't reach server: {e}", type="negative")
            return
        # ⭐ THE CONTEXT IS THE MODEL'S, set in its config when the server started — a conversation is created at that
        #   size and costs the card nothing more (its cache is allocated once, at boot, and shared). What the dialog
        #   shows is that window and what the conversation takes in RAM once it is put away, at its fullest.
        max_tokens = int(est0.get("max_tokens", 0))
        gb = float(est0.get("gb", 0.0))

        dlg = ui.dialog()
        with dlg, ui.card().classes("min-w-[420px] gap-2"):
            ui.label("New conversation").classes("text-lg font-bold")
            name_in = ui.input("Name", value=f"chat {len(existing) + 1}").classes("w-full")
            if max_tokens:
                ui.label(f"Context window: {max_tokens:,} tokens — the model's, set in its config").classes("text-sm")
                ui.label(f"Saved, it takes up to {gb:.2f} GB of RAM").classes("text-xs opacity-70")
                cards = int(est0.get("num_gpus", 1) or 1)
                if cards > 1 and est0.get("per_card"):
                    ui.label(f"Over {cards} cards: {float(est0['per_card']['gb']):.2f} GB of its cache on each"
                             ).classes("text-xs opacity-70")

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                create_btn = ui.button("Create").props("color=primary unelevated")

            async def do_create():
                create_btn.disable()
                try:
                    resp = await client.create(name=name_in.value or None)
                    dlg.close()
                    await activate_cart(resp["id"])
                except httpx.HTTPStatusError as ex:
                    safe_notify(f"create failed: {ex.response.text}", type="negative")
                    create_btn.enable()
                except Exception as ex:
                    safe_notify(f"create failed: {ex}", type="negative")
                    create_btn.enable()

            create_btn.on_click(do_create)

        dlg.open()

    async def rename_cart(cart_id: str, new_name: str):
        try:
            await client.rename(cart_id, new_name)
            if state["active_id"] == cart_id:
                state["active_name"] = new_name
                render_chat()
            await render_convs()
        except Exception as e:
            safe_notify(f"rename failed: {e}", type="negative")

    async def delete_cart(cart_id: str):
        try:
            await client.delete(cart_id)
            if state["active_id"] == cart_id:
                state["active_id"] = None
                state["history"] = []
                render_chat()
            await render_convs()
        except Exception as e:
            safe_notify(f"delete failed: {e}", type="negative")

    async def export_cart(cart_id: str):
        # ⛳ THROUGH AN HTTP ROUTE, NOT THE WEBSOCKET: an archive can be a few hundred MB (a conversation's recurrent
        #   state and cache), and sent as bytes over the page's websocket it drops the connection
        try:
            page_client.download(client.export_ticket(cart_id), f"{cart_id}.silvann.tar", "application/x-tar")
        except Exception as e:
            safe_notify(f"export failed: {e}", type="negative")

    async def on_upload(e):
        try:
            # NiceGUI 3.x upload event shape (see on_attach_upload note).
            blob = await e.file.read()
            resp = await client.import_bytes(blob, filename=e.file.name or "cart.tar")
            safe_notify(f"imported {resp['id']}", type="positive")
            await render_convs()
        except httpx.HTTPStatusError as ex:
            safe_notify(f"import refused: {ex.response.text}", type="negative")
        except Exception as ex:
            safe_notify(f"import failed: {ex}", type="negative")

    async def prompt_password(cart_id: str) -> Optional[str]:
        """Modal dialog asking for the password to unlock a protected cartridge. Returns the
        entered string (may be empty) or None if cancelled.

        Wraps the dialog construction in `with page_client:` so element creation works even when
        this is called from an async event-handler task (which doesn't auto-inherit the page's
        slot stack)."""
        result: dict = {"value": None}
        with page_client:
            dlg = ui.dialog()
            with dlg, ui.card():
                ui.label(f"Enter password for {cart_id}").classes("font-bold")
                inp = ui.input(password=True, password_toggle_button=True).classes("w-80")
                with ui.row().classes("justify-end w-full"):
                    ui.button("cancel", on_click=dlg.close).props("flat")
                    def ok():
                        result["value"] = inp.value
                        dlg.close()
                    ui.button("unlock", on_click=ok).props("color=primary")
        dlg.open()
        await dlg
        return result["value"]

    def open_password_dialog():
        """Opened by clicking the lock icon. Set a new password (when unprotected), change it
        (when protected, enter new+confirm), or remove protection. Saves happen automatically
        after every turn so there's no separate save action here."""
        if not state["active_id"]:
            safe_notify("activate a conversation first", type="warning")
            return
        cart_id = state["active_id"]
        is_protected = bool(state.get("protected_active", False))
        with page_client, ui.dialog() as dlg, ui.card().classes("w-96"):
            if is_protected:
                ui.label("Change password").classes("font-bold")
                ui.label("This conversation is locked. Enter a new password to change it, or click "
                         "Remove to make it public.").classes("text-xs opacity-70")
            else:
                ui.label("Set a password").classes("font-bold")
                ui.label("Lock this conversation so other users on the same server can't read it. "
                         "You'll be prompted for the password whenever you open it from a fresh "
                         "tab/device.").classes("text-xs opacity-70")
            pw1 = ui.input("new password", password=True, password_toggle_button=True
                           ).classes("w-full")
            pw2 = ui.input("confirm", password=True, password_toggle_button=True
                           ).classes("w-full")
            err = ui.label("").classes("text-red-500 text-sm")

            async def apply(password: Optional[str]):
                try:
                    resp = await client.save(cart_id, password=password)
                except httpx.HTTPStatusError as ex:
                    err.text = f"server refused: {ex.response.status_code} {ex.response.text}"
                    return
                except Exception as ex:
                    err.text = f"failed: {ex}"
                    return
                if password:
                    client.set_password(cart_id, password)
                else:
                    client.set_password(cart_id, None)
                state["protected_active"] = bool(resp.get("protected", False))
                protected_icon.props(f"icon={'lock' if state['protected_active'] else 'lock_open'}")
                safe_notify(
                    "password set" if state["protected_active"] else "password removed",
                    type="positive")
                dlg.close()
                await render_convs()

            async def submit_set():
                if not pw1.value:
                    err.text = "password cannot be empty"
                    return
                if pw1.value != pw2.value:
                    err.text = "passwords don't match"
                    return
                await apply(pw1.value)

            async def submit_remove():
                await apply("")

            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                if is_protected:
                    ui.button("remove", on_click=submit_remove).props("flat color=red")
                ui.button(("change" if is_protected else "set"),
                          on_click=submit_set).props("color=primary")
        dlg.open()

    protected_icon.on("click", open_password_dialog)

    def open_rename_dialog(cart_id: str, cur: str):
        with page_client, ui.dialog() as dlg, ui.card():
            ui.label(f"Rename {cart_id}").classes("font-bold")
            inp = ui.input(value=cur).classes("w-80")
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                async def ok():
                    await rename_cart(cart_id, inp.value)
                    dlg.close()
                ui.button("rename", on_click=ok)
        dlg.open()

    def open_delete_dialog(cart_id: str):
        with page_client, ui.dialog() as dlg, ui.card():
            ui.label(f"Delete {cart_id}?").classes("font-bold")
            ui.label("This is irreversible — the cartridge dir is removed from disk."
                     ).classes("text-sm opacity-70")
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                async def ok():
                    await delete_cart(cart_id)
                    dlg.close()
                ui.button("delete", on_click=ok).props("color=red")
        dlg.open()

    # ----- Conversations sub-tree under the Chat menu -----
    async def render_convs():
        # Wrap the whole thing in page_client so element creation works regardless of which task
        # we're called from (event handlers don't auto-inherit the slot stack).
        with page_client:
            convs_col.clear()
            with convs_col:
                try:
                    carts = await client.list_cartridges()
                except Exception as e:
                    ui.label(f"server down: {e}").classes("text-red-500 text-xs px-2")
                    # Still show controls so user can create/import once server is up.
                    with ui.row().classes("w-full gap-1 px-2 py-1"):
                        ui.button("+ new", on_click=create_cart
                                  ).props("size=sm dense flat color=primary")
                    return
                # Stash for activate_cart's "is this cart protected?" lookup.
                state["_last_carts"] = carts
                if carts:
                    for c in carts:
                        active = (c["id"] == state["active_id"])
                        expanded = (c["id"] == state.get("expanded_cart_id"))
                        with ui.column().classes(
                            "w-full silvann-conv-item gap-0"
                            + (" silvann-conv-active" if active else "")
                        ):
                            with ui.row().classes("w-full items-center gap-1 no-wrap min-w-0"):
                                with ui.row().classes("flex-grow items-center gap-1 cursor-pointer min-w-0"
                                                       ).on("click",
                                                            lambda cid=c["id"]: asyncio.create_task(activate_cart(cid))):
                                    if c.get("protected"):
                                        ui.icon("lock", size="xs").classes("text-slate-500")
                                    elif c["loaded"]:
                                        ui.icon("memory", size="xs").classes("text-green-500")
                                    else:
                                        ui.icon("chat_bubble_outline", size="xs").classes("opacity-50")
                                    ui.label(c["name"] or c["id"][:8]).classes(
                                        "text-sm flex-grow truncate"
                                    ).style("min-width: 0;")
                                    if not c["compatible"]:
                                        ui.icon("warning", size="xs").classes("text-orange-500")
                                ui.button(icon="expand_less" if expanded else "expand_more",
                                          on_click=lambda cid=c["id"]: toggle_conv_menu(cid)
                                          ).props("flat dense round size=xs").classes("opacity-70")
                            if expanded:
                                with ui.column().classes("w-full gap-0 pl-5 py-1"):
                                    ui.button("Rename", icon="edit",
                                              on_click=lambda cid=c["id"], n=c["name"]:
                                                  open_rename_dialog(cid, n)
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Export full session", icon="archive",
                                              on_click=lambda cid=c["id"]:
                                                  asyncio.create_task(export_cart(cid))
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Export chat history (.md)", icon="description",
                                              on_click=lambda cid=c["id"], n=c["name"]:
                                                  asyncio.create_task(export_chat_history(cid, n))
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Delete", icon="delete",
                                              on_click=lambda cid=c["id"]:
                                                  open_delete_dialog(cid)
                                              ).props("flat dense size=sm align=left color=red").classes("w-full justify-start")
                else:
                    ui.label("(no conversations yet)").classes("opacity-50 text-xs px-2 py-1")
                ui.separator().classes("my-1")
                with ui.column().classes("w-full gap-1 px-1 py-1"):
                    ui.button("+ new", on_click=create_cart
                              ).props("color=primary unelevated").classes("w-full")
                    ui.upload(on_upload=on_upload, auto_upload=True, max_files=1
                              ).props('label="import" flat').classes("w-full")

    # ----- Chat input wiring -----
    def set_send_btn(streaming: bool):
        if streaming:
            send_btn.props("icon=stop color=negative")
        else:
            send_btn.props("icon=send color=primary")

    set_send_btn(False)

    async def send():
        if state["streaming"]:
            return
        txt = (input_box.value or "").strip()
        attached = list(_cart_pending()) if state["active_id"] else []
        if not txt and not attached:
            return
        # a message typed with no conversation open goes to a new one, and one not online is brought online: typing is
        # the user's say-so, as the Activate button is
        if state["active_id"] is None and not await start_conversation():
            return
        if not state.get("active_bound"):
            await bind_active()
            if not state.get("active_bound"):
                return
        if state.get("ask"):
            safe_notify("The model waits on your answer to its ask — the buttons under its answer", type="warning")
            return
        # Explicit set_value pushes the empty string back to the client. Plain assignment doesn't
        # always trigger the value-update event for q-input/textarea, hence the explicit call.
        input_box.set_value("")
        state["streaming"] = True
        set_send_btn(True)
        image_ids = [a["image_id"] for a in attached]
        display_user = txt
        if attached:
            display_user = "📎 " + ", ".join(a["filename"] for a in attached) + (
                f"\n\n{txt}" if txt else "")
        state["history"].append({"role": "user", "content": display_user})
        state["history"].append({"role": "assistant", "content": ""})
        add_messages(len(state["history"]) - 2)
        # The engine serializes turns, so if another conversation is mid-turn the server holds our
        # request until it frees up. Surface that as "waiting my turn" rather than a dead UI; the
        # first received event clears the placeholder. Cheap single poll — no busy-loop.
        waiting = False
        try:
            b = await client.busy()
            other = b.get("cart")
            if b.get("busy") and other and other != state["active_id"]:
                state["history"][-1]["content"] = "_⏳ waiting my turn…_"
                waiting = True
                mark_live()
        except Exception:
            pass
        # Speed counter: walltime starts at the *first* received token to exclude server prefill
        # time (which can dominate on long conversations / offload mode) from the rate.
        first_token_at = None
        n_tokens = 0
        speed_label.text = "queued…" if waiting else "…"
        try:
            async for kind, payload in client.stream_message(
                    state["active_id"], txt, int(max_in.value or 2048),
                    images=image_ids):
                if kind == "waiting":                      # the card is serving another turn; asked again in 5 s
                    if not waiting:
                        waiting = True
                        state["history"][-1]["content"] = "_⏳ waiting my turn…_"
                        speed_label.text = "queued…"
                        mark_live()
                    continue
                if waiting:
                    waiting = False
                    state["history"][-1]["content"] = ""   # drop the "waiting my turn" placeholder
                if kind == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    n_tokens += 1
                    state["history"][-1]["content"] += payload.get("text", "")
                    if "used" in payload:
                        show_context(used=payload["used"])
                    elapsed = time.monotonic() - first_token_at
                    if elapsed > 0.3 and n_tokens > 1:
                        rate = (n_tokens - 1) / elapsed   # exclude the first token from the divisor
                        speed_label.text = f"{rate:.1f} tok/s"
                    mark_live()
                elif not on_tool_event(kind, payload):
                    break
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 409 and (state.get("ask") or "waits on an answer" in ex.response.text):
                # the model's ask is still open — a page that was put away and back may not have known it: its buttons
                # again, and the typed message kept for after the answer
                state["history"].pop()  # the empty assistant message
                state["history"].pop()  # the user message they typed
                input_box.set_value(txt)
                try:
                    state["ask"] = (await client.activate(state["active_id"])).get("ask") or state.get("ask")
                except Exception:
                    pass
                safe_notify("The model waits on your answer to its ask — the buttons under its answer", type="warning")
            elif ex.response.status_code == 409:
                # Server says cart isn't bound anymore — almost certainly evicted (we went idle
                # past the heartbeat window). Drop the assistant placeholder, flip back to
                # banner-mode, and let the user re-Activate.
                state["history"].pop()  # the empty assistant message
                state["history"].pop()  # the user message they typed
                state["active_bound"] = False
                bind_banner_label.text = "Session evicted (idle). Send again, or click Activate, to bring it back online."
                input_box.set_value(txt)  # restore the typed message so it's not lost
                safe_notify("Session was evicted while idle — send again, or click Activate.", type="warning")
            elif ex.response.status_code == 423:
                state["history"][-1]["content"] += "\n\n_[server: in use by another session]_"
            else:
                state["history"][-1]["content"] += f"\n\n_[stream error: HTTP {ex.response.status_code}]_"
        except Exception as e:
            state["history"][-1]["content"] += f"\n\n_[stream error: {e}]_"
        finally:
            if first_token_at is not None and n_tokens > 1:
                elapsed = time.monotonic() - first_token_at
                rate = (n_tokens - 1) / elapsed if elapsed > 0 else 0
                speed_label.text = f"{n_tokens} tok · {elapsed:.1f}s · {rate:.1f} tok/s"
            else:
                speed_label.text = "—"
            state["streaming"] = False
            set_send_btn(False)
            # Clear any consumed images from the chip row. Server has either consumed
            # them (success) or rejected the request (error); either way the IDs are
            # one-shot and re-using them would 404.
            if attached:
                cart_id = state.get("active_id")
                if cart_id:
                    state["pending_images"][cart_id] = []
                render_pending_images()
            finish_live()

    def on_tool_event(kind, payload):
        """An answer's event other than a token, into the conversation's last message: a call to the model's tool and
        its value, the ask that stops it, or the server's error. False where the stream is over."""
        if kind == "tool":
            state["history"][-1]["content"] += payload.get("text", "")
            mark_live()
        elif kind == "progress":                          # a picture model's, as the picture is made: said beside the box
            speed_label.text = payload.get("text", "")
        elif kind == "ask":
            state["ask"] = payload
        elif kind == "done" and "used" in payload:
            show_context(used=payload["used"], context=payload.get("context"))
        elif kind == "error":
            state["history"][-1]["content"] += f"\n\n_[server error: {payload.get('error')}]_"
            return False
        return True

    async def answer_ask(choice: str):
        """The user's answer to the model's ask, then the rest of its answer, into the conversation's last message."""
        if state["streaming"] or not state.get("ask") or not state["active_id"]:
            return
        # a conversation put away while the ask waited is brought back first, as sending brings it back
        if not state.get("active_bound"):
            await bind_active()
            if not state.get("active_bound"):
                return
        asked = state["ask"]
        state["ask"] = None
        state["streaming"] = True
        set_send_btn(True)
        if not state["history"] or state["history"][-1]["role"] != "assistant":
            state["history"].append({"role": "assistant", "content": ""})
        render_chat()
        try:
            async for kind, payload in client.stream_answer(state["active_id"], choice, int(max_in.value or 2048)):
                if kind == "token":
                    state["history"][-1]["content"] += payload.get("text", "")
                    if "used" in payload:
                        show_context(used=payload["used"])
                    mark_live()
                elif not on_tool_event(kind, payload):
                    break
        except httpx.HTTPStatusError as ex:
            state["history"][-1]["content"] += f"\n\n_[stream error: HTTP {ex.response.status_code}]_"
            if ex.response.status_code in (409, 423):
                state["ask"] = asked                  # not answered: its buttons stay
        except Exception as e:
            state["history"][-1]["content"] += f"\n\n_[stream error: {e}]_"
        finally:
            state["streaming"] = False
            set_send_btn(False)
            finish_live()

    async def stop():
        if not state["streaming"] or not state["active_id"]:
            return
        try:
            await client.stop_generation(state["active_id"])
            safe_notify("stop signal sent — finishing current token", type="info")
        except httpx.HTTPStatusError as ex:
            # Older server doesn't have /stop. The stream will continue server-side; client UI
            # can't cleanly interrupt without it. Surface the limitation.
            if ex.response.status_code == 404:
                safe_notify("server doesn't support stop (restart it to pick up the new endpoint)",
                          type="warning")
            else:
                safe_notify(f"stop failed: {ex}", type="negative")
        except Exception as e:
            safe_notify(f"stop failed: {e}", type="negative")

    async def send_or_stop():
        if state["streaming"]:
            await stop()
        else:
            await send()

    send_btn.on("click", send_or_stop)

    # Plain Enter sends — `.exact.prevent` stops the browser from inserting a newline locally
    # before set_value("") round-trips back from the server. Shift+Enter has no listener attached
    # so the browser's default newline behavior runs.
    input_box.on("keydown.enter.exact.prevent",
                 lambda e: asyncio.create_task(send_or_stop()))

    # ----- Options view: server-control wiring (only if LOCAL_AVAILABLE) -----
    if LOCAL_AVAILABLE:
        async def refresh_status():
            # Probe the configured endpoint so externally-started servers (e.g. ones spawned by a
            # previous UI instance and outliving it) get surfaced too, not just subprocesses we own.
            external = None
            try:
                h = await client.health()
                external = f"alive at {client.endpoint} (model={h.get('model_id', '?')})"
            except Exception:
                external = None
            if SERVER.failure and not SERVER.is_running():
                if state.get("failure_shown") != SERVER.failure:
                    state["failure_shown"] = SERVER.failure
                    safe_notify(("the server ended: " if SERVER.started else "not started: ") + SERVER.failure,
                                type="negative")
                setup_lbl.text = ("⛔ the server ended: " if SERVER.started else "⛔ not started: ") + SERVER.failure
                setup_lbl.classes(replace="text-sm whitespace-pre-wrap text-red-700")
            elif SERVER.setup:
                setup_lbl.text = SERVER.setup
                setup_lbl.classes(replace="text-sm whitespace-pre-wrap text-slate-700")
            else:
                setup_lbl.text = ""
            if SERVER.is_running():
                status_lbl.text = f"status: {SERVER.status_text()}"
            elif SERVER.failure:
                status_lbl.text = "status: ended" if SERVER.started else "status: not started"
            elif external:
                status_lbl.text = (f"status: external server {external} — not under UI control "
                                   f"(kill it from the box that started it)")
            else:
                status_lbl.text = "status: stopped"
            start_btn.set_enabled(not SERVER.is_running())
            stop_btn.set_enabled(SERVER.is_running())

        async def do_start():
            if not model_select.value:
                safe_notify("Pick a model first", type="warning")
                return
            if not config_select.value:
                safe_notify("Pick a config first", type="warning")
                return
            try:
                model_dir = str(MODELS_DIR / model_select.value)
                # a --local app starts it again when it is next started, once it has booted
                SERVER.start(model_dir, config_select.value, int(port_in.value),
                             remember={"model": model_select.value, "config": config_select.value,
                                       "port": int(port_in.value)} if LOCAL_APP else None)
                state["server_port"] = int(port_in.value)
                state["selected_model"] = model_select.value
                state["selected_config"] = config_select.value
                new_ep = f"http://localhost:{int(port_in.value)}"
                state["endpoint"] = new_ep
                endpoint_in.value = new_ep
                server_sel.set_value("local")
                client.set_endpoint(new_ep)
                safe_notify(f"started: {model_select.value} / {config_select.value} "
                            f"on port {int(port_in.value)}", type="positive")
            except Exception as e:
                safe_notify(f"start failed: {e}", type="negative")
            await refresh_status()

        async def do_stop():
            try:
                # stop() now saves conversations first (blocking, can be ~min for big windows) —
                # run it off the UI event loop so the page doesn't freeze during the save.
                await asyncio.to_thread(SERVER.stop)
                safe_notify("stopped", type="info")
            except Exception as e:
                safe_notify(f"stop failed: {e}", type="negative")
            await refresh_status()

        start_btn.on("click", do_start)
        stop_btn.on("click", do_stop)

        def pump_log():
            snap = SERVER.snapshot()
            cur = state["log_cursor"]
            if SERVER.rewrites != state.get("log_rewrites"):   # a line replaced in place: the pane again, whole
                state["log_rewrites"] = SERVER.rewrites
                cur = len(snap) + 1
            if cur > len(snap):
                log_pane.clear()
                cur = 0
            for line in snap[cur:]:
                log_pane.push(line)
            state["log_cursor"] = len(snap)

        log_pane.clear()
        state["log_cursor"] = 0
        pump_log()
        ui.timer(0.5, pump_log)
        ui.timer(1.0, refresh_status)
        asyncio.create_task(refresh_status())

    def open_config_editor(filename: Optional[str]):
        """Edit the selected model's configs/<filename>.json (filename=None means New)."""
        model = state.get("selected_model") or ""
        if not model:
            safe_notify("Pick a model first", type="warning")
            return
        cfg_dir = model_configs_dir(model)
        is_new = not (filename and (cfg_dir / filename).exists())
        if not is_new:
            body = (cfg_dir / filename).read_text(encoding="utf-8")
        else:
            body = json.dumps(new_config(model), indent=2)
        with page_client, ui.dialog().props("maximized") as dlg, ui.card().classes("w-full h-full"):
            ui.label(("New config" if is_new else f"Edit {filename}")
                     ).classes("font-bold")
            name_in = ui.input(label="filename",
                               value=(filename or "new_config.json")
                               ).classes("w-80")
            body_in = ui.textarea(value=body
                                  ).classes("w-full font-mono text-xs flex-grow"
                                            ).props("outlined autogrow rows=30")
            err = ui.label("").classes("text-red-500 text-sm")

            async def save():
                try:
                    parsed = json.loads(body_in.value)
                except json.JSONDecodeError as e:
                    err.text = f"JSON parse error: {e}"
                    return
                target = cfg_dir / name_in.value
                target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
                safe_notify(f"saved {target.name}", type="positive")
                dlg.close()
                ui.navigate.reload()

            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("config guide", icon="help_outline",
                          on_click=lambda: (dlg.close(), open_config_guide())).props("flat")
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("save", on_click=save).props("color=primary")
        dlg.open()

    def forget_autotune():
        """The setup the boot chose for the selected model on this machine forgotten (its `autotune.json`): the next start
        tests the setups again, and what the config leaves on auto is chosen afresh. The configs are not touched."""
        model = state.get("selected_model") or ""
        if not model:
            safe_notify("Pick a model first", type="warning")
            return
        target = MODELS_DIR / model / "autotune.json"
        if not target.exists():
            safe_notify("Nothing to forget: the next start tests the setups anyway", type="info")
            return
        with page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label("Autotune again?").classes("font-bold")
            ui.label("Forgets the setup chosen for this machine — where the experts sit, the tier on the card, the "
                     "context. The next start tests the setups again (a minute or so) and chooses afresh. The configs "
                     "are not changed." + (" A server running now keeps its setup until it is started again."
                                           if SERVER.is_running() else "")).classes("text-xs opacity-70 max-w-md")

            def do_forget():
                try:
                    target.unlink()
                except Exception as e:
                    safe_notify(f"could not forget it: {e}", type="negative")
                    dlg.close()
                    return
                safe_notify("forgotten: the next start tests the setups again", type="positive")
                dlg.close()

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("autotune again", on_click=do_forget).props("color=primary")
        dlg.open()

    def delete_config(filename: Optional[str]):
        """Delete the selected model's configs/<filename>.json (with a confirm step)."""
        model = state.get("selected_model") or ""
        if not model:
            safe_notify("Pick a model first", type="warning")
            return
        if not filename:
            safe_notify("No config selected", type="warning")
            return
        target = model_configs_dir(model) / filename
        if not target.exists():
            safe_notify(f"{filename} not found", type="warning")
            return
        with page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label(f"Delete {filename}?").classes("font-bold")
            ui.label("Permanently removes this config file. The model's weights are "
                     "not affected.").classes("text-xs opacity-70 max-w-md")

            def do_delete():
                try:
                    target.unlink()
                except Exception as e:
                    safe_notify(f"delete failed: {e}", type="negative")
                    dlg.close()
                    return
                safe_notify(f"deleted {filename}", type="positive")
                dlg.close()
                ui.navigate.reload()

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("delete", on_click=do_delete).props("color=negative")
        dlg.open()

    # Initial render
    update_view()
    render_chat()
    await ping()
    await render_convs()
    # a streaming answer drawn at most ten times a second, whatever the tokens' rate
    ui.timer(0.1, lambda: refresh_live() if live_answer.get("dirty") else None)

    # Keep the status dot live — re-ping every 5s.
    def _periodic_ping():
        asyncio.create_task(ping())
    ui.timer(5.0, _periodic_ping)

    # Heartbeat: tell the server we're still here so the idle-eviction loop doesn't unbind us.
    # Every 30s — well under the 2-min eviction window (gives ~4 missed pings of slack).
    async def _heartbeat():
        try:
            resp = await client.ping()
            # If our active cart isn't in the server's owned list, the server evicted us —
            # flip the UI back to banner-mode so the user can Activate again.
            if state.get("active_bound") and state.get("active_id"):
                if state["active_id"] not in (resp.get("owned") or []):
                    state["active_bound"] = False
                    bind_banner_label.text = "Session evicted by server (idle). Click Activate to reload."
        except Exception:
            pass

    def _periodic_heartbeat():
        asyncio.create_task(_heartbeat())
    ui.timer(30.0, _periodic_heartbeat)


# ---------- CLI ----------

def _start_last():
    """The model a --local UI started last, started again as it was — the app opens where it was left."""
    try:
        last = json.loads(LAST_START.read_text())
    except (OSError, ValueError):
        return
    model, config = last.get("model") or "", last.get("config") or ""
    if model_present(model) and config and (model_configs_dir(model) / config).is_file():
        SERVER.start(str(MODELS_DIR / model), config, int(last.get("port", 8765)))


def _stop_all():
    """A --local UI stopping (a container's `docker stop`): the download stopped, the conversations saved, the server
    stopped — ServerProcess.stop asks the server to save them first."""
    DOWNLOADS.cancel()
    SERVER.stop()


def main():
    global LOCAL_APP
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080, help="port for the NiceGUI app (default 8080)")
    p.add_argument("--host", default=None, help="the address the UI listens on (default NiceGUI's own)")
    p.add_argument("--endpoint", default=None,
                   help="default SiLVANN server endpoint (override on first launch)")
    p.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    # ⛳ off by default: a reload restarts this process, and a server it started goes with it — a boot of minutes lost to
    #   a file saved beside the UI
    p.add_argument("--reload", action="store_true", help="restart the UI when its files change (development)")
    p.add_argument("--local", action="store_true",
                   help="the UI as the app over this machine's models: the local panel first, "
                        "the model started last started again, and the server stopped with the UI")
    args = p.parse_args()

    if args.endpoint:
        os.environ["SILVANN_DEFAULT_ENDPOINT"] = args.endpoint
    if args.local:
        LOCAL_APP = True
        app.on_startup(_start_last)
        app.on_shutdown(_stop_all)

    ui.run(port=args.port, title="SiLVANN", show=not args.no_browser, storage_secret="silvann-ui",
           reload=args.reload and not args.local, **({"host": args.host} if args.host else {}))


if __name__ in {"__main__", "__mp_main__"}:
    main()
