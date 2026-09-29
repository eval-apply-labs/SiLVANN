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
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
from nicegui import ui, app


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


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
MODELS_DIR = REPO_ROOT / "models"                    # main/models/<name>/ — self-contained model folders
# the documents: docs/ beside the UI in a release, backstage/release/ in the development tree
RELEASE_DIR = REPO_ROOT if (REPO_ROOT / "docs" / "users.md").exists() else REPO_ROOT.parent / "backstage" / "release"
DOCS_DIR = RELEASE_DIR / "docs"


def list_models():
    """The self-contained model folders under main/models/ (folder names; __pycache__ excluded)."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name != "__pycache__")


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


LOCAL_AVAILABLE = SERVER_SCRIPT.exists() and bool(list_models())

# Help-view documents: (label, path). path=None -> the built-in UI guide (HELP_MARKDOWN below).
# The .md files are read from the repo at view time; only present when the UI runs from the repo root.
HELP_DOCS = [
    ("UI Guide", None),
    ("Overview", RELEASE_DIR / "README.md"),
    ("Running SiLVANN", DOCS_DIR / "users.md"),
    ("Working on SiLVANN", DOCS_DIR / "developers.md"),
    ("The ideas behind it", DOCS_DIR / "concepts.md"),
    ("How the weights are packed", DOCS_DIR / "compression.md"),
    ("Where it could go", DOCS_DIR / "directions.md"),
]


class ServerProcess:
    """Single subprocess + a shared line buffer that log panes poll."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: Optional[list] = None
        self.lines: deque[str] = deque(maxlen=2000)
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status_text(self) -> str:
        if not self.is_running():
            return "stopped"
        return f"running (pid {self.proc.pid}, cmd: {' '.join(self.cmd or [])})"

    def start(self, model_dir: str, config_name: str, port: int):
        if self.is_running():
            raise RuntimeError("already running — stop first")
        if not SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"server script not found at {SERVER_SCRIPT}")
        self.lines.clear()
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
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if not line:
                    break
                self._log(line.rstrip())
        finally:
            self._log("[ui] subprocess output stream closed.")

    def _log(self, line: str):
        with self.lock:
            self.lines.append(line)

    def snapshot(self) -> list:
        with self.lock:
            return list(self.lines)


SERVER = ServerProcess()


# ---------- Per-browser session state ----------

def _session_default():
    return {
        "endpoint": os.environ.get("SILVANN_DEFAULT_ENDPOINT", "http://localhost:8765"),
        "active_id": None,
        "active_name": None,
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
        # Engine toggle for the active conversation: whether the model is asked to think at all.
        # Synced from the server on activate (the canonical value lives in the cartridge sidecar).
        "active_thinking_mode": True,
        # Nav state: which main view is selected and whether the chat sub-tree is expanded.
        # The drawer itself defaults to mini-with-hover-expand; that's a fixed UX, not stored.
        "view": "chat",            # chat | options | help
        "convs_expanded": True,
        # Options-view prefs (was the Server tab)
        "server_mode": "remote",   # remote | local
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
    ui.add_head_html("""
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
        .q-drawer--mini .silvann-endpoint,
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
    </style>
    """)

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

        ui.space()
        ui.separator()

        # --- Endpoint settings (drawer bottom, hidden in mini mode) ---
        with ui.column().classes("w-full px-2 py-2 gap-1 silvann-endpoint"):
            endpoint_in = ui.input(label="Endpoint", value=state["endpoint"]
                                   ).props("dense outlined").classes("w-full")
            connect_btn = ui.button("connect").props("flat dense color=primary").classes("w-full")

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
            thinking_mode_toggle = ui.switch("thinking mode",
                                              value=state.get("active_thinking_mode", True)
                                              ).props("dense").classes("text-xs")
            show_think_toggle = ui.switch("show thinking",
                                           value=state.get("show_thinking", False)
                                           ).props("dense").classes("text-xs")
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
        # Banner shown when the active cart is loaded into UI memory but NOT bound to VRAM.
        # User must click Activate to /bind. Hidden once bound; reappears after server eviction.
        with ui.row().classes("w-full items-center gap-2 p-3 rounded bg-amber-50 border border-amber-300"
                              ).bind_visibility_from(state, "active_bound", lambda b: not b
                                                     ) as bind_banner:
            ui.icon("info", size="sm").classes("text-amber-700")
            bind_banner_label = ui.label("Conversation loaded. Click Activate to bring the session online."
                                          ).classes("text-sm flex-grow")
            bind_btn = ui.button("Activate", icon="play_arrow"
                                 ).props("color=primary unelevated")
        chat_scroll = ui.scroll_area().classes("w-full flex-grow min-h-0 border rounded")
        with chat_scroll:
            chat_history_col = ui.column().classes("w-full p-4 gap-3")
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
            # with `accepted_files="image/*"` and bound to the active cart's /images endpoint.
            # auto_upload fires on file selection so the user doesn't have to click upload twice.
            attach_upload = ui.upload(auto_upload=True, max_files=1,
                                       max_file_size=10 * 1024 * 1024
                                       ).props('label="" accept="image/*" flat hide-upload-btn '
                                               'color=primary')
            attach_upload.classes("hidden")
            attach_btn = ui.button(icon="attach_file"
                                   ).props("color=primary"
                                           ).tooltip("Attach an image to the next message")
            attach_btn.set_visibility(False)       # images are not accepted by this release's runtime (the server answers 501)
            input_box = ui.textarea(placeholder="Type a message — Enter to send, Shift+Enter for newline"
                                    ).classes("flex-grow").props("autogrow outlined")
            max_in = ui.number(label="max tokens", value=2048, min=1, max=4096).classes("w-32")
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
        with ui.row().classes("items-center gap-4 mb-2"):
            ui.label("Mode:")
            mode_toggle = ui.toggle(
                {"remote": "Remote (connect to existing)",
                 "local":  "Local (start a server on this box)"},
                value=state["server_mode"],
            ).on_value_change(lambda e: state.__setitem__("server_mode", e.value))

        # Remote panel
        with ui.column().bind_visibility_from(mode_toggle, "value",
                                              lambda v: v == "remote"
                                              ).classes("w-full gap-2"):
            ui.label(
                "Use the Endpoint field in the left drawer to set the server URL. "
                "You can't start or stop a remote server from here — sign in to that box and use "
                "start_silvann_server.py directly, or systemd/tmux for persistent runs."
            ).classes("opacity-70 max-w-2xl")

        # Local panel
        with ui.column().bind_visibility_from(mode_toggle, "value",
                                              lambda v: v == "local"
                                              ).classes("w-full gap-2"):
            if not LOCAL_AVAILABLE:
                ui.label(
                    f"Local mode unavailable: this UI process can't find start_silvann_server.py "
                    f"or any model folders under {MODELS_DIR}. Run the UI from main/ (the runtime root) "
                    f"to enable local control."
                ).classes("text-orange-600 max-w-2xl")
            else:
                ui.label(f"Models dir: {MODELS_DIR}").classes("text-xs opacity-70")

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
                config_select = ui.select(
                    _init_cfgs, value=_init_cfg, label="Config",
                ).classes("w-80")
                config_desc_label = ui.label(_config_desc(_init_model, _init_cfg)).classes(
                    "text-xs opacity-70 max-w-2xl whitespace-normal")

                def _on_config_change(e):
                    state["selected_config"] = e.value
                    config_desc_label.set_text(_config_desc(state["selected_model"], e.value))
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
                model_select.on_value_change(_on_model_change)
                with ui.row().classes("gap-2 items-end"):
                    port_in = ui.number(label="Port", value=state["server_port"],
                                        min=1, max=65535).classes("w-32")
                    ui.button("edit selected",
                              on_click=lambda: open_config_editor(config_select.value)
                              ).props("dense outline")
                    ui.button("new config",
                              on_click=lambda: open_config_editor(None)
                              ).props("dense outline")
                    ui.button("delete selected",
                              on_click=lambda: delete_config(config_select.value)
                              ).props("dense outline color=negative")
                status_lbl = ui.label("status: ...").classes("font-mono text-sm")
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

    # --- Help view: a document picker over the bundled .md docs (UI guide + role guides). ---
    help_view = ui.column().classes("silvann-main w-full p-6 overflow-auto")
    with help_view:
        with ui.column().classes("max-w-4xl mx-auto gap-3 w-full"):
            def _load_help_doc(label):
                path = dict(HELP_DOCS).get(label)
                if path is None:
                    return HELP_MARKDOWN
                try:
                    return path.read_text(encoding="utf-8")
                except Exception as e:
                    return (f"_Couldn't read `{path.name}` — run the UI from the SiLVANN repo root "
                            f"to view the bundled docs._\n\n`{e}`")
            help_doc_select = ui.select([lbl for lbl, _ in HELP_DOCS], value="UI Guide",
                                        label="Document").classes("w-80")
            help_doc_md = ui.markdown(HELP_MARKDOWN)
            help_doc_select.on_value_change(
                lambda e: help_doc_md.set_content(_load_help_doc(e.value)))

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

    async def ping():
        try:
            h = await client.health()
            status_dot.classes(replace="text-green-500").style("font-size: 12px")
            status_tip.text = (f"OK — {h['model_id']} · layers {h['pp_slice']} · "
                               f"gpus {h['num_gpus']} · tenants {h['tenants_loaded']}"
                               + (f" · active {h['active'][:8]}" if h['active'] else ""))
        except Exception as e:
            status_dot.classes(replace="text-red-500").style("font-size: 12px")
            status_tip.text = f"down: {type(e).__name__}: {e}"

    async def connect_clicked():
        state["endpoint"] = endpoint_in.value
        client.set_endpoint(endpoint_in.value)
        await ping()
        await render_convs()

    connect_btn.on("click", connect_clicked)

    # ----- Chat helpers -----
    def render_chat():
        chat_history_col.clear()
        if state["active_id"]:
            chat_header.text = state.get("active_name") or state["active_id"]
            chat_header.classes(replace="text-lg font-semibold flex-grow")
        else:
            chat_header.text = "(no conversation active — pick one from the menu or click '+ new')"
            chat_header.classes(replace="text-sm opacity-60 flex-grow")
        show_think = bool(state.get("show_thinking", False))
        with chat_history_col:
            if not state["history"] and state["active_id"]:
                ui.label("(empty — say hi)").classes("opacity-50 text-sm self-center")
            for m in state["history"]:
                content = m["content"] or ""
                if m["role"] == "assistant" and not show_think:
                    content = THINK_RE.sub("", content)
                _sent = (m["role"] == "user")
                # sent=user gives the bubble its style; self-end/self-start does the actual L/R
                # alignment (the column's align-items:flex-start otherwise keeps both on the left),
                # capped at 85% width so the bubbles don't span the whole row.
                with ui.chat_message(name=m["role"], sent=_sent
                                     ).classes("self-end" if _sent else "self-start"
                                     ).style("max-width: 85%"):
                    ui.markdown(content or "_(empty)_")
            if state["streaming"]:
                ui.label("…").classes("opacity-50 self-center animate-pulse")
        chat_scroll.scroll_to(percent=1.0)

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
            new_thinking = bool(resp.get("enable_thinking", True))
            state["active_thinking_mode"] = new_thinking
            thinking_mode_toggle.set_value(new_thinking)
            state["protected_active"] = bool(resp.get("protected", False))
            protected_icon.props(f"icon={'lock' if state['protected_active'] else 'lock_open'}")
            # bound_by_self happens if the cart is already bound to OUR session (e.g., we
            # navigated away and back within the heartbeat window). Treat that as already-bound.
            state["active_bound"] = bool(resp.get("bound_by_self", False))
            if resp.get("bound_by_other"):
                bind_banner_label.text = "Conversation is held by another session. Click Activate to take it over (will fail if they're still pinging)."
            else:
                bind_banner_label.text = "Conversation loaded. Click Activate to bring the session online (~3s, allocates VRAM)."
            render_chat()
            await render_convs()
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 401:
                safe_notify("wrong password", type="negative")
                client.set_password(cart_id, None)
            else:
                safe_notify(f"activate failed: {ex}", type="negative")
        except Exception as e:
            safe_notify(f"activate failed: {e}", type="negative")

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
        if state["active_id"] is None:
            safe_notify("Pick a conversation first (or click '+ new' under Chat)", type="warning")
            return
        if not state.get("active_bound"):
            safe_notify("Click 'Activate' first to bring the session online", type="warning")
            return
        txt = (input_box.value or "").strip()
        attached = list(_cart_pending())
        if not txt and not attached:
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
        render_chat()
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
                render_chat()
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
                        render_chat()
                    continue
                if waiting:
                    waiting = False
                    state["history"][-1]["content"] = ""   # drop the "waiting my turn" placeholder
                if kind == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    n_tokens += 1
                    state["history"][-1]["content"] += payload.get("text", "")
                    elapsed = time.monotonic() - first_token_at
                    if elapsed > 0.3 and n_tokens > 1:
                        rate = (n_tokens - 1) / elapsed   # exclude the first token from the divisor
                        speed_label.text = f"{rate:.1f} tok/s"
                    render_chat()
                elif kind == "error":
                    state["history"][-1]["content"] += (
                        f"\n\n_[server error: {payload.get('error')}]_")
                    break
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 409:
                # Server says cart isn't bound anymore — almost certainly evicted (we went idle
                # past the heartbeat window). Drop the assistant placeholder, flip back to
                # banner-mode, and let the user re-Activate.
                state["history"].pop()  # the empty assistant message
                state["history"].pop()  # the user message they typed
                state["active_bound"] = False
                bind_banner_label.text = "Session evicted (idle). Click Activate to bring it back online."
                input_box.set_value(txt)  # restore the typed message so it's not lost
                safe_notify("Session was evicted while idle. Click Activate.", type="warning")
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
            render_chat()

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
            if SERVER.is_running():
                status_lbl.text = f"status: {SERVER.status_text()}"
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
                SERVER.start(model_dir, config_select.value, int(port_in.value))
                state["server_port"] = int(port_in.value)
                state["selected_model"] = model_select.value
                state["selected_config"] = config_select.value
                new_ep = f"http://localhost:{int(port_in.value)}"
                state["endpoint"] = new_ep
                endpoint_in.value = new_ep
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
            body = json.dumps({
                "_comment": "New config — edit the values, then Save. Keys starting with '_' are "
                            "documentation only and are IGNORED by the engine: each '_<field>_desc' "
                            "describes the field above it. For the full offload + tiered-KV-cooling "
                            "reference, copy a shipped config (e.g. a *_offload_200k.json) instead.",

                "model_dir": "./model_data",
                "_model_dir_desc": "Weights directory, relative to this model folder. Almost always ./model_data.",
                "tokenizer_dir": "./model_data",
                "_tokenizer_dir_desc": "Tokenizer directory; self-booting model_data carries the tokenizer, so usually ./model_data.",
                "model": "python.silvann_model",
                "_model_desc": "Python package exporting this model's Composer + ChatAdapter (resolved from the model folder).",

                "is_master": True,
                "_is_master_desc": "True for a single box or the master of a pipeline-parallel ring; False for a PP worker.",
                "num_gpus": 1,
                "_num_gpus_desc": "GPUs on THIS box for tensor parallelism (1 = single card).",
                "visible_devices": "0",
                "_visible_devices_desc": "Comma-separated device indices this process may use, e.g. \"0\" or \"0,1,2,3\".",
                "pp_slice": [0, 40],
                "_pp_slice_desc": "[start, end) layer range this node runs. [0, N] = all N layers on one box.",
                "pipeline_next_ip": None,
                "listen_port": 5555,
                "next_port": 5556,
                "_pp_ring_desc": "pipeline_next_ip + listen_port/next_port wire a multi-box PP ring; leave as-is (ignored) on a single box.",

                "concurrent_dma_unsupported": True,
                "_concurrent_dma_unsupported_desc": "True on AMD/MI50 or any no-PCIe-P2P box (and the safe default). Set False on a CUDA/P2P box to enable the resident single-pass + op-major prefill paths.",
                "system_ram_offload": False,
                "_system_ram_offload_desc": "MoE only. \"data\" streams routed experts to host RAM (the 8GB-card path); False keeps all weights resident (needs a big card).",
                "tool_mailbox_size_mb": 2,

                "engine_config": {
                    "max_total_tokens": 8192,
                    "temperature": 0.7,
                },
                "_engine_config_desc": "max_total_tokens = the KV active-window span (older tokens roll off — NOT an output cap). temperature: 0 = greedy/deterministic, 0.7 = the chat default. A cooled (tiered-KV) setup adds a 'cooling' block (hot fp32 / warm q6 / cold q4) — see a shipped *_offload_200k.json for the full reference.",
            }, indent=2)
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
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("save", on_click=save).props("color=primary")
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


# ---------- Help content (markdown shown in the Help view) ----------

HELP_MARKDOWN = """\
# SiLVANN UI

A chat front-end for a SiLVANN server. The UI only talks HTTP; the model runs in `start_silvann_server.py`, on this
machine or another one.

## Navigation

The **left drawer** holds everything:

- **Chat** — the conversations, under the Chat row. Click one to open it.
- **Options** — start and stop a server on this machine, and edit its settings.
- **Help** — this page, and the release's documents.

The dot next to "SiLVANN" is the server's state: **gray** not asked yet, **green** answering, **red** unreachable.
The **Endpoint** field at the bottom is the server the UI talks to — `http://localhost:8765` when both run here.

## Chat

- **+ new** — a new conversation. Its context window is the model's, set in its config when the server started.
- **import** — a conversation exported earlier (`.silvann.tar`); refused if it was made with another model.
- **⋮** on a conversation — rename, export, delete, or protect it with a password.
- A ⚠ next to a name: the conversation belongs to another model than the one the server runs.
- **Enter** sends, **Shift+Enter** is a new line. **max tokens** caps an answer; the model stops earlier when it is
  done.
- One conversation is answered at a time. A message sent while another is being answered waits
  (**"⏳ waiting my turn…"**) and then streams.
- Images are not accepted yet.

## Options — a server on this machine

Pick a **model** from `models/`, then one of its **configs**, and **start**. The UI runs
`start_silvann_server.py --model models/<name> <config>`, shows its output, and points the Endpoint at it. A model
that is not there is fetched first with `python3 silvann_core.py download-model <name>`.

A config is `models/<name>/configs/<config>.json`:

```json
{ "max_context": 100000, "thinking": true, "lora": null, "sampler": "generation", "options": { } }
```

`max_context` is the longest conversation, in tokens; `options` is the model's own — the conversation cache's
tiers for the Qwen models, where the experts live and how a prompt is read for the ones that can run them on the
CPU. **Running SiLVANN**, in the list above, has every option.

## Where things are

- Conversations: `user_sessions/<model>/<id>/`, on the server's machine.
- A model and its configs: `models/<name>/`; the programs it runs, as written at its last start:
  `models/<name>/programs.lisp`.
- The UI's own settings (endpoint, the last conversation, passwords you typed): this browser's storage.
"""


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080, help="port for the NiceGUI app (default 8080)")
    p.add_argument("--endpoint", default=None,
                   help="default SiLVANN server endpoint (override on first launch)")
    p.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = p.parse_args()

    if args.endpoint:
        os.environ["SILVANN_DEFAULT_ENDPOINT"] = args.endpoint

    ui.run(port=args.port, title="SiLVANN", show=not args.no_browser,
           storage_secret="silvann-ui")


if __name__ in {"__main__", "__mp_main__"}:
    main()
