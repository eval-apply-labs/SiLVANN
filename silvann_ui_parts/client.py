"""The server's HTTP API as the UI calls it — one `Client` a page — and the export a browser downloads through the UI."""
import asyncio
import contextlib
import contextvars
import json
import secrets
import time
from typing import Optional

import httpx
from nicegui import app


# ⭐ WHAT A PAGE ASKS BY ITSELF IS NO USE OF ITS SIGN-IN: a sign-in is over ten minutes after its account last did
#   something (▶ silvann_runtime/users.py), and a page asks after its server every few seconds for as long as it is
#   open. Those requests say so — `X-Silvann-Unattended`, set on every request made within `unattended()` — and the
#   server answers them without counting them: a page left open keeps no one signed in.
UNATTENDED = contextvars.ContextVar("silvann_unattended", default=False)


@contextlib.contextmanager
def unattended():
    """The requests made within are the page's own — a timer's — and not the person's."""
    held = UNATTENDED.set(True)
    try:
        yield
    finally:
        UNATTENDED.reset(held)


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
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None),
                                      event_hooks={"request": [self._marked], "response": [self._seen]})
        self.passwords: dict = passwords if passwords is not None else {}
        self.web_user_id = web_user_id or ""
        self.on_signed_out = None          # called when a server with users answers 401: the sign-in is asked for

    # ── a server with users (NN-150): a sign-in, its token on every request ─────────────────────────────
    def use_token(self, token: Optional[str]):
        if token:
            self.http.headers["Authorization"] = "Bearer " + token
        else:
            self.http.headers.pop("Authorization", None)

    async def _marked(self, request):
        if UNATTENDED.get():
            request.headers["X-Silvann-Unattended"] = "1"

    async def used(self):
        """The person is at the page: said to a server elsewhere, whose sign-in counts its own ten minutes."""
        await self.http.get(f"{self.endpoint}/users/me")

    async def _seen(self, response):
        if response.status_code == 401 and not response.request.url.path.endswith("/users/sign_in") \
                and self.on_signed_out is not None:
            await self.on_signed_out()

    async def sign_in(self, name: str, password: str) -> dict:
        r = await self.http.post(f"{self.endpoint}/users/sign_in", json={"name": name, "password": password})
        r.raise_for_status()
        got = r.json()
        self.use_token(got["token"])
        return got

    async def sign_out(self):
        try:
            await self.http.post(f"{self.endpoint}/users/sign_out")
        finally:
            self.use_token(None)

    # ── single sign-on (NN-157): the server's provider, the page to send the browser to ──────────────────────────
    async def sso(self) -> dict:
        r = await self.http.get(f"{self.endpoint}/users/sso")
        r.raise_for_status()
        return r.json()

    async def sso_start(self) -> dict:
        r = await self.http.post(f"{self.endpoint}/users/sso/start")
        r.raise_for_status()
        return r.json()

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

    async def create(self, name: str, system: Optional[str] = None) -> dict:
        r = await self.http.post(f"{self.endpoint}/cartridges", json={"name": name, "system": system})
        r.raise_for_status()
        return r.json()

    async def estimate_vram(self) -> dict:
        """What a new conversation costs: the model's context, and what it takes in RAM put away. Used by the new-chat
        dialog."""
        r = await self.http.post(f"{self.endpoint}/cartridges/estimate")
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

    async def gone(self) -> dict:
        """This page has gone: with seats, the server puts its conversation away soon after its last answer."""
        r = await self.http.post(f"{self.endpoint}/sessions/gone", headers=self._hdr(None))
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
        """The server config's tools — a new conversation's mode, or "no" — the folders, and the machine's tools file's
        entries (▶ /tools)."""
        r = await self.http.get(f"{self.endpoint}/tools")
        r.raise_for_status()
        return r.json()

    async def rebuild_tools(self):
        """The server's skills read again, its tools file kept in step and its startup check run again
        (▶ /tools/rebuild): what it did, and the tools as `tools_settings` answers them."""
        r = await self.http.post(f"{self.endpoint}/tools/rebuild", timeout=httpx.Timeout(120.0))
        r.raise_for_status()
        return r.json()

    async def stream_answer(self, cart_id: str, choice: str):
        """The user's answer to the model's ask, then the rest of the answer — as `stream_message` yields it."""
        async for kind, payload in self._stream(f"{self.endpoint}/cartridges/{cart_id}/tools/answer", {"choice": choice}, cart_id):
            yield kind, payload

    async def stream_message(self, cart_id: str, text: str, images: Optional[list] = None, whole: Optional[list] = None):
        """Yield (event_kind, payload_dict) tuples from the server's SSE stream until done|error.

        `images` is an optional list of image_ids previously uploaded via
        `upload_image`; the server splices their cached embeddings into the prefill
        stream and pops them after a successful turn. `whole` names the documents among them
        the user asked for whole in the message, where the server would keep them as files.
        The answer runs until it ends, the stop button stops it, or it fills the conversation's context."""
        url = f"{self.endpoint}/cartridges/{cart_id}/messages"
        body = {"text": text}
        if images:
            body["images"] = list(images)
        if whole:
            body["whole"] = list(whole)
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
