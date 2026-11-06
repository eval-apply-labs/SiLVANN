"""THE REPL'S BROWSER — a Firefox in a companion container, driven over WebDriver BiDi for the REPL's `(browser …)`
(▶ tools.py, docker/companions/firefox).

⚖ 2026-10-31: *"we do not want the internet browser to be in the same container as the llm, i think it might be too
risky to download the wrong thing and all your private conversations get leaked"* · *"being pro-open-tools i would side
with firefox"*. The browser runs in a container of its own, its firewall letting it reach the public internet and nothing
private — not SiLVANN, not the machine — and SiLVANN connects to its control port; nothing it loads reaches SiLVANN's
files or its API. WebDriver BiDi is the W3C's protocol, built into Firefox; this speaks it over a WebSocket of the
standard library's sockets, so nothing is added to what SiLVANN installs.
```
  tab(address)                a conversation's tab in the server's one session on the companion (▶ `Browser`), opened
                              again where the companion was restarted — `Unreachable` where it does not answer
  .open(url)  .back()         a page loaded, the history stepped back
  .grab()                     the viewport as a Pillow image; .size(), .pointer() as the X11 screen's (▶ ui.py)
  .move(x, y) .click(b, n)    the pointer gliding to a pixel of the viewport, a button pressed where it is
  .type(text) .key("ctrl+l")  text typed, a key or a chord
  .scroll(n)                  n notches down where n > 0
  .text()  .elements()        the page's words; what can be clicked or typed into, each with its middle
  .where()                    (url, title)
  .install_userscript(url)    a public userscript installed in this session's Violentmonkey (▶ the companion's policies)
  .add_userscript(code)       a script's text — the model's own — run in this tab's pages as its header asks
```
"""
import base64
import io
import json
import math
import os
import socket
import struct
import threading
import urllib.parse

VIEWPORT = (1280, 800)
TIMEOUT = 60                      # seconds a command waits for its answer — a page's load among them
NOTCH = 120                       # pixels a scroll's notch moves the page
GLIDE_STEP = 15                   # pixels between the moves of the pointer's way — each element on it sees it pass
# ⛳ FIREFOX'S REMOTE AGENT ANSWERS ONLY ITS OWN HOST AND PORT: in the companion it listens on the container's loopback at
#   9223, socat handing it SiLVANN's connection on 9222 — so the handshake names "localhost:9223" whatever SiLVANN dialled
#   (a 400 for "localhost" or ":9222", MEASURED 2026-10-31 on node03, Firefox 153 ESR)
AGENT_HOST = "localhost:9223"
KEYS = {"enter": "", "return": "", "tab": "", "escape": "", "esc": "", "backspace": "",
        "delete": "", "del": "", "space": " ", "home": "", "end": "", "pageup": "",
        "pagedown": "", "up": "", "down": "", "left": "", "right": "", "insert": "",
        "f1": "", "f2": "", "f3": "", "f4": "", "f5": "", "f6": "", "f7": "",
        "f8": "", "f9": "", "f10": "", "f11": "", "f12": ""}
MODIFIERS = {"ctrl": "", "control": "", "shift": "", "alt": "", "meta": "", "cmd": "",
             "super": "", "win": ""}
# what a person can click or type into, each a line: its number, its kind, its words, its middle in the viewport, and
# what lies over that middle when something does — a person's click there reaches that instead (a click is a pointer
# pressed at the pixel, ▶ `Tab.click`). Besides what is named clickable — a link, a button, a field, a role, a handler
# written in the page, a popup's or a focusable element — what a script made clickable: an element that shows a pointer
# cursor its parent does not (the outermost of what shows one). Two at one middle are one, the named kept; a label and
# its own control over one another are neither under the other — a click on either is the control's. A checkbox's or a
# radio's words are its label's — its own, or the words beside it in its container — never its value ("on" for every
# one), and its kind says whether it is ticked, so what a click set can be read back.
ELEMENTS_JS = r"""(() => {
  const pick = 'a[href], button, input:not([type=hidden]), select, textarea, summary, label[for], [role=button], ' +
               '[role=link], [role=tab], [role=menuitem], [role=menuitemcheckbox], [role=menuitemradio], [role=option], ' +
               '[role=checkbox], [role=radio], [role=switch], [role=combobox], [role=treeitem], [contenteditable=true], ' +
               '[onclick], [aria-haspopup], [tabindex]:not([tabindex^="-"])';
  const box = e => e.tagName === 'INPUT' && (e.type === 'checkbox' || e.type === 'radio');
  const labels = e => [...(e.labels || [])].map(l => l.innerText).join(' ');
  const words = e => ([box(e) ? labels(e) || (e.parentElement || {}).innerText : '', e.innerText, box(e) ? '' : e.value,
    e.getAttribute('aria-label'), e.getAttribute('placeholder'), e.getAttribute('title'), e.getAttribute('alt'),
    box(e) ? '' : labels(e), e.getAttribute('name'), e.getAttribute('data-tooltip-content'), e.getAttribute('data-tooltip'),
    e.getAttribute('data-original-title'), (e.querySelector('img[alt]') || {}).alt, (e.querySelector('img[title]') || {}).title]
    .map(w => ((w || '') + '').trim()).find(w => w) || '').replace(/\s+/g, ' ').slice(0, 80);
  const state = e => box(e) ? (e.checked ? ', ticked' : ', not ticked') :
    e.hasAttribute('aria-checked') ? (e.getAttribute('aria-checked') === 'true' ? ', ticked' : ', not ticked') : '';
  const out = [], at = new Map();
  for (const e of document.body ? document.body.querySelectorAll('*') : []) {
    const named = e.matches(pick);
    if (!named && !(getComputedStyle(e).cursor === 'pointer' && e.parentElement &&
                    getComputedStyle(e.parentElement).cursor !== 'pointer')) continue;
    const r = e.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
    const st = getComputedStyle(e);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;
    const x = Math.round((Math.max(0, r.left) + Math.min(r.right, innerWidth)) / 2);
    const y = Math.round((Math.max(0, r.top) + Math.min(r.bottom, innerHeight)) / 2);
    const hit = document.elementFromPoint(x, y);
    const over = hit && hit !== e && !e.contains(hit) && !hit.contains(e) && hit !== e.control &&
                 !(hit.control && hit.control === e) ? hit : null;
    const row = [e.tagName.toLowerCase() + (e.type && e.tagName === 'INPUT' ? ':' + e.type : '') + state(e), words(e), x, y,
                 over ? over.tagName.toLowerCase() + (words(over) ? ' "' + words(over).slice(0, 30) + '"' : '') : '', named];
    const key = x + ',' + y;
    if (at.has(key)) {
      if (named && !out[at.get(key)][5]) out[at.get(key)] = row;
      continue;
    }
    at.set(key, out.length);
    out.push(row);
    if (out.length >= 150) break;
  }
  return JSON.stringify(out.map(r => r.slice(0, 5)));
})()"""


# The field the cursor is in, named as ELEMENTS_JS names an element — its kind and its words — and whether the page's
# pixel (x, y) lies inside it; null where the cursor is in no field.
FOCUSED_JS = r"""((x, y) => {
  const e = document.activeElement;
  if (!e || !e.matches('input:not([type=hidden]), textarea, select, [contenteditable=true]')) return null;
  const r = e.getBoundingClientRect();
  const kind = e.tagName.toLowerCase() + (e.type && e.tagName === 'INPUT' ? ':' + e.type : '');
  let words = (e.value || e.getAttribute('aria-label') || e.getAttribute('placeholder') || e.getAttribute('title') ||
               e.getAttribute('name') || '').trim().replace(/\s+/g, ' ').slice(0, 80);
  return JSON.stringify([kind, words, x >= r.left && x <= r.right && y >= r.top && y <= r.bottom]);
})(%d, %d)"""


# Where the view is on the page: its top, its height and the page's, in the page's pixels — a page taller than the view
# goes on where no look or element list reaches until it is scrolled
EXTENT_JS = r"""JSON.stringify([Math.round(scrollY), innerHeight,
  Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)])"""


VIOLENTMONKEY = "moz-extension://5ac1a7a2-6f4b-4f0e-9a51-7a1d1e0c0f00/"   # the companion's user.js fixes this address


def userscript_header(code):
    """A userscript's header — its `// @key value` lines between `==UserScript==` and `==/UserScript==` — as {key: [values]}."""
    out, inside = {}, False
    for line in code.splitlines():
        t = line.strip()
        if t.startswith("//") and "==UserScript==" in t:
            inside = True
        elif t.startswith("//") and "==/UserScript==" in t:
            break
        elif inside and t.startswith("//"):
            parts = t[2:].strip().split(None, 1)
            if parts and parts[0].startswith("@"):
                out.setdefault(parts[0][1:], []).append(parts[1].strip() if len(parts) > 1 else "")
    return out


# a userscript of the model's own, run as its header asks: in the pages @match or @include name and @exclude does not,
#   at @run-at's moment — with no GM_ functions (a public script installed through Violentmonkey has them)
USERSCRIPT_JS = r"""(() => {
  const toRe = (p) => {
    if (p.length > 2 && p.startsWith('/') && p.endsWith('/')) return new RegExp(p.slice(1, -1));
    if (p === '<all_urls>' || p === '*') return /.*/;
    return new RegExp('^' + p.split('*').map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$');
  };
  const take = %(take)s, skip = %(skip)s, at = %(at)s;
  const here = location.href;
  if (!take.some((p) => toRe(p).test(here)) || skip.some((p) => toRe(p).test(here))) return;
  const run = () => { try { %(body)s } catch (e) { console.error('userscript %(name)s:', e); } };
  if (at === 'document-start') run();
  else if (at === 'document-idle') addEventListener('load', run, { once: true });
  else if (document.readyState === 'loading') addEventListener('DOMContentLoaded', run, { once: true });
  else run();
})"""


class Unreachable(Exception):
    """The companion does not answer, or answered with an error: why."""


class _Socket:
    """A WebSocket client of the standard library's sockets: text frames, masked as a client's are (RFC 6455)."""

    def __init__(self, url, timeout=TIMEOUT):
        u = urllib.parse.urlparse(url)
        if u.scheme != "ws":
            raise Unreachable("the browser's address is ws://host:port, not %s" % url)
        try:
            self.s = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        except OSError as ex:
            raise Unreachable("the browser at %s does not answer (%s)" % (url, ex))
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (u.path or "/", AGENT_HOST, key)).encode())
        head = b""
        while b"\r\n\r\n" not in head:
            got = self.s.recv(4096)
            if not got:
                raise Unreachable("the browser at %s closed the connection" % url)
            head += got
        head, self.rest = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise Unreachable("the browser at %s refused the session: %s" % (url, head.split(b"\r\n", 1)[0].decode()))

    def send(self, text):
        data, mask = text.encode(), os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                                else bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n):
        while len(self.rest) < n:
            got = self.s.recv(1 << 16)
            if not got:
                raise Unreachable("the browser closed the connection")
            self.rest += got
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def recv(self):
        parts = []
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7f
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            op = b0 & 0x0f
            if op == 8:
                raise Unreachable("the browser closed the session")
            if op == 9:                               # a ping: its pong
                self.s.sendall(bytes([0x8a, 0x80]) + os.urandom(4))
                continue
            if op in (1, 2, 0):
                parts.append(data)
                if b0 & 0x80:
                    return b"".join(parts).decode()

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


class Browser:
    """The server's one BiDi session on the companion's Firefox — the companion gives each connection a Firefox of its
    own, so a server keeps one for its life and each conversation a tab in it (▶ `tab`)."""

    def __init__(self, address):
        self.address, self.lock, self.n = address, threading.Lock(), 0
        self.ws = _Socket(address.rstrip("/") + "/session")
        self.call("session.new", {"capabilities": {}})
        self.spare = [c["context"] for c in self.call("browsingContext.getTree", {}).get("contexts", [])]

    def call(self, method, params):
        with self.lock:
            self.n += 1
            n = self.n
            try:
                self.ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    m = json.loads(self.ws.recv())
                    if m.get("id") == n:
                        break                         # else an event, or an old answer
            except OSError as ex:
                raise Unreachable("the browser's connection broke (%s)" % ex)
        if m.get("type") == "error":
            raise Unreachable("%s: %s — %s" % (method, m.get("error"), m.get("message", "")))
        return m.get("result", {})

    def new_context(self):
        if self.spare:                                # Firefox's first tab, the first conversation's
            return self.spare.pop()
        return self.call("browsingContext.create", {"type": "tab"})["context"]


_BROWSERS = {}                    # address -> the server's session there
_BROWSERS_LOCK = threading.Lock()


def _browser(address, fresh=False):
    with _BROWSERS_LOCK:
        b = _BROWSERS.get(address)
        if b is None or fresh:
            if b is not None:
                b.ws.close()
            b = _BROWSERS[address] = Browser(address)
        return b


def session(address):
    """The server's session at `address`, opened where it has none."""
    return _browser(address)


def tab(address):
    """A tab of its own for a conversation, in the server's session at `address`."""
    return Tab(address)


class Tab:
    """A conversation's tab: its context, its viewport VIEWPORT, the pointer where it was put. A command finding the
    session gone — the companion restarted — opens the session again with a fresh tab, and says the page was lost."""

    def __init__(self, address):
        self.address, self.pointer_at = address, (VIEWPORT[0] // 2, VIEWPORT[1] // 2)
        self._open(_browser(address))

    def _open(self, b):
        self.b, self.context = b, b.new_context()
        self.b.call("browsingContext.setViewport", {"context": self.context,
                                                    "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]}})
        # the pointer where `pointer_at` says, the middle: a new context's starts at 0,0, and a glide sets out from it
        self.b.call("input.performActions", {"context": self.context, "actions": [{"type": "pointer", "id": "mouse",
                    "actions": [{"type": "pointerMove", "x": VIEWPORT[0] // 2, "y": VIEWPORT[1] // 2}]}]})

    def call(self, method, params):
        try:
            return self.b.call(method, params)
        except Unreachable as ex:
            if not any(w in str(ex) for w in ("connection broke", "closed", "no such frame")):
                raise
        self._open(_browser(self.address, fresh=True))
        self.pointer_at = (VIEWPORT[0] // 2, VIEWPORT[1] // 2)
        raise Unreachable("the browser was started again, and the page with it is gone: open it again")

    # ── the page ───────────────────────────────────────────────────────────────────────────────────────────────
    def open(self, url):
        self.call("browsingContext.navigate", {"context": self.context, "url": url, "wait": "complete"})
        return self.where()

    def back(self):
        self.call("browsingContext.traverseHistory", {"context": self.context, "delta": -1})
        return self.where()

    def evaluate(self, js):
        r = self.call("script.evaluate", {"expression": js, "target": {"context": self.context}, "awaitPromise": True})
        if r.get("type") == "exception":
            raise Unreachable("the page's script: %s" % r.get("exceptionDetails", {}).get("text", "failed"))
        return r.get("result", {}).get("value")

    def settle(self, url_before, grace=1.5, most=8.0):
        """Where an action leads to another page — a form sent, a link followed, a moment after the action — the new page's
        load waited for, `document.readyState` complete, up to `most` seconds; where the address has not changed within
        `grace` seconds, none. Answers whether it did. ⛳ A WINDOW, NOT A LOOK: Wikipedia's search, Enter pressed in its
        box, leaves its page well after the key (MEASURED 2026-10-31 on node03: a look at once found the old page)."""
        import time
        t0, moved = time.time(), False
        while time.time() - t0 < most:
            try:
                state = self.evaluate("JSON.stringify([location.href, document.readyState])")
            except Unreachable:                       # between two pages: their context briefly gone
                time.sleep(0.2)
                continue
            url, ready = json.loads(state) if state else ("", "")
            moved = moved or url != url_before
            if moved and ready == "complete":
                return True
            if not moved and time.time() - t0 > grace:
                return False
            time.sleep(0.2)
        return moved

    def where(self):
        got = self.evaluate("JSON.stringify([location.href, document.title])")
        return tuple(json.loads(got)) if got else ("", "")

    def text(self):
        return self.evaluate("document.body ? document.body.innerText : ''") or ""

    def elements(self):
        return json.loads(self.evaluate(ELEMENTS_JS) or "[]")

    def extent(self):
        """(the view's top, its height, the page's height), in the page's pixels."""
        return tuple(json.loads(self.evaluate(EXTENT_JS) or "[0, 0, 0]"))

    def focused(self, x, y):
        """The field the cursor is in — (kind, words, whether the pixel x, y lies inside it) — or None."""
        got = self.evaluate(FOCUSED_JS % (int(x), int(y)))
        return tuple(json.loads(got)) if got else None

    def install_userscript(self, url):
        """A public userscript installed in this session's Violentmonkey: its address opened — Violentmonkey takes it and
        opens its confirmation in a tab of its own — that tab's Install pressed, and the tab closed. Answers its name."""
        import time
        try:
            self.call("browsingContext.navigate", {"context": self.context, "url": url, "wait": "complete"})
        except Unreachable as ex:
            if "NS_ERROR_ABORT" not in str(ex):       # Violentmonkey took the address: the navigation is aborted
                raise
        confirm, t0 = None, time.time()
        while confirm is None and time.time() - t0 < 10:
            confirm = next((c["context"] for c in self.call("browsingContext.getTree", {}).get("contexts", [])
                            if c.get("url", "").startswith(VIOLENTMONKEY + "confirm/")), None)
            time.sleep(0.3)
        if confirm is None:
            raise Unreachable("Violentmonkey did not take %s: a userscript's address ends .user.js" % url)
        js = """(async () => {
          for (let i = 0; i < 50; i++) {
            const b = [...document.querySelectorAll('button')].find((b) => /^(install|reinstall|update)/i.test(b.innerText.trim()));
            const name = ((document.querySelector('h1, .name, [class*=name]') || {}).innerText || '').trim().split('\\n').pop();
            if (b && !b.disabled) { b.click(); await new Promise((r) => setTimeout(r, 800)); return name.trim() || 'installed'; }
            await new Promise((r) => setTimeout(r, 200));
          }
          return '';
        })()"""
        got = self.b.call("script.evaluate", {"expression": js, "target": {"context": confirm}, "awaitPromise": True})
        name = got.get("result", {}).get("value", "")
        try:
            self.b.call("browsingContext.close", {"context": confirm})
        except Unreachable:
            pass
        if not name:
            raise Unreachable("Violentmonkey's confirmation had no Install to press")
        return name

    def add_userscript(self, code):
        """A userscript's text run in this tab's pages from the next one loaded, as its header asks (USERSCRIPT_JS).
        Answers (its name, the patterns it runs on)."""
        h = userscript_header(code)
        take = h.get("match", []) + h.get("include", [])
        if not take:
            raise ValueError("a userscript names the pages it runs on: // @match https://example.org/*")
        name = (h.get("name") or ["a userscript"])[0]
        fn = USERSCRIPT_JS % dict(take=json.dumps(take), skip=json.dumps(h.get("exclude", []) + h.get("exclude-match", [])),
                                  at=json.dumps((h.get("run-at") or ["document-end"])[0]), body=code,
                                  name=name.replace("'", "").replace("\\", ""))
        self.call("script.addPreloadScript", {"functionDeclaration": "() => { %s(); }" % fn, "contexts": [self.context]})
        return name, take

    def grab(self):
        from PIL import Image
        data = self.call("browsingContext.captureScreenshot", {"context": self.context})["data"]
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def size(self):
        return VIEWPORT

    def pointer(self):
        return self.pointer_at

    # ── the pointer and the keys ───────────────────────────────────────────────────────────────────────────────
    def _act(self, source):
        self.call("input.performActions", {"context": self.context, "actions": [source]})

    def move(self, x, y):
        """The pointer to the viewport's pixel (x, y), gliding: along the straight line from where it is, a move every
        GLIDE_STEP pixels, at a hand's pace — asked as 150 ms and half a millisecond a pixel, at most 0.6 s; `MEASURED`
        (node03, the companion's Firefox, each with the call's round trip) 100 px 0.24 s, 564 px 0.74-0.92 s. What lies
        on the way sees it pass as it sees a person's: a menu that closes when the pointer leaves it closes, where a
        jump straight to the target — one move, as before — leaves it open."""
        x, y = max(0, min(int(x), VIEWPORT[0] - 1)), max(0, min(int(y), VIEWPORT[1] - 1))
        x0, y0 = self.pointer_at
        d = math.hypot(x - x0, y - y0)
        n = max(1, int(d // GLIDE_STEP))
        each = int(min(600.0, 150.0 + d / 2) / n) if d else 0
        self._act({"type": "pointer", "id": "mouse", "actions": [
            {"type": "pointerMove", "x": round(x0 + (x - x0) * i / n), "y": round(y0 + (y - y0) * i / n), "duration": each}
            for i in range(1, n + 1)]})
        self.pointer_at = (x, y)
        return x, y

    def click(self, button=1, n=1):
        """`button` as the X11 screen numbers them — 1 left, 2 middle, 3 right — n times where the pointer is."""
        b, (x, y) = {1: 0, 2: 1, 3: 2}[button], self.pointer_at
        steps = [{"type": "pointerMove", "x": x, "y": y}]
        for _ in range(n):
            steps += [{"type": "pointerDown", "button": b}, {"type": "pointerUp", "button": b}]
        self._act({"type": "pointer", "id": "mouse", "actions": steps})

    def scroll(self, n):
        x, y = self.pointer_at
        self._act({"type": "wheel", "id": "wheel", "actions": [{"type": "scroll", "x": x, "y": y, "deltaX": 0,
                                                                  "deltaY": int(n) * NOTCH}]})
    def type(self, text):
        steps = []
        for ch in text:
            ch = "" if ch == "\n" else ch
            steps += [{"type": "keyDown", "value": ch}, {"type": "keyUp", "value": ch}]
        self._act({"type": "key", "id": "keyboard", "actions": steps})

    def key(self, chord):
        parts = [p for p in chord.replace(" ", "").split("+") if p] or ([chord] if chord == "+" else [])
        if not parts:
            raise ValueError("a key's name, or a chord as \"ctrl+l\"")
        *mods, last = parts
        values = []
        for m in mods:
            if m.lower() not in MODIFIERS:
                raise ValueError("%r is not a modifier: ctrl, shift, alt, meta" % m)
            values.append(MODIFIERS[m.lower()])
        k = KEYS.get(last.lower(), last if len(last) == 1 else None)
        if k is None:
            raise ValueError("no key is named %r" % last)
        steps = [{"type": "keyDown", "value": v} for v in values] + [{"type": "keyDown", "value": k},
                                                                      {"type": "keyUp", "value": k}]
        steps += [{"type": "keyUp", "value": v} for v in reversed(values)]
        self._act({"type": "key", "id": "keyboard", "actions": steps})
