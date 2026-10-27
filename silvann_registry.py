#!/usr/bin/env python3
"""THE SERVERS' REGISTRY — which SiLVANN servers are up, what each serves and where, for the subagents (NN-130).

    silvann registry [--port 8770] [--host 0.0.0.0] [--key-file <data>/registry/key]
    venv/bin/python silvann_registry.py …            # the same, from the release's folder

⚖ 2026-10-26 *"it can be a specific port to communicate via api the addition or removal of oneself to the registry
and the endpoint has to be configurable in the config part of the tools, as it is effectively a setup of the tool to
know who is the supplier of the info. it could be a small program alongside silvann and part of the same docker image"*.
A server whose tools file's `agent` entry names this registry offers itself as it starts — its address and what it
serves — renews the offer while it runs, and withdraws it as it stops; one that stops without withdrawing leaves the
listing TTL seconds after its last renewal. The servers read the listing for the REPL's (agent …) and (whoami); the
model never reaches the registry, and never sees its key. Python's own library alone: it runs where no engine is built.

Its API, each call but /health with `Authorization: Bearer <key>`:
  PUT    /servers   {"address": "http://host:port", "model_id": …, …}   offered or renewed → {"ok", "ttl"}
  DELETE /servers   {"address": "http://host:port"}                      withdrawn → {"ok", "removed"}
  GET    /servers                                                        {"servers": [{…, "seen": seconds ago}], "ttl"}
  GET    /health                                                         {"ok", "servers"}: a container's check

⛔ THE KEY DECIDES WHO MAY OFFER A SERVER: an address listed receives the subagents' tasks, and a task carries what the
conversation gives it. So the registry always has a key — `SILVANN_REGISTRY_KEY`, or its key file's, written at its
first start where there is none, readable by its user alone, and said by its path, never printed.
"""
import argparse
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TTL = 90                        # seconds an offer stays listed without a renewal; a server renews it every TTL / 3
MAX_SERVERS = 512
MAX_BODY = 64 * 1024
ADDRESS = re.compile(r"^https?://[A-Za-z0-9._:\[\]-]+:\d{1,5}$")
# what an offer may say of its server, by its type; anything else in it is dropped
FIELDS = {"model_id": str, "architecture": str, "lora": str, "loras": list, "max_context": int, "cards": int,
          "busy": bool, "pictures": bool, "reads_pictures": bool}


def say(line):
    print("[silvann_registry] %s" % line, flush=True)


def key_of(path):
    """The registry's key: `SILVANN_REGISTRY_KEY`'s, else the key file's — made there, its user's alone, where it is
    not. Answers (key, where it came from)."""
    env = os.environ.get("SILVANN_REGISTRY_KEY", "").strip()
    if env:
        return env, "SILVANN_REGISTRY_KEY"
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        if not key:
            raise SystemExit("[silvann_registry] ⛔ not started: the key file %s is empty" % path)
        if os.name == "posix" and os.stat(path).st_mode & 0o077:
            say("⚠ the key file %s is readable by other users than its own (chmod 600 it)" % path)
        return key, path
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    say("a new key written in %s — each server's tools file names it as the agent entry's key" % path)
    return key, path


class Registry:
    """The servers offered, by address, each with what its offer said and when it was last renewed."""

    def __init__(self, ttl=TTL):
        self.ttl, self.lock, self.servers = ttl, threading.Lock(), {}

    def offer(self, body, peer):
        """An offer, or its renewal: (status, answer)."""
        address = str(body.get("address") or "").rstrip("/")
        if not ADDRESS.match(address):
            return 422, {"error": "address is http://host:port or https://host:port, not %r" % address[:200]}
        if not isinstance(body.get("model_id"), str) or not body["model_id"]:
            return 422, {"error": "model_id is the name of the model the server serves"}
        entry = {"address": address, "from": peer}
        for name, kind in FIELDS.items():
            value = body.get(name)
            if isinstance(value, kind) and not (kind is int and isinstance(value, bool)):
                entry[name] = value[:200] if kind is str else [str(v)[:200] for v in value[:64]] if kind is list \
                    else value
        with self.lock:
            new = address not in self.servers
            if new and len(self.servers) >= MAX_SERVERS:
                return 507, {"error": "the registry lists %d servers, its most" % MAX_SERVERS}
            entry["at"] = time.time()
            self.servers[address] = entry
        if new:
            say("offered: %s serves %s (from %s)" % (address, entry["model_id"], peer))
        return 200, {"ok": True, "ttl": self.ttl}

    def withdraw(self, body):
        address = str(body.get("address") or "").rstrip("/")
        with self.lock:
            gone = self.servers.pop(address, None)
        if gone:
            say("withdrawn: %s" % address)
        return 200, {"ok": True, "removed": gone is not None}

    def sweep(self):
        """The offers not renewed for `ttl` seconds taken out of the listing."""
        now = time.time()
        with self.lock:
            old = [a for a, e in self.servers.items() if now - e["at"] > self.ttl]
            for a in old:
                del self.servers[a]
        for a in old:
            say("left: %s (not renewed for %d s)" % (a, self.ttl))

    def listing(self):
        self.sweep()
        now = time.time()
        with self.lock:
            return [dict({k: v for k, v in e.items() if k != "at"}, seen=round(now - e["at"], 1))
                    for _, e in sorted(self.servers.items())]


def handler(registry, key):
    class Handler(BaseHTTPRequestHandler):
        server_version = "silvann_registry"

        def log_message(self, *args):            # the offers and the leavings are said; each call is not
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _keyed(self):
            given = self.headers.get("Authorization", "")
            if given.startswith("Bearer ") and hmac.compare_digest(given[7:].strip().encode(), key.encode()):
                return True
            self._send(401, {"error": "the registry's key is asked: Authorization: Bearer <key>"})
            return False

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._send(413, {"error": "an offer is %d bytes at most" % MAX_BODY})
                return None
            try:
                got = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                got = None
            if not isinstance(got, dict):
                self._send(400, {"error": "the body is a JSON object"})
                return None
            return got

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "servers": len(registry.listing())})
            if self.path != "/servers":
                return self._send(404, {"error": "GET /servers, or /health"})
            if self._keyed():
                self._send(200, {"servers": registry.listing(), "ttl": registry.ttl})

        def do_PUT(self):
            if self.path != "/servers":
                return self._send(404, {"error": "PUT /servers"})
            if self._keyed():
                body = self._body()
                if body is not None:
                    self._send(*registry.offer(body, self.client_address[0]))

        def do_DELETE(self):
            if self.path != "/servers":
                return self._send(404, {"error": "DELETE /servers"})
            if self._keyed():
                body = self._body()
                if body is not None:
                    self._send(*registry.withdraw(body))

    return Handler


def main(argv=None):
    data = os.environ.get("SILVANN_DATA") or os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="The registry the SiLVANN servers of a network offer themselves to.")
    p.add_argument("--host", default="0.0.0.0", help="the address it listens on (default every one)")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--key-file", default=os.path.join(data, "registry", "key"),
                   help="its key's file, made at the first start (default <the data folder>/registry/key); "
                        "SILVANN_REGISTRY_KEY in the environment instead")
    p.add_argument("--ttl", type=int, default=TTL, help="seconds an offer stays listed without a renewal")
    a = p.parse_args(argv)
    key, where = key_of(a.key_file)
    registry = Registry(a.ttl)
    httpd = ThreadingHTTPServer((a.host, a.port), handler(registry, key))
    httpd.daemon_threads = True

    def sweeping():
        while True:
            time.sleep(max(1, a.ttl / 3))
            registry.sweep()

    threading.Thread(target=sweeping, daemon=True).start()
    say("listening on %s:%d, the key %s's, an offer listed %d s past its last renewal" % (a.host, a.port, where, a.ttl))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
