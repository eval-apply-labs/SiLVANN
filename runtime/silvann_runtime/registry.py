"""A SERVER IN THE REGISTRY (NN-130) — the client of main/silvann_registry.py: a server offering itself there while it
runs (`Offer`), and the listing the REPL's (agent …) and (whoami) read (`listing`). Both take the tools file's `agent`
entry's settings (▶ tools_config.built_in): `registry`, its address; `key`, its key; `offer`, whether this server offers
itself; `address`, where the others reach it. ▶ main/silvann_registry.py for the registry and its API.
"""
import ipaddress
import socket
import threading
import urllib.parse

import httpx

TIMEOUT = 5


class Unreached(Exception):
    """The registry did not answer, or refused: the line says why."""


def _call(method, registry, key, json=None):
    url = str(registry).rstrip("/") + "/servers"
    try:
        r = httpx.request(method, url, json=json, timeout=TIMEOUT, headers={"Authorization": "Bearer %s" % (key or "")})
    except httpx.HTTPError as ex:
        raise Unreached("%s does not answer (%s)" % (registry, type(ex).__name__))
    if r.status_code != 200:
        try:
            why = r.json().get("error")
        except ValueError:
            why = r.text[:200]
        raise Unreached("%s answered %d: %s" % (registry, r.status_code, why))
    return r.json()


def listing(settings):
    """The servers the registry of `settings` lists — [{address, model_id, architecture, loras, max_context, busy,
    pictures, …, seen}] — or [] where the entry names none. Unreached where it does not answer or refuses."""
    if not settings.get("registry"):
        return []
    return list(_call("GET", settings["registry"], settings.get("key")).get("servers") or [])


def _local(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _route_from(registry):
    """The address this machine's route to the registry leaves from — a UDP socket's own address once connected, no
    packet sent."""
    u = urllib.parse.urlsplit(str(registry))
    family = socket.AF_INET6 if u.hostname and ":" in u.hostname else socket.AF_INET
    s = socket.socket(family, socket.SOCK_DGRAM)
    try:
        s.connect((u.hostname, u.port or 80))
        return s.getsockname()[0]
    finally:
        s.close()


def address_of(settings, host, port):
    """Where the registry's servers reach this one, listening on `host`:`port` — (address, None), or (None, why) where
    they cannot. The entry's `address` where it names one; "auto": an address of its own, every one (0.0.0.0, ::) taken
    as the one the route to the registry leaves from — and 127.0.0.1 only where the registry is on this machine too."""
    said = str(settings.get("address") or "auto")
    if said != "auto":
        return said.rstrip("/"), None
    registry = str(settings["registry"])
    there = urllib.parse.urlsplit(registry).hostname or ""
    if _local(host) and not _local(there):
        return None, ("this server listens on %s alone, which %s cannot reach: start it with --host 0.0.0.0, or name "
                      "the agent entry's address" % (host, registry))
    if host in ("0.0.0.0", "::", ""):
        try:
            host = _route_from(registry)
        except OSError as ex:
            return None, "no route to %s from this machine (%s)" % (registry, ex)
    return "http://%s:%d" % ("[%s]" % host if ":" in host else host, port), None


class Offer(threading.Thread):
    """This server offered to the registry its settings name, renewed while it runs, withdrawn by `stop`. `settings()`
    is read at each renewal — a rebuild of the tools that names another registry, or none, is followed at the next one,
    or at once on `wake` — and `serving()` says what the offer carries. `said(line)` hears when the offer is made,
    fails, comes back or is withdrawn: not each renewal."""

    def __init__(self, settings, serving, host, port, said):
        super().__init__(daemon=True, name="registry offer")
        self.settings, self.serving, self.host, self.port, self.said = settings, serving, host, port, said
        self.woken, self.stopped = threading.Event(), threading.Event()
        self.lock = threading.Lock()              # a renewal and the withdrawal at the stop, one at a time
        self.at = None                            # (registry, key, address) where it is offered, or None
        self.state = None                         # the last line said, so a line is said once
        self.address = None                       # this server's address as offered, which the listing leaves out

    def _say(self, line):
        if line != self.state:
            self.state = line
            self.said(line)

    def _withdraw(self):
        if self.at is None:
            return
        registry, key, address = self.at
        self.at = None
        try:
            _call("DELETE", registry, key, {"address": address})
            self._say("withdrawn from the registry %s" % registry)
        except Unreached as ex:
            self._say("not withdrawn from the registry: %s — it leaves the listing as its offer runs out" % ex)

    def once(self):
        """One renewal — or the first offer, or the withdrawal the settings now ask for. Answers the seconds to the
        next."""
        with self.lock:
            return 0 if self.stopped.is_set() else self._once()

    def _once(self):
        s = self.settings()
        if not s.get("registry"):                     # none named: nothing to say, but a withdrawal
            self._withdraw()
            return 30
        if str(s.get("offer", "yes")).lower() == "no":
            self._withdraw()
            self._say("not offered to the registry %s: the agent entry's offer is \"no\"" % s["registry"])
            return 30
        address, why = address_of(s, self.host, self.port)
        if address is None:
            self._withdraw()
            self._say("not offered to the registry %s: %s" % (s["registry"], why))
            return 30
        at = (str(s["registry"]), s.get("key"), address)
        if self.at is not None and self.at != at:
            self._withdraw()
        try:
            got = _call("PUT", at[0], at[1], dict(self.serving(), address=address))
        except Unreached as ex:
            self._say("not offered: the registry %s" % ex)
            return 15
        self.at, self.address = at, address
        self._say("offered to the registry %s as %s" % (at[0], address))
        return max(1.0, float(got.get("ttl") or 90) / 3)

    def run(self):
        while not self.stopped.is_set():
            try:
                wait = self.once()
            except Exception as ex:                   # noqa: BLE001 — the offer is the server's aside, never its end
                self._say("the registry's offer failed: %s" % ex)
                wait = 30
            self.woken.wait(wait)
            self.woken.clear()

    def wake(self):
        self.woken.set()

    def stop(self):
        """Withdrawn and stopped — the server is stopping."""
        self.stopped.set()
        self.woken.set()
        with self.lock:
            self._withdraw()
