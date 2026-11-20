"""WHOSE NAME A REQUEST COMES UNDER — the UI's check of a request's `Host`, where it answers with no sign-in of its own:
a machine whose first account is still to be made (▶ silvann_ui_parts/sign_in.py's `first_account_page`), and a UI that
talks to a server elsewhere and keeps that server's sign-in for its browser.

⚖ 2026-11-19, the code review's row 1: *"that is true as a possible escalation vector"*. Where the UI answers whoever
reaches it, it listens on this machine alone — but a browser on this machine reaches it for any page it has open. A
page served under a name of its author's can have that name answer 127.0.0.1 a moment later (DNS rebinding): the
browser then takes this machine's UI for the page's own site, and the page's script asks it anything — the machine's
first account made the page's own, a message sent, and through the model's tools a command run. What gives such a page
away is the name: the browser sends it as the request's `Host`, and it is the author's, never this machine's.

So a request is answered when its `Host` is
- an address written as numbers (`127.0.0.1:8765`, `[::1]:8080`, `192.0.2.7:8090`) — a name is what rebinding needs;
- `localhost`, or a name under it;
- this machine's own name, as the system has it, bare, in full and under `.local`;
- the name the server or the UI was told to listen on (`--host`), where it is one;
- a name `SILVANN_HOSTS` lists (`SILVANN_HOSTS=chat.example.org,box`) — a name of your own for this machine, a
  container's service name;
and refused otherwise, saying so. A request with no `Host` is no browser's and is answered.

⛳ ONLY WHERE NO ONE SIGNS IN (`applies`): once the machine has its account, a request without a sign-in is given the
sign-in page whatever name it comes under, and a page of another site holds no sign-in of this one — while a machine
behind a proxy is reached under the proxy's name, which nothing here knows. The server asks every request for a
sign-in and has no use for this check.
"""
import ipaddress
import os
import socket


def own_names(listen_host=None, extra=None):
    """The names this machine answers to: its own, `listen_host` where it is a name, and `extra` — `SILVANN_HOSTS` in the
    environment when none is given — lowercase, without a last dot."""
    names = {"localhost"}
    try:
        host = socket.gethostname()
        names.update((host, host.split(".")[0], host.split(".")[0] + ".local", socket.getfqdn()))
    except OSError:
        pass
    if listen_host:
        names.add(str(listen_host))
    said = extra if extra is not None else os.environ.get("SILVANN_HOSTS", "")
    names.update(said.split(",") if isinstance(said, str) else said)
    return {n.strip().strip("[]").rstrip(".").lower() for n in names if n and n.strip()}


def host_of(header):
    """The host a `Host` header names, without its port: `"box:8765"` -> `"box"`, `"[::1]:8080"` -> `"::1"`."""
    text = (header or "").strip()
    if text.startswith("["):
        return text[1:].split("]", 1)[0].lower()
    name, colon, port = text.rpartition(":")
    return (name if colon and port.isdigit() else text).rstrip(".").lower()


def refused(header, names):
    """Why a request whose `Host` is `header` is not answered — None where it is: no `Host` at all, an address, a name
    under localhost, or one of `names`."""
    host = host_of(header)
    if not host or host in names or host.endswith(".localhost"):
        return None
    try:
        ipaddress.ip_address(host.split("%")[0])
        return None
    except ValueError:
        pass
    return ("this request names %r, which is not this machine: until it has its account SiLVANN answers under its own "
            "address and names alone. If %s is a name of yours for this machine, start it with SILVANN_HOSTS=%s"
            % (host, host, host))


class OwnHosts:
    """The check as a layer over an ASGI app (the UI's): a request, or a socket, that `refused` names is answered 403
    and goes no further. `applies()` says whether it is asked now — whether no one signs in."""

    def __init__(self, app, names, applies=lambda: True):
        self.app, self.names, self.applies = app, names, applies

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and self.applies():
            host = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k.lower() == b"host"), "")
            why = refused(host, self.names)
            if why and scope["type"] == "websocket":
                await receive()                                # its connect, which the close answers
                return await send({"type": "websocket.close", "code": 1008})
            if why:
                body = why.encode()
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                        (b"content-length", str(len(body)).encode())]})
                return await send({"type": "http.response.body", "body": body})
        await self.app(scope, receive, send)
