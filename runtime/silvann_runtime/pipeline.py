"""A MODEL SPLIT OVER MACHINES — a ring: each machine holds a run of the layers, and a position goes round it.

⚖ *"we need to test pipeline parallelism, to see if two 16gb pc with 32gb of ram can run glm or three 8gb pc can run
27b"* · 2026-10-14: *"the file should also have a pipeline_parallelism: no setting followed by pipeline_parallelism_next
… and the layers used … so you can realistically pull together two gaming pcs and run glm from ram"* · *"ring but first
is autodetected by starting at layer 0 so the setting is no/yes"*.

Every machine runs the server with the same model and a config naming its run of the layers and the machine after it:
```
  "pipeline_parallelism": "yes", "pipeline_parallelism_layers": "0-23", "pipeline_parallelism_next": "192.0.2.12:8766"
```
The machine whose layers begin at 0 is the FIRST: it holds the conversations and answers the UI, and it embeds the token.
The one whose layers end at the model's last is the LAST: it runs the head and draws the next token — on its card, with
the sampler the request carries — and hands it to its next, the first, which closes the ring. Each listens on
`pipeline_parallelism_port` (DEFAULT_PORT) for the machine before it.
```
  a position   token -> first's layers -> its residual -> the next's layers -> … -> the last's -> the token -> the first
  a prompt     the same, a chunk of rows at a time (the smallest chunk any machine reads)
```
What crosses a link is the residual: a row of halves a position (4 KB for the 35B, 10 KB for the 27B, GLM's streams 32
KB), or a chunk's rows. Each machine keeps its own layers' caches — so a conversation's state is spread over the machines,
and one brought back after another ran reads its tokens again (`Pipeline.PARKS`).
"""
import argparse
import json
import socket
import struct
import sys
import time

import numpy as np

from .model_folder import Refused

DEFAULT_PORT = 8766
HEAD = struct.Struct("<4sqqqq")          # what, three integers, the payload's length


def _exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        got = sock.recv(min(n - len(buf), 1 << 20))
        if not got:
            raise ConnectionError("the machine before this one closed the link")
        buf += got
    return bytes(buf)


def layers_of(text, total):
    """`"first-last"` (inclusive) as the layers' list, within the model's `total`."""
    try:
        a, b = (int(x) for x in str(text).split("-"))
    except ValueError:
        raise Refused("pipeline_parallelism_layers is first-last, as \"0-23\", not %r" % (text,))
    if not 0 <= a <= b < total:
        raise Refused("the layers %d-%d are not within the model's 0-%d" % (a, b, total - 1))
    return list(range(a, b + 1))


def address(text):
    """`"host:port"` (or `"host"`, on DEFAULT_PORT) as (host, port)."""
    host, _, port = str(text).rpartition(":") if ":" in str(text) else (str(text), "", "")
    return (host or str(text)), int(port or DEFAULT_PORT)


def chunk_of(model):
    """The most rows the model reads a prompt in at once."""
    return int(getattr(model, "CH", None) or getattr(model, "CHUNK", None) or 64)


class Link:
    """This machine's place in the ring: it listens for the machine before it and connects to the one after."""

    def __init__(self, port, nxt):
        self.nxt = address(nxt)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("0.0.0.0", int(port)))
        self.listener.listen(1)
        self.out = self.inn = None

    def join(self, say=print):
        """Connected to the next machine — tried until it listens — then the one before accepted."""
        say("[pipeline] connecting to the next machine, %s:%d" % self.nxt)
        while self.out is None:
            try:
                self.out = socket.create_connection(self.nxt, timeout=5)
                self.out.settimeout(None)
            except OSError:
                time.sleep(1)
        self.out.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        say("[pipeline] waiting for the machine before this one, on port %d" % self.listener.getsockname()[1])
        self.inn, peer = self.listener.accept()
        self.inn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        say("[pipeline] the ring is joined: %s before, %s:%d after" % (peer[0], self.nxt[0], self.nxt[1]))

    def send(self, what, a=0, b=0, c=0, payload=b""):
        self.out.sendall(HEAD.pack(what, a, b, c, len(payload)) + payload)

    def receive(self):
        what, a, b, c, n = HEAD.unpack(_exact(self.inn, HEAD.size))
        return what, a, b, c, _exact(self.inn, n)


def _hello(model, last):
    return {"model": model.folder.identity(), "max_context": int(model.max_context), "chunk": chunk_of(model),
            "last": bool(last)}


def stage(model, link, last, say=print):
    """A machine past the first, for as long as the ring stands: each request from the machine before is run through this
    machine's layers and handed on — the last's as the token (`TOKN`), or the logits for a sampler on the first (`LOGT`).
      HELO   what each machine runs, checked against the first's — the chunk the smallest of all
      RSET   a new conversation · SEEN the recent tokens, for the last's sampler · QUIT the ring taken down
      STEP   a position (a, its residual) · ROWS a chunk (a its first position, b its rows) — the head expression the
             last runs, its `c` bytes, before the rows"""
    link.join(say)
    while True:
        try:
            what, a, b, c, payload = link.receive()
        except ConnectionError:
            say("[pipeline] the ring is down: the machine before this one closed the link")
            return
        if what == b"HELO":
            want = json.loads(payload)
            mine = _hello(model, last)
            if want["model"] != mine["model"] or want["max_context"] != mine["max_context"]:
                raise Refused("the ring's machines run different models or contexts: %s here, %s on the first"
                              % ({k: mine[k] for k in ("model", "max_context")}, {k: want[k] for k in ("model", "max_context")}))
            want["chunk"] = min(want["chunk"], mine["chunk"])
            want["last"] = want["last"] or last
            link.send(b"HELO", payload=json.dumps(want).encode())
        elif what == b"RSET":
            model.reset()
            link.send(b"RSET")
        elif what == b"SEEN":
            if last:
                from . import sampling
                sampling.remember(model.m, np.frombuffer(payload, dtype="<i8").tolist())
            link.send(b"SEEN", payload=payload)
        elif what in (b"STEP", b"ROWS"):
            head, x = payload[:c].decode(), payload[c:]
            if what == b"STEP":
                got = model.forward(a, x=x, head=(head or True) if last else False)
            else:
                got = model.forward_rows(a, xs=x, n=b, head=(head or False) if last else False)
            if not last:
                link.send(what, a, b, c, payload[:c] + got)
            elif not head:
                link.send(b"DONE")                                  # a prompt's chunk before its last: no token yet
            elif head == "(head_logits)":
                link.send(b"LOGT", payload=np.ascontiguousarray(model.logits(), dtype="<f4").tobytes())
            else:
                link.send(b"TOKN", int(got))
        elif what == b"QUIT":
            link.send(b"QUIT")
            return
        else:
            raise Refused("the ring sent %r, which no machine sends" % (what,))


class Pipeline:
    """The ring's first machine as the server's model: its own layers here, the rest round the ring. A model's `reset`,
    `step`, `prefill` and `logits`; the rest is the first machine's own model's (`vision`, `folder`, …)."""
    GENERATES = False                # a token at a time round the ring: no answer loop on one card
    PARKS = False                    # its state is spread over the machines: a conversation brought back reads its tokens again

    def __init__(self, model, link, say=print):
        self.model, self.link = model, link
        link.join(say)
        link.send(b"HELO", payload=json.dumps(_hello(model, False)).encode())
        what, _, _, _, payload = link.receive()
        if what != b"HELO":
            raise Refused("the ring answered %r to its hello" % (what,))
        ring = json.loads(payload)
        if not ring["last"]:
            raise Refused("no machine in the ring runs the model's last layer")
        self.chunk = ring["chunk"]
        self._logits = None
        say("[pipeline] the ring answers: chunks of %d rows, the context %d" % (self.chunk, model.max_context))

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _round(self, what, a=0, b=0, payload=b"", head=""):
        """A request round the ring, and the last machine's answer to it."""
        h = head.encode()
        self.link.send(what, a, b, len(h), h + payload)
        got, value, _, _, back = self.link.receive()
        if got == b"LOGT":
            self._logits = np.frombuffer(back, dtype="<f4")
            return None
        return value if got == b"TOKN" else None

    @staticmethod
    def _head(greedy, sample):
        return sample or ("(head)" if greedy else "(head_logits)")

    def reset(self):
        self.model.reset()
        self._round(b"RSET")

    def remember(self, tokens):
        """The recent tokens, for the last machine's sampler's penalty (▶ `sampling.remember`)."""
        self._round(b"SEEN", payload=np.asarray(tokens, dtype="<i8").tobytes())

    def step(self, token, pos, greedy=True, sample=None, lag=0):
        if lag:
            raise Refused("a Qwen model's positions after a picture are read on one machine, not round a ring")
        if pos >= self.model.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.model.max_context))
        x = self.model.forward(pos, token=token)
        return self._round(b"STEP", pos, 0, x, self._head(greedy, sample))

    def prefill(self, tokens, first, greedy=True, sample=None, positions=None, rows=None):
        if positions is not None:
            raise Refused("a Qwen model's pictures are read on one machine, not round a ring")
        got = None
        for at in range(0, len(tokens), self.chunk):
            part = tokens[at:at + self.chunk]
            mine = {i - at: r for i, r in (rows or {}).items() if at <= i < at + len(part)}
            xs = self.model.forward_rows(first + at, tokens=part, rows=mine or None)
            last = at + len(part) >= len(tokens)
            got = self._round(b"ROWS", first + at, len(part), xs, self._head(greedy, sample) if last else "")
        return got

    def logits(self):
        return self._logits

    def vision(self):
        """The first machine's tower: GLM's pictures go round the ring as rows over their pads; a Qwen model's need their
        rotary positions on every machine, and are read on one machine only, for now."""
        v = self.model.vision()
        if getattr(v, "mrope", False):
            raise Refused("a Qwen model's pictures are read on one machine, not round a ring")
        return v

    def state_regions(self, length):
        return []

    def close(self):
        self.link.send(b"QUIT")
        self.link.receive()


def main(argv):
    """A machine of the ring from the command line, as the server runs one (a check's stage)."""
    ap = argparse.ArgumentParser(prog="silvann_runtime.pipeline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True, help="first-last, inclusive")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--next", required=True, help="host:port of the machine after this one")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--options", default="{}")
    a = ap.parse_args(argv)
    from . import open_model
    from .model_folder import ModelFolder
    total = model_layers(ModelFolder(a.model))
    layers = layers_of(a.layers, total)
    if layers[0] == 0:
        raise SystemExit("the first machine is the server's (start_silvann_server.py), not a stage")
    model = open_model(a.model, max_context=a.max_context, layers=layers, **json.loads(a.options))
    link = Link(a.port, a.next)
    print("stage ready on port %d" % a.port, flush=True)
    stage(model, link, layers[-1] == total - 1)
    model.m.shutdown()


def model_layers(folder):
    """How many layers the model has (its config's `num_hidden_layers`)."""
    c = folder.config
    return int((c.get("text_config") or c)["num_hidden_layers"])


if __name__ == "__main__":
    main(sys.argv[1:])
