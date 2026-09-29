"""A MODEL SPLIT OVER MACHINES — each holds a run of the layers, and a position goes through them in turn.

⚖ *"we need to test pipeline parallelism, to see if two 16gb pc with 32gb of ram can run glm or three 8gb pc can run
27b"*. A STAGE is a process on one machine: it boots the model with its layers only (`layers=`), the first stage also
embedding the token and the last also running the head. The COORDINATOR — the process that holds the conversation, the
server's — talks to each stage in turn over TCP:
```
  a position   token -> stage 0 -> its residual row -> stage 1 -> … -> the last stage -> the next token
  a prompt     the same, a chunk of rows at a time (the first stage takes the chunk's tokens)
```
What crosses the link is the residual: a row of H halves a position (4 KB for the 35B, 10 KB for the 27B), or a chunk's
rows. Each stage keeps its own layers' caches, so a conversation's state is spread over the machines.

  a stage:  python3 -m silvann_runtime.pipeline stage --model DIR --layers 0-21 --port 5001 [--first] [--last]
                                                      [--max-context N] [--options JSON]
"""
import argparse
import json
import socket
import struct
import sys

import numpy as np

HEAD = struct.Struct("<4sqqqq")          # what, three integers, the payload's length


def _exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        got = sock.recv(min(n - len(buf), 1 << 20))
        if not got:
            raise ConnectionError("the other end closed the link")
        buf += got
    return bytes(buf)


def send(sock, what, a=0, b=0, c=0, payload=b""):
    sock.sendall(HEAD.pack(what, a, b, c, len(payload)) + payload)


def receive(sock):
    what, a, b, c, n = HEAD.unpack(_exact(sock, HEAD.size))
    return what, a, b, c, _exact(sock, n)


# ── a stage ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def serve(model, port, first, last):
    """Answer one coordinator until it says QUIT: RSET · STEP (pos, token) · ROWS (first position, n)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(1)
    print("stage ready on port %d" % port, flush=True)
    sock, _ = listener.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        what, a, b, c, payload = receive(sock)
        if what == b"QUIT":
            send(sock, b"OKAY")
            return
        if what == b"RSET":
            model.reset()
            send(sock, b"OKAY")
        elif what == b"STEP":
            got = model.forward(a, token=b if first else None, x=None if first else payload, head=last)
            send(sock, b"TOKN", got) if last else send(sock, b"XROW", payload=got)
        elif what == b"ROWS":
            tokens = np.frombuffer(payload, dtype="<i4").tolist() if first else None
            got = model.forward_rows(a, tokens=tokens, xs=None if first else payload, n=b, head=last)
            send(sock, b"TOKN", got) if last else send(sock, b"XROW", payload=got)
        else:
            send(sock, b"FAIL", payload=("no such request %r" % what).encode())


# ── the coordinator ──────────────────────────────────────────────────────────────────────────────────────────────────
class Pipeline:
    """The stages, in order, as one model: `reset`, `step`, `prefill` as a model has them."""

    def __init__(self, stages, chunk=64):
        self.socks = []
        for host, port in stages:
            s = socket.create_connection((host, port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socks.append(s)
        self.chunk = chunk

    def _through(self, what, a, b, first_payload):
        payload = first_payload
        for s in self.socks:
            send(s, what, a, b, payload=payload)
            got, value, _, _, payload = receive(s)
            if got == b"FAIL":
                raise RuntimeError(payload.decode())
            if got == b"TOKN":
                return value
        raise RuntimeError("the last stage answered rows, not a token — is it started with --last?")

    def reset(self):
        for s in self.socks:
            send(s, b"RSET")
            receive(s)

    def step(self, token, pos):
        return self._through(b"STEP", pos, token, b"")

    def prefill(self, tokens, first):
        nxt = None
        for at in range(0, len(tokens), self.chunk):
            chunk = tokens[at:at + self.chunk]
            nxt = self._through(b"ROWS", first + at, len(chunk), np.asarray(chunk, dtype="<i4").tobytes())
        return nxt

    def close(self):
        for s in self.socks:
            try:
                send(s, b"QUIT")
                receive(s)
            finally:
                s.close()


def _layers(text):
    a, b = text.split("-")
    return list(range(int(a), int(b) + 1))


def main(argv):
    ap = argparse.ArgumentParser(prog="silvann_runtime.pipeline")
    sub = ap.add_subparsers(dest="role", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--model", required=True)
    st.add_argument("--layers", required=True, help="first-last, inclusive")
    st.add_argument("--port", type=int, required=True)
    st.add_argument("--first", action="store_true")
    st.add_argument("--last", action="store_true")
    st.add_argument("--max-context", type=int, default=4096)
    st.add_argument("--options", default="{}")
    a = ap.parse_args(argv)
    from . import open_model
    model = open_model(a.model, max_context=a.max_context, layers=_layers(a.layers), **json.loads(a.options))
    serve(model, a.port, a.first, a.last)
    model.m.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
