"""THE PICTURE WORKER — a process of its own that makes a picture model's pictures, one at a time, for the server.

A picture's step is one engine call of seconds, and an engine call holds Python's lock: in the server's own process the
stream, the UI's heartbeat and every other request would wait out each step. So the server (▶ `image_session`) starts
this process once and hands it each picture: the engine is booted here, never in the server, and a fault ends this
process rather than the server. Its 8-bit table (▶ `nn_unpack_model._levels`) is made when it starts, before the first
picture asks for it.

The two talk over a socket pair, as pickled tuples (`multiprocessing.connection.Connection`):
  server -> worker   ("picture", prompt, settings, folder) · ("stop",) while one is being made · ("quit",)
  worker -> server   ("ready",) once · ("progress", stage, done, total) · ("done", name, seconds) · ("stopped", why)
                     · ("error", text)
"""
import os
import sys
import threading
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np


def serve(conn, path, silicon=None):
    from .qwen_image import Pipeline, Stopped
    pipeline = Pipeline(path, silicon=silicon or None)
    threading.Thread(target=lambda: __import__("nn_unpack_model")._levels(8), daemon=True).start()
    conn.send(("ready",))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "quit":
            return
        if msg[0] != "picture":
            continue
        _, prompt, s, folder = msg
        stop = []

        def tell(stage, done, total):
            while conn.poll():                       # a stop asked between steps
                if conn.recv()[0] == "stop":
                    stop.append(True)
            conn.send(("progress", stage, done, total))
            return not stop
        t0 = time.time()
        try:
            from PIL import Image
            pixels = pipeline.picture(prompt, height=s["height"], width=s["width"], steps=s["steps"], seed=s["seed"],
                                      progress=tell)
            os.makedirs(folder, exist_ok=True)
            name = "%s_%d.png" % (time.strftime("%Y%m%d-%H%M%S"), s["seed"])
            Image.fromarray(np.round((np.clip(pixels, -1, 1) / 2 + 0.5) * 255).astype(np.uint8), "RGBA").save(
                os.path.join(folder, name))
            conn.send(("done", name, time.time() - t0))
        except Stopped as ex:
            conn.send(("stopped", str(ex)))
        except Exception as ex:                      # noqa: BLE001 — said to the server, which says it to the UI
            traceback.print_exc()
            conn.send(("error", "%s: %s" % (type(ex).__name__, ex)))


def main():
    """`python -m silvann_runtime.image_worker PACK FD [SILICON]` — FD the worker's end of the socket pair."""
    conn = Connection(int(sys.argv[2]))
    serve(conn, sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)


if __name__ == "__main__":
    main()
