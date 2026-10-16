"""SiLVANN HTTP server — a model folder booted by `silvann_runtime`, and its conversations served over HTTP.

    python3 start_silvann_server.py --model models/<name> [<config.json>] [--port 8000] [--host 0.0.0.0]

The model folder is a packed bundle (`pack.json`, `layers/`, `global.pkl`, the model's own `config.json` and tokenizer);
one that is missing is asked for on a terminal and fetched from Hugging Face (`silvann_runtime.fetch`). The config — a
file in the folder's `configs/`, or a path — sets the runtime:
```
  {"max_context": 32768, "lora": null, "options": {...}, "sampler": {"temperature": 0.7, ...} | null, "thinking": true}
```
`lora` is the adapter the model boots with (⚖ *"at boot is enough … we say it in the json which lora we want"*), and a
conversation created with its own `lora` swaps it in for its turns. `options` go to the model's runtime (`kv`,
`fused_moe`, `sockets`, `experts_int8`, …). `sampler: null` is greedy.

Conversations are CARTRIDGES, one on the card at a time: a turn keeps the card until its answer is done, and a turn
asked for meanwhile is answered 503 with `Retry-After: 5` — ⚖ *"retry in 5 s is what the silvann ui will do (or the
requestor in general), it is the convention and it is the caller that will do it"*. On disk each lives in
`user_sessions/<model>/<id>/`: `cartridge.npz` (its tokens, and its state when it was put away), `meta.json` (what
model it belongs to) and `server_meta.json` (its name, history, system prompt and password).
"""
import argparse
import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
# the conversations go beside the release, or in the folder SILVANN_DATA names (the Docker image's volume)
DATA = Path(os.environ.get("SILVANN_DATA") or ROOT)
# the runtime beside this file in a release, or the development tree's
RUNTIME = Path(os.environ.get("SILVANN_RUNTIME", ROOT / "runtime" if (ROOT / "runtime").is_dir()
                              else ROOT.parent / "backstage" / "python"))
# the silvann_core.py beside this file, imported before the runtime leads the path: the development tree's runtime
# folder holds another package of that name
from silvann_core import default_config                                   # noqa: E402
sys.path.insert(0, str(RUNTIME))
import silvann_runtime as SR                                              # noqa: E402
from silvann_runtime import autotune, experts_file, tools              # noqa: E402
from silvann_runtime.chat import Chat, ToolMarkup                         # noqa: E402
from silvann_runtime.fetch import ensure_model                            # noqa: E402
from silvann_runtime.session import Busy, Cartridge, Runtime, sampler, state_bytes     # noqa: E402
from silvann_runtime import image_session                                 # noqa: E402
from silvann_runtime.qwen_image import Stopped, image_pack                 # noqa: E402
from silvann_runtime.vision import MAX_VIDEO_TOKENS                       # noqa: E402

IDLE_EVICTION_SECONDS = 120.0
EVICTION_TICK_SECONDS = 30.0


# ── requests ────────────────────────────────────────────────────────────────────────────────────────────────
class CreateConvReq(BaseModel):
    name: Optional[str] = None
    system: Optional[str] = None
    lora: Optional[str] = None
    cooling: Optional[dict] = None          # the old engine's cache knobs: accepted, and the runtime's cache used
    kv_cache_size: Optional[int] = None


class EstimateReq(BaseModel):
    cooling: Optional[dict] = None
    kv_cache_size: Optional[int] = None


class MessageReq(BaseModel):
    text: str
    max_new_tokens: int = 256
    images: list[str] = []


class UpdateReq(BaseModel):
    name: Optional[str] = None
    enable_thinking: Optional[bool] = None
    tools: Optional[bool] = None            # the model's tool: on or off, as the thinking switch (▶ _tools_on)
    tools_mode: Optional[str] = None        # what its commands may do unasked: "ask", "chill" or "full-auto"


class AnswerReq(BaseModel):
    choice: str                             # one of the ask's choices: "folder", "once", "refuse"
    max_new_tokens: int = 2048


class SaveReq(BaseModel):
    password: Optional[str] = None


# ── passwords: scrypt, as before ────────────────────────────────────────────────────────────────────────────
def _hash_password(password):
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return {"alg": "scrypt", "salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(h).decode(),
            "n": 16384, "r": 8, "p": 1, "dklen": 32}


def _verify_password(password, stored):
    try:
        h = hashlib.scrypt(password.encode(), salt=base64.b64decode(stored["salt"]), n=stored.get("n", 16384),
                           r=stored.get("r", 8), p=stored.get("p", 1), dklen=stored.get("dklen", 32))
        return hmac.compare_digest(h, base64.b64decode(stored["hash"]))
    except Exception:
        return False


# ── the server's state ──────────────────────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.model = self.runtime = self.chat = None
        self.config = {}
        self.model_id = ""
        self.dir = DATA / "user_sessions"
        self.owner = {}                     # cartridge id -> the web user who has it open
        self.last_ping = {}
        self.turns = {}                     # cartridge id -> its turn in flight
        self.busy_cart = None
        self.images = {}                    # cartridge id -> {image id: a picture read, waiting for its message}
        self.lock = asyncio.Lock()          # the endpoints that touch the card, one at a time
        self.tools = "chill"                # the config's tools: a new conversation's mode for its tool, or "no"
        self.classifier = None              # tools.json's folders, every conversation's (each its own mode)
        self.workspace = DATA / "workspace"
        self.tool_sessions = {}             # cartridge id -> its REPL
        self.asks = {}                      # cartridge id -> its answer, waiting on the user's answer to an ask
        self.image = False                  # the model is a picture model (▶ `_boot_image`): a message makes a picture


S = State()


def _sidecar(cid):
    p = S.dir / cid / "server_meta.json"
    try:
        return json.load(open(p)) if p.exists() else {}
    except Exception:
        return {}


def _write_sidecar(cid, meta):
    (S.dir / cid).mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(S.dir / cid / "server_meta.json", "w"), indent=2)


def _check_password(cid, provided):
    stored = _sidecar(cid).get("password")
    if not stored:
        return
    if not provided:
        raise HTTPException(401, "password required")
    if not _verify_password(provided, stored):
        raise HTTPException(401, "wrong password")


def _meta(cid):
    p = S.dir / cid / "meta.json"
    return json.load(open(p)) if p.exists() else None


def _persist(c, with_state=True):
    """A cartridge to its folder: its tokens (and its state, read off the card if it is there) and what it belongs to."""
    d = S.dir / c.id
    d.mkdir(parents=True, exist_ok=True)
    S.runtime.save(c, str(d / "cartridge.npz"), with_state=with_state)
    json.dump({"model_id": S.model_id, "identity": c.identity, "lora": c.lora, "current_pos": c.length},
              open(d / "meta.json", "w"), indent=2)


async def _idle():
    """Wait for the turn on the card, if there is one: the card's memory is read only between turns."""
    while S.runtime.busy():
        await asyncio.sleep(0.05)


async def _put_away(cid, persist=True):
    """A cartridge off the server's hands: saved with its state, and forgotten until it is bound again."""
    c = S.runtime.cartridges.get(cid)
    S.owner.pop(cid, None)
    if c is None:
        return
    await _idle()
    if persist:
        await asyncio.to_thread(_persist, c)
    if S.runtime.active is c:
        S.runtime.active = None              # what the card holds for it is no one's now
    S.runtime.cartridges.pop(cid, None)


async def _room_for(c):
    """Before `c` takes the card: the conversation on it is read off into RAM when another takes the card, and is put
    away to its folder instead — as the idle eviction puts one away — when RAM has no room for two of the largest
    beside the spare. ⚖ 2026-10-14: *"put it away when another conversation opens if there is no free space in ram to
    hold two (each conv is supposed to be the same max size), this is mainly a single player setup"* — the largest is a
    conversation at the full context (`state_bytes` at `max_context`), what any of them may grow to."""
    a = S.runtime.active
    if a is None or a is c or S.runtime.busy():
        return
    largest = state_bytes(S.model, S.model.max_context)
    room = experts_file.free()
    if room >= 2 * largest + experts_file.SPARE:
        return
    print("[put away] %s: %.1f GB free, no room for two conversations of %.1f GB" % (a.id, room / 2**30, largest / 2**30),
          flush=True)
    await _put_away(a.id)


def _load(cid):
    """A cartridge from its folder into the runtime, refused if it belongs to another model."""
    c = S.runtime.cartridges.get(cid)
    if c is not None:
        return c
    p = S.dir / cid / "cartridge.npz"
    if not p.exists():
        raise HTTPException(404, "cartridge %r not found" % cid)
    try:
        return S.runtime.adopt(Cartridge.load(str(p)))
    except SR.Refused as ex:
        raise HTTPException(409, str(ex))


def _describe(cid, x_web_user_id=None):
    side, meta = _sidecar(cid), _meta(cid) or {}
    extra = side.get("extra", {})
    c = S.runtime.cartridges.get(cid)
    owner = S.owner.get(cid)
    return {"id": cid, "name": side.get("name", cid), "history": extra.get("history", []),
            "next_pos": c.length if c else int(meta.get("current_pos", 0)),
            "enable_thinking": extra.get("enable_thinking", S.config.get("thinking", True)),
            "tools": _tools_on(extra), "tools_mode": _tools_mode(extra),
            "ask": S.asks[cid].ask if cid in S.asks else None,
            "protected": bool(side.get("password")), "lora": (meta.get("lora") or {}).get("name"),
            "bound": c is not None, "bound_by_self": owner is not None and owner == x_web_user_id,
            "bound_by_other": owner is not None and owner != x_web_user_id}


# ── boot ────────────────────────────────────────────────────────────────────────────────────────────────────
@contextmanager
def _one_boot_at_a_time():
    """Held from the setup's trials to the model on the card: two servers started together on one machine each try the
    card as the other leaves it, not both an empty card (a lock file in the machine's temporary folder; none on a system
    without `fcntl`, which runs one server)."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    path = os.path.join(tempfile.gettempdir(), "silvann_boot.lock")
    try:
        f = open(path, "a")
    except OSError:
        f = open(path)                                # another user's lock file: a lock on it reads as well
    with f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[silvann_server] another server on this machine is booting: this one boots after it", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _boot(args):
    folder = Path(ensure_model(args.model))
    if args.config:
        p = Path(args.config)
        if not p.exists():
            p = folder / "configs" / args.config
    else:
        p = default_config(folder)                                           # the folder's own, written if absent
    cfg = json.load(open(p))
    S.config = cfg
    t0 = time.time()
    # the card this machine runs on, where it has several: its index among them (`"auto"`: the first); `"cpu"` none —
    #   the CPU's cores and RAM run it, as a machine without a card would (a ring's stage in RAM beside one on the card)
    card = cfg.get("card", "auto")
    if card not in (None, "auto"):
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ[name] = "-1" if str(card).lower() == "cpu" else str(card)
    if image_pack(str(folder)):
        return _boot_image(folder, cfg, t0)
    # ⭐ THE SETUP FOR THIS MACHINE (NN-83): what the config leaves out — where the experts sit, the card's tier of them,
    #   the context the card holds — chosen by trying it on the card, or as chosen before (`autotune.json`)
    asked = cfg.get("max_context", 32768)       # a number, or "auto" ("max"): what the card holds, to the model's length
    asked = "max" if str(asked).lower() in ("max", "auto") else int(asked)
    # a video's tokens at most, and what reading one holds of RAM beside the experts (▶ vision.MAX_VIDEO_TOKENS)
    S.video_tokens = int(cfg.get("max_tokens_per_video", MAX_VIDEO_TOKENS))
    options = dict(cfg.get("options", {}))
    # the model's tool (▶ the classifier): what its commands may do unasked in a new conversation — or "no", no
    #   conversation has it
    S.tools = str(cfg.get("tools", "chill")).lower()
    if S.tools not in ("no",) + tools.MODES:
        raise SR.Refused("tools is \"no\" or one of %s, not %r" % (", ".join(tools.MODES), cfg.get("tools")))
    # ⭐ A RING OF MACHINES (▶ silvann_runtime/pipeline.py): this one runs its run of the layers; the one that begins at
    #   layer 0 is the first, and answers the UI — every other is a stage, and serves the ring until it is taken down
    ring = str(cfg.get("pipeline_parallelism", "no")).lower() in ("yes", "true", "1")
    if ring:
        from silvann_runtime import pipeline
        from silvann_runtime.model_folder import ModelFolder
        total = pipeline.model_layers(ModelFolder(str(folder)))
        layers = pipeline.layers_of(cfg.get("pipeline_parallelism_layers", ""), total)
        if not cfg.get("pipeline_parallelism_next"):
            raise SR.Refused("pipeline_parallelism_next names the machine after this one in the ring, as host:port")
        if asked == "max":
            raise SR.Refused("a ring's machines share one context: max_context is the same number in each one's config, "
                             "not \"auto\"")
        options["layers"] = layers
    with _one_boot_at_a_time():
        named = options                                   # the config's own: what a second choice is made around
        options, max_context, why = autotune.choose(str(folder), asked, named, video_tokens=S.video_tokens)
        S.why = why
        print("[silvann_server] the setup: %s" % why, flush=True)
        try:
            S.model = SR.open_model(str(folder), lora=cfg.get("lora"), max_context=max_context, **options)
        except Exception as ex:                               # noqa: BLE001 — the engine's refusal is a plain error
            if "kept in autotune.json" not in why or not autotune.refused_by_card(ex):
                raise
            options, max_context, why = autotune.choose(str(folder), asked, named, fresh=True,
                                                        video_tokens=S.video_tokens)
            S.why = why
            print("[silvann_server] the card refused the setup kept for it; chosen again: %s" % why, flush=True)
            S.model = SR.open_model(str(folder), lora=cfg.get("lora"), max_context=max_context, **options)
    if ring:
        if not hasattr(S.model, "forward"):
            raise SR.Refused("%s runs on one machine only: its model has no stages" % folder.name)
        link = pipeline.Link(int(cfg.get("pipeline_parallelism_port", pipeline.DEFAULT_PORT)),
                             cfg["pipeline_parallelism_next"])
        say = lambda text: print("[silvann_server] " + text.replace("[pipeline] ", ""), flush=True)   # noqa: E731
        if layers[0] != 0:
            say("a stage of the ring: layers %d-%d of %d%s" % (layers[0], layers[-1], total,
                                                                ", the head" if layers[-1] == total - 1 else ""))
            pipeline.stage(S.model, link, layers[-1] == total - 1, say)
            S.model.m.shutdown()
            raise SystemExit(0)
        say("the first machine of the ring: layers %d-%d of %d" % (layers[0], layers[-1], total))
        S.model = pipeline.Pipeline(S.model, link, say)
    S.runtime = Runtime(S.model)
    S.chat = Chat(S.model.folder)
    # the model's REPL starts in the workspace, where it reads unasked (▶ the classifier); tools.json keeps the folders
    #   allowed since
    S.workspace = Path(cfg.get("tools_folder") or DATA / "workspace")
    S.classifier = tools.Classifier(str(DATA / "tools.json"), "ask" if S.tools == "no" else S.tools,
                                    allowed=[str(S.workspace)])
    S.model_id = folder.name
    S.dir = DATA / "user_sessions" / folder.name
    S.dir.mkdir(parents=True, exist_ok=True)
    print("[silvann_server] %s (%s) booted in %.0f s — context %d, LoRA %s"
          % (folder.name, S.model.folder.architecture, time.time() - t0, S.model.max_context, cfg.get("lora")), flush=True)


def _boot_image(folder, cfg, t0):
    """⭐ A PICTURE MODEL (v0.10.0, ▶ silvann_runtime/image_session.py): a conversation's message is a prompt and its
    answer a picture, made a boot at a time when the message comes — nothing is on the card until then. The config's
    `image` holds a picture's settings (`width`, `height`, `steps`, `seed`); the model has no tool."""
    S.image = True
    S.tools = "no"
    S.model = image_session.ImageModel(str(folder), settings=cfg.get("image"))
    S.runtime = image_session.ImageRuntime(S.model)
    threading.Thread(target=S.runtime.start, daemon=True).start()     # the picture worker, ready before the first prompt
    S.workspace = Path(cfg.get("tools_folder") or DATA / "workspace")
    S.classifier = tools.Classifier(str(DATA / "tools.json"), "ask", allowed=[str(S.workspace)])
    S.model_id = folder.name
    S.dir = DATA / "user_sessions" / folder.name
    S.dir.mkdir(parents=True, exist_ok=True)
    print("[silvann_server] %s (a picture model, %s) ready in %.0f s — a picture %s"
          % (folder.name, S.model.folder.architecture, time.time() - t0, S.model.settings), flush=True)


def _sampler():
    s = S.config.get("sampler", "generation")
    if s is None:
        return None                          # greedy
    if s == "generation":                    # the model's own generation_config
        g = S.model.folder.generation
        s = {k: g[k] for k in ("temperature", "top_k", "top_p", "repetition_penalty") if k in g}
    return sampler(**s)


async def _eviction_loop():
    while True:
        try:
            now = time.time()
            stale = [cid for cid, owner in list(S.owner.items())
                     if now - S.last_ping.get(owner, 0) >= IDLE_EVICTION_SECONDS]
            for cid in stale:
                async with S.lock:
                    print("[evict] %s" % cid, flush=True)
                    await _put_away(cid)
        except Exception as ex:
            print("[evict] sweep failed: %s" % ex, flush=True)
        await asyncio.sleep(EVICTION_TICK_SECONDS)


@asynccontextmanager
async def lifespan(app):
    sweep = asyncio.create_task(_eviction_loop())
    try:
        yield
    finally:
        sweep.cancel()
        if S.runtime is not None:
            S.runtime.close()
            for cid in list(S.runtime.cartridges):
                try:
                    _persist(S.runtime.cartridges[cid])
                    print("[shutdown] saved %s" % cid, flush=True)
                except Exception as ex:
                    print("[shutdown] save failed for %s: %s" % (cid, ex), flush=True)

app = FastAPI(title="SiLVANN server", lifespan=lifespan)


# ── endpoints ───────────────────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    m = S.model
    return {"ok": True, "model_id": S.model_id, "architecture": m.folder.architecture if m else None,
            "lora": m.active_lora if m else None, "loras": m.folder.loras() if m else [],
            "tenants_loaded": len(S.runtime.cartridges) if S.runtime else 0,
            "active": S.runtime.active.id if S.runtime and S.runtime.active else None,
            "model_dir": m.folder.path if m else None, "pp_slice": "all", "num_gpus": getattr(m, "CARDS", 1),
            "max_context": m.max_context if m else 0,
            "adapter": "silvann_runtime", "images": _reads_pictures(), "files": not S.image, "pictures": S.image,
            "picture_settings": S.model.settings if S.image else None}


def _reads_pictures():
    """Whether the model reads pictures: its runtime has a vision tower and the pack carries it (▶ vision.py, a Qwen 3.5
    with M-RoPE; vision_glm.py, GLM 5.3) — the model's `vision()` refuses otherwise."""
    m = S.model
    if m is None or not hasattr(m, "vision"):
        return False
    try:
        m.vision()
        return True
    except SR.Refused:
        return False


@app.get("/busy")
def busy():
    return {"busy": S.runtime.busy(), "cart": S.busy_cart}


@app.post("/admin/save-all")
async def save_all():
    saved = []
    async with S.lock:
        await _idle()
        for cid, c in list(S.runtime.cartridges.items()):
            await asyncio.to_thread(_persist, c)
            saved.append(cid)
    return {"saved": saved}


@app.get("/cartridges")
def list_cartridges():
    out = []
    for d in sorted(S.dir.iterdir()) if S.dir.exists() else []:
        meta = _meta(d.name) if d.is_dir() else None
        if meta is None:
            continue
        side = _sidecar(d.name)
        out.append({"id": d.name, "name": side.get("name", d.name), "model_id": meta.get("model_id", "?"),
                    "compatible": meta.get("model_id") == S.model_id, "current_pos": meta.get("current_pos", 0),
                    "loaded": d.name in S.runtime.cartridges, "protected": bool(side.get("password")),
                    "lora": (meta.get("lora") or {}).get("name")})
    return {"cartridges": out}


@app.post("/cartridges")
async def create_cartridge(req: CreateConvReq, x_web_user_id: Optional[str] = Header(default=None)):
    async with S.lock:
        if x_web_user_id:                     # one open conversation a web user, as before
            for other in [c for c, o in list(S.owner.items()) if o == x_web_user_id]:
                await _put_away(other)
        try:
            c = S.runtime.new_cartridge(lora=req.lora if req.lora is not None else S.config.get("lora"))
        except SR.Refused as ex:
            raise HTTPException(409, str(ex))
        name = req.name or "chat-%s" % time.strftime("%H%M%S")
        _write_sidecar(c.id, {"id": c.id, "name": name, "created": time.time(),
                              "extra": {"history": [], "system_prompt": req.system,
                                        "enable_thinking": S.config.get("thinking", True)}})
        await asyncio.to_thread(_persist, c, False)
        if x_web_user_id:
            S.owner[c.id] = x_web_user_id
            S.last_ping[x_web_user_id] = time.time()
    return {"id": c.id, "name": name, "enable_thinking": S.config.get("thinking", True)}


@app.post("/cartridges/estimate")
async def estimate_cartridge(req: EstimateReq):
    """What a conversation costs. ⛳ The card's share is allocated once at boot for the whole context and shared by
    every conversation — only the one being served is on it — so a new conversation costs the card nothing; put away,
    it costs its state in RAM, up to the figure here."""
    m = S.model
    total = state_bytes(m, m.max_context)
    return {"mode": "flat", "bytes": int(total), "gb": total / 2 ** 30, "breakdown": {"state": int(total)},
            "num_gpus": getattr(m, "CARDS", 1),
            "defaults": dict(chunk_tokens=m.max_context, hot_chunks=1, warm_chunks=0, cold_chunks=0, system_chunks=0),
            "max_tokens": int(m.max_context), "page_size": 1}


@app.post("/cartridges/{cid}/activate")
async def activate_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None),
                             x_web_user_id: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    meta = _meta(cid)
    if meta is None:
        raise HTTPException(404, "cartridge %r not found" % cid)
    if meta.get("model_id") != S.model_id:
        raise HTTPException(409, "cartridge model %r != server %r" % (meta.get("model_id"), S.model_id))
    return _describe(cid, x_web_user_id)


@app.post("/cartridges/{cid}/bind")
async def bind_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None),
                         x_web_user_id: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    if not x_web_user_id:
        raise HTTPException(400, "missing X-Web-User-Id header")
    async with S.lock:
        for other in [c for c, o in list(S.owner.items()) if o == x_web_user_id and c != cid]:
            await _put_away(other)
        owner = S.owner.get(cid)
        if owner and owner != x_web_user_id:
            if time.time() - S.last_ping.get(owner, 0) < IDLE_EVICTION_SECONDS:
                raise HTTPException(423, "in use by another session")
            await _put_away(cid)
        _load(cid)
        S.owner[cid] = x_web_user_id
        S.last_ping[x_web_user_id] = time.time()
    return _describe(cid, x_web_user_id)


@app.post("/cartridges/{cid}/unbind")
async def unbind_cartridge(cid: str, x_web_user_id: Optional[str] = Header(default=None)):
    async with S.lock:
        if cid not in S.runtime.cartridges:
            return {"id": cid, "bound": False, "freed": False}
        owner = S.owner.get(cid)
        if owner and owner != x_web_user_id:
            raise HTTPException(423, "not your session")
        await _put_away(cid)
    return {"id": cid, "bound": False, "freed": True}


@app.post("/sessions/ping")
async def session_ping(x_web_user_id: Optional[str] = Header(default=None)):
    if not x_web_user_id:
        raise HTTPException(400, "missing X-Web-User-Id header")
    S.last_ping[x_web_user_id] = time.time()
    return {"session": x_web_user_id, "owned": [c for c, o in S.owner.items() if o == x_web_user_id],
            "idle_eviction_seconds": IDLE_EVICTION_SECONDS}


@app.delete("/cartridges/{cid}")
async def delete_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    async with S.lock:
        await _put_away(cid, persist=False)
        if (S.dir / cid).exists():
            shutil.rmtree(S.dir / cid)
        S.tool_sessions.pop(cid, None)                        # its REPL, and an ask it left waiting
        S.asks.pop(cid, None)
    return {"deleted": cid}


@app.patch("/cartridges/{cid}")
async def update_cartridge(cid: str, req: UpdateReq, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    side = _sidecar(cid)
    if req.name is not None:
        side["name"] = req.name
    if req.enable_thinking is not None:
        side.setdefault("extra", {})["enable_thinking"] = bool(req.enable_thinking)
    if req.tools is not None:
        extra = side.setdefault("extra", {})
        if req.tools and S.tools == "no":
            raise HTTPException(409, "this server's config turns the model's tool off (\"tools\": \"no\")")
        if req.tools and extra.get("history") and not extra.get("tools_declared"):
            raise HTTPException(409, "this conversation began without the tool, which is declared in a conversation's "
                                     "first turn: a new conversation has it")
        extra["tools"] = bool(req.tools)
    if req.tools_mode is not None:
        if req.tools_mode not in tools.MODES:
            raise HTTPException(422, "the mode is one of %s" % ", ".join(tools.MODES))
        side.setdefault("extra", {})["tools_mode"] = req.tools_mode
        if cid in S.tool_sessions:
            S.tool_sessions[cid].classifier.mode = req.tools_mode
    _write_sidecar(cid, side)
    return {"id": cid, "name": req.name, "enable_thinking": req.enable_thinking, "tools": req.tools,
            "tools_mode": req.tools_mode}


@app.get("/cartridges/{cid}/export")
async def export_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    c = S.runtime.cartridges.get(cid)
    if c is not None:                          # an open one is saved first, its state with it
        async with S.lock:
            await _idle()
            await asyncio.to_thread(_persist, c)
    d = S.dir / cid
    if not d.exists():
        raise HTTPException(404, "cartridge %r not found" % cid)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for f in sorted(d.iterdir()):
            tar.add(str(f), arcname=f.name)
    return Response(content=buf.getvalue(), media_type="application/x-tar",
                    headers={"Content-Disposition": 'attachment; filename="%s.silvann.tar"' % cid})


@app.post("/cartridges/import")
async def import_cartridge(file: UploadFile = File(...)):
    blob = await file.read()
    new_id = uuid.uuid4().hex[:12]
    d = S.dir / new_id
    d.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile() and "/" not in member.name and member.name in ("cartridge.npz", "meta.json", "server_meta.json"):
                    open(d / member.name, "wb").write(tar.extractfile(member).read())
        c = Cartridge.load(str(d / "cartridge.npz"))
        if not c.matches(S.runtime.identity):
            raise HTTPException(409, "the cartridge belongs to %s, and this server runs %s" % (c.identity, S.runtime.identity))
    except HTTPException:
        shutil.rmtree(d)
        raise
    except Exception as ex:
        shutil.rmtree(d)
        raise HTTPException(400, "invalid cartridge archive: %s" % ex)
    c.id = new_id
    c.save(str(d / "cartridge.npz"))
    json.dump({"model_id": S.model_id, "identity": c.identity, "lora": c.lora, "current_pos": c.length},
              open(d / "meta.json", "w"), indent=2)
    side = _sidecar(new_id)
    side["id"] = new_id
    _write_sidecar(new_id, side)
    return {"id": new_id, "name": side.get("name", new_id), "model_id": S.model_id}


@app.post("/cartridges/{cid}/stop")
async def stop_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    t = S.turns.get(cid)
    if t is not None and not t.done.is_set():
        t.cancelled = True
        return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False, "reason": "no generation in flight"}


@app.post("/cartridges/{cid}/images")
async def upload_image(cid: str, file: UploadFile = File(...), x_cart_password: Optional[str] = Header(default=None)):
    """An attachment for the conversation's next message, kept under the id this answers until a message names it —
    `{image_id, kind, merged_thw, n_placeholders}`: a picture or a video read through the model's vision
    tower; or a document — Word, PDF, text — whose text the message carries, for any model (▶ silvann_runtime/documents.py)."""
    _check_password(cid, x_cart_password)
    if S.runtime.cartridges.get(cid) is None:
        raise HTTPException(409, "not bound — call /bind first")
    if S.image:
        raise HTTPException(415, "%s makes pictures from a prompt; reference pictures come with a later release" % S.model_id)
    data = await file.read()
    from silvann_runtime import documents
    if documents.is_document(data, file.filename):
        try:
            doc = await asyncio.to_thread(lambda: documents.read_document(data, file.filename or "file"))
        except SR.Refused as ex:
            raise HTTPException(415, str(ex))
        image_id = uuid.uuid4().hex[:12]
        S.images.setdefault(cid, {})[image_id] = doc
        return {"image_id": image_id, "kind": "document", "merged_thw": [], "n_placeholders": 0, "chars": len(doc.text)}
    if not hasattr(S.model, "vision"):
        raise HTTPException(415, "%s reads no pictures" % S.model_id)
    def read(data):
        vision = S.model.vision()
        vision.max_video_tokens = S.video_tokens
        return vision.read(data)

    try:
        image = await asyncio.to_thread(read, data)
    except SR.Refused as ex:
        raise HTTPException(415, str(ex))
    except Exception as ex:                                   # not a picture Pillow reads
        raise HTTPException(400, "not read as a picture: %s" % ex)
    image_id = uuid.uuid4().hex[:12]
    S.images.setdefault(cid, {})[image_id] = image
    m = S.model.vision().merge
    pairs = getattr(image, "pairs", None)                   # a video: its pairs of frames, each read as a picture
    return {"image_id": image_id, "merged_thw": [image.grid[0], image.grid[1] // m, image.grid[2] // m],
            "n_placeholders": image.tokens, "kind": "video" if pairs is not None else "picture",
            "frames": 2 * len(pairs) if pairs is not None else 1}


@app.post("/cartridges/{cid}/save")
async def save_cartridge(cid: str, req: SaveReq, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    c = S.runtime.cartridges.get(cid)
    if c is not None:
        async with S.lock:
            await _idle()
            await asyncio.to_thread(_persist, c)
    side = _sidecar(cid)
    if req.password is not None:
        if req.password == "":
            side.pop("password", None)
        else:
            side["password"] = _hash_password(req.password)
        _write_sidecar(cid, side)
    return {"id": cid, "protected": bool(side.get("password"))}


@app.post("/cartridges/{cid}/messages")
async def post_message(cid: str, req: MessageReq, x_cart_password: Optional[str] = Header(default=None),
                       x_web_user_id: Optional[str] = Header(default=None)):
    """The answer as text/event-stream: `event: token` {"token_id", "text"} a token, then `event: done` or `event: error`
    — and in a conversation with the model's tool, `event: tool` a call and `event: ask` an ask (▶ `_Answer`).
    Where the prompt opened the answer's think block (a thinking Qwen's, GLM's), the first token's text begins with its
    `<think>`, so the answer's text is whole: its thinking, `</think>`, the answer.
    ⛔ A turn already on the card answers 503 with `Retry-After: 5`; the caller asks again."""
    _check_password(cid, x_cart_password)
    c = S.runtime.cartridges.get(cid)
    if c is None:
        raise HTTPException(409, "not bound — call /bind first")
    owner = S.owner.get(cid)
    if owner and owner != x_web_user_id:
        raise HTTPException(423, "in use by another session")
    if x_web_user_id:
        S.last_ping[x_web_user_id] = time.time()
    if S.image:
        return _picture_message(cid, req)
    side = _sidecar(cid)
    extra = side.setdefault("extra", {})
    waiting = S.images.get(cid, {})
    if any(i not in waiting for i in req.images):
        raise HTTPException(404, "a picture this message names was not uploaded to this conversation")
    items = [waiting[i] for i in req.images]
    # the message's pictures before its videos, as the chat frames them; a video's pairs of frames each a picture's pads;
    # its documents' text before its own, each cut to what the context has room for
    docs = [im for im in items if hasattr(im, "framed")]
    pics = [im for im in items if not hasattr(im, "pairs") and not hasattr(im, "framed")]
    vids = [im for im in items if hasattr(im, "pairs")]
    text = req.text
    if docs:
        room = S.model.max_context - len(c.tokens) - len(S.chat.encode(req.text)) - sum(im.tokens for im in pics + vids) - 512
        framed = []
        for d in docs:
            ids = S.chat.encode(d.text)
            share = max(0, room // len(docs))
            body = d.text if len(ids) <= share else (
                S.chat.tok.decode(ids[:share]) + "\n[… the rest of %s did not fit the conversation]" % d.name)
            framed.append(d.framed(body))
        text = "".join(framed) + req.text
    first = not c.tokens
    if first:                                                 # the tool is declared in the first turn, or never
        extra["tools_declared"] = _tools_on(extra)
        _write_sidecar(cid, side)
    try:
        new = S.chat.turn(c.tokens, text, extra.get("system_prompt"),
                          extra.get("enable_thinking", S.config.get("thinking", True)), images=[im.tokens for im in pics],
                          videos=[[(im.tokens, t) for im, t in v.pairs] for v in vids],
                          tools=first and extra["tools_declared"])
    except SR.Refused as ex:
        raise HTTPException(415, str(ex))
    images = pics + [im for v in vids for im, _ in v.pairs]
    placed = list(zip(S.chat.image_starts(new, S.model.vision().pads), images)) if images else []
    if c.tokens and c.tokens[-1] == S.chat.call_end:         # a call left unanswered (a stopped answer): closed
        new = S.chat.close_call() + new
        placed = [(at + len(S.chat.close_call()), im) for at, im in placed]
    room = S.model.max_context - len(c.tokens) - len(new)
    if room <= 0:
        raise HTTPException(413, "the conversation has reached the context of %d" % S.model.max_context)
    if cid in S.asks:
        raise HTTPException(409, "the model waits on an answer to its ask: %s" % S.asks[cid].ask["question"])
    a = _Answer(cid, c, req.max_new_tokens, {"role": "user", "content": req.text, "images": len(req.images)})
    async with S.lock:
        await _room_for(c)
    try:
        a.begin(new, placed)
    except Busy:
        raise HTTPException(503, "busy — another conversation's turn is on the card", headers={"Retry-After": "5"})
    for i in req.images:                                      # a picture is the message's now, in the conversation
        waiting.pop(i, None)
    return StreamingResponse(a.stream(), media_type="text/event-stream")


# ── a picture model's message (v0.10.0) ─────────────────────────────────────────────────────────────────────
def _picture_message(cid, req):
    """A message to a picture model: the picture its prompt asks for, as text/event-stream — `event: progress`
    {"stage", "done", "total", "text"} as it is made (the prompt read, a step at a time, the decode), then `event: token`
    {"text"} the answer's text, which names the picture (`![picture](picture:NAME)` and what made it), then `event:
    done`; `event: error` where it fails. The picture is in the conversation's folder, `pictures/NAME`
    (▶ `/cartridges/{cid}/pictures/{name}`). ⛔ A picture already being made answers 503 with `Retry-After: 5`."""
    if S.runtime.busy():
        raise HTTPException(503, "busy — another picture is being made", headers={"Retry-After": "5"})
    prompt, settings = S.runtime.settings(req.text)
    if not prompt:
        raise HTTPException(422, "a picture needs a prompt")
    queue, loop = asyncio.Queue(), asyncio.get_running_loop()

    def emit(kind, payload):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, json.dumps(payload)))

    def progress(stage, done, total):
        text = {"encode": "reading the prompt", "step": "step %d of %d" % (done, total), "decode": "decoding"}[stage]
        emit("progress", {"stage": stage, "done": done, "total": total, "text": text})

    def make():
        try:
            name, seconds = S.runtime.picture(prompt, settings, str(S.dir / cid / "pictures"), progress)
            answer = image_session.caption(name, settings, seconds)
            emit("token", {"token_id": -1, "text": answer})
        except Stopped as ex:
            answer = "_[the picture was stopped: %s]_" % ex
            emit("token", {"token_id": -1, "text": answer})
        except Exception as ex:                       # noqa: BLE001 — said to the UI
            emit("error", {"error": str(ex)})
            return
        finally:
            S.busy_cart = None
        side = _sidecar(cid)
        history = side.setdefault("extra", {}).setdefault("history", [])
        history += [{"role": "user", "content": req.text},
                    {"role": "assistant", "content": answer, "picture": dict(settings, prompt=prompt)}]
        _write_sidecar(cid, side)
        emit("done", {})

    S.busy_cart = cid
    job = threading.Thread(target=make, daemon=True)
    job.start()
    S.turns[cid] = _PictureTurn()

    async def stream():
        while True:
            kind, payload = await queue.get()
            yield "event: %s\ndata: %s\n\n" % (kind, payload)
            if kind in ("done", "error"):
                return
    return StreamingResponse(stream(), media_type="text/event-stream")


class _PictureTurn:
    """What `/stop` asks of a turn, for a picture: its job's stop."""

    @property
    def done(self):
        job = S.runtime._job
        return job.done if job is not None else threading.Event()

    @property
    def cancelled(self):
        return S.runtime._job is not None and S.runtime._job.cancelled

    @cancelled.setter
    def cancelled(self, value):
        if S.runtime._job is not None:
            S.runtime._job.cancelled = value


@app.get("/cartridges/{cid}/pictures/{name}")
async def get_picture(cid: str, name: str, format: str = "png", x_cart_password: Optional[str] = Header(default=None)):
    """A picture a conversation's answer made: the PNG, or `?format=jpeg` a smaller copy to show."""
    _check_password(cid, x_cart_password)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.png", name):
        raise HTTPException(404, "no picture %r" % name)
    p = S.dir / cid / "pictures" / name
    if not p.exists():
        raise HTTPException(404, "no picture %r in this conversation" % name)
    if format == "jpeg":
        from PIL import Image
        buf = io.BytesIO()
        Image.open(p).convert("RGB").save(buf, "JPEG", quality=88)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    return Response(content=p.read_bytes(), media_type="image/png")


# ── the model's tool (NN-88) ────────────────────────────────────────────────────────────────────────────────
# ⭐ A CONVERSATION BEGUN WITH THE TOOL (the config's "tools" anything but "no", or the conversation's own switch
#   before its first message) has a Lisp REPL (▶ silvann_runtime/tools.py): the answer stops where the model calls it,
#   the form runs here, and its value is read back in as the template frames a tool's answer — the answer goes on from
#   there. A command the classifier stops is asked of the user: the stream ends with `event: ask`, and `/tools/answer`
#   runs the user's answer in the same REPL and streams the rest. What the UI is shown of a call is the same for every
#   model: `<tool_call>FORM</tool_call><tool_response>VALUE</tool_response>` in the answer's text, an ask and the user's
#   answer between them as `<tool_ask>QUESTION</tool_ask><tool_answer>CHOICE</tool_answer>`.
TOOL_CALLS_MAX = 64                       # the calls one message's answer may make


def _tools_on(extra):
    """The conversation's switch, as the thinking switch: on unless turned off, where the config does not say "no" —
    and, once the conversation has begun, only if its first turn declared the tool."""
    if S.tools == "no" or extra.get("history") and not extra.get("tools_declared"):
        return False
    return bool(extra.get("tools", True))


def _tools_mode(extra):
    """What the conversation's commands may do unasked: its own choice, or the config's."""
    return extra.get("tools_mode") or (S.tools if S.tools in tools.MODES else "ask")


def _tool_session(cid):
    s = S.tool_sessions.get(cid)
    if s is None:
        S.workspace.mkdir(parents=True, exist_ok=True)
        mode = _tools_mode(_sidecar(cid).get("extra", {}))
        s = S.tool_sessions[cid] = tools.Session(str(S.workspace), S.classifier.conversation(mode))
    return s


class _Answer:
    """One message's answer in the making: the model's stretches of answer, the calls between them run and answered,
    streamed as text/event-stream — `event: token` {"token_id", "text"} a token, `event: tool` {"form", "answer",
    "text"} a call, then `event: done`, `event: ask` {"question", "choices"} (and done) or `event: error`."""

    def __init__(self, cid, c, budget, user):
        self.cid, self.c, self.budget, self.user = cid, c, budget, user
        self.extra = _sidecar(cid).get("extra", {})
        self.thinking = self.extra.get("enable_thinking", S.config.get("thinking", True))
        # a conversation that declared the tool stops at every call; one switched off since answers it "off"
        self.declared, self.tools = bool(self.extra.get("tools_declared")), _tools_on(self.extra)
        self.stop = set(S.chat.stop) | ({S.chat.call_end} if self.declared and S.chat.call_end is not None else set())
        self.text = ""                         # what the UI has been shown of the answer
        self.written = False                   # whether the history holds this answer yet
        self.calls = 0
        self.forms, self.answers = [], []      # the stretch's calls, and the values of those run
        self.ask = None
        self.queue = self.loop = None

    # the stream
    def _emit(self, kind, payload):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, json.dumps(payload)))

    def _show(self, kind, text, **payload):
        self.text += text
        self._emit(kind, dict(payload, text=text))

    async def stream(self):
        while True:
            kind, payload = await self.queue.get()
            yield "event: %s\ndata: %s\n\n" % (kind, payload)
            if kind in ("done", "error"):
                return

    # a stretch of answer on the card
    def _turn(self, new, images=(), retry=False):
        room = S.model.max_context - len(self.c.tokens) - len(new)
        if room <= 0:
            raise SR.Refused("the conversation has reached the context of %d" % S.model.max_context)
        self.answer, self.held, self.shown = [], False, ""
        self.opened = S.chat.opened(new)       # the stretch begins inside the think block its prompt opened
        while True:
            try:
                t = S.runtime.turn(self.c, new, min(self.budget, room), sampler=_sampler(), stop=self.stop,
                                   on_token=self._on_token, images=images)
                break
            except Busy:
                if not retry:
                    raise
                time.sleep(0.2)
        S.turns[self.cid], S.busy_cart = t, self.cid
        return t

    def _on_token(self, tok):
        self.answer.append(tok)
        if self.declared and tok == S.chat.call_start:
            self.held = True                   # a call: shown as every model's is, once it is read
        if self.held:
            return
        text = self.opened + S.chat.decode(self.answer)
        delta, self.shown = text[len(self.shown):], text
        self._show("token", delta, token_id=tok)

    def begin(self, new, placed):
        """The first stretch on the card now (`Busy` where another turn has it); the rest in a thread of its own."""
        self.queue, self.loop = asyncio.Queue(), asyncio.get_running_loop()
        t = self._turn(new, placed)
        threading.Thread(target=self._go, args=(t,), daemon=True).start()

    def resume(self, choice, value):
        """The user's answer to the ask: its value is the waiting call's, and the answer goes on."""
        self.queue, self.loop = asyncio.Queue(), asyncio.get_running_loop()
        self._show("tool", "<tool_answer>%s</tool_answer>" % choice, choice=choice)
        threading.Thread(target=self._go, args=(None, value), daemon=True).start()

    def _go(self, t, value=None):
        try:
            if t is None:                      # after an ask: the waiting call's value, then the rest of the stretch
                S.asks.pop(self.cid, None)
                self.ask = None
                pending = _tool_session(self.cid).pending
                if pending is not None:        # the form, run again, asked again
                    self._ask(pending[1])
                    return
                self._answered(value)
                if self._calls() is not None:
                    return
                t = self._turn(S.chat.tool_response(self.answers, self.thinking), retry=True)
            while True:
                t.wait()
                self.budget -= len(self.answer)
                forms = S.chat.tool_calls(self.answer) if self.declared and not t.cancelled else []
                if not forms or self.budget <= 0:
                    if self.held:              # a call never made, shown as the model wrote it
                        self._show("token", (self.opened + S.chat.decode(self.answer))[len(self.shown):], token_id=-1)
                    break
                self.forms, self.answers = forms, []
                if self._calls() is not None:
                    return
                t = self._turn(S.chat.tool_response(self.answers, self.thinking), retry=True)
            self._write()
            self._emit("done", {})
        except Exception as ex:                # noqa: BLE001 — said to the UI
            if self.written or self.calls:     # an answer that made calls keeps what it did
                self._write()
            self._emit("error", {"error": str(ex)})
        finally:
            S.busy_cart = None

    def _calls(self):
        """Run the stretch's calls not yet answered; the ask, where one stops — the answer then waits for the user."""
        session = _tool_session(self.cid)
        while len(self.answers) < len(self.forms):
            form = self.forms[len(self.answers)]
            self.calls += 1
            shown = ";; a call that does not read" if isinstance(form, ToolMarkup) else form
            self._show("tool", "<tool_call>%s</tool_call>" % shown, form=str(form))
            if isinstance(form, ToolMarkup):
                value = tools.write([tools.Sym("error"), tools.Text(str(form))])
            elif not self.tools:
                value = tools.write([tools.Sym("error"), tools.Text("the tool is off: the user turned it off — answer "
                                                                    "without it")])
            elif self.calls > TOOL_CALLS_MAX:
                value = tools.write([tools.Sym("error"), tools.Text("%d calls in one answer: answer the user now"
                                                                    % TOOL_CALLS_MAX)])
            else:
                value = session.run(form)
                if session.pending is not None:
                    return self._ask(session.pending[1])
            self._answered(value)
        return None

    def _ask(self, asked):
        """The answer waits on the user: the ask said, the history written, the stream ended."""
        self.ask = {"question": asked.question, "choices": list(asked.choices)}
        S.asks[self.cid] = self
        self._show("tool", "<tool_ask>%s</tool_ask>" % asked.question, ask=asked.question)
        self._write()
        self._emit("ask", self.ask)
        self._emit("done", {})
        return self.ask

    def _answered(self, value):
        self.answers.append(value)
        self._show("tool", "<tool_response>%s</tool_response>" % value, answer=value)

    def _write(self):
        """The conversation's history: the user's message and the answer so far."""
        side = _sidecar(self.cid)
        history = side.setdefault("extra", {}).setdefault("history", [])
        if not self.written:
            history += [self.user, {"role": "assistant", "content": self.text}]
            self.written = True
        elif history and history[-1].get("role") == "assistant":
            history[-1]["content"] = self.text
        _write_sidecar(self.cid, side)


@app.post("/cartridges/{cid}/tools/answer")
async def answer_ask(cid: str, req: AnswerReq, x_cart_password: Optional[str] = Header(default=None),
                     x_web_user_id: Optional[str] = Header(default=None)):
    """The user's answer to the model's ask — "folder", "once" or "refuse" — run in the conversation's REPL as the
    user's; the rest of the answer streamed as `/messages` streams it."""
    _check_password(cid, x_cart_password)
    a = S.asks.get(cid)
    if a is None:
        raise HTTPException(409, "the model is not waiting on an answer")
    owner = S.owner.get(cid)
    if owner and owner != x_web_user_id:
        raise HTTPException(423, "in use by another session")
    if req.choice not in a.ask["choices"]:
        raise HTTPException(422, "the answer is one of %s" % ", ".join(a.ask["choices"]))
    a.c = S.runtime.cartridges.get(cid)                       # the conversation as it is now (put away and back)
    if a.c is None:
        raise HTTPException(409, "not bound — call /bind first")
    value = _tool_session(cid).run("(answer '%s)" % req.choice, user=True)
    a.budget = req.max_new_tokens
    a.resume(req.choice, value)
    return StreamingResponse(a.stream(), media_type="text/event-stream")


@app.get("/tools")
def tools_settings():
    """The config's tools — a new conversation's mode ("ask", "chill", "full-auto"), or "no": no conversation has the
    tool — the modes, the folders every conversation reads unasked, and where its REPL starts."""
    return {"tools": S.tools, "modes": list(tools.MODES), "allowed": S.classifier.allowed,
            "workspace": str(S.workspace)}


def main():
    p = argparse.ArgumentParser(description="Start a SiLVANN server on a packed model folder.")
    p.add_argument("config", nargs="?", help="a config in the folder's configs/, or a path to one")
    p.add_argument("--model", required=True, help="the model folder (fetched from Hugging Face if it is not there)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)          # the port the UI looks for by default
    args = p.parse_args()
    # ⛔ A BOOT THAT CANNOT BE MADE IS SAID IN ONE LINE, `⛔ not started:` and why — the UI that started this server shows
    #   it in its status (▶ silvann_ui.py, `ServerProcess.failure`)
    try:
        _boot(args)
    except SR.Refused as ex:                                  # a model not here, a config not found, a card too small
        raise SystemExit("[silvann_server] ⛔ not started: %s" % ex)
    except Exception as ex:                                   # noqa: BLE001 — the engine's refusal is a plain error
        if not autotune.refused_by_card(ex):
            raise
        raise SystemExit("[silvann_server] ⛔ not started: the card has no room for this setup (%s) — in the config, set "
                         "experts_on and experts_from to \"auto\" in its options, and lower max_context or set it to "
                         "\"auto\", as long as the card holds" % getattr(S, "why", ""))
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
