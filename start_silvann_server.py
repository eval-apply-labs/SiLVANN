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
import secrets
import shutil
import sys
import tarfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
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
sys.path.insert(0, str(RUNTIME))
import silvann_runtime as SR                                              # noqa: E402
from silvann_runtime import autotune, experts_file                       # noqa: E402
from silvann_runtime.chat import Chat                                     # noqa: E402
from silvann_runtime.fetch import ensure_model                            # noqa: E402
from silvann_runtime.session import Busy, Cartridge, Runtime, sampler     # noqa: E402
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
    conversation at the full context (`state_regions(max_context)`), what any of them may grow to."""
    a = S.runtime.active
    if a is None or a is c or S.runtime.busy():
        return
    largest = sum(n for _, n in S.model.state_regions(S.model.max_context))
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
            "protected": bool(side.get("password")), "lora": (meta.get("lora") or {}).get("name"),
            "bound": c is not None, "bound_by_self": owner is not None and owner == x_web_user_id,
            "bound_by_other": owner is not None and owner != x_web_user_id}


# ── boot ────────────────────────────────────────────────────────────────────────────────────────────────────
def _boot(args):
    folder = Path(ensure_model(args.model))
    if args.config:
        p = Path(args.config)
        if not p.exists():
            p = folder / "configs" / args.config
    else:
        from silvann_core import default_config                              # the folder's own, written if absent
        p = default_config(folder)
    cfg = json.load(open(p))
    S.config = cfg
    t0 = time.time()
    # the card this machine runs on, where it has several: its index among them (`"auto"`: the first); `"cpu"` none —
    #   the CPU's cores and RAM run it, as a machine without a card would (a ring's stage in RAM beside one on the card)
    card = cfg.get("card", "auto")
    if card not in (None, "auto"):
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ[name] = "-1" if str(card).lower() == "cpu" else str(card)
    # ⭐ THE SETUP FOR THIS MACHINE (NN-83): what the config leaves out — where the experts sit, the card's tier of them,
    #   the context the card holds — chosen by trying it on the card, or as chosen before (`autotune.json`)
    asked = cfg.get("max_context", 32768)       # a number, or "auto" ("max"): what the card holds, to the model's length
    asked = "max" if str(asked).lower() in ("max", "auto") else int(asked)
    # a video's tokens at most, and what reading one holds of RAM beside the experts (▶ vision.MAX_VIDEO_TOKENS)
    S.video_tokens = int(cfg.get("max_tokens_per_video", MAX_VIDEO_TOKENS))
    options = dict(cfg.get("options", {}))
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
    options, max_context, why = autotune.choose(str(folder), asked, options, video_tokens=S.video_tokens)
    S.why = why
    print("[silvann_server] the setup: %s" % why, flush=True)
    try:
        S.model = SR.open_model(str(folder), lora=cfg.get("lora"), max_context=max_context, **options)
    except Exception as ex:                                   # noqa: BLE001 — the engine's refusal is a plain error
        if "kept in autotune.json" not in why or not autotune.refused_by_card(ex):
            raise
        options, max_context, why = autotune.choose(str(folder), asked, options, fresh=True, video_tokens=S.video_tokens)
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
    S.model_id = folder.name
    S.dir = DATA / "user_sessions" / folder.name
    S.dir.mkdir(parents=True, exist_ok=True)
    print("[silvann_server] %s (%s) booted in %.0f s — context %d, LoRA %s"
          % (folder.name, S.model.folder.architecture, time.time() - t0, S.model.max_context, cfg.get("lora")), flush=True)


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
            "model_dir": m.folder.path if m else None, "pp_slice": "all", "num_gpus": 1, "max_context": m.max_context if m else 0,
            "adapter": "silvann_runtime", "images": _reads_pictures(), "files": True}


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
    total = sum(n for _, n in m.state_regions(m.max_context))
    return {"mode": "flat", "bytes": int(total), "gb": total / 2 ** 30, "breakdown": {"state": int(total)}, "num_gpus": 1,
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
    return {"deleted": cid}


@app.patch("/cartridges/{cid}")
async def update_cartridge(cid: str, req: UpdateReq, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    side = _sidecar(cid)
    if req.name is not None:
        side["name"] = req.name
    if req.enable_thinking is not None:
        side.setdefault("extra", {})["enable_thinking"] = bool(req.enable_thinking)
    _write_sidecar(cid, side)
    return {"id": cid, "name": req.name, "enable_thinking": req.enable_thinking}


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
    """The answer as text/event-stream: `event: token` {"token_id", "text"} a token, then `event: done` or `event: error`.
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
    try:
        new = S.chat.turn(c.tokens, text, extra.get("system_prompt"),
                          extra.get("enable_thinking", S.config.get("thinking", True)), images=[im.tokens for im in pics],
                          videos=[[(im.tokens, t) for im, t in v.pairs] for v in vids])
    except SR.Refused as ex:
        raise HTTPException(415, str(ex))
    images = pics + [im for v in vids for im, _ in v.pairs]
    placed = list(zip(S.chat.image_starts(new, S.model.vision().pads), images)) if images else []
    room = S.model.max_context - len(c.tokens) - len(new)
    if room <= 0:
        raise HTTPException(413, "the conversation has reached the context of %d" % S.model.max_context)
    queue, loop = asyncio.Queue(), asyncio.get_running_loop()
    answer, shown = [], [""]
    opened = S.chat.opened(new)                               # the answer begins inside the think block the prompt opened

    def on_token(tok):
        answer.append(tok)
        text = opened + S.chat.decode(answer)
        delta, shown[0] = text[len(shown[0]):], text
        loop.call_soon_threadsafe(queue.put_nowait, ("token", json.dumps({"token_id": tok, "text": delta})))

    async with S.lock:
        await _room_for(c)
    try:
        turn = S.runtime.turn(c, new, min(req.max_new_tokens, room), sampler=_sampler(), stop=S.chat.stop, on_token=on_token,
                              images=placed)
    except Busy:
        raise HTTPException(503, "busy — another conversation's turn is on the card", headers={"Retry-After": "5"})
    S.turns[cid], S.busy_cart = turn, cid
    for i in req.images:                                      # a picture is the message's now, in the conversation
        waiting.pop(i, None)

    def finish():
        try:
            turn.wait()
            extra.setdefault("history", []).append({"role": "user", "content": req.text, "images": len(req.images)})
            extra["history"].append({"role": "assistant", "content": opened + S.chat.decode(answer)})
            _write_sidecar(cid, side)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", "{}"))
        except Exception as ex:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", json.dumps({"error": str(ex)})))
        finally:
            S.busy_cart = None
    threading.Thread(target=finish, daemon=True).start()

    async def stream():
        while True:
            kind, payload = await queue.get()
            yield "event: %s\ndata: %s\n\n" % (kind, payload)
            if kind in ("done", "error"):
                return
    return StreamingResponse(stream(), media_type="text/event-stream")


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
