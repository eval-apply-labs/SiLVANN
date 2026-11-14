"""SiLVANN HTTP server — a model folder booted by `silvann_runtime`, and its conversations served over HTTP.

    python3 start_silvann_server.py --model models/<name> [<config.json>] [--port 8000] [--host 0.0.0.0]

The model folder is a packed bundle (`pack.json`, `layers/`, `global.pkl`, the model's own `config.json` and tokenizer);
one that is missing is asked for on a terminal and fetched from Hugging Face (`silvann_runtime.fetch`). The config — a
file in the folder's `configs/`, or a path — sets the runtime:
```
  {"max_context": "auto", "lora": null, "options": {...}, "sampler": {"temperature": 0.7, ...} | null, "thinking": true}
```
`lora` is the adapter the model boots with (⚖ *"at boot is enough … we say it in the json which lora we want"*), and a
conversation created with its own `lora` swaps it in for its turns. `options` go to the model's runtime (`kv`,
`fused_moe`, `sockets`, `experts_int8`, …). `sampler: null` is greedy.

Conversations are CARTRIDGES, one on the card at a time: a turn keeps the card until its answer is done, and a turn
asked for meanwhile is answered 503 with `Retry-After: 5` — ⚖ *"retry in 5 s is what the silvann ui will do (or the
requestor in general), it is the convention and it is the caller that will do it"*. A model holding several conversations
at once — `"multi_user": "yes"` with `num_cards` above 1, a conversation a card, or `seats` in its options, several on
one — answers them together (▶ session's `SeatedRuntime`): a turn is 503 only when every seat is answering. On disk each lives in
`user_sessions/<model>/<id>/`: `cartridge.npz` (its tokens, and its state when it was put away), `meta.json` (what
model it belongs to), `server_meta.json` (its name, history and password) and `conversation_config.json` (its own
settings: a system prompt, where the config's `user_system_prompt` lets its creator give one). With users made
(`silvann users add`, ▶ silvann_runtime/users.py) every request but /health carries a sign-in, and a user's conversations
are `users/<name>/user_sessions/<model>/<id>/`, theirs alone.
"""
import argparse
import asyncio
import base64
import contextvars
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
# the skills the release ships (▶ silvann_runtime/skills.py): skills/ beside this file, or the development tree's
SKILLS_SHIPPED = ROOT / "skills" if (ROOT / "skills").is_dir() else ROOT.parent / "release" / "skills"
# the tool's guide, which the REPL's (guide) answers: docs/ beside this file, or the development tree's
TOOLS_GUIDE = ROOT / "docs" / "tools.md" if (ROOT / "docs").is_dir() else ROOT.parent / "release" / "docs" / "tools.md"
# the conversations go beside the release, or in the folder SILVANN_DATA names (the Docker image's volume)
DATA = Path(os.environ.get("SILVANN_DATA") or ROOT)
# the runtime beside this file in a release, or the development tree's
RUNTIME = Path(os.environ.get("SILVANN_RUNTIME", ROOT / "runtime" if (ROOT / "runtime").is_dir()
                              else ROOT.parent / "python"))
# the silvann_core.py beside this file, imported before the runtime leads the path: the development tree's runtime
# folder holds another package of that name
from silvann_core import default_config                                   # noqa: E402
sys.path.insert(0, str(RUNTIME))
import silvann_runtime as SR                                              # noqa: E402
from silvann_runtime import autotune, experts_file, registry, skills, tools, tools_config     # noqa: E402
from silvann_runtime import sso, users                                    # noqa: E402
from silvann_runtime.chat import Chat, ToolMarkup                         # noqa: E402
from silvann_runtime.fetch import ensure_model                            # noqa: E402
from silvann_runtime.machine import CPU_FAMILIES                         # noqa: E402
from silvann_runtime.session import Busy, Cartridge, Runtime, SeatedRuntime, _state_read, sampler, state_bytes     # noqa: E402
from silvann_runtime import image_session                                 # noqa: E402
from silvann_runtime.qwen_image import Stopped, image_pack                 # noqa: E402
from silvann_runtime.vision import MAX_VIDEO_TOKENS                       # noqa: E402

IDLE_EVICTION_SECONDS = 120.0         # a page that stopped asking after its conversation without saying it went
GONE_PARK_SECONDS = 20.0              # seats: a conversation whose page went, put away this long after its last answer
PANIC_GRACE_SECONDS = 2.0             # a panic's error on its way to the UI before the server ends (▶ `_stopped`)
PANIC_END_SECONDS = 15.0              # and the HTTP server's own shutdown, before the process is ended without it
PICTURE_WAIT_SECONDS = 600.0          # how long (image …) waits on a picture server drawing another's picture
EVICTION_TICK_SECONDS = 5.0


# ── requests ────────────────────────────────────────────────────────────────────────────────────────────────
class CreateConvReq(BaseModel):
    name: Optional[str] = None
    system: Optional[str] = None
    lora: Optional[str] = None
    tools_allowed: Optional[list[str]] = None   # a subagent's REPL: the commands its launch allows (▶ agents.py)
    agent: bool = False                     # a subagent's conversation: its REPL launches no agents of its own


class MessageReq(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None    # the most tokens the answer runs; none: until it ends, is stopped or fills the context
    images: list[str] = []


class UpdateReq(BaseModel):
    name: Optional[str] = None
    enable_thinking: Optional[bool] = None
    tools: Optional[bool] = None            # the model's tool: on or off, as the thinking switch (▶ _tools_on)
    tools_mode: Optional[str] = None        # what its commands may do unasked: "ask", "chill" or "full-auto"


class AnswerReq(BaseModel):
    choice: str                             # one of the ask's choices: "folder", "once", "refuse"
    max_new_tokens: Optional[int] = None    # as a message's


class SaveReq(BaseModel):
    password: Optional[str] = None


# ── passwords: scrypt, as before ────────────────────────────────────────────────────────────────────────────
def _cards(m):
    """How many cards the model is split over: the Qwen runtime's `CARDS`, GLM's and DeepSeek's `P`."""
    return int(getattr(m, "CARDS", None) or getattr(m, "P", None) or 1)


def _loading():
    """`progress(done, total, what)` for a model's load: a dot every 2.5% and the tens, the whole line so far each time it
    grows (`loading weights: ...10...20...30.`) — the UI shows it as one line growing — so a long load (GLM's experts
    copied into RAM, minutes) is seen to move. `what` names what is loading: the weights, unless the load says the
    engrams (an n-gram memory's tables, ▶ ram_table.steps), and each has a line of its own. A load of the weights in
    passes — the dense part, then the experts copied into RAM — climbs again after the pass before, on the same line
    (`loading weights: ...10...100 · ...10.`): a pass begins where `done` falls back."""
    said, last, before, named = [0], [0], [""], [None]

    def tick(done, total, what="weights"):
        if what != named[0]:                              # another thing loading: a line of its own
            named[0], before[0], said[0] = what, "", 0
        elif done < last[0]:                              # a new pass: the line so far kept, this one after it
            before[0] += _dots(said[0]) + " · "
            said[0] = 0
        last[0] = done
        step = done * 40 // max(1, total)                 # 40 steps of 2.5%
        if step > said[0]:
            said[0] = step
            print("[silvann_server] loading %s: %s%s" % (what, before[0], _dots(step)), flush=True)
    return tick


def _dots(step):
    """A pass's share of the loading line: a dot every 2.5%, the tens written."""
    return "".join(str(i * 10 // 4) if i % 4 == 0 else "." for i in range(1, step + 1))


_hash_password, _verify_password = users.hash_password, users.verify_password   # a conversation's, as a user's


# ── the server's state ──────────────────────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.model = self.runtime = self.chat = None
        self.config = {}
        self.model_id = ""
        self.dir = DATA / "user_sessions"
        self.owner = {}                     # cartridge id -> the web user who has it open
        self.where = {}                     # cartridge id -> (its folder, its user or None), as found (▶ `_locate`)
        self.user_boxes = {}                # a user -> their tools over the machine's (▶ users.UserToolbox)
        self.last_ping = {}
        self.gone = {}                      # web user -> when its page went (▶ `/sessions/gone`)
        self.answering = set()              # cartridge ids whose answer is being made
        self.answered = {}                  # cartridge id -> when its last answer ended
        self.turns = {}                     # cartridge id -> its turn in flight
        self.busy_cart = None
        self.images = {}                    # cartridge id -> {image id: a picture read, waiting for its message}
        self.lock = asyncio.Lock()          # the endpoints that touch the card, one at a time
        self.tools = "chill"                # the config's tools: a new conversation's mode for its tool, or "no"
        self.classifier = None              # tools.json's folders, every conversation's (each its own mode)
        self.workspace = DATA / "workspace"
        self.library = DATA / "tools_library.json"   # the definitions the REPL kept, every conversation's
        self.skills = [DATA / "skills", SKILLS_SHIPPED]   # the folders the REPL's skills are in
        self.tool_sessions = {}             # cartridge id -> its REPL
        self.asks = {}                      # cartridge id -> its answer, waiting on the user's answer to an ask
        self.image = False                  # the model is a picture model (▶ `_boot_image`): a message makes a picture
        self.setup = None                   # what the text model booted with, so a switch boots it again the same
        self.picture = None                 # the picture model the REPL's (image …) draws with (▶ `_tool_picture`)
        self.no_picture = ""                # why there is none: what (image …) answers the model at once
        self.toolbox = None                 # what the REPL reaches on this machine, as the tools file sets it up and
                                            #   the startup check found it (▶ tools_config.Toolbox), every session's
        self.web = None                     # how the REPL's fetch goes onto the web (▶ tools.Web)
        self.listen = ("127.0.0.1", 8765)   # the address and port the server listens on
        self.offer = None                   # this server offered to the agent entry's registry (▶ registry.Offer)
        self.panic = None                   # the evaluator's stop the server is ending on (▶ `_stopped`)
        self.http = None                    # the HTTP server, which a panic ends (▶ `_stopped`)


S = State()

# ── the users (NN-150, ▶ silvann_runtime/users.py) ─────────────────────────────────────────────
# ⚖ 2026-11-02 *"users have a folder with at least a password and they get in and only see their chat folder"*. With no
#   user made, a server is as it was: no sign-in, every conversation the machine's. With one, every request but /health
#   and the sign-in carries a token (`Authorization: Bearer …`, ▶ `_UsersGate`), and a user's conversations are the ones
#   in their folder — another user's answers 404, as one that is not there.
USERS = users.Users(DATA)
SSO = sso.SignOn(USERS, DATA)                       # a provider to sign on with (NN-157), where sso.json names one
_ME = contextvars.ContextVar("silvann_user", default=None)    # who the request in hand is from, set by the gate
CID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")                     # a conversation's id: a name, never a path


def _me():
    return _ME.get()


def _folder_of(name):
    """Where `name`'s conversations of this model are — a user's folder — or, for None, the machine's."""
    return Path(USERS.sessions(name, S.model_id)) if name else S.dir


def _locate(cid):
    """A conversation's folder and its user — the machine's (None) or the user whose folder holds it — as remembered
    from its making (▶ `_made_by`) or found on disk; one moved since (`silvann users adopt`) found where it is now.
    ⛔ ONE THE SERVER KNOWS IS NEVER ANOTHER'S: its folder gone from under it, it is made again where it was, its user's;
    only a conversation never made is the folder of whoever asks."""
    if not CID.match(cid or ""):
        raise HTTPException(404, "cartridge %r not found" % cid)
    got = S.where.get(cid)
    if got is not None and got[0].is_dir():
        return got
    for name in [None] + USERS.names():
        d = _folder_of(name) / cid
        if d.is_dir():
            S.where[cid] = (d, name)
            return S.where[cid]
    if got is not None:
        return got
    return _folder_of(_me()) / cid, _me()


def _made_by(cid):
    """A conversation just made or imported: its folder and its user, the one who asked — remembered from now on."""
    S.where[cid] = (_folder_of(_me()) / cid, _me())


def _cdir(cid):
    return _locate(cid)[0]


def _owner(cid):
    return _locate(cid)[1]


def _sidecar(cid):
    p = _cdir(cid) / "server_meta.json"
    try:
        return json.load(open(p)) if p.exists() else {}
    except Exception:
        return {}


def _write_json(path, obj):
    """`obj` to `path` whole or not at all: written beside it, then put in its place — a process killed mid-write
    leaves the last one."""
    part = str(path) + ".part"
    with open(part, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(part, path)


def _write_sidecar(cid, meta):
    _cdir(cid).mkdir(parents=True, exist_ok=True)
    _write_json(_cdir(cid) / "server_meta.json", meta)


def _user_system():
    """Whether a conversation's creator may give its own system prompt: the config's `user_system_prompt`, `"no"` by
    default — the system prompt is the server's."""
    return S.config.get("user_system_prompt", "no") in ("yes", True)


def _conversation_config(cid):
    """A conversation's own settings, `conversation_config.json` in its folder — exported and imported with it."""
    p = _cdir(cid) / "conversation_config.json"
    try:
        return json.load(open(p)) if p.exists() else {}
    except Exception:
        return {}


def _system(cid):
    """A conversation's system prompt: its own where the config lets its creator give one, else the config's `system`,
    else None, the model's own (▶ Chat.turn)."""
    own = _conversation_config(cid).get("system") if _user_system() else None
    return own or S.config.get("system")


def _check_password(cid, provided):
    stored = _sidecar(cid).get("password")
    if not stored:
        return
    if not provided:
        raise HTTPException(401, "password required")
    if not _verify_password(provided, stored):
        raise HTTPException(401, "wrong password")


def _meta(cid):
    p = _cdir(cid) / "meta.json"
    return json.load(open(p)) if p.exists() else None


def _persist(c, with_state=True):
    """A cartridge to its folder: its tokens (and its state, read off the card if it is there) and what it belongs to."""
    d = _cdir(c.id)
    d.mkdir(parents=True, exist_ok=True)
    S.runtime.save(c, str(d / "cartridge.npz"), with_state=with_state)
    _write_json(d / "meta.json", {"model_id": S.model_id, "identity": c.identity, "lora": c.lora,
                                  "current_pos": c.length, "used": len(c.tokens)})


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
    if hasattr(S.runtime, "forget"):
        S.runtime.forget(c)                  # a seat's, the same
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
    p = _cdir(cid) / "cartridge.npz"
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
            "used": len(c.tokens) if c else int(meta.get("used", meta.get("current_pos", 0))),
            "context": _context(),
            "enable_thinking": extra.get("enable_thinking", S.config.get("thinking", True)),
            "tools": _tools_on(extra), "tools_mode": _tools_mode(extra),
            "ask": S.asks[cid].ask if cid in S.asks else None,
            "protected": bool(side.get("password")), "lora": (meta.get("lora") or {}).get("name"),
            "bound": c is not None, "bound_by_self": owner is not None and owner == x_web_user_id,
            "bound_by_other": owner is not None and owner != x_web_user_id}


def _context():
    """The longest conversation the model takes, in tokens, which the chat's meter measures a conversation against: 0
    for a picture model, and while no model is up."""
    return 0 if S.image else int(getattr(S.model, "max_context", 0) or 0)


# ── boot ────────────────────────────────────────────────────────────────────────────────────────────────────
@contextmanager
def _one_boot_at_a_time():
    """Held from the setup's trials to the model on the card: two servers started together on one machine each try the
    card as the other leaves it, not both an empty card (a lock file in the machine's temporary folder; none on a system
    without `fcntl`, which runs one server)."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is None:                                 # outside the except: what the boot raises is not chained to it
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
    asked = cfg.get("max_context", "auto")      # a number, or "auto" ("max"): what the card holds, to the model's length
    asked = "max" if str(asked).lower() in ("max", "auto") else int(asked)
    # a video's tokens at most, and what reading one holds of RAM beside the experts (▶ vision.MAX_VIDEO_TOKENS)
    S.video_tokens = int(cfg.get("max_tokens_per_video", MAX_VIDEO_TOKENS))
    # how many of the machine's cards the model runs on, and whether they serve one conversation split over them or a
    #   conversation each (`multi_user`) — the placement a config names (▶ `SR.with_cards`)
    options = SR.with_cards(str(folder), cfg.get("num_cards"), cfg.get("options", {}), cfg.get("multi_user"))
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
        # a ring's machines share one context: the same number in each config that names one, and every machine left to
        #   "auto" boots with what its card holds beside its layers — the ring then holds the least of them (▶ `pipeline._agree`)
        options["layers"] = layers
    with _one_boot_at_a_time():
        named = options                                   # the config's own: what a second choice is made around
        options, max_context, why = autotune.choose(str(folder), asked, named, video_tokens=S.video_tokens,
                                                    margin_mb=cfg.get("vram_margin_mb"))
        arch = SR.ARCHITECTURES.get(SR.architecture_of(str(folder)))
        apart = getattr(arch, "CONVERSATION_A_CARD_OPTION", "cards")          # a conversation a card's option (▶ with_cards)
        n = int(options.get("tensor_parallel") or len(options.get("cards") or options.get(apart) or ()) or 1)
        if n > 1:
            seated = getattr(arch, "CARDS_ARE_SEATS", False) or (apart in options and getattr(arch, "CONVERSATION_A_CARD",
                                                                                              False))
            why = ("a conversation on each of %d cards; %s" if seated else "every layer over %d cards; %s") % (n, why)
        S.why = why
        print("[silvann_server] the setup: %s" % why, flush=True)
        try:
            S.model = SR.open_model(str(folder), lora=cfg.get("lora"), max_context=max_context, progress=_loading(), **options)
        except Exception as ex:                               # noqa: BLE001 — the engine's refusal is a plain error
            if "kept in autotune.json" not in why or not autotune.refused_by_card(ex):
                raise
            options, max_context, why = autotune.choose(str(folder), asked, named, fresh=True,
                                                        video_tokens=S.video_tokens, margin_mb=cfg.get("vram_margin_mb"))
            S.why = why
            print("[silvann_server] the card refused the setup kept for it; chosen again: %s" % why, flush=True)
            S.model = SR.open_model(str(folder), lora=cfg.get("lora"), max_context=max_context, progress=_loading(), **options)
    if ring:
        if not hasattr(S.model, "forward"):
            raise SR.Refused("%s runs on one machine only: its model has no stages" % folder.name)
        link = pipeline.Link(int(cfg.get("pipeline_parallelism_port", pipeline.DEFAULT_PORT)),
                             cfg["pipeline_parallelism_next"])
        say = lambda text: print("[silvann_server] " + text.replace("[pipeline] ", ""), flush=True)   # noqa: E731
        if layers[0] != 0:
            say("a stage of the ring: layers %d-%d of %d%s" % (layers[0], layers[-1], total,
                                                                ", the head" if layers[-1] == total - 1 else ""))
            pipeline.stage(S.model, link, layers[-1] == total - 1, say, fixed=asked != "max")
            S.model.m.shutdown()
            raise SystemExit(0)
        say("the first machine of the ring: layers %d-%d of %d" % (layers[0], layers[-1], total))
        S.model = pipeline.Pipeline(S.model, link, say, fixed=asked != "max")
        max_context = S.model.max_context                 # the ring's, agreed
    S.setup = dict(folder=str(folder), lora=cfg.get("lora"), max_context=max_context, options=options, ring=ring)
    S.runtime = _runtime_for(S.model)
    S.chat = Chat(S.model.folder)
    S.model_id = folder.name
    # the model's REPL starts in the workspace, where it reads unasked (▶ the classifier); tools.json keeps the folders
    #   allowed since, tools_library.json the definitions kept (▶ tools' `keep`) — every conversation's
    S.workspace = Path(cfg.get("tools_folder") or DATA / "workspace")
    S.library = DATA / "tools_library.json"
    S.skills = [DATA / "skills", SKILLS_SHIPPED]         # the user's first: theirs takes a shipped skill's name
    # ⭐ THE TOOLS FILE (NN-127): what the REPL reaches on this machine, set up in one place — `tools/default.json` in the
    #   data folder, or the one the config's `tools_config` names; its folder is never the model's to read or write
    # the tools the model adds (▶ tools' add_tool) go in skills_added/, read after the rest; both folders made now, so the
    #   model's commands find them read-only from the start
    for place in (DATA / "skills", DATA / "skills_added"):
        place.mkdir(parents=True, exist_ok=True)
    # the Python the image's server imports as it starts (`python/`, docker/install_nonfree.sh's), made now in a
    #   container, so the model's commands find it read-only before anything is there and never make it themselves
    if tools._jail.in_container():
        (DATA / "python").mkdir(parents=True, exist_ok=True)
    S.toolbox = tools_config.Toolbox(_tools_file(cfg), S.skills, added=DATA / "skills_added", code=[DATA / "python"])
    S.classifier = tools.Classifier(str(DATA / "tools.json"), "ask" if S.tools == "no" else S.tools,
                                    allowed=[str(S.workspace)], sealed=[S.toolbox.file.folder()])
    S.web = tools.Web()
    for line in _tools_ready(cfg):
        print("[silvann_server] %s" % line, flush=True)
    S.dir = DATA / "user_sessions" / folder.name
    S.dir.mkdir(parents=True, exist_ok=True)
    S.where.clear()
    lent = S.model.lent() if hasattr(S.model, "lent") else None
    print("[silvann_server] %s (%s) booted in %.0f s — context %d, LoRA %s%s"
          % (folder.name, S.model.folder.architecture, time.time() - t0, S.model.max_context, cfg.get("lora"),
             "; the tier holds experts in the cache's unwritten rows, %.1f GB, given back 8K positions at a time as a "
             "conversation grows" % lent[1] if lent else ""), flush=True)


def _tools_file(cfg):
    """The machine's tools file: `tools_config` names it — a name, `tools/<name>.json` in the data folder, or a path —
    and `"default"` is `tools/default.json`. ⛔ One its reader cannot take refuses the boot, a line what."""
    name = str(cfg.get("tools_config") or "default")
    path = Path(name) if os.path.isabs(name) or name.endswith(".json") else DATA / "tools" / (name + ".json")
    f = tools_config.ToolsFile(path)
    try:
        wrong = f.problems()
    except (OSError, ValueError) as ex:
        raise SR.Refused("the tools file %s is not read: %s" % (path, ex))
    if wrong:
        raise SR.Refused("; ".join(wrong))
    return f


def _tools_ready(cfg):
    """⭐ THE TOOLS FILE KEPT, AND THE STARTUP CHECK (NN-127) — at boot and on `POST /tools/rebuild`: the file's entries
    made to match the commands and skills installed (▶ tools_config.ToolsFile.sync), the picture model found, each
    skill's check called with its settings. ⚖ 2026-10-26 *"a startup check could see if searXNG is in a container it
    can reach and same for qwen image and doing so will comment a line on the skills if it is not available"*: a command
    its entry leaves on "auto" that nothing answers is a comment in (help), with why (▶ tools_config.Toolbox's
    `checked`); one set "yes" stays listed, and the check's reason is said here. What it finds reaches every session at
    its next call. Answers what it says, a line each."""
    box = S.toolbox
    changed = box.refresh()
    said = ["the tools file %s: %s" % (box.file.path, "; ".join(changed) if changed else "as it was")]
    S.web.named(box.config("fetch").get("contact"))
    missing, cache = {}, {}
    switch = box.enabled("image")
    S.picture = _picture_model(cfg, switch)
    if S.picture is None and switch == "auto":
        missing["image"] = S.no_picture
    p = S.picture
    if p:
        where = ("" if p["host"] else ", on card %s" % p["card"] if p["card"] is not None
                 else ", on the text model's card: the model is put down while it draws")
        said.append("the REPL's (image …): draws with %s%s" % (p["name"], where))
    else:
        said.append("the REPL's (image …): %s" % _not_ready(switch, S.no_picture))
    reg = box.config("agent").get("registry")
    if box.enabled("agent") == "no":
        said.append("the REPL's (agent …): switched off in the tools file")
    elif reg:
        said.append("the REPL's (agent …): the servers the registry %s lists too" % reg)
    if S.offer is not None:
        S.offer.wake()                                # the offer follows the agent entry now, not at its next renewal
    switch = box.enabled("ui")                    # the screen (NN-139): a display that answers
    if switch != "no":
        from silvann_runtime import ui as ui_
        display = box.config("ui").get("display")
        why = ui_.why_not(None if display in (None, "auto") else str(display))
        if why and switch == "auto":
            missing["ui"] = why
        said.append("the REPL's (ui …): %s" % (_not_ready(switch, why) if why else
                                              "the screen %s" % (display if display not in (None, "auto") else
                                                                 os.environ.get("DISPLAY"))))
    else:
        said.append("the REPL's (ui …): switched off in the tools file")
    switch = box.enabled("browser")               # the browser companion: its session opened now, and kept
    if switch != "no":
        try:
            where, why = tools.browser_address(box), None
        except Exception as ex:                   # noqa: BLE001 — whatever stops it, said as the reason
            where, why = None, str(ex)
        if why and switch == "auto":
            missing["browser"] = why
        said.append("the REPL's (browser …): %s" % (_not_ready(switch, why) if why else "the companion at %s" % where))
    else:
        said.append("the REPL's (browser …): switched off in the tools file")
    if box.enabled("bash") == "no":
        said.append("the REPL's (bash …): switched off in the tools file")
    else:                                             # where the model's commands reach (▶ tools.confinement)
        c = tools.confinement(box, S.classifier.sealed, str(S.workspace))
        said.append("the REPL's (bash …): %s" % (
            "the machine but its tools file and the server's process (Landlock ABI %d)" % c.rules["abi"]
            if c.rules is not None and c.shape == "machine" else
            "the workspace alone and what is linked at its top (Landlock ABI %d)" % c.rules["abi"]
            if c.rules is not None else "refused — %s" % c.why if c.wanted else
            "everything the server's user reaches (a direct install: the tools file's bash entry, confined, says so)"))
    for sk in box.skills:
        switch = box.enabled(sk.name)
        why = None if switch == "no" else skills.run_check(sk, box.config(sk.name), cache)
        if why and switch == "auto":
            missing.update((c.name, why) for c in sk.commands)
        if switch == "no" or sk.check:
            said.append("the %s skill: %s" % (sk.name, _not_ready(switch, why) if switch == "no" or why else "ready"))
    box.checked(missing)
    return said


def _not_ready(switch, why):
    """A tool the startup check found nothing for, as the boot says it: by its entry's switch."""
    return {"no": "switched off in the tools file", "auto": "not available — %s" % why}.get(switch, "on, though %s" % why)


def _picture_model(cfg, switch="auto"):
    """The picture model a text model's REPL draws with — the tools file's `image` entry (▶ tools_config.BUILT_IN):
    `host` "local", or another SiLVANN server's address, a picture model's (▶ `_picture_from`); `model` a folder in
    `models/` or a path, "auto" the first picture model downloaded there; `card` a card of its own, or null the text
    model's (▶ `_tool_picture`) — a model config's `image_model` and `image_card`, the older keys, over the entry's
    where they name one. Or None: switched off, none, or a folder that is not a picture model's — `S.no_picture` says
    why (said, not refused)."""
    entry = S.toolbox.config("image")
    name = cfg.get("image_model")
    name = entry.get("model") or "auto" if name in (None, "auto") else name
    fetch = "the user can download one, Qwen-Image 2.1, from the UI's Options"
    if switch == "no":
        S.no_picture = "the tools file switches (image …) off on this machine"
        return None
    if not name or str(name).lower() in ("no", "none", "false"):
        S.no_picture = "pictures are turned off in this model's config (image_model)"
        return None
    host = str(entry.get("host") or "local").rstrip("/")
    if host.lower() != "local":
        return _picture_server(host)
    if str(name).lower() == "auto":
        found = sorted(d for d in (DATA / "models").glob("*") if d.is_dir() and image_pack(str(d)))
        if not found:
            S.no_picture = "no picture model is installed on this machine (%s)" % fetch
            return None
        name = str(found[0])
    path = Path(name) if os.path.isabs(str(name)) else DATA / "models" / str(name)
    if not image_pack(str(path)):
        S.no_picture = "the picture model named (%s) is not installed (%s)" % (path.name, fetch)
        return None
    settings = cfg.get("image")
    if settings is None:                              # the picture model's own config's
        try:
            settings = json.load(open(path / "configs" / "default.json")).get("image")
        except (OSError, ValueError):
            settings = None
    card = cfg["image_card"] if cfg.get("image_card") is not None else entry.get("card")
    if card is None and S.setup["ring"]:
        S.no_picture = "a model run over several machines draws only on a card of its own (the tools file's image card)"
        return None
    return dict(path=str(path), settings=settings or {}, card=card, host=None, name=path.name)


def _picture_server(host):
    """The picture server the tools file's image `host` names, as `_picture_model` answers one, where it answers
    and serves a picture model — else None, `S.no_picture` why."""
    import httpx
    if not host.startswith(("http://", "https://")):
        S.no_picture = "the tools file's image host is \"local\" or a picture server's address, http://host:port, not %r" % host
        return None
    try:
        h = httpx.get(host + "/health", timeout=5).json()
    except (httpx.HTTPError, ValueError) as ex:
        S.no_picture = "the picture server the tools file names, %s, does not answer (%s)" % (host, type(ex).__name__)
        return None
    if not h.get("pictures"):
        S.no_picture = "the server the tools file names, %s, serves %s, not a picture model" % (host, h.get("model_id"))
        return None
    return dict(path=None, settings={}, card=None, host=host, name="%s at %s" % (h.get("model_id"), host))


def _tool_picture(cid, text):
    """⭐ THE REPL'S (image "prompt") — a picture from the picture model, in the conversation's `pictures/`.
    ⚖ 2026-10-16: *"do a scheduler switch where the current model goes down, loads the image model, responds and then it
    reloads the old model and the old conversation"*. On the text model's card: the conversation on the card read off
    into RAM, the text model shut down, the picture made by the picture worker, the worker ended, the text model booted
    again as it booted first (`S.setup`), and the runtime handed it — the conversation's state written back when its
    answer goes on (▶ `Runtime._activate`). On a card of its own (the tools file's image `card`) the text model stays up. While it draws,
    the runtime is busy: another conversation's turn is answered 503 and the eviction waits. Answers the model's text."""
    if S.picture is None:                             # answered at once, so the model can tell the user
        raise SR.Refused("no picture can be made: %s. Do not look for one or try to install it: tell the user now that "
                         "you cannot make pictures here, and why." % S.no_picture)
    p, session = S.picture, S.tool_sessions.get(cid)
    answering = getattr(session, "answering", None)
    if p["host"]:
        return _picture_from(p, cid, text, answering)
    model = image_session.ImageModel(p["path"], silicon=getattr(S.model.m, "silicon", None), settings=p["settings"])
    rt = image_session.ImageRuntime(model, card=p["card"])
    prompt, settings = rt.settings(text)
    if not prompt:
        raise SR.Refused("a picture needs a prompt")
    rt_text = S.runtime
    holder = _Holding()
    rt_text._running = holder                         # busy: no other turn takes the card meanwhile
    switch = p["card"] is None
    t0 = time.time()

    def progress(stage, done, total, which=0):
        if answering is not None:
            answering._emit("progress", _picture_progress(stage, done, total, which, settings["count"]))

    try:
        if switch:
            _put_down()
            print("[picture] %s: the text model put down to draw (%.1f s)" % (cid, time.time() - t0), flush=True)
        try:
            # on the text model's card, which is free now, every card this process sees draws; on a card of its own, that
            #   card alone, the pictures one after another
            made = image_session.pictures(rt, prompt, settings, str(_cdir(cid) / "pictures"), progress,
                                          cards=_picture_cards() if switch else None)
        finally:
            rt.close()
            if switch:
                t1 = time.time()
                if answering is not None:
                    answering._emit("progress", {"stage": "reload", "done": 0, "total": 0,
                                                 "text": "the model coming back"})
                _bring_back()
                print("[picture] %s: the text model back in %.0f s" % (cid, time.time() - t1), flush=True)
    finally:
        holder.done.set()
    if answering is not None:                         # the pictures in the answer the user reads, as a picture model's
        answering._show("token", "\n\n" + "\n\n".join(image_session.caption(name, st, seconds)
                                                         for name, seconds, st in made) + "\n\n", token_id=-1)
    return "; ".join("a picture of %d×%d, %d steps, seed %d, made in %d s, is shown to the user in the conversation "
                     "(pictures/%s)" % (st["width"], st["height"], st["steps"], st["seed"], round(seconds), name)
                     for name, seconds, st in made)


def _picture_from(p, cid, text, answering):
    """⭐ (image "prompt") DRAWN BY ANOTHER SiLVANN SERVER — the tools file's image `host`, whose model is a picture
    model: a conversation of its own there, the prompt its message, its progress said here as it comes, each picture it
    made copied into this conversation's `pictures/`, and its conversation there deleted. The text model stays up. A
    server drawing another picture answers 503: asked again as it says, for PICTURE_WAIT_SECONDS at most."""
    import httpx
    host, said, t0 = p["host"], "", time.time()
    http = httpx.Client(base_url=host, timeout=httpx.Timeout(30.0, read=None),
                        headers={"X-Web-User-Id": "silvann-%s-%s" % (S.model_id, cid)})
    try:
        made = http.post("/cartridges", json={"name": "a picture for %s" % S.model_id})
        made.raise_for_status()
        there = made.json()["id"]
        try:
            http.post("/cartridges/%s/bind" % there).raise_for_status()
            while not said:
                with http.stream("POST", "/cartridges/%s/messages" % there, json={"text": text}) as r:
                    if r.status_code == 503 and time.time() - t0 < PICTURE_WAIT_SECONDS:
                        if answering is not None:
                            answering._emit("progress", {"stage": "wait", "done": 0, "total": 0,
                                                         "text": "the picture server is drawing another picture"})
                        time.sleep(float(r.headers.get("Retry-After") or 5))
                        continue
                    if r.status_code != 200:
                        raise SR.Refused("the picture server %s: HTTP %d %s" % (host, r.status_code, r.read()[:300]))
                    kind = None
                    for line in r.iter_lines():
                        if line.startswith("event:"):
                            kind = line[6:].strip()
                        elif line.startswith("data:"):
                            d = json.loads(line[5:])
                            if kind == "progress" and answering is not None:
                                answering._emit("progress", d)
                            elif kind == "token":
                                said += d.get("text", "")
                            elif kind == "error":
                                raise SR.Refused("the picture server %s: %s" % (host, d.get("error")))
            names = re.findall(r"!\[picture\]\(picture:([A-Za-z0-9_.-]+\.png)\)", said)
            if not names:
                raise SR.Refused("the picture server %s made no picture: %s" % (host, said[:300]))
            folder = _cdir(cid) / "pictures"
            folder.mkdir(parents=True, exist_ok=True)
            for name in names:
                got = http.get("/cartridges/%s/pictures/%s" % (there, name))
                got.raise_for_status()
                (folder / name).write_bytes(got.content)
        finally:
            try:
                http.delete("/cartridges/%s" % there)
            except httpx.HTTPError:
                pass
    except httpx.HTTPError as ex:
        raise SR.Refused("the picture server %s: %s" % (host, ex))
    finally:
        http.close()
    if answering is not None:                         # the pictures in the answer the user reads, its captions theirs
        answering._show("token", "\n\n" + said.strip() + "\n\n", token_id=-1)
    made = dict(zip(names, re.findall(r"_(\d+)×(\d+) · (\d+) steps · seed (\d+) · (\d+) s_", said)))
    told = []
    for n in names:
        w, h, steps, seed, seconds = made.get(n) or ("?",) * 5
        told.append("a picture of %s×%s, %s steps, seed %s, made by the picture server %s in %s s, is shown to the user in "
                    "the conversation (pictures/%s)" % (w, h, steps, seed, host, seconds, n))
    return "; ".join(told)


def _tool_agent(cid, model, config, task):
    """⭐ THE REPL'S (agent 'model "config" "task") — a subagent's session (▶ silvann_runtime/agents.py, the design in
    src/docs/design/agents.md): a conversation of the server that serves `model` — this one, or a worker of the
    config's `workers` — under the config's LoRA and its allowed commands; the caller is then inside it until (exit)."""
    from silvann_runtime import agents
    model = S.model_id if model in ("self", "") else model
    conf = _agent_config(model, config)
    session = S.tool_sessions.get(cid)
    answering = getattr(session, "answering", None)
    say = (lambda text: answering._emit("progress", {"stage": "agent", "done": 0, "total": 0, "text": text})) \
        if answering is not None else None
    if model != S.model_id:
        endpoint = _worker_for(model)
    elif _switches_to(conf.get("lora")):
        endpoint = S.endpoint
    else:                                             # its own model under a LoRA it cannot switch to: booted again
        endpoint = None
        conf = dict(conf, max_context=conf.get("max_context", S.setup["max_context"]),
                    options=dict(S.setup["options"], **(conf.get("options") or {})))
    restore = None
    if endpoint is None:                              # in place: the card lent for the session
        folder = Path(S.setup["folder"]) if model == S.model_id else Path(model) if os.path.isabs(model) \
            else DATA / "models" / model
        if not (folder / "pack.json").exists() or image_pack(str(folder)):
            if model in _reachable():
                raise SR.Refused("%s is another server's model, and that server is busy: ask again later, or name "
                                 "another model" % model)
            raise SR.Refused(_unknown(model))
        if S.setup["ring"]:
            raise SR.Refused("a model run over several machines does not lend its card: a worker must serve %s" % model)
        if say:
            say("the card lent to %s%s" % (model, " under the LoRA %s" % conf["lora"] if conf.get("lora") else ""))
        rt, restore = _lend(folder, conf)
        endpoint = S.endpoint
    busy = S.busy_cart
    try:
        owner = _owner(cid)                           # a user's conversation: its subagent's on this server is theirs
        token = USERS.issue(owner) if owner and endpoint == S.endpoint else None
        link = agents.Link(endpoint, model, conf, task, user="agent-%s" % cid, progress=say, token=token)
    finally:
        if restore is not None:
            restore()                                 # the card back to the caller, who reads the answer
        S.busy_cart = busy                            # the caller's answer is still the one on the card
    return _Kept(link, busy, (folder, conf, rt) if restore is not None else None)


class _Kept:
    """A subagent's link, handing the caller's answer back the card's `busy_cart` after each of its turns."""

    def __init__(self, link, busy, lent=None):
        self.link, self.busy, self.lent = link, busy, lent    # lent: (folder, config, runtime) where it runs in place
        self.first, self.name = link.first, link.name

    @property
    def ask(self):
        return self.link.ask

    def _kept(self, fn, *args):
        restore = None
        try:
            if self.lent is not None:                 # in place: the card lent for this exchange
                _, restore = _lend(*self.lent)
            self.link.http.post("/cartridges/%s/bind" % self.link.cid)   # back if the eviction put it away meanwhile
            return fn(*args)
        finally:
            if restore is not None:
                restore()
            S.busy_cart = self.busy

    def say(self, text):
        return self._kept(self.link.say, text)

    def answer(self, choice):
        return self._kept(self.link.answer, choice)

    def shown(self, reply):
        return self.link.shown(reply)

    def close(self):
        if self.lent is None:
            return self.link.close()
        folder, _, rt = self.lent                     # in place: its conversation dropped without lending the card again
        cid = self.link.cid
        rt.cartridges.pop(cid, None)
        rt.active = None if rt.active is not None and rt.active.id == cid else rt.active
        S.tool_sessions.pop(cid, None)
        S.owner.pop(cid, None)
        shutil.rmtree(DATA / "user_sessions" / Path(folder).name / cid, ignore_errors=True)
        self.link.cid = None
        self.link.close()


def _whoami(cid):
    """⭐ THE REPL'S (whoami): which model this is, where, and what a subagent can be — so the model knows itself."""
    m, c = S.model, S.runtime.cartridges.get(cid)
    session = S.tool_sessions.get(cid)
    cards = _cards(m)
    lines = ["you are %s (%s), LoRA %s" % (S.model_id, m.folder.architecture, getattr(m, "active_lora", None) or "none"),
             "you run on %s: %s" % ("%d cards" % cards if cards > 1 else "one card" if m.m.silicon not in ("host",) + CPU_FAMILIES
                                    else "the CPU", m.m.silicon),
             "your context: %d positions; this conversation has used %d" % (m.max_context, c.length if c else 0),
             "your tool: %s%s" % (session.classifier.mode if session else "?",
                                  ", inside a subagent's session" if session and session.inside else "")]
    if not _sidecar(cid).get("extra", {}).get("agent"):
        loras = m.folder.loras()
        lines.append("(agent 'self …): your own model, which starts at once, or under one of your LoRAs (the config's "
                     "\"lora\"):" if loras else "(agent 'self …) starts at once: your own model (it has no LoRAs)")
        for name in loras:
            lines.append("  %s — %s; %s" % (name, m.folder.lora_about(name)["description"] or "no description",
                                             "starts at once" if _switches_to(name) else
                                             "your model is loaded again under it, in your place (minutes each way)"))
        here = sorted(d.name for d in (DATA / "models").glob("*")
                      if d.is_dir() and d.name != S.model_id and (d / "pack.json").exists() and not image_pack(str(d)))
        if here:
            lines.append("other models on this machine, loaded in your place (minutes each way): %s" % ", ".join(here))
        for w in S.config.get("workers") or []:
            import httpx
            url = w if "://" in w else "http://" + w
            try:
                h = httpx.get(url + "/health", timeout=5).json()
                lines.append("(agent '%s \"\" \"…\") starts at once on the worker %s%s" % (
                    h.get("model_id"), w, ", its LoRAs %s" % ", ".join(h["loras"]) if h.get("loras") else ""))
            except (httpx.HTTPError, ValueError):
                lines.append("worker %s does not answer" % w)
        servers, why = _registry_servers()
        if why:
            lines.append("the registry of servers: %s" % why)
        for e in servers:
            lines.append("(agent '%s \"\" \"…\") starts at once on the server %s, the registry's%s, its context %d%s"
                         % (e.get("model_id"), e["address"],
                            ", its LoRAs %s" % ", ".join(e["loras"]) if e.get("loras") else "",
                            e.get("max_context") or 0, " — busy now" if e.get("busy") else ""))
        lines.append("pictures: %s" % ("(image …) draws with %s" % S.picture["name"] if S.picture else S.no_picture))
    return "\n".join(lines)


def _switches_to(lora):
    """Whether a conversation of this server's model can run under `lora` without booting the model again: an adapter
    is written into the model's matrices between conversations, while a rank-1 mask is applied beside them and is
    chosen at boot (▶ Qwen35._ensure). No LoRA means the config's, and a LoRA the model does not have is refused when
    the conversation is made."""
    loras = S.model.folder.loras()
    if lora is None or lora not in loras:
        return True
    boot = S.setup["lora"]
    if boot in loras and S.model.folder.lora_about(boot)["kind"] == "rank1-mask":
        return lora == boot
    return S.model.folder.lora_about(lora)["kind"] != "rank1-mask"


def _agent_config(model, config):
    """A subagent's config: a JSON object written as text, a config's name in the model's `configs/`, or the name of one
    of its LoRAs — the model under it, the rest the config's defaults."""
    text = str(config).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except ValueError as ex:
            raise SR.Refused("the agent's config does not read as JSON: %s" % ex)
    folder = Path(S.setup["folder"]) if model == S.model_id else DATA / "models" / model
    if not text:
        return {}
    if not folder.is_dir():
        if model in _reachable():
            raise SR.Refused("%s is another server's model: its config is JSON text, or \"\" for its own — (agent '%s "
                             "\"\" \"the task\")" % (model, model))
        raise SR.Refused(_unknown(model))
    path = folder / "configs" / (text if text.endswith(".json") else text + ".json")
    try:
        return json.load(open(path))
    except OSError:
        if any((folder / "loras" / text / f).is_file() for f in ("pack.json", "mask.json")):
            return {"lora": text}
        have = sorted(p.stem for p in (folder / "configs").glob("*.json")) if (folder / "configs").is_dir() else []
        loras = sorted(d.name for d in (folder / "loras").iterdir()
                       if (d / "pack.json").is_file() or (d / "mask.json").is_file()) if (folder / "loras").is_dir() else []
        raise SR.Refused("%s has no config or LoRA %r (its configs: %s; its LoRAs: %s)"
                         % (model, text, ", ".join(have) or "none", ", ".join(loras) or "none"))


def _worker_for(model):
    """The first server that runs `model` and is not busy, as http://…: of the config's `workers` (host:port), then of
    the servers the registry lists (▶ `_registry_servers`) — each asked as it is chosen; or None."""
    import httpx
    urls = [w if "://" in w else "http://" + w for w in S.config.get("workers") or []]
    urls += [e["address"] for e in _registry_servers()[0] if e.get("model_id") == model]
    for url in urls:
        try:
            h = httpx.get(url + "/health", timeout=5).json()
            if h.get("model_id") == model and not httpx.get(url + "/busy", timeout=5).json().get("busy"):
                return url
        except (httpx.HTTPError, ValueError):
            continue
    return None


def _reachable():
    """The models (agent …) names besides 'self, for a refusal that says them: this machine's other text models, the
    config's workers' and the registry's servers'."""
    import httpx
    names = sorted(d.name for d in (DATA / "models").glob("*")
                   if d.is_dir() and d.name != S.model_id and (d / "pack.json").exists() and not image_pack(str(d)))
    for w in S.config.get("workers") or []:
        try:
            names.append(httpx.get((w if "://" in w else "http://" + w) + "/health", timeout=5).json()["model_id"])
        except (httpx.HTTPError, ValueError, KeyError):
            continue
    names += [e.get("model_id") for e in _registry_servers()[0]]
    return list(dict.fromkeys(n for n in names if n))


def _unknown(model):
    """The refusal of a model (agent …) does not reach: what its first argument names, and the call."""
    names = _reachable()
    return ("%s is no model (agent …) reaches: its first argument is the model's name — 'self, or %s — as (agent '%s "
            "\"\" \"the task\")" % (model, ", ".join(names) or "no other here or on another server",
                                    names[0] if names else "self"))


def _offer_settings():
    """The tools file's `agent` entry — the registry, its key, the offer and this server's address — as the offer and
    the listing read it at each use; a picture model's server, with no REPL, reads the file itself."""
    if S.toolbox is not None:
        return S.toolbox.config("agent")
    try:
        return tools_config.config_of(_tools_file(S.config), "agent")
    except SR.Refused:
        return {}


def _serving():
    """What this server's offer to the registry says it serves (▶ registry.Offer)."""
    m = S.model
    return {"model_id": S.model_id, "architecture": m.folder.architecture if m else "",
            "lora": getattr(m, "active_lora", None) or "", "loras": m.folder.loras() if m else [],
            "max_context": getattr(m, "max_context", 0) or 0, "cards": _cards(m), "pictures": bool(S.image),
            "busy": bool(S.runtime.busy()) if S.runtime else False}


def _registry_servers():
    """The text models' servers the registry of the agent entry lists, this one left out: (servers, None); ([], why)
    where it does not answer or refuses; ([], None) where the entry names none."""
    try:
        got = registry.listing(_offer_settings())
    except registry.Unreached as ex:
        return [], str(ex)
    mine = S.offer.address if S.offer is not None else None
    return [e for e in got if e.get("address") != mine and not e.get("pictures")], None


def _stopped(panic):
    """⛔ THE EVALUATOR STOPPED, AND THE SERVER STOPS WITH IT (⚖ 2026-11-03: *"considering it was not handled at all i
    think it makes sense to show it like this and stop the server... the ui is a different thing, you can maintain the ui
    up and use that to restart the server"*). The machine is shut down already (▶ silvann_runtime's `Panic`), and
    nothing boots it again: the request that met the stop answers its error, the word first, as any failure does; what
    the card held is let go of, so each conversation is saved from its tokens; the log's last line says why, and the
    process ends a moment later — once the error is on its way — with the HTTP server's own shutdown (the conversations
    saved), then `main`'s exit 1. Starting it again is the user's choice: the UI's start (▶ silvann_ui.py,
    `ServerProcess`, which shows the line)."""
    if S.panic is not None:
        return
    S.panic = panic
    if S.runtime is not None:
        S.runtime.card_lost()
    print("[silvann_server] ⛔ stopped: %s" % panic, flush=True)

    def end():
        time.sleep(PANIC_GRACE_SECONDS)
        if S.http is not None:
            S.http.should_exit = True                 # ⛳ ASKED, NOT SIGNALLED: a signal it caught it raises again as it
        time.sleep(PANIC_END_SECONDS)                 #   ends, and the process would end on it and not on `main`'s exit
        os._exit(3)                                   # a stream that would not close
    threading.Thread(target=end, daemon=True).start()


def _put_down():
    """The text model off the card: the conversation on it read off into RAM (its state written back when its answer
    goes on, ▶ `Runtime._activate`), and the model shut down. ▶ `_bring_back`."""
    if hasattr(S.runtime, "put_down"):          # seats: every seat's answer finished and its conversation read off
        S.runtime.put_down()
        S.model.m.shutdown()
        return
    a = S.runtime.active
    if a is not None and getattr(S.model, "PARKS", True):
        a.state = _state_read(S.model, a.length)
    S.runtime.active = None
    S.model.m.shutdown()


def _runtime_for(model):
    """The runtime a model's conversations take turns in: its seats', where it holds more than one conversation on the
    card at once (▶ session's `SeatedRuntime`, the model's `seats` option), else one on the card at a time."""
    return SeatedRuntime(model) if getattr(model, "SEATS", 1) > 1 else Runtime(model)


def _bring_back():
    """The text model booted again as it booted first (`S.setup`), and the runtime handed it."""
    up = S.setup
    S.model = SR.open_model(up["folder"], lora=up["lora"], max_context=up["max_context"], progress=_loading(), **up["options"])
    S.runtime.model = S.model


def _lend(folder, conf, rt=None):
    """⭐ THE CARD LENT TO A SUBAGENT FOR AN EXCHANGE — another model, no worker free: the caller's model put down, the
    subagent's booted with its config (its setup chosen as at a boot) into `rt`, its runtime kept from the exchange
    before (its conversation parked in it), and the server serving it — its conversations, its folder. Answers (rt, the
    restore): the subagent's conversation parked and its model shut down, the server's own state back and its model
    booted again. The caller thinks between exchanges, so the card is the caller's between them. Meanwhile the caller's
    runtime holds every other turn of its model off."""
    saved = dict(model_id=S.model_id, runtime=S.runtime, chat=S.chat, dir=S.dir)
    holder = _Holding()
    S.runtime._running = holder
    t0 = time.time()
    _put_down()
    try:
        asked = conf.get("max_context", "auto")
        asked = "max" if str(asked).lower() in ("max", "auto") else int(asked)
        options, ctx, why = autotune.choose(str(folder), asked, SR.with_cards(str(folder), conf.get("num_cards"),
                                                                          conf.get("options") or {},
                                                                          conf.get("multi_user")),
                                            video_tokens=S.video_tokens,
                                            margin_mb=conf.get("vram_margin_mb", S.config.get("vram_margin_mb")))
        m = SR.open_model(str(folder), lora=conf.get("lora"), max_context=ctx, progress=_loading(), **options)
    except Exception:
        _bring_back()
        holder.done.set()
        raise
    if rt is None:
        rt = _runtime_for(m)
    else:
        rt.model = m
    S.model, S.runtime, S.chat, S.model_id = m, rt, Chat(m.folder), Path(folder).name
    S.dir = DATA / "user_sessions" / S.model_id
    S.dir.mkdir(parents=True, exist_ok=True)
    S.where.clear()                                       # another model's conversations, found again
    print("[agent] the card lent to %s in %.0f s (%s)" % (S.model_id, time.time() - t0, why), flush=True)

    def restore():
        t1 = time.time()
        try:
            _put_down()                               # the subagent's conversation parked in its runtime, its model off
        finally:
            S.model_id, S.runtime, S.chat, S.dir = saved["model_id"], saved["runtime"], saved["chat"], saved["dir"]
            S.where.clear()
            try:
                _bring_back()
            finally:
                holder.done.set()
        print("[agent] the card back to %s in %.0f s" % (S.model_id, time.time() - t1), flush=True)
    return rt, restore


def _picture_cards():
    """The cards this process sees, by index — several pictures draw one on each (▶ image_session.pictures)."""
    return list(range(len(autotune.card_memory() or [None])))


def _picture_progress(stage, done, total, which, count):
    """A picture's progress as the stream says it — which of `count` pictures, where there are several."""
    said = {"encode": "reading the prompt", "step": "step %d of %d" % (done, total), "decode": "decoding"}[stage]
    if count > 1:
        said = "picture %d of %d: %s" % (which + 1, count, said)
    return {"stage": stage, "done": done, "total": total, "text": said, "picture": which}


class _Holding:
    """What the runtime's turn in flight is while a picture is drawn for a call: never done until it ends."""

    def __init__(self):
        self.done = threading.Event()
        self.cancelled = False


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
    S.library = DATA / "tools_library.json"
    S.skills = [DATA / "skills", SKILLS_SHIPPED]         # the user's first: theirs takes a shipped skill's name
    S.model_id = folder.name
    S.dir = DATA / "user_sessions" / folder.name
    S.dir.mkdir(parents=True, exist_ok=True)
    S.where.clear()
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


def _seen(user):
    """The web user's page is there: it asked after its conversations now."""
    if user:
        S.last_ping[user] = time.time()
        S.gone.pop(user, None)


def _parked_now(cid, owner, now):
    """Whether the idle eviction puts the conversation `cid` away now (⚖ 2026-11-13 *"i would start the timer of eviction
    after the response has completed... and then give it 20 seconds before it gets parked in a multi seat setup or until
    the user changes conversation if it is single seat"*). Never while its answer is being made: the time counts from
    the end of its last answer. With seats, `GONE_PARK_SECONDS` after its page went (the UI says so: `/sessions/gone`).
    With one seat, the conversation on the card stays until its user opens another (`/bind` puts the one before away);
    one read off into RAM when another took the card goes as before. Either way, a page that stopped asking after it
    without a word (`/sessions/ping`) leaves it `IDLE_EVICTION_SECONDS` after that or its answer, whichever is later."""
    if owner is None or cid in S.answering:
        return False
    ended = S.answered.get(cid, 0.0)
    if hasattr(S.runtime, "held"):                            # seats
        gone = S.gone.get(owner)
        if gone is not None and now - max(gone, ended) >= GONE_PARK_SECONDS:
            return True
    elif getattr(S.runtime, "active", None) is not None and S.runtime.active is S.runtime.cartridges.get(cid):
        return False
    return now - max(S.last_ping.get(owner, 0.0), ended) >= IDLE_EVICTION_SECONDS


async def _eviction_loop():
    while True:
        try:
            now = time.time()
            stale = [cid for cid, owner in list(S.owner.items()) if _parked_now(cid, owner, now)]
            for cid in stale:
                async with S.lock:
                    if not _parked_now(cid, S.owner.get(cid), time.time()):   # bound again, or answering, meanwhile
                        continue
                    print("[evict] %s" % cid, flush=True)
                    await _put_away(cid)
        except Exception as ex:
            print("[evict] sweep failed: %s" % ex, flush=True)
        await asyncio.sleep(EVICTION_TICK_SECONDS)


@asynccontextmanager
async def lifespan(app):
    sweep = asyncio.create_task(_eviction_loop())
    S.offer = registry.Offer(_offer_settings, _serving, *S.listen,
                             said=lambda line: print("[silvann_server] %s" % line, flush=True))
    S.offer.start()
    try:
        yield
    finally:
        sweep.cancel()
        await asyncio.to_thread(S.offer.stop)         # out of the listing before the conversations are saved
        if S.runtime is not None:
            S.runtime.close()
            for cid in list(S.runtime.cartridges):
                try:
                    _persist(S.runtime.cartridges[cid])
                    print("[shutdown] saved %s" % cid, flush=True)
                except Exception as ex:
                    print("[shutdown] save failed for %s: %s" % (cid, ex), flush=True)

app = FastAPI(title="SiLVANN server", lifespan=lifespan)


def _bearer(value):
    """The token of an `Authorization: Bearer …` header, or None."""
    value = (value or "").strip()
    return value[7:].strip() or None if value[:7].lower() == "bearer " else None


class _UsersGate:
    """⭐ EVERY REQUEST THROUGH ONE DOOR (NN-150): who it is from — the sign-in its `Authorization: Bearer` token names —
    set for the request (`_me`); with users on, one without a sign-in refused 401 but /health and the sign-in itself, and
    a conversation that is not the signer's answered 404, as one that is not there. One door rather than a check in each
    endpoint, so an endpoint added later is behind it too."""
    OPEN = ("/health", "/users/sign_in", "/admin/save-all",  # the last saves what is open and reads nothing: a UI
            "/users/sso", "/users/sso/start", "/users/sso/finish",   # stopping the server it started calls it unsigned;
            "/users/sso/device", "/users/sso/device/poll")           # the sign-on's steps, a sign-in's (▶ sso.py)
    NOT_A_CONVERSATION = ("estimate", "import")           # /cartridges/estimate and /cartridges/import

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        me = USERS.signed(_bearer(headers.get("authorization")))
        path = scope.get("path", "")
        if USERS.on():
            if me is None and path not in self.OPEN:
                return await self._refuse(send, 401, "sign in: this server has users (POST /users/sign_in)")
            m = re.match(r"^/cartridges/([^/]+)(?:/|$)", path)
            if m and m.group(1) not in self.NOT_A_CONVERSATION:
                cid = m.group(1)
                if not CID.match(cid) or _owner(cid) != me or not _cdir(cid).is_dir():
                    return await self._refuse(send, 404, "cartridge %r not found" % cid)
        held = _ME.set(me)
        try:
            await self.app(scope, receive, send)
        finally:
            _ME.reset(held)

    @staticmethod
    async def _refuse(send, status, detail):
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


app.add_middleware(_UsersGate)


class SignInReq(BaseModel):
    name: str
    password: str


@app.post("/users/sign_in")
def users_sign_in(req: SignInReq):
    """A sign-in: a token the requests that follow carry (`Authorization: Bearer <token>`), good until it is unused for
    twelve hours, its user's password changes, or the server stops."""
    if not USERS.on():
        raise HTTPException(409, "this server has no users: nothing to sign in to")
    name = req.name.strip().lower()
    token = USERS.sign_in(name, req.password)
    if token is None:
        raise HTTPException(401, "wrong name or password")
    return {"token": token, "name": name}


# ── single sign-on (NN-157, ▶ silvann_runtime/sso.py): the provider says who, this server checks it, then a token as above
class SsoFinishReq(BaseModel):
    code: str
    state: str


class SsoPollReq(BaseModel):
    handle: str


def _sso(step, *args):
    """A sign-on's step: refused 403 with the reason, the provider not answering 502."""
    try:
        return step(*args)
    except users.Refused as ex:
        raise HTTPException(403, str(ex))
    except sso.Unreachable as ex:
        raise HTTPException(502, "the provider: %s" % ex)


@app.get("/users/sso")
def users_sso():
    """Whether this server signs on through a provider: the UI's button's label, and whether a terminal can by a code."""
    return SSO.public()


@app.post("/users/sso/start")
def users_sso_start():
    """The provider's page to send the browser to, and the state the UI keeps to know the browser coming back."""
    return _sso(SSO.start)


@app.post("/users/sso/finish")
def users_sso_finish(req: SsoFinishReq):
    """The browser back from the provider with its code: the account checked here, then a token as a password's."""
    name = _sso(SSO.finish, req.code, req.state)
    return {"token": USERS.issue(name), "name": name}


@app.post("/users/sso/device")
def users_sso_device():
    """A terminal's sign-on: a code the user types at the provider's page, and the handle to ask after it with."""
    return _sso(SSO.device)


@app.post("/users/sso/device/poll")
def users_sso_device_poll(req: SsoPollReq):
    """`{"pending": true}` until the user has finished at the provider's page; then a token as a password's."""
    name = _sso(SSO.device_poll, req.handle)
    if name is None:
        return {"pending": True}
    return {"token": USERS.issue(name), "name": name}


@app.post("/users/sign_out")
def users_sign_out(authorization: Optional[str] = Header(default=None)):
    USERS.sign_out(_bearer(authorization))
    return {"signed_out": True}


@app.get("/users/me")
def users_me():
    """Who is signed in, how many conversations they hold, and what their settings ask that the machine keeps."""
    me = _me()
    if me is None:
        return {"user": None, "users": USERS.on()}
    return {"user": me, "users": True, "conversations": USERS.conversations(me), "kept": _user_box(me).kept()}


# ── endpoints ───────────────────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    m = S.model
    return {"ok": True, "model_id": S.model_id, "architecture": m.folder.architecture if m else None,
            "lora": m.active_lora if m else None, "loras": m.folder.loras() if m else [],
            "tenants_loaded": len(S.runtime.cartridges) if S.runtime else 0,
            "active": S.runtime.active.id if S.runtime and S.runtime.active else None,
            "seats": [c.id if c else None for c in S.runtime.held] if hasattr(S.runtime, "held") else None,
            "model_dir": m.folder.path if m else None, "pp_slice": "all", "num_gpus": _cards(m),
            "max_context": m.max_context if m else 0,
            "adapter": "silvann_runtime", "images": _reads_pictures(), "files": not S.image, "pictures": S.image,
            "picture_settings": S.model.settings if S.image else None, "user_system_prompt": _user_system(),
            "users": USERS.on(), "user": _me()}


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
    return {"saved": saved} if not USERS.on() or _me() else {"saved": len(saved)}   # no ids to a caller not signed in


@app.get("/cartridges")
def list_cartridges():
    out = []
    here = _folder_of(_me())                              # a user's own, or the machine's where there are no users
    for d in sorted(here.iterdir()) if here.exists() else []:
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
        _made_by(c.id)
        name = req.name or "chat-%s" % time.strftime("%H%M%S")
        _write_sidecar(c.id, {"id": c.id, "name": name, "created": time.time(),
                              "extra": {"history": [],
                                        "enable_thinking": S.config.get("thinking", True),
                                        "tools_allowed": req.tools_allowed, "agent": req.agent}})
        if _user_system() and req.system:
            _write_json(_cdir(c.id) / "conversation_config.json", {"system": req.system})
        await asyncio.to_thread(_persist, c, False)
        if x_web_user_id:
            S.owner[c.id] = x_web_user_id
            _seen(x_web_user_id)
    return {"id": c.id, "name": name, "enable_thinking": S.config.get("thinking", True)}


@app.post("/cartridges/estimate")
async def estimate_cartridge():
    """What a conversation costs. ⛳ The card's share is allocated once at boot for the whole context and shared by
    every conversation — only the one being served is on it — so a new conversation costs the card nothing; put away,
    it costs its state in RAM, up to the figure here. Under tensor parallelism each card holds its share of that state,
    `per_card`."""
    m = S.model
    total, cards = state_bytes(m, m.max_context), _cards(m)
    return {"bytes": int(total), "gb": total / 2 ** 30, "num_gpus": cards,
            "per_card": {"bytes": int(total // cards), "gb": total / cards / 2 ** 30}, "max_tokens": int(m.max_context)}


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
            if owner not in S.gone and time.time() - S.last_ping.get(owner, 0) < IDLE_EVICTION_SECONDS:
                raise HTTPException(423, "in use by another session")
            # a page that went: the conversation handed to the one that came back as it is — on the card or in RAM,
            #   its answer still being made — rather than put away and read again
        _load(cid)
        S.owner[cid] = x_web_user_id
        _seen(x_web_user_id)
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
    _seen(x_web_user_id)
    return {"session": x_web_user_id, "owned": [c for c, o in S.owner.items() if o == x_web_user_id],
            "idle_eviction_seconds": IDLE_EVICTION_SECONDS}


@app.post("/sessions/gone")
async def session_gone(x_web_user_id: Optional[str] = Header(default=None)):
    """The web user's page has gone — its browser left and did not come back within the UI's reconnect window. With
    seats, its conversation is put away `GONE_PARK_SECONDS` after its last answer (▶ `_parked_now`); its next ping, bind
    or message says it is back."""
    if not x_web_user_id:
        raise HTTPException(400, "missing X-Web-User-Id header")
    S.gone[x_web_user_id] = time.time()
    return {"session": x_web_user_id, "gone": True, "gone_park_seconds": GONE_PARK_SECONDS}


@app.delete("/cartridges/{cid}")
async def delete_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    async with S.lock:
        await _put_away(cid, persist=False)
        if _cdir(cid).exists():
            shutil.rmtree(_cdir(cid))
        S.where.pop(cid, None)
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
    d = _cdir(cid)
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
    _made_by(new_id)
    d = _cdir(new_id)
    d.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile() and "/" not in member.name and member.name in ("cartridge.npz", "meta.json", "server_meta.json",
                                                                                   "conversation_config.json"):
                    open(d / member.name, "wb").write(tar.extractfile(member).read())
        c = Cartridge.load(str(d / "cartridge.npz"))
        if not c.matches(S.runtime.identity):
            raise HTTPException(409, "the cartridge belongs to %s, and this server runs %s" % (c.identity, S.runtime.identity))
    except HTTPException:
        shutil.rmtree(d)
        S.where.pop(new_id, None)
        raise
    except Exception as ex:
        shutil.rmtree(d)
        S.where.pop(new_id, None)
        raise HTTPException(400, "invalid cartridge archive: %s" % ex)
    c.id = new_id
    c.save(str(d / "cartridge.npz"))
    _write_json(d / "meta.json", {"model_id": S.model_id, "identity": c.identity, "lora": c.lora,
                                  "current_pos": c.length, "used": len(c.tokens)})
    side = _sidecar(new_id)
    side["id"] = new_id
    _write_sidecar(new_id, side)
    return {"id": new_id, "name": side.get("name", new_id), "model_id": S.model_id}


@app.post("/cartridges/{cid}/stop")
async def stop_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    _check_password(cid, x_cart_password)
    t = S.turns.get(cid)
    if t is not None and not t.done.is_set():
        if hasattr(t, "cancel"):                     # the stop (NN-160): the answer ends at the token it is on
            t.cancel()
        else:
            t.cancelled = True
        return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False, "reason": "no generation in flight"}


@app.post("/cartridges/{cid}/kill")
async def kill_cartridge(cid: str, x_cart_password: Optional[str] = Header(default=None)):
    """⭐ THE PKILL (NN-160): whatever of the conversation's turn is in flight torn down — its queued work off every queue,
    its running step ended at its next move or slice — where `/stop` lets it end at the token it is on. Its answer so far
    is kept, and its state is read again from its tokens at its next message."""
    _check_password(cid, x_cart_password)
    t = S.turns.get(cid)
    if t is not None and not t.done.is_set():
        if hasattr(t, "kill"):
            t.kill()
        else:
            t.cancelled = True
        return {"ok": True, "killed": True}
    return {"ok": True, "killed": False, "reason": "no generation in flight"}


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
    _seen(x_web_user_id)
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
        new = S.chat.turn(c.tokens, text, _system(cid),
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

    def progress(stage, done, total, which=0):
        emit("progress", _picture_progress(stage, done, total, which, settings["count"]))

    def make():
        try:
            made = image_session.pictures(S.runtime, prompt, settings, str(_cdir(cid) / "pictures"), progress,
                                          cards=_picture_cards())
            answer = "\n\n".join(image_session.caption(name, st, seconds) for name, seconds, st in made)
            emit("token", {"token_id": -1, "text": answer})
        except Stopped as ex:
            answer = "_[the picture was stopped: %s]_" % ex
            emit("token", {"token_id": -1, "text": answer})
        except Exception as ex:                       # noqa: BLE001 — said to the UI
            emit("error", {"error": str(ex), "stopped": S.panic is not None})
            return
        finally:
            S.busy_cart = None
            S.answering.discard(cid)
            S.answered[cid] = time.time()
        side = _sidecar(cid)
        history = side.setdefault("extra", {}).setdefault("history", [])
        history += [{"role": "user", "content": req.text},
                    {"role": "assistant", "content": answer, "picture": dict(settings, prompt=prompt)}]
        _write_sidecar(cid, side)
        emit("done", {})

    S.busy_cart = cid
    S.answering.add(cid)
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
    p = _cdir(cid) / "pictures" / name
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


def _user_box(name):
    """A user's tools — the machine's with their settings over them (▶ users.UserToolbox) — made once a machine's toolbox."""
    got = S.user_boxes.get(name)
    if got is None or got.machine is not S.toolbox:
        got = S.user_boxes[name] = users.UserToolbox(S.toolbox, USERS.tools_file(name), data=DATA)
    return got


def _repl_place(cid):
    """Where a conversation's REPL works and what fences it: the machine's workspace, classifier and tools for a
    conversation of the machine's; for a user's, their own workspace and kept definitions, a classifier that allows
    their workspace and seals what is not theirs (▶ users.Fence), and their tools. (workspace, classifier, library, box)."""
    owner = _owner(cid)
    if owner is None:
        return S.workspace, S.classifier, S.library, S.toolbox
    home = Path(USERS.workspace(owner))
    k = tools.Classifier(str(Path(USERS.folder(owner)) / "tools_allowed.json"), S.classifier.mode, allowed=[str(home)],
                         sealed=[S.toolbox.file.folder()])
    # ⛔ A CHROOT'S VIEW (▶ users.Fence): the user's workspace to write; to read, besides, the system's programs, the
    #   tools they have switched on, the runtime and the venv — looked at afresh at each command
    box = _user_box(owner)
    k.sealed = users.Fence(USERS, owner, machine_sealed=list(k.sealed), keep=[os.path.dirname(tools.RUNNER), sys.prefix],
                           tools=box.tool_folders)
    return home, k, Path(USERS.folder(owner)) / "tools_library.json", box


def _tool_session(cid):
    s = S.tool_sessions.get(cid)
    if s is None:
        home, classifier, library, box = _repl_place(cid)
        home.mkdir(parents=True, exist_ok=True)
        extra = _sidecar(cid).get("extra", {})
        mode = _tools_mode(extra)
        s = S.tool_sessions[cid] = tools.Session(str(home), classifier.conversation(mode),
                                                 library=str(library),
                                                 guide=str(TOOLS_GUIDE),
                                                 image=lambda text: _tool_picture(cid, text),
                                                 agent=None if extra.get("agent") else
                                                 lambda model, config, task: _tool_agent(cid, model, config, task),
                                                 allowed=extra.get("tools_allowed"),
                                                 whoami=lambda: _whoami(cid), web=S.web, toolbox=box,
                                                 sees=lambda: hasattr(S.model, "vision") and S.chat.sees_in_answers())
    return s


class _Answer:
    """One message's answer in the making: the model's stretches of answer, the calls between them run and answered,
    streamed as text/event-stream — `event: token` {"token_id", "text", "used"} a token, `event: tool` {"form",
    "answer", "text"} a call, then `event: done` {"used", "context"}, `event: ask` {"question", "choices"} (and done) or
    `event: error`. `used` is the conversation's tokens so far, `context` the most it takes."""

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
        self.pictures = []                     # what each answered call showed the model ((ui 'look …)), read as pictures
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
                t = S.runtime.turn(self.c, new, room if self.budget is None else min(self.budget, room), sampler=_sampler(), stop=self.stop,
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
        # a character whose bytes the next token ends decodes as U+FFFD until then: shown once it is whole — and the text a
        #   call's markup opens with before its first token (DeepSeek's `<`), once the next token says it is not a call
        text = (self.opened + S.chat.decode(self.answer)).rstrip("�")
        if self.declared and S.chat.call_lead and text.endswith(S.chat.call_lead):
            text = text[:-len(S.chat.call_lead)]
        delta, self.shown = text[len(self.shown):], text
        self._show("token", delta, token_id=tok, used=len(self.c.tokens))

    def begin(self, new, placed):
        """The first stretch on the card now (`Busy` where another turn has it); the rest in a thread of its own."""
        self.queue, self.loop = asyncio.Queue(), asyncio.get_running_loop()
        t = self._turn(new, placed)
        S.answering.add(self.cid)
        threading.Thread(target=self._go, args=(t,), daemon=True).start()

    def resume(self, choice, value):
        """The user's answer to the ask: its value is the waiting call's, and the answer goes on."""
        self.queue, self.loop = asyncio.Queue(), asyncio.get_running_loop()
        self._show("tool", "<tool_answer>%s</tool_answer>" % choice, choice=choice)
        S.answering.add(self.cid)
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
                t = self._turn(*self._response(), retry=True)
            while True:
                t.wait()
                if self.budget is not None:
                    self.budget -= len(self.answer)
                forms = S.chat.tool_calls(self.answer) if self.declared and not t.cancelled else []
                if not forms or (self.budget is not None and self.budget <= 0):
                    if self.held:              # a call never made, shown as the model wrote it
                        self._show("token", (self.opened + S.chat.decode(self.answer))[len(self.shown):], token_id=-1)
                    break
                self.forms, self.answers, self.pictures = forms, [], []
                if self._calls() is not None:
                    return
                t = self._turn(*self._response(), retry=True)
            self._write()
            self._done()
        except Exception as ex:                # noqa: BLE001 — said to the UI
            if self.written or self.calls:     # an answer that made calls keeps what it did
                self._write()
            # `stopped`: the evaluator stopped, and the server is ending with it (▶ `_stopped`) — a client stops asking
            self._emit("error", {"error": str(ex), "stopped": S.panic is not None})
        finally:
            S.busy_cart = None
            S.answering.discard(self.cid)      # an answer that waits on an ask has ended too, until it is answered
            S.answered[self.cid] = time.time()

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
                session.answering = self          # what a call's own progress is said on (▶ `_tool_picture`)
                value = session.run(form)
                if session.pending is not None:
                    return self._ask(session.pending[1])
            self._answered(value)
        return None

    def _ask(self, asked):
        """The answer waits on the user: the ask said, the history written, the stream ended."""
        self.ask = {"question": asked.question, "choices": list(asked.choices), "touches": asked.touches,
                    "target": asked.target}
        S.asks[self.cid] = self
        self._show("tool", "<tool_ask>%s</tool_ask>" % asked.question, ask=asked.question)
        self._write()
        self._emit("ask", self.ask)
        self._done()
        return self.ask

    def _done(self):
        """The stream's end, with the conversation's tokens and the context they are counted against."""
        self._emit("done", {"used": len(self.c.tokens), "context": _context()})

    def _answered(self, value):
        self.answers.append(value)
        self.pictures.append(self._pictures())
        self._show("tool", "<tool_response>%s</tool_response>" % value, answer=value)

    def _pictures(self):
        """What the call just answered showed the model — `(ui 'look …)`'s pictures — read through its vision tower."""
        got = _tool_session(self.cid).take_pictures()
        if not got or not hasattr(S.model, "vision"):
            return []
        v = S.model.vision()
        return [v.read(png) for png in got]

    def _response(self):
        """The calls' answers as the model's next tokens, and the pictures they showed it placed at their pads, as a
        message's are (▶ chat's `tool_response`)."""
        pics = self.pictures + [[]] * (len(self.answers) - len(self.pictures))
        new = S.chat.tool_response(self.answers, self.thinking, pictures=[[im.tokens for im in ps] for ps in pics])
        flat = [im for ps in pics for im in ps]
        return new, (list(zip(S.chat.image_starts(new, S.model.vision().pads), flat)) if flat else [])

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
    tool — the modes, the folders every conversation reads unasked, where its REPL starts, and the machine's tools file
    (▶ `_tools_listed`)."""
    me = _me()
    if me:                                                # a user's own workspace and the folders their REPL allowed
        home = Path(USERS.workspace(me))
        k = tools.Classifier(str(Path(USERS.folder(me)) / "tools_allowed.json"), S.classifier.mode, allowed=[str(home)])
        return {"tools": S.tools, "modes": list(tools.MODES), "allowed": k.allowed, "workspace": str(home),
                **_tools_listed()}
    return {"tools": S.tools, "modes": list(tools.MODES), "allowed": S.classifier.allowed,
            "workspace": str(S.workspace), **_tools_listed()}


@app.post("/tools/rebuild")
async def tools_rebuild():
    """⭐ "REBUILD THE TOOLS" (⚖ 2026-10-26 *"maybe a rebuild skill tree button would also be good"*): the skills read
    again, the tools file's entries kept in step with them, and the startup check run again — what a skill installed or
    a setting changed since the boot needs. A conversation's REPL reads its skills again at its `(reset)`; what the
    check found reaches every one at once. Answers what it did, `said`, and the tools as `GET /tools` does."""
    if S.toolbox is None:
        raise HTTPException(409, "%s is a picture model: it has no tools" % S.model_id)
    try:
        said = await asyncio.to_thread(_tools_ready, S.config)
    except (OSError, ValueError) as ex:
        raise HTTPException(422, "the tools file %s: %s" % (S.toolbox.file.path, ex))
    for line in said:
        print("[silvann_server] %s" % line, flush=True)
    return {"said": said, **_tools_listed()}


def _tools_listed():
    """The tools file — where it is — and each tool's entry: its switch, what the startup check found, what it is, and
    its settings, a key, a password or a token said only as set or not (the file holds them, the UI never shows them)."""
    if S.toolbox is None:
        return {"tools_file": None, "entries": {}}
    me = _me()
    box = _user_box(me) if me else S.toolbox             # a user sees the tools as they have them
    out = box.listing()
    for e in out.values():
        e["config"] = {k: ("(set)" if v not in (None, "") else None) if _secret(k) else v for k, v in e["config"].items()}
    got = {"tools_file": S.toolbox.file.path, "entries": out}
    if me:
        got.update(user_tools_file=USERS.tools_file(me), kept=box.kept())
    return got


def _secret(key):
    return re.search(r"key|password|passwd|secret|token", str(key), re.I) is not None



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
    except (SR.Refused, SR.Panic) as ex:                      # a model not here, a config not found, a card too small;
        raise SystemExit("[silvann_server] ⛔ not started: %s" % ex)   # the evaluator stopped while the model loaded
    except Exception as ex:                                   # noqa: BLE001 — the engine's refusal is a plain error
        if autotune.refused_by_processor(ex):
            raise SystemExit("[silvann_server] ⛔ not started: %s (%s)" % (autotune.refused_by_processor(ex),
                                                                        getattr(S, "why", "")))
        if not autotune.refused_by_card(ex):
            raise
        raise SystemExit("[silvann_server] ⛔ not started: the card has no room for this setup (%s) — in the config, set "
                         "experts_on and experts_from to \"auto\" in its options, and lower max_context or set it to "
                         "\"auto\", as long as the card holds" % getattr(S, "why", ""))
    import uvicorn
    S.endpoint = "http://127.0.0.1:%d" % args.port   # the server itself, where a subagent of its own model talks to it
    S.listen = (args.host, args.port)
    SR.Machine.on_panic = _stopped
    S.http = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))
    S.http.run()
    if not S.http.started:                                    # as uvicorn.run ends one that never listened: a port taken
        raise SystemExit(3)
    if S.panic is not None:
        raise SystemExit(1)                                   # the line above it said why: `⛔ stopped:`


if __name__ == "__main__":
    main()
