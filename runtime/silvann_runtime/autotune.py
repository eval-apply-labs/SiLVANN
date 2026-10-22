"""THE SETUP A MODEL BOOTS WITH, CHOSEN FOR THE MACHINE IT BOOTS ON — the options a config leaves out (NN-83).

A model's speed turns on where its parts sit: its experts on the card or on the CPU, read from RAM or from the disk, how
much of the card holds the experts used most (a tier), how long a conversation the card's cache is sized for. `choose`
picks them by trying them on the card and measuring the RAM:

  the card     its pool is allocated when the machine boots, so a boot it cannot hold is refused at once — `MEASURED`
               2026-10-14 on an RTX 5060 Ti (measurements/2026-10-14_nn83_autotune.md): a refusal in about 1.5 s, and the
               same process boots again after it; on an MI50 (gfx906) in 2.3-2.7 s
               (measurements/2026-10-14_v080rc1_on_node03.md). Candidates are tried fastest first.
  the RAM      never tried: an allocation the system grants and cannot back ends in the OOM killer. The experts go to RAM
               when they, the process's own memory beside them and 500 MB to spare fit in what this process may hold
               (`experts_file.in_ram`; `experts_file.limit`: its cgroup's limit — a container's, a capped scope's — else
               what the system has available), else they are read from the disk.

  ① the experts on the card, where the card boots with them
  ② else on the CPU: from RAM, the largest tier the card boots with beside the context, in steps of STEP_GB; from the disk, no tier (a tier copies its experts from RAM), and GLM's prompt DISK_CHUNK positions at a
     time — `MEASURED` 2026-10-14, held to 28 GB on the RTX 5060 Ti: 10.6 positions/s against 8.3 at 256, the answer
     the same
  ③ the context halved, down to MIN_CONTEXT, while nothing boots
  every trial with the config's `vram_margin_mb` (VRAM_MARGIN_MB by default) of each card held beside it (`_held`), so
  what is chosen leaves that much free for whatever else draws on the card
  and GLM's tier exclusive (each expert on the card or in RAM) where it is a quarter of the experts or more and the card
  copies both ways at once — else inclusive (⚖ 2026-10-14)

The context comes first: the conversation's cache for the whole context is the card's before any tier (a tier a little
smaller costs a little speed; a cache too small refuses a conversation). A config's `max_context` of `"max"` takes that
further (⚖ 2026-10-14): the context grows to what the card holds beside the model, up to the model's own length
(`model_length`), and a tier only once the context is whole. The experts stay on the card at the longest context it
holds beside them where that is CARD_CONTEXT or more (⚖ 2026-10-17, NN-93) — else they go to the CPU for the whole
length, and the tier is lent the cache's rows a conversation has not reached (▶ qwen3_5's and glm5's `_lend`), so
the context costs the tier only what a conversation uses of it. Where the whole length does not fit beside the model,
a tier so lent is kept at the least the model takes — a slot a layer — and the context takes the rest of the card
(⚖ 2026-10-17: *"you set up the max tokens against the available space … it sizes the l1 cache against the actual
tokens on the moe setups"*).

The choice is kept in the model folder (`autotune.json`) under what it was made for — the card (its family, and its name
and memory where its vendor's tool says them), the machine's RAM (its cgroup's limit where there is one), the context and the config's options,
and the release (`_release`: an update chooses again) —
and the next boot on the same takes it without trying; a boot the card refuses all the same chooses again (`fresh`).

A config's options win: what it names of the placement (`experts_on`, `experts_from`, `card_experts_gb`) is kept and the
rest chosen around it — `"auto"` names nothing, so `{"experts_on": "cpu", "experts_from": "auto"}` chooses where from and
the tier — and its other options are kept in every candidate; a placement it names in full is booted as it is. A machine with no card is not tried — its "card" is the CPU's memory — nor an
integrated GPU (`khronos_opencl1_igpu`), whose memory is the PC's RAM — where pictures are read on the CPU unless the config's
`picture_pixels` gives them room.
"""
import collections
import datetime
import glob
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess

from . import experts_file
from .machine import Machine, own_memory
from .model_folder import ModelFolder, Refused
from .system import LIBRARY_SUFFIX, memory
from .vision import video_ram

PLACEMENT = ("experts_on", "experts_from", "card_experts_gb")
STEP_GB = 0.25           # ⚖ 2026-10-14 "128/256mb is good": the tier's step
VRAM_MARGIN_MB = 50      # what each card keeps free unless the config's `vram_margin_mb` says — ⚖ 2026-10-17 "we should
                         # have a vram_margin_mb parameter instead that tells us how many mb to leave available on the gpu
                         # when booting a model, so that it can be configured rather than taken at 256 even for system that
                         # run headless" · "set it as default at 50mb of margin, it should be more than enough". Held by
                         # every trial boot (`_held`), so whatever is chosen leaves it free
POOL = re.compile(r"nn__expert_cache_l1__size_mb:(\d+)")
CTX_STEP = 1024          # the context's step, where it takes what the card holds (`"max"`)
MIN_CONTEXT = 4096
NO_CARD_CONTEXT = 32768  # `"max"` on a machine with no card of its own, and a model that names no length
NOT_A_CARD = ("x86_avx2", "host", "khronos_opencl1_igpu")   # silicon whose pool is the PC's RAM: never tried — but an
                         # OpenCL 1.2 card whose memory is its own (an NVIDIA card below compute 7.5, NN-96) is a card
                         # (`machine.own_memory`), and is tried as one
TIER_MAX_GB = 128        # the binary search's ceiling, past any card's memory
DISK_CHUNK = 1024        # GLM's prompt rows a step with its experts on the disk (Qwen's chunk is at most 256)
FREED_GB = 1             # a card this much emptier than when a kept choice was made chooses again (`_freed`)
RULES = 6                # the rules' version, in what a kept choice was made for: a change to them chooses again
CARD_CONTEXT = 32768     # growing, the experts stay on the card where it holds this much context beside them (NN-93)
FILES_SHARE = 0.10       # what DeepSeek's experts in RAM leave of what this process may hold, beside the spare: the system's
                         # own and the files it reads in place (its n-gram tables, its layers mapped from their files).
                         # MEASURED node03 2026-10-21: with none left, 38 of its 40 layers in RAM filled the swap
                         # (measurements/2026-10-21_deepseek_chat_node03.md)


def _a_card(silicon, machine):
    """Whether a worker of `silicon` is a card of its own memory: not the CPU's, and not an OpenCL device that shares the
    PC's RAM (▶ `machine.own_memory`)."""
    return silicon not in NOT_A_CARD or (silicon == machine.silicon and own_memory(machine))


def refused_by_card(ex):
    """Whether a boot's error is the card's refusal of the pool's room."""
    return _refused_by_card(ex)


def _refused_by_card(ex):
    return "would not allocate the pool" in str(ex) or "allocator refused the pool" in str(ex)


def _held(params, workers, mb, machine):
    """A trial boot's sections with each card's pool `mb` larger, so what boots leaves `mb` of every card free when it
    boots for real. `params` one section (the card's) or {worker: section}; `workers` (silicon, device) a worker."""
    grow = lambda text: POOL.sub(lambda m: "nn__expert_cache_l1__size_mb:%d" % (int(m.group(1)) + mb), text, count=1)  # noqa: E731
    if not mb:
        return params
    if isinstance(params, dict):
        return {w: grow(t) if w < len(workers) and _a_card(workers[w][0], machine) else t for w, t in params.items()}
    return grow(params)


class _Prober:
    def __init__(self, folder, arch, silicon, log, margin_mb=VRAM_MARGIN_MB):
        self.folder, self.arch, self.silicon, self.log = folder, arch, silicon, log
        self.margin_mb = int(margin_mb)
        self.tried = 0
        self.video_tokens = None
        self.answer = None

    def model(self, max_context, options):
        """The architecture's model for these options, unbooted: what it plans, and refuses, before the card."""
        return self.arch(self.folder, Machine(silicon=self.silicon), max_context=max_context, **options)

    def boots(self, max_context, options, then=None):
        """Whether the card holds this setup's pool: booted, then shut down. A refusal of anything but the pool's room is
        raised — the setup is wrong, not too large. `then(model)`, run while it is booted, leaves its answer in
        `self.answer`."""
        model = self.model(max_context, options)
        machine = model.m
        params = _held(model.device_params(), model.workers(), self.margin_mb, machine)
        self.tried += 1
        try:
            if isinstance(params, dict):
                machine.boot(params[0], workers=model.workers(), worker_params=params)
            else:
                machine.boot(params)
            self.answer = then(model) if then is not None else None
            return True
        except Exception as ex:                                       # noqa: BLE001 — the engine's refusal is a plain error
            if _refused_by_card(ex):
                return False
            raise
        finally:
            machine.shutdown()

    def duplex(self, max_context, options):
        """The card's duplex ratio ×1000 for this setup's tier — a copy of an expert to the card and one back run at once,
        over the copy to the card alone (▶ `expert_tier.duplex`) — measured on a boot of it; None where it does not boot."""
        def probe(model):
            model.m.act_as(0)                                          # the card is worker 0
            return model.m.must("(nn__expert_tier__duplex %d)" % model.CPLAN.slot_bytes, "the duplex probe")
        return self.answer if self.boots(max_context, options, probe) else None

    def position_bytes(self, max_context, options):
        """What one more position costs a conversation's state near `max_context`, in bytes — the model's own sizes,
        before any boot. The pool may hold more a position than the state (GLM's index beside its latents), which only
        leaves the card's margin wider."""
        model = self.model(max_context, options)
        zeros = collections.defaultdict(int)
        size = lambda n: sum(b for _, b in model.state_regions(n, zeros))          # noqa: E731
        return (size(max_context) - size(max_context - CTX_STEP)) / CTX_STEP


def _vendor_cards():
    """(the cards' names and memory as their vendor's tool says them, "" where there is none; the bytes each card holds
    now, [] where the tool does not say) — what a card holds now is not in what a choice is kept under, which a card's
    own idle memory would change from one boot to the next."""
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"]):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            rows = [[c.strip() for c in l.split(",")] for l in out.splitlines() if l.strip()]
            if cmd[0] == "nvidia-smi":                 # name, total MiB, used MiB
                col, scale = 2, 2**20
            else:                                      # a header, then a card a row; the used memory in bytes
                head, rows = rows[0] if rows else [], rows[1:]
                col, scale = next((i for i, h in enumerate(head) if "Used" in h), None), 1
                rows = [head] + rows if col is None else rows
            try:
                used = [int(r[col]) * scale for r in rows] if col is not None else []
            except (IndexError, ValueError):
                used = []
            return " | ".join(", ".join(c for i, c in enumerate(r) if i != col) for r in rows), used
    return "", []


def card_memory():
    """Each card's memory in bytes, as its vendor's tool says it, in the order the runtime numbers the cards — the
    tool's order, narrowed by the `*_VISIBLE_DEVICES` list the runtime honours — or None where no tool says.
    ASSUMED: the vendor tool lists the cards in the runtime's order (both by PCI bus on the boxes we run); a tool that
    did not would compare the wrong cards."""
    for cmd in (["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                ["rocm-smi", "--showmeminfo", "vram", "--csv"]):
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        rows = [[c.strip() for c in l.split(",")] for l in out.splitlines() if l.strip()]
        try:
            if cmd[0] == "nvidia-smi":                 # total MiB, a card a row
                sizes, visible = [int(r[0]) << 20 for r in rows], "CUDA_VISIBLE_DEVICES"
            else:                                      # a header, then a card a row
                col = next(i for i, h in enumerate(rows[0]) if "Total Memory" in h and "Used" not in h)
                sizes, visible = [int(r[col]) for r in rows[1:]], "HIP_VISIBLE_DEVICES"
        except (IndexError, ValueError, StopIteration):
            continue
        for name in (visible, "ROCR_VISIBLE_DEVICES") if visible != "CUDA_VISIBLE_DEVICES" else (visible,):
            said = os.environ.get(name)
            if said:
                try:
                    sizes = [sizes[int(k)] for k in said.split(",") if k.strip()]
                except (IndexError, ValueError):
                    return None
        return sizes or None
    return None


def _freed(then, now):
    """Whether a card holds FREED_GB less now than when a kept choice was made — another program, or another server,
    held it then — so the choice is made again for the card as it is."""
    return bool(then and now and len(then) == len(now) and any(a - b > FREED_GB * 2**30 for a, b in zip(then, now)))


def _ram_gb():
    """The machine's RAM as a boot sees it, steady from one boot to the next: the cgroup's limit where there is one,
    else the total (not what is free now, which moves with the system's cache)."""
    total = memory()[0]
    try:
        with open("/proc/self/cgroup") as f:
            path = next(l.strip().split(":", 2)[2] for l in f if l.startswith("0::"))
        with open(os.path.join("/sys/fs/cgroup", path.lstrip("/"), "memory.max")) as f:
            said = f.read().strip()
        if said != "max":
            total = min(total, int(said))
    except (OSError, StopIteration, ValueError):
        pass
    return total >> 30


def _release(engine, silicon):
    """The code a boot runs, as a digest of its files: the engine module (`engine`, its path), the family that drives the
    card — `silicon_families/` beside the module in a release, `build/silicon_families/` in the development tree — and the
    runtime's Python. ⚖ 2026-10-17 *"autotune should run every time if it is not too slow, or at least after every
    update"*: an update can change the memory a boot takes while the rules stay, so a choice is kept for one release only.
    `MEASURED` 2026-10-17 on the precision (measurements/2026-10-17_v0110rc2_artifacts, server_tests.log): the 35B started
    in 54 and 66 s choosing against 34-40 s keeping."""
    lib = os.path.dirname(os.path.abspath(engine))
    family = "libsilvann_%s%s" % (silicon, LIBRARY_SUFFIX)
    files = ([engine] + [os.path.join(lib, d, family) for d in ("silicon_families", os.path.join("build", "silicon_families"))]
             + sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))))
    h = hashlib.md5()
    for f in files:
        if os.path.isfile(f):
            h.update(os.path.basename(f).encode())
            with open(f, "rb") as data:
                for block in iter(lambda: data.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:16]


def _key(where, cards, max_context, options, video_tokens=None, release=None):
    return json.dumps({"rules": RULES, "silicon": where, "cards": cards, "ram_gb": _ram_gb(), "release": release,
                       "max_context": max_context, "options": options, "video_tokens": video_tokens}, sort_keys=True)


def _remembered(folder, key):
    try:
        with open(os.path.join(folder.path, "autotune.json")) as f:
            got = json.load(f)
        if got.get("key") == key:
            return got
    except (OSError, ValueError):
        pass
    return None


def _remember(folder, key, options, max_context, why, used):
    try:
        with open(os.path.join(folder.path, "autotune.json"), "w") as f:
            json.dump({"key": key, "options": options, "max_context": max_context, "why": why, "card_used": used,
                       "chosen": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    except OSError:
        pass                          # a read-only folder chooses again at each boot


def choose(path, max_context, options=None, silicon=None, fresh=False, log=print, video_tokens=None, margin_mb=None):
    """(options, max_context, why): the config's `options`, completed for this machine, the context the card holds, and
    one line saying what was chosen and why. `max_context` a number of positions, or `"max"`: as many as the card holds
    beside the model, up to the model's own length — and a tier of experts only once the context is that whole length
    (⚖ 2026-10-14). `video_tokens`: the config's `max_tokens_per_video`, which the server's RAM holds while a video is
    read (▶ `vision.video_ram`). `fresh` tries again what the folder remembers. `margin_mb`: the config's `vram_margin_mb`,
    what each card keeps free (VRAM_MARGIN_MB where it says nothing)."""
    margin_mb = VRAM_MARGIN_MB if margin_mb is None else int(margin_mb)
    options = {k: v for k, v in (options or {}).items() if not (k in PLACEMENT and v == "auto")}
    # ⚖ 2026-10-14 *"it should be auto sized and just be a yes/no"*: `card_experts` "yes" (the default) has the card's
    #   tier of experts sized here, "no" has none; `card_experts_gb` pins a size instead
    if str(options.pop("card_experts", "yes")).lower() in ("no", "false", "0") and "card_experts_gb" not in options:
        options["card_experts_gb"] = 0
    folder = ModelFolder(path)
    from . import ARCHITECTURES
    arch = ARCHITECTURES.get(folder.architecture)
    if arch is None:
        raise Refused("no runtime for a %r model" % folder.architecture)
    takes = set(inspect.signature(arch.__init__).parameters)
    # a tier's size, or the CPU's integer arithmetic, named without `experts_on`: the experts on the CPU, where they are
    #   the only place they apply, and the rest chosen around them (a Qwen model's experts default to the card)
    if "experts_on" in takes and "experts_on" not in options and (
            float(options.get("card_experts_gb") or 0) or options.get("experts_int8")):
        options["experts_on"] = "cpu"
    most = max_context == "max"
    top = model_length(folder) if most else int(max_context)
    placement = [k for k in PLACEMENT if k in takes]
    named = placement and all(k in options for k in placement) or options.get("experts_on") == "card"
    if named and not most:
        return options, top, "the config's placement (%s), as it is" % ", ".join(
            "%s %s" % (k, options[k]) for k in PLACEMENT if k in options)
    machine = Machine(silicon=silicon)
    where = machine.silicon
    if where in NOT_A_CARD and not own_memory(machine):
        ctx = min(top, NO_CARD_CONTEXT) if most else top
        why = "%s, not a card of its own: the model's own defaults, the context %d" % (where, ctx)
        # an integrated GPU's pool is refused the pictures' room the model sets aside by default (`MEASURED` 2026-10-14
        #   on the precision: the 27B through OpenCL 1.2 boots with picture_pixels 0, and not with its 4 MP — since
        #   v0.7.0): pictures there are read on the CPU unless the config gives them room
        if where == "khronos_opencl1_igpu" and "picture_pixels" in takes and "picture_pixels" not in options:
            options = dict(options, picture_pixels=0)
            why += ", pictures read on the CPU (picture_pixels in the config's options gives them room in the pool)"
        return options, ctx, why
    names, used = _vendor_cards()
    key = _key(where, [c["card"] for c in machine.e.cards()] + [names], max_context, dict(options, vram_margin_mb=margin_mb),
               video_tokens, _release(machine.e.__file__, where))
    got = None if fresh else _remembered(folder, key)
    if got and _freed(got.get("card_used"), used):
        log("[autotune] the card holds less than when its setup was chosen (%s): chosen again" % got["chosen"])
        got = None
    if got:
        return got["options"], got["max_context"], "%s — as chosen %s, kept in autotune.json" % (got["why"], got["chosen"])
    p = _Prober(folder, arch, silicon, log, margin_mb)
    p.video_tokens = video_tokens
    if most:
        found = _setup(p, top, options, takes, grow=True)
        if found is None:
            raise Refused("%s does not fit this card even at a %d-token context: a larger card, or a smaller pack"
                          % (folder.path, MIN_CONTEXT))
        chosen, ctx, why = found
        why += "; the context %d, %s" % (ctx, "the model's whole length" if ctx == top else
                                         "as much as the card holds beside the model")
    else:
        ctx = top
        while True:
            found = _setup(p, ctx, options, takes)
            if found is not None:
                chosen, _, why = found
                if ctx < top:
                    why += "; the context %d, the largest of %d halved that boots" % (ctx, top)
                break
            if ctx <= MIN_CONTEXT:
                moe = bool(getattr(p.model(ctx, options), "MOE", False)) or "experts_on" not in takes and "card_experts_gb" in takes
                how = " with its experts on the CPU" if moe else ""
                raise Refused("%s does not fit this card even at a %d-token context%s: a larger card, or a smaller pack"
                              % (folder.path, ctx, how))
            ctx = max(MIN_CONTEXT, ctx // 2)
    why = "%s; %d MB of the card kept free (%d boots tried)" % (why, margin_mb, p.tried)
    _remember(folder, key, chosen, ctx, why, used)
    return chosen, ctx, why


def model_length(folder):
    """The longest conversation the model was made for: its configuration's `max_position_embeddings` — or less where its
    runtime holds less (its architecture's `longest_context`: DeepSeek's, to its candidate blocks)."""
    c = folder.config
    length = int((c.get("text_config") or c).get("max_position_embeddings") or NO_CARD_CONTEXT)
    from . import ARCHITECTURES
    longest = getattr(ARCHITECTURES.get(folder.architecture), "longest_context", None)
    held = longest(c) if longest else None
    return min(length, held) if held else length


def _most(p, options, top):
    """The largest context up to `top` the card boots with these options, the margin held: `top` itself where it boots;
    else the largest in steps of CTX_STEP that does — None where not even MIN_CONTEXT boots."""
    if p.boots(top, options):
        return top
    if not p.boots(MIN_CONTEXT, options):
        return None
    lo, hi = MIN_CONTEXT // CTX_STEP, -(-top // CTX_STEP)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p.boots(mid * CTX_STEP, options):
            lo = mid
        else:
            hi = mid
    return lo * CTX_STEP


def _least(p, ctx, options):
    """The least tier the model takes, a slot a layer, in steps of STEP_GB — where a tier at the context `ctx` would be
    lent the cache's rows a conversation has not reached (its `LEND`, NN-93); None where it would not."""
    model = p.model(ctx, dict(options, card_experts_gb=TIER_MAX_GB))
    if not getattr(model, "LEND", None):
        return None
    return -(-model.TIER_LEAST // int(STEP_GB * 2**30)) * STEP_GB


def _setup(p, ctx, options, takes, grow=False):
    """The fastest setup the card boots with, as (options, context, why); None when none does. `ctx` is the context —
    or, with `grow`, the most it may be: the context then takes what the card holds beside the model up to it, and a
    tier gets what is left only once the context is that whole length. What `options` names of the placement is kept,
    and the rest chosen around it."""
    def sized(opts):
        """The context these options boot with: `ctx`, or with `grow` the most the card holds up to it; None if none."""
        if grow:
            return _most(p, opts, ctx)
        return ctx if p.boots(ctx, opts) else None

    # DeepSeek's experts: on the CPU, the first layers' in RAM and the rest mapped from their files — as many layers in
    #   RAM as fit beside the server's own memory, the spare and FILES_SHARE (`experts_file.in_ram`'s rule, a layer at a
    #   time, less the share the files it reads in place need)
    if "experts_in_ram" in takes and "experts_from" not in options:
        whole = p.model(MIN_CONTEXT, dict(options, experts_from="ram"))
        layers, held = len(whole.run_layers), experts_file.limit()
        a_layer = whole.cpu_arena // layers
        room = held * (1 - FILES_SHARE) - experts_file.OWN - experts_file.SPARE - video_ram(p.folder.config, p.video_tokens)
        n = int(max(0, min(layers, room // a_layer)))
        placed = (dict(options, experts_from="ram") if n == layers else
                  dict(options, experts_from="mapped", experts_in_ram=n))
        why = ("the experts on the CPU, in RAM" if n == layers else
               "the experts on the CPU, %d of %d layers' in RAM and the rest mapped from their files" % (n, layers))
        got = sized(placed)
        return None if got is None else (placed, got, why + " (%.1f GB a layer, %.0f GB this process may hold)"
                                         % (a_layer / 2**30, held / 2**30))
    first = p.model(MIN_CONTEXT if grow else ctx, options)
    moe = bool(getattr(first, "MOE", False)) or "experts_on" not in takes and "card_experts_gb" in takes
    if not moe:
        got = sized(options)
        return None if got is None else (options, got, "on the card")
    if options.get("experts_on") == "card":
        got = sized(options)
        return None if got is None else (options, got, "the experts on the card, as the config says")
    if "experts_on" in takes and "experts_on" not in options:
        if p.boots(ctx, dict(options, experts_on="card")):
            return dict(options, experts_on="card"), ctx, "the experts on the card"
        # ⚖ 2026-10-17 (NN-93): growing, the experts on the card beat a longer context — they stay there at the longest
        #   context the card holds beside them, if that is CARD_CONTEXT or more, before the CPU takes them for the whole
        #   length (a 24 GB card keeps the 35B whole on it rather than reading 262,144 positions with a tier)
        if grow:
            got = _most(p, dict(options, experts_on="card"), ctx)
            if got is not None and got >= CARD_CONTEXT:
                return dict(options, experts_on="card"), got, "the experts on the card"
    cpu = dict(options, experts_on="cpu") if "experts_on" in takes else dict(options)
    need = p.model(MIN_CONTEXT, dict(cpu, experts_from="ram")).cpu_arena
    held = experts_file.limit()
    gb = "%.0f GB of experts, %.0f GB this process may hold" % (need / 2**30, held / 2**30)
    if "experts_from" in options:
        source, gb = options["experts_from"], gb + "; where from, as the config says"
    else:
        # where the experts live: RAM when they fit in what this process may hold, beside its own memory and the spare
        source = "ram" if experts_file.in_ram(need, held, video_ram(p.folder.config, p.video_tokens)) else "disk"
    cpu["experts_from"] = source
    if source != "ram" or "card_experts_gb" not in takes or "card_experts_gb" in options:
        why = "the experts on the CPU, from the %s (%s)" % ("RAM" if source == "ram" else "disk", gb)
        if "card_experts_gb" in options:
            why += (", %s GB of the card a tier of them, as the config says" % options["card_experts_gb"]
                    if float(options["card_experts_gb"]) else ", no tier on the card, as the config says")
        # GLM's prompt from the disk: more rows a step read each expert for more of them (its chunk reaches past 256)
        if source == "disk" and "experts_on" not in takes and "chunk" in takes and "chunk" not in options:
            got = sized(dict(cpu, chunk=DISK_CHUNK))
            if got is not None:
                return dict(cpu, chunk=DISK_CHUNK), got, why + ", the prompt %d positions a step" % DISK_CHUNK
        got = sized(cpu)
        return None if got is None else (cpu, got, why)
    got = sized(cpu)
    if got is None:
        return None
    if got < ctx:
        # the whole length does not fit beside the model: a tier lent the cache's unreached rows keeps the least it takes
        #   and the context the rest, so a short conversation has the card's free memory for its experts and a long one
        #   its cache; a tier not lent any leaves the card to the context
        least = _least(p, ctx, cpu)
        most = _most(p, dict(cpu, card_experts_gb=least), ctx) if least else None
        if most is not None:
            return (dict(cpu, card_experts_gb=least), most, "the experts on the CPU from RAM (%s), a tier of %g GB of its "
                    "own lent the cache's rows a conversation has not reached" % (gb, least))
        return cpu, got, "the experts on the CPU from RAM (%s), no tier: the context has the card" % gb
    # from RAM, the context whole: the largest tier the card boots with beside it, the margin held, in steps of STEP_GB
    lo, hi = 0, int(TIER_MAX_GB / STEP_GB)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p.boots(ctx, dict(cpu, card_experts_gb=mid * STEP_GB)):
            lo = mid
        else:
            hi = mid
    tier = lo * STEP_GB
    if not tier:
        return cpu, ctx, "the experts on the CPU from RAM (%s), no room on the card for a tier" % gb
    cpu["card_experts_gb"] = tier
    why = "the experts on the CPU from RAM (%s), %g GB of the card a tier of them" % (gb, tier)
    # ⚖ 2026-10-14 *"if the l1 is less than 1/4 of the l2 it stays inclusive, and if the full duplex test fails it also
    #   stays inclusive"*: an exclusive tier (each expert on the card or in RAM, never both — GLM's) gives back the tier's
    #   RAM for a swap each promotion, worth it only where the tier is a large share and the card copies both ways at once
    if "card_exclusive" in takes and "card_exclusive" not in options:
        if tier * 2**30 < need / 4:
            why += "; inclusive, the tier under a quarter of the experts"
        else:
            line = int(options.get("card_duplex_line", inspect.signature(p.arch.__init__).parameters["card_duplex_line"].default))
            ratio = p.duplex(ctx, dict(cpu, card_exclusive=True))
            if ratio is not None and ratio < line:
                cpu["card_exclusive"] = True
                why += "; exclusive, the card's copies both ways at once (%.2f)" % (ratio / 1000)
            else:
                why += "; inclusive, the card's copies not both ways at once (%s)" % (
                    "%.2f" % (ratio / 1000) if ratio is not None else "the exclusive tier does not boot")
    return cpu, ctx, why
