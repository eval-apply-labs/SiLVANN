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
  ② else on the CPU: from RAM, the largest tier the card boots with beside the context, in steps of STEP_GB, less
     MARGIN_GB; from the disk, no tier (a tier copies its experts from RAM), and GLM's prompt DISK_CHUNK positions at a
     time — `MEASURED` 2026-10-14, held to 28 GB on the RTX 5060 Ti: 10.6 positions/s against 8.3 at 256, the answer
     the same
  ③ the context halved, down to MIN_CONTEXT, while nothing boots
  and GLM's tier exclusive (each expert on the card or in RAM) where it is a quarter of the experts or more and the card
  copies both ways at once — else inclusive (⚖ 2026-10-14)

The context comes first: the conversation's cache for the whole context is the card's before any tier (a tier a little
smaller costs a little speed; a cache too small refuses a conversation). A config's `max_context` of `"max"` takes that
further (⚖ 2026-10-14): the context grows to what the card holds beside the model, up to the model's own length
(`model_length`), the experts on the card only where they boot with that whole length, and a tier only once the
context is whole.

The choice is kept in the model folder (`autotune.json`) under what it was made for — the card (its family, and its name
and memory where its vendor's tool says them), the machine's RAM (its cgroup's limit where there is one), the context and the config's options —
and the next boot on the same takes it without trying; a boot the card refuses all the same chooses again (`fresh`).

A config's options win: what it names of the placement (`experts_on`, `experts_from`, `card_experts_gb`) is kept and the
rest chosen around it — `"auto"` names nothing, so `{"experts_on": "cpu", "experts_from": "auto"}` chooses where from and
the tier — and its other options are kept in every candidate; a placement it names in full is booted as it is. A machine with no card is not tried — its "card" is the CPU's memory — nor an
integrated GPU (`khronos_opencl1_igpu`), whose memory is the PC's RAM.
"""
import collections
import datetime
import inspect
import json
import os
import shutil
import subprocess

from . import experts_file
from .machine import Machine
from .model_folder import ModelFolder, Refused
from .vision import video_ram

PLACEMENT = ("experts_on", "experts_from", "card_experts_gb")
STEP_GB = 0.25           # ⚖ 2026-10-14 "128/256mb is good": the tier's step
MARGIN_GB = 0.25         # what the card keeps free — ⚖ 2026-10-14 "we already have the video card occupation when we
                         # load, leave 200mb extra there and it will be fine (one 256mb step)": a trial boots beside
                         # whatever else holds the card then
CTX_STEP = 1024          # the context's step, where it takes what the card holds (`"max"`)
MIN_CONTEXT = 4096
NO_CARD_CONTEXT = 32768  # `"max"` on a machine with no card of its own, and a model that names no length
NOT_A_CARD = ("x86_avx2", "host", "khronos_opencl1_igpu")   # silicon whose pool is the PC's RAM: never tried
TIER_MAX_GB = 128        # the binary search's ceiling, past any card's memory
DISK_CHUNK = 1024        # GLM's prompt rows a step with its experts on the disk (Qwen's chunk is at most 256)
RULES = 4                # the rules' version, in what a kept choice was made for: a change to them chooses again


def refused_by_card(ex):
    """Whether a boot's error is the card's refusal of the pool's room."""
    return _refused_by_card(ex)


def _refused_by_card(ex):
    return "would not allocate the pool" in str(ex) or "allocator refused the pool" in str(ex)


class _Prober:
    def __init__(self, folder, arch, silicon, log):
        self.folder, self.arch, self.silicon, self.log = folder, arch, silicon, log
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
        params = model.device_params()
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
    """The cards' names and memory as their vendor's tool says them, or "" where there is none."""
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"]):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
                return " | ".join(l.strip() for l in out.splitlines() if l.strip())
            except (OSError, subprocess.SubprocessError):
                pass
    return ""


def _ram_gb():
    """The machine's RAM as a boot sees it, steady from one boot to the next: the cgroup's limit where there is one,
    else the total (not what is free now, which moves with the system's cache)."""
    total = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
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


def _key(where, cards, max_context, options, video_tokens=None):
    return json.dumps({"rules": RULES, "silicon": where, "cards": cards, "ram_gb": _ram_gb(),
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


def _remember(folder, key, options, max_context, why):
    try:
        with open(os.path.join(folder.path, "autotune.json"), "w") as f:
            json.dump({"key": key, "options": options, "max_context": max_context, "why": why,
                       "chosen": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    except OSError:
        pass                          # a read-only folder chooses again at each boot


def choose(path, max_context, options=None, silicon=None, fresh=False, log=print, video_tokens=None):
    """(options, max_context, why): the config's `options`, completed for this machine, the context the card holds, and
    one line saying what was chosen and why. `max_context` a number of positions, or `"max"`: as many as the card holds
    beside the model, up to the model's own length — and a tier of experts only once the context is that whole length
    (⚖ 2026-10-14). `video_tokens`: the config's `max_tokens_per_video`, which the server's RAM holds while a video is
    read (▶ `vision.video_ram`). `fresh` tries again what the folder remembers."""
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
    most = max_context == "max"
    top = model_length(folder) if most else int(max_context)
    placement = [k for k in PLACEMENT if k in takes]
    named = placement and all(k in options for k in placement) or options.get("experts_on") == "card"
    if named and not most:
        return options, top, "the config's placement (%s), as it is" % ", ".join(
            "%s %s" % (k, options[k]) for k in PLACEMENT if k in options)
    machine = Machine(silicon=silicon)
    where = machine.silicon
    if where in NOT_A_CARD:
        ctx = min(top, NO_CARD_CONTEXT) if most else top
        return options, ctx, "%s, not a card of its own: the model's own defaults, the context %d" % (where, ctx)
    key = _key(where, [c["card"] for c in machine.e.cards()] + [_vendor_cards()], max_context, options, video_tokens)
    got = None if fresh else _remembered(folder, key)
    if got:
        return got["options"], got["max_context"], "%s — as chosen %s, kept in autotune.json" % (got["why"], got["chosen"])
    p = _Prober(folder, arch, silicon, log)
    p.video_tokens = video_tokens
    if most:
        found = _setup(p, top, options, takes, grow=True)
        if found is None:
            raise Refused("%s does not fit this card even at a %d-token context: a larger card, or a smaller pack"
                          % (folder.path, MIN_CONTEXT))
        chosen, ctx, why = found
        why += "; the context %d, %s" % (ctx, "the model's whole length" if ctx == top else
                                         "as much as the card holds beside the model, %g GB kept free" % MARGIN_GB)
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
    why = "%s (%d boots tried)" % (why, p.tried)
    _remember(folder, key, chosen, ctx, why)
    return chosen, ctx, why


def model_length(folder):
    """The longest conversation the model was made for: its configuration's `max_position_embeddings`."""
    c = folder.config
    return int((c.get("text_config") or c).get("max_position_embeddings") or NO_CARD_CONTEXT)


def _most(p, options, top):
    """The largest context up to `top` the card boots with these options: `top` itself where it boots; else the largest
    in steps of CTX_STEP that does, less MARGIN_GB's worth of positions — None where not even MIN_CONTEXT boots."""
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
    ctx = lo * CTX_STEP
    spare = int(MARGIN_GB * 2**30 / max(1.0, p.position_bytes(ctx, options)))
    return max(MIN_CONTEXT, (ctx - spare) // CTX_STEP * CTX_STEP)


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

    first = p.model(MIN_CONTEXT if grow else ctx, options)
    moe = bool(getattr(first, "MOE", False)) or "experts_on" not in takes and "card_experts_gb" in takes
    if not moe:
        got = sized(options)
        return None if got is None else (options, got, "on the card")
    if options.get("experts_on") == "card":
        got = sized(options)
        return None if got is None else (options, got, "the experts on the card, as the config says")
    if "experts_on" in takes and "experts_on" not in options and p.boots(ctx, dict(options, experts_on="card")):
        return dict(options, experts_on="card"), ctx, "the experts on the card"
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
        return cpu, got, "the experts on the CPU from RAM (%s), no tier: the context has the card" % gb
    # from RAM, the context whole: the largest tier the card boots with beside it, in steps of STEP_GB, less MARGIN_GB
    lo, hi = 0, int(TIER_MAX_GB / STEP_GB)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p.boots(ctx, dict(cpu, card_experts_gb=mid * STEP_GB)):
            lo = mid
        else:
            hi = mid
    tier = max(0, lo - round(MARGIN_GB / STEP_GB)) * STEP_GB
    if not tier:
        return cpu, ctx, "the experts on the CPU from RAM (%s), no room on the card for a tier" % gb
    cpu["card_experts_gb"] = tier
    why = "the experts on the CPU from RAM (%s), %g GB of the card a tier of them (%g booted, %g kept free)" % (
        gb, tier, lo * STEP_GB, MARGIN_GB)
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
