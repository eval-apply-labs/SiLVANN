"""The runtime on the Eval-Apply Generalized Interpreter: a model folder booted onto a machine, and the conversations it
serves.

  model_folder   a packed bundle, the model's own files, its LoRAs, and the identity a conversation records
  machine        the engine, booted once, and the conveniences every model needs
  qwen3_5        the Qwen 3.5 family, dense or with routed experts — its tables, its procedures, a token's program
  glm5           GLM 5.3 Flash: the fixed matrices on the card, the routed experts on the CPU
  session        conversations as cartridges, a turn at a time
  chat           a turn's tokens, framed as the model's chat template frames them
  fetch          a model folder that is not there, asked for and downloaded
  composer       text in, a program in the engine's heap out
"""
import glob
import hashlib
import inspect
import json
import os

from .model_folder import ModelFolder, Refused, architecture_of
from .machine import Machine, EngineError, Panic
from .qwen3_5 import Qwen35
from .glm5 import Glm5
from .mistral3 import Ministral3
from .deepseek4 import DeepSeek4
from .qwen4_exp import Qwen4Exp

ARCHITECTURES = {"qwen3_5_text": Qwen35, "qwen3_5_moe_text": Qwen35, "qwen3_vl_text": Qwen35, "glm5_next_text": Glm5, "ministral3": Ministral3,
                 "deepseek_v41_text": DeepSeek4, "qwen4_exp_text": Qwen4Exp}


def _yes_no(value, name):
    """A config's switch: "yes" or "no", as the boot's other switches are written, or true or false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("yes", "no"):
        return value.lower() == "yes"
    raise Refused("%s is \"yes\" or \"no\", not %r" % (name, value))


def _routed(path):
    """Whether a model folder's model has routed experts, from its config.json alone."""
    whole = json.load(open(os.path.join(os.path.abspath(path), "config.json")))
    c = whole.get("text_config", whole)
    return any(int(c.get(k) or 0) > 0 for k in ("num_experts", "n_routed_experts", "num_local_experts"))


def with_cards(path, num_cards, options, multi_user=None):
    """A config's `num_cards` and `multi_user` as its family's options. ⭐ `multi_user` says what several cards are
    (⚖ 2026-11-04, seats.md): "no" (the default) one conversation's model split over the machine's first `num_cards`
    cards — `tensor_parallel` for the Qwen models and GLM, the `cards` list for DeepSeek, every layer over the group;
    "yes" a conversation a card, for a family that seats one on each (`CARDS_ARE_SEATS`: Flash-Next's `cards`), the
    session answering them at once. With "yes" the routed experts are the inclusive set (⚖ 2026-11-04): every expert in
    RAM where the cards are seats, and each card's tier a copy of some of them — never an exclusive tier.
    ⚖ 2026-11-04 *"only consider the one gpu pc with two conversation as the limit case, otherwise everyone runs with one
    conversation per card"*: on ONE card "yes" seats two conversations there (`seats` 2, the card on one's dense part
    while the CPU computes the other's experts), with no tier — two seats share the card and the tier is one seat's — the
    limit case, worth it as far as the card's dense part takes as long as the CPU's experts; a family that seats none
    keeps one, its tier inclusive. Over several cards, a conversation a card: each card keeps a tier of its own for its
    conversation where the family builds one (`TIER_A_CARD`, ⚖ 2026-11-02 *"one conversation per card cleanly, jack up the
    l1 cache"*) — the config's `card_experts_gb`, or the autotune's — and none where it does not yet. More cards refused for a family that runs on one or seats none, or where `options`
    already names another number. `options`' own `tensor_parallel` is still read where `num_cards` is left out (the configs written before
    it). `num_cards` and `multi_user` are the config's own keys, beside `options`: inside them refused, with where they
    go."""
    options = dict(options)
    for key in ("num_cards", "multi_user"):
        if key in options:
            raise Refused("%s is a key of the config, beside options — not one of them: move \"%s\": %s out of options"
                          % (key, key, json.dumps(options[key])))
    many = _yes_no("no" if multi_user is None else multi_user, "multi_user")
    num_cards = 1 if num_cards is None else num_cards
    n = int(num_cards)
    if n < 1:
        raise Refused("num_cards is 1 or more, not %r" % (num_cards,))
    if n == 1 and not many:
        return options
    arch = ARCHITECTURES.get(architecture_of(path))
    takes = inspect.signature(arch.__init__).parameters if arch is not None else {}
    seats = bool(getattr(arch, "CARDS_ARE_SEATS", False))
    # a family that seats two conversations on one card — Flash-Next's, and a family that says so (`SEATS_ON_A_CARD`, once
    #   its seats have answered through the server) — for a model with routed experts: a dense model's card and CPU have
    #   no two parts of a token to overlap (NN-155)
    two = (seats or bool(getattr(arch, "SEATS_ON_A_CARD", False))) and _routed(path)
    name = os.path.basename(path.rstrip(os.sep))
    if many:
        if options.get("card_exclusive"):
            raise Refused("multi_user \"yes\" keeps the experts inclusive, each card's tier a copy of what RAM holds: no "
                          "card_exclusive")
        if "card_exclusive" in takes:
            options["card_exclusive"] = False
        if n == 1 and not two:
            return options
        if options.get("experts_from", "auto") not in ("auto", "ram"):
            raise Refused("several conversations take every expert in RAM: experts_from \"ram\", not %r"
                          % (options["experts_from"],))
        if n == 1:
            if float(options.get("card_experts_gb") or 0):
                raise Refused("two conversations on one card share it with no tier of experts, the tier being one seat's: "
                              "no card_experts_gb with multi_user \"yes\" on one card")
            if int(options.get("seats", 2)) != 2:
                raise Refused("multi_user \"yes\" on one card seats two conversations there, not seats %s"
                              % options["seats"])
            options["experts_from"], options["seats"], options["card_experts_gb"] = "ram", 2, 0
            if "experts_on" in takes:                   # a seat's experts the CPU's, the card on the other's dense part
                options["experts_on"] = "cpu"
            return options
        apart = bool(getattr(arch, "CONVERSATION_A_CARD", False)) and _routed(path)   # a card each
        tiered = apart and bool(getattr(arch, "TIER_A_CARD", False))                 # ...and a tier on each (NN-166)
        if not (seats or apart):
            raise Refused("%s holds one conversation over its cards: num_cards 1 with multi_user \"yes\", or multi_user "
                          "\"no\" to split it over %d" % (name, n))
        if apart and not tiered and float(options.get("card_experts_gb") or 0):
            raise Refused("a conversation a card takes no tier of experts on its card yet: no card_experts_gb with "
                          "multi_user \"yes\" over %d cards" % n)
        # the option a conversation a card is: `cards`, or the family's own where `cards` is its group's (DeepSeek's)
        key = getattr(arch, "CONVERSATION_A_CARD_OPTION", "cards") if apart else "cards"
        if key in options and list(options[key]) != list(range(n)):
            raise Refused("num_cards %d and options.%s %s say two things: num_cards alone" % (n, key, options[key]))
        options["experts_from"], options[key] = "ram", list(range(n))
        if apart:                                       # the experts the CPU's, each card's tier its own or none
            if not tiered:
                options["card_experts_gb"] = 0
            if "experts_on" in takes:
                options["experts_on"] = "cpu"
        return options
    if seats:
        raise Refused("%s splits nothing over its cards, it seats a conversation on each: multi_user \"yes\", or num_cards 1"
                      % name)
    if "tensor_parallel" in takes:
        if int(options.get("tensor_parallel", n)) != n:
            raise Refused("num_cards %d and options.tensor_parallel %s say two numbers: num_cards alone" % (n, options["tensor_parallel"]))
        options["tensor_parallel"] = n
    elif "cards" in takes:
        options["cards"] = list(range(n))
    else:
        raise Refused("%s runs on one card: num_cards 1" % name)
    return options


def open_model(path, lora=None, max_context=4096, silicon=None, progress=None, **options):
    """A model folder booted onto a card, loaded and ready for a conversation; `progress(done, total, what)` called as its
    load goes, a layer at a time."""
    folder = ModelFolder(path, lora=lora)
    arch = ARCHITECTURES.get(folder.architecture)
    if arch is None:
        raise Refused("no runtime for a %r model" % folder.architecture)
    machine = Machine(silicon=silicon)
    model = arch(folder, machine, max_context=max_context, **options)
    # ⭐ THE PROGRAMS ARE READ, NOT COMPOSED, when `programs.lisp` was written for this key
    path = os.path.join(folder.path, "programs.lisp")
    key = programs_key(folder, folder.architecture, max_context, options, machine.silicon)
    machine.programs_file = (path, key)
    params = model.device_params()
    if isinstance(params, dict):          # a section a worker: the card and the CPU's
        machine.boot(params[0], workers=model.workers(), worker_params=params, bindings=getattr(model, "BINDINGS", 512))
    else:
        machine.boot(params)
    model.load(progress)
    # ⭐ THE MODEL AS THE PROGRAMS IT RUNS, beside it — written when they were composed, for the boots that follow
    if machine.read_from is None:
        try:
            machine.write_programs(path,
                                   "%s (%s) — the procedures silvann_runtime/%s composed\n"
                                   ";; for max_context %d, options %s, LoRA %s — other settings compose other procedures,\n"
                                   ";; and a boot with these reads them from here rather than composing them again"
                                   % (os.path.basename(folder.path.rstrip(os.sep)), folder.architecture,
                                      arch.__module__.split(".")[-1] + ".py", max_context,
                                      json.dumps(options, sort_keys=True), folder.lora),
                                   key)
        except OSError:
            pass                   # a read-only folder keeps its model; the programs are composed at each boot
    return model


def programs_key(folder, architecture, max_context, options, silicon):
    """What a model's programs are composed from, as one digest: the pack, the settings, the silicon family, the model's
    Lisp files, and the source of the code that binds what they read — this package and the bundle loader whose type ids
    and plane offsets they carry. ⛳ The silicon is in it since a composer reads it: `embed_on` "auto" puts the embedding
    on a CPU worker through the OpenCL 1.2 families alone, and a dense model's programs then hand it work — `MEASURED`
    2026-10-26, a CPU-alone boot of the 27B read a file an OpenCL 1.2 boot had written and refused at its worker 1."""
    from .model_folder import NL
    h = hashlib.sha256()
    with open(os.path.join(folder.path, "pack.json"), "rb") as f:
        h.update(f.read())
    h.update(json.dumps([architecture, max_context, folder.lora, options, silicon], sort_keys=True, default=str).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for src in sorted(glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(folder.lisp, "*.lisp"))) + [NL.__file__]:
        with open(src, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:32]
