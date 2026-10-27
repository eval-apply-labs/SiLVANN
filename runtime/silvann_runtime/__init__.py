"""The runtime on the Eval-Apply GPU Interpreter: a model folder booted onto a machine, and the conversations it serves.

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

from .model_folder import ModelFolder, Refused
from .machine import Machine, EngineError
from .qwen3_5 import Qwen35
from .glm5 import Glm5
from .mistral3 import Ministral3
from .deepseek4 import DeepSeek4

ARCHITECTURES = {"qwen3_5_text": Qwen35, "qwen3_5_moe_text": Qwen35, "qwen3_vl_text": Qwen35, "glm5_next_text": Glm5, "ministral3": Ministral3,
                 "deepseek_v41_text": DeepSeek4}


def with_cards(path, num_cards, options):
    """A config's `num_cards` as its family's option: `tensor_parallel` for the Qwen models and GLM, the `cards` list for
    DeepSeek — every layer split over the machine's first `num_cards` cards. One card adds nothing (every family's
    default); more refused for a family that runs on one, or where `options` already names another number. `options`'
    own `tensor_parallel` is still read where `num_cards` is left out (the configs written before it). `num_cards` is
    the config's own key, beside `options`: inside them it is refused, with where it goes."""
    options = dict(options)
    if "num_cards" in options:
        raise Refused("num_cards is a key of the config, beside options — not one of them: move \"num_cards\": %s out of "
                      "options" % (options["num_cards"],))
    num_cards = 1 if num_cards is None else num_cards
    n = int(num_cards)
    if n < 1:
        raise Refused("num_cards is 1 or more, not %r" % (num_cards,))
    if n == 1:
        return options
    arch = ARCHITECTURES.get(ModelFolder(path).architecture)
    takes = inspect.signature(arch.__init__).parameters if arch is not None else {}
    if "tensor_parallel" in takes:
        if int(options.get("tensor_parallel", n)) != n:
            raise Refused("num_cards %d and options.tensor_parallel %s say two numbers: num_cards alone" % (n, options["tensor_parallel"]))
        options["tensor_parallel"] = n
    elif "cards" in takes:
        options["cards"] = list(range(n))
    else:
        raise Refused("%s runs on one card: num_cards 1" % os.path.basename(path.rstrip(os.sep)))
    return options


def open_model(path, lora=None, max_context=4096, silicon=None, progress=None, **options):
    """A model folder booted onto a card, loaded and ready for a conversation; `progress(done, total)` called as its
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
    on a CPU worker through khronos_opencl1_igpu alone, and a dense model's programs then hand it work — `MEASURED`
    2026-10-26, a CPU-alone boot of the 27B read a file an OpenCL 1.2 boot had written and refused at its worker 1."""
    from .model_folder import NL
    h = hashlib.sha256()
    with open(os.path.join(folder.path, "pack.json"), "rb") as f:
        h.update(f.read())
    h.update(json.dumps([architecture, max_context, folder.lora, options, silicon], sort_keys=True, default=str).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for src in sorted(glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(folder.path, "lisp", "*.lisp"))) + [NL.__file__]:
        with open(src, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:32]
