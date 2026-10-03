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
import json
import os

from .model_folder import ModelFolder, Refused
from .machine import Machine, EngineError
from .qwen3_5 import Qwen35
from .glm5 import Glm5

ARCHITECTURES = {"qwen3_5_text": Qwen35, "qwen3_5_moe_text": Qwen35, "glm5_next_text": Glm5}


def open_model(path, lora=None, max_context=4096, silicon=None, **options):
    """A model folder booted onto a card, loaded and ready for a conversation."""
    folder = ModelFolder(path, lora=lora)
    arch = ARCHITECTURES.get(folder.architecture)
    if arch is None:
        raise Refused("no runtime for a %r model" % folder.architecture)
    machine = Machine(silicon=silicon)
    model = arch(folder, machine, max_context=max_context, **options)
    # ⭐ THE PROGRAMS ARE READ, NOT COMPOSED, when `programs.lisp` was written for this key
    path = os.path.join(folder.path, "programs.lisp")
    key = programs_key(folder, folder.architecture, max_context, options)
    machine.programs_file = (path, key)
    params = model.device_params()
    if isinstance(params, dict):          # a section a worker: the card and the CPU's
        machine.boot(params[0], workers=model.workers(), worker_params=params)
    else:
        machine.boot(params)
    model.load()
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


def programs_key(folder, architecture, max_context, options):
    """What a model's programs are composed from, as one digest: the pack, the settings, and the source of the code that
    composes them — this package and the bundle loader whose type ids and plane offsets they carry. ⛳ The silicon is not
    in it: no composer reads it (`REASONED` from qwen3_5.py and glm5.py, where it only names the card's worker)."""
    from .model_folder import NL
    h = hashlib.sha256()
    with open(os.path.join(folder.path, "pack.json"), "rb") as f:
        h.update(f.read())
    h.update(json.dumps([architecture, max_context, folder.lora, options], sort_keys=True, default=str).encode())
    for src in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))) + [NL.__file__]:
        with open(src, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:32]
