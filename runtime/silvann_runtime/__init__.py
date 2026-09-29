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
    params = model.device_params()
    if isinstance(params, dict):          # a section a worker: the card and the CPU's
        machine.boot(params[0], workers=model.workers(), worker_params=params)
    else:
        machine.boot(params)
    model.load()
    # ⭐ THE MODEL AS THE PROGRAMS IT RUNS, beside it: what the runtime wrote for this machine and these settings
    try:
        machine.write_programs(os.path.join(folder.path, "programs.lisp"),
                               "%s (%s) — the procedures silvann_runtime/%s wrote at this boot\n"
                               ";; for max_context %d, options %s — other settings write other tables and procedures"
                               % (os.path.basename(folder.path.rstrip(os.sep)), folder.architecture,
                                  arch.__module__.split(".")[-1] + ".py", max_context, json.dumps(options, sort_keys=True)))
    except OSError:
        pass                       # a read-only folder keeps its model; the programs are a view, not a need
    return model
