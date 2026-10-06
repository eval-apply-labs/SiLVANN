"""SiLVANN's command line for what is not the server.

    python3 silvann_core.py download-model <name>     a model into models/<name>/, from Hugging Face
    python3 silvann_core.py list-models               the models this release knows how to fetch, and which are here

A model is fetched only when it is asked for, so the release carries no weights; the server refuses a model that is not
here and names this command. A fetched folder gets `configs/default.json` — the runtime settings the server boots with
when no other config is named — unless it has one.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# the runtime beside this file in a release, or the development tree's
_HERE = ROOT / "runtime" if (ROOT / "runtime").is_dir() else ROOT.parent / "backstage" / "python"
sys.path.insert(0, os.environ.get("SILVANN_RUNTIME", str(_HERE)))
from silvann_runtime.fetch import KNOWN, ensure_model, present     # noqa: E402
from silvann_runtime.model_folder import Refused                   # noqa: E402

MODELS = ROOT / "models"
DEFAULT_CONFIG = {"max_context": 32768, "lora": None, "thinking": True, "sampler": "generation", "options": {}}

# how many more positions of a conversation 100 MB of the card holds, past the first 4,352 (kept at 16 bits);
# measured from the cache each model sizes at two contexts
TOKENS_PER_100MB = {
    "Qwen3.6-35B-A3B_silvann_tq_D8E4": 10000,
    "Qwen3.8-27B_silvann_tq_D4": 3200,
    "GLM-5.3-Flash_silvann_tq_D4E4": 8800,
    "GLM-5.3-Flash_silvann_tq_D8E4": 8800,         # the same cache: a pack's widths are its weights', not its cache's
    "Ministral-3-14B_silvann_tq_D4": 625,          # every position in halves: 40 layers' keys and values
    "Ministral-3-14B_silvann_tq_D8": 625,
}

# a model's context unless its config says otherwise, where the release's default does not suit it: Mistral 3's cache
# is every position in halves, so on a 16 GB card its D4 leaves room for 32k positions and its D8 for 8k
MAX_CONTEXT = {
    "Ministral-3-14B_silvann_tq_D4": 32768,
    "Ministral-3-14B_silvann_tq_D8": 8192,
}


# the runtime options a model boots with unless its config says otherwise: the 122B's experts (54 GB) and the 397B's do not fit a card
# it is meant for, so they start on the CPU, in RAM or read from the disk by the memory there is
OPTIONS = {
    "Qwen3.5-122B-A10B_silvann_tq_D8E4": {"experts_on": "cpu", "experts_from": "auto"},
    "Qwen3.5-122B-A10B_silvann_tq_D4E4": {"experts_on": "cpu", "experts_from": "auto"},
    "Qwen3.5-397B-A17B_silvann_tq_D4E4": {"experts_on": "cpu", "experts_from": "auto"},
}


def default_config(folder):
    """`configs/default.json` in a model folder, written when it has none."""
    cfg = Path(folder) / "configs" / "default.json"
    if not cfg.exists():
        config = dict(DEFAULT_CONFIG, options=dict(OPTIONS.get(Path(folder).name, {})))
        if Path(folder).name in MAX_CONTEXT:
            config["max_context"] = MAX_CONTEXT[Path(folder).name]
        per = TOKENS_PER_100MB.get(Path(folder).name)
        if per:
            config = {"_note": "max_context is the longest conversation, and its cache is on the card from boot: "
                               "every 100 MB of the card's memory holds about %d positions of it. Lower it if "
                               "the model does not fit, raise it if there is room." % per, **config}
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(config, indent=2) + "\n")
    return cfg


def main(argv):
    if len(argv) >= 2 and argv[0] == "download-model":
        try:
            folder = ensure_model(str(MODELS / argv[1]), yes=True)
        except Refused as ex:
            raise SystemExit(str(ex))
        print("%s is in %s — configs: %s" % (argv[1], folder, default_config(folder)))
        return 0
    if argv[:1] == ["list-models"]:
        for name, (repo, size) in sorted(KNOWN.items()):
            here = present(str(MODELS / name))
            print("%-36s %-8s %s  %s" % (name, size, "here" if here else "     ", "huggingface.co/" + repo))
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
