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

MODELS = ROOT / "models"
DEFAULT_CONFIG = {"max_context": 32768, "lora": None, "thinking": True, "sampler": "generation", "options": {}}

# how many more positions of a conversation 100 MB of the card holds, past the first 4,352 (kept at 16 bits);
# measured from the cache each model sizes at two contexts
TOKENS_PER_100MB = {
    "Qwen3.6-35B-A3B_silvann_tq_D8E4": 10000,
    "Qwen3.8-27B_silvann_tq_D4": 3200,
    "GLM-5.3-Flash_silvann_tq_D4E4": 8800,
}


def default_config(folder):
    """`configs/default.json` in a model folder, written when it has none."""
    cfg = Path(folder) / "configs" / "default.json"
    if not cfg.exists():
        config = dict(DEFAULT_CONFIG)
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
        folder = ensure_model(str(MODELS / argv[1]), yes=True)
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
