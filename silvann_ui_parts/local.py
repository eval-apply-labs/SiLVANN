"""This machine: where the UI's files, the data folder, the models and their configs are."""
import json
import os
from pathlib import Path
from typing import Optional


# ---------- This machine's folders, and a model's configs ----------

REPO_ROOT = Path(__file__).resolve().parent.parent   # main/ — the runtime root, the folder of silvann_ui.py
SERVER_SCRIPT = REPO_ROOT / "start_silvann_server.py"
CORE_SCRIPT = REPO_ROOT / "silvann_core.py"          # lists and downloads the models this release knows
# what the machine keeps — the models and the conversations: beside the UI, or where SILVANN_DATA says (a container's
# volume), as the server and silvann_core.py read it
DATA_DIR = Path(os.environ.get("SILVANN_DATA") or REPO_ROOT)
MODELS_DIR = DATA_DIR / "models"                     # models/<name>/ — self-contained model folders
LAST_START = DATA_DIR / "ui_last_start.json"         # the model a --local UI starts again when it starts
TOKEN_FILE = DATA_DIR / ".huggingface_token"         # the Hugging Face token, as silvann_core.py's hf-token saves it
# the documents: docs/ beside the UI in a release, release/ in the development tree
RELEASE_DIR = REPO_ROOT if (REPO_ROOT / "docs" / "users.md").exists() else REPO_ROOT.parent / "release"
DOCS_DIR = RELEASE_DIR / "docs"
# the engine's source and the licence: in a release beside the docs, in the development tree src and its root
ENGINE_SRC = RELEASE_DIR / "engine" / "src" if (RELEASE_DIR / "engine" / "src").is_dir() else REPO_ROOT.parent / "src"
LICENCE_DIR = RELEASE_DIR if (RELEASE_DIR / "LICENSE").exists() else REPO_ROOT.parent


def list_models():
    """The model folders under MODELS_DIR that are there whole — a download still running is not one yet."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir() and model_present(d.name))


def model_configs_dir(model: str) -> Path:
    return MODELS_DIR / model / "configs"


def list_model_configs(model: str):
    """The run configs (*.json) inside a model folder's configs/ dir."""
    cdir = model_configs_dir(model)
    return sorted(p.name for p in cdir.glob("*.json")) if cdir.is_dir() else []


def model_default_config(model: str) -> Optional[str]:
    """The config a model folder declares as its default (pack.json `default_config`).
    Drives the Options-view config selector so it lands on the model's intended setup (e.g. the
    200k cooled config) rather than alphabetical-first. None if pack.json is absent/unreadable."""
    try:
        pack = json.loads((MODELS_DIR / model / "pack.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    dflt = pack.get("default_config")
    return dflt if dflt in list_model_configs(model) else None


# ⭐ A NEW CONFIG: every key the server reads, at its default — settings only. Help → "A model's config" (docs/config.md)
#   says every key and every model's option, and the Options view's **config guide** opens it. Options are left out: a
#   placement named in `options` is kept as it is, where one left out is chosen for the machine at each boot.
CONFIG_GUIDE_DOC = "Using SiLVANN · A model's config"


def new_config(model: str) -> dict:
    """A new config for `model`: every key the server reads at its default."""
    folder = MODELS_DIR / model
    if (folder / "model_index.json").exists():                  # a picture model
        return {"card": "auto",
                "image": {"width": 1024, "height": 1024, "steps": 40, "seed": None}}
    try:
        c = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        layers = (c.get("text_config") or c).get("num_hidden_layers")
    except (OSError, ValueError):
        layers = None
    return {
            # what the card holds beside the model, up to its length, a tier sharing the card with the cache (NN-93)
            "max_context": "auto",
            "max_tokens_per_video": 8192, "thinking": True, "sampler": "generation",
            "tools": "chill", "tools_folder": None, "lora": None, "num_cards": 1, "multi_user": "no", "card": "auto",
            "vram_margin_mb": 50, "tools_config": "default", "workers": [],
            "options": {},
            "pipeline_parallelism": "no",
            "pipeline_parallelism_layers": "0-%d" % (layers - 1) if layers else "",
            "pipeline_parallelism_next": "192.0.2.2:8766", "pipeline_parallelism_prev": "192.0.2.1",
            "pipeline_parallelism_port": 8766}


def ensure_config(model: str) -> tuple:
    """(the config to select, what was done): a model folder with no config gets `configs/default.json`, every key at
    its default (`new_config`) — ⚖ 2026-11-12 *"it tells me to pick a config when there is none (like deepseek) i suggest
    in that case to generate an empty one by default"*. (None, why) where the folder will not take it."""
    if not model or list_model_configs(model):
        return None, ""
    target = model_configs_dir(model) / "default.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(new_config(model), indent=2), encoding="utf-8")
    except OSError as e:
        return None, "no config, and none could be made in %s: %s" % (target.parent, e)
    return "default.json", "no config: made configs/default.json, every setting at its default"


# local mode needs the server beside the UI; a machine with no model yet downloads one from the same panel
LOCAL_AVAILABLE = SERVER_SCRIPT.exists()
LOCAL_APP = {"on": False}                            # --local: the UI as the app over this machine's models
LISTEN = {"host": None, "argv": None}                # where this UI listens and how it was asked to (▶ main)
START_AGAIN = DATA_DIR / "ui_start_again.json"       # the model a switch stopped, started by the UI started after it
# the runtime beside the UI in a release, the development tree's otherwise (as silvann_tools_repl.py finds it)
RUNTIME_DIR = Path(os.environ.get("SILVANN_RUNTIME") or (REPO_ROOT / "runtime" if (REPO_ROOT / "runtime").is_dir()
                                                         else REPO_ROOT.parent / "python"))


def model_present(name: str) -> bool:
    """A model folder that is there, as silvann_runtime.fetch.present says: a text model's pack and config, or a picture
    model's — diffusers' model_index.json and its transformer packed."""
    d = MODELS_DIR / name
    try:
        return ((d / "pack.json").is_file() and (d / "config.json").is_file()
                or (d / "model_index.json").is_file() and (d / "transformer" / "pack.json").is_file())
    except OSError:                    # a folder this process may not look into — a disk's own lost+found — is no model
        return False
