"""SiLVANN's command line for what is not the server.

    python3 silvann_core.py download-model <name>     a model into models/<name>/, from Hugging Face
    python3 silvann_core.py list-models [--json]      the models this release knows how to fetch, and which are here
    python3 silvann_core.py hf-token                  whether a Hugging Face token is saved, and whose it is
    python3 silvann_core.py hf-token --set            save one, asked for without echoing it (or: hf-token <token>)
    python3 silvann_core.py hf-token --forget         remove it

A token makes Hugging Face's downloads faster than a client that is not signed in gets, and reaches a repository whose
terms its account accepted. It is saved in `.huggingface_token` beside `models/` (only its owner may read it), which the
UI's Options save too; `HF_TOKEN` in the environment comes before it.

`models/` and `user_sessions/` are beside this file, or in the folder `SILVANN_DATA` names (the Docker image's volume).

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

# what a user's machine keeps — the models and the conversations: beside the release, or where SILVANN_DATA says
DATA = Path(os.environ.get("SILVANN_DATA") or ROOT)
MODELS = DATA / "models"
TOKEN_FILE = DATA / ".huggingface_token"
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


def saved_token():
    """The Hugging Face token saved in TOKEN_FILE, or None."""
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


def save_token(token):
    """`token` into TOKEN_FILE, readable by its owner alone; an empty one removes the file."""
    if not token:
        TOKEN_FILE.unlink(missing_ok=True)
        return
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token.strip() + "\n")
    os.chmod(TOKEN_FILE, 0o600)


def token_owner(token):
    """The account a token signs in as; raises Refused when Hugging Face refuses it, and answers None when it could not
    be asked (no network)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    try:
        return HfApi().whoami(token=token).get("name") or "?"
    except HfHubHTTPError as ex:
        if ex.response is not None and ex.response.status_code in (401, 403):
            raise Refused("Hugging Face refused the token (%d)" % ex.response.status_code)
        return None
    except Exception:
        return None


def masked(token):
    return token[:3] + "…" + token[-4:] if len(token) > 10 else "…"


def hf_token(args):
    if args == ["--forget"]:
        save_token("")
        print("the saved token is removed")
        return 0
    if args:
        if args == ["--set"]:
            import getpass
            token = getpass.getpass("Hugging Face token (huggingface.co/settings/tokens, read access): ").strip()
        else:
            token = args[0].strip()
        try:
            who = token_owner(token)
        except Refused as ex:
            raise SystemExit("not saved: %s" % ex)
        save_token(token)
        print("saved in %s — %s" % (TOKEN_FILE, "signed in as %s" % who if who else "not checked: Hugging Face did not answer"))
        return 0
    token = saved_token()
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is set in the environment, and the downloads use it%s"
              % (" (a saved token is there too: %s)" % masked(token) if token else ""))
    elif token:
        try:
            who = token_owner(token)
            state = "signed in as %s" % who if who else "not checked: Hugging Face did not answer"
        except Refused as ex:
            state = str(ex)
        print("a token is saved in %s: %s — %s" % (TOKEN_FILE, masked(token), state))
    else:
        print("no token: the downloads go as a client that is not signed in. `hf-token --set` saves one")
    return 0


def main(argv):
    if argv[:1] == ["hf-token"]:
        return hf_token(argv[1:])
    if len(argv) >= 2 and argv[0] == "download-model":
        if not os.environ.get("HF_TOKEN") and saved_token():
            os.environ["HF_TOKEN"] = saved_token()          # huggingface_hub reads it at each request
        try:
            folder = ensure_model(str(MODELS / argv[1]), yes=True)
        except Refused as ex:
            raise SystemExit(str(ex))
        print("%s is in %s — configs: %s" % (argv[1], folder, default_config(folder)))
        return 0
    if argv[:1] == ["list-models"]:
        if argv[1:] == ["--json"]:
            print(json.dumps([{"name": name, "repo": repo, "size": size, "here": present(str(MODELS / name))}
                              for name, (repo, size) in sorted(KNOWN.items())]))
            return 0
        for name, (repo, size) in sorted(KNOWN.items()):
            here = present(str(MODELS / name))
            print("%-36s %-8s %s  %s" % (name, size, "here" if here else "     ", "huggingface.co/" + repo))
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
