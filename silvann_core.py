"""SiLVANN's command line for what is not the server.

    python3 silvann_core.py download-model <name>     a model into models/<name>/, from Hugging Face — or, when it is
                                                      there, the files of its pack it lacks or holds otherwise
    python3 silvann_core.py check-model <name> [--json]   what download-model would fetch for a model that is here
    python3 silvann_core.py check-versions [--json]   each model here against its pack's version: update or done
    python3 silvann_core.py update-programs           each model here gets this release's program from its pack —
                                                      what an update of SiLVANN runs once it is installed
    python3 silvann_core.py list-models [--json]      the models this release knows how to fetch, and which are here
    python3 silvann_core.py split-experts <name>     a two-socket machine's: the model's folder split in place, each
                                                      socket's share of its experts in its experts/, as the first boot
                                                      over two sockets does (▶ silvann_runtime/experts_split.py)
    python3 silvann_core.py stitch-experts <name>    the pack put back in place, as the first boot over one socket does
    python3 silvann_core.py split-experts <name> DIR0 DIR1   each socket's share in a folder on that socket's own drive,
                                                      which a boot with `experts_dir` [DIR0, DIR1] reads
    python3 silvann_core.py stitch-experts DIR0 DIR1 OUT   the pack put back together from them, byte for byte
    python3 silvann_core.py hf-token                  whether a Hugging Face token is saved, and whose it is
    python3 silvann_core.py hf-token --set            save one, asked for without echoing it (or: hf-token <token>)
    python3 silvann_core.py hf-token --forget         remove it
    python3 silvann_core.py users                     the machine's mode and its accounts, each with how many
                                                      conversations it holds, the admins marked
    python3 silvann_core.py users add <name> [--admin]   an account: a folder of its own and a password (asked twice,
                                                      without echoing it; read from stdin when that is no terminal).
                                                      A machine's first is its admin, and takes what the machine held
                                                      before accounts — a single user's conversations, workspace and
                                                      definitions (users/__local/)
    python3 silvann_core.py users admin <name> on|off   an admin or not: the machine's settings, users and models
    python3 silvann_core.py users multi               multi user: the accounts signed in side by side, each one's
                                                      model in its own folder, the UI on the network
    python3 silvann_core.py users single              back to single user: one account at a time, an admin's model
                                                      reaching the machine, the UI on this machine alone
    python3 silvann_core.py users passwd <name>       a new password: their sign-ins end — the way back in for an
                                                      admin locked out
    python3 silvann_core.py users space <name> [<size>|none]   the most the account's folder may hold — 5GB, 500MB —
                                                      or none; with no size, what it holds now and what occupies it
    python3 silvann_core.py users remove <name> [--with-conversations]
    python3 silvann_core.py users adopt <name>        the conversations made before there were users moved into
                                                      <name>'s folder, every model's
    python3 silvann_core.py users link <name> <issuer> <subject>    <name> signs on with that account of the provider's
                                                      (sso.json; the subject is what a refused sign-on says)
    python3 silvann_core.py users unlink <name> <issuer> <subject>  no longer: their sign-ins end

A token makes Hugging Face's downloads faster than a client that is not signed in gets, and reaches a repository whose
terms its account accepted. It is saved in `.huggingface_token` beside `models/` (only its owner may read it), which the
UI's Options save too; `HF_TOKEN` in the environment comes before it.

`models/` and `user_sessions/` are beside this file, or in the folder `SILVANN_DATA` names (the Docker image's volume);
a user's folder is `users/<name>/` there (▶ silvann_runtime/users.py).

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
_HERE = ROOT / "runtime" if (ROOT / "runtime").is_dir() else ROOT.parent / "python"
sys.path.insert(0, os.environ.get("SILVANN_RUNTIME", str(_HERE)))
from silvann_runtime.fetch import (KNOWN, PACKS_PAGE, differences, ensure_model, offered, present, said, state,  # noqa: E402
                                   update_program)
from silvann_runtime.fetch import size as fetch_size                                # noqa: E402
from silvann_runtime.model_folder import Refused                   # noqa: E402
from silvann_runtime import users as _users                         # noqa: E402
from silvann_runtime import space as _space                         # noqa: E402

# what a user's machine keeps — the models and the conversations: beside the release, or where SILVANN_DATA says
DATA = Path(os.environ.get("SILVANN_DATA") or ROOT)
MODELS = DATA / "models"
TOKEN_FILE = DATA / ".huggingface_token"
DEFAULT_CONFIG = {"max_context": "auto", "max_tokens_per_video": 8192, "num_cards": 1, "multi_user": "no", "card": "auto",
                  "vram_margin_mb": 50, "lora": None, "thinking": True,
                  "sampler": "generation", "options": {},
                  "pipeline_parallelism": "no", "pipeline_parallelism_layers": "", "pipeline_parallelism_next": "192.0.2.2:8766",
                  "pipeline_parallelism_prev": "192.0.2.1", "pipeline_parallelism_port": 8766, "tools": "chill", "tools_config": "default",
                  "workers": []}
IMAGE_CONFIG = {"card": "auto", "image": {"width": 1024, "height": 1024, "steps": 40, "seed": None}}


def default_config(folder):
    """`configs/default.json` in a model folder, written when it has none: settings only — docs/config.md (the UI's
    Help → "A model's config") says what each does."""
    cfg = Path(folder) / "configs" / "default.json"
    if not cfg.exists() and (Path(folder) / "model_index.json").exists():       # a picture model's
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(IMAGE_CONFIG, indent=2) + "\n")
    if not cfg.exists():
        # no options: where the experts sit and how much of the card holds them are chosen for the machine at each boot
        # (silvann_runtime/autotune.py), unless the config is edited to name them
        # ⚖ 2026-10-17 *"ideally auto should always be the default, as it sizes the tokens against the remaining space
        #   after the model and it sizes the l1 cache against the actual tokens on the moe setups"*
        config = dict(DEFAULT_CONFIG, options={})
        if not config.get("pipeline_parallelism_layers"):
            try:
                c = json.loads((Path(folder) / "config.json").read_text())
                config["pipeline_parallelism_layers"] = "0-%d" % ((c.get("text_config") or c)["num_hidden_layers"] - 1)
            except (OSError, ValueError, KeyError):
                pass
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


def _password(name):
    """A password for `name`: asked twice without echoing it at a terminal, or the first line of stdin."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")
    import getpass
    first = getpass.getpass("%s's password: " % name)
    if getpass.getpass("again: ") != first:
        raise SystemExit("the two passwords differ: nothing changed")
    return first


def users_command(argv):
    """`users …` (▶ the module's note): the data folder's accounts, made, given a new password, removed, given the
    conversations made before there were users; and the machine's mode."""
    u = _users.Users(DATA)
    what, rest = (argv[0] if argv else "list"), argv[1:]
    try:
        if what == "list":
            names = u.names()
            if names or u.ready():
                print("%s user%s" % (u.mode(), ": one account at a time — %s has the machine now" % u.active()
                                     if u.active() else ": one account at a time" if u.mode() == "single" else ""))
            for name in names:
                n = u.conversations(name)
                print("%-32s %d conversation%s%s · %s%s" % (
                    name, n, "" if n == 1 else "s", " · admin" if u.admin(name) else "", _space.said(_space.used(u, name)),
                    " of %s" % _space.said(u.space(name)) if u.space(name) else ""))
                for iss, sub, email in u.accounts(name):
                    print("%-32s   signs on as %s at %s (subject %s)" % ("", email or "-", iss, sub))
            if not names and u.ready():
                print("no users yet: a server on this data folder asks for a sign-in, and its provider (sso.json) makes "
                      "a user at their first sign-on where its allow lets them in")
            elif not names:
                print("no account yet: `silvann users add <name>` makes this machine's first, its admin — or the first "
                      "page of its UI does. Until then nothing of SiLVANN answers")
            if names and not u.admins():
                print("no admin: `silvann users admin <name> on` names one — until then no one changes the machine's "
                      "settings in the UI")
            return 0
        if what == "multi":
            u.multi()
            print("multi user: the accounts are signed in side by side, each one's model in its own folder; a UI started "
                  "from now on listens on the network")
            return 0
        if what == "single":
            u.single()
            print("single user: one account at a time, every sign-in over — the next to sign in has the machine; a UI "
                  "started from now on listens on this machine alone")
            return 0
        if not rest:
            raise SystemExit("users %s <name>" % what)
        name = rest[0].strip().lower()
        if what == "add":
            if not u.names():
                took = u.first(name, _password(name))
                print("%s is this machine's account, its admin, their folder %s%s. SiLVANN asks for this sign-in: in the "
                      "UI, in silvann_cli (--sign-in %s) and from a script (POST /users/sign_in)"
                      % (name, u.folder(name), " — with what the machine held before: %s" % ", ".join(took) if took
                         else "", name))
                return 0
            u.add(name, _password(name), admin="--admin" in rest[1:])
            print("%s added%s, their folder %s" % (name, " as an admin" if "--admin" in rest[1:] else "", u.folder(name)))
            return 0
        if what == "admin":
            if rest[1:2] not in (["on"], ["off"]):
                raise SystemExit("users admin <name> on|off")
            u.set_admin(name, rest[1] == "on")
            print("%s %s" % (name, "is an admin" if rest[1] == "on" else "is an admin no more"))
            return 0
        if what == "passwd":
            u.passwd(name, _password(name))
            print("%s's password changed; their sign-ins on a running server end" % name)
            return 0
        if what == "space":
            if u._record(name) is None:
                raise SystemExit("%s is no user here" % name)
            if rest[1:]:
                u.set_space(name, _space.parse(rest[1]))
                _space.forget(u, name)
            g = _space.usage(u, name)
            print("%s: %s used%s" % (name, _space.said(g["used"]), " of %s — %s left" % (_space.said(g["limit"]), _space.said(g["left"]))
                                     if g["limit"] else " — no limit"))
            for c in g["conversations"]:
                print("  %-28s %-10s transcript %-9s state %-9s files %-9s%s" % (
                    c["name"][:28], _space.said(c["bytes"]), _space.said(c["transcript"]), _space.said(c["state"]),
                    _space.said(c["files"]), " · archived" if c["archived"] else ""))
            print("  %-28s %-10s of it the shared folder, attached/: %s" % ("the workspace", _space.said(g["workspace"]),
                                                                           _space.said(g["shared"])))
            print("  %-28s %s" % ("settings and the rest", _space.said(g["other"])))
            return 0
        if what == "remove":
            u.remove(name, with_conversations="--with-conversations" in rest[1:])
            print("%s removed, their folder with them" % name)
            return 0
        if what in ("link", "unlink"):
            if len(rest) != 3:
                raise SystemExit("users %s <name> <issuer> <subject>" % what)
            if what == "link":
                u.link(name, rest[1], rest[2])
                print("%s signs on with %s's account %s from now on" % (name, rest[1], rest[2]))
            else:
                u.unlink(name, rest[1], rest[2])
                print("%s no longer signs on with %s's account %s; their sign-ins on a running server end" % (
                    name, rest[1], rest[2]))
            return 0
        if what == "adopt":
            moved = u.adopt(name, u.unowned())
            print("%d conversation%s moved into %s's folder%s" % (len(moved), "" if len(moved) == 1 else "s", name,
                  "" if not moved else ": " + ", ".join("%s/%s" % m for m in moved)))
            return 0
    except _users.Refused as ex:
        raise SystemExit(str(ex))
    raise SystemExit("users: list, add, admin, multi, single, passwd, space, remove, adopt, link or unlink — not %r" % what)


def main(argv):
    if argv[:1] == ["hf-token"]:
        return hf_token(argv[1:])
    if argv[:1] == ["users"]:
        return users_command(argv[1:])
    if len(argv) >= 2 and argv[0] == "download-model":
        if not os.environ.get("HF_TOKEN") and saved_token():
            os.environ["HF_TOKEN"] = saved_token()          # huggingface_hub reads it at each request
        try:
            folder = ensure_model(str(MODELS / argv[1]), yes=True)
        except Refused as ex:
            raise SystemExit(str(ex))
        print("%s is in %s — configs: %s" % (argv[1], folder, default_config(folder)))
        return 0
    if len(argv) >= 2 and argv[0] == "check-model":
        if not os.environ.get("HF_TOKEN") and saved_token():
            os.environ["HF_TOKEN"] = saved_token()
        folder, name = MODELS / argv[1], argv[1]
        if name not in KNOWN:
            raise SystemExit("%r is not a model this release knows how to fetch — known: %s" % (name, ", ".join(sorted(KNOWN))))
        if not present(str(folder)):
            raise SystemExit("%s is not here: `silvann_core.py download-model %s` fetches it whole" % (name, name))
        d = differences(str(folder), KNOWN[name][0])
        if "--json" in argv[2:]:
            print(json.dumps(d))
        else:
            print("%s: %s" % (name, said(d)))
            for f, size in d["files"]:
                print("    %-60s %10s" % (f, fetch_size(size)))
        return 0
    if argv[:1] == ["update-programs"]:
        if not os.environ.get("HF_TOKEN") and saved_token():
            os.environ["HF_TOKEN"] = saved_token()
        here = [(name, repo) for name, (repo, _, _) in sorted(KNOWN.items()) if present(str(MODELS / name))]
        for name, repo in here:
            got = update_program(str(MODELS / name), repo, say=False)
            line = ("fetched %s" % ", ".join(got["fetched"]) if got["fetched"] else got["why"] or "")
            if got["kept"]:
                line += "%skept as changed here: %s (`silvann download %s` fetches the pack's)" % (
                    "; " if line else "", ", ".join(got["kept"]), name)
            if got["newer"]:
                line += "; the pack has a program for a later release — a newer SiLVANN is out"
            print("   %-44s %s" % (name, line), flush=True)
        if not here:
            print("   no model here yet")
        return 0
    if argv[:1] == ["check-versions"]:
        if not os.environ.get("HF_TOKEN") and saved_token():
            os.environ["HF_TOKEN"] = saved_token()
        states = {name: state(str(MODELS / name), repo) for name, (repo, _, _) in sorted(KNOWN.items())
                  if present(str(MODELS / name))}
        if "--json" in argv[1:]:
            print(json.dumps(states))
            return 0
        for name, st in states.items():
            v = lambda d: "v%s" % d["version"] if d else "none"       # noqa: E731
            print("%-44s %-8s here %s, the pack %s%s" % (name, st["state"], v(st.get("here")), v(st.get("pack")),
                                                       ("; " + st["why"]) if st.get("why") else
                                                       "; a newer SiLVANN is out" if st.get("newer") else ""))
        return 0
    if argv[:1] in (["split-experts"], ["stitch-experts"]):
        from silvann_runtime import experts_split
        return experts_split.main([argv[0].split("-")[0]] + argv[1:], models=MODELS)
    if argv[:1] == ["list-models"]:
        if argv[1:] == ["--json"]:
            print(json.dumps([{"name": name, "repo": KNOWN[name][0], "size": KNOWN[name][1], "base": KNOWN[name][2],
                               "here": present(str(MODELS / name))} for name in offered(MODELS)]))
            return 0
        shown = offered(MODELS)                    # the packs a user is offered, and any other that is here
        wide = max(len(name) for name in shown)
        for name in shown:
            repo, size, base = KNOWN[name]
            here = present(str(MODELS / name))
            print("%-*s %-8s %s  huggingface.co/%s, made from huggingface.co/%s"
                  % (wide, name, size, "here" if here else "     ", repo, base))
        print("to see all available packs go to %s — `silvann download <name>` fetches any of them" % PACKS_PAGE)
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
