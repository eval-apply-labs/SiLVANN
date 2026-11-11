"""A MODEL FOLDER — a packed bundle, the model's own files beside it, and the LoRAs made for it.

```
  FOLDER/pack.json              the bundle's manifest: format, rotation, codec, policy, where it came from
  FOLDER/global.pkl · layers/   the weights
  FOLDER/config.json · tokenizer.json · chat_template.jinja · generation_config.json …   what the model is
  FOLDER/loras/NAME/            an adapter merged into the matrices it adapts (`nn_lora_overlay.py`), or a rank-1 mask the
                                verbs apply beside them (`nn_rank1_mask.py`)
```
The folder answers what the runtime needs before it touches a card: the architecture and its sizes, the tokenizer,
which LoRAs exist, and the IDENTITY a conversation records, so a conversation saved on one model is refused by
another.
"""
import hashlib
import json
import os
import re
import sys

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
import nn_load_bundle as NL                                         # noqa: E402
from .version import RELEASE, parse                                 # noqa: E402


class Refused(Exception):
    pass


# ⭐ THE WORDS A PUBLISHED PROGRAM CAN NAME THAT THIS ENGINE DOES NOT HAVE, each with the word that does its work. A pack's
#   lisp/ travels with the pack, so a pack downloaded for an older release keeps that release's program; one that names
#   such a word is refused at the boot with how to fetch the pack's own again, where it would otherwise stop at its
#   first token on a word the evaluator does not know. A program that defines the word itself (a `defun` of it) answers
#   it, and boots.
RETIRED_WORDS = {"sys__compute": "(sys__scheduler__add cu names program [answer]), its value by (sys__wait_for_result answer)",
                 "sys__completed": "(sys__result answer), which answers #t once the answer is filled",
                 "sys__wait": "(sys__wait_for_result answer) — what an asked computation came to, read from its answer — "
                              "or (sys__wait_for_device cu), a block waited for until it is idle,"}
_RETIRED = re.compile(r"(?<![\w-])(%s)(?![\w-])" % "|".join(RETIRED_WORDS))


def retired_words(text):
    """The retired words a program's text names and does not define, in the order it first names them — its `;`
    comments left out, so a comment that mentions one refuses nothing."""
    text = re.sub(r";[^\n]*", "", text)
    named = []
    for m in _RETIRED.finditer(text):
        word = m.group(1)
        if word not in named and not re.search(r"\(defun\s*\(\s*%s[\s)]" % re.escape(word), text):
            named.append(word)
    return named


# ⭐ A PACK CARRIES A PROGRAM FOR EACH RELEASE WHOSE PROGRAM CHANGED, `lisp/v<X.Y.Z>/`, and a boot reads the one of the
#   highest release at or below its own: an older release keeps reading the program written for its words, so a pack
#   an older engine reads never needs a `defun` of a word a newer one retired. ⚖ 2026-11-06 *"lisp would have
#   subfolders and you only pick ones with your version number or lower … this way we dont need to backport older verbs
#   as defun either"*. `lisp/dev/` — a development box's, the tree's files (▶ scripts/model_lisp.py) — comes before
#   them all; a `lisp/` with neither holds its files itself, a pack's layout before v0.24.0.
DEV = "dev"


def choose(names, release=None):
    """Of a `lisp/`'s subfolder names, the `v<X.Y.Z>` of the highest release at or below `release` (this one's,
    `version.RELEASE`); None when none is."""
    top = parse(release or RELEASE)
    found = [(parse(n), n) for n in names if n.startswith("v") and parse(n) is not None and parse(n) <= top]
    return max(found)[1] if found else None


def newer(names, release=None):
    """Of a `lisp/`'s subfolder names, those of a release above `release` (this one's), oldest first: programs for a
    later engine than this."""
    top = parse(release or RELEASE)
    return [n for _, n in sorted((parse(n), n) for n in names
                                 if n.startswith("v") and parse(n) is not None and parse(n) > top)]


def program_dir(lisp, release=None):
    """The folder within a `lisp/` whose files this engine reads as the program: `dev/` when there is one, else the
    `v<X.Y.Z>/` of the highest release at or below this one's, else `lisp/` itself."""
    if os.path.isdir(os.path.join(lisp, DEV)):
        return os.path.join(lisp, DEV)
    names = [n for n in os.listdir(lisp) if os.path.isdir(os.path.join(lisp, n))] if os.path.isdir(lisp) else []
    chosen = choose(names, release)
    return os.path.join(lisp, chosen) if chosen else lisp


def architecture_of(path):
    """A model folder's architecture, read from its config.json alone — what `ModelFolder.architecture` answers, for a
    caller deciding a boot's options before the bundle is opened."""
    cfg = os.path.join(os.path.abspath(path), "config.json")
    if not os.path.isfile(cfg):
        raise Refused("%s has no config.json" % os.path.abspath(path))
    whole = json.load(open(cfg))
    return whole.get("text_config", whole).get("model_type", "")


class ModelFolder:
    def __init__(self, path, lora=None):
        self.path = os.path.abspath(path)
        cfg = os.path.join(self.path, "config.json")
        if not os.path.isfile(cfg):
            raise Refused("%s has no config.json — carry the model's files into it with "
                          "`nn_pack_model.py --raw RAW --out %s --carry-only`" % (self.path, self.path))
        whole = json.load(open(cfg))
        self.config = whole.get("text_config", whole)
        # the end of a turn where no generation config names it (DeepSeek-V4.1's repository carries none), as Hugging Face's
        # generate falls back to the model's config
        self.eos = whole.get("eos_token_id", self.config.get("eos_token_id", []))
        self.generation = {}
        gen = os.path.join(self.path, "generation_config.json")
        if os.path.isfile(gen):
            self.generation = json.load(open(gen))
        self.bundle = NL.Bundle(self.path, lora=lora)
        self.manifest = self.bundle.manifest
        self.lora = lora

    @property
    def architecture(self):
        return self.config.get("model_type", "")

    @property
    def lisp(self):
        """The folder this engine reads the model's program from (▶ `program_dir`): `lisp/dev/`, `lisp/v<X.Y.Z>/` or
        `lisp/`."""
        return program_dir(os.path.join(self.path, "lisp"))

    def has_program(self, name):
        """Whether the program this engine reads has the file `<name>.lisp` — a part a release before it did not have."""
        return os.path.isfile(os.path.join(self.lisp, name + ".lisp"))

    def program(self, name):
        """`<name>.lisp` of the program this engine reads in this folder (▶ `lisp`) — a file of the model's program,
        which travels with the model: the boot reads it and the evaluator runs it. Refused when the folder has none."""
        path = os.path.join(self.lisp, name + ".lisp")
        shown = os.path.relpath(path, self.path)
        model = os.path.basename(self.path.rstrip(os.sep))
        if not os.path.isfile(path):
            raise Refused("%s has no %s — a file of the model's program. `silvann download %s` fetches what the folder "
                          "lacks of its pack, this release's program with it; a pack that does not carry the file is its "
                          "uploader's to fix. A folder of your own takes its family's files from the release's lisp/"
                          % (self.path, shown, model))
        with open(path, encoding="utf-8", errors="replace") as f:
            named = retired_words(f.read())
        if named:
            raise Refused("%s's %s names %s, which this engine does not have: the program was written for an older "
                          "release. `silvann download %s` fetches this release's program from the pack; a program of "
                          "your own uses %s" % (self.path, shown, ", ".join(named), model,
                                                "; ".join("%s for %s" % (RETIRED_WORDS[w], w) for w in named)))
        return path

    def loras(self):
        d = os.path.join(self.path, "loras")
        return sorted(n for n in os.listdir(d)
                      if any(os.path.isfile(os.path.join(d, n, f)) for f in ("pack.json", "mask.json"))) if os.path.isdir(d) else []

    def lora_about(self, name):
        """A LoRA of this folder: its `kind` — "adapter", whose matrices are written into the model's own between
        conversations, or "rank1-mask", applied beside them and so chosen when the model boots — and the `description`
        its maker wrote of what it is for, from its `pack.json` or `mask.json` ("" when it has none)."""
        d = os.path.join(self.path, "loras", name)
        if os.path.isfile(os.path.join(d, "mask.json")):
            return dict(kind="rank1-mask", description=json.load(open(os.path.join(d, "mask.json"))).get("description", ""))
        return dict(kind="adapter", description=json.load(open(os.path.join(d, "pack.json"))).get("description", ""))

    def identity(self):
        """What a conversation records and checks: the model — the raw weights it was packed from and how — and
        the LoRA it runs with, or none."""
        m = self.manifest
        how = json.dumps(dict(rotation=m["rotation"], codec=m["codec"], policy=m["policy"], residual=m.get("residual")),
                         sort_keys=True)
        ident = dict(model=m["source"]["index_sha"], packing=hashlib.sha256(how.encode()).hexdigest()[:16],
                     architecture=self.architecture, lora=None)
        if self.bundle.lora is not None:
            ident["lora"] = dict(name=self.bundle.lora["name"], weights=self.bundle.lora["adapter"]["weights_sha"])
        return ident

    def end_tokens(self):
        eos = self.generation.get("eos_token_id", self.eos)
        return set(eos if isinstance(eos, list) else [eos])
