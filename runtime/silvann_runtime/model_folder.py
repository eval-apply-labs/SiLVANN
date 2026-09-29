"""A MODEL FOLDER — a packed bundle, the model's own files beside it, and the LoRAs made for it.

```
  FOLDER/pack.json              the bundle's manifest: format, rotation, codec, policy, where it came from
  FOLDER/global.pkl · layers/   the weights
  FOLDER/config.json · tokenizer.json · chat_template.jinja · generation_config.json …   what the model is
  FOLDER/loras/NAME/            an adapter merged into the matrices it adapts (`nn_lora_overlay.py`)
```
The folder answers what the runtime needs before it touches a card: the architecture and its sizes, the tokenizer,
which LoRAs exist, and the IDENTITY a conversation records, so a conversation saved on one model is refused by
another.
"""
import hashlib
import json
import os
import sys

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
import nn_load_bundle as NL                                         # noqa: E402


class Refused(Exception):
    pass


class ModelFolder:
    def __init__(self, path, lora=None):
        self.path = os.path.abspath(path)
        cfg = os.path.join(self.path, "config.json")
        if not os.path.isfile(cfg):
            raise Refused("%s has no config.json — carry the model's files into it with "
                          "`nn_pack_model.py --raw RAW --out %s --carry-only`" % (self.path, self.path))
        whole = json.load(open(cfg))
        self.config = whole.get("text_config", whole)
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

    def loras(self):
        d = os.path.join(self.path, "loras")
        return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "pack.json"))) if os.path.isdir(d) else []

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
        eos = self.generation.get("eos_token_id", [])
        return set(eos if isinstance(eos, list) else [eos])
