"""A CONVERSATION'S TURNS AS TOKENS — the model's own chat framing, a turn at a time.

A cartridge keeps every token it has seen, so a turn adds only what is new: the first turn its opening (the system
prompt, the user's text, the assistant's header), a later one the user's text and the header again, framed the way the
model's chat template frames it. What a turn ended on — the stop token the model chose — is already in the cartridge,
and a later turn continues from it.
```
  Qwen 3.5    <|im_start|>system … <|im_end|>\\n<|im_start|>user\\n … <|im_end|>\\n<|im_start|>assistant\\n
              a later turn: \\n<|im_start|>user\\n …   (the answer ended on <|im_end|>)
              thinking off: an empty think block after the header
  GLM 5.3     [gMASK]<sop><|system|>Reasoning Effort: Max<|system|> … <|user|> … <|assistant|><think>
              a later turn: … <|assistant|><think>    (the answer ended on <|user|>, which opens the next)
              thinking off: Reasoning Effort: Low
  Mistral 3   <s>[SYSTEM_PROMPT] … [/SYSTEM_PROMPT][INST] … [/INST]
              a later turn: [INST] … [/INST]    (the answer ended on </s>)
              no thinking to turn off; with no system prompt given, Mistral's own (`SYSTEM_PROMPT.txt`, its dates filled
              in as Mistral's tooling does — the chat template leaves `{today}` unfilled)
```
"""
import datetime
import os

from .model_folder import Refused


class Chat:
    def __init__(self, folder):
        from tokenizers import Tokenizer
        self.arch = folder.architecture
        self.tok = Tokenizer.from_file(os.path.join(folder.path, "tokenizer.json"))
        self.stop = set(folder.end_tokens())
        if self.arch.startswith("qwen3_5"):
            self.stop.add(self.tok.token_to_id("<|im_end|>"))
        elif self.arch.startswith("glm5"):
            self.user = self.tok.token_to_id("<|user|>")
        elif self.arch.startswith("ministral3"):
            sp = os.path.join(folder.path, "SYSTEM_PROMPT.txt")
            self.default_system = open(sp, encoding="utf-8").read() if os.path.isfile(sp) else None
        else:
            raise Refused("no chat framing for a %r model" % self.arch)
        self.stop.discard(None)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode([i for i in ids if i not in self.stop], skip_special_tokens=False)

    def turn(self, seen, text, system=None, thinking=True):
        """The tokens a turn adds to a conversation that has seen `seen`."""
        first = not seen
        if self.arch.startswith("qwen3_5"):
            if first:
                s = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
                     % (system or "You are a helpful AI assistant.", text))
            else:
                s = "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % text
            if not thinking:
                s += "<think>\n\n</think>\n\n"
            return self.encode(s)
        if self.arch.startswith("ministral3"):
            s = ""
            if first:
                if system is None and self.default_system:
                    today = datetime.date.today()
                    system = self.default_system.replace("{today}", today.isoformat()).replace(
                        "{yesterday}", (today - datetime.timedelta(days=1)).isoformat())
                s = "<s>" + ("[SYSTEM_PROMPT]%s[/SYSTEM_PROMPT]" % system if system else "")
            return self.encode(s + "[INST]%s[/INST]" % text)
        # GLM 5.3
        s = ""
        if first:
            s = "[gMASK]<sop><|system|>Reasoning Effort: %s" % ("Max" if thinking else "Low")
            if system:
                s += "<|system|>" + system
        if first or seen[-1] != self.user:
            s += "<|user|>"
        return self.encode(s + text + "<|assistant|><think>")
