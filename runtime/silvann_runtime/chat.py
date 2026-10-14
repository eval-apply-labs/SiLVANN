"""A CONVERSATION'S TURNS AS TOKENS — the model's own chat framing, a turn at a time.

A cartridge keeps every token it has seen, so a turn adds only what is new: the first turn its opening (the system
prompt, the user's text, the assistant's header), a later one the user's text and the header again, framed the way the
model's chat template frames it. What a turn ended on — the stop token the model chose — is already in the cartridge,
and a later turn continues from it.
```
  Qwen 3.5    <|im_start|>system … <|im_end|>\\n<|im_start|>user\\n … <|im_end|>\\n<|im_start|>assistant\\n<think>\\n
              a later turn: \\n<|im_start|>user\\n …   (the answer ended on <|im_end|>)
              thinking off: an empty think block after the header; thinking on, the block opened, and the template's
              reasoning-effort instruction before the system prompt where it has one (the 27B's, Qwen 3.8: xhigh's)
  GLM 5.3     [gMASK]<sop><|system|>Reasoning Effort: Max<|system|> … <|user|> … <|assistant|><think>
              a later turn: … <|assistant|><think>    (the answer ended on <|user|>, which opens the next)
              thinking off: Reasoning Effort: Low
  Mistral 3   <s>[SYSTEM_PROMPT] … [/SYSTEM_PROMPT][INST] … [/INST]
              a later turn: [INST] … [/INST]    (the answer ended on </s>)
              no thinking to turn off; with no system prompt given, Mistral's own (`SYSTEM_PROMPT.txt`, its dates filled
              in as Mistral's tooling does — the chat template leaves `{today}` unfilled)
```
A turn with pictures puts each before the text, as the chat template renders a message whose content is the pictures
then the text: Qwen's `<|vision_start|>`, a pad a token of the picture, `<|vision_end|>` (▶ `vision.py`); GLM's
`<|begin_of_image|>`, an `<|image|>` a token, `<|end_of_image|>` (▶ `vision_glm.py`). Its videos come after its pictures,
each pair of frames framed as Hugging Face's processor frames it in place of the template's video pad: Qwen's
`<|vision_start|>`, then a pair's time `<0.2 seconds>` (the middle of its two frames), `<|vision_start|>`, a
`<|video_pad|>` a token, `<|vision_end|>` a pair, then `<|vision_end|>`; GLM's `<|begin_of_video|>`, then a pair's
`<|begin_of_image|>`, an `<|image|>` a token, `<|end_of_image|>` and its time "0.0 seconds" (its first frame's), then
`<|end_of_video|>`.
"""
import datetime
import os
import re

import numpy as np

from .model_folder import Refused


class Chat:
    def __init__(self, folder):
        from tokenizers import Tokenizer
        self.arch = folder.architecture
        self.tok = Tokenizer.from_file(os.path.join(folder.path, "tokenizer.json"))
        self.stop = set(folder.end_tokens())
        if self.arch.startswith("qwen3_5"):
            self.stop.add(self.tok.token_to_id("<|im_end|>"))
            self.reasoning = _reasoning(folder.path)
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

    def image_starts(self, tokens, pads):
        """Where each run of picture pads begins in `tokens`, in order — `pads` a pad's id, or the ids a run's pads may be
        (a picture's, a video's pair's: the vision's `pads`)."""
        pads = set(np.atleast_1d(pads).tolist())
        return [i for i, t in enumerate(tokens) if t in pads and (i == 0 or tokens[i - 1] != t)]

    def decode(self, ids):
        return self.tok.decode([i for i in ids if i not in self.stop], skip_special_tokens=False)

    def opened(self, tokens):
        """The think block a turn's tokens leave open — Qwen's `<think>\\n`, GLM's `<think>` — which its answer's text
        begins inside, or "" where they closed it or opened none."""
        tail = self.decode(tokens[-4:])
        at = tail.rfind("<think>")
        return tail[at:] if at >= 0 and "</think>" not in tail[at:] else ""

    def turn(self, seen, text, system=None, thinking=True, images=(), videos=()):
        """The tokens a turn adds to a conversation that has seen `seen`; `images` the token counts of the pictures the
        user's message carries before its text, `videos` its videos after them, each a list of (tokens, time in seconds)
        a pair of frames."""
        first = not seen
        if images and not self.arch.startswith(("qwen3_5", "glm5")):
            raise Refused("a %r model is not given pictures here" % self.arch)
        if videos and not self.arch.startswith(("qwen3_5", "glm5")):
            raise Refused("a %r model is not given videos here" % self.arch)
        if self.arch.startswith("glm5"):
            # a video as its template and Hugging Face's processor frame it: each pair of frames a picture, its time after it
            text = ("".join("<|begin_of_image|>" + "<|image|>" * n + "<|end_of_image|>" for n in images)
                    + "".join("<|begin_of_video|>" + "".join("<|begin_of_image|>" + "<|image|>" * n + "<|end_of_image|>"
                                                              + "%.1f seconds" % t for n, t in v) + "<|end_of_video|>"
                              for v in videos) + text)
        else:
            text = ("".join("<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>" for n in images)
                    + "".join("<|vision_start|>" + "".join("<%.1f seconds><|vision_start|>" % t + "<|video_pad|>" * n
                                                           + "<|vision_end|>" for n, t in v) + "<|vision_end|>"
                              for v in videos) + text)
        if self.arch.startswith("qwen3_5"):
            if first:
                system = system or "You are a helpful AI assistant."
                if thinking and self.reasoning:
                    system = self.reasoning + "\n\n" + system
                s = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
                     % (system, text))
            else:
                s = "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % text
            s += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
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


def _reasoning(path):
    """The reasoning-effort instruction a Qwen template puts before the system prompt when it thinks at its default
    effort (the 27B's, Qwen 3.8: "Reasoning effort is set to xhigh. …"), read from the folder's chat_template.jinja —
    or "" where it has none (the 35B's and the 122B's)."""
    try:
        template = open(os.path.join(path, "chat_template.jinja"), encoding="utf-8").read()
    except OSError:
        return ""
    effort = re.search(r"reasoning_effort\s*\|\s*default\(\s*'(\w+)'\s*\)", template)
    said = effort and re.search(r"==\s*'%s'\s*%%\}\s*\{%%-?\s*set reasoning_instructions\s*=\s*'([^']*)'" % effort.group(1),
                                template)
    return said.group(1) if said else ""
