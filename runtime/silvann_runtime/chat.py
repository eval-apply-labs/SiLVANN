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

⭐ THE MODEL'S TOOL (NN-88, ▶ `tools.py`): a conversation begun with tools has one declared in its first turn's system
prompt, as the template declares a tool — `lisp`, its one argument a form. The model calls it in its own markup, and the
call is answered as the template frames a tool's answer:
```
  Qwen 3.5    <tool_call>\n<function=lisp>\n<parameter=form>\n(ls)\n</parameter>\n</function>\n</tool_call>   (it stops there)
              <|im_end|>\n<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n
  GLM 5.3     <tool_call>lisp<arg_key>form</arg_key><arg_value>(ls)</arg_value></tool_call>                  (it stops there)
              <|observation|><tool_response>…</tool_response><|assistant|><think>
  Mistral 3   [TOOL_CALLS]lisp[ARGS]{"form": "(ls)"}</s>
              [TOOL_RESULTS]…[/TOOL_RESULTS]
```
A Qwen or GLM answer stops at `</tool_call>` (`call_end`, one token), so a call is answered before the model writes on.
"""
import datetime
import json
import os
import re

import numpy as np

from .model_folder import Refused

TOOL = {"type": "function", "function": {
    "name": "lisp",
    "description": "A Lisp REPL on the user's machine, in a working folder the user gave it: one form a call, and its "
                   "value is the answer. Start with (help), which lists the commands; a path is absolute, or relative to "
                   "the working folder. A command that reads elsewhere or writes may be asked of the user first: call it, and the "
                   "answer comes back once the user has allowed it, or refused. To make a command of your own, a skill in Python or "
                   "a program that stays for later conversations, (guide) says how. (whoami) says which model you are.",
    "parameters": {"type": "object", "properties": {"form": {
        "type": "string", "description": "one Lisp form, such as (help) or (ls \"docs\")"}}, "required": ["form"]}}}

_QWEN_TOOLS = ("# Tools\n\nYou have access to the following functions:\n\n<tools>\n%s\n</tools>\n\nIf you choose to call a "
               "function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>"
               "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value "
               "for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
               "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
               "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You "
               "may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT "
               "after\n- If there is no function call available, answer the question like normal with your current "
               "knowledge and do not tell the user about function calls\n</IMPORTANT>")
_GLM_TOOLS = ("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are "
              "provided with function signatures within <tools></tools> XML tags:\n<tools>\n%s\n</tools>\n\nFor each "
              "function call, output the function name and arguments within the following XML format:\n<tool_call>"
              "{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}"
              "</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")


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
        # a call's first token, and the one it ends on where that is not the turn's end (Mistral's ends on `</s>`)
        mistral = self.arch.startswith("ministral3")
        self.call_start = self.tok.token_to_id("[TOOL_CALLS]" if mistral else "<tool_call>")
        self.call_end = None if mistral else self.tok.token_to_id("</tool_call>")

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

    def turn(self, seen, text, system=None, thinking=True, images=(), videos=(), tools=False):
        """The tokens a turn adds to a conversation that has seen `seen`; `images` the token counts of the pictures the
        user's message carries before its text, `videos` its videos after them, each a list of (tokens, time in seconds)
        a pair of frames; `tools`, in a first turn, declares the model's tool."""
        first = not seen
        tools = tools and first
        if tools and self.call_start is None:
            raise Refused("a %r model's tokenizer has no tool call" % self.arch)
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
                if tools:
                    system = _QWEN_TOOLS % json.dumps(TOOL, ensure_ascii=False) + "\n\n" + system
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
                if tools:
                    s += "[AVAILABLE_TOOLS]%s[/AVAILABLE_TOOLS]" % json.dumps([TOOL], ensure_ascii=False)
            return self.encode(s + "[INST]%s[/INST]" % text)
        # GLM 5.3
        s = ""
        if first:
            s = "[gMASK]<sop><|system|>Reasoning Effort: %s" % ("Max" if thinking else "Low")
            if tools:
                s += _GLM_TOOLS % json.dumps(TOOL["function"], ensure_ascii=False)
            if system:
                s += "<|system|>" + system
        if first or seen[-1] != self.user:
            s += "<|user|>"
        return self.encode(s + text + "<|assistant|><think>")

    def tool_calls(self, answer):
        """The forms an answer's last stretch calls the tool with, in order — [] where it ended without a call. Its one
        argument is the form whatever the model named it. A call that cannot be read is a `ToolMarkup`, said back to the
        model in place of an answer."""
        if self.call_start is None or self.call_start not in answer:
            return []
        if answer[-1] != self.call_end and (self.call_end is not None or answer[-1] not in self.stop):
            return []                                  # a call ends the stretch: at `</tool_call>`, or Mistral's `</s>`
        at = len(answer) - 1 - answer[::-1].index(self.call_start)
        if self.call_end is None:                      # Mistral's: every call since the answer's last text
            at = answer.index(self.call_start)
        text = self.tok.decode(answer[at:], skip_special_tokens=False)
        if self.arch.startswith("ministral3"):
            calls = []
            for part in text.split("[TOOL_CALLS]")[1:]:
                name, _, args = part.partition("[ARGS]")
                args = args.split("</s>")[0]
                try:
                    got = json.loads(args)
                    one = got["form"] if "form" in got else next(iter(got.values())) if len(got) == 1 else got["form"]
                    form = one if name.strip() == "lisp" else ToolMarkup("the one tool is lisp, not %s" % name.strip())
                except (ValueError, KeyError, TypeError, AttributeError):
                    form = ToolMarkup("the call's arguments do not read as {\"form\": \"…\"}: %s" % args[:200])
                calls.append(form if isinstance(form, ToolMarkup) else str(form))
            return calls
        if self.arch.startswith("glm5"):
            got = re.match(r"<tool_call>\s*([^<]*?)\s*<arg_key>[^<]*</arg_key><arg_value>(.*)</arg_value>\s*</tool_call>$",
                           text, re.S)
            name, form = (got.group(1), got.group(2)) if got else (None, None)
        else:
            got = re.match(r"<tool_call>\s*<function=([^>]*)>\s*<parameter=[^>]*>\n?(.*?)\n?</parameter>\s*</function>"
                           r"\s*</tool_call>$", text, re.S)
            name, form = (got.group(1), got.group(2)) if got else (None, None)
        if form is None:
            return [ToolMarkup("the call does not read as the tool lisp with its one argument, form: %s" % text[:200])]
        if name.strip() != "lisp":
            return [ToolMarkup("the one tool is lisp, not %s" % name.strip())]
        return [form]

    def close_call(self):
        """The tokens that end a turn after a call left unanswered (an answer stopped there), where the template has the
        turn end after the call: Qwen's `<|im_end|>` — the next turn's framing follows it."""
        return self.encode("<|im_end|>") if self.arch.startswith("qwen3_5") else []

    def tool_response(self, answers, thinking=True):
        """The tokens that answer a stretch's calls, `answers` the value of each, written — and open the model's turn
        again, as the template's generation prompt does."""
        if self.arch.startswith("ministral3"):
            return self.encode("".join("[TOOL_RESULTS]%s[/TOOL_RESULTS]" % a for a in answers))
        if self.arch.startswith("glm5"):
            return self.encode("<|observation|>" + "".join("<tool_response>%s</tool_response>" % a for a in answers)
                               + "<|assistant|><think>")
        return self.encode("<|im_end|>\n<|im_start|>user" + "".join("\n<tool_response>\n%s\n</tool_response>" % a
                                                                  for a in answers)
                           + "<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n"))


class ToolMarkup(str):
    """A call the model wrote that cannot be read: what to say back to it."""


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
