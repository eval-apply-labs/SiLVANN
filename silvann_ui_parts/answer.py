"""An answer's text as the parts it is drawn in: its thinking, its tool calls, its text."""
import re


# an answer's think block: closed, or still open while the model thinks (the server sends the `<think>` its prompt opened)
THINK_PART_RE = re.compile(r"<think>(.*?)(</think>|\Z)", re.DOTALL)
# the model's tool (NN-88): the server writes every model's calls the same way in the answer's text
TOOL_RE = re.compile(r"<tool_(call|response|ask|answer)>(.*?)</tool_\1>", re.DOTALL)
ASK_LABELS = {"folder": "allow the folder", "once": "allow once", "refuse": "refuse"}
# a picture model's answer names its picture (v0.10.0): `![picture](picture:NAME)`, drawn from the server's copy
PICTURE_RE = re.compile(r"!\[([^\]]*)\]\(picture:([A-Za-z0-9_.-]+)\)")


def _tool_parts(text: str) -> list:
    """Text outside a think block as ("text", markdown) and ("tools", lines) parts: a run of calls to the model's tool is
    one part — a form after `>`, its value under it, an ask and the user's answer as comments."""
    parts, block, pos = [], [], 0

    def flush():
        if block:
            parts.append(("tools", list(block)))
            block.clear()
    for m in TOOL_RE.finditer(text):
        gap = text[pos:m.start()]
        if gap.strip():
            flush()
            parts.append(("text", gap))
        kind, body = m.group(1), m.group(2)
        block.append({"call": "> " + body, "response": body, "ask": ";; asked: " + body,
                      "answer": ";; you answered: " + ASK_LABELS.get(body, body)}[kind])
        pos = m.end()
    flush()
    if text[pos:].strip():
        parts.append(("text", text[pos:]))
    return parts


def answer_parts(content: str) -> list:
    """An answer's text as the parts it is drawn in, in order: ("think", text, closed) a think block — `closed` False while
    the model is still in it —, ("tools", lines) a run of calls to the model's tool, ("text", markdown) the rest. A
    `</think>` with no `<think>` before it closes a block the answer began in (a conversation saved without the tag)."""
    parts, pos = [], 0
    first_open, first_close = content.find("<think>"), content.find("</think>")
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        parts.append(("think", content[:first_close], True))
        pos = first_close + len("</think>")
    for m in THINK_PART_RE.finditer(content, pos):
        parts += _tool_parts(content[pos:m.start()])
        parts.append(("think", m.group(1), m.group(2) == "</think>"))
        pos = m.end()
    parts += _tool_parts(content[pos:])
    return parts


def tools_block(lines: list) -> str:
    """A run of tool calls as the markdown a tools section shows: one `<!tools> … </tools>` block."""
    return "```lisp\n<!tools>\n%s\n</tools>\n```" % "\n".join(lines)
