"""A PROMPT FOR FORMS — the reading main/'s two REPLs share: `silvann_base_repl.py` (the engine's language) and
`silvann_tools_repl.py` (the model's tool). Lines are read as they come — a prompt in a terminal, a line at a time from a
pipe — and gathered until the forms in them are whole, so a form over several lines is handed on once, and a line that
starts with `,` between forms is the REPL's own command.
"""
import re
import sys

GREEN, RESET = "\033[32m", "\033[0m"

# a text literal and a comment, whose parentheses open and close no form
_INERT = re.compile(r'"(?:[^"\\]|\\.)*"|;[^\n]*', re.S)


def open_forms(text):
    """How many forms the text leaves open: 0 when it is whole, more while a form waits for its `)`."""
    bare = _INERT.sub(" ", text)
    return bare.count("(") - bare.count(")")


def prompt_text(word):
    """`word> ` in green, the colour's codes between \\001 and \\002 so readline counts only the word's characters."""
    return "\001%s\002%s> \001%s\002" % (GREEN, word, RESET)


def texts(word):
    """What the input holds, as it comes: ("command", line) for a line starting with `,` between forms, ("forms", text)
    once the text's forms are whole, and ("unfinished", text) when the input ends inside a form. At a prompt (a
    terminal), `word> ` asks for a form and `  ... ` for its rest, Ctrl-C drops what was typed, and the line editing is
    readline's where Python has it; from a pipe, Ctrl-C is raised."""
    prompting = sys.stdin.isatty()
    if prompting:
        try:
            import readline  # noqa: F401 — line editing and history for input()
        except ImportError:
            pass
    pending = ""
    while True:
        try:
            if prompting:
                line = input(prompt_text(word) if not pending else "  ... ")
            else:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError
        except EOFError:
            if prompting:
                sys.stderr.write("\n")
            if pending.strip():
                yield "unfinished", pending
            return
        except KeyboardInterrupt:
            if not prompting:
                raise
            sys.stderr.write("\n")
            pending = ""
            continue
        if not pending and line.strip().startswith(","):
            yield "command", line.strip()
            continue
        pending += line if line.endswith("\n") else line + "\n"
        if open_forms(pending) > 0:
            continue
        text, pending = pending, ""
        if text.strip():
            yield "forms", text
