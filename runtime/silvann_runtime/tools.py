"""THE MODEL'S REPL — a Lisp session a model opens in its answer, its commands the machine's (NN-88).

⚖ 2026-10-14: *"every tool call is itself a REPL that allows the model to interact with external tools... this way we
just need to implement a list of commands and the agent by invoking the repl can then ask for the command list and it
will know what he can do"* · *"i am actually fine doing it in python … python can lower it back into the original lisp
if we so choose and it can therefore call c methods"*. ▶ `src/docs/design/pending/tool_calling.md`.

A model calls one tool, `lisp`, whose one argument is a form; the server hands the form's text to its conversation's
session and the answer back (`Session.run`):
```
  (help)                       -> the commands, a line each          (help 'read) -> one of them
  (ls "docs")                  -> ("models.md" "users.md")
  (define n (length (ls)))     -> n                                  n -> 7
  (read "docs/users.md" 140 3) -> "…three lines…"                    (cat "docs/users.md") -> read's (ALIASES)
  (reset)                      -> a fresh session: the names gone, the first folder; files and allowed folders stay
  (keep 'f "what it does")     -> the library, a JSON file every conversation shares: (import 'f) in another session
```
The forms are read by the composer's reader (`composer.read`: names, numbers, `"text"`, quotes) and evaluated here:
`quote` `if` `define` `set!` `let` `lambda` `begin` `and` `or`; numbers, text, lists, `#t`, `#f` and `nil`. A value
is written back in the reader's syntax, so the model reads what it would write; an error is the value `(error "…")`.
"""
import html.parser
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request

from . import skills as _skills
from .composer import ComposerError, Text, read, to_text

ANSWER_MAX = 16384               # the most characters an answer carries back; the rest is said to be cut
READ_LINES = 200                 # the lines `read` answers when it is not told how many
RUN_SECONDS = 60                 # a command's time unless `bash` is told another, at most RUN_SECONDS_MAX
RUN_SECONDS_MAX = 600
RUN_BYTES = 1 << 20              # a command printing more is stopped there
FETCH_SECONDS = 30
FETCH_BYTES = 4 << 20            # the most of a page `fetch` reads
# ⭐ THE NAMES MODELS REACH FOR — another Lisp's or Python's word for a command, answered as that command: of 159 calls
#   the models made in the tests, 25 named something not here, and 18 of those were these (`MEASURED`,
#   measurements/2026-10-16_tool_call_names_precision.md). Each is its command's own procedure, so it asks as the
#   command asks; `(help)` names them. `exit`, a REPL's way out, is `reset`: the session's next form starts afresh.
ALIASES = {"read-file": "read", "slurp-file": "read", "read-all": "read", "cat": "read", "slurp": "read",
           "system": "bash", "shell": "bash", "sh": "bash", "exit": "reset"}


class Sym(str):
    """A quoted name — `'read` — as a value."""


class Fault(Exception):
    """A form that cannot be evaluated: said back to the model as `(error "…")`."""


class Closure:
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env


class Env(dict):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

    def find(self, name):
        e = self
        while e is not None:
            if name in e:
                return e
            e = e.parent
        raise Fault("%s is not defined — (help) lists what is" % name)


# ── values written back ───────────────────────────────────────────────────────────────────────────────────────────
def write(v):
    """A value in the reader's syntax."""
    if v is True:
        return "#t"
    if v is False:
        return "#f"
    if v is None:
        return "nil"
    if isinstance(v, Text):
        return to_text(v)
    if isinstance(v, str):                              # a Sym, or a name
        return str(v)
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(write(x) for x in v) + ")"
    if isinstance(v, (Closure,)) or callable(v):
        return "#<procedure>"
    return repr(v) if isinstance(v, float) else str(v)


def source(form):
    """A form as the reader reads it again — a quote written as `'x`, which `write` would give as a list."""
    if isinstance(form, tuple) and form and form[0] == "quote":
        return "'" + source(form[1])
    if isinstance(form, list):
        return "(" + " ".join(source(x) for x in form) + ")"
    return write(form)


# ── the evaluator ─────────────────────────────────────────────────────────────────────────────────────────────────
CONSTANTS = {"#t": True, "#f": False, "nil": None}
SPECIAL = {"quote", "if", "define", "set!", "lambda", "let", "begin", "and", "or"}     # the forms evaluate reads itself


def _truth(v):
    return v is not False and v is not None


def evaluate(x, env, session):
    while True:                                          # a tail call loops here, not in Python's stack
        if isinstance(x, Text) or isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            if x in CONSTANTS:
                return CONSTANTS[x]
            return env.find(x)[x]
        if isinstance(x, tuple) and x and x[0] == "quote":
            return _quoted(x[1])
        if not isinstance(x, list):
            raise Fault("not a form: %r" % (x,))
        if not x:
            return None
        head = x[0]
        if head == "if":
            if len(x) not in (3, 4):
                raise Fault("(if test then [else])")
            if _truth(evaluate(x[1], env, session)):
                x = x[2]
            elif len(x) == 4:
                x = x[3]
            else:
                return None
            continue
        if head == "define":                             # its source kept, for the library (▶ `keep`)
            if len(x) >= 3 and isinstance(x[1], list) and x[1]:       # (define (f a b) body…)
                env[x[1][0]] = Closure(x[1][1:], x[2:], env)
                session.sources[x[1][0]] = source(x)
                return Sym(x[1][0])
            if len(x) != 3 or not isinstance(x[1], str):
                raise Fault("(define name value) or (define (name args…) body…)")
            env[x[1]] = evaluate(x[2], env, session)
            session.sources[x[1]] = source(x)
            return Sym(x[1])
        if head == "set!":
            if len(x) != 3 or not isinstance(x[1], str):
                raise Fault("(set! name value)")
            env.find(x[1])[x[1]] = evaluate(x[2], env, session)
            return env.find(x[1])[x[1]]
        if head == "lambda":
            if len(x) < 3 or not isinstance(x[1], list):
                raise Fault("(lambda (args…) body…)")
            return Closure(x[1], x[2:], env)
        if head == "let":
            if len(x) < 3 or not isinstance(x[1], list):
                raise Fault("(let ((name value)…) body…)")
            inner = Env(env)
            for pair in x[1]:
                if not (isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)):
                    raise Fault("(let ((name value)…) body…)")
                inner[pair[0]] = evaluate(pair[1], env, session)
            env, x = inner, ["begin"] + x[2:]
            continue
        if head == "begin":
            for form in x[1:-1]:
                evaluate(form, env, session)
            if len(x) == 1:
                return None
            x = x[-1]
            continue
        if head in ("and", "or"):
            v = head == "and"
            for form in x[1:]:
                v = evaluate(form, env, session)
                if _truth(v) != (head == "and"):
                    return v
            return v
        f = evaluate(head, env, session)
        args = [evaluate(a, env, session) for a in x[1:]]
        if isinstance(f, Closure):
            env = _bind(f, args)
            x = ["begin"] + f.body
            continue
        if callable(f):
            try:
                return f(*args)
            except TypeError as ex:                       # the arguments a builtin does not take: said by its name
                raise Fault("(%s …) given %d argument%s it does not take: %s" % (
                    head if isinstance(head, str) else "a procedure", len(args), "" if len(args) == 1 else "s",
                    str(ex).split(") ", 1)[-1]))
        raise Fault("%s is not a procedure" % write(f))


def _quoted(v):
    if isinstance(v, Text) or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return CONSTANTS.get(v, Sym(v))
    if isinstance(v, tuple) and v and v[0] == "quote":
        return [Sym("quote"), _quoted(v[1])]
    return [_quoted(e) for e in v]


def _bind(f, args):
    inner = Env(f.env)
    if len(args) != len(f.params):
        raise Fault("a procedure of %d arguments given %d" % (len(f.params), len(args)))
    for name, value in zip(f.params, args):
        inner[name] = value
    return inner


def apply(session, f, args):
    """`f` applied — a procedure of the session's, or a builtin — for the builtins that call one (`map`)."""
    if isinstance(f, Closure):
        return evaluate(["begin"] + f.body, _bind(f, args), session)
    if callable(f):
        return f(*args)
    raise Fault("%s is not a procedure" % write(f))


# ── the builtins: numbers, lists, text ────────────────────────────────────────────────────────────────────────────
def _num(*xs):
    for v in xs:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Fault("%s is not a number" % write(v))
    return xs


def _text(v, what="a text"):
    if not isinstance(v, str) or isinstance(v, Sym):
        raise Fault("%s is not %s" % (write(v), what))
    return v


def _list(v):
    if v is None:
        return []
    if not isinstance(v, list):
        raise Fault("%s is not a list" % write(v))
    return v


def _builtins(session):
    def sub(a, *bs):
        _num(a, *bs)
        return -a if not bs else a - sum(bs)

    def div(a, b):
        _num(a, b)
        if b == 0:
            raise Fault("division by zero")
        return a // b if isinstance(a, int) and isinstance(b, int) and a % b == 0 else a / b

    def cmp(op):
        def f(*xs):
            _num(*xs)
            return all(op(a, b) for a, b in zip(xs, xs[1:]))
        return f

    return {
        "+": lambda *xs: sum(_num(*xs)), "-": sub, "*": lambda *xs: __import__("math").prod(_num(*xs)), "/": div,
        "=": lambda a, b: a == b, "<": cmp(lambda a, b: a < b), ">": cmp(lambda a, b: a > b),
        "<=": cmp(lambda a, b: a <= b), ">=": cmp(lambda a, b: a >= b), "not": lambda v: not _truth(v),
        "list": lambda *xs: list(xs), "car": lambda v: _list(v)[0] if _list(v) else None,
        "cdr": lambda v: _list(v)[1:] or None, "cons": lambda a, v: [a] + _list(v), "length": lambda v: len(_list(v))
        if not isinstance(v, str) else len(v), "null?": lambda v: v is None or v == [],
        "append": lambda *vs: [e for v in vs for e in _list(v)] or None,
        "nth": lambda v, i: _list(v)[_num(i)[0]] if 0 <= i < len(_list(v)) else None,
        "map": lambda f, v: [apply(session, f, [e]) for e in _list(v)] or None,
        "filter": lambda f, v: [e for e in _list(v) if _truth(apply(session, f, [e]))] or None,
        "string-append": lambda *ts: Text("".join(_text(t) for t in ts)),
        "substring": lambda t, a, b=None: Text(_text(t)[a:b]),
        "string=?": lambda a, b: _text(a) == _text(b),
        "contains?": lambda t, part: _text(part) in _text(t),
        "split": lambda t, sep=None: [Text(s) for s in _text(t).split(sep if sep is None else _text(sep))] or None,
        "join": lambda v, sep=Text(" "): Text(_text(sep).join(_text(e) for e in _list(v))),
        "lines": lambda t: [Text(s) for s in _text(t).split("\n")] or None,
        "number->string": lambda n: Text(write(_num(n)[0])),
        "string->number": lambda t: _number(_text(t)),
    }


def _run(shell, command, cwd, seconds):
    """(what it printed, its exit code, why it was stopped or None): the command in a session of its own, so that it and
    what it started are stopped together — after `seconds`, or once it has printed RUN_BYTES."""
    p = subprocess.Popen([shell, "-c", command], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=hasattr(os, "killpg"))
    out, why = bytearray(), []

    def stop(reason):
        why.append(reason)
        try:
            os.killpg(p.pid, signal.SIGKILL) if hasattr(os, "killpg") else p.kill()
        except OSError:
            pass

    def drain():
        while True:
            b = p.stdout.read1(65536)
            if not b:
                return
            out.extend(b)
            if len(out) >= RUN_BYTES:
                return stop("\n[stopped: it printed more than %d MB]" % (RUN_BYTES >> 20))
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        p.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        stop("\n[stopped after %d s]" % seconds)
        p.wait()
    reader.join(1)                                # what it left running may hold the output open: not waited for
    return bytes(out[:RUN_BYTES]), p.returncode, why[0] if why else None


class _Words(html.parser.HTMLParser):
    """A page's words: its text without markup, scripts or styles, a block's on a line of its own."""
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table"}

    def __init__(self):
        super().__init__()
        self.parts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        self.skipping += tag in self.SKIP
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        self.skipping -= tag in self.SKIP and self.skipping > 0
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def _page_text(page):
    w = _Words()
    w.feed(page)
    lines = (" ".join(l.split()) for l in "".join(w.parts).split("\n"))
    return "\n".join(l for l in lines if l)


def _number(t):
    try:
        return int(t)
    except ValueError:
        try:
            return float(t)
        except ValueError:
            return False


# ── the classifier: what a command may do unasked ─────────────────────────────────────────────────────────────────
# ⚖ 2026-10-14: *"the classifier should check for example if the reading folder is in the allowed list and if not it
#   should ask what to do, if adding the folder to the allowed list, read just once or refuse, or if the classifier is
#   set to chill just let all reads go and writes asked and if the classifier is set to full-auto then everything goes"*
MODES = ("ask", "chill", "full-auto")


class Asked(Exception):
    """A command the classifier stopped: the question for the user, its choices, and what a yes would allow."""

    def __init__(self, question, choices, touches, target):
        super().__init__(question)
        self.question, self.choices, self.touches, self.target = question, choices, touches, target


class Classifier:
    """What a command may do without asking: its mode — a conversation's, from the server's config's `tools` until the
    user chooses another — and the folders a read goes unasked in, every conversation's, kept in `tools.json` (the data
    folder's; every model's) as `{"allowed": [folders]}`.
      ask        a read in an allowed folder goes; one elsewhere asks — the folder allowed, this once, or refused;
                 a write, a run or the network asks — this once, or refused
      chill      every read goes; a write, a run or the network asks
      full-auto  everything goes
    """

    def __init__(self, path=None, mode="ask", allowed=()):
        self.path, self.mode, self.allowed, self.once = path, mode, [os.path.realpath(a) for a in allowed], set()
        if path and os.path.isfile(path):
            import json
            with open(path) as f:
                got = json.load(f)
            self.allowed += [os.path.realpath(a) for a in got.get("allowed", []) if os.path.realpath(a) not in self.allowed]
        if self.mode not in MODES:
            raise Fault("the mode is one of %s, not %r" % (", ".join(MODES), self.mode))

    def conversation(self, mode):
        """A conversation's classifier: its own mode and what it allowed once; the folders this one's, shared."""
        if mode not in MODES:
            raise Fault("the mode is one of %s, not %r" % (", ".join(MODES), mode))
        k = Classifier.__new__(Classifier)
        k.path, k.mode, k.allowed, k.once = self.path, mode, self.allowed, set()
        return k

    def save(self):
        if self.path:
            import json
            with open(self.path, "w") as f:
                json.dump({"allowed": self.allowed}, f, indent=2)

    def inside(self, target):
        return any(target == a or target.startswith(a.rstrip(os.sep) + os.sep) for a in self.allowed)

    def check(self, touches, target):
        """Nothing when the command may go; `Asked` when the user is to say."""
        if touches is None or (touches, target) in self.once:
            self.once.discard((touches, target))
            return
        if self.mode == "full-auto" or touches == "read" and (self.mode == "chill" or self.inside(target)):
            return
        if touches == "read":
            raise Asked("read %s" % target, ("folder", "once", "refuse"), touches, target)
        raise Asked("%s %s" % ({"net": "fetch"}.get(touches, touches), target), ("once", "refuse"), touches, target)

    def check_call(self, touches, target):
        """A skill's command (▶ `skills.py`): asked as a built-in command of the same `touches` is outside the allowed
        folders — `none` never, a read in 'ask', a write, a run or the network in 'ask' and 'chill', nothing in
        'full-auto' — this once, or refused."""
        if touches == "none" or (touches, target) in self.once:
            self.once.discard((touches, target))
            return
        if self.mode == "full-auto" or touches == "read" and self.mode == "chill":
            return
        raise Asked("%s %s" % ({"net": "fetch"}.get(touches, touches), target), ("once", "refuse"), touches, target)

    def decide(self, asked, choice):
        if choice == "folder":
            folder = asked.target if os.path.isdir(asked.target) else os.path.dirname(asked.target)
            if folder not in self.allowed:
                self.allowed.append(folder)
                self.save()
        elif choice == "once":
            self.once.add((asked.touches, asked.target))


# ── the library: definitions kept for every conversation ───────────────────────────────────────────────────────
# ⭐ A JSON FILE BESIDE tools.json — `{name: {"source": "(define …)", "about": "…", "kept": "…"}}`, every conversation's
#   and every model's: what a session defined and kept is there after a `(reset)`, a restart, in another conversation,
#   as the folders allowed are. `(import 'name)` evaluates its source in the session, the kept names it uses first; the
#   commands it calls are asked as they run, as any form's are.
LIBRARY_SOURCE_MAX = 16384       # the longest definition the library keeps


def _library_read(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        got = json.load(f)
    return got if isinstance(got, dict) else {}


def _library_write(path, lib):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(path + ".tmp", path)


def _names_in(form):
    """Every name a form mentions."""
    if isinstance(form, Text) or not isinstance(form, (str, list, tuple)):
        return set()
    if isinstance(form, str):
        return {str(form)}
    return set().union(*(_names_in(x) for x in form)) if form else set()


# ── the commands: what touches the machine ────────────────────────────────────────────────────────────────────────
class Command:
    """A command the model may call: its procedure, a line and a page about it, and what it touches (`read`, `write`,
    `run`, `net`, or None) — which the classifier will weigh (▶ the design's classifier)."""

    def __init__(self, name, fn, line, page, touches=None):
        self.name, self.fn, self.line, self.page, self.touches = name, fn, line, page, touches


def _names(names):
    """`a`, `a and b`, `a, b and c`."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _commands(session):
    def help_(what=None):
        if what is None:
            also = {}
            for alias, name in ALIASES.items():
                also.setdefault(name, []).append(alias)
            skilled = ["(%s) %s" % (c.name, c.line) for c in session.skill_commands.values()]
            return Text("\n".join("(%s) %s" % (c.name, c.line) for c in session.commands.values())
                        + ("\nSkills — (skills) says whose:\n" + "\n".join(skilled) if skilled else "")
                        + "\nAlso: " + "; ".join("%s %s %s" % (_names(a), "is" if len(a) == 1 else "are", n)
                                                 for n, a in also.items()) + "."
                        + "\nAnd Lisp: define, set!, let, lambda, if, begin, and, or, quote; + - * / = < >; list car cdr"
                        " cons length nth map filter append; string-append substring string=? contains? split join"
                        " lines number->string string->number. (help 'name) says more about a command.")
        name = ALIASES.get(str(what), str(what))
        c = session.commands.get(name) or session.skill_commands.get(name)
        if c is None:
            raise Fault("no command %s — (help) lists them" % write(what))
        said = "(%s) is (%s). " % (what, name) if name != str(what) else ""
        if name in session.skill_commands:
            said += "the %s skill's, touching %s: " % (c.skill, c.touches)
        return Text(said + "(%s) %s\n%s" % (c.name, c.line, c.page))

    def ls(path=Text(".")):
        p = session.path(path)
        if not os.path.isdir(p):
            raise Fault("%s is not a folder" % path)
        session.classifier.check("read", p)
        return [Text(n + "/" if os.path.isdir(os.path.join(p, n)) else n) for n in sorted(os.listdir(p))] or None

    def read_(path, start=1, count=READ_LINES):
        p = session.path(path)
        if not os.path.isfile(p):
            raise Fault("%s is not a file" % path)
        _num(start, count)
        session.classifier.check("read", p)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        part = lines[max(0, start - 1):max(0, start - 1) + count]
        more = len(lines) - (start - 1) - len(part)
        return Text("\n".join(part) + ("\n[… %d more lines: (read %s %d)]" % (more, to_text(Text(path)), start + count)
                                       if more > 0 else ""))

    def write_(path, text, how=None):
        p = session.path(path)
        _text(text)
        if how not in (None, "append") and str(how) != "append":
            raise Fault("(write \"path\" \"text\") replaces the file; (write \"path\" \"text\" 'append) adds to it")
        session.classifier.check("write", p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a" if how is not None else "w", encoding="utf-8") as f:
            f.write(text)
        return Text("%s: %d characters %s" % (path, len(text), "added" if how is not None else "written"))

    def bash(command, seconds=RUN_SECONDS):
        _text(command, "a command")
        _num(seconds)
        session.classifier.check("run", command)
        out, code, said = _run(shutil.which("bash") or "/bin/sh", command, session.cwd,
                               max(1, min(seconds, RUN_SECONDS_MAX)))
        text, note = out.decode("utf-8", errors="replace"), said or ("" if code == 0 else "\n[exit %d]" % code)
        keep = ANSWER_MAX // 2 - len(note) - 64       # cut here, not by the answer's cap: the note is kept, and a text
        if len(text) > keep:                           # written back may grow by its escapes
            text = text[:keep] + "\n[… %d more characters]" % (len(text) - keep)
        return Text(text + note)

    def fetch(url):
        _text(url, "a URL")
        if not url.startswith(("http://", "https://")):
            raise Fault("fetch reads http:// and https:// addresses, not %s" % url)
        session.classifier.check("net", url)
        req = urllib.request.Request(url, headers={"User-Agent": "SiLVANN-tool/1"})
        try:
            with urllib.request.urlopen(req, timeout=FETCH_SECONDS) as r:
                body, kind = r.read(FETCH_BYTES), r.headers.get_content_type()
                charset = r.headers.get_content_charset() or "utf-8"
        except OSError as ex:                          # urllib's errors are OSErrors: said, the session lives on
            raise Fault("%s: %s" % (url, ex))
        text = body.decode(charset, errors="replace")
        return Text(_page_text(text) if kind in ("text/html", "application/xhtml+xml") else text)

    def pwd():
        return Text(session.cwd)

    def cd(path):
        p = session.path(path)
        if not os.path.isdir(p):
            raise Fault("%s is not a folder" % path)
        session.cwd = os.path.realpath(p)
        return Text(session.cwd)

    def shelf():
        if not session.library:
            raise Fault("this REPL has no library")
        return session.library

    def keep(name, about=None):
        name = str(name)
        if name not in session.sources:
            raise Fault("%s was not defined in this session — (define …) it, then (keep '%s)" % (name, name))
        src = session.sources[name]
        if len(src) > LIBRARY_SOURCE_MAX:
            raise Fault("%s is %d characters; the library keeps definitions of %d at most"
                        % (name, len(src), LIBRARY_SOURCE_MAX))
        lib = _library_read(shelf())
        lib[name] = {"source": src, "about": "" if about is None else _text(about, "what it does"),
                     "kept": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _library_write(shelf(), lib)
        return Text("%s kept: (import '%s) brings it into any conversation's session" % (name, name))

    def library():
        lib = _library_read(shelf())
        if not lib:
            return Text("the library is empty: (keep 'name \"what it does\") keeps a definition")
        lines = []
        for name in sorted(lib):
            form = read(lib[name]["source"])
            head = source(form[1]) if len(form) > 1 and isinstance(form[1], list) else name
            lines.append(head + (" — " + lib[name]["about"] if lib[name].get("about") else ""))
        return Text("\n".join(lines))

    def import_(*names):
        lib, done, seen = _library_read(shelf()), [], set()

        def one(name):
            if name in seen:
                return
            seen.add(name)
            if name not in lib:
                raise Fault("%s is not in the library — (library) lists it" % name)
            form = read(lib[name]["source"])
            for used in sorted(_names_in(form) & set(lib) - {name}):     # the kept names it uses, first
                one(used)
            evaluate(form, session.env, session)
            done.append(Sym(name))
        for n in names:
            one(str(n))
        return done or None

    def forget(name):
        name, lib = str(name), _library_read(shelf())
        if name not in lib:
            raise Fault("%s is not in the library — (library) lists it" % name)
        del lib[name]
        _library_write(shelf(), lib)
        users = sorted(n for n in lib if name in _names_in(read(lib[n]["source"])))
        return Text("%s forgotten: the library no longer has it" % name
                    + ("; %s still use%s it" % (_names(users), "s" if len(users) == 1 else "") if users else ""))

    def skills_():
        if not session.skills and not session.skill_problems:
            return Text("no skills: a skill is a folder with a SKILL.md, in %s"
                        % (" or ".join(session.skill_folders) or "a skills folder this REPL was not given"))
        lines = []
        for sk in session.skills:
            lines.append("%s — %s" % (sk.name, sk.description or "(no description)"))
            lines += ["  %s   [%s]" % (c.usage, c.touches) for c in sk.commands
                      if getattr(session.skill_commands.get(c.name), "skill", None) == sk.name]
        return Text("\n".join(lines + ["⚠ " + p for p in session.skill_problems]))

    def guide():
        mine = session.skill_folders[0] if session.skill_folders else None
        try:
            text = open(session.guide, encoding="utf-8").read() if session.guide else None
        except OSError:
            text = None
        if text is None:
            return Text("no guide here: this REPL was not given docs/tools.md")
        head = ("Your skills folder on this machine: %s\n"
                "A command of your own is a folder there: (write \"%s/NAME/SKILL.md\" \"…\") and its .py or program "
                "beside it — a write outside the workspace, asked as one. Then (reset) reads it, (skills) says whether it "
                "was taken and (help 'command) shows its page.\n\n" % (mine, mine)) if mine else ""
        return Text(head + text)

    def image(prompt):
        try:
            return Text(session.image(_text(prompt, "a prompt")))
        except (Asked, Fault):
            raise
        except Exception as ex:                       # noqa: BLE001 — the picture's failure: an error the model reads
            raise Fault("(image …): %s" % ex)

    def whoami():
        if session.whoami is None:
            raise Fault("this REPL does not know which model it serves")
        return Text(session.whoami())

    def agent(model, config, task):
        if session.agent is None:
            raise Fault("this REPL launches no agents")
        link = session.agent(str(model), _text(config, "a config"), _text(task, "a task"))
        session.inside = link
        return Text(link.shown(link.first))

    def reset():
        session.fresh()
        return Text("a fresh session: the names it defined are gone and the working folder is %s again; the files, the "
                    "folders allowed, the mode and the library stay" % session.cwd)

    table = [
        Command("help", help_, "the commands, or one of them: (help 'read)", "(help) lists every command a line each;"
                " (help 'name) says one in full."),
        Command("ls", ls, "a folder's names, a folder's ending in /: (ls \"docs\")", "(ls) the working folder's; "
                "(ls \"path\") another's, relative to the working folder or absolute.", "read"),
        Command("read", read_, "a file's lines: (read \"path\" [first-line] [count])", "(read \"path\") the first %d "
                "lines; (read \"path\" 140 20) twenty from line 140. A file longer says how to read on." % READ_LINES,
                "read"),
        Command("write", write_, "a file written: (write \"path\" \"text\" ['append])", "(write \"path\" \"text\") "
                "makes or replaces the file; with 'append it adds to the end. Folders it needs are made.", "write"),
        Command("bash", bash, "a shell command run in the working folder: (bash \"ls -la\" [seconds])",
                "(bash \"command\") runs it with bash in the working folder and answers what it printed, its errors "
                "too, and [exit N] when it fails; it is stopped after %d seconds, or (bash \"command\" 300) after 300 "
                "(at most %d). Nothing is typed into it." % (RUN_SECONDS, RUN_SECONDS_MAX), "run"),
        Command("fetch", fetch, "a web page's text: (fetch \"https://…\")", "(fetch \"url\") reads an http or https "
                "address and answers its text — a page's words without its markup."
                " Read on in a long page with Lisp: (substring (fetch \"…\") 16000 32000).", "net"),
        Command("pwd", pwd, "the working folder", "(pwd) the folder relative paths start from."),
        Command("cd", cd, "change the working folder: (cd \"path\")", "(cd \"path\") relative or absolute; answers "
                "the folder it is now.", "read"),
        Command("reset", reset, "a fresh session: (reset)", "(reset) forgets every name the session defined and goes "
                "back to the working folder it started in; what is outside it stays — the files, the folders allowed, "
                "the mode, the library."),
        Command("skills", skills_, "the skills installed and their commands: (skills)", "(skills) lists every "
                "skill — a folder with a SKILL.md, read when the session begins and at (reset) — what it is for, its "
                "commands and what each touches, and what was left out and why. (help 'name) says one of their "
                "commands in full."),
        Command("guide", guide, "how to make a command of your own, a skill: (guide)", "(guide) answers the tool's "
                "guide — the REPL, the library, and a skill step by step: a Python function or a program, its SKILL.md, "
                "how values cross between Lisp and Python, what is asked — and where your skills folder is."),
        Command("keep", keep, "a definition kept for every conversation: (keep 'name [\"what it does\"])",
                "(keep 'name \"what it does\") puts the session's definition of name in the library, where it stays "
                "after (reset), a restart, in every conversation; kept again, it replaces the one there."),
        Command("library", library, "the definitions kept: (library)", "(library) lists the library, a definition a "
                "line — its name and arguments, and what it does."),
        Command("import", import_, "kept definitions into this session: (import 'name …)", "(import 'name) defines "
                "it in the session as it was kept, and first the kept names it uses; it answers the names defined."),
        Command("forget", forget, "a kept definition removed: (forget 'name)", "(forget 'name) takes it out of the "
                "library; a session that imported it keeps its own."),
    ]
    if session.whoami is not None:
        table.append(Command("whoami", whoami, "which model you are, and what (agent …) can reach: (whoami)",
                             "(whoami) answers which model you are and under which LoRA, the cards you run on, your "
                             "context and how much of it this conversation has used, your tool's mode — and the models "
                             "(agent …) can start: your own LoRAs ('self), each with what it is for and whether it "
                             "starts at once, the other models on this machine, and the workers'."))
    if session.agent is not None:                     # the server launches subagents
        table.append(Command("agent", agent, "a subagent's session: (agent 'model \"config\" \"task\")",
                             "(agent 'model \"config\" \"task\") starts a conversation with another model — or this "
                             "one under a LoRA — and answers its answer to the task. config is the name of one of the "
                             "model's LoRAs ((whoami) lists yours), as (agent 'self \"its-name\" \"task\"); or a "
                             "config's name in that model's configs/ (its LoRA, "
                             "and tools_allowed: the commands its REPL has; read-only ones by default); or a JSON "
                             "object as text. You are then inside its session: each call of yours "
                             "is a message to it, (answer 'once) or (answer 'refuse) answers what it asks, and (exit) "
                             "ends the session and brings you back. You can allow only what you could do unasked "
                             "yourself; anything more is asked of the user. Choose the model with its cost in mind: "
                             "'self — your own model, or under a LoRA it switches to — starts at once, while another "
                             "model that no worker serves, or your own under a LoRA it is loaded again for (a rank-1 "
                             "mask), is loaded on this machine in your place for each exchange, and you are loaded "
                             "back from the disk after: that can take several minutes each way. (whoami) lists your "
                             "LoRAs, what each is for and which start at once: prefer one of those when it suits the "
                             "task. A subagent starts with nothing of this conversation: to carry it on "
                             "under a LoRA, put what it needs — the conversation, or a summary of it — in the task."))
    if session.image is not None:                     # a picture model beside this one (the server's config)
        table.append(Command("image", image, "a picture from a prompt, shown to the user: (image \"a red apple on a "
                             "table\")", "(image \"prompt\") draws the picture the prompt describes with the picture "
                             "model and shows it to the user in the conversation; it answers what was made. Write the "
                             "prompt as a description of the picture, in detail. Settings go in it: --size 768x1024, "
                             "--steps 20, --seed 7, --count 4 (four pictures, seeds 7, 8, 9, 10 — drawn at once where "
                             "the machine has several cards). It takes minutes: the user sees its progress."))
    return {c.name: c for c in table}


# ── the skills: commands from folders outside the runtime (▶ skills.py) ──────────────────────────────────────────
def _to_python(v):
    """A REPL value as a skill's Python takes it: text and names as str, lists as lists, nil as None."""
    if isinstance(v, list):
        return [_to_python(x) for x in v]
    if isinstance(v, Closure) or callable(v):
        raise Fault("a procedure cannot be handed to a skill")
    return str(v) if isinstance(v, str) else v


def _to_repl(x):
    """What a skill's Python answered, as the REPL's values: a str as text, a dict as (key value) pairs, empty as
    nil."""
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return Text(x)
    if isinstance(x, dict):
        return [[Text(str(k)), _to_repl(v)] for k, v in x.items()] or None
    if isinstance(x, (list, tuple)):
        return [_to_repl(e) for e in x] or None
    return Text(str(x))


def _skill_command(session, skill, cmd):
    """The procedure a skill's command is: asked as its `touches` says, then its Python function or its program."""
    def call(*args):
        target = "%s: (%s)" % (skill.name, " ".join([cmd.name] + [write(a) for a in args]))
        session.classifier.check_call(cmd.touches, target)
        if ".py:" in cmd.run:
            try:
                fn = _skills.load_function(cmd, skill.folder, session.skill_cache)
            except Exception as ex:                       # noqa: BLE001 — a skill's file that does not load: said
                raise Fault("%s: %s: %s" % (target, type(ex).__name__, ex))
            kw = {}
            if _skills.takes_context(fn):
                kw["context"] = {"cwd": session.cwd, "workspace": session.home, "folder": skill.folder,
                                 "check": lambda touches, what: session.classifier.check(
                                     touches, session.path(Text(what)) if touches in ("read", "write") else what)}
            try:
                return _to_repl(fn(*[_to_python(a) for a in args], **kw))
            except (Asked, Fault):
                raise
            except Exception as ex:                       # noqa: BLE001 — what it raised: an error the model reads
                raise Fault("(%s …): %s: %s" % (cmd.name, type(ex).__name__, ex))
        argv = _skills.program_argv(cmd, skill.folder, [str(a) if isinstance(a, str) else write(a) for a in args])
        out, code, said = _run(shutil.which("bash") or "/bin/sh", argv, session.cwd, RUN_SECONDS)
        text = out.decode("utf-8", errors="replace")
        if code != 0 or said:
            raise Fault("(%s …) failed%s: %s" % (cmd.name, said.strip() if said else " [exit %d]" % code,
                                                 text.strip()[-600:] or "it printed nothing"))
        return Text(text)
    return call


# ── a session ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Session:
    """A conversation's REPL: its environment, its working folder, its classifier, the form an ask holds, and whether it
    was ended. The model's forms run as the model's (`run`); the user's — the answer to an ask — as the user's
    (`run(text, user=True)`), where `answer`, `allow` and `mode` are, which a model's form never reaches."""

    def __init__(self, cwd, classifier=None, library=None, skills=None, guide=None, image=None, agent=None,
                 allowed=None, whoami=None):
        self.home = self.cwd = os.path.realpath(cwd)
        self.whoami = whoami                           # the server's: () -> what the model is, here and now
        self.agent = agent                             # the server's subagents: (model, config, task) -> a session
        self.allowed = None if allowed is None else set(allowed)   # the commands this REPL has, or None: every one
        self.inside = None                             # the subagent's session the model is inside (▶ `_inside`)
        self.image = image                             # the server's picture: text -> the answer's text, or None
        self.guide = guide                             # the tool's guide, docs/tools.md (▶ `guide`), or None
        self.library = library                         # the library's file (▶ `keep`), or None
        self.skill_folders = [str(f) for f in (skills or ())]   # where skills are (▶ skills.py), the first's first
        self.classifier = classifier if classifier is not None else Classifier(allowed=[self.cwd])
        self.fresh()

    def fresh(self):
        """The session as it began — every name the model defined gone, the working folder the first, no ask waiting and
        nothing allowed once. What is outside the session stays: the files, the folders allowed (`tools.json`, every
        conversation's), the conversation's mode."""
        if getattr(self, "inside", None) is not None:  # a subagent's session left open ends with the session
            self.inside.close()
            self.inside = None
        self.cwd, self.pending = self.home, None       # (the form's text, its Asked) while the user is to answer
        self.sources = {}                              # a name defined -> its definition's source, for `keep`
        self.classifier.once = set()
        self.commands = _commands(self)
        self.env = Env()
        self.env.update(_builtins(self))
        for name, c in self.commands.items():
            self.env[name] = c.fn
        for alias, name in ALIASES.items():           # another Lisp's word for a command: the command itself
            self.env[alias] = self.commands[name].fn
        if self.allowed is not None:                  # a subagent's REPL: the commands its launch allowed, and Lisp
            for name in list(self.commands):
                if name not in self.allowed:
                    del self.commands[name]
                    self.env.pop(name, None)
            for alias, name in ALIASES.items():
                if name not in self.allowed:
                    self.env.pop(alias, None)
        # the skills, read afresh: a command takes a name nothing has, the first skill's first
        self.skills, self.skill_cache = _skills.read_all(self.skill_folders), {}
        self.skill_commands, self.skill_problems = {}, []
        taken = set(self.env) | SPECIAL | {"answer", "allow", "mode"}
        for sk in self.skills:
            self.skill_problems += ["%s: %s" % (sk.name, p) for p in sk.problems]
            for c in sk.commands:
                if c.name in taken:
                    self.skill_problems.append("%s: (%s) left out — the name is taken" % (sk.name, c.name))
                    continue
                if self.allowed is not None and c.name not in self.allowed:
                    continue
                taken.add(c.name)
                cmd = Command(c.name, _skill_command(self, sk, c), "%s: %s" % (_skills.first_sentence(c.page), c.usage),
                              c.page, c.touches)
                cmd.skill, cmd.usage = sk.name, c.usage
                self.skill_commands[c.name] = cmd
                self.env[c.name] = cmd.fn
        self.user_env = Env(self.env)
        self.user_env.update({"answer": self._answer, "allow": self._allow, "mode": self._mode})

    def path(self, p):
        p = _text(p, "a path")
        return os.path.realpath(os.path.join(self.cwd, os.path.expanduser(p)))

    # the user's side
    def _answer(self, choice):
        """(answer 'once) — the ask's form runs again, as the model's, with what the choice allows; 'refuse ends it."""
        if self.pending is None:
            raise Fault("nothing is waiting on an answer")
        text, asked = self.pending
        choice = str(choice)
        if choice not in asked.choices:
            raise Fault("the answer is one of %s" % " ".join(asked.choices))
        self.pending = None
        if choice == "refuse":
            return [Sym("error"), Text("the user refused: %s" % asked.question)]
        self.classifier.decide(asked, choice)
        return _Written(self.run(text))

    def _inside(self, text):
        """⭐ INSIDE A SUBAGENT'S SESSION (▶ `agent`): the model's call is a message to the subagent, as a person's line is
        in silvann_cli — `(exit)` ends the session, `(answer 'once)` / `(answer 'refuse)` answers its ask. Allowing is the
        model's only where it could do the same unasked itself: otherwise its allow is asked of the user (`Asked`), and the
        subagent is answered once the user allows it. `folder` is never the model's: it would allow a folder for every
        conversation."""
        link, t = self.inside, text.strip()
        if re.fullmatch(r"\((exit|reset)\)", t):
            self.inside = None
            link.close()
            return Text("the session with the agent %s is over: you are back in your own REPL" % link.name)
        m = re.fullmatch(r"\(answer\s+'?\s*(\w+)\s*\)", t)
        if m:
            choice = m.group(1)
            if link.ask is None:
                raise Fault("the agent asks nothing: a call is a message to it, (exit) ends the session")
            if choice not in ("once", "refuse"):
                raise Fault("you answer the agent's ask with 'once or 'refuse")
            if choice == "once":
                self.classifier.check(link.ask["touches"], link.ask["target"])   # the model's own authority, or the user's
            return Text(link.shown(link.answer(choice)))
        m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"', t, re.S)
        message = m.group(1).replace('\\"', '"').replace("\\n", "\n") if m else t
        try:                                          # (agent … "text") inside a session: the text, to the agent there
            form = read(t)
            if isinstance(form, list) and form and form[0] == "agent" and isinstance(form[-1], str):
                message = str(form[-1])
        except ComposerError:
            pass
        if link.ask is not None:
            raise Fault("the agent waits on your answer to its ask (%s): (answer 'once) or (answer 'refuse)"
                        % link.ask["question"])
        return Text(link.shown(link.say(message)))

    def _allow(self, folder):
        """(allow "folder") — reads there go unasked from now on (tools.json)."""
        p = self.path(folder)
        self.classifier.decide(Asked("", (), "read", p if os.path.isdir(p) else os.path.dirname(p)), "folder")
        return Text(p)

    def _mode(self, mode):
        """(mode 'chill) — the classifier's mode."""
        if str(mode) not in MODES:
            raise Fault("the mode is one of %s" % " ".join(MODES))
        self.classifier.mode = str(mode)
        return Sym(str(mode))

    def run(self, text, user=False):
        """A form's text -> its value written back: `(error "…")` for what goes wrong, `(ask "…" (choices…))` when the
        classifier stops a command (the form waits for the user's answer), at most ANSWER_MAX characters."""
        if self.pending is not None and not user:
            got = write([Sym("error"), Text("waiting on the user's answer: %s" % self.pending[1].question)])
            return got
        try:
            if self.inside is not None and not user:
                value = self._inside(text)
            else:
                value = evaluate(read(text), self.user_env if user else self.env, self)
            got = value.text if isinstance(value, _Written) else write(value)
        except Asked as ex:
            self.pending = (text, ex)
            got = write([Sym("ask"), Text(ex.question), [Sym(c) for c in ex.choices]])
        except ComposerError as ex:
            got = write([Sym("error"), Text("the form does not read: %s" % ex)])
        except Fault as ex:
            got = write([Sym("error"), Text(str(ex))])
        except RecursionError:
            got = write([Sym("error"), Text("the form recursed too deep")])
        except Exception as ex:                           # noqa: BLE001 — a builtin's arity, a file it may not open:
            got = write([Sym("error"), Text("%s: %s" % (type(ex).__name__, ex))])   # said, and the session lives on
        if len(got) > ANSWER_MAX:
            got = got[:ANSWER_MAX] + "\n[… cut: %d more characters]" % (len(got) - ANSWER_MAX)
        return got


class _Written:
    """An answer already written (the ask's form, run again), handed back as it is."""

    def __init__(self, text):
        self.text = text
