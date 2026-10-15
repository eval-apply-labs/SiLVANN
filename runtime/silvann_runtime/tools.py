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
  (read "docs/users.md" 140 3) -> "…three lines…"
  (exit)                       -> the session ends
```
The forms are read by the composer's reader (`composer.read`: names, numbers, `"text"`, quotes) and evaluated here:
`quote` `if` `define` `set!` `let` `lambda` `begin` `and` `or`; numbers, text, lists, `#t`, `#f` and `nil`. A value
is written back in the reader's syntax, so the model reads what it would write; an error is the value `(error "…")`.
"""
import html.parser
import os
import shutil
import signal
import subprocess
import threading
import urllib.request

from .composer import ComposerError, Text, read, to_text

ANSWER_MAX = 16384               # the most characters an answer carries back; the rest is said to be cut
READ_LINES = 200                 # the lines `read` answers when it is not told how many
RUN_SECONDS = 60                 # a command's time unless `bash` is told another, at most RUN_SECONDS_MAX
RUN_SECONDS_MAX = 600
RUN_BYTES = 1 << 20              # a command printing more is stopped there
FETCH_SECONDS = 30
FETCH_BYTES = 4 << 20            # the most of a page `fetch` reads


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


# ── the evaluator ─────────────────────────────────────────────────────────────────────────────────────────────────
CONSTANTS = {"#t": True, "#f": False, "nil": None}


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
        if head == "define":
            if len(x) >= 3 and isinstance(x[1], list) and x[1]:       # (define (f a b) body…)
                env[x[1][0]] = Closure(x[1][1:], x[2:], env)
                return Sym(x[1][0])
            if len(x) != 3 or not isinstance(x[1], str):
                raise Fault("(define name value) or (define (name args…) body…)")
            env[x[1]] = evaluate(x[2], env, session)
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

    def decide(self, asked, choice):
        if choice == "folder":
            folder = asked.target if os.path.isdir(asked.target) else os.path.dirname(asked.target)
            if folder not in self.allowed:
                self.allowed.append(folder)
                self.save()
        elif choice == "once":
            self.once.add((asked.touches, asked.target))


# ── the commands: what touches the machine ────────────────────────────────────────────────────────────────────────
class Command:
    """A command the model may call: its procedure, a line and a page about it, and what it touches (`read`, `write`,
    `run`, `net`, or None) — which the classifier will weigh (▶ the design's classifier)."""

    def __init__(self, name, fn, line, page, touches=None):
        self.name, self.fn, self.line, self.page, self.touches = name, fn, line, page, touches


def _commands(session):
    def help_(what=None):
        if what is None:
            return Text("\n".join("(%s) %s" % (c.name, c.line) for c in session.commands.values())
                        + "\nAnd Lisp: define, set!, let, lambda, if, begin, and, or, quote; + - * / = < >; list car cdr"
                        " cons length nth map filter append; string-append substring string=? contains? split join"
                        " lines number->string string->number. (help 'name) says more about a command.")
        c = session.commands.get(str(what))
        if c is None:
            raise Fault("no command %s — (help) lists them" % write(what))
        return Text("(%s) %s\n%s" % (c.name, c.line, c.page))

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

    def exit_():
        session.closed = True
        return Sym("bye")

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
        Command("exit", exit_, "end the session", "(exit) ends the session — as an answer without a call does."),
    ]
    return {c.name: c for c in table}


# ── a session ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Session:
    """A conversation's REPL: its environment, its working folder, its classifier, the form an ask holds, and whether it
    was ended. The model's forms run as the model's (`run`); the user's — the answer to an ask — as the user's
    (`run(text, user=True)`), where `answer`, `allow` and `mode` are, which a model's form never reaches."""

    def __init__(self, cwd, classifier=None):
        self.cwd = os.path.realpath(cwd)
        self.closed = False
        self.classifier = classifier if classifier is not None else Classifier(allowed=[self.cwd])
        self.pending = None                            # (the form's text, its Asked) while the user is to answer
        self.commands = _commands(self)
        self.env = Env()
        self.env.update(_builtins(self))
        for name, c in self.commands.items():
            self.env[name] = c.fn
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
