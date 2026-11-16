#!/usr/bin/env python3
"""THE TOOLS REPL — the model's tool at a person's prompt: the session a conversation's model calls through its one tool
(`lisp`), its commands, skills and kept definitions, with no model — so a person runs the forms a model runs, tries a
tool's script before a model does, and uses the scripts written for the models to do the same things themselves.

⚖ 2026-11-15, the architect: *"alongside that i would add a silvann_tools_repl.py, so that a user can test the tool
calling scripts by himself / act as if he was a repl, so humans can recycle the same scripts used by the model to do
things"*.

    silvann_tools_repl                         a prompt in a terminal; from a pipe, a line for each form it is sent
    silvann_tools_repl FILE … [-i]             each file's forms in order, its last answer written; -i, a prompt after
    silvann_tools_repl -e FORM [-e FORM …]     each form's answer, a line each
    venv/bin/python silvann_tools_repl.py …    the same, from the release's folder

The session is the one the server opens for a conversation of the machine's (▶ start_silvann_server.py's
`_tool_session`), over the same files: the workspace (`$SILVANN_DATA/workspace`, where it starts and reads unasked), the
folders allowed (`tools.json`), the definitions kept (`tools_library.json`), the tools file (`tools/default.json`, or
`--tools-config`) and the skills (`skills/`, the release's and the user's). What a person keeps or allows here, every
conversation has. Each form runs as the model's would, and its answer is written as the model reads it — one line, a
text's line breaks written `\\n` — except at a prompt, where an answer that is a text is written as its text.
⛳ WHAT THE MODEL WOULD ASK ITS USER IS ASKED HERE (`--mode`, "ask" by default, as the chat's): at a prompt, a line —
o once, f the folder from now on, Enter refuses; from a pipe the ask is the answer, `(ask "…" (once folder refuse))`,
and the program answers it with `(answer 'once)`. `(answer …)`, `(allow "folder")` and `(mode 'chill)` run as the
user's, as in the chat; every other form as the model's. A picture `(ui 'look)` or `(browser 'look)` shows is written
to a file, and stderr says where. With no model here, `(image …)` and `(agent …)` answer an error.
A line that starts with `,` is this program's: `,help`, `,quit`. Ctrl-C at the prompt drops the form being typed; while
a form runs, it ends it, `(error "interrupted")`.
A batch — the files and the `-e` forms — stops at the first answer that is not a value: an error, or an ask nobody is
there to answer.

  exit: 0 · 1 the last answer an error, or an ask unanswered · 2 text that does not read · 130 left by Ctrl-C
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# the runtime beside this file in a release, or the development tree's
_HERE = ROOT / "runtime" if (ROOT / "runtime").is_dir() else ROOT.parent / "python"
sys.path.insert(0, os.environ.get("SILVANN_RUNTIME", str(_HERE)))
from silvann_runtime import tools, tools_config                      # noqa: E402
from silvann_runtime.composer import ComposerError, Text, read, read_all  # noqa: E402
from silvann_runtime.prompt import texts                             # noqa: E402

# the server's places (▶ start_silvann_server.py): the skills the release ships, the tool's guide, and the data — the
#   folder SILVANN_DATA names, else the one `silvann data` named (data_folder), else this one
SKILLS_SHIPPED = ROOT / "skills" if (ROOT / "skills").is_dir() else ROOT.parent / "release" / "skills"
TOOLS_GUIDE = ROOT / "docs" / "tools.md" if (ROOT / "docs").is_dir() else ROOT.parent / "release" / "docs" / "tools.md"
_NAMED = ROOT / "data_folder"
DATA = Path(os.environ.get("SILVANN_DATA") or
            (_NAMED.read_text(encoding="utf-8").strip() if _NAMED.is_file() and _NAMED.stat().st_size else "") or ROOT)

VALUE, ERROR, REFUSED = 0, 1, 2
DIM, RED, RESET = "\033[2m", "\033[31m", "\033[0m"
USERS_WORDS = ("answer", "allow", "mode")          # the user's in the chat, never the model's (▶ tools.Session.run)
HELP = ("the model's commands: (help) lists them, (help 'read) says one — (guide) the tool's guide; the user's, as in "
        "the chat: (answer 'once) (allow \"folder\") (mode 'chill) — this program's: ,help ,quit")


def one_line(text):
    return " ".join(str(text).split())


def no_image(text):
    raise tools.Fault("(image …) draws with the server's picture model, and silvann_tools_repl runs no model")


class ToolsRepl:
    def __init__(self, a):
        self.a = a
        self.status = VALUE
        self.tty_in, self.tty_out = sys.stdin.isatty(), sys.stdout.isatty()
        self.pictures_dir, self.pictures = None, 0
        path = Path(a.tools_config) if os.path.isabs(a.tools_config) or a.tools_config.endswith(".json") \
            else DATA / "tools" / (a.tools_config + ".json")
        tools_file = tools_config.ToolsFile(path)
        wrong = tools_file.problems()
        if wrong:
            raise ValueError("; ".join(wrong))
        workspace = DATA / "workspace"
        for place in (workspace, DATA / "skills", DATA / "skills_added"):
            place.mkdir(parents=True, exist_ok=True)
        box = tools_config.Toolbox(tools_file, [DATA / "skills", SKILLS_SHIPPED], added=DATA / "skills_added",
                                   code=[DATA / "python"])
        box.refresh()
        web = tools.Web()
        web.named(box.config("fetch").get("contact"))
        classifier = tools.Classifier(str(DATA / "tools.json"), a.mode, allowed=[str(workspace)],
                                      sealed=[box.file.folder()])
        self.session = tools.Session(str(workspace), classifier.conversation(a.mode),
                                     library=str(DATA / "tools_library.json"), guide=str(TOOLS_GUIDE),
                                     image=no_image, agent=None, whoami=lambda: "a person at silvann_tools_repl, no "
                                     "model: the forms are theirs", web=web, toolbox=box, sees=lambda: True)

    # ── a form ────────────────────────────────────────────────────────────────────────────────────────────────────────
    def run(self, node):
        """One form, as the model's or — `answer`, `allow`, `mode` — the user's: (its answer, status), an ask answered at
        the prompt when there is one."""
        user = isinstance(node, list) and bool(node) and node[0] in USERS_WORDS
        try:
            got = self.session.run(tools.source(node), user=user)
            while self.session.pending is not None and self.tty_in:
                got = self.session.run("(answer '%s)" % self.decide(self.session.pending[1]), user=True)
        except KeyboardInterrupt:
            sys.stderr.write("\n")
            got = tools.write([tools.Sym("error"), Text("interrupted")])
        self.keep_pictures()
        if self.session.pending is not None:
            return got, ERROR
        return got, ERROR if got.startswith("(error ") else VALUE

    def decide(self, asked):
        """The ask, on a line: its first letters choose — o once, f the folder — and Enter or anything else refuses."""
        keys = {c[0]: c for c in asked.choices if c != "refuse"}
        said = " · ".join("%s %s" % (k, c) for k, c in keys.items())
        try:
            line = input("\001%s\002[silvann] the tool asks: %s — %s · Enter refuses: \001%s\002"
                         % (DIM, asked.question, said, RESET))
        except (EOFError, KeyboardInterrupt):
            line = ""
        return keys.get(line.strip()[:1].lower(), "refuse")

    def keep_pictures(self):
        for png in self.session.take_pictures():
            if self.pictures_dir is None:
                self.pictures_dir = tempfile.mkdtemp(prefix="silvann_tools_repl_")
            self.pictures += 1
            path = os.path.join(self.pictures_dir, "look_%d.png" % self.pictures)
            with open(path, "wb") as f:
                f.write(png)
            sys.stderr.write("[silvann] the picture: %s\n" % path)

    def forms(self, text, write_each=True):
        try:
            nodes = read_all(text)
        except ComposerError as ex:
            return self.write(tools.write([tools.Sym("error"), Text("the text does not read: %s" % ex)]), REFUSED)
        for i, node in enumerate(nodes):
            got, status = self.run(node)
            if write_each or status != VALUE or i == len(nodes) - 1:
                self.write(got, status)
            if status != VALUE:
                return status
        return VALUE

    def write(self, got, status):
        if self.tty_out:
            try:
                value = read(got)
            except ComposerError:
                value = None
            if isinstance(value, Text):              # a text, as its text — what a person reads
                got = str(value)
            elif status != VALUE:
                got = RED + got + RESET
        else:
            got = one_line(got) if "\n" in got else got   # an answer cut long says so after a line break
        print(got, flush=True)
        self.status = status
        return status

    def loop(self):
        if self.tty_in:
            sys.stderr.write("%s[silvann] the model's tool, at your prompt — %s, the mode %s · (help) · ,help · ,quit%s\n"
                             % (DIM, self.session.home, self.session.classifier.mode, RESET))
        for kind, text in texts("tools"):
            if kind == "command":
                word = text[1:].strip().split(" ")[0]
                if word in ("quit", "exit", "q"):
                    return
                if word == "help":
                    self.write(HELP, VALUE)
                else:
                    self.write(tools.write([tools.Sym("error"), Text("no command ,%s: ,help says them" % word)]), ERROR)
            elif kind == "forms":
                self.forms(text)
            else:
                self.write(tools.write([tools.Sym("error"), Text("the input ends inside a form")]), REFUSED)


def main():
    ap = argparse.ArgumentParser(description="The model's tool at your prompt: its commands and skills, with no model.")
    ap.add_argument("files", nargs="*", help="files of forms run in order; each file's last answer is written")
    ap.add_argument("-e", "--eval", dest="forms", action="append", default=[], metavar="FORM",
                    help="a form to run, its answer written (again for more)")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="after the files and the -e forms, read forms as they come — a prompt, or a pipe")
    ap.add_argument("--mode", choices=tools.MODES, default="ask",
                    help="what the commands may do unasked, as the chat's mode (default: ask)")
    ap.add_argument("--tools-config", dest="tools_config", default="default",
                    help="the tools file: a name in the data folder's tools/, or a path (default: default)")
    a = ap.parse_args()
    try:
        repl = ToolsRepl(a)
    except (OSError, ValueError) as ex:
        sys.stderr.write("[silvann] the tools are not set up: %s\n" % one_line(ex))
        sys.exit(REFUSED)
    try:
        going = True
        for path in a.files:                      # a file answers once: its last form's answer
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as ex:
                repl.write(tools.write([tools.Sym("error"), Text(one_line(ex))]), REFUSED)
                going = False
            going = going and repl.forms(text, write_each=False) == VALUE
            if not going:
                break
        for form in a.forms if going else ():
            if repl.forms(form) != VALUE:
                going = False
                break
        if going and (a.interactive or not (a.files or a.forms)):
            repl.loop()
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        repl.status = 130
    except BrokenPipeError:                       # whoever read the answers has gone
        os._exit(repl.status)
    sys.exit(repl.status)


if __name__ == "__main__":
    main()
