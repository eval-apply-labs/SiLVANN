"""THE MODEL'S SKILLS — commands for its REPL (▶ `tools.py`) that live outside the runtime, a folder each, read when a
session begins and again at its `(reset)`.

⚖ 2026-10-16: *"it also maps to the skills.md pipeline, what a skills.md is here is a series of commands with their
explaination, and those commands can be links to python scripts or c code or whatever is needed (most likely python as
that requires no recomp and tool calling is not as ms sensitive as the token generation steps)"*.

A skill is a folder with a `SKILL.md`: a front matter naming it and saying what it is for, a text about it, then a
heading a command — the command as it is called, its `touches` and `run` lines, and its explanation, which `(help
'name)` answers:
```
  ---
  name: files
  description: find files by name under a folder
  ---
  What the skill is for, for whoever reads the file.

  ## (find "folder" "pattern")
  touches: read
  run: files.py:find
  The names under the folder, and under its folders, that match the pattern — `*.txt`, `notes*` — a path a line.
```
  touches   what the command does to the machine, so it is asked as a built-in command would be outside the allowed
            folders: `none` never asks; `read` asks in 'ask'; `write`, `run` and `net` ask in 'ask' and 'chill';
            'full-auto' asks nothing. A command that says nothing is asked as `run`.
  run       `file.py:function` — a Python function in the skill's folder, called in the server's process with the
            form's arguments (texts as str, numbers, lists, nil as None) and, when it takes one, `context` (its working
            folder `cwd`, the `workspace`, the skill's `folder`, and `check(touches, target)`, the classifier's own ask
            for a path a command chooses); what it answers comes back as the REPL's values (str as text, a dict as
            (key value) pairs), and what it raises as an error.
            `program` — a file in the skill's folder run as a command (a `.py` by this Python, a `.sh` by bash, a
            `.ps1` by PowerShell, any other as it is), its arguments the form's as texts, in the session's working folder, stopped after a
            command's time: what it prints is the answer, and a failing exit an error.
A skill's commands never take a built-in's name, nor one an earlier skill took: `(skills)` says which were left out,
and why. A skill is code the user installs: it runs as the server's user, as the REPL's commands do.

The front matter may also say, for the machine's tools file (▶ `tools_config.py`, NN-127):
```
  enabled: auto                                   the entry a new install gets: yes (the default), no, or auto — on
                                                  where its check finds what it needs
  check: search.py:available                      a function called with the entry's `config` as the server starts:
                                                  None where the skill can answer, else why not — said in (help)
  config: {"searxng": "auto", "brave_key": null}  its settings, as JSON: the entry's `config` until the user sets them;
                                                  a Python command gets them in `context["config"]`, a program in the
                                                  environment's SILVANN_SKILL_CONFIG — the model never sees them
```
"""
import importlib.util
import inspect
import json
import os
import re
import sys

TOUCHES = ("none", "read", "write", "run", "net")
_COMMAND = re.compile(r"^##\s+\((\S+?)(?:\s+(.*))?\)\s*$")


class Command:
    """A skill's command as its SKILL.md declares it."""

    def __init__(self, skill, name, usage, touches, run, page):
        self.skill, self.name, self.usage, self.touches, self.run, self.page = skill, name, usage, touches, run, page


class Skill:
    """A skill's folder read: its name, what it is for, its commands, and what was wrong with the file."""

    def __init__(self, folder):
        self.folder, self.name, self.description = folder, os.path.basename(folder), ""
        self.commands, self.problems = [], []
        self.enabled, self.check, self.config = "yes", "", {}   # its tools file entry's defaults, and its startup check


def read_skill(folder):
    """A skill's folder -> a `Skill`, its problems listed rather than raised."""
    s = Skill(folder)
    path = os.path.join(folder, "SKILL.md")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as ex:
        s.problems.append("no SKILL.md: %s" % ex.strerror)
        return s
    lines = text.split("\n")
    at = 0
    if lines and lines[0].strip() == "---":                      # the front matter
        for at in range(1, len(lines)):
            if lines[at].strip() == "---":
                break
            key, _, value = lines[at].partition(":")
            key, value = key.strip(), value.strip()
            if key == "name" and value:
                s.name = value
            elif key == "description":
                s.description = value
            elif key == "enabled":
                if value in ("yes", "no", "auto"):
                    s.enabled = value
                else:
                    s.problems.append("enabled is yes, no or auto, not %r" % value)
            elif key == "check":
                s.check = value
            elif key == "config":
                try:
                    s.config = json.loads(value)
                except ValueError as ex:
                    s.problems.append("config is not JSON: %s" % ex)
                if not isinstance(s.config, dict):
                    s.problems.append("config is a JSON object, its settings by name")
                    s.config = {}
        at += 1
    current = None
    for line in lines[at:]:
        m = _COMMAND.match(line.strip())
        if m:
            current = {"name": m.group(1), "usage": line.strip()[3:].strip(), "keys": {}, "page": []}
            s.commands.append(current)
            continue
        if line.startswith("## "):                             # a heading that is not a command ends the one before
            current = None
            continue
        if current is None:
            continue
        key, sep, value = line.partition(":")
        if sep and not current["page"] and key.strip() in ("touches", "run"):
            current["keys"][key.strip()] = value.strip()
        else:
            current["page"].append(line)
    commands = []
    for c in s.commands:
        touches = c["keys"].get("touches", "run")
        run = c["keys"].get("run", "")
        if touches not in TOUCHES:
            s.problems.append("(%s): touches is one of %s, not %r" % (c["name"], " ".join(TOUCHES), touches))
            continue
        if not run:
            s.problems.append("(%s): no run line" % c["name"])
            continue
        if not os.path.isfile(os.path.join(folder, run.split(":", 1)[0] if ".py:" in run else run)):
            s.problems.append("(%s): %s is not in the skill's folder" % (c["name"], run))
            continue
        page = "\n".join(c["page"]).strip()
        commands.append(Command(s.name, c["name"], c["usage"], touches, run, page))
    s.commands = commands
    if not commands and not s.problems:
        s.problems.append("no command: a command is a heading `## (name args…)`")
    return s


def read_all(folders):
    """Every skill in `folders`, in their order, a folder's skills by name — the first of a name wins."""
    skills, seen = [], set()
    for root in folders or ():
        if not root or not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            folder = os.path.join(root, entry)
            if not os.path.isfile(os.path.join(folder, "SKILL.md")):
                continue
            s = read_skill(folder)
            if s.name in seen:
                continue
            seen.add(s.name)
            skills.append(s)
    return skills


def run_check(skill, config, cache):
    """A skill's startup check (`check: file.py:function`) called with its settings: None where it can answer, else why
    not — what it raised, said as the reason."""
    if not skill.check:
        return None
    try:
        fn = load_function(Command(skill.name, "check", "", "none", skill.check, ""), skill.folder, cache)
        why = fn(config)
    except Exception as ex:                           # noqa: BLE001 — a check that fails says so; the server goes on
        return "its check failed: %s: %s" % (type(ex).__name__, ex)
    return None if not why else str(why)


def first_sentence(page, most=110):
    one = re.split(r"(?<=[.:;])\s", " ".join(page.split()), maxsplit=1)[0] if page else ""
    return one if len(one) <= most else one[:most - 1] + "…"


# ── the calls ─────────────────────────────────────────────────────────────────────────────────────────────────────
def load_function(cmd, folder, cache):
    """The Python function a command names, its module read once a session (`cache`)."""
    file, _, name = cmd.run.partition(":")
    path = os.path.join(folder, file)
    module = cache.get(path)
    if module is None:
        spec = importlib.util.spec_from_file_location("silvann_skill_%s_%s" % (re.sub(r"\W", "_", cmd.skill),
                                                                              re.sub(r"\W", "_", file[:-3])), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cache[path] = module
    fn = getattr(module, name, None)
    if not callable(fn):
        raise LookupError("%s has no function %s" % (file, name))
    return fn


def takes_context(fn):
    try:
        return "context" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def program_argv(cmd, folder, args, bash="bash", powershell="powershell"):
    """A program's command as a list: a `.py` by this Python, a `.sh` by `bash`, a `.ps1` by `powershell`, anything
    else run itself."""
    path = os.path.join(folder, cmd.run)
    if path.endswith(".py"):
        head = [sys.executable, path]
    elif path.endswith(".sh"):
        head = [bash, path]
    elif path.endswith(".ps1"):
        head = [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", path]
    else:
        head = [path]
    return head + [str(a) for a in args]
