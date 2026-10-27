"""THE TOOLS FILE — what a model's REPL can reach on this machine, set up in one place (NN-127).

⚖ 2026-10-26: *"have the startup engine have a tools section, so it can be pre-populated. it can be a separate thing from
the config, it can be called tools and point at them as a config, and we ignore those that start with _"* · *"each enabled
no simply rebinds the function as a lisp program that throws an error saying "command" is disabled, and placing the
passwords here means they are not read by the llm but passed as arguments … so python gets it but the llm is oblivious
to it"* · *"installing a new skill would add an entry here so you only set up things, never add or remove them"*.

The data folder's `tools/default.json` — a model's config names another with `tools_config` — an entry a tool:
```
  "search": {"enabled": "auto", "config": {"searxng": "auto", "brave_key": null}, "_about": "…"}
```
  enabled   "yes"; "no", and the command answers that it is disabled; or "auto", on where its startup check finds what
            it needs
  config    its settings, handed to the tool and never to the model — a key, a password, an address
  _…        a key starting with `_` is skipped: the file's `_note`, an entry's `_about` (what the tool is, the server's to
            write), `_gone` (the entries of tools no longer installed)
The server keeps the entries — one for every built-in command a machine may not have (`BUILT_IN`) and every skill
installed — as it starts, on the UI's "Rebuild the tools", and where a session's `(reset)` finds a skill installed since:
a tool with none gets one with its defaults; one there keeps its switch and its settings and gains the settings its tool
has added since; the entry of a tool no longer installed is set aside under `_gone`, from where it comes back with its
settings if the tool does. So whoever runs the machine sets the entries up and never adds or removes one.
`SILVANN_TOOL_<NAME>_<SETTING>` in the environment overrides a setting and `SILVANN_TOOL_<NAME>_ENABLED` the switch — a
container's way, with no file to edit.

⭐ ONLY THE SERVER'S PYTHON READS THE FILE (`Toolbox`, ⚖ 2026-10-26 *"make it so that help and the repl go through python
instead of going through the config, so the model never needs to read the file itself … help can query the python for
the list of commands and python can generate the response on the spot with what's available, what's off and what asks
for permission"*): it reads it as the server starts and on "Rebuild the tools", and every session asks it, at each call
and each (help), what is installed, what is on and with which settings.
"""
import json
import os
import re
import threading

from . import skills as _skills
from . import system as _system

SWITCHES = ("yes", "no", "auto")
DEFAULT_REPO = "https://github.com/eval-apply-labs/silvann_agentic_skills"   # the project's own repository of tools


def built_in():
    """⭐ THE BUILT-IN COMMANDS A MACHINE MAY NOT HAVE, their entries' defaults by the system the server runs on, and what
    they are; a skill's come from its SKILL.md (▶ skills.py). ⚖ 2026-10-26 *"i would prefer for safety reasons to set
    powershell and bash as off on windows, then it is the user who did it if he wants them on. i think the python can
    decide which os is in and set up the default accordingly"*: on Windows the shell and `add_tool` start switched off."""
    on = "no" if _system.WINDOWS else "yes"
    return {
        "image": {"enabled": "auto", "config": {"model": "auto", "card": None, "host": "local"},
                  "about": "(image \"prompt\"): a picture drawn by a picture model. model: a folder in models/, or "
                           "\"auto\" the first picture model there; card: its own card, or null the text model's, put "
                           "down while it draws; host: \"local\", or another SiLVANN server serving a picture model, "
                           "\"http://host:port\""},
        "bash": {"enabled": on, "config": {"confined": "auto"},
                 "about": "(bash \"command\"), and (powershell \"command\") on Windows: a shell command, as the "
                          "server's user — on Windows switched off until you switch it on. confined: \"auto\" — in a "
                          "container, the container but this file's folder and the server's own process; on a direct "
                          "install, a developer's machine, all of it; \"machine\" the container's way anywhere; "
                          "\"workspace\" the REPL's workspace alone, and what a link at its top points to — ln -s "
                          "/a/folder <workspace>/name, made from a console, is the model's way to that folder; \"no\" "
                          "all of it. Confined needs Linux's Landlock (5.13 or later): where it is not, the command is "
                          "refused (▶ docs/tools.md)"},
        "add_tool": {"enabled": on, "config": {"ask": "mode"},
                     "about": "(add_tool \"path/to/name.py\") and (remove_tool \"name\"): the model adds a tool it "
                              "wrote in one .py — its DOC, its CONFIG, its functions — to the tools added "
                              "(skills_added/), and this file gains its entry, name, whose config you set; or takes one "
                              "it added away. A tool runs as the model's (bash …) does, and is asked as it is. ask: "
                              "\"mode\" — adding, replacing and removing asked of a person as the conversation's mode "
                              "asks a write, so not in full-auto — or \"always\", a person asked in every mode. \"no\" "
                              "where "
                              "the server serves several people, or the model should add nothing; on Windows switched "
                              "off until you switch it on"},
        "repos": {"enabled": "yes", "config": {"sources": [DEFAULT_REPO]},
                  "about": "(list_repo), (list_package \"name\"), (get_tool \"package/tool\") and (add_repo \"address\"): "
                           "tools others wrote, in repositories of packages — the model lists them, downloads one into "
                           "its workspace to read, and adds it with (add_tool …). sources: the repositories read, each "
                           "an https:// git address (the server clones it; git's credentials answer for a private one) "
                           "or a folder; (add_repo …) adds one here (▶ docs/tools.md)"},
        "agent": {"enabled": "yes", "config": {"registry": None, "key": None, "offer": "yes", "address": "auto"},
                  "about": "(agent 'model \"config\" \"task\"): a subagent — this model, another on this machine, or "
                           "one another SiLVANN server serves; \"no\" switches it off. registry: a registry of SiLVANN "
                           "servers (silvann registry), \"http://host:8770\", or null none — (agent …) and (whoami) "
                           "reach the servers it lists, beside the config's workers; key: the registry's key, never "
                           "shown to the model; offer: \"yes\", this server offers itself there while it runs, or "
                           "\"no\", it only reads the listing; address: where the others reach this server — \"auto\" "
                           "from the address it listens on (with --host 0.0.0.0, the one its route to the registry "
                           "leaves from), or \"http://host:port\", a container's: its machine's address and the port "
                           "published (▶ docs/tools.md)"},
        "fetch": {"enabled": "yes", "config": {"contact": None},
                  "about": "(fetch \"url\"): a web page's words. contact: who a site writes to about the fetches, in "
                           "their User-Agent — an e-mail address or a page; null the project's page"},
    }


BUILT_IN = built_in()                 # the built-ins' names (their defaults: `built_in()`, asked when they are used)
NOTE = ("What the model's REPL can reach on this machine, an entry a tool. The server keeps the entries — it adds one for "
        "each tool and skill installed as it starts, at a session's (reset) and on the UI's \"Rebuild the tools\" — and "
        "you set them up: enabled \"yes\", \"no\" or \"auto\" (on where the server finds what the tool needs), and "
        "config, the tool's settings, handed to the tool and never shown to the model. A key starting with _ is "
        "ignored. docs/tools.md says what each tool takes.")


def env_name(*parts):
    """`SILVANN_TOOL_SEARCH_BRAVE_KEY` for ("search", "brave_key")."""
    return "SILVANN_TOOL_" + "_".join(re.sub(r"\W", "_", p).upper() for p in parts)


def _env_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def tools_of(skills):
    """What `sync` keeps entries for: every built-in of `BUILT_IN` and every skill read (▶ skills.read_all)."""
    tools = {name: dict(t) for name, t in built_in().items()}
    for s in skills:
        tools.setdefault(s.name, {"enabled": s.enabled, "config": dict(s.config),
                                  "about": "the %s skill: %s" % (s.name, s.description or "(no description)")})
    return tools


class ToolsFile:
    """A machine's tools file: its entries read, and kept in step with what is installed. None of it is the model's: the
    REPL reads and writes no file of its folder (▶ tools.Classifier's `sealed`), and what a session knows of it, the
    server's `Toolbox` tells it."""

    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.Lock()

    def folder(self):
        return os.path.dirname(os.path.abspath(self.path))

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                got = json.load(f)
        except FileNotFoundError:
            return {}
        if not isinstance(got, dict):
            raise ValueError("%s is a JSON object, an entry a tool" % self.path)
        return got

    def entries(self):
        """Every tool's entry, the keys starting with `_` left out."""
        return {k: v for k, v in self.read().items() if not k.startswith("_") and isinstance(v, dict)}

    def problems(self):
        """What the file says that its reader cannot take, a line each — the server refuses to start on one."""
        out = []
        for name, e in self.entries().items():
            if str(e.get("enabled", "yes")).lower() not in SWITCHES:
                out.append("%s: %s's enabled is \"yes\", \"no\" or \"auto\", not %r"
                           % (self.path, name, e.get("enabled")))
            if not isinstance(e.get("config", {}), dict):
                out.append("%s: %s's config is a JSON object, its settings by name" % (self.path, name))
        return out

    def set_setting(self, name, key, value):
        """A tool's setting written into its entry — the server's Python's, as (add_repo …) does; the file kept whole."""
        with self.lock:
            got = self.read()
            entry = got.setdefault(name, {})
            if not isinstance(entry.get("config"), dict):
                entry["config"] = {}
            entry["config"][key] = value
            os.makedirs(self.folder(), exist_ok=True)
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(got, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(self.path + ".tmp", self.path)

    def sync(self, tools):
        """The entries made to match `tools` — {name: {"enabled", "config", "about"}} — and the file written where that
        changed anything (▶ the module's note). Answers what it did, a line a change."""
        with self.lock:
            got = self.read()
            before = json.dumps(got, sort_keys=True)
            gone = got.get("_gone") if isinstance(got.get("_gone"), dict) else {}
            said, kept = [], {}
            for name in sorted(tools, key=lambda n: (n not in BUILT_IN, n)):
                t, e = tools[name], got.get(name)
                if not isinstance(e, dict):
                    e = gone.pop(name, None)
                    if isinstance(e, dict):
                        said.append("%s came back, its settings as they were" % name)
                    else:
                        e = {"enabled": t["enabled"], "config": dict(t["config"])}
                        said.append("%s added, enabled %s" % (name, t["enabled"]))
                config = e.get("config") if isinstance(e.get("config"), dict) else {}
                for key, value in t["config"].items():
                    if key not in config:
                        config[key] = value
                        said.append("%s gained the setting %s" % (name, key))
                kept[name] = {"enabled": e.get("enabled", t["enabled"]), "config": config, "_about": t["about"]}
            for name in sorted(k for k in got if not k.startswith("_") and k not in tools):
                gone[name] = got[name]
                said.append("%s set aside under _gone: no tool of that name is installed" % name)
            out = {"_note": NOTE}
            out.update(kept)
            out.update({k: v for k, v in got.items() if k.startswith("_") and k not in ("_note", "_gone")})
            if gone:
                out["_gone"] = gone
            if json.dumps(out, sort_keys=True) != before:
                os.makedirs(self.folder(), exist_ok=True)
                with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                os.replace(self.path + ".tmp", self.path)
            return said


def _switch(name, said, default):
    """A tool's switch: the environment's, else the file's, else its own default."""
    env = os.environ.get(env_name(name, "enabled"))
    got = str(env if env is not None else said if said is not None else default).lower()
    return got if got in SWITCHES else default


def _settings(name, defaults, mine):
    """A tool's settings: its defaults, the file's over them (no `_` keys), the environment's over those."""
    got = dict(defaults or {})
    if isinstance(mine, dict):
        got.update({k: v for k, v in mine.items() if not str(k).startswith("_")})
    for key in list(got):
        said = os.environ.get(env_name(name, key))
        if said is not None:
            got[key] = _env_value(said)
    return got


def config_of(tools_file, name):
    """A built-in's settings from the file and the environment, with no `Toolbox` — a picture model's server, which has
    no REPL, reads its `agent` entry so."""
    entry = tools_file.entries().get(name, {}) if tools_file is not None else {}
    return _settings(name, built_in()[name]["config"], entry.get("config"))


def _signature(skills):
    """What a session binds of the skills: a change in it binds them again."""
    return [(s.name, s.folder, s.enabled, [(c.name, c.usage, c.touches, c.run, c.page) for c in s.commands], s.problems)
            for s in skills]


class Toolbox:
    """⭐ THE MACHINE'S TOOLS AS THE SERVER HOLDS THEM — the skills installed, each tool's switch and settings as the tools
    file and the environment said when they were last read, and why a command cannot answer here (the startup check's,
    `checked`). Sessions ask it at each call and each (help) (▶ tools.Session), so a tool switched off on a rebuild is off
    in every conversation at once, and a skill installed is in each conversation's (help) at its next call. With no file,
    every tool is on with its own settings."""

    def __init__(self, tools_file=None, skill_folders=(), added=None, code=()):
        self.file = tools_file
        self.skill_folders = [str(f) for f in skill_folders or ()]
        self.added_folder = str(added) if added else None   # where (add_tool …) puts what the model adds, read last
        # the Python the server imports as it starts, kept in the data folder (the image's `python/`: what
        #   requirements-nonfree.txt installs) — a module the model's commands wrote there would run in the server's own
        #   process, beside the tools file, so they read it and never write it, as the skill folders
        self.code_folders = [str(f) for f in code or ()]
        self.lock = threading.Lock()
        self.skills, self.tools, self.held, self.unavailable = [], {}, {}, {}
        self.generation = 0           # changes when the skills or a tool's switch or settings do

    def refresh(self, keep=False):
        """The skills read again and the file kept in step with them (▶ ToolsFile.sync), and each tool's switch and
        settings read from the file and the environment — with `keep`, a tool's held already kept as it was: a
        session's start and (reset) find the skills installed since, and the settings change on a rebuild alone.
        Answers the file's changes, a line each."""
        found = _skills.read_all(self.skill_folders + ([self.added_folder] if self.added_folder else []))
        tools = tools_of(found)
        changed = self.file.sync(tools) if self.file is not None else []
        entries = self.file.entries() if self.file is not None else {}
        with self.lock:
            held = {}
            for name, t in tools.items():
                if keep and name in self.held:
                    held[name] = self.held[name]
                    continue
                e = entries.get(name, {})
                held[name] = (_switch(name, e.get("enabled"), t["enabled"]),
                              _settings(name, t["config"], e.get("config")))
            if held != self.held or _signature(found) != _signature(self.skills):
                self.generation += 1
            self.skills, self.tools, self.held = found, tools, held
        return changed

    def added(self, skill):
        """Whether the model added this skill (▶ tools' `add_tool`): its folder is in the added folder."""
        if not self.added_folder:
            return False
        root = os.path.realpath(self.added_folder)
        return os.path.realpath(skill.folder).startswith(root.rstrip(os.sep) + os.sep)

    def writable_folders(self):
        """The skill folders and the server's own Python (`code_folders`) the server's user can write — the ones a model's
        command must not (▶ tools' classifier, jail.py): only (add_tool …) writes a skill folder."""
        return [os.path.realpath(f) for f in self.skill_folders + ([self.added_folder] if self.added_folder else [])
                + self.code_folders if os.path.isdir(f) and os.access(f, os.W_OK)]

    def checked(self, unavailable):
        """What the startup check found nothing to answer: {command: why}."""
        with self.lock:
            self.unavailable = dict(unavailable)

    def enabled(self, name):
        """The tool's switch, "yes", "no" or "auto"; a name that is no tool's, "yes"."""
        return self.held.get(name, ("yes", {}))[0]

    def config(self, name, defaults=None):
        """The tool's settings, over `defaults`. Never the model's: a tool's own Python and programs get them."""
        got = dict(defaults or {})
        got.update(self.held.get(name, ("yes", {}))[1])
        return got

    def why_not(self, tool, command):
        """Why `command`, of `tool`, cannot answer here — switched off, or the startup check's reason — or None."""
        if self.enabled(tool) == "no":
            return "disabled on this machine (its tools file)"
        why = self.unavailable.get(command)
        return "not available here: %s" % why if why else None

    def listing(self):
        """Each tool: its switch, its settings, what it is, and why it cannot answer here, if it cannot."""
        commands = {s.name: [c.name for c in s.commands] for s in self.skills}
        return {name: {"enabled": self.enabled(name), "config": self.config(name), "about": t["about"],
                       "unavailable": next((self.unavailable[c] for c in commands.get(name, [name])
                                            if c in self.unavailable), None)}
                for name, t in self.tools.items()}
