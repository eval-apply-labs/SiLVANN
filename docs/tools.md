# The model's tool, and adding your own commands

A conversation's model has a tool: a Lisp REPL on the server's machine. The model writes one form at a time in its
answer, `(read "notes/a.md")`, and reads its value back before it writes on. This page has two parts:

- **[The REPL](#the-repl):** what the model can call.
- **[Adding a command](#adding-a-command-a-skill):** how to give the REPL commands of your own, written in Python or as
  a program, so the model calls them from Lisp like the built-in ones.

What the tool may do without asking is set by the config's `tools` ([`config.md`](config.md)) and by the UI's switch
beside each conversation ([`users.md`](users.md#the-models-tool)).

## The REPL

| command | what it does |
|---|---|
| `(help)`, `(help 'name)` | Lists every command, or gives one's page. |
| `(ls "folder")` | The names in a folder; a folder's name ends in `/`. |
| `(read "file")`, `(read "file" from count)` | A file's first 200 lines, or `count` of its lines from line `from`; a last line says how to read on. |
| `(write "file" "text")`, `(write "file" "text" 'append)` | Writes the file, or adds to its end, making the folders it needs. |
| `(bash "command")`, `(bash "command" seconds)` | Runs a shell command in the working folder. It is stopped after 60 seconds, or after `seconds` (at most 600). On Windows it is Git for Windows' bash, where it is installed — never WSL's. |
| `(powershell "command")`, `(powershell "command" seconds)` | On Windows: runs a PowerShell command in the working folder, stopped as `bash` is. `(help)` then opens by saying the machine runs Windows, so the model picks it. |
| `(fetch "https://…")` | A web page's words, without its markup. |
| `(cd "folder")`, `(pwd)` | The working folder. |
| `(reset)` (or `(exit)`) | A fresh REPL: the names defined, the working folder and any once-allowances are gone. Files, allowed folders, the mode and the library stay. |
| `(keep 'name "what it does")`, `(library)`, `(import 'name)`, `(forget 'name)` | The library: definitions kept across conversations in `tools_library.json` (below). |
| `(skills)` | The skills installed, their commands, and any left out with the reason. |
| `(image "prompt")` | A picture drawn by the picture model and shown to the user. Settings go in the prompt: `--size 768x1024`, `--steps 20`, `--seed 7`, `--count 4` (four pictures, seeds 7, 8, 9 and 10). Without a picture model it answers at once that none is installed, so the model can say so ([`config.md`](config.md#a-picture-from-the-text-model)). |
| `(whoami)` | Which model this is and under which LoRA, its cards, its context and how much this conversation used, its tool's mode, and what `(agent …)` can reach: its LoRAs, the other models here, the workers. |
| `(agent 'model "config" "task")` | A subagent's session: another model, or this one under a LoRA, given a task ([below](#subagents)). |
| `(guide)` | This page, headed by where the skills folder is on this machine: how the model makes a command of its own. |

**Lisp to put the commands together:** `define`, `let`, `lambda`, `if`, `map`, `filter`, `length`, `split`, and text and
list words.

**Other names for the same commands:** `read-file`, `slurp-file`, `read-all`, `cat` and `slurp` are `read`; `system`,
`shell` and `sh` are `bash` — on Windows `system`, `shell`, `cmd` and `pwsh` are `powershell`, and `sh` is `bash`.

**State:** a conversation's REPL keeps what the model defines from one call to the next until `(reset)`. A server's
restart starts every REPL afresh.

### The library

The library keeps a definition the model wrote so another conversation can use it:

1. **Keep it.** `(define (word-count path) (length (split (read path) " ")))`, then
   `(keep 'word-count "how many words a file has")`, stores the definition's source in `tools_library.json` beside the
   conversations.
2. **Find it.** `(library)` lists what is kept.
3. **Use it.** `(import 'word-count)` brings it into any conversation's REPL. Kept names it uses come in first.
4. **Remove it.** `(forget 'word-count)` takes it out.

The file is plain JSON: you can read it, edit it or share it.

## Subagents

`(agent 'model "config" "task")` starts a conversation with another model, or with the same model under a LoRA, and
answers its answer to the task:

- **`'model`** is a model folder's name, or `'self` for the model's own.
- **`"config"`** is the name of one of that model's LoRAs, such as `"research"` (the model under it, the rest by
  default); the name of a config in that model's `configs/`; or a JSON object written as text, such as
  `"{\"lora\": \"research\"}"`. A config sets the LoRA, a `system` prompt (where the subagent's server lets a
  conversation have its own: its config's `user_system_prompt`), and `tools_allowed`, the commands the
  subagent's REPL has. By default it gets the read-only commands: `help`, `ls`, `read`, `pwd`, `cd`, `library`,
  `import`, `skills`, `guide` and `find`. A config's name comes before a LoRA's of the same name.
- **`"task"`** is its first message. The subagent starts with nothing of the caller's conversation, so a task that
  continues it under a LoRA carries what it needs: the conversation, or a summary of it.

**The session is a REPL inside the REPL.** After `(agent …)` the model is inside the subagent's session, as a person is
inside `silvann_cli`:

- each call is a message to the subagent;
- `(answer 'once)` or `(answer 'refuse)` answers what it asks;
- `(exit)` ends the session, deletes the subagent's conversation, and brings the model back to its own REPL.

**Asks:** the subagent's REPL is "reads unasked" with only its allowed commands. When it asks, the model answers, but it
can allow only what it could do itself without asking. Anything more is asked of the user first, as the model's own
command would be. A subagent cannot start subagents.

**Where it runs, cheapest first:**

1. **`'self`, with no LoRA or one the model switches to:** a new conversation on the same model. It starts at once. An
   adapter switches: its matrices are written into the model's between conversations.
2. **Another model a worker serves:** the config's `workers` ([`config.md`](config.md)), other SiLVANN servers. The
   subagent runs there, and its REPL reads the worker's files.
3. **Another model with no worker free, or `'self` under a LoRA it does not switch to:** this machine lends its card
   for each exchange. For every message, the model is put down, the subagent's model is loaded (the same model, under
   the LoRA, as it was set up) and answers, and the model is loaded back from the disk. Each way can take minutes.

A rank-1 mask is the LoRA a model does not switch to: it is applied beside the matrices, so it is chosen when the model
is loaded, and a model loaded under a mask switches to no other LoRA. `(whoami)` lists the model's LoRAs, each with
what it is for and whether it starts at once, and `(help 'agent)` tells the model to prefer one that does where it
suits the task.

**A LoRA's description** is what `(whoami)` says it is for: the `description` in its `pack.json` (an adapter) or its
`mask.json` (a mask), in the model's `loras/NAME/`. The model chooses a LoRA by it, so write what the LoRA is good at.

## Adding a command: a skill

A **skill** is a folder that gives the REPL new commands. It holds:

- **`SKILL.md`**, which names the skill and declares each command: how it is called, what it touches, what runs it, and
  its page for `(help 'name)`;
- **what runs each command:** a Python function in a `.py` file, or a program (a `.sh`, a `.py`, any executable).

**Where skills are read from:**

- `skills/` beside the conversations: in the install's folder, beside `start_silvann_server.py`, or in the folder
  `SILVANN_DATA` names (`/data/skills` in a container built from `docker/`). Yours go here.
- `skills/` in the install, the ones the release ships. `files` is one, and a working example.

**When they are read:** at the start of a conversation's REPL, and again at its `(reset)`. A skill added while the
server runs is there after a `(reset)`, without a restart. Editing a skill's Python takes effect the same way.

### Step by step: a Python command

This walks through a skill with one command, `(csv-column "file" "column")`, which answers one column of a CSV file.

**1. Make the folder:** `skills/csv/`.

**2. Write the Python:** `skills/csv/csv_tools.py`.

```python
import csv
import os


def column(path, name, context=None):
    """The values of one column of a CSV file, top to bottom."""
    full = os.path.join(context["cwd"], path)
    context["check"]("read", full)                   # asked as (read …) would be, outside the allowed folders
    with open(full, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and name not in rows[0]:
        raise ValueError("no column %r: the columns are %s" % (name, ", ".join(rows[0])))
    return [r[name] for r in rows]
```

**3. Declare it:** `skills/csv/SKILL.md`.

```markdown
---
name: csv
description: read the columns of CSV files
---
A skill for tables: its commands read CSV files in the working folder.

## (csv-column "file" "column")
touches: none
run: csv_tools.py:column
The values of one column of a CSV file, as a list of texts, top to bottom. The file's first line names its columns.
```

**4. Load it.** In a conversation, `(reset)` makes the REPL read the skills again; a new conversation reads them
anyway. Check:

- `(skills)` lists `csv` and its command, or says why something was left out;
- `(help)` names the command, and `(help 'csv-column)` gives its page.

**5. Use it from Lisp.** The model calls it like any built-in, and puts it together with the rest:

```lisp
(csv-column "people.csv" "city")                  ; ("Rome" "Oslo" "Rome")
(length (csv-column "people.csv" "city"))         ; 3
(filter (lambda (c) (= c "Rome")) (csv-column "people.csv" "city"))   ; ("Rome" "Rome")
```

A definition built on it can be kept in the library like any other: `(keep 'rome-count "…")`.

### The model making its own

The model can make a command itself, and the same steps apply:

1. **Read the guide.** `(guide)` answers this page, headed by the skills folder's path on this machine. The tool's
   description tells the model that `(guide)` is there.
2. **Write the files.** `(write ".../skills/NAME/SKILL.md" "…")` and its `.py` beside it. `write` makes the folders it
   needs.
3. **Load it.** `(reset)`, then `(skills)` to see it taken, then `(help 'command)` to see its page.
4. **Use it.** From then on it is a command in every conversation, kept like a library definition but in Python.

The skills folder is outside the workspace, so each write is asked of the user in "ask first" and "reads unasked": the
user sees the skill's code before it exists. In "full auto" nothing is asked.

A definition in Lisp needs no skill: `(keep 'name "…")` keeps it ([the library](#the-library)). A skill is for what Lisp
and the built-in commands cannot do, such as a Python library, a file format, or a program on the machine.

### How values cross between Lisp and Python

**Arguments** go in order, as the form wrote them:

| Lisp | Python |
|---|---|
| text, a name (`'abc`) | `str` |
| a number | `int` or `float` |
| a list | `list`, its elements converted the same way |
| `nil` | `None` |
| a procedure | refused: a procedure cannot be handed to a skill |

**`context`:** a function that has a parameter named `context` gets a dict:

- `cwd`: the REPL's working folder. A path the model gives is relative to it.
- `workspace`: the REPL's starting folder.
- `folder`: the skill's own folder, for files it ships.
- `check(touches, path)`: asks as the REPL's own commands ask (below). It raises when the user refuses, and the model is
  told.

**Answers:**

| Python | Lisp |
|---|---|
| `str` | text |
| `int`, `float`, `bool`, `None` | the same value; `None` is `nil` |
| `list`, `tuple` | a list; an empty one is `nil` |
| `dict` | a list of `(key value)` pairs |
| anything else | its `str()` as text |

**Errors:** an exception the function raises reaches the model as an error, with its type and message. Write its
message for the model, as `ValueError("no column …")` does above.

**Where it runs:** in the server's own process. The module is loaded once per session and again at `(reset)`. A
function that takes a long time holds the conversation's turn until it returns.

### A program instead of Python

`run:` can name a file in the skill's folder rather than `file.py:function`:

- **How it runs:** a `.sh` by bash, a `.py` by the server's Python, a `.ps1` by PowerShell, any other file as an
  executable. It runs in the
  REPL's working folder, with the form's arguments as its command-line arguments, all as texts.
- **The answer:** what it prints, as text.
- **Failure:** a failing exit is an error, with the end of its output.
- **Time:** it is stopped after a command's 60 seconds.

```markdown
## (note-search "word")
touches: read
run: search.sh
The lines of notes.md that hold the word.
```

with `search.sh`: `grep -i -- "$1" notes.md || true`.

### What a command touches, and when the user is asked

`touches:` says what the command does to the machine, so the REPL asks before it runs as it would for a built-in
command:

| `touches` | asked in "ask first" | asked in "reads unasked" (the default) | asked in "full auto" |
|---|---|---|---|
| `none` | no | no | no |
| `read` | yes | no | no |
| `write`, `run`, `net` | yes | yes | no |

**A command that leaves `touches` out** is asked as `run`.

**A command that chooses its own paths** declares `none` and asks for each path itself with `context["check"]`, as
`csv-column` and the shipped `find` do. Then a read in the workspace or an allowed folder goes unasked, and a read
elsewhere is asked like `(read …)`.

### Names

**A command's name must be new.** A skill's command never takes the name of a built-in or an alias, nor one an earlier
skill took. It is left out, and `(skills)` says so.

**The order skills are read in:** your `skills/` folder before the release's, and within a folder by folder name. So a
skill of yours named like a shipped one takes its place.

⚠ **A skill is code you install.** Its Python runs inside the server and its programs as the server's user, with the
user's files and network. In a container, those are the container's.
