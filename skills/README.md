# Skills

Commands for the model's REPL that live outside the runtime: a folder each, with a `SKILL.md` that names the skill and
declares its commands — how each is called, what it touches (so it is asked as the REPL's own commands are), what runs
it (a Python function or a program in the folder) and what it does. The REPL reads the skills here and in `skills/`
beside the conversations when a conversation's session begins, and again at its `(reset)`; `(skills)` lists them.

How to write one: [`../docs/tools.md`](../docs/tools.md#adding-a-command). `files/` is one the release ships.
