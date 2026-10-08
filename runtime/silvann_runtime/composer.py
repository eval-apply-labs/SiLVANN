"""THE COMPOSER — text in, a program in the engine's heap out.

⚖ ARCHITECT: *"the composer is beyond the engine's eval apply loop, we emit our own in python and we
simply expose a entry point in c where you can start an execution specifying a bindings address and a
list ... our way is just a python convenience wrapper on top of the basics that remain accesible in c."*

So this file is a convenience and nothing more: everything it does, a C caller can do through the same two
doors it uses — `create_executable` and `execute`. What it adds is the part a person wants and a machine
does not care about: writing `(+ 1 2)` instead of naming kinds, verbs and symbol numbers by hand.

── WHAT IT IS FOR, AND WHAT IT IS NOT ──────────────────────────────────────────────────────────────
It is the engine's front end, and the runtime writes each model with it: any verb the built engine publishes
can be named — the language's, `nn`'s and a model package's (`ai_qwen_3__…`, `ai_glm_5_3__…`) — so what is
written here is a model's forward pass as much as any other lisp program.

── HOW A PROGRAM IS BUILT ──────────────────────────────────────────────────────────────────────────
Bottom up, one crossing per form. A form's cells are made first — numbers, names, and the CELLS that the
inner forms came back as — and then the form itself is one call. Nesting is putting one answer into the
next call's cells.
⛔ AND EACH PIECE IS GIVEN BACK AS IT GOES IN. `create_executable` takes nothing — it holds what it was
handed on its own account — so the hold this composer got when it built an inner form is still its own and
goes back at the moment that form is placed. What comes back holds everything under it, so the caller of
`compose` still releases exactly one thing.

── WHAT IS LOWERED HERE, AND WHY IT HAS TO BE ──────────────────────────────────────────────────────
    (if c a b)              the arms are QUOTED, because the scan must not reduce both of them
    (let ((n v) ...) b ...) becomes `(sys__let env 'n v ... '(begin b ...))`
    (set! n v)              becomes `(sys__bindings__set env 'n v)` — the name quoted, the value not
    (defun (f a) b ...)     becomes `(sys__defun env 'f '(a) '(begin b ...))`
⛳ THE ENVIRONMENT IS AN ARGUMENT TO ALL FOUR, because it is one to the verbs they become. A program
written here names no environment; the composer puts the one it was given into the cells.
⛳ AND A BODY OF SEVERAL FORMS IS WRAPPED IN A `begin`, because those verbs take ONE body and a sequence
needs a head or there is nothing to apply.
"""
import re

# The words a program writes, and the verbs they publish as. ⛳ A PROGRAM MAY ALSO WRITE THE PUBLISHED
# SPELLING — `sys__add` works wherever `+` does — so this table is sugar and never a gate: a name it does
# not know is looked for among the published verbs before it is treated as a name to read.
SURFACE = {
    "+": "sys__add",
    "-": "sys__sub",
    "*": "sys__mul",
    "<": "sys__less",
    "eq?": "sys__eq",
    "if": "sys__if",
    "begin": "sys__begin",
    "prog1": "sys__prog1",
    "clone": "sys__clone",
    "type": "sys__type",
    "create": "sys__create",
    "register-get": "sys__system_register__get",
    "register-set": "sys__system_register__set",
    "add-bindings": "sys__add_bindings",
    "remove-bindings": "sys__remove_bindings",
    "unquote": "sys__unquote", "while": "sys__while",
    "nth": "sys__node_array__get",
}

# What the composer lowers itself rather than passing through. Three of them take the environment as their
# first argument, so a program does not write one; `if` is here for a different reason — its arms are
# QUOTED, and a program that wrote them quoted would be writing down what the engine requires rather than
# what it means.
# ⛳ THE SPELLING A PROGRAM WRITES -> THE `_lower_*` THAT HANDLES IT. It was a tuple and the method was
# derived by stripping punctuation, which cannot name `=>` — and a table says what a derivation only
# implies.
LOWERED = {"let": "let", "set!": "set", "defun": "defun", "if": "if", "=>": "thread"}

_TOKENS = re.compile(r"""\(|\)|'|[^\s()']+""")


class ComposerError(Exception):
    """A program that cannot be composed, said before anything is built."""


def read(text):
    """Text -> nested lists of atoms. A quote becomes ("quote", <what follows>)."""
    tokens = _TOKENS.findall(text)
    if not tokens:
        raise ComposerError("nothing to compose")
    pos = [0]

    def one():
        if pos[0] >= len(tokens):
            raise ComposerError("the text ends inside a form")
        tok = tokens[pos[0]]
        pos[0] += 1
        if tok == "'":
            return ("quote", one())
        if tok == "(":
            out = []
            while True:
                if pos[0] >= len(tokens):
                    raise ComposerError("a form is never closed")
                if tokens[pos[0]] == ")":
                    pos[0] += 1
                    return out
                out.append(one())
        if tok == ")":
            raise ComposerError("a form is closed that was never opened")
        try:
            return int(tok)
        except ValueError:
            pass
        if any(ch.isdigit() for ch in tok) and tok[0] in "+-.0123456789":
            try:
                return float(tok)                    # a real: `0.5`, `1e-05` — a name never starts with a digit
            except ValueError:
                pass
        return tok

    form = one()
    if pos[0] != len(tokens):
        raise ComposerError("there is more text after the program")
    return form


def read_all(text):
    """A source file -> its top-level forms, in order. A `;` starts a comment that runs to the end of its line."""
    lines = [line.split(";", 1)[0] for line in text.split("\n")]
    tokens = _TOKENS.findall("\n".join(lines))
    forms, depth, start = [], 0, 0
    for i, tok in enumerate(tokens):
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                raise ComposerError("a form is closed that was never opened")
        if depth == 0 and tok != "'":
            forms.append(read(" ".join(tokens[start:i + 1]).replace("' ", "'")))
            start = i + 1
    if depth != 0:
        raise ComposerError("the source ends inside a form")
    return forms


def to_text(node):
    """A read form back into text, one line — what `read` would read again."""
    if isinstance(node, tuple) and node and node[0] == "quote":
        return "'" + to_text(node[1])
    if isinstance(node, list):
        return "(" + " ".join(to_text(n) for n in node) + ")"
    return repr(node) if isinstance(node, float) else str(node)


class Composer:
    """Composes against one engine and one environment.

    ⛳ THE SYMBOL NUMBERS ARE THE MACHINE'S, NOT THIS OBJECT'S. The engine deals one per name the first
    time any composer asks, and never gives it back (`packages/sys/cpu/symbol.cuh`), so two composers on one
    machine agree on every name. What this object keeps is a cache, which saves a crossing per mention.
    ⛔ SO A NUMBER CAN BE LARGER THAN THE ENVIRONMENT IS WIDE — another composer's names sit below these.
    `symbols` says how wide an environment would have to be to hold every name here by number, and `names`
    how many names there are.
    ⚖ AND A NAME WHOSE NUMBER DOES NOT FIT IS SPELLED: *"the dictionary is used for the values known at
    compile time/binding time"* was said of the array, and the cutoff is the width of the environment this
    composer writes for (⚖ *"env width"*). Such a name's cell holds the interned string instead, and the
    environment finds it in the dictionary behind its hatch. With no environment, the cutoff is the widest
    an environment can be.
    """

    def __init__(self, engine, env=None):
        self.e = engine
        self.env = env
        self.kinds = engine.kinds()
        self.verbs = engine.verbs()
        self._symbols = {}
        self._strings = {}
        self.width = engine.bindings_symbols(env) if env else engine.BINDINGS_WIDTH_MAX
        if self.width == 0:
            raise ComposerError("the environment this composer was given is not one")

    # ── the pieces a cell is made of ────────────────────────────────────────────────────────────────
    def symbol(self, name):
        """The number this name is known by on the machine — asked of the engine the first time this
        composer meets it."""
        if name not in self._symbols:
            self._symbols[name], self._strings[name] = self.e.intern_string(name)
        return self._symbols[name]

    def spelled(self, name):
        """Whether this name's cells hold its string rather than its number — its number does not fit."""
        return self.symbol(name) >= self.width

    @property
    def names(self):
        """How many distinct names the programs composed so far have used."""
        return len(self._symbols)

    @property
    def symbols(self):
        """How wide an environment would have to be to hold every name composed so far by number: one past
        the largest."""
        return max(self._symbols.values()) + 1 if self._symbols else 0

    def _verb_named(self, word):
        spelling = SURFACE.get(word, word)
        return self.verbs.get(spelling)

    def _cell_verb(self, word):
        return (self.kinds["sys__standard"], self._verb_named(word), 0)

    @property
    def _counted(self):
        """The kinds whose cell holds what it names — `sys__heap_node__carries_reference`, derived from the
        published table rather than typed, so a new one arrives here without this file learning its name.

        ⛔ AND THERE IS NO `if k in self.kinds` GUARD, WHICH THERE WAS UNTIL IT COST A DAY. The guard
        made a name this table does not carry into a SILENTLY SMALLER SET rather than an error — so when
        the published spellings gained their package prefix, every name here missed, this set came back
        EMPTY, and `_emit` released nothing. The suite did not fail where the names are; it failed 200
        programs later, as a pool running out.
        ⇒ ★ A LOOKUP THAT TOLERATES A NAME IT DOES NOT KNOW CANNOT TELL A NEW KIND FROM A TYPO, and the
        thing it was protecting against — a kind that is not published yet — is not a thing this suite
        ever meets. A KeyError here names the row to fix."""
        return {self.kinds[k] for k in ("sys__object_reference", "sys__quoted_list",
                                        "sys__procedure_reference", "sys__quoted_array")}

    def _cell_int(self, n):
        return (self.kinds["sys__value_int"], 0, n)

    def _cell_float(self, x):
        """A real as the engine holds one: its float64 bits in the value word."""
        import struct
        return (self.kinds["sys__value_float"], 0, struct.unpack("<Q", struct.pack("<d", float(x)))[0])

    def _cell_name(self, name):
        if self.spelled(name):
            return (self.kinds["sys__string_binding_reference"], 0, self._strings[name])
        return (self.kinds["sys__binding_reference"], 0, self.symbol(name))

    def _cell_quoted_name(self, name):
        if self.spelled(name):
            return (self.kinds["sys__quoted_string_name"], 0, self._strings[name])
        return (self.kinds["sys__quoted_name"], 0, self.symbol(name))

    def _quoted(self, cell):
        """The same reference, left alone by the scan. ⚖ ARCHITECT: *"the default list is executable and
        you only quote the wrapper."* So quoting is a tag on the cell and never a copy."""
        return (self.kinds["sys__quoted_list"], 0, cell[2])

    def _emit(self, cells):
        """One form, one crossing — and the pieces are given back as they go in.

        ⛔⛔ `create_executable` TAKES NOTHING. It holds what it was handed on its own account, so the hold
        this composer got when it built an inner form is still this composer's and has to go back here.
        ⚖ ARCHITECT: *"make them agree, create_executable should not consume."* ⇒ what comes back holds
        everything under it, and the caller of `compose` releases exactly one thing, as before.
        ⛳ EACH OBJECT APPEARS ONCE PER FORM, which is what makes one release each correct: `_quoted`
        RETAGS a cell rather than copying it, so quoting a form does not put a second reference to it in
        the same list."""
        cells = list(cells)
        got = self.e.create_executable(cells)
        for kind, _verb, value in cells:
            if kind in self._counted:
                self.e.release(value)
        return got

    # ── the program ─────────────────────────────────────────────────────────────────────────────────
    def compose(self, text):
        """Compose one program and answer the cell that names it. The caller releases that cell's value
        when it is done, and everything under it goes with it."""
        return self._form(read(text) if isinstance(text, str) else text)

    def _body(self, forms):
        """One form, or several wrapped in a `begin` — the verbs that take a body take exactly one.
        ⛔ A BODY THAT IS ONE BARE VALUE IS WRAPPED TOO, for `if`'s reason below: the body is quoted, and a quoted
        name or number is not a list — `(let ((e EPS)) e)` answered nothing and raised a fault."""
        if not forms:
            raise ComposerError("a body with nothing in it")
        if len(forms) == 1:
            return self._form(forms[0] if isinstance(forms[0], list) else ["begin", forms[0]])
        return self._emit([self._cell_verb("begin")] + [self._form(f) for f in forms])

    def _form(self, node):
        if isinstance(node, int):
            return self._cell_int(node)
        if isinstance(node, float):
            return self._cell_float(node)
        if isinstance(node, str):
            return self._cell_name(node)
        if isinstance(node, tuple) and node and node[0] == "quote":
            what = node[1]
            if isinstance(what, str):
                return self._cell_quoted_name(what)
            return self._quoted(self._form(what))
        if not isinstance(node, list):
            raise ComposerError("a form that is neither a number, a name nor a list")
        if not node:
            raise ComposerError("an empty form")

        head = node[0]
        if isinstance(head, str) and head in LOWERED:
            return getattr(self, "_lower_" + LOWERED[head])(node)
        if isinstance(head, str) and self._verb_named(head) is not None:
            return self._emit([self._cell_verb(head)] + [self._form(a) for a in node[1:]])
        # ⭐ ANYTHING ELSE AT THE HEAD IS A CALL, and it is written exactly as it is read: a name the
        #   evaluator resolves, and the arguments after it. That is how a procedure calls itself.
        return self._emit([self._form(head)] + [self._form(a) for a in node[1:]])

    # ── the four the composer lowers ────────────────────────────────────────────────────────────────
    def _env_cell(self):
        """⭐ 0: THE ENVIRONMENT OF WHATEVER COMPUTATION RUNS THE FORM — `let`, `set!` and `defun` resolve it from their
        computing base. It was this composer's own environment, so a picture composed here and handed to another block
        bound into the HOST's names there, and read its own, where nothing was bound. On the host it is the same one."""
        if self.env is None:
            raise ComposerError("this form needs an environment and the composer was given none")
        return self._cell_int(0)

    def _lower_let(self, node):
        if len(node) < 3 or not isinstance(node[1], list):
            raise ComposerError("a let is (let ((name value) ...) body ...)")
        cells = [self._cell_verb("sys__let"), self._env_cell()]
        for pair in node[1]:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ComposerError("a let binding is (name value)")
            cells.append(self._cell_quoted_name(pair[0]))
            cells.append(self._form(pair[1]))
        cells.append(self._quoted(self._body(node[2:])))
        return self._emit(cells)

    def _lower_if(self, node):
        """`(if test then else)` — the arms quoted, so the scan reduces the test and leaves both of them.

        ⛔ AN ARM THAT IS NOT A FORM IS WRAPPED IN ONE, and that is not a convenience: quoting is a tag on
        a cell, so tagging a number would say "this integer is a list" and the engine would follow it as
        one. `if` takes quoted LISTS, so an arm that is a bare value becomes `(begin value)`."""
        if len(node) != 4:
            raise ComposerError("an if is (if test then else)")
        arms = [a if isinstance(a, list) else ["begin", a] for a in node[2:4]]
        return self._emit([self._cell_verb("if"), self._form(node[1])] +
                          [self._quoted(self._form(a)) for a in arms])

    def _lower_thread(self, node):
        """`(=> s1 s2 ... sn)` — a chain written flat, LOWERED TO NESTING. Each step's answer becomes the
        FIRST argument of the next, so `(=> (f a) (g b) (h c))` composes as `(h (g (f a) b) c)`.

        ⚖ ARCHITECT: *"implementing tail call => can improve the result significantly, maybe resorting
        to 5 lists in total per layer like `(=>(attention) (ffn))`"*, and earlier: *"we can write an
        opcode that on a `(cascade f1 f2 f3 f4)` it executes the cdr and then moves the first element of
        the resulting list into the second position."*

        ⛔⛔ AND IT IS NOT AN OPCODE, WHICH IS THE WHOLE POINT OF THIS METHOD. `MEASURED`
        before writing one, 40 chains of 5 steps:
        ```
          flat, named intermediates + set!    1766 us/chain    353 us/step
          NESTED                               599 us/chain    120 us/step   2.95x cheaper
        ```
        120 us/step IS the bare-list rate (`NN-18`), so **a nested chain already costs ONE LIST PER STEP
        and nothing more — the evaluator threads at the floor.** It has always done so: the scan reduces
        an inner form and writes its answer into the enclosing form's argument slot, which is exactly
        what a threading verb would do.
        ⇒ ★★ A `=>` VERB WOULD HAVE BEEN STRICTLY WORSE. Rewriting `(=> ...)` step by step costs one
        RE-APPLICATION of `=>` per element on top of each element's own list — a verb cannot evaluate,
        so every step has to go back through the scan. The construct that looked like the optimisation
        was the one the engine was already doing for free.
        ⇒ ⭐ SO WHAT `=>` BUYS IS **NOTATION AND NOTHING ELSE**, and it buys it at exactly zero: this
        lowering emits the same cells a reader would have hand-nested. The 2.95x is real and is against
        the spelling the layer program uses TODAY — named intermediates threaded through `let` — not
        against nesting.
        ⛳ A BARE NAME IS A CALL ON THE ANSWER ALONE (`(=> a b c)` -> `(c (b (a)))`), which is the
        architect's `(=> attention_tq8 attention_tq4 attention_fp32 vector_add)`: those steps take what
        the previous one made and whatever else they read from their bindings."""
        if len(node) < 3:
            raise ComposerError("a => is (=> step step ...) with at least two steps")
        first = node[1]
        inner = first if isinstance(first, list) else [first]
        for step in node[2:]:
            if isinstance(step, list):
                if not step:
                    raise ComposerError("a => step cannot be an empty form")
                inner = [step[0], inner] + step[1:]
            elif isinstance(step, str):
                inner = [step, inner]
            else:
                raise ComposerError("a => step is a form or a name")
        return self._form(inner)

    def _lower_set(self, node):
        if len(node) != 3 or not isinstance(node[1], str):
            raise ComposerError("a set! is (set! name value)")
        return self._emit([self._cell_verb("sys__bindings__set"), self._env_cell(),
                           self._cell_quoted_name(node[1]), self._form(node[2])])

    def _lower_defun(self, node):
        if len(node) < 3 or not isinstance(node[1], list) or not node[1]:
            raise ComposerError("a defun is (defun (name param ...) body ...)")
        name, params = node[1][0], node[1][1:]
        if not isinstance(name, str) or any(not isinstance(p, str) for p in params):
            raise ComposerError("a defun's name and parameters are names")
        # ⛔ THE PARAMETERS ARE A QUOTED LIST OF QUOTED NAMES, AND AN EMPTY ONE IS AN EMPTY LIST rather
        #   than nil. `defun` takes quoted lists and refuses anything else, and it is what turns them into
        #   a picture — where an empty list BECOMES nil, which is what a procedure of no parameters holds.
        #   So nil is the right thing inside a procedure and the wrong thing to hand `defun`.
        # ⛳ AND AN EMPTY LIST IS THE ONE SHAPE THE ONE-CROSSING DOOR CANNOT MAKE, because a list of no
        #   cells is refused there — so it is made through the door that takes them one at a time, which
        #   is what that door is still for.
        if params:
            names = self._quoted(self._emit([self._cell_quoted_name(p) for p in params]))
        else:
            empty = self.e.list_create()
            if empty == 0:
                raise ComposerError("the pool would not give an empty parameter list")
            names = (self.kinds["sys__quoted_list"], 0, empty)
        return self._emit([self._cell_verb("sys__defun"), self._env_cell(),
                           self._cell_quoted_name(name), names, self._quoted(self._body(node[2:]))])

    # ── running one ─────────────────────────────────────────────────────────────────────────────────
    def run(self, text):
        """Compose, execute, and let the program go — which is everything under it too.

        ⛔ A PROGRAM IS USED UP BY RUNNING IT. Evaluation REWRITES the list, so a program that has run is
        not a program any more: to run one twice, compose it twice, or make it a `defun` and call it."""
        cell = self.compose(text)
        try:
            pic = self.e.freeze(cell)          # a program travels as a picture, and survives its own run
            kind, value = self.e.execute((self.kinds["sys__object_reference"], 0, self.env), pic)
            self.e.release(pic[2])
        finally:
            self.e.release(cell[2])
        return kind, value
