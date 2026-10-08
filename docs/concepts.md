# The ideas behind SiLVANN

SiLVANN's design did not fall out of a spec. Several of its choices started as intuitions about what a transformer
is "really doing", and those intuitions still explain why the engine looks the way it does. Read the physics and
the philosophy here as lenses, not mechanism: the engine runs ordinary linear algebra on ordinary silicon.

## Two halves, and a line between them that can be checked

| | |
|---|---|
| **the Eval-Apply GPU Interpreter** — `engine`, `sys` | the language and its evaluator: a small Lisp, its environment and the heap it lives in. A program is an s-expression; the evaluator walks it and hands the arithmetic to the silicon that holds the data. |
| **the Neural Network library** — `nn` | what a model is built with: the arithmetic primitives, the structures that hold weights and experts, and the words a model's program is written in. |

**The interpreter does not know what a transformer is, and the library does not schedule anything.** That is the
whole of the split, and it is checkable: the interpreter's code names no layer, and the library's names no
program.

## The model as a collapse of meaning

Before a word is generated, its meaning is not yet one thing. The up-projection inside an MLP or an expert spreads
a token across a wide hidden space — many possible senses active at once, a superposition of meanings. "Bank" is,
for the moment, river *and* institution *and* tilt, all held together and weighted.

Generation is the act that ends this. Attention, conditioned on the surrounding context, collapses the
superposition: of all the meanings a token could carry, context selects one and a single token is emitted. It is
useful to picture this as a measurement — the wide front of candidate meaning narrowing to a point the instant it
is observed.

The KV cache extends the collapse through time. A token's meaning is not settled when it is first produced: a later
token can reach back and re-collapse an earlier ambiguity. You read "…along the river" three words on, and the
earlier "bank" resolves. The past is continually re-measured by the present. That is the concrete reason the stored
keys and values are a first-class design surface here: they are what the model re-interprets every step, so how
they are kept — the tiers of the cache, what is held at full precision and what is compressed — matters as much as
the arithmetic that reads them.

## Why this is more than a cute analogy

The picture above is, almost line for line, the post-structuralist account of meaning:

- Meaning is not intrinsic to a word; it is produced relationally, by context and by a word's differences from
  other words (Saussure's differential value; Derrida's *différance* — meaning is never present in the sign, only
  deferred and relational).
- A text has no single determinate meaning until it is read; the reader actualises one reading out of many
  (Barthes' "death of the author"; reader-response criticism).

A language model makes this computational. It holds a superposition of latent meaning and collapses it, token by
token, into determinate text — and, through the cache, keeps re-collapsing the past as new context arrives. Under
this lens a language model is less a database of facts than a meaning-collapse engine: it does not retrieve
meaning, it produces it, contextually, in the moment of generation. A perspective, not a metaphysics — offered for
how it reframes the engineering.

## The engine as SICP's Chapter 5

That section is about the model. This one is about the machine. The design was reached by reasoning about what an
evaluator must be, and it converged on what *Structure and Interpretation of Computer Programs* builds in its
fourth and fifth chapters: an evaluator — `eval` and `apply` — over an environment of bindings, with a set of
primitive procedures it does not look inside.

| SICP | SiLVANN |
|---|---|
| the evaluator, `eval` / `apply` | the Eval-Apply GPU Interpreter's evaluator, on the host |
| the environment, frames of bindings | the bindings a program runs in: tables, buffers, procedures, pictures |
| primitive procedures | the verbs of `nn` and of a model's package — each launches its arithmetic on a card or on the CPU's cores |
| special forms (`if`, `define`, `let`, `lambda`) | `if`, `defun`, `let`, `while` |
| a program is data | a model is a Lisp file it carries (`models/<name>/lisp/`), read at boot into lists the evaluator walks |

Three things follow that a fixed inference graph does not have:

1. **A program can rewrite itself as it runs.** `while` is not a loop in C: each turn it rewrites its own form —
   the action, then the test again — so a loop of any length holds what one turn holds.
2. **A program can hand a program to another worker.** `sys__compute` gives a frozen program (a picture) to
   another worker — a CPU socket, when a card runs the rest of the model — and `sys__result` collects its answer.
   That is how GLM 5.3 Flash's experts run on the CPU while its other layers run on the card.
3. **The model's output can be a program the same machine runs.** The day a model emits a plan as data, the engine
   that runs the model also runs the plan — code and code generator on one machine. ▶ `directions.md`.

## The KV cache as a tape

A register machine with an unbounded, randomly addressable memory is a Turing machine; the only question is where
the tape is. In an autoregressive model the tape is the context itself, stored as the KV cache:

| a Turing machine | an autoregressive language model |
|---|---|
| the tape | the cached keys and values — the context the model attends over |
| the read head | attention, and it is random-access: any step may read any past cell |
| writing a cell | the autoregressive step appending the new token's keys and values |
| the transition function | one decoding step — the fixed weights — applied over and over |

Run that loop with enough tape and you have, in principle, a universal computer. Transformers are Turing-complete
given unbounded precision and steps (Pérez et al., 2019), and a language model coupled to an external read/write
memory can simulate a universal Turing machine (Schuurmans, 2023). Chain-of-thought is the same observation from
the practitioner's side: letting the model write intermediate tokens and read them back hands the loop a working
tape.

Two honest caveats. A real deployment has a finite context and finite precision, so strictly it is a very large
finite-state machine; universality is the statement as the tape grows without bound. And attention's cache is
append-only — a past cell is never overwritten — so it is most precisely an unbounded log with random read;
overwriting is simulated by appending tokens that supersede earlier ones.

## How the engine is extended

- **A capability is a new verb or a program rewrite, not a fork of the engine.** The evaluator stays small; what a
  model does lives in its programs and its package's verbs. A prompt read as rows, a model split over machines,
  experts on another worker — each was a new program or a new verb, not a new evaluator.
- **A small, sharp primitive plus a transform beats a special-cased feature.** When something new is needed, look
  for the one primitive that makes it expressible, and write the rest as a program over it.
- **Prototype, then promote, then prove equivalence.** Write the new behaviour first as a program over the verbs
  that exist; when it is right, promote the hot part into a verb; then show that the two agree before switching.
  That is how the prompt-as-rows path was built: every model's rows are checked against the same prompt read a
  position at a time, and against the reference implementation.

## A speculative aside

Strip away the metaphors and one bet remains: that intelligence may be less biological magic and more a matter of
navigating high-dimensional semantic geometry well — and that the geometry can be run efficiently, on hardware an
individual can afford. That is a personal hypothesis, flagged as exactly that. Nothing in the engine depends on it
being true; the code stands on its own.
