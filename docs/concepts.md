# The concepts SiLVANN is built on

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/concepts_dark.svg">
  <img src="images/concepts_light.svg" alt="One token's life, top to bottom: on the host, the Eval-Apply Generalized
  Interpreter's evaluator walks the model's program a form at a time (generate, a token's position, every layer this
  machine runs, then the sampler), and each verb launches its arithmetic on the worker that holds the data. On the
  card, worker 0, through a silicon family: the token's embedding read from a table in RAM, the rotary angles, the
  DeltaNet or attention mixer, the router, the layer's close, the final norm and the logits, and the sampler's
  repetition penalty, temperature, top k and draw. On a CPU socket, worker 1, through x86_avx2: the routed experts,
  sys__scheduler__add handing it a frozen program, the picture xq, whose answer the card waits on with
  sys__wait_for_result and copies across. Only the token comes back to the host, each the next one's input, until an
  end token.">
</picture>

SiLVANN is an interpreter and a neural network library, with a line between them that can be checked, and a model
is a program the one runs over the other's words. This page is the concepts that follow from that: what to hold in
mind when reading the engine or extending it. What inspired them is in [`inspiration.md`](inspiration.md).

## Two halves, and a line between them that can be checked

| | |
|---|---|
| **the Eval-Apply Generalized Interpreter** — `engine`, `sys` | the language and its evaluator: a small Lisp, its environment and the heap it lives in. A program is an s-expression; the evaluator walks it and hands the arithmetic to the silicon that holds the data. |
| **the Neural Network library** — `nn` | what a model is built with: the arithmetic primitives, the structures that hold weights and experts, and the words a model's program is written in. |

**The interpreter does not know what a transformer is, and the library does not schedule anything.** That is the
whole of the split, and it is checkable: the interpreter's code names no layer, and the library's names no
program.

## The engine as SICP's Chapter 5

The design was reached by reasoning about what an evaluator must be, and it converged on what *Structure and
Interpretation of Computer Programs* builds in its fourth and fifth chapters: an evaluator — `eval` and `apply` —
over an environment of bindings, with a set of primitive procedures it does not look inside.

| SICP | SiLVANN |
|---|---|
| the evaluator, `eval` / `apply` | the Eval-Apply Generalized Interpreter's evaluator, on the host |
| the environment, frames of bindings | the bindings a program runs in: tables, buffers, procedures, pictures |
| primitive procedures | the verbs of `nn` and of a model's package — each launches its arithmetic on a card or on the CPU's cores |
| special forms (`if`, `define`, `let`, `lambda`) | `if`, `defun`, `let`, `while` |
| a program is data | a model is a Lisp file it carries (`models/<name>/lisp/`), read at boot into lists the evaluator walks |

Three things follow that a fixed inference graph does not have:

1. **A program can rewrite itself as it runs.** `while` is not a loop in C: each turn it rewrites its own form —
   the action, then the test again — so a loop of any length holds what one turn holds.
2. **A program can hand a program to another worker.** `sys__scheduler__add` gives a frozen program (a picture) to
   another worker — a CPU socket, when a card runs the rest of the model — and `sys__wait_for_result` reads its answer.
   That is how GLM 5.3 Flash's experts run on the CPU while its other layers run on the card.
3. **The model's output can be a program the same machine runs.** The day a model emits a plan as data, the engine
   that runs the model also runs the plan — code and code generator on one machine. ▶ `directions.md`.

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
