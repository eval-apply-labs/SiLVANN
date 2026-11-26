# ai_thinkingmachines__inkling_2 — Inkling 2's own verbs

⚖ 2026-11-24 (the architect): *"each model should implement its own expert verb, so the shared verbs go into nn and the
collection of shared verbs used by a model go to their publisher, in this case ai_thinkingmachines__inkling_2__"*.

Inkling 2 (Thinking Machines; `inkling2_loop_text`) runs as `lisp/inkling2.lisp`, a program in nn's words. What the
package holds is the model's own:

- `ai_thinkingmachines__inkling_2__experts hand picks table routed` — a position's routed experts: nn's routine
  (`nn__routed__position`) over this model's table, its hand — the experts' input rotated, then the router's weights — and
  its activation, SwiGLU soft-capped by the table's two caps (`contracts/objects/experts.cuh`).
- `ai_thinkingmachines__inkling_2__before_experts layer application work with_messages pos ring_row picks` and
  `ai_thinkingmachines__inkling_2__after_experts layer application work routed` — a layer's application, in two verbs
  around its experts'. ⚖ 2026-11-24: *"the threshold that makes a breakage earn its keep is if we do have a parameter to
  pass that the evaluator can change during the execution"* — where the experts run is the one thing a layer hands the
  evaluator. The first takes the stream to the router's picks (what the attention reads, the attention with its bias a
  distance and its sink, the short convolutions, the experts' own mix, the router, the hand filled); the second brings
  the experts' sum back into the stream. Three tables: the layer's weights, the application's own memory, the work
  every application shares (`contracts/objects/layer.cuh`). C over nn's doors, in the model's forward's order.

The program's other sites are nn's words; they become this package's verbs when the model's rate is taken up
(`src/docs/design/pending/inkling2_mini.md`).

Checked by `test/src_inkling2_runtime.py`: the program against `scripts/inkling2_oracle.py`, a site at a time.
