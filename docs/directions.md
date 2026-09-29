# Where it could go

Suggestions from the author for whoever picks this software up. They are directions, not features: each says what
exists in this preview and what does not.

## A model is a program, so a model can change how it runs

In this engine a model is not a fixed graph. It is a program in a small Lisp: lists of forms that name verbs,
over tables of weights, run by an evaluator that a program can build, inspect and rewrite while it runs — `while`
rewrites its own form every turn, `if` chooses what runs next. The machine that runs the model is the same
machine that runs any other program written in that language.

That opens a path that a fixed graph does not have: the model's output can itself be a program, run by the engine
between tokens. The precise version is a **fixed set of operations** the engine exposes — the scalars below, which
adapter is loaded, the sampling settings — and a model that emits programs over them; the engine runs those and
nothing else. Letting generated programs rewrite the weights themselves is a different thing, with its own
questions (what signal it learns from, what keeps it stable), and is open research.

*Today:* the language, the evaluator, `if` and `while` are there, and every model in this preview already runs as
programs over its tables. A model emitting programs, and the operations it would be allowed, are not built.

The paths below are the ones this makes concrete.

### 1. Sizing the experts after they answer

Today, in both model families here, a routed expert's output is multiplied by a weight the router chose from the
input, before any expert ran (Qwen: the router's softmax over the picked experts; GLM: its sigmoid scores and a
routing scale), and the weighted outputs are summed onto the residual.

The residual stream is a sum, so every contribution can be resized by one scalar before it is added — for the
price of a multiply, not of changing any weights. The suggestion is to choose those scalars **after** the experts
have run: activation → the experts → their outputs, and then, with the outputs' magnitudes and their agreement in
hand, the scalar each is given before they are added into the next activation.

Trained that way, an expert would learn a *direction* and the model would learn its *magnitude* separately — so
one expert could stand for several strengths of the same representation, where today it may take several experts
to hold them. (The spirit is the spectral theorem's: an operation that would cost a matrix costs a scalar along
the right direction. It is an analogy, not the theorem.) A model not trained this way was trained with every
scale at one, so small changes are safe and large ones need measuring; the norm before each next site absorbs part
of any change to the stream's overall size.

*Today:* the weighted sums happen in `ai_qwen_3__experts` / `ai_qwen_3__moe` and `ai_glm_5_3__experts`; the
scalars would go between each expert's output and that sum.

### 2. Choosing an adapter at the end of a sentence

A LoRA adapter specialises a model cheaply. At the end of a sentence the model could choose the adapter best
suited to what comes next — a second layer of expert selection, choosing among specialisations above the router's
choice among feed-forward blocks. The choice could be made by a small classifier on the last hidden state, by the
model scoring the last sentence under each adapter, or by a tag the model emits itself. Switching costs a rewrite
of the adapted matrices, and the cache built under the previous adapter stays as it was.

*Today:* the Qwen runtime loads and unloads adapters in place between turns (`Qwen35.load_lora` / `unload_lora`),
and a conversation remembers which one it runs with. Choosing automatically is not built.

### 3. Layers replaced by simpler programs

With `if` in the language, a layer does not have to be the same computation for every token. Parts of a model —
a whole layer, an expert, a site — could be replaced by simpler programs that test a condition and take a cheaper
path: skipping what a token does not need, stopping early when the answer is settled, or standing in for a block
with a smaller rule where that rule was measured to agree.

*Today:* the conditionals are in the language; every layer of the shipped models runs in full.

### 4. Multi-token prediction as a program

These models were trained with a multi-token prediction layer: from the last hidden state and the token just
chosen, it predicts the token after. Used for speculative decoding, the loop is: draft the next few tokens with
that layer, run them through the model at once as rows, keep the drafted tokens the model agrees with, and go on
from the first it does not. Engines usually bake that loop into their C++. Here it is a program — `while` there
are drafts to check, `if` a draft agrees — over verbs that already exist (the layer's sites, the rows forms), so
how many tokens to draft, when to stop drafting and what to do on a miss are lines of Lisp a user can change
without rebuilding anything.

*Today:* GLM 5.3 Flash's published bundle carries its prediction layer (the 46th, `eh_proj` / `enorm` /
`hnorm` around an attention and experts site); the Qwen bundles were packed without theirs. The draft-and-verify
program is not written.

### 5. The model setting its own budget

The cheapest version of a model steering its own execution needs nothing new: the model emits, before it reasons,
a small estimate of the effort a question needs — how long to think, how much context to keep at full precision,
how hot to sample — and the runtime sets those levers for the turn. Temperature is the most legible of them: high
for relaxed or creative conversation, low for precise work.

Two rules make it safe and one makes it honest. Every lever is clamped to a range and falls back to its default on a
malformed estimate, so a model cannot break its own run. The estimate should come first, as a short preamble,
because a budget set halfway through the reasoning is set too late. And it only pays if the model's estimate of its
own difficulty is calibrated: measure it against a fixed default and against the best setting in hindsight, over
questions from easy to hard, before believing it.

*Today:* the levers exist (the sampling settings, the thinking switch, the cache's tiers at boot); a model setting
them is not built.

### 6. Deliberating by simulation — drafters with their own adapters, judged by the big model

A model could try several continuations before committing to one. The cheap way to produce them is not the big
model itself but **smaller drafter models, each with its own LoRA adapter** — a speculative adapter that reads the
problem from a particular angle. They draft in parallel; the big model then reads each draft as a prompt — at the
speed it reads a prompt, which is many times its speed of writing — and scores it token by token.

What the big model does with a draft is the decision:

- **accept it up to the point where it stops agreeing** — the draft's tokens are kept while the big model's own
  probabilities stay above a threshold, as in speculative decoding, and the big model writes on from the first it
  rejects;
- **hand it back for a rewrite** — rather than take over, the big model tells the drafter where it went wrong and
  hands down a few pointers, and the small model rewrites that part itself;
- **choose among angles, or have them summarised** — with *n* drafters on *n* adapters reading the question from
  different sides, the big model either picks one, or asks a sub-agent — a model with an adapter it chooses, given
  an angle it chooses — to summarise them into one answer.

Each of those is a program over machinery the engine already has: models booted side by side, adapters swapped in
place, a prompt read as rows, one worker handing another a program. What is optimised at that moment matters too:
choosing among drafts needs only forward passes and fits the engine; changing the weights from the outcome needs a
backward pass and an optimiser — a training engine, a different project. Between the two sits updating a small
state from the outcome, which the delta-rule layers of these models (Qwen's DeltaNet, GLM's KDA) already do in
closed form every token.

*Today:* adapters swap in place and a prompt reads as rows; drafters, and the judging and handing back between
models, are not built.

### 7. Decoding as search

The head need not collapse to one token. It can produce a frontier of candidate continuations, each with its
score, and the decoding policy becomes a selector over that stream: greedy takes the best, beam search keeps *k*,
sampling draws by weight, and backtracking search (McCarthy's `amb`) descends depth-first and resumes a sibling
when a branch scores badly. In this engine each of those would be a program, not orchestration outside it.

The memory makes branching cheap in one place and costly in another. Attention's cache is append-only, so every
branch can share the committed prefix without copying it and write only its own suffix. The recurrent state of the
delta-rule layers is overwritten every token, so each branch needs its own copy of it — the dominant cost of
branching a hybrid model. Build it slow and correct first — branch by saving and restoring whole conversations,
which works today — and make it fast only once the search logic is right.

*Today:* a conversation can be saved and restored whole; branching and search are not built.

### 8. One machine, many conversations

The weights are the same for every conversation; only the cache differs. A server could hold the weights once and
apply a layer to whoever sends an activation, each client keeping its own cache — one box serving many
conversations, or many small machines sharing one large model. The pipeline that splits a model over machines
(`silvann_runtime/pipeline.py`) already passes activations between processes; this turns it around.

*Today:* a server answers one conversation at a time.
