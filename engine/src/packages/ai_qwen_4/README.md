# ai_qwen_4 — Qwen3.8-Flash-Next's layer sites, a verb each

⚖ 2026-10-31 (the architect): *"then try to speed it up with the verbs/ see what is the tps and check if it is in line
with the rest of the models"*. The package the design names (`src/docs/design/qwen4_exp.md` ③): Flash-Next
(`qwen4_exp`) has no input norm before a sublayer — a site reads the gated residual's mix as it is, and its answer is
injected into four streams — so its sites are a package's own, not ai_qwen_3's.

The verbs, over a layer's table (the cells `python/silvann_runtime/qwen4_exp.py` writes) and the model's shape table
(`contracts/objects/layer.cuh`, which says each one's arguments):
- `ai_qwen_4__gated_pre` — the gated residual's before-step over the four fp32 streams: each stream normed by its slice
  of the gain, the mix's down and up maps (the rank rotated as its 256 and 128), the weighted mean, the injection weights;
- `ai_qwen_4__attention` — Qwen Sparse Attention's layer: q and its gate, k and v, each head normed and roped, the cache,
  the indexer's keys (a position's, and a block's when the position ends one); while every block is chosen the attention
  over every position, past that over the blocks the indexer scores best and the unfinished block's; the gated output
  projection;
- `ai_qwen_4__deltanet` — the Gated DeltaNet at a position, every head's step in one launch, its output gated by σ(z);
- `ai_qwen_4__shared` — the shared expert, gated by its own sigmoid;
- `ai_qwen_4__route` — the router's logits and its picks into the hand the CPU's routed experts read.

A prompt's chunk, its rows at once: `ai_qwen_4__gated_pre_rows`, `ai_qwen_4__inject_rows` (each row's streams plus its
injection weights times its answer), `ai_qwen_4__deltanet_rows`, `ai_qwen_4__attention_rows`, `ai_qwen_4__shared_rows`,
`ai_qwen_4__route_rows` — held to a position at a time's by a bound, not bits (CLAUDE.md: two forms).

Each launches the nn doors its site's words in `lisp/qwen4_exp.lisp` launch, in their order, so a program that calls
the verbs computes what the words compute, to the bit; the program chooses them with `VERBS` (▶ the program's header).
The routed experts are ai_qwen_3's on the CPU (`experts`, `experts_rows`, `experts_up` / `experts_down`).
Checked by `test/src_qwen4_exp_runtime.py` (HF's trace), `test/src_qwen4_exp_qsa.py` (the indexer past 2,051 positions,
against the numpy oracle) and the logits of the words and the verbs compared.
