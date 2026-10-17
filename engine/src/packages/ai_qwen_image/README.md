# ai_qwen_image — Qwen-Image 2.1's diffusion transformer, a block a verb

⚖ 2026-10-15 (the architect): *"make qwen image 0.10.0, tensor parallelism after"*; the block a package verb, as the
vision tower's.

One verb, `ai_qwen_image__block`, over a block's plane table (`contracts/objects/block.cuh`): a block of the
single-stream transformer over the sequence's rows — the prompt's, then the generated picture's — its residual in place.
It launches nn's doors: `layernorm_rows` (its weight the step's 1 + scale), the ladder's rotation and `expert_rows` for
every matrix, `rmsnorm_rows` a head, `rope_rows` from the sequence's three-axis table, `attention_limits` (a key limit a
row: the prompt causal, a picture whole, the generated picture every row), `vector_gated_add_rows` (the tanh gates),
`swiglu_combine`, `vector_copy` for the prompt's keys and values kept from the first step.

Four verbs do what a step does around the blocks, over the transformer's own table (`contracts/objects/step.cuh`):
`ai_qwen_image__time` (a timestep's modulation of the blocks — 1 + scale, nn's `vector_tanh` of the gates — and the last
norm's scale), `ai_qwen_image__text` (a run of the prompt's text rows into the sequence), `ai_qwen_image__latents` (a
picture's latent rows into it), `ai_qwen_image__out` (the prediction, and the flow's Euler step on the latents).

Two verbs are the picture decoder, its VAE (diffusers' `AutoencoderKLQwenImage21` on one frame), over the tables in
`contracts/objects/vae.cuh`: `ai_qwen_image__conv` (a convolution, by nn's `unfold_rows` and a product of the rows —
its input normed and through SiLU or upsampled first, its output's channels moved to space, or the output added onto
`y`, as its table says) and `ai_qwen_image__vae_attention` (the middle block's one head over every pixel). The
decoder's program is `lisp/qwen_image_vae.lisp`, `(vae_decode h w)`, and its runtime
`python/silvann_runtime/qwen_image_vae.py`.

The arithmetic is diffusers' (`QwenImage21TransformerBlock`, `QwenImage21Transformer2DModel`, diffusers 0.41.0). The
runtime (`python/silvann_runtime/qwen_image.py`) loads the pack, reorders q and k a head for nn's rotary pairing, lays
out the sequence, the rotary table, the key limits and the schedule, and the program (`lisp/qwen_image.lisp`) runs the
steps: `(denoise steps)`. Checked against diffusers by `test/src_qwen_image_block.py` and `test/src_qwen_image_dit.py`.
▶ `src/docs/design/pending/tool_calling.md` (NN-89).
