# How the weights are packed

The models are packed in the `silvann-packed-v3` format: every matrix quantised to a few bits a weight with a fixed
codebook, after a rotation that makes its rows look alike. The weights are never expanded back to 16 bits in
memory — the matrix kernels decode each weight from its code as they multiply.

## The codec

For a matrix `W` whose rows are multiplied with an activation `x`:

1. **Rotate each row along its input axis**, by a randomized Hadamard transform `R = H·S` — `S` a fixed pattern of
   random signs, `H` the Hadamard matrix. The rotation is orthogonal, so `W·x = (W·Rᵀ)·(R·x)`: the rotated weights
   against the rotated activation give the same product. What the rotation buys is that every row, whatever its
   outliers, comes out looking Gaussian — an "incoherent" row, in QuIP's term.
2. **Scale each row** to unit variance, and keep the scale — one half-precision number a row.
3. **Quantise each value** to the nearest level of a fixed codebook fitted once to a unit Gaussian (a Lloyd-Max
   quantiser for the normal distribution; at 4 bits it is the NF4 codebook). Because every row now looks the same,
   one codebook serves every tensor, and nothing is calibrated on data.

At inference the kernels rotate the activation with the same `R` (a fast Hadamard transform), and decode each
weight inline as `level[code] · scale[row]`.

In the literature: the rotation is the randomized Hadamard transform of QuIP and QuIP#; the fixed Gaussian
codebook without calibration is TurboQuant's method; rotating activations and the cache too is QuaRot's. The
conversation's cache uses the same codec for its older positions — 8 bits, then 4.

## What is per tensor, and what is per row

| | granularity | why |
|---|---|---|
| the **width** — bits a weight: 4, 5 or 8, or 16 for lossless | a tensor | the rows inside a tensor are too alike for a per-row width to pay: measured on the 35B, a per-row allocation would save at most 0.06 bits a weight |
| the **scale** | a row | it is what makes the fixed codebook fit the row at all; one scale for a whole tensor costs from a few percent to many times the error, and more at more bits |

A model's widths are chosen by name when it is packed: the published models put their routed experts at 4 bits
and the rest at 8 (`D8E4`, the 35B) or at 4 (`D4`, the 27B; `D4E4`, GLM 5.3 Flash) — ▶ the section below for why the
dense part is the one worth the bits. The embedding and the output
head are at 8 bits; one-dimensional tensors — norms, biases — are kept at 16.

## Why the dense part gets more bits than the experts

The rotation does more than make one codebook fit every row. It makes the quantisation error behave like small,
independent Gaussian noise added to each weight, the same for every row and with no preferred direction: no feature,
head or channel is hit harder than another. What that noise does to the model then depends on where the weight sits.

**A routed expert's error is added once.** A token's experts are a weighted sum: eight of them for these models, each
weighted by the router. Each expert's error is independent noise, so in the sum it stays additive and partly averages
out. And each expert serves only some of the tokens.

**A dense matrix's error is multiplied.** Every token goes through every dense matrix — the attention's projections,
the shared expert — and through all of them in turn, layer after layer. An error introduced there is carried into the
next matrix and multiplied by it, and by the one after that, so it compounds with depth instead of adding once.

So a clean base makes up for slightly noisier experts, and the measurements agree. Perplexity on WikiText-2's test split
(16 windows of 512 tokens), against the original weights:

| | the 35B | against the original |
|---|---|---|
| original (bf16) | 8.886 | — |
| D8E8 — everything at 8 bits | 8.887 | +0.01% |
| D8E5 | 8.900 | +0.15% |
| **D8E4** — the dense part at 8 bits, the experts at 4 | **8.959** | **+0.8%** |
| D4E4 — everything at 4 bits | 9.430 | +6.1% |

On the 122B the gap between the two published packs is wider: D8E4 5.926, D4E4 7.017, 18% higher. Taking the experts
past 4 bits buys well under 1%; taking the dense part from 4 to 8 bits buys 6% on the 35B and 18% on the 122B. The
experts are most of a model's bytes, so D8E4 keeps nearly all of the size saving and loses almost none of the quality.

**5 bits are not published.** The format defines a 5-bit width and the D8E5 row above was measured with it, but it buys
0.15 points of perplexity over D8E4 for 25% more expert bytes, and the card's and the CPU's fast kernels take 4 and 8 bits
only — at 5 bits the generic ones run, at half the speed. So no 5-bit pack is offered; the width stays in the format, and
fast 5-bit kernels can be added if a model ever wants it.

**Choosing a pack:** take D8E4 wherever it fits. D4E4 exists for cards too small to hold D8E4's dense part (the 122B
on an 8 GB card); it works, at the cost the table shows. The CPU's `experts_int8` arithmetic changes the 35B's
perplexity by 0.02%.

## The files

```
<model>/
  pack.json                the manifest: format, rotation, codebook, the width policy, the source
  global.pkl               the tensors outside the layers (embedding, head, final norm), and the sign pattern
  layers/NNN/weights.bin   a layer's records, each aligned to 4096 bytes
  layers/NNN/index.pkl     where each record is, its width, rows, columns and row size
  config.json, tokenizer.json, …   the model's own files, unchanged
```

A layer's routed experts are one record — the whole stacked tensor — and an expert's rows are found by arithmetic,
which is what lets the runtime copy one expert into a slot, or map a file of them, without an index of its own.

The reader is `runtime/nn_load_bundle.py`, and `runtime/nn_unpack_model.py` decodes a record back to numbers. The
packer that writes bundles from a model's original weights is not part of this preview.
