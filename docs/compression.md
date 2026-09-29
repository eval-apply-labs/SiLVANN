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
and the rest at 8 (`D8E4`, the 35B) or at 4 (`D4`, the 27B; `D4E4`, GLM 5.3 Flash). The embedding and the output
head are at 8 bits; one-dimensional tensors — norms, biases — are kept at 16.

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
