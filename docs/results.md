# Results

Every number on this page but the NVIDIA section's was measured on one machine, a Dell R730: two Xeon E5-2680 v4, two
AMD Instinct MI50, 256 GB of RAM. Disk speed 5 GB/s (2.6 GB/s where a row says so), RAM speed 126 GB/s. The README's
table of machines is drawn from this page. The NVIDIA section's machine is a desktop with an RTX 5060 Ti:
[▶ below](#on-an-nvidia-card-the-rtx-5060-ti).

**How it is measured.** On the card: a prompt of about 2,000 positions, then 32-64 tokens. With the experts on the CPU:
about 1,000 positions, then 128 tokens (32 in GLM's earlier rows). "All in RAM" means every expert is already in memory;
the other rows start cold, under the memory limit shown, with the expert files dropped from the operating system's cache
first. The answer's rate is measured after the prompt, from its 10th token on. The rows with a model entirely on one
MI50, and the 35B's and the 122B D4E4's with the experts on the CPU all in RAM, were measured on v0.7.0 before its
pictures were added (its answers since within 1% of them), whose runtime keeps the process out of the kernel's NUMA
balancing (on this two-socket machine its sampling cost a card's run 1.5-3%, and the 35B's prompt with its experts on
the CPU 5%); the 27B's row through OpenCL is v0.5.1's (the same on v0.7.0 before its pictures); the others were measured
with v0.3's card kernels, and where a card does part of the work they are a floor.

## How good each pack is: perplexity

The same text through every pack: the first 16 windows of 512 tokens of WikiText-2's test split, each read from a fresh
conversation, every next token scored from the model's own logits. **Perplexity** compares packs of one model, or of
models sharing a tokenizer (the Qwen ones); **bits per byte** — the same score over the text's bytes instead of its
tokens — compares any two models. Lower is better in both. The original weights' score (bf16, through Hugging Face's own
implementation) is given where the checkpoint fits this machine's memory.

| model | pack | weights | perplexity | bits per byte | against the original |
|---|---|---|---|---|---|
| GLM 5.3 Flash | D8E4 | 154 GiB | 3.018 | 0.3584 | |
| GLM 5.3 Flash | D4E4 (published) | 151 GiB | 3.095 | 0.3666 | |
| Qwen 3.5 397B-A17B | D4E4 (published) | 193 GiB | 3.548 | 0.4167 | |
| Qwen 3.5 122B-A10B | D8E4 (published) | 63 GiB | 5.926 | 0.5855 | |
| Qwen 3.5 122B-A10B | D4E4 (published) | 61 GiB | 7.017 | 0.6411 | |
| Qwen 3.6 35B-A3B | original (bf16) | 67 GiB | 8.886 | 0.7188 | — |
| Qwen 3.6 35B-A3B | D8E8 | 34 GiB | 8.887 | 0.7189 | +0.01% |
| Qwen 3.6 35B-A3B | D8E5 | 23 GiB | 8.900 | 0.7193 | +0.15% |
| Qwen 3.6 35B-A3B | D8E4 (published) | 18 GiB | 8.958 | 0.7215 | +0.8% |
| Qwen 3.6 35B-A3B | D4E4 (published) | 18 GiB | 9.430 | 0.7384 | +6.1% |
| Qwen 3.8 27B | D8 | 26 GiB | 9.053 | 0.7250 | |
| Qwen 3.8 27B | D4 (published) | 15 GiB | 9.259 | 0.7324 | |
| Ministral 3 14B | original (bf16) | 25 GiB | 9.807 | 0.7623 | — |
| Ministral 3 14B | D8 (published) | 12.6 GiB | 9.814 | 0.7626 | +0.07% |
| Ministral 3 14B | D4 (published) | 6.9 GiB | 10.275 | 0.7779 | +4.8% |

The dense part's precision is what costs: 8 bits there instead of 4 gains the 35B 6%, the 122B 18%, GLM 2.5%, the 27B
(all dense) 2.2% — while the experts past 4 bits gain under 1% (▶ `compression.md` for why). Between models the order is
the one their sizes suggest, but read it as how well each predicts encyclopedia text, which rewards having seen and
kept more of it; it does not measure reasoning, code or following instructions. The 35B with its experts on the CPU
scores 8.962 exact and 8.963 with `experts_int8`.

## Qwen 3.6 35B-A3B

| Where it runs | Prompt | Answer |
|---|---|---|
| one MI50 | 199 positions/s | 90 tokens/s (105 on a short prompt) |
| one MI50 for the dense part, the experts on the CPU, all in RAM | 88 positions/s | 28.4 tokens/s |
| the same, the experts read from the disk, 8 GB of RAM | 32 positions/s† | 11.9 tokens/s — 90% of expert reads from memory |
| the same, 6 GB of RAM | 31 positions/s† | 10.2 tokens/s — 83% from memory |
| `D4E4`, the CPU alone, no card, all in RAM | — | 15.4 tokens/s on a short prompt |
| `D4E4`, the CPU alone, the experts read from the disk, 12 GB of RAM (8 GB of it the experts' cache) | 21 positions/s† | 11.7 tokens/s — 95% from memory, 24 MB read a token |

## Qwen 3.8 27B

| Where it runs | Prompt | Answer |
|---|---|---|
| one MI50 | 102 positions/s | 31.5 tokens/s (34 on a short prompt) |
| one MI50 through OpenCL | 52.5 positions/s | 10.4 tokens/s (13.1 on a short prompt) |
| three stages over machines (pipeline parallel) | — | 6.8 tokens/s, the same tokens |

## Ministral 3 14B

| Pack | Where it runs | Prompt | Answer |
|---|---|---|---|
| D4 | one MI50 | 162 positions/s | 42 tokens/s (56 on a short prompt) |
| D8 | one MI50 | 158 positions/s | 30.6 tokens/s (38 on a short prompt) |

Ministral reads its prompt as rows, through nn's attention over many rows (v0.4.0, a position at a time, read 53
positions a second on the D4 pack). Its chat template adds about 560 positions, so its rows read about 2,600, and a short
prompt about 620.

## Qwen 3.5 122B-A10B

| Pack | Where it runs | Prompt | Answer |
|---|---|---|---|
| `D4E4` | one MI50 for the dense part, the experts on the CPU, all in RAM | 43 positions/s | 14.4 tokens/s |
| `D4E4` | the same, the experts read from the disk, 32 GB of RAM | 11 positions/s† | 4.7 tokens/s — 87% from memory |
| `D4E4` | the same, 16 GB of RAM | 10 positions/s† | 3.2 tokens/s — 73% from memory |
| `D8E4` | one MI50 for the dense part, the experts on the CPU, all in RAM | 25 positions/s | 6.1 tokens/s |
| `D4E4` | one MI50, the experts on the CPU, all in RAM — measured beside the tier rows | 38 positions/s | 10.9 tokens/s |
| `D4E4` | the same, 24 GB of the card the tier of experts (v0.3.0) | 52 positions/s | 19.9 tokens/s |
| `D4E4` | one MI50 through OpenCL, the experts on the CPU, all in RAM | 28 positions/s | 9.7 tokens/s |
| `D4E4` | the same, 18 GB of the card the tier of experts (OpenCL allocates at most 85% of the card at once) | 32 positions/s | 11.7 tokens/s |
| `D8E4` | the same, the experts read from the disk, 32 GB of RAM | 11 positions/s† | 3.9 tokens/s — 85% from memory |

## GLM 5.3 Flash

One MI50 for the dense part, the experts on the CPU.

| Experts | Disk | Prompt | Answer |
|---|---|---|---|
| all in RAM | — | 23.9 positions/s | 6.7 tokens/s |
| all in RAM, measured beside the tier rows | — | 18.1 positions/s | 6.8 tokens/s |
| all in RAM, 18 GB of the card the tier of experts (v0.3.0) | — | 25.0 positions/s | 8.1 tokens/s |
| all in RAM, 24 GB of the card the tier of experts (v0.3.1, from token 224) | — | — | 10.7 tokens/s |
| all in RAM, a 300,000-position cache and 6 GB of tier — 15.0 GB of the card in all | — | 26.5 positions/s | 7.5 tokens/s |
| 128 GB of RAM | 5 GB/s | 13.2 positions/s† | 5.7 tokens/s — 97% from memory, 132 MB read a token |
| 128 GB of RAM | 2.6 GB/s | — | 4.2 tokens/s — 97% from memory ◦ |
| 64 GB of RAM | 2.6 GB/s | — | 1.9 tokens/s — 79% from memory ◦ |
| 64 GB of RAM | 5 GB/s, overheating‡ | 4.9 positions/s† | 1.55 tokens/s — 80% from memory, 804 MB read a token ◦ |
| 32 GB of RAM, before grouped attention (v0.2.0) | 2.6 GB/s | 11 positions/s† | 1.1 tokens/s — 61% from memory ◦ |

With its experts on the CPU, a model reads a prompt a chunk of positions at a time through each layer — 256 for the
35B, 1,024 for GLM — so each expert is read about once a chunk and runs once over every position that picked it.
† Measured from a cold start, so reading the prompt is also when the experts are first read from the disk into
memory; the answer that follows runs on what the prompt left in memory.
◦ Measured before this release's card kernels — the rows on the 2.6 GB/s disk and the overheating 64 GB row; the card's
share of a GLM token has since become faster, so they are a floor.
‡ See below: the disk sits where the case's air does not reach it, and slows itself down within a minute of
sustained reading.

⚠ **Reading a prompt is not fully optimized yet.** It runs a chunk of positions through each layer at once — every
matrix once a chunk, on the card through a tiled kernel and on the CPU in 8-bit integers, and GLM's latent attention
over every row of the chunk at once, its 64 heads sharing each read of the cache.

The tier of experts starts empty in each of these runs; the prompt fills most of it, and the answer's rate climbs as the
rest fills — GLM's reaches about 9 tokens a second by its 240th token with 18 GB. Each tier run answers with the same
words as its run without, until a near-tie in the 122B's ~20 words in: the card sums its experts in another order than
the CPU does.

## On an NVIDIA card: the RTX 5060 Ti

A Dell Precision 5820: one Xeon W-2123 (Skylake, four cores, AVX-512), 62 GB of DDR4-2400 in two channels, an NVIDIA
GeForce RTX 5060 Ti 16 GB (sm_120, 36 SMs, 448 GB/s), CUDA 12.9, driver 595 — measured with v0.5.0. A prompt of about
2,000 positions (2,568 for Ministral), then 64 tokens (32 on the CPU alone), every expert in RAM; the answer's rate from
its 10th token on, and on a short prompt from the check against Hugging Face's answer.

| Model | Where it runs | Prompt | Answer |
|---|---|---|---|
| Ministral 3 14B `D4` | the card | 958 positions/s | 39.2 tokens/s (45.9 on a short prompt) |
| Ministral 3 14B `D8` | the card | 873 positions/s | 24.9 tokens/s (27.5) |
| Qwen 3.8 27B `D4` | the card | 549 positions/s | 23.6 tokens/s (24.4) |
| Qwen 3.8 27B `D4` | the card, a prompt of 6,033 positions — past the 4,352 the cache keeps at full precision | 472 positions/s | 21.8 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the dense part on the card, the experts on the CPU | 55 positions/s | 27.8 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the same, 8 GB of the card the tier of experts | 117 positions/s | 51.9 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the CPU alone | 20.8 positions/s | 8.6 tokens/s (9.0) |
| Qwen 3.6 35B-A3B `D8E4` | the CPU alone with AVX2 (`SILVANN_CPU_AVX512=0`) | — | 1.7 tokens/s on a short prompt |
| Qwen 3.5 122B-A10B `D4E4` | the dense part on the card, the experts on the CPU, read from the disk, 32 GB of RAM | 19.0 positions/s | 8.2 tokens/s — 87% from memory |
| GLM 5.3 Flash `D8E4` | the same, 60 GB of RAM | 10.0 positions/s | 3.0 tokens/s — 80% from memory |
| GLM 5.3 Flash `D8E4` | the same, 32 GB of RAM | 10.6 positions/s | 2.3 tokens/s — 65% from memory |

Against one MI50 the 27B reads a prompt 5.4 times as fast here (549 positions a second against 101) — the card's tensor
cores run the prompt's matrix products — and answers more slowly (23.6 tokens a second against 30.5): a token reads every
weight once, so the answer follows the memory's speed, 448 GB/s here against the MI50's 1,024. With the experts on the
CPU the four cores are what limits the 35B (55 positions a second against the R730's 83, whose 28 cores read the
experts). The tier run is one run: which experts the card holds follows when its copies land, so it varies from
run to run. Before v0.5.0's kernels, through the library's generic ones, the 27B read a prompt at 19 positions a second
and answered at 7.8 tokens a second; Ministral `D4` read its prompt a position at a time at 34.

The three rows with the disk were measured with v0.5.1, as the R730's were: a prompt of about 1,000 positions from a cold
start, then 128 tokens, the process held to the RAM it names, the experts read from an NVMe disk reading 5 GB/s.
Holding the disk to 5 GB/s changed neither model's answer rate; holding it to 2.5 GB/s slowed GLM's by 16% and the
122B's prompt by 25%. GLM's answer here is limited by reading its experts from RAM: the two channels give the four
cores at most 36 GB/s, and its rate follows that bandwidth from one core to four.

## What limits the experts on the disk: the drive

With the experts on the CPU, each is kept in the engine's own memory once it has been used, and one that is not is read
from its file straight into memory (`O_DIRECT`) while the others compute. A miss then costs what its bytes cost to read.
One GLM 5.3 Flash MoE layer on one socket, one of its 8 picks not in memory, measured on the test machine's single
PCIe 3 drive:

```
scale: 20 characters = 1 ms          0 ms                1.0                 2.0       2.5
                                     |-------------------|-------------------|---------|
the 7 experts in memory              [============================]                        1.4 ms of compute
the missing one, read (6.3 MB)       [===============================================]     2.4 ms — PCIe 3 x4 NVMe, 2.6 GB/s
  then computed                                                                    [====]  0.2 ms
                                                          ^ the layer waits ~1 ms for the disk

the same read, a drive twice as fast [========================]                            ~1.2 ms — a PCIe 4 NVMe:
  then computed                                               [====]                       nothing waited for
```

Where most of a token's experts are in memory — GLM with 128 GB — a fast disk removes most of the wait: from the
5 GB/s disk GLM answers 5.7 tokens a second, 85% of its rate with every expert in RAM (6.7). Where
many come from the disk — GLM with 64 GB, ~800 MB of reads a token — the rate follows the drive's almost directly.
The 35B on the CPU alone with 12 GB of RAM reads about 24 MB a token, under 1 GB/s at its peak, so any NVMe drive that
sustains 2.5 GB/s serves it with room to spare.

**NVMe drives slow down when they are hot.** The 5 GB/s disk keeps its rate for about 30 seconds of reading without
a pause; past 75 °C it throttles and holds a steady 2.4 GB/s, less than the 2.6 GB/s disk. A sustained read is what
GLM with 64 GB does for its whole answer, which is the row marked ‡ above. Give an NVMe drive air on the side that carries its controller.

Projected from the R730's rows: with the disk holding 5 GB/s, GLM with 64 GB should answer about **2.5 tokens a
second**. A token there is ~270 ms of computing plus its wait on reads, and the wait scales with the bytes read over
the drive's rate — fitted on the two 64 GB rows above, and checked against the 128 GB row, which it reproduces. Past
about 5 GB/s the computing dominates: 6 GB/s gives ~2.65, and a disk that took no time at all ~3.7. Measured since on
another machine (an RTX 5060 Ti, a four-core Xeon, ▶ above): with 60 GB and a 5 GB/s disk GLM answers at **3.0 tokens a
second**, and holding that disk to 2.5 GB/s costs it 16% — past 5 GB/s the computing dominates there too.

## How much memory a model needs

How much memory a model needs for few misses depends on its router. Counted over 600 tokens of one prompt, 90% of a
layer's picks go to 29% of the 35B's experts, 36% of the 122B's and 41% of GLM's — and each model's cache, holding
about that share, answers close to 90% of its reads from memory: the 35B with 8 GB (about a third of its experts) 90%,
the 122B with 32 GB (about half) 87%, GLM with 64 GB (about 40%) 77-80%. GLM's busiest experts are as concentrated as
the Qwens'; its tail is longer, so it needs a larger share held for the same rate.
