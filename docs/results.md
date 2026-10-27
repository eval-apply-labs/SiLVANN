# Results

Every number on this page was measured on one of these machines, and every row names the version it was measured with.
The README's table of machines is drawn from this page. The `D4E4` packs named below were retired in v0.18.0, and the
397B's pack is no longer published: the R730's rows of a retired pack were measured again on `D8E4` with v0.18.1, the
RTX 5060 Ti's with v0.19.0, and the others stay as they were measured.

| | The R730 (the test server) | The Precision 5820 (a desktop) | A Ryzen desktop |
|---|---|---|---|
| Card | two AMD Instinct MI50 32 GB (1,024 GB/s) | an NVIDIA GeForce RTX 5060 Ti 16 GB (sm_120, 36 SMs, 448 GB/s) | an NVIDIA GeForce RTX 5070 Ti 16 GB on PCIe 4 x8 |
| CPU | two Xeon E5-2680 v4 (28 cores, AVX2) | a Xeon W-2123 (Skylake, four cores, AVX-512) | a Ryzen 7 5800X (AVX2) |
| RAM | 256 GB, 126 GB/s | 62 GB of DDR4-2400 in two channels | 32 GB |
| Disk | NVMe, 5 GB/s (2.6 GB/s where a row says so) | NVMe, 5 GB/s | — |
| Below | [▶ AMD Instinct MI50](#on-the-r730-amd-instinct-mi50) | [▶ the RTX 5060 Ti](#the-rtx-5060-ti) | [▶ the RTX 5070 Ti](#the-rtx-5070-ti) |

Tensor parallelism was measured on a fourth: an EPYC 7532 with four MI50 32 GB on PCIe 4
([▶ below](#over-several-cards-tensor-parallelism)).

**How it is measured.** `test/src_runtime_bench.py`: on the card a prompt of about 2,000 positions, then 32-128
tokens; with the experts on the CPU about 1,000 positions, then 128 tokens (32 in GLM's earliest rows). Each token is
the most likely one, and the answer's rate is measured after the prompt, from its 10th token on. "All in RAM" means
every expert is already in memory; the other rows start cold, under the memory limit shown, with the expert files
dropped from the operating system's cache first. A rate in brackets is on a short prompt, from the check against
Hugging Face's answer. A row of an older version has not been measured again since; the rows of v0.3 were measured with
its card kernels, and where a card does part of the work they are a floor.

**Through the server**, which the chat's meter shows, each token is drawn with the model's sampler — its repetition
penalty, temperature, top k and top p — on the card, and the draw costs a token next to nothing: the server answers at
the rates below. Up to v0.11.0 that held on AMD cards through ROCm alone. Elsewhere the draw's top k ran nn's
generic body, which reads the whole vocabulary once for each of the `k` picks, and on one CPU lane also asks every value
whether an earlier pick took it. Every family now has its own top k but `khronos_opencl1_igpu`, the same picks and
weights to the bit (v0.11.1):

| Qwen 3.6 35B-A3B (248,320 logits, a top 20) | a draw's top k, before → after | the answer drawn as the server draws it, before → after |
|---|---|---|
| the RTX 5060 Ti, its experts in RAM and an 11.25 GB tier | 13.8 ms → 0.024 ms | 32.4-35.0 → 58.4-68.0 tokens/s (Ministral 3 14B `D4`: 35.0-36.9 → 46.1-48.5) |
| one MI50 through OpenCL | 50.2 ms → 0.89 ms | 11.9 → 29.9 tokens/s on a short prompt |
| the CPU alone, the R730 (`D8E4`) | 22.0 ms → 0.034 ms | 11.4 → 16.6 tokens/s on a short prompt |
| the CPU alone, the RTX 5060 Ti's desktop (`D8E4`) | → 0.031 ms | 8.4-8.7 → 10.2-10.9 tokens/s in the chat |
| one MI50 through ROCm | its own already | 99.7 tokens/s drawn, 100.5 greedy, on a short prompt |

## How good each pack is: perplexity

The same text through every pack: the first 16 windows of 512 tokens of WikiText-2's test split, each read from a fresh
conversation, every next token scored from the model's own logits. **Perplexity** compares packs of one model, or of
models sharing a tokenizer (the Qwen ones); **bits per byte** — the same score over the text's bytes instead of its
tokens — compares any two models. Lower is better in both. The original weights' score (bf16, through Hugging Face's own
implementation) is given where the checkpoint fits the R730's memory.

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

## On the R730: AMD Instinct MI50

The runs since v0.7.0 keep the process out of the kernel's NUMA balancing, whose sampling cost a card's run on this
two-socket machine 1.5-3%, and the 35B's prompt with its experts on the CPU 5%.

### Qwen 3.6 35B-A3B

| Where it runs | Prompt | Answer | Version |
|---|---|---|---|
| one MI50 | 379 positions/s | 88 tokens/s | v0.21.0 |
| one MI50, a prompt of 32,033 positions — its cache past the window kept at 8 bits | 132 positions/s | 54.8 tokens/s | v0.21.0 (v0.20.0: 42.0 · 17.8) |
| one MI50 for the dense part, the experts on the CPU, all in RAM | 116 positions/s | 32.1-32.8 tokens/s | v0.20.0 |
| the same, a 4 GB tier on the card and the prompt streamed onto it (`stream`), 2,035 positions | 450 positions/s (232 a chunk at a time) | 37.6 tokens/s | v0.20.0 |
| the same, the experts read from the disk, 8 GB of RAM | 32 positions/s† | 11.9 tokens/s — 90% of expert reads from memory | v0.3 |
| the same, 6 GB of RAM | 31 positions/s† | 10.2 tokens/s — 83% from memory | v0.3 |
| the CPU alone, no card, all in RAM | 65-68 positions/s | 16.1 tokens/s | v0.20.0 |
| the CPU alone, the experts on the CPU read from the disk, 12 GB of RAM (8 GB of it the experts' cache) | 45-46 positions/s† | 11.7-12.7 tokens/s — 94% from memory, 28 MB read a token | v0.20.0 |

v0.19.0's prompt GEMM on an MI50 stages a tile's slice once and four elements a load, and sums a long row's tile 8 rows
by 4 pairs a lane: the 35B reads a 2,000-position prompt at 337 positions a second where v0.18.1 read 226, the 27B at
199 where it read 107 — the answers the same words. v0.20.0 takes the grouped causal scores in tiles too: the 27B 206,
the 35B 379 — the 35B's on the v0.20.0 tree in its container and on the host, interleaved, the same words
(`measurements/2026-10-26_nn119_v0200_image_node03.md`; the 27B's `measurements/2026-10-24_nn114_tile_staging_node03.md`).
The 32,033-position row, v0.21.0's TurboQuant passes over the cache's 8-bit tier in slices against v0.20.0's, each
installed: `measurements/2026-10-27_v0210_mi50_rows_node03.md`.
Streamed, a layer's experts cross to the card once for the whole prompt, and the card computes them all
(`measurements/2026-10-25_nn115_qwen_router_rows_node03.md`).

### Qwen 3.8 27B

| Where it runs | Prompt | Answer | Version |
|---|---|---|---|
| one MI50 | 206 positions/s | 31.8 tokens/s | v0.20.0 (the answer v0.19.0's) |
| one MI50, a prompt of 32,098 positions — its cache past the window kept at 8 bits | 69.1 positions/s | 21.3 tokens/s | v0.21.0 (v0.20.0: 25.3 · 9.95) |
| one MI50, a prompt of 100,081 positions | 34.7 positions/s | 13.4 tokens/s | v0.21.0 |
| one MI50 through OpenCL | 52.5 positions/s | 10.4 tokens/s (13.1 on a short prompt) | v0.5.1 |
| three stages over machines (pipeline parallel) | — | 6.8 tokens/s, the same tokens | v0.3 |

### Ministral 3 14B

| Pack | Where it runs | Prompt | Answer | Version |
|---|---|---|---|---|
| D4 | one MI50 | 162 positions/s | 42 tokens/s (56 on a short prompt) | v0.7.0 |
| D8 | one MI50 | 158 positions/s | 30.6 tokens/s (38 on a short prompt) | v0.7.0 |

Ministral reads its prompt as rows, through nn's attention over many rows (v0.4.0, a position at a time, read 53
positions a second on the D4 pack). Its chat template adds about 560 positions, so its rows read about 2,600, and a short
prompt about 620.

### Qwen 3.5 122B-A10B

| Pack | Where it runs | Prompt | Answer | Version |
|---|---|---|---|---|
| `D8E4` | one MI50 for the dense part, the experts on the CPU, all in RAM | 48-50 positions/s | 12.7-15.4 tokens/s | v0.20.0 |
| `D8E4` | the same, 24 GB of the card the tier of experts, held to 80 GB of RAM | 91-94 positions/s | 25.0 tokens/s | v0.20.0 |
| `D8E4` | the same, the experts read from the disk, 32 GB of RAM | 33.4 positions/s† | 7.8 tokens/s — 85% from memory, 257 MB read a token | v0.20.0 |
| `D8E4` | the same, 16 GB of RAM | 31.2 positions/s† | 5.6-5.9 tokens/s — 72% from memory, 492 MB read a token | v0.20.0 |
| `D8E4` | one MI50 through OpenCL 2.0, the experts on the CPU, all in RAM | 36-37 positions/s | 10.3 tokens/s | v0.20.0 |
| `D8E4` | the same, 18 GB of the card the tier of experts (OpenCL allocates at most 85% of the card at once) | 49-51 positions/s | 13.5 tokens/s | v0.20.0 |

The `D8E4` rows were measured again with v0.20.0, two runs each (`measurements/2026-10-26_v0200_r730_rows_node03.md`),
as v0.18.1's were when the `D4E4` pack was retired (`measurements/2026-10-23_nn111_d8e4_rows_node03.md`): the prompts
read 11-63% faster, the answers within a few percent — from the disk at 32 GB 5% below, the same reads a token. The two
runs without a tier answered at 12.7 and 15.4, as v0.18.1's did at 12.9 and 15.6. With
v0.3.0, which brought the tier, the 122B `D4E4` answered at 10.9 tokens a second without it and 19.9 with 24 GB of it,
measured side by side.

### GLM 5.3 Flash

One MI50 for the dense part, the experts on the CPU.

| Experts | Disk | Prompt | Answer | Version |
|---|---|---|---|---|
| `D8E4`: all in RAM, a 300,000-position cache and 17.75 GB of the card the tier of experts — the setup chosen at boot | — | 54 positions/s | 10.8 tokens/s | v0.20.0 |
| `D8E4`: all in RAM, a 300,000-position cache and 6 GB of tier, lent 3.1 GB of the cache's unreached rows — 20.1 GiB of the card in all | — | 50.4 positions/s | 9.1-9.6 tokens/s | v0.20.0 |
| `D8E4`: the same RAM and cache on a 16 GB card's room (16 GiB of the MI50 held) — a 1.5 GB tier, lent 3.1 GB, the setup chosen at boot | — | 49.6 positions/s | 9.2 tokens/s | v0.20.0 |
| `D8E4`: all in RAM, a 10,020-position prompt streamed onto the card (`stream`) | — | 138 positions/s (45 a chunk at a time) | — | v0.20.0 |
| `D8E4`: the same over two MI50s' tiers | — | 152 positions/s | — | v0.20.0 |
| `D8E4`: over two MI50s (`"num_cards": 2`), all in RAM, 14 GB of each card a tier, the model's whole 1,048,576-position context — the setup chosen at boot; in the chat 12.7-14.5 tokens/s | — | 54.6-56.4 positions/s | 11.2-11.3 tokens/s | v0.21.0 |
| `D8E4`: the same with the tier exclusive (`card_exclusive`); in the chat 11.3-11.9 | — | 40.9-41.1 positions/s | 10.1-10.3 tokens/s | v0.21.0 |
| `D8E4`: the same, held to 120 GiB — a 128 GB machine's room: 114 GB of the experts in RAM, the cards' 28 GB not; in the chat 11.4-11.8 | — | 36.3 positions/s | 10.1 tokens/s | v0.21.0 |
| `D8E4`: the CPU alone, no card, the experts read from the disk, 64 GB of RAM (held to 60 GB); the answer from token 200 of 300 | 6.6 GB/s array | 6.4 positions/s | 1.4 tokens/s — 74% from memory, 1.1 GB read a token | v0.21.2 |
| `D8E4`: the same, 32 GB of RAM (held to 30 GB) | 6.6 GB/s array | 6.8 positions/s | 1.2 tokens/s — 49% from memory, 2.1 GB read a token | v0.21.2 |
| all in RAM | — | 23.9 positions/s | 6.7 tokens/s | v0.3 |
| all in RAM, 18 GB of the card the tier of experts | — | 25.0 positions/s | 8.1 tokens/s (6.8 without, side by side) | v0.3.0 |
| all in RAM, 24 GB of the card the tier of experts, from token 224 | — | — | 10.7 tokens/s | v0.3.1 |
| 128 GB of RAM | 5 GB/s | 13.2 positions/s† | 5.7 tokens/s — 97% from memory, 132 MB read a token | v0.3 |
| 128 GB of RAM | 2.6 GB/s | — | 4.2 tokens/s — 97% from memory ◦ | before v0.3 |
| 64 GB of RAM | 2.6 GB/s | — | 1.9 tokens/s — 79% from memory ◦ | before v0.3 |
| 64 GB of RAM | 5 GB/s, overheating‡ | 4.9 positions/s† | 1.55 tokens/s — 80% from memory, 804 MB read a token ◦ | before v0.3 |
| 32 GB of RAM, before grouped attention | 2.6 GB/s | 11 positions/s† | 1.1 tokens/s — 61% from memory ◦ | v0.2.0 |

The first two rows were held to 150 GB of RAM, a prompt of 1,000 positions, then 128 tokens. The two-card rows: the machine's 251 GB, a
prompt of 2,037 positions, then 64 tokens, the chat's meter over three answers of 256, two rounds each (`measurements/2026-10-27_v0210_glm_two_cards_node03.md`). The CPU-alone rows: a 1,020-position prompt, then 300 tokens, the
array's drives slowing themselves when hot as DeepSeek's below; GLM's dense part takes 9.6 GB of the RAM there, so 30 GB
is its least (`measurements/2026-10-27_nn133_deepseek_disk_node03.md`). The streamed rows: each
layer's experts cross to the cards once for the whole prompt (`measurements/2026-10-25_nn115_group_stream_node03.md`; a
chunk at a time `measurements/2026-10-25_nn115_glm_prompt_kernels_node03.md`).
With its experts on the CPU, a model reads a prompt a chunk of positions at a time through each layer — 256 for the
35B, 1,024 for GLM — so each expert is read about once a chunk and runs once over every position that picked it.
† Measured from a cold start, so reading the prompt is also when the experts are first read from the disk into
memory; the answer that follows runs on what the prompt left in memory.
◦ Measured before v0.3's card kernels; the card's share of a GLM token has since become faster, so they are a floor.
‡ See below: the disk sits where the case's air does not reach it, and slows itself down within a minute of
sustained reading.

The tier of experts starts empty in each of these runs; the prompt fills most of it, and the answer's rate climbs as the
rest fills — GLM's reaches about 9 tokens a second by its 240th token with 18 GB. Each tier run answers with the same
words as its run without, until a near-tie in the 122B's ~20 words in: the card sums its experts in another order than
the CPU does.

### DeepSeek-V4.1-Flash

Both MI50s, every layer split over them (`"num_cards": 2`), the experts on the CPU and every one of them in RAM. Its
routed experts are 272 GB, more than the R730's RAM: 22.5 GB of each card is an exclusive tier of them — an expert on a
card or in RAM, never both — the most a card holds beside the model's whole 1,048,576-position context, so each card's
share is RAM given back and all 40 layers' experts fit (on one card the server's setup keeps 34 layers' in RAM and maps
6 from the disk). The config:

```json
{"max_context": 1048576, "num_cards": 2,
 "options": {"experts_from": "ram", "card_experts_gb": 22.5, "card_exclusive": true}}
```

| Where it runs | In the chat | Prompt | Answer | Version |
|---|---|---|---|---|
| two MI50s, every expert in RAM, 22.5 GB of each card an exclusive tier, a 1,048,576-position context | 10.5-13.7 tokens/s (6.8-9.6 the first answer after a boot) | 21.8-23.7 positions/s | 6.5-7.9 tokens/s | v0.21.0 |

Two rounds, interleaved: the server on that config, the chat's meter over three answers of 256 tokens; then a prompt of
2,029 positions and the most likely token after it, from the 10th. The first answer after a boot is the slowest while
the tiers learn which experts are picked; the answer after the long prompt is slower than the chat's, its tiers having
seen the prompt's picks alone. The server boots it in about 6.5 minutes
(`measurements/2026-10-27_v0210_deepseek_rates_node03.md`).

**One card, the experts from the disk** (v0.21.2). Where RAM does not hold its 272 GB of experts, the server reads them
from the disk into a cache in RAM, an expert at a time straight from the pack (`experts_from: "disk"`, chosen at boot).
A 16 or 12 GB card's room is an MI50 with the rest of it held, and the CPU alone no card at all — its dense part in the
same RAM; the process held to the RAM a row names (14, 30 or 60
GB: about what a machine of 16, 32 or 64 GB leaves it); a prompt of 1,012 positions, then 300 tokens, the answer from
token 200; the range of two or three runs:

| Card | RAM | Prompt | Answer | Version |
|---|---|---|---|---|
| 16 GB | 64 GB | 9.8-11.0 positions/s | 2.3-2.8 tokens/s | v0.21.2 |
| 16 GB | 32 GB | 7.4-7.6 positions/s | 2.4-2.6 tokens/s | v0.21.2 |
| 16 GB | 16 GB | 7.1-7.7 positions/s | 1.4-1.8 tokens/s | v0.21.2 |
| 12 GB | 64 GB | 8.2 positions/s | 2.7 tokens/s | v0.21.2 |
| 12 GB | 32 GB | 7.7 positions/s | 2.4 tokens/s | v0.21.2 |
| none: the CPU alone | 64 GB | 3.4 positions/s | 1.2 tokens/s | v0.21.2 |
| none: the CPU alone | 32 GB | 3.1 positions/s | 0.7 tokens/s | v0.21.2 |
| 32 GB (one MI50) | 251 GB, the cache all of it | 14.3-14.5 positions/s | 4.7-5.0 tokens/s | v0.21.2 |
| 32 GB (one MI50) | 251 GB as v0.21.1 placed it: 34 layers in RAM, 6 mapped | 11.0-13.5 positions/s | 5.3-5.8 tokens/s | v0.21.1 |

The drives — the R730's six NVMe drives in one array — slow themselves when they are hot, and reached 79-81 °C in
every run: the spread between a row's lowest and highest figure is that. With the same RAM the disk reads a prompt and
answers about 7 times as fast as the layers mapped through the system's page cache (v0.21.1's way); the server starts
in about 20 seconds, where copying 34 layers into RAM took 6 minutes. A 16 GB card's answer waits about a quarter of a
second a token on the disk, a layer's reads after another's — the next lever is reading a layer's experts while the
one before it computes (`measurements/2026-10-27_nn133_deepseek_disk_node03.md`).

### Over several cards: tensor parallelism

The 27B `D4`'s layers split over the cards of one machine (`"tensor_parallel"`, v0.11.0), on an EPYC 7532 with four
MI50 32 GB on PCIe 4, each held to 155 W; a prompt of 2,097 positions, then 32 tokens:

| Cards | Prompt | Answer |
|---|---|---|
| one | 88 positions/s | 30.4 tokens/s |
| two | 158 positions/s (1.8 times) | 38.4 tokens/s |
| four | 194 positions/s (2.2 times) | 37.2 tokens/s |

## On NVIDIA cards

### The RTX 5060 Ti

In the Precision 5820, every row measured again with v0.20.0 on 2026-10-26 unless its version says otherwise
(`measurements/2026-10-26_v0200_precision_map.md`); the 122B's on `D8E4` with v0.19.0 on 2026-10-24, two rounds each
(`measurements/2026-10-24_nn111_d8e4_rows_precision.md`); every row with a tier fed from RAM measured again with v0.21.0
on 2026-10-27, as its candidate installs (`measurements/2026-10-27_v0210_tier_rows_precision.md`). With 62 GB the server reads the 122B's experts from the disk
unless its config says `"experts_from": "ram"`: they (54 GB), the server's own memory beside them and the spare miss what
the system has free by under a gigabyte. `D8E4`'s dense part takes 4.66 GB of the card against the retired `D4E4`'s 2.37,
so with its experts in RAM its tier is smaller and it answers about 15% below `D4E4`'s 20.4 in the chat and 18.2 after a
prompt; from the disk the two are within 2-4%.

**In the chat.** The server as it starts with the model's own config — its sampler, the setup it chooses for the
machine — and the answer's rate as the chat's meter reads it: a new conversation, a story of 512 tokens (256 for the
122B, GLM and the CPU alone), from its second token on; three answers a boot, two boots (one for the 122B and GLM), the
process held to the RAM a row names with no swap, a smaller card's room the rest of the card held. Its conversations are
short — from 54 positions, Ministral's from 583 (its own system prompt) — so it answers faster than the rows after a
prompt below.

| Model | The machine | The setup the server chooses | In the chat |
|---|---|---|---|
| Ministral 3 14B `D4` | the card, 62 GB of RAM | on the card | 44.8 tokens/s |
| Ministral 3 14B `D8` | the same | on the card | 27.7 tokens/s |
| Qwen 3.8 27B `D4` | the same | on the card, a 35,840-token context | 24.5 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the same | the dense part on the card, the experts on the CPU in RAM, an 11.75 GB tier | 72.0 tokens/s (62 the first answer after a boot; v0.21.0) |
| Qwen 3.6 35B-A3B `D8E4` | the same, a config with `"card_experts": "no"` | the experts on the CPU in RAM, no tier | 30.4 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the same, a config with `"card": "cpu"` | the CPU alone | 9.2 tokens/s |
| Qwen 3.5 122B-A10B `D8E4` | the same | the experts on the CPU, read from the disk | 9.7 tokens/s (v0.19.0) |
| Qwen 3.5 122B-A10B `D8E4` | the same, a config with `"experts_from": "ram"`, held to 61 GB | the experts on the CPU in RAM (54 GB), a 5 GB tier lent 2.9 GB of the cache's rows (7.75 GB at a context of 32,768: 17.2 with v0.19.0) | 17.5 tokens/s (v0.21.0) |
| GLM 5.3 Flash `D8E4` | the same | the experts on the CPU, read from the disk | 3.3 tokens/s |
| Qwen 3.6 35B-A3B `D8E4` | the card, 28 GB of RAM | as with 62 GB | 71.7 tokens/s (62-63 the first; v0.21.0) |
| Qwen 3.6 35B-A3B `D8E4` | 11.6 GB of the card free — a 12 GB card's room — 62 GB of RAM | the experts in RAM, a 7.5 GB tier | 69.9 tokens/s (62 the first; v0.21.0) |
| Qwen 3.6 35B-A3B `D8E4` | 7.8 GB of the card free — an 8 GB card's room — 32 GB of RAM | the experts in RAM, a 4 GB tier | 62.3 tokens/s (58 the first; v0.21.0) |
| Ministral 3 14B `D4` | the same | on the card, a 4,096-token context | 44.9 tokens/s |
| Qwen 3.5 122B-A10B `D8E4` | 7.5 GB of the card free, 32 GB of RAM | the experts on the CPU, read from the disk, a context of 33,792 | 8.3 tokens/s (v0.19.0) |
| Qwen 3.5 122B-A10B `D8E4` | the card, 28 / 32 GB of RAM | the same | 7.8 / 8.3 tokens/s (v0.19.0) |
| GLM 5.3 Flash `D8E4` | the card, 28 / 32 / 60 GB of RAM | the experts on the CPU, read from the disk | 2.4 / 2.5 / 3.25 tokens/s |

**After a prompt.** A prompt of about 2,000 positions (2,568 for Ministral), then 64 tokens (32 on the CPU alone, 128
where a row says so), every expert in RAM; the answer's rate from its 10th token on, and on a short prompt from the
check against Hugging Face's answer; two rounds, their mean.

| Model | Where it runs | Prompt | Answer | Version |
|---|---|---|---|---|
| Ministral 3 14B `D4` | the card | 952 positions/s | 40.4 tokens/s (48.4 on a short prompt) | v0.20.0 |
| Ministral 3 14B `D8` | the card | 868 positions/s | 25.9 tokens/s (29.0) | v0.20.0 |
| Qwen 3.8 27B `D4` | the card | 562 positions/s | 24.0 tokens/s (25.0) | v0.20.0 |
| Qwen 3.8 27B `D4` | the card, a prompt of 6,033 positions — past the 4,352 the cache keeps at full precision | 486 positions/s | 22.2 tokens/s | v0.20.0 |
| Qwen 3.6 35B-A3B `D8E4` | the dense part on the card, the experts on the CPU, the context of 32,768 its config asks and an 11.75 GB tier — the server's setup; 128 tokens | 194 positions/s | 59.4 tokens/s | v0.21.0 |
| Qwen 3.6 35B-A3B `D8E4` | the same with a context of 262,144 — a 9.5 GB tier, lent 2.44 GB more of the cache's rows; 128 tokens | 193 positions/s | 59.4 tokens/s | v0.21.0 |
| Qwen 3.6 35B-A3B `D8E4` | the same, after a prompt of 16,037 positions | 227 positions/s | 43.7 tokens/s | v0.21.0 |
| Qwen 3.6 35B-A3B `D8E4` | the same, after a prompt of 32,033 positions | 207 positions/s | 32.4 tokens/s | v0.21.0 |
| Qwen 3.6 35B-A3B `D8E4` | the dense part on the card, the experts on the CPU, no tier | 61.3 positions/s | 29.4 tokens/s | v0.20.0 |
| Qwen 3.6 35B-A3B `D8E4` | the CPU alone | 28.0 positions/s | 8.7 tokens/s (9.25) | v0.20.0 |
| Qwen 3.6 35B-A3B `D8E4` | the CPU alone with AVX2 (`SILVANN_CPU_AVX512=0`), a prompt of 145 positions | 17.1 positions/s | 1.8 tokens/s | v0.20.0 |
| Qwen 3.6 35B-A3B `D8E4` | the card through OpenCL 1.2 (`khronos_opencl1_igpu`), the experts on the CPU in RAM, no tier; 128 tokens | 37.1 positions/s | 23.0 tokens/s | v0.20.0 |
| Qwen 3.8 27B `D4` | the card through OpenCL 1.2; 128 tokens | 15.1 positions/s | 14.4 tokens/s | v0.20.0 |
| Qwen 3.5 122B-A10B `D8E4` | 62 GB of RAM (held to 61): the dense part on the card, the experts on the CPU in RAM (54 GB), a 7.75 GB tier — the setup chosen with the experts in RAM at 32,768; 128 tokens | 56.2 positions/s | 15.9 tokens/s (six rounds, one at 45.7 · 12.8) | v0.21.0 |
| Qwen 3.5 122B-A10B `D8E4` | the same, the setup chosen at 32,768 as it is: the experts read from the disk; 128 tokens | 20.5 positions/s | 10.0 tokens/s — 99% from memory | v0.19.0 |

Against one MI50 the 27B reads a prompt 5.5 times as fast here (562 positions a second against 102) — the card's tensor
cores run the prompt's matrix products — and answers more slowly (24.0 tokens a second against 31.8): a token reads every
weight once, so the answer follows the memory's speed, 448 GB/s here against the MI50's 1,024. With the experts on the
CPU the four cores are what limits the 35B (61 positions a second against the R730's 88, whose 28 cores read the
experts). With a tier the 35B reads a prompt 1.7 times as fast as v0.11.1 did (194 positions a second against 115).
v0.21.0 no longer pins the experts' RAM for the card (`experts_register`), which v0.20.0 did: the prompt 194 against
182, 173 against 156 with 28 GB, 164 against 147 on an 8 GB card's room, and the 122B's 56 against 44 — and the answer
after a short prompt 1-4% lower (59.4 against 60.9, 52.2 against 54.6 on the 8 GB room), after 16,000 positions and
in the chat within 1-2%. Which experts the tier holds follows when its copies land, so a tier's rate varies
from run to run, and over an answer it warms: 60.9 tokens a second from the 10th token of 128, 59.2 over all of them.
Through OpenCL 1.2 the card decodes the embedding on the CPU and multiplies matrices by vectors in that family's own
kernels, but reads a prompt a position at a time — it is the family for a card without OpenCL 2.0; through CUDA the
same card answers 1.3 to 2.6 times as fast. Before v0.5.0's kernels, through the library's generic ones, the 27B read a
prompt at 19 positions a second and answered at 7.8 tokens a second; Ministral `D4` read its prompt a position at a time
at 34.

**Less RAM, or less of the card.** Each held to the RAM a row names, allowed no swap; the experts on the CPU from the
disk where a row says so, a prompt of about 1,000 positions from a cold start, then 128 tokens:

| Model | Where it runs | Prompt | Answer | Version |
|---|---|---|---|---|
| Qwen 3.6 35B-A3B `D8E4` | 28 GB of RAM: the experts on the CPU in RAM, 8 GB of the card the tier | 174 positions/s | 59.0 tokens/s (six rounds, one at 147 · 56.9) | v0.21.0 |
| Qwen 3.6 35B-A3B `D8E4` | 11.6 GB of the card free — a 12 GB card's room: the experts in RAM, a 6.75 GB tier, a prompt of 2,000 | 113 positions/s | 51.8 tokens/s (45.5 with AVX2) | v0.11.0 |
| Qwen 3.6 35B-A3B `D8E4` | 7.8 GB of the card free — an 8 GB card's room — and 32 GB of RAM: the experts in RAM, a 4 GB tier, the setup chosen there | 164 positions/s | 52.2 tokens/s | v0.21.0 |
| Ministral 3 14B `D4` | the same card's room: on the card, a 4,096-token context (2,568 positions) | 948 positions/s | 40.4 tokens/s | v0.20.0 |
| Qwen 3.5 122B-A10B `D8E4` | 8 GB of the card held, 32 GB of RAM: the experts read from the disk | 19.0 positions/s | 7.8 tokens/s — 86% from memory | v0.19.0 |
| Qwen 3.5 122B-A10B `D8E4` | 32 GB of RAM, the experts read from the disk | 19.0 positions/s | 7.8 tokens/s — 86% from memory, 249 MB read a token | v0.19.0 |
| Qwen 3.5 122B-A10B `D8E4` | 28 GB of RAM, the same | 18.8 positions/s | 7.5 tokens/s — 83% from memory, 287 MB read a token | v0.19.0 |
| GLM 5.3 Flash `D8E4` | 60 GB of RAM, the same | 10.8 positions/s | 3.1 tokens/s — 78% from memory | v0.20.0 |
| GLM 5.3 Flash `D8E4` | 32 GB of RAM, the same | 11.0 positions/s | 2.3 tokens/s — 61% from memory | v0.20.0 |
| GLM 5.3 Flash `D8E4` | 28 GB of RAM, the same | 11.1 positions/s | 2.2 tokens/s — 58% from memory | v0.20.0 |

Each card's room is a small program holding the rest of the card; 7.83 GB left is what a headless 8 GB card leaves,
and Ministral `D4` at 4,096 positions holds 7.7 GB of it, so on an 8 GB card that also drives a display it does not
start. On an 8 GB card's room the 35B's tier is 4 GB now (3.5 GB with v0.11.1, 48.8 tokens a second; 2 GB with v0.8.0,
42.2). The 12 GB room's rows ran with the experts in the 62 GB of RAM, and with AVX2's bodies alone (as a CPU without
AVX-512 has) the 35B answers 12% slower. Holding the disk to 5 GB/s changed neither the 122B's nor GLM's answer; holding
it to 2.5 GB/s slowed GLM's by 16% and the 122B's prompt by 25%. GLM's answer here is limited by reading its experts
from RAM: the two channels give the four cores at most 36 GB/s, and its rate follows that bandwidth from one core to
four. With 28 GB the 122B and GLM answer 4-5% more slowly than with 32 GB; with the desktop's 62 GB the server keeps the
122B's experts in RAM instead, 18.2 tokens a second after a prompt and 20.4 in the chat, against 8.2 from the disk.

### The RTX 5070 Ti

In a desktop with a Ryzen 7 5800X and 32 GB of RAM; v0.11.0, with the setup it chose at boot.

| Model | Where it runs | Prompt | Answer |
|---|---|---|---|
| Ministral 3 14B `D4` | the card (2,568 positions) | 1,682 positions/s | 59.0 tokens/s (74.6 on a short prompt) |
| Qwen 3.6 35B-A3B `D8E4` | the dense part on the card, the experts on the CPU in RAM, 11 GB of the card the tier (2,035 positions) | 155 positions/s | 68.3 tokens/s |

The card has twice the 5060 Ti's memory bandwidth (896 GB/s against 448, by NVIDIA's figures), and Ministral answers
1.5 times as fast. Through v0.11.0's server, which paid the draw's slow top k, the same PC streamed Ministral at 46.0
tokens a second and the 35B at 39.7; Docker Desktop on Windows on the same PC, read from the chat's meter, at 37 and 22.

**Natively on Windows** (v0.12.0, built from source with CUDA 13.4, the chat's meter): Ministral 3 14B `D4` on the card
at 58 tokens a second, on a PC with the same card, processor and RAM. The same server's reading under Linux on it is
still to be taken; until then it stands beside Linux's harness above (74.6 on a short prompt), which takes the most
likely token and has no sampler or server to pay.

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
the RTX 5060 Ti's desktop (▶ above): with 60 GB and a 5 GB/s disk GLM answers at **3.0 tokens a second**, and holding
that disk to 2.5 GB/s costs it 16% — past 5 GB/s the computing dominates there too.

## How much memory a model needs

How much memory a model needs for few misses depends on its router. Counted over 600 tokens of one prompt, 90% of a
layer's picks go to 29% of the 35B's experts, 36% of the 122B's and 41% of GLM's — and each model's cache, holding
about that share, answers close to 90% of its reads from memory: the 35B with 8 GB (about a third of its experts) 90%,
the 122B with 32 GB (about half) 87%, GLM with 64 GB (about 40%) 77-80%. GLM's busiest experts are as concentrated as
the Qwens'; its tail is longer, so it needs a larger share held for the same rate.
