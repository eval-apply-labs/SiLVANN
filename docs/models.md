# Models

Every model this release runs, every pack of each, and every machine each was measured on: a model's table in the
columns [`results.md`](results.md)'s are in, its card linking that kind of machine there. Which model suits a machine:
[`results.md`](results.md#which-model-for-your-machine).

## The packs

| Model | Pack | Weights | Download |
|---|---|---|---|
| Qwen 3.5 4B (dense) | `D4`, for the CPU alone and small cards | 3.5 GB — 4 bits | [eval-apply/silvann_tq_D4_qwen3.5_4b](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.5_4b) |
| Qwen 3.5 4B | `D8` | 5.3 GB — 8 bits | [eval-apply/silvann_tq_D8_qwen3.5_4b](https://huggingface.co/eval-apply/silvann_tq_D8_qwen3.5_4b) |
| Ministral 3 14B (dense, Mistral AI) | `D4`, for 8 GB cards | 7.4 GB — 4 bits | [eval-apply/silvann_tq_D4_ministral_3_14b](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) |
| Ministral 3 14B | `D8`, for 16 GB cards | 13.5 GB — 8 bits | [eval-apply/silvann_tq_D8_ministral_3_14b](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) |
| Qwen 3.8 27B (dense) | `D4` | 15 GB — 4 bits | [eval-apply/silvann_tq_D4_qwen3.8_27b](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) |
| Qwen 3.6 35B-A3B (mixture of experts, 3B active) | `D8E4` | 19 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) |
| Qwen 3.5 122B-A10B (mixture of experts, 10B active) | `D8E4`, for 12 GB cards | 67 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) |
| GLM 5.3 Flash | `D8E4` | 166 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_glm_5.3_flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) |
| DeepSeek-V4.1-Flash-1017 (384 experts, 6 active; text and pictures) | `D8E4` | 492 GB — experts 4 bits, the rest 8; 203 GB of it the n-gram memory's tables, read from the disk | [eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017](https://huggingface.co/eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017) |
| Qwen-Image 2.1 (pictures from a prompt: a 7B diffusion transformer, Qwen3-VL 8B's text model as its encoder) | `D8` | 18 GB — the transformer and the encoder 8 bits, the picture decoder 16 | [eval-apply/silvann_tq_D8_qwen_image_2.1](https://huggingface.co/eval-apply/silvann_tq_D8_qwen_image_2.1) |

Qwen 3.5 397B-A17B runs too, but its pack is not published: GLM 5.3 Flash and DeepSeek-V4.1-Flash do more at its size.
The `D4E4` packs of the 35B, the 122B and GLM were retired in v0.18.0 — their `D8E4` hold the same 4-bit experts. A
pack keeps each routed expert as a record of its own, one read from the disk
([`users.md`](users.md#5-the-experts-on-the-disk)); a pack downloaded before v0.18.1 is not supported:
download it again.

The name says the bits: `D8E4` is the dense matrices at 8 and the experts at 4; `D4` a dense model at 4. The 8-bit dense
part is the one that pays — ▶ [`compression.md`](compression.md), and how close each pack comes to the original weights
is in [`results.md`](results.md#how-good-each-pack-is-perplexity). `python3 silvann_core.py list-models` lists them all,
and `download-model NAME` fetches one into `models/`.

Each model keeps its own licence, which is in its folder once downloaded: Qwen's and Mistral's Apache-2.0, GLM's and
DeepSeek's MIT.
⚠ Qwen-Image 2.1's is the Qwen Research License Agreement: research and other non-commercial use only. A picture model
runs as a server's model of its own: [`users.md`](users.md#a-picture-model-qwen-image-21).

## Qwen 3.5 4B

A dense model of Qwen 3.5's family — the 27B's, smaller: 32 layers, three Gated DeltaNet layers to one of attention, its
embedding and its head one matrix. For a machine without a card, or a small one: its `D4` answers on four cores of a
2016 server Xeon at 4 tokens a second.

<!-- rates:model Qwen 3.5 4B -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8` · on the card | 964 positions/s | 75.8 tokens/s after 2k | — | v0.22.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 979 positions/s | 101.9 tokens/s after 2k | — | v0.22.0 |
| [none](results.md#no-graphics-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8` · the CPU alone | 51.4 positions/s | 11.7 tokens/s after 550 | — | v0.22.0 |
| [none](results.md#no-graphics-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · the CPU alone | 18.9 positions/s | 7.95 tokens/s after 8k | 8,320 | v0.22.0 |
| [none](results.md#no-graphics-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · the CPU alone | 55.3 positions/s | 17.8 tokens/s after 550 | — | v0.22.0 |
| [none](results.md#no-graphics-card) | 251 GB | Xeon E5-2680 v4, 4 threads | — | `D8` · the CPU alone | 9.8 positions/s | 2.3 tokens/s after 550 | — | v0.22.0 |
| [none](results.md#no-graphics-card) | 251 GB | Xeon E5-2680 v4, 4 threads | — | `D4` · the CPU alone | 10.1 positions/s | 4.0 tokens/s after 550 | — | v0.22.0 |
| [16 GB · RTX 5060 Ti, OpenCL 1.2](results.md#an-older-card-or-an-integrated-gpu) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 175.9 positions/s | 47.1 tokens/s after 8k | — | v0.22.0 |
<!-- /rates -->

Both packs answer Hugging Face's greedy continuation token for token, the `D4` its logits 0.33 off on average and the
`D8` 0.05 (`measurements/2026-10-28_qwen35_4b_node03.md`). Four of the R730's threads stand in for a small machine's CPU;
a recent four-core one is likely faster.

## Ministral 3 14B

<!-- rates:model Ministral 3 14B -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [8 GB · RTX 5060 Ti, 7.8 GB free](results.md#an-8-gb-card) | 32 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 948 positions/s | 40.4 tokens/s after 2.6k | 4,096 | v0.20.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 952 positions/s | 40.4 tokens/s after 2.6k | — | v0.20.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8` · on the card | 868 positions/s | 25.9 tokens/s after 2.6k | 8,192 | v0.20.0 |
| [16 GB · RTX 5070 Ti](results.md#a-16-gb-card) | 32 GB | Ryzen 7 5800X (8 cores) | — | `D4` · on the card | 1,682 positions/s | 59.0 tokens/s after 2.6k | — | v0.11.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 162 positions/s | 42 tokens/s after 2.6k | — | v0.7.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8` · on the card | 158 positions/s | 30.6 tokens/s after 2.6k | — | v0.7.0 |
<!-- /rates -->

Ministral reads its prompt as rows, through nn's attention over many rows (v0.4.0, a position at a time, read 53
positions a second on the `D4` pack). Its chat template adds about 560 positions, so its rows read about 2,600, and a
short prompt about 620. On an 8 GB card at 4,096 positions it holds 7.7 GB of the 7.83 a headless card leaves, so on an
8 GB card that also drives a display it does not start. The RTX 5070 Ti has twice the 5060 Ti's memory bandwidth (896
GB/s against 448, by NVIDIA's figures), and Ministral answers 1.5 times as fast there; natively on Windows (v0.12.0, CUDA
13.4, the chat's meter) it answered at 58 tokens a second on a PC with the same card, processor and RAM. In the chat
([`results.md`](results.md#in-the-chat)) `D4` answers at 44.8 tokens a second and `D8` at 27.7 on the RTX 5060 Ti.

## Qwen 3.8 27B

<!-- rates:model Qwen 3.8 27B -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 16 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card, the context the card holds | 221.3 positions/s | 14.32 tokens/s after 35k | 35,840 | v0.22.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 562 positions/s | 24.0 tokens/s after 2k | 35,840 | v0.20.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 486 positions/s | 22.2 tokens/s after 6k | 35,840 | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 206 positions/s | 31.8 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 69.1 positions/s | 21.3 tokens/s after 32k | — | v0.21.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 34.7 positions/s | 13.4 tokens/s after 100k | — | v0.21.0 |
| [4× 32 GB · MI50, 155 W](results.md#two-cards-or-more) | — | EPYC 7532 (32 cores) | — | `D4` · every layer over 4 cards | 194 positions/s | 37.2 tokens/s after 2k | — | v0.11.0 |
| [2× 32 GB · MI50, 155 W](results.md#two-cards-or-more) | — | EPYC 7532 (32 cores) | — | `D4` · every layer over 2 cards | 158 positions/s | 38.4 tokens/s after 2k | — | v0.11.0 |
| [16 GB · RTX 5060 Ti, OpenCL 1.2](results.md#an-older-card-or-an-integrated-gpu) | 62 GB | Xeon W-2123 (4 cores) | — | `D4` · on the card | 15.1 positions/s | 14.4 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50, OpenCL 2.0](results.md#an-older-card-or-an-integrated-gpu) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D4` · on the card | 52.5 positions/s | 10.4 tokens/s after 2k | — | v0.5.1 |
<!-- /rates -->

Against one MI50 the 27B reads a prompt 2.7 times as fast on the RTX 5060 Ti (562 positions a second against 206) — the
card's tensor cores run the prompt's matrix products — and answers more slowly (24.0 tokens a second against 31.8): a
token reads every weight once, so the answer follows the memory's speed, 448 GB/s there against the MI50's 1,024. v0.19.0's
prompt GEMM on an MI50 stages a tile's slice once and four elements a load: the 27B reads a 2,000-position prompt at 199
positions a second where v0.18.1 read 107, and v0.20.0's tiled grouped scores 206 — the same words
(`measurements/2026-10-24_nn114_tile_staging_node03.md`). Past 32,000 positions its cache beyond the window is kept at 8
bits; v0.21.0's TurboQuant passes over it in slices (32,098 positions: 69.1 · 21.3 against v0.20.0's 25.3 · 9.95;
`measurements/2026-10-27_v0210_mi50_rows_node03.md`). Over two or four MI50s of one machine its layers split
(`"tensor_parallel"`, v0.11.0): a prompt 1.8 and 2.2 times as fast as one card's 88. Through OpenCL 1.2 a card reads a
prompt a position at a time; through CUDA the same card answers 1.3 to 2.6 times as fast. Three machines in a pipeline
(v0.3) answered at 6.8 tokens a second, the same tokens. In the chat on the RTX 5060 Ti: 24.5 tokens a second, a
35,840-token context.

## Qwen 3.6 35B-A3B

<!-- rates:model Qwen 3.6 35B-A3B -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [none](results.md#no-graphics-card) | 12 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array | `D8E4` · the experts from the disk, 8 GB of them in RAM | 45-46 positions/s | 11.7-12.7 tokens/s after 1k | — | v0.20.0 |
| [none](results.md#no-graphics-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every expert in RAM | 65-68 positions/s | 16.1 tokens/s after 2k | — | v0.20.0 |
| [none](results.md#no-graphics-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · every expert in RAM | 28.0 positions/s | 8.7 tokens/s after 2k | — | v0.20.0 |
| [8 GB · RTX 5060 Ti, 7.8 GB free](results.md#an-8-gb-card) | 32 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 1.75 GB tier | 94.9 positions/s | 14.34 tokens/s after 100k | 262,144 | v0.22.0 |
| [8 GB · RTX 5060 Ti, 7.8 GB free](results.md#an-8-gb-card) | 32 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 4 GB tier | 164 positions/s | 52.2 tokens/s after 2k | — | v0.21.0 |
| [12 GB · RTX 5060 Ti, 11.6 GB free](results.md#a-12-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 6.75 GB tier | 113 positions/s | 51.8 tokens/s after 2k | — | v0.11.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, an 11.75 GB tier | 194 positions/s | 59.4 tokens/s after 2k | 32,768 | v0.21.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 9.5 GB tier | 193 positions/s | 59.4 tokens/s after 2k | 262,144 | v0.21.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 9.5 GB tier | 207 positions/s | 32.4 tokens/s after 32k | 262,144 | v0.21.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 28 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, an 8 GB tier | 174 positions/s | 59.0 tokens/s after 2k | — | v0.21.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, no tier | 61.3 positions/s | 29.4 tokens/s after 2k | — | v0.20.0 |
| [16 GB · RTX 5070 Ti](results.md#a-16-gb-card) | 32 GB | Ryzen 7 5800X (8 cores) | — | `D8E4` · the experts in RAM, an 11 GB tier | 155 positions/s | 68.3 tokens/s after 2k | — | v0.11.0 |
| [16 GB · RTX 5070 Ti, natively on Windows](results.md#a-16-gb-card) | 32 GB | Ryzen 7 5800X (8 cores) | — | `D8E4` · the server's setup | — | 48 tokens/s in the chat | — | v0.21.2 |
| [16 GB · MI50](results.md#a-16-gb-card) | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, a 10 GB tier | 65.8 positions/s | 25.9 tokens/s after 100k | 262,144 | v0.23.0 |
| [16 GB · MI50](results.md#a-16-gb-card) | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, a 10 GB tier | 236-239 positions/s | 47.9-51.6 tokens/s after 2k | 262,144 | v0.23.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · on the card | 379 positions/s | 88 tokens/s after 2k | — | v0.21.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · on the card | 132 positions/s | 54.8 tokens/s after 32k | — | v0.21.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, no tier | 116 positions/s | 32.1-32.8 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, a 4 GB tier, the prompt streamed | 450 positions/s | 37.6 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 8 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 32 positions/s | 11.9 tokens/s after 1k | — | v0.3 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 6 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 31 positions/s | 10.2 tokens/s after 1k | — | v0.3 |
| [2× 16 GB · MI50](results.md#two-cards-or-more) | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, the experts in RAM, a 12.5 GB tier on each | 85.3 positions/s | 30.5 tokens/s after 100k | 262,144 | v0.23.0 |
| [2× 16 GB · MI50](results.md#two-cards-or-more) | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, the experts in RAM, a 12.5 GB tier on each | 226-228 positions/s | 42.9-45.6 tokens/s after 2k | 262,144 | v0.23.0 |
| [16 GB · RTX 5060 Ti, OpenCL 1.2](results.md#an-older-card-or-an-integrated-gpu) | 62 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, no tier | 37.1 positions/s | 23.0 tokens/s after 2k | — | v0.20.0 |
<!-- /rates -->

With a tier — part of the card holding the experts used most — the 35B reads a prompt 1.7 times as fast as v0.11.1 did
on the RTX 5060 Ti; v0.21.0 no longer pins the experts' RAM for the card (`experts_register`), which read prompts a
little faster again (194 positions a second against 182) and answered after a short prompt 1-4% lower. Which experts
the tier holds follows when its copies land, so a tier's rate varies from run to run, and over an answer it warms. On an
8 GB card's room the tier is 4 GB (3.5 GB with v0.11.1, 48.8 tokens a second; 2 GB with v0.8.0, 42.2); on a 12 GB card's
room the rows ran with the experts in 62 GB of RAM, and with AVX2's bodies alone (as a CPU without AVX-512 has) it answers
12% slower. Streamed (`stream`), a layer's experts cross to the card once for the whole prompt
(`measurements/2026-10-25_nn115_qwen_router_rows_node03.md`). With its experts from the disk and a small cache the next
layer's experts are read ahead (`experts_ahead`, v0.22.0): with a 3 GB cache 16.4 tokens a second becomes 19.8
(`measurements/2026-10-28_expert_prediction_node03.md`). The CPU alone with only AVX2 (`SILVANN_CPU_AVX512=0`) read a
145-position prompt at 17.1 positions a second and answered at 1.8 tokens. In the chat: 72 tokens a second on the RTX
5060 Ti with its tier, 62.3 on an 8 GB card's room, 69.9 on a 12 GB card's, 30.4 without a tier, 9.2 on the CPU alone.
Over two 16 GB MI50s (`"num_cards": 2`, a 12.5 GB tier on each) it reads a 100,000-token prompt 1.30 times as fast as
one of them alone with its 10 GB tier, and answers after it 1.18 times as fast; after a 2,000-token prompt the one card
is the faster, by about 10% (`measurements/2026-10-29_qwen36_35b_two_16gb_mi50_node03.md`).

## Qwen 3.5 122B-A10B

<!-- rates:model Qwen 3.5 122B-A10B -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [8 GB · RTX 5060 Ti, 8 GB held](results.md#an-8-gb-card) | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 19.0 positions/s | 7.8 tokens/s after 1k | 33,792 | v0.19.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 32 GB | Xeon W-2123 (4 cores) | NVMe array, 5.7 GB/s | `D8E4` · the experts from the disk, a 30 GB cache | 17.6 positions/s | 5.65 tokens/s after 100k | 110,000 | v0.22.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 19.0 positions/s | 7.8 tokens/s after 1k | — | v0.19.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 28 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 18.8 positions/s | 7.5 tokens/s after 1k | — | v0.19.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 61 GB | Xeon W-2123 (4 cores) | — | `D8E4` · the experts in RAM, a 7.75 GB tier | 56.2 positions/s | 15.9 tokens/s after 2k | 32,768 | v0.21.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, no tier | 48-50 positions/s | 12.7-15.4 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 80 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, a 24 GB tier | 91-94 positions/s | 25.0 tokens/s after 2k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 32 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 33.4 positions/s | 7.8 tokens/s after 1k | — | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 16 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 31.2 positions/s | 5.6-5.9 tokens/s after 1k | — | v0.20.0 |
| [32 GB · MI50, OpenCL 2.0](results.md#an-older-card-or-an-integrated-gpu) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · the experts in RAM, an 18 GB tier | 49-51 positions/s | 13.5 tokens/s after 2k | — | v0.20.0 |
<!-- /rates -->

Its `D8E4` rows were measured again with v0.20.0, two runs each (`measurements/2026-10-26_v0200_r730_rows_node03.md`), as
v0.18.1's were when the `D4E4` pack was retired: the prompts read 11-63% faster, the answers within a few percent. Its
dense part takes 4.66 GB of a card against the retired `D4E4`'s 2.37, so with its experts in RAM its tier is smaller. With
62 GB the server reads its experts from the disk unless its config says `"experts_from": "ram"`: they (54 GB), the
server's own memory beside them and the spare miss what the system has free by under a gigabyte (from the disk 10.0
tokens a second after a prompt, 9.7 in the chat; from RAM 15.9 and 17.5). From the disk its cache keeps 83-86% of a
token's experts with 28-32 GB, 72% with 16 GB (249-492 MB read a token). Holding the disk to 5 GB/s did not change its
answer; to 2.5 GB/s it slowed its prompt by 25%. Through OpenCL 2.0 a card allocates at most 85% of itself at once.

## GLM 5.3 Flash

<!-- rates:model GLM 5.3 Flash -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [none](results.md#no-graphics-card) | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 6.4 positions/s | 1.4 tokens/s after 1k | — | v0.21.2 |
| [none](results.md#no-graphics-card) | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 6.8 positions/s | 1.2 tokens/s after 1k | — | v0.21.2 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 14 GB | Xeon W-2123 (4 cores) | NVMe array, 5.7 GB/s | `D8E4` · the experts from the disk, the next layer's read ahead | 9.1 positions/s | 2.21 tokens/s after 100k | 110,000 | v0.22.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 16 GB | Xeon W-2123 (4 cores) | NVMe array (4 drives) | `D8E4` · the experts from the disk, a 12 GiB cache, the next layer's read ahead | — | 2.31 tokens/s after 256 | 32,768 | v0.22.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 60 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 10.8 positions/s | 3.1 tokens/s after 1k | — | v0.20.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 11.0 positions/s | 2.3 tokens/s after 1k | — | v0.20.0 |
| [16 GB · RTX 5060 Ti](results.md#a-16-gb-card) | 28 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 11.1 positions/s | 2.2 tokens/s after 1k | — | v0.20.0 |
| [16 GB · MI50 held to 16 GiB](results.md#a-16-gb-card) | 150 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every expert in RAM, a 1.5 GB tier | 49.6 positions/s | 9.2 tokens/s after 1k | 300,000 | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 9 GB/s | `D8E4` · the experts from the disk, a 30 GB cache, the next layer's read ahead | — | 3.31 tokens/s after 256 | 576 | v0.22.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 9 GB/s | `D8E4` · the experts from the disk, an 8 GB cache, the next layer's read ahead | — | 2.18 tokens/s after 256 | 576 | v0.22.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 160 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every expert in RAM, a 9.5 GB tier | 72.1 positions/s | 9.86 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 150 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every expert in RAM, a 17.75 GB tier | 54 positions/s | 10.8 tokens/s after 1k | 300,000 | v0.20.0 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 128 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | `D8E4` · the experts from the disk | 13.2 positions/s | 5.7 tokens/s after 1k | — | v0.3 |
| [2× 32 GB · MI50](results.md#two-cards-or-more) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, every expert in RAM, a 14 GB tier on each | 54.6-56.4 positions/s | 11.2-11.3 tokens/s after 2k | 1,048,576 | v0.21.0 |
| [2× 32 GB · MI50](results.md#two-cards-or-more) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, every expert in RAM, a 14 GB tier on each | 74.4 positions/s | 11.2 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [2× 32 GB · MI50](results.md#two-cards-or-more) | 120 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, 114 GB of the experts in RAM, 14 GB exclusive tiers | 36.3 positions/s | 10.1 tokens/s after 2k | 1,048,576 | v0.21.0 |
| [6× 32 GB · MI50 over two machines](results.md#two-cards-or-more) | — | 2× Xeon E5-2680 v4 (28 cores) and an EPYC | — | `D8E4` · a ring of six, every expert on the cards | — | 20.1 tokens/s after short | 1,048,576 | v0.22.0 |
<!-- /rates -->

With its experts on the CPU GLM reads a prompt a chunk of 1,024 positions at a time through each layer, so each expert is
read about once a chunk and runs over every position that picked it. A row from the disk starts cold: reading the
prompt is also when the experts are first read into memory. Its rate follows how much of a token's experts the RAM
holds: 58-61% of them with 28-32 GB, 78% with 60 GB, 97% with 128 GB — 2-3 tokens a second until most of them fit, then
about 10. The next layer's experts read ahead (`experts_ahead`, v0.22.0) are +12-19% there: on a 16 GB machine 1.94
tokens a second becomes 2.31 (`measurements/2026-10-28_nn138_glm_16gb_precision.md`). A tier starts empty; the prompt
fills most of it, and the answer climbs as the rest fills. Over two MI50s (`"num_cards": 2`) every layer splits over
both, and an exclusive tier (`card_exclusive`) keeps each expert on a card or in RAM, never both — 28 GB of RAM given back
for a little speed (`measurements/2026-10-27_v0210_glm_two_cards_node03.md`). On the CPU alone its dense part takes 9.6 GB
of the RAM, so 30 GB is its least. In the chat on the RTX 5060 Ti from the disk: 2.4 tokens a second with 28 GB, 2.5 with
32, 3.25 with 60; over two MI50s 12.7-14.5. Its experts on the cards of six MI50s over two machines answer at 20.1
(`measurements/2026-10-28_glm_small_kernels_node03.md`).

## DeepSeek-V4.1-Flash

<!-- rates:model DeepSeek-V4.1-Flash -->
| GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|
| [none](results.md#no-graphics-card) | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 3.4 positions/s | 1.2 tokens/s after 1k | — | v0.21.2 |
| [none](results.md#no-graphics-card) | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 3.1 positions/s | 0.7 tokens/s after 1k | — | v0.21.2 |
| [12 GB · MI50 held to 20 GiB](results.md#a-12-gb-card) | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 8.2 positions/s | 2.7 tokens/s after 1k | — | v0.21.2 |
| [12 GB · MI50 held to 20 GiB](results.md#a-12-gb-card) | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 7.7 positions/s | 2.4 tokens/s after 1k | — | v0.21.2 |
| [16 GB · MI50 held to 16 GiB](results.md#a-16-gb-card) | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 9.8-11.0 positions/s | 2.3-2.8 tokens/s after 1k | — | v0.21.2 |
| [16 GB · MI50 held to 16 GiB](results.md#a-16-gb-card) | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 7.4-7.6 positions/s | 2.4-2.6 tokens/s after 1k | — | v0.21.2 |
| [16 GB · MI50 held to 16 GiB](results.md#a-16-gb-card) | 14 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk | 7.1-7.7 positions/s | 1.4-1.8 tokens/s after 1k | — | v0.21.2 |
| [32 GB · MI50](results.md#a-24-or-32-gb-card) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | `D8E4` · the experts from the disk, every byte of RAM the cache | 14.3-14.5 positions/s | 4.7-5.0 tokens/s after 1k | — | v0.21.2 |
| [2× 32 GB · MI50](results.md#two-cards-or-more) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, every expert in RAM, 22.5 GB exclusive tiers | 31.3 positions/s | 9.72 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [2× 32 GB · MI50](results.md#two-cards-or-more) | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | `D8E4` · every layer over both cards, every expert in RAM, 22.5 GB exclusive tiers | 21.8-23.7 positions/s | 6.5-7.9 tokens/s after 2k | 1,048,576 | v0.21.0 |
<!-- /rates -->

DeepSeek's model reads text and pictures: its vision tower (32 blocks and the aligner, 464 MB in the pack's
`vision.pkl`) runs on the card in about 620 MB of it, a picture at most 1,024 tokens. Its routed experts are 272 GB (6.8 GB a layer,
40 layers) and run on the CPU: in RAM where they all fit beside the server, a spare and a tenth of the memory for the
system and the files it reads in place; otherwise read from the disk into a cache in RAM, an expert at a time straight
from the pack, the ones used most kept. Its n-gram memory's tables (203 GB) stay on the disk and are read a row at a time as
each token asks for them, so the disk should be an NVMe drive. A conversation reaches the model's own length,
1,048,576 positions: past 16,384, layer 20's candidate blocks choose where the later layers look, as DeepSeek's own
code does. On the R730's two MI50s, with every expert in RAM — 22.5 GB of each card an exclusive tier of them, so
the rest fit beside 251 GB — it answers in the chat at 10.5-13.7 tokens a second and reads a prompt at 22-24 positions
a second ([`results.md`](results.md#two-cards-or-more)); its options are in
[`config.md`](config.md#deepseek-v41-flashs-options).

Its routed experts are 272 GB, more than the R730's RAM: over two cards 22.5 GB of each is an exclusive tier of them, the
most a card holds beside the whole context, so all 40 layers' experts fit beside 251 GB. The config:

```json
{"max_context": 1048576, "num_cards": 2,
 "options": {"experts_from": "ram", "card_experts_gb": 22.5, "card_exclusive": true}}
```

The first answer after a boot is the slowest while the tiers learn which experts are picked; the server boots it in
about 6.5 minutes (`measurements/2026-10-27_v0210_deepseek_rates_node03.md`). From the disk the R730's drives slowed
themselves at 79-81 °C in every run, the spread between a row's lowest and highest figure; with the same RAM the disk
reads a prompt and answers about 7 times as fast as the layers mapped through the system's page cache (v0.21.1's way:
34 layers in RAM, 6 mapped, 11.0-13.5 positions a second and 5.3-5.8 tokens; `measurements/2026-10-27_nn133_deepseek_disk_node03.md`).

## The card's tier of experts

With a model's experts on the CPU from RAM, part of the card holds the experts used most — on by default, sized when
the server starts, or `card_experts_gb` in its config pins the size; the
card computes those while the CPU computes the rest, and the CPU copies an expert over while the card works, so a token
never waits for one. The rows above that name a tier are this; they ran on the machine's 256 GB of RAM, and the RAM they
name is what the experts and the process need, not a limit they were held to. For GLM, `card_exclusive` keeps what the
card holds out of RAM — an 18 GB tier gives back about as much RAM, for about a fifth fewer tokens a second on a card that
cannot copy both ways at once. [`users.md`](users.md#the-cards-tier-of-experts) says how to set both.

## How long a conversation fits

The conversation's cache sits on the card from boot. A model's `configs/default.json` says `max_context` `"auto"`: as
many positions as the card holds beside the model, up to the model's own length, found when the server starts. A number
holds it there instead. The Qwen caches keep their first 256 positions and the latest 4,096 at full precision and the
rest at 8 bits; past the first 4,352, each 100 MB of the card holds about 10,000 more tokens of the 35B's conversation
and 3,200 of the 27B's. GLM 5.3 Flash's cache, a latent a position, is every position in halves: about 8,800 a 100 MB. Ministral 3's cache is every position at full precision —
about 625 positions a 100 MB: on a 16 GB card about 32,000 beside its `D4`. DeepSeek-V4.1-Flash keeps a window of 128
positions and compressed latents for the rest: about 90 MB for 16,384 positions, 5.5 GB for its whole length.

With a mixture-of-experts model's experts on the CPU, the whole length costs a short conversation nothing: the card's
tier of experts holds experts in the cache's rows no conversation has reached and gives them back as one grows
([`config.md`](config.md#the-context-and-the-tier-share-the-card)).
