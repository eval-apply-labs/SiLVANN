# Results

Every number on this page was measured, and every row names the version it was measured with. Each table has the same
columns: the card (its memory, and which card), the RAM the process was held to, the CPU, the disk where the experts are
read from it, the model's setup, how fast it reads a prompt (prefill) and answers after one (decode — after the length
the row names, 100,000 positions where the context holds them), the longest conversation the setup holds, and the
version. A model's own page has every machine it was measured on: [`models.md`](models.md).

## Which model for your machine

Find your machine — the card, then the RAM, then the CPU — and the model that suits it; each row links the section below
with every model measured on that kind of machine, and the model's own page.

<!-- rates:lookup -->
| GPU | RAM | CPU | Disk | Model | Setup | Prefill | Decode | Max context | Or |
|---|---|---|---|---|---|---|---|---|---|
| [none](results.md#no-graphics-card) | 16 GB | 4 cores | — | [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | the CPU alone | 18.9 positions/s | 7.95 tokens/s after 8k | 8,320 | — |
| [8 GB](results.md#an-8-gb-card) | 32 GB | 4 cores | — | [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | the experts in RAM, a tier on the card | 94.9 positions/s | 14.34 tokens/s after 100k | 262,144 | the 4B on the card |
| [16 GB](results.md#a-16-gb-card) | 32 GB | 4 cores | NVMe, 5.7 GB/s | [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | the experts from the disk, ~30 GB of them in RAM | 17.6 positions/s | 5.65 tokens/s after 100k | 110,000 | GLM 5.3 Flash from the disk |
| [16 GB](results.md#a-16-gb-card) | 14 GB | 4 cores | NVMe, 5.7 GB/s | [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | the experts from the disk | 9.1 positions/s | 2.21 tokens/s after 100k | 110,000 | — |
| [16 GB](results.md#a-16-gb-card) | 16 GB | 4 cores | — | [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | on the card | 221.3 positions/s | 14.32 tokens/s after 35k | 35,840 | — |
| [2× 16 GB](results.md#two-cards-or-more) | 62 GB | 28 cores | — | [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | every layer over both cards, a tier on each | 85.3 positions/s | 30.5 tokens/s after 100k | 262,144 | the 35B on one card, faster after a short prompt |
| [32 GB](results.md#a-24-or-32-gb-card) | 160 GB | 28 cores | — | [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | every expert in RAM, a tier on the card | 72.1 positions/s | 9.86 tokens/s after 100k | 1,048,576 | DeepSeek-V4.1-Flash over two cards |
| [2× 32 GB](results.md#two-cards-or-more) | 251 GB | 28 cores | — | [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | every expert in RAM, exclusive tiers | 31.3 positions/s | 9.72 tokens/s after 100k | 1,048,576 | GLM 5.3 Flash on one card |
<!-- /rates -->

- **The low end** — a mini PC, an old desktop, no card: the 4B on the CPU, or on an integrated GPU through OpenCL 1.2,
  which should be faster than its CPU and leaves the CPU's cores free — not yet measured on one.
- **An 8 GB card**: Qwen 3.6 35B-A3B, the dense part on the card and the experts in RAM.
- **The middle** — a gaming PC, a 16 GB card and 32 GB of RAM: Qwen 3.5 122B-A10B, its experts read from the disk into
  about 30 GB of RAM; or GLM 5.3 Flash from the disk, slower, with as little as 14 GB.
- **The card alone**: Qwen 3.8 27B, the whole model on a 16 GB card.
- **Two 16 GB cards**: Qwen 3.6 35B-A3B, every layer over both and a tier of its experts on each — faster than one of
  the cards at a long context (after 100k, 30.5 tokens a second against 25.9), slower by about a tenth after a short
  prompt.
- **The top end** — RAM for every expert: GLM 5.3 Flash on one 32 GB card, DeepSeek-V4.1-Flash over two.

## By machine

### No graphics card

The CPU alone: the model in RAM, or its experts read from the disk.

<!-- rates:machine none -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D8` | none | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the CPU alone | 51.4 positions/s | 11.7 tokens/s after 550 | — | v0.22.0 |
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | none | 62 GB | Xeon W-2123 (4 cores) | — | the CPU alone | 18.9 positions/s | 7.95 tokens/s after 8k | 8,320 | v0.22.0 |
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | none | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the CPU alone | 55.3 positions/s | 17.8 tokens/s after 550 | — | v0.22.0 |
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D8` | none | 251 GB | Xeon E5-2680 v4, 4 threads | — | the CPU alone | 9.8 positions/s | 2.3 tokens/s after 550 | — | v0.22.0 |
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | none | 251 GB | Xeon E5-2680 v4, 4 threads | — | the CPU alone | 10.1 positions/s | 4.0 tokens/s after 550 | — | v0.22.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | none | 12 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array | the experts from the disk, 8 GB of them in RAM | 45-46 positions/s | 11.7-12.7 tokens/s after 1k | — | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | none | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every expert in RAM | 65-68 positions/s | 16.1 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | none | 62 GB | Xeon W-2123 (4 cores) | — | every expert in RAM | 28.0 positions/s | 8.7 tokens/s after 2k | — | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | none | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 6.4 positions/s | 1.4 tokens/s after 1k | — | v0.21.2 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | none | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 6.8 positions/s | 1.2 tokens/s after 1k | — | v0.21.2 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | none | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 3.4 positions/s | 1.2 tokens/s after 1k | — | v0.21.2 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | none | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 3.1 positions/s | 0.7 tokens/s after 1k | — | v0.21.2 |
<!-- /rates -->

### An 8 GB card

<!-- rates:machine 8 -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Ministral 3 14B](models.md#ministral-3-14b) `D4` | 8 GB · RTX 5060 Ti, 7.8 GB free | 32 GB | Xeon W-2123 (4 cores) | — | on the card | 948 positions/s | 40.4 tokens/s after 2.6k | 4,096 | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 8 GB · RTX 5060 Ti, 7.8 GB free | 32 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 1.75 GB tier | 94.9 positions/s | 14.34 tokens/s after 100k | 262,144 | v0.22.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 8 GB · RTX 5060 Ti, 7.8 GB free | 32 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 4 GB tier | 164 positions/s | 52.2 tokens/s after 2k | — | v0.21.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 8 GB · RTX 5060 Ti, 8 GB held | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 19.0 positions/s | 7.8 tokens/s after 1k | 33,792 | v0.19.0 |
<!-- /rates -->

### A 12 GB card

<!-- rates:machine 12 -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 12 GB · RTX 5060 Ti, 11.6 GB free | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 6.75 GB tier | 113 positions/s | 51.8 tokens/s after 2k | — | v0.11.0 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 12 GB · MI50 held to 20 GiB | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 8.2 positions/s | 2.7 tokens/s after 1k | — | v0.21.2 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 12 GB · MI50 held to 20 GiB | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 7.7 positions/s | 2.4 tokens/s after 1k | — | v0.21.2 |
<!-- /rates -->

### A 16 GB card

<!-- rates:machine 16 -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Ministral 3 14B](models.md#ministral-3-14b) `D4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 952 positions/s | 40.4 tokens/s after 2.6k | — | v0.20.0 |
| [Ministral 3 14B](models.md#ministral-3-14b) `D8` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 868 positions/s | 25.9 tokens/s after 2.6k | 8,192 | v0.20.0 |
| [Ministral 3 14B](models.md#ministral-3-14b) `D4` | 16 GB · RTX 5070 Ti | 32 GB | Ryzen 7 5800X (8 cores) | — | on the card | 1,682 positions/s | 59.0 tokens/s after 2.6k | — | v0.11.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 16 GB · RTX 5060 Ti | 16 GB | Xeon W-2123 (4 cores) | — | on the card, the context the card holds | 221.3 positions/s | 14.32 tokens/s after 35k | 35,840 | v0.22.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 562 positions/s | 24.0 tokens/s after 2k | 35,840 | v0.20.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 486 positions/s | 22.2 tokens/s after 6k | 35,840 | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, an 11.75 GB tier | 194 positions/s | 59.4 tokens/s after 2k | 32,768 | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 9.5 GB tier | 193 positions/s | 59.4 tokens/s after 2k | 262,144 | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 9.5 GB tier | 207 positions/s | 32.4 tokens/s after 32k | 262,144 | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti | 28 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, an 8 GB tier | 174 positions/s | 59.0 tokens/s after 2k | — | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, no tier | 61.3 positions/s | 29.4 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5070 Ti | 32 GB | Ryzen 7 5800X (8 cores) | — | the experts in RAM, an 11 GB tier | 155 positions/s | 68.3 tokens/s after 2k | — | v0.11.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5070 Ti, natively on Windows | 32 GB | Ryzen 7 5800X (8 cores) | — | the server's setup | — | 48 tokens/s in the chat | — | v0.21.2 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · MI50 | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, a 10 GB tier | 65.8 positions/s | 25.9 tokens/s after 100k | 262,144 | v0.23.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · MI50 | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, a 10 GB tier | 236-239 positions/s | 47.9-51.6 tokens/s after 2k | 262,144 | v0.23.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 16 GB · RTX 5060 Ti | 32 GB | Xeon W-2123 (4 cores) | NVMe array, 5.7 GB/s | the experts from the disk, a 30 GB cache | 17.6 positions/s | 5.65 tokens/s after 100k | 110,000 | v0.22.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 16 GB · RTX 5060 Ti | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 19.0 positions/s | 7.8 tokens/s after 1k | — | v0.19.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 16 GB · RTX 5060 Ti | 28 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 18.8 positions/s | 7.5 tokens/s after 1k | — | v0.19.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 16 GB · RTX 5060 Ti | 61 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, a 7.75 GB tier | 56.2 positions/s | 15.9 tokens/s after 2k | 32,768 | v0.21.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · RTX 5060 Ti | 14 GB | Xeon W-2123 (4 cores) | NVMe array, 5.7 GB/s | the experts from the disk, the next layer's read ahead | 9.1 positions/s | 2.21 tokens/s after 100k | 110,000 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · RTX 5060 Ti | 16 GB | Xeon W-2123 (4 cores) | NVMe array (4 drives) | the experts from the disk, a 12 GiB cache, the next layer's read ahead | — | 2.31 tokens/s after 256 | 32,768 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · RTX 5060 Ti | 60 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 10.8 positions/s | 3.1 tokens/s after 1k | — | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · RTX 5060 Ti | 32 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 11.0 positions/s | 2.3 tokens/s after 1k | — | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · RTX 5060 Ti | 28 GB | Xeon W-2123 (4 cores) | NVMe, 5 GB/s | the experts from the disk | 11.1 positions/s | 2.2 tokens/s after 1k | — | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 16 GB · MI50 held to 16 GiB | 150 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every expert in RAM, a 1.5 GB tier | 49.6 positions/s | 9.2 tokens/s after 1k | 300,000 | v0.20.0 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 16 GB · MI50 held to 16 GiB | 60 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 9.8-11.0 positions/s | 2.3-2.8 tokens/s after 1k | — | v0.21.2 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 16 GB · MI50 held to 16 GiB | 30 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 7.4-7.6 positions/s | 2.4-2.6 tokens/s after 1k | — | v0.21.2 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 16 GB · MI50 held to 16 GiB | 14 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk | 7.1-7.7 positions/s | 1.4-1.8 tokens/s after 1k | — | v0.21.2 |
<!-- /rates -->

### A 24 or 32 GB card

<!-- rates:machine 32 -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D8` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 964 positions/s | 75.8 tokens/s after 2k | — | v0.22.0 |
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 979 positions/s | 101.9 tokens/s after 2k | — | v0.22.0 |
| [Ministral 3 14B](models.md#ministral-3-14b) `D4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 162 positions/s | 42 tokens/s after 2.6k | — | v0.7.0 |
| [Ministral 3 14B](models.md#ministral-3-14b) `D8` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 158 positions/s | 30.6 tokens/s after 2.6k | — | v0.7.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 206 positions/s | 31.8 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 69.1 positions/s | 21.3 tokens/s after 32k | — | v0.21.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 34.7 positions/s | 13.4 tokens/s after 100k | — | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 379 positions/s | 88 tokens/s after 2k | — | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 132 positions/s | 54.8 tokens/s after 32k | — | v0.21.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, no tier | 116 positions/s | 32.1-32.8 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, a 4 GB tier, the prompt streamed | 450 positions/s | 37.6 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 8 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | the experts from the disk | 32 positions/s | 11.9 tokens/s after 1k | — | v0.3 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 32 GB · MI50 | 6 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | the experts from the disk | 31 positions/s | 10.2 tokens/s after 1k | — | v0.3 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, no tier | 48-50 positions/s | 12.7-15.4 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 32 GB · MI50 | 80 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, a 24 GB tier | 91-94 positions/s | 25.0 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 32 GB · MI50 | 32 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | the experts from the disk | 33.4 positions/s | 7.8 tokens/s after 1k | — | v0.20.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 32 GB · MI50 | 16 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | the experts from the disk | 31.2 positions/s | 5.6-5.9 tokens/s after 1k | — | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 9 GB/s | the experts from the disk, a 30 GB cache, the next layer's read ahead | — | 3.31 tokens/s after 256 | 576 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 9 GB/s | the experts from the disk, an 8 GB cache, the next layer's read ahead | — | 2.18 tokens/s after 256 | 576 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 32 GB · MI50 | 160 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every expert in RAM, a 9.5 GB tier | 72.1 positions/s | 9.86 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 32 GB · MI50 | 150 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every expert in RAM, a 17.75 GB tier | 54 positions/s | 10.8 tokens/s after 1k | 300,000 | v0.20.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 32 GB · MI50 | 128 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe, 5 GB/s | the experts from the disk | 13.2 positions/s | 5.7 tokens/s after 1k | — | v0.3 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | NVMe array, 6.6 GB/s | the experts from the disk, every byte of RAM the cache | 14.3-14.5 positions/s | 4.7-5.0 tokens/s after 1k | — | v0.21.2 |
<!-- /rates -->

### Two cards or more

A model's every layer split over the cards of one machine (`num_cards`), or its layers over machines in a ring
(`pipeline_parallelism`).

<!-- rates:machine multi -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 4× 32 GB · MI50, 155 W | — | EPYC 7532 (32 cores) | — | every layer over 4 cards | 194 positions/s | 37.2 tokens/s after 2k | — | v0.11.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 2× 32 GB · MI50, 155 W | — | EPYC 7532 (32 cores) | — | every layer over 2 cards | 158 positions/s | 38.4 tokens/s after 2k | — | v0.11.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 2× 16 GB · MI50 | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, the experts in RAM, a 12.5 GB tier on each | 85.3 positions/s | 30.5 tokens/s after 100k | 262,144 | v0.23.0 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 2× 16 GB · MI50 | 62 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, the experts in RAM, a 12.5 GB tier on each | 226-228 positions/s | 42.9-45.6 tokens/s after 2k | 262,144 | v0.23.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 2× 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, every expert in RAM, a 14 GB tier on each | 54.6-56.4 positions/s | 11.2-11.3 tokens/s after 2k | 1,048,576 | v0.21.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 2× 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, every expert in RAM, a 14 GB tier on each | 74.4 positions/s | 11.2 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 2× 32 GB · MI50 | 120 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, 114 GB of the experts in RAM, 14 GB exclusive tiers | 36.3 positions/s | 10.1 tokens/s after 2k | 1,048,576 | v0.21.0 |
| [GLM 5.3 Flash](models.md#glm-53-flash) `D8E4` | 6× 32 GB · MI50 over two machines | — | 2× Xeon E5-2680 v4 (28 cores) and an EPYC | — | a ring of six, every expert on the cards | — | 20.1 tokens/s after short | 1,048,576 | v0.22.0 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 2× 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, every expert in RAM, 22.5 GB exclusive tiers | 31.3 positions/s | 9.72 tokens/s after 100k | 1,048,576 | v0.22.0 |
| [DeepSeek-V4.1-Flash](models.md#deepseek-v41-flash) `D8E4` | 2× 32 GB · MI50 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | every layer over both cards, every expert in RAM, 22.5 GB exclusive tiers | 21.8-23.7 positions/s | 6.5-7.9 tokens/s after 2k | 1,048,576 | v0.21.0 |
<!-- /rates -->

### An older card or an integrated GPU

Through OpenCL: 2.0 for a card without its vendor's own family, 1.2 for an older card or an integrated GPU
(`khronos_opencl1_igpu`).

<!-- rates:machine opencl -->
| Model | GPU | RAM | CPU | Disk | Setup | Prefill | Decode | Max context | Version |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen 3.5 4B](models.md#qwen-35-4b) `D4` | 16 GB · RTX 5060 Ti, OpenCL 1.2 | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 175.9 positions/s | 47.1 tokens/s after 8k | — | v0.22.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 16 GB · RTX 5060 Ti, OpenCL 1.2 | 62 GB | Xeon W-2123 (4 cores) | — | on the card | 15.1 positions/s | 14.4 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.8 27B](models.md#qwen-38-27b) `D4` | 32 GB · MI50, OpenCL 2.0 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | on the card | 52.5 positions/s | 10.4 tokens/s after 2k | — | v0.5.1 |
| [Qwen 3.6 35B-A3B](models.md#qwen-36-35b-a3b) `D8E4` | 16 GB · RTX 5060 Ti, OpenCL 1.2 | 62 GB | Xeon W-2123 (4 cores) | — | the experts in RAM, no tier | 37.1 positions/s | 23.0 tokens/s after 2k | — | v0.20.0 |
| [Qwen 3.5 122B-A10B](models.md#qwen-35-122b-a10b) `D8E4` | 32 GB · MI50, OpenCL 2.0 | 251 GB | 2× Xeon E5-2680 v4 (28 cores) | — | the experts in RAM, an 18 GB tier | 49-51 positions/s | 13.5 tokens/s after 2k | — | v0.20.0 |
<!-- /rates -->

## The machines

| | The R730 (the test server) | The Precision 5820 (a desktop) | A Ryzen desktop |
|---|---|---|---|
| Card | two AMD Instinct MI50 32 GB (1,024 GB/s) | an NVIDIA GeForce RTX 5060 Ti 16 GB (sm_120, 36 SMs, 448 GB/s) | an NVIDIA GeForce RTX 5070 Ti 16 GB on PCIe 4 x8 |
| CPU | two Xeon E5-2680 v4 (28 cores, AVX2) | a Xeon W-2123 (Skylake, four cores, AVX-512) | a Ryzen 7 5800X (AVX2) |
| RAM | 256 GB, 126 GB/s | 62 GB of DDR4-2400 in two channels | 32 GB |
| Disk | six NVMe drives in one array, ~9 GB/s cool and 6.6 GB/s hot (older rows: one drive, 5 or 2.6 GB/s) | NVMe, 5 GB/s; four drives in one array since 2026-10-28 | — |

Tensor parallelism was measured on a fourth: an EPYC 7532 with four MI50 32 GB on PCIe 4, each held to 155 W.

**How it is measured.** `test/src_runtime_bench.py`: a prompt of the length the decode column names (`2k` about 2,000
positions; the lineup's rows 100,000 where the setup's context holds them), then 32-128 tokens; with the experts on the
CPU about 1,000 positions, then 128 tokens (32 in GLM's earliest rows). Each token is
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

## In the chat

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
GLM with 64 GB does for its whole answer: its rows with a 5 GB/s disk from before v0.3 slowed so. Give an NVMe drive air on the side that carries its controller.

Projected from the R730's rows: with the disk holding 5 GB/s, GLM with 64 GB should answer about **2.5 tokens a
second**. A token there is ~270 ms of computing plus its wait on reads, and the wait scales with the bytes read over
the drive's rate — fitted on two 64 GB runs from before v0.3 (2.6 GB/s 1.9 tokens/s, 5 GB/s 1.55), and checked against the 128 GB row ([GLM 5.3 Flash](models.md#glm-53-flash)), which it reproduces. Past
about 5 GB/s the computing dominates: 6 GB/s gives ~2.65, and a disk that took no time at all ~3.7. Measured since on
the RTX 5060 Ti's desktop ([a 16 GB card](#a-16-gb-card)): with 60 GB and a 5 GB/s disk GLM answers at **3.0 tokens a second**, and holding
that disk to 2.5 GB/s costs it 16% — past 5 GB/s the computing dominates there too.

## How much memory a model needs

How much memory a model needs for few misses depends on its router. Counted over 600 tokens of one prompt, 90% of a
layer's picks go to 29% of the 35B's experts, 36% of the 122B's and 41% of GLM's — and each model's cache, holding
about that share, answers close to 90% of its reads from memory: the 35B with 8 GB (about a third of its experts) 90%,
the 122B with 32 GB (about half) 87%, GLM with 64 GB (about 40%) 77-80%. GLM's busiest experts are as concentrated as
the Qwens'; its tail is longer, so it needs a larger share held for the same rate.
