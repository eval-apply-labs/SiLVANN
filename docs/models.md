# Models

Every model this release runs, every pack of each, and what each needs. The README recommends one model a kind of
machine; this page has the rest.

## The packs

| Model | Pack | Weights | Download |
|---|---|---|---|
| Ministral 3 14B (dense, Mistral AI) | `D4`, for 8 GB cards | 7.4 GB — 4 bits | [eval-apply/Ministral-3-14B_silvann_tq_D4](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D4) |
| Ministral 3 14B | `D8`, for 16 GB cards | 13.5 GB — 8 bits | [eval-apply/Ministral-3-14B_silvann_tq_D8](https://huggingface.co/eval-apply/Ministral-3-14B_silvann_tq_D8) |
| Qwen 3.8 27B (dense) | `D4` | 15 GB — 4 bits | [eval-apply/Qwen3.8-27B_silvann_tq_D4](https://huggingface.co/eval-apply/Qwen3.8-27B_silvann_tq_D4) |
| Qwen 3.6 35B-A3B (mixture of experts, 3B active) | `D8E4` | 19 GB — experts 4 bits, the rest 8 | [eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4) |
| Qwen 3.6 35B-A3B | `D4E4`, for the CPU alone | 19 GB — 4 bits | [eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4](https://huggingface.co/eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4) |
| Qwen 3.5 122B-A10B (mixture of experts, 10B active) | `D8E4`, for 12 GB cards | 67 GB — experts 4 bits, the rest 8 | [eval-apply/Qwen3.5-122B-A10B_silvann_tq_D8E4](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D8E4) |
| Qwen 3.5 122B-A10B | `D4E4`, for 8 GB cards | 65 GB — 4 bits | [eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4](https://huggingface.co/eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4) |
| Qwen 3.5 397B-A17B (mixture of experts, 17B active) | `D4E4` | 207 GB — 4 bits | [eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4](https://huggingface.co/eval-apply/Qwen3.5-397B-A17B_silvann_tq_D4E4) |
| GLM 5.3 Flash (288 experts, 8 active) | `D4E4` | 162 GB — 4 bits | [eval-apply/GLM-5.3-Flash_silvann_tq_D4E4](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D4E4) |
| GLM 5.3 Flash | `D8E4` | 166 GB — experts 4 bits, the rest 8 | [eval-apply/GLM-5.3-Flash_silvann_tq_D8E4](https://huggingface.co/eval-apply/GLM-5.3-Flash_silvann_tq_D8E4) |

The name says the bits: `D8E4` is the dense matrices at 8 and the experts at 4; `D4` a dense model at 4. The 8-bit dense
part is the one that pays — ▶ [`compression.md`](compression.md), and how close each pack comes to the original weights
is in [`results.md`](results.md#how-good-each-pack-is-perplexity). `python3 silvann_core.py list-models` lists them all,
and `download-model NAME` fetches one into `models/`.

Each model keeps its own licence, which is in its folder once downloaded: Qwen's and Mistral's Apache-2.0, GLM's MIT.

## What each needs, measured

From a machine with no graphics card to a model near the frontier on salvaged server parts:

| Your machine | Model, and how it runs | Answer |
|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk (2.5 GB/s) | Qwen 3.6 35B-A3B (`D4E4`) on the CPU alone, its experts read from the disk, 8 GB of them kept in memory | 11.7 tokens/s |
| An 8 GB card | Ministral 3 14B (`D4`) entirely on the card, at a 4,096-token context | 52 tokens/s ◦ |
| A 16 GB card | Qwen 3.8 27B entirely on the card, at a 32,000-token context (14 GB) | 20.3 tokens/s (22.1 on a short prompt) |
| A 16 GB card | Ministral 3 14B (`D8`) entirely on the card, at an 8,192-token context | 26 tokens/s ◦ |
| A 24 GB card | Qwen 3.6 35B-A3B entirely on the card (17 GB at a 32,000-token context) | 38 tokens/s (42 on a short prompt) |
| A 4 GB card, 8 GB of RAM, an NVMe disk | Qwen 3.6 35B-A3B — the dense part on the card, the experts on the CPU, read from the disk as they are needed | 11.9 tokens/s ◦ |
| An 8 GB card, 32 GB of RAM, an NVMe disk | Qwen 3.5 122B-A10B (`D4E4`) — the same split | 4.7 tokens/s ◦ |
| A 16 GB card, 64 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | about 2.5 tokens/s, projected |
| A 16 GB card, 128 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | 5.7 tokens/s |
| A 16 GB card, 150 GB of RAM | GLM 5.3 Flash with all of its experts (142 GB) in RAM, a 300,000-token context, and 6 GB of the card the tier of experts used most | 7.5 tokens/s (6.8 without the tier) |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B (`D4E4`), its experts in RAM and 24 GB of them on the card | 19.9 tokens/s (10.9 without the tier) |
| A 32 GB card, 160 GB of RAM | GLM 5.3 Flash, its experts in RAM and 24 GB of them on the card | 10.7 tokens/s |
| A 32 GB card, 256 GB of RAM | Qwen 3.5 397B-A17B, its experts in RAM and 24 GB of them on the card | 9.0 tokens/s (6.3 without the tier) |

Measured on a Dell R730 (two Xeon E5-2680 v4, AMD Instinct MI50 cards; disk speed 5 GB/s, RAM speed 126 GB/s), each
row with its process held to the memory it names — leave room beside it for the operating system — and from a cold
start where the experts come from the disk. ◦ Ran on a 32 GB MI50: the model fits the smaller card, but a card that size
has not been measured. The projected row is worked out from GLM's measured rows at 64 GB and 128 GB. Every measurement,
the prompt rates, and what limits a model reading its experts from the disk: [`results.md`](results.md).

## The card's tier of experts

With a model's experts on the CPU, `card_experts_gb` in its config gives part of the card to the experts used most; the
card computes those while the CPU computes the rest, and the CPU copies an expert over while the card works, so a token
never waits for one. The rows above that name a tier are this; they ran on the machine's 256 GB of RAM, and the RAM they
name is what the experts and the process need, not a limit they were held to. For GLM, `card_exclusive` keeps what the
card holds out of RAM — an 18 GB tier gives back about as much RAM, for about a fifth fewer tokens a second on a card that
cannot copy both ways at once. [`users.md`](users.md#the-cards-tier-of-experts) says how to set both.

## How long a conversation fits

The conversation's cache sits on the card from boot, so `max_context` in a model's `configs/default.json` is set to what
the card has room for. The Qwen and GLM caches keep their first 256 positions and the latest 4,096 at full precision and
the rest at 8 bits; past the first 4,352, each 100 MB of the card holds about 10,000 more tokens of the 35B's
conversation, 3,200 of the 27B's and 8,800 of GLM 5.3 Flash's. Ministral 3's cache is every position at full precision —
about 625 positions a 100 MB; its `D4` boots with 32,768 and its `D8` with 8,192, which fit a 16 GB card beside it. On an
8 GB card, set the `D4`'s to about 4,096.
