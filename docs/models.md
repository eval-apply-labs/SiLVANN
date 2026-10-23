# Models

Every model this release runs, every pack of each, and what each needs. The README recommends one model a kind of
machine; this page has the rest.

## The packs

| Model | Pack | Weights | Download |
|---|---|---|---|
| Ministral 3 14B (dense, Mistral AI) | `D4`, for 8 GB cards | 7.4 GB — 4 bits | [eval-apply/silvann_tq_D4_ministral_3_14b](https://huggingface.co/eval-apply/silvann_tq_D4_ministral_3_14b) |
| Ministral 3 14B | `D8`, for 16 GB cards | 13.5 GB — 8 bits | [eval-apply/silvann_tq_D8_ministral_3_14b](https://huggingface.co/eval-apply/silvann_tq_D8_ministral_3_14b) |
| Qwen 3.8 27B (dense) | `D4` | 15 GB — 4 bits | [eval-apply/silvann_tq_D4_qwen3.8_27b](https://huggingface.co/eval-apply/silvann_tq_D4_qwen3.8_27b) |
| Qwen 3.6 35B-A3B (mixture of experts, 3B active) | `D8E4` | 19 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b) |
| Qwen 3.5 122B-A10B (mixture of experts, 10B active) | `D8E4`, for 12 GB cards | 67 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b](https://huggingface.co/eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b) |
| GLM 5.3 Flash | `D8E4` | 166 GB — experts 4 bits, the rest 8 | [eval-apply/silvann_tq_D8E4_glm_5.3_flash](https://huggingface.co/eval-apply/silvann_tq_D8E4_glm_5.3_flash) |
| DeepSeek-V4.1-Flash-1017 (384 experts, 6 active; text and pictures) | `D8E4` | 492 GB — experts 4 bits, the rest 8; 203 GB of it the n-gram memory's tables, read from the disk | [eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017](https://huggingface.co/eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017) |
| Qwen-Image 2.1 (pictures from a prompt: a 7B diffusion transformer, Qwen3-VL 8B's text model as its encoder) | `D8` | 18 GB — the transformer and the encoder 8 bits, the picture decoder 16 | [eval-apply/silvann_tq_D8_qwen_image_2.1](https://huggingface.co/eval-apply/silvann_tq_D8_qwen_image_2.1) |

Qwen 3.5 397B-A17B runs too, but its pack is not published: GLM 5.3 Flash and DeepSeek-V4.1-Flash do more at its size.
The `D4E4` packs of the 35B, the 122B and GLM were retired in v0.18.0 — their `D8E4` hold the same 4-bit experts. Since
v0.18.0 a Qwen MoE pack keeps each routed expert as a record of its own, one read from the disk; one downloaded before
keeps working ([`users.md`](users.md#5-the-experts-on-the-disk)).

The name says the bits: `D8E4` is the dense matrices at 8 and the experts at 4; `D4` a dense model at 4. The 8-bit dense
part is the one that pays — ▶ [`compression.md`](compression.md), and how close each pack comes to the original weights
is in [`results.md`](results.md#how-good-each-pack-is-perplexity). `python3 silvann_core.py list-models` lists them all,
and `download-model NAME` fetches one into `models/`.

Each model keeps its own licence, which is in its folder once downloaded: Qwen's and Mistral's Apache-2.0, GLM's and
DeepSeek's MIT.
⚠ Qwen-Image 2.1's is the Qwen Research License Agreement: research and other non-commercial use only. A picture model
runs as a server's model of its own: [`users.md`](users.md#a-picture-model-qwen-image-21).

## What each needs, measured

From a machine with no graphics card to a model near the frontier on salvaged server parts:

| Your machine | Model, and how it runs | Answer |
|---|---|---|
| No graphics card, 12 GB of RAM, an NVMe disk (2.5 GB/s) | Qwen 3.6 35B-A3B (`D8E4`†) on the CPU alone, its experts read from the disk, 8 GB of them kept in memory | 11.7 tokens/s |
| An 8 GB card | Ministral 3 14B (`D4`) entirely on the card, at a 4,096-token context | 42 tokens/s (56 on a short prompt) ◦ |
| A 16 GB card | Qwen 3.8 27B entirely on the card, at a 32,000-token context (14 GB) | 30.5 tokens/s (34 on a short prompt) |
| A 16 GB card | Ministral 3 14B (`D8`) entirely on the card, at an 8,192-token context | 30 tokens/s (37 on a short prompt) ◦ |
| A 24 GB card | Qwen 3.6 35B-A3B entirely on the card (17 GB at a 32,000-token context) | 87 tokens/s (102 on a short prompt) |
| A 4 GB card, 8 GB of RAM, an NVMe disk | Qwen 3.6 35B-A3B — the dense part on the card, the experts on the CPU, read from the disk as they are needed | 11.9 tokens/s ◦ |
| An 8 GB card, 32 GB of RAM, an NVMe disk | Qwen 3.5 122B-A10B (`D8E4`†) — the same split | 4.7 tokens/s ◦ |
| An 8 GB card (an RTX 5060 Ti with 7.5 GB of it held), 32 GB of RAM | Qwen 3.6 35B-A3B (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 3.5 GB of the card the tier of experts used most | 56 tokens/s ‡ |
| An 8 GB card (the same), 32 GB of RAM | Ministral 3 14B (`D4`) entirely on the card, at a 4,096-token context | 47 tokens/s ‡ |
| A 12 GB card (an RTX 5060 Ti with 4 GB of it held), 62 GB of RAM | Qwen 3.6 35B-A3B (`D8E4`) — the same split, and 7 GB of the card the tier | 65 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 28 GB of RAM | Qwen 3.6 35B-A3B (`D8E4`) — the dense part on the card, the experts on the CPU in RAM, and 11 GB of the card the tier of experts used most | 68 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 28 GB of RAM, an NVMe disk reading 5 GB/s | Qwen 3.5 122B-A10B (`D8E4`†) — the dense part on the card, the experts on the CPU, read from the disk | 8.1 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 28 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash (`D8E4`) — the same split | 2.3 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 32 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | 2.4 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 60 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | 3.2 tokens/s ‡ |
| A 16 GB card (an RTX 5060 Ti), 62 GB of RAM | Qwen 3.5 122B-A10B (`D8E4`†) — the dense part on the card, the experts on the CPU in RAM (54 GB), and 9.5 GB of the card the tier of experts used most | 20 tokens/s ‡ |
| A 16 GB card, 128 GB of RAM, an NVMe disk reading 5 GB/s | GLM 5.3 Flash — the same split | 5.7 tokens/s |
| A 16 GB card, 150 GB of RAM | GLM 5.3 Flash with all of its experts (142 GB) in RAM, a 300,000-token context, and 6 GB of the card the tier of experts used most | 8.0 tokens/s |
| A 32 GB card, 80 GB of RAM | Qwen 3.5 122B-A10B (`D8E4`†), its experts in RAM and 24 GB of them on the card | 24.5 tokens/s |
| A 32 GB card, 160 GB of RAM | GLM 5.3 Flash, its experts in RAM and 24 GB of them on the card | 10.7 tokens/s |

Measured on a Dell R730 (two Xeon E5-2680 v4, AMD Instinct MI50 cards; disk speed 5 GB/s, RAM speed 126 GB/s), each row
with its process held to the memory it names — leave room beside it for the operating system — and from a cold start
where the experts come from the disk. ◦ Ran on a 32 GB MI50: the model fits the smaller card, but a card that size has
not been measured. The rows with a model entirely on the card, the 122B's with 24 GB of tier and GLM's with 6 GB are
v0.7.0's and v0.8.0's; the others were measured with v0.3's card kernels, and where a card does part of the work they
are a floor ([`results.md`](results.md) names each row's version). ‡ The RTX 5060 Ti rows are another machine's, a
four-core Xeon with two channels of DDR4: the answer in the chat — the server with the model's own config, held to the
RAM its row names, as the chat's meter reads a short conversation — measured with v0.11.1
(2026-10-18), a smaller card's room the rest of the card held (7.5 GB held leaves what a headless 8 GB card does); from
that disk the 122B with 32 GB of RAM answers at 8.5 tokens/s. † Measured with the model's `D4E4` pack, retired in
v0.18.0: the `D8E4` holds the same 4-bit experts and its dense part at 8 bits (the 122B's layer matrices 4.66 GB on the
card against 2.37, the 35B's 1.45 against 0.75), and its rate there is to be measured again. On that card in the chat Ministral `D4` answers at 47
tokens/s, the 27B at 26, and the 35B with its experts on the four-core Xeon and the tier the setup chooses at 67; after
a 2,000-position prompt at 39, 23.5 and 57 ([`results.md`](results.md#the-rtx-5060-ti)).
Every measurement, the prompt rates, and what limits a model reading its experts from the disk:
[`results.md`](results.md).

## DeepSeek-V4.1-Flash

DeepSeek's model reads text and pictures: its vision tower (32 blocks and the aligner, 464 MB in the pack's
`vision.pkl`) runs on the card in about 620 MB of it, a picture at most 1,024 tokens. A pack downloaded before v0.17.0
has no `vision.pkl` — it reads text, and a picture is refused — and this fetches the file alone:
`venv/bin/hf download eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017 vision.pkl --local-dir
models/silvann_tq_D8E4_deepseek_v4.1_flash_1017`. Its routed experts are 272 GB (6.8 GB a layer,
40 layers) and run on the CPU: as many layers in RAM as fit beside the server, a spare and a tenth of the memory for the
system and the files it reads in place; the rest are mapped from the pack and read through the system's page cache
(a pack downloaded before v0.18.1 writes them once beside it, in `models/<name>/experts/`). Its n-gram memory's tables (203 GB) stay on the disk and are read a row at a time as
each token asks for them, so the disk should be an NVMe drive. A conversation reaches the model's own length,
1,048,576 positions: past 16,384, layer 20's candidate blocks choose where the later layers look, as DeepSeek's own
code does. Its rates come with a later release;
its options are in [`config.md`](config.md#deepseek-v41-flashs-options).

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
