# A model's config

A model's config is a JSON file in its folder's `configs/`. The server reads it when it starts:

- `models/<name>/configs/default.json` by default, written when the model is downloaded;
- or another file, named as the server's first argument:
  `python3 start_silvann_server.py --model models/<name> long.json` — a file in the model's `configs/`, or a path.

The UI's **Options** view edits these files. Its **new config** button starts a file with the server's keys at their
defaults (`image` aside), and the model's options listed in `_options` for you to copy into `options`.

## How a config is written

- **Leave a key out** to take its default. A config of `{}` is a valid config.
- **Keys starting with `_` are notes** and are ignored (`_note`, `_note_tools` …). The downloaded `default.json` keeps
  its explanations there.
- **A config sets a machine's whole setup:**
  - the keys at the top are the server's;
  - `options` are handed to the model's runtime, and each family takes its own (below).
- **An option the model does not take stops the server** when it starts, with the option's name in the message.
  `kv`, `chunk`, `experts_on`, `experts_from` and `picture_weights` refuse a value they do not list, the same way.

```json
{
  "max_context": "auto",
  "thinking": true,
  "sampler": "generation",
  "tools": "chill",
  "options": { "kv": "tiered" }
}
```

## The server's keys

| key | default | what it sets |
|---|---|---|
| `max_context` | `"auto"` | The longest conversation, in positions. Its cache is on the card from boot, sized for it. `"auto"` (or `"max"`) means as many as the card holds beside the model, up to the model's own length — 262,144 for the Qwen models and Ministral, 1,048,576 for GLM — found when the server starts and kept for the conversations after. A number holds it there instead. A Qwen MoE model's or GLM's tier of experts holds experts in the part of that cache a conversation has not reached, and gives it back as the conversation grows ([below](#the-context-and-the-tier-share-the-card)). The `_note` of a downloaded config says how many positions 100 MB of the card holds for that model. |
| `max_tokens_per_video` | `8192` | The most tokens a video attached to a message becomes. Its frames share them. Reading a video holds 57–74 KB of RAM a token, which the server counts when it chooses whether the experts fit in RAM. |
| `thinking` | `true` | Whether the model reasons before it answers, in a new conversation. A conversation's own switch (the UI's **thinking mode**, `silvann_cli --nothink`) turns it off for that conversation. |
| `sampler` | `"generation"` | How the next token is chosen. `"generation"` uses the model's own `generation_config.json`. `null` takes the most likely token (greedy). A dict sets its own: `{"temperature", "top_k", "top_p", "repetition_penalty", "seed"}`; what it leaves out is temperature 1.0, top_k 20, top_p 0.95, no penalty, a new seed each time. A `seed` gives the same answer to the same conversation. |
| `tools` | `"chill"` | What the model's tool, a Lisp REPL on the server's machine, may do without asking in a new conversation. `"ask"`: read in the workspace and allowed folders. `"chill"`: read anywhere. `"full-auto"`: everything. `"no"`: no conversation has the tool. |
| `tools_folder` | `workspace/` | Where the REPL starts, and where it reads without asking. By default it is `workspace/` beside the conversations. |
| `lora` | `null` | A LoRA in the model's `loras/`, loaded at boot: the one a conversation runs under unless it names its own. An adapter is switched between conversations; a rank-1 mask stays for every conversation ([`tools.md`](tools.md#subagents)). |
| `vram_margin_mb` | `50` | How much of each card the setup chosen at boot leaves free, in MB, for whatever else draws on it (a desktop, a browser). `0` on a card nothing else uses. |
| `card` | `"auto"` | Which of the machine's cards to run on: `0`, `1` …, `"auto"` the first, or `"cpu"`, which uses no card (the CPU and its RAM). Under tensor parallelism leave it `"auto"`: the group is the first cards. |
| `options` | `{}` | Handed to the model's runtime, below. |
| `image_model` | `"auto"` | The picture model the REPL's `(image "…")` draws with: a folder in `models/`, or a path. `"auto"` takes a picture model downloaded into `models/`, if there is one. `"no"` turns pictures off, and `(image …)` says so. |
| `image` | the picture model's own | The settings of a picture the REPL draws, as a picture model's config has them (below). |
| `image_card` | `null` | The card the picture is drawn on. `null` uses the text model's card: the model is put down while it draws and comes back after (below). A card of its own (`1` …) keeps the text model up. |
| `workers` | `[]` | Other SiLVANN servers, `"host:port"`, that a subagent of another model runs on ([`tools.md`](tools.md#subagents)). With none free, this machine lends its card. |
| `pipeline_parallelism` | `"no"` | `"yes"` runs the model over several machines in a ring, each with a run of its layers ([`users.md` §6](users.md#6-a-model-over-several-machines)). |
| `pipeline_parallelism_layers` | — | This machine's run of the layers, first-last, as `"0-23"`. Needed with `pipeline_parallelism` `"yes"`. |
| `pipeline_parallelism_next` | — | The machine after this one, `host:port`. The last machine names the first. |
| `pipeline_parallelism_port` | `8766` | Where this machine listens for the one before it. |

### A picture from the text model

A text model's REPL has `(image "prompt")` ([`tools.md`](tools.md)). When the config finds no picture model, or
`image_model` is `"no"`, it answers at once with an error that says why, so the model tells the user.
The model calls it like any command, the user sees the picture's progress, and the picture appears in the
conversation.

**On the text model's card** (`image_card: null`), drawing a picture switches what is on the card:

1. The conversation on the card is read off into RAM.
2. The text model shuts down.
3. The picture model boots, draws, and ends.
4. The text model boots again as it did at the start.
5. The conversation goes on where the call stopped.

**What it costs** is the text model's boot on top of the picture. MEASURED on an MI50: the 27B and the 35B came back
in 16–17 s, and their answers went on to the same tokens as without the switch. While the picture is drawn, other
conversations wait: the server answers them 503 with `Retry-After`.

**On a card of its own** (`image_card: 1`), none of that happens, and that is better where the machine has a free
card.

A picture model (Qwen-Image 2.1) reads `card` and `image`:

| key | default | what it sets |
|---|---|---|
| `image` | `{"width": 1024, "height": 1024, "steps": 40, "seed": null, "count": 1}` | A picture's settings. `width` and `height` are multiples of 32 from 256 to 2,048. More `steps` are finer and slower. `seed: null` takes a new seed each picture. `count` (1 to 8) is how many pictures a prompt makes, seeds `seed`, `seed + 1`, …: drawn at once, one on each card, where the machine has several cards, and one after another on one. A message may set its own: `--size 768x1024 --steps 20 --seed 7 --count 4`. |

## What the config leaves out is chosen for the machine

A model whose `options` do not name its whole placement — `experts_on`, `experts_from` and `card_experts_gb` — has one
tried on the card when the server starts, fastest first:

1. **The experts on the card**, if it holds them.
2. **Otherwise the experts on the CPU**, in RAM when they fit there with about 3 GB to spare. That spare is the
   server's own 2 GB, what reading a video holds, and 500 MB more; a container's memory limit counts. Otherwise the
   experts are read from the disk.
3. **With the experts in RAM**, the card's tier of experts gets the largest size that boots beside the context's cache,
   in steps of 256 MB.

Every trial boot holds `vram_margin_mb` of each card beside it, so what is chosen leaves that much free for whatever else
draws on the card — a desktop, a browser. 50 MB by default; `0` on a card nothing else uses.

**The cache comes first.** The whole context's cache is sized before the tier: a smaller tier costs a little speed, but
a cache too small refuses a conversation. If nothing boots, the context is halved, down to 4,096. With `max_context`
`"auto"` the experts stay on the card at the longest context it holds beside them, if that is 32,768 or more; otherwise
they go to the CPU, and the context grows to the model's whole length as far as the card holds it. Where it is whole, the
tier gets what is left of the card. Where it is not, a tier lent the cache's unreached rows (below) keeps the least the
model takes, a slot a layer, and the context the rest: GLM on 16 GB of an MI50 gets 790,528 positions and a 0.5 GB tier
of its own, lent the rows a conversation has not reached. A tier lent nothing leaves the card to the context.

**Without a card of its own** — the CPU alone (`"card": "cpu"`), or an integrated GPU (`khronos_opencl1_igpu`), whose
memory is the PC's RAM — nothing is tried: the model's own defaults, and `"auto"` is 32,768 positions. An integrated GPU
reads pictures on the CPU unless the config's `picture_pixels` gives them room.

### The context and the tier share the card

A Qwen model's cache, and GLM's, is sized at boot for the whole context, but a conversation only writes the positions it
has reached. With the experts on the CPU, the card's tier of experts also holds experts in the rest of the cache, the
rows no conversation has reached: on an RTX 5060 Ti 16 GB, the 35B-A3B at 262,144 positions lends its tier 2.4 GB of
them; on a 32 GB MI50, GLM 5.3 Flash at its 1,048,576 lends its tier 11 GB. GLM's exclusive tier (`card_exclusive`) is
lent none: what it gives back is RAM, sized at boot.
As a conversation grows, the tier gives those rows back 8,192 positions at a time, before the conversation writes there.
When a shorter conversation comes onto the card, or one starts afresh, the tier is lent them again. So `"auto"` needs no
tier size and no context to be set: a short conversation has nearly the whole card for its experts, and a long one
has its cache. `"lend_cache": false` in the options keeps the tier to its own memory.

**The choice is kept** in the model's folder as `autotune.json`:

- The server's log says what was chosen and why.
- The next start with the same config on the same machine takes it without trying, until SiLVANN is updated: an
  update chooses again, as it may change what a boot needs. Delete the file to choose again sooner.
- If 1 GB more of a card is free than when the choice was made (another program held it then), or the card refuses
  the kept setup, the choice is made again.

**What the config names is kept.** A placement named in full, or `"experts_on": "card"`, boots as it is, untried, at
a `max_context` the config gives as a number; with `"auto"` its context is still found by trying. Otherwise the rest
is chosen around what is named, and `"auto"` names nothing. A tier's size or `experts_int8` named alone puts a Qwen
model's experts on the CPU, the only place they apply. For example,
`{"experts_on": "cpu", "experts_from": "auto"}` has where the experts come from and the tier chosen, and
`"card_experts": "no"` asks for no tier.

## The Qwen models' options

These options apply to Qwen 3.6 35B-A3B, Qwen 3.8 27B, and Qwen 3.5 122B-A10B and 397B-A17B.

| option | values | what it sets |
|---|---|---|
| `kv` | `"tiered"` by default, or `"fp16"` | The conversation's cache. `tiered` keeps the first 256 and the latest positions in halves, and the rest at 8 bits. |
| `tiers` | `[sink, hot, warm]` | The tiers' sizes; `warm` may be `"rest"`. The default is `[256, 4096, "rest"]`. `[256, 4096, N]` keeps the `N` positions before the latest 4,096 at 8 bits and the older ones at 4 bits, for small cards. |
| `chunk` | 1 to 256, default 256 | How many positions of a prompt go through a layer at once. |
| `experts_on` | `"card"`, `"cpu"` | The routed experts on the card, or on the CPU (for a card too small to hold them). It is chosen at boot when the config does not say. |
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | With the experts on the CPU, where they are read from. `ram` copies them into RAM. `disk` reads them from a file as they are needed and keeps the ones used in the engine's memory (`experts_ram_gb` sets how much). `mapped` leaves that to the system's page cache, and is slower. `auto` chooses by the memory there is. |
| `experts_ram_gb` | gigabytes | With `experts_from: "disk"`, how much RAM keeps the experts used. By default, what the process may hold less a quarter of it (at least 3 GB, at most 8 GB) kept for everything else. |
| `card_experts` | `"yes"` by default, or `"no"` | With the experts on the CPU from RAM, whether the card keeps its own tier of them, sized at boot ([`users.md`](users.md#the-cards-tier-of-experts)). |
| `card_experts_gb` | gigabytes | The tier's size, rather than the size chosen at boot. Named alone, it puts the experts on the CPU. |
| `lend_cache` | `true` by default, or `false` | With a tier, whether it also holds experts in the cache's rows a conversation has not reached, given back as it grows ([above](#the-context-and-the-tier-share-the-card)). |
| `experts_int8` | `false` by default, or `true` | With the experts on the CPU, a generated token's experts are multiplied in integers: faster, and slightly less exact. Named alone, it puts the experts on the CPU. |
| `experts_dir` | a folder | With the experts read from the disk, where they are written when they cannot be read in place ([`users.md` §5](users.md#5-the-experts-on-the-disk)): `experts/` in the model's folder by default — another drive, say. |
| `picture_pixels` | default `4194304` (2048 × 2048) | The largest picture the boot makes room for on the card, about 23 KB a 16 × 16 patch. A larger picture is made smaller. `0` keeps no room and reads pictures on the CPU. |
| `picture_weights` | `"auto"` by default, or `"card"`, `"ram"` | Where the vision tower's 0.6 GB of weights sit. In RAM, they are written to the card a block at a time as a picture is read. `auto` puts them on the card when the experts are on the CPU, and in RAM otherwise. |
| `tensor_parallel` | `1` by default, or `2`, `4` … | **Developer preview: tested on AMD (gfx906) only.** Every layer split over this many cards, each card with its share of the heads and of the MLP; see the conditions below. |
| `mtp` | `"no"` by default, or `"yes"` | The model's multi-token prediction layer, in the Qwen 3.6 packs: it drafts the token after the next one, and the model checks the draft and the next token together, as GLM's `mtp` below. A greedy answer is the same tokens either way. On the 35B-A3B, 90–92% of the drafts were kept and a 128-token answer ran at 1.12× with the experts on the CPU, but 0.91× with them on an MI50, where a check of two positions still costs about 1.8 tokens' time. One machine's every layer on one card, without a card tier of experts. It adds the MTP layer's weights (about 0.45 GB of the card for the 35B), and a sampled answer runs without it. |

### Tensor parallelism

`tensor_parallel` splits every layer's heads over the cards. It runs three kinds of model:

- **a dense model** (the 27B), with every layer on the cards;
- **a Qwen MoE** (the 35B, the 122B, the 397B), with its experts on the CPU (`experts_on: "cpu"`). A
  `card_experts_gb` tier is then that many gigabytes on each card, and a position's experts go to the cards that
  hold them, the rest to the CPU;
- **GLM**, with its experts in RAM. Each card keeps its own tier of experts, and 10% of a card's slots are experts
  every card holds. `MEASURED` on four MI50s: 109 ms a token on one card, 100.7 on two, 80.9–86.5 on four.

The server refuses it at boot when:

- the cards **are not all the same size**: each card holds the same share, so the group is sized as one of them;
- the cards' memory cannot be read: `rocm-smi` or `nvidia-smi` says it;
- two cards cannot reach each other's memory (peer access);
- the cards do not split the model's heads evenly;
- a LoRA is loaded, or the model is a ring's stage;
- for GLM: `mtp` is on, `card_exclusive` is `true`, or `fused` is off.

Pictures are not read under tensor parallelism yet: the model boots, and a message's picture is refused. A GLM group
reads a prompt one position at a time, and a conversation's state is not saved, so a conversation brought back reads
its tokens again.

**Its program:** a Qwen model's folder needs `lisp/qwen3_5_tp.lisp`; GLM's runs the pack's own `lisp/glm5.lisp`. `silvann download <model>` fetches the pack's `lisp/`
afresh, and a pack that does not carry the file yet cannot run tensor parallelism.

**Which cards:** the group is the machine's first cards. Use `HIP_VISIBLE_DEVICES` (AMD) or `CUDA_VISIBLE_DEVICES` to
choose them.

**Memory:** every number of the context is per card. The `"auto"` context is what one card of the group holds beside
its share of the model. The estimate (`/cartridges/estimate`, and the UI's **New conversation**) gives two figures:

- `per_card`: each card's share of a conversation's cache;
- `bytes`: what the conversation takes in RAM when it is put away, every card's share together.

**Moving between setups:** a conversation put away under tensor parallelism comes back on one card, or on another
number of cards.

**Measured:** on 4 MI50s (gfx906) the 27B generated a token in 29.9 ms on one card, 23.1 ms on two and 24.9 ms on
four.

## GLM 5.3 Flash's options

| option | values | what it sets |
|---|---|---|
| `experts_from` | `"auto"`, `"ram"`, `"disk"`, `"mapped"` | Where its routed experts (142 GB) are read from. `ram` holds them in RAM. `disk` reads them from a file as they are needed and keeps the ones used in the engine's memory (`experts_ram_gb` sets how much). `mapped` leaves that to the system's page cache. `auto` puts them in RAM when the available memory holds them with about 3 GB to spare, which is about 145 GB. |
| `experts_ram_gb` | gigabytes | With `experts_from: "disk"`, how much RAM keeps the experts used. |
| `chunk` | positions, default 256 | How many positions of a prompt go through a layer at once. With the experts on the disk 1024 reads a prompt faster, and is chosen at boot. |
| `experts_dir` | a folder | As the Qwen models'. |
| `sockets` | `1`, `2` | The CPU sockets the experts are split over. By default, all of them. |
| `experts_int8` | `true` by default, or `false` | The experts are multiplied in integers on the CPU. |
| `card_experts` | `"yes"` by default, or `"no"` | With the experts in RAM, whether the card keeps its own tier of experts, sized at boot. |
| `card_experts_gb` | gigabytes | The tier's size, rather than the size chosen at boot. |
| `lend_cache` | `true` by default, or `false` | As the Qwen models': an inclusive tier also holds experts in the latent cache's rows a conversation has not reached. |
| `card_exclusive` | chosen at boot; `false`, `true` | With `experts_from: "ram"`, an expert the card's tier holds is not also kept in RAM ([`users.md`](users.md#the-cards-tier-of-experts)). Chosen where the tier is a quarter of the experts or more and the card copies both ways at once. `true` needs `card_experts_gb`. |
| `picture_pixels` | default `4194304` | As the Qwen models'. |
| `mtp` | `"no"` by default, or `"yes"` | The model's multi-token prediction layer, which is in every GLM pack: it drafts the token after the next one, and the model checks the draft and the next token together, keeping the draft when it is the token the model would have picked. A greedy answer is the same tokens either way; whether it is faster depends on the machine and the context. Through the server, a 33-token chat prompt and 159 tokens of answer: 1.18× on two Xeon sockets with the experts in RAM, 0.99× on one socket with the experts from the disk. On a 1,600-token prompt it was slower, 0.92–0.94×, though 81% of the drafts were kept: a check reads both positions over the whole context. A sampled answer runs without it, for now. It adds the MTP layer's weights (about 127 MB of the card) and its experts beside the others, about 141 MB of the card for the check, and to the prompt about 8–12 ms a position (about +10%). |

Both families take a few more options, for measuring the tier rather than for use (`card_landing`, `card_line`, and GLM's
`card_hot`, `card_free`, `card_duplex`, `card_duplex_line`, `card_landing_line`, `grid`, `fused`, Qwen's `fused_moe`):
their defaults are the ones to run with.

## Ministral 3 14B's options

Ministral takes no options. Its cache is every position in halves, about 625 positions a 100 MB of the card, so
`"auto"` gives it what the card holds beside it: on a 16 GB card about 32,000 positions beside its D4.

## Examples

**The 35B with its experts on the CPU, in RAM, and older positions at 4 bits:**

```json
{ "max_context": 131072, "options": { "experts_on": "cpu", "experts_from": "ram", "tiers": [256, 4096, 65536] } }
```

**The 27B over two cards of one size:**

```json
{ "max_context": "auto", "options": { "tensor_parallel": 2 } }
```

**GLM with its experts in RAM and the card's tier at 18 GB:**

```json
{ "max_context": 65536, "options": { "experts_from": "ram", "card_experts_gb": 18 } }
```

**GLM with its experts on an NVMe drive:**

```json
{ "max_context": 65536, "options": { "experts_from": "disk" } }
```

**Greedy answers, no thinking, no tool:**

```json
{ "sampler": null, "thinking": false, "tools": "no" }
```
