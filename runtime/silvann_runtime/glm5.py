"""GLM 5.3 FLASH — four residual streams, Kimi Delta Attention and latent attention, 288 experts — on a card and the CPU.

⚖ *"at 151gb the point is that experts stay on ram and fixed matrices go on the card"* · *"it is the cpu that computes
the ram experts"*. So the machine has two workers: the card (worker 0) holds every matrix a token always uses — the
mixers, the dense MLPs, the shared experts, the routers, the hyper-connections and the head, and reads a token's row of
the embedding from a table in RAM — and the CPU (worker 1, and worker 2 on a second socket) holds the routed experts in its own memory and runs the eight a token picks. A layer is three short
programs:
```
  card   a{l}   the attention site: hc pre, norm, the mixer, hc post · the MLP site's hc pre and norm, and then either
                the dense MLP and hc post, or the router's picks and weights and the shared expert
  CPU    x{l}   the input and the weights copied over, the eight picked experts, their weighted sum
  card   c{l}   the sum copied back, added to the shared expert's, hc post
```
That is the unfused layer, nn's words a layer. The fused one, the default, is the program in `lisp/glm5.lisp` — the
model folder's — a loop over
the layers, each ai_z__glm_5_3's sites over its tables, every CPU socket handed its share of a layer's picks. Either is
`(position tok pos)`, so an answer is `generate` on the card (▶ lisp/generate.lisp), only its tokens coming back.

The mixer is the model's: KDA (`nn__kda__step` over a conv window and a state a layer) or MLA — the query absorbed into
a cache of rotated 512-wide latents, one kv head for 64 query heads (`nn__attention__absorb` · `decode` · `expand`).
Past 2048 positions the DSA indexer chooses which of them a query attends, as the model's does.

⭐ THE ROUTED EXPERTS FROM RAM OR FROM THE DISK. With memory for them they are copied in at the load; without it
(`experts_from="disk"`, or `"auto"` when they would not fit) each socket's share is written once as a file in its slots'
own layout — `experts/partNofP.bin` beside the bundle, from the bundle, which stays as it was published — and that file is
mapped over the slots. An expert is then read from the disk the first time a token picks it and stays for as long as the
system keeps the file's pages, so the memory holds the experts in use and the disk the rest.

Every matrix is a TurboQuant record rotated along its input, so each gemv is handed its input rotated; the lossless
ones (d = 16: the routers, the maps, the norms) take it as it is.
"""
import json
import os
import pickle

import numpy as np

from . import experts_file, expert_tier, sampling
from .vision import video_ram
from .generate import GENERATE_MOST, Generates
from .autotune import card_memory
from .machine import own_memory
from .model_folder import NL, Refused

ARCHITECTURES = ("glm5_next_text",)
CARD, CPU = 0, 1
EXPERT = "mlp.experts.%d.%s_proj.weight"


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


class Glm5(Generates):
    PROGRAM = ("glm5", "glm5_cpu", "sampling", "generate")    # the model folder's lisp/ files (▶ `ModelFolder.program`)
    SAMPLES_ON_CARD = True          # `head_sample` — ▶ `sampling`

    def __init__(self, folder, machine, max_context=2048, layers=None, grid=True, fused=True, sockets=None, experts_int8=True,
                 experts_lock=False, experts_register="auto", experts_from="auto", experts_dir=None, chunk=256, experts_ram_gb=None, card_experts_gb=0, card_hot=None,
                 card_landing="auto", card_line=None, card_exclusive=False, card_free=16, card_duplex_line=1200, card_duplex=None, card_landing_line=3,
                 lend_cache=True, picture_pixels=1 << 22, mtp=False, mtp_hidden="hn", tensor_parallel=1, card_shared="auto", card_visit_first=False,
                 stream=False, embed_on="auto",
                 ):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b, self.c = folder, machine, b, c
        self.H, self.LAYERS, self.MULT = c["hidden_size"], c["num_hidden_layers"], c["hc_mult"]
        self.ITERS, self.HC_EPS, self.EPS = c["hc_sinkhorn_iters"], float(c["hc_eps"]), float(c["rms_norm_eps"])
        la = c["linear_attn_config"]
        self.KH, self.KD, self.LOWER = la["num_heads"], la["head_dim"], float(la["gate_lower_bound"])
        self.KW = self.KH * self.KD
        self.QH, self.QR, self.LAT = c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
        self.NOPE, self.VD = c["qk_nope_head_dim"], c["v_head_dim"]
        if c["qk_rope_head_dim"] != 0:
            raise Refused("this MLA has a rotary part (%d) — the latent cache here carries none" % c["qk_rope_head_dim"])
        self.INTER, self.MINTER = c["intermediate_size"], c["moe_intermediate_size"]
        self.EXPERTS, self.TOPK = c["n_routed_experts"], c["num_experts_per_tok"]
        # a hand row, as ai_z__glm_5_3 lays it out: the input, the weights, the picks as words — AI_Z__GLM_5_3__HAND
        self.HAND = ((2 * self.H + 2 * self.TOPK + 7) & ~7) + 8 * self.TOPK
        self.RSCALE, self.LIMIT = float(c["routed_scaling_factor"]), float(c["swiglu_limit"])
        self.max_context = max_context
        # the DSA indexer: past `topk` positions a query attends the top pools of `kpool` and the pool still filling
        self.IH, self.ID, self.KP, self.ITOPK = c["index_n_heads"], c["index_head_dim"], c["index_kpool"], c["index_topk"]
        self.GRID = grid               # a token as one program over both blocks, or a program a part driven from here
        self.FUSED = fused             # the layer as ai_z__glm_5_3's verbs, a verb a device seam — or as nn's words
        self.run_layers = list(range(self.LAYERS)) if layers is None else list(layers)
        self.kinds = {l: ("f" if l in la["full_attn_layers"] else "d") for l in range(self.LAYERS)}
        self.moe = {l: "mlp.gate.weight" in b.names(l) for l in range(self.LAYERS)}
        # ⭐ THE MULTI-TOKEN PREDICTION LAYER (layer 45, `num_nextn_predict_layers` 1) — packed with the model and, until
        #   this, never loaded: `eh_proj` over [enorm(the next token's embedding) | hnorm(the model's hidden)], a DSA
        #   attention and an MoE on one plain residual (no hyper-connection of its own), `shared_head.norm`, the model's
        #   head — the token after next, a draft the model then checks. Held with the layers (its weights, its latent
        #   cache, its experts), run by its own program (`mtp`), never by the layers' loop. ▶ `_mtp_procedures`
        self.MTP = None
        # ⚖ 2026-10-19: *"make mtp a options field (no or yes, defaults no)"*
        if isinstance(mtp, str):
            if mtp.lower() not in ("yes", "no", "true", "false", "1", "0"):
                raise Refused("mtp is 'yes' or 'no', not %r" % (mtp,))
            mtp = mtp.lower() in ("yes", "true", "1")
        # the MTP layer's history: the last position its cache has (`_mtp_have`), the position whose head state is in hnB
        # (`_mtp_hid`), and (pos, token, draft) to go on with
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None
        # how much of a prompt the MTP layer is given (`MTP_HISTORY`): None every row, k the last k rows, 0 none but the last
        # row's head state (which a draft needs); and what generate's drafts did, {"passes", "kept"}.
        # ⭐ THE LAST 8 ROWS BY DEFAULT — `MEASURED` 2026-10-19 node-01 (test/src_glm5_mtp_history.py, a 1,600-token prompt,
        #   the MTP layer's caches cleared between arms): drafts kept, none 76%, the last 8 79%, the last 64 76%, every row
        #   81%; every row +11.7 s to the prompt (~7 ms a row, +12% — the precision ~12 ms, +10%), the last 8 +1.9 s. One
        #   prompt: the differences are a prompt's noise as much as the history's
        self.MTP_HISTORY, self.mtp_stats = 8, dict(passes=0, kept=0)
        if mtp:
            L = self.LAYERS
            if int(c.get("num_nextn_predict_layers", 0)) < 1 or b.n_layers <= L or "eh_proj.weight" not in b.names(L):
                raise Refused("%s carries no multi-token prediction layer (layer %d)" % (folder.path, L))
            if L - 1 not in self.run_layers:
                raise Refused("the MTP layer drafts from the last layer's hidden state, which this machine does not run")
            if not fused:
                raise Refused("the MTP layer runs in the fused layer — fused=True")
            self.MTP = L
            self.kinds[L], self.moe[L] = "f", "mlp.gate.weight" in b.names(L)
            self.run_layers.append(L)
        # which of the head's states is "the model's hidden" the MTP layer reads: the streams' mean after the final norm
        # (`hn`, what the head reads) or before it (`hm`). ⛳ REASONED, not read in any reference code: the drafts' acceptance
        # tells them apart (▶ test/src_glm5_mtp.py)
        if mtp_hidden not in ("hn", "hm"):
            raise Refused("mtp_hidden is 'hn' or 'hm', not %r" % (mtp_hidden,))
        self.MTP_HIDDEN = mtp_hidden
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        self.head_rec, self.emb_rec = g["lm_head.weight"], g["model.language_model.embed_tokens.weight"]
        self.VOCAB = int(self.head_rec["rows"])
        # ⭐ THE ROUTED EXPERTS ON EVERY SOCKET, EACH ITS SHARE IN ITS OWN MEMORY: a CPU worker a NUMA node (the x86 family's
        #   devices 1 and 2). `MEASURED`: one worker's arena landed whole on one node — the loader's thread touches every
        #   page first — so half its threads read across the socket link.
        #   ⭐⭐ AND THE SHARE IS HALF OF EVERY EXPERT, NOT HALF OF THE EXPERTS — ⚖ *"go ahead with the balanced split"*: a
        #   socket holds its part of each expert's inner width (its gate and up rows, and the same columns of its down
        #   rows), runs all eight picks at that width, and the parts' sums add up to the layer's. `MEASURED` before it,
        #   split by expert: the sockets took 1.08 and 2.95 ms for one layer, as the picks happened to fall.
        #   One worker holding all of them is `sockets=1`, and the words' path takes nothing else.
        if sockets is None:
            sockets = 2 if fused and machine is not None and machine.e.devices("x86_avx2") >= 3 else 1
        if not fused and max_context > self.ITOPK:
            raise Refused("past %d positions GLM's attention needs the indexer, which ai_z__glm_5_3's verbs run — not nn's words"
                          % self.ITOPK)
        if sockets > 1 and not fused:
            raise Refused("the experts split over %d sockets are reached by ai_z__glm_5_3's verbs only" % sockets)
        self.PARTS = sockets
        # ⭐ A TENSOR-PARALLEL GROUP OF `tensor_parallel` CARDS (NN-99 ④): every card the same cut of each matrix — the
        #   attention by heads (a KDA layer's channels, an MLA layer's heads), the dense MLP and the shared expert by their
        #   inner width — each site's output the group's sum (ai_z__glm_5_3's verbs' `GROUP sum` pair, ▶ ai_deepseek_4's),
        #   every card a tier of routed experts sharing a position's picks (▶ nn's `nn__expert_tier__visit_group`), the
        #   streams, the latents, the indexer and the router replicated, so every card's are the same. Card 0 holds the
        #   head and the embedding and hands the sockets their share. A prompt as rows, a chunk at a time: every card's rows
        #   sites take the group's sum of the chunk's rows, and the routed experts are the sockets' and card 0's tier's.
        #   `card_shared`: of a card's slots a layer, the share every card keeps (NN-99 ③) — "auto" from this machine's own
        #   speeds at boot (▶ `expert_tier.slice_of`), a number as it says
        self.P = int(tensor_parallel or 1)
        # a group's state is every card's share, which `state_regions` does not lay out yet: a conversation put away is
        #   its tokens, read again when it comes back (▶ session's PARKS, as a ring's)
        self.PARKS = self.P == 1
        self.CARD_SHARED = None if card_shared in (None, "auto") else float(card_shared)
        if card_exclusive and self.P > 1 and self.CARD_SHARED is None:
            self.CARD_SHARED = 0.0             # an exclusive group's slice is dealt at boot, before the speeds are timed
        self.CARD_SPEEDS, self._TIER = None, {}
        # `card_visit_first`: the tier's picks marked and the sockets started before the card's experts are queued (▶
        #   glm5.lisp `tier_and_sockets`) — off, as it was slower on the MI50 box (NN-99 ④); a switch to measure it elsewhere
        self.VISIT_FIRST = bool(card_visit_first)
        self.GROUP = 0
        if self.P > 1:
            if not fused:
                raise Refused("a tensor-parallel group runs ai_z__glm_5_3's verbs — fused=True")
            if mtp:
                raise Refused("the MTP layer does not run in a tensor-parallel group yet — mtp 'no'")
            # ⚖ 2026-10-22 the exclusive tier in a group: "a landing slot for each card, so each card can swap experts
            #   independently from the others" — every expert outside the slice on one card or in RAM, the slice's on every
            #   card and in no RAM slot (NN-106: ▶ nn's SHARED)
            if card_hot:
                raise Refused("a tensor-parallel group's tiers start empty — no card_hot")
            if layers is not None:
                raise Refused("a tensor-parallel group runs every layer — not a ring's stage yet")
            if self.KH % self.P or self.QH % self.P or (self.KW // self.P) % 512 or (self.QH // self.P * self.VD) % 512:
                raise Refused("%d KDA heads and %d MLA heads do not split into %d parts of whole rotation blocks"
                              % (self.KH, self.QH, self.P))
            if self.INTER % (512 * self.P) or self.MINTER % (512 * self.P):
                raise Refused("the MLPs' widths %d and %d do not split into %d parts of whole rotation blocks"
                              % (self.INTER, self.MINTER, self.P))
            # ⭐ CARDS OF ONE SIZE, as qwen3_5's group (⚖ 2026-10-16): every card holds the same share and the same tier,
            #   so the group is sized as one card of them. Peer access is the group's join (▶ nn's `nn__partials__join`)
            sizes = card_memory()
            if sizes is None or len(sizes) < self.P:
                raise Refused("tensor parallelism over %d cards: the cards' memory cannot be read (rocm-smi or "
                              "nvidia-smi says it)%s" % (self.P, "" if sizes is None else ": %d cards here" % len(sizes)))
            if len(set(sizes[:self.P])) > 1:
                raise Refused("tensor parallelism takes cards of one size, and these %d are not: %s GB"
                              % (self.P, ", ".join("%.1f" % (z / 2 ** 30) for z in sizes[:self.P])))
            self.BINDINGS = 512 * self.P + 256
        # a card's share of each width: all of it on one card
        self.KHC, self.QHC = self.KH // self.P, self.QH // self.P
        self.KWC, self.IC, self.MC = self.KHC * self.KD, self.INTER // self.P, self.MINTER // self.P
        self.EXPERTS_INT8 = experts_int8   # the routed experts' products in integers on the CPU (nn's int8 gemvs)
        # ⭐ THE CARD'S TIER OF ROUTED EXPERTS (NN-48): `card_experts_gb` of whole experts on the card, which computes the
        #   picks it holds while the CPU computes the rest. An expert the CPU computed earns a slot by nn's qualifier — three
        #   calls, the older of its last two newer than the card's least recent resident's — and is stitched from the
        #   sockets' parts in RAM into it while the card computes (`card_landing` a layer's visit at most, "auto" from this
        #   machine's own speeds at boot: ▶ `_from_speeds`). `card_hot`, a
        #   router log (▶ `test/src_glm5_router_log.py`), fills it at boot with each layer's most picked; without one it
        #   starts empty and fills itself.
        self.CARD_GB, self.CARD_HOT = float(card_experts_gb or 0), card_hot
        self.CARD_LANDING = None if card_landing in (None, "auto") else int(card_landing)
        # a prompt chunk's line — the rows past which an expert is worth its copy to the card — measured at boot and bound
        # as `nn__expert_major__vram_promotion_threshold_picks` (▶ `_calibrate_line`); `card_line` fixes it instead
        self.CARD_LINE = int(card_line) if card_line else None
        # ⭐ AN EXCLUSIVE TIER (NN-48): each expert on the card OR in the sockets' memory, never both, so the card's share
        #   is RAM given back. The card starts holding all but `card_free` of its slots — the landing slots a promotion
        #   lands in — and each promotion sends one back, written into the RAM slots the promoted one gave up. Whether
        #   the two copies run at once is bound in Lisp at boot (▶ `expert_tier.duplex`): `card_duplex` a ratio ×1000 to
        #   bind instead of measuring, `card_duplex_line` the one below which they do.
        self.CARD_EXCLUSIVE, self.CARD_FREE = bool(card_exclusive), int(card_free)
        self.CARD_DUPLEX_LINE, self.CARD_DUPLEX = int(card_duplex_line), card_duplex
        self.CARD_LANDING_LINE = int(card_landing_line)    # ▶ `expert_tier.tier`: above this many hits a visit lands one
        # ⭐ THE LATENT CACHE'S ROWS NO CONVERSATION HAS REACHED, LENT TO AN INCLUSIVE TIER (NN-93) — ▶ `_lend`. An exclusive
        #   tier is lent none: what it gives back is RAM, sized at boot, and an expert withheld from it would have no RAM
        #   slot to go to. `lend_cache` false keeps the tier to its own run
        self.LEND_ON = bool(lend_cache)
        # `stream`: a prompt read layer-major, its experts streamed onto the card through a ring (NN-115 ② ③, ▶ qwen3_5.py's,
        #   `_prefill_streamed`) — a slice SLICE_CHUNKS chunks of rows: eight with `true`, or as many as a number of rows
        #   needs, each expert then crossing once for that many positions
        self.STREAM_ON, self.STREAM_CH = bool(stream), max(1, min(chunk, max_context)) if fused else 0
        self.SLICE_CHUNKS = 8 if stream is True or not self.STREAM_CH else max(1, -(-int(stream) // self.STREAM_CH))
        if self.CARD_EXCLUSIVE and not self.CARD_GB:
            raise Refused("an exclusive tier needs a card tier — card_experts_gb")
        if self.CARD_GB and not fused:
            raise Refused("the card's tier of experts runs in the fused layer — fused=True")
        # ⭐ A RANK-1 MASK, where the folder's LoRA is one (▶ `nn_rank1_mask.py`, `nn_rank1_mask_from_lora.py`): `W' = W + r·vᵀ`
        #   beside the attention's output projections, the shared experts' downs and the routed experts' downs, each a site's
        #   cells past its table (▶ the package's `*__MASK` and `_mask_layout`). Chosen at boot, as Qwen's is.
        if b.overlay is not None:
            raise Refused("GLM 5.3 Flash takes a LoRA as a rank-1 mask, not as an adapter's matrices")
        if b.mask is not None:
            if not fused:
                raise Refused("a rank-1 mask is applied by ai_z__glm_5_3's verbs — fused=True")
            if self.MTP is not None:
                raise Refused("the MTP layer's draft check does not take a mask yet — mtp 'no'")
        self.active_lora = folder.lora if b.mask is not None else None
        self._lora_cache = {}
        # ⭐ THE PROGRAM IS lisp/glm5.lisp — the fused layer's: a loop over the layers, each its sites over its tables. The
        #   unfused layer, nn's words a layer, is composed here (`_composed_procedures`), for the router's log
        self.SOURCE = fused
        self.CPUS = [dict(w=self.P + i, dev=(1 + i) if sockets > 1 else 0, part=i) for i in range(sockets)]
        # ⭐ THE EMBEDDING ON THE CPU (as Qwen's, ▶ qwen3_5.py): the first socket keeps the table in its RAM and decodes a
        #   token's row, which the card copies across and un-rotates — where the card's family keeps mapped RAM as a copy on
        #   the card (khronos_opencl1_igpu), so the table neither takes the card's room nor crosses whole each token.
        if embed_on not in ("auto", "card", "cpu"):
            raise Refused("embed_on is 'auto', 'card' or 'cpu', not %r" % (embed_on,))
        if embed_on == "cpu" and self.P > 1:
            raise Refused("embed_on 'cpu' is one card's: a group's card 0 reads the table in place")
        self.EMBED_CPU = embed_on == "cpu" or (embed_on == "auto" and self.P == 1 and machine.silicon == "khronos_opencl1_igpu")
        self._plan()
        self.EXPERTS_FROM = experts_file.choose(experts_from, self.cpu_arena, video_ram(folder.config))
        # `experts_lock`: the sockets' experts in RAM locked there, so the system cannot swap them out (▶ `experts_file.lock_check`)
        self.EXPERTS_LOCK, self.LOCKED_GB = bool(experts_lock), 0.0
        if self.EPLAN is not None:
            experts_file.lock_check(self.EXPERTS_LOCK, self.EXPERTS_FROM, self.EPLAN.slot_bytes * self.EXPERTS * self.PARTS
                                    * sum(1 for l in self.run_layers if self.moe[l]))
        # `experts_register`: the sockets' experts in RAM registered with the tier's cards, a promotion copied straight from
        #   where it lies (▶ `experts_file.register_of`: NVIDIA's cards) — "auto" on one socket only: an expert split
        #   between sockets is stitched a down row a piece, past nn's NN__EXPERT__IN_PLACE_PIECES, and staged however it
        #   is registered.
        #   `MEASURED` node03 2026-10-26 (measurements/2026-10-26_v0200_r730_rows_node03.md): two sockets, 142 GB
        #   registered, 0 of 7,461 promotions in place — and at 150 GB with a 17.75 GB tier the load took 1,036 s against
        #   251 s unregistered, registering nothing in the end
        register = False if experts_register == "auto" and self.PARTS > 1 else experts_register
        self.EXPERTS_REGISTER, self.REGISTERED_GB = experts_file.register_of(register, self.EXPERTS_FROM, self.CARD_GB,
                                                                             machine.silicon), 0.0
        self.REGISTER_REQUIRED = experts_register is True
        if self.CARD_EXCLUSIVE and self.EXPERTS_FROM != "ram":
            raise Refused("an exclusive tier moves experts between the card and the sockets' memory — experts_from 'ram', "
                          "not %r" % self.EXPERTS_FROM)
        # an inclusive tier copies an expert from its place in a socket's memory, which only the experts loaded into RAM
        # have (`SRC`): from the disk or mapped it would ask for experts and never hold one — `MEASURED` 2026-10-12 from
        # the disk (measurements/2026-10-12_nvme_array_122b_glm.md), `REASONED` mapped, from `SRC`
        if self.CARD_GB and self.EXPERTS_FROM != "ram":
            raise Refused("a card's tier of experts is copied from the experts in RAM — experts_from 'ram', not %r"
                          % self.EXPERTS_FROM)
        # ⭐ FROM THE DISK, THE EXPERTS ARE A CACHE IN EACH WORKER'S OWN MEMORY — read from the worker's file into a slot
        #   when a token picks one that is not there, straight from the drive (`O_DIRECT`), the layer's least recently used
        #   given up for it; and a prompt's, a batch at a time, the next batch read while one computes. ⚖ *"the engine
        #   layer should be in charge of dealing with this … we just get the data directly from the disk"*. `"mapped"`
        #   maps the file instead, the system's page cache the cache. ▶ nn's loader, ai_z__glm_5_3's experts verbs.
        self.SLOTS = None
        # ⛔ the unfused layer names each expert's planes itself (`x%d`), so every expert must be resident: from the disk
        #   they are a cache, admitted only by ai_z__glm_5_3's experts verbs, and a plane of one not there answers a refusal
        #   the next verb meets as a type fault (OPCT) at the first MoE layer
        if self.EXPERTS_FROM == "disk" and self.EPLAN is not None and not self.FUSED:
            raise Refused("the unfused layer reads every expert in place, and from the disk they are a cache — "
                          "fused=True, or experts_from 'ram' or 'mapped'")
        if self.EXPERTS_FROM == "disk" and self.EPLAN is not None:
            n_moe = sum(self.moe[l] for l in self.run_layers)
            budget = experts_file.cache_bytes(self.cpu_arena, experts_ram_gb)
            # ⛳ WITH NO CARD OF ITS OWN, THE DENSE PART IS IN THE SAME RAM — its pool and two gigabytes for its buffers and
            #   the pictures' room come out of what the cache may take: `MEASURED` on node03 held to 60 GB, the CPU alone
            #   with the cache at the process's limit less 8 GB was killed for memory in its prompt (62.4 GB resident,
            #   measurements/2026-10-27_nn133_deepseek_disk_node03.md)
            if experts_ram_gb is None and (machine.silicon in ("x86_avx2", "host") or (
                    machine.silicon == "khronos_opencl1_igpu" and not own_memory(machine))):
                budget -= self.arena * self.P + (2 << 30)
            per = max(0, budget) // self.PARTS // self.EPLAN.slot_bytes
            self.SLOTS = min(self.EXPERTS * n_moe, per)
            if self.SLOTS < 3 * 16 + self.TOPK:
                raise Refused("%d experts a worker is too few for a cache of them — a prompt reads two batches of 16 at once"
                              % self.SLOTS)
        # ⭐ A PROMPT AS ROWS, `chunk` positions at a time through each layer (ai_z__glm_5_3's `…_rows` verbs): a routed expert
        #   is read once for every row of the chunk that picked it. ⛳ `REASONED`: with the experts on the disk the chunk
        #   wants to be large — a chunk's picks reach most of a layer's experts, so each is streamed about once a chunk
        self.CH = max(1, min(chunk, max_context)) if fused else 0
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        self._mask_layout()
        # ⭐ PICTURES ON THE CARD (NN-81): the vision tower's weights in a collection of their own and its scratch in another,
        #   sized for pictures of up to `picture_pixels` (0: none, and none of the card's memory for them — the tower then
        #   runs on the host, ▶ vision_glm.py) — where the pack carries the tower, a prompt goes as rows (the only way a
        #   picture goes in) and this machine holds the first layers
        self.PICTURE_PIXELS = int(picture_pixels or 0)
        self.VIS = bool(self.PICTURE_PIXELS and self.CH and self.EPLAN is not None and 0 in self.run_layers and self.P == 1
                        and "model.visual.blocks.0.attn.qkv.weight" in g)
        if self.VIS:
            self._vision_plan()

    @staticmethod
    def cn(i, name):
        """A CPU worker's buffer by name: `c_…` on the first, `c1_…` on the second."""
        return ("c_" if i == 0 else "c%d_" % i) + name

    @staticmethod
    def sfx(name, k):
        """Card `k`'s own name for a buffer or a table: card 0's is the name itself."""
        return name if k == 0 else "%s_%d" % (name, k)

    def _cut(self, l, n, s):
        """Card `s`'s part of layer `l`'s matrix `n` in a group: None for whole, ("rows", runs) or ("cols", (first, end)).
        A KDA layer's heads are channels [s·KWC, +KWC) of q, k, v, their convolutions, the gates' second projections and
        dt_bias, heads of β and A_log, and those columns of its output; an MLA layer's heads are rows of q_b and of the
        latent's expansion (a head's NOPE + VD rows together), and its output's columns. The rest is every card's whole."""
        if self.P == 1:
            return None
        a, rows, cols = "self_attn.", lambda at, w: ("rows", [(s * w + at, w)]), lambda w: ("cols", (s * w, (s + 1) * w))
        if self.kinds[l] == "d" and not n.startswith("mlp."):
            KWC = self.KWC
            cut = {a + x: rows(0, KWC) for x in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "q_conv1d.weight",
                                                  "k_conv1d.weight", "v_conv1d.weight", "f_b_proj.weight", "g_b_proj.weight")}
            cut.update({a + "b_proj.weight": rows(0, self.KHC), a + "A_log": cols(self.KHC), a + "dt_bias": cols(KWC),
                        a + "o_proj.weight": cols(KWC)})
            return cut.get(n)
        if self.kinds[l] == "f" and n.startswith(a) and ".indexer." not in n:
            return {a + "q_b_proj.weight": rows(0, self.QHC * self.NOPE),
                    a + "kv_b_proj.weight": rows(0, self.QHC * (self.NOPE + self.VD)),
                    a + "o_proj.weight": cols(self.QHC * self.VD)}.get(n)
        dense, shared = "mlp.%s_proj.weight", "mlp.shared_experts.%s_proj.weight"
        return {dense % "gate": rows(0, self.IC), dense % "up": rows(0, self.IC), dense % "down": cols(self.IC),
                shared % "gate": rows(0, self.MC), shared % "up": rows(0, self.MC), shared % "down": cols(self.MC)}.get(n)

    def _part(self, l, n, s):
        """Card `s`'s bytes of layer `l`'s matrix `n` (data, scales), or None where it holds it whole."""
        c = self._cut(l, n, s)
        return None if c is None else NL.dense_part(self.b, n, l, **{c[0]: c[1]})

    # ── ① the collections ─────────────────────────────────────────────────────────────────────────────────────
    def dense_names(self, l):
        """A layer's matrices that live on the card: all but the routed experts, and the indexer's place bias, which is
        decoded on the host."""
        return [n for n in self.b.names(l) if ".experts." not in n and not n.endswith("index_kpool_compress_ape")]

    def _plan(self):
        b, run = self.b, self.run_layers
        # a tensor is known by its layer's kind and its name: `o_proj` is 8192 wide in a KDA layer and 16384 in an MLA one
        keys = sorted({self.key(l, n) for l in run for n in self.dense_names(l)})
        first = {k: next(l for l in run if self.key(l, k[1]) == k and k[1] in self.dense_names(l)) for k in keys}
        # ⛳ in a group every card's part of a matrix is the same shape (▶ `_cut`), so card 0's plan is every card's
        plans = {}
        for k in keys:
            got = self._part(first[k], k[1], 0)
            plans[k] = NL.dense_plan(b, k[1], first[k]) if got is None else NL.SlotPlan(len(got[0]), 0, 0, len(got[1]))
        # ⭐ ONE COLLECTION A SHAPE, a slot a tensor of it in each layer — q, k and v share one, the two maps another —
        #   which is the collections' own design, and what keeps GLM's forty-odd tensors under the ruled thirty-two
        groups = {}
        for k in keys:
            groups.setdefault((plans[k].up_data, plans[k].down_lut), []).append(k)
        self.TYPE, self.WHICH, self.PLAN, self.WIDTH = {}, {}, {}, {}
        for t, members in enumerate(groups.values()):
            for i, k in enumerate(members):
                self.TYPE[k], self.WHICH[k], self.PLAN[k] = t, i, plans[k]
            self.WIDTH[t] = len(members)
        slots = {t: sum(1 for l in run for n in self.dense_names(l) if self.TYPE[self.key(l, n)] == t) for t in self.WIDTH}
        t = len(groups)
        # ⭐ THE EMBEDDING STAYS IN RAM, a table the card reads a row of in place; the card keeps the head
        self.T_HEAD, self.T_STATE, self.T_KV = t, t + 1, t + 2

        def plan_of(rec):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, self.BPLAN = plan_of(self.emb_rec)
        # what the program reads of the two tables once the engine holds them and this program does not
        self.EMB_ROW_BYTES, self.EMB_BITS = int(self.emb_rec["row_bytes"]), int(self.emb_rec["d"])
        self.EMB_DATA_BYTES, self.HEAD_BITS = len(self.emb_data), int(self.head_rec["d"])
        self.TABLE_BYTES = (len(self.emb_data) + len(self.emb_lut) + 255) // 256 * 256      # the rows, then the scales
        # KDA: the state (heads · head_dim² floats) in the up half, the conv window (q · k · v, three taps) in the down
        self.S_BYTES, self.W_BYTES = self.KHC * self.KD * self.KD * 4, 3 * self.KWC * 3 * 2
        self.SPLAN = NL.SlotPlan(self.S_BYTES, 0, 0, self.W_BYTES)
        # MLA: the rotated latents in the up half, the latent's expansion decoded to halves in the down
        self.C_BYTES, self.KVB_BYTES = self.max_context * self.LAT * 2, self.QHC * (self.NOPE + self.VD) * self.LAT * 2
        # and the indexer's caches after the latents: a key and a gate in a ring of `kpool` (read only to finish their
        # pool), a pooled key a pool
        ik = self.KP * self.ID * 2
        self.IK_AT, self.IG_AT, self.IP_AT = self.C_BYTES, self.C_BYTES + ik, self.C_BYTES + 2 * ik
        self.UP_BYTES = self.IP_AT + (self.max_context // self.KP + 1) * self.ID * 2
        self.KVPLAN = NL.SlotPlan(self.UP_BYTES, 0, 0, self.KVB_BYTES)
        n_d = sum(1 for l in run if self.kinds[l] == "d")
        plan_of_type = {self.TYPE[k]: self.PLAN[k] for k in keys}
        self.collections = ([(t, plan_of_type[t], slots[t]) for t in sorted(self.WIDTH)]
                            + [(self.T_HEAD, self.HPLAN, 1),
                               (self.T_STATE, self.SPLAN, max(1, n_d)), (self.T_KV, self.KVPLAN, max(1, len(run) - n_d))])
        self.arena = sum(p.slot_bytes * n for _, p, n in self.collections)
        # the CPU's one collection: an expert a slot — gate and up rows as one matrix, then down
        moe = [l for l in run if self.moe[l]]
        self.EPLAN = None
        if moe:
            r = lambda w: b.record(moe[0], EXPERT % (0, w))
            P, g, dn = self.PARTS, r("gate"), r("down")
            # a part: its inner rows of gate and of up, and its columns of every down row — a rotation block's multiple,
            # so each part's columns are rotated on their own and a row's one scale serves every part
            self.PI = self.MINTER // P
            if self.MINTER % P or self.PI % 512 or dn.row_bytes % P:
                raise Refused("an expert's inner width %d does not split into %d parts of whole rotation blocks" % (self.MINTER, P))
            self.EPLAN = NL.SlotPlan(2 * self.PI * g.row_bytes, 2 * self.PI * 2, dn.rows * (dn.row_bytes // P), dn.rows * 2)
            self.E_D = g.d
        self.cpu_arena = self.EPLAN.slot_bytes * self.EXPERTS * len(moe) * self.PARTS if moe else 0
        # the card's tier: whole experts — every inner row, every down column — a slot each, a collection after the KV's
        self.CPLAN, self.T_CEXP, self.CARD_SLOTS = None, None, 0
        self.T_STREAM, self.T_SLICE, self.SLICE = None, None, None
        if self.CARD_GB and moe:
            self.CPLAN = NL.SlotPlan(2 * self.MINTER * g.row_bytes, 2 * self.MINTER * 2, dn.rows * dn.row_bytes, dn.rows * 2)
            self.T_CEXP = self.T_KV + 1
            self.CARD_SLOTS = int(self.CARD_GB * (1 << 30)) // self.CPLAN.slot_bytes
            self.TIER_LEAST = len(moe) * self.CPLAN.slot_bytes             # the least tier it takes: a slot a MoE layer
            if self.CARD_SLOTS < len(moe):
                raise Refused("%.1f GB holds %d experts, fewer than one a MoE layer" % (self.CARD_GB, self.CARD_SLOTS))
            self.collections.append((self.T_CEXP, self.CPLAN, self.CARD_SLOTS))
            # ⭐ THE RING AND THE SLICE A PROMPT IS STREAMED AND READ LAYER-MAJOR THROUGH (NN-115 ② ③, ▶ qwen3_5.py's): the
            #   ring two of the largest batch of whole experts; the slice SLICE_CHUNKS chunks held on the card from one layer to
            #   the next — its rows' streams, hands, carries and routed rows, and the experts' scratch over all its rows
            #   — on every card of a group, each streaming its share of a layer's experts over its own link (NN-115 ④)
            if self.STREAM_ON:
                if self.CARD_EXCLUSIVE or not self.STREAM_CH:
                    raise Refused("stream takes an inclusive tier, the layers fused")
                self.T_STREAM, self.T_SLICE = self.T_CEXP + 1, self.T_CEXP + 2
                self.collections.append((self.T_STREAM, self.CPLAN, 2 * expert_tier.STREAM_BATCH_MAX))
                S = self.SLICE_CHUNKS * self.STREAM_CH
                P, H, I = S * self.TOPK, self.H, self.MINTER
                parts = [("srows", S * self.MULT * H * 2), ("hands", S * self.HAND), ("carries", S * (256 + 2 * H)),
                         ("rrows", S * H * 2),
                         ("scratch", 2 * S * H + 16 * P + 2 * P + 4 * P * I + 2 * P * I + 2 * P * I + 4 * P * H + 4096)]
                self.SLICE, at = dict(rows=S), 0
                for name, size in parts:
                    self.SLICE[name] = (at, size)
                    at = (at + size + 4095) // 4096 * 4096
                self.collections.append((self.T_SLICE, NL.SlotPlan(at, 0, 0, 256), 1))
                self.arena += self.CPLAN.slot_bytes * 2 * expert_tier.STREAM_BATCH_MAX + self.collections[-1][1].slot_bytes
            # exclusive: each MoE layer's even share of all but the landing slots held on the card, and not in RAM
            self.CARD_PER, self.CARD_SLICE, self.RAM_PER = 0, 0, self.EXPERTS
            if self.CARD_EXCLUSIVE:
                # in a group every card its own share, none of them the same expert, and the slice — `card_shared` of a
                #   card's slots a layer — on every card: what the cards hold together is RAM given back
                each = max(0, self.CARD_SLOTS - self.CARD_FREE) // len(moe)
                if self.P > 1:
                    self.CARD_SLICE = min(each, int(round((self.CARD_SHARED or 0.0) * self.CARD_SLOTS / len(moe))))
                self.CARD_PER = min((self.EXPERTS - self.TOPK - self.CARD_SLICE) // self.P, each - self.CARD_SLICE)
                self.RAM_PER = self.EXPERTS - self.CARD_SLICE - self.P * self.CARD_PER
                self.cpu_arena = self.EPLAN.slot_bytes * self.RAM_PER * len(moe) * self.PARTS
            self.arena += self.CPLAN.slot_bytes * self.CARD_SLOTS
            self.GATE_ROW, self.DOWN_ROW = g.row_bytes, dn.row_bytes
            if self.EPLAN.at_up_data != 0 or self.CPLAN.at_up_data != 0:
                raise Refused("a slot's gate-and-up codes are expected at its start")
        # ⭐ WHAT THE TIER MAY BE LENT (NN-93): each MLA layer's latents past those a conversation has written, a region a
        #   layer, `most` slots deep at the most
        self.LEND = None
        if self.CPLAN is not None and not self.CARD_EXCLUSIVE and self.LEND_ON:
            attn, most = sum(1 for l in run if self.kinds[l] == "f"), self.C_BYTES // self.CPLAN.slot_bytes
            if attn and most:
                self.LEND = dict(row=self.LAT * 2, regions=attn, most=most)
        self.SRC = {}                  # (layer, part) -> every expert's address in that socket's memory, as loaded

    @staticmethod
    def _section(big, arena, types, kinds, many=128, bigs=3, mains=2, work=0, works=1):
        typed = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in types)
        n_d = kinds.count("d")
        return ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
                "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:%d\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:%d\n"
                "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:%d\n"
                "nn__buffer_partials__size_bytes:%d\nnn__buffer_partials__qty:%d\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:%d\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                % (mains, many, big, bigs, work or 1048576, works if work else 8, (arena + (256 << 20)) >> 20, len(kinds) - n_d, n_d,
                   kinds, len(types), typed))

    def workers(self):
        return [(self.m.silicon, k) for k in range(self.P)] + [("x86_avx2", c["dev"]) for c in self.CPUS]

    def device_params(self):
        """The card's section and each CPU worker's: {worker: text}."""
        # every layer the index has a place for — and the MTP layer after them, where it is held
        kinds = "".join(self.kinds[l] for l in range(self.LAYERS + (1 if self.MTP is not None else 0)))
        big = max(8 << 20, 2 * self.QH * min(self.max_context, self.ITOPK + self.KP) * 4, self.VOCAB * 2)
        bigs = 3
        if self.CH:
            # a chunk's rows: streams, hands, carries, the routed rows (a socket each) and the tokens' rows
            big = max(big, self.CH * self.MULT * self.H * 2, self.CH * self.HAND, self.CH * (256 + 2 * self.H))
            bigs += 5 + len(self.CPUS)
        # a mask's buffer, one of the big ones: asked for past the 4 MiB class so it lands in theirs (▶ `_mask_layout`)
        ask = max(self.MASK_ASK.get(k, 0) for k in range(self.P))
        if ask:
            big, bigs = max(big, ask), bigs + 1
        # ⛳ the rows verbs' `work` buffer is the one buffer of its class, sized past every other big one so no other lands there
        self.WORK = max(self._work_bytes(), big + 4096) if self.CH else 0
        works = 1
        if self.CH and self.CPLAN is not None:
            # the card's tier over a chunk: its rows' share among the big ones, and its `experts_rows` scratch in the work
            # class beside `rwork`, which is then sized for the larger of the two
            n, P, I, H = self.CH, self.CH * self.TOPK, self.MINTER, self.H
            self.KRSCRATCH = 2 * n * H + 16 * P + 2 * P + 4 * P * I + 2 * P * I + 2 * P * I + 4 * P * H + 4096
            self.WORK, works, bigs = max(self.WORK, self.KRSCRATCH), 2, bigs + 1
        # ⛳ a draft's check keeps each KDA layer's state (4 MiB) after its first position: a main buffer each (▶ `_card_buffers`)
        kept = sum(1 for l in self.run_layers if self.kinds[l] == "d") if self.MTP is not None else 0
        card = self._section(big, self.arena, self.collections, kinds, bigs=bigs,
                             mains=2 + (4 + len(self.CPUS) if self.CH else 0) + kept,
                             work=self.WORK, works=works)
        if self.LEND:                                  # the links for the slots the tier may be lent (▶ nn's `__lendable`)
            card += "nn__expert__type%d__lendable:%d\n" % (self.T_CEXP, self.LEND["regions"] * self.LEND["most"])
        # the embedding table's RAM class — ⛳ with the other two RAM classes, which nn takes all or none of
        ram = ("nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
               "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
               "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % self.TABLE_BYTES)
        card += ram.replace("size_bytes:%d\n" % self.TABLE_BYTES, "size_bytes:256\n") if self.EMBED_CPU else ram
        # ⛳ a section with no collection is refused, so a run with no MoE layer gives the CPU one of a single slot
        out = {CARD: card}
        # a group's other cards: the same collections and classes, and no embedding to hold
        for k in range(1, self.P):
            out[k] = card.replace("nn__ram_buffer_table__size_bytes:%d\n" % self.TABLE_BYTES, "nn__ram_buffer_table__size_bytes:256\n")
        moe = sum(self.moe[l] for l in self.run_layers)
        for c in self.CPUS:
            slots = self.SLOTS if self.SLOTS is not None else getattr(self, "RAM_PER", self.EXPERTS) * moe
            experts = [(0, self.EPLAN, slots)] if self.EPLAN else [(0, NL.SlotPlan(4096, 0, 0, 4096), 1)]
            cbig, cbigs = 1 << 20, 3
            if self.CH and self.EPLAN:
                P = self.CH * self.TOPK
                cbig = max(cbig, 2 * self.CH * self.H + 16 * P + 2 * P + 8 * P * self.PI + 4 * P * self.H + 64,
                           self.CH * self.HAND)
                cbigs += 3
            if self.MASK_ASK.get(c["w"]):
                cbig, cbigs = max(cbig, self.MASK_ASK[c["w"]]), cbigs + 1
            out[c["w"]] = self._section(cbig, self.EPLAN.slot_bytes * slots if self.EPLAN else 0, experts,
                                        kinds, many=16, bigs=cbigs, mains=2 + (3 if self.CH else 0))
            if self.EMBED_CPU and c is self.CPUS[0]:   # the embedding's table, the first socket's (▶ `__init__`)
                out[c["w"]] += ram
        return out

    def _work_bytes(self):
        """What the busiest rows verb takes of its `work` buffer for a chunk — ▶ ai_z__glm_5_3's `…_rows` verbs, whose carving
        this follows, a 256-byte alignment a region."""
        n, H, KW, KD, KH = self.CH, self.H, self.KW, self.KD, self.KH
        QR, LAT, QH = self.QR, self.LAT, self.QH
        common = 24 * n + 256 * n + 3 * 2 * n * H
        kda = common + 2 * n * (3 * KW + 2 * KD + KH) + 2 * n * 3 * KW + 2 * 2 * n * KD + 4 * 2 * n * KW + 4 * n * KW
        # ⛳ and a block of the rows that attend every position before them: its absorbed queries, its answers in the latent,
        #   and its scores over at most the indexer's `topk` positions and a pool — ▶ AI_Z__GLM_5_3__MLA__ROWS_BLOCK
        blk, span = min(32, n), min(self.max_context, self.ITOPK + self.KP)
        mla = (common + 2 * n * (QR + LAT) + 2 * 2 * n * QR + 2 * n * LAT + 2 * n * QH * self.NOPE + 2 * 2 * n * QH * self.VD
               + 2 * n * (self.IH * self.ID + 2 * self.ID + self.IH) + 2 * 2 * blk * QH * LAT + 4 * blk * QH * span)
        # ⛳ and a block of the rows past the indexer's `topk`, chosen and attended at once: each row's pools' scores, its
        #   positions and their count, its latents gathered and its scores over them — ▶ ai_z__glm_5_3's `mla_rows`
        most = self.ITOPK + self.KP
        mla += (4 * blk * (self.max_context // self.KP + 2) + 4 * blk * most + 4 * blk + 2 * blk * most * LAT
                + 4 * blk * QH * most)
        mlp = common + 2 * n * 2 * self.INTER + 2 * 2 * n * self.INTER
        route = 24 * n + 3 * 2 * n * H + 2 * n * 2 * self.MINTER + 2 * 2 * n * self.MINTER + 2 * n * self.EXPERTS
        return max(kda, mla, mlp, route) + 32 * 256

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        self.state_of = {}
        for card in range(self.P):                    # card 0 first: the others' buffers take its signs
            m.act_as(card)
            loader = m.loader()
            if card == 0:
                self._head_and_embedding(loader)
                if self.VIS:
                    self._vision_load(loader)
            at = self.state_of[card] = {}
            for l in self.run_layers:
                for t in sorted({self.TYPE[self.key(l, n)] for n in self.dense_names(l)}):
                    loader.layer_width(l, t, self.WIDTH[t])
                for n in self.dense_names(l):
                    k = self.key(l, n)
                    got = self._part(l, n, card)
                    if got is None:
                        loader.load_dense(b, self.PLAN[k], self.TYPE[k], n, l, which=self.WHICH[k])
                    else:
                        loader.load_bytes(self.PLAN[k], self.TYPE[k], l, got[0], got[1], which=self.WHICH[k])
                t = self.T_STATE if self.kinds[l] == "d" else self.T_KV
                loader.layer_width(l, t, 1)
                at[l] = loader.reserve(t)
                loader.assign(l, t, 0, at[l])
            if self.CPLAN is not None:
                self._card_experts(loader, card)
            if loader.handed() != 0:
                raise Refused("card %d: %d slots were reserved and never published" % (card, loader.handed()))
            if card == 0:
                self._card_buffers()
                self.MASK_CELLS = {}
                self._mask_buffers(CARD, "mk")
            else:
                self._group_buffers(card)
                self._mask_buffers(card, self.sfx("mk", card))
            # every MLA layer's latent expansion — a card's heads of it — decoded once into halves beside its cache
            for l in self.run_layers:
                if self.kinds[l] == "f":
                    n = "self_attn.kv_b_proj.weight"
                    r = b.record(l, n)
                    m.must("(begin (nn__turboquant__decode %s %s %s %d %d %d) (type x))"
                           % (self.plane(l, n, "data"), self.plane(l, n, "lut"), self.kv(l, "kvb"),
                              self.QHC * (self.NOPE + self.VD), r.cols, r.d), "card %d: layer %d's latent expansion" % (card, l))
        self.state_at = self.state_of[0]
        if self.EPLAN is not None:
            {"disk": self._cache_experts, "mapped": self._map_experts}.get(self.EXPERTS_FROM, self._load_experts)(progress)
            for i, c in enumerate(self.CPUS):
                m.act_as(c["w"])
                self._cpu_buffers(i)
                self._mask_buffers(c["w"], self.cn(i, "mk"))
        for card in range(1, self.P):
            m.act_as(card)
            self._fused_tables(card)
        m.act_as(CARD)
        self._tables()
        if self.P > 1:
            # ⭐ THE GROUP: each card joins it on its own worker, card 0 first (▶ nn's `nn__partials__join`)
            for card in range(self.P):
                m.act_as(card)
                m.must("(nn__partials__join %d %d %d %d)" % (self.GROUP, card, self.P, min(4, self.P)),
                       "card %d joining the group" % card, "sys__value_true")
            m.act_as(CARD)
        if self.CPLAN is not None:
            self._lend()
        if self.VIS:
            self._vision_tables()
        if self.SOURCE:
            self._bindings()
        else:
            m.bind_values(dict(VOCAB=self.VOCAB, STOPS=0, SEEN=sampling.SEEN))      # what the sampler and `generate` read
        m.define_programs(self._procedures)
        if self.CPLAN is not None and self.CH and self.EPLAN is not None:
            self._calibrate_line()
        if self.CPLAN is not None and ((self.P > 1 and self.CARD_SHARED is None) or self.CARD_LANDING is None):
            self._from_speeds()                     # after the line: a visit's landings from its copy reading
        if self.CPLAN is not None and self.CH and self.EPLAN is not None and self.P > 1:
            # the other cards' blocks read the bindings through `names`, taken when the programs were defined: again, so
            #   their tiers' notes see the threshold just bound
            m.view("names")
        self.global_ = None                         # what was left of global.pkl, read by the buffers
        self.b.release()                            # every layer's file let go, now on the card or in RAM (▶ `Bundle.release`)

    def _head_and_embedding(self, loader):
        """The head into its slot and the embedding into its RAM buffer, before any layer — then the program lets go of
        its own copies of them: global.pkl's two records and the bytes made of them. The layers then load beside the
        engine's copy alone, so the load holds one tensor at a time above the model, not the two tables.
        ⛳ THE HEAD HAS A COLLECTION OF ITS OWN AND THE EMBEDDING IS THE ONE RAM BUFFER, so neither comes first at another
          slot's or buffer's expense: every address is what it was with the tables loaded last.
        `MEASURED` (▶ measurements/2026-10-13_cpu_without_avx2.md): every slot and buffer at its offset on the 14B, the 27B,
        the 35B and GLM; the 14B through x86_avx2 holds 7.36 GB after its load, against 9.98, and peaks at 7.38, against 10.58."""
        m = self.m
        loader.layer_width(0, self.T_HEAD, 1)
        self.HEAD_AT = loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
        if self.EMBED_CPU:                          # the first socket's own (▶ `__init__`)
            m.act_as(self.CPUS[0]["w"])
        self.EMB_AT = m.buffer(self.TABLE_BYTES, self.cn(0, "embedding") if self.EMBED_CPU else "EMBEDDING", ram=True)
        m.write(self.EMB_AT, self.emb_data)
        m.write(self.EMB_AT + self.EMB_DATA_BYTES, self.emb_lut)
        if self.EMBED_CPU:
            m.act_as(CARD)
        for name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
            del self.global_[name]
        self.head_rec = self.emb_rec = None
        self.head_data = self.head_lut = self.emb_data = self.emb_lut = None

    def _calibrate_line(self):
        """`nn__expert_major__vram_promotion_threshold_picks`, measured and bound (▶ `expert_tier.calibrate`): one socket's
        `experts_rows` stands for the CPUs, the sockets running theirs at once."""
        H, K = self.H, self.TOPK
        first = next(l for l in self.run_layers if self.moe[l])
        held = set(self.CARD_HELD.get(first, ())) if self.CARD_EXCLUSIVE else set()
        self.CALIBRATION = expert_tier.calibrate(
            self.m, CARD, self.CPUS[0]["w"], self.KRS_AT, self.CPLAN.slot_bytes, self.at[self.cn(0, "hands")], self.HAND,
            2 * H, (2 * H + 2 * K + 7) & ~7, K, self.EXPERTS,
            lambda n: "(ai_z__glm_5_3__experts_rows %s exr%d_0 %s %d)" % (self.cn(0, "hands"), next(l for l in self.run_layers if self.moe[l]),
                                                                       self.cn(0, "rrows"), n),
            self.CH, fixed=self.CARD_LINE, pool=[x for x in range(self.EXPERTS) if x not in held])
        self.EXPERT_LINE = self.CALIBRATION["line"]

    def _int_array(self, name, values):
        """A node array of integers, bound by name — filled a hundred at a time, as a program must fit one heap chunk."""
        m = self.m
        m.node_array(name, len(values))
        for at in range(0, len(values), 100):
            m.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set %s %d %d)" % (name, i, v)
                                                     for i, v in enumerate(values[at:at + 100], at)),
                   "the array %s" % name, "sys__value_true")

    # ── the latent cache's unreached rows lent to the tier (NN-93) ─────────────────────────────────────────────
    # ⚖ 2026-10-17: *"progressively shrink the l1 cache as the context grows, maybe in chunks of 8k tokens... this way we
    # dont need the users to set a max tokens or to size the l1 cache at all"*. An MLA layer's latents are the context's
    # whole length and position p's row is written at p (▶ `layer`, the rows' `at`), read only below the conversation's
    # count, so the rows past them are memory nothing reads. The tier holds experts there — slots numbered from each
    # region's end — and gives back LEND_CHUNK rows' worth before a conversation reaches them: `_room` before every write,
    # and when a conversation comes onto the card (`reset`, `state_write`) the slots its length leaves are lent again.
    # ▶ qwen3_5's, the same over its warm ring.
    LEND_CHUNK = 8192

    def _lend(self):
        """The tier lent every MLA layer's latents' tail — the regions' ends on the slots' alignment, each `depth` slots
        deep — then all of them live, the cache holding nothing yet. In a group every card's tier its own card's cache, as
        deep as the least of them, so every card holds as many lent slots at every length (NN-99)."""
        L, b = self.LEND, self.CPLAN.slot_bytes
        if L is None:
            return
        starts = [[self.state_of[k][l] + self.KVPLAN.at_up_data for l in self.run_layers if self.kinds[l] == "f"]
                  for k in range(self.P)]
        ends = [[(s + self.C_BYTES) // 4096 * 4096 for s in ss] for ss in starts]
        depth = min((e - (s + 4095) // 4096 * 4096) // b for ss, es in zip(starts, ends) for s, e in zip(ss, es))
        if depth <= 0:
            self.LEND = None
            return
        for k in range(self.P):
            self.m.act_as(k)
            self._int_array(self.sfx("lend_ends", k), ends[k])
            self.m.must("(nn__expert__lend %d %d %s)" % (self.T_CEXP, depth, self.sfx("lend_ends", k)),
                        "card %d's cache rows lent to its tier" % k)
        self.m.act_as(CARD)
        L.update(starts=starts[0], ends=ends[0], depth=depth, live=0, until=-1)
        self.LENT_GB = depth * len(ends[0]) * b / 2 ** 30
        self._room(0, exact=True)

    def _room(self, n, exact=False):
        """The cache's first `n` positions free to be written: the lent slots in their rows withheld — the rows rounded up
        to LEND_CHUNK, so the tier gives way a chunk at a time. `exact`: set for a conversation of `n` positions coming
        onto the card, the slots past its rows lent again."""
        L = self.LEND
        if L is None or (not exact and n <= L["until"]):
            return
        rows = -(-max(0, n) // self.LEND_CHUNK) * self.LEND_CHUNK
        b, need = self.CPLAN.slot_bytes, rows * L["row"]
        deep = max(0, min(L["depth"], min((e - s - need) // b for s, e in zip(L["starts"], L["ends"]))))
        live = deep * len(L["ends"])
        if live != L["live"]:
            for k in range(self.P):                 # every card's tier gives way together
                self.m.act_as(k)
                got = self.m.must("(nn__expert__lent %d %d)" % (self.T_CEXP, live), "card %d's tier's lent slots" % k)
                if got != live:
                    raise Refused("card %d's tier holds %d lent slots, not %d" % (k, got, live))
            self.m.act_as(CARD)
            L["live"] = live
        free = min(e - deep * b - s for s, e in zip(L["starts"], L["ends"]))
        L["until"] = min(self.max_context, free // L["row"])

    def lent(self):
        """The cache's rows lent to the tier now: (slots live, GB), or None where the tier is lent none."""
        L = self.LEND
        return None if L is None else (L["live"], L["live"] * self.CPLAN.slot_bytes / 2 ** 30)

    def _card_experts(self, loader, card=0):
        """Card `card`'s tier, stood at boot: each MoE layer's band of every expert — and, given a router log, its most
        picked ones written whole into the card's slots, an even share of them a layer. `CARD_HELD` every card's together:
        in a group each exclusive tier starts with the slice — the layer's `CARD_SLICE` most picked, or its first ones
        without a log, on every card — and its own share of the rest, card k the k-th."""
        m, b = self.m, self.b
        moe = [l for l in self.run_layers if self.moe[l]]
        for l in moe:
            loader.layer_width(l, self.T_CEXP, self.EXPERTS)
            if self.T_STREAM is not None:               # ...and the ring's, which every MoE layer streams through
                loader.layer_width(l, self.T_STREAM, self.EXPERTS)
        if self.T_SLICE is not None:                    # the slice a prompt is read layer-major in, its one slot — every card's
            loader.layer_width(0, self.T_SLICE, 1)
            loader.assign(0, self.T_SLICE, 0, loader.reserve(self.T_SLICE))
        if card == 0:
            self.CARD_HELD = {}
        if not self.CARD_HOT and not self.CARD_EXCLUSIVE:
            return
        log = np.load(self.CARD_HOT) if self.CARD_HOT else None
        logged = [int(l) for l in log["layers"]] if log is not None else []
        per = self.CARD_PER if self.CARD_EXCLUSIVE else self.CARD_SLOTS // len(moe)
        s = self.CARD_SLICE if self.CARD_EXCLUSIVE else 0
        for l in moe:
            if log is None:
                # exclusive, no router log: the slice and the card's share, until it learns
                hot = list(range(s)) + list(range(s + card * per, s + (card + 1) * per))
            else:
                if l not in logged:
                    raise Refused("the router log has no picks for layer %d" % l)
                counts = np.bincount(log["picks"][:, logged.index(l), :].reshape(-1).astype(np.int64), minlength=self.EXPERTS)
                order = [int(x) for x in np.argsort(-counts, kind="stable")]
                # an exclusive group's cards dealt the most picked in turn past the slice, so none holds another's
                hot = order[:s] + order[s:][card::self.P][:per] if self.CARD_EXCLUSIVE else order[:per]
            view = np.frombuffer(b.blob(l), dtype=np.uint8)
            for x in hot:
                at = loader.reserve(self.T_CEXP)
                for into, piece in self._slot_pieces(view, l, x, 0, whole=True):
                    m.write_from(at + into, piece.ctypes.data, piece.nbytes)
                loader.assign(l, self.T_CEXP, x, at)
            self.CARD_HELD[l] = self.CARD_HELD.get(l, []) + (hot[s:] if card else hot)   # the slice named once

    def _load_experts(self, progress):
        """Every MoE layer's routed experts into the CPU workers' memory, each its part of every expert, straight from the
        mapped bundle — a layer at a time for every worker, so a layer's file is read once."""
        m, b, p, P, I = self.m, self.b, self.EPLAN, self.PARTS, self.PI
        loaders = {}
        for c in self.CPUS:
            m.act_as(c["w"])
            loaders[c["w"]] = m.loader()
        moe = [l for l in self.run_layers if self.moe[l]]
        for i, l in enumerate(moe):
            view = np.frombuffer(b.blob(l), dtype=np.uint8)
            for c in self.CPUS:
                m.act_as(c["w"])
                loader, s = loaders[c["w"]], c["part"]
                loader.layer_width(l, 0, self.EXPERTS)
                held = set(self.CARD_HELD.get(l, ())) if self.CARD_EXCLUSIVE else set()
                for x in range(self.EXPERTS):
                    if x in held:                 # exclusive: on the card, so not here — its source 0
                        self.SRC.setdefault((l, s), []).append(0)
                        continue
                    at = loader.reserve(0)
                    for into, piece in self._slot_pieces(view, l, x, s):
                        m.write_from(at + into, piece.ctypes.data, piece.nbytes)
                    loader.assign(l, 0, x, at)
                    self.SRC.setdefault((l, s), []).append(at)
            view = piece = None                   # a slice left bound keeps the map open: `close` refuses it
            if b.release(l):                      # its experts copied out: the file let go (▶ `Bundle.release`)
                raise Refused("layer %d's file is still mapped after its experts were copied out" % l)
            if progress:
                progress(i + 1, len(moe))
        for c in self.CPUS:
            m.act_as(c["w"])
            if loaders[c["w"]].handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loaders[c["w"]].handed())
        self.LOCKED_GB = experts_file.lock_experts(self.EXPERTS_LOCK, [a for xs in self.SRC.values() for a in xs], p.slot_bytes)
        self.REGISTERED_GB = experts_file.register_experts(m, self.EXPERTS_REGISTER, range(self.P),
                                                           [a for xs in self.SRC.values() for a in xs], p.slot_bytes,
                                                           self.REGISTER_REQUIRED)

    def _slot_pieces(self, view, l, x, s, whole=False):
        """Expert `x` of layer `l`, part `s`, as (offset in its slot, bytes) pieces (▶ `expert_tier.part_pieces`). `whole`:
        the expert entire, as the card's tier holds it."""
        p, P, I = (self.CPLAN, 1, self.MINTER) if whole else (self.EPLAN, self.PARTS, self.PI)
        g, u, d = (self.b.record(l, EXPERT % (x, w)) for w in ("gate", "up", "down"))
        return expert_tier.part_pieces(view, g, u, d, p, s, P, I)

    def _cache_experts(self, progress):
        """Every MoE layer's routed experts as a cache in each socket's memory, read from its file (▶ `experts_file`)."""
        moe = [l for l in self.run_layers if self.moe[l]]
        self.BACKING = experts_file.cache_experts(self.m, self.b, [(c["w"], c["part"], self.PARTS) for c in self.CPUS], self.EPLAN,
                                                  moe, self.EXPERTS, self.experts_dir, self._slot_pieces,
                                                  [l for l in range(self.LAYERS) if self.moe[l]], progress=progress)

    def _backing(self, w, l):
        """The experts table's last cell where the experts are a cache: worker `w`'s backing of layer `l`."""
        return [self.BACKING[(w, l)]] if self.SLOTS is not None else []

    def _experts_tail(self, w, l):
        """An experts table's cells past its fixed length: its backing — an integer 0 in its place where there is none and a
        mask follows — and the mask's cells for worker `w`'s share of layer `l` (▶ AI_Z__GLM_5_3__EXP__MASK)."""
        back = self._backing(w, l)
        return back if self.b.mask is None else (back or [0]) + self._masked(w, l, "moe")

    def _map_experts(self, progress):
        """Every MoE layer's routed experts from each socket's file, mapped over its slots (▶ `experts_file`)."""
        moe = [l for l in self.run_layers if self.moe[l]]
        experts_file.map_experts(self.m, self.b, [(c["w"], c["part"], self.PARTS) for c in self.CPUS], self.EPLAN, moe,
                                 self.EXPERTS, self.experts_dir, self._slot_pieces,
                                 [l for l in range(self.LAYERS) if self.moe[l]], progress=progress)

    def _card_buffers(self):
        m, H, KW = self.m, self.H, self.KW
        self.at = {}                   # a buffer's address by name, for reading one back
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        self.signs = signs
        final_w = np.frombuffer(np.asarray(self.global_["model.language_model.norm.weight"]["vbr_data"]).tobytes(),
                                dtype="<f2").astype(np.float64)[:H]
        m.buffer(1024, "signs", _half(signs[:512]))
        m.buffer(1024, "ones512", _half(np.ones(512)))
        m.buffer(H * 2, "signs_row", _half(np.tile(signs[:512], H // 512)))
        m.buffer(H * 2, "fnw", _half(final_w))                   # GLM's norm is a plain gain
        m.buffer(H * 2, "zero_h", _half(np.zeros(H)))
        self.STREAMS_AT = m.buffer(self.MULT * H * 2, "streams")
        if self.P > 1:
            # ⛳ the group's sums land in a buffer of their own: between two reduces the other cards read it, so nothing but a
            #   reduce writes it (▶ nn's `contracts/objects/partials.cuh`)
            m.buffer(self.MULT * H * 2, "tsum")
            if self.CH:
                m.buffer(self.CH * H * 2, "tsums")         # a chunk's rows' sums, the same
        for nm, n in (("u", H), ("hu", H), ("e", H), ("xin", H), ("h", H), ("hr", H), ("y", H), ("sh", H),
                      ("routed", H), ("hm", H), ("hn", H), ("hnr", H),
                      ("qkv", 3 * KW), ("conved", 3 * KW), ("fb", KW + self.KH), ("fa", self.KD), ("far", self.KD),
                      ("ga", self.KD), ("gar", self.KD), ("gate", KW), ("ko", KW), ("kor", KW),
                      ("qa", self.QR), ("qn", self.QR), ("qnr", self.QR), ("q", self.QH * self.NOPE),
                      ("ckv", self.LAT), ("ckn", self.LAT), ("qabs", self.QH * self.LAT), ("olat", self.QH * self.LAT),
                      ("vh", self.QH * self.VD), ("vhr", self.QH * self.VD),
                      ("gu", 2 * self.INTER), ("act", self.INTER), ("actr", self.INTER),
                      ("router", self.EXPERTS), ("sgu", 2 * self.MINTER), ("sact", self.MINTER), ("sactr", self.MINTER)):
            self.at[nm] = m.buffer(n * 2, nm)
        if self.MTP is not None:
            # the MTP layer's: its input's two halves and their rotation, its one residual, its head's norm and rotation
            for nm, n in (("mcat", 2 * H), ("mcatr", 2 * H), ("mres", H), ("mhn", H), ("mhnr", H)):
                self.at[nm] = m.buffer(n * 2, nm)
            # ⭐ A DRAFT'S CHECK, TWO POSITIONS A LAYER (▶ `_check_procedures`): the second position's streams, hand and
            #   routed sum, its MoE sites' own scratch (a site's carry lives in it from the router to the close), the head's
            #   state at each of the two positions — and each KDA layer's state and window as they were after the first
            #   position, given back when the draft is refused
            self.STREAMS2_AT = m.buffer(self.MULT * H * 2, "streams2")
            for nm, n in (("hand2", H + self.TOPK), ("routed2", H), ("hnA", H), ("hnB", H)):
                self.at[nm] = m.buffer(n * 2, nm)
            m.buffer(1 << 19, "gscratch2")
            m.node_array("picks2", self.TOPK)
            m.node_array("CHECK", 2)
            for i in range(1, len(self.CPUS)):       # another socket's share of the second position's routed sum
                m.buffer(2 * H, "routed2_%d" % i)
            for l in self.run_layers:
                if self.kinds[l] == "d":
                    m.buffer(self.S_BYTES, "kbS%d" % l)
                    m.buffer(self.W_BYTES, "kbW%d" % l)
        m.buffer(256, "mix")
        m.buffer(64, "weights")
        m.buffer(64, "res")
        sampling.buffers(m)
        for i in range(1, len(self.CPUS)):
            m.buffer(2 * H, "routed%d" % i)                    # another socket's share of the routed sum
        if self.CPLAN is not None:
            m.buffer(2 * H, "routed_k")                        # the card's tier's share
            m.buffer(2 * 4 * self.TOPK * self.MINTER, "kscratch")
            if self.CH:
                # a chunk's: its rows' share, and `experts_rows`' scratch — the rows' inputs, the pairs and weights, gate-and-up,
                # the activations twice, the fp32 sum
                m.buffer(2 * self.CH * H, "rrows_k")
                self.KRS_AT = m.buffer(self.KRSCRATCH, "krscratch")
        self._answer_buffers()                      # what `generate` made, the end tokens it watches for
        m.buffer(1 << 19, "gscratch")                          # ai_z__glm_5_3's scratch, its first bytes the MLP site's carry
        self.at["hand"] = m.buffer(2 * (H + self.TOPK), "hand")
        # ⛳ a query attends at most `topk` positions and the filling pool's, so the scores need no more than that
        self.ATTEND = min(self.max_context, self.ITOPK + self.KP)
        m.buffer(2 * self.QHC * self.ATTEND * 4, "scores")
        # the indexer's: the pools' scores, the chosen positions (and their count), their latents gathered, and each MLA
        # layer's place bias, decoded here — it is added to the gates, not multiplied, so the card needs it as it is
        m.buffer((self.max_context // self.KP + 1) * 4, "iscores")
        self.at["iindex"] = m.buffer((self.ITOPK + self.KP + 1) * 4, "iindex")
        m.buffer((self.ITOPK + self.KP) * self.LAT * 2, "igather")
        import nn_unpack_model as NU
        for l in self.run_layers:
            if self.kinds[l] == "f":
                r = self.b.record(l, "self_attn.indexer.index_kpool_compress_ape")
                ape = np.asarray(NU.decode_record(r.rec, r.bundle.blob(l)), dtype=np.float64).reshape(self.KP, self.ID)
                m.buffer(self.KP * self.ID * 2, "ape%d" % l, _half(ape))
        self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        m.node_array("picks", self.TOPK)
        if self.CH:
            # a chunk's rows: its streams, the hands and carries of its MoE sites, the routed sums back, its tokens
            CH = self.CH
            self.SROWS_AT = m.buffer(CH * self.MULT * H * 2, "srows")
            m.buffer(CH * self.HAND, "hands")
            m.buffer(CH * (256 + 2 * H), "carries")
            m.buffer(CH * H * 2, "rrows")
            for i in range(1, len(self.CPUS)):
                m.buffer(CH * H * 2, "rrows%d" % i)
            m.node_array("PROMPT", CH)            # a chunk's tokens
            m.node_array("nrows", 1)
            m.buffer(self.WORK, "rwork")                         # the rows verbs' — ▶ `_work_bytes`

    def _group_buffers(self, k):
        """Card `k`'s buffers in a group, by its own names (▶ `sfx`): the streams (and zeros, its part of the embedding's
        sum), the group's sum, the sites' scratch, the attention's scores and the indexer's, the hand, the picks and its
        tier's share — what its tables and its layers' program name. ⛳ The indexer's place biases are every card's: each
        card chooses the positions an MLA query attends, the same ones."""
        m, H, n = self.m, self.H, lambda name: self.sfx(name, k)
        m.buffer(1024, n("signs"), _half(self.signs[:512]))
        m.buffer(H * 2, n("zero_h"), _half(np.zeros(H)))
        self.at[n("streams")] = m.buffer(self.MULT * H * 2, n("streams"))
        m.buffer(self.MULT * H * 2, n("zstreams"), _half(np.zeros(self.MULT * H)))
        m.buffer(self.MULT * H * 2, n("tsum"))
        m.buffer(1 << 19, n("gscratch"))
        m.buffer(2 * (H + self.TOPK), n("hand"))
        m.node_array(n("picks"), self.TOPK)
        m.buffer(2 * self.QHC * self.ATTEND * 4, n("scores"))
        m.buffer((self.max_context // self.KP + 1) * 4, n("iscores"))
        m.buffer((self.ITOPK + self.KP + 1) * 4, n("iindex"))
        m.buffer((self.ITOPK + self.KP) * self.LAT * 2, n("igather"))
        import nn_unpack_model as NU
        for l in self.run_layers:
            if self.kinds[l] == "f":
                r = self.b.record(l, "self_attn.indexer.index_kpool_compress_ape")
                ape = np.asarray(NU.decode_record(r.rec, r.bundle.blob(l)), dtype=np.float64).reshape(self.KP, self.ID)
                m.buffer(self.KP * self.ID * 2, n("ape%d" % l), _half(ape))
        if self.CPLAN is not None:
            m.buffer(2 * H, n("routed_k"))
            m.buffer(2 * 4 * self.TOPK * self.MINTER, n("kscratch"))
        if self.CH:
            # a chunk's rows, as card 0's: the streams and their zeros (its part of the chunk's embedding's sum), the hands
            #   and carries its route writes, the zeros its close adds in place of a routed sum, the sums, the work
            CH, MH = self.CH, self.MULT * H
            m.buffer(CH * MH * 2, n("srows"))
            m.buffer(CH * MH * 2, n("zsrows"), _half(np.zeros(CH * MH)))
            m.buffer(CH * self.HAND, n("hands"))
            m.buffer(CH * (256 + 2 * H), n("carries"))
            m.buffer(CH * H * 2, n("zrows"), _half(np.zeros(CH * H)))
            m.buffer(CH * H * 2, n("tsums"))
            m.buffer(self.WORK, n("rwork"))
            if self.CPLAN is not None and self.CARD_EXCLUSIVE:
                # an exclusive tier's share of a chunk's routed rows, and `experts_rows`' scratch (as card 0's)
                m.buffer(2 * CH * H, n("rrows_k"))
                m.buffer(self.KRSCRATCH, n("krscratch"))

    def _cpu_buffers(self, i):
        m, H, c = self.m, self.H, lambda name: self.cn(i, name)
        m.buffer(1024, c("signs"), _half(self.signs[:512]))
        for nm, n in (("hr", H), ("gu", 2 * self.MINTER), ("act", self.MINTER), ("actr", self.MINTER), ("e", H), ("routed", H)):
            self.at[c(nm)] = m.buffer(n * 2, c(nm))
        m.buffer(64, c("w"))
        if self.EMBED_CPU and i == 0:                 # a token's decoded row of the embedding (▶ `__init__`)
            m.buffer(H * 2, c("emb"))
        m.buffer(2 * (H + self.TOPK), c("hand"))
        m.buffer(2 * H, c("zero"), _half(np.zeros(H)))
        if self.MTP is not None:                      # a draft's check: the second position's hand and routed sum
            m.buffer(2 * (H + self.TOPK), c("hand2"))
            m.buffer(2 * H, c("routed2"))
        m.buffer(8 * self.TOPK * self.MINTER, c("scratch"))
        if self.CH:
            # a chunk's hands, its routed rows, and `experts_rows`' scratch: the inputs, the pairs and weights, the gate-and-up,
            # activation and rotated rows of every pick, and a fp32 sum for every pick — ▶ ai_z__glm_5_3__experts_rows
            CH, P, I = self.CH, self.CH * self.TOPK, self.PI
            self.at[c("hands")] = m.buffer(CH * self.HAND, c("hands"))
            m.buffer(CH * H * 2, c("rrows"))
            m.buffer(2 * CH * H + 16 * P + 2 * P + 8 * P * I + 4 * P * H + 64, c("rscratch"))

    # ── the rank-1 mask, when the folder's LoRA is one (▶ the package's `*__MASK` cells) ───────────────────────────────
    def _mask_layout(self):
        """Where each worker keeps the mask's tensors: the card the attention's and the shared expert's `r` and `v`, each
        socket the routed experts' `r` and its part of every expert's `V` — the columns of its share of the inner width,
        whole rotation blocks, so they meet the activation it rotated on its own — and, with a card's tier, the card the
        routed experts' `r` and the whole of `V`, since the tier holds whole experts. In a group each card its own columns of
        the attention's and the shared expert's `v` — its heads', its share of the inner width — so its term is a partial the
        group's sum adds as it adds the matrix's (NN-99), and with a tier the routed experts' as one card's.
        `MASK_PLACE[(worker, l, site)]` the two tensors' (start, count) in halves, after the scratch the verbs' dots take;
        `MASK_ASK` each buffer's request."""
        self.MASK_PLACE, self.MASK_ASK, self._mask_data = {}, {CARD: 0}, {}
        if self.b.mask is None:
            return
        from safetensors.numpy import load_file
        t = load_file(self.b.mask["path"])
        # the verbs' dots: a chunk's picks 8 bytes each, a chunk's rows 4, a position's picks 12
        self._mask_scratch = 8 * max(self.CH, 1) * self.TOPK + 256
        at = {k: self._mask_scratch for k in range(self.P)}
        for l in self.run_layers:
            if l == self.MTP:
                continue
            attn_in = self.KW if self.kinds[l] == "d" else self.QH * self.VD
            for site, cols in (("attn", attn_in), ("shared", self.MINTER)):
                if "layers.%d.%s.r" % (l, site) in t:
                    r, v = t["layers.%d.%s.r" % (l, site)], t["layers.%d.%s.v" % (l, site)]
                    if v.size != cols:
                        raise Refused("the mask's layer %d %s is %d wide, where this model's is %d" % (l, site, v.size, cols))
                    w = cols // self.P                  # a card's columns: its heads', its share of the inner width
                    for k in range(self.P):
                        self._mask_put(at, k, l, site, r, v.reshape(-1)[k * w:(k + 1) * w], w)
            if "layers.%d.moe.r" % l in t:
                V = t["layers.%d.moe.V" % l]
                if not self.moe[l] or self.EPLAN is None or V.shape != (self.EXPERTS, self.MINTER):
                    raise Refused("the mask's layer %d routed experts are %s, where this model runs %s"
                                  % (l, V.shape, (self.EXPERTS, self.MINTER) if self.moe[l] else "none"))
                for c in self.CPUS:
                    at.setdefault(c["w"], self._mask_scratch)
                    part = V[:, c["part"] * self.PI:(c["part"] + 1) * self.PI]
                    self._mask_put(at, c["w"], l, "moe", t["layers.%d.moe.r" % l], part, self.EXPERTS * self.PI)
                if self.CPLAN is not None:
                    for k in range(self.P):
                        self._mask_put(at, k, l, "moe", t["layers.%d.moe.r" % l], V, self.EXPERTS * self.MINTER)
        for w, n in at.items():
            if any(k[0] == w for k in self.MASK_PLACE):
                self.MASK_ASK[w] = max(n, (4 << 20) + 256)

    def _mask_put(self, at, w, l, site, r, v, cols):
        """A site's `r` and `v` placed in worker `w`'s mask buffer, refused where they are not this model's widths."""
        if r.size != self.H or v.size != cols:
            raise Refused("the mask's layer %d %s is %d and %d wide, where this model's is %d and %d"
                          % (l, site, r.size, v.size, self.H, cols))
        place = []
        for a in (r, v):
            place.append((at[w] // 2, a.size))
            at[w] += (2 * a.size + 255) & ~255
        self.MASK_PLACE[(w, l, site)] = place
        self._mask_data[(w, l, site)] = (r, v)

    def _mask_buffers(self, w, name):
        """Worker `w`'s mask buffer, filled, and each of its sites' three cells: views of its `r`, its `v` and the scratch."""
        if not self.MASK_ASK.get(w):
            return
        base = self.m.buffer(self.MASK_ASK[w], name)
        for (w_, l, site), place in self.MASK_PLACE.items():
            if w_ != w:
                continue
            for (start, _), a in zip(place, self._mask_data[(w, l, site)]):
                self.m.write(base + 2 * start, np.ascontiguousarray(a, dtype="<f2").tobytes())
            self.MASK_CELLS[(w, l, site)] = [self.rng(name, *place[0]), self.rng(name, *place[1]),
                                             self.rng(name, 0, self._mask_scratch // 2)]

    def _masked(self, w, l, site):
        """A table's mask cells for worker `w`'s site of layer `l`: none where the model has no mask, an integer 0 where it
        leaves this site alone."""
        if self.b.mask is None:
            return []
        return list(self.MASK_CELLS.get((w, l, site), [0]))

    # ── the planes ────────────────────────────────────────────────────────────────────────────────────────────
    def key(self, l, n):
        return (self.kinds[l], n)

    def plane(self, l, n, which):
        k = self.key(l, n)
        p = self.PLAN[k]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d %d %d %d)" % (l, self.TYPE[k], self.WHICH[k], at, size)

    def state(self, l, which, at=0, size=None):
        """A KDA layer's state (`S`) or conv window (`W`), or a part of the window."""
        p = self.SPLAN
        base, whole = (p.at_up_data, self.S_BYTES) if which == "S" else (p.at_down_data, self.W_BYTES)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_STATE, base + at, whole if size is None else size)

    def kv(self, l, which, at=None, size=None):
        """An MLA layer's latent cache, one row of it (`at` an expression in bytes), or its decoded expansion."""
        p = self.KVPLAN
        if which == "kvb":
            return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, p.at_down_data, self.KVB_BYTES)
        if which in ("ikeys", "igates", "ipool"):
            at, n = dict(ikeys=(self.IK_AT, self.IG_AT - self.IK_AT), igates=(self.IG_AT, self.IP_AT - self.IG_AT),
                         ipool=(self.IP_AT, self.UP_BYTES - self.IP_AT))[which]
            return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, p.at_up_data + at, n)
        if at is None:
            return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, p.at_up_data, self.C_BYTES)
        return "(nn__expert__plane %d %d 0 (sys__add %d %s) %d)" % (l, self.T_KV, p.at_up_data, at, size)

    def gemv(self, l, n, x, out):
        """`out = W x` for a layer's matrix — its input rotated unless the matrix is lossless."""
        r = self.b.record(l, n)
        if r.d == 16:
            return "(nn__expert__multiply_fp16 %s %s %s %d %d)" % (self.plane(l, n, "data"), x, out, r.rows, r.cols)
        return ("(nn__turboquant__gemv %s %s %s %s %d %d %d)"
                % (self.plane(l, n, "data"), self.plane(l, n, "lut"), x, out, r.rows, r.cols, r.d))

    @staticmethod
    def rng(buf, at, n):
        return "(nn__vector__range %s %d %d)" % (buf, at, n)

    # ── ③ the procedures ─────────────────────────────────────────────────────────────────────────────────────
    def _hc(self, l, site):
        p = lambda n: self.plane(l, "hc_%s_%s" % (site, n), "data")
        return ("(nn__hyper__pre streams %s %s %s xin mix %d %d %d %r %r)"
                % (p("fn"), p("base"), p("scale"), self.H, self.MULT, self.ITERS, self.EPS, self.HC_EPS))

    def _post(self):
        return "(nn__hyper__post streams y mix streams %d %d)" % (self.H, self.MULT)

    def _kda(self, l):
        H, KW, KH, KD, s = self.H, self.KW, self.KH, self.KD, "self_attn."
        conv = " ".join("(nn__deltanet__conv_step %s %s %s %s %d)"
                        % (self.plane(l, s + "%s_conv1d.weight" % c, "data"), self.state(l, "W", i * KW * 3 * 2, KW * 3 * 2),
                           self.rng("qkv", i * KW, KW), self.rng("conved", i * KW, KW), KW) for i, c in enumerate("qkv"))
        return (" ".join(self.gemv(l, s + "%s_proj.weight" % c, "hr", self.rng("qkv", i * KW, KW)) for i, c in enumerate("qkv"))
                + " " + conv
                + " " + self.gemv(l, s + "f_a_proj.weight", "hr", "fa") + " (nn__hadamard__rotate fa signs far %d) " % KD
                + self.gemv(l, s + "f_b_proj.weight", "far", self.rng("fb", 0, KW)) + " "
                + self.gemv(l, s + "b_proj.weight", "hr", self.rng("fb", KW, KH)) + " "
                + self.gemv(l, s + "g_a_proj.weight", "hr", "ga") + " (nn__hadamard__rotate ga signs gar %d) " % KD
                + self.gemv(l, s + "g_b_proj.weight", "gar", "gate")
                + " (nn__kda__step %s conved fb %s %s gate %s ko %d %d %r %r)"
                % (self.state(l, "S"), self.plane(l, s + "dt_bias", "data"), self.plane(l, s + "A_log", "data"),
                   self.plane(l, s + "o_norm.weight", "data"), KH, KD, self.LOWER, self.EPS)
                + " (nn__hadamard__rotate ko signs kor %d) " % KW + self.gemv(l, s + "o_proj.weight", "kor", "y"))

    def _mla(self, l):
        s, LAT = "self_attn.", self.LAT
        stride = self.NOPE + self.VD
        return (self.gemv(l, s + "q_a_proj.weight", "hr", "qa")
                + " (nn__rmsnorm__apply qa %s qn %d %r) (nn__hadamard__rotate qn signs qnr %d) "
                % (self.plane(l, s + "q_a_layernorm.weight", "data"), self.QR, self.EPS, self.QR)
                + self.gemv(l, s + "q_b_proj.weight", "qnr", "q") + " "
                + self.gemv(l, s + "kv_a_proj_with_mqa.weight", "hr", "ckv")
                + " (nn__rmsnorm__apply ckv %s ckn %d %r) (nn__hadamard__rotate ckn signs %s %d)"
                % (self.plane(l, s + "kv_a_layernorm.weight", "data"), LAT, self.EPS, self.kv(l, "c", "at", LAT * 2), LAT)
                # ⛳ the attention's own scale is 1/√latent and the model's 1/√head, so the absorbed query carries their ratio
                + " (nn__attention__absorb q %s qabs %d %d %d %d %r)" % (self.kv(l, "kvb"), self.QH, self.NOPE, LAT, stride,
                                                                        float(np.sqrt(LAT / self.NOPE)))
                + " (nn__attention__decode qabs %s %s scores olat %d 1 %d 0 len)" % (self.kv(l, "c"), self.kv(l, "c"), self.QH, LAT)
                + " (nn__attention__expand olat %s vh %d %d %d %d %d)" % (self.kv(l, "kvb"), self.QH, self.VD, LAT, stride, self.NOPE)
                + " (nn__hadamard__rotate vh signs vhr %d) " % (self.QH * self.VD)
                + self.gemv(l, s + "o_proj.weight", "vhr", "y"))

    def _swiglu(self, l, pre, gu, act, actr, x, out, width):
        return (self.gemv(l, pre + "gate_proj.weight", x, self.rng(gu, 0, width)) + " "
                + self.gemv(l, pre + "up_proj.weight", x, self.rng(gu, width, width))
                + " (nn__swiglu__clamped %s %s %d %r) (nn__hadamard__rotate %s signs %s %d) "
                % (gu, act, width, self.LIMIT, act, actr, width)
                + self.gemv(l, pre + "down_proj.weight", actr, out))

    def _tables(self):
        """ai_z__glm_5_3's plane tables: a layer's attention site (`at`), its MLP site (`ml` dense, `mo` with experts), and on
        the CPU its experts' (`ex`, and `exr` over a prompt's chunk). ▶ the package's `contracts/objects/layer.cuh` for what
        each cell is. Made at every boot; the programs that name them are `_procedures`'."""
        if self.FUSED:
            self._fused_tables()
        if self.CH and self.EPLAN is not None:
            self._rows_tables()

    def _tier(self, k):
        """Card `k`'s tier table (▶ `expert_tier.tier`), with the share of its slots a layer every card keeps in a group —
        none until it is known (`_from_speeds`); an exclusive group's, the slice dealt at boot."""
        moe = sum(1 for l in self.run_layers if self.moe[l])
        part = self.CARD_SHARED or 0.0
        shared = int(round(part * self.CARD_SLOTS / moe)) if self.P > 1 and moe else 0
        if self.CARD_EXCLUSIVE:
            shared = self.CARD_SLICE
        t = self._TIER[k]
        expert_tier.tier(self.m, self.sfx("tier_card", k), self.T_CEXP, self.EXPERTS, self.TOPK, self.H, self.MINTER, self.CPLAN,
                         self.LAYERS, t["fed"], self.PARTS, self.EPLAN, self.GATE_ROW, self.DOWN_ROW, self.CARD_LANDING or 1,
                         t["exclusive"], self.CARD_LANDING_LINE, ranks=self.P, shared=shared,
                         stream=self.T_STREAM, stream_batch=expert_tier.STREAM_BATCH)

    def _slice_plane(self, name):
        """A part of the slice (▶ `_plan`'s SLICE) as a plane of its slot: a view the rows' words read and write."""
        at, size = self.SLICE[name]
        return "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_SLICE, at, size)

    def _from_speeds(self):
        """`card_shared` "auto" in a group and `card_landing` "auto": a CPU pick against a card expert and an expert's
        copy, timed here (▶ `expert_tier.speeds` — the first MoE layer's first experts in RAM on the first socket, the head's
        product on card 0 priced to a routed expert's bytes; the copy `_calibrate_line`'s, or `expert_tier.copy_seconds`
        where the line is fixed), the slice and a visit's landings from them, and every card's tier made again with them."""
        m, l, c = self.m, next(l for l in self.run_layers if self.moe[l]), self.CPUS[0]
        held = set(self.CARD_HELD.get(l, ())) if self.CARD_EXCLUSIVE else set()   # an exclusive tier's are not in RAM
        cpu_s, card_s = expert_tier.speeds(
            m, CARD, c["w"], "picks", [x for x in range(self.EXPERTS) if x not in held][:self.TOPK],
            "(ai_z__glm_5_3__experts %s picks ex%d_%d %s)" % (self.cn(0, "hand"), l, 0, self.cn(0, "routed")),
            self.CPLAN.slot_bytes / expert_tier.head_bytes(self.HPLAN))
        calibrated = (getattr(self, "CALIBRATION", None) or {}).get("copy_ms")
        copy_s = calibrated / 1e3 if calibrated else expert_tier.copy_seconds(m, CARD, self.HEAD_AT + self.HPLAN.at_up_data,
                                                                              min(self.CPLAN.slot_bytes, self.HPLAN.up_data))
        if self.CARD_SHARED is None:
            self.CARD_SHARED = expert_tier.slice_of(cpu_s, card_s)
        self.CARD_SPEEDS = dict(cpu_ms=cpu_s * 1e3, card_ms=card_s * 1e3, ratio=cpu_s / max(card_s, 1e-12), shared=self.CARD_SHARED)
        self.CARD_SPEEDS["copy_ms"] = copy_s * 1e3
        if self.CARD_LANDING is None:
            self.CARD_LANDING = expert_tier.landing_of(cpu_s, copy_s)
        self.CARD_SPEEDS["landing"] = self.CARD_LANDING
        for k in range(self.P):
            m.act_as(k)
            self._tier(k)
        m.act_as(CARD)
        expert_tier.bind(m, {c: self.sfx("tier_card", c) for c in range(self.P)}, len(self.workers()))

    def _fused_tables(self, k=0):
        """The fused layer's tables — card `k`'s in a group, by its own names (▶ `sfx`), over its parts' planes: its heads,
        its share of the MLPs' widths, its tier. The CPU's experts tables are card 0's to make."""
        m, b, H, s, n = self.m, self.b, self.H, "self_attn.", lambda name: self.sfx(name, k)
        pair = lambda l, n: [self.plane(l, n, "data"), self.plane(l, n, "lut")]
        data = lambda l, n: self.plane(l, n, "data")
        d = lambda l, n: b.record(l, n).d
        # ⛳ the MTP layer's sites are one plain residual (`MULT` 0 in its tables): its hyper-connection cells are never read
        hc = lambda l, site: (["signs"] * 3 if l == self.MTP else [data(l, "hc_%s_%s" % (site, x)) for x in ("fn", "base", "scale")])
        fed = {}                       # the tier's layers fed from the sockets' memory, by layer
        heads = {l: [H, 0 if l == self.MTP else self.MULT, self.ITERS] for l in self.run_layers}
        for l in self.run_layers:
            head = heads[l]
            if self.kinds[l] == "d":
                mats = [s + n + ".weight" for n in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj",
                                                    "f_b_proj", "g_b_proj", "o_proj")]
                m.table(n("at%d" % l), hc(l, "attn") + [data(l, "input_layernorm.weight")] + [c for x in mats for c in pair(l, x)]
                        + [data(l, s + "%s_conv1d.weight" % c) for c in "qkv"]
                        + [data(l, s + "A_log"), data(l, s + "dt_bias"), data(l, s + "o_norm.weight"),
                           self.state(l, "S"), self.state(l, "W"), n("signs"), n("gscratch")]
                        + head + [self.KHC, self.KD] + [d(l, x) for x in mats] + [self.EPS, self.HC_EPS, self.LOWER]
                        + self._masked(k, l, "attn"))
            else:
                mats = [s + n + ".weight" for n in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")]
                ix = [s + "indexer." + n for n in ("wq_b.weight", "wk.weight", "weights_proj.weight", "index_kpool_compress_gate")]
                m.table(n("at%d" % l), hc(l, "attn") + [data(l, "input_layernorm.weight")] + [c for x in mats for c in pair(l, x)]
                        + [data(l, s + "q_a_layernorm.weight"), data(l, s + "kv_a_layernorm.weight"), self.kv(l, "kvb"),
                           self.kv(l, "c"), n("scores"), n("signs"), n("gscratch")]
                        + head + [self.QHC, self.QR, self.LAT, self.NOPE, self.VD] + [d(l, x) for x in mats]
                        + [self.EPS, self.HC_EPS]
                        # the indexer — ▶ the package's AI_Z__GLM_5_3__IDX__ cells
                        + [c for x in ix for c in pair(l, x)]
                        + [data(l, s + "indexer.k_norm.weight"), data(l, s + "indexer.k_norm.bias"), n("ape%d" % l),
                           self.kv(l, "ikeys"), self.kv(l, "igates"), self.kv(l, "ipool"), n("iscores"), n("iindex"), n("igather")]
                        + [self.IH, self.ID, self.KP, self.ITOPK] + [d(l, x) for x in ix]
                        + self._masked(k, l, "attn"))
            if not self.moe[l]:
                mats = ["mlp.%s_proj.weight" % n for n in ("gate", "up", "down")]
                m.table(n("ml%d" % l), hc(l, "ffn") + [data(l, "post_attention_layernorm.weight")] + [c for x in mats for c in pair(l, x)]
                        + [n("signs"), n("gscratch")] + head + [self.IC] + [d(l, x) for x in mats]
                        + [self.EPS, self.HC_EPS, self.LIMIT])
                continue
            if self.CPLAN is not None:
                cp = self.CPLAN
                # the tier's layer, fed from the sockets' memory where every expert is there (▶ `nn__expert_tier`)
                if all((l, c["part"]) in self.SRC for c in self.CPUS):
                    if k == (1 if self.P > 1 else 0):   # every card's tier reads the same sources: the first card to table them makes them
                        self._int_array("src%d" % l, [self.SRC[(l, c["part"])][x] for x in range(self.EXPERTS) for c in self.CPUS])
                    fed[l] = "src%d" % l
                cells = [n("signs"), n("zero_h"), n("kscratch"), l, self.T_CEXP, cp.at_up_lut - cp.at_up_data,
                         cp.at_down_data - cp.at_up_data, cp.at_down_lut - cp.at_up_data, H, self.MINTER,
                         self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS, 0, self.LIMIT, 0] + self._masked(k, l, "moe")
                m.table(n("exk%d" % l), cells)
                # a prompt chunk's, its own scratch: card 0's tier takes a chunk's picks — and in a group of exclusive tiers
                #   every card's its own, each the experts only it holds
                if self.CH and (k == 0 or self.CARD_EXCLUSIVE):
                    m.table(n("exkr%d" % l), [n("signs"), n("zero_h"), n("krscratch")] + cells[3:])
                if self.T_SLICE is not None:                # a slice's: the experts' scratch over all its rows, the card's own
                    m.table(n("exks%d" % l), [n("signs"), n("zero_h"), self._slice_plane("scratch")] + cells[3:])
            mats = ["mlp.shared_experts.%s_proj.weight" % n for n in ("gate", "up", "down")]
            # ⛳ a draft's check runs a layer's MoE site for two positions at once: the second its own table over the same
            #   planes, its scratch its own (`gscratch2`), so the first's carry is not written over (▶ `_check_procedures`)
            for name, scratch in ((n("mo%d" % l), n("gscratch")),) + ((("mo%db" % l, "gscratch2"),) if self.MTP is not None and l != self.MTP else ()):
                m.table(name, hc(l, "ffn") + [data(l, "post_attention_layernorm.weight"), data(l, "mlp.gate.weight"),
                                              data(l, "mlp.gate.e_score_correction_bias")]
                        + [c for x in mats for c in pair(l, x)] + [n("signs"), scratch]
                        + head + [self.EXPERTS, self.TOPK, self.MC] + [d(l, x) for x in mats]
                        + [self.EPS, self.HC_EPS, self.RSCALE, self.LIMIT] + self._masked(k, l, "shared"))
        if self.CPLAN is not None:
            # ⭐ THE CARD'S TIER, ONE FOR THE CARD: how one of its experts is cut, where the sockets keep every expert's parts
            #   a layer at a time, and the binding `nn__expert_tier` the programs hand the verbs (▶ `expert_tier.bind`)
            exclusive = None
            if self.CARD_EXCLUSIVE:
                if set(fed) != {l for l in self.run_layers if self.moe[l]}:
                    raise Refused("an exclusive tier needs every MoE layer's sources")
                if k == (1 if self.P > 1 else 0):  # bound once, by the first card tabled: card 0's link measured (▶ `expert_tier.duplex`)
                    self.DUPLEX = expert_tier.duplex(m, CARD, self.CPLAN.slot_bytes, self.CARD_DUPLEX_LINE, self.CARD_DUPLEX)
                    m.act_as(k)
                exclusive = dict(part_worker=self.CPUS[0]["w"], part_type=0)
            # in a group every card a tier of its own, the picks shared out among them, and a slice of each layer's most picked
            #   every card keeps (NN-99 ③)
            self._TIER[k] = dict(fed=fed, exclusive=exclusive)
            self._tier(k)
            if k == 0:
                expert_tier.bind(m, {c: self.sfx("tier_card", c) for c in range(self.P)}, len(self.workers()))
        if self.EPLAN is not None and k == 0:
            p = self.EPLAN
            for i, cpu in enumerate(self.CPUS):
                m.act_as(cpu["w"])
                c = lambda name: self.cn(i, name)
                for l in self.run_layers:
                    if self.moe[l]:
                        m.table("ex%d_%d" % (l, i), [c("signs"), c("zero"), c("scratch"), l, 0, p.at_up_lut - p.at_up_data,
                                                     p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.PI,
                                                     self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS,
                                                     1 if self.EXPERTS_INT8 else 0, self.LIMIT] + self._experts_tail(cpu["w"], l))
            m.act_as(CARD)

    def _fused_body(self):
        """The fused token's pictures — what each CPU runs of a layer — and its body, a layer after another."""
        m, H = self.m, self.H
        if self.EPLAN is not None:
            for i, cpu in enumerate(self.CPUS):
                m.act_as(cpu["w"])
                c = lambda name: self.cn(i, name)
                for l in self.run_layers:
                    if self.moe[l]:
                        m.picture("xf%d_%d" % (l, i), "(begin (nn__buffer__copy hand %d %s %d) (ai_z__glm_5_3__experts %s picks ex%d_%d %s))"
                                  % (CARD, c("hand"), 2 * (H + self.TOPK), c("hand"), l, i, c("routed")))
            m.act_as(CARD)
        layers = []
        for l in self.run_layers:
            layers.append("(ai_z__glm_5_3__kda streams at%d)" % l if self.kinds[l] == "d" else "(ai_z__glm_5_3__mla streams at%d pos)" % l)
            if self.moe[l]:
                # every socket handed its share of the picks, the shared expert on the card meanwhile, the shares summed
                ws = [(i, c["w"]) for i, c in enumerate(self.CPUS)]
                layers.append("(ai_z__glm_5_3__route streams mo%d hand picks) " % l
                              + ("(ai_z__glm_5_3__experts hand picks exk%d routed_k %s) " % (l, expert_tier.BINDING) if self.CPLAN is not None else "")
                              + " ".join("(sys__compute %d names xf%d_%d)" % (w, l, i) for i, w in ws)
                              + " (ai_z__glm_5_3__shared hand mo%d) " % l
                              + " ".join("(sys__result %d)" % w for _, w in ws) + " "
                              # the CPUs done: an exclusive tier's writes back to RAM, held on half duplex, go now
                              + ("(nn__expert_tier__written %s) " % expert_tier.BINDING if self.CPLAN is not None else "")
                              + " ".join("(nn__buffer__copy %s %d %s %d)" % (self.cn(i, "routed"), w, "routed" if i == 0 else "routed%d" % i,
                                                                            2 * H) for i, w in ws) + " "
                              + " ".join("(nn__vector__add routed routed%d routed %d)" % (i, H) for i, _ in ws if i > 0)
                              + (" (nn__vector__add routed routed_k routed %d)" % H if self.CPLAN is not None else "")
                              + " (ai_z__glm_5_3__close streams mo%d routed)" % l)
            else:
                layers.append("(ai_z__glm_5_3__mlp streams ml%d)" % l)
        return " ".join(x for x in layers if x)

    # ── ③ the names the program is written against, and the program ──────────────────────────────────────────
    def _bindings(self):
        """What `lisp/glm5.lisp` reads: the shape, whether the card holds a tier, a layer's tables gathered into arrays
        indexed by its place in the layers this machine runs, and each socket's worker, buffers and experts tables."""
        m, H, run, K = self.m, self.H, self.run_layers, self.TOPK
        head = lambda at, size: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_HEAD, at, size)
        tier = self.CPLAN is not None
        m.bind_values(dict(HIDDEN=H, HIDDEN_BYTES=2 * H, MULT=self.MULT, MEAN=1.0 / self.MULT, VOCAB=self.VOCAB, EPS=self.EPS,
                           RUN=len(run) - (1 if self.MTP is not None else 0), EMB_HALVES=self.EMB_ROW_BYTES // 2, EMB_SCALES=self.EMB_DATA_BYTES // 2,
                           EMB_BITS=self.EMB_BITS, STOPS=0, SEEN=sampling.SEEN,
                           HEAD_CODES=head(self.HPLAN.at_up_data, self.HPLAN.up_data),
                           HEAD_SCALES=head(self.HPLAN.at_down_lut, self.HPLAN.down_lut), HEAD_BITS=self.HEAD_BITS,
                           HAND=self.HAND, HAND_PICKS_AT=self.HAND - 8 * K, HAND_BYTES=2 * (H + K),
                           ROWS_HANDS_BYTES=self.CH * self.HAND, ROWS_BYTES=2 * H * self.CH, ROWS_HALVES=H * self.CH,
                           CARD_TIER=tier, CARD=CARD, SOCKETS=len(self.CPUS), FIRST_SOCKET=self.P, CARDS=self.P, GROUP=self.GROUP,
                           VISIT_FIRST=self.VISIT_FIRST, STREAM=self.T_SLICE is not None, CH=self.CH,
                           EMBED_ON_CPU=bool(self.EMBED_CPU),
                           NOTE_STREAM=expert_tier.NOTE_STREAM))
        if self.T_SLICE is not None:
            m.bind_values(dict({name.upper() + "_S": self._slice_plane(name) for name in ("srows", "hands", "carries", "rrows")},
                               SROW_HALVES=self.MULT * H, HAND_HALVES=self.HAND // 2, CARRY_HALVES=(256 + 2 * H) // 2))
            for k in range(1, self.P):          # every other card's slice, by its own names, made on its own worker
                m.act_as(k)
                m.bind_values({self.sfx(name.upper() + "_S", k): self._slice_plane(name) for name in ("srows", "hands", "carries", "rrows")})
            m.act_as(CARD)
        if self.P > 1:
            m.node_array("TP_POS", 1)           # the position the group's other cards run (▶ `_group_procedures`)
            if self.CH:
                m.node_array("TP_ROWS", 2)      # a chunk's first position and its rows, for the same
            if self.T_SLICE is not None:
                m.node_array("TP_SLICE", 4)     # a slice's first position, its chunks, its last chunk's rows and a chunk's
        moe = lambda name: [name % l if self.moe[l] else 0 for l in run]
        m.array("KDA", [self.kinds[l] == "d" for l in run])
        m.array("MOE", [bool(self.moe[l]) for l in run])
        m.array("LAYER", list(run))
        m.array("AT", ["at%d" % l for l in run])
        m.array("MO", moe("mo%d"))
        m.array("ML", [0 if self.moe[l] else "ml%d" % l for l in run])
        if tier:
            m.array("EXK", moe("exk%d"))
            if self.CH:
                m.array("EXKR", moe("exkr%d"))
            if self.T_SLICE is not None:
                m.array("EXKS", moe("exks%d"))
        socket = range(len(self.CPUS))
        m.array("SOCKET", [c["w"] for c in self.CPUS])
        m.array("ROUTED_INTO", ["routed" if i == 0 else "routed%d" % i for i in socket])
        m.array("C_HAND", [self.cn(i, "hand") for i in socket])
        m.array("C_ROUTED", [self.cn(i, "routed") for i in socket])
        for i in socket:
            m.array("EX_%d" % i, [("ex%d_" + str(i)) % l if self.moe[l] else 0 for l in run])
        m.array("EX", ["EX_%d" % i for i in socket])
        if self.MTP is not None:                     # a draft's check: each socket's second hand and routed sum
            m.array("C_HAND2", [self.cn(i, "hand2") for i in socket])
            m.array("C_ROUTED2", [self.cn(i, "routed2") for i in socket])
        if self.CH:
            m.array("RROWS_INTO", ["rrows" if i == 0 else "rrows%d" % i for i in socket])
            m.array("C_HANDS", [self.cn(i, "hands") for i in socket])
            m.array("C_RROWS", [self.cn(i, "rrows") for i in socket])
            for i in socket:
                m.array("EXR_%d" % i, [("exr%d_" + str(i)) % l if self.moe[l] else 0 for l in run])
            m.array("EXR", ["EXR_%d" % i for i in socket])
        m.node_array("HERE", 1)             # the layer a socket's picture runs (▶ lisp/glm5_cpu.lisp)
        m.node_array("EMB_TOK", 1)          # the token the first socket decodes the row of (EMBED_ON_CPU)

    def _procedures(self):
        """The program: `lisp/glm5.lisp`, the sampler's and the sockets' pictures — or, unfused, `_composed_procedures`.
        Read from `programs.lisp` instead by the boots that have the same key (▶ `Machine.define_programs`)."""
        if not self.SOURCE:
            return self._composed_procedures()
        self.m.load_source(self.folder.program("glm5"))
        self.m.load_source(self.folder.program("sampling"))
        self.m.load_source(self.folder.program("generate"))
        if self.EPLAN is not None:
            self.m.worker_moved = {1: self.P}          # the file's worker 1 the first socket, after a group's cards
            self.m.load_source(self.folder.program("glm5_cpu"))
        if self.P > 1:
            self._group_procedures()
        if self.MTP is not None:
            self._mtp_procedures()

    def _group_layers(self, k):
        """Every layer at the position in TP_POS on card `k` past the first, by its own names: the same sites as card 0's,
        each output the group's sum; a MoE site its tier's share of the picks, or none of the routed sum (card 0 adds the
        sockets')."""
        n, pos, tier = (lambda name: self.sfx(name, k)), "(nth TP_POS 0)", self.CPLAN is not None
        out = []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                out.append("(ai_z__glm_5_3__kda %s %s GROUP %s)" % (n("streams"), n("at%d" % l), n("tsum")))
            else:
                out.append("(ai_z__glm_5_3__mla %s %s %s GROUP %s)" % (n("streams"), n("at%d" % l), pos, n("tsum")))
            if not self.moe[l]:
                out.append("(ai_z__glm_5_3__mlp %s %s GROUP %s)" % (n("streams"), n("ml%d" % l), n("tsum")))
                continue
            mine = ("(ai_z__glm_5_3__experts %s %s %s %s %s) " % (n("hand"), n("picks"), n("exk%d" % l), n("routed_k"),
                                                                 expert_tier.BINDING) if tier else "")
            # its close waits for the group's sum, card 0's after the sockets: the CPUs are done — a held promotion goes
            out.append("(ai_z__glm_5_3__route %s %s %s %s) %s(ai_z__glm_5_3__shared %s %s) "
                       "(ai_z__glm_5_3__close %s %s %s GROUP %s)%s"
                       % (n("streams"), n("mo%d" % l), n("hand"), n("picks"), mine, n("hand"), n("mo%d" % l),
                          n("streams"), n("mo%d" % l), n("routed_k") if tier else n("zero_h"), n("tsum"),
                          " (nn__expert_tier__written %s)" % expert_tier.BINDING if tier else ""))
        return " ".join(out)

    def _group_procedures(self):
        """A tensor-parallel group's program (NN-99 ④): card 0's sites redefined to take the group's sums — the rest of
        lisp/glm5.lisp as it is — and every other card's layers, a picture each it starts on them for a position: from card
        0's embedding (`tp_first`, its streams zeros, the group's sum of them the embedding on every card); and for a
        prompt's chunk the same as rows (`tp_rows`, the chunk's place and length in TP_ROWS)."""
        m, MH = self.m, self.MULT * self.H
        for k in range(1, self.P):
            m.act_as(k)
            n = lambda name: self.sfx(name, k)
            m.picture(n("tp_first"), "(begin (nn__partials__reduce GROUP %s %s %d) %s (type %s))"
                      % (n("zstreams"), n("streams"), MH, self._group_layers(k), n("streams")))
            if self.T_SLICE is not None:
                srows = self.sfx("SROWS_S", k)
                m.picture(n("tp_slice"), "(let ((first (nth TP_SLICE 0)) (k (nth TP_SLICE 1)) (last_n (nth TP_SLICE 2)) (w (nth TP_SLICE 3)) "
                          "(c 0) (n 0) (all 0)) (set! all (+ (* (- k 1) w) last_n)) (nn__vector__zero %s (* all SROW_HALVES)) "
                          "(nn__partials__reduce GROUP %s %s (* all SROW_HALVES)) %s (type %s))"
                          % (srows, srows, srows, self._group_slice(k), srows))
            if self.CH:
                m.picture(n("tp_rows"), "(begin (nn__partials__reduce GROUP %s %s (* (nth TP_ROWS 1) %d)) %s (type %s))"
                          % (n("zsrows"), n("srows"), MH, self._group_rows(k), n("srows")))
        m.act_as(CARD)
        m.view("names")
        m.defun("(defun (mixer i pos) (if (nth KDA i) (ai_z__glm_5_3__kda streams (nth AT i) GROUP tsum) "
                "(ai_z__glm_5_3__mla streams (nth AT i) pos GROUP tsum)))")
        m.defun("(defun (moe i) (let ((s 0)) (ai_z__glm_5_3__route streams (nth MO i) hand picks) "
                "(tier_and_sockets i) "
                "(ai_z__glm_5_3__shared hand (nth MO i)) "
                "(while '(< s SOCKETS) '(begin (sys__result (nth SOCKET s)) (set! s (+ s 1)))) "
                "(if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0) "
                "(sockets_sum) (if CARD_TIER (nn__vector__add routed routed_k routed HIDDEN) 0) "
                "(ai_z__glm_5_3__close streams (nth MO i) routed GROUP tsum)))")
        m.defun("(defun (layer i pos) (begin (mixer i pos) (if (nth MOE i) (moe i) (ai_z__glm_5_3__mlp streams (nth ML i) GROUP tsum)) "
                "(type streams)))")
        start = " ".join("(sys__compute %d names %s)" % (k, self.sfx("tp_first", k)) for k in range(1, self.P))
        wait = " ".join("(sys__result %d)" % k for k in range(1, self.P))
        m.defun("(defun (position tok pos) (begin (sys__node_array__set TP_POS 0 pos) %s (embed tok) "
                "(nn__partials__reduce GROUP streams streams %d) (layers_f pos) %s (type streams)))" % (start, MH, wait))
        if self.CH:
            # a chunk's rows: lisp/glm5.lisp's rows sites, each taking the group's sum of the chunk's rows (card 0's MoE
            #   close its sockets' and its tier's routed rows, the other cards' none)
            m.defun("(defun (rows_layer i first n) (begin (if (nth KDA i) (ai_z__glm_5_3__kda_rows srows (nth AT i) n rwork GROUP tsums) "
                    "(ai_z__glm_5_3__mla_rows srows (nth AT i) first n rwork GROUP tsums)) "
                    "(if (nth MOE i) (moe_rows i n) (ai_z__glm_5_3__mlp_rows srows (nth ML i) n rwork GROUP tsums)) (type srows)))")
            m.defun("(defun (moe_rows i n) (begin (ai_z__glm_5_3__route_rows srows (nth MO i) hands carries n rwork) "
                    "(if CARD_TIER (nn__expert_tier__note hands nn__expert_tier (nth LAYER i) n HAND HAND_PICKS_AT "
                    "nn__expert_major__vram_promotion_threshold_picks) 0) "
                    "(sockets_start i xr) "
                    "(if CARD_TIER (ai_z__glm_5_3__experts_rows hands (nth EXKR i) rrows_k n nn__expert_tier) 0) "
                    "(sockets_sum_rows) (if CARD_TIER (nn__vector__add rrows rrows_k rrows (* n HIDDEN)) 0) "
                    "(ai_z__glm_5_3__close_rows srows (nth MO i) rrows carries n rwork GROUP tsums)))")
            start = " ".join("(sys__compute %d names %s)" % (k, self.sfx("tp_rows", k)) for k in range(1, self.P))
            m.defun("(defun (rows first n) (let ((i 0)) (sys__node_array__set TP_ROWS 0 first) (sys__node_array__set TP_ROWS 1 n) "
                    "%s (nn__partials__reduce GROUP srows srows (* n %d)) (sys__node_array__set nrows 0 n) "
                    "(while '(< i RUN) '(begin (rows_layer i first n) (set! i (+ i 1)))) %s (type srows)))" % (start, MH, wait))
        if self.T_SLICE is not None:
            # ⭐ A SLICE READ LAYER-MAJOR BY THE GROUP (NN-115 ④): lisp/glm5.lisp's `slice_layer`, each site taking the group's
            #   sum of a chunk's rows, every card's tier noting the slice and streaming its share of the layer's experts into
            #   its own routed rows, which the close's group sum adds (▶ nn's `nn__expert_tier__chunk`)
            g = " GROUP tsums"
            part = lambda v, halves: "(part_rows %s c n %s w)" % (v, halves)
            m.defun("(defun (slice_layer i first k last_n w) (let ((c 0) (n w) (all (+ (* (- k 1) w) last_n))) "
                    "(while '(< c k) '(begin (set! n (if (eq? c (- k 1)) last_n w)) "
                    "(if (nth KDA i) (ai_z__glm_5_3__kda_rows %s (nth AT i) n rwork%s) "
                    "(ai_z__glm_5_3__mla_rows %s (nth AT i) (+ first (* c w)) n rwork%s)) "
                    "(if (nth MOE i) (ai_z__glm_5_3__route_rows %s (nth MO i) %s %s n rwork) "
                    "(ai_z__glm_5_3__mlp_rows %s (nth ML i) n rwork%s)) (set! c (+ c 1)))) "
                    "(if (nth MOE i) (begin (nn__expert_tier__note HANDS_S nn__expert_tier (nth LAYER i) all HAND HAND_PICKS_AT "
                    "(+ nn__expert_major__vram_promotion_threshold_picks NOTE_STREAM)) "
                    "(ai_z__glm_5_3__experts_rows HANDS_S (nth EXKS i) RROWS_S all nn__expert_tier) (set! c 0) "
                    "(while '(< c k) '(begin (set! n (if (eq? c (- k 1)) last_n w)) "
                    "(ai_z__glm_5_3__close_rows %s (nth MO i) %s %s n rwork%s) (set! c (+ c 1)))) (type SROWS_S)) "
                    "(type SROWS_S)) (type SROWS_S)))"
                    % (part("SROWS_S", "SROW_HALVES"), g, part("SROWS_S", "SROW_HALVES"), g, part("SROWS_S", "SROW_HALVES"),
                       part("HANDS_S", "HAND_HALVES"), part("CARRIES_S", "CARRY_HALVES"), part("SROWS_S", "SROW_HALVES"), g,
                       part("SROWS_S", "SROW_HALVES"), part("RROWS_S", "HIDDEN"), part("CARRIES_S", "CARRY_HALVES"), g))
            start = " ".join("(sys__compute %d names %s)" % (k, self.sfx("tp_slice", k)) for k in range(1, self.P))
            m.defun("(defun (rows_slice first k last_n w) (let ((i 0) (all (+ (* (- k 1) w) last_n))) "
                    "(sys__node_array__set TP_SLICE 0 first) (sys__node_array__set TP_SLICE 1 k) (sys__node_array__set TP_SLICE 2 last_n) "
                    "(sys__node_array__set TP_SLICE 3 w) %s (nn__partials__reduce GROUP SROWS_S SROWS_S (* all SROW_HALVES)) "
                    "(sys__node_array__set nrows 0 w) (while '(< i RUN) '(begin (slice_layer i first k last_n w) (set! i (+ i 1)))) "
                    "%s (type SROWS_S)))" % (start, wait))

    def _group_slice(self, k):
        """A slice through every layer on card `k` past the first, by its own names — its place, its chunks and their rows
        in TP_SLICE: the same sites as card 0's `slice_layer`, each output the group's sum; a MoE site its tier noting the
        slice — its share of the layer's experts streamed — its routed rows those experts', which the close's group sum
        adds to the other cards'."""
        n, S = (lambda name: self.sfx(name, k)), (lambda name: self.sfx(name.upper() + "_S", k))
        g = " GROUP %s" % n("tsums")
        part = lambda v, halves: "(part_rows %s c n %s w)" % (v, halves)
        chunks = "(set! c 0) (while '(< c k) '(begin (set! n (if (eq? c (- k 1)) last_n w)) %s (set! c (+ c 1))))"
        out = []
        for l in self.run_layers:
            rows = part(S("srows"), "SROW_HALVES")
            if self.kinds[l] == "d":
                mixer = "(ai_z__glm_5_3__kda_rows %s %s n %s%s)" % (rows, n("at%d" % l), n("rwork"), g)
            else:
                mixer = "(ai_z__glm_5_3__mla_rows %s %s (+ first (* c w)) n %s%s)" % (rows, n("at%d" % l), n("rwork"), g)
            if not self.moe[l]:
                out.append(chunks % (mixer + " (ai_z__glm_5_3__mlp_rows %s %s n %s%s)" % (rows, n("ml%d" % l), n("rwork"), g)))
                continue
            out.append(chunks % (mixer + " (ai_z__glm_5_3__route_rows %s %s %s %s n %s)"
                                 % (rows, n("mo%d" % l), part(S("hands"), "HAND_HALVES"), part(S("carries"), "CARRY_HALVES"), n("rwork"))))
            out.append("(nn__expert_tier__note %s %s %d all HAND HAND_PICKS_AT (+ nn__expert_major__vram_promotion_threshold_picks "
                       "NOTE_STREAM)) (ai_z__glm_5_3__experts_rows %s %s %s all %s)"
                       % (S("hands"), expert_tier.BINDING, l, S("hands"), n("exks%d" % l), S("rrows"), expert_tier.BINDING))
            out.append(chunks % ("(ai_z__glm_5_3__close_rows %s %s %s %s n %s%s)"
                                 % (rows, n("mo%d" % l), part(S("rrows"), "HIDDEN"), part(S("carries"), "CARRY_HALVES"), n("rwork"), g)))
        return " ".join(out)

    def _group_rows(self, k):
        """A chunk's rows through every layer on card `k` past the first, by its own names — its place and length in
        TP_ROWS: the same rows sites as card 0's, each output the group's sum; a MoE site its share of the shared expert and
        none of the routed rows (card 0 adds the sockets' and its tier's) — but with an exclusive tier, the routed rows of
        the experts only this card holds, its tier told of the chunk as card 0's is."""
        n, first, rows = (lambda name: self.sfx(name, k)), "(nth TP_ROWS 0)", "(nth TP_ROWS 1)"
        tp = " GROUP %s" % n("tsums")
        out = []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                out.append("(ai_z__glm_5_3__kda_rows %s %s %s %s%s)" % (n("srows"), n("at%d" % l), rows, n("rwork"), tp))
            else:
                out.append("(ai_z__glm_5_3__mla_rows %s %s %s %s %s%s)" % (n("srows"), n("at%d" % l), first, rows, n("rwork"), tp))
            if not self.moe[l]:
                out.append("(ai_z__glm_5_3__mlp_rows %s %s %s %s%s)" % (n("srows"), n("ml%d" % l), rows, n("rwork"), tp))
                continue
            routed = n("zrows")
            out.append("(ai_z__glm_5_3__route_rows %s %s %s %s %s %s)" % (n("srows"), n("mo%d" % l), n("hands"), n("carries"), rows, n("rwork")))
            if self.CPLAN is not None and self.CARD_EXCLUSIVE:
                routed = n("rrows_k")
                out.append("(nn__expert_tier__note %s %s %d %s HAND HAND_PICKS_AT nn__expert_major__vram_promotion_threshold_picks) "
                           "(ai_z__glm_5_3__experts_rows %s %s %s %s %s)"
                           % (n("hands"), expert_tier.BINDING, l, rows, n("hands"), n("exkr%d" % l), routed, rows, expert_tier.BINDING))
            # its close waits for the group's sum, card 0's after the sockets: the CPUs are done — a held write goes
            out.append("(ai_z__glm_5_3__close_rows %s %s %s %s %s %s%s)%s"
                       % (n("srows"), n("mo%d" % l), routed, n("carries"), rows, n("rwork"), tp,
                          " (nn__expert_tier__written %s)" % expert_tier.BINDING
                          if self.CPLAN is not None and self.CARD_EXCLUSIVE else ""))
        return " ".join(out)

    def _mtp_procedures(self):
        """`(mtp tok pos)` -> the MTP layer's greedy draft of the token after `tok`: at position `pos` of its own cache,
        `tok` the token at `pos + 1` and the model's hidden at `pos` the head's state the last `(logits)` left
        (`MTP_HIDDEN`). Its MoE is the layers' own sockets and tier, at the MTP layer's place in the arrays (RUN). ⛳ It
        leaves its logits in `logits_h`: the model's head is read first."""
        m, H, l, s = self.m, self.H, self.MTP, "self_attn."
        p = lambda n: self.plane(l, n, "data")
        m.defun("(defun (mtp_input tok hid) (begin (embed_e tok) (nn__rmsnorm__apply e %s %s %d %r) (nn__rmsnorm__apply hid %s %s %d %r) "
                "(nn__hadamard__rotate mcat signs mcatr %d) %s (type mres)))"
                % (p("enorm.weight"), self.rng("mcat", 0, H), H, self.EPS, p("hnorm.weight"), self.rng("mcat", H, H),
                   H, self.EPS, 2 * H, self.gemv(l, "eh_proj.weight", "mcatr", "mres")))
        moe = ("(ai_z__glm_5_3__route mres (nth MO i) hand picks) "
               "(if CARD_TIER (ai_z__glm_5_3__experts hand picks (nth EXK i) routed_k nn__expert_tier) 0) "
               "(sockets_start i xf) (ai_z__glm_5_3__shared hand (nth MO i)) "
               "(while '(< s SOCKETS) '(begin (sys__result (nth SOCKET s)) (set! s (+ s 1)))) "
               "(if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0) (sockets_sum) "
               "(if CARD_TIER (nn__vector__add routed routed_k routed HIDDEN) 0) (ai_z__glm_5_3__close mres (nth MO i) routed)")
        # the layer alone — what a prompt's positions give it, its cache's row written, no draft read
        m.defun("(defun (mtp_cache_from tok pos hid) (let ((i RUN) (s 0)) (mtp_input tok hid) (ai_z__glm_5_3__mla mres (nth AT i) pos) %s "
                "(type mres)))" % moe)
        m.defun("(defun (mtp_logits_from tok pos hid) (begin (mtp_cache_from tok pos hid) "
                "(nn__rmsnorm__apply mres %s mhn %d %r) (nn__hadamard__rotate mhn signs mhnr %d) "
                "(nn__turboquant__gemv HEAD_CODES HEAD_SCALES mhnr logits_h VOCAB HIDDEN HEAD_BITS) (type mhn)))"
                % (p("shared_head.norm.weight"), H, self.EPS, H))
        if self.CH:
            # a prompt row's head state — the streams' mean, the final norm — into hnB, as `(logits)` leaves it for a position
            MH, rr = self.MULT * H, lambda j: "(nn__vector__range srows (+ (* r %d) %d) %d)" % (self.MULT * H, j * H, H)
            m.defun("(defun (row_hidden r) (begin (nn__vector__add %s %s hm %d) %s (nn__vector__scale hm %r hm %d) "
                    "(nn__rmsnorm__apply hm fnw hn %d %r) (nn__vector__copy %s hnB %d) (type hnB)))"
                    % (rr(0), rr(1), H, " ".join("(nn__vector__add hm %s hm %d)" % (rr(j), H) for j in range(2, self.MULT)),
                       1.0 / self.MULT, H, H, self.EPS, self.MTP_HIDDEN, H))
        m.defun("(defun (mtp_logits tok pos) (mtp_logits_from tok pos %s))" % self.MTP_HIDDEN)
        m.defun("(defun (mtp_from tok pos hid) (begin (mtp_logits_from tok pos hid) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))")
        m.defun("(defun (mtp tok pos) (mtp_from tok pos %s))" % self.MTP_HIDDEN)
        self._check_procedures()

    def _check_procedures(self):
        """⭐ A DRAFT'S CHECK, TWO POSITIONS A LAYER — composed of the one-position verbs (⚖ 2026-10-19: *"a lisp rewrite to do
        two straight experts compute in a row rather than verbs"*; a verb of ai_z__glm_5_3 once it stands). `(check a b p q)`:
        the token `a` at `p` and the draft `b` at `q` = p + 1 through every layer — each layer's attention for `p` then
        `q` (a KDA layer's state and window kept between the two), both MoE routers, one hand-off to each socket for both
        positions' experts while the card runs both shared experts, both closes — then both heads: CHECK holds the
        greedy token after each, hnA and hnB the head's state at each (what the MTP layer reads). `(restore)` gives each
        KDA layer its state as it was after `p`. ⛳ An MLA layer needs nothing given back: a position reads only rows it
        has written, and `q`'s row is written again when `q` is run for real. MEASURED: ▶ test/src_glm5_mtp_generate.py"""
        m, H, M, CPU0 = self.m, self.H, self.MULT, self.CPUS[0]["w"]
        if self.CPLAN is not None:
            raise Refused("a draft's check runs without a card tier of experts, for now")
        # ⛔ NO BUFFER OR BINDING HERE — this composes programs, and a boot that reads them from programs.lisp never runs it:
        #   the second position's buffers are `_card_buffers`', its arrays `_bindings`'. MEASURED 2026-10-19: made here, a
        #   boot whose programs were read had no `routed2_1` and no C_HAND2, and its checks faulted OPCT at `xfb`
        for cpu in self.CPUS:
            m.act_as(cpu["w"])
            m.picture("xfb", "(let ((s (- (sys__worker) 1))) "
                      "(nn__buffer__copy hand %d (nth C_HAND s) HAND_BYTES) (nn__buffer__copy hand2 %d (nth C_HAND2 s) HAND_BYTES) "
                      "(ai_z__glm_5_3__experts (nth C_HAND s) picks (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED s)) "
                      "(ai_z__glm_5_3__experts (nth C_HAND2 s) picks2 (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED2 s)))" % (CARD, CARD))
        m.act_as(CARD)
        m.view("names")
        rng = self.rng
        m.defun("(defun (embed2 tok) (begin (embed_e tok) %s (type e)))"
                % " ".join("(nn__vector__add e zero_h %s %d)" % (rng("streams2", j * H, H), H) for j in range(M)))
        main = [l for l in self.run_layers if l != self.MTP]
        for i, l in enumerate(main):
            if self.kinds[l] == "d":
                mix = ("(ai_z__glm_5_3__kda streams at%d) (nn__vector__copy %s kbS%d %d) (nn__vector__copy %s kbW%d %d) "
                       "(ai_z__glm_5_3__kda streams2 at%d)" % (l, self.state(l, "S"), l, self.S_BYTES // 2, self.state(l, "W"), l,
                                                           self.W_BYTES // 2, l))
            else:
                mix = "(ai_z__glm_5_3__mla streams at%d p) (ai_z__glm_5_3__mla streams2 at%d q)" % (l, l)
            if self.moe[l]:
                sums = []
                for k, cpu in enumerate(self.CPUS):
                    into, into2 = ("routed", "routed2") if k == 0 else ("routed%d" % k, "routed2_%d" % k)
                    sums.append("(nn__buffer__copy %s %d %s %d) (nn__buffer__copy %s %d %s %d)"
                                % (self.cn(k, "routed"), cpu["w"], into, 2 * H, self.cn(k, "routed2"), cpu["w"], into2, 2 * H))
                    if k > 0:
                        sums.append("(nn__vector__add routed %s routed %d) (nn__vector__add routed2 %s routed2 %d)" % (into, H, into2, H))
                ffn = ("(ai_z__glm_5_3__route streams mo%d hand picks) (ai_z__glm_5_3__route streams2 mo%db hand2 picks2) "
                       "(sys__node_array__set HERE 0 %d) %s (ai_z__glm_5_3__shared hand mo%d) (ai_z__glm_5_3__shared hand2 mo%db) %s %s "
                       "(ai_z__glm_5_3__close streams mo%d routed) (ai_z__glm_5_3__close streams2 mo%db routed2)"
                       % (l, l, i, " ".join("(sys__compute %d names xfb)" % c["w"] for c in self.CPUS), l, l,
                          " ".join("(sys__result %d)" % c["w"] for c in self.CPUS), " ".join(sums), l, l))
            else:
                ffn = "(ai_z__glm_5_3__mlp streams ml%d) (ai_z__glm_5_3__mlp streams2 ml%d)" % (l, l)
            m.defun("(defun (ck%d p q) (begin %s %s (type streams)))" % (l, mix, ffn))

        def head(buf, cell, keep):
            mean = ("(nn__vector__add %s %s hm %d) " % (rng(buf, 0, H), rng(buf, H, H), H)
                    + " ".join("(nn__vector__add hm %s hm %d)" % (rng(buf, j * H, H), H) for j in range(2, M))
                    + " (nn__vector__scale hm %r hm %d)" % (1.0 / M, H))
            return (mean + " (nn__rmsnorm__apply hm fnw hn %d %r) (nn__hadamard__rotate hn signs hnr %d) "
                    "(nn__turboquant__gemv HEAD_CODES HEAD_SCALES hnr logits_h VOCAB HIDDEN HEAD_BITS) "
                    "(nn__argmax__find logits_h res VOCAB) (sys__node_array__set CHECK %d (nn__buffer__read res)) "
                    "(nn__vector__copy %s %s %d)" % (H, self.EPS, H, cell, self.MTP_HIDDEN, keep, H))
        m.defun("(defun (check_heads) (begin %s %s (type hnB)))" % (head("streams", 0, "hnA"), head("streams2", 1, "hnB")))
        m.defun("(defun (check a b p q) (begin (embed a) (embed2 b) %s (check_heads) (type streams)))"
                % " ".join("(ck%d p q)" % l for l in main))
        for l in main:
            if self.kinds[l] == "d":
                m.defun("(defun (rs%d) (begin (nn__vector__copy kbS%d %s %d) (nn__vector__copy kbW%d %s %d) (type streams)))"
                        % (l, l, self.state(l, "S"), self.S_BYTES // 2, l, self.state(l, "W"), self.W_BYTES // 2))
        m.defun("(defun (restore) (begin %s (type streams)))" % " ".join("(rs%d)" % l for l in main if self.kinds[l] == "d"))

    # ── a draft's check, from the host ──────────────────────────────────────────────────────────────────────────
    def mtp_prime(self, tokens):
        """A prompt a position at a time, the MTP layer given each position's history as it goes: answers the greedy token
        after the prompt and the MTP layer's draft of the one after that. ⛳ A position at a time: the rows' path does not
        leave each row's head state yet."""
        m, nxt = self.m, None
        for q, tok in enumerate(tokens):
            if q > 0:      # the MTP layer at q-1: the token at q, the model's state at q-1 (kept in hnB)
                self._run("(begin (mtp_logits_from %d %d hnB) (type mhn))" % (tok, q - 1), "the MTP layer at %d" % (q - 1))
            nxt = self._run("(begin (position %d %d) (head))" % (tok, q), "position %d" % q)
            self._run("(begin (nn__vector__copy %s hnB %d) (type hnB))" % (self.MTP_HIDDEN, self.H), "the state kept")
        P = len(tokens)
        draft = self._run("(mtp_from %d %d hnB)" % (nxt, P - 1), "the first draft")
        self._mtp_have, self._mtp_hid, self._mtp_next = P - 1, P - 1, (P, nxt, draft)
        return nxt, draft

    def generate_mtp(self, token, pos, draft, n, stop=()):
        """`n` greedy tokens from `token` at `pos` with the MTP layer's `draft` of the one after: each pass checks the
        token and the draft together (`check`) — the draft kept when the model's own greedy token after `token` is the
        draft, the token after the draft then the next; refused, the KDA states given back. Answers (tokens, passes)."""
        m, out, passes = self.m, [], 0
        # ⛳ WHAT `generate` PROMISES (▶ session.py): every token it answers but the last has been run — so a pass that would
        #   run more than it answers gives the KDA states back to after `token`: a kept draft that is an end token, or
        #   one with no room left for the token after it
        while len(out) < n:
            self._run("(begin (check %d %d %d %d) (type streams))" % (token, draft, pos, pos + 1), "the check at %d" % pos)
            passes += 1
            y1 = m.must("(sys__node_array__get CHECK 0)", "the model's token after the token")
            y2 = m.must("(sys__node_array__get CHECK 1)", "the model's token after the draft")
            if y1 == draft and (draft in stop or len(out) + 1 >= n):
                out.append(draft)
                self._run("(restore)", "the KDA states given back")
                self._run("(begin (nn__vector__copy hnA hnB %d) (type hnB))" % self.H, "the state at %d" % pos)
                self._mtp_have, self._mtp_hid, self._mtp_next = pos - 1, pos, None
                return out, passes
            if y1 == draft:
                out += [draft, y2]
                # the MTP layer at pos (the draft, the state at pos) and at pos + 1 (y2, the state at pos + 1): the next draft
                self._run("(begin (mtp_cache_from %d %d hnA) (type mres))" % (draft, pos), "the MTP layer at %d" % pos)
                draft = self._run("(mtp_from %d %d hnB)" % (y2, pos + 1), "the draft at %d" % (pos + 1))
                token, pos, self._mtp_have, self._mtp_hid = y2, pos + 2, pos + 1, pos + 1
            else:
                out.append(y1)
                self._run("(restore)", "the KDA states given back")
                self._run("(begin (nn__vector__copy hnA hnB %d) (type hnB))" % self.H, "the state at %d" % pos)
                draft = self._run("(mtp_from %d %d hnA)" % (y1, pos), "the draft at %d" % pos)
                token, pos, self._mtp_have, self._mtp_hid = y1, pos + 1, pos, pos
            if out[-1] in stop:
                break
        self._mtp_next = (pos, token, draft)
        return out[:n], passes

    def _composed_procedures(self):
        """The unfused layer's procedures, nn's words a layer — and the fused layer's for a run that takes directions out
        or captures the streams — composed here."""
        m, H = self.m, self.H
        # the token's row of the embedding table, un-rotated (Rᵀu = S ⊙ H·u), into e — and into every stream
        if self.EMBED_CPU:                          # the first socket decodes the row (▶ lisp/glm5_cpu.lisp `xe`)
            m.act_as(self.CPUS[0]["w"])
            m.picture("xe", "(let ((tok (nth EMB_TOK 0))) (nn__turboquant__decode (nn__vector__range c_embedding (* tok %d) %d) "
                      "(nn__vector__range c_embedding (+ %d tok) 1) c_emb 1 %d %d))"
                      % (self.EMB_ROW_BYTES // 2, self.EMB_ROW_BYTES // 2, self.EMB_DATA_BYTES // 2, H, self.EMB_BITS))
            m.act_as(CARD)
            m.defun("(defun (embed_e tok) (begin (sys__node_array__set EMB_TOK 0 tok) (sys__compute %d names xe) "
                    "(sys__result %d) (nn__buffer__copy c_emb %d u %d) (nn__hadamard__rotate u ones512 hu %d) "
                    "(nn__vector__pointwise_mul hu signs_row e %d)))"
                    % (self.CPUS[0]["w"], self.CPUS[0]["w"], self.CPUS[0]["w"], 2 * H, H, H))
        else:
            m.defun("(defun (embed_e tok) (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok %d) %d) "
                    "(nn__vector__range EMBEDDING (+ %d tok) 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                    "(nn__vector__pointwise_mul hu signs_row e %d)))"
                    % (self.EMB_ROW_BYTES // 2, self.EMB_ROW_BYTES // 2, self.EMB_DATA_BYTES // 2, H, self.EMB_BITS, H, H))
        m.defun("(defun (embed tok) (begin (embed_e tok) %s (type e)))"
                % " ".join("(nn__vector__add e zero_h %s %d)" % (self.rng("streams", j * H, H), H) for j in range(self.MULT)))
        # the head: the streams' mean, the final norm, the logits
        mean = ("(nn__vector__add %s %s hm %d) " % (self.rng("streams", 0, H), self.rng("streams", H, H), H)
                + " ".join("(nn__vector__add hm %s hm %d)" % (self.rng("streams", j * H, H), H) for j in range(2, self.MULT))
                + " (nn__vector__scale hm %r hm %d)" % (1.0 / self.MULT, H))
        logits = (mean + " (nn__rmsnorm__apply hm fnw hn %d %r) (nn__hadamard__rotate hn signs hnr %d) " % (H, self.EPS, H)
                  + "(nn__turboquant__gemv (nn__expert__plane 0 %d 0 %d %d) (nn__expert__plane 0 %d 0 %d %d) hnr logits_h %d %d %d)"
                  % (self.T_HEAD, self.HPLAN.at_up_data, self.HPLAN.up_data, self.T_HEAD, self.HPLAN.at_down_lut,
                     self.HPLAN.down_lut, self.VOCAB, H, self.HEAD_BITS))
        m.defun("(defun (head) (begin %s (nn__argmax__find logits_h res %d) (nn__buffer__read res)))" % (logits, self.VOCAB))
        m.defun("(defun (head_logits) (begin %s (type hm)))" % logits)
        m.defun("(defun (logits) (begin %s))" % logits)      # what `head_sample` reads (▶ lisp/sampling.lisp)
        m.load_source(self.folder.program("sampling"))
        m.load_source(self.folder.program("generate"))      # `generate` over the `position` below
        for l in self.run_layers:
            mixer = self._kda(l) if self.kinds[l] == "d" else self._mla(l)
            body = (self._hc(l, "attn") + " (nn__rmsnorm__apply xin %s h %d %r) (nn__hadamard__rotate h signs hr %d) "
                    % (self.plane(l, "input_layernorm.weight", "data"), H, self.EPS, H)
                    + mixer + " " + self._post() + " " + self._hc(l, "ffn")
                    + " (nn__rmsnorm__apply xin %s h %d %r) (nn__hadamard__rotate h signs hr %d) "
                    % (self.plane(l, "post_attention_layernorm.weight", "data"), H, self.EPS, H))
            if not self.moe[l]:
                body += self._swiglu(l, "mlp.", "gu", "act", "actr", "hr", "y", self.INTER) + " " + self._post()
            else:
                body += (self.gemv(l, "mlp.gate.weight", "h", "router")
                         + " (nn__vector__top_k_biased router %d %d picks weights %s %r) "
                         % (self.EXPERTS, self.TOPK, self.plane(l, "mlp.gate.e_score_correction_bias", "data"), self.RSCALE))
            m.defun("(defun (a%d at len) (begin %s (type streams)))" % (l, body))
            if self.moe[l]:
                # the shared expert, which the card runs while the CPU runs the routed ones
                m.defun("(defun (s%d) (begin %s (type sh)))"
                        % (l, self._swiglu(l, "mlp.shared_experts.", "sgu", "sact", "sactr", "hr", "sh", self.MINTER)))
                m.defun("(defun (c%d) (begin (nn__buffer__copy c_routed %d routed %d) (nn__vector__add sh routed y %d) %s "
                        "(type streams)))" % (l, CPU, H * 2, H, self._post()))
        if self.EPLAN is not None:
            m.act_as(CPU)
            p, I = self.EPLAN, self.MINTER
            for l in self.run_layers:
                if not self.moe[l]:
                    continue
                pl = lambda at, size: "(nn__expert__plane %d 0 x %d %d)" % (l, at, size)
                experts = " ".join(
                    "(let ((x (sys__node_array__get picks %d))) "
                    "(nn__turboquant__gemv %s %s c_hr c_gu %d %d %d) (nn__swiglu__clamped c_gu c_act %d %r) "
                    "(nn__hadamard__rotate c_act c_signs c_actr %d) (nn__turboquant__gemv %s %s c_actr c_e %d %d %d) "
                    "(nn__vector__scale_at c_e c_w %d c_e %d) (nn__vector__add c_routed c_e c_routed %d))"
                    % (j, pl(p.at_up_data, p.up_data), pl(p.at_up_lut, p.up_lut), 2 * I, H, self.E_D, I, self.LIMIT, I,
                       pl(p.at_down_data, p.down_data), pl(p.at_down_lut, p.down_lut), H, I, self.E_D, j, H, H)
                    for j in range(self.TOPK))
                m.defun("(defun (x%d) (begin (nn__buffer__copy hr %d c_hr %d) (nn__buffer__copy weights %d c_w %d) "
                        "(nn__vector__zero c_routed %d) %s (type c_routed)))" % (l, CARD, H * 2, CARD, 2 * self.TOPK, H, experts))
                m.picture("xw%d" % l, "(x%d)" % l)
            m.act_as(CARD)
        # ⭐ A TOKEN AS ONE PROGRAM: the card runs every layer, and hands each MoE layer's experts to the CPU's block —
        #   `sys__compute` with that layer's picture over a snapshot of the names, `sys__result` to collect it
        fused = self._fused_body() if self.FUSED else None
        fused_tok = fused
        if self.CH and self.EPLAN is not None:
            self._rows_procedures()
        if self.EPLAN is not None:
            m.view("names")
        # ⭐ A POSITION, AND THEN ITS HEAD — the step runs `(begin (position tok pos) <head>)`, the head greedy, drawn on
        #   the card or the logits left (▶ `step`)
        if fused is not None:
            m.defun("(defun (position tok pos) (begin (embed tok) %s (type streams)))" % fused_tok)
            # a pipeline stage's own layers, the streams in and out (▶ `forward`)
            m.defun("(defun (layers_f pos) (begin %s (type streams)))" % fused)
        if fused is None:
            # the layers as nn's words, a layer's latent row `at` and the positions it attends `len`
            layers = " ".join(("(a%d at len) (sys__compute %d names xw%d) (s%d) (sys__result %d) (c%d)" % (l, CPU, l, l, CPU, l))
                              if self.moe[l] else "(a%d at len)" % l for l in self.run_layers)
            m.defun("(defun (position tok pos) (let ((at (* pos %d)) (len (+ pos 1))) (embed tok) %s (type streams)))"
                    % (self.LAT * 2, layers))

    def _rows_tables(self):
        """Each CPU's experts table over a prompt's chunk."""
        m, H, p = self.m, self.H, self.EPLAN
        for i, cpu in enumerate(self.CPUS):
            m.act_as(cpu["w"])
            c = lambda name: self.cn(i, name)
            for l in self.run_layers:
                if self.moe[l]:
                    m.table("exr%d_%d" % (l, i), [c("signs"), c("zero"), c("rscratch"), l, 0, p.at_up_lut - p.at_up_data,
                                                  p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.PI,
                                                  self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS,
                                                  1 if self.EXPERTS_INT8 else 0, self.LIMIT] + self._experts_tail(cpu["w"], l))
        m.act_as(CARD)

    def _rows_procedures(self):
        """A chunk of a prompt as rows: each CPU's picture over the chunk, and a procedure a layer."""
        m, H = self.m, self.H
        for i, cpu in enumerate(self.CPUS):
            m.act_as(cpu["w"])
            c = lambda name: self.cn(i, name)
            for l in self.run_layers:
                if self.moe[l]:
                    m.picture("xr%d_%d" % (l, i), "(begin (nn__buffer__copy hands %d %s %d) "
                              "(ai_z__glm_5_3__experts_rows %s exr%d_%d %s (sys__node_array__get nrows 0)))"
                              % (CARD, c("hands"), self.CH * self.HAND, c("hands"), l, i, c("rrows")))
        m.act_as(CARD)
        ws = [(i, cpu["w"]) for i, cpu in enumerate(self.CPUS)]
        for l in self.run_layers:
            body = ("(ai_z__glm_5_3__kda_rows srows at%d n rwork)" % l if self.kinds[l] == "d"
                    else "(ai_z__glm_5_3__mla_rows srows at%d first n rwork)" % l)
            if self.moe[l]:
                body += (" (ai_z__glm_5_3__route_rows srows mo%d hands carries n rwork) " % l
                         # the card's tier told of the chunk's picks, so the prompt warms it (NN-48)
                         + ("(nn__expert_tier__note hands %s %d n %d %d %s) " % (expert_tier.BINDING, l, self.HAND, self.HAND - 8 * self.TOPK,
                                                                       expert_tier.THRESHOLD)
                            if self.CPLAN is not None else "")
                         + " ".join("(sys__compute %d names xr%d_%d)" % (w, l, i) for i, w in ws) + " "
                         # the card's share computed while the CPUs compute theirs
                         + ("(ai_z__glm_5_3__experts_rows hands exkr%d rrows_k n %s) " % (l, expert_tier.BINDING) if self.CPLAN is not None else "")
                         # ⛳ WAITED FOR WITHOUT A BOUND: `sys__result` gives up after ~15 s, and a chunk's experts read
                         #   from the disk can take longer — so the card asks `sys__completed` until it says so, then
                         #   collects. `(< 1 0)` is #f: the loop runs while completed is #f
                         + " ".join("(while '(sys__eq (sys__completed %d) (< 1 0)) '(< 0 1)) (sys__result %d)" % (w, w)
                                    for _, w in ws) + " "
                         + ("(nn__expert_tier__written %s) " % expert_tier.BINDING if self.CPLAN is not None else "")
                         + " ".join("(nn__buffer__copy %s %d %s %d)" % (self.cn(i, "rrows"), w, "rrows" if i == 0 else "rrows%d" % i,
                                                                       2 * H * self.CH) for i, w in ws) + " "
                         + " ".join("(nn__vector__add rrows rrows%d rrows %d)" % (i, H * self.CH) for i, _ in ws if i > 0)
                         + (" (nn__vector__add rrows rrows_k rrows %d)" % (H * self.CH) if self.CPLAN is not None else "")
                         + " (ai_z__glm_5_3__close_rows srows mo%d rrows carries n rwork)" % l)
            else:
                body += " (ai_z__glm_5_3__mlp_rows srows ml%d n rwork)" % l
            m.defun("(defun (r%d first n) (begin %s (type srows)))" % (l, body))
        # a token's row, un-rotated, into every stream of a row of the chunk, its streams at `at0 ..`
        m.defun("(defun (embed_row tok %s) (begin (embed_e tok) %s (type e)))"
                % (" ".join("at%d" % j for j in range(self.MULT)),
                   " ".join("(nn__vector__add e zero_h (nn__vector__range srows at%d %d) %d)" % (j, H, H)
                            for j in range(self.MULT))))

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: every KDA layer's state and window to zero. The latent caches need nothing — a
        position reads only rows it has written."""
        zeros = []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                zeros += ["(nn__vector__zero %s %d)" % (self.state(l, "S"), self.S_BYTES // 2),
                          "(nn__vector__zero %s %d)" % (self.state(l, "W"), self.W_BYTES // 2)]
        for k in reversed(range(self.P)):
            self.m.act_as(k)
            self.m.must("(begin %s (type %s))" % (" ".join(zeros), self.sfx("streams", k)), "card %d's reset" % k)
        self._room(0, exact=True)                       # a conversation from nothing: every lent slot the tier's again
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None

    def layer(self, l, pos):
        """One layer at one position, the streams in and out on the card."""
        m = self.m
        m.must("(a%d %d %d)" % (l, pos * self.LAT * 2, pos + 1), "layer %d at %d" % (l, pos))
        if self.moe[l]:
            m.must("(s%d)" % l, "layer %d's shared expert at %d" % (l, pos))
            m.act_as(CPU)
            m.must("(x%d)" % l, "layer %d's experts at %d" % (l, pos))
            m.act_as(CARD)
            m.must("(c%d)" % l, "layer %d's close at %d" % (l, pos))

    def step(self, token, pos, greedy=True, sample=None):
        """One position: the token in, the next token out — greedy, or drawn on the card by `sample` (a `sampling.head`)
        — or the logits left for a sampler on the host (answers None)."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._room(pos + 1)
        head = sample or ("(head)" if greedy else "(head_logits)")
        if self.FUSED or self.GRID:
            value = self._run("(begin (position %d %d) %s)" % (token, pos, head), "position %d" % pos)
            return value if greedy or sample else None
        m = self.m
        self._embedding("(embed %d)" % token, "the embedding at %d" % pos)
        for l in self.run_layers:
            self.layer(l, pos)
        got = m.must(head, "the head at %d" % pos)
        return got if greedy or sample else None

    def generate(self, token, pos, n, stop=(), sampler=None, lag=0):
        """`Generates.generate`, the cache's rows it may write withheld from the tier first — or, with the MTP layer and a
        greedy answer, two tokens a pass where its draft is kept (`generate_mtp`). ⛳ The draft goes on from where the last
        prompt or answer left it; anywhere else (another conversation ran between) the answer is the plain one."""
        self._room(pos + min(n, GENERATE_MOST) + 2)
        n = min(n, GENERATE_MOST, self.max_context - pos - 1)
        if self.MTP is not None and sampler is None and not lag and n > 0:
            draft = None
            if self._mtp_next is not None and self._mtp_next[:2] == (pos, token):
                draft = self._mtp_next[2]
            elif self._mtp_hid == pos - 1:          # a gap in the MTP layer's history only makes its drafts worse
                draft = self._run("(mtp_from %d %d hnB)" % (token, pos - 1), "the draft at %d" % (pos - 1))
            if draft is not None:
                self._watch(())
                out, passes = self.generate_mtp(token, pos, draft, n, set(stop))
                self.mtp_stats["passes"] += passes
                self.mtp_stats["kept"] += len(out) - passes
                return out
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None   # the plain path runs on without the MTP layer
        return Generates.generate(self, token, pos, n, stop, sampler, lag)

    def _run(self, text, what):
        """A program's answer, run with every block standing: it hands the CPU sockets their experts (`sys__compute`), and
        a program that does not would find their blocks never started."""
        return self.m.grid(text)[1]

    def _prompt(self, tokens):
        """A prompt chunk's tokens into PROMPT, which `embed_rows` reads — sixty-four to a program, as a program is one
        form and a form must fit one of the heap's chunks."""
        for at in range(0, len(tokens), 64):
            self.m.must("(begin %s 0)" % " ".join("(sys__node_array__set PROMPT %d %d)" % (i, t)
                                                  for i, t in enumerate(tokens[at:at + 64], at)), "the prompt's tokens")

    # ── pictures: the vision tower on the card ─────────────────────────────────────────────────────────────────
    def _vision_plan(self):
        """Where the tower's planes go, and its scratch's: a collection each, after the model's. Every record of global.pkl's
        `model.visual.*` is a plane as the pack keeps it — a matrix's codes and its scales, a vector's halves — but the two
        convolutions, whose rows of 14 and of 2 no product takes: the patch embedding becomes a matrix of halves (1024
        rows of 1176), the downsampling one of 4096 rows of 4096, its columns laid out for the four patches as they lie
        (a patch's 1024 channels, then the next patch's). A block's gate and up biases are one plane, side by side."""
        vc = json.load(open(os.path.join(self.folder.path, "config.json")))["vision_config"]
        self.VC = vc
        g, self.VREC, self.D_VIS = self.global_, {}, {}
        H, I, out, P = vc["hidden_size"], vc["intermediate_size"], vc["out_hidden_size"], vc["patch_size"]
        M = vc["spatial_merge_size"] ** 2
        self.VGROUPS = ["embed"] + list(range(vc["depth"])) + ["tail"]
        group_of = lambda short: ("embed" if short.startswith("patch_embed.")
                                  else int(short.split(".")[1]) if short.startswith("blocks.") else "tail")
        size, at = {k: 0 for k in self.VGROUPS}, 0

        def place(group, name, nbytes, d=16):
            self.VREC[name] = (group, size[group], nbytes)
            self.D_VIS[name] = d
            size[group] += (nbytes + 255) // 256 * 256
        for name in sorted(k for k in g if k.startswith("model.visual.")):
            short, rec = name[len("model.visual."):], g[name]
            if short in ("patch_embed.proj.weight", "downsample.weight"):
                shape = [int(x) for x in rec["original_shape"]]
                place(group_of(short), short, 2 * int(np.prod(shape)))
                continue
            if short.endswith(("mlp.gate_proj.bias", "mlp.up_proj.bias")):
                continue                                       # ▶ `mlp.gu_bias` below
            d = int(rec["d"])
            place(group_of(short), short, len(np.asarray(rec["vbr_data"]).tobytes()), d)
            if d != 16:
                place(group_of(short), short + ":lut", 2 * int(rec["rows"]))
        for i in range(vc["depth"]):
            place(i, "blocks.%d.mlp.gu_bias" % i, 2 * 2 * I)
        self.VCARD = {}
        for k in self.VGROUPS:
            self.VCARD[k] = at
            at += size[k]
        self.VIS_BYTES = at
        self.VPLAN = NL.SlotPlan(at, 0, 0, 256)               # every plane in the up half (nn takes no empty half)
        self.VIS_ROWS = max(M, self.PICTURE_PIXELS // (P * P) // M * M)     # the most patches a picture is made into
        self.VIS_CHUNK = 1024                                                # the MLPs' rows a step
        n, hd, PI = self.VIS_ROWS, H // vc["num_heads"], vc["projection_intermediate_size"]
        IN, tokens = 3 * vc["temporal_patch_size"] * P * P, n // M
        pairs = lambda rows: (24 * rows + 255) & ~255
        step, step2 = min(self.VIS_CHUNK, n), min(self.VIS_CHUNK, tokens)
        # ▶ ai_z__glm_5_3's `vision_block` and `vision_tail`: their scratch, one after the other in the same room
        block = pairs(n) + 2 * (n * H + max(n * H, step * I) + 3 * n * H + 2 * step * I + 2 * step * I + step * H) + 4 * n * hd
        big = max(n * H, tokens * out)
        tail = pairs(tokens) + 2 * (big + tokens * out + max(tokens * out, step2 * PI) + 2 * step2 * PI + 2 * step2 * PI)
        self.VSCR, at = {}, 0
        for name, nbytes in (("x", 2 * n * H), ("pix", 2 * n * IN), ("rc", 8 * n), ("out", 2 * tokens * out),
                             ("scratch", max(block, tail, pairs(n)))):
            self.VSCR[name] = (at, nbytes)
            at += (nbytes + 255) // 256 * 256
        self.VSPLAN = NL.SlotPlan(at, 0, 0, 256)
        self.T_VIS = max(t for t, _, _ in self.collections) + 1
        self.T_VSCR = self.T_VIS + 1
        self.collections += [(self.T_VIS, self.VPLAN, 1), (self.T_VSCR, self.VSPLAN, 1)]
        self.arena += self.VPLAN.slot_bytes + self.VSPLAN.slot_bytes

    def _vision_load(self, loader):
        """The tower's planes into its slot, each record let go of once it is there; the scratch's slot reserved."""
        from .vision import decoded
        g, vc = self.global_, self.VC
        loader.layer_width(0, self.T_VIS, 1)
        self.VIS_AT = loader.reserve(self.T_VIS)

        def put(name, data):
            group, at, _ = self.VREC[name]
            loader.write(self.VIS_AT + self.VCARD[group] + at, data)
        bias = {}
        for name in sorted(k for k in g if k.startswith("model.visual.")):
            short, rec = name[len("model.visual."):], g.pop(name)
            if short == "patch_embed.proj.weight":
                put(short, decoded(rec).reshape(int(rec["original_shape"][0]), -1).astype("<f2").tobytes())
            elif short == "downsample.weight":
                w = decoded(rec)                                # out, channels, the block's row, its column
                put(short, w.reshape(w.shape[0], w.shape[1], -1).transpose(0, 2, 1).reshape(w.shape[0], -1).astype("<f2").tobytes())
            elif short.endswith(("mlp.gate_proj.bias", "mlp.up_proj.bias")):
                bias[short] = np.asarray(rec["vbr_data"]).tobytes()
            else:
                put(short, np.asarray(rec["vbr_data"]).tobytes())
                if int(rec["d"]) != 16:
                    put(short + ":lut", np.asarray(rec["vbr_lut"]).astype("<f2").tobytes())
        for i in range(vc["depth"]):
            p = "blocks.%d.mlp." % i
            put(p + "gu_bias", bias[p + "gate_proj.bias"] + bias[p + "up_proj.bias"])
        loader.assign(0, self.T_VIS, 0, self.VIS_AT)
        loader.layer_width(0, self.T_VSCR, 1)
        self.VSCR_AT = loader.reserve(self.T_VSCR)
        loader.assign(0, self.T_VSCR, 0, self.VSCR_AT)

    def _vision_tables(self):
        """The tower's tables — the embedding's, a block's each, the tail's — and the scratch's planes."""
        m, vc = self.m, self.VC
        H, heads = vc["hidden_size"], vc["num_heads"]
        eps, limit = float(vc.get("rms_norm_eps", 1e-5)), float(vc.get("swiglu_limit", 10.0))

        def w(name):
            group, at, nbytes = self.VREC[name]
            return "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_VIS, self.VCARD[group] + at, nbytes)
        sp = lambda name: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_VSCR,) + self.VSCR[name])
        d = lambda name: int(self.D_VIS[name])
        # a matrix's codes and scales; at 16 bits the scales' cell is not read, and the codes stand in
        pair = lambda n: [w(n), w(n + ":lut") if d(n) != 16 else w(n)]
        IN = 3 * vc["temporal_patch_size"] * vc["patch_size"] ** 2
        m.table("GVE", pair("patch_embed.proj.weight") + [w("patch_embed.proj.bias"), sp("scratch"), H, IN, 16])
        for i in range(vc["depth"]):
            p = "blocks.%d." % i
            m.table("GVB%d" % i, [w(p + "norm1.weight")] + pair(p + "attn.qkv.weight") + [w(p + "attn.qkv.bias"),
                                  w(p + "attn.q_norm.weight"), w(p + "attn.k_norm.weight")]
                    + pair(p + "attn.proj.weight") + [w(p + "attn.proj.bias"), w(p + "norm2.weight")]
                    + pair(p + "mlp.gate_proj.weight") + pair(p + "mlp.up_proj.weight") + [w(p + "mlp.gu_bias")]
                    + pair(p + "mlp.down_proj.weight") + [w(p + "mlp.down_proj.bias"), "signs", sp("scratch"),
                       H, heads, H // heads, vc["intermediate_size"], d(p + "attn.qkv.weight"), d(p + "attn.proj.weight"),
                       d(p + "mlp.gate_proj.weight"), d(p + "mlp.down_proj.weight"),
                       int((vc.get("rope_parameters") or {}).get("rope_theta", 10000)), self.VIS_CHUNK, eps, limit])
        m.table("GVT", [w("post_layernorm.weight")] + pair("downsample.weight") + [w("downsample.bias")]
                + pair("merger.proj.weight") + [w("merger.post_projection_norm.weight"), w("merger.post_projection_norm.bias")]
                + pair("merger.gate_proj.weight") + pair("merger.up_proj.weight") + pair("merger.down_proj.weight")
                + ["signs", sp("scratch"), H, vc["spatial_merge_size"] ** 2, vc["out_hidden_size"],
                   vc["projection_intermediate_size"], 16, d("merger.proj.weight"), d("merger.gate_proj.weight"),
                   d("merger.down_proj.weight"), self.VIS_CHUNK, eps, 1e-5, limit])
        self.VPLANES = {k: sp(k) for k in ("x", "pix", "rc", "out")}

    def vision_rows(self, pixel_values, rowcol, blocks=None):
        """A picture's patches through the tower on the card: `pixel_values` (patches x 1176) and each patch's row and
        column (`rowcol`, patches x 2) in; the tokens' rows (patches / 4 x the text model's width, float32) out — or, given
        `blocks`, the patches' residual after that many blocks (a check's stage)."""
        n, vc = len(pixel_values), self.VC
        M = vc["spatial_merge_size"] ** 2
        if n > self.VIS_ROWS:
            raise Refused("%d patches, and this boot holds %d (picture_pixels %d)" % (n, self.VIS_ROWS, self.PICTURE_PIXELS))
        m, base = self.m, self.VSCR_AT
        m.write(base + self.VSCR["pix"][0], np.ascontiguousarray(pixel_values, dtype="<f2").tobytes())
        m.write(base + self.VSCR["rc"][0], np.ascontiguousarray(rowcol, dtype="<u4").tobytes())
        P = self.VPLANES
        depth = vc["depth"] if blocks is None else blocks
        words = (["(ai_z__glm_5_3__vision_embed %s GVE %d %s)" % (P["pix"], n, P["x"])]
                 + ["(ai_z__glm_5_3__vision_block %s %s GVB%d %d)" % (P["x"], P["rc"], i, n) for i in range(depth)]
                 + (["(ai_z__glm_5_3__vision_tail %s GVT %d %s)" % (P["x"], n, P["out"])] if blocks is None else []))
        for word in words:
            # a word's answer is its out plane, or the error that says why it did not run
            if m.must("(type %s)" % word, "the tower's word %s" % word.split()[0][1:]) == self.m.K["sys__error"]:
                raise Refused("the tower's word %s answered an error" % word.split()[0][1:])
        if blocks is not None:
            x = m.read(base + self.VSCR["x"][0], n * vc["hidden_size"] * 2)
            return np.frombuffer(x, dtype="<f2").reshape(n, -1).astype(np.float32)
        out = m.read(base + self.VSCR["out"][0], (n // M) * vc["out_hidden_size"] * 2)
        return np.frombuffer(out, dtype="<f2").reshape(n // M, -1).astype(np.float32)

    # ── pictures: the tower's rows in place of their pads ───────────────────────────────────────────────────────
    def vision(self):
        """The model's vision tower (▶ `vision_glm.GlmVision`), made when the first picture comes. GLM's attention has no
        rotary part, so a picture's tokens take their places in the cache as text does: they need only their rows."""
        if getattr(self, "_vision", None) is None:
            if not (self.CH and self.EPLAN is not None):
                raise Refused("a picture goes in with a prompt read as rows: the experts on the CPU, `fused` on")
            from .vision_glm import GlmVision
            v = GlmVision(self.folder, card=self if self.VIS else None)
            if v.out != self.H:
                raise Refused("the tower's rows are %d wide and the model's %d" % (v.out, self.H))
            self._vision = v
        return self._vision

    def _rows_in(self, rows, n):
        """A chunk's rows that are a picture's — `rows` {index in the chunk: the tower's row} — written into every stream
        of their rows of srows, over their pads' embeddings, a run of consecutive rows at a time."""
        H, M = self.H, self.MULT
        mine = sorted(r for r in rows if 0 <= r < n)
        runs, start = [], None
        for k, r in enumerate(mine):
            if start is None:
                start = r
            if k + 1 == len(mine) or mine[k + 1] != r + 1:
                runs.append((start, r + 1))
                start = None
        for a, b in runs:
            block = np.stack([rows[r] for r in range(a, b)]).astype("<f2")
            self.m.write(self.SROWS_AT + a * M * H * 2, np.repeat(block[:, None, :], M, axis=1).tobytes())

    def prefill(self, tokens, first, greedy=True, sample=None, rows=None):
        """Positions `first ..` for `tokens`: as rows, a chunk at a time, when the experts are the CPU's and there is more
        than one; then the next token from the last row (greedy), or its logits left for a sampler. A picture's tokens
        come with `rows` {index in `tokens`: the tower's row, in place of that pad's embedding}."""
        pictures = rows or {}
        rows = bool(self.CH) and self.EPLAN is not None and len(tokens) >= 2
        if pictures and not rows:
            raise Refused("a picture goes in with a prompt read as rows: the experts on the CPU, two tokens or more")
        if not rows:
            out = None
            for i, tok in enumerate(tokens):
                self._mtp_prime(first + i, tok)
                out = self.step(tok, first + i, greedy, sample if i == len(tokens) - 1 else None)
                if self.MTP is not None:
                    self._run("(begin (nn__vector__copy %s hnB %d) (type hnB))" % (self.MTP_HIDDEN, self.H), "the state kept")
                    self._mtp_hid = first + i
            return out
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        if self.T_SLICE is not None and not pictures and self.MTP is None and self.SOURCE:
            return self._prefill_streamed(tokens, first, greedy, sample)
        out = None
        for at in range(0, len(tokens), self.CH):
            chunk = tokens[at:at + self.CH]
            mine = {i - at: r for i, r in pictures.items() if at <= i < at + len(chunk)}
            out = self.forward_rows(first + at, tokens=chunk, head=at + self.CH >= len(tokens), greedy=greedy, sample=sample,
                                    rows=mine)
            if self.MTP is not None:
                # ⭐ THE MTP LAYER GIVEN THE CHUNK'S HISTORY: each row's head state from its streams, the layer at the row
                #   before with the row's token — a position at a time, the layer's cache written, no draft read. With
                #   `mtp_history` k, only the prompt's last k rows (and the head state of the one before them)
                start = first if self.MTP_HISTORY is None else first + len(tokens) - self.MTP_HISTORY
                for r, tok in enumerate(chunk):
                    p = first + at + r
                    if p < start - 1 and p != first + len(tokens) - 1:
                        continue
                    if p >= start:
                        self._mtp_prime(p, tok)
                    self._run("(row_hidden %d)" % r, "row %d's state" % r)
                    self._mtp_hid = p
        return out

    def _mtp_prime(self, p, tok):
        """The MTP layer at `p - 1` from the token at `p` and the head state at `p - 1`, where that state is kept."""
        if self.MTP is not None and p > 0 and self._mtp_hid == p - 1:
            self._run("(begin (mtp_cache_from %d %d hnB) (type mres))" % (tok, p - 1), "the MTP layer at %d" % (p - 1))
            self._mtp_have = p - 1

    # ── a pipeline stage (▶ `pipeline.py`): this machine's layers, the streams crossing between stages ─────────────
    def forward(self, pos, token=None, x=None, head=False):
        """One position through this stage's layers: the token embedded (the first stage) or the streams that came across
        (`x`, MULT·H halves as bytes) set; then the next token (the last stage) or the streams to hand on."""
        m = self.m
        self._room(pos + 1)
        if token is not None:
            self._embedding("(begin (embed %d) (type streams))" % token, "the embedding at %d" % pos)
        else:
            m.write(self.STREAMS_AT, x)
        m.grid("(layers_f %d)" % pos)
        if head:                                     # the greedy head, or the head expression it is (`sampling.head`)
            return m.must(head if isinstance(head, str) else "(head)", "the head at %d" % pos)
        return m.read(self.STREAMS_AT, self.MULT * self.H * 2)

    def _embedding(self, text, what):
        """A program that embeds: with the table the first socket's (EMBED_CPU) a grid's, every worker standing for the
        hand-off (▶ `Machine.grid`), else the card's alone."""
        if self.EMBED_CPU:
            return self.m.grid(text)[1]
        return self.m.must(text, what)

    def forward_rows(self, first, tokens=None, xs=None, n=None, head=False, greedy=True, sample=None, rows=None):
        """A chunk of rows through this stage's layers: its tokens embedded (the first stage) — a picture's `rows` {index in
        the chunk: the tower's row} over its pads' — or its rows' streams that came across (`xs`, `n` rows of MULT·H
        halves) written; then the next token from its last row (the last stage), or the rows' streams to hand on."""
        m, H, M = self.m, self.H, self.MULT
        n = len(tokens) if tokens is not None else n
        if n > self.CH:
            raise Refused("a chunk is at most %d rows, not %d" % (self.CH, n))
        self._room(first + n)
        if tokens is not None:
            if self.SOURCE:
                self._prompt(tokens)
                self._embedding("(embed_rows %d)" % n, "the chunk's embeddings")
            else:
                # ⛳ the embeddings in programs of their own, a hundred rows each: one program is one form, and a form
                #   must fit in one of the heap's chunks
                for e0 in range(0, n, 128):
                    self._embedding("(begin %s (type srows))" % " ".join(
                        "(embed_row %d %s)" % (tokens[i], " ".join(str((i * M + j) * H) for j in range(M)))
                        for i in range(e0, min(n, e0 + 128))), "the chunk's embeddings")
            if rows:
                self._rows_in(rows, n)
        else:
            m.write(self.SROWS_AT, xs)
        if self.SOURCE:
            m.grid("(begin (rows %d %d) (type srows))" % (first, n))
        else:
            layers = " ".join("(r%d %d %d)" % (l, first, n) for l in self.run_layers)
            m.grid("(begin (sys__node_array__set nrows 0 %d) %s (type srows))" % (n, layers))
        if not head:
            return m.read(self.SROWS_AT, n * M * H * 2)
        # the last row's streams, then the head
        last = (n - 1) * M * H
        m.must("(begin %s (type streams))" % " ".join("(nn__vector__add %s zero_h %s %d)" % (
            self.rng("srows", last + j * H, H), self.rng("streams", j * H, H), H) for j in range(M)), "the prompt's last row")
        return m.must(head if isinstance(head, str) else sample or ("(head)" if greedy else "(head_logits)"), "the head")

    def _prefill_streamed(self, tokens, first, greedy, sample):
        """⭐ A PROMPT READ LAYER-MAJOR (NN-115 ③) — `prefill`'s, with `stream` (▶ qwen3_5.py's): a slice of SLICE_CHUNKS
        chunks at a time, its chunks' embeddings into the slice's streams, then every layer once over the slice — each
        chunk's attention and router, the layer's experts once over all the slice's rows, streamed onto the card, each
        chunk's close (▶ `rows_slice`). Then the next token from the last row."""
        m, H, M, step = self.m, self.H, self.MULT, self.CH
        self._room(first + len(tokens))
        span, last = step * self.SLICE_CHUNKS, 0
        for s0 in range(0, len(tokens), span):
            part = tokens[s0:s0 + span]
            k = (len(part) + step - 1) // step
            for c in range(k):
                chunk = part[c * step:(c + 1) * step]
                self._prompt(chunk)
                self._embedding("(embed_slice %d %d)" % (c * step, len(chunk)), "the slice's embeddings")
            last_n = len(part) - (k - 1) * step
            m.grid("(begin (rows_slice %d %d %d %d) (type SROWS_S))" % (first + s0, k, last_n, step))
            last = len(part) - 1
        at = last * M * H
        m.must("(begin %s (type streams))" % " ".join("(nn__vector__add (nn__vector__range SROWS_S %d %d) zero_h %s %d)" % (
            at + j * H, H, self.rng("streams", j * H, H), H) for j in range(M)), "the prompt's last row")
        return m.must(sample or ("(head)" if greedy else "(head_logits)"), "the head")

    def logits(self):
        return np.frombuffer(self.m.read(self.LOGITS_AT, self.VOCAB * 2), dtype="<f2").astype(np.float32)

    def read(self, name, n):
        """`n` halves of a buffer, by name — the card's, or the CPU's for a `c_` one, read as that worker."""
        if name.startswith("c"):
            head = name.split("_", 1)[0]
            if head == "c" or (head[1:].isdigit() and head.startswith("c")):
                self.m.act_as(CPU + (int(head[1:]) if head[1:] else 0))
        try:
            return np.frombuffer(self.m.read(self.at[name], n * 2), dtype="<f2").astype(np.float64)
        finally:
            self.m.act_as(CARD)

    def streams(self):
        return np.frombuffer(self.m.read(self.STREAMS_AT, self.MULT * self.H * 2), dtype="<f2").astype(np.float32)

    def set_streams(self, a):
        self.m.write(self.STREAMS_AT, _half(np.asarray(a).reshape(-1)))

    # ── LoRAs: a rank-1 mask chosen at boot, or none (▶ `_mask_layout`) ────────────────────────────────────────
    def load_lora(self, name):
        """The LoRA a conversation runs under: the one booted, which is all this model swaps to. Answers the seconds taken."""
        if name == self.active_lora:
            return 0.0
        if self.b.mask is not None:
            # ⛳ a rank-1 mask is chosen at boot: its cells are in the tables the procedures were written over
            raise Refused("a rank-1 mask is chosen at boot, not swapped: boot the folder with lora %r" % (name,))
        raise Refused("GLM 5.3 Flash takes a LoRA only as a rank-1 mask, chosen at boot")

    def unload_lora(self):
        if self.active_lora is not None:
            raise Refused("a rank-1 mask is chosen at boot, not taken off: boot the folder without it")

    def lora_identity(self):
        if self.active_lora is None:
            return None
        return dict(name=self.b.lora["name"], weights=self.b.lora["adapter"]["weights_sha"])

    def _ensure(self, name):
        """The LoRA `name` known, as a conversation adopted from disk asks (▶ `session.adopt`): the booted mask only."""
        if name != self.active_lora:
            raise Refused("this model runs the LoRA %r, chosen at boot, not %r" % (self.active_lora, name))
        self._lora_cache.setdefault(name, dict(ident=self.lora_identity()))

    def state_write(self, length, blobs):
        """A conversation's state, as `state_regions` lays it out, written back — its rows withheld from the tier first."""
        if self.P > 1:
            raise Refused("a conversation's state is saved and restored on one card — a group's is every card's, not yet")
        self._room(length, exact=True)
        for (at, _), blob in zip([r for r in self.state_regions(length) if r[1]], blobs):
            self.m.write(at, blob)
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None   # another conversation's: the MTP layer's held nothing

    def state_regions(self, length, where=None):
        """(address, bytes) of everything a conversation of `length` positions has written: each KDA layer's state
        and window whole, each MLA layer's latents `0..length` — and then each MLA layer's indexer caches, the ring of keys
        and gates and the pooled keys to its last pool, which choose what a position past `index_topk` attends. They come
        last, so a conversation saved without them still finds its latents and states where they were. `where`: each
        layer's state's address — the load's by default; a table of zeros answers the sizes before the boot."""
        if self.P > 1 and where is None:
            raise Refused("a conversation's state is saved and restored on one card — a group's is every card's, not yet")
        out, index = [], []
        for l in self.run_layers:
            at = (self.state_at if where is None else where)[l]
            if self.kinds[l] == "d":
                out += [(at + self.SPLAN.at_up_data, self.S_BYTES), (at + self.SPLAN.at_down_data, self.W_BYTES)]
            else:
                out.append((at + self.KVPLAN.at_up_data, length * self.LAT * 2))
                index += [(at + self.KVPLAN.at_up_data + self.IK_AT, self.IP_AT - self.IK_AT),
                          (at + self.KVPLAN.at_up_data + self.IP_AT, -(-length // self.KP) * self.ID * 2)]
        return out + index
