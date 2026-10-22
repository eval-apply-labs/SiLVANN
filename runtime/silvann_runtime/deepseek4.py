"""DEEPSEEK-V4.1-FLASH — four residual streams, attention over a window and compressed latents, 384 experts — on a card
and the CPU (NN-95).

The card (worker 0) holds every matrix a token always uses — the attention, the shared experts, the routers, the
hyper-connections and the head — and reads a token's row of the embedding from a table in RAM; the CPU (worker 1) holds
the routed experts in its own memory and runs the six a token picks. A position is ai_deepseek_4's sites a layer after
another (▶ `src/packages/ai_deepseek_4/`), each site collapsing the streams by the mix its previous site wrote:
```
  card   (ai_deepseek_4__attention streams at{l} pos)       the window's ring, a sink a head, the grouped output
  card   (ai_deepseek_4__route streams mo{l} hand picks)    the router's six picks and the rotated input in the hand
  CPU    xf{l}: (ai_deepseek_4__experts ...)                 the hand copied over, the six experts, their weighted sum
  card   (ai_deepseek_4__shared hand mo{l}) · (ai_deepseek_4__close streams mo{l} routed)
```
A compressing layer attends its window and its source's compressed latents — all of them up to `index_topk` (512), and
past that the best 512 its index source's indexer scores. ⛳ The indexer's first level, the candidate blocks layer 20
picks for the index sources after it, changes nothing until a layer has more latents than those blocks hold (16,384), so
the context stops there until it is here (▶ `src/docs/design/pending/deepseek_v41.md` ③). The n-gram memory (engram, layers 1 and 14) is
hashed and its rows read on the host each position (▶ `engram_ds4.py`), from the tables' files — `engram_from`, the
checkpoint's folder until a pack carries them. The routed experts are in one socket's RAM: the inner width (2304) is not a
whole number of rotation blocks in two equal parts, so two sockets need parts of their own widths.

Every matrix is a TurboQuant record rotated along its input, so each gemv is handed its input rotated; the lossless ones
(d = 16: the routers, the maps, the norms, the sinks) take it as it is.

⭐ THE STREAMS ARE FLOATS — and the collapse a site norms, and the head's: a sink token's pass fp16's range (the
begin-of-sentence token's largest |stream| 48,104 after layer 20, 307,505 after layer 39 in the model's own code,
`MEASURED` node03 2026-10-21, measurements/2026-10-21_deepseek_streams_fp32_node03.md), which DeepSeek's bf16 holds and a
half does not. A sublayer's input is normed before it is a half, and its output is added to the streams as a float.
"""
import os
import pickle

import numpy as np

from . import experts_file, expert_tier, sampling
from .generate import Generates
from .model_folder import NL, Refused

ARCHITECTURES = ("deepseek_v41_text",)
CARD = 0                           # the first card is worker 0; the CPU worker comes after the cards (`self.CPU`)
EXPERT = "mlp.experts.%d.%s_proj.weight"
MIX_BYTES = 256                    # ▶ AI_DEEPSEEK_4__MIX_BYTES


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


def angles(n, rot, theta, original=0, factor=1.0, beta_fast=32, beta_slow=1):
    """The rotary's cosines and sines of positions 0 .. n: a row of `rot` floats a position — `rot/2` cosines, then the
    sines — as the model's `precompute_freqs_cis` makes them, in fp32: with `original` > 0, YaRN — the dimensions whose
    wavelength fits the training context keep their frequency, those far past it are divided by `factor`, and the band
    between is faded across. `MEASURED` 2026-10-18 node-01: within one fp32 ulp (6e-8) of the model's own at θ 10,000
    and at θ 160,000 with YaRN (factor 16 over 65,536), 4,096 positions."""
    import math
    freqs = (1.0 / (np.float32(theta) ** (np.arange(0, rot, 2, dtype=np.float32) / np.float32(rot)))).astype(np.float32)
    if original > 0:
        dim = lambda turns: rot * math.log(original / (turns * 2 * math.pi)) / (2 * math.log(theta))
        low, high = max(math.floor(dim(beta_fast)), 0), min(math.ceil(dim(beta_slow)), rot - 1)
        ramp = np.clip((np.arange(rot // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1).astype(np.float32)
        smooth = (1 - ramp).astype(np.float32)
        freqs = (freqs / np.float32(factor) * (1 - smooth) + freqs * smooth).astype(np.float32)
    a = np.outer(np.arange(n, dtype=np.float32), freqs).astype(np.float32)
    return np.concatenate([np.cos(a), np.sin(a)], axis=1).astype("<f4")


class DeepSeek4(Generates):
    PROGRAM = ("sampling", "generate")       # the model folder's lisp/ files (▶ `ModelFolder.program`)
    SAMPLES_ON_CARD = True
    # ⛳ AN ANSWER A TOKEN AT A TIME FROM THE HOST: the n-gram memory's rows are hashed and read on the host before each
    #   position (`engram_rows`), which `generate`'s loop on the card would leave the last position's — so a session steps
    #   every token (▶ session.py's `GENERATES`)
    GENERATES = False

    @staticmethod
    def longest_context(config):
        """The longest conversation this runtime holds: past `candidate_topk_blocks · candidate_block_size` latents, an
        index source after the candidate source scores inside the candidate blocks, which are not here yet (▶ the refusal
        in `__init__`) — None where no layer reaches them."""
        layers, cand = int(config["num_hidden_layers"]), int(config.get("candidate_source_layer_id", -1))
        reach = int(config.get("candidate_topk_blocks", 0)) * int(config.get("candidate_block_size", 0))
        index = list(config["index_source_layer_ids"])
        most = [reach * r + r - 1 for l, r in enumerate(list(config["compress_ratios"])[:layers])
                if r and 0 <= cand < max((s for s in index if s <= l), default=-1)]
        return min(most) if most else None

    def __init__(self, folder, machine, max_context=512, layers=None, experts_int8=True, engram_from=None,
                 experts_from="ram", experts_in_ram=0, experts_lock=False, experts_dir=None, socket=None, card=0, cards=None,
                 card_experts=0, chunk=128, card_experts_gb=0, card_shared="auto", card_landing=1, card_landing_line=3):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b, self.c = folder, machine, b, c
        self.H, self.LAYERS, self.MULT = c["hidden_size"], c["num_hidden_layers"], c["hc_mult"]
        self.ITERS, self.HC_EPS, self.EPS = c["hc_sinkhorn_iters"], float(c["hc_eps"]), float(c["rms_norm_eps"])
        self.QH, self.HD, self.RD = c["num_attention_heads"], c["head_dim"], c["qk_rope_head_dim"]
        self.QR, self.OR, self.G, self.W = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"], c["sliding_window"]
        self.EXPERTS, self.TOPK, self.MINTER = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
        self.RSCALE, self.LIMIT, self.THETA = float(c["routed_scaling_factor"]), float(c["swiglu_limit"]), float(c["rope_theta"])
        if c.get("scoring_func") != "sqrtsoftplus" or c.get("n_shared_experts") != 1:
            raise Refused("the router is %r with %r shared experts — this family runs sqrtsoftplus with one"
                          % (c.get("scoring_func"), c.get("n_shared_experts")))
        self.RATIO = list(c["compress_ratios"])[:self.LAYERS]
        self.ENGRAM = set(c.get("engram_layer_ids", ()))
        self.SOURCES = list(c["kv_source_layer_ids"])
        self.INDEXERS = list(c["index_source_layer_ids"])
        self.TOPK_C = int(c["index_topk"])
        self.IH, self.ID = int(c["index_n_heads"]), int(c["index_head_dim"])
        self.CAND_FROM = int(c.get("candidate_source_layer_id", -1))
        self.CAND_REACH = int(c.get("candidate_topk_blocks", 0)) * int(c.get("candidate_block_size", 0))
        rs = c.get("rope_scaling") or {}
        self.YARN = dict(theta=float(c["compress_rope_theta"]), original=int(rs.get("original_max_position_embeddings", 0)),
                         factor=float(rs.get("factor", 1.0)), beta_fast=rs.get("beta_fast", 32), beta_slow=rs.get("beta_slow", 1))
        self.max_context = max_context
        self.EXPERTS_INT8 = experts_int8
        self.run_layers = list(range(self.LAYERS)) if layers is None else list(layers)
        self.ENG = None
        if any(l in self.ENGRAM for l in self.run_layers):
            # ⭐ THE TABLES IN THE PACK: `engram/` beside its layers (the checkpoint's two shards that hold them, ~190 GiB,
            #   and their index) — so a pack stands alone; `engram_from` names another folder of them (a faster drive)
            if engram_from is None and os.path.isfile(os.path.join(folder.path, "engram", "model.safetensors.index.json")):
                engram_from = os.path.join(folder.path, "engram")
            if engram_from is None:
                raise Refused("layers %s read the n-gram memory: the pack has no engram/, and engram_from names no folder of "
                              "its tables" % sorted(self.ENGRAM))
            from .engram_ds4 import Engram
            self.ENG = Engram(c, os.path.join(folder.path, "tokenizer.json"), engram_from)
        self.history = []                 # the conversation's tokens, which the n-gram memory hashes
        for l in self.run_layers:
            r = self.RATIO[l]
            if r and self.source(l) not in self.run_layers:
                raise Refused("layer %d reads layer %d's compressed latents, which this machine does not run" % (l, self.source(l)))
            if r and self.indexer(l) not in self.run_layers:
                raise Refused("layer %d reads layer %d's selection of latents, which this machine does not run" % (l, self.indexer(l)))
            # ⛳ an index source past the candidate source scores only inside its candidate blocks, which hold every latent
            #   up to CAND_REACH of them — so up to there the candidates change nothing, and past it they are not here yet
            if r and 0 <= self.CAND_FROM < self.indexer(l) and max_context // r > self.CAND_REACH:
                raise Refused("layer %d: past %d latents its index source scores inside the candidate blocks, which are not "
                              "here yet — max_context %d" % (l, self.CAND_REACH, max_context))
        if self.MINTER % 2 or (self.QH * self.HD) % self.G:
            raise Refused("the shapes do not divide: inner %d, %d heads of %d in %d groups" % (self.MINTER, self.QH, self.HD, self.G))
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        self.head_rec, self.emb_rec = g["lm_head.weight"], g["model.language_model.embed_tokens.weight"]
        self.VOCAB = int(self.head_rec["rows"])
        # ⭐ THE CARDS — one (`card`, which of the machine's), or a TENSOR-PARALLEL GROUP of them (`cards`): each card holds
        #   its share of every layer's query heads and output groups and of the shared expert's inner width, and the same
        #   columns of the matrices back, so its attention and its MoE are partial sums the verbs add over the cards' own
        #   memory (▶ nn's contracts/objects/partials.cuh). The rest — the window's ring, the compressor, the indexer, the
        #   router, the hyper-connections, the n-gram memory — every card runs whole, the same on each. Card 0 embeds and
        #   runs the head, and hands the CPU its picks. ▶ `_cut`
        self.CARD_DEVS = [int(card)] if cards is None else [int(x) for x in cards]
        self.P, self.GROUP = len(self.CARD_DEVS), 0
        self.CPU = self.P                         # the CPU worker: after the cards
        n512, tail = divmod(self.MINTER, 512)
        if self.QH % self.P or self.G % self.P or n512 % self.P:
            raise Refused("%d cards do not split %d heads, %d output groups and %d 512-blocks of the shared expert evenly"
                          % (self.P, self.QH, self.G, n512))
        self.QHC, self.GC = self.QH // self.P, self.G // self.P
        # ⛳ THE SHARED EXPERT'S INNER WIDTH IN WHOLE ROTATION BLOCKS: nn rotates a width in 512-blocks, then a 256 and a 128
        #   tail (2304: four 512s and a 256), so a card's part is whole 512-blocks and the last card's also has the tail —
        #   rotated alone it is rotated as the whole was (two cards: 1024 and 1280)
        each = 512 * (n512 // self.P)
        self.I_AT = [s * each for s in range(self.P)]
        self.I_OF = [each + (tail if s == self.P - 1 else 0) for s in range(self.P)]
        # the routed experts' CPU worker: the whole machine, or — `socket` k, a stage of a pipeline a socket — that socket's
        #   device, its threads and its memory on that socket's node (▶ x86_avx2's devices: 0 the machine, 1 + k a socket)
        self.CPUS = [dict(w=self.CPU, dev=0 if socket is None else 1 + int(socket), part=0)]
        # ⭐ THE ROUTED EXPERTS IN RAM, OR MAPPED FROM A FILE ON A DRIVE (▶ `experts_file`) — the 40 layers' are 272 GB and
        #   node-01 has 251 GB of RAM. `"mapped"`: the first `experts_in_ram` layers this machine runs copied into RAM as
        #   `"ram"` copies them all, the rest written once in their slots' layout to a file in `experts_dir` and mapped
        #   over their slots — the page cache holds the ones in use, the drive the rest
        if experts_from not in ("ram", "mapped"):
            raise Refused("experts_from is 'ram' or 'mapped', not %r" % (experts_from,))
        self.EXPERTS_FROM, self.EXPERTS_IN_RAM = experts_from, int(experts_in_ram)
        self.EXPERTS_LOCK, self.LOCKED_GB = bool(experts_lock), 0.0      # ▶ `_plan`
        # ⭐ THE CARDS' SHARE OF THE ROUTED EXPERTS — ⚖ 2026-10-19: *"each card keeps their half of the expert they are
        #   computing, and at the end we add up the two partials"*. Experts 0 .. card_experts-1 of every layer on the cards,
        #   each card its part of their inner width (the shared expert's cut: whole rotation blocks), the rest in the CPU's
        #   memory: a pick on the cards is a partial on each, added in the group's reduce with the CPU's sum of the others.
        #   ⛳ A FIXED SHARE, NOT A CACHE: the router's bias keeps the experts' load level, so any card_experts of them take
        #   about that share of the picks — and the cards stay in step without deciding anything between them.
        self.NK = int(card_experts)
        if not 0 <= self.NK <= self.EXPERTS:
            raise Refused("card_experts is 0 .. %d, not %d" % (self.EXPERTS, self.NK))
        # ⭐ A TIER A CARD (NN-99): `card_experts_gb` of each card's memory a cache of whole routed experts, promoted from the
        #   CPU's RAM as they are picked (▶ nn's `nn__expert_tier`) — in a group every card a tier of its own, a position's
        #   picks shared out among them and `card_shared` of each card's slots a layer the most picked, kept on every card;
        #   the CPU computes the rest. A layer mapped from its file feeds no tier: its picks stay the CPU's. `card_landing`
        #   and `card_landing_line` as GLM's (▶ `expert_tier.tier`).
        self.CARD_GB = float(card_experts_gb or 0)
        self.CARD_SHARED = None if card_shared in (None, "auto") else float(card_shared)   # "auto": ▶ `_shared_from_speeds`
        self.CARD_SPEEDS, self._FED = None, {}
        self.CARD_LANDING, self.CARD_LANDING_LINE = int(card_landing), int(card_landing_line)
        if self.CARD_GB and self.NK:
            raise Refused("card_experts (a fixed share on the cards) or card_experts_gb (a tier a card), not both")
        if self.CARD_GB and self.TOPK > 12:
            raise Refused("a tier takes at most 12 picks a position, not %d" % self.TOPK)
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        self._plan()
        # ⭐ A PROMPT AS ROWS (▶ `prefill`): `chunk` positions a step, a layer at a time — each row's attention in order,
        #   the routed experts once over the chunk on the CPU; over a group every card's rows sites taking the group's sum of
        #   the chunk's rows. A card's share of the experts reads a prompt a position at a time
        self.CHUNK = int(chunk) if not self.NK else 0
        if self.CHUNK > 4096:
            raise Refused("chunk is at most 4096 positions, not %d" % self.CHUNK)
        self.HANDB = ((2 * self.H + 2 * self.TOPK + 7) & ~7) + 8 * self.TOPK    # a hand's bytes: ▶ AI_DEEPSEEK_4__HAND
        # `experts_rows`' scratch: the rows' inputs, each pick's pair and weight, its gate-and-up, activation and rotated
        #   rows, and its fp32 sum (▶ nn's `nn__routed__rows`), P the chunk's picks
        P = self.CHUNK * self.TOPK
        self.RSCRATCH = 2 * self.CHUNK * self.H + 18 * P + 8 * P * self.MINTER + 4 * P * self.H + 64
        # the rows verbs' work (▶ the package's `ai_deepseek_4__rows__zzprivate_take`): the pairs, then the attention
        #   site's rows of each step — its input and rotated input, wq_a's and wkv's, the rank normed and rotated, the
        #   query (its answer in place), the answer rotated, the groups' answer, the compressor's latents and gates, the
        #   indexer's queries and weights — or the MoE close's: the shared expert's gate-and-up, activation, rotated, down
        R, QW, ON, I = self.CHUNK, self.QH * self.HD, self.G * self.OR, self.MINTER
        att = [8 * R * (3 + self.G), 2 * R * self.H, 2 * R * self.H, 2 * R * (self.QR + self.HD), 2 * R * self.QR, 2 * R * self.QR,
               2 * R * QW, 2 * R * QW, 2 * R * ON, 4 * R * self.HD, 2 * R * self.IH * self.ID, 2 * R * self.IH]
        close = [24 * R, 4 * R * I, 2 * R * I, 2 * R * I, 2 * R * self.H]
        self.AWORK = max(sum(att) + 256 * len(att), sum(close) + 256 * len(close), 2 * R * self.H) if R else 0
        self.ERWS = {}                    # each engram layer's rows for a chunk, on every card: (card, layer) -> address
        self.ISEL_AT = {}                 # each index source's choices, on card 0: a row's `index_topk` and their count
        # ⭐ A MASK, where the folder's LoRA is one (▶ `nn_rank1_mask_from_projection.py`): `W' = W + Σ r_i·v_iᵀ` beside wo_b
        #   and the shared expert's down, each a site's cells past its table (▶ the package's AI_DEEPSEEK_4__MASK__* and
        #   `_mask_layout`). Chosen at boot.
        self.active_lora = folder.lora if b.mask is not None else None
        self._lora_cache = {}
        self._mask_layout()

    def indexer(self, l):
        """The layer whose selection of latents layer `l` attends past `index_topk`: the last index source at or before it."""
        return max(s for s in self.INDEXERS if s <= l)

    def source(self, l):
        """The layer whose compressed latents layer `l` reads: the last compressing source at or before it."""
        return max(s for s in self.SOURCES if s <= l)

    @staticmethod
    def sfx(name, s):
        """Card `s`'s own name for a buffer or a table: card 0's is the name itself."""
        return name if s == 0 else "%s_%d" % (name, s)

    def _cut(self, n, s):
        """Card `s`'s part of the matrix `n`: None for whole, ("rows", runs) or ("cols", (first, end))."""
        if self.P == 1:
            return None
        a, QW, GW, i0, ip = "self_attn.", self.QHC * self.HD, self.GC * self.OR, self.I_AT[s], self.I_OF[s]
        sh = "mlp.shared_experts.%s_proj.weight"
        return {a + "wq_b.weight":  ("rows", [(s * QW, QW)]),
                a + "attn_sink":    ("cols", (s * self.QHC, (s + 1) * self.QHC)),
                a + "wo_a.weight":  ("rows", [(s * GW, GW)]),
                a + "wo_b.weight":  ("cols", (s * GW, (s + 1) * GW)),
                sh % "gate":        ("rows", [(i0, ip)]),
                sh % "up":          ("rows", [(i0, ip)]),
                sh % "down":        ("cols", (i0, i0 + ip))}.get(n)

    def _part(self, l, n, s):
        """Card `s`'s bytes of layer `l`'s matrix `n` (data, scales), or None when it holds it whole."""
        c = self._cut(n, s)
        return None if c is None else NL.dense_part(self.b, n, l, **{c[0]: c[1]})

    @staticmethod
    def cn(i, name):
        return ("c_" if i == 0 else "c%d_" % i) + name

    # ── ① the collections ─────────────────────────────────────────────────────────────────────────────────────
    def dense_names(self, l):
        """A layer's matrices that live on the card: all but the routed experts."""
        return [n for n in self.b.names(l) if ".experts." not in n]

    def _plan(self):
        b, run = self.b, self.run_layers
        keys = sorted({n for l in run for n in self.dense_names(l)})
        first = {k: next(l for l in run if k in self.dense_names(l)) for k in keys}
        # each card's plan of each matrix: its part's, or the whole's
        self.PLANS = [{} for _ in range(self.P)]
        for k in keys:
            for s in range(self.P):
                got = self._part(first[k], k, s)
                self.PLANS[s][k] = NL.dense_plan(b, k, first[k]) if got is None else NL.SlotPlan(len(got[0]), 0, 0, len(got[1]))
        # ⭐ ONE COLLECTION A SHAPE, a slot a tensor of it in each layer — the shape on every card, so a type is the same
        #   number on each
        groups = {}
        for k in keys:
            groups.setdefault(tuple((self.PLANS[s][k].up_data, self.PLANS[s][k].down_lut) for s in range(self.P)), []).append(k)
        self.TYPE, self.WHICH, self.WIDTH = {}, {}, {}
        for t, members in enumerate(groups.values()):
            for i, k in enumerate(members):
                self.TYPE[k], self.WHICH[k] = t, i
            self.WIDTH[t] = len(members)
        self.PLAN = self.PLANS[0]
        slots = {t: sum(1 for l in run for n in self.dense_names(l) if self.TYPE[n] == t) for t in self.WIDTH}
        t = len(groups)
        self.T_HEAD, self.T_RING = t, t + 1

        def plan_of(rec):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, _ = plan_of(self.emb_rec)
        self.EMB_ROW_BYTES, self.EMB_BITS = int(self.emb_rec["row_bytes"]), int(self.emb_rec["d"])
        self.EMB_DATA_BYTES, self.HEAD_BITS = len(self.emb_data), int(self.head_rec["d"])
        self.TABLE_BYTES = (len(self.emb_data) + len(self.emb_lut) + 255) // 256 * 256
        # a layer's window: its ring of `W` normed latents, their tails turned, in the up half
        self.RING_BYTES = self.W * self.HD * 2
        self.RPLAN = NL.SlotPlan(self.RING_BYTES, 0, 0, 256)
        # a source layer's compressed latents, a row a group, in the up half; its partial group's ring — the latents,
        # then their gates — in the down half
        self.T_CACHE = t + 2
        self.CACHE_ROWS = {l: self.max_context // self.RATIO[l] + 1 for l in run if l in self.SOURCES}
        rmax = max([self.RATIO[l] for l in self.CACHE_ROWS] or [1])
        rows = max([n for n in self.CACHE_ROWS.values()] or [1])
        self.KEYS_AT = rows * self.HD * 2                        # and after the latents, their index keys, a row a latent
        self.CPLAN = NL.SlotPlan(self.KEYS_AT + rows * self.ID * 2, 0, 0, 2 * rmax * self.HD * 2)
        # every card's collections — the head card 0's alone, a slot of a few bytes on the others
        self.COLLECTIONS = [([(t, self.PLANS[s][members[0]], slots[t]) for t, members in enumerate(groups.values())]
                             + [(self.T_HEAD, self.HPLAN if s == 0 else NL.SlotPlan(256, 0, 0, 256), 1),
                                (self.T_RING, self.RPLAN, max(1, len(run))), (self.T_CACHE, self.CPLAN, max(1, len(self.CACHE_ROWS)))])
                            for s in range(self.P)]
        self.ARENAS = [sum(p.slot_bytes * n for _, p, n in cs) for cs in self.COLLECTIONS]
        self.collections, self.arena = self.COLLECTIONS[0], self.ARENAS[0]
        # the CPU's one collection: an expert a slot — gate and up rows as one matrix, then down
        r = lambda w: b.record(run[0], EXPERT % (0, w))
        gr, dn = r("gate"), r("down")
        self.EPLAN = NL.SlotPlan(2 * self.MINTER * gr.row_bytes, 2 * self.MINTER * 2, dn.rows * dn.row_bytes, dn.rows * 2)
        self.E_D = gr.d
        # the CPU's slots: a layer copied into RAM leaves out the cards' experts; a mapped layer is its file, every expert
        ram = run if self.EXPERTS_FROM == "ram" else run[:self.EXPERTS_IN_RAM]
        self.cpu_slots = sum(self.EXPERTS - self.NK if l in ram else self.EXPERTS for l in run)
        self.cpu_arena = self.EPLAN.slot_bytes * self.cpu_slots
        # `experts_lock`: the layers copied into RAM locked there, so the system cannot swap them out — a mapped layer is
        #   its file's (▶ `experts_file.lock_check`)
        experts_file.lock_check(self.EXPERTS_LOCK, "ram" if ram else self.EXPERTS_FROM,
                                self.EPLAN.slot_bytes * (self.EXPERTS - self.NK) * len(ram))
        # each card's slot of an expert: its part of the gate and up rows, their scales, its columns of every down row, every
        # down row's scale
        if any(dn.row_bytes * ip % self.MINTER for ip in self.I_OF):
            raise Refused("a card's part of the down rows does not end on a byte")
        self.KPLANS = [NL.SlotPlan(2 * ip * gr.row_bytes, 2 * ip * 2, dn.rows * (dn.row_bytes * ip // self.MINTER), dn.rows * 2)
                       for ip in self.I_OF]
        self.T_CEXP = self.T_CACHE + 1
        self.TIER_LEAST = len(run) * self.EPLAN.slot_bytes            # the least tier it takes: a slot a layer
        self.CARD_SLOTS = 0
        if self.CARD_GB:
            # the tier's slots hold whole experts, as the CPU's do: gate and up rows first, then their scales, the down codes
            #   and their scales — the layout nn's tier copies from RAM as it is (▶ `nn__expert_tier__stitch`)
            if self.EPLAN.at_up_data != 0:
                raise Refused("an expert's slot does not start with its gate rows, which a tier copies as they are")
            self.CARD_SLOTS = int(self.CARD_GB * (1 << 30)) // self.EPLAN.slot_bytes
            if self.CARD_SLOTS < len(run):
                raise Refused("%.1f GB holds %d experts, fewer than one a layer" % (self.CARD_GB, self.CARD_SLOTS))
            for k in range(self.P):
                self.COLLECTIONS[k].append((self.T_CEXP, self.EPLAN, self.CARD_SLOTS))
            self.ARENAS = [sum(p.slot_bytes * n for _, p, n in cs) for cs in self.COLLECTIONS]
            self.collections, self.arena = self.COLLECTIONS[0], self.ARENAS[0]
            self.GATE_ROW, self.DOWN_ROW = gr.row_bytes, dn.row_bytes
        if self.NK:
            for k in range(self.P):
                self.COLLECTIONS[k].append((self.T_CEXP, self.KPLANS[k], self.NK * len(run)))
            self.ARENAS = [sum(p.slot_bytes * n for _, p, n in cs) for cs in self.COLLECTIONS]
            self.collections, self.arena = self.COLLECTIONS[0], self.ARENAS[0]

    @staticmethod
    def _section(big, arena, types, kinds, many=128, bigs=3, mains=2):
        typed = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in types)
        return ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
                "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:%d\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:%d\n"
                "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:%d\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:8\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:0\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                % (mains, many, big, bigs, (arena + (256 << 20)) >> 20, len(kinds), kinds, len(types), typed))

    def workers(self):
        return [(self.m.silicon, d) for d in self.CARD_DEVS] + [("x86_avx2", c["dev"]) for c in self.CPUS]

    def device_params(self):
        """The card's section and the CPU worker's: {worker: text}."""
        kinds = "f" * self.LAYERS
        big = max(8 << 20, self.max_context * self.RD * 4, self.VOCAB * 2)
        # a chunk's rows on card 0, a big buffer each: their streams, the hands, the experts' answers, the n-gram rows
        R = self.CHUNK
        rows = [R * self.MULT * self.H * 4, R * self.HANDB, R * self.H * 2, R * self.H * 2, self.AWORK] + (
            [R * self.ENG.width * 2 for l in self.run_layers if l in self.ENGRAM] if self.ENG is not None else []) if R else []
        out = {}
        # a group's other cards: their rows' streams and work, the n-gram rows — the rest fit the smaller classes
        other = [R * self.MULT * self.H * 4, self.AWORK] + (
            [R * self.ENG.width * 2 for l in self.run_layers if l in self.ENGRAM] if self.ENG is not None else []) if R else []
        for k in range(self.P):
            ask = self.MASK_ASK.get(k, 0)
            mine = rows if k == 0 else other
            out[k] = self._section(max([big, ask] + mine), self.ARENAS[k], self.COLLECTIONS[k], kinds,
                                   bigs=(4 if ask else 3) + len(mine))
            out[k] += ("nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                       "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                       "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % (self.TABLE_BYTES if k == 0 else 256))
        for c in self.CPUS:
            out[c["w"]] = self._section(max(1 << 20, R * self.HANDB, R * self.H * 2, self.RSCRATCH if R else 0), self.cpu_arena,
                                        [(0, self.EPLAN, self.cpu_slots)], kinds, many=16, bigs=3 + (3 if R else 0))
        return out

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        self.ring_at, self.cache_at, self.CB, self.MASK_CELLS = {}, {}, {}, {}
        for k in range(self.P):
            m.act_as(k)
            loader = m.loader()
            if k == 0:
                loader.layer_width(0, self.T_HEAD, 1)
                loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
                self.EMB_AT = m.buffer(self.TABLE_BYTES, "EMBEDDING", ram=True)
                m.write(self.EMB_AT, self.emb_data)
                m.write(self.EMB_AT + self.EMB_DATA_BYTES, self.emb_lut)
            for l in self.run_layers:
                for t in sorted({self.TYPE[n] for n in self.dense_names(l)}):
                    loader.layer_width(l, t, self.WIDTH[t])
                for n in self.dense_names(l):
                    got = self._part(l, n, k)
                    if got is None:
                        loader.load_dense(b, self.PLANS[k][n], self.TYPE[n], n, l, which=self.WHICH[n])
                    else:
                        loader.load_bytes(self.PLANS[k][n], self.TYPE[n], l, got[0], got[1], which=self.WHICH[n])
                loader.layer_width(l, self.T_RING, 1)
                self.ring_at[(k, l)] = loader.reserve(self.T_RING)
                loader.assign(l, self.T_RING, 0, self.ring_at[(k, l)])
                if l in self.CACHE_ROWS:
                    loader.layer_width(l, self.T_CACHE, 1)
                    self.cache_at[(k, l)] = loader.reserve(self.T_CACHE)
                    loader.assign(l, self.T_CACHE, 0, self.cache_at[(k, l)])
            if self.NK:
                self._load_card_experts(k, loader)
            if self.CARD_GB:
                for l in self.run_layers:
                    loader.layer_width(l, self.T_CEXP, self.EXPERTS)
            if loader.handed() != 0:
                raise Refused("card %d: %d slots were reserved and never published" % (k, loader.handed()))
            self._card_buffers(k)
            self._mask_buffers(k)
        self.head_data = self.head_lut = self.emb_data = self.emb_lut = None
        self._load_experts(progress)
        m.act_as(self.CPU)
        self._cpu_buffers(0)
        m.bind_values(dict(VOCAB=self.VOCAB, STOPS=0, SEEN=sampling.SEEN, GROUP=self.GROUP))
        for k in range(self.P):
            m.act_as(k)
            self._tables(k)
        if self.CARD_GB:
            m.act_as(CARD)
            expert_tier.bind(m, {k: self.sfx("tier_card", k) for k in range(self.P)}, len(self.workers()))
        if self.P > 1:
            # ⭐ THE GROUP: each card joins it on its own worker, card 0 first (▶ nn's `nn__partials__join`)
            m.node_array("TP_POS", 1)
            if self.CHUNK:
                m.node_array("TP_ROWS", 2)          # a chunk's first position and its rows (▶ `_tp_rows`)
            for k in range(self.P):
                m.act_as(k)
                m.must("(nn__partials__join %d %d %d %d)" % (self.GROUP, k, self.P, min(4, self.P)),
                       "card %d joining the group" % k, "sys__value_true")
        m.act_as(CARD)
        m.define_programs(self._procedures)
        if self.CARD_GB and self.P > 1 and self.CARD_SHARED is None:
            self._shared_from_speeds()
        self.global_ = None
        self.b.release()                            # every layer's file let go (▶ `Bundle.release`)

    def _card_pieces(self, view, l, x, k):
        """Card `k`'s part of expert `x` of layer `l` as its slot's pieces: rows I_AT[k] .. + I_OF[k] of gate and up and
        their scales, the same columns of every down row, every down row's scale."""
        g, u, d = (self.b.record(l, EXPERT % (x, w)) for w in ("gate", "up", "down"))
        i0, ip, p = self.I_AT[k], self.I_OF[k], self.KPLANS[k]
        span = lambda at, n: view[at:at + n]
        b0, rb = d.row_bytes * i0 // self.MINTER, d.row_bytes * ip // self.MINTER
        d_at = d.data_span()[0]
        cols = np.ascontiguousarray(view[d_at:d_at + d.rows * d.row_bytes].reshape(d.rows, d.row_bytes)[:, b0:b0 + rb]).reshape(-1)
        return [(p.at_up_data, span(g.data_span()[0] + i0 * g.row_bytes, ip * g.row_bytes)),
                (p.at_up_data + ip * g.row_bytes, span(u.data_span()[0] + i0 * u.row_bytes, ip * u.row_bytes)),
                (p.at_up_lut, span(g.lut_span()[0] + i0 * 2, ip * 2)),
                (p.at_up_lut + ip * 2, span(u.lut_span()[0] + i0 * 2, ip * 2)),
                (p.at_down_data, cols), (p.at_down_lut, span(*d.lut_span()))]

    def _load_card_experts(self, k, loader):
        """Card `k`'s parts of experts 0 .. NK-1 of every layer, into its collection of them."""
        for l in self.run_layers:
            view = np.frombuffer(self.b.blob(l), dtype=np.uint8)
            loader.layer_width(l, self.T_CEXP, self.EXPERTS)
            for x in range(self.NK):
                at = loader.reserve(self.T_CEXP)
                for into, piece in self._card_pieces(view, l, x, k):
                    loader.write(at + into, piece.tobytes())
                loader.assign(l, self.T_CEXP, x, at)

    def _expert_pieces(self, view, l, x, part):
        """Expert `x` of layer `l` as its slot's pieces (▶ `experts_file`)."""
        g, u, d = (self.b.record(l, EXPERT % (x, w)) for w in ("gate", "up", "down"))
        return expert_tier.part_pieces(view, g, u, d, self.EPLAN, 0, 1, self.MINTER)

    def _load_experts(self, progress):
        """Every layer's routed experts into the CPU worker's slots: copied in from the mapped bundle, or — `"mapped"`, past
        the first `experts_in_ram` layers — their file mapped over them."""
        m, b, p = self.m, self.b, self.EPLAN
        ram = self.run_layers if self.EXPERTS_FROM == "ram" else self.run_layers[:self.EXPERTS_IN_RAM]
        mapped = [l for l in self.run_layers if l not in ram]
        # ⭐ A FILE A LAYER, so a stage of a pipeline, or a test's few layers, maps the same files as the whole model: each
        #   written once (from the bundle on the RAID) if it is not there already
        paths = {l: experts_file.ensure(b, [(self.CPU, 0, 1)], p, [l], self.EXPERTS, self.experts_dir, self._expert_pieces,
                                        list(range(self.LAYERS)))[0] for l in mapped}
        m.act_as(self.CPU)
        loader, at = m.loader(), {}
        self.SRC = {}                     # a layer copied into RAM: its experts' addresses there, which a tier copies from
        for i, l in enumerate(self.run_layers):
            view = np.frombuffer(b.blob(l), dtype=np.uint8) if l in ram else None
            loader.layer_width(l, 0, self.EXPERTS)
            # ⛳ nn NUMBERS A WORKER'S EXPERTS FROM ITS `first`: the CPU's, past the cards' NK, are expert x at x - NK. A mapped
            #   layer's slots are its file's, every expert's one after another: the cards' ones are published past the end,
            #   where no pick reaches
            for x in range(self.NK if l in ram else 0, self.EXPERTS):
                at[(l, x)] = loader.reserve(0)
                if view is not None:
                    for into, piece in self._expert_pieces(view, l, x, 0):
                        m.write_from(at[(l, x)] + into, piece.ctypes.data, piece.nbytes)
                loader.assign(l, 0, x - self.NK if x >= self.NK else self.EXPERTS - self.NK + x, at[(l, x)])
            if l in ram:
                self.SRC[l] = [at[(l, x)] for x in range(self.EXPERTS)] if not self.NK else None
                # the layer's file let go once its experts are copied out — the loop's last slice of its map with it — so
                #   its pages do not sit beside the experts in RAM until the load ends (▶ `Bundle.release`)
                view = piece = None
                b.release(l)
            if progress and l in ram:
                progress(i + 1, len(self.run_layers))
        if loader.handed() != 0:
            raise Refused("%d expert slots were reserved and never published" % loader.handed())
        for l in mapped:
            lo = at[(l, 0)]
            if any(at[(l, x)] - lo != x * p.slot_bytes for x in range(self.EXPERTS)):
                raise Refused("layer %d's slots were not handed out one after another, as its file is laid out" % l)
            # ⛳ A LAYER'S SLOTS ARE A WHOLE NUMBER OF PAGES (384 × 17,714,176 bytes), so each layer starts on a page
            m.map_file(lo, p.slot_bytes * self.EXPERTS, paths[l], 0)
        self.LOCKED_GB = experts_file.lock_experts(self.EXPERTS_LOCK, [at[(l, x)] for l in ram for x in range(self.NK, self.EXPERTS)],
                                                   p.slot_bytes)
        m.act_as(CARD)

    def _card_buffers(self, k):
        """Card `k`'s buffers, by its own names (▶ `sfx`): card 0's are the names themselves, and only card 0 has the head's,
        the sampler's and the answer's."""
        m, H, n = self.m, self.H, lambda name: self.sfx(name, k)
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        self.signs = signs
        final_w = np.frombuffer(np.asarray(self.global_["model.language_model.norm.weight"]["vbr_data"]).tobytes(),
                                dtype="<f2").astype(np.float64)[:H]
        cb = self.CB[k] = {}
        m.buffer(1024, n("signs"), _half(signs[:512]))
        m.buffer(1024, n("ones512"), _half(np.ones(512)))
        m.buffer(H * 2, n("signs_row"), _half(np.tile(signs[:512], H // 512)))
        m.buffer(H * 2, n("fnw"), _half(final_w))
        m.buffer(H * 2, n("zero_h"), _half(np.zeros(H)))
        cb["streams"] = m.buffer(self.MULT * H * 4, n("streams"))
        for nm in ("u", "hu", "e", "hn", "hnr", "routed"):
            cb[nm] = m.buffer(H * 2, n(nm))
        cb["hm"] = m.buffer(H * 4, n("hm"))             # the head's collapse, floats like the streams
        # the two sites' mixes, handed from one site to the next, and the first layer's one-hot `pre`
        cb["mixa"] = m.buffer(MIX_BYTES, n("mixa"))
        cb["mixb"] = m.buffer(MIX_BYTES, n("mixb"))
        # what the first layer this machine runs collapses by: the model's one-hot `pre` at layer 0, or — a later stage of a
        # pipeline — the mix the stage before it handed on with the streams (▶ `forward`)
        first = np.zeros(MIX_BYTES // 4, dtype="<f4")
        first[0] = 1.0
        cb["mix_in"] = m.buffer(MIX_BYTES, n("mix_in"), first.tobytes())
        if self.P > 1:
            # the group's sums land here, and nothing else writes it (▶ ai_deepseek_4's `zzprivate_tp`); every card past the
            # first takes a position's embedding as the group's sum of its zeros and card 0's (`_procedures`)
            cb["tsum"] = m.buffer(H * 2, n("tsum"))
        if k == 0:
            m.buffer(64, "weights")
            m.buffer(64, "res")
            sampling.buffers(m)
            self._answer_buffers()
            self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        cb["angles"] = m.buffer(self.max_context * self.RD * 4, n("angles"), angles(self.max_context, self.RD, self.THETA).tobytes())
        if any(self.RATIO[l] for l in self.run_layers):
            m.buffer(self.max_context * self.RD * 4, n("angles_c"), angles(self.max_context, self.RD, **self.YARN).tobytes())
            m.buffer((self.W + self.TOPK_C) * self.HD * 2, n("gather"))
            m.buffer(2 * 64 * self.HD, n("czero"), _half(np.zeros(64 * self.HD)))
            m.buffer(4 * (max(self.CACHE_ROWS.values()) + 1), n("iscores"))
            for l in self.run_layers:
                if l in self.INDEXERS:
                    # a choice a row: a chunk's rows each keep theirs (▶ the package's A PROMPT AS ROWS)
                    at = m.buffer(4 * (self.TOPK_C + 1) * max(1, self.CHUNK), n("isel%d" % l))
                    if k == 0:
                        self.ISEL_AT[l] = at
        m.buffer(self.QH * (self.W + self.TOPK_C) * 4, n("scores"))
        QW, ON, I = self.QH * self.HD, self.G * self.OR, self.MINTER
        m.buffer(2 * (5 * H + 3 * self.QR + self.HD + 3 * QW + 2 * ON + 4 * self.HD + self.ID + self.IH * self.ID + self.IH) + 256,
                 n("ascratch"))
        m.buffer(2 * (4 * H + self.EXPERTS) + 8 * I + 256, n("mscratch"))
        cb["hand"] = m.buffer(2 * (H + self.TOPK), n("hand"))
        if k == 0 and self.CHUNK:
            # a chunk's rows: their streams and the two sites' mixes, the hands the CPU reads, what its experts give back,
            #   the n-gram memory's rows and the tokens
            R = self.CHUNK
            m.buffer(R * self.MULT * H * 4, "srows")
            m.buffer(R * MIX_BYTES, "amix")
            m.buffer(R * MIX_BYTES, "bmix")
            m.buffer(R * self.HANDB, "hands")
            m.buffer(R * H * 2, "rrows")
            m.buffer(R * H * 2, "xrows")
            m.buffer(self.AWORK, "awork")
            m.node_array("TOKS", R)
            m.node_array("ROWS_N", 1)
        if self.CHUNK and self.P > 1:
            # a group's chunk on every card: the sums of its rows (▶ ai_deepseek_4's `..._rows__zzprivate_sum`) and the rows'
            #   embedding — card 0's, the group's sum of its own and the others' zeros; every other card its rows as card 0's
            R = self.CHUNK
            m.buffer(R * H * 2, n("tsums"))
            m.buffer(R * H * 2, n("erows_e"))
            if k > 0:
                m.buffer(R * H * 2, n("zerows"), _half(np.zeros(R * H)))
                m.buffer(R * self.MULT * H * 4, n("srows"))
                m.buffer(R * MIX_BYTES, n("amix"))
                m.buffer(R * MIX_BYTES, n("bmix"))
                m.buffer(R * self.HANDB, n("hands"))
                m.buffer(R * H * 2, n("xrows"))
                m.buffer(R * H * 2, n("zrows"), _half(np.zeros(R * H)))
                m.buffer(self.AWORK, n("awork"))
        if self.NK or self.CARD_GB:
            m.buffer(8 * self.TOPK * self.MINTER, n("kscratch"))
            cb["routed_k"] = m.buffer(H * 2, n("routed_k"))
        m.node_array(n("picks"), self.TOPK)
        if self.ENG is not None:
            cb["eng"] = {}
            for l in self.run_layers:
                if l in self.ENGRAM:
                    cb["eng"][l] = m.buffer(self.ENG.width * 2, n("eng%d" % l))
                    rec = lambda name: self.b.record(l, name)
                    import nn_unpack_model as NU
                    q, kk = (np.asarray(NU.decode_record(rec(x).rec, self.b.blob(l)), dtype=np.float64)
                             for x in ("engram.q_weight", "engram.k_weight"))
                    m.buffer(self.MULT * H * 2, n("engw%d" % l), _half(q * kk))
                    if self.CHUNK:
                        self.ERWS[(k, l)] = m.buffer(self.CHUNK * self.ENG.width * 2, n("erows%d" % l))
            m.buffer(2 * (self.ENG.width + (self.MULT + 1) * H) + 256, n("escratch"))
        if k == 0:
            # card 0's, by the names the rest of this runtime reads them by
            self.at = {nm: cb[nm] for nm in ("u", "hu", "e", "hm", "hn", "hnr", "routed", "hand")}
            self.STREAMS_AT, self.MIXA_AT, self.MIXB_AT, self.MIXIN_AT = cb["streams"], cb["mixa"], cb["mixb"], cb["mix_in"]
            self.ANGLES_AT = cb["angles"]
            self.ENG_AT = cb.get("eng", {})

    def _cpu_buffers(self, i):
        m, H, c = self.m, self.H, lambda name: self.cn(i, name)
        m.buffer(1024, c("signs"), _half(self.signs[:512]))
        for nm in ("e", "routed"):
            self.at[c(nm)] = m.buffer(H * 2, c(nm))
        m.buffer(2 * (H + self.TOPK), c("hand"))
        if self.CHUNK:
            m.buffer(self.CHUNK * self.HANDB, c("hands"))
            m.buffer(self.CHUNK * H * 2, c("rrows"))
            m.buffer(self.RSCRATCH, c("rscratch"))
        m.buffer(2 * H, c("zero"), _half(np.zeros(H)))
        m.buffer(8 * self.TOPK * self.MINTER, c("scratch"))

    # ── the planes ────────────────────────────────────────────────────────────────────────────────────────────
    def plane(self, l, n, which, k=0):
        p = self.PLANS[k][n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d %d %d %d)" % (l, self.TYPE[n], self.WHICH[n], at, size)

    def ring(self, l):
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_RING, self.RPLAN.at_up_data, self.RING_BYTES)

    def cache(self, l, which):
        """Source layer `l`'s compressed latents (`c`) or its partial group's ring (`r`)."""
        p, s = self.CPLAN, self.source(l)
        if which == "c":
            return "(nn__expert__plane %d %d 0 %d %d)" % (s, self.T_CACHE, p.at_up_data, self.CACHE_ROWS[s] * self.HD * 2)
        if which == "k":
            return "(nn__expert__plane %d %d 0 %d %d)" % (s, self.T_CACHE, p.at_up_data + self.KEYS_AT, self.CACHE_ROWS[s] * self.ID * 2)
        return "(nn__expert__plane %d %d 0 %d %d)" % (s, self.T_CACHE, p.at_down_data, 2 * self.RATIO[s] * self.HD * 2)

    def rng(self, buf, at, n):
        return "(nn__vector__range %s %d %d)" % (buf, at, n)

    # ── a mask (▶ `nn_rank1_mask_from_projection.py`) ─────────────────────────────────────────────────────────
    def _mask_layout(self):
        """Where each card keeps the mask's tensors: every card the whole of each site's `r` and its own columns of `v` — at
        the attention its groups' part of wo_b's input, at the shared expert its part of the inner width, whole rotation
        blocks both — so a card's mask is a partial the group's sum adds as it adds the matrix's. `MASK_PLACE[(card, l,
        site)]` the two tensors' (start, count) in halves, after the scratch the verbs' dots take; `MASK_ASK` each card's
        buffer's request."""
        self.MASK_PLACE, self.MASK_ASK, self._mask_data = {}, {}, {}
        if self.b.mask is None:
            return
        from safetensors.numpy import load_file
        t = load_file(self.b.mask["path"])
        for key in t:
            parts = key.split(".")
            if len(parts) != 4 or parts[0] != "layers" or parts[2] not in ("attn", "shared") or parts[3] not in ("r", "v"):
                raise Refused("the mask's %s is not wo_b's or the shared expert's down, which DeepSeek-V4.1 masks" % key)
        self._mask_scratch = 256                               # the dots: a float a row of `v`
        cuts = lambda k: {"attn": (k * self.GC * self.OR, (k + 1) * self.GC * self.OR, self.G * self.OR),
                          "shared": (self.I_AT[k], self.I_AT[k] + self.I_OF[k], self.MINTER)}
        for k in range(self.P):
            at = self._mask_scratch
            for l in self.run_layers:
                for site, (a, z, whole) in cuts(k).items():
                    name = "layers.%d.%s" % (l, site)
                    if name + ".r" not in t:
                        continue
                    r, v = t[name + ".r"], t[name + ".v"]
                    if r.ndim != 2 or v.ndim != 2 or r.shape != (v.shape[0], self.H) or v.shape[1] != whole \
                            or not 1 <= v.shape[0] <= 8:
                        raise Refused("the mask's layer %d %s is %s and %s, where this model's is (rank, %d) and (rank, %d)"
                                      % (l, site, r.shape, v.shape, self.H, whole))
                    place = []
                    for x in (r, np.ascontiguousarray(v[:, a:z])):
                        place.append((at // 2, x.size))
                        at += (2 * x.size + 255) & ~255
                    self.MASK_PLACE[(k, l, site)] = place
                    self._mask_data[(k, l, site)] = (r, v[:, a:z])
            if any(p[0] == k for p in self.MASK_PLACE):
                self.MASK_ASK[k] = max(at, (4 << 20) + 256)

    def _mask_buffers(self, k):
        """Card `k`'s mask buffer, filled, and each of its sites' three cells: views of its `r`, its `v` and the scratch."""
        if not self.MASK_ASK.get(k):
            return
        name = self.sfx("mk", k)
        base = self.m.buffer(self.MASK_ASK[k], name)
        for (k_, l, site), place in self.MASK_PLACE.items():
            if k_ != k:
                continue
            for (start, _), x in zip(place, self._mask_data[(k, l, site)]):
                self.m.write(base + 2 * start, np.ascontiguousarray(x, dtype="<f2").tobytes())
            self.MASK_CELLS[(k, l, site)] = [self.rng(name, *place[0]), self.rng(name, *place[1]),
                                             self.rng(name, 0, self._mask_scratch // 2)]

    def _masked(self, k, l, site):
        """A table's mask cells for card `k`'s site of layer `l`: none where the model has no mask, an integer 0 where it
        leaves this site alone."""
        if self.b.mask is None:
            return []
        return list(self.MASK_CELLS.get((k, l, site), [0]))

    def _tables(self, k):
        """Card `k`'s ai_deepseek_4 plane tables: a layer's attention site (`at`), its MoE site (`mo`) — its own heads, groups
        and inner width under tensor parallelism — and, with card 0's, the CPU's experts' (`ex`). ▶ the package's
        `contracts/objects/layer.cuh` for what each cell is."""
        m, b, H, s, n = self.m, self.b, self.H, "self_attn.", lambda name: self.sfx(name, k)
        pair = lambda l, x: [self.plane(l, x, "data", k), self.plane(l, x, "lut", k)]
        data = lambda l, x: self.plane(l, x, "data", k)
        d = lambda l, x: b.record(l, x).d
        hc = lambda l, site: [data(l, "hc_%s_%s" % (site, x)) for x in ("fn", "base", "scale")]
        head = [H, self.MULT, self.ITERS]
        for i, l in enumerate(self.run_layers):
            earlier = n("mix_in") if i == 0 else n("mixb")
            mats = [s + x + ".weight" for x in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")]
            m.table(n("at%d" % l), hc(l, "attn") + [data(l, "input_layernorm.weight"), n("mixa"), earlier]
                    + [c for x in mats for c in pair(l, x)]
                    + [data(l, s + "q_norm.weight"), data(l, s + "kv_norm.weight"), data(l, s + "attn_sink"), self.ring(l),
                       n("angles_c") if self.RATIO[l] else n("angles"), n("scores"), n("signs"), n("ascratch")]
                    + head + [self.QHC, self.HD, self.RD, self.QR, self.OR, self.GC, self.W] + [d(l, x) for x in mats]
                    + [self.EPS, self.HC_EPS] + self._compressed(l, k) + self._masked(k, l, "attn"))
            mats = ["mlp.shared_experts.%s_proj.weight" % x for x in ("gate", "up", "down")]
            m.table(n("mo%d" % l), hc(l, "ffn") + [data(l, "post_attention_layernorm.weight"), n("mixb"), n("mixa"),
                                                   data(l, "mlp.gate.weight"), data(l, "mlp.gate.e_score_correction_bias")]
                    + [c for x in mats for c in pair(l, x)] + [n("signs"), n("mscratch")]
                    + head + [self.EXPERTS, self.TOPK, self.I_OF[k]] + [d(l, x) for x in mats]
                    + [self.EPS, self.HC_EPS, self.RSCALE, self.LIMIT] + self._masked(k, l, "shared"))
        for l in self.run_layers:
            if l in self.ENGRAM:
                m.table(n("en%d" % l), [n("eng%d" % l)] + pair(l, "engram.wkv.weight") + [n("engw%d" % l), n("signs"), n("escratch")]
                        + [H, self.MULT, self.ENG.width, d(l, "engram.wkv.weight"), self.EPS])
        if self.NK:
            kp = self.KPLANS[k]
            for l in self.run_layers:
                m.table(n("ek%d" % l), [n("signs"), n("zero_h"), n("kscratch"), l, self.T_CEXP, kp.at_up_lut - kp.at_up_data,
                                        kp.at_down_data - kp.at_up_data, kp.at_down_lut - kp.at_up_data, H, self.I_OF[k],
                                        self.EXPERTS, self.TOPK, self.E_D, 0, self.NK, 0, self.LIMIT])
        if self.CARD_GB:
            # the card's tier: a layer's experts table over every expert (its picks the ones the tier holds and the visit
            #   gives this card), the layers copied into RAM its sources — their arrays made once, read by every card's
            #   tier — and the tier itself, a layer's slice every card keeps in a group
            p = self.EPLAN
            fed = {}
            for l in self.run_layers:
                m.table(n("exk%d" % l), [n("signs"), n("zero_h"), n("kscratch"), l, self.T_CEXP, p.at_up_lut - p.at_up_data,
                                         p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.MINTER,
                                         self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS, 0, self.LIMIT])
                if self.SRC.get(l):
                    if k == 0:
                        self._int_array("src%d" % l, self.SRC[l])
                    fed[l] = "src%d" % l
            self._FED[k] = fed
            self._tier(k)
        if k != 0:
            return
        p = self.EPLAN
        m.act_as(self.CPU)
        c = lambda name: self.cn(0, name)
        for l in self.run_layers:
            cells = [l, 0, p.at_up_lut - p.at_up_data, p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.MINTER,
                     self.EXPERTS, self.TOPK, self.E_D, self.NK, self.EXPERTS - self.NK, 1 if self.EXPERTS_INT8 else 0, self.LIMIT]
            m.table("ex%d_0" % l, [c("signs"), c("zero"), c("scratch")] + cells)
            if self.CHUNK:                    # a chunk's: the same experts, `experts_rows`' scratch
                m.table("exr%d_0" % l, [c("signs"), c("zero"), c("rscratch")] + cells)
        m.act_as(k)

    def _tier(self, k):
        """Card `k`'s tier table (▶ `expert_tier.tier`), with the share of its slots a layer every card keeps in a group —
        none until it is known (`_shared_from_speeds`)."""
        p, part = self.EPLAN, self.CARD_SHARED or 0.0
        shared = int(round(part * self.CARD_SLOTS / len(self.run_layers))) if self.P > 1 else 0
        expert_tier.tier(self.m, self.sfx("tier_card", k), self.T_CEXP, self.EXPERTS, self.TOPK, self.H, self.MINTER, p, self.LAYERS,
                         self._FED[k], 1, p, self.GATE_ROW, self.DOWN_ROW, self.CARD_LANDING, None, self.CARD_LANDING_LINE,
                         ranks=self.P, shared=shared)

    def _shared_from_speeds(self):
        """`card_shared` "auto" in a group: a CPU pick against a card expert, timed here (▶ `expert_tier.speeds` — the
        first fed layer's first experts, the head's product on card 0 priced to a routed expert's bytes), the slice from
        them, and every card's tier made again with it."""
        m = self.m
        l = next((l for l in self.run_layers if l in self._FED[0]), self.run_layers[0])
        cpu_s, card_s = expert_tier.speeds(m, CARD, self.CPU, "picks", list(range(self.TOPK)),
                                           "(ai_deepseek_4__experts c_hand picks ex%d_0 c_routed)" % l,
                                           self.EPLAN.slot_bytes / expert_tier.head_bytes(self.HPLAN))
        self.CARD_SHARED = expert_tier.slice_of(cpu_s, card_s)
        self.CARD_SPEEDS = dict(cpu_ms=cpu_s * 1e3, card_ms=card_s * 1e3, ratio=cpu_s / max(card_s, 1e-12), shared=self.CARD_SHARED)
        for k in range(self.P):
            m.act_as(k)
            self._tier(k)
        m.act_as(CARD)
        expert_tier.bind(m, {k: self.sfx("tier_card", k) for k in range(self.P)}, len(self.workers()))

    def _int_array(self, name, values):
        """A node array of integers, bound by name — filled a hundred at a time, as a program must fit one heap chunk."""
        m = self.m
        m.node_array(name, len(values))
        for at in range(0, len(values), 100):
            m.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set %s %d %d)" % (name, i, v)
                                                     for i, v in enumerate(values[at:at + 100], at)),
                   "the array %s" % name, "sys__value_true")

    def _compressed(self, l, k=0):
        """A compressing layer's cells past its table (▶ AI_DEEPSEEK_4__CMP__*): its source's cache, the source's own
        compressor where this is the source (its maps, norm and ring), and the rows it gathers to attend — card `k`'s."""
        r, n = self.RATIO[l], lambda name: self.sfx(name, k)
        if not r:
            return []
        cw = lambda x: "self_attn.compressor.%s.weight" % x
        if l in self.SOURCES:
            gate = [self.plane(l, cw("wgate"), "data", k), self.plane(l, cw("wgate"), "lut", k)] if r > 1 else [n("signs"), n("signs")]
            comp = ([self.plane(l, cw("wkv"), "data", k), self.plane(l, cw("wkv"), "lut", k)] + gate
                    + [self.plane(l, cw("norm"), "data", k), self.cache(l, "r"), n("czero")])
            d = [self.b.record(l, cw("wkv")).d, self.b.record(l, cw("wgate")).d if r > 1 else 0]
        else:
            comp, d = [n("signs")] * 7, [0, 0]
        ix, data = "self_attn.indexer.", lambda x: self.plane(l, x, "data", k)
        pair = lambda x: [self.plane(l, x, "data", k), self.plane(l, x, "lut", k)]
        own = [ix + "wq_b.weight", ix + "weights_proj.weight"]
        idx = ([self.cache(l, "k")] + (pair(own[0]) + pair(own[1]) if l in self.INDEXERS else [n("signs")] * 4)
               + (pair(ix + "wk.weight") + [data(ix + "k_norm.weight")] if l in self.SOURCES else [n("signs")] * 3)
               + [n("iscores"), n("isel%d" % self.indexer(l)), 1 if l in self.INDEXERS else 0, self.IH, self.ID]
               + [self.b.record(l, x).d if l in self.INDEXERS else 0 for x in own]
               + [self.b.record(l, ix + "wk.weight").d if l in self.SOURCES else 0])
        return ([self.cache(l, "c")] + comp + [n("gather"), r, 1 if l in self.SOURCES else 0, self.CACHE_ROWS[self.source(l)],
                                               self.TOPK_C] + d + idx)

    # ── ③ the program ────────────────────────────────────────────────────────────────────────────────────────
    def _layer(self, l):
        """Layer `l` at the position `pos` on card 0: its attention site, its MoE site with the CPU's experts — under tensor
        parallelism each site's output the group's sum."""
        H, CPU = self.H, self.CPU
        tp = " GROUP tsum" if self.P > 1 else ""
        engram = "(ai_deepseek_4__engram streams en%d) " % l if l in self.ENGRAM else ""
        mine = "(ai_deepseek_4__experts hand picks ek%d routed_k) " % l if self.NK else ""
        add = " (nn__vector__add routed routed_k routed %d)" % H if self.NK or self.CARD_GB else ""
        # ⭐ A TIER'S CARD FIRST: its visit marks the picks the cards compute before the CPU reads them, and the card's share
        #   is queued; the CPU starts after it, and the writes a promotion held back go once the CPU is done
        tier = "(ai_deepseek_4__experts hand picks exk%d routed_k %s) " % (l, expert_tier.BINDING) if self.CARD_GB else ""
        written = " (nn__expert_tier__written %s)" % expert_tier.BINDING if self.CARD_GB else ""
        return (engram + "(ai_deepseek_4__attention streams at%d pos%s) (ai_deepseek_4__route streams mo%d hand picks) %s"
                "(sys__compute %d names xf%d_0) %s(ai_deepseek_4__shared hand mo%d) (sys__result %d)%s "
                "(nn__buffer__copy c_routed %d routed %d)%s (ai_deepseek_4__close streams mo%d routed%s)"
                % (l, tp, l, tier, CPU, l, mine, l, CPU, written, CPU, 2 * H, add, l, tp))

    def _tp_layers(self, k):
        """Every layer at the position in TP_POS on card `k` past the first, by its own names: the same sites as card 0's,
        its partials added in the group's reduces, the routed experts none of its share (card 0 adds the CPU's sum)."""
        n, pos = (lambda name: self.sfx(name, k)), "(nth TP_POS 0)"
        out = []
        for l in self.run_layers:
            if l in self.ENGRAM:
                out.append("(ai_deepseek_4__engram %s %s)" % (n("streams"), n("en%d" % l)))
            mine = ("(ai_deepseek_4__experts %s %s %s %s) " % (n("hand"), n("picks"), n("ek%d" % l), n("routed_k")) if self.NK else
                    "(ai_deepseek_4__experts %s %s %s %s %s) " % (n("hand"), n("picks"), n("exk%d" % l), n("routed_k"), expert_tier.BINDING)
                    if self.CARD_GB else "")
            out.append("(ai_deepseek_4__attention %s %s %s GROUP %s) (ai_deepseek_4__route %s %s %s %s) %s(ai_deepseek_4__shared %s %s) "
                       "(ai_deepseek_4__close %s %s %s GROUP %s)%s"
                       % (n("streams"), n("at%d" % l), pos, n("tsum"), n("streams"), n("mo%d" % l), n("hand"), n("picks"), mine,
                          n("hand"), n("mo%d" % l), n("streams"), n("mo%d" % l), n("routed_k") if self.NK or self.CARD_GB else n("zero_h"),
                          n("tsum"), " (nn__expert_tier__written %s)" % expert_tier.BINDING if self.CARD_GB else ""))
        return " ".join(out)

    def _procedures(self):
        m, H = self.m, self.H
        # the token's row of the embedding table, un-rotated (Rᵀu = S ⊙ H·u), into e — and into every stream
        m.defun("(defun (embed_e tok) (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok %d) %d) "
                "(nn__vector__range EMBEDDING (+ %d tok) 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                "(nn__vector__pointwise_mul hu signs_row e %d)))"
                % (self.EMB_ROW_BYTES // 2, self.EMB_ROW_BYTES // 2, self.EMB_DATA_BYTES // 2, H, self.EMB_BITS, H, H))
        m.defun("(defun (embed tok) (begin (embed_e tok) %s (type e)))" % self._spread(0))
        # the head: the streams collapsed by the last site's mix (the model's `hc_pre` before its norm), the norm, the logits
        logits = ("(nn__hyper__collapse_fp32 streams mixb hm %d %d) (nn__rmsnorm__apply_fp32 hm fnw hn %d %r) "
                  "(nn__hadamard__rotate hn signs hnr %d) "
                  "(nn__turboquant__gemv (nn__expert__plane 0 %d 0 %d %d) (nn__expert__plane 0 %d 0 %d %d) hnr logits_h %d %d %d)"
                  % (H, self.MULT, H, self.EPS, H, self.T_HEAD, self.HPLAN.at_up_data, self.HPLAN.up_data, self.T_HEAD,
                     self.HPLAN.at_down_lut, self.HPLAN.down_lut, self.VOCAB, H, self.HEAD_BITS))
        m.defun("(defun (head) (begin %s (nn__argmax__find logits_h res %d) (nn__buffer__read res)))" % (logits, self.VOCAB))
        m.defun("(defun (head_logits) (begin %s (type hm)))" % logits)
        m.defun("(defun (logits) (begin %s))" % logits)
        m.load_source(self.folder.program("sampling"))
        m.load_source(self.folder.program("generate"))
        m.act_as(self.CPU)
        for l in self.run_layers:
            m.picture("xf%d_0" % l, "(begin (nn__buffer__copy hand %d c_hand %d) (ai_deepseek_4__experts c_hand picks ex%d_0 c_routed))"
                      % (CARD, 2 * (self.H + self.TOPK), l))
            if self.CHUNK:                    # a chunk's experts: its hands across, every expert they picked once
                m.picture("xr%d_0" % l, "(begin (nn__buffer__copy hands %d c_hands (* (nth ROWS_N 0) %d)) "
                          "(ai_deepseek_4__experts_rows c_hands exr%d_0 c_rrows (nth ROWS_N 0)))" % (CARD, self.HANDB, l))
        # ⭐ EVERY OTHER CARD OF THE GROUP: its layers, a picture — from card 0's embedding (`tp_first`: the group's reduce of
        #   its zeros with card 0's embedding, into every stream) or from the streams already there (`tp_given`)
        for k in range(1, self.P):
            m.act_as(k)
            n = lambda name: self.sfx(name, k)
            m.picture(n("tp_first"), "(begin (nn__partials__reduce GROUP %s %s %d) %s %s (type %s))"
                      % (n("zero_h"), n("e"), H, self._spread(k), self._tp_layers(k), n("streams")))
            m.picture(n("tp_given"), "(begin %s (type %s))" % (self._tp_layers(k), n("streams")))
            if self.CHUNK:
                m.picture(n("tp_rows"), self._tp_rows(k))
        m.act_as(CARD)
        m.view("names")
        for l in self.run_layers:
            m.defun("(defun (lay%d pos) (begin %s (type streams)))" % (l, self._layer(l)))
        lays = " ".join("(lay%d pos)" % l for l in self.run_layers)
        if self.P == 1:
            m.defun("(defun (layers pos) (begin %s (type streams)))" % lays)
            m.defun("(defun (position tok pos) (begin (embed tok) %s (type streams)))" % lays)
            if self.CHUNK:
                self._rows_procedures()
            return
        start = lambda pic: " ".join("(sys__compute %d names %s)" % (k, self.sfx(pic, k)) for k in range(1, self.P))
        wait = " ".join("(sys__result %d)" % k for k in range(1, self.P))
        m.defun("(defun (layers pos) (begin (sys__node_array__set TP_POS 0 pos) %s %s %s (type streams)))"
                % (start("tp_given"), lays, wait))
        m.defun("(defun (position tok pos) (begin (sys__node_array__set TP_POS 0 pos) %s (embed_e tok) "
                "(nn__partials__reduce GROUP e e %d) %s %s %s (type streams)))" % (start("tp_first"), H, self._spread(0), lays, wait))
        if self.CHUNK:
            self._rows_procedures(start("tp_rows"), wait)

    def _rows_procedures(self, start="", wait=""):
        """A chunk of positions as rows (▶ `prefill`): `(rows_chunk first n)` embeds the chunk's tokens (TOKS) into the
        rows' streams, every row's first mix the one-hot `mix_in`, runs every layer over them — `(rl{l} first n)`: the
        n-gram memory a row at a time where the layer has it; the attention site over the rows (`attention_rows`, its
        matrices once a chunk, each row's attention in position order, its mix in its record of `amix`); the router over
        every row (`route_rows`, the picks in the rows' hands, the rotated inputs in `xrows`); the routed experts on the
        CPU once over the chunk; the shared expert and the close over the rows (`close_rows`) — and leaves the last row's
        streams and mix where the head reads them. A chunk's experts can take longer than `sys__result` waits, so the card
        asks `sys__completed` until the CPU says so, then collects (as GLM's chunks do). Over a group (`start` and `wait`
        the other cards' `tp_rows`, ▶ `_tp_rows`): the chunk's place and length in TP_ROWS, its embedding the group's sum,
        and both sites' rows each the group's sum."""
        m, H, MH2, MIXH = self.m, self.H, 2 * self.MULT * self.H, MIX_BYTES // 2
        tp = " GROUP tsums" if self.P > 1 else ""
        row = lambda buf, n: "(nn__vector__range %s (* t %d) %d)" % (buf, n, n)
        stream = row("srows", MH2)
        for l in self.run_layers:
            eng = ""
            if l in self.ENGRAM:
                W = self.ENG.width
                eng = ("(while '(< t n) '(begin (nn__vector__copy %s eng%d %d) (ai_deepseek_4__engram %s en%d) (set! t (+ t 1)))) "
                       % (row("erows%d" % l, W), l, W, stream, l))
            m.defun("(defun (rl%d first n) (let ((t 0)) %s"
                    "(ai_deepseek_4__attention_rows srows at%d first n amix bmix awork%s)"
                    " (ai_deepseek_4__route_rows srows mo%d hands n amix bmix xrows awork)"
                    " (sys__compute %d names xr%d_0) (while '(sys__eq (sys__completed %d) (< 1 0)) '(< 0 1)) (sys__result %d)"
                    " (nn__buffer__copy c_rrows %d rrows (* n %d))"
                    " (ai_deepseek_4__close_rows srows mo%d xrows rrows n bmix awork%s)"
                    " n))"
                    % (l, eng, l, tp, l, self.CPU, l, self.CPU, self.CPU, self.CPU, 2 * H, l, tp))
        if self.P == 1:
            spread = " ".join("(nn__vector__widen e (nn__vector__range srows (+ (* t %d) %d) %d) %d)" % (MH2, 2 * j * H, 2 * H, H)
                              for j in range(self.MULT))
            m.defun("(defun (rows_embed n) (let ((t 0)) (while '(< t n) '(begin (embed_e (nth TOKS t)) %s"
                    " (nn__vector__copy mix_in %s %d) (set! t (+ t 1)))) n))" % (spread, row("bmix", MIXH), MIXH))
        else:
            # the rows' embeddings, then the group's sum of them — every other card's zeros — spread as one card's are
            m.defun("(defun (rows_embed n) (let ((t 0)) (while '(< t n) '(begin (embed_e (nth TOKS t)) (nn__vector__copy e %s %d) "
                    "(set! t (+ t 1)))) (nn__partials__reduce GROUP erows_e erows_e (* n %d)) %s n))"
                    % (row("erows_e", H), H, H, self._rows_spread(0)))
        m.defun("(defun (rows_chunk first n) (begin (sys__node_array__set TP_ROWS 0 first) (sys__node_array__set TP_ROWS 1 n) %s "
                "(rows_embed n) %s"
                " (nn__vector__copy (nn__vector__range srows (* (- n 1) %d) %d) streams %d)"
                " (nn__vector__copy (nn__vector__range bmix (* (- n 1) %d) %d) mixb %d) %s (type streams)))"
                % (start, " ".join("(rl%d first n)" % l for l in self.run_layers), MH2, MH2, MH2, MIXH, MIXH, MIXH, wait)
                if self.P > 1 else
                "(defun (rows_chunk first n) (begin (rows_embed n) %s"
                " (nn__vector__copy (nn__vector__range srows (* (- n 1) %d) %d) streams %d)"
                " (nn__vector__copy (nn__vector__range bmix (* (- n 1) %d) %d) mixb %d) (type streams)))"
                % (" ".join("(rl%d first n)" % l for l in self.run_layers), MH2, MH2, MH2, MIXH, MIXH, MIXH))

    def _rows_spread(self, k):
        """Card `k`'s chunk's embeddings (`erows_e`, a row of halves each) into every one of its rows' streams (floats), and
        every row's first mix the one-hot `mix_in` — the rows `0 .. (nth TP_ROWS 1)`."""
        H, MH2, MIXH, n = self.H, 2 * self.MULT * self.H, MIX_BYTES // 2, (lambda name: self.sfx(name, k))
        spread = " ".join("(nn__vector__widen (nn__vector__range %s (* t %d) %d) (nn__vector__range %s (+ (* t %d) %d) %d) %d)"
                          % (n("erows_e"), H, H, n("srows"), MH2, 2 * j * H, 2 * H, H) for j in range(self.MULT))
        return ("(set! t 0) (while '(< t (nth TP_ROWS 1)) '(begin %s (nn__vector__copy %s (nn__vector__range %s (* t %d) %d) %d) "
                "(set! t (+ t 1))))" % (spread, n("mix_in"), n("bmix"), MIXH, MIXH, MIXH))

    def _tp_rows(self, k):
        """A chunk's rows on card `k` past the first, by its own names — its place and length in TP_ROWS: the chunk's
        embedding the group's sum (its zeros, card 0's rows), then every layer's rows sites as card 0's, each output the
        group's sum, the routed experts none of its share (card 0 adds the CPU's)."""
        n, H, MH2 = (lambda name: self.sfx(name, k)), self.H, 2 * self.MULT * self.H
        first, rows = "(nth TP_ROWS 0)", "(nth TP_ROWS 1)"
        out = ["(nn__partials__reduce GROUP %s %s (* %s %d))" % (n("zerows"), n("erows_e"), rows, H), self._rows_spread(k)]
        for l in self.run_layers:
            if l in self.ENGRAM:
                W = self.ENG.width
                out.append("(set! t 0) (while '(< t %s) '(begin (nn__vector__copy (nn__vector__range %s (* t %d) %d) %s %d) "
                           "(ai_deepseek_4__engram (nn__vector__range %s (* t %d) %d) %s) (set! t (+ t 1))))"
                           % (rows, n("erows%d" % l), W, W, n("eng%d" % l), W, n("srows"), MH2, MH2, n("en%d" % l)))
            out.append("(ai_deepseek_4__attention_rows %s %s %s %s %s %s %s GROUP %s)"
                       % (n("srows"), n("at%d" % l), first, rows, n("amix"), n("bmix"), n("awork"), n("tsums")))
            out.append("(ai_deepseek_4__route_rows %s %s %s %s %s %s %s %s)"
                       % (n("srows"), n("mo%d" % l), n("hands"), rows, n("amix"), n("bmix"), n("xrows"), n("awork")))
            out.append("(ai_deepseek_4__close_rows %s %s %s %s %s %s %s GROUP %s)"
                       % (n("srows"), n("mo%d" % l), n("xrows"), n("zrows"), rows, n("bmix"), n("awork"), n("tsums")))
        return "(let ((t 0)) %s (type %s))" % (" ".join(out), n("srows"))

    def _spread(self, k):
        """Card `k`'s embedding `e` (halves) into every one of its streams (floats)."""
        H, n = self.H, lambda name: self.sfx(name, k)
        return " ".join("(nn__vector__widen %s %s %d)" % (n("e"), self.rng(n("streams"), 2 * j * H, 2 * H), H)
                        for j in range(self.MULT))

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: every layer's ring to zero (a position reads only the rows it has written, so this
        is for a clean state rather than for the answer)."""
        zeros = ["(nn__vector__zero %s %d)" % (self.ring(l), self.RING_BYTES // 2) for l in self.run_layers]
        zeros += ["(nn__vector__zero %s %d)" % (self.cache(l, "c"), self.CACHE_ROWS[l] * self.HD) for l in self.CACHE_ROWS]
        for k in range(self.P):
            self.m.act_as(k)
            self.m.must("(begin %s (type %s))" % (" ".join(zeros), self.sfx("streams", k)), "card %d's reset" % k)
        self.m.act_as(CARD)
        self.history = []

    def engram_rows(self, token, pos):
        """The n-gram memory's rows for `token` at `pos` written into each engram layer's buffer — before the position runs."""
        self.history[pos:] = [token]
        if self.ENG is None:
            return
        ids = self.ENG.ids(self.history, pos)
        rows = self.ENG.all_rows(ids)
        for i, l in enumerate(self.ENG.layers):
            if l in self.ENG_AT:
                row = _half(rows[l])
                for k in range(self.P):
                    self.m.act_as(k)
                    self.m.write(self.CB[k]["eng"][l], row)
        self.m.act_as(CARD)

    def layer(self, l, pos):
        """One layer at one position, the streams in and out on the card."""
        self.m.grid("(lay%d %d)" % (l, pos))
        f = self.m.e.sys.fault()
        if f["raised"] != 0:
            self.m.e.sys.fault_clear()
            raise Refused("layer %d at %d raised %s" % (l, pos, f))

    def step(self, token, pos, greedy=True, sample=None):
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        head = sample or ("(head)" if greedy else "(head_logits)")
        self.engram_rows(token, pos)
        value = self._run("(begin (position %d %d) %s)" % (token, pos, head), "position %d" % pos)
        return value if greedy or sample else None

    def prefill(self, tokens, first, greedy=True, sample=None, rows=None):
        """A prompt from `first`: as rows, `chunk` positions a step (▶ `_rows_procedures`), where the machine has them; else a
        position at a time. The last position's head as `step`'s."""
        if not self.CHUNK or len(tokens) < 2:
            out = None
            for i, tok in enumerate(tokens):
                out = self.step(tok, first + i, greedy, sample if i == len(tokens) - 1 else None)
            return out
        out = None
        for at in range(0, len(tokens), self.CHUNK):
            part = tokens[at:at + self.CHUNK]
            last = at + len(part) == len(tokens)
            out = self._rows(part, first + at, (sample or ("(head)" if greedy else "(head_logits)")) if last else None)
        return out if greedy or sample else None

    def _rows(self, part, pos, head):
        """Positions `pos` .. of `part` as one chunk: the n-gram memory's rows of each written for the chunk, the tokens, the
        chunk's program, and `head` after it where it is not None."""
        n = len(part)
        if pos + n > self.max_context:
            raise Refused("positions %d-%d are past the context of %d" % (pos, pos + n - 1, self.max_context))
        self.history[pos:] = list(part)
        if self.ENG is not None:
            rows = self.ENG.chunk_rows(self.history, pos, n)
            for (k, l), at in self.ERWS.items():
                self.m.act_as(k)
                self.m.write(at, _half(rows[l]))
            self.m.act_as(CARD)
        self.m.must("(begin %s (sys__node_array__set ROWS_N 0 %d) 0)"
                    % (" ".join("(sys__node_array__set TOKS %d %d)" % (i, t) for i, t in enumerate(part)), n), "the chunk's tokens")
        return self._run("(begin (rows_chunk %d %d) %s)" % (pos, n, head or "0"), "positions %d-%d" % (pos, pos + n - 1))

    # ── a pipeline stage: this machine's layers, the streams and the last site's mix crossing between stages ─────────
    def forward(self, pos, token=None, x=None, mix=None, head=False):
        """One position through this stage's layers: the token embedded (the first stage) or what the stage before handed on
        (`x`), then the next token (the last stage) or what to hand on. ⭐ WHAT CROSSES BETWEEN STAGES IS ONE BLOB — the
        streams (MULT·H floats), the mix the stage before ended on (MIX_BYTES) and the position's token (8 bytes), which the
        n-gram memory hashes — so a ring (▶ `pipeline`) carries it as any model's residual. `x` and `mix` apart are taken
        too."""
        m, SB = self.m, self.MULT * self.H * 4
        if x is not None and mix is None:
            x, mix, given = x[:SB], x[SB:SB + MIX_BYTES], int(np.frombuffer(x[SB + MIX_BYTES:SB + MIX_BYTES + 8], "<i8")[0])
            token = given if token is None else token
        if token is None:
            raise Refused("position %d: no token, and none came with the streams" % pos)
        self.engram_rows(token, pos)
        if x is None:
            m.grid("(begin (position %d %d) (type streams))" % (token, pos))
        else:
            for k in range(self.P):          # every card of the group starts from the streams and the mix that came across
                m.act_as(k)
                m.write(self.CB[k]["streams"], x)
                m.write(self.CB[k]["mix_in"], mix)
            m.act_as(CARD)
            m.grid("(begin (layers %d) (type streams))" % pos)
        if head:
            return m.must(head if isinstance(head, str) else "(head)", "the head at %d" % pos)
        return m.read(self.STREAMS_AT, SB) + m.read(self.MIXB_AT, MIX_BYTES) + np.array([token], "<i8").tobytes()

    def _run(self, text, what):
        return self.m.grid(text)[1]

    def logits(self):
        return np.frombuffer(self.m.read(self.LOGITS_AT, self.VOCAB * 2), dtype="<f2").astype(np.float32)

    def read(self, name, n):
        """`n` halves of a buffer, by name — the card's, or the CPU's for a `c_` one, read as that worker."""
        if name.startswith("c_"):
            self.m.act_as(self.CPU)
        try:
            return np.frombuffer(self.m.read(self.at[name], n * 2), dtype="<f2").astype(np.float64)
        finally:
            self.m.act_as(CARD)

    # ── a conversation's state: put away and brought back (▶ session.py's `_state_read` / `_state_write`) ─────────
    def state_regions(self, length, where=None, card=0):
        """(address, bytes) on card `card` of everything a conversation of `length` positions has written there that a
        later position reads: each layer's window ring, its rows written so far — and each source layer's compressed
        latents and their index keys to its last whole group, and its partial group's ring whole. A position reads only
        rows it has written, so what lies past them is no one's. `where` (a table of zeros) answers the sizes before the
        boot. Every card of a group holds the same, each in its own slots."""
        ring = (lambda l: 0) if where is not None else (lambda l: self.ring_at[(card, l)])
        cache = (lambda l: 0) if where is not None else (lambda l: self.cache_at[(card, l)])
        out = [(ring(l) + self.RPLAN.at_up_data, min(length, self.W) * self.HD * 2) for l in self.run_layers]
        for s in sorted(self.CACHE_ROWS):
            at, n = cache(s) + self.CPLAN.at_up_data, length // self.RATIO[s]
            out += [(at, n * self.HD * 2), (at + self.KEYS_AT, n * self.ID * 2),
                    (cache(s) + self.CPLAN.at_down_data, 2 * self.RATIO[s] * self.HD * 2)]
        return out

    def state_read(self, length):
        """A conversation's state off the card, as `state_regions` lays it out — card 0's, which every card of a group
        holds the same — and its tokens, which the n-gram memory hashes at the positions after it."""
        self.m.act_as(CARD)
        return ([self.m.read(at, n) for at, n in self.state_regions(length) if n]
                + [np.asarray(self.history[:length], dtype="<i8").tobytes()])

    def state_write(self, length, blobs):
        """A conversation's state, as `state_read` read it, written back — onto every card of a group — and its tokens."""
        *rows, tokens = blobs
        for k in range(self.P):
            self.m.act_as(k)
            for (at, _), blob in zip([r for r in self.state_regions(length, card=k) if r[1]], rows):
                self.m.write(at, blob)
        self.m.act_as(CARD)
        self.history = [int(t) for t in np.frombuffer(tokens, dtype="<i8")]

    def streams(self):
        return np.frombuffer(self.m.read(self.STREAMS_AT, self.MULT * self.H * 4), dtype="<f4").copy()

    def set_streams(self, a):
        self.m.write(self.STREAMS_AT, np.asarray(a, dtype="<f4").reshape(-1).tobytes())

    def mix(self, which):
        """A site's mix as floats: `pre`, `post`, then `comb` (▶ AI_DEEPSEEK_4__MIX_BYTES) — `a` the attention's, `b` the MoE's."""
        at = self.MIXA_AT if which == "a" else self.MIXB_AT
        return np.frombuffer(self.m.read(at, MIX_BYTES), dtype="<f4").astype(np.float64)

    # ── LoRAs: a mask chosen at boot, or none (▶ `_mask_layout`) ──────────────────────────────────────────────
    def load_lora(self, name):
        """The LoRA a conversation runs under: the one booted, which is all this model swaps to. Answers the seconds taken."""
        if name == self.active_lora:
            return 0.0
        if self.b.mask is not None:
            # ⛳ a mask is chosen at boot: its cells are in the tables the procedures were written over
            raise Refused("a mask is chosen at boot, not swapped: boot the folder with lora %r" % (name,))
        raise Refused("DeepSeek-V4.1 takes a LoRA only as a mask, chosen at boot")

    def unload_lora(self):
        if self.active_lora is not None:
            raise Refused("a mask is chosen at boot, not taken off: boot the folder without it")

    def lora_identity(self):
        if self.active_lora is None:
            return None
        return dict(name=self.b.lora["name"], weights=self.b.lora["adapter"]["weights_sha"])

    def _ensure(self, name):
        """The LoRA `name` known, as a conversation adopted from disk asks (▶ `session.adopt`): the booted mask only."""
        if name != self.active_lora:
            raise Refused("this model runs the LoRA %r, chosen at boot, not %r" % (self.active_lora, name))
        self._lora_cache.setdefault(name, dict(ident=self.lora_identity()))
