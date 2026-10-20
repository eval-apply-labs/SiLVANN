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

    def __init__(self, folder, machine, max_context=512, layers=None, experts_int8=True, engram_from=None,
                 experts_from="ram", experts_in_ram=0, experts_dir=None, socket=None, card=0, cards=None,
                 card_experts=0):
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
        # ⭐ THE CARDS' SHARE OF THE ROUTED EXPERTS — ⚖ 2026-10-19: *"each card keeps their half of the expert they are
        #   computing, and at the end we add up the two partials"*. Experts 0 .. card_experts-1 of every layer on the cards,
        #   each card its part of their inner width (the shared expert's cut: whole rotation blocks), the rest in the CPU's
        #   memory: a pick on the cards is a partial on each, added in the group's reduce with the CPU's sum of the others.
        #   ⛳ A FIXED SHARE, NOT A CACHE: the router's bias keeps the experts' load level, so any card_experts of them take
        #   about that share of the picks — and the cards stay in step without deciding anything between them.
        self.NK = int(card_experts)
        if not 0 <= self.NK <= self.EXPERTS:
            raise Refused("card_experts is 0 .. %d, not %d" % (self.EXPERTS, self.NK))
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        self._plan()

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
        # each card's slot of an expert: its part of the gate and up rows, their scales, its columns of every down row, every
        # down row's scale
        if any(dn.row_bytes * ip % self.MINTER for ip in self.I_OF):
            raise Refused("a card's part of the down rows does not end on a byte")
        self.KPLANS = [NL.SlotPlan(2 * ip * gr.row_bytes, 2 * ip * 2, dn.rows * (dn.row_bytes * ip // self.MINTER), dn.rows * 2)
                       for ip in self.I_OF]
        self.T_CEXP = self.T_CACHE + 1
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
        out = {}
        for k in range(self.P):
            out[k] = self._section(big, self.ARENAS[k], self.COLLECTIONS[k], kinds)
            out[k] += ("nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                       "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                       "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % (self.TABLE_BYTES if k == 0 else 256))
        for c in self.CPUS:
            out[c["w"]] = self._section(1 << 20, self.cpu_arena, [(0, self.EPLAN, self.cpu_slots)],
                                        kinds, many=16)
        return out

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        self.ring_at, self.CB = {}, {}
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
                    loader.assign(l, self.T_CACHE, 0, loader.reserve(self.T_CACHE))
            if self.NK:
                self._load_card_experts(k, loader)
            if loader.handed() != 0:
                raise Refused("card %d: %d slots were reserved and never published" % (k, loader.handed()))
            self._card_buffers(k)
        self.head_data = self.head_lut = self.emb_data = self.emb_lut = None
        self._load_experts(progress)
        m.act_as(self.CPU)
        self._cpu_buffers(0)
        m.bind_values(dict(VOCAB=self.VOCAB, STOPS=0, SEEN=sampling.SEEN, GROUP=self.GROUP))
        for k in range(self.P):
            m.act_as(k)
            self._tables(k)
        if self.P > 1:
            # ⭐ THE GROUP: each card joins it on its own worker, card 0 first (▶ nn's `nn__partials__join`)
            m.node_array("TP_POS", 1)
            for k in range(self.P):
                m.act_as(k)
                m.must("(nn__partials__join %d %d %d %d)" % (self.GROUP, k, self.P, min(4, self.P)),
                       "card %d joining the group" % k, "sys__value_true")
        m.act_as(CARD)
        m.define_programs(self._procedures)
        self.global_ = None

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
        cb["streams"] = m.buffer(self.MULT * H * 2, n("streams"))
        for nm in ("u", "hu", "e", "hm", "hn", "hnr", "routed"):
            cb[nm] = m.buffer(H * 2, n(nm))
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
            # first starts a position's streams from zeros, which card 0's embedding is added to (`_procedures`)
            cb["tsum"] = m.buffer(H * 2, n("tsum"))
            if k > 0:
                m.buffer(self.MULT * H * 2, n("zstreams"), _half(np.zeros(self.MULT * H)))
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
                    m.buffer(4 * (self.TOPK_C + 1), n("isel%d" % l))
        m.buffer(self.QH * (self.W + self.TOPK_C) * 4, n("scores"))
        QW, ON, I = self.QH * self.HD, self.G * self.OR, self.MINTER
        m.buffer(2 * (4 * H + 3 * self.QR + self.HD + 3 * QW + 2 * ON + 4 * self.HD + self.ID + self.IH * self.ID + self.IH) + 256,
                 n("ascratch"))
        m.buffer(2 * (3 * H + self.EXPERTS) + 8 * I + 256, n("mscratch"))
        cb["hand"] = m.buffer(2 * (H + self.TOPK), n("hand"))
        if self.NK:
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
                    + [self.EPS, self.HC_EPS] + self._compressed(l, k))
            mats = ["mlp.shared_experts.%s_proj.weight" % x for x in ("gate", "up", "down")]
            m.table(n("mo%d" % l), hc(l, "ffn") + [data(l, "post_attention_layernorm.weight"), n("mixb"), n("mixa"),
                                                   data(l, "mlp.gate.weight"), data(l, "mlp.gate.e_score_correction_bias")]
                    + [c for x in mats for c in pair(l, x)] + [n("signs"), n("mscratch")]
                    + head + [self.EXPERTS, self.TOPK, self.I_OF[k]] + [d(l, x) for x in mats]
                    + [self.EPS, self.HC_EPS, self.RSCALE, self.LIMIT])
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
        if k != 0:
            return
        p = self.EPLAN
        m.act_as(self.CPU)
        c = lambda name: self.cn(0, name)
        for l in self.run_layers:
            m.table("ex%d_0" % l, [c("signs"), c("zero"), c("scratch"), l, 0, p.at_up_lut - p.at_up_data,
                                   p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.MINTER,
                                   self.EXPERTS, self.TOPK, self.E_D, self.NK, self.EXPERTS - self.NK, 1 if self.EXPERTS_INT8 else 0,
                                   self.LIMIT])
        m.act_as(k)

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
        add = " (nn__vector__add routed routed_k routed %d)" % H if self.NK else ""
        return (engram + "(ai_deepseek_4__attention streams at%d pos%s) (ai_deepseek_4__route streams mo%d hand picks) "
                "(sys__compute %d names xf%d_0) %s(ai_deepseek_4__shared hand mo%d) (sys__result %d) "
                "(nn__buffer__copy c_routed %d routed %d)%s (ai_deepseek_4__close streams mo%d routed%s)"
                % (l, tp, l, CPU, l, mine, l, CPU, CPU, 2 * H, add, l, tp))

    def _tp_layers(self, k):
        """Every layer at the position in TP_POS on card `k` past the first, by its own names: the same sites as card 0's,
        its partials added in the group's reduces, the routed experts none of its share (card 0 adds the CPU's sum)."""
        n, pos = (lambda name: self.sfx(name, k)), "(nth TP_POS 0)"
        out = []
        for l in self.run_layers:
            if l in self.ENGRAM:
                out.append("(ai_deepseek_4__engram %s %s)" % (n("streams"), n("en%d" % l)))
            mine = ("(ai_deepseek_4__experts %s %s %s %s) " % (n("hand"), n("picks"), n("ek%d" % l), n("routed_k"))
                    if self.NK else "")
            out.append("(ai_deepseek_4__attention %s %s %s GROUP %s) (ai_deepseek_4__route %s %s %s %s) %s(ai_deepseek_4__shared %s %s) "
                       "(ai_deepseek_4__close %s %s %s GROUP %s)"
                       % (n("streams"), n("at%d" % l), pos, n("tsum"), n("streams"), n("mo%d" % l), n("hand"), n("picks"), mine,
                          n("hand"), n("mo%d" % l), n("streams"), n("mo%d" % l), n("routed_k") if self.NK else n("zero_h"), n("tsum")))
        return " ".join(out)

    def _procedures(self):
        m, H = self.m, self.H
        # the token's row of the embedding table, un-rotated (Rᵀu = S ⊙ H·u), into e — and into every stream
        m.defun("(defun (embed_e tok) (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok %d) %d) "
                "(nn__vector__range EMBEDDING (+ %d tok) 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                "(nn__vector__pointwise_mul hu signs_row e %d)))"
                % (self.EMB_ROW_BYTES // 2, self.EMB_ROW_BYTES // 2, self.EMB_DATA_BYTES // 2, H, self.EMB_BITS, H, H))
        m.defun("(defun (embed tok) (begin (embed_e tok) %s (type e)))"
                % " ".join("(nn__vector__add e zero_h %s %d)" % (self.rng("streams", j * H, H), H) for j in range(self.MULT)))
        # the head: the streams collapsed by the last site's mix (the model's `hc_pre` before its norm), the norm, the logits
        logits = ("(nn__hyper__collapse streams mixb hm %d %d) (nn__rmsnorm__apply hm fnw hn %d %r) "
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
        # ⭐ EVERY OTHER CARD OF THE GROUP: its layers, a picture — from card 0's embedding (`tp_first`: its streams zeros, the
        #   group's reduce of them with card 0's the embedding on every card) or from the streams already there (`tp_given`)
        MH = self.MULT * H
        for k in range(1, self.P):
            m.act_as(k)
            n = lambda name: self.sfx(name, k)
            m.picture(n("tp_first"), "(begin (nn__partials__reduce GROUP %s %s %d) %s (type %s))"
                      % (n("zstreams"), n("streams"), MH, self._tp_layers(k), n("streams")))
            m.picture(n("tp_given"), "(begin %s (type %s))" % (self._tp_layers(k), n("streams")))
        m.act_as(CARD)
        m.view("names")
        for l in self.run_layers:
            m.defun("(defun (lay%d pos) (begin %s (type streams)))" % (l, self._layer(l)))
        lays = " ".join("(lay%d pos)" % l for l in self.run_layers)
        if self.P == 1:
            m.defun("(defun (layers pos) (begin %s (type streams)))" % lays)
            m.defun("(defun (position tok pos) (begin (embed tok) %s (type streams)))" % lays)
            return
        start = lambda pic: " ".join("(sys__compute %d names %s)" % (k, self.sfx(pic, k)) for k in range(1, self.P))
        wait = " ".join("(sys__result %d)" % k for k in range(1, self.P))
        m.defun("(defun (layers pos) (begin (sys__node_array__set TP_POS 0 pos) %s %s %s (type streams)))"
                % (start("tp_given"), lays, wait))
        m.defun("(defun (position tok pos) (begin (sys__node_array__set TP_POS 0 pos) %s (embed tok) "
                "(nn__partials__reduce GROUP streams streams %d) %s %s (type streams)))" % (start("tp_first"), MH, lays, wait))

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
        out = None
        for i, tok in enumerate(tokens):
            out = self.step(tok, first + i, greedy, sample if i == len(tokens) - 1 else None)
        return out

    # ── a pipeline stage: this machine's layers, the streams and the last site's mix crossing between stages ─────────
    def forward(self, pos, token=None, x=None, mix=None, head=False):
        """One position through this stage's layers: the token embedded (the first stage) or what the stage before handed on
        (`x`), then the next token (the last stage) or what to hand on. ⭐ WHAT CROSSES BETWEEN STAGES IS ONE BLOB — the
        streams (MULT·H halves), the mix the stage before ended on (MIX_BYTES) and the position's token (8 bytes), which the
        n-gram memory hashes — so a ring (▶ `pipeline`) carries it as any model's residual. `x` and `mix` apart are taken
        too."""
        m, SB = self.m, self.MULT * self.H * 2
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

    def streams(self):
        return np.frombuffer(self.m.read(self.STREAMS_AT, self.MULT * self.H * 2), dtype="<f2").astype(np.float32)

    def set_streams(self, a):
        self.m.write(self.STREAMS_AT, _half(np.asarray(a).reshape(-1)))

    def mix(self, which):
        """A site's mix as floats: `pre`, `post`, then `comb` (▶ AI_DEEPSEEK_4__MIX_BYTES) — `a` the attention's, `b` the MoE's."""
        at = self.MIXA_AT if which == "a" else self.MIXB_AT
        return np.frombuffer(self.m.read(at, MIX_BYTES), dtype="<f4").astype(np.float64)

    active_lora = None

    def load_lora(self, name):
        if name is not None:
            raise Refused("DeepSeek-V4.1 has no LoRA overlays in this runtime yet")
        return 0.0

    def unload_lora(self):
        pass

    def lora_identity(self):
        return None
