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
the layers, each ai_glm_5_3's sites over its tables, every CPU socket handed its share of a layer's picks. Either is
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
                 experts_from="auto", experts_dir=None, chunk=256, experts_ram_gb=None, card_experts_gb=0, card_hot=None,
                 card_landing=1, card_line=None, card_exclusive=False, card_free=16, card_duplex_line=1200, card_duplex=None, card_landing_line=3,
                 lend_cache=True, picture_pixels=1 << 22,
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
        # a hand row, as ai_glm_5_3 lays it out: the input, the weights, the picks as words — AI_GLM_5_3__HAND
        self.HAND = ((2 * self.H + 2 * self.TOPK + 7) & ~7) + 8 * self.TOPK
        self.RSCALE, self.LIMIT = float(c["routed_scaling_factor"]), float(c["swiglu_limit"])
        self.max_context = max_context
        # the DSA indexer: past `topk` positions a query attends the top pools of `kpool` and the pool still filling
        self.IH, self.ID, self.KP, self.ITOPK = c["index_n_heads"], c["index_head_dim"], c["index_kpool"], c["index_topk"]
        self.GRID = grid               # a token as one program over both blocks, or a program a part driven from here
        self.FUSED = fused             # the layer as ai_glm_5_3's verbs, a verb a device seam — or as nn's words
        self.run_layers = list(range(self.LAYERS)) if layers is None else list(layers)
        self.kinds = {l: ("f" if l in la["full_attn_layers"] else "d") for l in range(self.LAYERS)}
        self.moe = {l: "mlp.gate.weight" in b.names(l) for l in range(self.LAYERS)}
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
            raise Refused("past %d positions GLM's attention needs the indexer, which ai_glm_5_3's verbs run — not nn's words"
                          % self.ITOPK)
        if sockets > 1 and not fused:
            raise Refused("the experts split over %d sockets are reached by ai_glm_5_3's verbs only" % sockets)
        self.PARTS = sockets
        self.EXPERTS_INT8 = experts_int8   # the routed experts' products in integers on the CPU (nn's int8 gemvs)
        # ⭐ THE CARD'S TIER OF ROUTED EXPERTS (NN-48): `card_experts_gb` of whole experts on the card, which computes the
        #   picks it holds while the CPU computes the rest. An expert the CPU computed earns a slot by nn's qualifier — three
        #   calls, the older of its last two newer than the card's least recent resident's — and is stitched from the
        #   sockets' parts in RAM into it while the card computes (`card_landing` a layer's visit at most). `card_hot`, a
        #   router log (▶ `test/src_glm5_router_log.py`), fills it at boot with each layer's most picked; without one it
        #   starts empty and fills itself.
        self.CARD_GB, self.CARD_HOT, self.CARD_LANDING = float(card_experts_gb or 0), card_hot, int(card_landing)
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
        if self.CARD_EXCLUSIVE and not self.CARD_GB:
            raise Refused("an exclusive tier needs a card tier — card_experts_gb")
        if self.CARD_GB and not fused:
            raise Refused("the card's tier of experts runs in the fused layer — fused=True")
        # ⭐ THE PROGRAM IS lisp/glm5.lisp — the fused layer's: a loop over the layers, each its sites over its tables. The
        #   unfused layer, nn's words a layer, is composed here (`_composed_procedures`), for the router's log
        self.SOURCE = fused
        self.CPUS = [dict(w=CPU + i, dev=(1 + i) if sockets > 1 else 0, part=i) for i in range(sockets)]
        self._plan()
        self.EXPERTS_FROM = experts_file.choose(experts_from, self.cpu_arena, video_ram(folder.config))
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
        #   maps the file instead, the system's page cache the cache. ▶ nn's loader, ai_glm_5_3's experts verbs.
        self.SLOTS = None
        # ⛔ the unfused layer names each expert's planes itself (`x%d`), so every expert must be resident: from the disk
        #   they are a cache, admitted only by ai_glm_5_3's experts verbs, and a plane of one not there answers a refusal
        #   the next verb meets as a type fault (OPCT) at the first MoE layer
        if self.EXPERTS_FROM == "disk" and self.EPLAN is not None and not self.FUSED:
            raise Refused("the unfused layer reads every expert in place, and from the disk they are a cache — "
                          "fused=True, or experts_from 'ram' or 'mapped'")
        if self.EXPERTS_FROM == "disk" and self.EPLAN is not None:
            n_moe = sum(self.moe[l] for l in self.run_layers)
            per = experts_file.cache_bytes(self.cpu_arena, experts_ram_gb) // self.PARTS // self.EPLAN.slot_bytes
            self.SLOTS = min(self.EXPERTS * n_moe, per)
            if self.SLOTS < 3 * 16 + self.TOPK:
                raise Refused("%d experts a worker is too few for a cache of them — a prompt reads two batches of 16 at once"
                              % self.SLOTS)
        # ⭐ A PROMPT AS ROWS, `chunk` positions at a time through each layer (ai_glm_5_3's `…_rows` verbs): a routed expert
        #   is read once for every row of the chunk that picked it. ⛳ `REASONED`: with the experts on the disk the chunk
        #   wants to be large — a chunk's picks reach most of a layer's experts, so each is streamed about once a chunk
        self.CH = max(1, min(chunk, max_context)) if fused else 0
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        # ⭐ PICTURES ON THE CARD (NN-81): the vision tower's weights in a collection of their own and its scratch in another,
        #   sized for pictures of up to `picture_pixels` (0: none, and none of the card's memory for them — the tower then
        #   runs on the host, ▶ vision_glm.py) — where the pack carries the tower, a prompt goes as rows (the only way a
        #   picture goes in) and this machine holds the first layers
        self.PICTURE_PIXELS = int(picture_pixels or 0)
        self.VIS = bool(self.PICTURE_PIXELS and self.CH and self.EPLAN is not None and 0 in self.run_layers
                        and "model.visual.blocks.0.attn.qkv.weight" in g)
        if self.VIS:
            self._vision_plan()

    @staticmethod
    def cn(i, name):
        """A CPU worker's buffer by name: `c_…` on the first, `c1_…` on the second."""
        return ("c_" if i == 0 else "c%d_" % i) + name

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
        plans = {k: NL.dense_plan(b, k[1], first[k]) for k in keys}
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
        self.S_BYTES, self.W_BYTES = self.KH * self.KD * self.KD * 4, 3 * self.KW * 3 * 2
        self.SPLAN = NL.SlotPlan(self.S_BYTES, 0, 0, self.W_BYTES)
        # MLA: the rotated latents in the up half, the latent's expansion decoded to halves in the down
        self.C_BYTES, self.KVB_BYTES = self.max_context * self.LAT * 2, self.QH * (self.NOPE + self.VD) * self.LAT * 2
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
        if self.CARD_GB and moe:
            self.CPLAN = NL.SlotPlan(2 * self.MINTER * g.row_bytes, 2 * self.MINTER * 2, dn.rows * dn.row_bytes, dn.rows * 2)
            self.T_CEXP = self.T_KV + 1
            self.CARD_SLOTS = int(self.CARD_GB * (1 << 30)) // self.CPLAN.slot_bytes
            self.TIER_LEAST = len(moe) * self.CPLAN.slot_bytes             # the least tier it takes: a slot a MoE layer
            if self.CARD_SLOTS < len(moe):
                raise Refused("%.1f GB holds %d experts, fewer than one a MoE layer" % (self.CARD_GB, self.CARD_SLOTS))
            self.collections.append((self.T_CEXP, self.CPLAN, self.CARD_SLOTS))
            # exclusive: each MoE layer's even share of all but the landing slots held on the card, and not in RAM
            self.CARD_PER = 0
            if self.CARD_EXCLUSIVE:
                self.CARD_PER = min(self.EXPERTS - self.TOPK, max(0, self.CARD_SLOTS - self.CARD_FREE) // len(moe))
                self.cpu_arena = self.EPLAN.slot_bytes * (self.EXPERTS - self.CARD_PER) * len(moe) * self.PARTS
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
        return [(self.m.silicon, 0)] + [("x86_avx2", c["dev"]) for c in self.CPUS]

    def device_params(self):
        """The card's section and each CPU worker's: {worker: text}."""
        kinds = "".join(self.kinds[l] for l in range(self.LAYERS))
        big = max(8 << 20, 2 * self.QH * min(self.max_context, self.ITOPK + self.KP) * 4, self.VOCAB * 2)
        bigs = 3
        if self.CH:
            # a chunk's rows: streams, hands, carries, the routed rows (a socket each) and the tokens' rows
            big = max(big, self.CH * self.MULT * self.H * 2, self.CH * self.HAND, self.CH * (256 + 2 * self.H))
            bigs += 5 + len(self.CPUS)
        # ⛳ the rows verbs' `work` buffer is the one buffer of its class, sized past every other big one so no other lands there
        self.WORK = max(self._work_bytes(), big + 4096) if self.CH else 0
        works = 1
        if self.CH and self.CPLAN is not None:
            # the card's tier over a chunk: its rows' share among the big ones, and its `experts_rows` scratch in the work
            # class beside `rwork`, which is then sized for the larger of the two
            n, P, I, H = self.CH, self.CH * self.TOPK, self.MINTER, self.H
            self.KRSCRATCH = 2 * n * H + 16 * P + 2 * P + 4 * P * I + 2 * P * I + 2 * P * I + 4 * P * H + 4096
            self.WORK, works, bigs = max(self.WORK, self.KRSCRATCH), 2, bigs + 1
        card = self._section(big, self.arena, self.collections, kinds, bigs=bigs, mains=2 + (4 + len(self.CPUS) if self.CH else 0),
                             work=self.WORK, works=works)
        if self.LEND:                                  # the links for the slots the tier may be lent (▶ nn's `__lendable`)
            card += "nn__expert__type%d__lendable:%d\n" % (self.T_CEXP, self.LEND["regions"] * self.LEND["most"])
        # the embedding table's RAM class — ⛳ with the other two RAM classes, which nn takes all or none of
        card += ("nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                 "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                 "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % self.TABLE_BYTES)
        # ⛳ a section with no collection is refused, so a run with no MoE layer gives the CPU one of a single slot
        out = {CARD: card}
        moe = sum(self.moe[l] for l in self.run_layers)
        for c in self.CPUS:
            slots = self.SLOTS if self.SLOTS is not None else (self.EXPERTS - getattr(self, "CARD_PER", 0)) * moe
            experts = [(0, self.EPLAN, slots)] if self.EPLAN else [(0, NL.SlotPlan(4096, 0, 0, 4096), 1)]
            cbig, cbigs = 1 << 20, 3
            if self.CH and self.EPLAN:
                P = self.CH * self.TOPK
                cbig = max(cbig, 2 * self.CH * self.H + 16 * P + 2 * P + 8 * P * self.PI + 4 * P * self.H + 64,
                           self.CH * self.HAND)
                cbigs += 3
            out[c["w"]] = self._section(cbig, self.EPLAN.slot_bytes * slots if self.EPLAN else 0, experts,
                                        kinds, many=16, bigs=cbigs, mains=2 + (3 if self.CH else 0))
        return out

    def _work_bytes(self):
        """What the busiest rows verb takes of its `work` buffer for a chunk — ▶ ai_glm_5_3's `…_rows` verbs, whose carving
        this follows, a 256-byte alignment a region."""
        n, H, KW, KD, KH = self.CH, self.H, self.KW, self.KD, self.KH
        QR, LAT, QH = self.QR, self.LAT, self.QH
        common = 24 * n + 256 * n + 3 * 2 * n * H
        kda = common + 2 * n * (3 * KW + 2 * KD + KH) + 2 * n * 3 * KW + 2 * 2 * n * KD + 4 * 2 * n * KW
        # ⛳ and a block of the rows that attend every position before them: its absorbed queries, its answers in the latent,
        #   and its scores over at most the indexer's `topk` positions and a pool — ▶ AI_GLM_5_3__MLA__ROWS_BLOCK
        blk, span = min(32, n), min(self.max_context, self.ITOPK + self.KP)
        mla = (common + 2 * n * (QR + LAT) + 2 * 2 * n * QR + 2 * n * LAT + 2 * n * QH * self.NOPE + 2 * 2 * n * QH * self.VD
               + 2 * n * (self.IH * self.ID + 2 * self.ID + self.IH) + 2 * 2 * blk * QH * LAT + 4 * blk * QH * span)
        mlp = common + 2 * n * 2 * self.INTER + 2 * 2 * n * self.INTER
        route = 24 * n + 3 * 2 * n * H + 2 * n * 2 * self.MINTER + 2 * 2 * n * self.MINTER
        return max(kda, mla, mlp, route) + 32 * 256

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        m.act_as(CARD)
        loader = m.loader()
        self._head_and_embedding(loader)
        if self.VIS:
            self._vision_load(loader)
        self.state_at = {}
        for l in self.run_layers:
            for t in sorted({self.TYPE[self.key(l, n)] for n in self.dense_names(l)}):
                loader.layer_width(l, t, self.WIDTH[t])
            for n in self.dense_names(l):
                k = self.key(l, n)
                loader.load_dense(b, self.PLAN[k], self.TYPE[k], n, l, which=self.WHICH[k])
            t = self.T_STATE if self.kinds[l] == "d" else self.T_KV
            loader.layer_width(l, t, 1)
            self.state_at[l] = loader.reserve(t)
            loader.assign(l, t, 0, self.state_at[l])
        if self.CPLAN is not None:
            self._card_experts(loader)
        if loader.handed() != 0:
            raise Refused("%d card slots were reserved and never published" % loader.handed())
        self._card_buffers()
        # every MLA layer's latent expansion, decoded once into halves beside its cache
        for l in self.run_layers:
            if self.kinds[l] == "f":
                n = "self_attn.kv_b_proj.weight"
                r = b.record(l, n)
                m.must("(begin (nn__turboquant__decode %s %s %s %d %d %d) (type x))"
                       % (self.plane(l, n, "data"), self.plane(l, n, "lut"), self.kv(l, "kvb"), r.rows, r.cols, r.d),
                       "layer %d's latent expansion" % l)
        if self.EPLAN is not None:
            {"disk": self._cache_experts, "mapped": self._map_experts}.get(self.EXPERTS_FROM, self._load_experts)(progress)
            for i, c in enumerate(self.CPUS):
                m.act_as(c["w"])
                self._cpu_buffers(i)
        m.act_as(CARD)
        self._tables()
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
        self.global_ = None                         # what was left of global.pkl, read by the buffers

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
        loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
        self.EMB_AT = m.buffer(self.TABLE_BYTES, "EMBEDDING", ram=True)
        m.write(self.EMB_AT, self.emb_data)
        m.write(self.EMB_AT + self.EMB_DATA_BYTES, self.emb_lut)
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
            lambda n: "(ai_glm_5_3__experts_rows %s exr%d_0 %s %d)" % (self.cn(0, "hands"), next(l for l in self.run_layers if self.moe[l]),
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
        deep — then all of them live, the cache holding nothing yet."""
        L, b = self.LEND, self.CPLAN.slot_bytes
        if L is None:
            return
        starts = [self.state_at[l] + self.KVPLAN.at_up_data for l in self.run_layers if self.kinds[l] == "f"]
        ends = [(s + self.C_BYTES) // 4096 * 4096 for s in starts]
        depth = min((e - (s + 4095) // 4096 * 4096) // b for s, e in zip(starts, ends))
        if depth <= 0:
            self.LEND = None
            return
        self.m.act_as(CARD)
        self._int_array("lend_ends", ends)
        self.m.must("(nn__expert__lend %d %d lend_ends)" % (self.T_CEXP, depth), "the cache's rows lent to the tier")
        L.update(starts=starts, ends=ends, depth=depth, live=0, until=-1)
        self.LENT_GB = depth * len(ends) * b / 2 ** 30
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
            self.m.act_as(CARD)
            got = self.m.must("(nn__expert__lent %d %d)" % (self.T_CEXP, live), "the tier's lent slots")
            if got != live:
                raise Refused("the tier holds %d lent slots, not %d" % (got, live))
            L["live"] = live
        free = min(e - deep * b - s for s, e in zip(L["starts"], L["ends"]))
        L["until"] = min(self.max_context, free // L["row"])

    def lent(self):
        """The cache's rows lent to the tier now: (slots live, GB), or None where the tier is lent none."""
        L = self.LEND
        return None if L is None else (L["live"], L["live"] * self.CPLAN.slot_bytes / 2 ** 30)

    def _card_experts(self, loader):
        """The card's tier, stood at boot: each MoE layer's band of every expert — and, given a router log, its most picked
        ones written whole into the card's slots, an even share of them a layer."""
        m, b = self.m, self.b
        moe = [l for l in self.run_layers if self.moe[l]]
        for l in moe:
            loader.layer_width(l, self.T_CEXP, self.EXPERTS)
        self.CARD_HELD = {}
        if not self.CARD_HOT and not self.CARD_EXCLUSIVE:
            return
        log = np.load(self.CARD_HOT) if self.CARD_HOT else None
        logged = [int(l) for l in log["layers"]] if log is not None else []
        per = self.CARD_PER if self.CARD_EXCLUSIVE else self.CARD_SLOTS // len(moe)
        for l in moe:
            if log is None:
                hot = list(range(per))        # exclusive with no router log: the first of each layer, until it learns
            else:
                if l not in logged:
                    raise Refused("the router log has no picks for layer %d" % l)
                counts = np.bincount(log["picks"][:, logged.index(l), :].reshape(-1).astype(np.int64), minlength=self.EXPERTS)
                hot = [int(x) for x in np.argsort(-counts, kind="stable")[:per]]
            view = np.frombuffer(b.blob(l), dtype=np.uint8)
            for x in hot:
                at = loader.reserve(self.T_CEXP)
                for into, piece in self._slot_pieces(view, l, x, 0, whole=True):
                    m.write_from(at + into, piece.ctypes.data, piece.nbytes)
                loader.assign(l, self.T_CEXP, x, at)
            self.CARD_HELD[l] = hot

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
            if progress:
                progress(i + 1, len(moe))
        for c in self.CPUS:
            m.act_as(c["w"])
            if loaders[c["w"]].handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loaders[c["w"]].handed())

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
        m.buffer(1 << 19, "gscratch")                          # ai_glm_5_3's scratch, its first bytes the MLP site's carry
        self.at["hand"] = m.buffer(2 * (H + self.TOPK), "hand")
        # ⛳ a query attends at most `topk` positions and the filling pool's, so the scores need no more than that
        self.ATTEND = min(self.max_context, self.ITOPK + self.KP)
        m.buffer(2 * self.QH * self.ATTEND * 4, "scores")
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

    def _cpu_buffers(self, i):
        m, H, c = self.m, self.H, lambda name: self.cn(i, name)
        m.buffer(1024, c("signs"), _half(self.signs[:512]))
        for nm, n in (("hr", H), ("gu", 2 * self.MINTER), ("act", self.MINTER), ("actr", self.MINTER), ("e", H), ("routed", H)):
            self.at[c(nm)] = m.buffer(n * 2, c(nm))
        m.buffer(64, c("w"))
        m.buffer(2 * (H + self.TOPK), c("hand"))
        m.buffer(2 * H, c("zero"), _half(np.zeros(H)))
        m.buffer(8 * self.TOPK * self.MINTER, c("scratch"))
        if self.CH:
            # a chunk's hands, its routed rows, and `experts_rows`' scratch: the inputs, the pairs and weights, the gate-and-up,
            # activation and rotated rows of every pick, and a fp32 sum for every pick — ▶ ai_glm_5_3__experts_rows
            CH, P, I = self.CH, self.CH * self.TOPK, self.PI
            self.at[c("hands")] = m.buffer(CH * self.HAND, c("hands"))
            m.buffer(CH * H * 2, c("rrows"))
            m.buffer(2 * CH * H + 16 * P + 2 * P + 8 * P * I + 4 * P * H + 64, c("rscratch"))

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
        """ai_glm_5_3's plane tables: a layer's attention site (`at`), its MLP site (`ml` dense, `mo` with experts), and on
        the CPU its experts' (`ex`, and `exr` over a prompt's chunk). ▶ the package's `contracts/objects/layer.cuh` for what
        each cell is. Made at every boot; the programs that name them are `_procedures`'."""
        if self.FUSED:
            self._fused_tables()
        if self.CH and self.EPLAN is not None:
            self._rows_tables()

    def _fused_tables(self):
        m, b, H, s = self.m, self.b, self.H, "self_attn."
        pair = lambda l, n: [self.plane(l, n, "data"), self.plane(l, n, "lut")]
        data = lambda l, n: self.plane(l, n, "data")
        d = lambda l, n: b.record(l, n).d
        hc = lambda l, site: [data(l, "hc_%s_%s" % (site, x)) for x in ("fn", "base", "scale")]
        fed = {}                       # the tier's layers fed from the sockets' memory, by layer
        head = [H, self.MULT, self.ITERS]
        for l in self.run_layers:
            if self.kinds[l] == "d":
                mats = [s + n + ".weight" for n in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj",
                                                    "f_b_proj", "g_b_proj", "o_proj")]
                m.table("at%d" % l, hc(l, "attn") + [data(l, "input_layernorm.weight")] + [c for n in mats for c in pair(l, n)]
                        + [data(l, s + "%s_conv1d.weight" % c) for c in "qkv"]
                        + [data(l, s + "A_log"), data(l, s + "dt_bias"), data(l, s + "o_norm.weight"),
                           self.state(l, "S"), self.state(l, "W"), "signs", "gscratch"]
                        + head + [self.KH, self.KD] + [d(l, n) for n in mats] + [self.EPS, self.HC_EPS, self.LOWER])
            else:
                mats = [s + n + ".weight" for n in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")]
                ix = [s + "indexer." + n for n in ("wq_b.weight", "wk.weight", "weights_proj.weight", "index_kpool_compress_gate")]
                m.table("at%d" % l, hc(l, "attn") + [data(l, "input_layernorm.weight")] + [c for n in mats for c in pair(l, n)]
                        + [data(l, s + "q_a_layernorm.weight"), data(l, s + "kv_a_layernorm.weight"), self.kv(l, "kvb"),
                           self.kv(l, "c"), "scores", "signs", "gscratch"]
                        + head + [self.QH, self.QR, self.LAT, self.NOPE, self.VD] + [d(l, n) for n in mats]
                        + [self.EPS, self.HC_EPS]
                        # the indexer — ▶ the package's AI_GLM_5_3__IDX__ cells
                        + [c for n in ix for c in pair(l, n)]
                        + [data(l, s + "indexer.k_norm.weight"), data(l, s + "indexer.k_norm.bias"), "ape%d" % l,
                           self.kv(l, "ikeys"), self.kv(l, "igates"), self.kv(l, "ipool"), "iscores", "iindex", "igather"]
                        + [self.IH, self.ID, self.KP, self.ITOPK] + [d(l, n) for n in ix])
            if not self.moe[l]:
                mats = ["mlp.%s_proj.weight" % n for n in ("gate", "up", "down")]
                m.table("ml%d" % l, hc(l, "ffn") + [data(l, "post_attention_layernorm.weight")] + [c for n in mats for c in pair(l, n)]
                        + ["signs", "gscratch"] + head + [self.INTER] + [d(l, n) for n in mats]
                        + [self.EPS, self.HC_EPS, self.LIMIT])
                continue
            if self.CPLAN is not None:
                cp = self.CPLAN
                # the tier's layer, fed from the sockets' memory where every expert is there (▶ `nn__expert_tier`)
                if all((l, c["part"]) in self.SRC for c in self.CPUS):
                    self._int_array("src%d" % l, [self.SRC[(l, c["part"])][x] for x in range(self.EXPERTS) for c in self.CPUS])
                    fed[l] = "src%d" % l
                cells = ["signs", "zero_h", "kscratch", l, self.T_CEXP, cp.at_up_lut - cp.at_up_data,
                         cp.at_down_data - cp.at_up_data, cp.at_down_lut - cp.at_up_data, H, self.MINTER,
                         self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS, 0, self.LIMIT, 0]
                m.table("exk%d" % l, cells)
                if self.CH:
                    m.table("exkr%d" % l, ["signs", "zero_h", "krscratch"] + cells[3:])   # a prompt chunk's, its own scratch
            mats = ["mlp.shared_experts.%s_proj.weight" % n for n in ("gate", "up", "down")]
            m.table("mo%d" % l, hc(l, "ffn") + [data(l, "post_attention_layernorm.weight"), data(l, "mlp.gate.weight"),
                                                data(l, "mlp.gate.e_score_correction_bias")]
                    + [c for n in mats for c in pair(l, n)] + ["signs", "gscratch"]
                    + head + [self.EXPERTS, self.TOPK, self.MINTER] + [d(l, n) for n in mats]
                    + [self.EPS, self.HC_EPS, self.RSCALE, self.LIMIT])
        if self.CPLAN is not None:
            # ⭐ THE CARD'S TIER, ONE FOR THE CARD: how one of its experts is cut, where the sockets keep every expert's parts
            #   a layer at a time, and the binding `nn__expert_tier` the programs hand the verbs (▶ `expert_tier.bind`)
            exclusive = None
            if self.CARD_EXCLUSIVE:
                if set(fed) != {l for l in self.run_layers if self.moe[l]}:
                    raise Refused("an exclusive tier needs every MoE layer's sources")
                self.DUPLEX = expert_tier.duplex(m, CARD, self.CPLAN.slot_bytes, self.CARD_DUPLEX_LINE, self.CARD_DUPLEX)
                exclusive = dict(part_worker=self.CPUS[0]["w"], part_type=0)
            expert_tier.tier(m, "tier_card", self.T_CEXP, self.EXPERTS, self.TOPK, H, self.MINTER, self.CPLAN, self.LAYERS,
                             fed, self.PARTS, self.EPLAN, self.GATE_ROW, self.DOWN_ROW, self.CARD_LANDING, exclusive,
                             self.CARD_LANDING_LINE)
            expert_tier.bind(m, {CARD: "tier_card"}, len(self.workers()))
        if self.EPLAN is not None:
            p = self.EPLAN
            for i, cpu in enumerate(self.CPUS):
                m.act_as(cpu["w"])
                c = lambda name: self.cn(i, name)
                for l in self.run_layers:
                    if self.moe[l]:
                        m.table("ex%d_%d" % (l, i), [c("signs"), c("zero"), c("scratch"), l, 0, p.at_up_lut - p.at_up_data,
                                                     p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data, H, self.PI,
                                                     self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS,
                                                     1 if self.EXPERTS_INT8 else 0, self.LIMIT] + self._backing(cpu["w"], l))
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
                        m.picture("xf%d_%d" % (l, i), "(begin (nn__buffer__copy hand %d %s %d) (ai_glm_5_3__experts %s picks ex%d_%d %s))"
                                  % (CARD, c("hand"), 2 * (H + self.TOPK), c("hand"), l, i, c("routed")))
            m.act_as(CARD)
        layers = []
        for l in self.run_layers:
            layers.append("(ai_glm_5_3__kda streams at%d)" % l if self.kinds[l] == "d" else "(ai_glm_5_3__mla streams at%d pos)" % l)
            if self.moe[l]:
                # every socket handed its share of the picks, the shared expert on the card meanwhile, the shares summed
                ws = [(i, c["w"]) for i, c in enumerate(self.CPUS)]
                layers.append("(ai_glm_5_3__route streams mo%d hand picks) " % l
                              + ("(ai_glm_5_3__experts hand picks exk%d routed_k %s) " % (l, expert_tier.BINDING) if self.CPLAN is not None else "")
                              + " ".join("(sys__compute %d names xf%d_%d)" % (w, l, i) for i, w in ws)
                              + " (ai_glm_5_3__shared hand mo%d) " % l
                              + " ".join("(sys__result %d)" % w for _, w in ws) + " "
                              # the CPUs done: an exclusive tier's writes back to RAM, held on half duplex, go now
                              + ("(nn__expert_tier__written %s) " % expert_tier.BINDING if self.CPLAN is not None else "")
                              + " ".join("(nn__buffer__copy %s %d %s %d)" % (self.cn(i, "routed"), w, "routed" if i == 0 else "routed%d" % i,
                                                                            2 * H) for i, w in ws) + " "
                              + " ".join("(nn__vector__add routed routed%d routed %d)" % (i, H) for i, _ in ws if i > 0)
                              + (" (nn__vector__add routed routed_k routed %d)" % H if self.CPLAN is not None else "")
                              + " (ai_glm_5_3__close streams mo%d routed)" % l)
            else:
                layers.append("(ai_glm_5_3__mlp streams ml%d)" % l)
        return " ".join(x for x in layers if x)

    # ── ③ the names the program is written against, and the program ──────────────────────────────────────────
    def _bindings(self):
        """What `lisp/glm5.lisp` reads: the shape, whether the card holds a tier, a layer's tables gathered into arrays
        indexed by its place in the layers this machine runs, and each socket's worker, buffers and experts tables."""
        m, H, run, K = self.m, self.H, self.run_layers, self.TOPK
        head = lambda at, size: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_HEAD, at, size)
        tier = self.CPLAN is not None
        m.bind_values(dict(HIDDEN=H, HIDDEN_BYTES=2 * H, MULT=self.MULT, MEAN=1.0 / self.MULT, VOCAB=self.VOCAB, EPS=self.EPS,
                           RUN=len(run), EMB_HALVES=self.EMB_ROW_BYTES // 2, EMB_SCALES=self.EMB_DATA_BYTES // 2,
                           EMB_BITS=self.EMB_BITS, STOPS=0, SEEN=sampling.SEEN,
                           HEAD_CODES=head(self.HPLAN.at_up_data, self.HPLAN.up_data),
                           HEAD_SCALES=head(self.HPLAN.at_down_lut, self.HPLAN.down_lut), HEAD_BITS=self.HEAD_BITS,
                           HAND=self.HAND, HAND_PICKS_AT=self.HAND - 8 * K, HAND_BYTES=2 * (H + K),
                           ROWS_HANDS_BYTES=self.CH * self.HAND, ROWS_BYTES=2 * H * self.CH, ROWS_HALVES=H * self.CH,
                           CARD_TIER=tier, CARD=CARD, SOCKETS=len(self.CPUS)))
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
        socket = range(len(self.CPUS))
        m.array("SOCKET", [c["w"] for c in self.CPUS])
        m.array("ROUTED_INTO", ["routed" if i == 0 else "routed%d" % i for i in socket])
        m.array("C_HAND", [self.cn(i, "hand") for i in socket])
        m.array("C_ROUTED", [self.cn(i, "routed") for i in socket])
        for i in socket:
            m.array("EX_%d" % i, [("ex%d_" + str(i)) % l if self.moe[l] else 0 for l in run])
        m.array("EX", ["EX_%d" % i for i in socket])
        if self.CH:
            m.array("RROWS_INTO", ["rrows" if i == 0 else "rrows%d" % i for i in socket])
            m.array("C_HANDS", [self.cn(i, "hands") for i in socket])
            m.array("C_RROWS", [self.cn(i, "rrows") for i in socket])
            for i in socket:
                m.array("EXR_%d" % i, [("exr%d_" + str(i)) % l if self.moe[l] else 0 for l in run])
            m.array("EXR", ["EXR_%d" % i for i in socket])
        m.node_array("HERE", 1)             # the layer a socket's picture runs (▶ lisp/glm5_cpu.lisp)

    def _procedures(self):
        """The program: `lisp/glm5.lisp`, the sampler's and the sockets' pictures — or, unfused, `_composed_procedures`.
        Read from `programs.lisp` instead by the boots that have the same key (▶ `Machine.define_programs`)."""
        if not self.SOURCE:
            return self._composed_procedures()
        self.m.load_source(self.folder.program("glm5"))
        self.m.load_source(self.folder.program("sampling"))
        self.m.load_source(self.folder.program("generate"))
        if self.EPLAN is not None:
            self.m.load_source(self.folder.program("glm5_cpu"))

    def _composed_procedures(self):
        """The unfused layer's procedures, nn's words a layer — and the fused layer's for a run that takes directions out
        or captures the streams — composed here."""
        m, H = self.m, self.H
        # the token's row of the embedding table, un-rotated (Rᵀu = S ⊙ H·u), into e — and into every stream
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
                                                  1 if self.EXPERTS_INT8 else 0, self.LIMIT] + self._backing(cpu["w"], l))
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
                              "(ai_glm_5_3__experts_rows %s exr%d_%d %s (sys__node_array__get nrows 0)))"
                              % (CARD, c("hands"), self.CH * self.HAND, c("hands"), l, i, c("rrows")))
        m.act_as(CARD)
        ws = [(i, cpu["w"]) for i, cpu in enumerate(self.CPUS)]
        for l in self.run_layers:
            body = ("(ai_glm_5_3__kda_rows srows at%d n rwork)" % l if self.kinds[l] == "d"
                    else "(ai_glm_5_3__mla_rows srows at%d first n rwork)" % l)
            if self.moe[l]:
                body += (" (ai_glm_5_3__route_rows srows mo%d hands carries n rwork) " % l
                         # the card's tier told of the chunk's picks, so the prompt warms it (NN-48)
                         + ("(nn__expert_tier__note hands %s %d n %d %d %s) " % (expert_tier.BINDING, l, self.HAND, self.HAND - 8 * self.TOPK,
                                                                       expert_tier.THRESHOLD)
                            if self.CPLAN is not None else "")
                         + " ".join("(sys__compute %d names xr%d_%d)" % (w, l, i) for i, w in ws) + " "
                         # the card's share computed while the CPUs compute theirs
                         + ("(ai_glm_5_3__experts_rows hands exkr%d rrows_k n %s) " % (l, expert_tier.BINDING) if self.CPLAN is not None else "")
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
                         + " (ai_glm_5_3__close_rows srows mo%d rrows carries n rwork)" % l)
            else:
                body += " (ai_glm_5_3__mlp_rows srows ml%d n rwork)" % l
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
        self.m.must("(begin %s (type streams))" % " ".join(zeros), "the reset")
        self._room(0, exact=True)                       # a conversation from nothing: every lent slot the tier's again

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
        m.must("(embed %d)" % token, "the embedding at %d" % pos)
        for l in self.run_layers:
            self.layer(l, pos)
        got = m.must(head, "the head at %d" % pos)
        return got if greedy or sample else None

    def generate(self, token, pos, n, stop=(), sampler=None, lag=0):
        """`Generates.generate`, the cache's rows it may write withheld from the tier first."""
        self._room(pos + min(n, GENERATE_MOST) + 1)
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
        # ▶ ai_glm_5_3's `vision_block` and `vision_tail`: their scratch, one after the other in the same room
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
        words = (["(ai_glm_5_3__vision_embed %s GVE %d %s)" % (P["pix"], n, P["x"])]
                 + ["(ai_glm_5_3__vision_block %s %s GVB%d %d)" % (P["x"], P["rc"], i, n) for i in range(depth)]
                 + (["(ai_glm_5_3__vision_tail %s GVT %d %s)" % (P["x"], n, P["out"])] if blocks is None else []))
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
                out = self.step(tok, first + i, greedy, sample if i == len(tokens) - 1 else None)
            return out
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        out = None
        for at in range(0, len(tokens), self.CH):
            chunk = tokens[at:at + self.CH]
            mine = {i - at: r for i, r in pictures.items() if at <= i < at + len(chunk)}
            out = self.forward_rows(first + at, tokens=chunk, head=at + self.CH >= len(tokens), greedy=greedy, sample=sample,
                                    rows=mine)
        return out

    # ── a pipeline stage (▶ `pipeline.py`): this machine's layers, the streams crossing between stages ─────────────
    def forward(self, pos, token=None, x=None, head=False):
        """One position through this stage's layers: the token embedded (the first stage) or the streams that came across
        (`x`, MULT·H halves as bytes) set; then the next token (the last stage) or the streams to hand on."""
        m = self.m
        self._room(pos + 1)
        if token is not None:
            m.must("(begin (embed %d) (type streams))" % token, "the embedding at %d" % pos)
        else:
            m.write(self.STREAMS_AT, x)
        m.grid("(layers_f %d)" % pos)
        if head:                                     # the greedy head, or the head expression it is (`sampling.head`)
            return m.must(head if isinstance(head, str) else "(head)", "the head at %d" % pos)
        return m.read(self.STREAMS_AT, self.MULT * self.H * 2)

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
                m.must("(embed_rows %d)" % n, "the chunk's embeddings")
            else:
                # ⛳ the embeddings in programs of their own, a hundred rows each: one program is one form, and a form
                #   must fit in one of the heap's chunks
                for e0 in range(0, n, 128):
                    m.must("(begin %s (type srows))" % " ".join(
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

    # ── LoRAs: none yet for this family, so a conversation runs on the base alone ────────────────────────────
    active_lora = None

    def load_lora(self, name):
        if name is not None:
            raise Refused("GLM 5.3 Flash has no LoRA overlays in this runtime yet")
        return 0.0

    def unload_lora(self):
        pass

    def lora_identity(self):
        return None

    def state_write(self, length, blobs):
        """A conversation's state, as `state_regions` lays it out, written back — its rows withheld from the tier first."""
        self._room(length, exact=True)
        for (at, _), blob in zip([r for r in self.state_regions(length) if r[1]], blobs):
            self.m.write(at, blob)

    def state_regions(self, length, where=None):
        """(address, bytes) of everything a conversation of `length` positions has written: each KDA layer's state
        and window whole, each MLA layer's latents `0..length` — and then each MLA layer's indexer caches, the ring of keys
        and gates and the pooled keys to its last pool, which choose what a position past `index_topk` attends. They come
        last, so a conversation saved without them still finds its latents and states where they were. `where`: each
        layer's state's address — the load's by default; a table of zeros answers the sizes before the boot."""
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
