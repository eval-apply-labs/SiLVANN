"""GLM 5.3 FLASH — four residual streams, Kimi Delta Attention and latent attention, 288 experts — on a card and the CPU.

⚖ *"at 151gb the point is that experts stay on ram and fixed matrices go on the card"* · *"it is the cpu that computes
the ram experts"*. So the machine has two workers: the card (worker 0) holds every matrix a token always uses — the
mixers, the dense MLPs, the shared experts, the routers, the hyper-connections, the embedding and the head — and the
CPU (worker 1, and worker 2 on a second socket) holds the routed experts in its own memory and runs the eight a token picks. A layer is three short
programs:
```
  card   a{l}   the attention site: hc pre, norm, the mixer, hc post · the MLP site's hc pre and norm, and then either
                the dense MLP and hc post, or the router's picks and weights and the shared expert
  CPU    x{l}   the input and the weights copied over, the eight picked experts, their weighted sum
  card   c{l}   the sum copied back, added to the shared expert's, hc post
```
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
import os
import pickle

import numpy as np

from . import experts_file, expert_tier
from .model_folder import NL, Refused

ARCHITECTURES = ("glm5_next_text",)
CARD, CPU = 0, 1
EXPERT = "mlp.experts.%d.%s_proj.weight"


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


class Glm5:
    def __init__(self, folder, machine, max_context=2048, layers=None, grid=True, fused=True, sockets=None, experts_int8=True,
                 experts_from="auto", experts_dir=None, chunk=256, experts_ram_gb=None, card_experts_gb=0, card_hot=None,
                 card_landing=2, card_line=None):
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
        if self.CARD_GB and not fused:
            raise Refused("the card's tier of experts runs in the fused layer — fused=True")
        self.CPUS = [dict(w=CPU + i, dev=(1 + i) if sockets > 1 else 0, part=i) for i in range(sockets)]
        self._plan()
        self.EXPERTS_FROM = experts_file.choose(experts_from, self.cpu_arena)
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
        # ⭐ THE EMBEDDING STAYS IN RAM — a token needs one row of it, which the host writes to the card each step; the
        #   card keeps only the head
        self.T_HEAD, self.T_STATE, self.T_KV = t, t + 1, t + 2

        def plan_of(rec):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, self.BPLAN = plan_of(self.emb_rec)
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
            if self.CARD_SLOTS < len(moe):
                raise Refused("%.1f GB holds %d experts, fewer than one a MoE layer" % (self.CARD_GB, self.CARD_SLOTS))
            self.collections.append((self.T_CEXP, self.CPLAN, self.CARD_SLOTS))
            self.arena += self.CPLAN.slot_bytes * self.CARD_SLOTS
            self.GATE_ROW, self.DOWN_ROW = g.row_bytes, dn.row_bytes
            if self.EPLAN.at_up_data != 0 or self.CPLAN.at_up_data != 0:
                raise Refused("a slot's gate-and-up codes are expected at its start")
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
        return [("amd_rocm6_wave64", 0)] + [("x86_avx2", c["dev"]) for c in self.CPUS]

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
            self.KRSCRATCH = 2 * n * H + 16 * P + 2 * P + 4 * P * I + 2 * P * I + 2 * P * I + 4 * n * H + 4096
            self.WORK, works, bigs = max(self.WORK, self.KRSCRATCH), 2, bigs + 1
        card = self._section(big, self.arena, self.collections, kinds, bigs=bigs, mains=2 + (4 + len(self.CPUS) if self.CH else 0),
                             work=self.WORK, works=works)
        # ⛳ a section with no collection is refused, so a run with no MoE layer gives the CPU one of a single slot
        out = {CARD: card}
        moe = sum(self.moe[l] for l in self.run_layers)
        for c in self.CPUS:
            slots = self.SLOTS if self.SLOTS is not None else self.EXPERTS * moe
            experts = [(0, self.EPLAN, slots)] if self.EPLAN else [(0, NL.SlotPlan(4096, 0, 0, 4096), 1)]
            cbig, cbigs = 1 << 20, 3
            if self.CH and self.EPLAN:
                P = self.CH * self.TOPK
                cbig = max(cbig, 2 * self.CH * self.H + 16 * P + 2 * P + 8 * P * self.PI + 4 * self.CH * self.H + 64,
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
        loader.layer_width(0, self.T_HEAD, 1)
        loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
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
        m.define_programs(self._procedures)
        if self.CPLAN is not None and self.CH and self.EPLAN is not None:
            self._calibrate_line()

    def _calibrate_line(self):
        """`nn__expert_major__vram_promotion_threshold_picks`, measured and bound (▶ `expert_tier.calibrate`): one socket's
        `experts_rows` stands for the CPUs, the sockets running theirs at once."""
        H, K = self.H, self.TOPK
        self.CALIBRATION = expert_tier.calibrate(
            self.m, CARD, self.CPUS[0]["w"], self.KRS_AT, self.CPLAN.slot_bytes, self.at[self.cn(0, "hands")], self.HAND,
            2 * H, (2 * H + 2 * K + 7) & ~7, K, self.EXPERTS,
            lambda n: "(ai_glm_5_3__experts_rows %s exr%d_0 %s %d)" % (self.cn(0, "hands"), next(l for l in self.run_layers if self.moe[l]),
                                                                       self.cn(0, "rrows"), n),
            self.CH, fixed=self.CARD_LINE)
        self.EXPERT_LINE = self.CALIBRATION["line"]

    def _int_array(self, name, values):
        """A node array of integers, bound by name — filled a hundred at a time, as a program must fit one heap chunk."""
        m = self.m
        m.node_array(name, len(values))
        for at in range(0, len(values), 100):
            m.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set %s %d %d)" % (name, i, v)
                                                     for i, v in enumerate(values[at:at + 100], at)),
                   "the array %s" % name, "sys__value_true")

    def _card_experts(self, loader):
        """The card's tier, stood at boot: each MoE layer's band of every expert — and, given a router log, its most picked
        ones written whole into the card's slots, an even share of them a layer."""
        m, b = self.m, self.b
        moe = [l for l in self.run_layers if self.moe[l]]
        for l in moe:
            loader.layer_width(l, self.T_CEXP, self.EXPERTS)
        self.CARD_HELD = {}
        if not self.CARD_HOT:
            return
        log = np.load(self.CARD_HOT)
        logged = [int(l) for l in log["layers"]]
        per = self.CARD_SLOTS // len(moe)
        for l in moe:
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
                for x in range(self.EXPERTS):
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
        """Expert `x` of layer `l`, part `s`, as (offset in its slot, bytes) pieces out of the layer's mapped blob `view`: its
        part's gate rows and up rows as one matrix, their scales, its part's columns of every down row, the down scales.
        `whole`: the expert entire, as the card's tier holds it."""
        b, p, P, I = (self.b, self.CPLAN, 1, self.MINTER) if whole else (self.b, self.EPLAN, self.PARTS, self.PI)
        g, u, d = (b.record(l, EXPERT % (x, w)) for w in ("gate", "up", "down"))
        span = lambda at, n: view[at:at + n]
        rb, (d_at, _) = d.row_bytes // P, d.data_span()
        # this part's columns of every down row, cut out of the rows — the one piece that is not contiguous
        cols = np.ascontiguousarray(view[d_at:d_at + d.rows * d.row_bytes].reshape(d.rows, d.row_bytes)[:, s * rb:(s + 1) * rb]).reshape(-1)
        return [(p.at_up_data, span(g.data_span()[0] + s * I * g.row_bytes, I * g.row_bytes)),
                (p.at_up_data + I * g.row_bytes, span(u.data_span()[0] + s * I * u.row_bytes, I * u.row_bytes)),
                (p.at_up_lut, span(g.lut_span()[0] + s * I * 2, I * 2)),
                (p.at_up_lut + I * 2, span(u.lut_span()[0] + s * I * 2, I * 2)),
                (p.at_down_data, cols),
                (p.at_down_lut, span(*d.lut_span()))]

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
                      ("logits", self.EXPERTS), ("sgu", 2 * self.MINTER), ("sact", self.MINTER), ("sactr", self.MINTER)):
            self.at[nm] = m.buffer(n * 2, nm)
        m.buffer(256, "mix")
        m.buffer(64, "weights")
        m.buffer(64, "res")
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
        self.EMB_AT = m.buffer(int(self.emb_rec["row_bytes"]), "emb_codes")      # the token's row, written by the host
        self.EMB_SCALE_AT = m.buffer(64, "emb_scales")
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
            self.EMB_ROWS_AT = m.buffer(CH * int(self.emb_rec["row_bytes"]), "emb_rows")
            self.EMB_RSCALES_AT = m.buffer(max(64, CH * 2), "emb_rscales")
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
            # activation and rotated rows of every pick, the fp32 sum — ▶ ai_glm_5_3__experts_rows
            CH, P, I = self.CH, self.CH * self.TOPK, self.PI
            self.at[c("hands")] = m.buffer(CH * self.HAND, c("hands"))
            m.buffer(CH * H * 2, c("rrows"))
            m.buffer(2 * CH * H + 16 * P + 2 * P + 8 * P * I + 4 * CH * H + 64, c("rscratch"))

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
                cp, ep = self.CPLAN, self.EPLAN
                cells = ["signs", "zero_h", "kscratch", l, self.T_CEXP, cp.at_up_lut - cp.at_up_data,
                         cp.at_down_data - cp.at_up_data, cp.at_down_lut - cp.at_up_data, H, self.MINTER,
                         self.EXPERTS, self.TOPK, self.E_D, 0, self.EXPERTS, 0, self.LIMIT, 0, 1]
                # fed from the sockets' memory where every expert is there (▶ AI_GLM_5_3__EXP__SOURCES)
                if all((l, c["part"]) in self.SRC for c in self.CPUS):
                    self._int_array("src%d" % l, [self.SRC[(l, c["part"])][x] for x in range(self.EXPERTS) for c in self.CPUS])
                    cells += ["src%d" % l, self.PARTS, ep.at_up_lut, ep.at_down_data, ep.at_down_lut, self.GATE_ROW,
                              self.DOWN_ROW, self.CARD_LANDING]
                m.table("exk%d" % l, cells)
                if self.CH:
                    m.table("exkr%d" % l, ["signs", "zero_h", "krscratch"] + cells[3:])   # a prompt chunk's, its own scratch
            mats = ["mlp.shared_experts.%s_proj.weight" % n for n in ("gate", "up", "down")]
            m.table("mo%d" % l, hc(l, "ffn") + [data(l, "post_attention_layernorm.weight"), data(l, "mlp.gate.weight"),
                                                data(l, "mlp.gate.e_score_correction_bias")]
                    + [c for n in mats for c in pair(l, n)] + ["signs", "gscratch"]
                    + head + [self.EXPERTS, self.TOPK, self.MINTER] + [d(l, n) for n in mats]
                    + [self.EPS, self.HC_EPS, self.RSCALE, self.LIMIT])
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
                              + ("(ai_glm_5_3__experts hand picks exk%d routed_k) " % l if self.CPLAN is not None else "")
                              + " ".join("(sys__compute %d names xf%d_%d)" % (w, l, i) for i, w in ws)
                              + " (ai_glm_5_3__shared hand mo%d) " % l
                              + " ".join("(sys__result %d)" % w for _, w in ws) + " "
                              + " ".join("(nn__buffer__copy %s %d %s %d)" % (self.cn(i, "routed"), w, "routed" if i == 0 else "routed%d" % i,
                                                                            2 * H) for i, w in ws) + " "
                              + " ".join("(nn__vector__add routed routed%d routed %d)" % (i, H) for i, _ in ws if i > 0)
                              + (" (nn__vector__add routed routed_k routed %d)" % H if self.CPLAN is not None else "")
                              + " (ai_glm_5_3__close streams mo%d routed)" % l)
            else:
                layers.append("(ai_glm_5_3__mlp streams ml%d)" % l)
        return " ".join(layers)

    def _procedures(self):
        """Every procedure and picture GLM runs — composed here once, then read from `programs.lisp` by the boots that have
        the same key (▶ `Machine.define_programs`). Nothing but programs: the tables are `_tables`'."""
        m, H = self.m, self.H
        row = int(self.emb_rec["row_bytes"])
        # the token's row, un-rotated (Rᵀu = S ⊙ H·u), into every stream
        m.defun("(defun (embed codes scale) (begin (nn__turboquant__decode (nn__vector__range emb_codes codes %d) "
                "(nn__vector__range emb_scales scale 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                "(nn__vector__pointwise_mul hu signs_row e %d) %s (type e)))"
                % (row // 2, H, int(self.emb_rec["d"]), H, H,
                   " ".join("(nn__vector__add e zero_h %s %d)" % (self.rng("streams", j * H, H), H) for j in range(self.MULT))))
        # the head: the streams' mean, the final norm, the logits
        mean = ("(nn__vector__add %s %s hm %d) " % (self.rng("streams", 0, H), self.rng("streams", H, H), H)
                + " ".join("(nn__vector__add hm %s hm %d)" % (self.rng("streams", j * H, H), H) for j in range(2, self.MULT))
                + " (nn__vector__scale hm %r hm %d)" % (1.0 / self.MULT, H))
        logits = (mean + " (nn__rmsnorm__apply hm fnw hn %d %r) (nn__hadamard__rotate hn signs hnr %d) " % (H, self.EPS, H)
                  + "(nn__turboquant__gemv (nn__expert__plane 0 %d 0 %d %d) (nn__expert__plane 0 %d 0 %d %d) hnr logits_h %d %d %d)"
                  % (self.T_HEAD, self.HPLAN.at_up_data, self.HPLAN.up_data, self.T_HEAD, self.HPLAN.at_down_lut,
                     self.HPLAN.down_lut, self.VOCAB, H, int(self.head_rec["d"])))
        m.defun("(defun (head) (begin %s (nn__argmax__find logits_h res %d) (nn__buffer__read res)))" % (logits, self.VOCAB))
        m.defun("(defun (head_logits) (begin %s (type hm)))" % logits)
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
                body += (self.gemv(l, "mlp.gate.weight", "h", "logits")
                         + " (nn__vector__top_k_biased logits %d %d picks weights %s %r) "
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
        if self.CH and self.EPLAN is not None:
            self._rows_procedures()
        if self.EPLAN is not None:
            m.view("names")
        if fused is not None:
            m.defun("(defun (token_f codes scale pos) (begin (embed codes scale) %s (head)))" % fused)
            # a pipeline stage's own layers, the streams in and out (▶ `forward`)
            m.defun("(defun (layers_f pos) (begin %s (type streams)))" % fused)
            m.defun("(defun (token_f_logits codes scale pos) (begin (embed codes scale) %s (head_logits)))" % fused)
        layers = " ".join(("(a%d at len) (sys__compute %d names xw%d) (s%d) (sys__result %d) (c%d)" % (l, CPU, l, l, CPU, l))
                          if self.moe[l] else "(a%d at len)" % l for l in self.run_layers)
        m.defun("(defun (token codes scale at len) (begin (embed codes scale) %s (head)))" % layers)
        m.defun("(defun (token_logits codes scale at len) (begin (embed codes scale) %s (head_logits)))" % layers)

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
                         + ("(ai_glm_5_3__experts_note hands exk%d n nn__expert_major__vram_promotion_threshold_picks) " % l if self.CPLAN is not None else "")
                         + " ".join("(sys__compute %d names xr%d_%d)" % (w, l, i) for i, w in ws) + " "
                         # the card's share computed while the CPUs compute theirs
                         + ("(ai_glm_5_3__experts_rows hands exkr%d rrows_k n) " % l if self.CPLAN is not None else "")
                         # ⛳ WAITED FOR WITHOUT A BOUND: `sys__result` gives up after ~15 s, and a chunk's experts read
                         #   from the disk can take longer — so the card asks `sys__completed` until it says so, then
                         #   collects. `(< 1 0)` is #f: the loop runs while completed is #f
                         + " ".join("(while '(sys__eq (sys__completed %d) (< 1 0)) '(< 0 1)) (sys__result %d)" % (w, w)
                                    for _, w in ws) + " "
                         + " ".join("(nn__buffer__copy %s %d %s %d)" % (self.cn(i, "rrows"), w, "rrows" if i == 0 else "rrows%d" % i,
                                                                       2 * H * self.CH) for i, w in ws) + " "
                         + " ".join("(nn__vector__add rrows rrows%d rrows %d)" % (i, H * self.CH) for i, _ in ws if i > 0)
                         + (" (nn__vector__add rrows rrows_k rrows %d)" % (H * self.CH) if self.CPLAN is not None else "")
                         + " (ai_glm_5_3__close_rows srows mo%d rrows carries n rwork)" % l)
            else:
                body += " (ai_glm_5_3__mlp_rows srows ml%d n rwork)" % l
            m.defun("(defun (r%d first n) (begin %s (type srows)))" % (l, body))
        row = int(self.emb_rec["row_bytes"])
        # a token's row, un-rotated, into every stream of row `at` of the chunk
        m.defun("(defun (embed_row codes scale %s) (begin (nn__turboquant__decode (nn__vector__range emb_rows codes %d) "
                "(nn__vector__range emb_rscales scale 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                "(nn__vector__pointwise_mul hu signs_row e %d) %s (type e)))"
                % (" ".join("at%d" % j for j in range(self.MULT)), row // 2, H, int(self.emb_rec["d"]), H, H,
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

    def step(self, token, pos, greedy=True):
        """One position: the token in, the next token out (greedy), or the logits left for a sampler (answers None)."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        row = int(self.emb_rec["row_bytes"])
        self.m.write(self.EMB_AT, self.emb_data[token * row:(token + 1) * row])     # the embedding lives in RAM
        self.m.write(self.EMB_SCALE_AT, self.emb_lut[token * 2:token * 2 + 2])
        codes, scale = 0, 0
        if self.FUSED:
            kind, value = self.m.grid("(%s %d %d %d)" % ("token_f" if greedy else "token_f_logits", codes, scale, pos))
            return value if greedy else None
        if self.GRID:
            kind, value = self.m.grid("(%s %d %d %d %d)" % ("token" if greedy else "token_logits", codes, scale,
                                                           pos * self.LAT * 2, pos + 1))
            return value if greedy else None
        m = self.m
        m.must("(embed %d %d)" % (codes, scale), "the embedding at %d" % pos)
        for l in self.run_layers:
            self.layer(l, pos)
        return m.must("(head)" if greedy else "(head_logits)", "the head at %d" % pos)

    def prefill(self, tokens, first, greedy=True):
        """Positions `first ..` for `tokens`: as rows, a chunk at a time, when the experts are the CPU's and there is more
        than one; then the next token from the last row (greedy), or its logits left for a sampler."""
        if not self.CH or self.EPLAN is None or len(tokens) < 2:
            out = None
            for i, tok in enumerate(tokens):
                out = self.step(tok, first + i, greedy)
            return out
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        out = None
        for at in range(0, len(tokens), self.CH):
            chunk = tokens[at:at + self.CH]
            out = self.forward_rows(first + at, tokens=chunk, head=at + self.CH >= len(tokens), greedy=greedy)
        return out

    # ── a pipeline stage (▶ `pipeline.py`): this machine's layers, the streams crossing between stages ─────────────
    def forward(self, pos, token=None, x=None, head=False):
        """One position through this stage's layers: the token embedded (the first stage) or the streams that came across
        (`x`, MULT·H halves as bytes) set; then the next token (the last stage) or the streams to hand on."""
        m = self.m
        if token is not None:
            row = int(self.emb_rec["row_bytes"])
            m.write(self.EMB_AT, self.emb_data[token * row:(token + 1) * row])
            m.write(self.EMB_SCALE_AT, self.emb_lut[token * 2:token * 2 + 2])
            m.must("(begin (embed 0 0) (type streams))", "the embedding at %d" % pos)
        else:
            m.write(self.STREAMS_AT, x)
        m.grid("(layers_f %d)" % pos)
        if head:
            return m.must("(head)", "the head at %d" % pos)
        return m.read(self.STREAMS_AT, self.MULT * self.H * 2)

    def forward_rows(self, first, tokens=None, xs=None, n=None, head=False, greedy=True):
        """A chunk of rows through this stage's layers: its tokens embedded (the first stage) or its rows' streams that came
        across (`xs`, `n` rows of MULT·H halves) written; then the next token from its last row (the last stage), or the
        rows' streams to hand on."""
        m, H, M = self.m, self.H, self.MULT
        n = len(tokens) if tokens is not None else n
        if n > self.CH:
            raise Refused("a chunk is at most %d rows, not %d" % (self.CH, n))
        if tokens is not None:
            row = int(self.emb_rec["row_bytes"])
            m.write(self.EMB_ROWS_AT, b"".join(self.emb_data[t * row:(t + 1) * row] for t in tokens))
            m.write(self.EMB_RSCALES_AT, b"".join(self.emb_lut[t * 2:t * 2 + 2] for t in tokens))
            # ⛳ the embeddings in programs of their own, a hundred rows each: one program is one form, and a form
            #   must fit in one of the heap's chunks
            for e0 in range(0, n, 128):
                m.must("(begin %s (type srows))" % " ".join(
                    "(embed_row %d %d %s)" % (i * row // 2, i, " ".join(str((i * M + j) * H) for j in range(M)))
                    for i in range(e0, min(n, e0 + 128))), "the chunk's embeddings")
        else:
            m.write(self.SROWS_AT, xs)
        layers = " ".join("(r%d %d %d)" % (l, first, n) for l in self.run_layers)
        m.grid("(begin (sys__node_array__set nrows 0 %d) %s (type srows))" % (n, layers))
        if not head:
            return m.read(self.SROWS_AT, n * M * H * 2)
        # the last row's streams, then the head
        last = (n - 1) * M * H
        m.must("(begin %s (type streams))" % " ".join("(nn__vector__add %s zero_h %s %d)" % (
            self.rng("srows", last + j * H, H), self.rng("streams", j * H, H), H) for j in range(M)), "the prompt's last row")
        return m.must("(head)", "the head") if greedy else m.must("(head_logits)", "the head")

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

    def state_regions(self, length):
        """(address, bytes) of everything a conversation of `length` positions has written: each KDA layer's state
        and window whole, each MLA layer's latents `0..length`."""
        out = []
        for l in self.run_layers:
            at = self.state_at[l]
            if self.kinds[l] == "d":
                out += [(at + self.SPLAN.at_up_data, self.S_BYTES), (at + self.SPLAN.at_down_data, self.W_BYTES)]
            else:
                out.append((at + self.KVPLAN.at_up_data, length * self.LAT * 2))
        return out
