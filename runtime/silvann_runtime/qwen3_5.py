"""QWEN 3.5 — the DeltaNet-and-attention family, dense (the 27B) or with routed experts (the 35B-A3B), on one card.

Every size comes from the model folder's `config.json`. A layer is `ai_qwen_3`'s words over plane tables the boot
fills: the mixer — `ai_qwen_3__deltanet` or `ai_qwen_3__attention`, its residual included — then the dense MLP
(`ai_qwen_3__mlp`) or the MoE in three (`pre_expert` · `experts` · `post_expert`). Each layer is a procedure,
defined once at boot, and a token is one short program:
```
  (begin (embed codes scale) (nn__rope__angles cs pos theta rot) (layers pos) (head))
```
— the token's embedding row written to the card by the host first: the embedding stays in RAM.
The residual stream `x` never leaves the card; `head` answers the greedy token, or `head_logits` leaves the logits
for a sampler on the host.

A conversation's STATE is the DeltaNet layers' recurrent state and conv window and the attention layers' keys and
values up to its length — `state_regions` says where, so a conversation can be put away and brought back.
"""
import os
import pickle

import numpy as np

from . import experts_file
from .model_folder import NL, Refused

GU, DN = NL.GATE_UP, NL.DOWN
CARD, CPU = 0, 1                  # the workers: the card, and — with the experts there — the CPU
ARCHITECTURES = ("qwen3_5_text", "qwen3_5_moe_text")


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


class Qwen35:
    def __init__(self, folder, machine, max_context=4096, fused_moe=True, kv="tiered", tiers=None, experts_on="card",
                 experts_from="auto", experts_dir=None, layers=None, chunk=256, experts_ram_gb=None):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b = folder, machine, b
        self.H, self.LAYERS = c["hidden_size"], c["num_hidden_layers"]
        self.KH, self.VH, self.LH = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"]
        if c["linear_value_head_dim"] != self.LH:
            raise Refused("the DeltaNet's key and value heads differ in width (%d, %d)" % (self.LH, c["linear_value_head_dim"]))
        self.QH, self.KVH, self.AH = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        rope = c.get("rope_parameters") or {}
        self.ROT = int(self.AH * (rope.get("partial_rotary_factor") or c["partial_rotary_factor"]))
        self.THETA = int(rope.get("rope_theta") or c["rope_theta"])
        self.MOE = bool(c.get("num_experts"))
        if self.MOE:
            self.EXPERTS, self.TOPK, self.INTER = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
            if c.get("shared_expert_intermediate_size", self.INTER) != self.INTER:
                raise Refused("the shared expert is wider than the routed ones — ai_qwen_3's MoE takes one width")
        else:
            self.INTER = c["intermediate_size"]
        self.max_context = max_context
        self.FUSED = fused_moe          # the MoE as ai_qwen_3__moe, one word, over the rows' table — rather than three
        # ⭐ THE KV CACHE: "tiered", rotated from the start — the first `sink` and the most recent `hot` positions in
        #   halves, the next most recent `warm` at TurboQuant 8, the rest at TurboQuant 4 — or "fp16", every position
        #   whole. ⚖ *"default to fp16 4k and then tq8 to fill the space but let an override possible in the boot
        #   settings"*: by default `warm` is every position the other two leave, so nothing reaches tq4. `tiers` =
        #   (sink, hot, warm) overrides it — `warm` may be "rest" — and ⚖ *"we need to keep the current split"* for small
        #   cards, where (256, 4096, max_context // 4) puts the oldest three quarters at tq4. Fixed at boot.
        self.TIERED = kv == "tiered"
        if self.TIERED:
            sink, hot, warm = tiers or (256, 4096, "rest")
            self.SINK, self.HOT = int(sink), int(hot)
            self.WARM = max(1, max_context - self.SINK - self.HOT) if warm == "rest" else int(warm)
            self.COLD = max(0, max_context - self.SINK - self.HOT - self.WARM)
        # ⭐ A PACK THAT KEEPS ITS RESIDUAL ROTATED: the verbs are told so by a cell past their tables, the embedding is
        #   decoded straight into the residual and the head reads it normed and unrotated
        self.ROTATED = b.residual_rotated
        self.kinds = {l: ("f" if "self_attn.q_proj.weight" in b.names(l) else "d") for l in range(self.LAYERS)}
        # ⭐ THE LAYERS THIS MACHINE HOLDS — all of them, or a stage's run of them when the model is split over machines
        #   (▶ `pipeline`): the card is sized for these, and `layers` / `rows` run these
        self.run_layers = list(range(self.LAYERS)) if layers is None else sorted(layers)
        self.CHUNK_WANT = chunk          # a prompt's rows a layer at a time — at most ai_qwen_3's AI_QWEN_3__ROWS_MAX (256)
        self.KV_WIDE = self.KVH * self.AH
        self.CONV_CH = self.KH * self.LH * 2 + self.VH * self.LH
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        self.head_rec, self.emb_rec = g["lm_head.weight"], g["model.language_model.embed_tokens.weight"]
        self.VOCAB = int(self.head_rec["rows"])
        # ⭐ THE ROUTED EXPERTS ON THE CARD, OR ON THE CPU — ⚖ *"the potato tier should fit with an l2 of 8gb and the rest
        #   on disk"*: `experts_on="cpu"` leaves the card the rest of the model and a CPU worker the experts, in RAM
        #   (`experts_from="ram"`) or mapped from their file (`"disk"`, ▶ `experts_file`), `"auto"` choosing by the
        #   memory there is. A token's MoE is then the card's router and shared expert, the CPU's eight, the card's sum.
        if experts_on not in ("card", "cpu"):
            raise Refused("experts_on is 'card' or 'cpu', not %r" % (experts_on,))
        self.CPU_EXPERTS = experts_on == "cpu"
        # a prompt's hand row with the experts on the CPU: the hand, then its picks as words — ▶ AI_QWEN_3__HAND__ROW
        self.HAND_ROW = ((2 * (self.H + 16 + self.INTER) + 7) & ~7) + 8 * self.TOPK if self.MOE else 0
        if self.CPU_EXPERTS and not self.MOE:
            raise Refused("experts on the CPU: this model has no routed experts")
        if self.CPU_EXPERTS:
            self.FUSED = False          # the split words: the router on the card, the experts on the CPU
        self._plan()
        if self.CPU_EXPERTS:
            self.EXPERTS_FROM = experts_file.choose(experts_from, self.cpu_arena)
            self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        # ⭐ FROM THE DISK, THE EXPERTS ARE A CACHE IN THE CPU WORKER'S OWN MEMORY — read from the file into a slot when a
        #   token picks one that is not there, straight from the drive, the layer's least recently used given up for it.
        #   `"mapped"` maps the file instead. ▶ glm5.py's, and nn's loader.
        self.SLOTS = None
        if self.CPU_EXPERTS and self.EXPERTS_FROM == "disk":
            total = self.EXPERTS * len(self.run_layers)
            self.SLOTS = min(total, experts_file.cache_bytes(self.cpu_arena, experts_ram_gb) // self.EPLAN.slot_bytes)
            if self.SLOTS < 4 * self.TOPK:
                raise Refused("%d experts is too few for a cache of them" % self.SLOTS)
            self.cpu_arena = self.SLOTS * self.EPLAN.slot_bytes

    # ── ① the collections: a type a dense tensor, the experts, the head, the embedding, the two states ──────────
    def _plan(self):
        b, L, run = self.b, self.LAYERS, self.run_layers
        names = sorted({n for l in run for n in b.names(l)} - ({GU, DN} if self.MOE else set()))
        self.TYPE = {n: i for i, n in enumerate(names)}
        first = {n: next(l for l in run if n in b.names(l)) for n in names}
        self.PLAN = {n: NL.dense_plan(b, n, first[n]) for n in names}
        slots = {n: sum(1 for l in run if n in b.names(l)) for n in names}
        t = len(names)
        on_card = self.MOE and not self.CPU_EXPERTS
        self.T_EXPERT = t if on_card else None
        t += 1 if on_card else 0
        # ⭐ THE EMBEDDING STAYS IN RAM — a token needs one row of it, which the host writes to the card each step (its
        #   packed row and its scale, decoded there as before); the card keeps only the head
        self.T_HEAD, self.T_STATE, self.T_KV = t, t + 1, t + 2

        def plan_of(rec):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, self.BPLAN = plan_of(self.emb_rec)
        self.S_BYTES, self.W_BYTES = self.VH * self.LH * self.LH * 4, self.CONV_CH * 3 * 2
        self.SPLAN = NL.SlotPlan(self.S_BYTES, 0, 0, self.W_BYTES)
        if self.TIERED:
            # a slot a layer: keys in the up half, values in the down, each [sink | hot | warm codes | warm scales |
            # cold codes | cold scales]
            rb8, rb4 = (self.AH + 1) // 2 * 2, (self.AH + 3) // 4 * 2
            w = self.KV_WIDE * 2
            self.TIER = dict(sink=(0, self.SINK * w), hot=(self.SINK * w, self.HOT * w))
            off = (self.SINK + self.HOT) * w
            for nm, n in (("warm_c", self.WARM * self.KVH * rb8), ("warm_s", self.WARM * self.KVH * 2),
                          ("cold_c", max(1, self.COLD) * self.KVH * rb4), ("cold_s", max(1, self.COLD) * self.KVH * 2)):
                self.TIER[nm] = (off, n)
                off += (n + 255) // 256 * 256
            self.KVPLAN = NL.SlotPlan(off, 0, 0, off)
        else:
            kv = self.max_context * self.KV_WIDE * 2
            self.KVPLAN = NL.SlotPlan(kv, 0, 0, kv)
        n_d = sum(1 for l in run if self.kinds[l] == "d")
        coll = [(self.TYPE[n], self.PLAN[n], slots[n]) for n in names]
        self.cpu_arena = 0
        if self.MOE:
            self.EPLAN = NL.expert_plan(b, 0, self.EXPERTS)
            if on_card:
                coll.append((self.T_EXPERT, self.EPLAN, self.EXPERTS * len(run)))
            else:
                self.cpu_arena = self.EPLAN.slot_bytes * self.EXPERTS * len(run)
        coll += [(self.T_HEAD, self.HPLAN, 1), (self.T_STATE, self.SPLAN, n_d),
                 (self.T_KV, self.KVPLAN, len(run) - n_d)]
        self.collections = coll
        self.arena = sum(p.slot_bytes * n for _, p, n in coll)

    def device_params(self):
        """The boot's device section: the buffer pools, the arena and every collection's shape."""
        if not 1 <= self.CHUNK_WANT <= 256:
            raise Refused("chunk is 1 to 256 rows, not %r" % (self.CHUNK_WANT,))
        self.CHUNK = min(self.CHUNK_WANT, self.max_context)
        # a prompt's rows: their scratch (the tiered rows verb's is the larger), and — tiered — the window the cache's fp16
        # positions are gathered into; the one-cache rows verb's scores span the whole context instead
        qw, kvw = self.QH * self.AH, self.KVH * self.AH
        self.ROWS_SCRATCH = max(self.CHUNK * 80 * 1024, 16 * 256 + self.CHUNK * 2 * (3 * self.H + 8 * qw + 2 * kvw),
                                16 * 256 + self.CHUNK * 2 * (3 * self.H + 4 * self.INTER))    # the dense MLP's rows
        self.WINDOW = ((self.SINK + self.HOT) * kvw * 2) if self.TIERED else 0
        self.WINDOW_SCORES = (self.CHUNK * self.QH * (self.SINK + self.HOT) * 4) if self.TIERED else 0
        big = max(8 << 20, self.QH * self.max_context * 4, self.VOCAB * 2,
                  (self.CHUNK * self.QH * self.max_context * 4) if self.MOE and not self.TIERED else 0,
                  self.ROWS_SCRATCH, self.WINDOW, self.WINDOW_SCORES)
        # ⭐ a rank-1 mask lives in one buffer a worker, of the largest class, which it grows by one
        self._mask_layout()
        big = max(big, self.MASK_BYTES[CARD])
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in self.collections)
        n_d = sum(1 for k in self.kinds.values() if k == "d")
        # ⛳ four mid-size buffers: a wide model's chunk of rows (the 27B's `xs` and `h1s` at 256 rows, 2.6 MB each) lands here
        card = ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
                "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:4\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:64\n"
                "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:%d\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:8\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:%d\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                % (big, 6 + (self.MASK_BYTES[CARD] > 0), (self.arena + (256 << 20)) >> 20, self.LAYERS - n_d, n_d,
                   "".join(self.kinds[l] for l in range(self.LAYERS)), len(self.collections), types))
        if not self.CPU_EXPERTS:
            return card
        # the CPU worker's section: its one collection, the experts, and small pools for the hand-off's buffers — and a
        # prompt's: its hand rows, its routed rows and `experts_rows`' scratch (▶ ai_qwen_3's `experts_rows`)
        p = self.EPLAN
        P, I = self.CHUNK * self.TOPK, self.INTER
        self.RSCRATCH = 2 * self.CHUNK * self.H + 16 * P + 2 * P + 8 * P * I + 4 * self.CHUNK * self.H + 64
        cbig = max(1 << 20, self.RSCRATCH, self.CHUNK * self.HAND_ROW, self.MASK_BYTES[CPU])
        cpu = ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
               "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:1\n"
               "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:16\n"
               "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:%d\n"
               "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:8\n"
               "nn__expert_cache_l1__size_mb:%d\n"
               "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
               "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
               "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
               "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
               "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:%d\n"
               "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
               "nn__model__layer_types:%s\nnn__expert__types:1\n"
               "nn__expert__type0__up_bytes:%d\nnn__expert__type0__down_bytes:%d\nnn__expert__type0__slots:%d\n"
               % (cbig, 5 + (self.MASK_BYTES[CPU] > 0), (self.cpu_arena + (256 << 20)) >> 20, self.LAYERS - n_d, n_d,
                  "".join(self.kinds[l] for l in range(self.LAYERS)), p.up_bytes, p.down_bytes,
                  self.SLOTS if self.SLOTS is not None else self.EXPERTS * len(self.run_layers)))
        return {CARD: card, CPU: cpu}

    def workers(self):
        """The machine's workers: the card, and the CPU when the experts are there."""
        return [(self.m.silicon, 0)] + ([("x86_avx2", 0)] if self.CPU_EXPERTS else [])

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        loader = m.loader()
        self.state_at = {}
        self.slot_at = {}              # (layer, name) -> the slot a dense matrix lives in, for a LoRA swap
        for i, l in enumerate(self.run_layers):
            for n in b.names(l):
                if self.MOE and n in (GU, DN):
                    continue
                loader.layer_width(l, self.TYPE[n], 1)
                self.slot_at[(l, n)] = loader.load_dense(b, self.PLAN[n], self.TYPE[n], n, l)
            if self.MOE and not self.CPU_EXPERTS:
                loader.layer_width(l, self.T_EXPERT, self.EXPERTS)
                for x in range(self.EXPERTS):
                    loader.load_expert(b, self.EPLAN, self.T_EXPERT, l, x, self.EXPERTS)
            t = self.T_STATE if self.kinds[l] == "d" else self.T_KV
            loader.layer_width(l, t, 1)
            self.state_at[l] = loader.reserve(t)
            loader.assign(l, t, 0, self.state_at[l])
            if progress:
                progress(i + 1, len(self.run_layers))
        loader.layer_width(0, self.T_HEAD, 1)
        loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())
        self._buffers()
        self.MASK_CELLS = {}
        self._mask_buffers(cpu=False)
        self._tables()
        if self.CPU_EXPERTS:
            self._cpu_experts(progress)
        self._procedures()
        self.active_lora, self._lora_cache = self.folder.lora, {}     # a LoRA chosen at boot is in the slots already

    def _expert_pieces(self, view, l, x, part):
        """Expert `x` of layer `l` as its slot's four pieces (▶ `experts_file`) — the loader's own slices."""
        b, p = self.b, self.EPLAN
        gd, gl = b.record(l, GU).expert_slices(x, self.EXPERTS)
        dd, dl = b.record(l, DN).expert_slices(x, self.EXPERTS)
        u8 = lambda raw: np.frombuffer(raw, dtype=np.uint8)
        return [(p.at_up_data, u8(gd)), (p.at_up_lut, u8(gl)), (p.at_down_data, u8(dd)), (p.at_down_lut, u8(dl))]

    def _cpu_experts(self, progress):
        """The CPU worker: every layer's experts in its memory or mapped from their file, the hand-off's buffers, each
        layer's experts table and the picture the card hands it."""
        m, H, e = self.m, self.H, self.EPLAN
        layers = self.run_layers
        backing = {}
        if self.EXPERTS_FROM == "disk":
            backing = experts_file.cache_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                                 self._expert_pieces, list(range(self.LAYERS)), progress=progress)
        elif self.EXPERTS_FROM == "mapped":
            experts_file.map_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                     self._expert_pieces, list(range(self.LAYERS)), progress=progress)
        else:
            m.act_as(CPU)
            loader = m.loader()
            for l in layers:
                loader.layer_width(l, 0, self.EXPERTS)
                for x in range(self.EXPERTS):
                    loader.load_expert(self.b, e, 0, l, x, self.EXPERTS)
            if loader.handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loader.handed())
        m.act_as(CPU)
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        m.buffer(1024, "c_signs", _half(signs[:512]))
        m.buffer(H * 2, "c_zero", _half(np.zeros(H)))
        m.buffer(2 * (H + 16), "c_hand")
        m.buffer(H * 2, "c_routed")
        m.buffer(131072, "c_scratch")
        m.buffer(self.CHUNK * self.HAND_ROW, "c_hands")
        m.buffer(self.CHUNK * H * 2, "c_rrows")
        m.buffer(self.RSCRATCH, "c_rscratch")
        self._mask_buffers(cpu=True)
        for l in layers:
            m.table("exc%d" % l, ["c_signs", "c_scratch", "c_zero", l, 0, e.at_up_lut, e.at_down_data, e.at_down_lut,
                                  H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d),
                                  backing.get((CPU, l), 0)] + self._masked(l, "moe", 16, 14))
            # a prompt's: the chunk's scratch in place of a position's, and a picture over the chunk
            m.table("exrc%d" % l, ["c_signs", "c_rscratch", "c_zero", l, 0, e.at_up_lut, e.at_down_data, e.at_down_lut,
                                   H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d),
                                   backing.get((CPU, l), 0)] + self._masked(l, "moe", 16, 14))
            m.picture("xr%d" % l, "(begin (nn__buffer__copy hands %d c_hands %d) "
                      "(ai_qwen_3__experts_rows c_hands exrc%d c_rrows (sys__node_array__get nrows 0)))"
                      % (CARD, self.CHUNK * self.HAND_ROW, l))
            m.picture("xq%d" % l, "(begin (nn__buffer__copy hand %d c_hand %d) (ai_qwen_3__experts c_hand picks exc%d c_routed))"
                      % (CARD, 2 * (H + 16), l))
        m.act_as(CARD)

    def _buffers(self):
        m, H = self.m, self.H
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        final_w = np.frombuffer(np.asarray(self.global_["model.language_model.norm.weight"]["vbr_data"]).tobytes(),
                                dtype="<f2").astype(np.float64)[:H]
        m.buffer(1024, "signs", _half(signs[:512]))
        m.buffer(1024, "ones512", _half(np.ones(512)))
        m.buffer(64, "minus1", _half(-np.ones(32)))
        m.buffer(64, "one", _half(np.ones(32)))
        m.buffer(H * 2, "signs_row", _half(np.tile(signs[:512], H // 512)))
        m.buffer(H * 2, "fnw", _half(1.0 + final_w))           # the `1 + w` convention
        self.X_AT = m.buffer(H * 2, "x")
        for nm in ("h", "h1", "routed"):
            m.buffer(H * 2, nm)
        m.buffer(H * 2, "zero_h", _half(np.zeros(H)))
        m.buffer(self.ROT * 4, "cs")
        m.buffer(self.QH * self.max_context * 4, "scores")
        if self.TIERED:
            rw = 4 * self.QH * (self.AH + 2)
            m.buffer(4 * rw + 2 * (2 * self.QH * self.AH + 3 * self.KV_WIDE), "tier_scratch")
        m.buffer(262144, "mix_scratch")
        row = int(self.emb_rec["row_bytes"])
        self.EMB_AT = m.buffer(self.CHUNK * row, "emb_codes")               # the rows the host writes, a step's or a chunk's
        self.EMB_SCALE_AT = m.buffer(max(64, self.CHUNK * 2), "emb_scales")
        self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        m.buffer(64, "res")
        # ⭐ A PROMPT'S ROWS — every model reads its prompt as rows: the scratch, the angles, the rows and their h1s, and
        #   the tiered cache's window, or the one-cache rows' scores
        m.buffer(self.ROWS_SCRATCH, "pf_scratch")
        m.buffer(self.CHUNK * self.ROT * 4, "cs_rows")
        if self.TIERED:
            m.buffer(self.WINDOW, "win_k")
            m.buffer(self.WINDOW, "win_v")
            m.buffer(3 * self.CHUNK * 4 * self.QH * (self.AH + 2), "rows_res")
            m.buffer(self.WINDOW_SCORES, "window_scores")
        else:
            m.buffer(self.CHUNK * self.QH * self.max_context * 4, "scores_rows")
        self.XS_AT = m.buffer(self.CHUNK * H * 2, "xs")
        m.buffer(self.CHUNK * H * 2, "h1s")
        if self.MOE:
            # the router's hand, its picks and its scratch
            m.buffer(2 * (H + 16 + self.INTER), "hand")
            m.buffer(65536, "pre_scratch")
        if self.CPU_EXPERTS:
            # a prompt's hand rows and the routed rows the CPU hands back
            m.buffer(self.CHUNK * self.HAND_ROW, "hands")
            m.buffer(self.CHUNK * H * 2, "rrows")
            m.node_array("nrows", 1)
            m.buffer(131072, "exp_scratch")
            m.node_array("picks", self.TOPK)

    def plane(self, l, n, which):
        p = self.PLAN[n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.TYPE[n], at, size)

    def tier(self, l, which, half):
        """A tier of an attention layer's cache, keys (`up`) or values (`down`)."""
        at, size = self.TIER[which]
        base = self.KVPLAN.at_up_data if half == "up" else self.KVPLAN.at_down_data
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, base + at, size)

    def state(self, l, which):
        t, p = (self.T_STATE, self.SPLAN) if self.kinds[l] == "d" else (self.T_KV, self.KVPLAN)
        at, size = (p.at_up_data, p.up_bytes) if which == "up" else (p.at_down_data, p.down_bytes)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, t, at, size)

    def _mixer_cells(self, l, scratch, cs="cs", scores="scores"):
        """A mixer's plane table over the scratch named: the DeltaNet's or the attention's, as its verbs read it."""
        pl, d = self.plane, (lambda l_, n: self.b.record(l_, n).d)
        if self.kinds[l] == "d":
            ns = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
                  "linear_attn.in_proj_b.weight"]
            return ([pl(l, "input_layernorm.weight", "data")] + [c for n in ns for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                    + [pl(l, "linear_attn.conv1d.weight", "data"), pl(l, "linear_attn.A_log", "data"),
                       pl(l, "linear_attn.dt_bias", "data"), pl(l, "linear_attn.norm.weight", "data"),
                       pl(l, "linear_attn.out_proj.weight", "data"), pl(l, "linear_attn.out_proj.weight", "lut"),
                       self.state(l, "up"), self.state(l, "down"), "signs", "minus1", scratch,
                       self.H, self.KH, self.VH, self.LH] + [d(l, n) for n in ns] + [d(l, "linear_attn.out_proj.weight")])
        ns = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
        return ([pl(l, "input_layernorm.weight", "data")] + [c for n in ns for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                + [pl(l, "self_attn.q_norm.weight", "data"), pl(l, "self_attn.k_norm.weight", "data"),
                   self.state(l, "up"), self.state(l, "down"), cs, scores, "signs", scratch,
                   self.H, self.QH, self.KVH, self.AH, self.ROT] + [d(l, n) for n in ns])

    def _tiered_cells(self, l):
        """An attention layer's tiered table: the one-cache cells with the sink in the cache's place, then the tiers."""
        cells = self._mixer_cells(l, "mix_scratch")
        cells[11], cells[12] = self.tier(l, "sink", "up"), self.tier(l, "sink", "down")
        return (cells + [self.THETA, 1 if self.ROTATED else 0]
                + [self.tier(l, "hot", "up"), self.tier(l, "hot", "down"),
                   self.tier(l, "warm_c", "up"), self.tier(l, "warm_s", "up"),
                   self.tier(l, "warm_c", "down"), self.tier(l, "warm_s", "down"),
                   self.tier(l, "cold_c", "up"), self.tier(l, "cold_s", "up"),
                   self.tier(l, "cold_c", "down"), self.tier(l, "cold_s", "down"),
                   "tier_scratch", self.SINK, self.HOT, self.WARM])

    # ── the rank-1 mask, when the folder's LoRA is one (▶ `nn_rank1_mask.py`, the `rank1_*` doors) ─────────────────
    def _mask_layout(self):
        """Where each of the mask's tensors goes: `MASK_PLACE[(worker, l, site)]` its `r` and `v` as (start, count) in
        halves inside that worker's one mask buffer, and `MASK_BYTES` each buffer's size — the scratch for the verbs'
        dots first. The mixers and the shared expert are the card's; the routed experts go wherever they run."""
        self.MASK_BYTES, self.MASK_PLACE = {CARD: 0, CPU: 0}, {}
        mask = self.b.mask
        if mask is None:
            return
        from safetensors.numpy import load_file
        self._mask_tensors = t = load_file(mask["path"])
        scratch = 8 * self.CHUNK * max(self.TOPK, 1) + 256
        at = {CARD: scratch, CPU: scratch}
        for l in self.run_layers:
            for site in ("attn", "shared", "moe"):
                if "layers.%d.%s.r" % (l, site) not in t:
                    continue
                w = CPU if site == "moe" and self.CPU_EXPERTS else CARD
                place = []
                for a in (t["layers.%d.%s.r" % (l, site)], t["layers.%d.%s.%s" % (l, site, "V" if site == "moe" else "v")]):
                    place.append((at[w] // 2, a.size))
                    at[w] += (a.nbytes + 255) & ~255
                self.MASK_PLACE[(w, l, site)] = place
        for w in (CARD, CPU):
            if any(k[0] == w for k in self.MASK_PLACE):
                self.MASK_BYTES[w] = at[w]
        self._mask_scratch = scratch

    def _mask_buffers(self, cpu):
        """This worker's mask buffer, filled, and each site's three table cells: views of its `r`, its `v` and the
        scratch."""
        w = CPU if cpu else CARD
        if not self.MASK_BYTES[w]:
            return
        m, t, name = self.m, self._mask_tensors, ("c_mk" if cpu else "mk")
        base = m.buffer(self.MASK_BYTES[w], name)
        view = lambda start, count: "(nn__vector__range %s %d %d)" % (name, start, count)
        for (w_, l, site), place in self.MASK_PLACE.items():
            if w_ != w:
                continue
            for (start, count), a in zip(place, (t["layers.%d.%s.r" % (l, site)],
                                                 t["layers.%d.%s.%s" % (l, site, "V" if site == "moe" else "v")])):
                m.write(base + 2 * start, np.ascontiguousarray(a, dtype="<f2").tobytes())
            self.MASK_CELLS[(l, site)] = [view(*place[0]), view(*place[1]), view(0, self._mask_scratch // 2)]

    def _masked(self, l, site, at, have=None, both=False):
        """A table's mask cells for layer `l`'s site at cell `at` — zeros up to it from `have` cells — or nothing when the
        folder has no mask. A site the mask leaves alone is a 0. `both`: the routed experts' three, then the shared's."""
        if not self.MASK_CELLS and self.b.mask is None:
            return []
        pad = [0] * (at - have) if have is not None else []
        cells = lambda s_: list(self.MASK_CELLS[(l, s_)]) if (l, s_) in self.MASK_CELLS else [0, 0, 0]
        return pad + cells(site) + (cells("shared") if both else [])

    def _tables(self):
        m, b, pl = self.m, self.b, self.plane
        d = lambda l, n: b.record(l, n).d
        for l in self.run_layers:
            # ⭐ a mask's cells, when the folder's LoRA is a rank-1 mask: past each table's longest form, padded up to them
            mk = lambda cells, at: cells + self._masked(l, "attn", at, len(cells))
            if self.kinds[l] == "d":
                m.table("mx%d" % l, mk(self._mixer_cells(l, "mix_scratch") + ([1] if self.ROTATED else []), 30))
            elif self.TIERED:
                tiered = self._tiered_cells(l)
                m.table("mx%d" % l, mk(tiered, 48))
                # a prompt's rows through the same tiers: ▶ ai_qwen_3__attention_tiered_rows
                m.table("atr%d" % l, mk(tiered + ["pf_scratch", "cs_rows", "win_k", "win_v", "rows_res", "window_scores"], 48))
            else:
                m.table("mx%d" % l, mk(self._mixer_cells(l, "mix_scratch")
                                       + ([self.THETA, 1] if self.ROTATED else [self.THETA, 0] if self.b.mask is not None else []), 48))
            # the rows' mixer tables, over the prompt's scratch
            flag = [1] if self.ROTATED else []
            if self.kinds[l] == "d":
                m.table("dr%d" % l, mk(self._mixer_cells(l, "pf_scratch") + flag, 30))
            elif not self.TIERED:
                m.table("ar%d" % l, mk(self._mixer_cells(l, "pf_scratch", "cs_rows", "scores_rows") + [self.THETA] + flag, 48))
            if not self.MOE:
                ms = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
                for name, scratch in (("ml%d", "mix_scratch"), ("mlr%d", "pf_scratch")):     # a position's, and the rows'
                    m.table(name % l, [pl(l, "post_attention_layernorm.weight", "data")]
                            + [c for n in ms for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                            + ["signs", "one", scratch, self.H, self.INTER] + [d(l, n) for n in ms]
                            + ([1] if self.ROTATED else []))
                continue
            sh = ["mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                  "mlp.shared_expert_gate.weight"]
            m.table("pre%d" % l, [pl(l, "post_attention_layernorm.weight", "data"), pl(l, "mlp.gate.weight", "data"),
                                  pl(l, "mlp.gate.weight", "lut")]
                    + [c for n in sh for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                    + ["signs", "pre_scratch", self.H, self.INTER, self.EXPERTS, self.TOPK, d(l, "mlp.gate.weight"),
                       d(l, "mlp.shared_expert.down_proj.weight")] + ([1] if self.ROTATED else []))
            m.table("post%d" % l, [pl(l, "mlp.shared_expert.down_proj.weight", "data"),
                                   pl(l, "mlp.shared_expert.down_proj.weight", "lut"),
                                   self.H, self.INTER, self.TOPK, d(l, "mlp.shared_expert.down_proj.weight")]
                    + self._masked(l, "shared", 6))
            if self.CPU_EXPERTS:
                continue                # the experts' table is the CPU's (▶ `_cpu_experts`)
            # the fused MoE's rows table, its streaming cells empty
            e = self.EPLAN
            m.table("mr%d" % l, [pl(l, "post_attention_layernorm.weight", "data"), pl(l, "mlp.gate.weight", "data"),
                                 pl(l, "mlp.gate.weight", "lut")]
                    + [c for n in ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                                   "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight")
                       for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                    + ["signs", "pf_scratch", l, self.T_EXPERT, e.at_up_lut, e.at_down_data, e.at_down_lut,
                       self.H, self.INTER, self.EXPERTS, self.TOPK, d(l, "mlp.gate.weight"),
                       d(l, "mlp.shared_expert.down_proj.weight"), d(l, GU), 0, 0, 0]
                    + (flag or ([0] if self.b.mask is not None else [])) + self._masked(l, "moe", 29, 29, both=True))
            m.table("ex%d" % l, ["signs", "exp_scratch", "zero_h", l, self.T_EXPERT, e.at_up_lut, e.at_down_data,
                                 e.at_down_lut, self.H, self.INTER, self.EXPERTS, self.TOPK, d(l, GU), 0]
                    + self._masked(l, "moe", 16, 14))

    # ── ③ the procedures, defined once ────────────────────────────────────────────────────────────────────────
    def _procedures(self):
        m, H = self.m, self.H
        row = int(self.emb_rec["row_bytes"])
        if self.ROTATED:        # the decoded row IS R·e — the residual's own basis
            m.defun("(defun (embed codes scale) (begin (nn__turboquant__decode (nn__vector__range emb_codes codes %d) "
                    "(nn__vector__range emb_scales scale 1) x 1 %d %d) (type x)))"
                    % (row // 2, H, int(self.emb_rec["d"])))
        else:                   # un-rotated: Rᵀu = S ⊙ H·u
            m.defun("(defun (embed codes scale) (let ((u (nn__buffer__getnew %d)) (hu (nn__buffer__getnew %d))) "
                    "(nn__turboquant__decode (nn__vector__range emb_codes codes %d) (nn__vector__range emb_scales scale 1) u 1 %d %d) "
                    "(nn__hadamard__rotate u ones512 hu %d) (nn__vector__pointwise_mul hu signs_row x %d) (type x)))"
                    % (H * 2, H * 2, row // 2, H, int(self.emb_rec["d"]), H, H))
        normed = ("(nn__rmsnorm__apply x fnw h %d) " % H) if self.ROTATED else \
                 ("(nn__rmsnorm__apply x fnw h %d) (nn__hadamard__rotate h signs h1 %d) " % (H, H))
        logits = (normed + "(nn__turboquant__gemv (nn__expert__plane 0 %d 0 %d %d) (nn__expert__plane 0 %d 0 %d %d) %s logits_h %d %d %d)"
                  % (self.T_HEAD, self.HPLAN.at_up_data, self.HPLAN.up_data, self.T_HEAD, self.HPLAN.at_down_lut,
                     self.HPLAN.down_lut, "h" if self.ROTATED else "h1", self.VOCAB, H, int(self.head_rec["d"])))
        # a prompt's rows — every model has them
        into = "(nn__vector__range xs at %d)" % H
        if self.ROTATED:
            m.defun("(defun (embed_row codes scale at) (begin (nn__turboquant__decode (nn__vector__range emb_codes codes %d) "
                    "(nn__vector__range emb_scales scale 1) %s 1 %d %d) (type xs)))"
                    % (row // 2, into, H, int(self.emb_rec["d"])))
        else:
            m.defun("(defun (embed_row codes scale at) (let ((u (nn__buffer__getnew %d)) (hu (nn__buffer__getnew %d))) "
                    "(nn__turboquant__decode (nn__vector__range emb_codes codes %d) (nn__vector__range emb_scales scale 1) u 1 %d %d) "
                    "(nn__hadamard__rotate u ones512 hu %d) (nn__vector__pointwise_mul hu signs_row %s %d) (type xs)))"
                    % (H * 2, H * 2, row // 2, H, int(self.emb_rec["d"]), H, into, H))
        for l in self.run_layers:
            mixer = ("(ai_qwen_3__deltanet_rows xs dr%d h1s n)" % l if self.kinds[l] == "d"
                     else ("(ai_qwen_3__attention_tiered_rows xs atr%d first h1s n)" if self.TIERED
                           else "(ai_qwen_3__attention_rows xs ar%d first h1s n)") % l)
            if self.CPU_EXPERTS:
                # ⭐ THE EXPERTS ON THE CPU, A CHUNK AT A TIME: the card routes every row into its hand row, the CPU runs each
                #   expert once over the rows that picked it, the card closes every row. ⛳ Waited for without a bound, as
                #   GLM's: a chunk's experts read from the disk can outlast `sys__result`'s wait
                mlp = ("(ai_qwen_3__pre_expert_rows h1s pre%d hands n) (sys__compute %d names xr%d) "
                       "(while '(sys__eq (sys__completed %d) (< 1 0)) '(< 0 1)) (sys__result %d) "
                       "(nn__buffer__copy c_rrows %d rrows %d) (ai_qwen_3__post_expert_rows h1s hands rrows post%d xs n)"
                       % (l, CPU, l, CPU, CPU, CPU, self.CHUNK * H * 2, l))
            else:
                mlp = ("(ai_qwen_3__moe_rows h1s mr%d xs n)" if self.MOE else "(ai_qwen_3__mlp_rows h1s mlr%d xs n)") % l
            m.defun("(defun (rows%d first n) (begin %s %s (type xs)))" % (l, mixer, mlp))
        m.defun("(defun (rows first n) (begin %s (type xs)))"
                % " ".join("(rows%d first n)" % l for l in self.run_layers))
        m.defun("(defun (head) (begin %s (nn__argmax__find logits_h res %d) (nn__buffer__read res)))" % (logits, self.VOCAB))
        m.defun("(defun (head_logits) (begin %s (type x)))" % logits)
        for l in self.run_layers:
            attention = "ai_qwen_3__attention_tiered" if self.TIERED else "ai_qwen_3__attention"
            mixer = ("(ai_qwen_3__deltanet x mx%d %s)" if self.kinds[l] == "d" else "(" + attention + " x mx%d pos %s)")
            if self.MOE and self.FUSED:
                body = mixer % (l, "h1") + " (ai_qwen_3__moe h1 mr%d x)" % l
            elif self.CPU_EXPERTS:
                body = (mixer % (l, "h1") + " (ai_qwen_3__pre_expert h1 pre%d hand picks) (sys__compute %d names xq%d) "
                        "(sys__result %d) (nn__buffer__copy c_routed %d routed %d) (ai_qwen_3__post_expert h1 hand routed post%d x)"
                        % (l, CPU, l, CPU, CPU, 2 * self.H, l))
            elif self.MOE:
                body = (mixer % (l, "h1") + " (ai_qwen_3__pre_expert h1 pre%d hand picks) "
                        "(ai_qwen_3__experts hand picks ex%d routed) (ai_qwen_3__post_expert h1 hand routed post%d x)"
                        % (l, l, l))
            else:
                body = mixer % (l, "h") + " (ai_qwen_3__mlp h ml%d x)" % l
            m.defun("(defun (layer%d pos) (begin %s (type x)))" % (l, body))
        m.defun("(defun (layers pos) (begin %s (type x)))" % " ".join("(layer%d pos)" % l for l in self.run_layers))
        if self.CPU_EXPERTS:
            m.view("names")             # what a CPU block sees: the pictures, the tables, the card's hand and picks

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: every DeltaNet layer's state and window to zero. The attention caches need
        nothing — a position reads only rows it has written."""
        zeros = []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                zeros += ["(nn__vector__zero %s %d)" % (self.state(l, "up"), self.S_BYTES // 2),
                          "(nn__vector__zero %s %d)" % (self.state(l, "down"), self.W_BYTES // 2)]
        self.m.must("(begin %s (type x))" % " ".join(zeros), "the reset")

    def step(self, token, pos, greedy=True):
        """One position: the token in, the next token out (greedy), or the logits left for a sampler (answers None)."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._stage([token])
        head = "(head)" if greedy else "(head_logits)"
        return self._run("(begin (embed 0 0) (nn__rope__angles cs %d %d %d) (layers %d) %s)"
                         % (pos, self.THETA, self.ROT, pos, head), "position %d" % pos)

    def _run(self, text, what):
        """A program's answer. ⛳ With the experts on the CPU it runs with every block standing — it hands them work
        (`sys__compute`) — and a program that does not would find the CPU's block never started."""
        if self.CPU_EXPERTS:
            return self.m.grid(text)[1]
        return self.m.must(text, what)

    def _stage(self, tokens):
        """The tokens' packed rows and scales from the host's embedding into the card's staging buffers, in order."""
        row = int(self.emb_rec["row_bytes"])
        self.m.write(self.EMB_AT, b"".join(self.emb_data[t * row:(t + 1) * row] for t in tokens))
        self.m.write(self.EMB_SCALE_AT, b"".join(self.emb_lut[t * 2:t * 2 + 2] for t in tokens))

    def prefill(self, tokens, first, greedy=True):
        """Positions `first ..` for `tokens` in chunks of rows — each layer one call a chunk — then the next token from
        the last row (greedy), or its logits left for a sampler."""
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        row = int(self.emb_rec["row_bytes"])
        step = min(self.CHUNK, self.HOT) if self.TIERED else self.CHUNK      # a chunk never outgrows the hot ring
        for at in range(0, len(tokens), step):
            chunk = tokens[at:at + step]
            self._stage(chunk)
            embeds = " ".join("(embed_row %d %d %d)" % (i * row // 2, i, i * self.H) for i in range(len(chunk)))
            count = "(sys__node_array__set nrows 0 %d) " % len(chunk) if self.CPU_EXPERTS else ""
            self._run("(begin %s%s (rows %d %d) (type xs))" % (count, embeds, first + at, len(chunk)), "the rows from %d" % (first + at))
            last = len(chunk) - 1
        head = "(head)" if greedy else "(head_logits)"
        return self._run("(begin (nn__vector__add (nn__vector__range xs %d %d) zero_h x %d) %s)"
                         % (last * self.H, self.H, self.H, head), "the prompt's last row")

    # ── a stage of a pipeline: this machine's run of layers, its input and output the residual rows ─────────────
    # ⚖ *"we need to test pipeline parallelism"*. A model split over machines (▶ `pipeline`) runs each machine's layers
    # on what the one before handed on: the first machine starts from the token, the last ends with the head.
    def forward(self, pos, token=None, x=None, head=False):
        """Position `pos` through this machine's layers, from `token` (the first machine) or the residual row `x` (the
        bytes the machine before answered). Answers the next token when `head`, else this machine's residual row."""
        if token is not None:
            self._stage([token])
            start = "(embed 0 0)"
        else:
            self.m.write(self.X_AT, x)
            start = ""
        end = "(head)" if head else "(type x)"
        got = self._run("(begin %s (nn__rope__angles cs %d %d %d) (layers %d) %s)"
                        % (start, pos, self.THETA, self.ROT, pos, end), "position %d" % pos)
        return got if head else self.m.read(self.X_AT, self.H * 2)

    def forward_rows(self, first, tokens=None, xs=None, n=None, head=False):
        """A chunk of positions `first ..` as rows through this machine's layers, from `tokens` or from the rows `xs`.
        Answers the next token from the last row when `head`, else this machine's rows."""
        if self.CPU_EXPERTS:
            raise Refused("a stage with its experts on the CPU reads a prompt a position at a time")
        m, H = self.m, self.H
        if tokens is not None:
            n = len(tokens)
            row = int(self.emb_rec["row_bytes"])
            self._stage(tokens)
            start = " ".join("(embed_row %d %d %d)" % (i * row // 2, i, i * H) for i in range(n))
        else:
            m.write(self.XS_AT, xs)
            start = ""
        m.must("(begin %s (rows %d %d) (type xs))" % (start, first, n), "the rows from %d" % first)
        if not head:
            return m.read(self.XS_AT, n * H * 2)
        return m.must("(begin (nn__vector__add (nn__vector__range xs %d %d) zero_h x %d) (head))" % ((n - 1) * H, H, H),
                      "the chunk's last row")

    def logits(self):
        return np.frombuffer(self.m.read(self.LOGITS_AT, self.VOCAB * 2), dtype="<f2").astype(np.float32)

    # ── ⑤ LoRAs, swapped in place ─────────────────────────────────────────────────────────────────────────────
    # ⚖ *"would it be a nightmare to do hot swapping of loras at the cartridge level? … if there is a unload_lora and
    # load_lora it might be good"*. An adapter replaces whole matrices, and each lives in a slot the load remembered,
    # so a swap is writing an adapter's records into those slots — and the base's back to unload it — straight from
    # the mapped files through nn's C door. Nothing is kept in RAM but the page cache.
    def _spans(self, bundle, pairs):
        """(slot address, host pointer, bytes) of each matrix's codes and scales in a mapped bundle."""
        out = []
        for l, n in pairs:
            r = bundle.record(l, n)
            view = np.frombuffer(r.bundle.blob(l), dtype=np.uint8)
            base = view.ctypes.data
            at, p = self.slot_at[(l, n)], self.PLAN[n]
            (d_at, d_n), (l_at, l_n) = r.data_span(), r.lut_span()
            out += [(at + p.at_up_data, base + d_at, d_n), (at + p.at_down_lut, base + l_at, l_n)]
        return out

    def _ensure(self, name):
        """An adapter's and the base's spans for the matrices it replaces, and its identity. The bundles stay open."""
        if name in self._lora_cache:
            return
        over = NL.Bundle(self.folder.path, lora=name)
        if over.mask is not None or self.b.mask is not None:
            # ⛳ a rank-1 mask is chosen at boot: its cells are in the tables the procedures were written over
            raise Refused("a rank-1 mask is chosen at boot, not swapped: boot the folder with lora %r" % (name,))
        om = over.overlay
        pairs = [(l, k.split("layers.%d." % l, 1)[1]) for l in om.manifest["overlay"]["layers"] for k in om.index(l)]
        base = self._lora_cache.setdefault(None, NL.Bundle(self.folder.path))
        self._lora_cache[name] = dict(bundles=(over, base), lora=self._spans(over, pairs), base=self._spans(base, pairs),
                                      ident=dict(name=name, weights=over.lora["adapter"]["weights_sha"]))

    def load_lora(self, name):
        """Write an adapter's matrices into their slots, putting back the one in use first. Answers the seconds the
        swap took."""
        import time
        if name == self.active_lora:
            return 0.0
        t0 = time.time()
        self.unload_lora()
        if name is not None:
            self._ensure(name)
            self._write_spans(self._lora_cache[name]["lora"])
            self.active_lora = name
        return time.time() - t0

    def unload_lora(self):
        """The base's matrices back into the slots the adapter in use took."""
        if self.active_lora is None:
            return
        self._ensure(self.active_lora)
        self._write_spans(self._lora_cache[self.active_lora]["base"])
        self.active_lora = None

    def lora_identity(self):
        if self.active_lora is None:
            return None
        if self.b.mask is not None:
            return dict(name=self.b.lora["name"], weights=self.b.lora["adapter"]["weights_sha"])
        self._ensure(self.active_lora)
        return self._lora_cache[self.active_lora]["ident"]

    def _write_spans(self, spans):
        for at, pointer, n in spans:
            self.m.write_from(at, pointer, n)

    # ── ⑥ a conversation's state, for putting it away ─────────────────────────────────────────────────────────
    def state_regions(self, length):
        """(address, bytes) of everything a conversation of `length` positions has written: each DeltaNet layer's
        state and window whole, each attention layer's key and value rows `0..length`."""
        out = []
        for l in self.run_layers:
            at = self.state_at[l]
            if self.kinds[l] == "d":
                out += [(at + self.SPLAN.at_up_data, self.S_BYTES), (at + self.SPLAN.at_down_data, self.W_BYTES)]
            elif self.TIERED:
                past = max(0, length - self.SINK)
                n_rows = dict(sink=min(length, self.SINK), hot=min(past, self.HOT),
                              warm=min(max(0, past - self.HOT), self.WARM), cold=max(0, past - self.HOT - self.WARM))
                w, rb8, rb4 = self.KV_WIDE * 2, (self.AH + 1) // 2 * 2, (self.AH + 3) // 4 * 2
                sizes = dict(sink=n_rows["sink"] * w, hot=n_rows["hot"] * w, warm_c=n_rows["warm"] * self.KVH * rb8,
                             warm_s=n_rows["warm"] * self.KVH * 2, cold_c=n_rows["cold"] * self.KVH * rb4,
                             cold_s=n_rows["cold"] * self.KVH * 2)
                for half in (self.KVPLAN.at_up_data, self.KVPLAN.at_down_data):
                    out += [(at + half + self.TIER[nm][0], n) for nm, n in sizes.items()]
            else:
                n = length * self.KV_WIDE * 2
                out += [(at + self.KVPLAN.at_up_data, n), (at + self.KVPLAN.at_down_data, n)]
        return out
