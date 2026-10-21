"""QWEN 3.5 — the DeltaNet-and-attention family, dense (the 27B) or with routed experts (the 35B-A3B), on one card.

Every size comes from the model folder's `config.json`. The PROGRAM is Lisp, written by hand — `lisp/qwen3_5.lisp`,
which the model folder carries and the boot reads: a layer is `ai_qwen_3`'s words over plane tables the boot fills —
the mixer, `ai_qwen_3__deltanet` or `ai_qwen_3__attention`, its residual included, then the dense MLP (nn's
`nn__mlp__apply`) or the MoE — and `layers` is a loop over them. This class makes what the program reads — the
collections, the load, the buffers, the tables, and the names the source is written against (`_bindings`) — and a token
is one short program:
```
  (begin (position tok pos) (head))
```
— the embedding a table in RAM the card reads a row of: only a row crosses. An answer is `generate` (▶ `generate`),
the same a token at a time on the card until an end token.
The residual stream `x` never leaves the card; `head` answers the greedy token, `head_sample` one drawn there
(▶ `sampling`), or `head_logits` leaves the logits for a sampler on the host.

A conversation's STATE is the DeltaNet layers' recurrent state and conv window and the attention layers' keys and
values up to its length — `state_regions` says where, so a conversation can be put away and brought back; `state_read`
and `state_write` take it whole, in one card's form however many cards hold it.
"""
import json
import re
import os
import pickle

import numpy as np

from . import experts_file, expert_tier, sampling
from .autotune import card_memory
from .vision import video_ram
from .generate import GENERATE_MOST, Generates
from .model_folder import NL, Refused

GU, DN = NL.GATE_UP, NL.DOWN
CARD, CPU = 0, 1                  # the workers: the card, and — with the experts there — the CPU
ARCHITECTURES = ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_vl_text")


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


class Qwen35(Generates):
    PROGRAM = ("qwen3_5", "qwen3_5_cpu", "sampling", "generate", "qwen3_5_pictures", "qwen3_5_tp")   # the pack's lisp/ files (▶ `ModelFolder.program`)
    SAMPLES_ON_CARD = True          # `head_sample` — ▶ `sampling`

    def __init__(self, folder, machine, max_context=4096, fused_moe=True, kv="tiered", tiers=None, experts_on="card",
                 experts_from="auto", experts_dir=None, layers=None, chunk=256, experts_ram_gb=None, card_experts_gb=0,
                 card_landing=1, card_line=None, card_shared=0.1, experts_int8=False, picture_pixels=1 << 22,
                 picture_weights="auto", tensor_parallel=1, lend_cache=True, mtp="no", mtp_rows=False, mtp_pairs=True):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b = folder, machine, b
        self.H, self.LAYERS = c["hidden_size"], c["num_hidden_layers"]
        # ⭐ QWEN 3 (Qwen3-VL's text model, Qwen-Image's encoder) is this family's attention layers alone: no DeltaNet, every
        #   head's rotary whole, no output gate (▶ ai_qwen_3's `AI_QWEN_3__AT__UNGATED`) — its prompt read as rows, one cache
        self.KH, self.VH, self.LH = c.get("linear_num_key_heads", 0), c.get("linear_num_value_heads", 0), c.get("linear_key_head_dim", 0)
        if c.get("linear_value_head_dim", 0) != self.LH:
            raise Refused("the DeltaNet's key and value heads differ in width (%d, %d)" % (self.LH, c["linear_value_head_dim"]))
        self.QH, self.KVH, self.AH = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        self.ROT = int(self.AH * (rope.get("partial_rotary_factor") or c.get("partial_rotary_factor") or 1.0))
        self.GATED = bool(c.get("attn_output_gate", folder.architecture.startswith("qwen3_5")))
        if kv not in ("tiered", "fp16"):
            raise Refused("kv is 'tiered' or 'fp16', not %r" % (kv,))
        if not self.GATED and kv != "fp16":
            raise Refused("an attention without an output gate reads its prompt through one cache: kv 'fp16', not %r" % (kv,))
        # M-RoPE: a token's three positions (time, row, column), interleaved by frequency — the row's sections, then the
        #   column's, in nn's `rope_angles_mrope_rows` word; 0 when the model has none (its pictures are refused)
        section = rope.get("mrope_section") or []
        self.MROPE = (section[1] | (section[2] << 16)) if len(section) == 3 and rope.get("mrope_interleaved") else 0
        self.THETA = int(rope.get("rope_theta") or c["rope_theta"])
        # The norms' epsilon: the program's own norm takes it from here, and ai_qwen_3's words hold it as
        # AI_QWEN_3__RMS_NORM_EPS, so a config that says otherwise is refused rather than half-honoured.
        self.EPS = float(c["rms_norm_eps"])
        if self.EPS != 1e-6:
            raise Refused("the norms' epsilon is %r; ai_qwen_3's words are built for 1e-06" % self.EPS)
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
        # ⭐ TENSOR PARALLEL (NN-87): every layer split over `tensor_parallel` cards — each its share of the heads and of
        #   the MLP's 512-blocks, and the same columns of each projection back, so a card's mixer and MLP are partial sums
        #   its verbs add over the cards' own memory (▶ lisp/qwen3_5_tp.lisp, nn's partials). From here on the head counts
        #   are a card's; FULL keeps the model's. A dense model, the embedding and the head card 0's.
        self.CARDS, self.GROUP = int(tensor_parallel or 1), 0
        self.KV_REP = 1                                     # the cards a key-value head is copied onto (▶ below)
        self.CPU = self.CARDS if self.CARDS > 1 else CPU    # the CPU worker: after a group's cards
        self.BINDINGS = 512 * self.CARDS            # every card's tables are names in the one environment
        self.FULL = dict(KH=self.KH, VH=self.VH, QH=self.QH, KVH=self.KVH)
        if self.CARDS > 1:
            P = self.CARDS
            # ⭐ AN MoE OVER THE CARDS (NN-99 ①): its mixers split by heads as a dense model's, its experts the CPU's — the
            #   MoE block whole on card 0, which hands every card the layer's answer through the group
            #   ⭐ AND A TIER ON EVERY CARD (NN-99 ②): `card_experts_gb` a card, the router run on every card, a position's
            #   picks shared out among the cards that hold them and the rest the CPU's (▶ nn's `nn__expert_tier__visit_group`)
            if (self.MOE and experts_on != "cpu") or (not self.MOE and experts_on != "card") \
                    or layers is not None or b.mask is not None:
                raise Refused("tensor parallelism is a dense model's, its every layer on the cards, or an MoE's with its "
                              "experts on the CPU (experts_on='cpu') — without a mask")
            dense_cut = not self.MOE and (self.INTER % 512 or self.INTER // 512 < P)
            # ⭐ MORE CARDS THAN KEY-VALUE HEADS (the 35B's two over four cards): each card a copy of the one its query
            #   heads read — KV_REP cards a head — so its cache holds that head, and a conversation's state takes it once
            kv_cut = self.KVH % P and P % self.KVH
            if any(n % P for n in (self.KH, self.VH, self.QH)) or kv_cut or dense_cut:
                raise Refused("%d cards do not split %s's heads (%d, %d, %d, %d) and 512-blocks evenly"
                              % (P, folder.path, self.KH, self.VH, self.QH, self.KVH))
            # ⭐ CARDS OF ONE SIZE (⚖ 2026-10-16: "it should only work when the cards have the same size otherwise it is
            #   a nightmare"): every card holds the same share, so the group is sized as one card of them
            sizes = card_memory()
            if sizes is None or len(sizes) < P:
                raise Refused("tensor parallelism over %d cards: the cards' memory cannot be read (rocm-smi or "
                              "nvidia-smi says it)%s" % (P, "" if sizes is None else ": %d cards here" % len(sizes)))
            if len(set(sizes[:P])) > 1:
                raise Refused("tensor parallelism takes cards of one size, and these %d are not: %s GB"
                              % (P, ", ".join("%.1f" % (s / 2 ** 30) for s in sizes[:P])))
            self.KV_REP = max(1, P // self.KVH)
            self.KH, self.VH, self.QH, self.KVH = self.KH // P, self.VH // P, self.QH // P, max(1, self.KVH // P)
            blocks = max(self.INTER // 512, P)          # an MoE's MLP is not cut: card 0's whole
            each = [blocks // P + (1 if k < blocks % P else 0) for k in range(P)]
            self.INTER_AT = [512 * sum(each[:k]) for k in range(P)]
            self.INTER_OF = [512 * n for n in each]
        else:
            self.INTER_AT, self.INTER_OF = [0], [self.INTER]
        self.KV_WIDE = self.KVH * self.AH
        self.CONV_CH = self.KH * self.LH * 2 + self.VH * self.LH
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        # ⭐ A MODEL WITHOUT ITS HEAD — an encoder, its pack made with the head left out (`nn_pack_model.py --leave-out`):
        #   its rows are what it answers (`forward_rows`), and the head's words are never bound
        self.head_rec, self.emb_rec = g.get("lm_head.weight"), g["model.language_model.embed_tokens.weight"]
        self.HEADLESS = self.head_rec is None
        self.VOCAB = int((self.emb_rec if self.HEADLESS else self.head_rec)["rows"])
        # ⭐ THE MULTI-TOKEN PREDICTION LAYER (`mtp_num_hidden_layers` 1) — packed with the model, filed under layer 0 as
        #   `mtp.layers.0.*` (▶ nn_load_bundle's `MTP_PREFIX`), and its `fc` and three norms in global.pkl: `fc` over
        #   [norm(the next token's embedding) | norm(the model's hidden)], an attention layer and its MoE on the residual,
        #   `mtp.norm`, the model's head — the token after next, a draft the model then checks with the next token
        #   (▶ `_mtp_procedures`). Held as the layer after the last (LAYERS) at the end of `run_layers`, so its tables,
        #   its cache and its experts are made as any layer's; `layers` and `rows` run to RUN, before it.
        #   ⚖ 2026-10-19: *"ok do the qwen mtp as a stretch goal"*; its option as GLM's, "no" by default
        self.MTP = None
        if isinstance(mtp, str):
            if mtp.lower() not in ("yes", "no", "true", "false", "1", "0"):
                raise Refused("mtp is 'yes' or 'no', not %r" % (mtp,))
            mtp = mtp.lower() in ("yes", "true", "1")
        # the MTP layer's history, as GLM's (▶ glm5.py): the last position its cache has, the position whose head state is
        # in hB, and (pos, token, draft) to go on with; the prompt's last MTP_HISTORY rows given to it; what drafts did
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None
        self.MTP_HISTORY, self.mtp_stats = 8, dict(passes=0, kept=0)
        # the experts on the CPU: a check's hand-off as two rows (▶ `_mtp_procedures`) — off: ⚖ MEASURED 2026-10-20 node-01,
        # the 35B, two rows' hand-off 0.84x the plain rate and two one-position hand-offs in one picture 1.12x
        self.CHECK_ROWS = bool(mtp_rows)
        self.CHECK_PAIRS = bool(mtp_pairs)            # a check's both positions in one word a site (`deltanet_n`, `moe_n`)
        if mtp:
            L = self.LAYERS
            if int(c.get("mtp_num_hidden_layers", 0)) < 1 or "mtp.fc.weight" not in g \
                    or "self_attn.q_proj.weight" not in b.names(L):
                raise Refused("%s carries no multi-token prediction layer" % folder.path)
            if layers is not None or int(tensor_parallel or 1) > 1 or card_experts_gb or b.mask is not None:
                raise Refused("the MTP layer runs on one machine's every layer, on one card, without a card tier of "
                              "experts or a mask")
            if self.HEADLESS or b.residual_rotated:
                raise Refused("the MTP layer reads the model's head and its residual unrotated")
            self.MTP = L
            self.kinds[L] = "f"
            self.run_layers.append(L)
            fc = g["mtp.fc.weight"]
            self.fc_data = np.asarray(fc["vbr_data"]).tobytes()
            self.fc_lut = np.asarray(fc["vbr_lut"]).astype("<f2").tobytes()
            self.FCPLAN, self.FC_BITS = NL.SlotPlan(len(self.fc_data), 0, 0, len(self.fc_lut)), int(fc["d"])
        # ⭐ PICTURES' FORMS, a file of their own that a runtime before them never reads — so a pack's qwen3_5.lisp stays
        #   one every runtime runs; a folder without it reads text, and refuses pictures (▶ `vision`). Decided at the boot,
        #   as a boot that reads `programs.lisp` never calls `_procedures`; the file is in that file's key, so what it
        #   holds was composed with the pictures exactly when the folder has them
        self.PICTURES = os.path.isfile(os.path.join(folder.path, "lisp", "qwen3_5_pictures.lisp"))
        # ⭐ PICTURES ON THE CARD (NN-78): the vision tower's weights in a collection of their own and its scratch in another,
        #   sized for pictures of up to `picture_pixels` (0: none, and none of the card's memory for them) — where the
        #   folder has the program's pictures file, the pack the tower, the model M-RoPE, and this machine the first layers
        self.VIS = bool(self.PICTURES and self.MROPE and picture_pixels and (layers is None or 0 in layers)
                        and "model.visual.blocks.0.attn.qkv.weight" in g)
        if self.CARDS > 1:
            self.VIS = False                # the vision tower is one card's: not under tensor parallelism yet
        self.PICTURE_PIXELS = int(picture_pixels or 0)
        if picture_weights not in ("auto", "card", "ram"):
            raise Refused("picture_weights is 'auto', 'card' or 'ram', not %r" % (picture_weights,))
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
        # ⭐ THE TOWER'S WEIGHTS (~0.6 GB packed) ON THE CARD OR IN RAM — `REASONED`: with the experts on the CPU the RAM is
        #   theirs and the card has room, so the card; with the model on the card the card is the scarce memory (the 27B
        #   at 32,768 positions leaves a 16 GB card 1 GB), so RAM, a word's planes written to the card before it runs; a
        #   machine whose worker is a CPU has the one memory, so its slot
        self.VIS_RAM = self.VIS and (picture_weights == "ram" or (picture_weights == "auto" and not self.CPU_EXPERTS
                                                                  and machine.silicon not in ("host", "x86_avx2")))
        # ⭐ THE CARD'S TIER OF ROUTED EXPERTS (NN-48) — the experts on the CPU, and `card_experts_gb` of them on the card as
        #   well: the card computes the picks it holds, the CPU the rest; an expert the CPU computed earns a slot by nn's
        #   qualifier and is copied whole from the CPU's memory while the card computes (`card_landing` a visit at most); a
        #   prompt chunk gives the card every expert past `nn__expert_major__vram_promotion_threshold_picks`, measured at boot
        #   (`card_line` fixes it). ▶ glm5.py's, and nn's `nn__expert_tier__*`
        self.CARD_GB, self.CARD_LANDING = float(card_experts_gb or 0), int(card_landing)
        self.CARD_LINE = int(card_line) if card_line else None
        # ⭐ A GROUP'S SHARED SLICE (NN-99 ③): `card_shared` of a card's slots a layer kept on every card of a tensor-parallel
        #   group — the hottest, chosen again from decayed picks (▶ nn's `NN__EXPERT_TIER__SHARED`); 0, none
        self.CARD_SHARED = float(card_shared)
        # ⭐ THE CACHE'S ROWS NO CONVERSATION HAS REACHED, LENT TO THE TIER (NN-93) — ▶ `_lend`. `lend_cache` false keeps
        #   the tier to its own run
        self.LEND_ON = bool(lend_cache)
        # a position's experts on the CPU multiplied in integers (nn's int8 gemvs, as GLM's) rather than exactly — the
        # prompt's rows there are in integers either way. ▶ AI_QWEN_3__EXP__INT8
        self.EXPERTS_INT8 = bool(experts_int8)
        if self.EXPERTS_INT8 and not self.CPU_EXPERTS:
            raise Refused("experts_int8 is the CPU's arithmetic: experts_on 'cpu'")
        if self.CARD_GB and not self.CPU_EXPERTS:
            raise Refused("a card's tier of experts is beside the CPU's — experts_on='cpu'")
        if self.CARD_GB and b.mask is not None:
            raise Refused("a card's tier of experts does not take a mask yet")
        self._plan()
        if self.CPU_EXPERTS:
            self.EXPERTS_FROM = experts_file.choose(experts_from, self.cpu_arena, video_ram(folder.config))
            self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        # a card's tier copies an expert from its place in the CPU's memory, which only the experts loaded into RAM have
        # (`SRC`): from the disk or mapped it would ask for experts and never hold one — `MEASURED` 2026-10-12 from the disk
        # (measurements/2026-10-12_nvme_array_122b_glm.md), `REASONED` mapped, from `SRC`
        if self.CARD_GB and self.EXPERTS_FROM != "ram":
            raise Refused("a card's tier of experts is copied from the experts in RAM — experts_from 'ram', not %r"
                          % self.EXPERTS_FROM)
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

    # ── tensor parallel: how card `s` cuts a matrix ───────────────────────────────────────────────────────────────
    def _cut(self, n, s):
        """Card `s`'s part of the matrix `n`: None for whole (a norm, one card), ("rows", runs) or ("cols", (first, end)).
        Every cut is whole rows, or whole 512-blocks of a row — the blocks its input is rotated in — so no part pads."""
        if self.CARDS == 1 or n.endswith("norm.weight") or (self.MOE and n.startswith("mlp.")):
            return None                 # an MoE's block is card 0's, whole
        F, LH, AH = self.FULL, self.LH, self.AH
        kp, vp, qp, kvp = self.KH, self.VH, self.QH, self.KVH
        i0, ip = self.INTER_AT[s], self.INTER_OF[s]
        chans = [(s * kp * LH, kp * LH), (F["KH"] * LH + s * kp * LH, kp * LH), (2 * F["KH"] * LH + s * vp * LH, vp * LH)]
        return {"linear_attn.in_proj_qkv.weight": ("rows", chans),
                "linear_attn.conv1d.weight":      ("rows", chans),
                "linear_attn.in_proj_z.weight":   ("rows", [(s * vp * LH, vp * LH)]),
                "linear_attn.in_proj_a.weight":   ("rows", [(s * vp, vp)]),
                "linear_attn.in_proj_b.weight":   ("rows", [(s * vp, vp)]),
                "linear_attn.A_log":              ("cols", (s * vp, (s + 1) * vp)),
                "linear_attn.dt_bias":            ("cols", (s * vp, (s + 1) * vp)),
                "linear_attn.out_proj.weight":    ("cols", (s * vp * LH, (s + 1) * vp * LH)),
                "self_attn.q_proj.weight":        ("rows", [(s * qp * 2 * AH, qp * 2 * AH)]),
                "self_attn.k_proj.weight":        ("rows", [(s // self.KV_REP * kvp * AH, kvp * AH)]),
                "self_attn.v_proj.weight":        ("rows", [(s // self.KV_REP * kvp * AH, kvp * AH)]),
                "self_attn.o_proj.weight":        ("cols", (s * qp * AH, (s + 1) * qp * AH)),
                "mlp.gate_proj.weight":           ("rows", [(i0, ip)]),
                "mlp.up_proj.weight":             ("rows", [(i0, ip)]),
                "mlp.down_proj.weight":           ("cols", (i0, i0 + ip))}[n]

    def _moe_block(self, n):
        """Whether `n` is an MoE layer's block after its mixer — its norm, router and shared expert, card 0's alone."""
        return n.startswith("mlp.") or n == "post_attention_layernorm.weight"

    def _part(self, l, n, s):
        """Card `s`'s bytes of layer `l`'s matrix `n` (data, scales), or None when it holds it whole. An MLP part of fewer
        512-blocks than card 0's is padded at its end to card 0's size — the slot's — and its table's width says how much
        of it is read."""
        c = self._cut(n, s)
        if c is None:
            return None
        part = lambda k: NL.dense_part(self.b, n, l, **{c[0]: self._cut(n, k)[1]})
        data, lut = part(s)
        if s > 0 and n.startswith("mlp."):
            data0, lut0 = part(0)
            data, lut = data + bytes(len(data0) - len(data)), lut + bytes(len(lut0) - len(lut))
        return data, lut

    # ── ① the collections: a type a dense tensor, the experts, the head, the embedding, the two states ──────────
    def _plan(self):
        b, L, run = self.b, self.LAYERS, self.run_layers
        names = sorted({n for l in run for n in b.names(l)} - ({GU, DN} if self.MOE else set()))
        # ⭐ A COLLECTION A MATRIX NAME AND WIDTH: a pack that keeps some layers' matrices exact (16 bits — Qwen-Image's
        #   encoder, its last layers) holds them in a collection of their own, the slots of each the same size
        self.KEY = {(l, n): (n, b.record(l, n).d) for l in run for n in b.names(l) if n in names}
        keys = sorted(set(self.KEY.values()))
        self.TYPE = {k: i for i, k in enumerate(keys)}
        first = {k: next(l for l in run if self.KEY.get((l, k[0])) == k) for k in keys}
        self.PLAN = {}
        for k in keys:
            got = self._part(first[k], k[0], 0)
            self.PLAN[k] = NL.dense_plan(b, k[0], first[k]) if got is None else NL.SlotPlan(len(got[0]), 0, 0, len(got[1]))
        slots = {k: sum(1 for l in run if self.KEY.get((l, k[0])) == k) for k in keys}
        t = len(keys)
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
        if self.HEADLESS:                   # a slot of zeros where the head would be: the collections keep their shape
            self.head_data, self.head_lut, self.HPLAN = bytes(256), bytes(256), NL.SlotPlan(256, 0, 0, 256)
        else:
            self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, self.BPLAN = plan_of(self.emb_rec)
        # what the program reads of the two tables once the engine holds them and this program does not
        self.EMB_ROW_BYTES, self.EMB_BITS = int(self.emb_rec["row_bytes"]), int(self.emb_rec["d"])
        self.EMB_DATA_BYTES, self.HEAD_BITS = len(self.emb_data), 8 if self.HEADLESS else int(self.head_rec["d"])
        # ⭐ THE EMBEDDING, A TABLE IN RAM THE CARD READS IN PLACE: the rows, then the scales — a RAM buffer of its own class
        self.TABLE_BYTES = (len(self.emb_data) + len(self.emb_lut) + 255) // 256 * 256
        self.S_BYTES, self.W_BYTES = self.VH * self.LH * self.LH * 4, self.CONV_CH * 3 * 2
        # a model with no DeltaNet layer (Qwen 3) keeps the collection, one slot of a few bytes: nn takes none empty
        self.SPLAN = NL.SlotPlan(self.S_BYTES, 0, 0, self.W_BYTES) if self.S_BYTES else NL.SlotPlan(256, 0, 0, 256)
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
        coll = [(self.TYPE[k], self.PLAN[k], slots[k]) for k in keys]
        self.cpu_arena = 0
        if self.MOE:
            self.EPLAN = NL.expert_plan(b, 0, self.EXPERTS)
            if on_card:
                coll.append((self.T_EXPERT, self.EPLAN, self.EXPERTS * len(run)))
            else:
                self.cpu_arena = self.EPLAN.slot_bytes * self.EXPERTS * len(run)
        coll += [(self.T_HEAD, self.HPLAN, 1), (self.T_STATE, self.SPLAN, max(n_d, 1)),
                 (self.T_KV, self.KVPLAN, len(run) - n_d)]
        after = self.T_KV + 1
        if self.VIS:
            self.T_VIS, self.T_VSCR = after, after + 1
            self._vision_plan()
            coll += [(self.T_VIS, self.VPLAN, 1), (self.T_VSCR, self.VSPLAN, 1)]
            after += 2
        # the card's tier: whole experts, laid out as the CPU's, so a promotion copies a slot as it is
        self.CPLAN, self.T_CEXP, self.CARD_SLOTS, self.SRC = None, None, 0, {}
        if self.CARD_GB and self.MOE:
            self.CPLAN, self.T_CEXP = self.EPLAN, after
            self.CARD_SLOTS = int(self.CARD_GB * (1 << 30)) // self.EPLAN.slot_bytes
            self.TIER_LEAST = len(run) * self.EPLAN.slot_bytes             # the least tier it takes: a slot a layer
            if self.CARD_SLOTS < len(run):
                raise Refused("%.1f GB holds %d experts, fewer than one a layer" % (self.CARD_GB, self.CARD_SLOTS))
            coll.append((self.T_CEXP, self.CPLAN, self.CARD_SLOTS))
        # ⭐ WHAT THE TIER MAY BE LENT (NN-93): the tail of each attention layer's warm keys and values — the rows of the
        #   warm ring past those a conversation has written — `most` slots a region at the most, two regions a layer
        self.LEND = None
        if self.CPLAN is not None and self.TIERED and self.CARDS == 1 and self.LEND_ON:
            row, attn = self.KVH * ((self.AH + 1) // 2 * 2), sum(1 for l in run if self.kinds[l] == "f")
            most = self.WARM * row // self.CPLAN.slot_bytes
            if attn and most:
                self.LEND = dict(row=row, regions=2 * attn, most=most)
        # the MTP layer's `fc`, a collection of its own as the head's: it is no layer's record
        self.T_FC = None
        if self.MTP is not None:
            self.T_FC = max(t for t, _, _ in coll) + 1
            coll.append((self.T_FC, self.FCPLAN, 1))
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
                  (self.CHUNK * self.QH * self.max_context * 4) if not self.TIERED else 0,
                  self.ROWS_SCRATCH, self.WINDOW, self.WINDOW_SCORES)
        # ⭐ a rank-1 mask lives in one buffer a worker, of the largest class, which it grows by one
        self._mask_layout()
        big = max(big, self.MASK_BYTES[CARD])
        # the card's tier over a chunk: `experts_rows`' scratch and the rows' share, two more of the largest class
        tier = 0
        if self.CPLAN is not None:
            P, n = self.CHUNK * self.TOPK, self.CHUNK
            self.KRSCRATCH = 2 * n * self.H + 16 * P + 2 * P + 8 * P * self.INTER + 4 * n * self.TOPK * self.H + 4096
            big, tier = max(big, self.KRSCRATCH), 2
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in self.collections)
        if self.LEND:                                  # the links for the slots the tier may be lent (▶ nn's `__lendable`)
            types += "nn__expert__type%d__lendable:%d\n" % (self.T_CEXP, self.LEND["regions"] * self.LEND["most"])
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
                "nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n"
                % (big, 6 + (self.MASK_BYTES[CARD] > 0) + tier, (self.arena + (256 << 20)) >> 20, len(self.kinds) - n_d, n_d,
                   self._layer_types(), len(self.collections), types, self.TABLE_BYTES))
        card = self._pools(card, self._mtp_asks())
        cards = {k: card for k in range(self.CARDS)}          # every card the same collections, each its own part
        if not self.CPU_EXPERTS:
            return cards if self.CARDS > 1 else card
        # the CPU worker's section: its one collection, the experts, and small pools for the hand-off's buffers — and a
        # prompt's: its hand rows, its routed rows and `experts_rows`' scratch (▶ ai_qwen_3's `experts_rows`)
        p = self.EPLAN
        P, I = self.CHUNK * self.TOPK, self.INTER
        self.RSCRATCH = 2 * self.CHUNK * self.H + 16 * P + 2 * P + 8 * P * I + 4 * self.CHUNK * self.TOPK * self.H + 64
        cbig = max(1 << 20, self.RSCRATCH, self.CHUNK * self.HAND_ROW, self.MASK_BYTES[self.CPU])
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
               % (cbig, 5 + (self.MASK_BYTES[self.CPU] > 0), (self.cpu_arena + (256 << 20)) >> 20, len(self.kinds) - n_d, n_d,
                  self._layer_types(), p.up_bytes, p.down_bytes,
                  self.SLOTS if self.SLOTS is not None else self.EXPERTS * len(self.run_layers)))
        cpu = self._pools(cpu, [2 * (self.H + 16), self.H * 2] if self.MTP is not None else [])   # c_hand2, c_routed2
        return {**cards, self.CPU: cpu}

    def _index_layers(self):
        """The model's layers, as the experts' file names them — the MTP layer's place after them where it is held."""
        return list(range(self.LAYERS + (1 if self.MTP is not None else 0)))

    def _layer_types(self):
        """Every layer's kind, the index's places — and the MTP layer's after them, where it is held."""
        return "".join(self.kinds[l] for l in range(self.LAYERS + (1 if self.MTP is not None else 0)))

    @staticmethod
    def _pools(section, asks):
        """A device section with a buffer more in the class each of `asks` (bytes) lands in: nn takes the smallest class
        that fits and never a larger one (▶ nn's `zzprivate_take`), the first of two the same size."""
        sizes, order = {}, ("resid", "main", "attn_out", "logits", "partials")
        for k in order:
            sizes[k] = int(re.search(r"nn__buffer_%s__size_bytes:(\d+)" % k, section).group(1))
        more = dict.fromkeys(order, 0)
        for n in asks:
            fits = [k for k in order if sizes[k] >= n]
            if not fits:
                raise Refused("a %d-byte buffer fits none of the card's buffer classes" % n)
            more[min(fits, key=lambda k: (sizes[k], order.index(k)))] += 1
        for k in order:
            if more[k]:
                section = re.sub(r"(nn__buffer_%s__qty:)(\d+)" % k, lambda mt: mt.group(1) + str(int(mt.group(2)) + more[k]),
                                 section, count=1)
        return section

    def _mtp_asks(self):
        """The bytes of each card buffer the MTP layer and a draft's check make (▶ `_buffers`)."""
        if self.MTP is None:
            return []
        H, n_d = self.H, sum(1 for l in self.run_layers if self.kinds[l] == "d")
        small = [H * 2] * 11 + [H * 4] * 4 + ([2 * (H + 16 + self.INTER)] if self.MOE else [])   # xp and hp: two rows each
        return small + [self.S_BYTES] * n_d + [self.W_BYTES] * n_d + [4 * self.VOCAB]

    def workers(self):
        """The machine's workers: the card, and the CPU when the experts are there — or the cards of a tensor-parallel
        group, worker `k` device `k`."""
        if self.CARDS > 1:
            return [(self.m.silicon, k) for k in range(self.CARDS)] + ([("x86_avx2", 0)] if self.CPU_EXPERTS else [])
        return [(self.m.silicon, 0)] + ([("x86_avx2", 0)] if self.CPU_EXPERTS else [])

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        if self.CARDS > 1:
            return self._load_cards(progress)
        m, b = self.m, self.b
        loader = m.loader()
        self._head_and_embedding(loader)
        if self.MTP is not None:
            loader.layer_width(0, self.T_FC, 1)
            loader.load_bytes(self.FCPLAN, self.T_FC, 0, self.fc_data, self.fc_lut)
            self.fc_data = self.fc_lut = None
        if self.VIS:
            self._vision_load(loader)
        self.state_at = {}
        self.state_of = [self.state_at]          # each card's states, where a group's cards each hold their part
        self.slot_at = {}              # (layer, name) -> the slot a dense matrix lives in, for a LoRA swap
        for i, l in enumerate(self.run_layers):
            for n in b.names(l):
                if self.MOE and n in (GU, DN):
                    continue
                k = self.KEY[(l, n)]
                loader.layer_width(l, self.TYPE[k], 1)
                self.slot_at[(l, n)] = loader.load_dense(b, self.PLAN[k], self.TYPE[k], n, l)
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
        if self.CPLAN is not None:                     # the card's tier, every layer's band of it, empty at boot
            for l in self.run_layers:
                loader.layer_width(l, self.T_CEXP, self.EXPERTS)
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())
        self._buffers()
        self.MASK_CELLS = {}
        self._mask_buffers(cpu=False)
        self._tables()
        if self.CPU_EXPERTS:
            self._cpu_experts(progress)
        if self.CPLAN is not None:
            self._tier_tables()
            self._lend()
        self._bindings()
        if self.VIS:
            self._vision_bindings()
        m.define_programs(self._procedures)
        if self.CPLAN is not None:
            self.CALIBRATION = expert_tier.calibrate(
                m, CARD, self.CPU, self.KRS_AT, self.EPLAN.slot_bytes, self.C_HANDS_AT, self.HAND_ROW, 2 * self.H,
                (2 * (self.H + 16 + self.INTER) + 7) & ~7, self.TOPK, self.EXPERTS,
                lambda n: "(ai_qwen_3__experts_rows c_hands exrc%d c_rrows %d)" % (self.run_layers[0], n),
                self.CHUNK, fixed=self.CARD_LINE)
        self.active_lora, self._lora_cache = self.folder.lora, {}     # a LoRA chosen at boot is in the slots already
        self.global_ = None                         # what was left of global.pkl, read by the buffers
        self.b.release()                            # every layer's file let go (▶ `Bundle.release`)

    def _load_cards(self, progress=None):
        """Tensor parallel: each card its part of every layer, its buffers and its tables, as that card's worker; card 0
        the head and the embedding as well. Then every card joins the group — refused when two of them cannot reach each
        other's memory: ⚖ "tensor parallelism requires DMA" — and the program is bound and read once, as card 0."""
        m, b, P = self.m, self.b, self.CARDS
        self.state_of, self.slot_at = [{} for _ in range(P)], {}
        for k in range(P):
            m.act_as(k)
            loader = m.loader()
            if k == 0:
                self._head_and_embedding(loader)
            for i, l in enumerate(self.run_layers):
                for n in b.names(l):
                    if self.MOE and (n in (GU, DN) or (k > 0 and self.CPLAN is None and self._moe_block(n))):
                        continue                    # the experts the CPU's; the rest of the block card 0's, or every
                                                    # card's where each holds a tier (its router its own)
                    key = self.KEY[(l, n)]
                    loader.layer_width(l, self.TYPE[key], 1)
                    got = self._part(l, n, k)
                    if got is None:
                        loader.load_dense(b, self.PLAN[key], self.TYPE[key], n, l)
                    else:
                        loader.load_bytes(self.PLAN[key], self.TYPE[key], l, got[0], got[1])
                t = self.T_STATE if self.kinds[l] == "d" else self.T_KV
                loader.layer_width(l, t, 1)
                self.state_of[k][l] = loader.reserve(t)
                loader.assign(l, t, 0, self.state_of[k][l])
                if progress:
                    progress(k * len(self.run_layers) + i + 1, P * len(self.run_layers))
            if self.CPLAN is not None:                 # this card's tier, every layer's band of it, empty at boot
                for l in self.run_layers:
                    loader.layer_width(l, self.T_CEXP, self.EXPERTS)
            if loader.handed() != 0:
                raise Refused("card %d: %d slots were reserved and never published" % (k, loader.handed()))
            self._buffers(k)
            self.MASK_CELLS = {}
            self._tables(k)
        fan = min(4, P)
        for k in range(P):
            m.act_as(k)
            m.must("(nn__partials__join %d %d %d %d)" % (self.GROUP, k, P, fan), "card %d joining the group" % k,
                   "sys__value_true")
        if self.CPU_EXPERTS:
            self._cpu_experts(progress)
        if self.CPLAN is not None:
            for k in range(P):
                m.act_as(k)
                self._tier_tables(k)
            expert_tier.bind(m, {k: self._named("tier_card", k) for k in range(P)}, len(self.workers()))
        m.act_as(-1)
        self.state_at = self.state_of[0]
        self._bindings()
        m.define_programs(self._procedures)
        if self.CPLAN is not None:          # a prompt's rows through card 0's tier: its line, as one card's (▶ `load`)
            self.CALIBRATION = expert_tier.calibrate(
                m, CARD, self.CPU, self.KRS_AT, self.EPLAN.slot_bytes, self.C_HANDS_AT, self.HAND_ROW, 2 * self.H,
                (2 * (self.H + 16 + self.INTER) + 7) & ~7, self.TOPK, self.EXPERTS,
                lambda n: "(ai_qwen_3__experts_rows c_hands exrc%d c_rrows %d)" % (self.run_layers[0], n),
                self.CHUNK, fixed=self.CARD_LINE)
        self.active_lora, self._lora_cache = self.folder.lora, {}
        self.global_ = None
        self.b.release()                            # every layer's file let go (▶ `Bundle.release`)

    # ── pictures: the vision tower on the card ───────────────────────────────────────────────────────────────
    def _vision_plan(self):
        """Where the tower's planes go, and its scratch's. Every record of global.pkl's `model.visual.*` is a plane as the
        pack keeps it — a matrix's codes and its scales, a vector's halves — but the patch embedding, whose rows of 16 no
        product takes, which becomes a matrix of halves (1152 rows of 1536), and the position table, which stays on the
        host for each picture's resampling (▶ `vision.Vision`). The planes are in groups, a word's each — the embedding's,
        a block's, the merger's: on the card (`picture_weights` "card") the groups lie one after another in the tower's
        slot; in RAM ("ram") they lie so in the host's memory, and the slot is one group's room, which each word's planes
        are written into before it runs."""
        vc = json.load(open(os.path.join(self.folder.path, "config.json")))["vision_config"]
        self.VC = vc
        g, self.VREC, self.D_VIS = self.global_, {}, {}
        group_of = lambda short: ("embed" if short.startswith("patch_embed.") else "merger" if short.startswith("merger.")
                                  else int(short.split(".")[1]))
        self.VGROUPS = ["embed"] + list(range(vc["depth"])) + ["merger"]
        size = {k: 0 for k in self.VGROUPS}

        def place(group, name, nbytes):
            self.VREC[name] = (group, size[group], nbytes)
            size[group] += (nbytes + 255) // 256 * 256
        for name in sorted(k for k in g if k.startswith("model.visual.")):
            short, rec = name[len("model.visual."):], g[name]
            if short == "pos_embed.weight":
                continue
            if short == "patch_embed.proj.weight":
                rows = int(rec["original_shape"][0])
                place("embed", short, rows * (int(np.prod(rec["original_shape"])) // rows) * 2)
                self.D_VIS[short] = 16
                continue
            place(group_of(short), short, len(np.asarray(rec["vbr_data"]).tobytes()))
            self.D_VIS[short] = int(rec["d"])
            if int(rec["d"]) != 16:
                place(group_of(short), short + ":lut", 2 * int(rec["rows"]))
        self.VHOST, at = {}, 0                         # a group's place among them all, and its bytes
        for k in self.VGROUPS:
            self.VHOST[k] = (at, size[k])
            at += size[k]
        self.VIS_BYTES = at
        self.VCARD = ({k: 0 for k in self.VGROUPS} if self.VIS_RAM else {k: self.VHOST[k][0] for k in self.VGROUPS})
        room = max(size.values()) if self.VIS_RAM else at
        self.VPLAN = NL.SlotPlan(room, 0, 0, 256)          # every plane in the up half (nn takes no empty half)
        H, P = vc["hidden_size"], vc["patch_size"]
        M = vc["spatial_merge_size"] ** 2
        self.VIS_ROWS = max(M, self.PICTURE_PIXELS // (P * P) // M * M)     # the most patches a picture is made into
        self.VIS_CHUNK = 1024                                                # the MLP's rows a step
        n, I, hd, out = self.VIS_ROWS, vc["intermediate_size"], H // vc["num_heads"], vc["out_hidden_size"]
        IN = 3 * vc["temporal_patch_size"] * P * P
        pairs, step = (8 * n + 15) & ~15, min(self.VIS_CHUNK, n)
        # ▶ `ai_qwen_3__vision_block`'s scratch: the pairs, h, hrot (a row of the MLP's width where its second matrix is
        #   rotated), q k v, the MLP's two, the angles
        rot = max(n * H, step * I)
        scratch = pairs + 2 * (4 * n * H + rot + step * (I + H)) + 4 * n * hd
        self.VSCR, at = {}, 0
        for name, nbytes in (("x", 2 * n * H), ("pix", 2 * n * IN), ("pos", 2 * n * H), ("rc", 8 * n),
                             ("out", 2 * (n // M) * out), ("scratch", scratch)):
            self.VSCR[name] = (at, nbytes)
            at += (nbytes + 255) // 256 * 256
        self.VSPLAN = NL.SlotPlan(at, 0, 0, 256)

    def _vision_load(self, loader):
        """The tower's planes into its slot — or into the host's memory, its slot then a group's room — each record let go
        of once it is there; the position table decoded to the host's floats; the scratch's slot reserved."""
        from .vision import decoded
        g = self.global_
        loader.layer_width(0, self.T_VIS, 1)
        self.VIS_AT = loader.reserve(self.T_VIS)
        self.VIS_HOST = np.zeros(self.VIS_BYTES, np.uint8) if self.VIS_RAM else None

        def put(name, data):
            group, at, _ = self.VREC[name]
            if self.VIS_RAM:
                a = self.VHOST[group][0] + at
                self.VIS_HOST[a:a + len(data)] = np.frombuffer(data, np.uint8)
            else:
                loader.write(self.VIS_AT + self.VCARD[group] + at, data)
        for name in sorted(k for k in g if k.startswith("model.visual.")):
            short, rec = name[len("model.visual."):], g.pop(name)
            if short == "pos_embed.weight":
                self.POS_TABLE = decoded(rec).astype(np.float32)
                continue
            if short == "patch_embed.proj.weight":
                put(short, decoded(rec).reshape(int(rec["original_shape"][0]), -1).astype("<f2").tobytes())
                continue
            put(short, np.asarray(rec["vbr_data"]).tobytes())
            if int(rec["d"]) != 16:
                put(short + ":lut", np.asarray(rec["vbr_lut"]).astype("<f2").tobytes())
        loader.assign(0, self.T_VIS, 0, self.VIS_AT)
        loader.layer_width(0, self.T_VSCR, 1)
        self.VSCR_AT = loader.reserve(self.T_VSCR)
        loader.assign(0, self.T_VSCR, 0, self.VSCR_AT)

    def _vision_bindings(self):
        """The tower's tables — the embedding's, a block's each, the merger's — and the scratch's planes."""
        m, vc = self.m, self.VC
        H, heads = vc["hidden_size"], vc["num_heads"]

        def w(name):
            group, at, nbytes = self.VREC[name]
            return "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_VIS, self.VCARD[group] + at, nbytes)
        sp = lambda name: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_VSCR,) + self.VSCR[name])
        d = lambda name: int(self.D_VIS[name])
        IN = 3 * vc["temporal_patch_size"] * vc["patch_size"] ** 2
        # a matrix's scales below 16 bits; at 16 the cell is not read, and its bias stands in
        lut = lambda n: w(n + ":lut") if d(n) != 16 else w(n.replace("weight", "bias"))
        m.table("VE", [w("patch_embed.proj.weight"), w("patch_embed.proj.bias"), w("patch_embed.proj.bias"), sp("scratch"),
                       H, IN, 16])
        for i in range(vc["depth"]):
            p = "blocks.%d." % i
            m.table("VB%d" % i, [w(p + "norm1.weight"), w(p + "norm1.bias"),
                                 w(p + "attn.qkv.weight"), lut(p + "attn.qkv.weight"), w(p + "attn.qkv.bias"),
                                 w(p + "attn.proj.weight"), lut(p + "attn.proj.weight"), w(p + "attn.proj.bias"),
                                 w(p + "norm2.weight"), w(p + "norm2.bias"),
                                 w(p + "mlp.linear_fc1.weight"), lut(p + "mlp.linear_fc1.weight"),
                                 w(p + "mlp.linear_fc1.bias"),
                                 w(p + "mlp.linear_fc2.weight"), lut(p + "mlp.linear_fc2.weight"),
                                 w(p + "mlp.linear_fc2.bias"),
                                 "signs", sp("scratch"),
                                 H, heads, H // heads, vc["intermediate_size"], d(p + "attn.qkv.weight"),
                                 d(p + "attn.proj.weight"), d(p + "mlp.linear_fc1.weight"), d(p + "mlp.linear_fc2.weight"),
                                 int((vc.get("rope_parameters") or {}).get("rope_theta", 10000)), self.VIS_CHUNK])
        m.table("VM", [w("merger.norm.weight"), w("merger.norm.bias"),
                       w("merger.linear_fc1.weight"), lut("merger.linear_fc1.weight"), w("merger.linear_fc1.bias"),
                       w("merger.linear_fc2.weight"), lut("merger.linear_fc2.weight"), w("merger.linear_fc2.bias"),
                       "signs", sp("scratch"), H, vc["spatial_merge_size"] ** 2, vc["out_hidden_size"],
                       d("merger.linear_fc1.weight"), d("merger.linear_fc2.weight")])
        self.VPLANES = {k: sp(k) for k in ("x", "pix", "pos", "rc", "out")}

    def vision_rows(self, pixel_values, rowcol, pos, blocks=None):
        """A picture's patches through the tower on the card: `pixel_values` (patches x 1536), each patch's row and column
        (`rowcol`, patches x 2) and its position table's rows (`pos`, patches x 1152) in; the tokens' rows (patches / 4 x
        the text model's width, float32) out — or, given `blocks`, the patches' residual after that many blocks (a check's
        stage). With the weights in RAM, a word's planes are written to the card before it runs: a write follows the
        words before it on the card (the families' `memory_write`), so the slot is free when it is written."""
        n, vc = len(pixel_values), self.VC
        M = vc["spatial_merge_size"] ** 2
        if n > self.VIS_ROWS:
            raise Refused("%d patches, and this boot holds %d (picture_pixels %d)" % (n, self.VIS_ROWS, self.PICTURE_PIXELS))
        m, base = self.m, self.VSCR_AT
        m.write(base + self.VSCR["pix"][0], np.ascontiguousarray(pixel_values, dtype="<f2").tobytes())
        m.write(base + self.VSCR["pos"][0], np.ascontiguousarray(pos, dtype="<f2").tobytes())
        m.write(base + self.VSCR["rc"][0], np.ascontiguousarray(rowcol, dtype="<u4").tobytes())
        P = self.VPLANES
        depth = vc["depth"] if blocks is None else blocks
        words = ([("embed", "(ai_qwen_3__vision_embed %s %s VE %d %s)" % (P["pix"], P["pos"], n, P["x"]))]
                 + [(i, "(ai_qwen_3__vision_block %s %s VB%d %d)" % (P["x"], P["rc"], i, n)) for i in range(depth)]
                 + ([("merger", "(ai_qwen_3__vision_merger %s VM %d %s)" % (P["x"], n, P["out"]))] if blocks is None else []))
        for group, word in words:
            if self.VIS_RAM:
                at, nbytes = self.VHOST[group]
                m.write_from(self.VIS_AT, self.VIS_HOST.ctypes.data + at, nbytes)
            # a word's answer is its out plane, or the error that says why it did not run
            if m.must("(type %s)" % word, "the tower's word %s" % word.split()[0][1:]) == self.m.K["sys__error"]:
                raise Refused("the tower's word %s answered an error" % word.split()[0][1:])
        if blocks is not None:
            x = m.read(base + self.VSCR["x"][0], n * vc["hidden_size"] * 2)
            return np.frombuffer(x, dtype="<f2").reshape(n, -1).astype(np.float32)
        out = m.read(base + self.VSCR["out"][0], (n // M) * vc["out_hidden_size"] * 2)
        return np.frombuffer(out, dtype="<f2").reshape(n // M, -1).astype(np.float32)

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
            self.global_.pop(name, None)
        self.head_rec = self.emb_rec = None
        self.head_data = self.head_lut = self.emb_data = self.emb_lut = None

    def _expert_pieces(self, view, l, x, part):
        """Expert `x` of layer `l` as its slot's four pieces (▶ `experts_file`) — the loader's own slices: runs of the
        mapped layer `view` where the pack holds the record, so the disk tier can read them where they lie; a LoRA's
        record, in a file of its own, as a copy."""
        b, p, out = self.b, self.EPLAN, []
        for rec, data_into, lut_into in ((b.record(l, GU), p.at_up_data, p.at_up_lut),
                                         (b.record(l, DN), p.at_down_data, p.at_down_lut)):
            if rec.bundle is b:
                per = rec.expert_rows(self.EXPERTS)
                (d, _), (s, _) = rec.data_span(), rec.lut_span()
                d, s = d + x * per * rec.row_bytes, s + x * per * 2
                out += [(data_into, view[d:d + per * rec.row_bytes]), (lut_into, view[s:s + per * 2])]
            else:
                data, lut = rec.expert_slices(x, self.EXPERTS)
                out += [(data_into, np.frombuffer(data, dtype=np.uint8)), (lut_into, np.frombuffer(lut, dtype=np.uint8))]
        return out

    def _cpu_experts(self, progress):
        """The CPU worker: every layer's experts in its memory or mapped from their file, the hand-off's buffers, each
        layer's experts table and the picture the card hands it."""
        m, H, e = self.m, self.H, self.EPLAN
        layers = self.run_layers
        backing = {}
        if self.EXPERTS_FROM == "disk":
            backing = experts_file.cache_experts(m, self.b, [(self.CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                                 self._expert_pieces, self._index_layers(), progress=progress)
        elif self.EXPERTS_FROM == "mapped":
            experts_file.map_experts(m, self.b, [(self.CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                     self._expert_pieces, self._index_layers(), progress=progress)
        else:
            m.act_as(self.CPU)
            loader = m.loader()
            for l in layers:
                loader.layer_width(l, 0, self.EXPERTS)
                for x in range(self.EXPERTS):
                    loader.load_expert(self.b, e, 0, l, x, self.EXPERTS)
                    self.SRC.setdefault(l, []).append(loader.placed[-1][3])
            if loader.handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loader.handed())
        m.act_as(self.CPU)
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        m.buffer(1024, "c_signs", _half(signs[:512]))
        m.buffer(H * 2, "c_zero", _half(np.zeros(H)))
        m.buffer(2 * (H + 16), "c_hand")
        m.buffer(H * 2, "c_routed")
        if self.MTP is not None:                       # a draft's check: the second position's hand and sum
            m.buffer(2 * (H + 16), "c_hand2")
            m.buffer(H * 2, "c_routed2")
        m.buffer(131072, "c_scratch")
        self.C_HANDS_AT = m.buffer(self.CHUNK * self.HAND_ROW, "c_hands")
        m.buffer(self.CHUNK * H * 2, "c_rrows")
        m.buffer(self.RSCRATCH, "c_rscratch")
        self._mask_buffers(cpu=True)
        for l in layers:
            cells = ["c_signs", "c_scratch", "c_zero", l, 0, e.at_up_lut, e.at_down_data, e.at_down_lut,
                     H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d),
                     backing.get((self.CPU, l), 0)] + self._masked(l, "moe", 16, 14)
            if self.EXPERTS_INT8:                          # ▶ AI_QWEN_3__EXP__INT8, past the mask's cells
                cells += [0] * (19 - len(cells)) + [1]
            m.table("exc%d" % l, cells)
            # a prompt's: the chunk's scratch in place of a position's, and a picture over the chunk
            m.table("exrc%d" % l, ["c_signs", "c_rscratch", "c_zero", l, 0, e.at_up_lut, e.at_down_data, e.at_down_lut,
                                   H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d),
                                   backing.get((self.CPU, l), 0)] + self._masked(l, "moe", 16, 14))
        m.array("EXPERTS_CPU", ["exc%d" % l for l in layers])
        m.array("EXPERTS_CPU_ROWS", ["exrc%d" % l for l in layers])
        m.act_as(CARD)

    def _named(self, name, card):
        """A buffer's or a table's name on card `card` of a tensor-parallel group: card 0's are the one-card names."""
        return name if card == 0 else "%s_%d" % (name, card)

    def _buffers(self, card=0):
        m, H = self.m, self.H
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        if card > 0:
            # another card of a tensor-parallel group: what its layers read — its residual, its scratch, its angles
            nm = lambda n: self._named(n, card)
            m.buffer(1024, nm("signs"), _half(signs[:512]))
            m.buffer(64, nm("minus1"), _half(-np.ones(32)))
            m.buffer(64, nm("one"), _half(np.ones(32)))
            for n in ("x", "h"):
                m.buffer(H * 2, nm(n))
            m.buffer(H * 2, nm("zero_h"), _half(np.zeros(H)))
            m.buffer(self.ROT * 4, nm("cs"))
            m.buffer(self.QH * self.max_context * 4, nm("scores"))
            if self.TIERED:
                rw = 4 * self.QH * (self.AH + 2)
                m.buffer(4 * rw + 2 * (2 * self.QH * self.AH + 3 * self.KV_WIDE), nm("tier_scratch"))
            m.buffer(262144, nm("mix_scratch"))
            # a prompt chunk's rows, as card 0's below: the rows and their h1s, the scratch, the angles, the tiered window
            #   or the one-cache scores
            m.buffer(self.CHUNK * H * 2, nm("xs"))
            m.buffer(self.CHUNK * H * 2, nm("h1s"))
            m.buffer(self.ROWS_SCRATCH, nm("pf_scratch"))
            m.buffer(self.CHUNK * self.ROT * 4, nm("cs_rows"))
            if self.TIERED:
                m.buffer(self.WINDOW, nm("win_k"))
                m.buffer(self.WINDOW, nm("win_v"))
                m.buffer(3 * self.CHUNK * 4 * self.QH * (self.AH + 2), nm("rows_res"))
                m.buffer(self.WINDOW_SCORES, nm("window_scores"))
            else:
                m.buffer(self.CHUNK * self.QH * self.max_context * 4, nm("scores_rows"))
            if self.CPLAN is not None:
                # a tier on this card too: its router's hand and picks, and its share of a position's experts
                m.buffer(2 * (H + 16 + self.INTER), nm("hand"))
                m.buffer(65536, nm("pre_scratch"))
                m.node_array(nm("picks"), self.TOPK + 1)      # past the picks: 1 where this card adds the shared expert
                m.buffer(H * 2, nm("routed_k"))
                m.buffer(2 * (self.TOPK * 2 * self.INTER + 2 * self.TOPK * self.INTER), nm("kscratch"))
            return
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
        m.node_array("PROMPT", self.CHUNK)          # a prompt chunk's tokens
        self._answer_buffers()                      # what `generate` made, the end tokens it watches for
        self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        m.buffer(64, "res")
        sampling.buffers(m)
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
        self.H1S_AT = m.buffer(self.CHUNK * H * 2, "h1s")
        self.ROWPOS_AT = m.buffer(self.CHUNK * 12, "rowpos")      # a chunk's rotary positions, three uint32 a row
        if self.MOE:
            # the router's hand, its picks and its scratch, and the experts verb's — the MoE in three reads them on the
            # card as the CPU's experts read the hand and the picks
            m.buffer(2 * (H + 16 + self.INTER), "hand")
            m.buffer(65536, "pre_scratch")
            m.node_array("picks", self.TOPK + (1 if self.CARDS > 1 and self.CPLAN is not None else 0))
            m.buffer(131072, "exp_scratch")
        if self.MTP is not None:
            self._mtp_buffers()
        if self.CPU_EXPERTS:
            # a prompt's hand rows and the routed rows the CPU hands back
            m.buffer(self.CHUNK * self.HAND_ROW, "hands")
            m.buffer(self.CHUNK * H * 2, "rrows")
            m.node_array("nrows", 1)
        if self.CPLAN is not None:
            # the card's tier: a position's share and scratch, a chunk's rows' share and `experts_rows`' scratch
            m.buffer(H * 2, "routed_k")
            m.buffer(2 * (self.TOPK * 2 * self.INTER + 2 * self.TOPK * self.INTER), "kscratch")
            m.buffer(self.CHUNK * H * 2, "rrows_k")
            self.KRS_AT = m.buffer(self.KRSCRATCH, "krscratch")

    def _mtp_buffers(self):
        """The MTP layer's and a draft's check's (▶ `_mtp_procedures`): its three norms' gains (`1 + w`, as `fnw`), its input
        — the embedding, the two normed halves and their rotation — its head's norm, the head states of the check's two
        positions (hA, hB), the second position's residual and sums, each DeltaNet layer's state and window kept between
        the check's two positions, and CHECK, the model's greedy token after each. ⛔ Made here, never where the programs
        are composed: a boot that reads programs.lisp composes none (▶ glm5.py, MEASURED 2026-10-19). Counted for the
        pools by `_mtp_asks`."""
        m, H, g = self.m, self.H, self.global_
        gain = lambda n: _half(1.0 + np.frombuffer(np.asarray(g[n]["vbr_data"]).tobytes(), dtype="<f2").astype(np.float64)[:H])
        m.buffer(H * 2, "enw", gain("mtp.pre_fc_norm_embedding.weight"))
        m.buffer(H * 2, "hnw", gain("mtp.pre_fc_norm_hidden.weight"))
        m.buffer(H * 2, "mnw", gain("mtp.norm.weight"))
        for nm in ("me", "hA", "hB", "mh", "mhr", "x2", "h12", "routed2"):
            m.buffer(H * 2, nm)
        for nm in ("xp", "hp"):                          # the check's two positions' residuals, and its mixers' answers, side by side
            m.buffer(2 * H * 2, nm)
        m.buffer(2 * self.VOCAB * 2, "logits2")         # both heads' logits
        m.buffer(H * 4, "mcat")
        m.buffer(H * 4, "mcatr")
        if self.MOE:
            m.buffer(2 * (H + 16 + self.INTER), "hand2")
            m.node_array("picks2", self.TOPK)
        for l in self.run_layers:
            if self.kinds[l] == "d":
                m.buffer(self.S_BYTES, "kbS%d" % l)
                m.buffer(self.W_BYTES, "kbW%d" % l)
        m.node_array("CHECK", 2)

    def plane(self, l, n, which):
        k = self.KEY[(l, n)]
        p = self.PLAN[k]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.TYPE[k], at, size)

    def tier(self, l, which, half):
        """A tier of an attention layer's cache, keys (`up`) or values (`down`)."""
        at, size = self.TIER[which]
        base = self.KVPLAN.at_up_data if half == "up" else self.KVPLAN.at_down_data
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, base + at, size)

    def state(self, l, which):
        t, p = (self.T_STATE, self.SPLAN) if self.kinds[l] == "d" else (self.T_KV, self.KVPLAN)
        at, size = (p.at_up_data, p.up_bytes) if which == "up" else (p.at_down_data, p.down_bytes)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, t, at, size)

    def _mixer_cells(self, l, scratch, cs="cs", scores="scores", card=0):
        """A mixer's plane table over the scratch named: the DeltaNet's or the attention's, as its verbs read it — on card
        `card` of a tensor-parallel group, its buffers and its share of the heads."""
        pl, d = self.plane, (lambda l_, n: self.b.record(l_, n).d)
        signs, minus1 = self._named("signs", card), self._named("minus1", card)
        if self.kinds[l] == "d":
            ns = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
                  "linear_attn.in_proj_b.weight"]
            return ([pl(l, "input_layernorm.weight", "data")] + [c for n in ns for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                    + [pl(l, "linear_attn.conv1d.weight", "data"), pl(l, "linear_attn.A_log", "data"),
                       pl(l, "linear_attn.dt_bias", "data"), pl(l, "linear_attn.norm.weight", "data"),
                       pl(l, "linear_attn.out_proj.weight", "data"), pl(l, "linear_attn.out_proj.weight", "lut"),
                       self.state(l, "up"), self.state(l, "down"), signs, minus1, scratch,
                       self.H, self.KH, self.VH, self.LH] + [d(l, n) for n in ns] + [d(l, "linear_attn.out_proj.weight")])
        ns = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
        return ([pl(l, "input_layernorm.weight", "data")] + [c for n in ns for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                + [pl(l, "self_attn.q_norm.weight", "data"), pl(l, "self_attn.k_norm.weight", "data"),
                   self.state(l, "up"), self.state(l, "down"), cs, scores, signs, scratch,
                   self.H, self.QH, self.KVH, self.AH, self.ROT] + [d(l, n) for n in ns])

    def _tiered_cells(self, l, card=0):
        """An attention layer's tiered table: the one-cache cells with the sink in the cache's place, then the tiers."""
        nm = lambda n: self._named(n, card)
        cells = self._mixer_cells(l, nm("mix_scratch"), nm("cs"), nm("scores"), card)
        cells[11], cells[12] = self.tier(l, "sink", "up"), self.tier(l, "sink", "down")
        return (cells + [self.THETA, 1 if self.ROTATED else 0]
                + [self.tier(l, "hot", "up"), self.tier(l, "hot", "down"),
                   self.tier(l, "warm_c", "up"), self.tier(l, "warm_s", "up"),
                   self.tier(l, "warm_c", "down"), self.tier(l, "warm_s", "down"),
                   self.tier(l, "cold_c", "up"), self.tier(l, "cold_s", "up"),
                   self.tier(l, "cold_c", "down"), self.tier(l, "cold_s", "down"),
                   nm("tier_scratch"), self.SINK, self.HOT, self.WARM])

    # ── the rank-1 mask, when the folder's LoRA is one (▶ `nn_rank1_mask.py`, the `rank1_*` doors) ─────────────────
    def _mask_layout(self):
        """Where each of the mask's tensors goes: `MASK_PLACE[(worker, l, site)]` its `r` and `v` as (start, count) in
        halves inside that worker's one mask buffer, and `MASK_BYTES` each buffer's size — the scratch for the verbs'
        dots first. The mixers and the shared expert are the card's; the routed experts go wherever they run."""
        self.MASK_BYTES, self.MASK_PLACE = {CARD: 0, self.CPU: 0}, {}
        mask = self.b.mask
        if mask is None:
            return
        from safetensors.numpy import load_file
        self._mask_tensors = t = load_file(mask["path"])
        scratch = 8 * self.CHUNK * max(self.TOPK, 1) + 256
        at = {CARD: scratch, self.CPU: scratch}
        for l in self.run_layers:
            for site in ("attn", "shared", "moe"):
                if "layers.%d.%s.r" % (l, site) not in t:
                    continue
                w = self.CPU if site == "moe" and self.CPU_EXPERTS else CARD
                place = []
                for a in (t["layers.%d.%s.r" % (l, site)], t["layers.%d.%s.%s" % (l, site, "V" if site == "moe" else "v")]):
                    place.append((at[w] // 2, a.size))
                    at[w] += (a.nbytes + 255) & ~255
                self.MASK_PLACE[(w, l, site)] = place
        for w in (CARD, self.CPU):
            if any(k[0] == w for k in self.MASK_PLACE):
                self.MASK_BYTES[w] = at[w]
        self._mask_scratch = scratch

    def _mask_buffers(self, cpu):
        """This worker's mask buffer, filled, and each site's three table cells: views of its `r`, its `v` and the
        scratch."""
        w = self.CPU if cpu else CARD
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

    def _tables(self, card=0):
        m, b, pl = self.m, self.b, self.plane
        d = lambda l, n: b.record(l, n).d
        if card > 0:
            return self._card_tables(card)
        for l in self.run_layers:
            # ⭐ a mask's cells, when the folder's LoRA is a rank-1 mask: past each table's longest form, padded up to them;
            #   and past those, an attention without an output gate says so (▶ AI_QWEN_3__AT__UNGATED)
            masked = lambda cells, at: cells + self._masked(l, "attn", at, len(cells))
            mk = lambda cells, at: (masked(cells, at) if self.GATED or self.kinds[l] == "d"
                                    else (lambda c: c + [0] * (51 - len(c)) + [1])(masked(cells, at)))
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
                            + ["signs", "one", scratch, self.H, self.INTER_OF[0]] + [d(l, n) for n in ms] + [self.EPS]
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

    def _card_tables(self, card):
        """Another card of a tensor-parallel group: a layer's mixer and MLP tables, over that card's part, its buffers and
        its share of the heads and the MLP's width — a position's and a prompt chunk's."""
        m, b, pl, nm = self.m, self.b, self.plane, (lambda n: self._named(n, card))
        d = lambda l, n: b.record(l, n).d
        flag = [1] if self.ROTATED else []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                cells = self._mixer_cells(l, nm("mix_scratch"), card=card) + flag
            elif self.TIERED:
                cells = self._tiered_cells(l, card)
            else:
                cells = (self._mixer_cells(l, nm("mix_scratch"), nm("cs"), nm("scores"), card)
                         + ([self.THETA, 1] if self.ROTATED else []))
            ungated = lambda c: c if self.GATED or self.kinds[l] == "d" else c + [0] * (51 - len(c)) + [1]   # noqa: E731
            m.table(nm("mx%d" % l), ungated(cells))
            if self.kinds[l] == "d":
                rows = self._mixer_cells(l, nm("pf_scratch"), card=card) + flag
            elif self.TIERED:
                rows = cells + [nm("pf_scratch"), nm("cs_rows"), nm("win_k"), nm("win_v"), nm("rows_res"), nm("window_scores")]
            else:
                rows = self._mixer_cells(l, nm("pf_scratch"), nm("cs_rows"), nm("scores_rows"), card) + [self.THETA] + flag
            m.table(nm(("dr%d" if self.kinds[l] == "d" else "atr%d" if self.TIERED else "ar%d") % l), ungated(rows))
            if self.MOE:
                if self.CPLAN is not None:  # its own router, as card 0's (▶ `_tables`)
                    sh = ["mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                          "mlp.shared_expert_gate.weight"]
                    m.table(nm("pre%d" % l), [pl(l, "post_attention_layernorm.weight", "data"),
                                              pl(l, "mlp.gate.weight", "data"), pl(l, "mlp.gate.weight", "lut")]
                            + [c for n in sh for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                            + [nm("signs"), nm("pre_scratch"), self.H, self.INTER, self.EXPERTS, self.TOPK,
                               d(l, "mlp.gate.weight"), d(l, "mlp.shared_expert.down_proj.weight")] + flag)
                    # and the shared expert's down, for a position it falls to this card (NN-99 ③)
                    m.table(nm("post%d" % l), [pl(l, "mlp.shared_expert.down_proj.weight", "data"),
                                               pl(l, "mlp.shared_expert.down_proj.weight", "lut"),
                                               self.H, self.INTER, self.TOPK, d(l, "mlp.shared_expert.down_proj.weight")])
                continue                    # the rest of its MoE block is card 0's
            ms = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            for name, scratch in (("ml%d", "mix_scratch"), ("mlr%d", "pf_scratch")):     # a position's, and the rows'
                m.table(nm(name % l), [pl(l, "post_attention_layernorm.weight", "data")]
                        + [c for n in ms for c in (pl(l, n, "data"), pl(l, n, "lut"))]
                        + [nm("signs"), nm("one"), nm(scratch), self.H, self.INTER_OF[card]] + [d(l, n) for n in ms]
                        + [self.EPS] + flag)

    def _tier_tables(self, card=0):
        """The card's tier (▶ nn's `nn__expert_tier`): its experts tables, a position's and a chunk's, each with its own
        scratch; the tier, fed from the CPU's memory where every expert is there (the experts in RAM); and the binding —
        one card's, or (`card`, as that card's worker) a tensor-parallel group's card's, whose tier says the group's size
        and whose binding `_load_cards` makes for every card. A chunk's table is card 0's alone: a prompt's rows go
        through card 0's tier and the CPU (▶ lisp/qwen3_5_tp.lisp)."""
        m, e, nm = self.m, self.EPLAN, (lambda n: self._named(n, card))
        for l in self.run_layers:
            d = int(self.b.record(l, GU).d)
            if l in self.SRC and card == 0:
                self._int_array("src%d" % l, self.SRC[l])
            cells = [l, self.T_CEXP, e.at_up_lut, e.at_down_data, e.at_down_lut, self.H, self.INTER, self.EXPERTS, self.TOPK,
                     d, 0, 0, 0, 0, 0, 0]
            m.table(nm("exk%d" % l), [nm("signs"), nm("kscratch"), nm("zero_h")] + cells)
            if card == 0:
                m.table("exkr%d" % l, ["signs", "krscratch", "zero_h"] + cells)
        # a card slot laid out as the CPU's, so a promotion copies one whole
        expert_tier.tier(m, nm("tier_card"), self.T_CEXP, self.EXPERTS, self.TOPK, self.H, self.INTER, e, self.LAYERS,
                         {l: "src%d" % l for l in self.run_layers if l in self.SRC}, landing=self.CARD_LANDING,
                         ranks=self.CARDS, shared=int(round(self.CARD_SHARED * self.CARD_SLOTS / len(self.run_layers))))
        if self.CARDS == 1:
            expert_tier.bind(m, {CARD: "tier_card"}, len(self.workers()))

    # ── the cache's unreached rows lent to the tier (NN-93) ──────────────────────────────────────────────────────
    # ⚖ 2026-10-17: *"progressively shrink the l1 cache as the context grows, maybe in chunks of 8k tokens... this way we
    # dont need the users to set a max tokens or to size the l1 cache at all"* · *"it is the experts on vram"*. The warm
    # ring is the context's whole length and a conversation writes its rows from the first (▶ ai_qwen_3's
    # `attention_tiered`: position p's row is p - SINK - HOT, read only below the conversation's count), so the rows past
    # them are memory nothing reads. The tier holds experts there — slots numbered from each region's end — and gives
    # back LEND_CHUNK rows' worth before a conversation reaches them: `_room` before every write, and when a
    # conversation comes onto the card (`reset`, `state_write`) the slots its length leaves are lent again.
    LEND_CHUNK = 8192

    def _lend(self):
        """The tier lent every attention layer's warm keys' and values' tail — the regions' ends on the slots' alignment,
        each `depth` slots deep — then all of them live, the cache holding nothing yet."""
        L, b = self.LEND, self.CPLAN.slot_bytes
        if L is None:
            return
        starts, ends = [], []
        for l in self.run_layers:
            if self.kinds[l] != "f":
                continue
            for half in (self.KVPLAN.at_up_data, self.KVPLAN.at_down_data):
                start = self.state_at[l] + half + self.TIER["warm_c"][0]
                starts.append(start)
                ends.append((start + self.WARM * L["row"]) // 4096 * 4096)
        depth = min((e - (s + 4095) // 4096 * 4096) // b for s, e in zip(starts, ends))
        if depth <= 0:
            self.LEND = None
            return
        was = self.m.worker
        self.m.act_as(CARD)
        self._int_array("lend_ends", ends)
        self.m.must("(nn__expert__lend %d %d lend_ends)" % (self.T_CEXP, depth), "the cache's rows lent to the tier")
        self.m.act_as(was)
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
        rows = -(-max(0, n - self.SINK - self.HOT) // self.LEND_CHUNK) * self.LEND_CHUNK
        b, need = self.CPLAN.slot_bytes, rows * L["row"]
        deep = max(0, min(L["depth"], min((e - s - need) // b for s, e in zip(L["starts"], L["ends"]))))
        live = deep * len(L["ends"])
        if live != L["live"]:
            was = self.m.worker
            self.m.act_as(CARD)
            got = self.m.must("(nn__expert__lent %d %d)" % (self.T_CEXP, live), "the tier's lent slots")
            self.m.act_as(was)
            if got != live:
                raise Refused("the tier holds %d lent slots, not %d" % (got, live))
            L["live"] = live
        free = min(e - deep * b - s for s, e in zip(L["starts"], L["ends"]))
        L["until"] = min(self.max_context, self.SINK + self.HOT + free // L["row"])

    def lent(self):
        """The cache's rows lent to the tier now: (slots live, GB), or None where the tier is lent none."""
        L = self.LEND
        return None if L is None else (L["live"], L["live"] * self.CPLAN.slot_bytes / 2 ** 30)

    def _int_array(self, name, values):
        """A node array of integers, bound by name — filled a hundred at a time, as a program must fit one heap chunk."""
        m = self.m
        m.node_array(name, len(values))
        for at in range(0, len(values), 100):
            m.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set %s %d %d)" % (name, i, v)
                                                     for i, v in enumerate(values[at:at + 100], at)),
                   "the array %s" % name, "sys__value_true")

    # ── ③ the names the program is written against, and the program ──────────────────────────────────────────
    def _bindings(self):
        """What `lisp/qwen3_5.lisp` reads: the shape, the settings, and a layer's tables gathered into arrays indexed by its
        place in the layers this machine runs."""
        m, H, run = self.m, self.H, self.run_layers
        row = self.EMB_ROW_BYTES
        plane = lambda at, size: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_HEAD, at, size)
        values = dict(HIDDEN=H, HIDDEN_BYTES=2 * H, VOCAB=self.VOCAB, EPS=self.EPS, THETA=self.THETA, ROTARY=self.ROT,
                      RUN=len(run) - (1 if self.MTP is not None else 0), EMB_HALVES=row // 2, EMB_SCALES=self.EMB_DATA_BYTES // 2, EMB_BITS=self.EMB_BITS,
                      HEAD_BITS=self.HEAD_BITS, STOPS=0, SEEN=sampling.SEEN,
                      HEAD_CODES=plane(self.HPLAN.at_up_data, self.HPLAN.up_data),
                      HEAD_SCALES=plane(self.HPLAN.at_down_lut, self.HPLAN.down_lut),
                      LAG=0, ROWPOS=False, MROPE=self.MROPE,
                      ROTATED=bool(self.ROTATED), TIERED=bool(self.TIERED), ROUTED=bool(self.MOE),
                      FUSED=bool(self.MOE and self.FUSED), EXPERTS_ON_CPU=bool(self.CPU_EXPERTS),
                      CARD_TIER=self.CPLAN is not None, CARD=CARD, CPU=self.CPU)
        if self.CPU_EXPERTS:
            values.update(HAND_BYTES=2 * (H + 16), HANDS_BYTES=self.CHUNK * self.HAND_ROW, ROWS_BYTES=self.CHUNK * H * 2,
                          ROWS_HALVES=self.CHUNK * H, HAND_ROW=self.HAND_ROW, HAND_PICKS_AT=self.HAND_ROW - 8 * self.TOPK)
        if self.MTP is not None:
            fc = lambda at, size: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_FC, at, size)
            values.update(FC_CODES=fc(self.FCPLAN.at_up_data, self.FCPLAN.up_data),
                          FC_SCALES=fc(self.FCPLAN.at_down_lut, self.FCPLAN.down_lut), FC_BITS=self.FC_BITS)
        m.bind_values(values)
        m.array("DELTANET", [self.kinds[l] == "d" for l in run])
        m.array("LAYER", list(run))
        m.array("MIXER", ["mx%d" % l for l in run])
        m.array("MIXER_ROWS", [("dr%d" if self.kinds[l] == "d" else "atr%d" if self.TIERED else "ar%d") % l for l in run])
        if not self.MOE:
            m.array("MLP", ["ml%d" % l for l in run])
            m.array("MLP_ROWS", ["mlr%d" % l for l in run])
            if self.CARDS > 1:
                self._card_bindings()
            return
        m.array("PRE", ["pre%d" % l for l in run])
        m.array("POST", ["post%d" % l for l in run])
        if not self.CPU_EXPERTS:
            m.array("MOE", ["mr%d" % l for l in run])
            m.array("EXPERTS", ["ex%d" % l for l in run])
        else:
            m.node_array("HERE", 1)             # the layer a CPU picture runs (▶ lisp/qwen3_5_cpu.lisp)
        if self.CPLAN is not None:
            m.array("EXPERTS_TIER", ["exk%d" % l for l in run])
            m.array("EXPERTS_TIER_ROWS", ["exkr%d" % l for l in run])
        if self.CARDS > 1:
            self._card_bindings()

    def _card_bindings(self):
        """What lisp/qwen3_5_tp.lisp reads besides: the group, and each card's names in arrays indexed by its worker."""
        m, run, P = self.m, self.run_layers, self.CARDS
        m.bind_values(dict(CARDS=P, GROUP=self.GROUP, TOP_K=self.TOPK if self.MOE else 0))
        m.node_array("TP_POS", 1)
        m.node_array("TP_N", 1)
        m.node_array("TP_I", P)
        rows = lambda l: ("dr%d" if self.kinds[l] == "d" else "atr%d" if self.TIERED else "ar%d") % l   # noqa: E731
        for k in range(1, P):
            m.array(self._named("MIXER", k), [self._named("mx%d" % l, k) for l in run])
            m.array(self._named("MIXER_ROWS", k), [self._named(rows(l), k) for l in run])
            if not self.MOE:
                m.array(self._named("MLP", k), [self._named("ml%d" % l, k) for l in run])
                m.array(self._named("MLP_ROWS", k), [self._named("mlr%d" % l, k) for l in run])
        names = [("X_OF", "x"), ("H_OF", "h"), ("ZERO_OF", "zero_h"), ("CS_OF", "cs"), ("MIXER_OF", "MIXER"),
                 ("XS_OF", "xs"), ("H1S_OF", "h1s"), ("MIXER_ROWS_OF", "MIXER_ROWS")]
        if not self.MOE:
            names += [("MLP_OF", "MLP"), ("MLP_ROWS_OF", "MLP_ROWS")]
        else:                               # an MoE's block is card 0's: the other cards have no MLP, the names zeros
            m.array("MLP_OF", [0] * P)
            m.array("MLP_ROWS_OF", [0] * P)
            if self.CPLAN is not None:      # a tier on every card: its router's and its experts' names
                for k in range(1, P):
                    m.array(self._named("PRE", k), [self._named("pre%d" % l, k) for l in run])
                    m.array(self._named("EXPERTS_TIER", k), [self._named("exk%d" % l, k) for l in run])
                    m.array(self._named("POST", k), [self._named("post%d" % l, k) for l in run])
                names += [("PRE_OF", "PRE"), ("HAND_OF", "hand"), ("PICKS_OF", "picks"), ("ROUTED_K_OF", "routed_k"),
                          ("EXPERTS_TIER_OF", "EXPERTS_TIER"), ("POST_OF", "POST")]
            else:
                for name in ("PRE_OF", "HAND_OF", "PICKS_OF", "ROUTED_K_OF", "EXPERTS_TIER_OF", "POST_OF"):
                    m.array(name, [0] * P)
        for name, of in names:
            m.array(name, [self._named(of, k) for k in range(P)])

    def _procedures(self):
        """The program: `lisp/qwen3_5.lisp`, the sampler's, and — the experts on the CPU — the pictures the card hands it.
        Read from `programs.lisp` instead by the boots that have the same key (▶ `Machine.define_programs`)."""
        self.m.load_source(self.folder.program("qwen3_5"))
        if self.CARDS > 1:
            self.m.load_source(self.folder.program("qwen3_5_tp"))
        if self.PICTURES:
            self.m.load_source(self.folder.program("qwen3_5_pictures"))
        self.m.load_source(self.folder.program("sampling"))
        self.m.load_source(self.folder.program("generate"))
        if self.CPU_EXPERTS:
            self.m.worker_moved = {CPU: self.CPU}           # the file's worker 1 the CPU, after a group's cards
            self.m.load_source(self.folder.program("qwen3_5_cpu"))
        if self.MTP is not None:
            self._mtp_procedures()

    def _mtp_procedures(self):
        """⭐ THE MTP LAYER AND A DRAFT'S CHECK, composed (as GLM's, ▶ glm5.py `_mtp_procedures`):
          (mtp_from tok pos hid)   the MTP layer at `pos` of its own cache — `tok` the token at pos + 1, `hid` the model's
                                   normed hidden at `pos` — and its head: the greedy draft of the token at pos + 2
          (mtp_cache_from …)       the layer alone, its cache's row written: what a prompt's rows give it
          (row_hidden r)           a prompt row's normed hidden (its final norm) into hB
          (check a b p q)          the token `a` at `p` and the draft `b` at q = p + 1 through every layer, each layer's
                                   two positions in turn — its mixer's for p, then q's (a DeltaNet layer's state and window
                                   kept between them, the rotary's angles each position's), both MoEs (the experts on the
                                   CPU: one hand-off for both) — then both heads: CHECK the greedy token after each, hA and
                                   hB the normed hidden at each
          (restore)                every DeltaNet layer's state and window as they were after `p`
        ⛳ An attention layer needs nothing given back: a position reads only rows it has written, and q's row is written
        again when q is run for real.
        ⚖ MEASURED 2026-10-19 node-01, the 35B on one MI50 (scratchpad qwen_parts_cost.py): a step 8.6 ms, its mixers alone
        5.3 and its MoE 5.4 — and the rows verbs over 2 rows 40 + 50 ms (the MoE's rows verb reads every expert): so the
        check is two one-position passes, not a chunk of two rows."""
        m, H, run = self.m, self.H, self.run_layers
        main = [l for l in run if l != self.MTP]
        attn = "ai_qwen_3__attention_tiered" if self.TIERED else "ai_qwen_3__attention"
        rng = lambda b, at, n: "(nn__vector__range %s %d %d)" % (b, at, n)
        head = "(nn__turboquant__gemv HEAD_CODES HEAD_SCALES mhr logits_h VOCAB HIDDEN HEAD_BITS)"
        m.defun("(defun (mtp_input tok hid) (begin (embed_into tok me) (nn__rmsnorm__apply me enw %s HIDDEN EPS) "
                "(nn__rmsnorm__apply hid hnw %s HIDDEN EPS) (nn__hadamard__rotate mcat signs mcatr %d) "
                "(nn__turboquant__gemv FC_CODES FC_SCALES mcatr x HIDDEN %d FC_BITS) (type x)))"
                % (rng("mcat", 0, H), rng("mcat", H, H), 2 * H, 2 * H))
        m.defun("(defun (mtp_cache_from tok pos hid) (begin (mtp_input tok hid) (nn__rope__angles cs pos THETA ROTARY) "
                "(layer RUN pos) (type x)))")
        m.defun("(defun (mtp_logits_from tok pos hid) (begin (mtp_cache_from tok pos hid) (nn__rmsnorm__apply x mnw mh HIDDEN EPS) "
                "(nn__hadamard__rotate mh signs mhr HIDDEN) %s (type x)))" % head)
        m.defun("(defun (mtp_from tok pos hid) (begin (mtp_logits_from tok pos hid) (nn__argmax__find logits_h res VOCAB) "
                "(nn__buffer__read res)))")
        m.defun("(defun (row_hidden r) (begin (nn__rmsnorm__apply (nn__vector__range xs (* r HIDDEN) HIDDEN) fnw hB HIDDEN EPS) "
                "(type hB)))")
        # ⭐ THE EXPERTS ON THE CPU: the two positions are a prompt's two rows — the residuals rows 0 and 1 of xs, the mixers'
        #   answers of h1s — so the card routes both in one call into their hand rows, the CPU runs each expert once over
        #   the rows that picked it (`experts_rows`, the prompt's), and the card closes both. `mtp_rows` false keeps
        #   two one-position hand-offs in one picture instead
        rows2 = self.CPU_EXPERTS and self.CHECK_ROWS
        # ⭐ BOTH POSITIONS SIDE BY SIDE (`mtp_pairs`): a DeltaNet layer's two in one word, `deltanet_n`, which reads its
        #   matrices once for both and keeps its state and window as the first leaves them
        pairs = self.CHECK_PAIRS
        xb, hb = ("xs", "h1s") if rows2 else ("xp", "hp") if pairs else (None, None)
        x, x2 = (rng(xb, 0, H), rng(xb, H, H)) if xb else ("x", "x2")
        h1, h12 = (rng(hb, 0, H), rng(hb, H, H)) if hb else ("h1", "h12")
        if self.CPU_EXPERTS:
            m.act_as(self.CPU)
            m.picture("xqb", "(begin (nn__buffer__copy hand CARD c_hand HAND_BYTES) (nn__buffer__copy hand2 CARD c_hand2 HAND_BYTES) "
                      "(ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed) "
                      "(ai_qwen_3__experts c_hand2 picks2 (nth EXPERTS_CPU (nth HERE 0)) c_routed2))")
            m.picture("xr2", "(begin (nn__buffer__copy hands CARD c_hands %d) "
                      "(ai_qwen_3__experts_rows c_hands (nth EXPERTS_CPU_ROWS (nth HERE 0)) c_rrows 2))" % (2 * self.HAND_ROW))
            m.act_as(CARD)
            m.view("names")
        for i, l in enumerate(main):
            if self.kinds[l] == "d" and xb:
                mix = "(ai_qwen_3__deltanet_n %s mx%d %s 2 kbS%d kbW%d)" % (xb, l, hb, l, l)
            elif self.kinds[l] == "d":
                mix = ("(ai_qwen_3__deltanet %s mx%d %s) (nn__vector__copy %s kbS%d %d) (nn__vector__copy %s kbW%d %d) "
                       "(ai_qwen_3__deltanet %s mx%d %s)" % (x, l, h1, self.state(l, "up"), l, self.S_BYTES // 2,
                                                           self.state(l, "down"), l, self.W_BYTES // 2, x2, l, h12))
            else:
                mix = ("(nn__rope__angles cs p THETA ROTARY) (%s %s mx%d p %s) (nn__rope__angles cs q THETA ROTARY) (%s %s mx%d q %s)"
                       % (attn, x, l, h1, attn, x2, l, h12))
            if not self.MOE:
                ffn = "(nn__mlp__apply %s ml%d %s) (nn__mlp__apply %s ml%d %s)" % (h1, l, x, h12, l, x2)
            elif rows2:
                ffn = ("(ai_qwen_3__pre_expert_rows h1s pre%d hands 2) (sys__node_array__set HERE 0 %d) (sys__compute CPU names xr2) "
                       "(sys__result CPU) (nn__buffer__copy c_rrows CPU rrows %d) (ai_qwen_3__post_expert_rows h1s hands rrows post%d xs 2)"
                       % (l, i, 4 * H, l))
            elif self.CPU_EXPERTS:
                ffn = ("(ai_qwen_3__pre_expert %s pre%d hand picks) (ai_qwen_3__pre_expert %s pre%d hand2 picks2) "
                       "(sys__node_array__set HERE 0 %d) (sys__compute CPU names xqb) (sys__result CPU) "
                       "(nn__buffer__copy c_routed CPU routed HIDDEN_BYTES) (nn__buffer__copy c_routed2 CPU routed2 HIDDEN_BYTES) "
                       "(ai_qwen_3__post_expert %s hand routed post%d %s) (ai_qwen_3__post_expert %s hand2 routed2 post%d %s)"
                       % (h1, l, h12, l, i, h1, l, x, h12, l, x2))
            elif self.FUSED and pairs:
                ffn = "(ai_qwen_3__moe_n %s mr%d %s 2)" % (hb, l, xb)       # both positions, one wait for their picks
            elif self.FUSED:
                ffn = "(ai_qwen_3__moe %s mr%d %s) (ai_qwen_3__moe %s mr%d %s)" % (h1, l, x, h12, l, x2)
            else:
                ffn = ("(ai_qwen_3__pre_expert %s pre%d hand picks) (ai_qwen_3__experts hand picks ex%d routed) "
                       "(ai_qwen_3__post_expert %s hand routed post%d %s) "
                       "(ai_qwen_3__pre_expert %s pre%d hand2 picks2) (ai_qwen_3__experts hand2 picks2 ex%d routed2) "
                       "(ai_qwen_3__post_expert %s hand2 routed2 post%d %s)" % (h1, l, l, h1, l, x, h12, l, l, h12, l, x2))
            m.defun("(defun (ck%d p q) (begin %s %s (type x)))" % (l, mix, ffn))
        heads = " ".join("(nn__rmsnorm__apply %s fnw %s HIDDEN EPS) (nn__hadamard__rotate %s signs mhr HIDDEN) %s "
                         "(nn__argmax__find logits_h res VOCAB) (sys__node_array__set CHECK %d (nn__buffer__read res))"
                         % (xk, hid, hid, head, k) for k, (xk, hid) in enumerate(((x, "hA"), (x2, "hB"))))
        if pairs:
            # ⭐ both heads in one read of the head's matrix (`nn__turboquant__gemv_n`): each row's normed hidden rotated
            #   into mcatr's two rows, both rows' logits into logits2
            heads = ("(nn__rmsnorm__apply %s fnw hA HIDDEN EPS) (nn__rmsnorm__apply %s fnw hB HIDDEN EPS) "
                     "(nn__hadamard__rotate hA signs %s HIDDEN) (nn__hadamard__rotate hB signs %s HIDDEN) "
                     "(nn__turboquant__gemv_n HEAD_CODES HEAD_SCALES mcatr logits2 VOCAB HIDDEN HEAD_BITS 2) "
                     "(nn__argmax__find %s res VOCAB) (sys__node_array__set CHECK 0 (nn__buffer__read res)) "
                     "(nn__argmax__find %s res VOCAB) (sys__node_array__set CHECK 1 (nn__buffer__read res))"
                     % (x, x2, rng("mcatr", 0, H), rng("mcatr", H, H), rng("logits2", 0, self.VOCAB),
                        rng("logits2", self.VOCAB, self.VOCAB)))
        m.defun("(defun (check_heads) (begin %s (type hB)))" % heads)
        m.defun("(defun (check a b p q) (begin (embed_into a %s) (embed_into b %s) %s (check_heads) (type x)))"
                % (x, x2, " ".join("(ck%d p q)" % l for l in main)))
        back = ["(nn__vector__copy kbS%d %s %d) (nn__vector__copy kbW%d %s %d)"
                % (l, self.state(l, "up"), self.S_BYTES // 2, l, self.state(l, "down"), self.W_BYTES // 2)
                for l in main if self.kinds[l] == "d"]
        m.defun("(defun (restore) (begin %s (type x)))" % " ".join(back))

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: every DeltaNet layer's state and window to zero. The attention caches need
        nothing — a position reads only rows it has written."""
        zeros = []
        for l in self.run_layers:
            if self.kinds[l] == "d":
                zeros += ["(nn__vector__zero %s %d)" % (self.state(l, "up"), self.S_BYTES // 2),
                          "(nn__vector__zero %s %d)" % (self.state(l, "down"), self.W_BYTES // 2)]
        for k in range(1, self.CARDS):                  # every other card of a tensor-parallel group: its own states
            self.m.act_as(k)
            self.m.must("(begin %s (type x))" % " ".join(zeros), "the reset of card %d" % k)
        if self.CARDS > 1:
            self.m.act_as(-1)
        self.m.must("(begin %s (type x))" % " ".join(zeros), "the reset")
        self._room(0, exact=True)                       # a conversation from nothing: every lent slot the tier's again
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None

    def _one_card(self, what):
        """⛳ What a tensor-parallel group does not do yet: refused rather than done on card 0's part alone."""
        if self.CARDS > 1:
            raise Refused("%s is one card's: not under tensor parallelism yet (each card holds its part)" % what)

    def _needs_head(self):
        if self.HEADLESS:
            raise Refused("%s was packed without its head: it answers its rows (`forward_rows`), not tokens" % self.folder.path)

    def step(self, token, pos, greedy=True, sample=None, lag=0):
        """One position: the token in, the next token out — greedy, or drawn on the card by `sample` (a
        `sampling.head`) — or the logits left for a sampler on the host. `lag`: how far its rotary position is behind
        `pos` (an image before it took fewer positions than places)."""
        self._needs_head()
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._room(pos + 1)
        head = sample or ("(head)" if greedy else "(head_logits)")
        self._lag = lag                                    # what `generate` finds set (▶ generate.py)
        return self._run("(begin (set! LAG %d) (position %d %d) %s)" % (lag, token, pos, head), "position %d" % pos)

    def generate(self, token, pos, n, stop=(), sampler=None, lag=0):
        """`Generates.generate`, the cache's rows it may write withheld from the tier first — or, with the MTP layer and a
        greedy answer, two tokens a pass where its draft is kept (`generate_mtp`). ⛳ The draft goes on from where the last
        prompt or answer left it; anywhere else (another conversation ran between) the answer is the plain one."""
        self._room(pos + min(n, GENERATE_MOST) + 2)
        if self.MTP is not None and sampler is None and not lag:
            n = min(n, GENERATE_MOST, self.max_context - pos - 1)
            draft = None
            if n > 0 and self._mtp_next is not None and self._mtp_next[:2] == (pos, token):
                draft = self._mtp_next[2]
            elif n > 0 and self._mtp_hid == pos - 1:     # the head state at pos - 1 is hB's
                draft = self._run("(mtp_from %d %d hB)" % (token, pos - 1), "the draft at %d" % (pos - 1))
            if draft is not None:
                self._watch(())
                out, passes = self.generate_mtp(token, pos, draft, n, set(stop))
                self.mtp_stats["passes"] += passes
                self.mtp_stats["kept"] += len(out) - passes
                return out
            self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None   # the plain path runs on without the MTP layer
        return Generates.generate(self, token, pos, n, stop, sampler, lag)

    def generate_mtp(self, token, pos, draft, n, stop=()):
        """`n` greedy tokens from `token` at `pos` with the MTP layer's `draft` of the one after: each pass checks the
        token and the draft together (`check`) — the draft kept when the model's own greedy token after `token` is the
        draft, the token after the draft then the next; refused, the DeltaNet states given back. Answers (tokens,
        passes). ▶ glm5.py's, the same steps."""
        m, out, passes = self.m, [], 0
        # ⛳ WHAT `generate` PROMISES (▶ session.py): every token it answers but the last has been run — so a pass that would
        #   run more than it answers gives the states back to after `token`: a kept draft that is an end token, or one with
        #   no room left for the token after it
        while len(out) < n:
            self._run("(begin (check %d %d %d %d) (type x))" % (token, draft, pos, pos + 1), "the check at %d" % pos)
            passes += 1
            y1 = m.must("(sys__node_array__get CHECK 0)", "the model's token after the token")
            y2 = m.must("(sys__node_array__get CHECK 1)", "the model's token after the draft")
            if y1 == draft and (draft in stop or len(out) + 1 >= n):
                out.append(draft)
                self._run("(restore)", "the DeltaNet states given back")
                self._run("(begin (nn__vector__copy hA hB %d) (type hB))" % self.H, "the state at %d" % pos)
                self._mtp_have, self._mtp_hid, self._mtp_next = pos - 1, pos, None
                return out, passes
            if y1 == draft:
                out += [draft, y2]
                # the MTP layer at pos (the draft, the state at pos) and at pos + 1 (y2, the state at pos + 1): the next draft
                self._run("(begin (mtp_cache_from %d %d hA) (type x))" % (draft, pos), "the MTP layer at %d" % pos)
                draft = self._run("(mtp_from %d %d hB)" % (y2, pos + 1), "the draft at %d" % (pos + 1))
                token, pos, self._mtp_have, self._mtp_hid = y2, pos + 2, pos + 1, pos + 1
            else:
                out.append(y1)
                self._run("(restore)", "the DeltaNet states given back")
                self._run("(begin (nn__vector__copy hA hB %d) (type hB))" % self.H, "the state at %d" % pos)
                draft = self._run("(mtp_from %d %d hA)" % (y1, pos), "the draft at %d" % pos)
                token, pos, self._mtp_have, self._mtp_hid = y1, pos + 1, pos, pos
            if out[-1] in stop:
                break
        self._mtp_next = (pos, token, draft)
        return out[:n], passes

    def _mtp_prime(self, p, tok):
        """The MTP layer at `p - 1` from the token at `p` and the head state at `p - 1`, where hB holds it."""
        if self.MTP is not None and p > 0 and self._mtp_hid == p - 1:
            self._run("(begin (mtp_cache_from %d %d hB) (type x))" % (tok, p - 1), "the MTP layer at %d" % (p - 1))
            self._mtp_have = p - 1

    # ── pictures: the tower's rows in place of their pads, and M-RoPE's positions ─────────────────────────────
    def vision(self):
        """The model's vision tower (▶ `vision.Vision`), made when the first picture comes."""
        if getattr(self, "_vision", None) is None:
            self._one_card("reading pictures")
            if not self.MROPE:
                raise Refused("%s reads no pictures: its rotary has no M-RoPE sections" % self.folder.path)
            # ⛳ the program travels with the model (lisp/): a folder fetched before pictures has no qwen3_5_pictures.lisp
            if not self.PICTURES:
                raise Refused("%s has no lisp/qwen3_5_pictures.lisp — the program's pictures, which a folder fetched before "
                              "v0.7.0 lacks: `silvann download <model>` fetches the pack's lisp/ afresh" % self.folder.path)
            from .vision import Vision
            self._vision = Vision(self.folder, self if getattr(self, "VIS", False) else None)
        return self._vision

    def _rows_in(self, rows, at, n):
        """A chunk's rows that are a picture's — `rows` {index in the prompt: the tower's row} — written over their pads'
        embeddings in `xs`, through `h1s` and rotated into the residual's basis on the card when the pack keeps it
        rotated, a run of consecutive rows at a time."""
        H, mine = self.H, sorted(i - at for i in rows if at <= i < at + n)
        runs, start = [], None
        for k, r in enumerate(mine):
            if start is None:
                start = r
            if k + 1 == len(mine) or mine[k + 1] != r + 1:
                runs.append((start, r + 1))
                start = None
        for a, b in runs:
            block = np.stack([rows[at + r] for r in range(a, b)]).astype("<f2").tobytes()
            if not self.ROTATED:
                self.m.write(self.XS_AT + a * H * 2, block)
                continue
            self.m.write(self.H1S_AT + a * H * 2, block)
            self.m.must("(begin (nn__hadamard__rotate (nn__vector__range h1s %d %d) signs (nn__vector__range xs %d %d) %d) "
                        "(type xs))" % (a * H, (b - a) * H, a * H, (b - a) * H, (b - a) * H), "a picture's rows")

    def _run(self, text, what):
        """A program's answer. ⛳ With the experts on the CPU it runs with every block standing — it hands them work
        (`sys__compute`) — and a program that does not would find the CPU's block never started."""
        if self.CPU_EXPERTS or self.CARDS > 1:
            return self.m.grid(text)[1]
        return self.m.must(text, what)

    def _prompt(self, tokens):
        """A prompt chunk's tokens into PROMPT, which `embed_rows` reads — sixty-four to a program, as a program is one
        form and a form must fit one of the heap's chunks."""
        for at in range(0, len(tokens), 64):
            self.m.must("(begin %s 0)" % " ".join("(sys__node_array__set PROMPT %d %d)" % (i, t)
                                                  for i, t in enumerate(tokens[at:at + 64], at)), "the prompt's tokens")

    def prefill(self, tokens, first, greedy=True, sample=None, positions=None, rows=None):
        """Positions `first ..` for `tokens` in chunks of rows — each layer one call a chunk — then the next token from
        the last row (greedy, or drawn by `sample`), or its logits left for a sampler. A picture's tokens come with
        `rows` {index in `tokens`: the tower's row, in place of that pad's embedding} and every token with its three
        rotary `positions` (n x 3, ▶ `vision.positions`); without them a token's position is its place in the cache."""
        self._needs_head()
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        if (positions is not None or rows) and not (self.MROPE and self.PICTURES):
            raise Refused("%s reads no pictures here (▶ `vision`)" % self.folder.path)
        rows = rows or {}
        if self.CARDS > 1 and (positions is not None or rows):
            raise Refused("pictures are one card's: not under tensor parallelism yet")
        self._room(first + len(tokens))
        step = min(self.CHUNK, self.HOT) if self.TIERED else self.CHUNK      # a chunk never outgrows the hot ring
        for at in range(0, len(tokens), step):
            chunk = tokens[at:at + step]
            self._prompt(chunk)
            given = positions is not None
            if given:
                self.m.write(self.ROWPOS_AT, np.ascontiguousarray(positions[at:at + len(chunk)], dtype="<u4").tobytes())
            if any(at <= i < at + len(chunk) for i in rows):
                self.m.must("(begin (embed_rows %d) (type xs))" % len(chunk), "the rows' embeddings from %d" % (first + at))
                self._rows_in(rows, at, len(chunk))
                start = ""
            else:
                start = "(embed_rows %d)" % len(chunk)
            self._run("(begin (set! ROWPOS %s) %s (rows %d %d) (set! ROWPOS (< 1 0)) (type xs))"
                      % ("(< 0 1)" if given else "(< 1 0)", start, first + at, len(chunk)), "the rows from %d" % (first + at))
            last = len(chunk) - 1
            if self.MTP is not None and given:
                self._mtp_hid = -1                     # a picture's prompt: the MTP layer is given none of it
            if self.MTP is not None and not given:
                # ⭐ THE MTP LAYER GIVEN THE PROMPT'S LAST MTP_HISTORY ROWS (None: every row) — each row's normed hidden, the
                #   layer at the row before with the row's token, its cache's row written (▶ glm5.py's `prefill`)
                end = first + len(tokens)
                begin = first if self.MTP_HISTORY is None else end - self.MTP_HISTORY
                for r, tok in enumerate(chunk):
                    p = first + at + r
                    if p < begin - 1 and p != end - 1:
                        continue
                    if p >= begin:
                        self._mtp_prime(p, tok)
                    self._run("(row_hidden %d)" % r, "row %d's state" % r)
                    self._mtp_hid = p
        head = sample or ("(head)" if greedy else "(head_logits)")
        return self._run("(begin (nn__vector__add (nn__vector__range xs %d %d) zero_h x %d) %s)"
                         % (last * self.H, self.H, self.H, head), "the prompt's last row")

    # ── a stage of a pipeline: this machine's run of layers, its input and output the residual rows ─────────────
    # ⚖ *"we need to test pipeline parallelism"*. A model split over machines (▶ `pipeline`) runs each machine's layers
    # on what the one before handed on: the first machine starts from the token, the last ends with the head.
    def forward(self, pos, token=None, x=None, head=False):
        """Position `pos` through this machine's layers, from `token` (the first machine) or the residual row `x` (the
        bytes the machine before answered). Answers the next token when `head` — the greedy head, or the head expression
        it is (a sampler's draw, `sampling.head`; `(head_logits)`) — else this machine's residual row."""
        self._one_card("a stage of a pipeline")
        if head:
            self._needs_head()
        self._room(pos + 1)
        if token is not None:
            start = "(embed %d)" % token
        else:
            self.m.write(self.X_AT, x)
            start = ""
        end = (head if isinstance(head, str) else "(head)") if head else "(type x)"
        got = self._run("(begin %s (nn__rope__angles cs %d THETA ROTARY) (layers %d) %s)" % (start, pos, pos, end),
                        "position %d" % pos)
        return got if head else self.m.read(self.X_AT, self.H * 2)

    def forward_rows(self, first, tokens=None, xs=None, n=None, head=False, positions=None, rows=None):
        """A chunk of positions `first ..` as rows through this machine's layers, from `tokens` or from the rows `xs`.
        Answers the next token from the last row when `head`, else this machine's rows. A picture's: `rows` {index in
        the chunk: the tower's row} over its pads (with `tokens`) and the chunk's rotary `positions` (▶ `prefill`)."""
        self._one_card("a stage of a pipeline")
        if head:
            self._needs_head()
        if self.CPU_EXPERTS:
            # ⛳ the experts on the CPU read a prompt a position at a time (no rows words for them here): the chunk's
            #   positions through `forward` in turn — the same tokens, a pipeline's coordinator none the wiser
            H = self.H
            n = len(tokens) if tokens is not None else n
            rows, got = [], None
            for i in range(n):
                last = head and i == n - 1
                got = self.forward(first + i, token=tokens[i] if tokens is not None else None,
                                   x=None if tokens is not None else xs[i * H * 2:(i + 1) * H * 2], head=last)
                if not last:
                    rows.append(got)
            return got if head else b"".join(rows)
        m, H = self.m, self.H
        if (positions is not None or rows) and not (self.MROPE and self.PICTURES):
            raise Refused("%s reads no pictures here (▶ `vision`)" % self.folder.path)
        self._room(first + (len(tokens) if tokens is not None else n))
        if tokens is not None:
            n = len(tokens)
            self._prompt(tokens)
            start = "(embed_rows %d)" % n
            if rows:
                m.must("(begin (embed_rows %d) (type xs))" % n, "the rows' embeddings from %d" % first)
                self._rows_in(rows, 0, n)
                start = ""
        else:
            m.write(self.XS_AT, xs)
            start = ""
        if positions is not None:
            m.write(self.ROWPOS_AT, np.ascontiguousarray(positions[:n], dtype="<u4").tobytes())
        m.must("(begin (set! ROWPOS %s) %s (rows %d %d) (set! ROWPOS (< 1 0)) (type xs))"
               % ("(< 0 1)" if positions is not None else "(< 1 0)", start, first, n), "the rows from %d" % first)
        if not head:
            return m.read(self.XS_AT, n * H * 2)
        return m.must("(begin (nn__vector__add (nn__vector__range xs %d %d) zero_h x %d) %s)"
                      % ((n - 1) * H, H, H, head if isinstance(head, str) else "(head)"), "the chunk's last row")

    def logits(self):
        self._needs_head()
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
            at, p = self.slot_at[(l, n)], self.PLAN[self.KEY[(l, n)]]
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
        self._one_card("a LoRA swap")
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
    def _state_layout(self, length):
        """Every region a conversation of `length` positions has written, as (layer, offset in its state's slot, rows,
        sections): a row is its sections one after another, each section the bytes of this card's share of the heads —
        the DeltaNet's recurrent state one row of its value heads, its window one row of its q, k and v channels (three
        deep, a channel's window whole), a cache's region a row a position of its key-value heads. Cards' shares of a
        row, section by section in the cards' order, are one card's row."""
        out = []
        for l in self.run_layers:
            if l == self.MTP:                           # the MTP layer's cache is drafts', no conversation's: not saved
                continue
            if self.kinds[l] == "d":
                ch = lambda heads: heads * self.LH * 3 * 2                                  # noqa: E731
                out += [(l, self.SPLAN.at_up_data, 1, [self.S_BYTES]),
                        (l, self.SPLAN.at_down_data, 1, [ch(self.KH), ch(self.KH), ch(self.VH)])]
            elif self.TIERED:
                past = max(0, length - self.SINK)
                warm, cold = min(max(0, past - self.HOT), self.WARM), max(0, past - self.HOT - self.WARM)
                w, rb8, rb4 = self.KV_WIDE * 2, (self.AH + 1) // 2 * 2, (self.AH + 3) // 4 * 2
                rows = dict(sink=(min(length, self.SINK), w), hot=(min(past, self.HOT), w),
                            warm_c=(warm, self.KVH * rb8), warm_s=(warm, self.KVH * 2),
                            cold_c=(cold, self.KVH * rb4), cold_s=(cold, self.KVH * 2))
                for half in (self.KVPLAN.at_up_data, self.KVPLAN.at_down_data):
                    out += [(l, half + self.TIER[nm][0], n, [width]) for nm, (n, width) in rows.items()]
            else:
                out += [(l, half, length, [self.KV_WIDE * 2])
                        for half in (self.KVPLAN.at_up_data, self.KVPLAN.at_down_data)]
        return out

    def state_regions(self, length, where=None, card=0):
        """(address, bytes) of everything a conversation of `length` positions has written: each DeltaNet layer's
        state and window whole, each attention layer's key and value rows `0..length`. `where`: each layer's state's
        address — the load's by default; a table of zeros answers the sizes before the boot. Under tensor parallelism
        these are card `card`'s, its share of the heads: `state_read` and `state_write` take the whole."""
        at = (self.state_of[card] if where is None else where)
        return [(at[l] + off, rows * sum(secs)) for l, off, rows, secs in self._state_layout(length)]

    def state_read(self, length):
        """A conversation's state read off the card, region by region as one card holds it — so a conversation put away
        under tensor parallelism comes back on one card, or on another number of cards. Several cards' shares are
        stitched: each region's rows, section by section, card 0's share first."""
        if self.CARDS == 1:
            return [self.m.read(at, n) for at, n in self.state_regions(length) if n]
        parts = []
        for k in range(self.CARDS):
            self.m.act_as(k)
            parts.append([self.m.read(at, n) if n else b"" for at, n in self.state_regions(length, card=k)])
        self.m.act_as(-1)
        out = []
        for i, (l, _, rows, secs) in enumerate(self._state_layout(length)):
            if not rows * sum(secs):
                continue
            cut = np.cumsum([0] + secs)
            held = parts if self.kinds[l] == "d" else parts[::self.KV_REP]     # a copied head once
            mine = [np.frombuffer(p[i], dtype=np.uint8).reshape(rows, -1) for p in held]
            out.append(np.concatenate([c[:, cut[j]:cut[j + 1]] for j in range(len(secs)) for c in mine],
                                      axis=1).tobytes())
        return out

    def state_write(self, length, blobs):
        """A conversation's state, as `state_read` answers it, written back — each card its share of every region."""
        layout = [r for r in self._state_layout(length) if r[2] * sum(r[3])]
        self._room(length, exact=True)                  # its rows withheld from the tier before they are written
        P = self.CARDS
        shares = lambda l: P if self.kinds[l] == "d" else P // self.KV_REP          # noqa: E731 — a copied head once
        want = [rows * sum(secs) * shares(l) for l, _, rows, secs in layout]
        if [len(b) for b in blobs] != want:
            raise Refused("a state of %d regions (%d bytes) is not this model's at %d positions (%d regions, %d bytes)"
                          % (len(blobs), sum(map(len, blobs)), length, len(want), sum(want)))
        for k in range(P):
            at = self.state_of[k]
            if P > 1:
                self.m.act_as(k)
            for blob, (l, off, rows, secs) in zip(blobs, layout):
                if P == 1:
                    self.m.write(at[l] + off, blob)
                    continue
                full, cut = np.frombuffer(blob, dtype=np.uint8).reshape(rows, -1), np.cumsum([0] + secs)
                Q, q = shares(l), (k if self.kinds[l] == "d" else k // self.KV_REP)
                mine = [full[:, Q * cut[j] + q * secs[j]:Q * cut[j] + (q + 1) * secs[j]] for j in range(len(secs))]
                self.m.write(at[l] + off, np.concatenate(mine, axis=1).tobytes())
        if P > 1:
            self.m.act_as(-1)
        self._mtp_have, self._mtp_hid, self._mtp_next = -1, -1, None   # another conversation's: the MTP layer held nothing
