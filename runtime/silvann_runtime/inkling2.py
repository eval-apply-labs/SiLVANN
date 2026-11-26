"""INKLING 2 (`inkling2_loop_text`) — 24 layers of weights of which the middle twelve run three times, an attention whose
position is a bias a distance with a sink in its softmax, 128 experts of which a position takes 8 — the matrices on the
card, the routed experts on the CPU through the model's own verb (ai_thinkingmachines__inkling_2's) (NN-207).

⛳ THE PROGRAM IS nn's WORDS (▶ src/docs/design/pending/inkling2_mini.md, "Stage 3"): `lisp/inkling2.lisp`, a position at
a time, its arithmetic `scripts/inkling2_reference.py`'s. This class makes what the program reads — the collections, the
load, the buffers, the tables, the names (`_bindings`).

A conversation's STATE is, for each of the 48 applications of a layer, its keys and values and the three inputs before
the next position of its four short convolutions. A sliding layer's keys and values are a ring of its window's 1,024
positions, the same size whatever the context; a full layer's are a row a position — 8 applications, 32 KB a position.
The position bias's table is laid a row a distance for the program's `score` (▶ `_made`), and past position 8,191 a full
layer's scale is read from a half a position.

Answers on the card (`generate`, ▶ generate.py) and the card's sampler (▶ sampling.py), as the other families'.
"""
import math
import os
import pickle

import numpy as np

from . import experts_file, expert_tier, sampling
from .generate import GENERATE_MOST, STOPS_MOST, Generates
from .machine import CPU_FAMILIES, OPENCL1_FAMILIES
from .model_folder import NL, Refused
from .qwen4_exp import Qwen4Exp as _Sizes          # its buffer classes' arithmetic (`_classes`, `_pools`)

import nn_unpack_model as NU      # noqa: E402  (python/ on the path, as model_folder puts it)

GU, DN = NL.GATE_UP, NL.DOWN
CARD, CPU = 0, 1
ARCHITECTURES = ("inkling2_loop_text",)


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


PROJECTIONS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
               "self_attn.gate_attn.weight", "self_attn.wr_du.weight", "self_attn.attn_sink_proj.weight", "mlp.gate.weight")
VECTORS = ("self_attn.q_l2_clip.scale", "self_attn.rel_l2_clip.scale", "mlp_norm.weight", "mlp.gate.bias")
SLIDING = ("attn_norm.weight",)
FULL = ("q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight",
        "self_attn.k_sconv.weight_proj", "self_attn.v_sconv.weight_proj")
CONTROLLERS = tuple("controllers.%s.%s" % (w, p) for w in ("attn", "k", "v", "mlp")
                    for p in ("down_weight", "forget_weight", "gain_weight"))
# Matrices that read one vector, laid one after another in one slot — codes after codes, scales after scales — so the
# verbs take them in one product (▶ the package's contracts/objects/layer.cuh): the attention's over the query's input,
# the key's and the value's last since they read it only where theirs is the query's; a message gate's forget and gain.
# A layer whose members are not all there, or not all at one width, loads them one a slot.
JOINED = {"self_attn.query_input_products": ("self_attn.q_proj.weight", "self_attn.wr_du.weight",
                                             "self_attn.attn_sink_proj.weight", "self_attn.gate_attn.weight",
                                             "self_attn.k_proj.weight", "self_attn.v_proj.weight")}
JOINED.update({"controllers.%s.forget_gain" % w: ("controllers.%s.forget_weight" % w, "controllers.%s.gain_weight" % w)
               for w in ("attn", "k", "v", "mlp")})

# The three tables the model's verbs read — ▶ src/packages/ai_thinkingmachines__inkling_2/contracts/objects/layer.cuh:
# a layer's is its 22 matrices' buffers (codes, then row scales), its vectors and tables, then the matrices' widths and
# what kind of layer it is; a cell the layer has no use for holds any buffer.
LAYER_MATRICES, LAYER_CELLS = 22, 87
APPLICATION_CELLS = ("keys_%d", "values_%d", "key_history_%d", "value_history_%d", "attention_history_%d",
                     "experts_history_%d")
WORK_BUFFERS = ("stream", "states", "pass_states", "messages", "hand", "residual", "inputs", "attention_answer",
                "attention_convolved", "experts_mix", "scores", "bias", "mix_scores", "signs", "ones", "POSITION_SCALE",
                "work_scratch")
CONVOLUTIONS = (("key", "self_attn.k_sconv."), ("value", "self_attn.v_sconv."), ("attention", "attn_sconv."),
                ("experts", "mlp_sconv."))
# A prompt read as rows (▶ the package's layer.cuh, "A chunk"): the chunk's work table, laid as WORK — its buffers that
# hold a row a position the chunk's, the others a position's, the same as WORK's.
WORK_ROWS_BUFFERS = ("stream_rows", "states_rows", "pass_states_rows", "messages_rows", "hands", "residual_rows", "inputs",
                     "answer_rows", "attention_convolved", "experts_mix", "scores_rows", "bias", "mix_scores", "signs", "ones",
                     "POSITION_SCALE", "rows_scratch")
ROWS_MAX, STATES_ROWS = 256, 5         # the package's ROWS_MAX and STATES__ROWS


class Inkling2(Generates):
    PROGRAM = ("inkling2", "inkling2_cpu", "sampling", "generate")   # the pack's lisp/ files (▶ `ModelFolder.program`)
    BINDINGS = 1024
    SAMPLES_ON_CARD = True          # `head_sample` — ▶ `sampling`

    @staticmethod
    def longest_context(config):
        """The longest conversation this program holds: the model's own."""
        return int(config.get("text_config", config).get("model_max_length", 1024))

    def __init__(self, folder, machine, max_context=1024, experts_from="auto", experts_dir=None, experts_ram_gb=None,
                 experts_int8=False, experts_on="cpu", embed_on="auto", prompt_rows=ROWS_MAX, card_experts_gb=0,
                 card_line=None, card_landing=1, lend_cache=True):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b = folder, machine, b
        self.H, self.LAYERS = c["hidden_size"], c["num_hidden_layers"]
        self.QH, self.KVH, self.AD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.DREL, self.EXTENT, self.WINDOW = c["d_rel"], c["rel_extent"], c["sliding_window_size"]
        self.CONV_K, self.BLOCK = c["sconv_kernel_size"], c["n_layers_per_attn_res_block"]
        loop = c["loop_config"]
        self.LOOP_FROM, self.LOOP_TO, self.PASSES, self.RANK = loop["core_start"], loop["core_end"], loop["num_passes"], loop["gate_rank"]
        self.EXPERTS, self.TOPK, self.INTER = c["n_routed_experts"], c["num_experts_per_tok"], c["intermediate_size"]
        self.EPS, self.VOCAB = float(c["rms_norm_eps"]), int(c["unpadded_vocab_size"])
        self.FULL = [l for l in range(self.LAYERS) if l not in c["local_layer_ids"]]
        if max_context > int(c["model_max_length"]):
            raise Refused("max_context %d is past the model's own %d" % (max_context, c["model_max_length"]))
        if self.EXTENT != self.WINDOW:
            raise Refused("a sliding layer's window (%d) is not the position bias's extent (%d): this program reads one "
                          "ring by one table" % (self.WINDOW, self.EXTENT))
        self.TAU_ALPHA, self.TAU_FLOOR = float(c["log_scaling_alpha"]), int(c["log_scaling_n_floor"])
        if c.get("n_shared_experts") or loop.get("is_varloop") or self.CONV_K != 4 or self.PASSES != 3 \
                or (self.LOOP_TO - self.LOOP_FROM) % self.BLOCK or self.LOOP_FROM % self.BLOCK:
            raise Refused("a shape of Inkling 2 this program was not written for (a shared expert, a loop of varying "
                          "passes, another kernel, or a loop that does not begin and end on a block)")
        self.max_context = max_context
        self.GATE_CAP, self.UP_CAP = float(c["mlp_gate_softcap"]), float(c["mlp_up_softcap"])
        self.ROUTE_SCALE, self.LOGIT_CAP = float(c["route_scale"]), float(c["final_logit_softcapping"])
        self.QK_CLIP, self.REL_CLIP = float(c["qk_l2_clip_threshold"]), float(c["rel_l2_clip_threshold"])
        self.run_layers = list(range(self.LAYERS))
        # an application of a layer: layers before the loop once, the loop's a pass each, the layers after once
        span = self.LOOP_TO - self.LOOP_FROM
        self.APPLICATIONS = self.LOOP_FROM + self.PASSES * span + (self.LAYERS - self.LOOP_TO)
        self.global_ = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.EXPERTS_INT8 = bool(experts_int8)
        self.MOE = True                            # routed experts: what the boot's setup places (▶ autotune)
        # ⭐ WHERE THE ROUTED EXPERTS ARE (`experts_on`): "cpu" — in the CPU worker's memory (or a file's, `experts_from`),
        #   a layer's handed it by the card; "card" — every expert on the card beside the matrices, the whole model there,
        #   the card running the model's experts verb itself (12.3 GB of them at 4 bits: a 16 GB card's).
        if experts_on not in ("cpu", "card"):
            raise Refused("experts_on is \"cpu\" or \"card\", not %r" % (experts_on,))
        self.EXPERTS_ON = experts_on
        # ⭐ THE EMBEDDING ON THE CPU (`embed_on`), as Qwen's and Ministral's: a token's row decoded by the CPU worker from
        #   its own copy of the table and handed to the card, HIDDEN halves. Through an OpenCL 1.2 family the card's reading
        #   its row in place keeps the table twice and crosses it whole each token, both ways — `MEASURED` on the
        #   precision's 5060 Ti 463 MB a way, a token 161–170 ms for the card's 31.9 (measurements/2026-11-24_nn213_
        #   inkling2_rates_precision.md ⑧). `"auto"`: the CPU there alone; this program has the CPU worker either way.
        if embed_on not in ("auto", "card", "cpu"):
            raise Refused("embed_on is 'auto', 'card' or 'cpu', not %r" % (embed_on,))
        self.EMBED_CPU = embed_on == "cpu" or (embed_on == "auto" and machine.silicon in OPENCL1_FAMILIES)
        # ⭐ A PROMPT AS ROWS (`prompt_rows`): its positions a chunk of this many at a time, every application over the chunk
        #   at once (▶ inkling2.lisp `rows`); 0 reads it a position at a time, as a conversation's later positions are
        if not 0 <= int(prompt_rows) <= ROWS_MAX:
            raise Refused("prompt_rows is 0 … %d, not %r" % (ROWS_MAX, prompt_rows))
        self.ROWS = int(prompt_rows)
        # ⭐ A CARD'S TIER OF THE ROUTED EXPERTS (`card_experts_gb`, NN-48's, as Qwen's): the card keeps that many GB of the
        #   experts picked most, a slot an expert laid as the CPU's, promoted from the CPU's memory — the experts in RAM,
        #   or the disk's cache there — while the CPU keeps every one; a position's picks the card holds computed there,
        #   the rest by the CPU, the two sums added (▶ inkling2.lisp `experts`). `card_line` fixes the prompt's threshold
        #   of picks past which a chunk promotes an expert (else measured at boot, ▶ `expert_tier.calibrate`);
        #   `card_landing` the promotions a visit lands. ⛳ A routed sum split between the card and the CPU repeats from
        #   boot, not from one request to the next, as the tier fills (NN-70). `MEASURED` node03, the experts in RAM: a
        #   4 GB tier 24.1 tokens/s against 16.8, the prompt 132 positions a second against 44
        #   (measurements/2026-11-26_inkling2_card_tier_node03.md).
        if not isinstance(card_experts_gb, (int, float)) or isinstance(card_experts_gb, bool) or card_experts_gb < 0:
            raise Refused("card_experts_gb is the card's room for a tier of experts, in GB, not %r" % (card_experts_gb,))
        self.CARD_GB, self.CARD_LINE, self.CARD_LANDING = float(card_experts_gb), card_line, int(card_landing)
        if self.CARD_GB and experts_on == "card":
            raise Refused("with the experts on the card there is no tier of them: no card_experts_gb")
        if self.CARD_GB and machine.cpu not in CPU_FAMILIES:
            raise Refused("a card's tier of experts is promoted from a processor's memory: the CPU worker is %s, not a "
                          "processor's family" % (machine.cpu,))
        # ⭐ THE FULL LAYERS' ROWS NO CONVERSATION HAS REACHED, LENT TO THE TIER (NN-93, as Qwen's — ▶ `_lend`):
        #   `lend_cache` false keeps the tier to its own run. `MEASURED` node03, 16 GB of an MI50: the boot's choice at max
        #   440,320 positions, the tier lent 13.4 GB (measurements/2026-11-26_inkling2_lend_node03.md)
        self.LEND_ON = bool(lend_cache)
        self._plan()
        if experts_on == "card":
            experts_from = "ram"                   # none of them the CPU's: nothing read from a file
        self.EXPERTS_FROM = experts_file.choose(experts_from, self.EXPERTS * self.LAYERS * self.EPLAN.slot_bytes)
        if self.CARD_GB and self.EXPERTS_FROM not in ("ram", "disk"):
            raise Refused("a card's tier of experts comes from the experts in RAM or their cache there — experts_from 'ram' "
                          "or 'disk', not %r" % (self.EXPERTS_FROM,))
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        self.SLOTS = None
        if self.EXPERTS_FROM == "disk":
            self.SLOTS = min(self.EXPERTS * self.LAYERS,
                             experts_file.cache_bytes(self.EXPERTS * self.LAYERS * self.EPLAN.slot_bytes, experts_ram_gb)
                             // self.EPLAN.slot_bytes)
            if self.SLOTS < 4 * self.TOPK:
                raise Refused("%d experts is too few for a cache of them" % self.SLOTS)
        self.cpu_arena = (self.SLOTS if self.SLOTS is not None else self.EXPERTS * self.LAYERS) * self.EPLAN.slot_bytes

    # ── what goes where ─────────────────────────────────────────────────────────────────────────────────────────────
    def loops(self, l):
        return self.LOOP_FROM <= l < self.LOOP_TO

    def names_of(self, l):
        """The records of layer `l` the card holds as planes: its matrices and the vectors the program reads in place."""
        n = list(PROJECTIONS) + list(VECTORS) + (list(FULL) if l in self.FULL else list(SLIDING))
        return n + (list(CONTROLLERS) if self.loops(l) else [])

    def _loaded_of(self, l):
        """What layer `l` loads, a slot each: its records, the members of a join in their joined slot (▶ `JOINED`)."""
        names, joins = set(self.names_of(l)), []
        for j, members in JOINED.items():
            if all(m in names for m in members):
                ds = {self.b.record(l, m).d for m in members}
                if len(ds) == 1 and ds != {16}:
                    joins.append(j)
        joined = {m for j in joins for m in JOINED[j]}
        return [n for n in self.names_of(l) if n not in joined] + joins

    def _joined_plan(self, j, l):
        """A joined slot's plan — its members' codes one after another, then their scales — and where each member lies
        in it (`MEMBER`: the join, its codes' offset and bytes, its scales' offset and bytes)."""
        at_d = at_l = 0
        for m in JOINED[j]:
            r = self.b.record(l, m)
            d_n, l_n = r.data_span()[1], r.lut_span()[1]
            self.MEMBER[m] = (j, at_d, d_n, at_l, l_n)
            at_d, at_l = at_d + d_n, at_l + l_n
        return NL.SlotPlan(at_d, 0, 0, at_l)

    def _decode(self, l, n):
        """A layer's record as the values it stands for (the pack's reader's: exact ones are the halves themselves)."""
        r = self.b.record(l, n)
        return NU.decode_record(r.rec, self.b.blob(l)).astype(np.float64)

    def _global(self, name):
        """A global record kept exact, as its halves' values."""
        g = self.global_[name]
        if int(g["d"]) != 16:
            raise Refused("%s is packed at d=%d: this program reads it exact" % (name, g["d"]))
        return np.frombuffer(np.asarray(g["vbr_data"]).tobytes(), dtype="<f2").astype(np.float64).reshape(g["original_shape"])

    def _plan(self):
        b, H = self.b, self.H
        for n in sorted({n for l in self.run_layers for n in self.names_of(l)}):   # a matrix's width one for every layer:
            ds = {b.record(l, n).d for l in self.run_layers if n in self.names_of(l)}  # the program reads one `mv`
            if len(ds) != 1:
                raise Refused("%s is packed at widths %s across the layers" % (n, sorted(ds)))
        self.LOADED = {l: self._loaded_of(l) for l in self.run_layers}
        self.ALL = sorted({n for l in self.run_layers for n in self.LOADED[l]})
        self.TYPE = {n: i for i, n in enumerate(self.ALL)}
        first = {n: next(l for l in self.run_layers if n in self.LOADED[l]) for n in self.ALL}
        self.MEMBER = {}
        self.PLAN = {n: self._joined_plan(n, first[n]) if n in JOINED else NL.dense_plan(b, n, first[n]) for n in self.ALL}
        self.COUNT = {n: sum(1 for l in self.run_layers if n in self.LOADED[l]) for n in self.ALL}
        self.EPLAN = NL.expert_plan(b, 0, self.EXPERTS)
        head = self.global_["lm_head.weight"]
        data, lut = np.asarray(head["vbr_data"]).tobytes(), np.asarray(head["vbr_lut"]).astype("<f2").tobytes()
        self.GLOBAL = {"head": (data, lut, NL.SlotPlan(len(data), 0, 0, len(lut)))}
        self.T_GLOBAL = {k: len(self.ALL) + i for i, k in enumerate(self.GLOBAL)}
        self.T_EXPERTS = len(self.ALL) + len(self.GLOBAL)          # the card's own experts, where it holds them
        self.T_CEXP = self.T_EXPERTS                               # …or its tier's slots, where it keeps one
        self.CARD_TIER = self.CARD_GB > 0
        self.CARD_SLOTS = (max(self.LAYERS, int(self.CARD_GB * (1 << 30)) // self.EPLAN.slot_bytes)
                           if self.CARD_TIER else 0)            # a slot a layer at the least
        self.HEAD_BITS = int(head["d"])
        emb = self.global_["model.language_model.embed_tokens.weight"]
        self.emb_data = np.asarray(emb["vbr_data"]).tobytes()
        self.emb_lut = np.asarray(emb["vbr_lut"]).astype("<f2").tobytes()
        self.EMB_ROW_BYTES, self.EMB_BITS, self.EMB_DATA_BYTES = int(emb["row_bytes"]), int(emb["d"]), len(self.emb_data)
        self.TABLE_BYTES = (len(self.emb_data) + len(self.emb_lut) + 255) // 256 * 256
        self.arena = (sum(self.PLAN[n].slot_bytes * self.COUNT[n] for n in self.ALL)
                      + sum(p.slot_bytes for _, _, p in self.GLOBAL.values())
                      + (self.EXPERTS * self.LAYERS * self.EPLAN.slot_bytes if self.EXPERTS_ON == "card" else 0)
                      + self.CARD_SLOTS * self.EPLAN.slot_bytes)
        QW, KVW, RW, I, ctx = self.QH * self.AD, self.KVH * self.AD, self.QH * self.DREL, self.INTER, self.max_context
        # the card's working buffers, their bytes (▶ `_buffers`)
        # the verbs' scratch, as layer__abi.cuh lays it out: every part a site keeps to itself
        scratch = (3 * H + 6 * H + 4 * H + 2 * self.RANK + 2 * H + 3 * QW + 4 * KVW + self.KVH * self.CONV_K
                   + 4 * RW + self.QH + 5 * QW + 2 * H + self.EXPERTS + H)
        # a chunk of a prompt as rows: a row a position of the stream, its states, what a pass leaves, the messages, the
        #   hand (its picks in it), the residual, the attention's answer, the experts' sums; the scratch a position's
        #   parts and then the chunk's (▶ layer__abi.cuh `chunk`); nn's rows routine's scratch over the chunk's picks
        R = max(self.ROWS, 1)
        self.HAND_ROW = ((2 * (H + 16) + 7) & ~7) + 8 * self.TOPK
        per_row = (4 + 25 * H + 7 * QW + 4 * RW + self.QH + 4 * KVW + 2 * self.KVH * self.CONV_K + 2 * self.RANK
                   + self.EXPERTS)             # ▶ layer__abi.cuh `chunk`: its parts, a row a position
        picks = R * self.TOPK
        self.ROUTED_ROWS_SCRATCH = max(1 << 20, 2 * R * H + 18 * picks + 8 * picks * I + 4 * picks * H + 64)
        self.BUFS = dict(
            signs=1024, ones512=1024, signs_row=H * 2, ones=H * 2, zeros=H * 2, res=64, stream=H * 2, loop_start=H * 2,
            states=STATES_ROWS * H * 2, pass_states=2 * H * 2, messages=4 * H * 2, mix_scores=4 * 8 * 4, inputs=3 * H * 2,
            attention_answer=H * 2, attention_convolved=H * 2, residual=H * 2, experts_mix=H * 2, hand=2 * (H + 16),
            routed=H * 2, bias=self.QH * ctx * 4, scores=self.QH * ctx * 4,
            POSITION_SCALE=max(64, ctx * 2 if ctx > self.TAU_FLOOR else 64), work_scratch=scratch * 2,
            # the card's experts' scratch: a chunk's where it computes any; with a tier also an expert's copy, what the
            #   boot times a promotion by (▶ `_tier`)
            experts_scratch=(max(self.ROUTED_ROWS_SCRATCH, self.EPLAN.slot_bytes) if self.CARD_TIER else
                             self.ROUTED_ROWS_SCRATCH if self.EXPERTS_ON == "card" else 1 << 20),
            embedding_rotated=H * 2, embedding_unsigned=H * 2, embedding=H * 2, final_mix=H * 2,
            final_normed=H * 2, final_normed_rotated=H * 2, logits_h=self.VOCAB * 2)
        self.BUFS.update(
            stream_rows=R * H * 2, loop_start_rows=R * H * 2, states_rows=R * STATES_ROWS * H * 2,
            pass_states_rows=R * 2 * H * 2, messages_rows=R * 4 * H * 2, hands=R * self.HAND_ROW,
            residual_rows=R * H * 2, answer_rows=R * H * 2, routed_rows=R * H * 2,
            rows_scratch=(scratch + 4 + R * per_row + 2 * (self.WINDOW + R) * KVW) * 2,
            # the chunk's attention scores, a row's heads over its window: as many rows as 64 MB holds, one at the least
            scores_rows=max(self.QH * ctx * 4, min(R * self.QH * max(self.WINDOW, ctx) * 4, 64 << 20)))
        if self.CARD_TIER:                         # the card's share of a position's sum, and of a chunk's
            self.BUFS.update(routed_tier=H * 2, routed_rows_tier=R * H * 2)
        # what the program reads that the load makes: the convolutions' taps, the bias's table a row a distance, the
        # learned queries, the two global norms (▶ `_made`) — a name and its bytes, written once
        self.MADE = {}
        # a conversation's state: an application's keys and values — a full layer's a row a position, a sliding one's
        #   the ring of its window — and the three inputs before the next position of its four convolutions
        self.STATE = {}
        for a in range(self.APPLICATIONS):
            rows = ctx if self.layer_of(a) in self.FULL else min(ctx, self.WINDOW)
            self.STATE["keys_%d" % a] = self.STATE["values_%d" % a] = rows * KVW * 2
            self.STATE["key_history_%d" % a] = self.STATE["value_history_%d" % a] = 3 * KVW * 2
            self.STATE["attention_history_%d" % a] = self.STATE["experts_history_%d" % a] = 3 * H * 2
        # ⭐ WITH A TIER, THE FULL LAYERS' KEYS AND VALUES IN THE EXPERTS' ARENA (NN-93): a collection of their own, a slot
        #   an application — keys in its up half, values in its down — so the rows past a conversation's length, which
        #   nothing reads (▶ the package's layer__abi.cuh: a full layer reads `pos + 1` rows), are memory the tier may be
        #   lent: `most` slots a region at the most, two regions an application
        self.T_KV, self.KVPLAN, self.KV_APPS, self.LEND = None, None, [], None
        self.TIER_LEAST = self.LAYERS * self.EPLAN.slot_bytes           # the least tier it takes: a slot a layer
        full = [a for a in range(self.APPLICATIONS) if self.layer_of(a) in self.FULL]
        if self.CARD_TIER and self.LEND_ON and full:
            kv = ctx * KVW * 2
            self.T_KV, self.KVPLAN, self.KV_APPS = self.T_EXPERTS + 1, NL.SlotPlan(kv, 0, 0, kv), full
            for a in full:
                del self.STATE["keys_%d" % a], self.STATE["values_%d" % a]
            self.arena += len(full) * self.KVPLAN.slot_bytes
            most = ctx * KVW * 2 // self.EPLAN.slot_bytes
            if most:
                self.LEND = dict(row=KVW * 2, regions=2 * len(full), most=most)
        # a position's place in a ring, taken from the position by subtraction: the window times each power of two
        #   below the context, the largest first (▶ inkling2.lisp `ring`)
        self.RING_POW = []
        w = self.WINDOW
        while w < ctx:
            self.RING_POW.insert(0, w)
            w *= 2

    def layer_of(self, a):
        """The layer application `a` is of: the layers before the loop, the loop's a pass after another, the rest."""
        span = self.LOOP_TO - self.LOOP_FROM
        if a < self.LOOP_FROM:
            return a
        if a < self.LOOP_FROM + self.PASSES * span:
            return self.LOOP_FROM + (a - self.LOOP_FROM) % span
        return a - (self.PASSES - 1) * span

    def _bias_table(self, l):
        """Layer `l`'s position bias, [16, extent], laid a row a distance for the program's `score` — the last row it
        reads is distance 0, as a cache's last row is the position itself. A sliding layer's: the table, the furthest
        distance first, twice over — a ring standing at `slot` reads it from `window − 1 − slot`. A full layer's: as
        many rows as the context, the furthest first, zeros where the distance is past the extent."""
        far_first = np.ascontiguousarray(self._decode(l, "self_attn.rel_logits_proj.proj").T[::-1])
        if l not in self.FULL:
            return np.concatenate([far_first, far_first])
        rows = np.zeros((self.max_context, self.DREL))
        n = min(self.max_context, self.EXTENT)
        rows[-n:] = far_first[-n:]
        return rows

    def _made(self):
        """The buffers the load writes from the pack's exact records, by name."""
        H, made = self.H, {}
        for l in self.run_layers:
            full = l in self.FULL
            for key, rec in CONVOLUTIONS:               # a row a lag, this position's first
                if full and key in ("key", "value"):       # [4, channels]
                    taps = self._decode(l, rec + "static_w")
                else:                              # [channels, 1, 4]: the last tap this position's — lag 0
                    taps = self._decode(l, rec + "weight")[:, 0, ::-1].T
                made["%s_taps_%d" % (key, l)] = _half(np.ascontiguousarray(taps[:self.CONV_K]))
            made["bias_table_%d" % l] = _half(self._bias_table(l))
            if l:
                made["attention_query_%d" % l] = _half(self._decode(l, "attn_query"))
            made["experts_query_%d" % l] = _half(self._decode(l, "mlp_query"))
        qm = self._global("model.language_model.cross_pass_queries")
        qkv = self._global("model.language_model.cross_pass_kv_queries")
        for i, l in enumerate(range(self.LOOP_FROM, self.LOOP_TO)):   # to the query, the key, the value, the experts
            made["message_queries_%d" % l] = _half(np.stack([qm[2 * i], qkv[2 * i], qkv[2 * i + 1], qm[2 * i + 1]]))
        made["embedding_norm"] = _half(self._global("model.language_model.embed_norm.weight"))
        made["final_norm"] = _half(self._global("model.language_model.norm.weight"))
        made["final_query"] = _half(self._global("model.language_model.output_query"))
        return made

    def _made_sizes(self):
        """`_made`'s names and bytes without decoding a record: what the pools are sized by before the boot."""
        H, KVW, sizes = self.H, self.KVH * self.AD, {}
        for l in self.run_layers:
            for key, n in (("key", KVW), ("value", KVW), ("attention", H), ("experts", H)):
                sizes["%s_taps_%d" % (key, l)] = self.CONV_K * n * 2
            sizes["bias_table_%d" % l] = (self.max_context if l in self.FULL else 2 * self.EXTENT) * self.DREL * 2
            if l:
                sizes["attention_query_%d" % l] = (3 if l in self.FULL else 1) * H * 2
            sizes["experts_query_%d" % l] = H * 2
        for l in range(self.LOOP_FROM, self.LOOP_TO):
            sizes["message_queries_%d" % l] = 4 * H * 2
        sizes.update(embedding_norm=H * 2, final_norm=H * 2, final_query=H * 2)
        return sizes

    def device_params(self):
        """The boot's sections: the card's (its pools, its arena, its collections) and the CPU worker's (the experts)."""
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (self.TYPE[n], self.PLAN[n].up_bytes, self.TYPE[n], self.PLAN[n].down_bytes, self.TYPE[n],
                           self.COUNT[n]) for n in self.ALL)
        for k, t in self.T_GLOBAL.items():
            p = self.GLOBAL[k][2]
            types += ("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:1\n"
                      % (t, p.up_bytes, t, p.down_bytes, t))
        n_types = len(self.ALL) + len(self.T_GLOBAL)
        if self.EXPERTS_ON == "card" or self.CARD_TIER:          # every expert, or the tier's slots
            e = self.EPLAN
            types += ("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                      % (self.T_EXPERTS, e.up_bytes, self.T_EXPERTS, e.down_bytes, self.T_EXPERTS,
                         self.CARD_SLOTS if self.CARD_TIER else self.EXPERTS * self.LAYERS))
            n_types += 1
            if self.LEND:                          # the links for the slots the tier may be lent (▶ nn's `__lendable`)
                types += "nn__expert__type%d__lendable:%d\n" % (self.T_CEXP, self.LEND["regions"] * self.LEND["most"])
        if self.T_KV is not None:                  # the full layers' keys and values, a slot an application
            p = self.KVPLAN
            types += ("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                      % (self.T_KV, p.up_bytes, self.T_KV, p.down_bytes, self.T_KV, len(self.KV_APPS)))
            n_types += 1
        common = ("nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                  "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                  "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                  "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                  "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:0\n"
                  "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                  "nn__model__layer_types:%s\n" % (self.LAYERS, "f" * self.LAYERS))
        H = self.H
        self.MADE_BYTES = self._made_sizes()
        card_asks = list(self.BUFS.values()) + list(self.MADE_BYTES.values()) + list(self.STATE.values())
        reach = max(self.BUFS.values())
        # the card's and the CPU's classes the same sizes, name for name: a copy between the two reads the source
        #   buffer's room from its class in the copying worker's own table (▶ qwen4_exp.py `device_params`)
        self.CLASSES = _Sizes._classes(card_asks, 2, reach)
        card = (_Sizes._pools(self.CLASSES, card_asks, 2, reach)
                + "nn__expert_cache_l1__size_mb:%d\n" % ((self.arena + (256 << 20)) >> 20) + common
                + "nn__expert__types:%d\n%s" % (n_types, types)
                + "nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                  "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                  "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n"
                  % (256 if self.EMBED_CPU else self.TABLE_BYTES))
        p = self.EPLAN
        slots = self.SLOTS if self.SLOTS is not None else self.EXPERTS * self.LAYERS
        if self.EXPERTS_ON == "card":
            slots = 1                              # the CPU worker holds none
        self.cpu_arena = slots * p.slot_bytes
        self.CPU_BUFS = dict(cpu_signs=1024, cpu_zeros=H * 2, cpu_hand=2 * (H + 16), cpu_routed=H * 2,
                             cpu_scratch=self.ROUTED_ROWS_SCRATCH,
                             # a chunk's hands and the experts' sums over it (▶ inkling2_cpu.lisp)
                             cpu_hands=max(self.ROWS, 1) * self.HAND_ROW, cpu_routed_rows=max(self.ROWS, 1) * H * 2)
        if self.EMBED_CPU:
            self.CPU_BUFS["c_emb"] = H * 2         # a token's row, decoded from the CPU's table (▶ inkling2_cpu.lisp)
        cpu_asks = list(self.CPU_BUFS.values())
        cpu_classes = self.CLASSES[:-1] + [max(self.CLASSES[-1], _Sizes._room(max(cpu_asks)))]
        cpu = (_Sizes._pools(cpu_classes, cpu_asks, 6, max(cpu_asks))
               + "nn__expert_cache_l1__size_mb:%d\n" % ((self.cpu_arena + (256 << 20)) >> 20) + common
               + "nn__expert__types:1\nnn__expert__type0__up_bytes:%d\nnn__expert__type0__down_bytes:%d\n"
                 "nn__expert__type0__slots:%d\n" % (p.up_bytes, p.down_bytes, slots))
        if self.EMBED_CPU:                         # the table's room in the CPU worker's RAM, not the card's
            cpu += ("nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                    "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                    "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % self.TABLE_BYTES)
        return {CARD: card, CPU: cpu}

    def workers(self):
        return [(self.m.silicon, 0), (self.m.cpu, 0)]

    # ── the load ────────────────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        m.act_as(CARD)
        loader = m.loader()
        for i, l in enumerate(self.run_layers):
            for n in self.LOADED[l]:
                loader.layer_width(l, self.TYPE[n], 1)
                if n not in JOINED:
                    loader.load_dense(b, self.PLAN[n], self.TYPE[n], n, l)
                    continue
                parts = [NL.dense_part(b, member, l) for member in JOINED[n]]
                for member, (data, lut) in zip(JOINED[n], parts):
                    if (len(data), len(lut)) != (self.MEMBER[member][2], self.MEMBER[member][4]):
                        raise Refused("layer %d's %s is not the size of the first layer's" % (l, member))
                loader.load_bytes(self.PLAN[n], self.TYPE[n], l, b"".join(d for d, _ in parts), b"".join(u for _, u in parts))
            if self.CARD_TIER:                     # the tier's band of the layer, empty at boot
                loader.layer_width(l, self.T_CEXP, self.EXPERTS)
            if progress:
                progress(i + 1, len(self.run_layers))
        self.KV_AT = {}
        if self.T_KV is not None:                  # the full layers' keys and values, a slot an application
            loader.layer_width(0, self.T_KV, len(self.KV_APPS))
            for i, a in enumerate(self.KV_APPS):
                at = loader.reserve(self.T_KV)
                loader.assign(0, self.T_KV, i, at)
                self.KV_AT[a] = at
        for k, t in self.T_GLOBAL.items():
            data, lut, p = self.GLOBAL[k]
            loader.layer_width(0, t, 1)
            loader.load_bytes(p, t, 0, data, lut)
        self.GLOBAL = {k: v[2] for k, v in self.GLOBAL.items()}       # their bytes the engine's now
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())
        if self.EMBED_CPU:                         # the CPU worker's own (▶ `__init__`)
            m.act_as(CPU)
        self.EMB_AT = m.buffer(self.TABLE_BYTES, "c_embedding" if self.EMBED_CPU else "EMBEDDING", ram=True)
        m.write(self.EMB_AT, self.emb_data)
        m.write(self.EMB_AT + self.EMB_DATA_BYTES, self.emb_lut)
        m.act_as(CARD)
        self.emb_data = self.emb_lut = None
        self._buffers()
        self._tables()
        if self.EXPERTS_ON == "card":
            self._card_experts(progress)
        self._cpu_experts(progress)
        if self.CARD_TIER:
            self._tier()
            self._lend()
        self._bindings()
        m.define_programs(self._procedures)
        self.b.release()

    def _signs(self):
        return 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                         bitorder="little").astype(np.float64)

    def _buffers(self):
        m, H = self.m, self.H
        signs = self._signs()
        # a row's signs as the rotation's blocks lay them: each block the sign vector's first values
        row = np.concatenate([signs[:w] for w in NU.NC.blocks_or_refuse(H)])
        given = dict(signs=_half(signs[:512]), ones512=_half(np.ones(512)), signs_row=_half(row), ones=_half(np.ones(H)),
                     zeros=_half(np.zeros(H)))
        if self.max_context > self.TAU_FLOOR:      # a full layer's scale past position 8,191, a half a position
            at = np.arange(1, self.max_context + 1, dtype=np.float64)
            given["POSITION_SCALE"] = _half(1.0 + self.TAU_ALPHA * np.log(np.maximum(at / self.TAU_FLOOR, 1.0)))
        self.AT = {}
        for name, n in self.BUFS.items():
            self.AT[name] = m.buffer(n, name, given.get(name))
        made = self._made()
        if {k: len(v) for k, v in made.items()} != self.MADE_BYTES:
            raise Refused("the pack's exact records are not the sizes this program was booted for")
        for name, payload in made.items():
            self.AT[name] = m.buffer(len(payload), name, payload)
        for name, n in self.STATE.items():
            self.AT[name] = m.buffer(n, name)
        for a, at in self.KV_AT.items():           # with a tier, the arena's (▶ `_plan`)
            self.AT["keys_%d" % a] = at + self.KVPLAN.at_up_data
            self.AT["values_%d" % a] = at + self.KVPLAN.at_down_data
        self.reset()
        m.node_array("picks", self.TOPK)
        m.node_array("HERE", 2)
        m.node_array("PROMPT", max(self.ROWS, 1))   # a prompt chunk's tokens (▶ `_prompt`)
        self._answer_buffers()                      # what `generate` made, the end tokens it watches for
        sampling.buffers(m)

    def plane(self, l, n, which):
        if n in self.MEMBER and self.MEMBER[n][0] in self.LOADED[l]:     # a member of a join: its span of the joined slot
            j, at_d, d_n, at_l, l_n = self.MEMBER[n]
            p = self.PLAN[j]
            at, size = (p.at_up_data + at_d, d_n) if which == "data" else (p.at_down_lut + at_l, l_n)
            return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.TYPE[j], at, size)
        p = self.PLAN[n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.TYPE[n], at, size)

    def _matrix(self, l, n):
        return [self.plane(l, n, "data"), self.plane(l, n, "lut"), int(self.b.record(l, n).d)]

    def _tables(self):
        """The three tables the model's verbs read: a layer's weights, an application's own memory — its cache and its
        convolutions' inputs — and the work every application shares (▶ the package's contracts/objects/layer.cuh)."""
        m, H = self.m, self.H
        absent = ["zeros", "zeros", 16]             # a matrix the layer has no use for: any buffer, a width
        for l in self.run_layers:
            full, names = l in self.FULL, self.names_of(l)
            matrix = lambda n: self._matrix(l, n) if n in names else absent
            matrices = [matrix(n) for n in PROJECTIONS]
            matrices += [matrix(n) for n in FULL[3:5]] if full else [absent] * 2
            matrices += [matrix(n) for n in CONTROLLERS] if self.loops(l) else [absent] * len(CONTROLLERS)
            cells = [c for one in matrices for c in one[:2]]
            cells += [self.plane(l, n, "data") if n in names else "zeros"
                      for n in VECTORS + (FULL[:3] if full else SLIDING * 3)]
            cells += ["%s_taps_%d" % (key, l) for key, _ in CONVOLUTIONS]
            cells += ["bias_table_%d" % l, "attention_query_%d" % l if l else "zeros", "experts_query_%d" % l,
                      "message_queries_%d" % l if self.loops(l) else "zeros"]
            cells += [one[2] for one in matrices]
            # it attends the whole context · a block of six begins here · that block's row of the states · the block
            #   states its attention reads — those begun before it — and its experts, its own block's too · the first
            cells += [int(full), int(l % self.BLOCK == 0), l // self.BLOCK, (l + self.BLOCK - 1) // self.BLOCK,
                      l // self.BLOCK + 1, int(l == 0)]
            if len(matrices) != LAYER_MATRICES or len(cells) != LAYER_CELLS:
                raise Refused("layer %d's table is %d cells of %d" % (l, len(cells), LAYER_CELLS))
            m.table("layer_%d" % l, cells)
        m.array("LAYER_TABLES", ["layer_%d" % l for l in self.run_layers])
        for a in range(self.APPLICATIONS):
            cells = [cell % a for cell in APPLICATION_CELLS]
            if a in self.KV_APPS:                  # with a tier, its keys and values a slot of the arena's: a plane each
                i, p, kv = self.KV_APPS.index(a), self.KVPLAN, self.KVPLAN.up_data
                cells[:2] = ["(nn__expert__plane 0 %d %d %d %d)" % (self.T_KV, i, p.at_up_data, kv),
                             "(nn__expert__plane 0 %d %d %d %d)" % (self.T_KV, i, p.at_down_data, kv)]
            m.table("application_%d" % a, cells)
        m.array("APPLICATION_TABLES", ["application_%d" % a for a in range(self.APPLICATIONS)])
        scaled_from = self.TAU_FLOOR if self.max_context > self.TAU_FLOOR else self.max_context
        m.table("WORK", list(WORK_BUFFERS) + [
            H, self.QH, self.KVH, self.AD, self.DREL, self.RANK, self.EXPERTS, self.TOPK, self.CONV_K, self.WINDOW,
            self.max_context, scaled_from,
            float(self.EPS), float(self.QK_CLIP), float(self.REL_CLIP), float(self.ROUTE_SCALE), float(math.sqrt(self.DREL)),
            float(1.0 / math.sqrt(self.CONV_K + 1)), float(1.0 / math.sqrt(2 * self.CONV_K + 1))])
        m.array("RING_POWERS", self.RING_POW or [0])
        m.table("WORK_ROWS", list(WORK_ROWS_BUFFERS) + [        # the chunk's work, laid as WORK (▶ `WORK_ROWS_BUFFERS`)
            H, self.QH, self.KVH, self.AD, self.DREL, self.RANK, self.EXPERTS, self.TOPK, self.CONV_K, self.WINDOW,
            self.max_context, scaled_from,
            float(self.EPS), float(self.QK_CLIP), float(self.REL_CLIP), float(self.ROUTE_SCALE), float(math.sqrt(self.DREL)),
            float(1.0 / math.sqrt(self.CONV_K + 1)), float(1.0 / math.sqrt(2 * self.CONV_K + 1))])

    def _expert_pieces(self, view, l, x, part):
        """Expert `x` of layer `l` as its slot's four pieces (▶ `experts_file`), the runs of the layer `view` it lies in."""
        b, p, out = self.b, self.EPLAN, []
        for rec, data_into, lut_into in ((b.record(l, GU), p.at_up_data, p.at_up_lut),
                                         (b.record(l, DN), p.at_down_data, p.at_down_lut)):
            (d, nd), (s, ns) = rec.expert_spans(x, self.EXPERTS)
            out += [(data_into, view[d:d + nd]), (lut_into, view[s:s + ns])]
        return out

    def _experts_table(self, name, l, signs, zero, scratch, type_id, backing=0):
        """A layer's experts table, ai_thinkingmachines__inkling_2's cells (▶ its contracts/objects/experts.cuh): three
        buffers, the integers, the activation's two caps, the backing."""
        e = self.EPLAN
        self.m.table(name, [signs, zero, scratch, l, type_id, e.at_up_lut, e.at_down_data, e.at_down_lut,
                            self.H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d),
                            1 if self.EXPERTS_INT8 else 0, self.GATE_CAP, self.UP_CAP, backing])

    def _card_experts(self, progress):
        """Every layer's experts on the card, and each layer's experts table there."""
        m, e, layers = self.m, self.EPLAN, self.run_layers
        loader = m.loader()
        for i, l in enumerate(layers):
            loader.layer_width(l, self.T_EXPERTS, self.EXPERTS)
            for x in range(self.EXPERTS):
                loader.load_expert(self.b, e, self.T_EXPERTS, l, x, self.EXPERTS)
            if progress:
                progress(i + 1, len(layers), "experts")
        if loader.handed() != 0:
            raise Refused("%d expert slots were reserved and never published" % loader.handed())
        for l in layers:
            self._experts_table("exk%d" % l, l, "signs", "zeros", "experts_scratch", self.T_EXPERTS)

    def _cpu_experts(self, progress):
        """The CPU worker: every layer's experts in its memory, mapped, or a cache of them read from their file; the
        hand-off's buffers and each layer's experts table. Where the card holds the experts, its buffers alone."""
        m, H, e = self.m, self.H, self.EPLAN
        layers = self.run_layers
        backing = {}
        if self.EXPERTS_ON == "card":
            pass
        elif self.EXPERTS_FROM == "disk":
            backing = experts_file.cache_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                                 self._expert_pieces, layers, progress=progress)
        elif self.EXPERTS_FROM == "mapped":
            experts_file.map_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                     self._expert_pieces, layers, progress=progress)
        else:
            m.act_as(CPU)
            loader = m.loader()
            self.SRC = {}                          # every expert's address in the CPU's memory: what a card's tier copies
            for i, l in enumerate(layers):
                loader.layer_width(l, 0, self.EXPERTS)
                self.SRC[l] = [loader.load_expert(self.b, e, 0, l, x, self.EXPERTS) for x in range(self.EXPERTS)]
                if progress:
                    progress(i + 1, len(layers), "experts")
            if loader.handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loader.handed())
        m.act_as(CPU)
        given = dict(cpu_signs=_half(self._signs()[:512]), cpu_zeros=_half(np.zeros(H)))
        self.CPU_AT = {}
        for name, n in self.CPU_BUFS.items():
            self.CPU_AT[name] = m.buffer(n, name, given.get(name))
        for l in layers:
            self._experts_table("exc%d" % l, l, "cpu_signs", "cpu_zeros", "cpu_scratch", 0, backing.get((CPU, l), 0))
        m.array("EXPERTS_CPU", ["exc%d" % l for l in layers])
        if self.CARD_TIER and self.EXPERTS_FROM != "disk":   # from the disk the tier reads the cache's slots (NN-164)
            for l in layers:
                m.array("src%d" % l, self.SRC[l])
        m.act_as(CARD)
        m.array("EXPERTS_CARD", ["exk%d" % l if self.EXPERTS_ON == "card" else 0 for l in layers])

    def _tier(self):
        """The card's tier (▶ nn's `nn__expert_tier`, `card_experts_gb`): each layer's experts table over its slots, a
        position's and a chunk's alike, with the card's experts' scratch; the tier, fed from the CPU's memory — an
        expert's place in RAM, or the disk's cache there — its binding, and the prompt's threshold: `card_line`, or
        measured here, a promotion's copy against the CPU's rows verb over a chunk (▶ `expert_tier.calibrate`)."""
        m, e, layers = self.m, self.EPLAN, self.run_layers
        m.act_as(CARD)
        for l in layers:
            self._experts_table("ext%d" % l, l, "signs", "zeros", "experts_scratch", self.T_CEXP)
        m.array("EXPERTS_TIER", ["ext%d" % l for l in layers])
        disk = self.EXPERTS_FROM == "disk"
        expert_tier.tier(m, "tier_card", self.T_CEXP, self.EXPERTS, self.TOPK, self.H, self.INTER, e, self.LAYERS,
                         None if disk else {l: "src%d" % l for l in layers}, landing=self.CARD_LANDING,
                         slots=dict(worker=CPU, type=0) if disk else None)
        expert_tier.bind(m, {CARD: "tier_card"}, len(self.workers()))
        picks_at = (2 * (self.H + 16) + 7) & ~7     # the package's HAND__PICKS
        if self.ROWS:
            self.CALIBRATION = expert_tier.calibrate(
                m, CARD, CPU, self.AT["experts_scratch"], e.slot_bytes, self.CPU_AT["cpu_hands"], self.HAND_ROW, 2 * self.H,
                picks_at, self.TOPK, self.EXPERTS,
                lambda n: "(ai_thinkingmachines__inkling_2__experts_rows cpu_hands %d exc%d cpu_routed_rows)" % (n, layers[0]),
                self.ROWS, fixed=self.CARD_LINE)
        else:                                      # no prompt as rows: no chunk is noted
            m.must("(sys__add_bindings %d '%s %d)" % (m.env, expert_tier.THRESHOLD, 1), "the promotion threshold",
                   "sys__value_true")
        m.act_as(CARD)

    def _bindings(self):
        H = self.H
        plane = lambda at, n: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_GLOBAL["head"], at, n)
        hp = self.GLOBAL["head"]
        values = dict(
            HIDDEN=H, VOCAB=self.VOCAB, EPSILON=self.EPS, LAYERS=self.LAYERS, LOOP_FROM=self.LOOP_FROM,
            LOOP_TO=self.LOOP_TO, HIDDEN_BYTES=2 * H, HAND_BYTES=2 * (H + 16), RING_STEPS=len(self.RING_POW),
            STOPS=STOPS_MOST, SEEN=sampling.SEEN, EXPERTS_ON_CARD=self.EXPERTS_ON == "card", CARD=CARD, CPU=CPU,
            LOGIT_CAP=self.LOGIT_CAP, LOGIT_CAP_INVERSE=1.0 / self.LOGIT_CAP, EMBEDDING_HALVES=self.EMB_ROW_BYTES // 2,
            EMBEDDING_SCALES=self.EMB_DATA_BYTES // 2, EMBEDDING_BITS=self.EMB_BITS,
            HEAD_CODES=plane(hp.at_up_data, hp.up_data), HEAD_SCALES=plane(hp.at_down_lut, hp.down_lut),
            HEAD_BITS=self.HEAD_BITS, EMBED_ON_CPU=bool(self.EMBED_CPU), STATES_ROWS=STATES_ROWS,
            STATES_HALVES=STATES_ROWS * H, HAND_ROW_BYTES=self.HAND_ROW, CARD_TIER=self.CARD_TIER,
            HAND_PICKS_AT=(2 * (H + 16) + 7) & ~7)
        self.m.bind_values(values)
        if self.EMBED_CPU:
            self.m.node_array("EMB_TOK", 1)       # the token whose row the CPU decodes

    def _procedures(self):
        """The program: `lisp/inkling2.lisp`, and the picture the card hands the CPU (▶ `Machine.define_programs`)."""
        self.m.load_source(self.folder.program("inkling2"))
        self.m.load_source(self.folder.program("inkling2_cpu"))
        self.m.load_source(self.folder.program("sampling"))
        self.m.load_source(self.folder.program("generate"))

    # ── a position ──────────────────────────────────────────────────────────────────────────────────────────────────
    # ── a conversation's state ──────────────────────────────────────────────────────────────────────────────────────
    active_lora = None                             # no adapter: the session's `_activate` loads none

    def load_lora(self, name):
        if name is not None:
            raise Refused("Inkling 2 has no LoRA overlays in this runtime yet")
        return 0.0

    def unload_lora(self):
        pass

    def lora_identity(self):
        return None

    def state_regions(self, length, where=None):
        """(address, bytes) of everything a conversation of `length` positions has written: each application's keys and
        values — a full layer's `length` rows, a sliding one's ring as far as it is filled — and its convolutions'
        inputs. `where`: each buffer's address by name — the load's by default; zeros answer the sizes before the boot."""
        at = self.AT if where is None else where
        KVW, out = self.KVH * self.AD, []
        for a in range(self.APPLICATIONS):
            rows = length if self.layer_of(a) in self.FULL else min(length, self.WINDOW)
            out += [(at["keys_%d" % a], rows * KVW * 2), (at["values_%d" % a], rows * KVW * 2)]
            out += [(at[k % a], self.STATE[k % a] if length else 0)
                    for k in ("key_history_%d", "value_history_%d", "attention_history_%d", "experts_history_%d")]
        return out

    def reset(self):
        """A new conversation: the convolutions' inputs zeroed — a position reads no key or value before its own — and
        every lent slot the tier's again."""
        for name, n in self.STATE.items():
            if "history" in name:
                self.m.zero(self.AT[name], n)
        self._room(0, exact=True)

    def state_write(self, length, blobs):
        """A conversation's state, as `state_regions` lays it out, written back — its rows withheld from the tier first."""
        self._room(length, exact=True)
        for (at, _), blob in zip([r for r in self.state_regions(length) if r[1]], blobs):
            self.m.write(at, blob)

    # ── the full layers' unreached rows lent to the tier (NN-93) ───────────────────────────────────────────────────
    # ⚖ 2026-10-17: *"progressively shrink the l1 cache as the context grows, maybe in chunks of 8k tokens"* — as Qwen's
    # (▶ qwen3_5.py `_lend`). A full layer writes row `pos` and reads `pos + 1` rows, so the rows past a conversation's
    # length are memory nothing reads: the tier holds experts there, slots numbered from each region's end, and gives
    # back LEND_CHUNK rows' worth before a conversation reaches them — `_room` before every write, and when a
    # conversation comes onto the card (`reset`, `state_write`) the slots its length leaves are lent again.
    LEND_CHUNK = 8192

    def _lend(self):
        """The tier lent every full application's keys' and values' tail — the regions' ends on the slots' alignment, each
        `depth` slots deep — then all of them live, the cache holding nothing yet."""
        L, b = self.LEND, self.EPLAN.slot_bytes
        if L is None:
            return
        starts = [self.AT[k % a] for a in self.KV_APPS for k in ("keys_%d", "values_%d")]
        ends = [(s + self.max_context * L["row"]) // 4096 * 4096 for s in starts]
        depth = min((e - (s + 4095) // 4096 * 4096) // b for s, e in zip(starts, ends))
        if depth <= 0:
            self.LEND = None
            return
        m = self.m
        was = m.worker
        m.act_as(CARD)
        m.node_array("lend_ends", len(ends))
        for at in range(0, len(ends), 100):
            m.must("(begin %s (< 0 1))" % " ".join("(sys__node_array__set lend_ends %d %d)" % (i, v)
                                                     for i, v in enumerate(ends[at:at + 100], at)),
                   "the array lend_ends", "sys__value_true")
        m.must("(nn__expert__lend %d %d lend_ends)" % (self.T_CEXP, depth), "the cache's rows lent to the tier")
        m.act_as(was)
        self.LENT_GB = depth * len(ends) * b / 2 ** 30
        L.update(starts=starts, ends=ends, depth=depth, live=0, until=-1)
        self._room(0, exact=True)

    def _room(self, n, exact=False):
        """The cache's first `n` positions free to be written: the lent slots in their rows withheld — the rows rounded up
        to LEND_CHUNK, so the tier gives way a chunk at a time. `exact`: set for a conversation of `n` positions coming
        onto the card, the slots past its rows lent again."""
        L = self.LEND
        if L is None or "starts" not in L or (not exact and n <= L["until"]):
            return
        rows = -(-max(0, n) // self.LEND_CHUNK) * self.LEND_CHUNK
        b, need = self.EPLAN.slot_bytes, rows * L["row"]
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
        L["until"] = min(self.max_context, free // L["row"])

    def lent(self):
        """The cache's rows lent to the tier now: (slots live, GB), or None where the tier is lent none."""
        L = self.LEND
        return None if L is None or "live" not in L else (L["live"], L["live"] * self.EPLAN.slot_bytes / 2 ** 30)

    # ── a position ──────────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, token, pos, greedy=True, sample=None, lag=0, head=None):
        """One position: the token in, the next token out — greedy, or drawn on the card by `sample` (a `sampling.head`)
        — or the logits left for a sampler on the host (answers None). `head` replaces all three: False a prompt's inner
        position, which has none."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._room(pos + 1)
        if lag:
            raise Refused("a position of this model is its own place in the conversation: no lag")
        end = "(type stream)" if head is False else head or sample or ("(head)" if greedy else "(head_logits)")
        got = self._run("(begin (position %d %d) %s)" % (token, pos, end), "position %d" % pos)
        return got if (greedy or sample) and head is None else None

    def generate(self, token, pos, n, stop=(), sampler=None, lag=0):
        """`Generates.generate`, the rows its tokens write withheld from the tier first."""
        self._room(pos + min(n, GENERATE_MOST) + 2)
        return Generates.generate(self, token, pos, n, stop, sampler, lag)

    def _run(self, text, what):
        """A program's answer, run with every block standing: a layer's experts are handed the CPU
        (`sys__scheduler__add`), and a program that does not stand them would find its block never started."""
        return self.m.grid(text)[1]

    def _prompt(self, tokens):
        """A prompt chunk's tokens into PROMPT, which `embed_rows` reads — sixty-four to a program, as a program is one
        form and a form must fit one of the heap's chunks."""
        for at in range(0, len(tokens), 64):
            self.m.must("(begin %s 0)" % " ".join("(sys__node_array__set PROMPT %d %d)" % (i, t)
                                                  for i, t in enumerate(tokens[at:at + 64], at)), "the prompt's tokens")

    def prefill(self, tokens, first, greedy=True, sample=None, **_):
        """Positions `first ..` for `tokens`, read as rows a chunk of ROWS at a time — or a position at a time where ROWS
        is 0; the next token from the last."""
        tokens = [int(t) for t in tokens]
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        self._room(first + len(tokens))
        if not self.ROWS:
            for i, t in enumerate(tokens[:-1]):
                self.step(t, first + i, head=False)
            return self.step(tokens[-1], first + len(tokens) - 1, greedy, sample)
        end = sample or ("(head)" if greedy else "(head_logits)")
        got = None
        for at in range(0, len(tokens), self.ROWS):
            chunk = tokens[at:at + self.ROWS]
            self._prompt(chunk)
            last = at + len(chunk) == len(tokens)
            got = self._run("(begin (rows %d %d) %s)" % (first + at, len(chunk), end if last else "(type stream)"),
                            "positions %d to %d" % (first + at, first + at + len(chunk) - 1))
        return got if greedy or sample else None

    def logits(self):
        return np.frombuffer(self.m.read(self.AT["logits_h"], self.VOCAB * 2), dtype="<f2").astype(np.float32)
