"""MINISTRAL 3 — Mistral's dense models (the 14B; the 8B and 3B are the same shape), on one card, as nn's own words.

A layer is grouped-query attention and a SwiGLU MLP, each behind an RMS norm with a plain gain, and nothing else: no
gate on the attention, no norm on its queries and keys. So no package of its own — every step is one of nn's words:
```
  rmsnorm · rotate · q k v (v straight into its cache row) · RoPE on q, and on k into its cache row · attention over
  every position so far · rotate · o · residual · rmsnorm · rotate · gate up · swiglu · rotate · down · residual
```
— each matrix packed and read through `nn__turboquant__gemv`, whose input is rotated as the pack rotated its columns.
A token is one short program, `(begin (embed 0 0) (layers at len) (head))`, its embedding row written by the host.

⭐ YaRN, IN THE HOST: the rotary's frequencies are YaRN's (the 14B: factor 16 over an original 16,384, θ 10⁹), which
`nn__rope__angles` does not compute — so the host writes each position's 64 cosines and 64 sines into the angles buffer
the RoPE reads, 512 bytes a token. ⭐ And past the original 16,384 positions Mistral 3 scales its queries — Llama 4's
`1 + β log(1 + ⌊pos / 16384⌋)`, β 0.1 — which the host works out a position and every layer applies (`qs`).

The cache is every layer's keys and values in halves, `[position][kv head][head dim]`, up to `max_context`.
"""
import math
import os
import pickle

import numpy as np

from .model_folder import NL, Refused

ARCHITECTURES = ("ministral3",)
CARD = 0


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


def yarn_inv_freq(head_dim, rope):
    """The rotary's frequencies, as HF's `_compute_yarn_parameters` makes them, and the factor its cos and sin carry."""
    base, factor = float(rope["rope_theta"]), float(rope["factor"])
    orig = int(rope["original_max_position_embeddings"])
    beta_fast, beta_slow = float(rope.get("beta_fast", 32.0)), float(rope.get("beta_slow", 1.0))

    def mscale(scale, m=1.0):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
    ms, ms_all = rope.get("mscale"), rope.get("mscale_all_dim")
    attention = mscale(factor, ms) / mscale(factor, ms_all) if ms and ms_all else mscale(factor)
    if rope.get("attention_factor") is not None:
        attention = float(rope["attention_factor"])

    def correction(n_rot):
        return (head_dim * math.log(orig / (n_rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = correction(beta_fast), correction(beta_slow)
    if rope.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    extrapolation, interpolation = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    ramp = np.clip((np.arange(head_dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    keep = 1.0 - ramp                              # the share of a frequency that is extrapolated
    return interpolation * (1.0 - keep) + extrapolation * keep, attention


class Ministral3:
    def __init__(self, folder, machine, max_context=8192, layers=None):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b = folder, machine, b
        self.H, self.LAYERS, self.INTER = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
        self.QH, self.KVH = c["num_attention_heads"], c["num_key_value_heads"]
        self.AH = c.get("head_dim") or self.H // self.QH
        self.QW, self.KVW = self.QH * self.AH, self.KVH * self.AH
        self.EPS = float(c["rms_norm_eps"])
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if (rope.get("rope_type") or rope.get("type")) == "yarn":
            self.INV_FREQ, self.ROPE_SCALE = yarn_inv_freq(self.AH, rope)
            limit = int(rope["original_max_position_embeddings"])
        else:
            theta = float(rope.get("rope_theta") or c["rope_theta"])
            self.INV_FREQ, self.ROPE_SCALE = 1.0 / theta ** (np.arange(0, self.AH, 2, dtype=np.float64) / self.AH), 1.0
            limit = max_context
        # Llama 4's query scaling past the original context (1 everywhere when the config has none)
        self.Q_BETA, self.Q_SPAN = float(rope.get("llama_4_scaling_beta") or 0.0), limit
        if max_context > c.get("max_position_embeddings", max_context):
            raise Refused("max_context %d is past the model's %d" % (max_context, c["max_position_embeddings"]))
        self.max_context = int(max_context)
        self.MOE = False
        self.run_layers = list(range(self.LAYERS)) if layers is None else sorted(layers)
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        self.head_rec, self.emb_rec = g["lm_head.weight"], g["model.language_model.embed_tokens.weight"]
        self.VOCAB = int(self.head_rec["rows"])
        self._plan()

    # ── ① the collections: a type a shape of matrix, the head, the cache ─────────────────────────────────────────
    def _plan(self):
        b, run = self.b, self.run_layers
        names = sorted({n for l in run for n in b.names(l)})
        first = {n: next(l for l in run if n in b.names(l)) for n in names}
        plans = {n: NL.dense_plan(b, n, first[n]) for n in names}
        # one collection a shape, a slot a tensor of it in each layer — the norms share one, k and v another
        groups = {}
        for n in names:
            groups.setdefault((plans[n].up_data, plans[n].down_lut), []).append(n)
        self.TYPE, self.WHICH, self.PLAN, self.WIDTH = {}, {}, {}, {}
        for t, members in enumerate(groups.values()):
            for i, n in enumerate(members):
                self.TYPE[n], self.WHICH[n], self.PLAN[n] = t, i, plans[n]
            self.WIDTH[t] = len(members)
        t = len(groups)
        # ⭐ THE EMBEDDING STAYS IN RAM — a token needs one row of it, which the host writes each step; the card keeps
        #   the head
        self.T_HEAD, self.T_KV = t, t + 1

        def plan_of(rec):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        self.head_data, self.head_lut, self.HPLAN = plan_of(self.head_rec)
        self.emb_data, self.emb_lut, self.BPLAN = plan_of(self.emb_rec)
        # the cache, a slot a layer: its keys in the up half, its values in the down
        kv = self.max_context * self.KVW * 2
        self.KVPLAN = NL.SlotPlan(kv, 0, 0, kv)
        plan_of_type = {self.TYPE[n]: self.PLAN[n] for n in names}
        slots = {tt: len(run) * self.WIDTH[tt] for tt in self.WIDTH}
        self.collections = ([(tt, plan_of_type[tt], slots[tt]) for tt in sorted(self.WIDTH)]
                            + [(self.T_HEAD, self.HPLAN, 1), (self.T_KV, self.KVPLAN, len(run))])
        self.arena = sum(p.slot_bytes * n for _, p, n in self.collections)

    def device_params(self):
        """The boot's device section: the buffer pools, the arena and every collection's shape."""
        big = max(8 << 20, self.QH * self.max_context * 4, self.VOCAB * 2)
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in self.collections)
        return ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
                "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:4\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:64\n"
                "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:4\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:8\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:0\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                % (big, (self.arena + (256 << 20)) >> 20, self.LAYERS, "f" * self.LAYERS, len(self.collections), types))

    def workers(self):
        return [(self.m.silicon, 0)]

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        loader = m.loader()
        self.kv_at = {}
        for i, l in enumerate(self.run_layers):
            for t in sorted({self.TYPE[n] for n in b.names(l)}):
                loader.layer_width(l, t, self.WIDTH[t])
            for n in b.names(l):
                loader.load_dense(b, self.PLAN[n], self.TYPE[n], n, l, which=self.WHICH[n])
            loader.layer_width(l, self.T_KV, 1)
            self.kv_at[l] = loader.reserve(self.T_KV)
            loader.assign(l, self.T_KV, 0, self.kv_at[l])
            if progress:
                progress(i + 1, len(self.run_layers))
        loader.layer_width(0, self.T_HEAD, 1)
        loader.load_bytes(self.HPLAN, self.T_HEAD, 0, self.head_data, self.head_lut)
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())
        self._buffers()
        m.define_programs(self._procedures)

    def _buffers(self):
        m, H = self.m, self.H
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        final_w = np.frombuffer(np.asarray(self.global_["model.language_model.norm.weight"]["vbr_data"]).tobytes(),
                                dtype="<f2").astype(np.float64)[:H]
        m.buffer(1024, "signs", _half(signs[:512]))
        m.buffer(1024, "ones512", _half(np.ones(512)))
        m.buffer(H * 2, "signs_row", _half(np.tile(signs[:512], H // 512)))
        m.buffer(H * 2, "fnw", _half(final_w))                  # Mistral's norm is a plain gain
        m.buffer(H * 2, "zero_h", _half(np.zeros(H)))
        self.X_AT = m.buffer(H * 2, "x")
        for nm, n in (("u", H), ("hu", H), ("h", H), ("hr", H), ("y", H), ("hn", H), ("hnr", H),
                      ("qq", self.QW), ("kk", self.KVW), ("att", self.QW), ("attr", self.QW),
                      ("g", self.INTER), ("up", self.INTER), ("gr", self.INTER)):
            m.buffer(n * 2, nm)
        self.CS_AT = m.buffer(self.AH * 4, "cs")                # a position's cosines then sines, fp32, the host's
        m.buffer(self.QH * self.max_context * 4, "scores")
        row = int(self.emb_rec["row_bytes"])
        self.EMB_AT = m.buffer(row, "emb_codes")
        self.EMB_SCALE_AT = m.buffer(64, "emb_scales")
        self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        m.buffer(64, "res")

    # ── ③ the procedures ─────────────────────────────────────────────────────────────────────────────────────
    def plane(self, l, n, which):
        p = self.PLAN[n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d %d %d %d)" % (l, self.TYPE[n], self.WHICH[n], at, size)

    def gemv(self, l, n, x, out):
        """`out = W x` for a layer's matrix — its input rotated unless the matrix is lossless."""
        r = self.b.record(l, n)
        if r.d == 16:
            return "(nn__expert__multiply_fp16 %s %s %s %d %d)" % (self.plane(l, n, "data"), x, out, r.rows, r.cols)
        return ("(nn__turboquant__gemv %s %s %s %s %d %d %d)"
                % (self.plane(l, n, "data"), self.plane(l, n, "lut"), x, out, r.rows, r.cols, r.d))

    def cache(self, l, half, at=None):
        """Layer `l`'s keys (`up`) or values (`down`): all of them, or the row at `at` (an expression, in bytes)."""
        p = self.KVPLAN
        base = p.at_up_data if half == "up" else p.at_down_data
        if at is None:
            return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, base, self.max_context * self.KVW * 2)
        return "(nn__expert__plane %d %d 0 (sys__add %d %s) %d)" % (l, self.T_KV, base, at, self.KVW * 2)

    def _layer(self, l):
        H, s = self.H, "self_attn."
        norm = lambda n: self.plane(l, n, "data")
        return ("(nn__rmsnorm__apply x %s h %d %r) (nn__hadamard__rotate h signs hr %d) " % (norm("input_layernorm.weight"), H, self.EPS, H)
                + self.gemv(l, s + "q_proj.weight", "hr", "qq") + " (nn__vector__scale qq qs qq %d) " % self.QW
                + self.gemv(l, s + "k_proj.weight", "hr", "kk") + " "
                + self.gemv(l, s + "v_proj.weight", "hr", self.cache(l, "down", "at")) + " "
                + "(nn__rope__apply qq cs qq %d %d) (nn__rope__apply kk cs %s %d %d) "
                % (self.QH, self.AH, self.cache(l, "up", "at"), self.KVH, self.AH)
                + "(nn__attention__decode qq %s %s scores att %d %d %d 0 len) "
                % (self.cache(l, "up"), self.cache(l, "down"), self.QH, self.KVH, self.AH)
                + "(nn__hadamard__rotate att signs attr %d) " % self.QW
                + self.gemv(l, s + "o_proj.weight", "attr", "y") + " (nn__vector__add x y x %d) " % H
                + "(nn__rmsnorm__apply x %s h %d %r) (nn__hadamard__rotate h signs hr %d) "
                % (norm("post_attention_layernorm.weight"), H, self.EPS, H)
                + self.gemv(l, "mlp.gate_proj.weight", "hr", "g") + " " + self.gemv(l, "mlp.up_proj.weight", "hr", "up")
                + " (nn__swiglu__combine g up g %d) (nn__hadamard__rotate g signs gr %d) " % (self.INTER, self.INTER)
                + self.gemv(l, "mlp.down_proj.weight", "gr", "y") + " (nn__vector__add x y x %d)" % H)

    def _procedures(self):
        m, H = self.m, self.H
        row = int(self.emb_rec["row_bytes"])
        # the token's row, un-rotated (Rᵀu = S ⊙ H·u), into the residual
        m.defun("(defun (embed codes scale) (begin (nn__turboquant__decode (nn__vector__range emb_codes codes %d) "
                "(nn__vector__range emb_scales scale 1) u 1 %d %d) (nn__hadamard__rotate u ones512 hu %d) "
                "(nn__vector__pointwise_mul hu signs_row x %d) (type x)))" % (row // 2, H, int(self.emb_rec["d"]), H, H))
        logits = ("(nn__rmsnorm__apply x fnw hn %d %r) (nn__hadamard__rotate hn signs hnr %d) " % (H, self.EPS, H)
                  + "(nn__turboquant__gemv (nn__expert__plane 0 %d 0 %d %d) (nn__expert__plane 0 %d 0 %d %d) hnr logits_h %d %d %d)"
                  % (self.T_HEAD, self.HPLAN.at_up_data, self.HPLAN.up_data, self.T_HEAD, self.HPLAN.at_down_lut,
                     self.HPLAN.down_lut, self.VOCAB, H, int(self.head_rec["d"])))
        m.defun("(defun (head) (begin %s (nn__argmax__find logits_h res %d) (nn__buffer__read res)))" % (logits, self.VOCAB))
        m.defun("(defun (head_logits) (begin %s (type x)))" % logits)
        for l in self.run_layers:
            m.defun("(defun (layer%d at len qs) (begin %s (type x)))" % (l, self._layer(l)))
        m.defun("(defun (layers at len qs) (begin %s (type x)))" % " ".join("(layer%d at len qs)" % l for l in self.run_layers))

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: the cache needs nothing — a position reads only rows it has written."""

    def _angles(self, pos):
        ang = pos * self.INV_FREQ
        cs = np.concatenate([np.cos(ang), np.sin(ang)]) * self.ROPE_SCALE
        self.m.write(self.CS_AT, cs.astype("<f4").tobytes())

    def query_scale(self, pos):
        """Llama 4's factor on the queries at `pos`: 1 within the original context, then a step every `span` positions."""
        return 1.0 + self.Q_BETA * math.log(1.0 + math.floor(pos / self.Q_SPAN))

    def _stage(self, token):
        row = int(self.emb_rec["row_bytes"])
        self.m.write(self.EMB_AT, self.emb_data[token * row:(token + 1) * row])
        self.m.write(self.EMB_SCALE_AT, self.emb_lut[token * 2:token * 2 + 2])

    def step(self, token, pos, greedy=True):
        """One position: the token in, the next token out (greedy), or the logits left for a sampler (answers None)."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._stage(token)
        self._angles(pos)
        got = self.m.must("(begin (embed 0 0) (layers %d %d %r) %s)" % (pos * self.KVW * 2, pos + 1, self.query_scale(pos),
                                                                        "(head)" if greedy else "(head_logits)"),
                          "position %d" % pos)
        return got if greedy else None

    def prefill(self, tokens, first, greedy=True):
        """Positions `first ..` for `tokens`, a position at a time; then the next token (greedy) or the logits left."""
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        out = None
        for i, t in enumerate(tokens):
            out = self.step(t, first + i, greedy)
        return out

    def logits(self):
        return np.frombuffer(self.m.read(self.LOGITS_AT, self.VOCAB * 2), dtype="<f2").astype(np.float32)

    # ── LoRAs: none yet for this family ─────────────────────────────────────────────────────────────────────
    active_lora = None

    def load_lora(self, name):
        if name is not None:
            raise Refused("Ministral 3 has no LoRA overlays in this runtime yet")
        return 0.0

    def unload_lora(self):
        pass

    def lora_identity(self):
        return None

    def state_regions(self, length):
        """(address, bytes) of everything a conversation of `length` positions has written: each layer's keys and values."""
        p, n = self.KVPLAN, length * self.KVW * 2
        return [r for l in self.run_layers for r in ((self.kv_at[l] + p.at_up_data, n), (self.kv_at[l] + p.at_down_data, n))]
