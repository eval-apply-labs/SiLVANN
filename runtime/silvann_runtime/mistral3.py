"""MINISTRAL 3 — Mistral's dense models (the 14B; the 8B and 3B are the same shape), on one card, as nn's own words.

A layer is grouped-query attention and a SwiGLU MLP, each behind an RMS norm with a plain gain, and nothing else: no
gate on the attention, no norm on its queries and keys. So no package of its own — every step is one of nn's words:
```
  rmsnorm · rotate · q k v (v straight into its cache row) · RoPE on q, and on k into its cache row · attention over
  every position so far · rotate · o · residual · rmsnorm · rotate · gate up · swiglu · rotate · down · residual
```
— each matrix packed and read through `nn__turboquant__gemv`, whose input is rotated as the pack rotated its columns.
The PROGRAM is Lisp, written by hand — `lisp/mistral3.lisp`, which the model folder carries and the boot reads — `layers`
a loop over the layers; a token is one short program, `(begin (position tok pos) (head))`, and an answer is `generate`
(▶ `generate`), the same a token at a time on the card until an end token. The embedding is a table in RAM the card
reads a row of: only a row crosses.

⭐ YaRN, WORKED OUT BY THE HOST AT BOOT: the rotary's frequencies are YaRN's (the 14B: factor 16 over an original 16,384,
θ 10⁹), which `nn__rope__angles` does not compute — so the boot writes every position's 64 cosines and 64 sines, 512
bytes a position, into a table on the card the RoPE reads a row of (`CS_K`; the cache is ~160 KB a position). ⭐ And past
the original 16,384 positions Mistral 3 scales its queries — Llama 4's `1 + β log(1 + ⌊pos / 16384⌋)`, β 0.1 — which
the queries' table carries in its angles (`CS_Q`): a rotation and a scale commute, and where the scale is 1 the two
tables are the same bytes.

The cache is every layer's keys and values in halves, `[position][kv head][head dim]`, up to `max_context`.
"""
import math
import os
import pickle

import numpy as np

from . import sampling
from .generate import Generates
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


class Ministral3(Generates):
    PROGRAM = ("mistral3", "sampling", "generate")        # the model folder's lisp/ files (▶ `ModelFolder.program`)
    SAMPLES_ON_CARD = True          # `head_sample` — ▶ `sampling`

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
        # a prompt's rows a call (▶ nn's `nn__attention__rows`, `nn__mlp__rows`): nn's 256, or the context if shorter; the
        # attention's scores hold as many of them as fit 32 MiB — a row's are query heads × the context, fp32 — and the verb
        # attends the chunk that many rows at a time
        self.CHUNK = min(256, self.max_context)
        self.SCORES_ROWS = max(self.QH * self.max_context * 4, 32 << 20)
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
        # ⭐ THE EMBEDDING STAYS IN RAM, a table the card reads a row of in place; the card keeps the head
        self.T_HEAD, self.T_KV = t, t + 1

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
        # the largest class holds a prompt's three large buffers too — the MLP's and the attention's scratch, the scores
        T = self.CHUNK
        self.MLP_SCRATCH = max(2 * (2 * self.H + 4 * self.INTER), 8 * 256 + 2 * T * (3 * self.H + 4 * self.INTER))
        self.ATT_SCRATCH = 16 * 256 + 2 * T * (3 * self.H + 3 * self.QW + self.KVW)
        big = max(8 << 20, self.QH * self.max_context * 4, self.VOCAB * 2, self.MLP_SCRATCH, self.ATT_SCRATCH, self.SCORES_ROWS)
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, n) for t, p, n in self.collections)
        return ("nn__buffer_resid__size_bytes:2097152\nnn__buffer_resid__qty:8\n"
                "nn__buffer_main__size_bytes:4194304\nnn__buffer_main__qty:4\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:64\n"
                "nn__buffer_logits__size_bytes:%d\nnn__buffer_logits__qty:7\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:8\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:0\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                "nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n"
                % (big, (self.arena + (256 << 20)) >> 20, self.LAYERS, "f" * self.LAYERS, len(self.collections), types,
                   self.TABLE_BYTES))

    def workers(self):
        return [(self.m.silicon, 0)]

    # ── ② the load ───────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m, b = self.m, self.b
        loader = m.loader()
        self._head_and_embedding(loader)
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
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())
        self._buffers()
        self._bindings()
        m.define_programs(self._procedures)
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
        m.buffer(64, "one", _half(np.ones(32)))                 # the MLP's down's one weight (▶ nn's mlp.cuh)
        T = self.CHUNK
        # the MLP's scratch, a position's or a chunk's: the rows' pairs, then h · hrot · [gates · ups] · act · actr · ao
        m.buffer(self.MLP_SCRATCH, "mlp_scratch")
        # a chunk's: its rows in and out, the attention's scratch — two lists of pairs, then
        # h · hrot · [q · k] · att · attr · ao (▶ nn's attention.cuh) — its scores, its tokens
        m.buffer(T * H * 2, "xs")
        m.buffer(T * H * 2, "h1s")
        m.buffer(self.ATT_SCRATCH, "att_scratch")
        m.buffer(self.SCORES_ROWS, "scores_rows")
        m.node_array("PROMPT", T)
        self.X_AT = m.buffer(H * 2, "x")
        for nm, n in (("h1", H), ("u", H), ("hu", H), ("h", H), ("hr", H), ("y", H), ("hn", H), ("hnr", H),
                      ("qq", self.QW), ("kk", self.KVW), ("att", self.QW), ("attr", self.QW),
                      ("g", self.INTER), ("up", self.INTER), ("gr", self.INTER)):
            m.buffer(n * 2, nm)
        # every position's cosines then sines, fp32, a row each: the keys', and the queries' with their scale in them
        keys, queries = [], []
        for pos in range(self.max_context):
            ang = pos * self.INV_FREQ
            cs = np.concatenate([np.cos(ang), np.sin(ang)]) * self.ROPE_SCALE
            keys.append(cs.astype("<f4").tobytes())
            queries.append((cs * self.query_scale(pos)).astype("<f4").tobytes())
        m.buffer(self.max_context * self.AH * 4, "CS_K", b"".join(keys))
        m.buffer(self.max_context * self.AH * 4, "CS_Q", b"".join(queries))
        m.buffer(self.QH * self.max_context * 4, "scores")
        self._answer_buffers()                      # what `generate` made, the end tokens it watches for
        self.LOGITS_AT = m.buffer(self.VOCAB * 2, "logits_h")
        m.buffer(64, "res")
        sampling.buffers(m)

    # ── ③ the names the program is written against, and the program ──────────────────────────────────────────
    def plane(self, l, n, which):
        p = self.PLAN[n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d %d %d %d)" % (l, self.TYPE[n], self.WHICH[n], at, size)

    MATRICES = (("Q", "self_attn.q_proj.weight"), ("K", "self_attn.k_proj.weight"), ("V", "self_attn.v_proj.weight"),
                ("O", "self_attn.o_proj.weight"), ("GATE", "mlp.gate_proj.weight"), ("UP", "mlp.up_proj.weight"),
                ("DOWN", "mlp.down_proj.weight"))

    def _bindings(self):
        """What `lisp/mistral3.lisp` reads: the shape, each matrix's width, and a layer's planes gathered into arrays
        indexed by its place in the layers this machine runs."""
        m, H, run, b = self.m, self.H, self.run_layers, self.b
        head = lambda at, size: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_HEAD, at, size)
        values = dict(HIDDEN=H, INTER=self.INTER, EPS=self.EPS, VOCAB=self.VOCAB, RUN=len(run), Q_HEADS=self.QH,
                      KV_HEADS=self.KVH, HEAD_DIM=self.AH, Q_WIDTH=self.QW, KV_WIDTH=self.KVW,
                      EMB_HALVES=self.EMB_ROW_BYTES // 2, EMB_SCALES=self.EMB_DATA_BYTES // 2,
                      EMB_BITS=self.EMB_BITS, CS_HALVES=self.AH * 2, STOPS=0, SEEN=sampling.SEEN,
                      HEAD_CODES=head(self.HPLAN.at_up_data, self.HPLAN.up_data),
                      HEAD_SCALES=head(self.HPLAN.at_down_lut, self.HPLAN.down_lut), HEAD_BITS=self.HEAD_BITS,
                      KV_TYPE=self.T_KV, KEYS_AT=self.KVPLAN.at_up_data, VALUES_AT=self.KVPLAN.at_down_data,
                      KV_ROW_BYTES=self.KVW * 2, CACHE_BYTES=self.max_context * self.KVW * 2)
        for short, n in self.MATRICES:
            bits = {int(b.record(l, n).d) for l in run}
            if len(bits) != 1:
                raise Refused("%s is packed at %s bits across the layers; the program takes one width" % (n, sorted(bits)))
            values[short + "_BITS"] = bits.pop()
        values["MLP_WORDS"] = 16 in (values["GATE_BITS"], values["UP_BITS"], values["DOWN_BITS"])
        m.bind_values(values)
        m.array("LAYER", list(run))
        m.array("INPUT_NORM", [self.plane(l, "input_layernorm.weight", "data") for l in run])
        m.array("POST_NORM", [self.plane(l, "post_attention_layernorm.weight", "data") for l in run])
        for short, n in self.MATRICES:
            m.array(short + "_CODES", [self.plane(l, n, "data") for l in run])
            m.array(short + "_SCALES", [self.plane(l, n, "lut") for l in run])
        # each layer's MLP as nn's table (▶ nn's `contracts/objects/mlp.cuh`)
        ms = [n for short, n in self.MATRICES if short in ("GATE", "UP", "DOWN")]
        for l in run:
            m.table("ml%d" % l, [self.plane(l, "post_attention_layernorm.weight", "data")]
                    + [c for n in ms for c in (self.plane(l, n, "data"), self.plane(l, n, "lut"))]
                    + ["signs", "one", "mlp_scratch", H, self.INTER] + [values[s + "_BITS"] for s in ("GATE", "UP", "DOWN")]
                    + [self.EPS])
        m.array("MLP", ["ml%d" % l for l in run])
        # ⭐ A PROMPT AS ROWS: each layer's attention as nn's table (▶ nn's `contracts/objects/attention.cuh`), where every
        # matrix is TurboQuant's and the widths are whole blocks of 512; otherwise a prompt is read a position at a time
        self.ROWS = (16 not in [values[s + "_BITS"] for s, _ in self.MATRICES] and H % 512 == 0 and self.QW % 512 == 0)
        if self.ROWS:
            cache = lambda l, at: "(nn__expert__plane %d %d 0 %d %d)" % (l, self.T_KV, at, values["CACHE_BYTES"])
            ats = [n for short, n in self.MATRICES if short in ("Q", "K", "V", "O")]
            for l in run:
                m.table("at%d" % l, [self.plane(l, "input_layernorm.weight", "data")]
                        + [c for n in ats for c in (self.plane(l, n, "data"), self.plane(l, n, "lut"))]
                        + [cache(l, self.KVPLAN.at_up_data), cache(l, self.KVPLAN.at_down_data), "CS_Q", "CS_K", "signs",
                           "att_scratch", "scores_rows", H, self.QH, self.KVH, self.AH]
                        + [values[s + "_BITS"] for s in ("Q", "K", "V", "O")] + [self.EPS])
            m.array("ATTENTION", ["at%d" % l for l in run])

    def _procedures(self):
        """The program: `lisp/mistral3.lisp`, the sampler's and `generate`."""
        self.m.load_source(self.folder.program("mistral3"))
        self.m.load_source(self.folder.program("sampling"))
        self.m.load_source(self.folder.program("generate"))

    # ── ④ running ─────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A conversation from nothing: the cache needs nothing — a position reads only rows it has written."""

    def query_scale(self, pos):
        """Llama 4's factor on the queries at `pos`: 1 within the original context, then a step every `span` positions."""
        return 1.0 + self.Q_BETA * math.log(1.0 + math.floor(pos / self.Q_SPAN))

    def step(self, token, pos, greedy=True, sample=None, head=None):
        """One position: the token in, the next token out — greedy, or drawn on the card by `sample` (a `sampling.head`)
        — or the logits left for a sampler on the host (answers None). `head` replaces all three."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        head = head or sample or ("(head)" if greedy else "(head_logits)")
        got = self.m.must("(begin (position %d %d) %s)" % (token, pos, head), "position %d" % pos)
        return got if greedy or sample else None

    def _prompt(self, tokens):
        """A prompt chunk's tokens into PROMPT, which `embed_rows` reads — sixty-four to a program, as a program is one
        form and a form must fit one of the heap's chunks."""
        for at in range(0, len(tokens), 64):
            self.m.must("(begin %s 0)" % " ".join("(sys__node_array__set PROMPT %d %d)" % (i, t)
                                                  for i, t in enumerate(tokens[at:at + 64], at)), "the prompt's tokens")

    def prefill(self, tokens, first, greedy=True, sample=None):
        """Positions `first ..` for `tokens` in chunks of rows — each layer one call a chunk — then the next token from
        the last row (greedy, or drawn by `sample`), or its logits left; a position at a time where the pack keeps a
        layer's matrix in halves, the head only at the last, the others' logits never made."""
        if first + len(tokens) > self.max_context:
            raise Refused("positions to %d are past the context of %d" % (first + len(tokens), self.max_context))
        if not self.ROWS:
            for i, t in enumerate(tokens[:-1]):
                self.step(t, first + i, head="(type x)")
            return self.step(tokens[-1], first + len(tokens) - 1, greedy, sample)
        for at in range(0, len(tokens), self.CHUNK):
            chunk = tokens[at:at + self.CHUNK]
            self._prompt(chunk)
            self.m.must("(begin (embed_rows %d) (rows %d %d) (type xs))" % (len(chunk), first + at, len(chunk)),
                        "the rows from %d" % (first + at))
            last = len(chunk) - 1
        head = sample or ("(head)" if greedy else "(head_logits)")
        got = self.m.must("(begin (nn__vector__add (nn__vector__range xs %d %d) zero_h x %d) %s)"
                          % (last * self.H, self.H, self.H, head), "the prompt's last row")
        return got if greedy or sample else None

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

    def state_regions(self, length, where=None):
        """(address, bytes) of everything a conversation of `length` positions has written: each layer's keys and values.
        `where`: each layer's address — the load's by default; a table of zeros answers the sizes before the boot."""
        p, n, kv = self.KVPLAN, length * self.KVW * 2, self.kv_at if where is None else where
        return [r for l in self.run_layers for r in ((kv[l] + p.at_up_data, n), (kv[l] + p.at_down_data, n))]
