"""QWEN-IMAGE 2.1 — a picture from a prompt and reference pictures (NN-89), its diffusion transformer on one card.

The pack (`silvann_tq_D8_qwen_image_2.1`) holds diffusers' folders, the transformer and the text encoder packed: a block of
the transformer is `ai_qwen_image`'s verb over a plane table this class fills (▶ `packages/ai_qwen_image/contracts/objects/
block.cuh`), and the sequence it runs over is the prompt's rows — the encoder's, then each reference picture's latent
rows — and the generated picture's. The program reads the blocks one after another; the first step runs every row and keeps
each block's prompt keys and values, a later step the picture's rows alone (diffusers' `causal_condition`: the prompt is
modulated from t = 0 and causal, so it is the same at every step).

What this class makes for the verbs:
  - each block's planes in a slot of their own: q, k, v, the output, the MLP's three — the pack's codes and scales as they
    are, but q's and k's rows reordered a head, and their norms' weights likewise, so that nn's rotary (element k with
    k + 64) turns what the transformer turned as neighbours (2k with 2k + 1);
  - the prompt's keys and values a block, kept from the first step;
  - the transformer's own planes, for what a step does around the blocks (▶ `ai_qwen_image`'s `step.cuh`): the timestep's
    two matrices and the modulation's, the prompt's projection, the latents' (64 columns, below the ladder's 128, so
    written decoded, at 16 bits), the last norm's and the output's;
  - the rows' planes: the sequence `x`, the modulation (two rows — the prompt's, t = 0, then the picture's — of 1 + scale
    and tanh(gate) for each half of a block, which `ai_qwen_image__time` writes), the last norm's scale, the rotary table
    (a row of the head's width a position, its cosines then its sines) and each row's key limit (`layout`), the
    timesteps' sinusoids (`set_timesteps`), the encoder's rows, the latents and the prediction.
"""
import json
import os
import pickle

import numpy as np

from .model_folder import NL, Refused

# the block's matrices, by the pack's names, in the table's order
MATRICES = (("attn.to_q.weight", "q"), ("attn.to_k.weight", "k"), ("attn.to_v.weight", "v"), ("attn.to_out.0.weight", "out"),
            ("img_mlp.gate_layer.weight", "gate"), ("img_mlp.proj.weight", "up"), ("img_mlp.out.weight", "down"))
NORMS = (("attn.norm_q.weight", "norm_q"), ("attn.norm_k.weight", "norm_k"))
# the transformer's own matrices, by the pack's names, in the step table's order
GLOBALS = (("img_in.weight", "img"), ("txt_in.in_layer.weight", "txt_in"), ("txt_in.out_layer.weight", "txt_out"),
           ("time_text_embed.timestep_embedder.linear_1.weight", "t1"),
           ("time_text_embed.timestep_embedder.linear_2.weight", "t2"), ("modulation.1.weight", "mod"),
           ("norm_out.linear.weight", "norm_out"), ("proj_out.weight", "proj"))
THETA = 10000           # the rotary's base (diffusers' `QwenImage21Rope(theta=10000)`)


SYSTEM = "Comprehend and analyze the provided prompt."     # the pipeline's system message (QwenImage21Pipeline.sys_prompt)
PIPELINES = ("QwenImage21Pipeline",)                      # the diffusers pipelines a pack of this module's is made from


def image_pack(path):
    """Whether a folder is an image pack this module runs: diffusers' `model_index.json` naming one of `PIPELINES`, and
    the transformer packed."""
    try:
        index = json.load(open(os.path.join(path, "model_index.json")))
    except (OSError, ValueError):
        return False
    return index.get("_class_name") in PIPELINES and os.path.isfile(os.path.join(path, "transformer", "pack.json"))


class Stopped(Exception):
    """A picture stopped before it was done (its `progress` answered False)."""


class Encoder:
    """Qwen-Image 2.1's prompt encoder — the pack's Qwen3-VL text model (`text_encoder/`), booted as Qwen 3's runtime
    reads it (▶ qwen3_5.py: no output gate, every head's rotary whole, no head) — over the pipeline's template: the
    system message, the user's prompt, the assistant's turn begun. Its answer is the rows after the last layer, before
    the final norm, less the system message's (diffusers' `_get_qwen_prompt_embeds`)."""

    def __init__(self, path, max_context=2048, chunk=256, silicon=None):
        from tokenizers import Tokenizer
        from . import open_model
        self.model = open_model(os.path.join(path, "text_encoder"), max_context=max_context, silicon=silicon, kv="fp16",
                                chunk=chunk, picture_pixels=0)
        self.tok = Tokenizer.from_file(os.path.join(path, "processor", "tokenizer.json"))
        self.system = "<|im_start|>system\n%s<|im_end|>\n" % SYSTEM
        self.DROP = len(self.tok.encode(self.system, add_special_tokens=False).ids)

    def tokens(self, prompt):
        """The template's tokens for a prompt without pictures (an empty prompt is a space: Qwen has no first token)."""
        text = self.system + "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (prompt or " ")
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, prompt):
        """The prompt's rows (float32, HIDDEN wide) and which are pictures' slots (none, here)."""
        ids, m = self.tokens(prompt), self.model
        if len(ids) > m.max_context:
            raise Refused("the prompt is %d tokens, and this encoder holds %d" % (len(ids), m.max_context))
        rows = [np.frombuffer(m.forward_rows(at, tokens=ids[at:at + m.CHUNK]), "<f2").reshape(-1, m.H)
                for at in range(0, len(ids), m.CHUNK)]
        rows = np.concatenate(rows)[self.DROP:].astype(np.float32)
        return rows, np.zeros(len(rows), bool)


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


def _plane(align, size):
    return (size + align - 1) // align * align


class QwenImage:
    PROGRAM = ("qwen_image", "qwen_image_vae")      # the pack's lisp/ files: the transformer's, the picture decoder's

    def __init__(self, path, machine, blocks=None, max_rows=4096, prefix_max=1024, chunk=1024, max_steps=100, program=None):
        self.path, self.m = path, machine
        # the program's file: the pack's own (lisp/qwen_image.lisp), or the one given — a development box's
        self.PROGRAM_FILE = program or os.path.join(path, "lisp", self.PROGRAM[0] + ".lisp")
        folder = os.path.join(path, "transformer")
        self.config = c = json.load(open(os.path.join(folder, "config.json")))
        if c.get("_class_name") != "QwenImage21Transformer2DModel":
            raise Refused("%s is a %r, not Qwen-Image 2.1's transformer" % (folder, c.get("_class_name")))
        self.b = NL.Bundle(folder)
        self.HEADS, self.HD = c["num_attention_heads"], c["attention_head_dim"]
        self.H, self.INTER = self.HEADS * self.HD, self.HEADS * self.HD * c["mlp_ratio"]
        self.AXES = list(c["axes_dims_rope"])
        if sum(self.AXES) != self.HD or float(c["eps"]) != 1e-6 or not c.get("causal_condition"):
            raise Refused("this transformer's rotary axes, epsilon or conditioning are not the ones ai_qwen_image is built for")
        self.BLOCKS = c["num_layers"]
        self.run_blocks = list(range(self.BLOCKS)) if blocks is None else sorted(blocks)
        self.MAX_ROWS, self.PREFIX_MAX, self.CHUNK, self.MAX_STEPS = int(max_rows), int(prefix_max), int(chunk), int(max_steps)
        if self.PREFIX_MAX >= self.MAX_ROWS:
            raise Refused("the prompt's most rows (%d) leave no room for a picture in %d" % (self.PREFIX_MAX, self.MAX_ROWS))
        self.global_ = pickle.load(open(os.path.join(folder, "global.pkl"), "rb"))
        g = self.global_
        self.CONTEXT, self.CHANNELS = c["context_in_dim"], c["in_channels"]
        self.FREQ = int(g["time_text_embed.timestep_embedder.linear_1.weight"]["original_shape"][1])
        self._plan()

    # ── ① the collections: a block's planes a slot, the prompt's keys and values a slot, the rows' planes and scratch ──
    def _record(self, block, short):
        full = "transformer_blocks.%d.%s" % (block, short)
        idx = self.b.index(block)
        if full not in idx:
            raise Refused("block %d has no record %r" % (block, short))
        return NL.Record(self.b, block, full, idx[full])

    def _plan(self):
        H, I, hd = self.H, self.INTER, self.HD
        first = self.run_blocks[0]
        self.D, self.BREC, at = {}, {}, 0          # each matrix's width; each plane's (offset, bytes) in a block's slot
        for short, key in MATRICES:
            r = self._record(first, short)
            self.D[key] = r.d
            data = r.rows * r.row_bytes
            self.BREC[key] = (at, data)
            at += _plane(256, data)
            self.BREC[key + "_lut"] = (at, 2 * r.rows if r.d != 16 else 256)
            at += _plane(256, self.BREC[key + "_lut"][1])
        for _, key in NORMS:
            self.BREC[key] = (at, 2 * hd)
            at += _plane(256, 2 * hd)
        self.BPLAN = NL.SlotPlan(at, 0, 0, 256)      # every plane in the up half (nn takes no empty half)
        kv = self.PREFIX_MAX * H * 2
        self.KPLAN = NL.SlotPlan(kv, 0, 0, kv)       # the prompt's keys in the up half, its values in the down
        n, step, C, CH, W = self.MAX_ROWS, min(self.CHUNK, self.MAX_ROWS), self.CONTEXT, self.CHANNELS, max(H, self.CONTEXT)
        # ▶ the block's scratch: the pairs, then h, hrot, q, keys, values, a, m1, m2; the step verbs' (the pairs, then h,
        #   hrot, m: CHUNK rows each of the wider of HIDDEN and CONTEXT) is the same plane, and smaller
        rot = max(n * H, step * I)
        scratch = max(((8 * n + 15) & ~15) + 2 * (n * H + rot + n * H + 2 * n * H + n * H + 2 * step * I),
                      ((8 * step + 15) & ~15) + 6 * step * W)
        self.SCR, at = {}, 0
        for name, nbytes in (("x", 2 * n * H), ("modulation", 16 * H), ("scale", 2 * H), ("rotary", 4 * n * hd),
                             ("limits", 4 * n), ("sinusoids", 2 * (self.MAX_STEPS + 1) * self.FREQ),
                             ("encoder", 2 * self.PREFIX_MAX * C), ("conditions", 2 * self.PREFIX_MAX * CH),
                             ("latents", 2 * n * CH), ("prediction", 2 * n * CH),
                             ("zero", 2 * H), ("signs", 1024), ("scratch", scratch)):
            self.SCR[name] = (at, nbytes)
            at += _plane(256, nbytes)
        self.SPLAN = NL.SlotPlan(at, 0, 0, 256)
        # the transformer's own planes: each matrix's codes and scales — the latents' decoded, its rows 16 bits — then the
        # text norm's 1 + w, a row of ones, the modulation's ones where it is a scale
        self.GREC, self.DG, at = {}, {}, 0
        for name, key in GLOBALS:
            rec = self.global_[name]
            rows, cols = (int(v) for v in rec["original_shape"])
            self.DG[key] = 16 if key == "img" else int(rec["d"])
            data = rows * cols * 2 if self.DG[key] == 16 else len(np.asarray(rec["vbr_data"]).tobytes())
            self.GREC[key] = (at, data)
            at += _plane(256, data)
            self.GREC[key + "_lut"] = (at, 2 * rows if self.DG[key] != 16 else 256)
            at += _plane(256, self.GREC[key + "_lut"][1])
        for key, nbytes in (("txt_norm", 2 * C), ("ones", 2 * H), ("mod_one", 8 * H)):
            self.GREC[key] = (at, nbytes)
            at += _plane(256, nbytes)
        self.GPLAN = NL.SlotPlan(at, 0, 0, 256)
        self.T_BLOCK, self.T_PREFIX, self.T_SCR, self.T_GLOBAL = 0, 1, 2, 3
        nb = len(self.run_blocks)
        self.collections = [(self.T_BLOCK, self.BPLAN, nb), (self.T_PREFIX, self.KPLAN, nb), (self.T_SCR, self.SPLAN, 1),
                            (self.T_GLOBAL, self.GPLAN, 1)]
        self.arena = sum(p.slot_bytes * k for _, p, k in self.collections)

    def device_params(self):
        """The boot's device section: small buffer pools — every row the verb reads is in a slot — and the collections, a
        layer a block (the index's layers are the blocks this machine holds)."""
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, k) for t, p, k in self.collections)
        return ("nn__buffer_resid__size_bytes:1048576\nnn__buffer_resid__qty:2\n"
                "nn__buffer_main__size_bytes:1048576\nnn__buffer_main__qty:2\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:2\n"
                "nn__buffer_logits__size_bytes:1048576\nnn__buffer_logits__qty:1\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:1\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:0\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:%s\nnn__expert__types:%d\n%s"
                % ((self.arena + (256 << 20)) >> 20, len(self.run_blocks), "f" * len(self.run_blocks), len(self.collections),
                   types))

    # ── ② the load ─────────────────────────────────────────────────────────────────────────────────────────────────
    def _order(self):
        """The rows of q and k as nn's rotary wants them, a head at a time: the transformer's element 2k at k, 2k + 1 at
        k + HEAD_DIM / 2."""
        hd = self.HD
        within = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
        return within, (np.arange(self.HEADS)[:, None] * hd + within[None, :]).reshape(-1)

    def load(self):
        """Every block's planes into its slot, q's and k's rows reordered; the prompt's keys' and values' slots and the
        rows' slot reserved; the transformer's own planes; the zero row and the signs written."""
        from nn_unpack_model import decode_record
        m, loader = self.m, self.m.loader()
        within, rows = self._order()
        for i, block in enumerate(self.run_blocks):
            loader.layer_width(i, self.T_BLOCK, 1)
            at = loader.reserve(self.T_BLOCK)
            for short, key in MATRICES:
                r = self._record(block, short)
                if r.d != self.D[key]:
                    raise Refused("block %d's %s is %d bits, block %d's %d" % (block, short, r.d, self.run_blocks[0], self.D[key]))
                data = r.data_bytes()
                a, nbytes = r.lut_span()
                lut = bytes(self.b.blob(block)[a:a + nbytes])
                if key in ("q", "k"):                       # reordered a head; every other matrix as the file has it
                    data = np.frombuffer(data, np.uint8).reshape(r.rows, r.row_bytes)[rows].tobytes()
                    if r.d != 16:
                        lut = np.frombuffer(lut, "<f2")[rows].tobytes()
                loader.write(at + self.BREC[key][0], data)
                loader.write(at + self.BREC[key + "_lut"][0], lut)
            for short, key in NORMS:
                rec = self.b.index(block)["transformer_blocks.%d.%s" % (block, short)]
                w = decode_record(rec, self.b.blob(block)).reshape(-1)[within]
                loader.write(at + self.BREC[key][0], _half(w))
            loader.assign(i, self.T_BLOCK, 0, at)
            loader.layer_width(i, self.T_PREFIX, 1)
            loader.assign(i, self.T_PREFIX, 0, loader.reserve(self.T_PREFIX))
        loader.layer_width(0, self.T_SCR, 1)
        self.SCR_AT = loader.reserve(self.T_SCR)
        loader.assign(0, self.T_SCR, 0, self.SCR_AT)
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little")[:512].astype(np.float32)
        m.write(self.SCR_AT + self.SCR["zero"][0], bytes(2 * self.H))
        m.write(self.SCR_AT + self.SCR["signs"][0], _half(signs))
        self._load_global(loader)
        self._tables()
        self._bindings()

    def _load_global(self, loader):
        """The transformer's own planes into their slot: each matrix as the pack keeps it — the latents', decoded to 16 bits
        — the text norm's 1 + w (diffusers' `QwenImage21ZeroCenterRMSNorm` keeps w - 1), the ones."""
        from .vision import decoded
        loader.layer_width(0, self.T_GLOBAL, 1)
        at = loader.reserve(self.T_GLOBAL)
        for name, key in GLOBALS:
            rec = self.global_[name]
            if self.DG[key] == 16:
                loader.write(at + self.GREC[key][0], _half(decoded(rec).reshape(rec["original_shape"])))
                continue
            loader.write(at + self.GREC[key][0], np.asarray(rec["vbr_data"]).tobytes())
            loader.write(at + self.GREC[key + "_lut"][0], np.asarray(rec["vbr_lut"]).astype("<f2").tobytes())
        H = self.H
        loader.write(at + self.GREC["txt_norm"][0], _half(decoded(self.global_["txt_in.text_norm.weight"]).reshape(-1) + 1.0))
        loader.write(at + self.GREC["ones"][0], _half(np.ones(H)))
        loader.write(at + self.GREC["mod_one"][0], _half(np.tile(np.concatenate([np.ones(H), np.zeros(H)]), 2)))
        loader.assign(0, self.T_GLOBAL, 0, at)

    def _tables(self):
        """A block's table each, `QB<i>` for the i-th block this machine holds."""
        m, hd = self.m, self.HD
        sp = lambda name: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_SCR,) + self.SCR[name])
        for i in range(len(self.run_blocks)):
            w = lambda key: "(nn__expert__plane %d %d 0 %d %d)" % ((i, self.T_BLOCK) + self.BREC[key])
            kv = self.PREFIX_MAX * self.H * 2
            m.table("QB%d" % i, [w("q"), w("q_lut"), w("k"), w("k_lut"), w("v"), w("v_lut"), w("out"), w("out_lut"),
                                 w("gate"), w("gate_lut"), w("up"), w("up_lut"), w("down"), w("down_lut"),
                                 w("norm_q"), w("norm_k"),
                                 "(nn__expert__plane %d %d 0 %d %d)" % (i, self.T_PREFIX, self.KPLAN.at_up_data, kv),
                                 "(nn__expert__plane %d %d 0 %d %d)" % (i, self.T_PREFIX, self.KPLAN.at_down_lut, kv),
                                 sp("zero"), sp("signs"), sp("scratch"),
                                 self.H, self.HEADS, hd, self.INTER, self.D["q"], self.D["out"], self.D["gate"],
                                 self.D["down"], self.PREFIX_MAX, self.CHUNK])
        g = lambda key: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_GLOBAL,) + self.GREC[key])
        m.table("QG", [g("img"), g("img_lut"), g("txt_in"), g("txt_in_lut"), g("txt_out"), g("txt_out_lut"), g("t1"),
                       g("t1_lut"), g("t2"), g("t2_lut"), g("mod"), g("mod_lut"), g("norm_out"), g("norm_out_lut"),
                       g("proj"), g("proj_lut"), g("txt_norm"), g("ones"), g("mod_one"), sp("zero"), sp("signs"),
                       sp("scratch"), self.H, self.CONTEXT, self.CHANNELS, self.FREQ, min(self.CHUNK, self.MAX_ROWS)]
                + [self.DG[key] for _, key in GLOBALS])
        self.PLANES = {k: sp(k) for k in ("x", "modulation", "scale", "rotary", "limits", "sinusoids", "encoder", "conditions",
                                          "latents", "prediction")}

    def _bindings(self):
        """What `lisp/qwen_image.lisp` reads of the boot — the transformer's tables and the rows' planes — then the
        program itself."""
        m = self.m
        if not os.path.isfile(self.PROGRAM_FILE):
            raise Refused("%s has no %s — the model's program" % (self.path, os.path.relpath(self.PROGRAM_FILE, self.path)))
        m.bind_values(dict({"BLOCKS": len(self.run_blocks)}, **self.PLANES))
        m.array("BLOCK", ["QB%d" % i for i in range(len(self.run_blocks))])
        m.load_source(self.PROGRAM_FILE)

    # ── ③ what a picture's steps read ───────────────────────────────────────────────────────────────────────────────
    def schedule(self, steps, rows):
        """The sigmas of `steps` steps for a picture of `rows` latent rows, the last 0 — the pipeline's: `linspace(1,
        1/steps)`, shifted by `mu` (the scheduler's dynamic shift for the picture's rows, exponential) and stretched to
        end at its `shift_terminal` (diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps`), in float32 as it is."""
        sc = json.load(open(os.path.join(self.path, "scheduler", "scheduler_config.json")))
        if (sc.get("time_shift_type") != "exponential" or not sc.get("use_dynamic_shifting") or sc.get("invert_sigmas")
                or any(sc.get(k) for k in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas"))):
            raise Refused("the scheduler's settings are not the ones this schedule is written for")
        base, top = sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096)
        lo, hi = sc.get("base_shift", 0.5), sc.get("max_shift", 1.15)
        slope = (hi - lo) / (top - base)
        mu = rows * slope + (lo - slope * base)
        sig = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
        sig = (np.exp(mu) / (np.exp(mu) + (1.0 / sig - 1.0))).astype(np.float32)
        if sc.get("shift_terminal"):
            one_minus = np.float32(1.0) - sig
            sig = (np.float32(1.0) - one_minus / (one_minus[-1] / np.float32(1.0 - sc["shift_terminal"]))).astype(np.float32)
        return np.concatenate([sig, np.zeros(1, np.float32)])

    def bind_picture(self, encoder, slots, shapes, latents, conditions=None, sigmas=None):
        """A picture's inputs written and its names bound for `(denoise steps)`: the encoder's rows and which are
        pictures' slots, each picture's shape (the references', then the generated one's), the starting latents, the
        references' latents in order, and the sigmas (▶ `schedule`, the last 0). Answers the sequence's rows and the
        prompt's."""
        m = self.m
        pad, runs = self.sequence(slots, shapes)
        prefix = self.set_layout(pad, shapes)
        sigmas = np.asarray(sigmas, np.float32)
        self.set_timesteps(np.concatenate([[0.0], sigmas[:-1]]))
        self.write("encoder", np.asarray(encoder)[~np.asarray(slots, bool)])
        if conditions is not None:
            self.write("conditions", conditions)
        self.write("latents", latents)
        self.TARGET = len(pad) - prefix
        m.bind_values({"TOTAL": len(pad), "PREFIX": prefix, "TARGET": self.TARGET, "RUNS": len(runs)})
        m.array("RUN_TEXT", [kind == "text" for kind, _, _, _ in runs])
        m.array("RUN_FROM", [frm for _, frm, _, _ in runs])
        m.array("RUN_FIRST", [first for _, _, first, _ in runs])
        m.array("RUN_ROWS", [n for _, _, _, n in runs])
        m.array("DSIGMA", [repr(float(sigmas[i + 1]) - float(sigmas[i])) for i in range(len(sigmas) - 1)])
        return len(pad), prefix

    def steps(self, steps, progress):
        """`steps` steps of the bound picture as the program's `denoise` runs them, a call each so that `progress(step,
        steps)` is told after each — and a False from it stops them (`Stopped`)."""
        self.m.must("(type (ai_qwen_image__time sinusoids 0 modulation 0 scale QG))", "the prompt's modulation",
                    "sys__value_int")
        for s in range(steps):
            self.m.must("(type (step %d))" % s, "step %d" % s)
            if progress(s + 1, steps) is False:
                raise Stopped("stopped after step %d of %d" % (s + 1, steps))
        return self.read("latents", self.TARGET, self.CHANNELS)

    def denoise(self, steps):
        """`steps` steps of the bound picture (▶ `bind_picture`) on the card; its latents after them, float32."""
        self.m.must("(denoise %d)" % steps, "the picture's steps")           # the program answers its latents' kind
        return self.read("latents", self.TARGET, self.CHANNELS)

    def layout(self, pad, shapes):
        """The sequence's rotary table and key limits, and the prompt's rows — diffusers' `QwenImage21Rope.forward` and
        `build_token_metadata`. `pad` says of each row of the sequence whether it is a picture's; `shapes` each picture's
        (frames, height, width) in latent rows, in order, the generated one last and the sequence's last rows. A text row
        moves the position on all three axes; a picture holds the frame axis at the text's position, lays its rows out on
        a height-width grid centred on zero, and moves the position on by its larger side. A text row reads the keys up
        to itself; a reference picture's row up to its picture's end; the generated picture's every row."""
        pad = np.asarray(pad, bool)
        total = len(pad)
        frame, hh, ww, ends = [], [], [], []
        cursor = position = 0
        for _, h, w in shapes:
            start = int(np.flatnonzero(pad[cursor:])[0]) + cursor
            frame += range(position, position + start - cursor)
            position += start - cursor
            cursor = start + h * w
            ends.append(cursor)
            frame += [position] * (h * w)
            position += max(h, w)
            hh += [r for r in range(-(h - h // 2), h // 2) for _ in range(w)]
            ww += [c for _ in range(h) for c in range(-(w - w // 2), w // 2)]
        if cursor != total:
            raise Refused("the generated picture is not the sequence's last %d rows" % (shapes[-1][1] * shapes[-1][2]))
        frame = np.asarray(frame, np.int64)
        height, width = frame.copy(), frame.copy()
        height[pad], width[pad] = hh, ww
        angles = []
        for index, d in zip((frame, height, width), self.AXES):
            freqs = np.float32(1.0) / np.power(np.float32(THETA), np.arange(0, d, 2, dtype=np.float32) / np.float32(d))
            angles.append(np.outer(index.astype(np.float32), freqs))
        angles = np.concatenate(angles, axis=1)
        prefix = ends[-1] - shapes[-1][1] * shapes[-1][2]
        limits = np.full(total, total, np.uint32)
        block_end = np.zeros(total, np.int64)
        for (_, h, w), end in zip(shapes, ends):
            block_end[end - h * w:end] = end
        text = ~pad[:prefix]
        limits[:prefix] = np.where(text, np.arange(1, prefix + 1), block_end[:prefix])
        return np.cos(angles), np.sin(angles), limits, prefix

    @staticmethod
    def sequence(slots, shapes):
        """The sequence a prompt makes: `slots` says of each of the encoder's rows whether it is a picture's slot — a slot
        stands for 2 x 2 of the picture's latent rows — and `shapes` each picture's (frames, height, width) in latent rows,
        the references' in order, then the generated one's, whose rows follow the prompt. Answers each row's `pad` (a
        picture's or not) and the runs that fill the prompt's rows: ("text", from, first, rows), the encoder's text rows
        `from ..` (counted among its text rows alone) into the sequence's rows `first ..`; ("picture", from, first, rows),
        a reference's latent rows `from ..` (counted among the references' rows) — diffusers' `repeat_interleave` of the
        slots and `joint_hidden_states[:, image_pad_mask] = hidden_states`."""
        slots = np.asarray(slots, bool)
        target = shapes[-1][0] * shapes[-1][1] * shapes[-1][2]
        pad = np.concatenate([np.repeat(slots, np.where(slots, 4, 1)), np.ones(target, bool)])
        runs, row, text, refs, picture, i = [], 0, 0, 0, 0, 0
        n = len(pad) - target
        while row < n:
            if not pad[row]:
                end = row
                while end < n and not pad[end]:
                    end += 1
                runs.append(("text", text, row, end - row))
                text += end - row
            else:
                if picture >= len(shapes) - 1:
                    raise Refused("the prompt holds more pictures' slots than the %d references" % (len(shapes) - 1))
                f, h, w = shapes[picture]
                end = row + f * h * w
                if end > n or not pad[row:end].all():
                    raise Refused("picture %d's %d rows are not the prompt's slots" % (picture, f * h * w))
                runs.append(("picture", refs, row, end - row))
                refs, picture = refs + end - row, picture + 1
            row = end
        if picture != len(shapes) - 1:
            raise Refused("%d references, and the prompt holds slots for %d" % (len(shapes) - 1, picture))
        return pad, runs

    def set_layout(self, pad, shapes):
        """The rotary table and the key limits written for the sequence `pad`, `shapes` (▶ `layout`); answers the prompt's
        rows."""
        cos, sin, limits, prefix = self.layout(pad, shapes)
        if prefix > self.PREFIX_MAX:
            raise Refused("the prompt is %d rows, and this boot holds %d" % (prefix, self.PREFIX_MAX))
        self.set_rotary(cos, sin)
        self.set_limits(limits)
        return prefix

    def set_timesteps(self, timesteps):
        """Each timestep's sinusoid, a row each in its order: cos(1000·t·fₖ), then sin, fₖ = 10000^(-k/(FREQ/2)) —
        diffusers' `QwenImage21TemporalTimesteps`, in float32 as it computes them."""
        t = np.asarray(timesteps, np.float32)
        if len(t) > self.MAX_STEPS + 1:
            raise Refused("%d timesteps, and this boot holds %d" % (len(t), self.MAX_STEPS + 1))
        half = self.FREQ // 2
        freqs = np.exp(np.float32(-np.log(10000.0)) * np.arange(half, dtype=np.float32) / np.float32(half))
        args = (np.float32(1000.0) * t)[:, None] * freqs[None, :]
        self.m.write(self.SCR_AT + self.SCR["sinusoids"][0], _half(np.concatenate([np.cos(args), np.sin(args)], axis=1)))

    def set_modulation(self, prompt, picture):
        """The modulation written from the host — a check's, to hold a block to diffusers' own (`ai_qwen_image__time`
        writes it on the card): each of `prompt` (t = 0) and `picture` diffusers' row of four HIDDEN — scale₁, gate₁,
        scale₂, gate₂ — as the verb reads them, 1 + scale and tanh(gate)."""
        def row(p):
            p = np.asarray(p, np.float64).reshape(4, self.H)
            return np.stack([1.0 + p[0], np.tanh(p[1]), 1.0 + p[2], np.tanh(p[3])])
        self.m.write(self.SCR_AT + self.SCR["modulation"][0], _half(np.concatenate([row(prompt), row(picture)])))

    def set_rotary(self, cos, sin):
        """The sequence's rotary table: `cos` and `sin` a row a position, HEAD_DIM / 2 each (the transformer's angles,
        its pair k the elements 2k and 2k + 1)."""
        cos, sin = np.asarray(cos, np.float32), np.asarray(sin, np.float32)
        if len(cos) > self.MAX_ROWS:
            raise Refused("%d rows, and this boot holds %d" % (len(cos), self.MAX_ROWS))
        self.m.write(self.SCR_AT + self.SCR["rotary"][0], np.ascontiguousarray(np.concatenate([cos, sin], axis=1)).tobytes())

    def set_limits(self, limits):
        """Each row's key limit: the row reads the keys 0 .. its limit."""
        self.m.write(self.SCR_AT + self.SCR["limits"][0], np.ascontiguousarray(limits, dtype="<u4").tobytes())

    @staticmethod
    def limits(segments, total):
        """The key limits of a sequence of `total` rows whose prompt is `segments` — (start, end, is_text) runs, as
        diffusers' `_qwenimage21_prefix_segments` — and whose rows past the last are the picture's: a text row reads up
        to itself, a reference picture's row up to its picture's end, the generated picture's every row."""
        out = np.full(total, total, np.uint32)
        for start, end, is_text in segments:
            out[start:end] = np.arange(start + 1, end + 1) if is_text else end
        return out

    def write(self, plane, rows):
        """Rows of halves into one of the rows' planes (▶ `PLANES`), from its start."""
        rows = np.asarray(rows)
        if rows.size * 2 > self.SCR[plane][1]:
            raise Refused("%d values, and %s holds %d" % (rows.size, plane, self.SCR[plane][1] // 2))
        self.m.write(self.SCR_AT + self.SCR[plane][0], _half(rows))

    def read(self, plane, n, width):
        """`n` rows of `width` halves of one of the rows' planes, as float32."""
        raw = self.m.read(self.SCR_AT + self.SCR[plane][0], n * width * 2)
        return np.frombuffer(raw, "<f2").reshape(n, width).astype(np.float32)

    def write_rows(self, x):
        self.write("x", x)

    def read_rows(self, n):
        return self.read("x", n, self.H)

    def word(self, text, what):
        """One of the package's verbs, its answer checked: an error refuses."""
        if self.m.must("(type %s)" % text, what) == self.m.K["sys__error"]:
            raise Refused("%s answered an error" % what)

    def block(self, i, rows, first, prefix):
        """The i-th block this machine holds over `rows` rows of the sequence from `first`, the prompt's `prefix` rows."""
        P = self.PLANES
        self.word("(ai_qwen_image__block %s %s %s %s QB%d %d %d %d)" % (P["x"], P["modulation"], P["rotary"], P["limits"], i,
                                                                       rows, first, prefix), "the block %d" % i)


class Pipeline:
    """A picture from a prompt — diffusers' `QwenImage21Pipeline` without references: the encoder's boot (its rows, then
    shut down), the transformer's (the steps, then shut down), the decoder's (`lisp/qwen_image_vae.lisp` on the card, or
    numpy on the host). One boot at a time: the card holds one model's planes; sharing it with a text model is the next
    step (NN-89's done-when)."""

    def __init__(self, path, silicon=None, program=None, vae="card", vae_program=None):
        self.path, self.silicon, self.program, self.vae_program = path, silicon, program, vae_program
        if vae not in ("card", "host"):
            raise Refused("the VAE runs on the 'card' or the 'host', not %r" % (vae,))
        self.vae = vae

    def encode(self, prompt):
        enc = Encoder(self.path, silicon=self.silicon)
        try:
            return enc.encode(prompt)
        finally:
            enc.model.m.shutdown()

    def latents(self, rows, slots, height, width, steps, noise, progress=None):
        """The steps over the prompt's `rows` from `noise` (the picture's rows of latents): the latents after them —
        `progress(step, steps)` told after each step (▶ `QwenImage.steps`), the program's `denoise` in one call without."""
        from .machine import Machine
        h, w = height // 16, width // 16
        m = Machine(silicon=self.silicon)
        qi = QwenImage(self.path, m, max_rows=len(rows) + h * w, prefix_max=max(len(rows), 1), chunk=1024, max_steps=steps,
                       program=self.program)
        m.boot(qi.device_params())
        try:
            qi.load()
            qi.bind_picture(rows, slots, [(1, h, w)], noise, sigmas=qi.schedule(steps, h * w))
            return qi.denoise(steps) if progress is None else qi.steps(steps, progress)
        finally:
            m.shutdown()

    def picture(self, prompt, height=1024, width=1024, steps=40, seed=0, noise=None, progress=None):
        """The picture: H x W x 4 floats in [-1, 1] (RGBA). `noise` the starting latents (h·w x 64), else drawn from
        `seed`. Sides are multiples of 32, as the pipeline makes them. `progress(stage, done, total)` is told as it goes —
        "encode", then "step" a step at a time, then "decode" — and a False from it stops the picture (`Stopped`)."""
        tell = progress or (lambda *a: None)
        if height % 32 or width % 32:
            raise Refused("a picture's sides are multiples of 32, not %d x %d" % (width, height))
        h, w = height // 16, width // 16
        if noise is None:
            noise = np.random.default_rng(seed).standard_normal((h * w, 64)).astype(np.float32)
        tell("encode", 0, 1)
        rows, slots = self.encode(prompt)
        if tell("encode", 1, 1) is False:
            raise Stopped("stopped after the prompt was read")
        lat = self.latents(rows, slots, height, width, steps, noise,
                           None if progress is None else (lambda i, n: tell("step", i, n)))
        tell("decode", 0, 1)
        return self.decode(lat, h, w)

    def decode(self, lat, h, w):
        """The latents to a picture: on the card — a boot of its own, so the card holds one model's planes at a time — or
        on the host (`vae="host"`, or a machine whose silicon is the host's)."""
        from .machine import Machine
        from .qwen_image_vae import CardVae, Vae
        m = Machine(silicon=self.silicon) if self.vae == "card" else None
        if m is None or m.silicon == "host":
            return Vae(self.path).decode(lat, h, w)
        cv = CardVae(self.path, m, max_h=h, max_w=w, program=self.vae_program)
        m.boot(cv.device_params())
        try:
            cv.load()
            return cv.decode(lat, h, w)
        finally:
            m.shutdown()
