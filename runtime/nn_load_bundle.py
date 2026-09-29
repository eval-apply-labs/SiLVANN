#!/usr/bin/env python3
"""nn_load_bundle.py — THE HOST HALF OF THE LOAD PROTOCOL: bundle bytes into device slots.

⚖ ARCHITECT, on the protocol this drives: *"an abi method that loads an expert is what is needed …
the host asks for space, the device finds the appropriate slot and hands him a pointer and adds that
pointer to the pending list, the host writes with DMA and then calls the completion."*

⭐⭐ WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT. It is a **slicer and a driver**: it works out
which bytes of a v3 bundle belong to expert `i` of layer `L`, asks the card for a slot, writes them
and calls the completion. ⛔ **IT NEVER DECODES.** A loader that decoded would be a second
implementation of the codec, and the one thing a load path must not do is agree with the decoder
because they were written from each other.
⇒ ★ THE ORACLE FOR THIS FILE IS `nn_unpack_model.decode_record`, WHICH SHARES NO CODE WITH IT and
reads the same bytes by its own arithmetic. ▶ `test/src_weights_oracle.py`.

── ⛔⛔ AN EXPERT IS NOT A RECORD IN v3, AND THAT IS THE FACT THE WHOLE FILE IS SHAPED BY ─────────────
`MEASURED` by reading `layers/000/index.pkl` of `/mnt/data/bigdisk/qwen36_35b_v3`:
```
  mlp.experts.gate_up_proj   shape [256, 1024, 2048]   rows 262144  cols 2048  row_bytes 2048
  mlp.experts.down_proj      shape [256, 2048,  512]   rows 524288  cols  512  row_bytes  512
```
⇒ the packer stores **the whole stacked tensor as ONE record a layer** — 537 MB and 269 MB — so expert
`i` is not a key to look up, it is **four contiguous slices computed by arithmetic**. The deployed-122B
shape (`…_Down_ID<i>` as separate records) is superseded and nothing here looks for it.
⛳ AND THE ARITHMETIC IS ONLY VALID BECAUSE THE ROWS OF ONE EXPERT ARE CONTIGUOUS, which is a claim
about the packer's flattening order and not about the format: `[E, R, C] -> rows = E*R` puts expert
`i`'s `R` rows together, in order, at `i*R`. ⛔ `ASSUMED` until `expert_rows()` checks it — which it
does, by refusing any record whose `rows` is not `experts * rows_per_expert`. **What would falsify it:**
a packer that interleaved experts; the check above turns that into a refusal rather than a wrong slice.

── ⭐⭐ THE SLOT'S LAYOUT, AND WHY IT IS [data | lut] TWICE RATHER THAN [data data | lut lut] ─────────
A slot is two halves and the type table's `OFFSET` says where the first ends:
```
   0                     up_data    rows_up * row_bytes_up
   up_data               up_lut     rows_up * 2                       one fp16 scale a row
   OFFSET                down_data  rows_down * row_bytes_down        OFFSET = align_4k(up half)
   OFFSET + down_data    down_lut   rows_down * 2
```
⛳ EACH HALF IS SELF-CONTAINED, so `plane` can hand a compute verb one half and that half carries its
own scales. Interleaving all four would make the down half's address depend on the up half's row
count, which is exactly the coupling the one-collection-per-matrix-type ruling removed.

── ⭐ WHAT A DENSE TENSOR IS ────────────────────────────────────────────────────────────────────────
⚖ RULED: *"a dense tensor is a collection with 40 slots, one per layer."* So it is the
SAME object with the second half empty: one slice, `[data | lut]`, and `down_bytes` of zero.
⛔ HOW A PROGRAM *NAMES* ONE IS NOT RULED and this file takes no position — it writes bytes to an
address the card chose. ▶ `RESUME.md`, the open question.
"""

import json
import os
import pickle

# ⛳ THE FOUR NAMES THIS FILE KNOWS, AND IT KNOWS NO OTHERS. A model whose MoE is spelled differently
#   is a different loader, not a flag here — the point of naming them is that a typo is a KeyError at
#   load time and not a silently empty slot.
GATE_UP = "mlp.experts.gate_up_proj"
DOWN = "mlp.experts.down_proj"

FORMAT = "silvann-packed-v3"

# ══ ⭐⭐⭐ THE NORM GAIN CONVENTION — ⚖ RULED: *"the +1 goes in the loader, per tensor."* ══
#
# Hugging Face declares TWO RMSNorm classes with OPPOSITE parameter inits, and this model uses BOTH:
# ```
#   Qwen3_5MoeRMSNorm       nn.Parameter(torch.zeros(dim))   applied as `1 + w`   (:775 :955 :956 :1339)
#   Qwen3_5MoeRMSNormGated  nn.Parameter(torch.ones(size))   applied as plain `w` (:541, and ONLY :541)
# ```
# ⇒ ⭐ SO THE ATTENTION NORMS (`q_norm`, `k_norm`) TAKE THE `+1` LIKE EVERYTHING ELSE. The single
# exception in the whole model is the GatedDeltaNet's own internal norm.
#
# ⛔⛔ AND IT IS DECIDED BY THE TENSOR'S NAME, NEVER BY ITS STATISTICS. The means look separable —
# plain-`w` tensors sit near +0.88 and the rest in [-0.12, +0.40] — but `self_attn.k_norm` at +0.39 is
# already halfway, and a threshold that works on this checkpoint is a heuristic. ⇒ ★ A CONVENTION IS A
# FACT ABOUT THE CODE THAT WROTE THE FILE, SO READ IT FROM THE CODE, NOT FROM THE NUMBERS.
#
# ⛳ `src_old` DID EXACTLY THIS AND IS THE PRECEDENT: `composer.py` carries `self.rms_norm_offset`,
# derived from `model_type.startswith("qwen3")`, and adds it while mapping each norm to VRAM — never to
# `linear_attn.norm.weight`. Its own comment records the bug this guards, hit once: *"omitting it scaled
# Q/K by ~0 (raw delta) instead of ~1, collapsing attention scores in the 10 full_attention layers."*
GAIN_EXEMPT = ("linear_attn.norm.weight",)      # the gated norm — plain `w`, ▶ `modeling_…:541`


def is_norm(short_name):
    """Whether a tensor is an RMSNorm gain at all. ⛳ Both spellings, because `q_norm`/`k_norm` do not
    end in `norm.weight` and they are exactly the pair `src_old` once forgot."""
    return short_name.endswith("norm.weight") or "_norm.weight" in short_name


class Refused(Exception):
    """⛳ NAMED, AND NOT A RETURN CODE — the same choice `nn_unpack_model` makes and for the same
    reason: a bundle this loader cannot slice is not a degraded mode, and a caller that continued
    would write plausible bytes into a real slot."""


# ══ ⭐ THE BUNDLE ════════════════════════════════════════════════════════════════════════════════

class Bundle:
    """A v3 bundle, opened for slicing. Holds no weights — every read is a slice of an mmap."""

    def __init__(self, path, lora=None):
        self.path = path
        man_path = os.path.join(path, "pack.json")
        if not os.path.exists(man_path):
            raise Refused("no pack.json at %r — this is not a bundle" % (path,))
        self.manifest = json.load(open(man_path))
        if self.manifest.get("format") != FORMAT:
            raise Refused("format %r is not %s" % (self.manifest.get("format"), FORMAT))
        self.n_layers = int(self.manifest["n_layers"])
        self.norm_offset = self._norm_offset()
        self._index = {}
        self._blob = {}
        # ⭐ THE RESIDUAL KEPT ROTATED (`nn_pack_model.py --residual`): the norms' gains folded in, the residual writers
        #   rotated. ⛔ A value this reader does not know is refused, since reading it either way would be a guess.
        residual = self.manifest.get("residual")
        if residual not in (None, "rotated"):
            raise Refused("residual %r is not a form this reader knows" % (residual,))
        self.residual_rotated = residual == "rotated"
        # ⭐ A LoRA — an adapter merged into the matrices it adapts (`nn_lora_overlay.py`), kept in this bundle's
        #   `loras/NAME/`: a record it holds is read from it, every other from this bundle. ⚖ *"at boot is enough …
        #   we say it in the json which lora we want"*. ⛔ Refused unless it was made from this bundle's model and packed
        #   the way this bundle was — named by identity, because a record quantised another way decodes as a
        #   plausible wrong matrix. A path to an overlay elsewhere is accepted too; the identity decides either way.
        self.lora, self.overlay = None, None
        if lora:
            where = lora if os.sep in lora else os.path.join(path, "loras", lora)
            self.overlay = Bundle(where)
            om = self.overlay.manifest.get("overlay")
            if om is None:
                raise Refused("%r is a bundle, not a LoRA overlay" % (where,))
            mine = dict(index_sha=self.manifest["source"]["index_sha"], rotation=self.manifest["rotation"],
                        codec=self.manifest["codec"], policy=self.manifest["policy"], residual=self.manifest.get("residual"))
            for k, v in mine.items():
                if om["base"].get(k) != v:
                    raise Refused("the LoRA %r was made for another base: its %s is %r, this bundle's %r"
                                  % (lora, k, om["base"].get(k), v))
            self.lora = dict(name=om["name"], adapter=om["adapter"])
            self._overlaid = set(om["layers"])

    def _norm_offset(self):
        """The `1 + w` delta for this bundle's model family, or None when it cannot be derived.

        ⛔⛔ IT IS DERIVED FROM THE SOURCE MODEL'S `model_type` AND NOT DEFAULTED. A silent 1.0 would be
        right for every checkpoint this tree has ever seen and wrong for the first plain-convention one,
        with no symptom but a model that answers badly. ⇒ ★ A CONSTANT THAT IS RIGHT FOR EVERY INPUT SO
        FAR IS THE HARDEST KIND TO FIND WRONG. `None` here makes `gain_offset` REFUSE rather than guess.
        ⛳ THE PREDICATE IS `src_old`'s, verbatim in effect: `model_type.startswith("qwen3")`."""
        src = (self.manifest.get("source") or {}).get("path")
        if not src:
            return None
        cfg = os.path.join(src, "config.json")
        if not os.path.exists(cfg):
            return None
        try:
            man = json.load(open(cfg))
        except (ValueError, OSError):
            return None
        mt = str((man.get("text_config") or man).get("model_type") or man.get("model_type") or "")
        return 1.0 if mt.startswith("qwen3") else 0.0

    def gain_offset(self, short_name):
        """What to add to this tensor's weights before they go on the card. ⛳ 0.0 for anything that is
        not a norm, so a caller may ask about every tensor without branching."""
        if not is_norm(short_name) or short_name in GAIN_EXEMPT:
            return 0.0
        if self.norm_offset is None:
            raise Refused(
                "%s is an RMSNorm gain and this bundle's convention could not be derived — its "
                "`source.path` config is unreadable. Pass the offset explicitly rather than letting "
                "a loader guess: `1 + w` and `w` differ by a factor of ~30 on these weights."
                % (short_name,))
        return self.norm_offset

    def layer_dir(self, layer):
        return os.path.join(self.path, self.manifest["layout"]["layer_dir"].format(L=layer))

    def index(self, layer):
        """The layer's record table. Read once and kept — it is ~6 KB against a 1.6 GB blob."""
        if layer not in self._index:
            p = os.path.join(self.layer_dir(layer), self.manifest["layout"]["index"])
            if not os.path.exists(p):
                raise Refused("layer %d has no index at %r" % (layer, p))
            self._index[layer] = pickle.load(open(p, "rb"))["weights"]
        return self._index[layer]

    def blob(self, layer):
        """The layer's `weights.bin`, memory-mapped.

        ⛳ `mmap` AND NOT A READ, because a layer is 1.6 GB and a loader that read it whole would need
        the model's size in RAM to put the model on a card. Slicing an mmap touches only the pages a
        slice covers, which is what makes loading one expert cost one expert."""
        if layer not in self._blob:
            import mmap
            p = os.path.join(self.layer_dir(layer), self.manifest["layout"]["weights_bin"])
            f = open(p, "rb")
            self._blob[layer] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._blob[layer][1]

    def record(self, layer, short_name):
        """One record, by the name it carries after the layer prefix is stripped — the overlay's when it has one."""
        full = "model.language_model.layers.%d.%s" % (layer, short_name)
        if self.overlay is not None and layer in self._overlaid and full in self.overlay.index(layer):
            return Record(self.overlay, layer, full, self.overlay.index(layer)[full])
        idx = self.index(layer)
        if full not in idx:
            raise Refused("layer %d has no record %r" % (layer, short_name))
        return Record(self, layer, full, idx[full])

    def names(self, layer):
        """Every record this layer carries, short — the layer prefix stripped."""
        pre = "model.language_model.layers.%d." % (layer,)
        return sorted(k[len(pre):] for k in self.index(layer) if k.startswith(pre))

    def close(self):
        if self.overlay is not None:
            self.overlay.close()
        for f, m in self._blob.values():
            m.close()
            f.close()
        self._blob = {}


# ══ ⭐⭐ ONE RECORD, AND THE SLICES INSIDE IT ═══════════════════════════════════════════════════

class Record:
    """A packed record's geometry and its two byte planes.

    ⛔ EVERY FIELD EXTENT IS CHECKED AGAINST THE RECORD AND THE RECORD AGAINST THE FILE, which is the
    same refusal `nn_unpack_model` makes. A loader is the one consumer for which an over-long slice is
    silent: it would write a neighbouring record's bytes into a slot and every count would stay right."""

    def __init__(self, bundle, layer, name, rec):
        self.bundle = bundle
        self.layer = layer
        self.name = name
        self.rec = rec
        sc = rec["scalars"]
        if sc.get("storage") != "packed":
            raise Refused("%s: storage %r — v3 stores every record packed" % (name, sc.get("storage")))
        self.rows = int(sc["rows"])
        self.cols = int(sc["cols"])
        self.d = int(sc["d"])
        self.row_bytes = int(sc["row_bytes"])
        self.family = int(sc["family"])
        self.original_shape = list(sc["original_shape"])
        self.offset = int(rec["record_offset"])
        self.nbytes = int(rec["record_nbytes"])

    # ── the two planes, as byte ranges in the layer blob ──────────────────────────────────────────
    def _field(self, which):
        f = self.rec["fields"][which]
        rel, n = int(f["rel"]), int(f["nbytes"])
        if rel + n > self.nbytes:
            raise Refused("%s: field %s runs past its record" % (self.name, which))
        return self.offset + rel, n

    def data_span(self):
        """(absolute offset, bytes) of the packed codes."""
        return self._field("vbr_data")

    def lut_span(self):
        """(absolute offset, bytes) of the per-row fp16 scales."""
        return self._field("vbr_lut")

    def data_bytes(self, row0=0, nrows=None):
        """The codes for `nrows` rows starting at `row0`, as bytes."""
        at, n = self.data_span()
        if n < self.rows * self.row_bytes:
            raise Refused("%s: the codes plane is short: %d < %d"
                          % (self.name, n, self.rows * self.row_bytes))
        nrows = self.rows if nrows is None else nrows
        if row0 < 0 or nrows < 0 or row0 + nrows > self.rows:
            raise Refused("%s: rows [%d,%d) are not inside %d"
                          % (self.name, row0, row0 + nrows, self.rows))
        a = at + row0 * self.row_bytes
        return bytes(self.bundle.blob(self.layer)[a:a + nrows * self.row_bytes])

    def lut_bytes(self, row0=0, nrows=None):
        """The fp16 scales for `nrows` rows starting at `row0`, as bytes.

        ⛔ `d == 16` CARRIES NO SCALE — one fp16 total, and it is not a per-row plane. The packer still
        emits a two-byte field there, so slicing it per row would hand row 1 the bytes after the field.
        This refuses instead, because a zero-length scale plane and a one-entry one are different
        mistakes and only a refusal distinguishes them."""
        at, n = self.lut_span()
        nrows = self.rows if nrows is None else nrows
        if n < self.rows * 2:
            raise Refused("%s: the scale plane holds %d bytes, not %d — a d=%d record carries one "
                          "scale for the whole tensor, not one a row"
                          % (self.name, n, self.rows * 2, self.d))
        if row0 < 0 or nrows < 0 or row0 + nrows > self.rows:
            raise Refused("%s: rows [%d,%d) are not inside %d"
                          % (self.name, row0, row0 + nrows, self.rows))
        a = at + row0 * 2
        return bytes(self.bundle.blob(self.layer)[a:a + nrows * 2])

    # ── ⭐⭐ THE STACKED-EXPERT ARITHMETIC ─────────────────────────────────────────────────────────
    def expert_rows(self, experts):
        """How many rows one expert owns — and the check that makes the slicing legitimate.

        ⛔ IT REFUSES A RECORD THAT DOES NOT DIVIDE, which is what turns the contiguity ASSUMPTION at
        the top of this file into something a wrong packer trips over. `rows` must be exactly
        `experts * rows_per_expert`, and `original_shape[0]` must be the expert count the caller
        named — so a caller asking for 128 experts of a 256-expert record is refused rather than
        handed the first half."""
        if not self.original_shape or int(self.original_shape[0]) != int(experts):
            raise Refused("%s: this record stacks %r experts, not %d"
                          % (self.name, self.original_shape[:1], experts))
        if experts <= 0 or self.rows % experts != 0:
            raise Refused("%s: %d rows do not divide into %d experts"
                          % (self.name, self.rows, experts))
        return self.rows // experts

    def expert_slices(self, which, experts):
        """`(data, lut)` bytes for expert `which` of `experts`."""
        per = self.expert_rows(experts)
        if which < 0 or which >= experts:
            raise Refused("%s: expert %d is not inside %d" % (self.name, which, experts))
        return self.data_bytes(which * per, per), self.lut_bytes(which * per, per)


# ══ ⭐⭐ THE SLOT — WHAT THE FOUR SLICES LOOK LIKE ONCE THEY ARE ON THE CARD ══════════════════════

def align_up(n, to=4096):
    return ((n + to - 1) // to) * to


class SlotPlan:
    """Where each slice lands inside one slot, and how big the slot has to be.

    ⭐⭐ IT IS COMPUTED FROM THE RECORDS AND NOT WRITTEN DOWN, which is the same argument the codec
    makes about its block structure: a derived geometry cannot disagree with the one the bytes were
    packed under. The config keys this produces (`type<n>__up_bytes` etc.) are therefore an OUTPUT of
    reading a bundle, never a figure a human keeps in step with one."""

    def __init__(self, up_data, up_lut, down_data=0, down_lut=0):
        self.up_data, self.up_lut = up_data, up_lut
        self.down_data, self.down_lut = down_data, down_lut
        self.up_bytes = up_data + up_lut
        self.down_bytes = down_data + down_lut
        self.offset = align_up(self.up_bytes)
        self.slot_bytes = self.offset + align_up(self.down_bytes)

    # the four addresses, relative to the slot's own base
    @property
    def at_up_data(self):
        return 0

    @property
    def at_up_lut(self):
        return self.up_data

    @property
    def at_down_data(self):
        return self.offset

    @property
    def at_down_lut(self):
        return self.offset + self.down_data

    def __repr__(self):
        return ("SlotPlan(up=%d+%d, down=%d+%d, offset=%d, slot=%d)"
                % (self.up_data, self.up_lut, self.down_data, self.down_lut,
                   self.offset, self.slot_bytes))


def expert_plan(bundle, layer=0, experts=256):
    """The slot plan for one expert of this bundle — read off the records, not assumed."""
    gu = bundle.record(layer, GATE_UP)
    dn = bundle.record(layer, DOWN)
    return SlotPlan(gu.expert_rows(experts) * gu.row_bytes,
                    gu.expert_rows(experts) * 2,
                    dn.expert_rows(experts) * dn.row_bytes,
                    dn.expert_rows(experts) * 2)


def expert_part_plan(bundle, layer=0, experts=256, parts=2):
    """The slot plan for PART of every expert, split by output rows over `parts` workers: I/parts of its gate_up's
    pairs (their gate rows, then the same up rows) and H/parts of its down's rows. Whole rows, so the split never
    lands inside a row and no quantisation width needs padding. ▶ ai_qwen_3's `AI_QWEN_3__EXP__PART`."""
    gu = bundle.record(layer, GATE_UP)
    dn = bundle.record(layer, DOWN)
    gu_rows, dn_rows = gu.expert_rows(experts), dn.expert_rows(experts)
    if gu_rows % (2 * parts) != 0 or dn_rows % parts != 0:
        raise Refused("an expert of %d gate_up and %d down rows does not split into %d parts"
                      % (gu_rows, dn_rows, parts))
    return SlotPlan(gu_rows // parts * gu.row_bytes, gu_rows // parts * 2,
                    dn_rows // parts * dn.row_bytes, dn_rows // parts * 2)


def dense_part(bundle, short_name, layer, rows=None, cols=None):
    """A PART of a dense tensor, for tensor parallelism: `(data, lut)` bytes of the rows in `rows` — a list of
    `(first, count)` runs, concatenated — or of every row's columns `cols = (first, end)`. Whole rows, or whole
    containers of a row: a column cut that lands inside a container is refused (▶ padding, for a width whose
    container does not divide the cut). ⛔ NEVER A NORM: those carry the `1 + w` convention and stay whole."""
    r = bundle.record(layer, short_name)
    if bundle.gain_offset(short_name) != 0.0:
        raise Refused("%s is a norm — it is not cut, every part holds it whole" % short_name)
    d_at, d_n = r.data_span()
    l_at, l_n = r.lut_span()
    blob = r.bundle.blob(layer)          # the record's own bundle: an overlay's offsets are into the overlay
    data, lut = bytes(blob[d_at:d_at + d_n]), bytes(blob[l_at:l_at + l_n])
    per_row_lut = r.d != 16
    if rows is not None:
        out_d, out_l = b"", b""
        for first, count in rows:
            if first < 0 or count <= 0 or first + count > r.rows:
                raise Refused("%s: rows [%d, %d) are not inside %d" % (short_name, first, first + count, r.rows))
            out_d += data[first * r.row_bytes:(first + count) * r.row_bytes]
            if per_row_lut:
                out_l += lut[first * 2:(first + count) * 2]
        return out_d, (out_l if per_row_lut else lut)
    if cols is not None:
        c0, c1 = cols
        per = 16 // r.d
        if c0 < 0 or c1 > r.cols or c0 >= c1 or c0 % per != 0 or (c1 % per != 0 and c1 != r.cols):
            raise Refused("%s: columns [%d, %d) do not cut on a %d-value container" % (short_name, c0, c1, per))
        b0, b1 = c0 // per * 2, -(-c1 // per) * 2
        return b"".join(data[k * r.row_bytes + b0:k * r.row_bytes + b1] for k in range(r.rows)), lut
    return data, lut


def dense_plan(bundle, short_name, layer=0):
    """The slot plan for a dense tensor — `up = [codes]`, `down = [scales]`.

    ⚖ *"a dense tensor is a collection with 40 slots, one per layer."* ⭐⭐ AND IT NEEDS NO NEW SHAPE:
    a v3 record is `vbr_data` then `vbr_lut`, which is exactly a slot's two halves. So a dense
    collection is an ordinary one — both halves non-zero, both aligned, nothing optional.
    ⛔ THE OBVIOUS READING — one slice and `down_bytes: 0` — IS THE ONE THAT DOES NOT WORK, because
    `sys__settings__size` refuses a zero and the device cannot tell an absent key from a typo'd one.
    ▶ `expert__header.cuh`, which records the attempt and why it came out."""
    r = bundle.record(layer, short_name)
    # ⛔⛔ THE FIELD SIZES ARE READ OFF THE RECORD AND NOT RECOMPUTED, WHICH IS NOT PEDANTRY: at
    #   `d == 16` a row carries NO scale, so the lut field is **2 bytes for the whole tensor** and not
    #   `rows * 2`. `mlp.gate.weight` is 256 rows and a 2-byte lut. A plan that multiplied would ask
    #   for 512 bytes, and the loader would then read 510 bytes belonging to the next record.
    #   ⇒ ★ A DERIVED FIGURE IS ONLY SAFE WHERE THE DERIVATION HOLDS FOR EVERY WIDTH.
    return SlotPlan(r.data_span()[1], 0, 0, r.lut_span()[1])


# ══ ⭐⭐⭐ THE DRIVER — reserve · write · assign ══════════════════════════════════════════════════

class Loader:
    """Drives the three-verb protocol against a booted engine.

    ⛔⛔ IT TAKES THE ENGINE MODULE AND A `run` CALLABLE RATHER THAN IMPORTING EITHER. The device verbs
    are reached by building a lisp form, and the shape of that form belongs to whoever booted the
    machine — a loader that built its own would be a second harness, drifting from the one every test
    uses. ⇒ ★ A DRIVER SHOULD BORROW ITS TRANSPORT, NOT MINT ONE.

    `run(program_cells) -> (kind, value)` is the whole contract, plus the three cell makers."""

    def __init__(self, engine, run, verb, num, sub, listify, kinds):
        self.e = engine
        self._run = run
        self._vb = verb
        self._num = num
        self._sub = sub
        self._L = listify
        self.K = kinds
        self.placed = []          # (layer, which, address) — everything published, in order

    # ── the three verbs ──────────────────────────────────────────────────────────────────────────
    def reserve(self, type_id):
        """Ask for a slot of this collection. Answers its address, or raises.

        ⛔ A REFUSAL COMES BACK AS AN ERROR OBJECT AND NOT AS A ZERO, so this checks the KIND. A loader
        that read the value alone would take an error's payload for an address — and an address is the
        one thing the next call writes to."""
        kind, at = self._run(self._L(self._vb("nn__expert__reserve"), self._num(type_id)))
        if kind != self.K["sys__value_int"]:
            raise Refused("reserve(type=%d) did not answer an address — the collection is full, or "
                          "there is no such type" % (type_id,))
        return at

    def write(self, at, payload):
        """DMA. ⛳ The one step that is not a verb, because it does not touch the index.
        The cpu wheel reaches a card's memory through its `nn` module, so that is where the write goes."""
        (self.e.nn if hasattr(self.e, "nn") else self.e).write(at, payload)

    def assign(self, layer, type_id, which, at):
        """The completion. Publishes the slot at `[layer][type][which]` and clears the handed row.

        ⚖ THE TYPE IS IN THE PATH — *"[layer][type][expert]"* — and `assign` checks it
        against the one the RESERVATION recorded, so a mistyped band is an error rather than a lie."""
        kind, _ = self._run(self._L(self._vb("nn__expert__assign"), self._num(layer),
                                    self._num(type_id), self._num(which), self._num(at)))
        if kind == self.K["sys__object_reference"]:
            raise Refused("assign(%d, %d, %d, %#x) was refused" % (layer, type_id, which, at))
        self.placed.append((layer, type_id, which, at))
        return kind

    def handed(self):
        """How many slots are reserved and not yet published. ⭐ Zero is the whole leak check."""
        return self._run(self._L(self._vb("nn__expert__handed")))[1]

    def layer_width(self, layer, type_id, count):
        """Mint the `[layer][type]` band's expert array. ⛔ Without it `assign` refuses every expert.

        ⛳ IT ANSWERS *NULL* WHEN IT WORKED, which is the shape of a command rather than a question —
        there is no value to hand back, and `nothing` is what this language says instead of inventing
        a yes. So a caller tells the outcomes apart by KIND: null is done, an object is the refusal.
        ⇒ ★ CHECKING FOR A TRUE HERE WOULD FAIL ON SUCCESS, which is how this was first written."""
        kind, val = self._run(self._L(self._vb("nn__expert__layer"), self._num(layer),
                                      self._num(type_id), self._num(count)))
        if kind != self.K["sys__value_null"]:
            raise Refused("layer(%d, %d, %d) was refused" % (layer, type_id, count))
        return kind

    # ── ⭐⭐ THE TWO THINGS A CALLER ACTUALLY WANTS ───────────────────────────────────────────────
    def load_expert(self, bundle, plan, type_id, layer, which, experts=256):
        """One expert of one layer: four slices, one slot, one publication.

        ⛳ THE WRITES HAPPEN BETWEEN `reserve` AND `assign` AND THAT ORDER IS THE PROTOCOL, not a
        convenience: until `assign` runs the slot is on the handed list, so a crash mid-write leaves
        an address somebody can still account for. ⇒ ★ THE WINDOW IN WHICH BYTES ARE HALF-WRITTEN IS
        EXACTLY THE WINDOW IN WHICH THE SLOT IS OBSERVABLY OUTSTANDING."""
        gu = bundle.record(layer, GATE_UP)
        dn = bundle.record(layer, DOWN)
        gd, gl = gu.expert_slices(which, experts)
        dd, dl = dn.expert_slices(which, experts)
        for got, want, what in ((len(gd), plan.up_data, "gate_up codes"),
                                (len(gl), plan.up_lut, "gate_up scales"),
                                (len(dd), plan.down_data, "down codes"),
                                (len(dl), plan.down_lut, "down scales")):
            if got != want:
                raise Refused("%s: %d bytes, the plan says %d" % (what, got, want))
        at = self.reserve(type_id)
        self.write(at + plan.at_up_data, gd)
        self.write(at + plan.at_up_lut, gl)
        self.write(at + plan.at_down_data, dd)
        self.write(at + plan.at_down_lut, dl)
        self.assign(layer, type_id, which, at)
        return at

    def load_expert_part(self, bundle, plan, type_id, layer, which, part, parts, experts=256):
        """Part `part` of `parts` of one expert (▶ `expert_part_plan`): its gate rows and the same up rows, then its
        down rows — each worker loads only its own, into its own memory."""
        gu = bundle.record(layer, GATE_UP)
        dn = bundle.record(layer, DOWN)
        gu_rows, dn_rows = gu.expert_rows(experts), dn.expert_rows(experts)
        inter, n, m = gu_rows // 2, gu_rows // 2 // parts, dn_rows // parts
        g0, d0 = which * gu_rows, which * dn_rows
        gd = gu.data_bytes(g0 + part * n, n) + gu.data_bytes(g0 + inter + part * n, n)
        gl = gu.lut_bytes(g0 + part * n, n) + gu.lut_bytes(g0 + inter + part * n, n)
        dd, dl = dn.data_bytes(d0 + part * m, m), dn.lut_bytes(d0 + part * m, m)
        for got, want, what in ((len(gd), plan.up_data, "gate_up codes"), (len(gl), plan.up_lut, "gate_up scales"),
                                (len(dd), plan.down_data, "down codes"), (len(dl), plan.down_lut, "down scales")):
            if got != want:
                raise Refused("%s: %d bytes, the part plan says %d" % (what, got, want))
        at = self.reserve(type_id)
        self.write(at + plan.at_up_data, gd)
        self.write(at + plan.at_up_lut, gl)
        self.write(at + plan.at_down_data, dd)
        self.write(at + plan.at_down_lut, dl)
        self.assign(layer, type_id, which, at)
        return at

    def load_bytes(self, plan, type_id, layer, data, lut, which=0):
        """Bytes already cut for a slot — codes in the up half, scales in the down half — reserved, written, assigned."""
        if len(data) != plan.up_data or len(lut) != plan.down_lut:
            raise Refused("%d+%d bytes, the plan says %d+%d" % (len(data), len(lut), plan.up_data, plan.down_lut))
        at = self.reserve(type_id)
        self.write(at + plan.at_up_data, data)
        self.write(at + plan.at_down_lut, lut)
        self.assign(layer, type_id, which, at)
        return at

    def load_dense(self, bundle, plan, type_id, short_name, layer, which=0, at_layer=None):
        """One dense tensor of one layer: codes in the up half, scales in the down half.

        ⭐ IT APPLIES THE NORM GAIN CONVENTION — ▶ `Bundle.gain_offset` and `NN-6`. A caller does not
        choose: the tensor's NAME decides, so no site can forget the way `src_old`'s q_norm once did.

        ⚖ **`NN-5` IS RULED AND THIS IS THE SHAPE:** a dense tensor is `[layer][type][0]`,
        because the TYPE in the path is what keeps it off expert 0. So `at_layer` now defaults to the
        record's own layer and a caller normally passes neither it nor `which` — the band does the
        separating that a spare index position used to have to do.
        ⛳ `at_layer` SURVIVES for the one case that still wants it: filing a tensor under a layer that
        is not its own, which the oracles do to keep a fixture small."""
        r = bundle.record(layer, short_name)
        d_at, d_n = r.data_span()
        l_at, l_n = r.lut_span()
        blob = r.bundle.blob(layer)      # the record's own bundle: an overlay's offsets are into the overlay
        data, lut = bytes(blob[d_at:d_at + d_n]), bytes(blob[l_at:l_at + l_n])

        # ⭐⭐⭐ THE NORM GAIN CONVENTION, APPLIED HERE — ⚖ *"the +1 goes in the loader, per tensor."*
        # ⛔ ONLY AT `d == 16`, AND THE REFUSAL IS THE POINT: at that width the "codes" ARE the fp16
        #   weights, so adding 1 is an arithmetic on values. At any other width they are INDICES into a
        #   table, and adding 1 to an index is not adding 1 to a weight — it is picking the next level.
        #   ⇒ ★ THE SAME BYTES MEAN DIFFERENT THINGS AT DIFFERENT WIDTHS, so an offset that is correct
        #   at one is nonsense at the others. Every norm in this bundle is `d == 16`; a future packer
        #   that quantised one would land here and be told, rather than shipping a scrambled gain.
        gain = bundle.gain_offset(short_name)
        if gain != 0.0:
            import numpy as _np
            if r.d != 16:
                raise Refused("%s carries the `1 + w` convention but is d=%d — the offset is only "
                              "meaningful where the stored code IS the weight" % (short_name, r.d))
            data = (_np.frombuffer(data, dtype="<f2").astype(_np.float32)
                    + _np.float32(gain)).astype("<f2").tobytes()
        if len(data) != plan.up_data or len(lut) != plan.down_lut:
            raise Refused("%s: %d+%d bytes, the plan says %d+%d"
                          % (short_name, len(data), len(lut), plan.up_data, plan.down_lut))
        at = self.reserve(type_id)
        self.write(at + plan.at_up_data, data)
        self.write(at + plan.at_down_lut, lut)
        self.assign(layer if at_layer is None else at_layer, type_id, which, at)
        return at


# ══ ⛳ A LOOK AT A BUNDLE FROM THE COMMAND LINE — no device, no decode ════════════════════════════

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Describe a v3 bundle's slicing geometry.")
    ap.add_argument("path")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=256)
    a = ap.parse_args(argv)

    b = Bundle(a.path)
    print("bundle   %s  · %d layers · rotation %s" % (a.path, b.n_layers, b.manifest.get("rotation")))
    print("\nlayer %d holds %d records:" % (a.layer, len(b.index(a.layer))))
    for n in b.names(a.layer):
        r = b.record(a.layer, n)
        print("   %-50s d=%-3d rows=%-8d cols=%-5d row_bytes=%-5d shape=%s"
              % (n, r.d, r.rows, r.cols, r.row_bytes, r.original_shape))
    p = expert_plan(b, a.layer, a.experts)
    print("\none expert of %d: %r" % (a.experts, p))
    print("   up_data   at %9d  %9d bytes" % (p.at_up_data, p.up_data))
    print("   up_lut    at %9d  %9d bytes" % (p.at_up_lut, p.up_lut))
    print("   down_data at %9d  %9d bytes" % (p.at_down_data, p.down_data))
    print("   down_lut  at %9d  %9d bytes" % (p.at_down_lut, p.down_lut))
    print("   slot      %d bytes (%.3f MiB)" % (p.slot_bytes, p.slot_bytes / 1048576.0))
    b.close()
    return 0


if __name__ == "__main__":
    import sys as _s
    _s.exit(main(_s.argv[1:]))
