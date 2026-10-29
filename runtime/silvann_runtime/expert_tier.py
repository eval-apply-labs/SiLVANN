"""A CARD'S TIER OF ROUTED EXPERTS (NN-48) — what a model's runtime shares: the threshold its prompt programs name, measured
once a boot on this machine and bound in the Lisp environment.

`nn__expert_major__vram_promotion_threshold_picks` is the fewest of a prompt chunk's rows past which an expert costs the
CPUs more than its copy to the card. ⚖ *"make 12 be a lisp bound variable so it is calculated once and from the current
setup"* · *"make the lisp variable nn__expert_major__vram_promotion_threshold_picks"*.
"""
import time

import numpy as np

THRESHOLD = "nn__expert_major__vram_promotion_threshold_picks"
READINGS = 5


def _timed(fn):
    a = time.perf_counter()
    fn()
    return time.perf_counter() - a


def _half(a):
    return np.asarray(a, dtype=np.float64).astype("<f2").tobytes()


def _line(copy, weights, per_row, most_rows):
    """The fewest rows past which an expert costs the CPUs more than its copy: 1 where the copy is no slower than the
    expert's weights alone, `most_rows` at most."""
    return max(1, min(most_rows, int(np.ceil((copy - weights) / per_row)) if copy > weights else 1))


def calibrate(m, card, cpu, copy_to, slot_bytes, hands_at, hand_bytes, weights_at, picks_at, top_k, experts, rows_text,
              most_rows, fixed=None, pool=None, readings=READINGS):
    """The threshold, bound, and what it was measured from. A reading is the copy — one expert's bytes written to the
    card at `copy_to` (a scratch nothing else holds), best of three — and the CPU: its experts verb over a chunk of
    `rows_text(n)` — `top_k` experts each picked by every one of `n` hand rows written at `hands_at` (input zeros,
    weights at `weights_at`, picks at `picks_at` of a `hand_bytes` row) — for one row and for 64, best of three on
    experts not used before, so their weights come from RAM as a prompt's do — out of `pool`, the experts the CPU holds,
    where it does not hold every one (an exclusive tier). The line comes from the medians of `readings` readings, so no
    one reading sets it: a single reading's copy can measure no slower than an expert's weights, and its line of 1 sends
    every expert a prompt picks to the card (`MEASURED` once in a GLM boot, 186.7 promotions a token against ~45 —
    measurements/2026-10-07_S115_glm_on_its_lisp.md). `fixed` binds a number instead."""
    line, measured = fixed, {}
    if line is None:
        pool = list(range(experts)) if pool is None else list(pool)
        taken, x0 = [], 0
        for _ in range(readings):
            copy = copy_seconds(m, card, copy_to, slot_bytes)
            costs = {}
            m.act_as(cpu)
            for rows in (1, 64):
                best = None
                for _ in range(3):
                    hand = np.zeros(hand_bytes, np.uint8)
                    hand[weights_at:weights_at + 2 * top_k] = np.frombuffer(_half(np.full(top_k, 0.1)), np.uint8)
                    hand[picks_at:picks_at + 8 * top_k] = np.frombuffer(np.asarray(pool[x0:x0 + top_k], dtype="<u8").tobytes(), np.uint8)
                    # an exclusive tier that holds all but `top_k` of a layer's leaves the CPU those alone, read each time
                    x0 = (x0 + top_k) % (len(pool) - top_k) if len(pool) > top_k else 0
                    m.write(hands_at, np.tile(hand, rows).tobytes())
                    t = _timed(lambda: m.run(rows_text(rows)))
                    best = t if best is None else min(best, t)
                costs[rows] = best / top_k
            per_row = max((costs[64] - costs[1]) / 63, 1e-9)
            taken.append((copy, max(costs[1] - per_row, 0.0), per_row))
        m.act_as(card)
        copy, weights, per_row = (float(np.median([t[k] for t in taken])) for k in range(3))
        line = _line(copy, weights, per_row, most_rows)
        measured = dict(copy_ms=copy * 1e3, weights_ms=weights * 1e3, row_ms=per_row * 1e3,
                        lines=[_line(*t, most_rows) for t in taken])
    m.must("(sys__add_bindings %d '%s %d)" % (m.env, THRESHOLD, line), "the promotion threshold", "sys__value_true")
    measured["line"] = line
    return measured


BINDING = "nn__expert_tier"
DUPLEX = "nn__expert_tier_duplex"
DUPLEX_RATIO = "nn__expert_tier_duplex_ratio"
DUPLEX_LINE = "nn__expert_tier_duplex_line"
LANDING_LINE = "nn__expert_tier_landing_line"


def duplex(m, card, slot_bytes, line=1200, fixed=None):
    """Whether a copy to the card and one back run at once at full speed — three bindings, so the program decides and
    each can be set instead of measured. `nn__expert_tier_duplex_ratio`: both copies of an expert's `slot_bytes` at once,
    over the copy to the card alone, ×1000 (▶ `(nn__expert_tier__duplex bytes)`, run as the `card`). `…_duplex_line`: the
    ratio below which they count as running together, ×1000 — ⚖ *"the 1.2 should also be a lisp binding parameter"*.
    `nn__expert_tier_duplex`: #t below the line. `fixed`, a ratio, binds that instead of measuring it."""
    m.act_as(card)
    ratio = int(fixed) if fixed is not None else m.must("(nn__expert_tier__duplex %d)" % slot_bytes, "the duplex probe")
    for name, v in ((DUPLEX_RATIO, ratio), (DUPLEX_LINE, int(line))):
        m.must("(sys__add_bindings %d '%s %d)" % (m.env, name, v), "the binding %s" % name, "sys__value_true")
    m.must("(sys__add_bindings %d '%s (< %s %s))" % (m.env, DUPLEX, DUPLEX_RATIO, DUPLEX_LINE), "the binding %s" % DUPLEX,
           "sys__value_true")
    return dict(ratio=ratio, line=int(line), full=ratio < int(line))


# ⭐ THE CARD'S LANDING PAD (NN-115, ▶ sys's `pad.cuh`): two chunks up and two down, each the largest expert — a card
#   slot — and 1 MB for the messages beside it, whole pages. ⚖ 2026-10-24 *"just keep 4 slots per card of expert max size
#   for the full duplex"* · *"add like 1mb to the biggest expert when sizing for the message passing"*: two threads fill
#   each promotion and keep the link full (`MEASURED` measurements/2026-10-24_nn115_pad_node03.md).
#   ⚖ *"we have to keep the parking if we want to do it in full duplex"*: an exclusive tier's evictions wait in the down
#   way until their swap commits, a chunk a swap in flight — PARKING over a group, each card its share, the swaps nn's
#   evacuation channels allowed.
PAD_UP, PAD_DOWN, PARKING = 2, 2, 16


def pad_bytes(slot_bytes):
    """A pad chunk for experts of `slot_bytes`: the expert and 1 MB, whole pages."""
    return (slot_bytes + (1 << 20) + 4095) // 4096 * 4096


def pad_down(exclusive, ranks):
    """The pad's down chunks: an exclusive tier's card's share of the parking, else two."""
    return max(1, PARKING // max(1, int(ranks))) if exclusive else PAD_DOWN


# ⭐ A PROMPT STREAMED THROUGH A RING (NN-115 ②, ▶ nn's STREAM): a chunk's experts the tier does not hold promoted into a
#   ring of the card's own a batch ahead, the card computing one batch while the next lands — STREAM_BATCH experts a batch,
#   the ring two of the largest batch.
STREAM_BATCH, STREAM_BATCH_MAX = 4, 8
NOTE_STREAM = 1 << 62                   # a note's line with this bit streams, its line's experts into the tier first (▶ nn's)


def tier(m, name, type_, experts, top_k, hidden, inter, plan, layers, sources=None, parts=1, part_plan=None, gate_row=0,
         down_row=0, landing=0, exclusive=None, landing_line=3, ranks=1, shared=0, stream=None, stream_batch=STREAM_BATCH):
    """A card's tier of routed experts as nn reads it (▶ NN__EXPERT_TIER__), bound by `name`: the card's collection, the
    card slot's `plan`, and — where the CPUs keep every expert — `sources`, {layer: the name of a node array of that layer's
    addresses, part p of expert x at x · parts + p}, with a part slot's `part_plan` and the row bytes the stitch splices
    by. A layer without sources stays what boot made it. `exclusive`, {part_worker, part_type}: each expert on the card OR
    in RAM, a promotion swapping one back — sources on every layer it runs, and `nn__expert_tier_duplex` bound
    (▶ `duplex`), which the tier reads as its DUPLEX cell. `landing_line` binds `nn__expert_tier_landing_line`: the hits
    of a visit above which it starts one promotion, `landing` at or below it — ⚖ *"1 if the hits are more than 3/8 …
    two if it is below that (the compute on cpu masks the loading)"*. A program may rebind it and make the table again.
    `ranks` > 1: a tier on each of a tensor-parallel group's cards, a position's picks shared out among them (NN-99,
    ▶ nn's RANKS), `shared` of a layer's experts kept on every one of them (▶ nn's SHARED). `stream`: the ring's
    collection a chunk's experts are streamed through when its note's line is 0, `stream_batch` experts a batch.
    Made as the card's worker, it opens that card's landing pad, which every promotion is moved through (▶ `pad_bytes`)."""
    m.must("(sys__add_bindings %d '%s %d)" % (m.env, LANDING_LINE, int(landing_line)), "the binding " + LANDING_LINE,
           "sys__value_true")
    m.must("(sys__pad__open %d %d %d)" % (PAD_UP, pad_down(exclusive, ranks), pad_bytes(plan.slot_bytes)),
           "the card's landing pad")
    off = lambda p: [p.at_up_lut - p.at_up_data, p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data]
    pp = part_plan if part_plan is not None else plan
    if sources:
        m.table(name + "_sources", [sources.get(l, 0) for l in range(layers)])
    m.table(name, [name + "_sources" if sources else 0, type_, experts, top_k, hidden, inter] + off(plan) + [parts] + off(pp)
            + [gate_row, down_row, landing]
            + ([1, exclusive["part_worker"], exclusive["part_type"], "(if %s 1 0)" % DUPLEX] if exclusive else [0, 0, 0, 0])
            + [LANDING_LINE] + ([int(ranks), int(shared)] if ranks > 1 or stream is not None else [])
            + ([int(stream) + 1, int(stream_batch)] if stream is not None else []))


# ⭐ THE SHARED SLICE FROM THE MACHINE'S OWN SPEEDS (NN-99) — ⚖ 2026-10-21 *"it is likely better to have zero shared if we
#   have even one expert on the ram … let's keep that 10% cause if avx512 or an igpu reduces the gap to 4x then the shared
#   setup still pays"*: a slot every card keeps is a slot fewer of distinct experts, so a pick more sent to the CPUs,
#   and it pays only where a CPU's pick costs less than CROSSOVER card experts. `speeds` measures both at boot.
SHARED = 0.1
CROSSOVER = 4.0


def slice_of(cpu_s, card_s):
    """The share of a card's slots every card keeps, from a CPU pick's and a card expert's seconds."""
    return SHARED if card_s > 0 and cpu_s / card_s < CROSSOVER else 0.0


# ⭐ A VISIT'S LANDINGS FROM THE MACHINE'S OWN SPEEDS (NN-99) — ⚖ 2026-10-22 *"we can try landing 2 as an autotuned option
#   compared against the length of computing a cpu expert vs uploading it to vram, and see how many fit. we round to the
#   closest and set it at 1 if the result is 0"*: a promotion is copied while the CPUs compute the visit's picks the cards
#   do not hold, so as many fit as one expert's copy goes into a CPU expert's time. ⚖ *"reuse calibrate's copy time"*:
#   the copy is `calibrate`'s median reading where it measured one, `copy_seconds` where it did not.
def landing_of(cpu_s, copy_s):
    """A visit's promotions at most, from a CPU pick's seconds and one expert's copy to the card's: the nearest whole
    number, at least 1."""
    return max(1, int(round(cpu_s / copy_s))) if copy_s > 0 else 1


def copy_seconds(m, card, at, nbytes, readings=3):
    """One expert's copy to `card`, the best of `readings`: the `nbytes` of the card's memory at `at` read back once,
    then written back where they were from a host buffer — the card's bytes the same before and after, so `at` may be any
    span the card holds (a scratch, the head's codes). ⛳ A CARD AT REST: a promotion's write shares the bus with what the
    card computes meanwhile, so this is its fastest."""
    m.act_as(card)
    held = np.frombuffer(m.read(at, nbytes), dtype=np.uint8).copy()
    return min(_timed(lambda: m.write_from(at, held.ctypes.data, nbytes)) for _ in range(readings))


def speeds(m, card, cpu, picks, ids, cpu_text, card_scale, card_text="(head)", readings=3):
    """(a CPU pick's seconds, a card expert's seconds), each the best of `readings`. The CPU's: its experts verb
    (`cpu_text`, as the worker `cpu`) over the experts `ids`, written into the node array `picks`. The card's: an expert's
    bytes at the rate the card reads a large matrix of its own — `card_text`, the model's head and its greedy token read
    back, so the card is done — times `card_scale`, a routed expert's bytes over the head's. ⛳ A RATE, NOT AN EXPERT: a
    card's tier is empty at boot, and a matrix an expert's size takes mostly its launches' time, which a visit's picks
    share — so the head's hundreds of MB price an expert's bytes. `REASONED`: a card expert's product as fast a byte as
    the head's, its share of a visit's launches left out; the rule is a factor of 4."""
    m.act_as(card)
    m.must("(begin %s 0)" % " ".join("(sys__node_array__set %s %d %d)" % (picks, j, x) for j, x in enumerate(ids)), "the picks")
    best = lambda run: min(_timed(run) for _ in range(readings))
    card_s = best(lambda: m.must(card_text, "the card's reading")) * card_scale
    m.act_as(cpu)
    cpu_s = best(lambda: m.must("(begin %s 0)" % cpu_text, "a CPU's reading")) / len(ids)
    m.act_as(card)
    return cpu_s, card_s


def head_bytes(plan):
    """The bytes the head's product reads: its codes and its scales (▶ the models' HPLAN)."""
    return plan.up_data + plan.down_lut


def bind(m, tiers, workers):
    """The binding `nn__expert_tier`: an entry for every worker, the name of the tier its card holds (`tiers`, {worker:
    name}) or 0 — the programs hand it to the verbs that use it, and each reads its own worker's entry."""
    m.table(BINDING, [tiers.get(w, 0) for w in range(workers)])


def part_pieces(view, gate, up, down, plan, s, parts, inter):
    """Part `s` of `parts` of one expert as (offset in its slot, bytes) pieces out of a layer's mapped blob `view`, from its
    gate, up and down records: its `inter` gate rows and up rows as one matrix, their scales, its columns of every down
    row, every down row's scale — the layout nn's tier stitches whole again (▶ `nn__expert_tier__stitch`)."""
    span = lambda at, n: view[at:at + n]
    rb, (d_at, _) = down.row_bytes // parts, down.data_span()
    # this part's columns of every down row, cut out of the rows — the one piece that is not contiguous
    cols = np.ascontiguousarray(view[d_at:d_at + down.rows * down.row_bytes].reshape(down.rows, down.row_bytes)[:, s * rb:(s + 1) * rb]).reshape(-1)
    return [(plan.at_up_data, span(gate.data_span()[0] + s * inter * gate.row_bytes, inter * gate.row_bytes)),
            (plan.at_up_data + inter * gate.row_bytes, span(up.data_span()[0] + s * inter * up.row_bytes, inter * up.row_bytes)),
            (plan.at_up_lut, span(gate.lut_span()[0] + s * inter * 2, inter * 2)),
            (plan.at_up_lut + inter * 2, span(up.lut_span()[0] + s * inter * 2, inter * 2)),
            (plan.at_down_data, cols),
            (plan.at_down_lut, span(*down.lut_span()))]
