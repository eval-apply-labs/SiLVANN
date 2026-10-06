"""A CARD'S TIER OF ROUTED EXPERTS (NN-48) — what a model's runtime shares: the threshold its prompt programs name, measured
once a boot on this machine and bound in the Lisp environment.

`nn__expert_major__vram_promotion_threshold_picks` is the fewest of a prompt chunk's rows past which an expert costs the
CPUs more than its copy to the card. ⚖ *"make 12 be a lisp bound variable so it is calculated once and from the current
setup"* · *"make the lisp variable nn__expert_major__vram_promotion_threshold_picks"*.
"""
import time

import numpy as np

THRESHOLD = "nn__expert_major__vram_promotion_threshold_picks"


def _timed(fn):
    a = time.perf_counter()
    fn()
    return time.perf_counter() - a


def _half(a):
    return np.asarray(a, dtype=np.float64).astype("<f2").tobytes()


def calibrate(m, card, cpu, copy_to, slot_bytes, hands_at, hand_bytes, weights_at, picks_at, top_k, experts, rows_text,
              most_rows, fixed=None, pool=None):
    """The threshold, bound, and what it was measured from. The copy: one expert's bytes written to the card at `copy_to`
    (a scratch nothing else holds), best of three. The CPU: its experts verb over a chunk of `rows_text(n)` — `top_k`
    experts each picked by every one of `n` hand rows written at `hands_at` (input zeros, weights at `weights_at`, picks
    at `picks_at` of a `hand_bytes` row) — for one row and for 64, best of three on experts not used before, so their
    weights come from RAM as a prompt's do — out of `pool`, the experts the CPU holds, where it does not hold every one
    (an exclusive tier). `fixed` binds a number instead."""
    line, measured = fixed, {}
    if line is None:
        blank = np.zeros(slot_bytes, np.uint8)
        m.act_as(card)
        copy = min(_timed(lambda: m.write_from(copy_to, blank.ctypes.data, slot_bytes)) for _ in range(3))
        pool = list(range(experts)) if pool is None else list(pool)
        costs, x0 = {}, 0
        m.act_as(cpu)
        for rows in (1, 64):
            best = None
            for _ in range(3):
                hand = np.zeros(hand_bytes, np.uint8)
                hand[weights_at:weights_at + 2 * top_k] = np.frombuffer(_half(np.full(top_k, 0.1)), np.uint8)
                hand[picks_at:picks_at + 8 * top_k] = np.frombuffer(np.asarray(pool[x0:x0 + top_k], dtype="<u8").tobytes(), np.uint8)
                x0 = (x0 + top_k) % (len(pool) - top_k)
                m.write(hands_at, np.tile(hand, rows).tobytes())
                t = _timed(lambda: m.run(rows_text(rows)))
                best = t if best is None else min(best, t)
            costs[rows] = best / top_k
        m.act_as(card)
        per_row = max((costs[64] - costs[1]) / 63, 1e-9)
        weights = max(costs[1] - per_row, 0.0)
        line = max(1, min(most_rows, int(np.ceil((copy - weights) / per_row)) if copy > weights else 1))
        measured = dict(copy_ms=copy * 1e3, weights_ms=weights * 1e3, row_ms=per_row * 1e3)
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


def tier(m, name, type_, experts, top_k, hidden, inter, plan, layers, sources=None, parts=1, part_plan=None, gate_row=0,
         down_row=0, landing=0, exclusive=None, landing_line=3):
    """A card's tier of routed experts as nn reads it (▶ NN__EXPERT_TIER__), bound by `name`: the card's collection, the
    card slot's `plan`, and — where the CPUs keep every expert — `sources`, {layer: the name of a node array of that layer's
    addresses, part p of expert x at x · parts + p}, with a part slot's `part_plan` and the row bytes the stitch splices
    by. A layer without sources stays what boot made it. `exclusive`, {part_worker, part_type}: each expert on the card OR
    in RAM, a promotion swapping one back — sources on every layer it runs, and `nn__expert_tier_duplex` bound
    (▶ `duplex`), which the tier reads as its DUPLEX cell. `landing_line` binds `nn__expert_tier_landing_line`: the hits
    of a visit above which it starts one promotion, `landing` at or below it — ⚖ *"1 if the hits are more than 3/8 …
    two if it is below that (the compute on cpu masks the loading)"*. A program may rebind it and make the table again."""
    m.must("(sys__add_bindings %d '%s %d)" % (m.env, LANDING_LINE, int(landing_line)), "the binding " + LANDING_LINE,
           "sys__value_true")
    off = lambda p: [p.at_up_lut - p.at_up_data, p.at_down_data - p.at_up_data, p.at_down_lut - p.at_up_data]
    pp = part_plan if part_plan is not None else plan
    if sources:
        m.table(name + "_sources", [sources.get(l, 0) for l in range(layers)])
    m.table(name, [name + "_sources" if sources else 0, type_, experts, top_k, hidden, inter] + off(plan) + [parts] + off(pp)
            + [gate_row, down_row, landing]
            + ([1, exclusive["part_worker"], exclusive["part_type"], "(if %s 1 0)" % DUPLEX] if exclusive else [0, 0, 0, 0])
            + [LANDING_LINE])


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
