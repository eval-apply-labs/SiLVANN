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
              most_rows, fixed=None):
    """The threshold, bound, and what it was measured from. The copy: one expert's bytes written to the card at `copy_to`
    (a scratch nothing else holds), best of three. The CPU: its experts verb over a chunk of `rows_text(n)` — `top_k`
    experts each picked by every one of `n` hand rows written at `hands_at` (input zeros, weights at `weights_at`, picks
    at `picks_at` of a `hand_bytes` row) — for one row and for 64, best of three on experts not used before, so their
    weights come from RAM as a prompt's do. `fixed` binds a number instead."""
    line, measured = fixed, {}
    if line is None:
        blank = np.zeros(slot_bytes, np.uint8)
        m.act_as(card)
        copy = min(_timed(lambda: m.write_from(copy_to, blank.ctypes.data, slot_bytes)) for _ in range(3))
        costs, x0 = {}, 0
        m.act_as(cpu)
        for rows in (1, 64):
            best = None
            for _ in range(3):
                hand = np.zeros(hand_bytes, np.uint8)
                hand[weights_at:weights_at + 2 * top_k] = np.frombuffer(_half(np.full(top_k, 0.1)), np.uint8)
                hand[picks_at:picks_at + 8 * top_k] = np.frombuffer(np.arange(x0, x0 + top_k, dtype="<u8").tobytes(), np.uint8)
                x0 = (x0 + top_k) % (experts - top_k)
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
