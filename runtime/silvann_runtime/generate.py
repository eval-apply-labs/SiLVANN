"""AN ANSWER GENERATED ON THE CARD — `generate` (▶ the model folder's lisp/generate.lisp): up to `GENERATE_MOST` tokens a
call, each the next one's input, ending early after an end token; only the tokens come back.

A family whose program has `(position tok pos)` and a head answers with it by taking `Generates`: its boot calls
`_answer_buffers`, binds STOPS (`STOPS_MOST`) and SEEN with its shape, loads lisp/generate.lisp, and runs a program with
`_run`.
"""
from . import sampling
from .model_folder import Refused

GENERATE_MOST = 64                # the most tokens one `generate` makes — OUT's length
STOPS_MOST = 8                    # the most end tokens it watches for — STOP's length, and STOPS
NO_TOKEN = 1 << 30                # a STOP slot no end token fills: past every vocabulary, no token is it


class Generates:
    GENERATES = True                # `generate` — an answer's tokens on the card (▶ lisp/generate.lisp)

    def _answer_buffers(self):
        """`generate`'s names: what it made, and the end tokens it watches for."""
        self.m.node_array("OUT", GENERATE_MOST)
        self.m.node_array("STOP", STOPS_MOST)
        self._stops = None

    def generate(self, token, pos, n, stop=(), sampler=None, lag=0):
        """Up to `n` tokens from `token` at `pos`, each the next one's input, in one program on the card — ending early
        after one of `stop`. Greedy, or drawn by the card's sampler `sampler` (a `session.Sampler`). `lag`: how far the
        tokens' rotary positions are behind their places (a family whose program reads LAG). Answers the tokens."""
        if getattr(self, "HEADLESS", False):
            self._needs_head()
        n = min(n, GENERATE_MOST, self.max_context - pos)
        if n <= 0:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        self._watch(stop)
        if lag != getattr(self, "_lag", 0):
            self.m.must("(begin (set! LAG %d) 0)" % lag, "the rotary's lag")
            self._lag = lag
        if sampler is None:
            call = "(generate %d %d %d (< 1 0) 0 0 0 1.0 1.0 1 1.0 0)" % (token, pos, n)
        else:
            reads = sampler.repetition_penalty != 1.0
            call = "(generate %d %d %d (< 0 1) %d %d %d %r %r %d %r %d)" % (
                token, pos, n, 1 if reads else 0, min(pos, sampling.SEEN) if reads else 0, pos % sampling.SEEN,
                float(sampler.repetition_penalty), 1.0 / sampler.temperature, sampler.top_k, min(1.0, float(sampler.top_p)),
                sampler.seed)
        made = self._run(call, "the answer from %d" % pos)
        return [self.m.must("(nth OUT %d)" % j, "a token made") for j in range(made)]

    def _watch(self, stop):
        """The end tokens `generate` watches for, into STOP — when they are not the ones there already. Every slot is
        written, the ones past the end tokens `NO_TOKEN`, and STOPS stays as bound: a view (▶ `Machine.view`) holds its
        bindings as they stood when it was made, and a `set!` after it does not reach it, while STOP is one array that
        every view shares."""
        stop = tuple(sorted(stop))[:STOPS_MOST]
        if stop == self._stops:
            return
        slots = stop + (NO_TOKEN,) * (STOPS_MOST - len(stop))
        self.m.must("(begin %s 0)" % " ".join("(sys__node_array__set STOP %d %d)" % (j, t) for j, t in enumerate(slots)),
                    "the end tokens")
        self._stops = stop

    def _run(self, text, what):
        """A program's answer."""
        return self.m.must(text, what)
