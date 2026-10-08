"""A SAMPLER ON THE CARD — the next token drawn where the logits are, as one more procedure of the model's program.

The host's sampler (▶ `session.sampler`) reads every logit back and draws in numpy, ~2.8 ms a token on the 35B; this one is
nn's words after the head — `head_sample`, in `lisp/sampling.lisp` — and the token is the only thing that comes back.
Each model's program defines `(logits)` and binds VOCAB; this module makes the sampler's buffers, fills `seen` and writes
the call. `seen` holds the last `SEEN` tokens a position each, as the host's `seen[-256:]`; it is written only when the
penalty reads it. The draw's random number is `seed` and `pos` hashed, so a seed replays a conversation exactly.
"""

SEEN = 256          # the recent tokens the repetition penalty reads — the host sampler's window
K_MOST = 64         # nn's top-k bound, NN__VECTOR__TOP_K_MAX


def buffers(m):
    """The sampler's names: the recent tokens, the picks and their weights. Beside the model's `logits_h` and `res`."""
    m.node_array("seen", SEEN)
    m.node_array("sample_picks", K_MOST)
    m.buffer(2 * K_MOST, "sample_weights")


def remember(m, tokens):
    """The recent tokens of a conversation `tokens` long into `seen`, each at its position's slot — what a turn does first
    when its sampler's penalty reads them. Sixty-four to a program: a program is one form, and a form must fit a chunk."""
    first = max(0, len(tokens) - SEEN)
    sets = ["(sys__node_array__set seen %d %d)" % (p % SEEN, tokens[p]) for p in range(first, len(tokens))]
    for at in range(0, len(sets), 64):
        m.must("(begin %s 0)" % " ".join(sets[at:at + 64]), "the recent tokens")


def head(s, pos, token=None):
    """The head that draws with sampler `s` after position `pos`; `token`, the one fed at `pos`, goes into `seen` first
    when the penalty reads it (a prompt's were remembered before it)."""
    reads = s.repetition_penalty != 1.0
    note = "(sys__node_array__set seen %d %d) " % (pos % SEEN, token) if reads and token is not None else ""
    return note + "(head_sample %d %r %r %d %r %d %d)" % (min(pos + 1, SEEN) if reads else 0, float(s.repetition_penalty),
                                                       1.0 / s.temperature, s.top_k, min(1.0, float(s.top_p)), s.seed, pos)
