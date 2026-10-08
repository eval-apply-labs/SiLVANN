"""CONVERSATIONS — cartridges, one on the card at a time, the others parked until their turn is asked for.

⚖ *"i do want multi conversations tho, one cartridge at a time and the other cartridges wait in a queue so multi users
with max context is possible but the compute is time shared. the conversation cartridge also notes its model and lora
in its params so it refuses a load if the system is misaligned"* · *"i think it would be cool to have different agents
running on different loras"*.

A CARTRIDGE is a conversation: the model it belongs to, the LoRA it runs under, every token it has seen, and — while
it is not on the card — its state, the bytes the card had written for it. Every conversation can use the whole
context, because only the one being served occupies the card.
```
  activate(c)    the cartridge on the card is put away (its state read off the card), c's LoRA is written into the
                 slots if it is not the one there, and c's state written back — or, a cartridge saved without its
                 state, its tokens run again
  a turn         the new tokens run through, then the answer is generated, a token at a time
```
The SERVER hands turns to a `Runtime`. ⚖ *"keep the card until the response is completed and if someone asks return
that the system is busy"* — a turn has the card until its answer is done, and a turn asked for meanwhile is refused
with `Busy` at once. ⚖ *"retry in 5 s is what the silvann ui will do (or the requestor in general), it is the
convention and it is the caller that will do it"* — so the runtime keeps no queue and never retries.
"""
import json
import threading
import time
import uuid

import numpy as np

from . import sampling
from .model_folder import Refused

GENERATE_CALL = 8            # the tokens of an answer one `generate` makes, where the model generates on the card


class Busy(Exception):
    """A turn is on the card; ask again later."""


class Cartridge:
    def __init__(self, identity, lora=None, cid=None):
        self.id = cid or uuid.uuid4().hex[:12]
        self.identity = dict(identity)             # the model: its source weights, how it was packed, its architecture
        self.lora = lora                            # the adapter it runs under: {name, weights} or None
        self.tokens = []                            # every token it has seen, prompt and answers
        self.length = 0                             # the positions whose state the card has written
        self.state = None                           # the state's bytes, while it is off the card

    def matches(self, identity):
        return all(self.identity.get(k) == identity.get(k) for k in ("model", "packing", "architecture"))

    # ── on disk: a header naming what it belongs to, the tokens, and the state when it has one ─────────────────
    def save(self, path, with_state=True):
        header = dict(id=self.id, identity=self.identity, lora=self.lora, length=self.length,
                      has_state=bool(with_state and self.state is not None))
        arrays = dict(header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
                      tokens=np.asarray(self.tokens, dtype=np.int64))
        if header["has_state"]:
            for i, blob in enumerate(self.state):
                arrays["s%d" % i] = np.frombuffer(blob, dtype=np.uint8)
        with open(path, "wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        h = json.loads(bytes(z["header"]).decode())
        c = cls(h["identity"], h["lora"], h["id"])
        c.tokens = [int(t) for t in z["tokens"]]
        if h["has_state"]:
            c.state = [bytes(z["s%d" % i]) for i in range(len([k for k in z.files if k.startswith("s")]))]
            c.length = h["length"]
        return c


class Runtime:
    """A model on a card and the conversations it serves, a turn at a time."""

    def __init__(self, model):
        self.model = model
        ident = dict(model.folder.identity())
        ident.pop("lora", None)
        self.identity = ident
        self.active = None
        self.cartridges = {}
        self.stats = dict(switches=0, lora_swaps=0, lora_seconds=0.0, park_seconds=0.0)
        self._lock = threading.Lock()
        self._running = None

    # ── cartridges ──────────────────────────────────────────────────────────────────────────────────────────
    def new_cartridge(self, lora=None):
        if lora is not None and lora not in self.model.folder.loras():
            raise Refused("this model has no LoRA %r (it has %s)" % (lora, self.model.folder.loras()))
        c = Cartridge(self.identity, None if lora is None else dict(name=lora))
        self.cartridges[c.id] = c
        return c

    def adopt(self, c):
        """A cartridge from disk. ⛔ Refused unless it belongs to this model, and its LoRA is one this model has with
        the same weights — a state computed under other weights would be read as if it were this model's."""
        if not c.matches(self.identity):
            raise Refused("the cartridge %s belongs to %s, and this machine runs %s" % (c.id, c.identity, self.identity))
        if c.lora is not None:
            if c.lora["name"] not in self.model.folder.loras():
                raise Refused("the cartridge %s runs under the LoRA %r, which this model does not have" % (c.id, c.lora["name"]))
            self.model._ensure(c.lora["name"])
            have = self.model._lora_cache[c.lora["name"]]["ident"]["weights"]
            if c.lora.get("weights") not in (None, have):
                raise Refused("the cartridge %s ran under %r with weights %s; this model's %r has %s"
                              % (c.id, c.lora["name"], c.lora["weights"], c.lora["name"], have))
        self.cartridges[c.id] = c
        return c

    def _activate(self, c):
        if self.active is c:
            return
        m = self.model
        t0 = time.time()
        if self.active is not None:                   # the one on the card is put away, its state read off
            regions = m.state_regions(self.active.length)
            self.active.state = [m.m.read(at, n) for at, n in regions if n]
        self.stats["park_seconds"] += time.time() - t0
        want = c.lora["name"] if c.lora else None
        if want != m.active_lora:
            self.stats["lora_seconds"] += m.load_lora(want)
            self.stats["lora_swaps"] += 1
        if c.lora is not None:
            c.lora = m.lora_identity()                # the weights it runs under, recorded
        if c.state is not None:
            live = [(at, n) for at, n in m.state_regions(c.length) if n]
            for (at, _), blob in zip(live, c.state):
                m.m.write(at, blob)
            c.state = None
        else:
            m.reset()                                 # without its state, every token it has seen is run again
            c.length = 0
        self.active = c
        self.stats["switches"] += 1

    # ── turns ───────────────────────────────────────────────────────────────────────────────────────────────
    def turn(self, c, new_tokens, max_new, sampler=None, stop=(), on_token=None):
        """Start a turn: `new_tokens` into the conversation, then up to `max_new` tokens of answer. Answers an object
        whose `.wait()` gives the answer's tokens. ⛔ Raises `Busy` while another turn has the card."""
        with self._lock:
            if self._running is not None and not self._running.done.is_set():
                raise Busy("a turn is on the card")
            t = _Turn(c, list(new_tokens), max_new, sampler, set(stop), on_token)
            self._running = t
        threading.Thread(target=self._serve, args=(t,), daemon=True).start()
        return t

    def busy(self):
        return self._running is not None and not self._running.done.is_set()

    def _serve(self, t):
        try:
            t.result = self._run(t)
        except Exception as exc:                      # a turn that fails answers its failure
            t.error = exc
        t.done.set()

    def _run(self, t):
        """⛳ `tokens[length:]` ARE SEEN AND NOT YET RUN — the answer's last token, or a whole conversation brought
        back without its state — so a turn runs them first, then its own new tokens, then answers."""
        m, c = self.model, t.cartridge
        self._activate(c)
        pending = c.tokens[c.length:] + t.new_tokens
        if not pending:
            raise Refused("a turn needs a token to run")
        if c.length + len(pending) + t.max_new > m.max_context:
            raise Refused("the conversation would pass the context of %d" % m.max_context)
        c.tokens += t.new_tokens
        greedy = t.sampler is None or (t.sampler.temperature <= 0 and t.sampler.repetition_penalty == 1.0)
        # ⭐ DRAWN ON THE CARD where the model has the procedure and the sampler fits it — only the token comes back
        card = not greedy and t.sampler.card(m)
        if card and t.sampler.repetition_penalty != 1.0:
            sampling.remember(m.m, c.tokens)
        if len(pending) > 1:                          # a prompt: its rows at once, where the model has the words
            last = c.length + len(pending) - 1
            out = m.prefill(pending, c.length, greedy=greedy, sample=sampling.head(t.sampler, last) if card else None)
            c.length += len(pending)
        else:
            out = m.step(pending[0], c.length, greedy=greedy,
                         sample=sampling.head(t.sampler, c.length, pending[0]) if card else None)
            c.length += 1
        # ⭐ THE ANSWER GENERATED ON THE CARD where the model has the loop and the token is chosen there — `GENERATE_CALL`
        #   tokens a call, so a stream sees them in that many at a time and a cancel waits for at most that many
        on_card = getattr(m, "GENERATES", False) and (greedy or card)
        answer = []
        tok = out if greedy or card else t.sampler(m.logits(), c.tokens)
        while True:
            answer.append(tok)
            c.tokens.append(tok)
            if t.on_token:
                t.on_token(tok)
            if tok in t.stop or len(answer) >= t.max_new or t.cancelled:
                return answer
            if on_card:
                made = m.generate(tok, c.length, min(GENERATE_CALL, t.max_new - len(answer)), t.stop,
                                  t.sampler if card else None)
                c.length += len(made)                 # each made token's input was run: `tok`, then all but the last
                for x in made[:-1]:                   # ⛳ none of them an end token — the loop ends at one
                    answer.append(x)
                    c.tokens.append(x)
                    if t.on_token:
                        t.on_token(x)
                tok = made[-1]
                continue
            out = m.step(tok, c.length, greedy=greedy, sample=sampling.head(t.sampler, c.length, tok) if card else None)
            c.length += 1
            tok = out if greedy or card else t.sampler(m.logits(), c.tokens)

    def save(self, c, path, with_state=True):
        """A cartridge to disk; the one on the card has its state read first, and stays on the card."""
        if c is self.active and with_state:
            c.state = [self.model.m.read(at, n) for at, n in self.model.state_regions(c.length) if n]
            c.save(path, with_state)
            c.state = None
        else:
            c.save(path, with_state)

    def close(self):
        if self._running is not None:
            self._running.done.wait()


class _Turn:
    def __init__(self, cartridge, new_tokens, max_new, sampler, stop, on_token):
        self.cartridge, self.new_tokens, self.max_new = cartridge, new_tokens, max_new
        self.sampler, self.stop, self.on_token = sampler, stop, on_token
        self.done, self.result, self.error, self.cancelled = threading.Event(), None, None, False

    def wait(self, timeout=None):
        self.done.wait(timeout)
        if self.error is not None:
            raise self.error
        return self.result


class Sampler:
    """Repetition penalty, temperature, top-k, then top-p. Drawn on the card when the model has `head_sample` and the
    sampler fits it — a temperature above 0, a top-k of 1 to `sampling.K_MOST` (▶ `sampling`) — and on the host over
    the logits the card left otherwise, or with `on_card=False`. `seed` replays the card's draws exactly; the host's
    are numpy's, so the two agree in distribution and not token for token."""

    def __init__(self, temperature=1.0, top_k=20, top_p=0.95, repetition_penalty=1.0, seed=None, on_card=True):
        self.temperature, self.top_k, self.top_p = float(temperature), int(top_k), float(top_p)
        self.repetition_penalty, self.on_card = float(repetition_penalty), on_card
        self.seed = int(seed) % (1 << 63) if seed is not None else int(np.random.SeedSequence().entropy) % (1 << 63)
        self.rng = np.random.default_rng(seed)

    def card(self, model):
        return (self.on_card and getattr(model, "SAMPLES_ON_CARD", False) and self.temperature > 0
                and 0 < self.top_k <= sampling.K_MOST and self.repetition_penalty > 0)

    def __call__(self, logits, seen):
        """The host's draw, over the logits read back."""
        lg = logits.astype(np.float64).copy()
        if self.repetition_penalty != 1.0 and seen:
            ids = np.unique(np.asarray(seen[-sampling.SEEN:]))
            lg[ids] = np.where(lg[ids] > 0, lg[ids] / self.repetition_penalty, lg[ids] * self.repetition_penalty)
        if self.temperature <= 0:
            return int(np.argmax(lg))
        lg /= self.temperature
        k = self.top_k
        top = np.argpartition(-lg, k)[:k] if 0 < k < lg.size else np.arange(lg.size)
        p = np.exp(lg[top] - lg[top].max())
        order = np.argsort(-p)
        top, p = top[order], p[order] / p.sum()
        keep = int(np.searchsorted(np.cumsum(p), self.top_p) + 1)
        top, p = top[:keep], p[:keep] / p[:keep].sum()
        return int(self.rng.choice(top, p=p))


def sampler(temperature=1.0, top_k=20, top_p=0.95, repetition_penalty=1.0, seed=None, on_card=True):
    """A sampler: ▶ `Sampler`."""
    return Sampler(temperature, top_k, top_p, repetition_penalty, seed, on_card)
