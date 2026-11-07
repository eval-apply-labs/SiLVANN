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


def _state_read(m, length):
    """A conversation's state off the card, as one card holds it — a model over several cards stitches its cards' shares
    (▶ qwen3_5's `state_read`)."""
    if hasattr(m, "state_read"):
        return m.state_read(length)
    return [m.m.read(at, n) for at, n in m.state_regions(length) if n]


def state_bytes(m, length):
    """What a conversation of `length` positions holds of state, every card's share of it — none where the model keeps a
    conversation put away as its tokens (`PARKS` false: a ring's, a GLM group's)."""
    if not getattr(m, "PARKS", True):
        return 0
    return sum(n for _, n in m.state_regions(length)) * getattr(m, "CARDS", 1)


def _state_write(m, length, blobs):
    if hasattr(m, "state_write"):
        return m.state_write(length, blobs)
    live = [(at, n) for at, n in m.state_regions(length) if n]
    for (at, _), blob in zip(live, blobs):
        m.m.write(at, blob)


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
        # its pictures: where each one's pads begin in `tokens`, its grid, and the tower's rows that stand for them
        #   (halves, as the card takes them) — kept so a conversation brought back without its state runs them again
        self.images = []
        self.lag = 0                                # how far its next position is behind its length (▶ vision.positions)

    def matches(self, identity):
        return all(self.identity.get(k) == identity.get(k) for k in ("model", "packing", "architecture"))

    # ── on disk: a header naming what it belongs to, the tokens, and the state when it has one ─────────────────
    def save(self, path, with_state=True):
        header = dict(id=self.id, identity=self.identity, lora=self.lora, length=self.length,
                      has_state=bool(with_state and self.state is not None), lag=self.lag,
                      images=[dict(at=im["at"], grid=list(im["grid"])) for im in self.images])
        arrays = dict(header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
                      tokens=np.asarray(self.tokens, dtype=np.int64))
        for i, im in enumerate(self.images):
            arrays["image%d" % i] = np.asarray(im["rows"], dtype=np.float16)
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
        c.images = [dict(at=im["at"], grid=tuple(im["grid"]), rows=z["image%d" % i]) for i, im in enumerate(h.get("images", []))]
        if h["has_state"]:
            c.state = [bytes(z["s%d" % i]) for i in range(len([k for k in z.files if k.startswith("s")]))]
            c.length = h["length"]
            c.lag = h.get("lag", 0)
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
        if self.active is not None and getattr(m, "PARKS", True):    # the one on the card is put away, its state read off
            self.active.state = _state_read(m, self.active.length)
        elif self.active is not None:                 # a state that cannot be read off (a ring's): its tokens read again
            self.active.state, self.active.length, self.active.lag = None, 0, 0
        self.stats["park_seconds"] += time.time() - t0
        want = c.lora["name"] if c.lora else None
        if want != m.active_lora:
            self.stats["lora_seconds"] += m.load_lora(want)
            self.stats["lora_swaps"] += 1
        if c.lora is not None:
            c.lora = m.lora_identity()                # the weights it runs under, recorded
        if c.state is not None and getattr(m, "PARKS", True):
            _state_write(m, c.length, c.state)
            c.state = None
        else:
            m.reset()                                 # without its state, every token it has seen is run again — or with
            c.state = None                            #   one this model cannot take back (a GLM group: ▶ glm5's PARKS)
            c.length = 0
            c.lag = 0
        self.active = c
        self.stats["switches"] += 1

    # ── turns ───────────────────────────────────────────────────────────────────────────────────────────────
    def turn(self, c, new_tokens, max_new, sampler=None, stop=(), on_token=None, images=()):
        """Start a turn: `new_tokens` into the conversation, then up to `max_new` tokens of answer. Answers an object
        whose `.wait()` gives the answer's tokens. `images`: the pictures in `new_tokens`, each (where its pads begin
        in `new_tokens`, a `vision.Image`). ⛔ Raises `Busy` while another turn has the card."""
        with self._lock:
            if self._running is not None and not self._running.done.is_set():
                raise Busy("a turn is on the card")
            t = _Turn(c, list(new_tokens), max_new, sampler, set(stop), on_token)
            t.images = list(images)
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
        self._activate(t.cartridge)
        tok, how = self._prompt(t)
        return self._answer(t, tok, how)

    def _prompt(self, t):
        """⛳ `tokens[length:]` ARE SEEN AND NOT YET RUN — the answer's last token, or a whole conversation brought
        back without its state — so a turn runs them first, then its own new tokens. Answers the answer's first token and
        how the rest is drawn: (greedy, drawn on the card, the positions' lag)."""
        m, c = self.model, t.cartridge
        pending = c.tokens[c.length:] + t.new_tokens
        if not pending:
            raise Refused("a turn needs a token to run")
        if c.length + len(pending) + t.max_new > m.max_context:
            raise Refused("the conversation would pass the context of %d" % m.max_context)
        for offset, im in t.images:
            c.images.append(dict(at=len(c.tokens) + offset, grid=im.grid, rows=im.rows.astype(np.float16)))
        c.tokens += t.new_tokens
        greedy = t.sampler is None or (t.sampler.temperature <= 0 and t.sampler.repetition_penalty == 1.0)
        # ⭐ DRAWN ON THE CARD where the model has the procedure and the sampler fits it — only the token comes back
        card = not greedy and t.sampler.card(m)
        if card and t.sampler.repetition_penalty != 1.0:
            m.remember(c.tokens) if hasattr(m, "remember") else sampling.remember(m.m, c.tokens)
        # ⭐ PICTURES: their pads' rows are the tower's. A Qwen model's positions from the first picture on are M-RoPE's —
        #   its three a token, and a token's position no longer its place (▶ vision.positions), `lag` carrying the
        #   difference; GLM's tokens keep their places (▶ vision_glm.py), and only the rows go in
        shown = [im for im in c.images if im["at"] >= c.length]
        pictures = bool(shown) or c.lag > 0
        mrope = False
        if len(pending) > 1 or pictures:              # a prompt: its rows at once, where the model has the words
            last = c.length + len(pending) - 1
            extra = {}
            if pictures:
                v = m.vision()
                rows = {}
                for im in shown:
                    for j in range(len(im["rows"])):
                        rows[im["at"] - c.length + j] = im["rows"][j]
                extra = dict(rows=rows)
                mrope = getattr(v, "mrope", True)
                if mrope:
                    from .vision import positions
                    pos, after = positions(pending, [(im["at"] - c.length, im["grid"]) for im in shown], v.pads, v.merge,
                                           start=c.length - c.lag)
                    extra["positions"] = pos
            out = m.prefill(pending, c.length, greedy=greedy, sample=sampling.head(t.sampler, last) if card else None,
                            **extra)
            c.length += len(pending)
            if mrope:
                c.lag = c.length - after
        else:
            out = m.step(pending[0], c.length, greedy=greedy,
                         sample=sampling.head(t.sampler, c.length, pending[0]) if card else None)
            c.length += 1
        return (out if greedy or card else t.sampler(m.logits(), c.tokens)), (greedy, card, c.lag)

    def _answer(self, t, tok, how):
        """The answer from its first token `tok`, a token at a time, until an end token, its length or a cancel."""
        m, c, (greedy, card, lag) = self.model, t.cartridge, how
        lag = dict(lag=lag) if lag else {}
        # ⭐ THE ANSWER GENERATED ON THE CARD where the model has the loop and the token is chosen there — `GENERATE_CALL`
        #   tokens a call, so a stream sees them in that many at a time and a cancel waits for at most that many
        on_card = getattr(m, "GENERATES", False) and (greedy or card)
        answer = []
        while True:
            answer.append(tok)
            c.tokens.append(tok)
            if t.on_token:
                t.on_token(tok)
            if tok in t.stop or len(answer) >= t.max_new or t.cancelled:
                return answer
            if on_card:
                made = m.generate(tok, c.length, min(GENERATE_CALL, t.max_new - len(answer)), t.stop,
                                  t.sampler if card else None, **lag)
                c.length += len(made)                 # each made token's input was run: `tok`, then all but the last
                for x in made[:-1]:                   # ⛳ none of them an end token — the loop ends at one
                    answer.append(x)
                    c.tokens.append(x)
                    if t.on_token:
                        t.on_token(x)
                tok = made[-1]
                continue
            out = m.step(tok, c.length, greedy=greedy, sample=sampling.head(t.sampler, c.length, tok) if card else None,
                         **lag)
            c.length += 1
            tok = out if greedy or card else t.sampler(m.logits(), c.tokens)

    def card_lost(self):
        """The machine went down under the conversations (a panic, ▶ machine's `Panic`): what the card held is gone, so
        every conversation whose state was on it is read again from its tokens when it next runs, and none is on it. One
        whose state was read off keeps it."""
        for c in self.cartridges.values():
            if c.state is None:
                c.length, c.lag = 0, 0
        self._holds_nothing()

    def _holds_nothing(self):
        self.active = None

    def save(self, c, path, with_state=True):
        """A cartridge to disk; the one on the card has its state read first, and stays on the card — where the model's
        state can be read off (a ring's cannot: its tokens are read again)."""
        with_state = with_state and getattr(self.model, "PARKS", True)
        if c is self.active and with_state:
            c.state = _state_read(self.model, c.length)
            c.save(path, with_state)
            c.state = None
        else:
            c.save(path, with_state)

    def close(self):
        if self._running is not None:
            self._running.done.wait()


class SeatedRuntime(Runtime):
    """⭐ SEATS (NN-147): a model's conversations held on the card at once, one a seat (the model's `SEATS`), their turns
    run together. ⚖ 2026-10-30: *"4 conversation on the same card, two active during inference ad other two being rotated
    out so that the queue of conversations can continue"* · *"lets keep that a [prompt] takes the whole card as it does
    now"*.
    ```
      a turn       takes a seat no turn is running on: its conversation's own where it sits there, else an empty one,
                   else the one least recently used — whose conversation is read off into RAM first (its state, as a
                   cartridge is parked). Busy where every seat is answering — and the conversations told so wait their
                   turn: a seat that frees is theirs, oldest first, before a seated conversation's next turn (⚖ *"the
                   first one to end will evict and pass the ball to the waiting one"*); one that stops asking for
                   WAIT_FORGET seconds is forgotten
      a prompt     runs by itself, the whole card's, before the next step of any answer
      answers      two seats answering are stepped as a pair (▶ the model's `pair_step`: the card on one's attention while
                   the CPUs compute the other's experts), the two stepped longest ago; one answering alone a position at
                   a time
    ```
    One thread runs the model: every turn's work is its, in the order above. ⭐ A MODEL WHOSE SEATS STEP ON THEIR OWN
    (its `ALONE`, ⚖ 2026-11-04 the resident machine) has a thread a seat instead, each running its seat's turns — its
    prompt, then its answer a position at a time — so no seat waits for another's position: their steps interleave on
    their card's queue (▶ the model's chain), and a prompt holds its card's other seats only between its chunks."""
    WAIT_FORGET = 15.0                               # three of the callers' 5 s retries

    def __init__(self, model):
        super().__init__(model)
        n = int(getattr(model, "SEATS", 1))
        if n < 2:
            raise Refused("a seated runtime takes a model with two seats or more, not %d" % n)
        if not getattr(model, "PARKS", True):
            raise Refused("seats take a model whose conversation's state can be read off the card")
        self.held = [None] * n                       # the conversation each seat holds
        self.used = [0.0] * n                        # when a turn last ran on each seat
        self.stepped = [0.0] * n                     # when each seat's answer last took a step
        self.running = {}                            # seat -> its turn
        self.waiting = {}                            # a conversation told Busy -> [when first, when last], oldest first
        self._work = threading.Condition(self._lock)
        if getattr(model, "ALONE", False):
            for seat in range(n):
                threading.Thread(target=self._loop, args=(seat,), daemon=True).start()
        else:
            threading.Thread(target=self._loop, daemon=True).start()

    # the server reads one conversation on the card; with seats every seat's is, and none of them is "the" one
    @property
    def active(self):
        return None

    @active.setter
    def active(self, value):
        if value is not None:
            raise Refused("a seated runtime's conversations are its seats'")

    def seat_of(self, c):
        return next((k for k, h in enumerate(self.held) if h is c), None)

    def _holds_nothing(self):
        with self._lock:
            self.held = [None] * len(self.held)

    def forget(self, c):
        """`c` put away: what its seat holds is no one's now."""
        with self._lock:
            k = self.seat_of(c)
            if k is not None and k not in self.running:
                self.held[k] = None

    def turn(self, c, new_tokens, max_new, sampler=None, stop=(), on_token=None, images=()):
        """A turn on a seat (▶ the class); answers its object, whose `.wait()` gives the answer. ⛔ `Busy` where every seat
        is answering, or where `c` runs under another LoRA than the one the answering seats run under."""
        with self._lock:
            if self._holding():
                raise Busy("the card is lent")
            k = self.seat_of(c)
            if k is not None and k in self.running:
                raise Busy("a turn of this conversation is on the card")
            want = c.lora["name"] if c.lora else None
            if want != self.model.active_lora and self.running:
                raise Busy("the seats answering run under another LoRA")
            now = time.time()
            for cid in [cid for cid, (_, last) in self.waiting.items() if now - last > self.WAIT_FORGET]:
                del self.waiting[cid]
            free = [j for j in range(len(self.held)) if j not in self.running]
            ahead = [cid for cid in self.waiting if cid != c.id]
            if c.id in self.waiting:                 # waiting: the free seats go to the ones waiting longer first
                ahead = ahead[:list(self.waiting).index(c.id)]
            if len(ahead) >= len(free):
                self.waiting.setdefault(c.id, [now, now])[1] = now
                raise Busy("every seat is answering" if not free else "the conversations waiting take the free seats")
            self.waiting.pop(c.id, None)
            if k is None:
                empty = [j for j in free if self.held[j] is None]
                k = empty[0] if empty else min(free, key=lambda j: self.used[j])
            t = _Turn(c, list(new_tokens), max_new, sampler, set(stop), on_token)
            t.images, t.seat, t.phase = list(images), k, "new"
            self.running[k] = t
            self._running = t
            self._work.notify_all()
        return t

    def _holding(self):
        """Whether the server holds every turn off (its `_Holding` in `_running`: the card lent to another model)."""
        h = self._running
        return h is not None and not hasattr(h, "seat") and not h.done.is_set()

    def busy(self):
        return bool(self.running) or self._holding()

    def put_down(self):
        """Before the model leaves the card (lent to another): every seat's answer finished, and every seat's conversation
        read off into RAM — each written back when its next turn takes a seat."""
        while self.running:
            time.sleep(0.05)
        with self._lock:
            for k, c in enumerate(self.held):
                if c is not None:
                    c.state = self._seat_read(k, c.length)
                    self.held[k] = None

    def save(self, c, path, with_state=True):
        """A cartridge to disk; one a seat holds has its state read off that seat, and stays there."""
        with_state = with_state and getattr(self.model, "PARKS", True)
        k = self.seat_of(c)
        if k is not None and with_state:
            with self._lock:
                if k in self.running:
                    raise Busy("its turn is on the card")
                c.state = self._seat_read(k, c.length)
            c.save(path, with_state)
            c.state = None
        else:
            c.save(path, with_state)

    def close(self):
        while self.running:
            time.sleep(0.05)

    # ── the seats' state ──────────────────────────────────────────────────────────────────────────────────────
    def _seat_read(self, k, length):
        """Seat `k`'s conversation's state: its regions on the card — and what else the model keeps of it, where it reads
        it itself (DeepSeek's tokens, which its n-gram memory hashes: ▶ its `seat_state_read`)."""
        m = self.model
        if hasattr(m, "seat_state_read"):
            return m.seat_state_read(length, k)
        return [m.m.read(at, n) for at, n in m.seat_state_regions(length, k) if n]

    def _seat_write(self, k, length, blobs):
        m = self.model
        if hasattr(m, "seat_state_write"):
            return m.seat_state_write(length, blobs, k)
        for (at, _), blob in zip([(at, n) for at, n in m.seat_state_regions(length, k) if n], blobs):
            m.m.write(at, blob)

    def _seat(self, k, c):
        """Seat `k` given to `c`: the conversation it held read off into RAM, `c`'s written in — or, saved without its state,
        its seat reset and its tokens run again."""
        m, old = self.model, self.held[k]
        want = c.lora["name"] if c.lora else None
        if want != m.active_lora:                    # the model's one LoRA: every seat's turns run under it (▶ `turn`)
            self.stats["lora_seconds"] += m.load_lora(want)
            self.stats["lora_swaps"] += 1
        if c.lora is not None:
            c.lora = m.lora_identity()
        if old is c:
            return
        t0 = time.time()
        if old is not None:
            old.state = self._seat_read(k, old.length)
        self.held[k] = None
        self.stats["park_seconds"] += time.time() - t0
        with m.on_seat(k):
            if c.state is not None:
                self._seat_write(k, c.length, c.state)
            else:
                m.reset()
                c.length, c.lag = 0, 0
        c.state = None
        self.held[k] = c
        self.stats["switches"] += 1

    # ── the model's thread ────────────────────────────────────────────────────────────────────────────────────
    def _loop(self, seat=None):
        """The model's thread — or seat `seat`'s, stepping that seat alone."""
        def mine(k):
            return seat is None or k == seat
        while True:
            with self._lock:
                while not any(mine(k) for k in self.running):
                    self._work.wait()
                new = [(k, t) for k, t in self.running.items() if mine(k) and t.phase == "new"]
                answering = [(k, t) for k, t in self.running.items() if mine(k) and t.phase == "answer"]
            if new:                                  # a prompt takes the card by itself, before any answer's next step
                k, t = new[0]
                try:
                    self._seat(k, t.cartridge)
                    with self.model.on_seat(k):
                        tok, t.how = self._prompt(t)
                    t.phase, t.answer = "answer", []
                    self._emit(k, t, tok)
                except Exception as exc:             # a turn that fails answers its failure
                    self._end(k, t, exc)
                continue
            if len(answering) >= 2:
                group = sorted(answering, key=lambda kt: self.stepped[kt[0]])[:2]
                try:
                    self._pair(group)
                except Exception as exc:
                    for k, t in group:
                        self._end(k, t, exc)
            elif answering:
                k, t = answering[0]
                try:
                    self._single(k, t)
                except Exception as exc:
                    self._end(k, t, exc)

    def _head(self, t, tok):
        """A seat's head for its next token: the greedy one, the card's draw, or the host's over the logits read back."""
        greedy, card, _ = t.how
        if greedy:
            return "(head)"
        if card:
            return sampling.head(t.sampler, t.cartridge.length, tok)
        return lambda logits: t.sampler(logits, t.cartridge.tokens)

    def _single(self, k, t):
        m, c, tok = self.model, t.cartridge, t.answer[-1]
        h = self._head(t, tok)
        with m.on_seat(k):
            out = m.step(tok, c.length, greedy=h == "(head)", sample=h if isinstance(h, str) and h != "(head)" else None)
            if not isinstance(h, str):
                out = h(m.logits())
        c.length += 1
        self.stepped[k] = time.time()
        self._emit(k, t, out)

    def _pair(self, pair):
        (ka, ta), (kb, tb) = pair
        toks = (ta.answer[-1], tb.answer[-1])
        heads = (self._head(ta, toks[0]), self._head(tb, toks[1]))
        outs = self.model.pair_step(toks, (ta.cartridge.length, tb.cartridge.length), heads=heads, seats=(ka, kb))
        now = time.time()
        for (k, t), out in zip(pair, outs):
            t.cartridge.length += 1
            self.stepped[k] = now
            self._emit(k, t, out)

    def _emit(self, k, t, tok):
        """A token of `t`'s answer: in its conversation, to its stream — and the turn ended at an end token, its length or a
        cancel."""
        c = t.cartridge
        t.answer.append(tok)
        c.tokens.append(tok)
        if t.on_token:
            t.on_token(tok)
        if tok in t.stop or len(t.answer) >= t.max_new or t.cancelled:
            self._end(k, t)

    def _end(self, k, t, error=None):
        with self._lock:
            self.running.pop(k, None)
            self.used[k] = time.time()
            if error is not None and self.held[k] is t.cartridge:
                self.held[k] = None                  # its seat's state unknown after a failure: written again when it returns
                t.cartridge.state, t.cartridge.length, t.cartridge.lag = None, 0, 0
        t.result, t.error = (None if error is not None else t.answer), error
        t.done.set()


class _Turn:
    def __init__(self, cartridge, new_tokens, max_new, sampler, stop, on_token):
        self.cartridge, self.new_tokens, self.max_new = cartridge, new_tokens, max_new
        self.sampler, self.stop, self.on_token = sampler, stop, on_token
        self.done, self.result, self.error, self.cancelled = threading.Event(), None, None, False
        self.images = []

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
