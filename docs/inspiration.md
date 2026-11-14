# What inspired SiLVANN

SiLVANN's design did not fall out of a spec. Several of its choices started as intuitions about what a transformer
is "really doing", and those intuitions still explain why the engine looks the way it does. One of them comes
first, and the others follow from it: that meaning is not stored in words but negotiated between them. Read the
physics and the philosophy here as lenses, not mechanism: the engine runs ordinary linear algebra on ordinary
silicon. The concepts it is built on are in [`concepts.md`](concepts.md).

## Meaning as a negotiation

Post-structuralism holds that meaning is not inside a word. Saussure's sign has its value only by its differences
from the signs around it; Derrida's *différance* says meaning is never present in the sign at all, only deferred
along the chain of signs that follow it; and for Barthes and reader-response criticism a text has no single meaning
until a reader actualises one of its many. Meaning, on this view, is a negotiation: between a word and its
neighbours, and between a text and whoever reads it.

The rest of this page comes from that lens. If meaning lives in relations, the operation that matters is the one
that relates tokens to each other: attention. Before a word is generated its meaning is not yet one thing — the
up-projection inside an MLP or an expert spreads a token across a wide hidden space, many senses active at once.
"Bank" is, for the moment, river *and* institution *and* tilt. Attention, conditioned on the context, collapses it:
context selects one meaning, and a single token is emitted, like a measurement narrowing a wide front of candidates
to a point.

The collapse is never final. A later token can reach back and re-collapse an earlier one: "…along the river" three
words on, and "bank" resolves. That is Derrida's deferral, computed, and it lives in the keys and values the model
re-reads every step — which is why how they are kept, the tiers of the cache and what is held at full precision or
compressed, matters as much as the arithmetic that reads them. A language model, seen this way, does not retrieve
meaning: it produces it, in context, at the moment of generation.

## The map people agree on

People do not meet reality alone. They negotiate a version of it between themselves — what matters, what things are
called, what causes what — and find their way through the world by it. For most purposes that map is the territory:
it is what people act on. Baudrillard opens *Simulacra and Simulation* (1981) with Borges's fable of a map as large
as the empire it charts, and turns it round: now the map precedes the territory, and engenders it.

An agreed map does more than describe, because what people act on becomes true. The Cybernetic Culture Research
Unit (the CCRU, at the University of Warwick in the 1990s) called such fictions *hyperstitions*: ideas that bring
about the reality they describe. A currency is worth what everyone agrees it is worth; a run on a bank happens
because enough people believe it will.

Seen as geometry, the real has more dimensions than any agreement keeps. An agreement is a projection of it onto a
lower-dimensional, constrained surface — a manifold — that the people who hold it share, and different groups hold
different projections of the same real. People live in these projections as much as in the world they project, and
so, it turns out, do models.

## Pain and the loss

A map holds until the territory bites back. Byung-Chul Han argues in *The Palliative Society* (2020) that a society
which numbs every pain loses its hold on the real, because the real is felt through its resistance, and resistance
hurts. Pain is the data point a map cannot ignore, because ignoring it has a cost.

Training has the same structure. Its loss is that cost: a prediction the text contradicts is penalised, and the
weights move until it is not.

What training reads is not the territory but the maps: each text one person's or one group's projection of the
real, written from where they stood. From them it reconstructs an object the way a CT scanner reconstructs a body
from flat X-ray images, each taken from its own angle. No single projection shows the object; enough of them, from
enough angles, pin it down. Machine learning's manifold hypothesis — that real data lies near a low-dimensional
surface inside its high-dimensional space — reads, through this lens, as saying that for language the surface a
model learns is the one the agreements made.

The weights are that reconstruction, frozen when training stops: a frozen hyperstition, the maps people negotiated,
compressed into a geometry. It is what is meant when a model is said to carry the views of its training
([`local_ai.md`](local_ai.md#as-a-dipl-o-mat)).

## The negotiation reopens in a conversation

Once training stops, nothing bites the weights: running a model computes no loss, and the model cannot renegotiate
its projection on its own. It can do so only inside a conversation, where the context is a map negotiated between
it and the person it talks with, each token re-reading the others — the negotiation of the first section, held open
for as long as the conversation lasts. That is why the conversation, and not only the model, is what the engine is
built around: its context kept as long as the model allows, its cache kept in tiers.

A negotiation between two parties can settle on a map nothing outside them checks. Tools give the conversation a
point of contact outside the interlocutor: a file read, a command run, a page fetched — another projection of the
territory, one that is neither party's. That is what SiLVANN's REPL is for ([`tools.md`](tools.md#the-repl)).

What a conversation finds does not reach the weights on its own. A LoRA adapter is how it can: a small
renegotiation of the frozen agreement, which can be trained from what conversations found and swapped in and out
between turns, the base's own projection unchanged beneath it
([`directions.md`](directions.md#2-choosing-an-adapter-at-the-end-of-a-sentence)).

## The KV cache as a tape

A register machine with an unbounded, randomly addressable memory is a Turing machine; the only question is where
the tape is. In an autoregressive model the tape is the context itself, stored as the KV cache:

| a Turing machine | an autoregressive language model |
|---|---|
| the tape | the cached keys and values — the context the model attends over |
| the read head | attention, and it is random-access: any step may read any past cell |
| writing a cell | the autoregressive step appending the new token's keys and values |
| the transition function | one decoding step — the fixed weights — applied over and over |

Run that loop with enough tape and you have, in principle, a universal computer. Transformers are Turing-complete
given unbounded precision and steps (Pérez et al., 2019), and a language model coupled to an external read/write
memory can simulate a universal Turing machine (Schuurmans, 2023). Chain-of-thought is the same observation from
the practitioner's side: letting the model write intermediate tokens and read them back hands the loop a working
tape.

Two honest caveats. A real deployment has a finite context and finite precision, so strictly it is a very large
finite-state machine; universality is the statement as the tape grows without bound. And attention's cache is
append-only — a past cell is never overwritten — so it is most precisely an unbounded log with random read;
overwriting is simulated by appending tokens that supersede earlier ones.

## Why the angles have to stay many

A CT scanner reconstructs nothing from one angle. However sharp a single projection is, it flattens the body along
the line it was taken from: it is the difference between projections, taken from many places, that pins the object
down. The same holds for the maps a model is trained on. Nick Land, the CCRU's best-known voice, imagines the
opposite: "Meltdown" (1994) begins its story with the Earth "captured by a technocapital singularity", one process
that takes every other into itself. Through this lens a singularity is a scanner that keeps a single angle. His own
later politics, *The Dark Enlightenment* (2012), split where "Meltdown" merges: sovereign corporations competing for
residents who may leave but not speak — exit without voice — and the residents who leave, in the sociologist Roger
Burrows's reading, "quickly form themselves into, often racially based, microstates" ("On Neoreaction", *The
Sociological Review*, 2018). It is the same lens turned inside out: many projections, each kept in a room of its own,
and nothing that reconstructs across them. Rooms need not stay sealed. A negotiation is where the projections are put
back together, each side's map tested against the other's until both stand on the same ground — the work a model each
side runs for itself can do ([`local_ai.md`](local_ai.md#as-a-dipl-o-mat)).

Many angles is not the same as every angle weighing the same. The margins are often the first to feel what the
centre has not yet registered — the first alarm — but a viewpoint earns its weight by what it predicts, and by
whether the people who hold it can make it work: an island that stays afloat, not only a refusal of the mainland.
Given what has happened, which frame accounts for it best is Bayes' question, P(frame | fact) = P(fact | frame) ·
P(frame) / P(fact), where P(fact) is how expected the fact was over all the frames together: each fact the territory
delivers raises the frames that expected it more than the rest did and sinks the ones it bit. A CT scanner's
statistical reconstruction does the same with its rays, weighing each by how reliable its measurement is, and training
does it with its loss. The angles have to stay many so that there is something to weigh; the weighing
is what turns them into an object.

A bubble's frame is weighed by the same facts. The elites of each are tasked with crafting a cohesive narrative for
their own public while keeping it in touch with the real, and they can lose in two ways. They can offer a bad system,
and lose their members to bubbles that work better. Or they can read the real badly and become a colony of lemmings:
inwardly, in behaviour that destroys it; outwardly, in a stance its neighbours cannot live beside, until the conflict
is real — with the risk of being conquered and destroyed — or in any mix of the two. The territory delivers the fact
that sinks the frame either way; what a bubble chooses is whether it learns it at its own desk or on the field.

Machine learning has measured what a single angle does. Models trained on what earlier models generated lose the
tails of the distribution first — the rare, the improbable — and generation by generation narrow towards a blander
copy of what they began with (Shumailov et al., "AI models collapse when trained on recursively generated data",
*Nature*, 2024). A reconstruction fed its own projections does not converge on the real; it forgets it. What keeps
it honest is a steady supply of independent angles: people in their own places, under their own frictions, writing
from where they stand — and the models they run themselves, answering to them rather than to one centre. Those are
not a preference but the condition for the reconstruction to stay alive.

## A speculative aside

Strip away the metaphors and one bet remains: that intelligence may be less biological magic and more a matter of
navigating high-dimensional semantic geometry well — and that the geometry can be run efficiently, on hardware an
individual can afford. That is a personal hypothesis, flagged as exactly that. Nothing in the engine depends on it
being true; the code stands on its own.
