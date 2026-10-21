# Why AI on your own machine

SiLVANN runs a model on the machine in front of you: the conversation, and the files its tool reads, stay on that
machine. This page is the author's case for why that matters, why being able to do it is worth keeping, and what to
make of the bad actors who could run a model the same way.

## Work that cannot leave the room

Some professions are bound to keep what they hear to themselves. A therapist's or a psychiatrist's notes, and a lawyer's
case files and correspondence with a client, are confidential: by law, by privilege and by their profession's rules.
Sending them to a service on someone else's servers is a disclosure. Their rules may not allow it, and their clients
never agreed to it.

A model on their own machine lets them draft, summarise and search their own material without that disclosure, and
nothing leaves the room. It is not a guarantee of compliance on its own: what their rules allow is still theirs to judge.
It does remove the part that no rule can make safe, which is a copy of the client's words somewhere else.

## Beside the big models, not instead of them

A model you run at home is not a replacement for the large commercial models, least of all for writing code. They are
bigger, faster and better at most of what they do. A local model is something extra. It helps with the work those
services cannot be given, and it lets a product be built to keep working when they are out of reach.

Much of that work sits under a non-disclosure agreement: a client's code, a design not yet announced, a contract under
negotiation. A service can promise not to keep what it is sent, and the promise is a contract. A model on your own
machine enforces the secrecy by construction, because the material never leaves. When an NDA demands that nothing goes
out, only local AI can meet that demand on its own.

## The elephant in the room: bad actors

Anything that helps people helps bad actors too, and a model anyone can run is no exception. The question is where the
harm comes from, and what actually stops it.

### Weak systems, and the people who defend them

Much of what bad actors attack was built for a different threat model. Those systems assumed an attacker's time and skill
were scarce: a phone line that trusts a caller's voice, a website nobody had the time to audit, a process that checks a
document by looking at it. AI makes the attacker's time cheap, and those systems were not designed for that.

Keeping the tools from everyone does not fix this. Attackers do not ask permission, so it leaves them with the tools and
the defenders without. A model on the defender's own machine can read the code, the configurations and the procedures
they would never send anywhere else, and find the weak points first. Local models make those systems more resistant,
because the people who run them get the same help in private.

### Biological risk

The worry is that a model could help someone cause harm with biology. That risk is real, but it does not sit with a
person running a model at home. The hard parts of that harm are not words on a screen. They are specialised equipment,
controlled materials, laboratory skill, and the means to carry it out. The actors who can clear those barriers are
larger ones with access to specialist tools, and a model on a home PC is not what holds them back. SiLVANN runs models
their authors chose to publish, on ordinary hardware. It does not give a model abilities it did not have. Local hosting
is not what enables these actors; control of the specialist tools and materials is where that risk is met.

### Impersonation

Scammers have always pretended to be someone else: a bank, a relative, a boss. AI makes the voice and the face cheaper
to fake. A call can sound like your daughter, and a video call can show a colleague who is not there. Trying to spot
each fake is a race the fakes will keep winning. What works is a new way of recognising who you are dealing with, one
that leaves the impersonation nothing to work on:

- A call from a number you do not know goes to voicemail. If it matters, you call back.
- You call a business only on the number it publishes, such as its website or the back of your card, never on a number
  given in the call or the message.
- A family agrees on a word to ask for when someone calls in trouble.
- A person on a video call may not be real. A request for money, a password or access is confirmed another way.

None of this needs AI kept away from anyone. It closes the door the fake would have walked through.

## A change of habits, not for the worse

AI will change some of our habits, as the telephone and the internet did. That does not mean it has to change them for
the worse. Models that run on your own machine, from published weights and software anyone can build, keep the benefit
with the people who most need their work to stay private, and keep the defence in the hands of the defenders. That is
worth safeguarding, if we are smart about it.
