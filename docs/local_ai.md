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

### Backdoors

A backdoor is a weakness put into a product on purpose: a hidden account, a key that opens every copy, a way in that
only its makers are meant to know. It was always a bet that nobody else would find it. That bet is lost once large
models, and the state actors who run them, can reverse engineer compiled binaries: a backdoor shipped in a product is
then a backdoor your enemies can find and use.

So besides the ethics of mishandling the trust its users put in it, a company that plants a backdoor weakens its own
country's security and hands its enemies an easy target. A local actor cannot exploit such a weakness at scale. A state
actor can, and will keep it as leverage until the day it is useful.

The defence starts with the user. A single person with a local model has an expert beside them to understand the
threat: to read what a product does and what it reaches, and to work out how best to isolate it from attackers and to
mitigate a weakness until it is fixed. Containers and a "need to use" permission model — every app given what it needs
and nothing more ([below](#container-first)) — remove most of the attack surface.

That is not enough on its own. The industry has to acknowledge that the threat model is radically different now, and
patching vulnerabilities and offering mitigations should be of paramount importance to everyone who considers
themselves an IT professional.

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

## Authentication and anonymity

Impersonation points at a habit worth changing: how we prove who we are. A voice, a face or a password can be copied or
guessed. A signing key is something you hold rather than something you look or sound like, so a model cannot copy it
from a recording. Each person keeps a private key, on a phone or a small device of its own, and publishes the matching
public key. A message signed with the private key can only have come from its holder, and anyone can check it against
the public key. Logging in becomes signing a challenge instead of typing a password, and a payment, a contract or a
message from your bank carries a signature you can check, the way a website already carries a certificate.

It would also end the ever-growing pile of passwords. You can keep one key for everything, or a key for each service you
want kept apart (work and home, a bank and a forum), so that one cannot be linked to the other. Either way, no site
receives or stores a secret. The private key stays with you, and the one password left is the one that unlocks it on
your own device. Because you type it in one place and no server ever sees it, it can be long and hardened. A site
keeps only your public key and checks your signature on a challenge it chose, so a stolen copy of its database logs
nobody in. A careless implementation has no passwords to leak, and the system's owner has no password to hash, salt and
guard: nothing it stores can be turned into access. Logging in becomes safer and less painful at once. Passkeys already
work this way on many sites.

An electronic signature can then be the normal way to identify someone, without identifying anyone more than a task
needs. Proving you are over 18 does not need your name. An anonymous account, bought at a shop that checks your ID once
at the counter, the way a phone top-up card is sold today, can carry that one fact, signed by whoever issued it, and
nothing that leads back to you. Your full identity stays for the few things that need it.

Keys need somewhere to be looked up, and a way to be replaced when one is lost. A general ledger, kept by many parties
rather than one company, can hold that register: which public key belongs to which account, which keys were withdrawn,
which name points where. [go-ledger](https://github.com/eval-apply-labs/go-ledger), another Eval Apply Labs project, is
a small ledger of this kind. It binds names to accounts in a registry its validators agree on, and every entry is signed,
so a name can be checked without trusting a single company's database.

## Container first

A model that only answers can say something wrong. A model with tools can do something wrong: SiLVANN's REPL reads
files, runs commands and fetches from the network when it is allowed to, and an agent built on any model is the same.
Most of the harm such an app can do is decided before the model says a word, by what the process it runs in can reach.

So build and run AI apps container-first, and ask of each one what it can reach and what that would cost:

- **Its files.** A container sees only the folders it is given. A volume mounted read-only cannot be changed, and the
  rest of the machine is not there at all.
- **Its credentials.** Keys, tokens and logged-in sessions in its environment are the app's to use, and the model's to
  misuse. Keep them out of the image and out of the tool's reach, and give the app its own, limited to what it does.
- **Its network.** A container can have no network, or only the one service it needs. A model that cannot reach the
  internet cannot send anything to it.
- **The machine.** Run it rootless, as Podman does by default: root inside the container is an ordinary user outside
  it, so even an app that is fully taken over holds no more than that user does.

The same safeguards suit any app that might be compromised, not only one with a model in it: a browser, a mail client, a
game, a library with a flaw nobody has found yet. Firewalled by default, an exploit stays in one separate environment
and does not take the whole machine with it.

The browser can keep apps apart too. Put each app's container behind one small proxy that gives it a name under
localhost, such as `silvann.localhost` or `searxng.localhost`, instead of a port each. The ports stop colliding, the
address says what you are opening, and the browser keeps each app's cookies and stored data to its own name, which
ports never did ([`docker/README.md`](../docker/README.md#each-app-a-name-under-localhost)). The same suits most desktop
apps built on Electron, which are a web page and a backend: the backend in a container with only the folders it is
given, the page a name in the browser you already have.

Each of these is a flag on the command that starts the container, not a project: a volume, a `:ro`, a `--network`, a
rootless runtime. The risk that is left is the one you chose to give the app, and you can see it in one line. SiLVANN's
own images run this way ([`docker/README.md`](../docker/README.md)): rootless under Podman, the models and the
conversations in one folder of yours, the chat on this machine's address alone — and inside the container, the model's
own commands closed off from its tools file, where the keys are, and from the server's process
([`tools.md`](tools.md#where-the-models-commands-reach)).

## A change of habits, not for the worse

AI will change some of our habits and expectations, as the telephone and the internet did. That does not mean it has to
change them for the worse. Models that run on your own machine, from published weights and software anyone can build,
keep the benefit with the people who most need their work to stay private, and keep the defence in the hands of the
defenders. That is worth safeguarding, and if we are smart about it we can forge new habits where the expectation is
that users' security and privacy are at the top of the design list.
