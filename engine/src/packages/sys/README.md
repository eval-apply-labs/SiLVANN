# `sys` — the language, and the operating system it runs in

`sys` is the language of the Eval-Apply Generalized Interpreter — what a value is, the objects values are built
into, and the verbs a program calls — and the small operating system those programs run in, virtualized
over the host's. The evaluator that walks a program lives in `engine/`; the neural network library, `nn`,
is a second package written on top of this one.

**As an operating system.** A machine is a set of workers — runner threads, a block each — over one heap
in host memory. Every block has a run queue any block appends to, and a request is a program, the
read-only names it runs in and the answer it fills (▶ "Work waits its turn", below). The blocks stay up
from the boot to the shutdown; a step that waits for answers is parked while its block runs other work,
the block that fills the last of them wakes it, and an idle block sleeps on its door until work is queued
for it. `MEASURED` on node03: a request there and back ~9 µs on the host, and a parked step on a sleeping
block run within milliseconds of its last answer, where the doze alone would leave it up to 100 ms
(measurements/2026-11-05_callbacks_node03.md). ⛳ The scheduling is **cooperative multitasking**, as in Windows 3.x
or the classic Mac OS: a step runs to its end once begun, and gives its block back by ending or by queuing the step
that continues it (`sys__scheduler__continue`, the yield) or by moving the computation itself
(`sys__scheduler__continue_on`); a queue is first in, first out. Booted without a slice nothing preempts a step and
nothing bounds how long it runs, so a long step holds its block — that one worker, not the machine — until it ends.
⭐ **Booted with a slice it is preemptive multitasking, though not protected**: `sys__scheduler__slice_us:<n>` or
`sys__scheduler__slice_cycles:<n>` in `boot_commands` (0 or absent, the engine's default, for none; the runtime's
`Machine.boot` says `slice_us:50000`, 50 ms), and a computation that has run `n` microseconds — or `n` cycles of the
processor's clock — since its evaluation began gives its block up at the end of a verb — to the back of the queue, as
`(sys__scheduler__continue_on <its block>)` would — when something waits there and it may move: an answer to fill and
an environment of its own, never the host's. Each evaluation starts a whole slice, a moved one's where it goes on.
The slice is the machine's: only its `boot_commands` say it, and a worker booting on a section of its own (a model's
card and CPU do) leaves it as it is. ⛔ Not protected: a slice is as fine as the longest verb, and a program whose steps
keep a worker's scratch between verbs — a model's do — runs them inside `(sys__scheduler__compute_inline 'form)`: there
a spent slice gives nothing up (`(sys__scheduler__preemptive)` says `#f`), a section inside another leaves the outer one
inline, and a kill still ends it at its next slice. `MEASURED` on node03: two of DeepSeek's seats on one card at a
slice of 1 µs gave the block up 1,846 times mid-step and one answered another seat's tokens; with each step on the card
inline, both answered their own (measurements/2026-11-06_kill_node03.md). ⛔ **A verb never waits on another
computation** (⚖ 2026-11-07 *"c opcodes do not wait but return and let the wait be done in lisp"*): it returns, and the
program waits — `sys__result` in a `while`, or a step that ends continued on the answer (`sys__scheduler__continue …
'(r)`), its block free meanwhile. ⭐ **`sys__wait_for_result` parks** (NN-158): on an answer not there yet, a computation that may
move is handed to its own block's queue with the answer its gate, its form left to be applied again, and the block runs
what else waits until the answer fills; one that may not move — the host's, one inside `compute_inline` — waits where it
stands, up to fifteen seconds by the clock, and a wait that runs out answers `ANSR`, an error, never nothing.
⚠ A step that keeps a worker's scratch between verbs and waits mid-way runs inline: parked, another program on its
block would run in the gap. ⭐ **A process can be killed**: a request carries the
process it is of — the asker's, inherited by what it asks and what it goes on as, or a new one from the host's doors —
`(sys__scheduler__pid)` says which, and `(sys__scheduler__kill pid)` takes its requests off every queue, their answers
filled with `SCKL`; a block refuses one taken later and drops its parked steps; a step of it still running ends at its
next move or slice, and an answer it fills at its end is `SCKL`. Nothing it did is undone (measurements/
2026-11-06_kill_node03.md). ⭐ **A process can be asked to stop** — the milder of the two levels (NN-160):
`(sys__scheduler__stop pid)` remembers it, and its program reads `(sys__scheduler__stopped)` once a token (`generate`'s
loop does) and ends at the token it is on, everything whole; a killed process reads as stopped too. The host makes a
process of its own from the counter every process is made from (`process_new`), makes its requests as it
(`process_as`: its thread's, inherited as a program's), and kills it or asks it to stop through doors of its own
(`process_kill`, `process_stop`) — so a server's stop ends an answer at its token, and its kill tears a conversation's
work down (measurements/2026-11-08_processes_node03.md). `MEASURED` on node03: a short computation
queued behind a loop that never yields answered in 1.2 ms at a 1 ms slice, against 806 ms without; two such loops
interleave and take what the two take one after the other; the clock costs a loop 1-2%
(measurements/2026-11-06_slice_node03.md).

**What it asks of the host: primitive calls**, in one contract (`contracts/abi/os.cuh`) that each host
answers in one file (`../../operating_systems/linux/os.cuh`, `../../operating_systems/windows/os.cuh`):
threads and thread keys, locks, conditions and once; a clock, sleep and yield; the CPUs and pinning to
them; memory — mapped, aligned, bound to or interleaved over NUMA nodes, kept resident; files and paths;
and opening a library. A silicon family is such a library: it wraps a driver's functions — HIP, CUDA,
OpenCL — or the CPU's vector units behind the arithmetic `sys` says it needs (`contracts/abi/gpu.cuh`).

**What it knows of neural networks: nothing.** `nn` is written on top of `sys` and the model packages on
top of `nn`; `sys` names them only in its comments. The language and its operating system stand on their
own.

A program is a list. The evaluator scans it, and each time it reaches a form whose head is a verb it
hands that form to the verb, which rewrites it in place — with a value, or with more work. Everything
the evaluator and the verbs touch is made of one thing: a 64-byte **node**.

It runs on the host CPU. A machine is a set of runner threads, one per block, over one heap in host
memory. Graphics cards are reached through the tree's `silicon_families/`, never from here: `sys` only
states what it needs a card to do (`contracts/abi/gpu.cuh`), and each family answers it.

## The ideas, in the order you meet them

**Everything is a node.** A node is a kind, a count, an opcode and six 64-bit words; what the six mean
is the kind's business. The heap hands out nodes in chunks of 512, and a reference is a node OFFSET
into the heap, not a pointer. ▶ `contracts/objects/heap_node.cuh`, `cpu/heap__header.cuh`.

**A kind says what a node is.** Every kind has a published name — `sys__list`, `sys__string`,
`sys__value_int` — and a number, tagged with the package in its top byte so two packages can never
collide. From Python, `e.kinds()` is the whole map. ▶ `contracts/macros/language_contract__kinds.cuh`.

**Some kinds are objects.** An object is reference-counted: whoever holds one retains it and releases it
when done, and it is taken apart when the last hold goes. Each kind that can be allocated has one row
saying how it is released, where it is carved from, how it is cloned, how it is built and how many nodes
it takes — and
the switch over all of them is written once, over `PACKAGE_LIST`, and never edited again as packages arrive. ▶ `contracts/macros/language_contract__objects.cuh`, `cpu/heap_object__header.cuh`.

**The objects:**

| | what it is |
|---|---|
| list | the program's own shape: cells, with sublists; the evaluator walks these |
| node array | n contiguous nodes; a *picture* of a program is a list frozen into one, read-only and shareable |
| string | text held by the machine, with a 64-byte hash |
| symbol | a name registered once and numbered once, spelled by an interned string |
| bindings | what each name means right now, and what it meant before — a stack of meanings per name |
| procedure | what a name means when it names a function (`defun`) |
| dictionary | values found by key, in a balanced tree of small sorted leaves (why: `src/docs/design/dictionary.md`, in the development repository) |
| stack | a chunked last-in-first-out stack |
| error | what a computation hands back when it went wrong |
| computing base | the state machine that owns one computation: claimed, started, finished, read |

And three things that are not objects: the **C fixed stack** (an array of numbers and an index, in C —
the heap's free chunks are one), the **system register** (a few rows every block can name without being handed them), and
the **settings** (a config file read into a sealed dictionary at boot).

**Verbs are rows too.** Each verb is a row: the spelling a program uses, the function that runs it, and
a number, tagged like a kind. From Python, `e.verbs()` lists them. ▶
`contracts/macros/language_contract__verbs.cuh`, `cpu/opcodes/`.

```
arithmetic   add sub mul less eq, and their int__ / float__ forms
control      if begin prog1 while unquote
names        let defun bindings__set bindings__viewonly add_bindings remove_bindings
objects      create type clone
node arrays  node_array__get node_array__set
lists        list__shared_append list__shared_take list__shared_length
computation  compute result wait worker scheduler__continue scheduler__continue_on scheduler__gate scheduler__pid
             scheduler__kill scheduler__compute_inline scheduler__preemptive scheduler__stop scheduler__stopped
the machine  system_register__get system_register__set package__init
dictionary   dictionary__put dictionary__get dictionary__remove
files        file__open file__close
memory       garbage_collection__manual garbage_collection__collect
```

**Nothing is a special form.** A quoted list is a value, so both arms of an `if` sit untouched while its
test is reduced; a quoted name is a value, so an assignment is handed the name rather than what it
means. Quoting covers conditional evaluation and assignment, and no verb is treated specially.

**Truth is two values.** `#t` and `#f` are kinds of their own, and a program writes them so. Inside the
engine `sys__opcodes__truth` is the only thing that makes one, and `if` refuses anything that is neither.

**A teardown can wait.** After `(sys__garbage_collection__manual #t)` an object whose last hold goes waits
on the thread's worklist instead of being taken apart, and `(sys__garbage_collection__collect)`,
`(sys__garbage_collection__manual #f)`, or the evaluation's end takes everything waiting apart. The mode is
the thread's and stays on until the program turns it off; the verb answers the mode it replaced, so
`(let ((was (sys__garbage_collection__manual #t))) … (sys__garbage_collection__manual was))` scopes it. The
counts still decide what is dead; the program chooses when. ▶ `cpu/garbage_collection__header.cuh`.

**Work waits its turn, and its value has a home before it is asked for.** `(sys__scheduler__add cu names program)`
puts a program on block `cu`'s queue, which any number of blocks append to at once, and hands back the **answer** its
value will go into — or `(sys__scheduler__add cu names program answer)` into an answer the program already holds. The
block runs its queue in the order it arrived whenever its base is free, and files each value into its answer.
`(sys__result answer)` says `#t` once the value is there and `#f` while it is on its way, reading one word;
`(sys__wait_for_result answer)` hands over the value, waiting for it, bounded. Reading takes nothing away, so any number
of readers share one answer, and an answer lives as long as somebody holds it — the asker, a reader, a request on its
way. ⭐ **What a computation came to is read from its answer, never from the block that ran it** (⚖ 2026-11-11 *"no one
names a worker to read a computing variable, they just name the return variable"*): a block may hold several askers'
work, so the data's end and the block's are two verbs — `(sys__wait_for_device cu)` waits until block `cu` is idle, its queue
empty and its base free, a probe of the device that hands no value over. A step may go on: `(sys__scheduler__continue cu names program)` queues its
next step on block `cu`, this one among them, in `names` (0: the names it runs in), into the same answer, and only the
step that does not continue fills it. Asked with the names of answers — `(let ((ra (sys__scheduler__add …)) (rb
(sys__scheduler__add …))) (sys__scheduler__continue CARD names second_half '(ra rb)))` — the next step runs once every
one of them is filled, its block parking it and running what is behind it until the block that fills the last one
rings it, and finds `ra` and `rb` bound to what they came to in its environment — the results themselves, read
straight, the answers behind them let go of when the step ends. A computation may move instead:
`(sys__scheduler__continue_on cu)` names a block and nothing else — the computation itself goes there, its environment
(the scopes its block built over the names it was asked with, every `let`, `set!` and `defun` in them) and its program
as it stands, and goes on from its next form, into the same answer; nothing of it runs where it was after. Nothing in a
program can name a second program over that environment, so no two ever hold it (⚖ 2026-11-06 *"as long as we never
maintain two programs to reference the same live bindings we are good both to send the rest of the compute somewhere
else and to send it on the back of your own queue"* · *"it has to be itself so it has to be (sys__scheduler__continue_on
cu) and the bindings and the program are implicit. this guarantees no double live bindings"*). Moved onto its own block
it goes to the back of the queue. What it asked of each block and its collection mode go with it. Only that environment
moves: the host's, or one a `continue` in its own names shares with a step on another block, is refused (`SCSH`).
`MEASURED` on node03: a move between two blocks ~6.3 µs (measurements/2026-11-06_continue_on_node03.md). The host does the same from outside (`place`,
`poll`, `take`). A value that could not be computed arrives as an error, a value like any other. A shared list is
reached only through the `list__shared_` verbs, each the list's own lock around one append, take or count.
▶ `cpu/scheduler__header.cuh`, `cpu/answer.cuh`.

**When something refuses, it says why.** A verb that refuses leaves an error in the form it was handed,
and a refusal with nobody to return to lands on the fault channel as a four-letter word.
▶ `cpu/error.cuh`, `cpu/fault__header.cuh`. The program goes on: a refusal is a value.

**The evaluator stopping is a panic.** Its stops — a descent it has no room to record (`EVAD`), a form it
has no room to make (`EVNR`), a call it cannot make or a head that is not a verb (`EVAV`) — abandon the
evaluation where it stands, so the machine stops with it: the word is kept, and every door refuses until a shutdown and a boot.
▶ `engine/eval.cuh`, "A STOP IS A PANIC".

**Packages stand themselves up.** At boot every package's init is one call in a list, and the list is
evaluated like any program: an init that finds what it needs not ready yet goes back to the end of the
list. ▶ `cpu/package__header.cuh`.

## Layout

```
language_contract.cuh    this package's three lists, gathered: kinds, verbs, objects
manifest__header.cuh     what the package DECLARES, in order
manifest.cuh             what the package DEFINES
manifest__gpu.cuh        what a silicon family's binary takes of it: only its requests

contracts/               definitions only — nothing in here runs
  objects/               one file per object: its struct, layout, sizes and fault words; kind.cuh
  macros/                the three lists — language_contract__{kinds,verbs,objects}.cuh
  abi/                   the edges of the package:
    cpu.cuh                what a program outside the language may call (sys_abi_peek and the rest)
    gpu.cuh                what sys asks of every silicon family: memory, copies, waiting, devices
    silicon_family.cuh     the table a family's binary exports
  defaults.cuh           the numbers you can turn, and what they are when nobody turns them

cpu/                     everything that runs
  <object>__header.cuh   what you can use: the description, then the contract
  <object>__impl.cuh     how it works
  <object>.cuh           one file, where there is nothing to hide behind a header
  opcodes/               the verbs, and the bridge every verb is called through (verb_abi.cuh)
  silicon/               the evaluator's own primitives — which block am I, an atomic, a wait, scratch
                         room — and silicon_family.cuh, the program's table of families and the doors
                         sys__gpu__* called through it
  abi_surface__impl.cuh  the C interface as data, for a host language to bind
  todo.cuh               what is not built yet, declared so code can be written against it
```

## Using it from outside

A program that is not written in the language reaches `sys` through its C interface,
`contracts/abi/cpu.cuh`: looking at a heap node, turning an offset into an address, the allocator's
tallies, and the fault channel. The same calls are described as data by `sys_abi_surface()`, and the
Python wheel binds them from that table as `e.sys`:

```python
import silvann_engine_cpu as e
e.boot()
e.sys.peek(address)        # (kind, op_code, [six words])
e.sys.full_address(offset)
e.sys.counters()           # {'chunk_claims': …, 'chunk_refusals': …, 'deallocations': … or None}
e.sys.fault()              # {'raised': …, 'pc': …, 'op': …}
e.sys.fault_clear()
```

Building and running programs — lists, strings, names, `execute` — is the engine's interface, on the
module itself. ▶ `engine/abi/`.

## Conventions

### Names carry their package, their type and their visibility

A name is **package, type, rest**, separated by double underscores: `sys__stack__push`. Single
underscores stay inside a part, so a name can always be taken apart by rule — which matters most for a
package from outside whose name has underscores of its own. The double underscore is always in the
middle of a name, never at its start or end, which keeps clear of the names C reserves for the compiler.

The program is one translation unit, so `static` hides nothing, and the language this is written in has
no `private`. Visibility is carried by the name instead, as a marker after the type:

```
sys__stack__push                           public     anyone may call it
sys__stack__zzengine_compute_stack_push    engine     this package and the engine
sys__stack__zzpackage_construct            package    this package only
sys__stack__zzprivate_set_current_chunk    private    its own file only
```

`zz` sorts after every letter, so a type's public names list first. `scripts/src_c_subset_gate.py` (in the
development repository) finds each marked name's definition and fails on any reach from outside its tier;
`test/` is exempt, and the gate says so when it applies the exemption.

A method a kind takes from the object interface, rather than overriding, ends in `__default` or `__plain`
— `sys__heap_object__release__default`.

### Three kinds of comment

```
═════ … ═════      the file     what is true of everything below
───── … ─────      an area      the argument behind a group of functions; it opens and closes
/* plain text */   a function   about the thing directly underneath, and nothing more
```

A plain comment must not carry general context; if it outgrows its function, give it rules and move it
above the group it describes.

⛔ **Never write a path containing `*/` in a block comment — it TERMINATES THE COMMENT, and the words
after it become code.** Backticks do nothing for a compiler. Write the wildcard as a word, or name the
paths. The claim gate's `comment_glob` rule checks for it.

### AI temporary blocks

A block headed `AI TEMPORARY COMMENT` is a note from one agent to the next — usually what a file replaced,
so it is not restored by mistake. People are not meant to read them, and they are removed all together
at release, never one at a time.

- **ONE PER FILE, AT THE TOP**, so a reader collapses one region and is done.
- A sentence that compares a file against earlier code belongs in it, or in the commit message — not in
  the text a reader of the contract meets.

### Headers

A header has two parts, in this order: a **description** (what this is and why it is shaped this way),
then a **contract** (what the caller owes, what it promises back, what it does not do, and the
declarations — no implementation). A file earns a header when its surface has stopped moving and it has
callers who should not need to read the rest; a file that is all surface stays one file.

The clause to re-read whenever a file grows is **what it does not do**: a promise that something is
absent has nothing that will notice when it stops being true.

## Checks

From the development repository's root:

```
bash scripts/src_selftest.sh             the language, compiled for the host and run: ~1,650 checks
python3 test/src_device_selftest.py      the whole engine through the Python wheel
python3 scripts/src_c_subset_gate.py     the code stays C-shaped; names stay in their tiers
python3 scripts/src_claim_gate.py        every claim this tree writes down still holds
```

Run the gates for their rules; they print them, and no count of them is written here.
