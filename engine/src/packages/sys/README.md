# `sys` — the language

`sys` is the language of the Eval-Apply GPU Interpreter: what a value is, the objects values are built
into, and the verbs a program calls. The evaluator that walks a program lives in `engine/`; the neural
network library, `nn`, is a second package written on top of this one.

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
| dictionary | values found by key, in a balanced tree of small sorted leaves (why: `src/docs/design/dictionary.md`) |
| stack | a chunked last-in-first-out stack |
| error | what a computation hands back when it went wrong |
| computing base | the state machine that owns one computation: claimed, started, finished, read |

And three things that are not objects: the **carousel** (a ring of numbers in C — the heap's free-chunk
list is one), the **system register** (a few rows every block can name without being handed them), and
the **settings** (a config file read into a sealed dictionary at boot).

**Verbs are rows too.** Each verb is a row: the spelling a program uses, the function that runs it, and
a number, tagged like a kind. From Python, `e.verbs()` lists them. ▶
`contracts/macros/language_contract__verbs.cuh`, `cpu/opcodes/`.

```
arithmetic   add sub less eq, and their int__ / float__ forms
control      if begin prog1
names        let defun bindings__set bindings__viewonly add_bindings remove_bindings
objects      create type clone
computation  compute result completed
the machine  system_register__get system_register__set package__init
dictionary   dictionary__put dictionary__get
```

**Nothing is a special form.** A quoted list is a value, so both arms of an `if` sit untouched while its
test is reduced; a quoted name is a value, so an assignment is handed the name rather than what it
means. Quoting covers conditional evaluation and assignment, and no verb is treated specially.

**Truth is two values.** `#t` and `#f` are kinds of their own. `sys__opcodes__truth` is the only thing
that makes one, and `if` refuses anything that is neither.

**When something refuses, it says why.** A verb that refuses leaves an error in the form it was handed,
and a refusal with nobody to return to lands on the fault channel as a four-letter word.
▶ `cpu/error.cuh`, `cpu/fault__header.cuh`.

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

`zz` sorts after every letter, so a type's public names list first. `scripts/src_c_subset_gate.py` finds
each marked name's definition and fails on any reach from outside its tier; `test/` is exempt, and the
gate says so when it applies the exemption.

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

```
bash scripts/src_selftest.sh             the language, compiled for the host and run: ~1,650 checks
python3 test/src_device_selftest.py      the whole engine through the Python wheel
python3 scripts/src_c_subset_gate.py     the code stays C-shaped; names stay in their tiers
python3 scripts/src_claim_gate.py        every claim this tree writes down still holds
```

Run the gates for their rules; they print them, and no count of them is written here.
