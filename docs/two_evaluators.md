# Two evaluators, one Lisp

SiLVANN has two Lisp evaluators. The **engine's**, the Eval-Apply Generalized Interpreter written in C, runs a model's
program and is what `silvann_base_repl` gives you at a prompt ([`developers.md`](developers.md#the-interpreter-at-a-prompt)).
The **tool's**, written in Python beside the server, runs the forms a model writes to act on the machine and is what
`silvann_tools_repl` gives you ([`tools.md`](tools.md)). They read one core the same way and part where their jobs part.
This page says where, and why, so that a form moved from one to the other surprises nobody.

## Why there are two

|  | the engine's | the tool's |
|---|---|---|
| its job | computing: a model's layers, its sampler, on cards and CPU sockets | acting on the machine: files, the shell, the web, pictures, subagents |
| written in | C, the same source on every silicon family | Python |
| what it reaches | nothing past its machine: no file, no network | the host, behind the permissions the conversation's mode allows |
| a failure | some stop the machine (`#<panic …>`), and every form is refused until `,reboot` | an error is a value, `(error "…")`, and nothing ends a conversation |
| its time | processes that preempt each other; Ctrl-C kills the form running | one form at a time, waiting on whatever it reads or runs |
| its procedures | made by `defun` and looked up where they are called | `lambda` closures, looked up where they were made |

One evaluator for both would have to give the engine what only the tool needs (`lambda`, text, errors as values, verbs
that call Python behind the permissions), or give the tool the engine's failure, which a conversation cannot survive. So
they stay two, and one check, `test/src_one_syntax.py`, holds the core they share: a core form that comes to answer
differently in the two fails it. It is the arrangement of one language over two hosts, as Clojure runs on the JVM and
ClojureScript on JavaScript, each with its host's own functions.

## What both read the same

```
quote 'x   if   begin   let   set!   define   defun   while   and   or   ; comments
numbers  #t  #f  nil    + - * < > = eq?
```

- **`define`** binds in the frame it runs in: it sets the name where that frame has it and makes it where it does not.
  At the prompt that frame is the session, so `(define n 5)` is there for the next form, and a name defined again is
  still one binding. Inside a `let` or a procedure the definition is that frame's and goes when it ends, as in Scheme:
  a body's helpers never reach out and overwrite a name of the same spelling outside. `(define (f x) …)` is a `defun`.
  It answers the name.
- **`set!`** changes the nearest binding of a name, however far out, and is an error for a name nobody bound: the guard
  against a mistyped name. It answers `#t`. To change something outside a frame from inside it, `set!` is the way.
- **`while`** takes its test and body as written and answers `#f`.
- **`=`** is `eq?`: the same kind and the same value, so `(= 1 1.0)` is `#f`, and a list is `eq?` only to itself.
- **`+ - *`** take two numbers or more and work left to right: `(+ 9 16 144)` is `(+ (+ 9 16) 144)` in both, so a sum
  of floats rounds the same in both, to the last bit.

## Where they part, and what it means

| written | the engine answers | the tool answers |
|---|---|---|
| `(define (g) (+ y 1))` then `(let ((y 5)) (g))` | `6` | `(error "y is not defined …")` |
| `(and 1 2)` · `(if 0 'yes 'no)` | `#<error OPCT>` · `#<error OPCT>` | `2` · `yes` |
| `(+ 1 2.5)` · `(- 5)` · `(< 1 2 3)` | `#<error OPCT>` · `#<error OPCA>` · `#<error OPCA>` | `3.5` · `-5` · `#t` |
| `(define (add1 x) (+ x 1))` then `(add1 1 2)` | `#<panic EVAV>`, the machine stopped | `(error "a procedure of 1 arguments given 2")` |
| `(lambda (x) x)` | `#<panic EVAV>`, the machine stopped | a procedure |
| `"hello"` | `#<refused …>`, nothing ran | `"hello"` |
| `nobody` | `#<error BNDU>`, the machine kept | `(error "nobody is not defined …")` |

- **Scoping.** The engine's is dynamic: a procedure reads the names bound where it is *called*, so `g` sees the `let`'s
  `y`. Its closures are made on purpose instead, a picture and a view (frozen code and a snapshot of the bindings),
  which is what lets a computation move to another worker and carry everything it reads. The tool's is lexical: `g`
  reads the names where it was *made*, and a `lambda` keeps them. ⇒ In the engine, hand a procedure what it needs as
  arguments or bind it with `let` around the call, and do not count on it seeing the names around its `defun`.
- **Truth.** The engine's tests take `#t` or `#f` and nothing else: a number in a test is a mistake, and it says so.
  The tool counts everything but `#f` and `nil` as true, as Scheme does. ⇒ Code meant for both tests with truths.
- **Numbers.** The engine's integers and floats are two kinds, as its `=` says, and its arithmetic and comparisons
  take one kind at a time: `(+ 1 2.5)` is an error and `(+ 1.0 2.5)` is `3.5`. Its `-` takes two numbers or more and
  its `<` and `>` take two. The tool mixes the kinds as Python does, negates with `(- 5)` and chains `(< 1 2 3)`.
  ⇒ Code meant for both writes `1.0` where a float meets a whole number, and compares two numbers at a time.
- **Failure.** A call the engine cannot make (a head that is no verb and no procedure, a procedure given the wrong
  number of arguments) stops its machine, so a program that went wrong does not go on computing with it; `,reboot`
  boots a new one. An unbound name or a refused verb is an error and the machine is kept. The tool turns every
  failure into an error value the model reads and recovers from.
- **Text and `lambda`** are the tool's. The engine names things with quoted names (`'path`), its programs carry no
  `"…"` literal, and every procedure it has is named by a `defun`.
- **What each knows.** The engine's words are its verbs (sys's, nn's and a model's package's, in C; `,verbs` lists
  them). The tool's are its commands and its list and text words, in Python, `(help)` listing them. A name of one is
  not the other's.

## Between the two

- **`(engine '(form))`** in the tool runs the form in a `silvann_base_repl` its session keeps. The names the form uses
  that the session bound to a number, a truth, `nil`, a name or a list go with it by value. A procedure of the
  session's does not, and a form that calls one is refused before it goes: define it in the form. A `define` or a
  `defun` there stays for the next call. A form calling a head the engine has no verb or procedure for (`progn`,
  `lambda`, a procedure a stopped engine took with it) is refused before it goes too, so the failure that would stop the
  engine stops nothing. The files' verbs are refused, and a form past its time is ended ([`tools.md`](tools.md)).
- **`,tools FORM`** at the base REPL's prompt runs the form by the tool, in a session of its own.
- **What the engine keeps for itself.** `add-bindings`, `remove-bindings` and `unquote` bind by hand and run a quoted
  form; a `let` and the runtime are made of them. They are refused at the base REPL's prompt, as are the verbs `define`
  and `defun` become, which written by hand would skip the frame rule above. A program binds with `let` and `define`.

## Writing for both

- A form meant for both keeps to the shared core: truths in its tests, one kind of number in its arithmetic, no text,
  no `lambda`.
- A procedure defined in one is not in the other: define it where it runs.
- A value that has to outlive a `let` or a procedure is returned, not defined: a definition goes with its frame in both.
- When the two answer the same form differently and this page does not say why, that is a bug in one of them, and
  `test/src_one_syntax.py` is where it gets a check.
