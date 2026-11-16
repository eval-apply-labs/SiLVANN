#!/usr/bin/env python3
"""THE BASE REPL — the Eval-Apply Generalized Interpreter at a prompt: the engine booted with its own packages and no
model, each form read, run in one environment, and its answer written back.

⚖ 2026-11-15, the architect: *"a cli program but instead of doing the messaging it does expose the repl directly ... it
would then be something that a random program can call and execute, just like the jvm"* · *"it would also remain online
when getting a response, then the program can decide to compute again in the same environment or close the vm"*.

    silvann_base_repl                          a prompt in a terminal; from a pipe, a line for each form it is sent
    silvann_base_repl FILE … [-i]              each file's forms in order, its last answer written; -i, a prompt after
    silvann_base_repl -e FORM [-e FORM …]      each form's answer, a line each
    venv/bin/python silvann_base_repl.py …     the same, from the release's folder

The machine is the processors' own family, one worker (`--silicon` names another — a card's), booted with no device
parameters: the language alone, nn's buffer verbs answering `#<error NBNP>`. nn's pools are carved once at the boot and
no verb makes one later, so a program that wants nn's words asks at the boot: `--nn` for small pools, `--params FILE` for
the parameters a file names (⚖ 2026-11-15 *"empty by default with --nn"*). It stays up between forms, and every form
runs in the same environment: a `defun`, or a name defined with `(define n 5)` — set where it has a binding, made where
it has none, so a name defined again is one binding — is there for the next one, until the input ends or `,quit`. What
binds by hand and runs a quoted form — `add-bindings`, `remove-bindings`, `unquote` — is the engine's and the runtime's,
refused at this prompt (⚖ 2026-11-15, NN-193). The forms are the composer's Lisp (▶ silvann_runtime/composer.py), what a model's program is
written in: a file a model carries loads here as it loads at a model's boot, its verbs answering where the machine has
what they need.

── WHAT A PROGRAM DRIVING IT READS ──────────────────────────────────────────────────────────────────────────────────────
A line for every form it is sent, in order, once the form is whole — several forms on a line answer a line each, a form
over several lines answers once, and a blank line or a comment answers nothing:
    3   2.5   #t   #f   nil   name   (1 a (2 b))      a value, written as the reader reads it again
    #<nn__buffer @753>                                an object the reader cannot write: its kind, and where it is
    #<error BNDU>                                     an error — a value like any other: the machine and its names stay
    #<refused …>                                      text that is not a program, or a machine that is stopped: nothing ran
    #<panic EVAV>                                     the evaluator stopped on this form: the machine is shut down, and every
                                                      form is refused until `,reboot` boots a new one (⚖ 2026-11-03: *"a
                                                      panicked response should simply output to the repl/log that it failed
                                                      and not reinstate the heap ... the reboot is an user's explicit
                                                      choice"*) — a head that is no verb and no procedure stops it, as does
                                                      a procedure called with too few or too many
A line that starts with `,` is a command — `,help` `,quit` `,reboot` `,load FILE` `,verbs [TEXT]`, and `,tools FORM`,
the form run by the model's tool in a session of its own (silvann_tools_repl's: its commands, its asks) — and answers a
line too. ⛳ CTRL-C (SIGINT) KILLS THE FORM RUNNING, NOT THE MACHINE: each form runs as a process of its own
(`Machine.new_process`), and the kill answers `#<error SCKL>` with the machine and its names kept; a second Ctrl-C before
the form ends leaves the program. At the prompt, Ctrl-C drops the form being typed.
A batch — the files and the `-e` forms — stops at the first answer that is not a value: it writes it, and exits.

  exit: 0 · 1 the last answer an error · 2 refused · 3 the machine stopped (a panic, or a boot refused) · 130 left by Ctrl-C
"""
import argparse
import os
import struct
import sys
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
# the runtime beside this file in a release, or the development tree's
_HERE = os.path.join(ROOT, "runtime") if os.path.isdir(os.path.join(ROOT, "runtime")) else \
    os.path.join(os.path.dirname(ROOT), "python")
sys.path.insert(0, os.environ.get("SILVANN_RUNTIME", _HERE))
from silvann_runtime.composer import REFUSED_AT_A_PROMPT, ComposerError, Text, read_all, to_text   # noqa: E402
from silvann_runtime.machine import Machine, Panic                             # noqa: E402
from silvann_runtime.prompt import texts                                       # noqa: E402

VALUE, ERROR, REFUSED, PANIC = 0, 1, 2, 3            # an answer's status, and the exit code a batch ends with
DIM, RED, RESET = "\033[2m", "\033[31m", "\033[0m"

# ⛳ `--nn`'s buffer pools, small: `nn__buffer__getnew` takes 1 MiB at most, from 32. The pools are one span nn's package
#   init sums from these lines and the family allocates at the boot, carved into a free list a class; a buffer goes back
#   to its list when its last holder lets go, so a `let` bounds a buffer's life and the boot its pool's size. nn's span
#   boots only with a model's shape named as well — four layers, three DeltaNet and one of full attention, as
#   test/src_answer_buffer.py boots — and refuses the boot with any of these lines left out ("the silicon family would
#   not allocate the pool"). `MEASURED` 2026-11-15 on node03 (x86_avx2): the boot with them takes 0.07 s, the same as
#   without; 1,000 buffers of 4 KiB taken one after another inside a `let`, from a class of 8.
NN_POOLS = """nn__buffer_resid__size_bytes:4096
nn__buffer_resid__qty:8
nn__buffer_main__size_bytes:1048576
nn__buffer_main__qty:32
nn__buffer_attn_out__size_bytes:4096
nn__buffer_attn_out__qty:4
nn__buffer_logits__size_bytes:8192
nn__buffer_logits__qty:1
nn__buffer_partials__size_bytes:4096
nn__buffer_partials__qty:4
nn__expert_cache_l1__size_mb:8
nn__cartridge_sink__tokens:4
nn__cartridge_sink__bytes_per_token_layer:512
nn__cartridge_tq4__tokens:16
nn__cartridge_tq4__bytes_per_token_layer:512
nn__cartridge_tq6__tokens:16
nn__cartridge_tq6__bytes_per_token_layer:512
nn__cartridge_fp32__tokens:16
nn__cartridge_fp32__bytes_per_token_layer:512
nn__model__kv_cache_layers_fullattn:1
nn__model__kv_cache_layers_deltanet:3
nn__conversation__text_size_kb:4
nn__deltanet_state__bytes_per_layer:4096
nn__model__layer_types:dddf
nn__ram_buffer_result__size_bytes:65536
nn__ram_buffer_result__qty:16
nn__ram_buffer_experts__size_bytes:4096
nn__ram_buffer_experts__qty:1
nn__ram_buffer_table__size_bytes:128
nn__ram_buffer_table__qty:1
"""

HELP = ("forms are the composer's Lisp: (+ 1 2) · (define n 5) · (define (sq x) (* x x)) · "
        "(let ((i 0)) (while (< i 3) (set! i (+ i 1))) i) — commands: ,quit ,reboot ,load FILE ,verbs [TEXT] "
        ",tools FORM ,help")

def one_line(text):
    return " ".join(str(text).split())


class Printer:
    """Answers written as text, read through sys's C interface (`full_address`, `peek`) by the layouts sys's contracts
    fix, and a frozen list read through the engine's `thaw` — nothing here reaches past what those publish."""
    # contracts/objects/list.cuh: a list's store names its first chunk of values; a chunk's head, one node before its first
    #   value, says how many values it holds and which chunk comes next; a sublist's payload, the store it views and how
    #   far along it begins
    STORE_VALUES, HEAD_USED, HEAD_NEXT = 0, 1, 2
    SUBLIST_STORE, SUBLIST_START = 0, 1
    ERROR_CODE = 0                     # contracts/objects/error.cuh: the payload's word
    MOST, DEEPEST = 1000, 32           # the cells written of one answer, and the lists inside lists, then "…"
    # the kinds whose cell holds what it names (sys's `carries_reference`): an answer of one is the asker's to release
    HOLDS = ("sys__object_reference", "sys__quoted_list", "sys__procedure_reference", "sys__quoted_array",
             "sys__frozen_list", "sys__node_array_cone")

    def __init__(self, m):
        self.m, self.e = m, m.e
        self.kinds = {v: k for k, v in m.K.items()}
        self.verbs = {v: k for k, v in m.V.items()}
        self.left = self.MOST

    def node(self, offset):
        return self.e.sys.peek(self.e.sys.full_address(offset))

    def answer(self, kind, value):
        """(text, whether it is an error) for an answer, and its hold given back."""
        self.left = self.MOST
        name = self.kinds.get(kind, str(kind))
        try:
            error = self.error_word(name, value)
            text = "#<error %s>" % error if error is not None else self.cell(name, 0, value, 0, top=True)
        except RuntimeError:                       # a peek refused: an offset that is no node
            error, text = None, "#<%s @%d>" % (name, value)
        if name in self.HOLDS and value:
            self.e.release(value)
        return text, error is not None

    def error_word(self, name, value):
        if name != "sys__object_reference" or not value or self.kinds.get(self.node(value - 1)[0]) != "sys__error":
            return None
        word = self.node(value)[2][self.ERROR_CODE]
        spelled = word.to_bytes(8, "big").lstrip(b"\0").decode("ascii", "replace")
        return spelled if spelled.isprintable() and 0 < len(spelled) <= 4 else "%#x" % word

    def cell(self, name, op, value, depth, top=False):
        if name == "sys__value_int":
            return str(value - (1 << 64) if value >> 63 else value)
        if name == "sys__value_float":
            return repr(struct.unpack("<d", struct.pack("<Q", value))[0])
        if name in ("sys__value_true", "sys__value_false", "sys__value_null"):
            return {"sys__value_true": "#t", "sys__value_false": "#f", "sys__value_null": "nil"}[name]
        if name in ("sys__quoted_name", "sys__binding_reference", "sys__quoted_string_name",
                    "sys__string_binding_reference"):
            try:
                spelled = (self.e.symbol_name(value) if name in ("sys__quoted_name", "sys__binding_reference")
                           else self.e.string_read(value)[0])
            except RuntimeError:
                return "#<%s %d>" % (name, value)
            text = spelled.decode("utf-8", "replace")
            return "'" + text if name.startswith("sys__quoted") and not top else text
        if name == "sys__standard":                # a verb, as a list holds one: written as it is published
            return self.verbs.get(op, "#<verb %d>" % op)
        if name == "sys__frozen_list":             # a list inside a picture, made a list to be read
            got = self.e.thaw((self.m.K["sys__frozen_list"], 0, value))
            try:
                return self.items(got[2], depth)
            finally:
                self.e.release(got[2])
        if name in ("sys__quoted_list", "sys__object_reference") and value:
            head = self.kinds.get(self.node(value - 1)[0])
            if head in ("sys__list", "sys__sublist"):
                text = self.items(value, depth)
                return "'" + text if name == "sys__quoted_list" and not top else text
            if head == "sys__error":
                return "#<error %s>" % self.error_word("sys__object_reference", value)
            if head == "sys__string":
                return to_text(Text(self.e.string_read(value)[0].decode("utf-8", "replace")))
            return "#<%s @%d>" % (head, value)
        return "#<%s @%d>" % (name, value)

    def items(self, offset, depth):
        if depth >= self.DEEPEST:
            return "(…)"
        payload = self.node(offset)[2]
        sublist = self.kinds.get(self.node(offset - 1)[0]) == "sys__sublist"
        store, start = (payload[self.SUBLIST_STORE], payload[self.SUBLIST_START]) if sublist else (offset, 0)
        chunk, seen, out = self.node(store)[2][self.STORE_VALUES], 0, []
        while chunk:
            head = self.node(chunk - 1)[2]
            for i in range(head[self.HEAD_USED]):
                if seen >= start:
                    if self.left <= 0:
                        return "(" + " ".join(out + ["…"]) + ")"
                    self.left -= 1
                    kind, op, args = self.node(chunk + i)
                    out.append(self.cell(self.kinds.get(kind, str(kind)), op, args[0], depth + 1))
                seen += 1
            chunk = head[self.HEAD_NEXT]
        return "(" + " ".join(out) + ")"


class Repl:
    def __init__(self, a):
        self.a = a
        self.m = None
        self.stopped = None              # the word the machine stopped with, while it is down
        self.status = VALUE              # the last answer's
        self.tools = None                # the tool's session `,tools` runs forms in, made at the first
        self.tty_out = sys.stdout.isatty()
        if a.params is not None:
            with open(a.params, encoding="utf-8") as f:
                self.params = f.read()
        else:
            self.params = NN_POOLS if a.nn else ""

    # ── the machine ───────────────────────────────────────────────────────────────────────────────────────────────────
    def boot(self):
        """A new machine: True, or False with the reason written to stderr."""
        try:
            m = Machine(silicon=self.a.silicon)
            m.boot(self.params, alloc_mb=self.a.memory_mb)
        except Exception as ex:
            sys.stderr.write("[silvann] the machine did not boot: %s\n" % one_line(ex))
            return False
        m.comp.refused = REFUSED_AT_A_PROMPT       # a person's text binds with define and let (NN-193)
        self.m, self.stopped, self.printer = m, None, Printer(m)
        return True

    def shutdown(self):
        if self.m is not None and self.m.booted:
            self.m.shutdown()

    # ── a form ────────────────────────────────────────────────────────────────────────────────────────────────────────
    def run(self, node):
        """One form run as a process of its own: (text, status)."""
        if self.m is None or self.stopped is not None:
            return "#<refused the machine is stopped (%s): ,reboot boots a new one>" % (self.stopped or "no boot"), REFUSED
        text = to_text(node)
        if not isinstance(node, list) or node[:1] == ["quote"]:   # a name, a number or a quoted form: a program needs
            text = "(begin %s)" % text                            # a head
        m, got, done = self.m, {}, threading.Event()
        pid = m.new_process()

        def evaluate():
            try:
                with m.as_process(pid):
                    got["answer"] = m.run(text)
            except BaseException as ex:            # noqa: B902 — whatever ends the form is its answer
                got["raised"] = ex
            finally:
                done.set()

        # ⛳ THE WAIT IS ON AN EVENT THE FORM'S THREAD SETS, NOT ON `join`: Python 3.10's `join`, interrupted by a Ctrl-C,
        #   can mark the thread stopped while it still runs (`_wait_for_tstate_lock`), and the answer is read before it is
        #   there. `MEASURED` 2026-11-15 on node03, venv's Python 3.10: a SIGINT during `join` came back as no answer.
        threading.Thread(target=evaluate, name="form", daemon=True).start()
        killed = False
        while not done.is_set():
            try:
                done.wait()
            except KeyboardInterrupt:
                if killed:
                    sys.stderr.write("\n[silvann] left with the form still running\n")
                    sys.stderr.flush()
                    os._exit(130)
                killed = True
                m.kill(pid)
                sys.stderr.write("\n[silvann] the form is killed — Ctrl-C again leaves\n")
        if "raised" in got:
            ex = got["raised"]
            if isinstance(ex, Panic):
                self.stopped = ex.word
                return "#<panic %s>" % ex.word, PANIC
            return "#<refused %s>" % one_line(ex), REFUSED
        kind, value = got["answer"]
        answer, error = self.printer.answer(kind, value)
        fault = m.e.sys.fault()
        if fault["raised"] and not error and self.a.interactive_notes:
            word = fault["op"].to_bytes(8, "big").lstrip(b"\0").decode("ascii", "replace")
            sys.stderr.write("%s; %d fault%s on the way, the last %s%s\n"
                             % (DIM, fault["raised"], "s" * (fault["raised"] != 1), word, RESET))
        m.e.sys.fault_clear()
        return answer, ERROR if error else VALUE

    def forms(self, text, write_each=True):
        """The text's forms run in order; each answer written, or only the last. Stops at the first that is not a value.
        Answers the last status."""
        try:
            nodes = read_all(text)
        except ComposerError as ex:
            return self.write("#<refused %s>" % one_line(ex), REFUSED)
        answer, status = None, VALUE
        for i, node in enumerate(nodes):
            answer, status = self.run(node)
            if write_each or status != VALUE or i == len(nodes) - 1:
                self.write(answer, status)
            if status != VALUE:
                break
        self.status = status
        return status

    def write(self, answer, status):
        if self.tty_out and status != VALUE:
            answer = RED + answer + RESET
        print(answer, flush=True)
        self.status = status
        return status

    # ── a command ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def command(self, line):
        """`,word rest`: False for `,quit`, else True."""
        word, _, rest = line[1:].strip().partition(" ")
        rest = rest.strip()
        if word in ("quit", "exit", "q"):
            return False
        if word == "help":
            self.write(HELP, VALUE)
        elif word == "reboot":
            self.shutdown()
            if self.boot():
                self.write("#t", VALUE)
            else:
                self.m = None
                self.write("#<refused the machine did not boot>", PANIC)
        elif word == "load":
            try:
                with open(os.path.expanduser(rest), encoding="utf-8") as f:
                    text = f.read()
            except OSError as ex:
                self.write("#<refused %s>" % one_line(ex), REFUSED)
            else:
                self.forms(text, write_each=False)
        elif word == "tools":
            self.tools_forms(rest)
        elif word == "verbs":
            names = sorted(n for n in (self.m.V if self.m is not None else {}) if rest in n)
            self.write("(" + " ".join(names) + ")", VALUE)
        else:
            self.write("#<refused no command ,%s: ,help says them>" % word, REFUSED)
        return True

    def tools_forms(self, text):
        """`,tools FORM` — the form run by the model's tool, in a session of silvann_tools_repl's made at the first: its
        commands, its asks (a line at a prompt, an `(ask …)` through a pipe), its answer written as the tool writes it."""
        if self.tools is None:
            try:
                import silvann_tools_repl            # beside this file
                self.tools = silvann_tools_repl.ToolsRepl(argparse.Namespace(mode="ask", tools_config="default"))
            except Exception as ex:                # noqa: BLE001 — whatever stops it is said, and the prompt goes on
                return self.write("#<refused the tool's session: %s>" % one_line(ex), REFUSED)
        try:
            nodes = read_all(text)
        except ComposerError as ex:
            return self.write("#<refused %s>" % one_line(ex), REFUSED)
        for node in nodes:
            got, status = self.tools.run(node)
            self.write(" ".join(got.split("\n")) if not self.tty_out else got, ERROR if status else VALUE)

    # ── the loop ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def loop(self):
        """Forms read as they come — a prompt in a terminal, a line at a time from a pipe — until the input ends or
        `,quit`."""
        if sys.stdin.isatty():
            sys.stderr.write("%s[silvann] the Eval-Apply Generalized Interpreter on %s — %d verbs · ,help · ,quit%s\n"
                             % (DIM, self.m.silicon if self.m else "no machine", len(self.m.V) if self.m else 0, RESET))
        for kind, text in texts("lisp"):
            if kind == "command":
                if not self.command(text):
                    return
            elif kind == "forms":
                self.forms(text)
            else:
                self.write("#<refused the input ends inside a form>", REFUSED)


def main():
    ap = argparse.ArgumentParser(description="The Eval-Apply Generalized Interpreter at a prompt: the engine booted "
                                             "with no model, a form at a time, in one environment.")
    ap.add_argument("files", nargs="*", help="Lisp files whose forms run in order; the last one's answer is written")
    ap.add_argument("-e", "--eval", dest="forms", action="append", default=[], metavar="FORM",
                    help="a form to run, its answer written (again for more)")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="after the files and the -e forms, read forms as they come — a prompt, or a pipe")
    ap.add_argument("--silicon", default="cpu",
                    help="the silicon family the machine stands up on (default: the processors' own, never a card)")
    pools = ap.add_mutually_exclusive_group()
    pools.add_argument("--nn", action="store_true",
                       help="nn's buffer pools, small (default: none — the language alone, nn's buffers refused)")
    pools.add_argument("--params", default=None, metavar="FILE", help="the device parameters it boots with")
    ap.add_argument("--memory-mb", dest="memory_mb", type=int, default=256, help="the heap, in MiB (default 256)")
    a = ap.parse_args()
    a.interactive_notes = sys.stdin.isatty() and sys.stderr.isatty()

    try:
        repl = Repl(a)
    except OSError as ex:
        sys.stderr.write("[silvann] the parameters: %s\n" % one_line(ex))
        sys.exit(REFUSED)
    if not repl.boot():
        sys.exit(PANIC)
    try:
        going = True
        for path in a.files:                      # a file answers once: its last form's answer
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as ex:
                repl.write("#<refused %s>" % one_line(ex), REFUSED)
                going = False
            going = going and repl.forms(text, write_each=False) == VALUE
            if not going:
                break
        for form in a.forms if going else ():
            if repl.forms(form) != VALUE:
                going = False
                break
        if going and (a.interactive or not (a.files or a.forms)):
            repl.loop()
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        repl.status = 130
    except BrokenPipeError:                       # whoever read the answers has gone
        os._exit(repl.status)
    repl.shutdown()
    sys.exit(repl.status)


if __name__ == "__main__":
    main()
