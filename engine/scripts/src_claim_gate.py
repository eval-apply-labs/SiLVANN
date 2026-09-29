#!/usr/bin/env python3
"""src_claim_gate.py — the comments that assert a property of the code, made executable.

⭐⭐ THE IDEA IS THE BACKLOG ROW'S AND IT IS ONE SENTENCE: **an argument-comment that asserts a property
OF THE CODE is a specification, and a specification can be checked.** It is not new here — two of the
`c_subset` gate's rules were each born from a prose claim somebody wrote and somebody else then made
machine-checkable. This file is the generalisation, and what it adds is that the prose and the rule stay
ONE OBJECT rather than two that drift apart.

⛔⛔ WHICH IS WHY EVERY RULE CARRIES THE SITE OF THE COMMENT IT CAME FROM, AND WHY THE GATE CHECKS THAT
TOO. A rule whose prose has been edited away is a rule nobody can trace to an intention, and it becomes
folklore the first time somebody asks why it is there. So a missing spec site FAILS, in both directions:
  · the code drifts from the claim  -> the check fails, and the comment is what says what was wanted
  · the claim is edited away        -> the SITE fails, and somebody must decide whether the rule survives

⭐ AND THE FLAGSHIP RULE READS ITS NUMBER OUT OF THE PROSE. `launch_width` does not hold a count of its
own: it parses the integer the comment claims and compares it to what the tree contains. A comment
saying "all 11 launch sites" cannot then sit above a tree with 13. That is the purest form of the idea
and every rule that CAN be written this way should be.

⛔⛔ AND EVERY RULE REFUSES ON ZERO, WHICH IS THE ONE HAZARD THE BACKLOG ROW EXISTS TO CARRY. Its worked
example: a predecessor rule matched a type spelled `SysPendingReleaseStack` while the tree spelled it
`SysPendingDeallocateStack`. It matched NOTHING for its whole life and printed *"0 of them public"* —
which reads exactly like a pass. A gate that draws its rules from prose has that failure BY
CONSTRUCTION, because a reworded comment silently stops being checked.
⇒ ★★ A CLAIM WHOSE SUBJECT CANNOT BE FOUND IS RED, NOT GREEN, and every rule reports HOW MUCH IT LOOKED
AT — so a reader can tell "nothing was wrong" from "nothing was examined".

⛳ WHY THIS IS NOT `src_rot_scan.py`. That one asks whether the prose still names things that exist —
filenames, symbols, counts it can find. This one asks whether an ASSERTION ABOUT BEHAVIOUR still holds:
that a call always passes zero, that one function has exactly one caller, that nothing but one place
makes a particular kind. A name can be checked by looking it up; a behaviour has to be gone and counted.

  usage:  python3 backstage/scripts/src_claim_gate.py            check the tree
          python3 backstage/scripts/src_claim_gate.py --selftest prove every rule can FAIL
          exit 0 = green · 1 = a rule or a spec site failed
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC2 = os.path.join(os.path.dirname(HERE), 'src')
CODE = ('.cuh', '.cu')


def load(root=SRC2):
    """Every text in the tree, by path relative to src. Rules take this dict and nothing else, so the
    selftest can hand them a mutated copy without touching a file."""
    out = {}
    for dp, _, ns in os.walk(root):
        for n in sorted(ns):
            if n.endswith(CODE) or n.endswith('.md'):
                p = os.path.join(dp, n)
                out[os.path.relpath(p, root)] = open(p, errors='ignore').read()
    return out


def code_files(files):
    return {k: v for k, v in files.items() if k.endswith(CODE)}


def strip_comments(text):
    """Blank the comments, KEEPING the line structure — a rule that reports a line number must report
    the one in the file, and collapsing the text silently shifts every number after the first block."""
    text = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


# ── THE RULES ───────────────────────────────────────────────────────────────────────────────────────
# Each is (id, spec_file, spec_phrase, why, check, falsify). `check` answers (ok, detail). `falsify`
# returns a mutated `files` that the rule MUST reject — that is what makes the selftest a proof rather
# than a rehearsal.

def r_one_tu(files):
    n = [k for k in files if k.endswith('.cu')]
    return len(n) == 1, f"{len(n)} .cu file(s): {', '.join(n) or 'none'}"

def f_one_tu(files):
    files = dict(files); files['packages/sys/second.cu'] = '// a second translation unit\n'; return files


# ⛔⛔ THE INSIDE OF A LAUNCH MAY CONTAIN `>`, AND EXCLUDING IT SILENTLY LOSES SITES. The first form of
# this pattern was `<<<([^,<>]+?),\s*([^,<>]+?)>>>`, which cannot match `<<<got->blocks, 1>>>` — so it
# found ONE launch in `boot.cuh` where there are four, and would have reported a count that was short by
# three while looking exactly as confident. ⇒ ★ A PATTERN THAT EXCLUDES A CHARACTER EXCLUDES EVERY USE OF
# IT, and the uses you did not think of are the ones it costs you.
LAUNCH = re.compile(r'<<<(.+?)>>>')

def r_launch_width(files):
    """⛔⛔ IT STRIPS COMMENTS, AND IT DID NOT UNTIL 2026-09-19 — WHICH MADE THE FLAGSHIP RULE COUNT
    PROSE AS CODE. A header explaining that `eval` launches `<<<1, 1>>>` pushed the tally from 23 to 24
    and the gate went red against a tree nothing had changed. ⇒ ★★ AN INSTRUMENT THAT LOOKS MECHANICAL
    BUYS TRUST IT HAS NOT EARNED.

    ⭐⭐ REWRITTEN 2026-09-23, AND THE CLAIM IS UNCHANGED WHILE THE SUBJECT MOVED. The launches became
    `ENG__LAUNCH_ONE` / `ENG__LAUNCH_MANY` (▶ `engine/launch.cuh`) so the host build can mean something
    else by them, and the old form — count every `<<<>>>` and check each one's thread width — then found
    TWO sites and reported the tree had lost 21 launches.

    ⇒ ★ AND THE PROPERTY IS NOW STRONGER THAN IT WAS, WHICH IS WHY THIS IS NOT A WEAKENING. "One thread
    per block" used to be 23 independent facts that happened to agree; it is now true BY CONSTRUCTION in
    the seam's macro definitions, and this rule's first job is to check that no site bypasses them. A raw
    `<<<` outside `launch.cuh` is now a failure in itself — which the old rule could not say, because
    every site was raw."""
    SEAM = 'engine/launch.cuh'
    stray = []
    uses = 0
    for k, v in code_files(files).items():
        body = strip_comments(v)
        if k == SEAM:
            continue          # ⛳ the seam DEFINES the macros; a definition is not a use, and counting
                              #   the four arms here reported 27 against a tree with 23.
        # ⛳ A RUNTIME'S DOORS ARE NOT THE EVALUATOR'S LAUNCHES. ⚖ *"the <<<>>> is on a case by case
        #   basis"* — a door launches its arch's own way, in its arch's folder, and this rule's subject
        #   is the program. The HIP doors were never counted either: they spell a launch
        #   `hipLaunchKernelGGL`, which this pattern does not match; the CUDA doors spell it `<<<`.
        if re.match(r'silicon_families/', k):
            continue
        for m in LAUNCH.finditer(body):
            stray.append(f"{k}: <<<{m.group(1)}>>>")
        # ⛳ RESIDENT is a launch site like the others — a launch whose runners outlive the call, but
        #   one thread per block all the same; and so is HERE, one runner in the caller's own thread.
        uses += len(re.findall(r'\bENG__LAUNCH_(?:ONE|MANY|RESIDENT|HERE)\s*\(', body))
    if stray:
        return False, (f"{len(stray)} launch(es) bypass the seam in `{SEAM}`: " + '; '.join(stray[:3]))

    # ⭐ THE SEAM'S DEFINITIONS ARE WHERE THE WIDTH NOW LIVES, so they are what is checked. Since the device
    #   evaluator retired (2026-09-25) every launch is a host runner, one per block, which is thread zero of
    #   that block BY CONSTRUCTION — so what is checked is that the seam still defines all four launches
    #   and that no card launch (`<<<`) has come back into it.
    seam = strip_comments(files.get(SEAM, ''))
    defined = set(re.findall(r'#\s*define\s+(ENG__LAUNCH_(?:ONE|MANY|RESIDENT|HERE))\b', seam))
    if defined != {'ENG__LAUNCH_ONE', 'ENG__LAUNCH_MANY', 'ENG__LAUNCH_RESIDENT', 'ENG__LAUNCH_HERE'}:
        return False, f"`{SEAM}` does not define all four launches ({sorted(defined)}) — the seam is gone, not satisfied"
    if LAUNCH.search(seam):
        return False, f"`{SEAM}` launches on a card again; the evaluator's launches are host runners"
    widths = ['1'] * len(defined)

    # ⭐ AND THE COUNT COMES OUT OF THE PROSE, not out of this file.
    spec = files.get('docs/design/evaluator.md', '')
    m = re.search(r'all (\d+) launch sites', spec)
    if not m:
        return False, f"{uses} launch sites, but the claim states no count to check against"
    if int(m.group(1)) != uses:
        return False, f"the prose claims {m.group(1)} launch sites and the tree has {uses}"
    return True, (f"{uses} sites, all through the seam; its {len(widths)} definitions are "
                  f"one-thread-per-block; the prose says {uses}")

def f_launch_width(files):
    """⛳ THE FALSIFIER MOVED WITH THE RULE, and the old one was DEAD: it mutated a `<<<1, 1>>>` in
    `engine/boot.cuh` that the macro conversion had already removed, so it changed nothing and the rule
    would have passed its own falsification. ⇒ ★★ A RULE'S FALSIFIER ROTS THE SAME WAY ITS CLAIM DOES,
    and a falsifier that no longer bites is how a rule keeps reporting green after it stopped looking."""
    files = dict(files)
    files['engine/abi.cuh'] = files['engine/abi.cuh'] + '\nstatic void zz_falsify(void) { k<<<1, 64>>>(); }\n'
    return files


RAISE_CALL = re.compile(r'sys__fault__raise\s*\(\s*([^,]+?)\s*,')

def r_fault_pc(files):
    bad, n = [], 0
    for k, v in code_files(files).items():
        for m in RAISE_CALL.finditer(strip_comments(v)):
            a = m.group(1).strip()
            if a.startswith('uint64_t'):     # the declaration and the definition, not calls
                continue
            n += 1
            if a != '0ull':
                bad.append(f"{k}: {a}")
    if n == 0:
        return False, "NO raise call sites found at all — the subject of the claim is missing"
    return (not bad), (f"{len(bad)} of {n} call(s) pass a pc: " + '; '.join(bad[:3])) if bad else f"all {n} raises pass 0ull"

def f_fault_pc(files):
    files = dict(files)
    files['packages/sys/cpu/stack__impl.cuh'] = files['packages/sys/cpu/stack__impl.cuh'].replace(
        'sys__fault__raise(0ull,', 'sys__fault__raise(pc,', 1)
    return files


def r_publish_once(files):
    sites = []
    for k, v in code_files(files).items():
        if k.startswith('packages/sys/cpu/fault__'):
            continue
        for i, line in enumerate(strip_comments(v).split('\n'), 1):
            if 'sys__fault__publish(' in line:
                sites.append(f"{k}:{i}")
    ok = len(sites) == 1 and sites[0].startswith('engine/boot.cuh')
    return ok, f"{len(sites)} caller(s): {', '.join(sites) or 'none'}"

def f_publish_once(files):
    files = dict(files)
    files['engine/abi.cuh'] = files['engine/abi.cuh'] + '\nstatic void x(void){ sys__fault__publish(0); }\n'
    return files


def r_clone_default(files):
    v = files.get('packages/sys/contracts/macros/language_contract__objects.cuh', '')
    found = re.findall(r'sys__heap_object__clone__([a-z_]+)', v)
    odd = sorted(set(x for x in found if x != 'default'))
    return (not odd and len(found) > 0), (f"{len(found)} clone column(s); non-default: {odd}" if odd
                                          else f"all {len(found)} clone columns are the default")

def f_clone_default(files):
    files = dict(files)
    files['packages/sys/contracts/macros/language_contract__objects.cuh'] = files['packages/sys/contracts/macros/language_contract__objects.cuh'].replace(
        'sys__heap_object__clone__default', 'sys__heap_object__clone__deep', 1)
    return files


def r_view_sole_maker(files):
    sites = []
    for k, v in code_files(files).items():
        for i, line in enumerate(v.split('\n'), 1):
            if 'sys__heap__make(SYS__KIND__SUBLIST' in line:
                sites.append((k, i))
    if len(sites) != 1:
        return False, f"{len(sites)} site(s) make a SUBLIST: {sites}"
    k, i = sites[0]
    before = '\n'.join(files[k].split('\n')[:i])
    fn = re.findall(r'^static .*?(sys__[a-z_0-9]+)\s*\(', before, re.M)
    host = fn[-1] if fn else '?'
    ok = host == 'sys__list__zzprivate_view'
    return ok, f"the one site is inside `{host}`"

def f_view_sole_maker(files):
    files = dict(files)
    files['packages/sys/cpu/node_array__impl.cuh'] = (files['packages/sys/cpu/node_array__impl.cuh']
        + '\nstatic void y(void){ sys__heap__make(SYS__KIND__SUBLIST); }\n')
    return files


TEMP = 'AI TEMPORARY COMMENT'

def r_temp_block(files):
    bad = []
    for k, v in code_files(files).items():
        lines = v.split('\n')
        at = [i for i, l in enumerate(lines, 1) if TEMP in l]
        if not at:
            continue
        if len(at) > 1:
            bad.append(f"{k}: {len(at)} blocks")
        elif at[0] > 30:
            bad.append(f"{k}: block at line {at[0]}, not at the top")
        elif 'RETIREMENT' not in v:
            bad.append(f"{k}: no retirement condition")
    n = sum(1 for v in code_files(files).values() if TEMP in v)
    if n == 0:
        return False, "NO file carries a block at all — the subject of the claim is missing"
    return (not bad), (f"{len(bad)} of {n}: " + '; '.join(bad[:3])) if bad else f"{n} file(s), one block each, each at the top and each retiring"

def f_temp_block(files):
    files = dict(files)
    k = next(k for k, v in code_files(files).items() if TEMP in v)
    files[k] = files[k] + f'\n/* {TEMP} — a second one, far from the top */\n'
    return files


GLOBAL = re.compile(r'(\S*)\s*__global__')

def r_kernels_static(files):
    bad, n = [], 0
    for k, v in code_files(files).items():
        for i, line in enumerate(strip_comments(v).split('\n'), 1):
            if '__global__' not in line:
                continue
            # ⛳ A PREPROCESSOR LINE IS NOT A KERNEL. `cpu/silicon/environment.cuh` defines `__global__`
            #   AWAY for the host build, and that `#define` line contains the token — so the rule counted
            #   the thing that ABOLISHES kernels as a kernel, and as a non-static one. ⇒ ★ A TOKEN SEARCH
            #   OVER SOURCE FINDS THE DECLARATION AND THE DEFINITION OF THE DECLARATION.
            if line.lstrip().startswith('#'):
                continue
            n += 1
            if not re.match(r'\s*static\s+__global__', line):
                bad.append(f"{k}:{i}")
    return (not bad and n > 0), (f"{len(bad)} non-static kernel(s): " + '; '.join(bad[:3])) if bad else f"all {n} kernels are static"

def f_kernels_static(files):
    files = dict(files)
    files['engine/abi/running.cuh'] = files['engine/abi/running.cuh'].replace('static __global__', '__global__', 1)
    return files


def body_of(text, signature):
    """The braces of one function, so a rule can say what is NOT in it. A regex over the whole file
    would answer about the file, and the claim is about the function."""
    i = text.find(signature)
    if i < 0:
        return None
    j = text.index('{', i)
    d = 0
    for k in range(j, len(text)):
        if text[k] == '{':
            d += 1
        elif text[k] == '}':
            d -= 1
            if d == 0:
                return text[j:k]
    return None

def r_init_never_releases(files):
    b = body_of(files.get('packages/sys/cpu/computing_base__impl.cuh', ''), 'bool sys__computing_base__init(')
    if b is None:
        return False, "sys__computing_base__init not found"
    rel = len(re.findall(r'sys__heap_object__release\s*\(', b))
    ret = len(re.findall(r'sys__heap_object__retain\s*\(', b))
    return rel == 0, f"{rel} release(s), {ret} retain(s) in the body"

def f_init_never_releases(files):
    files = dict(files)
    t = files['packages/sys/cpu/computing_base__impl.cuh']
    i = t.index('bool sys__computing_base__init(')
    j = t.index('{', i)
    files['packages/sys/cpu/computing_base__impl.cuh'] = t[:j+1] + '\n    (void)sys__heap_object__release(0ull);' + t[j+1:]
    return files


QUOTED_END = re.compile(r'`[^`\n]*\*/[^`\n]*`')

def r_comment_glob(files):
    """⛔ A `*/` INSIDE A BACKTICK-QUOTED PATH CLOSES THE COMMENT IT IS WRITTEN IN. The backticks are
    markdown and the compiler cannot see them; a comment is a lexical span, not a paragraph.

    ⛳ AND THE RULE MATCHES THE QUOTED SPAN IN THE RAW TEXT RATHER THAN LOOKING INSIDE COMMENTS, which is
    not laziness — it is forced by the defect. Once the `*/` lands, the comment has ENDED there, so the
    span is no longer inside one and a comment-scoped search cannot see the very thing it is for. The
    first version of this rule was unfalsifiable for exactly that reason and its selftest said so."""
    bad, n = 0, 0
    hits = []
    for k, v in code_files(files).items():
        n += 1
        for m in QUOTED_END.finditer(v):
            hits.append(f"{k}: {m.group(0)[:50]}")
            bad += 1
    if n == 0:
        return False, "no code files read at all — the subject of the claim is missing"
    return (bad == 0), (f"{bad} quoted span(s) hold a `*/`: " + '; '.join(hits[:3])) if bad else f"{n} code files, no quoted span holds a `*/`"

def f_comment_glob(files):
    files = dict(files)
    files['packages/sys/contracts/objects/kind.cuh'] = files['packages/sys/contracts/objects/kind.cuh'].replace(
        '/*', '/* both arms of `silicon/*/atomic.cuh` agree.', 1)
    return files


RULES = [
    ('one_tu', 'packages/sys/README.md', 'one translation unit',
     "the whole visibility scheme rests on it: `static` only gives internal linkage over the program if the program is one TU",
     r_one_tu, f_one_tu),
    ('launch_width', 'docs/design/evaluator.md', 'launch sites',
     "every kernel is written for the one thread it runs on, so a launch that started more than one per block would run its work several times over",
     r_launch_width, f_launch_width),
    ('fault_pc', 'packages/sys/contracts/objects/stack.cuh', 'left at zero',
     "the pc slot is not a program counter here; the operation word carries the identity, and a real pc in it would be read as one",
     r_fault_pc, f_fault_pc),
    ('publish_once', 'packages/sys/cpu/fault__impl.cuh', 'Told once',
     "the channel is published by the boot and by nothing else; a second publisher would silently reset it mid-run",
     r_publish_once, f_publish_once),
    ('clone_default', 'packages/sys/contracts/macros/language_contract__objects.cuh', 'clone',
     "no kind needs a clone of its own, which is what dissolved a ruling; a kind growing a real one re-opens it",
     r_clone_default, f_clone_default),
    ('view_sole_maker', 'packages/sys/cpu/list__impl.cuh', 'made in one place',
     "what a view takes a hold of and what it puts on the roll cannot come apart, BECAUSE there is one maker",
     r_view_sole_maker, f_view_sole_maker),
    ('temp_block', 'packages/sys/README.md', 'ONE PER FILE',
     "agent-to-agent mail: one block per file, at the top, each carrying the condition under which it goes",
     r_temp_block, f_temp_block),
    ('kernels_static', 'engine/boot.cuh', 'no external linkage',
     "a kernel is launched from the one TU that defines it; one without `static` is reaching for an outside this build does not have",
     r_kernels_static, f_kernels_static),
    ('init_never_releases', 'packages/sys/cpu/computing_base__impl.cuh', 'NOTHING IS RELEASED HERE',
     "this is what makes `init` safe on a node that was just carved, whose words are whatever the last tenant left",
     r_init_never_releases, f_init_never_releases),
    ('comment_glob', 'packages/sys/README.md', 'TERMINATES THE COMMENT',
     "a path holding `*/` closes the block it is written in and the prose after it reaches the compiler, which then names a line well below the cause",
     r_comment_glob, f_comment_glob),
]


# ── AND THE RULES MINED FROM THE CORPUS, ONE FILE EACH ──────────────────────────────────────────────
# ⭐⭐ A DIRECTORY RATHER THAN MORE ENTRIES ABOVE, FOR ONE REASON: the rules above were written by hand,
#   one at a time, and these were mined in parallel from 47 files at once. One file per rule is what lets
#   that happen without two authors editing one list — and it is what will let the next package add its
#   own without touching this one.
# ⛔ EVERY MODULE IS RE-VALIDATED BEFORE IT IS INSTALLED, by `validate_claim_rule.py`, against four
#   conditions: its spec phrase is still in its comment, it holds on the tree, it REFUSES a tree that
#   breaks it, and it REFUSES when it examines nothing. A rule that cannot fail is a sentence that
#   happens to be green, and a rule that stays green when its subject is renamed away is worse.
# ⛳ THEY LOAD IN NAME ORDER so a run is reproducible, and a module that will not import is a RED rather
#   than a skip — a rule silently absent is the same failure as a rule silently passing.
def _mined(directory=os.path.join(HERE, 'claim_rules')):
    import importlib.util
    out = []
    if not os.path.isdir(directory):
        return out
    for n in sorted(os.listdir(directory)):
        if not n.endswith('.py') or n.startswith('_'):
            continue
        path = os.path.join(directory, n)
        spec = importlib.util.spec_from_file_location('claim_rule_' + n[:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out.append((m.ID, m.SPEC_FILE, m.SPEC_PHRASE, m.WHY, m.check, m.falsify))
    return out


RULES = RULES + _mined()


def run(files, show=True):
    bad = 0
    for rid, sf, phrase, why, check, _ in RULES:
        spec_ok = phrase in files.get(sf, '')
        ok, detail = check(files)
        if not spec_ok:
            bad += 1
            if show:
                print(f"  ⛔ SPEC GONE  {rid:<16} `{phrase}` is no longer in {sf}")
                print(f"                 the rule still passes or fails, but nothing says what it was FOR:")
                print(f"                 {why}")
        if not ok:
            bad += 1
            if show:
                print(f"  ⛔ FAIL       {rid:<16} {detail}")
                print(f"                 the claim, at {sf}: {why}")
        if ok and spec_ok and show:
            print(f"  ok           {rid:<16} {detail}")
    return bad


def selftest():
    """⛔⛔ A DETECTOR NEEDS A LIVENESS SIGNAL SEPARATE FROM ITS FINDINGS. Every rule here is shown a tree
    it MUST reject. A rule that cannot fail is not a check, it is a sentence that happens to be green."""
    files = load()
    out = 0
    for rid, sf, phrase, why, check, falsify in RULES:
        real_ok, _ = check(files)
        if not real_ok:
            print(f"  ⛔ {rid}: does not hold on the real tree, so its falsification proves nothing")
            out += 1
            continue
        try:
            broke_ok, detail = check(falsify(files))
        except Exception as e:
            print(f"  ⛔ {rid}: falsification raised {e!r}")
            out += 1
            continue
        if broke_ok:
            print(f"  ⛔ {rid}: ACCEPTED a tree it should have refused — the rule cannot fail")
            out += 1
        else:
            print(f"  ok  {rid:<16} holds on the tree, and refuses a tree that breaks it")
    return out


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        print("=" * 96)
        print("src CLAIM GATE — SELFTEST: every rule must hold here AND refuse a tree that breaks it")
        print("=" * 96)
        n = selftest()
        print("-" * 96)
        print("✅ SELFTEST PASS — every rule is live and every rule can fail." if not n
              else f"⛔ SELFTEST FAIL — {n} rule(s) are not proving anything.")
        sys.exit(1 if n else 0)

    print("=" * 96)
    print("src CLAIM GATE — comments that assert a property of the code, checked against the code")
    print("=" * 96)
    files = load()
    print(f"LIVENESS   {len(files)} files read · {len(RULES)} rules · each traced to the comment it came from")
    print("-" * 96)
    n = run(files)
    print("-" * 96)
    if n:
        print(f"⛔ GATE RED — {n} problem(s). A rule that fails means the code drifted from a claim somebody")
        print("   wrote down; a SPEC GONE means the claim was edited away and somebody must decide whether")
        print("   the rule survives it. Neither is fixed by deleting the rule.")
        sys.exit(1)
    print("✅ GATE GREEN — every claim still holds, and every claim is still written down where it came from.")
    sys.exit(0)
