import re

ID = 'silicon_seam_conformance'
SPEC_FILE = 'packages/sys/cpu/silicon/silicon__header.cuh'
HOST_PRIMITIVES = ('atomic.cuh', 'block.cuh', 'memory.cuh')
# ⚖ *"primitives with the package"*: the arithmetic nn's kernels call is declared by nn, beside its doors
NN_SPEC_FILE = 'packages/nn/contracts/abi/gpu.cuh'
_FAMILY_FIELDS = ('device_count', 'bind_device', 'bound_device')
_EVALUATOR = 'evaluator'          # sys's cpu/silicon/: the evaluator's own primitives
_CPU_FAMILIES = ('host',)         # a family whose functions are plain host functions
SPEC_PHRASE = 'Each is DEFINED by exactly one backend, selected by the build.'
WHY = ('A backend that drops a verb, or defines one with a drifted parameter type, stops answering the '
       'seam silently: the link error only fires for a verb something CALLS, and the C++ harness the '
       'comment measured does not catch parameter drift at all (it overloads instead), so the mismatched '
       'definition becomes a second function nobody calls while every caller keeps the stand-in.')
MEASURED = ('2026-09-26: 24 verbs declared in silicon__header.cuh — 10 device and 14 host — and 13 in '
            "nn's gpu.cuh; backends amd_rocm6_wave64, evaluator, host, khronos_opencl2, nvidia_cuda12 and x86_avx2 "
            '(sys\'s cpu/silicon/ and each family under silicon_families/); 161 definitions found, 144 '
            'signature pairs compared, every backend set equal to the declared set, every signature '
            'identical. The scope is part of the comparison and not decoration: a host verb defined '
            '`__device__`, or a device verb defined without it, is a different function that C++ accepts '
            'and the caller never reaches. Re-derive: `python3 scripts/src_claim_gate.py | grep conformance`.')

_LEX = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.S)

# a declaration ends in ';', a definition in '{'; the parameter list may run over several lines.
# `__device__` is OPTIONAL and CAPTURED: the seam has two sides, and which side a verb is on is part of
# what a backend owes. Matching only the device form is how twelve host verbs sat in the seam unwatched.
_SIG = re.compile(r'\bstatic\s+(__device__\s+)?(?:inline|__noinline__|__forceinline__)\s+'
                  r'([A-Za-z_][A-Za-z0-9_ \t*&]*?)\b((?:sys__silicon|sys__gpu|nn__silicon)__\w+)\s*\(([^)]*)\)\s*([;{])', re.S)

_KEYWORD = {'void', 'int', 'char', 'unsigned', 'signed', 'long', 'short',
            'float', 'double', 'bool', 'const', 'volatile'}


def _strip_comments(text):
    return _LEX.sub(lambda m: m.group(0) if m.group(0)[0] in '"\'' else ' ', text)


def _norm(t):
    t = re.sub(r'\s+', ' ', t.strip())
    t = re.sub(r'\s*\*\s*', '* ', t)
    t = re.sub(r'\s*&\s*', '& ', t)
    return t.strip()


def _norm_param(p):
    p = _norm(p)
    if not p:
        return ''
    toks = p.split()
    # drop the parameter NAME; a bare type keyword or a pointer spelling is not a name
    if len(toks) >= 2 and re.fullmatch(r'[A-Za-z_]\w*', toks[-1]) and toks[-1] not in _KEYWORD:
        toks = toks[:-1]
    return ' '.join(toks)


def _params(raw):
    parts = [_norm_param(p) for p in raw.split(',')]
    parts = [p for p in parts if p]
    if parts == ['void']:
        parts = []
    return tuple(parts)


def _sig_for(family):
    """The signature pattern, widened to a silicon family's own prefix — `amd_rocm6_wave64__…` — which is how
    a family names the doors it defines. Its helpers — private to a file, or shared by the family's own files
    (`zzpackage`, which the C-subset gate keeps inside the family's folder) — and its `includes` are not doors."""
    return re.compile(_SIG.pattern.replace(r'((?:sys__silicon|sys__gpu|nn__silicon)__\w+)',
                                           r'((?:sys__silicon|sys__gpu|nn__silicon|%s)__(?!zzprivate|zzpackage|includes\b)\w+)' % re.escape(family)),
                      re.S)


def _harvest(text, terminator, family=None):
    """-> list of (name, (scope, return-type, param-type-tuple))"""
    out = []
    for dev, ret, name, raw, term in (_sig_for(family) if family else _SIG).findall(_strip_comments(text)):
        if term == terminator:
            out.append((name, ('device' if dev.strip() else 'host', _norm(ret), _params(raw))))
    return out


def _spell(sig):
    """The signature as a reader would write it, so a mismatch names both sides in one line."""
    scope, ret, params = sig
    return '%s %s(%s)' % (scope, ret, ', '.join(params))


def _requests(files):
    """-> (declared sys verbs, nn's requests), each {name: signature}. What every package asks of silicon."""
    declared = dict(_harvest(files.get(SPEC_FILE, ''), ';'))
    nn = {n: sig for n, sig in _harvest(files.get(NN_SPEC_FILE, ''), ';') if n.startswith('nn__silicon__')}
    return declared, nn


def _owed(family, declared, nn):
    """-> {name a family defines: what it answers}. sys's host verbs under the family's own prefix, the three
    device doors, and nn's requests — all of them for a card family, the arithmetic alone for one that runs
    no kernel. ONE definition, read by this rule and by the build (`unanswered`)."""
    owed = {}
    for n, sig in declared.items():
        if sig[0] == 'host':
            tail = n[len('sys__gpu__'):] if n.startswith('sys__gpu__') else n[len('sys__silicon__'):]
            owed['%s__%s' % (family, tail)] = n
    for w in _FAMILY_FIELDS:
        owed['%s__%s' % (family, w)] = w
    for m, sig in nn.items():
        if family not in _CPU_FAMILIES or sig[0] == 'device':
            owed[m] = m
    return owed


def _answers(parts):
    """The family whose seam this file is part of, or None. ⚖ *"a folder for primitives and one for overrides so
    it is clear what is expected to run and what are performance upgrades"*: a family's answers are the files
    directly in its folder, or in its `primitives/`; `overrides/` holds faster doors, which answer nothing the
    seam asks, so they are not read here."""
    if len(parts) == 3 and parts[0] == 'silicon_families':
        return parts[1]
    if len(parts) == 4 and parts[0] == 'silicon_families' and parts[2] == 'primitives':
        return parts[1]
    return None


def unanswered(files, family):
    """The requests `family` does not define, sorted — empty when it answers every package. The build asks
    this before compiling a family, and skips one that has gaps rather than failing on them."""
    declared, nn = _requests(files)
    seen = set()
    for path, text in files.items():
        parts = path.replace('\\', '/').split('/')
        if _answers(parts) == family and path.endswith('.cuh'):
            seen.update(n for n, _ in _harvest(text, '{', family))
    return sorted(set(_owed(family, declared, nn)) - seen)


def check(files):
    header = files.get(SPEC_FILE, '')
    declared = {}
    for name, sig in _harvest(header, ';'):
        declared[name] = sig
    if not declared:
        return False, ('examined NOTHING: no `static [__device__] inline ... sys__silicon__*(...);` '
                       'declaration found in %s — the seam has been renamed or moved, so this rule '
                       'cannot vouch for anything' % SPEC_FILE)

    backends = {}
    # nn's requests: the arithmetic, which every family owes — the host's too, because the program compiles
    # nn's bodies — and the launch, which a card family owes
    math_verbs = {n: sig for n, sig in _harvest(files.get(NN_SPEC_FILE, ''), ';') if n.startswith('nn__silicon__')}
    if not math_verbs:
        return False, ('examined NOTHING of nn: no `nn__silicon__*` declaration in %s — the arithmetic nn asks '
                       'for has moved or been renamed' % NN_SPEC_FILE)
    for path, text in files.items():
        parts = path.replace('\\', '/').split('/')
        if not path.endswith('.cuh'):
            continue
        # ⭐ A SILICON FAMILY IS A FOLDER OF THE TREE'S `silicon_families/`, and everything it defines is in
        #   the files directly inside it or in its `primitives/` (▶ `_answers`). The evaluator's own primitives sit in sys's cpu/silicon/, beside
        #   the seam's declaration, and are a different kind of backend (below).
        if _answers(parts) is not None:
            backends.setdefault(_answers(parts), []).extend(_harvest(text, '{', _answers(parts)))
        elif parts[:4] == ['packages', 'sys', 'cpu', 'silicon'] and parts[-1] in HOST_PRIMITIVES:
            backends.setdefault(_EVALUATOR, []).extend(_harvest(text, '{'))
    if not [b for b in backends if b != _EVALUATOR]:
        return False, ('examined NOTHING: no family folder found under silicon_families/ '
                       '(%d verbs declared, 0 implementations to compare them against)' % len(declared))

    # ⭐⭐⭐ TWO KINDS OF BACKEND, AND THE SEAM ALWAYS HAD BOTH HALVES.
    # ⚖ RULED 2026-09-23: *"sys and the evaluator go host."*
    #   THE EVALUATOR'S PRIMITIVES (sys's cpu/silicon/) owe the seam's device verbs — the atomics, the
    #   block identity, the arithmetic — HOST-scoped, because on a CPU the "device" side IS the host. They
    #   must NOT define the card doors: those are a family's.
    #   A SILICON FAMILY owes sys's doors, each under its own prefix — `<family>__memory_allocate` — and
    #   the arithmetic nn declares for its kernels: `__device__` on a card, a plain function on the host,
    #   which compiles the same bodies into the program.
    # ⛳ AND THE SCOPE FLIP IS THE POINT, NOT AN EXEMPTION: a device verb defined `__device__` among the
    #   evaluator's primitives would not compile for a CPU at all, so requiring HOST scope there is a
    #   stricter check than the card families get, not a looser one.
    problems = []
    skipped = []
    compared = 0
    device_side = {n for n, sig in declared.items() if sig[0] == 'device'}
    host_side   = set(declared) - device_side
    for backend in sorted(backends):
        seen = {}
        for name, sig in backends[backend]:
            if name in seen:
                problems.append('%s defines %s twice' % (backend, name))
            seen[name] = sig

        if backend == _EVALUATOR:
            owed = device_side
            forbidden = sorted(host_side & set(seen))
            if forbidden:
                problems.append('the evaluator\'s primitives define the CARD\'s doors %s — those are a '
                                'silicon family\'s' % ', '.join(forbidden))
            missing = sorted(owed - set(seen))
        else:
            # ⭐ sys's host verbs are spelled `sys__gpu__*` and take the family first; a family defines the
            #   same tail under its own prefix, with that first parameter taken off — the family IS it.
            prefixed = {}
            for n in host_side:
                tail = n[len('sys__gpu__'):] if n.startswith('sys__gpu__') else n[len('sys__silicon__'):]
                prefixed['%s__%s' % (backend, tail)] = n
            # ⛳ A family that runs no kernel owes nn's arithmetic — the program compiles nn's bodies — but
            #   not the launch, which only a card has.
            owed = (set(math_verbs) if backend not in _CPU_FAMILIES
                    else {m for m, sig in math_verbs.items() if sig[0] == 'device'})
            missing = sorted(set(_owed(backend, declared, math_verbs)) - set(seen))
            # ⛳ THE THREE A FAMILY'S TABLE HOLDS BESIDE sys's DOORS are declared by the program's table
            #   (`cpu/silicon/silicon_family.cuh`) rather than by the seam, and are a family's too.
            family = {'%s__%s' % (backend, w) for w in _FAMILY_FIELDS}
            extra = sorted(set(seen) - set(declared) - set(prefixed) - family - set(math_verbs))
            if extra:
                problems.append('%s defines undeclared %s' % (backend, ', '.join(extra)))
            for pref, pub in sorted(prefixed.items()):
                if pref in seen:
                    compared += 1
                    scope, ret, params = declared[pub]
                    if not params or params[0] != 'sys__silicon_family__id':
                        problems.append('%s is declared without its family as the first parameter' % pub)
                        continue
                    want = (scope, ret, params[1:])
                    if want != seen[pref]:
                        problems.append('%s %s: header says %s for %s, family says %s'
                                        % (backend, pref, _spell(declared[pub]), pub, _spell(seen[pref])))
        # ⭐ A CARD FAMILY WITH GAPS IS SKIPPED BY THE BUILD, NOT A BROKEN TREE. ⚖ *"can we skip that family
        #   without collapsing the build (and maybe log that the function x was missing)?"* So its gaps are
        #   reported here and do not fail the rule; what it DOES define is still held to the signatures.
        #   ⛔ The evaluator and a family that runs on the host are compiled into the program itself, so a
        #   gap there is a program that does not build, and fails.
        if missing and backend != _EVALUATOR and backend not in _CPU_FAMILIES:
            skipped.append('%s (does not answer %s)' % (backend, ', '.join(missing)))
        elif missing:
            problems.append('%s does not define %s' % (backend, ', '.join(missing)))

        for name in sorted(owed & set(seen)):
            compared += 1
            want = declared[name] if name in declared else math_verbs[name]
            # Among the evaluator's primitives, and in a family that runs on the host, the device set is
            # HOST-scoped — that is the whole port. Everything else must still match to the letter.
            if (backend == _EVALUATOR or backend in _CPU_FAMILIES) and want[0] == 'device':
                want = ('host',) + tuple(want[1:])
            if want != seen[name]:
                problems.append('%s %s: header says %s, backend says %s'
                                % (backend, name, _spell(want), _spell(seen[name])))
    if compared == 0:
        return False, ('examined NOTHING: %d verbs declared, %d backend(s), 0 signature pairs compared'
                       % (len(declared), len(backends)))

    sides = {'device': 0, 'host': 0}
    for sig in declared.values():
        sides[sig[0]] += 1
    # ⛳ BOTH SIDES COUNTED SEPARATELY, because the way this rule was blind before was that one side's
    # count read like the whole seam's. A number that cannot distinguish 10 verbs from 22 is not a
    # liveness signal, it is a number that happened to be true of the part the regex could see.
    if not sides['host'] or not sides['device']:
        return False, ('examined ONE SIDE ONLY: %d device verbs and %d host verbs declared. The seam has '
                       'both, so a zero here means the regex stopped matching a whole side rather than '
                       'that a side went away' % (sides['device'], sides['host']))

    detail = ('%d verbs declared in %s (%d device, %d host) and %d in nn\'s %s; backends %s; %d definitions '
              'found, %d signature pairs compared, scope included'
              % (len(declared), SPEC_FILE.split('/')[-1], sides['device'], sides['host'],
                 len(math_verbs), NN_SPEC_FILE.split('/')[-1],
                 '+'.join(sorted(backends)), sum(len(v) for v in backends.values()), compared))
    if problems:
        return False, detail + ' — ' + '; '.join(problems)
    if skipped:
        return True, detail + ' — every signature identical; the build SKIPS ' + '; '.join(skipped)
    return True, detail + ' — every backend set equal to the declared set, every signature identical'


def falsify(files):
    """Parameter-type drift in one backend: exactly the kind the header MEASURED as uncaught."""
    files = dict(files)
    path = 'silicon_families/amd_rocm6_wave64/nn.cuh'
    text = files[path]
    broken = re.sub(r'(nn__silicon__sqrtf\s*\()\s*float\b', r'\1double', text, count=1)
    assert broken != text, 'falsify() did not bite — sqrtf has moved'
    files[path] = broken
    return files
