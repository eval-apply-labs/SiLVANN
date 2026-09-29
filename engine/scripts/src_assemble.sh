#!/usr/bin/env bash
# Enrols new package folders (scripts/src_enroll.py), then writes src/packages/manifest.cuh and
# src/packages/manifest__gpu.cuh from the roster in manifest__header.cuh.
#
# The assembly is a pure function of the roster: one include per package per phase, in row order. A
# macro cannot emit an `#include` — `#define EMIT(x) #include x` is a diagnostic — but the preprocessor
# can expand the roster and this script can put the `#` back on. That is the whole of what it does.
#
# ⭐ THE EXPANSION IS DONE BY `cpp`, WHICH IS NOT A DETAIL. A script that parsed PACKAGE_LIST itself
# would be a SECOND implementation of what that macro means, free to disagree with the compiler about
# its own roster. Handing the real preprocessor the real file removes the possibility rather than
# guarding against it.
#
#   src_assemble.sh            rewrite manifest.cuh
#   src_assemble.sh --check    fail if the committed file is not what the roster produces
set -u
cd "$(dirname "$0")/.." || exit 2
PKGS=src/packages
LEAF=$PKGS/manifest__header.cuh
OUT=$PKGS/manifest.cuh
OUT_GPU=$PKGS/manifest__gpu.cuh
[ -f "$LEAF" ] || { echo "⛔ REFUSING: $LEAF does not exist, so there is no roster to read."; exit 2; }

# ⭐ FIRST, EVERY PACKAGE FOLDER ONTO THE LIST, with the next id — a folder is all it takes to add one. ▶ scripts/src_enroll.py. `--check` changes nothing and fails on a folder not enrolled.
if [ "${1:-}" = "--check" ]; then python3 scripts/src_enroll.py --check || exit 1
else python3 scripts/src_enroll.py || exit 1; fi

probe=$(mktemp) ; trap 'rm -f "$probe"' EXIT
printf '#include "manifest__header.cuh"\n#define SYS2_EMIT(id, name, path) @@path\nPACKAGE_LIST(SYS2_EMIT)\n' > "$probe"
paths=$(cpp -P -I"$PKGS" "$probe" 2>/dev/null | tr ' \t' '\n\n' | sed -n 's/^@@"\(.*\)"$/\1/p')
n=$(printf '%s\n' "$paths" | grep -c .)

# A generator that emits nothing looks exactly like a package list with nothing in it, and one of those
# is a tree that will not link. Refuse rather than write an empty assembly.
[ "$n" -gt 0 ] || { echo "⛔ REFUSING: the roster expanded to no packages. Is PACKAGE_LIST intact?"; exit 2; }

# ⛳ GUARDED BY NAME, NOT `#pragma once`: GCC takes two headers of the same size and modification time — every
#   header of a fresh checkout has the same time — for one file and skips the second.
emit() {
    printf '#ifndef SILVANN__PACKAGES_MANIFEST_CUH\n#define SILVANN__PACKAGES_MANIFEST_CUH\n'
    emit_body
    printf '\n#endif /* SILVANN__PACKAGES_MANIFEST_CUH */\n'
}
emit_body() {
    cat <<'HDR'
/* ⛔ GENERATED FILE — DO NOT EDIT. Your change will be overwritten the next time anything builds.
 *
 * Written by `scripts/src_assemble.sh` from the roster in `manifest__header.cuh`, which is where the
 * reasoning for both files lives; a package is added by its folder, which `scripts/src_enroll.py` enrols.
 * This one is a pure function of that row set: one include per package per phase, in row order.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "manifest__header.cuh"

/* ── PHASE ZERO — WHAT EVERY PACKAGE PUBLISHES INTO THE LANGUAGE ─────────────────────────────────
 *
 * A contract file needs nothing and expands nothing, so every package's rows can be read before
 * anything else in the tree has been. That is what makes this phase possible, and the kinds are why it
 * is necessary: a kind is compared inside declarations — whether a node carries a reference, what an
 * empty register row answers — so its constants have to exist before the first header, while a verb is
 * only ever compared inside an implementation and can be named after the last one. */
HDR
    printf '#include "%s/language_contract.cuh"\n' $paths
    cat <<'A0'

/* Every package has now written its rows, and the tag that makes a kind unique is applied once, here,
 * for all of them. */
#include "language_contract_kinds.cuh"

/* ── PHASE ONE — WHAT EVERY PACKAGE DECLARES ────────────────────────────────────────────────────── */
A0
    printf '#include "%s/manifest__header.cuh"\n' $paths
    cat <<'A1'

#define PACKAGE_HEADER_PRESENT_CHECK(id, name, ...) \
    static_assert(PACKAGE_##name##_HEADER_PRESENT == 1, \
                  "package '" #name "' is listed and its header phase is not included");
PACKAGE_LIST(PACKAGE_HEADER_PRESENT_CHECK)
#undef PACKAGE_HEADER_PRESENT_CHECK

/* ── THE VERBS, NAMED ────────────────────────────────────────────────────────────────────────────
 *
 * Every package has now written its rows, and nothing before this point asks what a verb is called. A
 * gather that reads EVERY package cannot sit inside one of them: the packages listed after it have not
 * been read yet, so it pastes a macro nobody has defined and the compiler reports a syntax error inside
 * whichever package happened to be next — pages away from the row that caused it. Anything that reads
 * all the packages belongs beside the list, and runs once they have all spoken. */
#include "language_contract_verbs.cuh"

/* ── PHASE TWO — WHAT EVERY PACKAGE DEFINES ─────────────────────────────────────────────────────── */
A1
    printf '#include "%s/manifest.cuh"\n' $paths
    cat <<'A2'

#define PACKAGE_PRESENT_CHECK(id, name, ...) \
    static_assert(PACKAGE_##name##_PRESENT == 1, \
                  "package '" #name "' is listed and its implementation is not included");
PACKAGE_LIST(PACKAGE_PRESENT_CHECK)
#undef PACKAGE_PRESENT_CHECK
A2
}

# ⭐ AND THE PACKAGES AS A SILICON FAMILY'S BINARY COMPILES THEM — the same roster, the same row order, and
#   the same reason to generate rather than write: a file outside every package that names each package is
#   the roster's to write. ⚖ *"this sits outside of the individual packages folder, if it has to be like this
#   it will need to be auto generated like manifest.cuh"*.
emit_gpu() {
    printf '#ifndef SILVANN__PACKAGES_MANIFEST__GPU_CUH\n#define SILVANN__PACKAGES_MANIFEST__GPU_CUH\n'
    emit_gpu_body
    printf '\n#endif /* SILVANN__PACKAGES_MANIFEST__GPU_CUH */\n'
}
emit_gpu_body() {
    cat <<'G0'
/* ⛔ GENERATED FILE — DO NOT EDIT. Your change will be overwritten the next time anything builds.
 *
 * Written by `scripts/src_assemble.sh` from the roster in `manifest__header.cuh`, beside `manifest.cuh`:
 * THE PACKAGES, AS A SILICON FAMILY'S BINARY COMPILES THEM. A family's binary holds each package's card
 * side — its doors and the kernels behind them — compiled against the family's answers; which headers
 * that takes is each package's business, named in its own `<pkg>/manifest__gpu.cuh`, and a family
 * includes this file and nothing else of the packages. ⛳ EVERY PACKAGE HAS ONE, even one that runs
 * nothing on a card: it carries the package's requests (`contracts/abi/gpu.cuh`), which every family's
 * table is gathered over. The order is the roster's: the lists first, then each package.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "manifest__header.cuh"          /* the roster, and each package's id */

/* The lists — they expand nothing, so a card side can take them. */
G0
    printf '#include "%s/language_contract.cuh"\n' $paths
    printf '#include "language_contract_kinds.cuh"\n\n/* Each package'"'"'s card side. */\n'
    printf '#include "%s/manifest__gpu.cuh"\n' $paths
}

if [ "${1:-}" = "--check" ]; then
    if ! emit_gpu | cmp -s - "$OUT_GPU"; then
        echo "⛔ STALE — $OUT_GPU is not what the roster produces. Run scripts/src_assemble.sh."
        emit_gpu | diff -u "$OUT_GPU" - | head -20
        exit 1
    fi
    if emit | cmp -s - "$OUT"; then
        echo "✅ ASSEMBLY CURRENT — $n package(s), and $OUT and $OUT_GPU are what the roster produces."
    else
        echo "⛔ STALE — $OUT is not what the roster produces. Run scripts/src_assemble.sh."
        emit | diff -u "$OUT" - | head -20
        exit 1
    fi
else
    emit > "$OUT"
    emit_gpu > "$OUT_GPU"
    echo "✅ WROTE $OUT and $OUT_GPU — $n package(s): $(printf '%s ' $paths)"
fi
