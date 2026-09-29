#ifndef SILVANN__PACKAGES_SYS_CPU_PROCEDURE_CUH
#define SILVANN__PACKAGES_SYS_CPU_PROCEDURE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh"
#include "node_array__header.cuh"
#include "heap__header.cuh"
#include "../contracts/objects/procedure.cuh" /* its constants and fault words */
/* ══ what a name means when it means a function ══════════════════════════════════════════════════════
 *
 * A procedure is a node array of two — the parameters and the body — and the whole of this file is that
 * sentence plus the one thing the array cannot say: that this particular array is a procedure.
 *
 * ⭐ THE KIND IS WHAT LETS THE OBJECT ANSWER FOR ITSELF. A tag on a CELL answers for a slot: it says
 * what is at the other end of the reference in it, and it survives a copy, so a procedure travelling
 * through lists stays one. What it cannot do is answer when the question is asked of the procedure —
 * `type` of the thing would say what it was carved as, which is an array of two like any other.
 * ⇒ ⭐ WHICH IS `defun`'S OWN SENTENCE, ASKED OF THE OTHER FIELD: *"TAGGED, NOT SHAPED. A procedure is
 * an array of two things, and a test of shape would be satisfied by any other array of two. The kind is
 * the answer, one word that nothing satisfies by accident."* It is true of the cell and it has to be
 * true of the object, because a shape test is what a reader falls back on when the object says nothing.
 *
 * ⛳ THE ROOM AND THE TEARDOWN STAY THE ARRAY'S, AND THAT IS WHY THIS IS CHEAP. The contract row names
 * `sys__node_array__zzpackage_release_internal` — which never asks what kind it is handed, because what
 * kind it is, is how the teardown reached it — so a procedure hands over its two elements exactly as an
 * array of two would. Nothing was added to the allocator, the copy or the drain to make that true.
 *
 * ⛔ AND IT IS NOT CONSTRUCTIBLE FROM A PROGRAM. Its row refuses `create`, because a procedure made out
 * of an arbitrary pair would be a thing the evaluator will try to CALL — the parameters must be names and
 * the body must be a form, and neither is checkable from a size. `defun` is the only door, and what that
 * door asks before it asks for the room is the two CONTAINERS: the parameters a quoted list or nil, the
 * body a quoted list.
 * ⛳ AND BOTH REQUIREMENTS ARE CHECKED THERE. The body half is that tag test. The parameter half walks the
 * elements, and refuses any that is not a quoted name, numbered or spelled — the same element test `let`
 * makes — so a procedure reaching the expansion carries names only, and the expansion hands each one to
 * `sys__bindings__add` as the cell it is. ⛔ Without the walk, a quoted list of integers would be adopted
 * as a procedure whose parameters are read as symbol numbers.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Whether a reference names one. ⛔ IT ASKS THE HEAD AND NOT A CELL, which is the distinction the kind
 * exists to make: `PROCEDURE_REFERENCE` is what a SLOT holding one says, and this is what the THING says. */
static __device__ inline bool sys__procedure__is(uint64_t offset) {
    if (!sys__heap_object__zzpackage_addressable(offset)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(offset), SYS__KIND__PROCEDURE);
}

/* Say that an array of two is a procedure. ⛳ IT IS A RESTAMP AND NOT A MOVE: the room, the count and the
 * two elements are exactly where the copy left them, and only the head's first word changes — so there is
 * no window in which a half-built procedure exists and nothing to undo if the caller then fails.
 * ⛔ `zzpackage_` BECAUSE IT TRUSTS ITS CALLER: it is `defun` saying what it just built, and a caller
 * handing it anything else has made a procedure out of it. */
static __device__ inline void sys__procedure__zzpackage_adopt(uint64_t offset) {
    sys__heap_node* head = sys__heap_node__zzpackage_head(offset);
    if (head != 0) head->dtype = SYS__KIND__PROCEDURE;
}

/* The two halves, read without taking a hold — the procedure holds them and the caller holds the
 * procedure, so a hold here would be a third one nobody releases. Both answer nothing when asked of
 * something that is not a procedure, and both raise saying so rather than returning a plausible cell. */
static __device__ inline sys__heap_node sys__procedure__zzprivate_part(uint64_t offset, uint64_t which) {
    if (!sys__procedure__is(offset)) {
        sys__fault__raise(0ull, SYS__PROCEDURE__FAULT_KIND);
        return sys__heap_node__nothing();
    }
    return *sys__node_array__zzpackage_at(offset, which);
}

static __device__ inline sys__heap_node sys__procedure__parameters(uint64_t offset) {
    return sys__procedure__zzprivate_part(offset, (uint64_t)SYS__PROCEDURE__PARAMETERS);
}

static __device__ inline sys__heap_node sys__procedure__body(uint64_t offset) {
    return sys__procedure__zzprivate_part(offset, (uint64_t)SYS__PROCEDURE__BODY);
}

/* Both halves for one question asked once. ⛳ IT IS `zzengine_` BECAUSE THE CALL EXPANSION IS THE ONLY
 * CALLER AND IT IS THE HOTTEST THING A PROGRAM DOES: a call reads the parameters and the body together,
 * always, so asking what this is twice is asking a settled question again. Package code that wants one
 * half without the other uses the two verbs above. */
static __device__ inline bool sys__procedure__zzengine_halves(uint64_t offset,
                                                              sys__heap_node* parameters,
                                                              sys__heap_node* body) {
    if (!sys__procedure__is(offset)) {
        sys__fault__raise(0ull, SYS__PROCEDURE__FAULT_KIND);
        return false;
    }
    *parameters = *sys__node_array__zzpackage_at(offset, (uint64_t)SYS__PROCEDURE__PARAMETERS);
    *body       = *sys__node_array__zzpackage_at(offset, (uint64_t)SYS__PROCEDURE__BODY);
    return true;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_PROCEDURE_CUH */
