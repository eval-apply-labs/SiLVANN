#ifndef SILVANN__PACKAGES_SYS_CPU_ERROR_CUH
#define SILVANN__PACKAGES_SYS_CPU_ERROR_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh"
#include "heap__header.cuh"
#include "../contracts/objects/error.cuh" /* its constants and fault words */
/* ══ what a computation hands back when it went wrong ════════════════════════════════════════════════
 *
 * An error is a VALUE. It is an object like any other — carved from the objects provider, counted, held,
 * released, handed over — and the only thing that makes it an error is its kind. There is no second
 * channel for failure and nothing has to be correlated against anything: a computation's result slot
 * holds what the program BECAME, and what it became may be an exception.
 *
 * ⭐ WHICH IS WHY THIS FILE IS SMALL AND WHY THAT IS THE POINT. A return code needs a convention at every
 * boundary it crosses; a value needs none, because every mechanism that already moves values moves this
 * one. `read_result` does not look at the kind. `release` takes it apart through the same row every kind
 * has. A list may hold one. Nothing was added anywhere to make that true.
 *
 * ⛳ IT CARRIES THE FAULT WORD IT WAS RAISED WITH, so what went wrong travels WITH the thing that went
 * wrong. The fault channel still gets the raise — a log is how you find out that something happened at
 * all — but a caller holding the object does not have to go looking for a matching entry to find out what.
 *
 * ⛔ IT HOLDS NOTHING ELSE, AND SO ITS ROW IS THE DEFAULT ONE. An error owes its own room and nothing
 * more, which is exactly what `sys__heap_object__release__default` does — so there is no teardown method
 * here, and there should not be one.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛳ THE NAME SAYS THE WORD AND THE CHECK IS THE NODE, and the tree asks those as two separate questions:
 * `sys__node_array__zzpackage_construct` raises the shared NO_VALUE when the parameter node itself is
 * null, and `sys__node_array__create` raises its own NARE when the count inside it is zero. Only the
 * first question is asked here. `REASONED` from the one caller: the create verb builds its parameter
 * node as a stack local and zero-fills every argument the form did not supply, so a program that asks
 * for an error and names no fault word arrives with `args[0] == 0` — it gets an error carrying code 0
 * and a raise of code 0, and the null node ERNC names is a shape no program reaches. */

/* Put one in the heap and say nothing. It is the making without the telling, and it exists for the one
 * caller shape that needs an error as a VALUE rather than as a report: a container that keeps one to
 * answer with, made once when the container is, long before anybody asks it anything.
 *
 * ⛳ IT IS PACKAGE-ONLY, AND THAT IS THE WHOLE OF WHY IT IS SAFE TO HAVE. Skipping the log at a real
 * failure is how a run goes wrong with nothing to read afterwards, so the quiet door is not offered to
 * callers reporting one — it is offered to code that is not reporting anything yet. */
static __device__ inline uint64_t sys__error__zzpackage_place(uint64_t fault) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__ERROR);
    if (head == 0) return 0ull;                    /* the heap already raised; there is nothing to add */
    head[1].args[SYS__ERROR__CODE] = fault;
    return sys__heap__offset(head + 1);
}

/* Make one, and say so. The fault word is the whole of what it carries.
 *
 * ⛳ IT RAISES AS WELL AS RETURNING, and the two are not redundant: the raise puts it in the log where
 * somebody watching a run will see it, the object puts it in the hands of whoever asked for the
 * computation. Either one alone leaves half the story with the wrong reader. */
static __device__ inline uint64_t sys__error__make(uint64_t fault) {
    const uint64_t at = sys__error__zzpackage_place(fault);
    if (at == 0ull) return 0ull;                   /* the placing already raised whatever went wrong */
    sys__fault__raise(0ull, fault);
    return at;
}

/* What a program gets when it asks for an error: the fault comes out of the parameters, and the offset
 * the C verb answers becomes a reference. ⛳ THE PARAMETERS ARE READ HERE AND NOWHERE ELSE, which is what
 * a constructor column is for — the C verb above takes a plain number and never learns it came from a
 * program. */
static __device__ inline sys__heap_node sys__error__zzpackage_construct(sys__heap_node* base,
                                                                    const sys__heap_node* parameters) {
    (void)base;
    if (parameters == 0) { sys__fault__raise(0ull, SYS__ERROR__FAULT_NO_CODE); return sys__heap_node__nothing(); }
    const uint64_t at = sys__error__make(parameters->args[0]);
    if (at == 0ull) return sys__heap_node__nothing();          /* the making already raised */
    return sys__heap_object__reference_to(at);
}

/* Whether a reference names one, and what it says. Both read and neither takes anything: an error is
 * immutable once made, so there is nothing to exclude anybody from. */
static __device__ inline bool sys__error__is(uint64_t offset) {
    if (!sys__heap_object__zzpackage_addressable(offset)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(offset), SYS__KIND__ERROR);
}

static __device__ inline uint64_t sys__error__code(uint64_t offset) {
    if (!sys__error__is(offset)) return 0ull;
    return sys__heap__object_full_address(offset)->args[SYS__ERROR__CODE];
}

#endif /* SILVANN__PACKAGES_SYS_CPU_ERROR_CUH */
