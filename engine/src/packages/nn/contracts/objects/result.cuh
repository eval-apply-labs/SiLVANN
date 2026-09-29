#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_RESULT_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_RESULT_CUH
/* ══ A VALUE A CARD HANDS BACK — ONE STAMPED 64-BIT WORD ══════════════════════════════════════════════
 * ⚖ RULED (shape B): a verb whose answer is a value — argmax, dot_product, at — writes it into
 * an `out` buffer like every other verb and answers that buffer; `nn__buffer__read` turns it into a value
 * when the program needs one. The word is
 *
 *     bits 63..56   the kind:  NN__RESULT__INT or NN__RESULT__FLOAT
 *     bits 55..32   the ticket: which call wrote it
 *     bits 31..0    the value:  a 32-bit index, or a float's bits
 *
 * The verb makes the top half — the STAMP — before it launches and records it on the `out` object; the
 * kernel stores stamp | value in one write; the read waits until the buffer holds the stamp its object
 * records. ⛳ A SENTINEL WOULD RACE: a second call into the same buffer cannot order its own "not yet"
 * against the first call's kernel still landing. A stamp cannot be mistaken for another call's.
 * ⛳ STAMP 0 IS "NOTHING WRITTEN" — no kind is 0 — which is what a fresh buffer or view carries. */

#define NN__RESULT__INT            1ull
#define NN__RESULT__FLOAT          2ull
#define NN__RESULT__KIND_SHIFT     56u
#define NN__RESULT__TICKET_SHIFT   32u
#define NN__RESULT__TICKET_MASK    0xFFFFFFull
#define NN__RESULT__STAMP_MASK     0xFFFFFFFF00000000ull
#define NN__RESULT__BYTES          8ull     /* how much of `out` the word takes */

#define NN__RESULT__FAULT_UNWRITTEN 0x4E525557ull   /* "NRUW" — nothing this buffer was given ever landed  */
#define NN__RESULT__FAULT_TOO_LONG  0x4E52544Cull   /* "NRTL" — an index that does not fit the word's 32 bits */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_RESULT_CUH */
