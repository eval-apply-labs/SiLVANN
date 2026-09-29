#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_NODE_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_NODE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kind.cuh"   /* what a thing IS — the first word of every node */

/* ══ THE NODE ════════════════════════════════════════════════════════════════════════════════════════
 *
 * Every value in this language is one of these. A kind, a count, an opcode and six words — and what the
 * six mean is the kind's business, which is why nothing beyond them is named here.
 *
 * ⭐⭐ SIXTY-FOUR BYTES IS THE ALLOCATION UNIT AND NOT A ROUNDING. The heap hands out nodes and a chunk
 * is a run of them, so an offset is a NODE count — `sys__heap__offset` is a pointer difference — and the
 * predicates that take one apart mask it against `SYS__HEAP__CHUNK_NODES`, never against a byte width;
 * the one place that turns an offset into a chunk NUMBER, `sys__heap__chunk_of`, divides, and says beside
 * itself why it is not a mask. ⇒ THE ADDRESSING RESTS ON A CHUNK BEING A POWER-OF-TWO MANY NODES, NOT ON
 * THIS WIDTH — which is exactly why the width has to be stated HERE: it is what a node costs, and no
 * addressing downstream would object to it changing. The fields come to exactly 64, so the alignment adds
 * nothing today; it is stated anyway, because it is what a seventh word would silently double.
 *
 * ⭐ THE ASSERT IS THE POINT OF THE FILE AS MUCH AS THE STRUCT IS. A node that grows past 64 does not
 * fail — it doubles the pool, whose bytes are a node count times this size, and doubles every copy,
 * because a node travels by value. ⛳ AND WHAT MAKES IT SILENT IS THE HALF THAT DOES NOT MOVE: a chunk's
 * capacity is a node count, so it still holds exactly `SYS__HEAP__CHUNK_NODES` of them and every figure
 * measured in nodes reads as before. Only the byte footprint changes, and every test still passes.
 * Nothing else in the tree would say so.
 *
 * ⛳ ONE PACKAGE FILE DOES MULTIPLY BY THIS WIDTH, AND IT CARRIES AN ASSERT OF ITS OWN. `string.cuh` holds
 * text, which is bytes, and its hash is exactly one node — so it needs the width, and it pins it: a node of
 * another width stops the build there as well as here. Everywhere else in the package, a byte count
 * beside a node count is the defect this paragraph describes.
 *
 * ⛳ AND THE KIND IS AN ENUM, so it costs this package four bytes and no layout. It has its own file
 * because its row list generates three things this struct is not one of — the enum the device switches
 * on, the table the C interface publishes, and through that the map a composer builds its programs out
 * of. This struct is just one more reader of the type, like everything else that asks a node what it
 * is. */

typedef struct alignas(64) {
    sys__kind dtype;
    uint32_t       num_args;
    uint64_t       op_code;
    uint64_t       args[6];
} sys__heap_node;

static_assert(sizeof(sys__heap_node) == 64,
              "sys__heap_node must stay 64 B — the heap hands out nodes and a chunk is a run of them, so this "
              "width is the allocation unit the addressing is built on. A seventh word makes it 128 and "
              "nothing else in the tree will tell you.");
static_assert(alignof(sys__heap_node) == 64,
              "no address in this package is ever masked, so no addressing depends on this — what depends on it "
              "is the assert above: the alignment is what makes a seventh word jump straight to 128 instead of "
              "creeping to 72. The two hold each other up.");

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_NODE_CUH */
