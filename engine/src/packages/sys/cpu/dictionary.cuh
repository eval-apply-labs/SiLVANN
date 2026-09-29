#ifndef SILVANN__PACKAGES_SYS_CPU_DICTIONARY_CUH
#define SILVANN__PACKAGES_SYS_CPU_DICTIONARY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"       /* how a refusal is said */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh" /* holds, references, and writing a cell */
#include "heap__header.cuh"        /* where the room comes from, and how wide an array granule is */
#include "stack__header.cuh"       /* how a dying tree hands over what it held */
#include "../contracts/objects/dictionary.cuh" /* its constants and fault words */
/* ══ a dictionary — values found by key, in a balanced tree of small sorted leaves ════════════════════
 *
 * ⚖ ARCHITECT: *"make sure that the dictionary is done in the shape of a tree."* Pages linked in order
 * would be a list, and every page past the first would be one more dependent load on every lookup. The
 * reasoning behind the whole structure is in `src/docs/design/dictionary.md`; what is here is the shape that was
 * built, and why each part of it is that shape.
 *
 * ── WHAT ONE IS MADE OF ─────────────────────────────────────────────────────────────────────────────
 *     the dictionary   a head and one node: `args[0]` the ROOT — a leaf, a routing node, or nothing —
 *                      and `args[1]` how many keys it holds
 *     a routing node   a head — `args[1]` LEFT, `args[2]` RIGHT, `args[3]` its HEIGHT — and one node
 *                      that is, whole, the LARGEST KEY ON ITS LEFT. That node is what a reference names.
 *     a leaf           an array granule: the head's `args[1]` says how many pairs, and pair i is the key
 *                      at element 2i and the value at element 2i + 1. Sixteen nodes hold seven pairs.
 * ⚖ ARCHITECT, on the routing node: *"heap nodes being sys__kind__tree_routing_node … the two pointers"*,
 * and on its bound, first as two keys and then as one: *"do we need 3 or can we simply do 2? after all we
 * only care if we are bigger or smaller than the left bit."* ⚖ And on the leaves: *"7 values is enough
 * because otherwise you have to delete and rewrite giant array chunks … an intermediate insert becomes
 * drama every time."*
 *
 * ── THE DESCENT, AND WHY ONE BOUND IS ENOUGH ────────────────────────────────────────────────────────
 * At a routing node a key no larger than the bound goes LEFT and any other goes RIGHT. That rule keeps
 * itself true: a key sent left is no larger than the bound, so the bound is still the left side's
 * largest; a key sent right is compared with nothing the right side stores. ⇒ ⭐ AN INSERT NEVER WRITES A
 * BOUND. A new routing node gets its bound once, from the split that made it, and never again.
 * ⭐⭐ AND A ROTATION NEVER READS ONE. The largest key of a left subtree always lies in that subtree's own
 * right part, and that part travels with the bound when a rotation moves it — so after a rotation, left
 * or right, every node's bound is still the largest key on its left. `REASONED` case by case for both
 * directions; a double rotation is two singles. Rebalancing is pointer swaps and heights, nothing else.
 * ⛳ A bound only has to be an UPPER bound for lookups to stay right, so a key that later leaves the tree
 * cannot make it wrong.
 *
 * ── THE KEY ORDER ───────────────────────────────────────────────────────────────────────────────────
 * A key is a whole node, compared field by field as unsigned numbers — kind, count, opcode, then the six
 * arguments. For a 512-bit hash that is simply a total order over its bytes. For a NUMBER, every field
 * but the first argument is the same in every key this file makes (`sys__dictionary__number_key`), so the
 * order is the number's own — which is what sends increasing symbol ids to the right edge every time.
 * ⛔ COMPARING RAW BYTES WOULD NOT: the bytes are little-endian, so 256 would sort before 1.
 *
 * ── WHEN A LEAF IS FULL ─────────────────────────────────────────────────────────────────────────────
 * ⚖ *"split and place the objects 50/50 when they dont land at the edge, and if they do there is nothing
 * to split. an item on the edge that is against a full chunk means a new chunk and that's it."*
 *     past its last key    a new leaf on the RIGHT holding the new pair, and the full one stays full
 *     before its first     a new leaf on the LEFT holding the new pair
 *     anywhere between     the eight pairs split four and four, the new leaf on the right
 * Either way a routing node takes the leaf's place, with the old leaf and the new one under it, and the
 * path back to the root is rebalanced.
 * ⛔ BOTH ALLOCATIONS ARE TAKEN BEFORE ANYTHING MOVES. The routing node is made first, and if the leaf
 * cannot be made the routing node is let go — so a refused insert leaves the tree exactly as it was.
 *
 * ── BALANCE ─────────────────────────────────────────────────────────────────────────────────────────
 * ⚖ *"go with avl."* A leaf has height zero and a routing node one more than its taller child. After an
 * insert, the path is walked back up: a height that did not change ends the walk, and so does a rotation,
 * because one single or double rotation puts the subtree back at the height it had before the insert.
 * ⛳ THE PATH IS KEPT ON THE WAY DOWN, IN A FIXED ARRAY. `SYS__DICTIONARY__DEPTH_MAX` routing nodes deep
 * needs at least a Fibonacci number of that order of leaves — far more than any heap holds — so the
 * limit is never met by a sound tree, and a descent that reaches it is refused rather than followed.
 *
 * ── OWNERSHIP ───────────────────────────────────────────────────────────────────────────────────────
 * Every link is exactly one hold: the dictionary holds its root and a routing node holds its two
 * children. A rotation moves links and so moves holds, and no count changes. A value is held the way an
 * array holds one — putting it in takes a hold, what it displaces gives one up, and reading one hands
 * back a hold the caller owes.
 * ⛔ A DYING LEAF HANDS OVER ITS VALUES AND NEVER LOOKS AT ITS KEYS. A hash key is raw bytes, and the
 * field a reader would take for its kind is hash — any kind at all can appear there. The parity of the
 * slot is what says which cells can hold a reference.
 *
 * ── READ-ONLY, WHICH IS WHAT LETS ONE BE SHARED ─────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"we do need to have a readonly dictionary too so we dont have to lock when reading, it is
 * necessary for the async compute function so let's add the deep copy and a flag to the head to be read
 * as readonly."* A flag in the head, set once by `seal`, and every write refuses while it is set. Readers
 * never look at it: what makes a sealed dictionary safe to read from sixty blocks at once is that nobody
 * can write it, not anything a reader does.
 * ⛳ THE COPY IS A WALK AND NOT A CLONE OF THE SHAPE: every key in order, put into a new dictionary. It
 * reuses the one verb that builds a tree, and keys arriving in order leave every leaf but the last full.
 *
 * ── WHAT IT DOES NOT DO ─────────────────────────────────────────────────────────────────────────────
 * It takes no lock: a dictionary is written by one owner, and one that is shared is sealed. A removal
 * writes a null value and keeps the key — the spec's cache — and never merges, shrinks or frees anything.
 * Nothing produces a null KEY yet, so nothing reclaims one. Its row refuses a `clone` — the copy is
 * reached by its own name. Keys are unique: putting a key that is there replaces its value.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE PIECES, READ AND WRITTEN ────────────────────────────────────────────────────────────────────
 * A routing node and a leaf are both named by the node after their head, like everything else. These
 * are the only places that know where in them each thing is.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

static __device__ inline bool sys__dictionary__zzprivate_is_routing(uint64_t node) {
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(node),
                                               SYS__KIND__TREE_ROUTING_NODE);
}

static __device__ inline uint64_t sys__dictionary__zzprivate_height(uint64_t node) {
    if (!sys__dictionary__zzprivate_is_routing(node)) return 0ull;
    return sys__heap_node__zzpackage_head(node)->args[SYS__TREE_ROUTING_NODE__HEIGHT];
}

static __device__ inline uint64_t* sys__dictionary__zzprivate_link(uint64_t routing, uint32_t side) {
    return &sys__heap_node__zzpackage_head(routing)->args[side];
}

static __device__ inline void sys__dictionary__zzprivate_refresh_height(uint64_t routing) {
    const uint64_t left  = sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(routing, SYS__TREE_ROUTING_NODE__LEFT));
    const uint64_t right = sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(routing, SYS__TREE_ROUTING_NODE__RIGHT));
    sys__heap_node__zzpackage_head(routing)->args[SYS__TREE_ROUTING_NODE__HEIGHT] = 1ull + (left > right ? left : right);
}

/* Where the pairs of a leaf begin, and how many there are. */
static __device__ inline sys__heap_node* sys__dictionary__zzprivate_pairs(uint64_t leaf) {
    return sys__heap__object_full_address(leaf);
}

static __device__ inline uint64_t* sys__dictionary__zzprivate_used(uint64_t leaf) {
    return &sys__heap_node__zzpackage_head(leaf)->args[SYS__DICTIONARY_LEAF__USED];
}

/* The order every key is kept in — see the top of this file. */
static __device__ inline int sys__dictionary__zzprivate_order(const sys__heap_node* a, const sys__heap_node* b) {
    if (a->dtype    != b->dtype)    return a->dtype    < b->dtype    ? -1 : 1;
    if (a->num_args != b->num_args) return a->num_args < b->num_args ? -1 : 1;
    if (a->op_code  != b->op_code)  return a->op_code  < b->op_code  ? -1 : 1;
    for (uint32_t i = 0u; i < 6u; ++i)
        if (a->args[i] != b->args[i]) return a->args[i] < b->args[i] ? -1 : 1;
    return 0;
}

/* The first pair whose key is not below `key`, and whether that pair's key IS `key`. Seven pairs at most,
 * so a scan: a binary search over seven buys nothing a reader should pay for. */
static __device__ inline uint64_t sys__dictionary__zzprivate_seat(const sys__heap_node* pairs, uint64_t used,
                                                                   const sys__heap_node* key, bool* found) {
    for (uint64_t i = 0ull; i < used; ++i) {
        const int order = sys__dictionary__zzprivate_order(&pairs[2ull * i], key);
        if (order >= 0) { *found = (order == 0); return i; }
    }
    *found = false;
    return used;
}

/* The leaf a key belongs in. Nothing when the dictionary is empty, and nothing — having raised — when the
 * descent runs past any depth a sound tree can have. */
static __device__ inline uint64_t sys__dictionary__zzprivate_leaf_for(uint64_t root, const sys__heap_node* key) {
    uint64_t node = root;
    for (uint32_t depth = 0u; node != 0ull && sys__dictionary__zzprivate_is_routing(node); ++depth) {
        if (depth == SYS__DICTIONARY__DEPTH_MAX) {
            sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_DEEP);
            return 0ull;
        }
        const uint32_t side = sys__dictionary__zzprivate_order(key, sys__heap__object_full_address(node)) <= 0
                            ? SYS__TREE_ROUTING_NODE__LEFT : SYS__TREE_ROUTING_NODE__RIGHT;
        node = *sys__dictionary__zzprivate_link(node, side);
    }
    return node;
}

/* A leaf with nothing in it, and a routing node with nothing under it. Both write every field they own,
 * because a carved chunk is not zeroed — and a routing node let go before it is linked must hand over
 * nothing. */
static __device__ inline uint64_t sys__dictionary__zzprivate_new_leaf(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__DICTIONARY_LEAF);
    if (head == 0) return 0ull;                    /* the heap already raised */
    head->args[SYS__DICTIONARY_LEAF__USED] = 0ull;
    return sys__heap__offset(head + 1);
}

static __device__ inline uint64_t sys__dictionary__zzprivate_new_routing(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__TREE_ROUTING_NODE);
    if (head == 0) return 0ull;                    /* the heap already raised */
    head->args[SYS__TREE_ROUTING_NODE__LEFT]   = 0ull;
    head->args[SYS__TREE_ROUTING_NODE__RIGHT]  = 0ull;
    head->args[SYS__TREE_ROUTING_NODE__HEIGHT] = 1ull;
    return sys__heap__offset(head + 1);
}

/* Write a pair into a cell pair nobody holds anything through: the key is copied, the value is held. */
static __device__ inline void sys__dictionary__zzprivate_place(sys__heap_node* pair, const sys__heap_node* key,
                                                               const sys__heap_node* value) {
    pair[0] = *key;
    pair[1] = *value;
    if (sys__heap_node__carries_reference(value->dtype)) (void)sys__heap_object__retain(value->args[0]);
}

/* ── BALANCE ─────────────────────────────────────────────────────────────────────────────────────────
 * Each rotation answers the node now at the top of the subtree it turned; the caller links that node
 * where the old top was. No bound is read or written — see the top of this file.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

static __device__ inline uint64_t sys__dictionary__zzprivate_rotate_right(uint64_t top) {
    const uint64_t left = *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__LEFT);
    *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__LEFT) =
        *sys__dictionary__zzprivate_link(left, SYS__TREE_ROUTING_NODE__RIGHT);
    *sys__dictionary__zzprivate_link(left, SYS__TREE_ROUTING_NODE__RIGHT) = top;
    sys__dictionary__zzprivate_refresh_height(top);
    sys__dictionary__zzprivate_refresh_height(left);
    return left;
}

static __device__ inline uint64_t sys__dictionary__zzprivate_rotate_left(uint64_t top) {
    const uint64_t right = *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__RIGHT);
    *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__RIGHT) =
        *sys__dictionary__zzprivate_link(right, SYS__TREE_ROUTING_NODE__LEFT);
    *sys__dictionary__zzprivate_link(right, SYS__TREE_ROUTING_NODE__LEFT) = top;
    sys__dictionary__zzprivate_refresh_height(top);
    sys__dictionary__zzprivate_refresh_height(right);
    return right;
}

/* Bring one routing node's height up to date, and rotate if its two sides now differ by two. A side that
 * leans the other way inside is turned first, which is the double rotation. */
static __device__ inline uint64_t sys__dictionary__zzprivate_balance(uint64_t top) {
    sys__dictionary__zzprivate_refresh_height(top);
    const uint64_t left  = *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__LEFT);
    const uint64_t right = *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__RIGHT);
    const int64_t lean = (int64_t)sys__dictionary__zzprivate_height(left)
                       - (int64_t)sys__dictionary__zzprivate_height(right);
    if (lean > 1) {
        if (sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(left, SYS__TREE_ROUTING_NODE__LEFT))
            < sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(left, SYS__TREE_ROUTING_NODE__RIGHT)))
            *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__LEFT) =
                sys__dictionary__zzprivate_rotate_left(left);
        return sys__dictionary__zzprivate_rotate_right(top);
    }
    if (lean < -1) {
        if (sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(right, SYS__TREE_ROUTING_NODE__RIGHT))
            < sys__dictionary__zzprivate_height(*sys__dictionary__zzprivate_link(right, SYS__TREE_ROUTING_NODE__LEFT)))
            *sys__dictionary__zzprivate_link(top, SYS__TREE_ROUTING_NODE__RIGHT) =
                sys__dictionary__zzprivate_rotate_right(right);
        return sys__dictionary__zzprivate_rotate_left(top);
    }
    return top;
}

/* Walk the path back up after a leaf became a routing node. `went_right` holds one bit per level: which
 * link of `path[j]` the descent took. */
static __device__ inline void sys__dictionary__zzprivate_rebalance(sys__heap_node* fields, const uint64_t* path,
                                                                    uint64_t went_right, uint32_t depth) {
    for (uint32_t j = depth; j-- > 0u; ) {
        const uint64_t node   = path[j];
        const uint64_t before = sys__dictionary__zzprivate_height(node);
        const uint64_t top    = sys__dictionary__zzprivate_balance(node);
        if (top != node) {
            if (j == 0u) fields->args[SYS__DICTIONARY__ROOT] = top;
            else *sys__dictionary__zzprivate_link(path[j - 1u], ((went_right >> (j - 1u)) & 1ull)
                                                                ? SYS__TREE_ROUTING_NODE__RIGHT
                                                                : SYS__TREE_ROUTING_NODE__LEFT) = top;
            return;
        }
        if (sys__dictionary__zzprivate_height(node) == before) return;
    }
}

/* ── THE DICTIONARY ──────────────────────────────────────────────────────────────────────────────────
 * Making one, asking it, and changing what it holds. Every verb that is handed a dictionary asks `is`
 * first, and every one that is handed a key or a value refuses a null one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* An empty dictionary. The caller holds it. */
static __device__ inline uint64_t sys__dictionary__create(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__DICTIONARY);
    if (head == 0) return 0ull;                    /* the heap already raised */
    head->args[SYS__DICTIONARY__READONLY] = 0ull;
    head[1].args[SYS__DICTIONARY__ROOT]  = 0ull;
    head[1].args[SYS__DICTIONARY__COUNT] = 0ull;
    return sys__heap__offset(head + 1);
}

/* Whether this names one at all. */
static __device__ inline bool sys__dictionary__is(uint64_t dictionary) {
    if (!sys__heap_object__zzpackage_addressable(dictionary)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(dictionary), SYS__KIND__DICTIONARY);
}

/* The one check every verb below opens with. */
static __device__ inline sys__heap_node* sys__dictionary__zzprivate_fields(uint64_t dictionary) {
    if (!sys__dictionary__is(dictionary)) {
        sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_KIND);
        return (sys__heap_node*)0;
    }
    return sys__heap__object_full_address(dictionary);
}

/* Whether nobody may write it any more. Something that is not a dictionary is not one, and says nothing. */
static __device__ inline bool sys__dictionary__readonly(uint64_t dictionary) {
    return sys__dictionary__is(dictionary)
        && sys__heap_node__zzpackage_head(dictionary)->args[SYS__DICTIONARY__READONLY] != 0ull;
}

/* Make it read-only, for good. There is no way back: a dictionary somebody may be reading without a lock
 * cannot become writable again without every reader knowing, and nothing could tell them. */
static __device__ inline bool sys__dictionary__seal(uint64_t dictionary) {
    if (sys__dictionary__zzprivate_fields(dictionary) == 0) return false;
    sys__heap_node__zzpackage_head(dictionary)->args[SYS__DICTIONARY__READONLY] = 1ull;
    return true;
}

/* The one check both writers make after the ones every verb makes. */
static __device__ inline bool sys__dictionary__zzprivate_writable(uint64_t dictionary) {
    if (sys__heap_node__zzpackage_head(dictionary)->args[SYS__DICTIONARY__READONLY] == 0ull) return true;
    sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_READONLY);
    return false;
}

/* A number as a key: an integer cell with every other field zeroed, so two number keys differ only in
 * their number and the order is the number's. */
static __device__ inline sys__heap_node sys__dictionary__number_key(uint64_t number) {
    sys__heap_node key;
    key.dtype    = SYS__KIND__VALUE_INT;
    key.num_args = 0u;
    key.op_code  = 0ull;
    key.args[0]  = number;
    for (uint32_t i = 1u; i < 6u; ++i) key.args[i] = 0ull;
    return key;
}

/* ⛳ ONE PLACE BUILDS THE KEY, so the two verbs cannot disagree about what a key IS. An INTEGER goes
 * through `number_key`, which zeroes every word the ordering compares — a cell that merely happens to
 * hold the right number in `args[0]` may carry rubbish in the rest, and the comparator reads them.
 * ⇒ ★ A KEY IS A CANONICAL FORM AND NOT "THE CELL THE CALLER PASSED", which is the kind of difference
 * that shows up as a lookup that misses a key it just stored. Anything else is passed as given, which is
 * what lets the hashed string keys the settings reader uses keep working through the same doors. */
static __device__ inline sys__heap_node sys__dictionary__zzpackage_key_of(const sys__heap_node* cell) {
    if (cell->dtype == SYS__KIND__VALUE_INT) return sys__dictionary__number_key(cell->args[0]);
    return *cell;
}

/* How many keys it holds — every key ever put, since a removal keeps its key. Zero, having raised, for
 * something that is not a dictionary. */
static __device__ inline uint64_t sys__dictionary__count(uint64_t dictionary) {
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    return fields == 0 ? 0ull : fields->args[SYS__DICTIONARY__COUNT];
}

/* How many routing nodes stand between the root and the deepest leaf — zero while it is one leaf. */
static __device__ inline uint64_t sys__dictionary__height(uint64_t dictionary) {
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0 || fields->args[SYS__DICTIONARY__ROOT] == 0ull) return 0ull;
    return sys__dictionary__zzprivate_height(fields->args[SYS__DICTIONARY__ROOT]);
}

/* The value a key holds, HELD — the caller gives that hold back. A key that is not there, a key whose
 * value was removed, and a refusal all answer nothing; only the refusal raises. */
static __device__ inline sys__heap_node sys__dictionary__get(uint64_t dictionary, const sys__heap_node* key) {
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0) return sys__heap_node__nothing();
    if (key == 0) {
        sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_NO_KEY);
        return sys__heap_node__nothing();
    }
    const uint64_t leaf = sys__dictionary__zzprivate_leaf_for(fields->args[SYS__DICTIONARY__ROOT], key);
    if (leaf == 0ull) return sys__heap_node__nothing();
    bool found = false;
    const sys__heap_node* pairs = sys__dictionary__zzprivate_pairs(leaf);
    const uint64_t seat = sys__dictionary__zzprivate_seat(pairs, *sys__dictionary__zzprivate_used(leaf), key, &found);
    if (!found) return sys__heap_node__nothing();
    const sys__heap_node value = pairs[2ull * seat + 1ull];
    if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__retain(value.args[0]);
    return value;
}

/* Make `key` mean `value`: a key that is there has its value replaced, and any other is inserted. The
 * dictionary takes a hold of the value and gives up its hold on whatever it displaced. Answers whether
 * it happened; a refusal raises and leaves the dictionary exactly as it was.
 * ⛳ OUT OF LINE: it is the one verb here that can allocate, rotate and walk a path, and it is reached from
 * wherever a name is registered. */
static __device__ __noinline__ bool sys__dictionary__put(uint64_t dictionary, const sys__heap_node* key,
                                                         const sys__heap_node* value) {
    sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0 || !sys__dictionary__zzprivate_writable(dictionary)) return false;
    if (key == 0)   { sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_NO_KEY);   return false; }
    if (value == 0) { sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_NO_VALUE); return false; }

    if (fields->args[SYS__DICTIONARY__ROOT] == 0ull) {
        const uint64_t leaf = sys__dictionary__zzprivate_new_leaf();
        if (leaf == 0ull) return false;
        sys__dictionary__zzprivate_place(sys__dictionary__zzprivate_pairs(leaf), key, value);
        *sys__dictionary__zzprivate_used(leaf) = 1ull;
        fields->args[SYS__DICTIONARY__ROOT]  = leaf;
        fields->args[SYS__DICTIONARY__COUNT] = 1ull;
        return true;
    }

    /* Down to the leaf, keeping the path. */
    uint64_t path[SYS__DICTIONARY__DEPTH_MAX];
    uint64_t went_right = 0ull;
    uint32_t depth = 0u;
    uint64_t node = fields->args[SYS__DICTIONARY__ROOT];
    while (sys__dictionary__zzprivate_is_routing(node)) {
        if (depth == SYS__DICTIONARY__DEPTH_MAX) {
            sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_DEEP);
            return false;
        }
        path[depth] = node;
        uint32_t side = SYS__TREE_ROUTING_NODE__LEFT;
        if (sys__dictionary__zzprivate_order(key, sys__heap__object_full_address(node)) > 0) {
            side = SYS__TREE_ROUTING_NODE__RIGHT;
            went_right |= 1ull << depth;
        }
        node = *sys__dictionary__zzprivate_link(node, side);
        ++depth;
    }

    const uint64_t leaf = node;
    sys__heap_node* pairs = sys__dictionary__zzprivate_pairs(leaf);
    const uint64_t used = *sys__dictionary__zzprivate_used(leaf);
    bool found = false;
    const uint64_t seat = sys__dictionary__zzprivate_seat(pairs, used, key, &found);

    if (found) {                                   /* a key that is there: only its value changes */
        sys__heap_object__set(&pairs[2ull * seat + 1ull], value);
        return true;
    }
    if (used < (uint64_t)SYS__DICTIONARY_LEAF__PAIRS) {
        for (uint64_t i = used; i > seat; --i) {
            pairs[2ull * i]       = pairs[2ull * i - 2ull];
            pairs[2ull * i + 1ull] = pairs[2ull * i - 1ull];
        }
        sys__dictionary__zzprivate_place(&pairs[2ull * seat], key, value);
        *sys__dictionary__zzprivate_used(leaf) = used + 1ull;
        fields->args[SYS__DICTIONARY__COUNT] += 1ull;
        return true;
    }

    /* A full leaf. Both allocations first, so a refusal changes nothing. */
    const uint64_t routing = sys__dictionary__zzprivate_new_routing();
    if (routing == 0ull) return false;
    const uint64_t fresh = sys__dictionary__zzprivate_new_leaf();
    if (fresh == 0ull) {
        (void)sys__heap_object__release(routing);
        return false;
    }
    sys__heap_node* moved = sys__dictionary__zzprivate_pairs(fresh);
    uint64_t left = leaf, right = fresh;
    sys__heap_node* bound = sys__heap__object_full_address(routing);

    if (seat == used) {                            /* past the last key: the full leaf stays full */
        sys__dictionary__zzprivate_place(moved, key, value);
        *sys__dictionary__zzprivate_used(fresh) = 1ull;
        *bound = pairs[2ull * (used - 1ull)];
    } else if (seat == 0ull) {                     /* before the first: the new leaf goes on the left */
        sys__dictionary__zzprivate_place(moved, key, value);
        *sys__dictionary__zzprivate_used(fresh) = 1ull;
        *bound = *key;
        left = fresh; right = leaf;
    } else {                                       /* between: eight pairs, four and four */
        const uint64_t all  = used + 1ull;
        const uint64_t half = all / 2ull;
        /* The right half first, read out of the leaf before anything in it moves. Pair j of the merged
         * run is the old pair j before the seat, the new pair at it, and the old pair j - 1 after it. */
        for (uint64_t j = half; j < all; ++j) {
            sys__heap_node* to = &moved[2ull * (j - half)];
            if (j < seat)       { to[0] = pairs[2ull * j];         to[1] = pairs[2ull * j + 1ull]; }
            else if (j == seat) { sys__dictionary__zzprivate_place(to, key, value); }
            else                { to[0] = pairs[2ull * (j - 1ull)]; to[1] = pairs[2ull * j - 1ull]; }
        }
        if (seat < half) {                         /* the new pair lands in the half that stays */
            for (uint64_t i = half - 1ull; i > seat; --i) {
                pairs[2ull * i]       = pairs[2ull * i - 2ull];
                pairs[2ull * i + 1ull] = pairs[2ull * i - 1ull];
            }
            sys__dictionary__zzprivate_place(&pairs[2ull * seat], key, value);
        }
        *sys__dictionary__zzprivate_used(leaf)  = half;
        *sys__dictionary__zzprivate_used(fresh) = all - half;
        *bound = pairs[2ull * (half - 1ull)];
    }

    *sys__dictionary__zzprivate_link(routing, SYS__TREE_ROUTING_NODE__LEFT)  = left;
    *sys__dictionary__zzprivate_link(routing, SYS__TREE_ROUTING_NODE__RIGHT) = right;
    if (depth == 0u) fields->args[SYS__DICTIONARY__ROOT] = routing;
    else *sys__dictionary__zzprivate_link(path[depth - 1u], ((went_right >> (depth - 1u)) & 1ull)
                                                            ? SYS__TREE_ROUTING_NODE__RIGHT
                                                            : SYS__TREE_ROUTING_NODE__LEFT) = routing;
    fields->args[SYS__DICTIONARY__COUNT] += 1ull;
    sys__dictionary__zzprivate_rebalance(fields, path, went_right, depth);
    return true;
}

/* Take a key's value away: the value's hold is given up and a null takes its place, and the KEY STAYS —
 * putting it again is a write and not an insert. Answers whether there was a value to take. A key that
 * is not there is not a mistake and raises nothing. */
static __device__ inline bool sys__dictionary__remove(uint64_t dictionary, const sys__heap_node* key) {
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0 || !sys__dictionary__zzprivate_writable(dictionary)) return false;
    if (key == 0) {
        sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_NO_KEY);
        return false;
    }
    const uint64_t leaf = sys__dictionary__zzprivate_leaf_for(fields->args[SYS__DICTIONARY__ROOT], key);
    if (leaf == 0ull) return false;
    bool found = false;
    sys__heap_node* pairs = sys__dictionary__zzprivate_pairs(leaf);
    const uint64_t seat = sys__dictionary__zzprivate_seat(pairs, *sys__dictionary__zzprivate_used(leaf), key, &found);
    if (!found || pairs[2ull * seat + 1ull].dtype == SYS__KIND__VALUE_NULL) return false;
    const sys__heap_node nothing = sys__heap_node__nothing();
    sys__heap_object__set(&pairs[2ull * seat + 1ull], &nothing);
    return true;
}

/* ── READING WITHOUT A HOLD, IN ORDER, AND WHOLE ──────────────────────────────────────────────────────
 * The two borrowed readers hand back a value and take nothing, so they are for a dictionary nobody writes
 * while the answer is in use: a sealed one, or one whose only writer is the caller.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* The value a key holds, BORROWED, and whether the key is there at all — which a null answer cannot say,
 * because a null is a value a key may hold. A key that is not there raises nothing. */
static __device__ inline sys__heap_node sys__dictionary__zzpackage_find(uint64_t dictionary,
                                                                       const sys__heap_node* key, bool* found) {
    if (found == 0) return sys__heap_node__nothing();
    *found = false;
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0) return sys__heap_node__nothing();
    if (key == 0) {
        sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_NO_KEY);
        return sys__heap_node__nothing();
    }
    const uint64_t leaf = sys__dictionary__zzprivate_leaf_for(fields->args[SYS__DICTIONARY__ROOT], key);
    if (leaf == 0ull) return sys__heap_node__nothing();
    const sys__heap_node* pairs = sys__dictionary__zzprivate_pairs(leaf);
    const uint64_t seat = sys__dictionary__zzprivate_seat(pairs, *sys__dictionary__zzprivate_used(leaf), key, found);
    return *found ? pairs[2ull * seat + 1ull] : sys__heap_node__nothing();
}

/* The first pair whose key comes after `key` — or the first pair of all, when `key` is null — copied out
 * with its value BORROWED. Answers whether there was one.
 * ⛳ ONE DESCENT, NOT A STACK OF THEM: going left past a bound remembers the right side as where the next key
 * is if the left has nothing after `key`, and the deepest such side is the nearest. Leaves are never
 * empty — a leaf is made holding a pair and a removal keeps its key — so the first pair of the leftmost leaf
 * under that side is the answer. */
static __device__ inline bool sys__dictionary__zzpackage_after(uint64_t dictionary, const sys__heap_node* key,
                                                               sys__heap_node* next_key, sys__heap_node* next_value) {
    const sys__heap_node* fields = sys__dictionary__zzprivate_fields(dictionary);
    if (fields == 0 || next_key == 0 || next_value == 0) return false;
    uint64_t node = fields->args[SYS__DICTIONARY__ROOT];
    uint64_t later = 0ull;
    for (uint32_t depth = 0u; node != 0ull && sys__dictionary__zzprivate_is_routing(node); ++depth) {
        if (depth == SYS__DICTIONARY__DEPTH_MAX) {
            sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_DEEP);
            return false;
        }
        if (key == 0 || sys__dictionary__zzprivate_order(key, sys__heap__object_full_address(node)) < 0) {
            later = *sys__dictionary__zzprivate_link(node, SYS__TREE_ROUTING_NODE__RIGHT);
            node  = *sys__dictionary__zzprivate_link(node, SYS__TREE_ROUTING_NODE__LEFT);
        } else {
            node  = *sys__dictionary__zzprivate_link(node, SYS__TREE_ROUTING_NODE__RIGHT);
        }
    }
    if (node == 0ull) return false;
    const sys__heap_node* pairs = sys__dictionary__zzprivate_pairs(node);
    const uint64_t used = *sys__dictionary__zzprivate_used(node);
    for (uint64_t i = 0ull; i < used; ++i) {
        if (key == 0 || sys__dictionary__zzprivate_order(&pairs[2ull * i], key) > 0) {
            *next_key   = pairs[2ull * i];
            *next_value = pairs[2ull * i + 1ull];
            return true;
        }
    }
    if (later == 0ull) return false;
    for (uint32_t depth = 0u; sys__dictionary__zzprivate_is_routing(later); ++depth) {
        if (depth == SYS__DICTIONARY__DEPTH_MAX) {
            sys__fault__raise(0ull, SYS__DICTIONARY__FAULT_DEEP);
            return false;
        }
        later = *sys__dictionary__zzprivate_link(later, SYS__TREE_ROUTING_NODE__LEFT);
    }
    const sys__heap_node* first = sys__dictionary__zzprivate_pairs(later);
    *next_key   = first[0];
    *next_value = first[1];
    return true;
}

/* A new dictionary holding the same keys and values — each value held once more — and writable, whatever
 * the source is. The caller holds it. A refusal answers nothing, having raised, and leaves nothing behind. */
static __device__ __noinline__ uint64_t sys__dictionary__copy(uint64_t dictionary) {
    if (sys__dictionary__zzprivate_fields(dictionary) == 0) return 0ull;
    const uint64_t copy = sys__dictionary__create();
    if (copy == 0ull) return 0ull;
    sys__heap_node at, key, value;
    bool first = true;
    while (sys__dictionary__zzpackage_after(dictionary, first ? (const sys__heap_node*)0 : &at, &key, &value)) {
        first = false;
        at = key;
        if (!sys__dictionary__put(copy, &key, &value)) {
            (void)sys__heap_object__release(copy);
            return 0ull;
        }
    }
    return copy;
}

/* ── THE ROW METHODS ─────────────────────────────────────────────────────────────────────────────────
 * What each of the three kinds hands over when its last hold goes. None of them lets go of anything
 * itself; each moves what it held onto the chain the release loop drains.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

static __device__ __noinline__ void sys__dictionary__zzpackage_release_internal(sys__heap_node* head,
                                                                               uint64_t* releaser_stack) {
    if (head == 0) return;
    const uint64_t root = head[1].args[SYS__DICTIONARY__ROOT];
    head[1].args[SYS__DICTIONARY__ROOT] = 0ull;
    if (root == 0ull) return;
    sys__heap_node reference = sys__heap_object__reference_to(root);
    sys__stack__transfer(releaser_stack, &reference);
}

static __device__ __noinline__ void sys__tree_routing_node__zzpackage_release_internal(sys__heap_node* head,
                                                                                      uint64_t* releaser_stack) {
    if (head == 0) return;
    const uint32_t sides[2] = { SYS__TREE_ROUTING_NODE__LEFT, SYS__TREE_ROUTING_NODE__RIGHT };
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint64_t child = head->args[sides[i]];
        head->args[sides[i]] = 0ull;
        if (child == 0ull) continue;               /* made and let go before it was linked */
        sys__heap_node reference = sys__heap_object__reference_to(child);
        sys__stack__transfer(releaser_stack, &reference);
    }
}

/* ⛔ THE VALUES ONLY, BY PARITY — never a key. See the top of this file. */
static __device__ __noinline__ void sys__dictionary_leaf__zzpackage_release_internal(sys__heap_node* head,
                                                                                    uint64_t* releaser_stack) {
    if (head == 0) return;
    sys__heap_node* pairs = head + 1;
    const uint64_t used = head->args[SYS__DICTIONARY_LEAF__USED];
    head->args[SYS__DICTIONARY_LEAF__USED] = 0ull;
    for (uint64_t i = 0ull; i < used; ++i)
        if (sys__heap_node__carries_reference(pairs[2ull * i + 1ull].dtype))
            sys__stack__transfer(releaser_stack, &pairs[2ull * i + 1ull]);
}

/* Building one from a program: it takes nothing and arrives empty. */
static __device__ inline sys__heap_node sys__dictionary__zzpackage_construct(sys__heap_node* base,
                                                                            const sys__heap_node* parameters) {
    (void)base; (void)parameters;
    const uint64_t at = sys__dictionary__create();
    if (at == 0ull) return sys__heap_node__nothing();    /* the making already raised */
    return sys__heap_object__reference_to(at);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_DICTIONARY_CUH */
