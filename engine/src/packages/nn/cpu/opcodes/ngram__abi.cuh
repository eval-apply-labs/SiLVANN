#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI_CUH
/* ══ AN N-GRAM MEMORY'S ROWS, READ INSIDE THE PROGRAM ══════════════════════════════════════════════════════════════
 * ▶ `contracts/objects/ngram.cuh` for the verb, its table and why it is one. The hash is the models' own in integers —
 * Flash-Next's (`ngram_qwen4.py`) and DeepSeek V4's (`engram_ds4.py`) — and a row's values are dequantised as theirs
 * are, in floats, then made halves as nn makes every half, so the rows are the host's bit for bit. */

#include <stdlib.h>
#include <string.h>
#include "../doors.cuh"
#include "../../contracts/objects/ngram.cuh"
#include "../../../sys/cpu/silicon/file.cuh"   /* a handle's ordinary descriptor: the reads go through the system's cache */

/* An FP8 E4M3 (fn) byte's value — bias 7, no infinities, 0x7F and 0xFF not a number (unsigned, as numpy's table has
 * them) — each value exact in a float. */
static inline float nn__ngram__zzprivate_e4m3(uint8_t b) {
    const uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
    uint32_t bits;
    if ((b & 0x7Fu) == 0x7Fu) {
        bits = 0x7FC00000u;
    } else if (e == 0u) {
        const float v = (float)m * 0.001953125f;                 /* m · 2⁻⁹: m / 8 · 2⁻⁶, exact */
        memcpy(&bits, &v, 4);
        bits |= (uint32_t)(b & 0x80u) << 24;
    } else {
        bits = ((e + 120u) << 23) | (m << 20) | ((uint32_t)(b & 0x80u) << 24);   /* (1 + m / 8) · 2^(e − 7) */
    }
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

/* An E8M0 byte's scale, 2^(s − 127), as numpy's `ldexp` of a float makes it: 0 the smallest subnormal power, 255 past
 * a float's range. */
static inline float nn__ngram__zzprivate_e8m0(uint8_t s) {
    const uint32_t bits = s == 0u ? 0x00400000u : s == 255u ? 0x7F800000u : (uint32_t)s << 23;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

/* One reader's share of a position's rows: every row whose index is `first` modulo the readers, values and scales. */
typedef struct nn__ngram__share {
    int data_fd, scale_fd;
    uint64_t data_at, scale_at, row_bytes, scale_bytes, rows, first;
    const uint64_t* ids;
    uint8_t *values, *scales;
    bool ok;
} nn__ngram__share;

static bool nn__ngram__zzprivate_read(int fd, void* to, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0ull;
    while (done < bytes) {
        const int64_t got = sys__os__file_read_at(fd, (char*)to + done, (size_t)(bytes - done), offset + done);
        if (got <= 0) return false;
        done += (uint64_t)got;
    }
    return true;
}

static void* nn__ngram__zzprivate_reader(void* arg) {
    nn__ngram__share* s = (nn__ngram__share*)arg;
    s->ok = true;
    for (uint64_t j = s->first; j < s->rows && s->ok; j += NN__NGRAM__READERS) {
        s->ok = nn__ngram__zzprivate_read(s->data_fd, s->values + j * s->row_bytes, s->row_bytes,
                                          s->data_at + s->ids[j] * s->row_bytes);
        if (s->ok && s->scale_bytes != 0ull)
            s->ok = nn__ngram__zzprivate_read(s->scale_fd, s->scales + j * s->scale_bytes, s->scale_bytes,
                                              s->scale_at + s->ids[j] * s->scale_bytes);
    }
    return 0;
}

static inline int64_t nn__ngram__zzprivate_int(uint64_t table, uint64_t i, bool* ok) {
    const sys__heap_node n = sys__node_array__borrow(table, i);
    if (n.dtype != SYS__KIND__VALUE_INT) *ok = false;
    return (int64_t)n.args[0];
}

/* `(nn__ngram__rows tok back table out)` -> `out`. ▶ `contracts/objects/ngram.cuh`. */
static sys__heap_node nn__ngram__zzabi_apply_rows(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t back_at = 0, back_room = 0, out_at = 0, out_room = 0;
    if (argv[0].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[1], &back_at, &back_room)
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !nn__primitives__room(&argv[3], &out_at, &out_room))
        return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    const uint64_t table = argv[2].args[0];
    if (sys__node_array__length(table) <= NN__NGRAM__MULT) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    bool ok = true;
    const int64_t N = nn__ngram__zzprivate_int(table, NN__NGRAM__N, &ok), H = nn__ngram__zzprivate_int(table, NN__NGRAM__HEADS, &ok);
    const int64_t row_bytes = nn__ngram__zzprivate_int(table, NN__NGRAM__ROW_BYTES, &ok);
    const uint64_t file = (uint64_t)nn__ngram__zzprivate_int(table, NN__NGRAM__FILE, &ok);
    const uint64_t scale_file = (uint64_t)nn__ngram__zzprivate_int(table, NN__NGRAM__SCALE_FILE, &ok);
    if (!ok || N < 2 || N > (int64_t)NN__NGRAM__N_MAX || H < 1 || H > (int64_t)NN__NGRAM__HEADS_MAX || row_bytes < 1
     || row_bytes > (int64_t)NN__NGRAM__ROW_MAX || (scale_file != 0ull && row_bytes % 32 != 0) || file == 0ull
     || sys__node_array__length(table) < NN__NGRAM__MULT + (uint64_t)(N + 2 * (N - 1) * H))
        return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    const uint64_t rows = (uint64_t)((N - 1) * H), values = rows * (uint64_t)row_bytes;
    if (back_room < 4ull * (uint64_t)(N - 1) || !nn__primitives__fits(values, out_room))
        return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    const int64_t stop = nn__ngram__zzprivate_int(table, NN__NGRAM__STOP, &ok), fill = nn__ngram__zzprivate_int(table, NN__NGRAM__FILL, &ok);
    const sys__heap_node map = sys__node_array__borrow(table, NN__NGRAM__MAP);
    const bool mapped = sys__heap_node__carries_reference(map.dtype) && sys__node_array__is(map.args[0]);
    if (!mapped && !(map.dtype == SYS__KIND__VALUE_INT && map.args[0] == 0ull)) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    float scale = 1.0f;
    if (scale_file == 0ull) {
        const sys__heap_node s = sys__node_array__borrow(table, NN__NGRAM__SCALE);
        if (s.dtype != SYS__KIND__VALUE_FLOAT) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
        scale = (float)sys__heap_node__real(s.args[0]);
    }
    /* ⭐ THE LOOK BACK: the position's token, then the ones before it newest first — `back` holds them oldest first */
    int32_t back[NN__NGRAM__N_MAX];
    if (!sys__gpu__memory_read(ctx->family, back, (const void*)(uintptr_t)back_at, 4u * (size_t)(N - 1)))
        return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    int64_t v[NN__NGRAM__N_MAX];
    bool dead = false;
    for (int64_t s = 0; s < N; ++s) {
        const int64_t t = s == 0 ? (int64_t)argv[0].args[0] : (int64_t)back[N - 1 - s];
        dead = dead || t < 0 || (s >= 1 && stop >= 0 && t == stop);
        if (dead) { v[s] = fill; continue; }
        if (!mapped) { v[s] = t; continue; }
        if ((uint64_t)t >= sys__node_array__length(map.args[0])) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
        const sys__heap_node m = sys__node_array__borrow(map.args[0], (uint64_t)t);
        if (m.dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
        v[s] = (int64_t)m.args[0];
    }
    /* ⭐ THE HASH, in 64-bit integers as numpy's: each token times its multiplier, xored one after another; each n-gram
     *   length's rolling value modulo each head's size (a floor's remainder, never negative), offset into the table */
    uint64_t ids[NN__NGRAM__ROWS_MAX];
    int64_t rolling = 0;
    for (int64_t s = 0; s < N && ok; ++s) {
        const int64_t p = (int64_t)((uint64_t)v[s] * (uint64_t)nn__ngram__zzprivate_int(table, NN__NGRAM__MULT + (uint64_t)s, &ok));
        rolling = s == 0 ? p : (rolling ^ p);
        if (s == 0) continue;
        for (int64_t h = 0; h < H; ++h) {
            const uint64_t j = (uint64_t)((s - 1) * H + h);
            const int64_t size = nn__ngram__zzprivate_int(table, NN__NGRAM__MULT + (uint64_t)N + j, &ok);
            const int64_t off = nn__ngram__zzprivate_int(table, NN__NGRAM__MULT + (uint64_t)N + rows + j, &ok);
            if (size <= 0 || off < 0) { ok = false; break; }
            int64_t r = rolling % size;
            if (r < 0) r += size;
            ids[j] = (uint64_t)(r + off);
        }
    }
    if (!ok) return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    /* ⭐ THE ROWS, every read in flight at once: NN__NGRAM__READERS threads, each a row in that many */
    const uint64_t scale_bytes = scale_file != 0ull ? (uint64_t)row_bytes / 32u : 0ull;
    uint8_t* raw = (uint8_t*)malloc((size_t)(values + rows * scale_bytes));
    uint16_t* halves = (uint16_t*)malloc((size_t)(2u * values));
    if (raw == 0 || halves == 0) { free(raw); free(halves); return sys__engine__abi__error(NN__NGRAM__FAULT_READ); }
    nn__ngram__share share[NN__NGRAM__READERS];
    sys__os__thread th[NN__NGRAM__READERS];
    bool started[NN__NGRAM__READERS];
    const uint64_t data_at = (uint64_t)nn__ngram__zzprivate_int(table, NN__NGRAM__DATA_AT, &ok);
    const uint64_t scale_at = (uint64_t)nn__ngram__zzprivate_int(table, NN__NGRAM__SCALE_AT, &ok);
    for (unsigned r = 0u; r < NN__NGRAM__READERS; ++r) {
        nn__ngram__share* s = &share[r];
        s->data_fd = sys__file__zzabi_descriptor(file);
        s->scale_fd = scale_file != 0ull ? sys__file__zzabi_descriptor(scale_file) : -1;
        s->data_at = data_at; s->scale_at = scale_at; s->row_bytes = (uint64_t)row_bytes; s->scale_bytes = scale_bytes;
        s->rows = rows; s->first = r; s->ids = ids; s->values = raw; s->scales = raw + values; s->ok = false;
        /* the first share is this thread's own, read while the others are */
        started[r] = r > 0u && (uint64_t)r < rows && sys__os__thread_start(&th[r], nn__ngram__zzprivate_reader, s);
    }
    for (unsigned r = 0u; r < NN__NGRAM__READERS; ++r)            /* a share no thread took is read here */
        if (!started[r] && (uint64_t)r < rows) (void)nn__ngram__zzprivate_reader(&share[r]);
    for (unsigned r = 0u; r < NN__NGRAM__READERS; ++r)
        if (started[r]) sys__os__thread_join(th[r]);
    for (unsigned r = 0u; r < NN__NGRAM__READERS && (uint64_t)r < rows; ++r) ok = ok && share[r].ok;
    /* ⭐ DEQUANTISED AS THE HOST DOES: a value times its scale in floats, then the half nearest */
    bool over = false;
    for (uint64_t j = 0u; ok && j < rows; ++j)
        for (uint64_t i = 0u; i < (uint64_t)row_bytes; ++i) {
            const float s = scale_bytes != 0ull ? nn__ngram__zzprivate_e8m0(raw[values + j * scale_bytes + i / 32u]) : scale;
            halves[j * (uint64_t)row_bytes + i] = nn__primitives__zzpackage_float_to_half(
                nn__ngram__zzprivate_e4m3(raw[j * (uint64_t)row_bytes + i]) * s, &over);
        }
    ok = ok && sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)out_at, halves, (size_t)(2u * values));
    free(raw);
    free(halves);
    if (!ok) return sys__engine__abi__error(NN__NGRAM__FAULT_READ);
    /* ⭐ `back` MOVED ON: the oldest out, the position's token in */
    for (int64_t s = 0; s + 1 < N - 1; ++s) back[s] = back[s + 1];
    back[N - 2] = (int32_t)(int64_t)argv[0].args[0];
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)back_at, back, 4u * (size_t)(N - 1)))
        return sys__engine__abi__error(NN__NGRAM__FAULT_TABLE);
    if (over) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_OVERFLOW);
    return nn__doors_answer(&argv[3]);
}
SYS__ENGINE__ABI__BRIDGE(nn__ngram__zzabi_adapter_rows, nn__ngram__zzabi_apply_rows)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_NGRAM__ABI_CUH */
