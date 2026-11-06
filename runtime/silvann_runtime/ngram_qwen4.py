"""Qwen3.8-Flash-Next's n-gram embedding on the host (`qwen4_exp`, NN-141): a position's 16 row ids and the rows they
name, from the table a pack carries in `ngram/` (▶ `python/nn_ngram_table.py`).

At a position the bigram and the trigram ending there are hashed, each into 8 heads: the tokens a step back times the
layer's multipliers, xored, then modulo each head's prime and offset into the one table. A token across an end of text
(`eos`) takes no part — it reads as `eos`, as do the places before the first position — so a conversation's segments
hash apart. The rows are FP8 E4M3, one scale for the table. HF's own is `Qwen4ExpTextNGramEmbedding`
(`transformers.models.qwen4_exp`), which `test/src_ngram_qwen4.py` checks this against.
"""
import json
import os

import numpy as np

SLICES = 64                  # the reads a call keeps in flight


class NGram:
    """The n-gram table of a Flash-Next pack: `ids(tokens, pos)` a position's row ids, `rows(ids)` the rows."""

    def __init__(self, folder):
        t = json.load(open(os.path.join(folder, "ngram", "table.json")))
        if t.get("format") != "silvann-ngram-v1":
            raise ValueError("%s/ngram is %r, not silvann-ngram-v1" % (folder, t.get("format")))
        self.layer, self.width, self.scale, self.eos = int(t["layer"]), int(t["width"]), np.float32(t["scale"]), int(t["eos"])
        self.N, self.per = int(t["ngram_size"]), int(t["heads_per_ngram"])
        self.mult = np.array(t["multipliers"], dtype=np.int64)                      # a token `s` steps back
        self.sizes = np.array(t["vocab_sizes"], dtype=np.int64).reshape(self.N - 1, self.per)
        self.offsets = np.array(t["offsets"], dtype=np.int64).reshape(self.N - 1, self.per)
        self.cols = (self.N - 1) * self.per
        self.rows_n = int(t["rows"])
        path = os.path.join(folder, "ngram", "rows.e4m3")
        if os.path.getsize(path) != self.rows_n * self.width:
            raise ValueError("%s is %d bytes, not %d rows of %d" % (path, os.path.getsize(path), self.rows_n, self.width))
        self.fd = os.open(path, os.O_RDONLY)
        # ⭐ A POSITION'S ROWS READ ALL AT ONCE, as DeepSeek's engram learned: each is a random place in 51 GB, and read
        #   one after another each waits on the drive. MEASURED node03 2026-10-31, the table on the SATA SSD, cold
        #   (posix_fadvise DONTNEED): 16 rows through a map 8.0 ms a position (median), as 16 preads at once 0.86 ms
        #   (measurements/2026-10-31_nn141_flash_next_node03.md).
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=SLICES)
        from nn_compressor import E4M3
        self.e4m3 = E4M3

    def _back(self, tokens, pos):
        """`[positions][N]`: each position's token and those before it in its segment, `eos` where there are none."""
        tok = np.asarray(tokens, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        back = np.empty((len(pos), self.N), dtype=np.int64)
        alive = np.ones(len(pos), dtype=bool)
        back[:, 0] = tok[pos]
        for s in range(1, self.N):
            p = pos - s
            got = tok[np.maximum(p, 0)]
            alive &= p >= 0
            alive &= np.where(p >= 0, got != self.eos, False)              # an end of text before ends the look back
            back[:, s] = np.where(alive, got, self.eos)
        return back

    def chunk_ids(self, tokens, first, n):
        """The row ids of positions `first` .. `first + n - 1`, `[position][cols]`: the bigram's heads, then the trigram's."""
        back = self._back(tokens[:first + n], np.arange(first, first + n))
        products = back * self.mult
        rolling, out = products[:, 0], []
        for k in range(1, self.N):
            rolling = np.bitwise_xor(rolling, products[:, k])
            out.append(rolling[:, None] % self.sizes[k - 1][None, :] + self.offsets[k - 1][None, :])
        return np.concatenate(out, axis=1)

    def ids(self, tokens, pos):
        """Position `pos`'s row ids, over `tokens` (the conversation's ids from position 0)."""
        return self.chunk_ids(tokens[:pos + 1], pos, 1)[0]

    def rows(self, ids):
        """The rows `ids` names, dequantised, end to end: `len(ids) · width` floats — every read in flight at once, a
        slice of them a thread."""
        offsets = (np.asarray(ids, dtype=np.int64).reshape(-1) * self.width).tolist()
        step = max(1, -(-len(offsets) // SLICES))
        w, fd = self.width, self.fd
        parts = [self.pool.submit(lambda o: b"".join(os.pread(fd, w, x) for x in o), offsets[a:a + step])
                 for a in range(0, len(offsets), step)]
        b = np.frombuffer(b"".join(f.result() for f in parts), dtype=np.uint8)
        return (self.e4m3[b] * self.scale).astype(np.float32)
