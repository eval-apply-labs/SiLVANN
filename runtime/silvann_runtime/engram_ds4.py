"""DeepSeek-V4.1's n-gram memory on the host: a position's hash ids, and the rows of the layer's table they name (NN-95).

At a position, the 2-, 3- … `max_ngram`-grams ending there are hashed — over a compressed vocabulary in which tokens that
normalise alike are one (NFKC, accents stripped, lower case, whitespace folded) — each into `heads` prime-sized ranges
of the layer's table, the primes drawn in order and never reused. The rows are FP8 E4M3 with an E8M0 scale a 32, read
in place from the tables' files and dequantised. The card does the rest (▶ ai_deepseek_4's `engram` verb).
The model's own is `inference/engram.py` (`build_compressed_token_map`, `EngramLayout`, `NgramHashState`).
"""
import json
import os
import struct

import numpy as np


def _is_prime(n):
    """Miller–Rabin with the bases that decide every n below 3.3·10^24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def token_map(tokenizer_json):
    """Every token id onto the compressed vocabulary, and its size — the model's `build_compressed_token_map`."""
    from tokenizers import Regex, Tokenizer, normalizers
    tok = Tokenizer.from_file(tokenizer_json)
    sentinel = ""
    norm = normalizers.Sequence([normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
                                 normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
                                 normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    key_to_new, n = {}, tok.get_vocab_size(with_added_tokens=True)
    lookup = np.zeros(n, dtype=np.int64)
    for t in range(n):
        text = tok.decode([t], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(t)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        lookup[t] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class Engram:
    """The n-gram memory of a DeepSeek-V4.1 model: `ids(tokens, pos)` a position's row ids a layer, `rows(layer, ids)` the
    rows. `tables` is the folder whose safetensors hold `layers.L.engram.embed.weight` and `.scale` (the checkpoint's)."""

    def __init__(self, config, tokenizer_json, tables):
        c = config
        self.layers = list(c["engram_layer_ids"])
        self.N, self.heads, self.dim = int(c["engram_max_ngram_size"]), int(c["engram_n_heads"]), int(c["engram_head_dim"])
        self.map, size = token_map(tokenizer_json)
        if size != int(c["engram_compressed_vocab_size"]):
            # every hash multiplier comes from this size, so another one would rehash the whole table
            raise ValueError("the compressed vocabulary is %d tokens, the model's %d" % (size, c["engram_compressed_vocab_size"]))
        self.pad = int(self.map[int(c["engram_pad_token_id"])])
        primes, seen = [], set()
        for _ in self.layers:
            per = []
            for _ in range(self.N - 1):
                sizes, cur = [], int(c["engram_vocab_size"]) - 1
                for _ in range(self.heads):
                    cur += 1
                    while not _is_prime(cur) or cur in seen:
                        cur += 1
                    seen.add(cur)
                    sizes.append(cur)
                per.append(sizes)
            primes.append(per)
        self.primes = np.array(primes, dtype=np.int64)                               # [layer, n-gram, head]
        flat = self.primes.reshape(len(self.layers), -1)
        self.offsets = np.concatenate([np.zeros((len(self.layers), 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1)
        bound = max(1, (np.iinfo(np.int64).max // size) // 2)
        self.mult = np.stack([np.random.default_rng(10007 * l).integers(0, bound, size=(self.N,), dtype=np.int64) * 2 + 1
                              for l in self.layers])                                 # [layer, lookback]
        self.cols = (self.N - 1) * self.heads
        self.width = self.cols * self.dim
        self._tables(tables)

    def _tables(self, raw):
        wm = json.load(open(os.path.join(raw, "model.safetensors.index.json")))["weight_map"]
        self.table, self.fds, self.at = {}, {}, {}
        # ⭐ A POSITION'S ROWS READ ALL AT ONCE: each is a random place in a table of 384 million (98 GB a layer), so read one
        #   after another through the map a position was one page fault after another — MEASURED 2026-10-19 node-01, the
        #   tables on the RAID: 1,057 ms a position (median, 40 positions of fresh tokens), the whole of a token's time.
        #   A pool of threads keeps every read in flight together, as many as the drives can serve.
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=64)
        for l in self.layers:
            got = []
            for what in ("weight", "scale"):
                name = "layers.%d.engram.embed.%s" % (l, what)
                path = os.path.join(raw, wm[name])
                with open(path, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    head = json.loads(fh.read(n))
                a, b = head[name]["data_offsets"]
                mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + a, shape=(b - a,))
                got.append(mm.reshape(head[name]["shape"]))
                # where a row is, for the reads `all_rows` makes at once: the file, its first row, a row's bytes
                fd = self.fds.setdefault(path, os.open(path, os.O_RDONLY))
                self.at[(l, what)] = (fd, 8 + n + a, int(np.prod(head[name]["shape"][1:])))
            self.table[l] = tuple(got)
        from nn_compressor import E4M3
        self.e4m3 = E4M3

    def ids(self, tokens, pos):
        """Position `pos`'s row ids in each engram layer, `[layer][cols]`: the n-grams ending there, over `tokens` (the
        conversation's ids from position 0), padded before the first."""
        back = [int(self.map[tokens[pos - s]]) if pos - s >= 0 else self.pad for s in range(self.N)]
        out = np.empty((len(self.layers), self.cols), dtype=np.int64)
        for i in range(len(self.layers)):
            products = np.array(back, dtype=np.int64) * self.mult[i]
            rolling, hashes = products[0], []
            for k in range(1, self.N):
                rolling = np.bitwise_xor(rolling, products[k])
                hashes.append(rolling % self.primes[i, k - 1])
            out[i] = np.concatenate(hashes) + self.offsets[i]
        return out

    def all_rows(self, ids):
        """Every engram layer's rows for `ids` (`[layer][cols]`, as `ids` answers them), dequantised — {layer: floats} —
        every read issued at once."""
        jobs = {}
        for i, l in enumerate(self.layers):
            for what in ("weight", "scale"):
                fd, base, rb = self.at[(l, what)]
                jobs[(l, what)] = [self.pool.submit(os.pread, fd, rb, base + int(r) * rb) for r in ids[i]]
        out = {}
        for i, l in enumerate(self.layers):
            w = np.frombuffer(b"".join(f.result() for f in jobs[(l, "weight")]), dtype=np.uint8).reshape(len(ids[i]), -1)
            sb = np.frombuffer(b"".join(f.result() for f in jobs[(l, "scale")]), dtype=np.uint8).reshape(len(ids[i]), -1)
            v = self.e4m3[w].astype(np.float32)
            sc = np.ldexp(np.float32(1), sb.astype(np.int32) - 127).astype(np.float32)
            out[l] = (v.reshape(len(ids[i]), -1, 32) * sc[:, :, None]).reshape(-1)
        return out

    def rows(self, layer, ids):
        """The rows `ids` names in `layer`'s table, dequantised: `cols · dim` floats."""
        w, s = self.table[layer]
        v = self.e4m3[np.asarray(w[ids])].astype(np.float32)                         # [cols, dim]
        sc = np.ldexp(np.float32(1), np.asarray(s[ids]).astype(np.int32) - 127).astype(np.float32)
        return (v.reshape(len(ids), -1, 32) * sc[:, :, None]).reshape(-1)
