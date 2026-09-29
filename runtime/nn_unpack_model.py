#!/usr/bin/env python3
"""nn_unpack_model.py — THE READER for a **silvann-packed-v3** bundle, and its round-trip gate.

⛔⛔ THE READER IS THE ONLY GATE A PACKER HAS THAT MEANS ANYTHING. A packer checked against itself
proves that its own arithmetic is self-consistent, which every wrong layout also is. What this file
does instead is read the bundle back **with no access to the packer's in-memory state** — only the
bytes, the offsets and the scalars that shipped — and check the weights against the SOURCE.
⇒ ★ A ROUND TRIP THROUGH THE FILE IS A DIFFERENT CLAIM FROM A ROUND TRIP THROUGH A FUNCTION.

── ⚖ WHAT IT REFUSES, AND WHY REFUSING IS THE POINT OF v3 ───────────────────────────────────────────
v2 discriminated by booleans, so a reader meeting an unknown record read it as whatever the missing
flags defaulted to and proceeded. v3 carries one required `storage` scalar, and this reader:
  · REFUSES a manifest whose `format` is not `silvann-packed-v3`
  · REFUSES a record with no `storage`
  · REFUSES a packed record whose `family` or `d` this build does not know
  · REFUSES a field whose extent runs past the record, or a record past the file
⛔ NONE OF THESE DEFAULT. Reading a family-3 file as family 2 produces plausible numbers, which is the
failure with no later symptom — ▶ `pack_container.md` §③.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn_compressor as NC                                            # noqa: E402

FORMAT = "silvann-packed-v3"
ROTATION = "blocked-512-256-128"   # ⛔ the ladder, and an absent key means PRE-PROMOTION


class Refused(Exception):
    """⛳ A NAMED EXCEPTION AND NOT A RETURN CODE, because every caller of this reader wants the
    process to stop. A bundle this build cannot read is not a degraded mode."""


def _fwht(a):
    """`H` at the packer's normalisation, over the last axis — the reader's OWN butterfly.

    ⛔⛔ THE FIRST VERSION OF THE UN-ROTATION CALLED THE REFERENCE PACKER'S PURE-PYTHON `unrotate`
    ONCE PER ROW. That is fine on the 32-row tensors the first sample happened to pick and unusable
    on an expert stack, which is 262,144 rows after flattening — the run had to be killed.
    ⇒ ★ A READER THAT IS ONLY FAST ENOUGH FOR THE SMALL RECORDS WILL BE TESTED ONLY ON THE SMALL
    RECORDS — the same selection effect that made the first sample alphabetical, arriving twice in
    one file from two unrelated decisions.
    ⛳ WRITING THE BUTTERFLY HERE RATHER THAN IMPORTING `NC.fwht` KEEPS THE INDEPENDENCE THAT MATTERS:
    what this file must not share with the packer is the LAYOUT — the offsets, the strides, the
    scalars. The transform is mathematics, and both sides computing `H` the same way is the claim
    being made, not a shortcut past one."""
    n = a.shape[-1]
    rows = a.shape[0]
    a = np.array(a, dtype=np.float64, copy=True)
    h = 1
    while h < n:
        a = a.reshape(rows, n // (2 * h), 2, h)
        x = a[:, :, 0, :].copy()
        y = a[:, :, 1, :].copy()
        a[:, :, 0, :] = x + y
        a[:, :, 1, :] = x - y
        a = a.reshape(rows, n)
        h *= 2
    return a / np.sqrt(n)


def _levels(d):
    import nn_gaussian_tables as T
    return np.array(T.table(d)[1], dtype=np.float64)


def decode_record(rec, blob):
    """One record's bytes -> the dense weights it stands for. The decoder's half, from the FILE.

    ⛳ THIS IS A THIRD IMPLEMENTATION OF THE LAYOUT — beside the packer and the device — and it is
    deliberately written from the format rather than by importing the packer's helpers. A reader that
    unpacks with the packer's own code tests only that one function is its own inverse."""
    sc = rec["scalars"]
    storage = sc.get("storage")
    if storage is None:
        raise Refused("record has no `storage` scalar — v3 never infers a shape from missing keys")

    if storage == "dense":
        f = rec["fields"]["dense"]
        raw = blob[rec["record_offset"] + f["rel"]:][:f["nbytes"]]
        return np.frombuffer(raw, dtype=np.dtype(f["dtype"])).reshape(f["shape"])

    if storage != "packed":
        raise Refused("unknown storage %r" % (storage,))
    if sc.get("family") != NC.FAMILY_GAUSSIAN:
        raise Refused("family %r is not one this build reads" % (sc.get("family"),))
    d = sc.get("d")
    if d not in NC.WIDTHS and d != 16:
        raise Refused("width %r is not one this build reads" % (d,))

    rows, cols, row_bytes = sc["rows"], sc["cols"], sc["row_bytes"]
    fw, fl = rec["fields"]["vbr_data"], rec["fields"]["vbr_lut"]
    if fw["rel"] + fw["nbytes"] > rec["record_nbytes"] or \
       fl["rel"] + fl["nbytes"] > rec["record_nbytes"]:
        raise Refused("a field runs past its record")
    base = rec["record_offset"]
    wb = blob[base + fw["rel"]:][:fw["nbytes"]]
    lb = blob[base + fl["rel"]:][:fl["nbytes"]]
    if len(wb) < rows * row_bytes:
        raise Refused("the weights plane is short: %d < %d" % (len(wb), rows * row_bytes))

    scale = np.frombuffer(lb, dtype="<f2")[:rows].astype(np.float64)
    per = 16 // d
    nunits = row_bytes // 2
    units = np.frombuffer(wb, dtype="<u2")[:rows * nunits].reshape(rows, nunits).astype(np.uint32)
    codes = np.empty((rows, nunits * per), dtype=np.uint32)
    for s in range(per):
        codes[:, s::per] = (units >> np.uint32(d * s)) & np.uint32((1 << d) - 1)
    codes = codes[:, :cols]
    if d == 16:
        # ⛔⛔ AND IT RETURNS HERE WITHOUT UN-ROTATING, WHICH IS THE HALF THIS ARM WAS MISSING. Reading
        #   key-as-value was already right; falling through to the inverse transform below was not,
        #   because the PACKER does not rotate at this width — there is no scale to divide by and no
        #   table to index, so there is nothing for a rotation to have made Gaussian.
        #   ⛔ IT WAS LATENT RATHER THAN WRONG UNTIL TODAY: the packer never emitted `d == 16`, so no
        #   record ever took this path. Promoting the width (⚖ *"promote the d==16 as the real user"*)
        #   is what would have turned it into every lossless tensor decoding to `H(S w)` while every
        #   consumer believed it held `w`.
        #   ⇒ ★★ A HALF-BUILT ARM FOR A CASE NOTHING PRODUCES IS NOT DEAD CODE, IT IS A TRAP ARMED FOR
        #   WHOEVER PRODUCES IT — and it reads as finished support, which is what makes it worse than
        #   an absent arm that would have raised.
        w = codes.astype("<u2").view("<f2").astype(np.float64)
        return w.reshape(sc["original_shape"])
    rot = _levels(d)[codes] * scale[:, None]
    # the rotation is UNDONE here, because a consumer wants the weights, not the rotated ones.
    # ⛔ `R^T = S .* (H u)` — the same two operations in the OTHER order. Not `R` again: `H` and `S`
    #    are each involutions and their composition is not.
    # ⭐⭐ PER BLOCK, WITH THE STRUCTURE DERIVED FROM `cols` — the same call the packer made. ⛳ THE
    #   RECORD CARRIES NO BLOCK FIELD, so this cannot read a structure that disagrees with the one the
    #   weights were rotated under: both sides ask `blocks_for(cols)` and there is nothing to keep in
    #   sync. ⇒ ★ A DERIVED STRUCTURE IS THE ONE KIND THAT CANNOT DRIFT.
    #   ⛔ This is a THIRD implementation of the layout by design (▶ the docstring above), so it calls
    #   `blocked_unrotate` for the BLOCKS and not for the transform: `_fwht` here stays this file's own.
    signs = NC.sign_master()
    dense = np.empty_like(rot)
    _a = 0
    for _b in NC.blocks_or_refuse(cols):
        dense[:, _a:_a + _b] = _fwht(rot[:, _a:_a + _b]) * signs[:_b][None, :]
        _a += _b
    return dense.reshape(sc["original_shape"])


def open_bundle(path):
    man = json.load(open(os.path.join(path, "pack.json")))
    if man.get("format") != FORMAT:
        raise Refused("format %r is not %s — v3 refuses rather than guessing"
                      % (man.get("format"), FORMAT))
    # ⛔⛔ AND THE ROTATION'S STRUCTURE, WHICH IS THE ONE THING A v3 BUNDLE CAN DISAGREE WITH THIS READER
    #   ABOUT WHILE LOOKING PERFECTLY WELL FORMED. Bundles packed before the 512 ladder was promoted
    #   rotated each row WHOLE; this reader un-rotates per block. `MEASURED`: a cols-2048 record then
    #   decodes with 9.8e-01 relative error while a cols-512 one is exact — so the damage is invisible on
    #   exactly the records a small sample is most likely to pick.
    #   ⇒ ★ REFUSING ON AN ABSENT KEY IS THE WHOLE POINT: there is no value a pre-promotion packer could
    #   have written here, so absence IS the discriminator, and it is the case v3 was designed to refuse.
    if man.get("rotation") != ROTATION:
        raise Refused(
            "this bundle's `rotation` is %r and this reader requires %r — a bundle packed before the "
            "512-block ladder was promoted rotated whole rows, and reading it per block decodes a "
            "cols-2048 record ~98%% wrong while leaving cols-512 exact. RE-PACK it."
            % (man.get("rotation"), ROTATION))
    return man


def verify(path, raw, layers=None, per_layer=3, verbose=True):
    """Decode records back out of the FILE and check them against the source safetensors."""
    man = open_bundle(path)
    g = pickle.load(open(os.path.join(path, "global.pkl"), "rb"))
    checks = fails = 0

    def report(ok, msg):
        nonlocal checks, fails
        checks += 1
        if not ok:
            fails += 1
            print("  FAIL  %s" % msg)
        elif verbose:
            print("  ok    %s" % msg)

    report("__signs__" in g, "the sign plane is in the bundle")
    if "__signs__" in g:
        s = g["__signs__"]
        bits = np.unpackbits(np.frombuffer(s["plane"], dtype=np.uint8), bitorder="little")
        want = (NC.sign_master() < 0).astype(np.uint8)
        report(len(s["plane"]) == s["bits"] // 8, "the plane's length matches its declared bit count")
        report(bool((bits[:len(want)] == want).all()),
               "the plane decodes to the packer's own sign vector, bit for bit")

    nl = man["n_layers"] if layers is None else min(layers, man["n_layers"])
    for L in range(nl):
        d = os.path.join(path, "layers", "%03d" % L)
        idx = pickle.load(open(os.path.join(d, "index.pkl"), "rb"))["weights"]
        blob = np.memmap(os.path.join(d, "weights.bin"), dtype=np.uint8, mode="r")
        # ⛔⛔ THE BIGGEST RECORDS, NOT THE FIRST ALPHABETICALLY, AND THE FIRST VERSION OF THIS LINE
        #   TOOK `sorted(idx)[:n]` — which is `A_log`, `dt_bias`, `in_proj_a` … and never touched an
        #   expert stack. An expert is `[E*R, C]` = 262,144 rows after flattening, and a stride or
        #   offset defect is invisible on a 32-row tensor and fatal on that one.
        #   ⇒ ★★ A SAMPLE ORDERED BY NAME IS A SAMPLE OF THE ALPHABET. Order by what would break.
        names = sorted(idx, key=lambda n: -idx[n]["record_nbytes"])[:per_layer]
        for name in names:
            rec = idx[name]
            got = decode_record(rec, blob)
            src = NC.read_tensor(raw, name).astype(np.float64)
            sc = rec["scalars"]
            if sc["storage"] == "dense":
                # ⛳ AGAINST THE SOURCE NARROWED TO THE DTYPE THE RECORD DECLARES, AND THAT IS STILL A
                #   BIT-EXACT CLAIM ABOUT THE FILE. A dense record is fp16 storage (⚖ *"fp16 storage,
                #   fp32 accumulators"*), so comparing it to the fp32 the reader widens bf16 into would
                #   assert something false and go red on a correct bundle. What this row is FOR is the
                #   LAYOUT — offset, stride, shape, padding — and equality against `dtype(src)` settles
                #   every one of those exactly as before.
                #   ⛔ IT IS NOT WEAKENED INTO A TOLERANCE, which is the tempting repair and the wrong
                #   one: `allclose` would pass a record written at the wrong OFFSET by a few bytes.
                #   ⛳ AND IT ASKS THE RECORD RATHER THAN ASSUMING fp16, so a bundle packed before the
                #   fp16 fix still verifies as what it is instead of failing for being old. ⇒ ★ THE
                #   RECORD SAYS ITS OWN DTYPE; A CHECKER THAT HARDCODES ONE IS A SECOND OPINION ABOUT
                #   THE FORMAT — and this file has just watched a hardcoded set fail green three times.
                stored = np.dtype(rec["fields"]["dense"]["dtype"])
                want = src.astype(stored).astype(np.float64)
                report(bool(np.array_equal(got.astype(np.float64), want)),
                       "L%02d %s — dense, bit for bit vs %s(src) (%s)"
                       % (L, name.split(".")[-1], stored.name, sc.get("dense_because")))
                continue
            # ⭐⭐⭐ THE LAYOUT IS CHECKED BY BYTE EQUALITY, WHICH IS BOTH THE STRONGEST AND THE
            #   CHEAPEST FORM OF IT. Re-pack the source and compare the PLANE BYTES to the ones in
            #   the file. If those are equal there is no offset, stride, container-shape or padding
            #   question left open — the file IS what the packer produced.
            #   ⛔⛔ AND IT AVOIDS THE SAMPLING TRAP THAT THIS FILE HAS NOW FALLEN INTO TWICE: first
            #   `sorted(idx)[:n]` sampled the ALPHABET, then a decode-and-compare sampled the first
            #   64 ROWS of a 262,144-row tensor. A byte comparison has no sample — it is the whole
            #   record or nothing. ⇒ ★★ THE FIX FOR A BAD SAMPLE IS OFTEN NOT A BETTER SAMPLE.
            planes, recon, pstats = NC.pack_tensor(src.reshape(-1, src.shape[-1]), sc["d"])
            base = rec["record_offset"]
            ok = True
            for key in ("vbr_data", "vbr_lut"):
                f = rec["fields"][key]
                onfile = bytes(blob[base + f["rel"]:base + f["rel"] + f["nbytes"]])
                ok = ok and onfile == planes[key]
            report(ok, "L%02d %-24s q%-2d %6.3f b/w  %8.2f MiB  plane bytes IDENTICAL to the packer"
                   % (L, name.split("layers.%d." % L)[-1][:24], sc["d"],
                      sc["bits_per_weight"], rec["record_nbytes"] / 2**20))
            # ⛳ AND THE SCALARS MUST DESCRIBE THOSE BYTES, because a correct plane under a wrong
            #   `rows`/`cols`/`row_bytes` decodes to the wrong shape and the byte check cannot see it.
            report((sc["rows"], sc["cols"], sc["row_bytes"])
                   == (pstats["rows"], pstats["cols"], pstats["row_bytes"]),
                   "L%02d %-24s ...and its scalars describe them (%d x %d, %d B a row)"
                   % (L, name.split("layers.%d." % L)[-1][:24],
                      sc["rows"], sc["cols"], sc["row_bytes"]))
            # ⛳ THE DECODE PATH, on a sample — a DIFFERENT claim from the layout, and the only one a
            #   sample is the right instrument for: that this reader reads back what is there.
            got_rows = got.reshape(-1, got.shape[-1])[:32]
            # ⛔ AT `d == 16` THERE IS NOTHING TO UN-ROTATE, and the reference's `unrotate` REFUSES a
            #   non-power-of-two length — so this row was the second place the promoted width tripped
            #   over an unconditional inverse transform. `recon` already IS the weights at this width,
            #   which is what makes the comparison stronger here and not weaker: the reference and the
            #   reader agree by both leaving the values alone.
            #   ⇒ ★ THE ARM AND ITS CROSS-CHECK BOTH ASSUMED A ROTATION, so fixing only the decoder
            #   would have swapped a silent wrong answer for a confident red in the row that exists to
            #   catch it. A width is not promoted until every consumer of the old assumption is found.
            # ⛳ AND THE REFERENCE IS ASKED PER BLOCK TOO, because `REF.unrotate` is a whole-vector
            #   routine and REFUSES a non-power-of-two length. Feeding it one block at a time is what
            #   keeps it usable as an independent second opinion on a blocked record.
            if sc["d"] == 16:
                want = np.asarray(recon[:32], dtype=np.float64)
            else:
                _bl = NC.blocks_or_refuse(src.shape[-1])
                _ms = NC.sign_master()
                want = np.empty((min(32, recon.shape[0]), src.shape[-1]), dtype=np.float64)
                for _i, _r in enumerate(recon[:32]):
                    _a = 0
                    for _b in _bl:
                        want[_i, _a:_a + _b] = NC.REF.unrotate(
                            list(np.asarray(_r[_a:_a + _b], dtype=np.float64)), list(_ms[:_b]))
                        _a += _b
            w = float(np.max(np.abs(got_rows - want)) / max(float(np.max(np.abs(want))), 1e-30))
            report(w < 1e-9, "L%02d %-24s ...and the reader decodes them (%.1e)"
                   % (L, name.split("layers.%d." % L)[-1][:24], w))
    print("\n%s — %d checks, %d failures" % ("PASS" if fails == 0 else "FAIL", checks, fails))
    return fails == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--raw", default=os.environ.get("SILVANN_RAW_MODEL",
                                                    "/mnt/data/bigdisk/qwen_36_35_a3b"))
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--per-layer", type=int, default=3)
    a = ap.parse_args()
    sys.exit(0 if verify(a.bundle, a.raw, a.layers, a.per_layer) else 1)
