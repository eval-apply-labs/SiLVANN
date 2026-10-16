"""QWEN-IMAGE 2.1's VAE ON THE HOST — a picture's pixels to its latents and back, in numpy (NN-89).

diffusers' `AutoencoderKLQwenImage21` for a single picture, written out over arrays a pixel a row (height x width x
channels, the transformer's packed layout): its "causal 3D" convolutions are 2D ones (the class folds the frame away),
and on a picture's one frame every temporal path is skipped — the time convolutions, the feature caches — so what
remains is 3 x 3 and 1 x 1 convolutions, a channel RMS norm, SiLU, nearest upsampling, stride-2 downsampling, one
single-head attention in the middle, and the residual paths that move channels to space and back (`DupUp3D`,
`AvgDown3D`, each taking its picture's one frame). Read from the pack's `vae/` as it ships (float32 safetensors, read
here without the `safetensors` package), in float32. The latents are the pipeline's: normalized by the config's means
and deviations, a pixel a row.
"""
import json
import os
import struct

import numpy as np

from .model_folder import Refused

CHUNK_FLOATS = 1 << 25        # the unfolded rows a convolution multiplies at a time: 128 MB of float32


def tensors(path):
    """Every tensor of a safetensors file, mapped from the disk: {name: float32 array}."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, v in head.items():
        if name == "__metadata__":
            continue
        if v["dtype"] != "F32":
            raise ValueError("%s is %s; this reader takes the VAE's float32" % (name, v["dtype"]))
        a, b = v["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype="<f4").reshape(v["shape"])
    return out


def conv(x, w, b, stride=1, pad=(0, 0, 0, 0)):
    """A 2D convolution over `x` (H x W x C): `w` (out, in, kh, kw), `b` (out); `pad` (top, bottom, left, right) zeros."""
    o, i, kh, kw = w.shape
    if kh == 1 and kw == 1 and stride == 1 and not any(pad):
        y = x.reshape(-1, i) @ w.reshape(o, i).T
        return (y + b).reshape(x.shape[0], x.shape[1], o)
    xp = np.pad(x, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(0, 1))[::stride, ::stride]   # Ho, Wo, C, kh, kw
    ho, wo = win.shape[:2]
    wm = w.transpose(1, 2, 3, 0).reshape(i * kh * kw, o)                                              # (C, kh, kw) x out
    out = np.empty((ho, wo, o), np.float32)
    rows = max(1, CHUNK_FLOATS // (wo * i * kh * kw))
    for r0 in range(0, ho, rows):
        r1 = min(ho, r0 + rows)
        out[r0:r1] = (win[r0:r1].reshape((r1 - r0) * wo, i * kh * kw) @ wm).reshape(r1 - r0, wo, o) + b
    return out


def rms_norm(x, gamma):
    """diffusers' `QwenImage21RMS_norm` a pixel at a time: x / ‖x‖ (over the channels) · √C · γ."""
    n = np.sqrt(np.einsum("hwc,hwc->hw", x, x, dtype=np.float64)).astype(np.float32)[..., None]
    return x / np.maximum(n, 1e-12) * np.float32(np.sqrt(x.shape[-1])) * gamma.reshape(-1)


def silu(x):
    return x / (np.float32(1.0) + np.exp(-x))


class Vae:
    def __init__(self, path):
        folder = os.path.join(path, "vae")
        self.config = c = json.load(open(os.path.join(folder, "config.json")))
        if c.get("_class_name") != "AutoencoderKLQwenImage21" or not c.get("is_residual") or c.get("patch_size") or c.get("attn_scales"):
            raise ValueError("%s is not the VAE this reader is written for" % folder)
        self.t = tensors(os.path.join(folder, "diffusion_pytorch_model.safetensors"))
        self.Z = c["z_dim"]
        self.mean = np.asarray(c["latents_mean"], np.float32)
        self.std = np.asarray(c["latents_std"], np.float32)
        self.down_t = list(c["temperal_downsample"])
        self.levels = len(c["dim_mult"])

    def _w(self, name):
        return self.t[name + ".weight"], self.t[name + ".bias"]

    def _conv(self, x, name, k=3, stride=1, pad=None):
        w, b = self._w(name)
        if pad is None:
            pad = (k // 2,) * 4
        return conv(x, w, b, stride, pad)

    def _resnet(self, x, name):
        """`QwenImage21ResidualBlock`: the shortcut (a 1 x 1 convolution where the width changes) plus norm, SiLU,
        convolution, twice."""
        h = self._conv(x, name + ".conv_shortcut", 1) if name + ".conv_shortcut.weight" in self.t else x
        y = self._conv(silu(rms_norm(x, self.t[name + ".norm1.gamma"])), name + ".conv1")
        y = self._conv(silu(rms_norm(y, self.t[name + ".norm2.gamma"])), name + ".conv2")
        return y + h

    def _attention(self, x, name):
        """`QwenImage21AttentionBlock`: one head over every pixel."""
        hh, ww, c = x.shape
        y = rms_norm(x, self.t[name + ".norm.gamma"])
        qkv = self._conv(y, name + ".to_qkv", 1).reshape(hh * ww, 3 * c)
        q, k, v = qkv[:, :c], qkv[:, c:2 * c], qkv[:, 2 * c:]
        s = (q @ k.T) * np.float32(1.0 / np.sqrt(c))
        s = np.exp(s - s.max(axis=1, keepdims=True))
        a = (s / s.sum(axis=1, keepdims=True)) @ v
        return self._conv(a.reshape(hh, ww, c), name + ".proj", 1) + x

    def _mid(self, x, name):
        x = self._resnet(x, name + ".resnets.0")
        x = self._attention(x, name + ".attentions.0")
        return self._resnet(x, name + ".resnets.1")

    @staticmethod
    def _dup_up(x, out, ft, fs=2):
        """`QwenImage21DupUp3D` on a picture's frame (`first_chunk`): output channel o at (fs·h + i, fs·w + j) is input
        channel (o·ft·fs² + (ft − 1)·fs² + i·fs + j) // repeats at (h, w)."""
        hh, ww, cin = x.shape
        rep = out * ft * fs * fs // cin
        o = np.arange(out)[:, None, None]
        i, j = np.arange(fs)[None, :, None], np.arange(fs)[None, None, :]
        src = (o * ft * fs * fs + (ft - 1) * fs * fs + i * fs + j) // rep                          # out x fs x fs
        y = x[:, :, src]                                                                           # H x W x out x fs x fs
        return y.transpose(0, 3, 1, 4, 2).reshape(hh * fs, ww * fs, out)

    @staticmethod
    def _avg_down(x, out, ft, fs=2):
        """`QwenImage21AvgDown3D` on a picture's frame: the frame padded with ft − 1 zero frames before it, each fs x fs
        patch's channels laid out (c, t, i, j), grouped `C·ft·fs² / out` to an output channel, and averaged."""
        hh, ww, c = x.shape
        y = np.zeros((hh // fs, ww // fs, c, ft, fs, fs), np.float32)
        for i in range(fs):
            for j in range(fs):
                y[:, :, :, ft - 1, i, j] = x[i::fs, j::fs, :]
        return y.reshape(hh // fs, ww // fs, out, c * ft * fs * fs // out).mean(axis=-1)

    # ── decode: the transformer's latents to a picture ─────────────────────────────────────────────────────────────
    def decode(self, latents, height, width):
        """The pipeline's latents (`height` x `width` of them, a pixel a row, normalized) to a picture: H x W x 4 floats in
        [-1, 1] (RGBA), 16 pixels a latent each way."""
        z = np.asarray(latents, np.float32).reshape(height, width, self.Z) * self.std + self.mean
        x = self._conv(z, "post_quant_conv", 1)
        x = self._conv(x, "decoder.conv_in")
        x = self._mid(x, "decoder.mid_block")
        dims = [self.config["decoder_base_dim"] * u for u in [self.config["dim_mult"][-1]] + self.config["dim_mult"][::-1]]
        up_t = self.down_t[::-1]
        for i in range(self.levels):
            name = "decoder.up_blocks.%d" % i
            x0 = x
            for r in range(self.config["num_res_blocks"] + 1):
                x = self._resnet(x, "%s.resnets.%d" % (name, r))
            if i != self.levels - 1:
                x = self._conv(x.repeat(2, axis=0).repeat(2, axis=1), name + ".upsampler.resample.1")
                x = x + self._dup_up(x0, dims[i + 1], 2 if up_t[i] else 1)
        x = silu(rms_norm(x, self.t["decoder.norm_out.gamma"]))
        return np.clip(self._conv(x, "decoder.conv_out"), -1.0, 1.0)

    # ── encode: a picture to its latents ───────────────────────────────────────────────────────────────────────────
    def encode(self, pixels):
        """A picture (H x W x 4 floats in [-1, 1], RGBA, sides multiples of 16) to the pipeline's latents: the
        posterior's mean, normalized, a pixel a row — (H/16 · W/16) x Z."""
        x = self._conv(np.asarray(pixels, np.float32), "encoder.conv_in")
        dims = [self.config["base_dim"] * u for u in [1] + self.config["dim_mult"]]
        for i in range(self.levels):
            name = "encoder.down_blocks.%d" % i
            last = i == self.levels - 1
            x0 = x
            for r in range(self.config["num_res_blocks"]):
                x = self._resnet(x, "%s.resnets.%d" % (name, r))
            if not last:
                x = self._conv(x, name + ".downsampler.resample.1", stride=2, pad=(0, 1, 0, 1))
            x = x + self._avg_down(x0, dims[i + 1], 2 if (not last and self.down_t[i]) else 1, 1 if last else 2)
        x = self._mid(x, "encoder.mid_block")
        x = silu(rms_norm(x, self.t["encoder.norm_out.gamma"]))
        x = self._conv(self._conv(x, "encoder.conv_out"), "quant_conv", 1)
        z = x[..., :self.Z]
        return ((z - self.mean) / self.std).reshape(-1, self.Z)


# ══ ⭐ THE VAE ON THE CARD — ai_qwen_image's `conv` and `vae_attention` over the same weights at 16 bits ══════════════
class CardVae:
    """The decode on a worker, as the pack's program says it (`lisp/qwen_image_vae.lisp`, `(vae_decode h w)`): each
    convolution `ai_qwen_image__conv` over a table (▶ ai_qwen_image's `contracts/objects/vae.cuh`), its weights in
    halves, a kernel a row in (dy, dx, c); the middle's attention its own verb. This class makes what the program reads —
    the weights' and the planes' collections, a table a convolution, the levels' arrays — writes the latents and reads
    the picture back. Four picture-sized planes carry it (X, Y, T and D), sized at boot for latents of up to `max_h` x
    `max_w`."""
    PROGRAM = "qwen_image_vae"      # the pack's lisp/ file

    def __init__(self, path, machine, max_h=64, max_w=64, chunk=4096, program=None):
        self.cpu = Vae(path)                   # the weights as the pack ships them, and the config
        self.m, self.MAX_H, self.MAX_W, self.CHUNK = machine, int(max_h), int(max_w), int(chunk)
        # the program's file: the pack's own, or the one given — a development box's
        self.PROGRAM_FILE = program or os.path.join(path, "lisp", self.PROGRAM + ".lisp")
        c = self.cpu.config
        self.dims = [c["decoder_base_dim"] * u for u in [c["dim_mult"][-1]] + c["dim_mult"][::-1]]
        self.up_t = self.cpu.down_t[::-1]
        self._plan()

    # ── the layers, the planes they read, the sizes ───────────────────────────────────────────────────────────────
    def _convs(self):
        """Every convolution of the decode: (name, weight (O, C, kh, kw) float32, bias, gamma or None, options)."""
        t, c = self.cpu.t, self.cpu.config
        out = []

        def conv(name, prefix, norm=None, **o):
            w, b = t[prefix + ".weight"], t[prefix + ".bias"]
            k = w.shape[-1]
            o.setdefault("pad", (k // 2,) * 4)
            out.append((name, w, b, None if norm is None else t[norm].reshape(-1), o))

        def resnet(name, prefix):
            if prefix + ".conv_shortcut.weight" in t:
                conv(name + "_short", prefix + ".conv_shortcut")
            conv(name + "_c1", prefix + ".conv1", prefix + ".norm1.gamma")
            conv(name + "_c2", prefix + ".conv2", prefix + ".norm2.gamma", add=1)
        conv("post_quant", "post_quant_conv")
        conv("conv_in", "decoder.conv_in")
        resnet("mid_r0", "decoder.mid_block.resnets.0")
        resnet("mid_r1", "decoder.mid_block.resnets.1")
        for i in range(self.cpu.levels):
            for r in range(c["num_res_blocks"] + 1):
                resnet("up%d_r%d" % (i, r), "decoder.up_blocks.%d.resnets.%d" % (i, r))
            if i != self.cpu.levels - 1:
                conv("up%d_up" % i, "decoder.up_blocks.%d.upsampler.resample.1" % i, up=2, add=1)
                cin, cout, ft = self.dims[i], self.dims[i + 1], 2 if self.up_t[i] else 1
                w = np.zeros((4 * cout, cin, 1, 1), np.float32)       # the dup-up's selection: out (i, j, o) <- in src
                rep = cout * ft * 4 // cin
                for i2 in range(2):
                    for j2 in range(2):
                        for o in range(cout):
                            w[(i2 * 2 + j2) * cout + o, (o * ft * 4 + (ft - 1) * 4 + i2 * 2 + j2) // rep, 0, 0] = 1.0
                out.append(("up%d_dup" % i, w, np.zeros(4 * cout, np.float32), None, {"pad": (0, 0, 0, 0), "shuffle": 2}))
        conv("conv_out", "decoder.conv_out", "decoder.norm_out.gamma")
        return out

    def _plan(self):
        """The weights' slot (every convolution's planes, the attention's) and the planes' slot (z, X, Y, T, D, the
        picture, the scratch) — sized by walking the decode at the largest latents."""
        self.CONVS = {name: (w, b, g, o) for name, w, b, g, o in self._convs()}
        self.WREC, at = {}, 0

        def place(key, nbytes):
            nonlocal at
            self.WREC[key] = (at, nbytes)
            at += (nbytes + 255) // 256 * 256
        for name, (w, b, g, o) in self.CONVS.items():
            place(name + ".w", w.size * 2)
            place(name + ".b", b.size * 2)
            place(name + ".g", (g.size if g is not None else 1) * 2)
        t = self.cpu.t
        C = self.dims[0]
        for part in ("q", "k", "v", "proj"):
            place("attn." + part + ".w", C * C * 2)
            place("attn." + part + ".b", C * 2)
        place("attn.g", C * 2)
        self.WPLAN_BYTES = at
        # the walk: the largest picture-sized plane, and the largest scratch any layer asks
        h, w = self.MAX_H, self.MAX_W
        big, scratch = h * w * max(64, self.dims[0]), 0

        def need(name, H, W):
            nonlocal big, scratch
            wt, _, g, o = self.CONVS[name]
            O, Cc, kh, kw = wt.shape
            up, st = o.get("up", 1), o.get("stride", 1)
            pt, pb, pl, pr = o["pad"]
            Ho, Wo = (H * up + pt + pb - kh) // st + 1, (W * up + pl + pr - kw) // st + 1
            out = Ho * Wo * O
            big = max(big, out, H * W * Cc)
            scratch = max(scratch, ((8 * self.CHUNK + 15) & ~15) + 2 * self.CHUNK * kh * kw * Cc + 2 * self.CHUNK * O
                          + (2 * H * W * Cc if g is not None else 0) + (4 * out if o.get("shuffle", 1) > 1 else 0))
            f = o.get("shuffle", 1)
            return Ho * f, Wo * f
        H, W = need("post_quant", h, w)
        need("conv_in", H, W)
        for name in ("mid_r0_c1", "mid_r0_c2", "mid_r1_c1", "mid_r1_c2"):
            need(name, H, W)
        for i in range(self.cpu.levels):
            for r in range(self.cpu.config["num_res_blocks"] + 1):
                for s in ("_short", "_c1", "_c2"):
                    if "up%d_r%d%s" % (i, r, s) in self.CONVS:
                        need("up%d_r%d%s" % (i, r, s), H, W)
            if i != self.cpu.levels - 1:
                need("up%d_dup" % i, H, W)
                H, W = need("up%d_up" % i, H, W)
        need("conv_out", H, W)
        N = self.MAX_H * self.MAX_W
        self.ATTN_SCRATCH = ((8 * N + 15) & ~15) + 12 * N * C + 2 * N * N
        self.SREC, at = {}, 0
        for name, nbytes in (("z", 2 * h * w * 64), ("X", 2 * big), ("Y", 2 * big), ("T", 2 * big), ("D", 2 * big),
                             ("scratch", max(scratch, self.ATTN_SCRATCH))):
            self.SREC[name] = (at, nbytes)
            at += (nbytes + 255) // 256 * 256
        self.SPLAN_BYTES = at

    def collections(self, first):
        """The two collections, from type `first`: [(type, plan, slots)]."""
        from .model_folder import NL
        self.T_W, self.T_S = first, first + 1
        return [(self.T_W, NL.SlotPlan(self.WPLAN_BYTES, 0, 0, 256), 1), (self.T_S, NL.SlotPlan(self.SPLAN_BYTES, 0, 0, 256), 1)]

    def device_params(self):
        """A boot of its own: small pools, its two collections."""
        coll = self.collections(0)
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (t, p.up_bytes, t, p.down_bytes, t, k) for t, p, k in coll)
        arena = sum(p.slot_bytes * k for _, p, k in coll)
        return ("nn__buffer_resid__size_bytes:1048576\nnn__buffer_resid__qty:2\n"
                "nn__buffer_main__size_bytes:1048576\nnn__buffer_main__qty:2\n"
                "nn__buffer_attn_out__size_bytes:1048576\nnn__buffer_attn_out__qty:2\n"
                "nn__buffer_logits__size_bytes:1048576\nnn__buffer_logits__qty:1\n"
                "nn__buffer_partials__size_bytes:1048576\nnn__buffer_partials__qty:1\n"
                "nn__expert_cache_l1__size_mb:%d\n"
                "nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                "nn__model__kv_cache_layers_fullattn:1\nnn__model__kv_cache_layers_deltanet:0\n"
                "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                "nn__model__layer_types:f\nnn__expert__types:%d\n%s" % ((arena + (256 << 20)) >> 20, len(coll), types))

    # ── the load and the tables ──────────────────────────────────────────────────────────────────────────────────
    def load(self, loader=None):
        loader = loader or self.m.loader()
        half = lambda a: np.ascontiguousarray(a, dtype="<f2").tobytes()
        loader.layer_width(0, self.T_W, 1)
        at = loader.reserve(self.T_W)
        for name, (w, b, g, o) in self.CONVS.items():
            loader.write(at + self.WREC[name + ".w"][0], half(w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)))
            loader.write(at + self.WREC[name + ".b"][0], half(b))
            if g is not None:
                loader.write(at + self.WREC[name + ".g"][0], half(g))
        t, C = self.cpu.t, self.dims[0]
        qkv_w, qkv_b = t["decoder.mid_block.attentions.0.to_qkv.weight"].reshape(3 * C, C), t["decoder.mid_block.attentions.0.to_qkv.bias"]
        for i, part in enumerate(("q", "k", "v")):
            loader.write(at + self.WREC["attn.%s.w" % part][0], half(qkv_w[i * C:(i + 1) * C]))
            loader.write(at + self.WREC["attn.%s.b" % part][0], half(qkv_b[i * C:(i + 1) * C]))
        loader.write(at + self.WREC["attn.proj.w"][0], half(t["decoder.mid_block.attentions.0.proj.weight"].reshape(C, C)))
        loader.write(at + self.WREC["attn.proj.b"][0], half(t["decoder.mid_block.attentions.0.proj.bias"]))
        loader.write(at + self.WREC["attn.g"][0], half(t["decoder.mid_block.attentions.0.norm.gamma"].reshape(-1)))
        loader.assign(0, self.T_W, 0, at)
        loader.layer_width(0, self.T_S, 1)
        self.S_AT = loader.reserve(self.T_S)
        loader.assign(0, self.T_S, 0, self.S_AT)
        self._tables()

    def _tables(self):
        m = self.m
        wp = lambda key: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_W,) + self.WREC[key])
        sp = lambda key: "(nn__expert__plane 0 %d 0 %d %d)" % ((self.T_S,) + self.SREC[key])
        for name, (w, b, g, o) in self.CONVS.items():
            O, Cc, kh, kw = w.shape
            pt, pb, pl, pr = o["pad"]
            m.table("VAE_" + name, [wp(name + ".w"), wp(name + ".b"), wp(name + ".g"), sp("scratch"), Cc, O, kh, kw,
                                    o.get("stride", 1), pt, pl, pb, pr, o.get("up", 1), 1 if g is not None else 0,
                                    o.get("shuffle", 1), o.get("add", 0), self.CHUNK])
        m.table("VAE_attn", [wp("attn.q.w"), wp("attn.q.b"), wp("attn.k.w"), wp("attn.k.b"), wp("attn.v.w"), wp("attn.v.b"),
                             wp("attn.proj.w"), wp("attn.proj.b"), wp("attn.g"), sp("scratch"), self.dims[0]])
        self._bindings({k: sp(k) for k in ("z", "X", "Y", "T", "D")})

    def _bindings(self, planes):
        """What `lisp/qwen_image_vae.lisp` reads of the boot — the planes, the levels' arrays of tables — then the program."""
        m, levels, R = self.m, self.cpu.levels, self.cpu.config["num_res_blocks"] + 1
        if not os.path.isfile(self.PROGRAM_FILE):
            raise Refused("%s has no %s — the picture decoder's program; `silvann download` fetches it"
                          % (os.path.dirname(os.path.dirname(self.PROGRAM_FILE)), os.path.basename(self.PROGRAM_FILE)))
        if any(n + "_short" in self.CONVS for n in ("mid_r0", "mid_r1")):
            raise Refused("this decoder's middle resnets change their width; the program's run in place")
        m.bind_values(dict(planes, LEVELS=levels, RESNETS=R))
        res = ["up%d_r%d" % (i, r) for i in range(levels) for r in range(R)]
        m.array("UP_C1", ["VAE_%s_c1" % n for n in res])
        m.array("UP_C2", ["VAE_%s_c2" % n for n in res])
        m.array("UP_HAS_SHORT", [n + "_short" in self.CONVS for n in res])
        m.array("UP_SHORT", ["VAE_%s_short" % n if n + "_short" in self.CONVS else 0 for n in res])
        m.array("UP_DUP", ["VAE_up%d_dup" % i if i < levels - 1 else 0 for i in range(levels)])
        m.array("UP_UP", ["VAE_up%d_up" % i if i < levels - 1 else 0 for i in range(levels)])
        m.load_source(self.PROGRAM_FILE)

    # ── the decode ────────────────────────────────────────────────────────────────────────────────────────────────
    def decode(self, latents, h, w):
        """The pipeline's latents (h·w x Z, normalized) to a picture on the card: H x W x 4 floats in [-1, 1]."""
        if h > self.MAX_H or w > self.MAX_W:
            raise Refused("latents of %d x %d, and this boot holds %d x %d" % (h, w, self.MAX_H, self.MAX_W))
        z = np.asarray(latents, np.float32).reshape(h, w, self.cpu.Z) * self.cpu.std + self.cpu.mean
        self.m.write(self.S_AT + self.SREC["z"][0], np.ascontiguousarray(z, dtype="<f2").tobytes())
        self.m.must("(vae_decode %d %d)" % (h, w), "the picture decoder")      # the program answers T's kind
        H, W = h << (self.cpu.levels - 1), w << (self.cpu.levels - 1)
        raw = self.m.read(self.S_AT + self.SREC["T"][0], H * W * 4 * 2)
        return np.clip(np.frombuffer(raw, "<f2").reshape(H, W, 4).astype(np.float32), -1.0, 1.0)
