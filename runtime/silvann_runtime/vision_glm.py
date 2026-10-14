"""A PICTURE AS GLM 5.3 FLASH READS IT — the picture made into patches and its vision tower over them.

    v = GlmVision(folder)                       # a GLM 5.3 Flash model folder whose global.pkl carries model.visual.*
    image = v.read(png_or_jpeg_bytes)           # .rows (tokens x the text model's width, float32), .grid (t, h, w)
    tokens = chat-framed text with image.tokens image pads between the picture's start and end tokens

GLM's text model reads a picture's tokens at their places, as it reads text: Hugging Face's model writes the tower's
rows over the pads and the positions stay the cache's (`Glm5NextModel.forward`: `masked_scatter`, then the language
model with the caller's positions). So the text side needs only the rows; the tower is all of a picture's arithmetic.

The picture is prepared as Hugging Face's processor (`Glm5NextImageProcessorPil`) prepares it, with the original
repository's settings (`processor_config.json`'s `image_processor`): the most and fewest tokens a picture may take
(8,000 and 16) decide an aligned canvas, multiples of 28 pixels; the picture is resized to fit inside it, its aspect
kept (Pillow's bicubic), and the rest of the canvas is zeros; then scaled by 1/255 and normalized with CLIP's mean and
spread, the still picture as two frames, cut into 14 x 14 patches in 2 x 2-block order.

The tower (`Glm5NextVisionModel`), in numpy here — float32, or float64 for the check against Hugging Face's
(`test/src_vision_reference_glm.py`):
  ① the patch embedding: each patch's 3 x 2 x 14 x 14 values times a 1024 x 1176 matrix, plus its bias;
  ② 24 blocks: RMSNorm -> q, k, v (with bias) -> each head's q and k RMSNormed (64 wide) -> the 2D rotary over the whole
    head (16 frequencies for the patch's row, 16 for its column, the 32 repeated) -> attention over the picture's
    patches, every patch to every patch -> the projection (with bias), added; RMSNorm -> gate and up (with bias), the
    gate at most 10 and the up within ±10, SiLU(gate) · up -> down (with bias), added. Every RMSNorm's eps 1e-5;
  ③ the last RMSNorm; each 2 x 2 block of patches (together in the processor's order) as one row of 4,096 — channel
    by channel, the block's four values of a channel side by side — times the 4,096 x 4,096 downsampling, plus its bias;
  ④ the merger: a 4,096 x 4,096 projection, LayerNorm (eps 1e-5, with bias), GELU (erf); gate and up to 10,240, clamped
    as the blocks' are, SiLU(gate) · up -> down to 4,096. No biases.
"""
import io
import json
import math
import os
import pickle

import numpy as np

from .model_folder import Refused
from .vision import Image, Video, decoded, frames_of

# the original repository's processor_config.json `image_processor`, which the packs do not carry
PROCESSOR = {"min_image_tokens": 16, "max_image_tokens": 8000,
             "image_mean": (0.48145466, 0.4578275, 0.40821073), "image_std": (0.26862954, 0.26130258, 0.27577711)}
# and its `video_processor`: frames sampled at `fps` a second, every frame's tokens within these bounds together
VIDEO = {"fps": 2.0, "min_image_tokens": 16, "max_image_tokens": 240000, "max_frames": 2048}
HOST_MAX_TOKENS = 1024


def smart_resize(num_frames, height, width, temporal_factor, factor, min_tokens, max_tokens):
    """Hugging Face's: the aligned canvas (multiples of `factor`) a picture of `height` x `width` goes into, within the
    token bounds (a token `temporal_factor · factor²` pixels over the frames)."""
    per_token = temporal_factor * factor ** 2
    min_pixels, max_pixels = min_tokens * per_token, max_tokens * per_token

    def align(value):
        return math.ceil(value / factor) * factor

    frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    h, w = align(height), align(width)
    if frames * h * w < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        h, w = align(max(1, math.ceil(height * scale))), align(max(1, math.ceil(width * scale)))
    if frames * h * w > max_pixels:
        if max_pixels < frames * factor ** 2:
            raise Refused("max_image_tokens %d is too few for one aligned patch" % max_tokens)
        low, high, h, w = 1, height, factor, factor
        while low <= high:                                    # the tallest content that fits the budget, as HF searches
            content = (low + high) // 2
            ch, cw = align(content), align(max(1, math.floor(width * content / height)))
            if frames * ch * cw <= max_pixels:
                h, w, low = ch, cw, content + 1
            else:
                high = content - 1
    return h, w


class GlmVision:
    mrope = False                   # its tokens keep their places in the cache: the text model has no rotary

    def __init__(self, folder, max_tokens=None, card=None):
        config = json.load(open(os.path.join(folder.path, "config.json")))
        vc = config.get("vision_config")
        if not vc or vc.get("model_type") != "glm5_next_vision":
            raise Refused("%s has no GLM 5.3 vision tower" % folder.path)
        self.folder = folder
        self.hidden, self.heads = vc["hidden_size"], vc["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.patch, self.temporal, self.merge = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
        self.depth, self.out = vc["depth"], vc["out_hidden_size"]
        self.eps, self.limit = float(vc.get("rms_norm_eps", 1e-5)), float(vc.get("swiglu_limit", 10.0))
        self.theta = float((vc.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        self.pad = config.get("image_token_id")
        self.pads = (self.pad,)                       # a picture's pads, and a video's pairs' too
        self.start, self.end = config.get("image_start_token_id"), config.get("image_end_token_id")
        # the tower on the card, where the boot put it there (▶ glm5.py `vision_rows`) — else in numpy, here
        self.card = card
        room = card.VIS_ROWS // self.merge ** 2 if card is not None else HOST_MAX_TOKENS
        self.max_tokens = min(PROCESSOR["max_image_tokens"], max_tokens or room)
        self._w = None

    # ── the picture into patches ────────────────────────────────────────────────────────────────────────────
    def patches(self, data):
        """`data` (a picture's file bytes, or a PIL image) -> (pixel_values: patches x 1176 float32, grid (1, h, w))."""
        from PIL import Image as PILImage
        img = data if hasattr(data, "convert") else PILImage.open(io.BytesIO(data))
        img = img.convert("RGB")
        factor = self.patch * self.merge
        th, tw = smart_resize(self.temporal, img.height, img.width, self.temporal, factor,
                              PROCESSOR["min_image_tokens"], self.max_tokens)
        per_token = self.temporal * factor ** 2
        scale = min(th / img.height, tw / img.width)
        if self.temporal * img.height * img.width >= per_token * PROCESSOR["min_image_tokens"]:
            scale = min(1.0, scale)                               # never enlarged when it is big enough already
        ch = max(1, min(th, math.floor(img.height * scale)))
        cw = max(1, min(tw, math.floor(img.width * scale)))
        if (ch, cw) != (img.height, img.width):
            img = img.resize((cw, ch), PILImage.BICUBIC)
        x = np.zeros((th, tw, 3), dtype=np.float32)
        x[:ch, :cw] = np.asarray(img, dtype=np.float32)       # the canvas's rest zeros, as HF pads before rescaling
        x = (x / 255.0 - np.asarray(PROCESSOR["image_mean"], np.float32)) / np.asarray(PROCESSOR["image_std"], np.float32)
        x = x.transpose(2, 0, 1)                                                      # channels, h, w
        gh, gw, p, m = th // self.patch, tw // self.patch, self.patch, self.merge
        x = x.reshape(3, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)  # blocks, in-block, c, p, p
        x = np.repeat(x[:, :, :, :, :, None], self.temporal, axis=5)                   # the still picture, two frames
        return x.reshape(gh * gw, 3 * self.temporal * p * p).astype(np.float32), (1, gh, gw)

    # ── the tower ───────────────────────────────────────────────────────────────────────────────────────────
    def _weights(self):
        """The tower's weights from global.pkl, decoded once, as float32 (about 1.2 GB)."""
        if self._w is None:
            g = pickle.load(open(os.path.join(self.folder.path, "global.pkl"), "rb"))
            w = {}
            for name in [k for k in g if k.startswith("model.visual.")]:
                w[name[len("model.visual."):]] = decoded(g.pop(name)).astype(np.float32)
            if not w:
                raise Refused("%s's global.pkl carries no vision tower" % self.folder.path)
            del g
            self._w = w
        return self._w

    def _positions(self, gh, gw):
        m = self.merge
        rows, cols = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        shape = (gh // m, m, gw // m, m)
        return (rows.reshape(shape).transpose(0, 2, 1, 3).ravel().astype(np.float64),
                cols.reshape(shape).transpose(0, 2, 1, 3).ravel().astype(np.float64))

    def tower(self, pixel_values, grid, dtype=np.float32, keep=None, host=False):
        """pixel_values (patches x 1176) -> the tokens' rows (patches / 4 x the text model's width): on the card where the
        boot holds the tower, else (or `host`, or `keep`) in numpy, in `dtype`. `keep`, a dict, receives the stages the
        check against Hugging Face's reads."""
        _, gh, gw = grid
        rows, cols = self._positions(gh, gw)
        if self.card is not None and not host and keep is None:
            return self.card.vision_rows(pixel_values, np.stack([rows, cols], 1).astype(np.uint32))
        w = self._weights() if dtype == np.float32 else {k: v.astype(dtype) for k, v in self._weights().items()}
        hid, heads, hd, eps, lim = self.hidden, self.heads, self.head_dim, self.eps, self.limit
        x = pixel_values.astype(dtype) @ w["patch_embed.proj.weight"].reshape(hid, -1).T + w["patch_embed.proj.bias"]
        if keep is not None:
            keep["embedded"] = x
        spatial = hd // 2
        inv = 1.0 / self.theta ** (np.arange(0, spatial, 2, dtype=np.float64) / spatial)
        f = np.concatenate([rows[:, None] * inv, cols[:, None] * inv], 1)
        f = np.concatenate([f, f], 1)
        cos, sin = np.cos(f).astype(dtype), np.sin(f).astype(dtype)
        n = len(x)
        for i in range(self.depth):
            p = "blocks.%d." % i
            h = _rms_norm(x, w[p + "norm1.weight"], eps)
            qkv = (h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]).reshape(n, 3, heads, hd)
            att = np.empty((n, heads, hd), dtype)
            for e in range(heads):                                   # a head at a time: n x n scores
                q = _rms_norm(qkv[:, 0, e], w[p + "attn.q_norm.weight"], eps)
                k = _rms_norm(qkv[:, 1, e], w[p + "attn.k_norm.weight"], eps)
                v = qkv[:, 2, e]
                q = q * cos + _rotate_half(q) * sin
                k = k * cos + _rotate_half(k) * sin
                s = (q @ k.T) / math.sqrt(hd)
                s = np.exp(s - s.max(1, keepdims=True))
                att[:, e] = (s / s.sum(1, keepdims=True)) @ v
            x = x + att.reshape(n, hid) @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"]
            h = _rms_norm(x, w[p + "norm2.weight"], eps)
            g = _swiglu(h @ w[p + "mlp.gate_proj.weight"].T + w[p + "mlp.gate_proj.bias"],
                        h @ w[p + "mlp.up_proj.weight"].T + w[p + "mlp.up_proj.bias"], lim)
            x = x + g @ w[p + "mlp.down_proj.weight"].T + w[p + "mlp.down_proj.bias"]
            if keep is not None:
                keep["block_%d" % i] = x
        x = _rms_norm(x, w["post_layernorm.weight"], eps)
        if keep is not None:
            keep["normed"] = x
        m2 = self.merge ** 2
        x = x.reshape(-1, m2, hid).transpose(0, 2, 1).reshape(-1, hid * m2)    # a block's channel, its four side by side
        x = x @ w["downsample.weight"].reshape(self.out, -1).T + w["downsample.bias"]
        if keep is not None:
            keep["downsampled"] = x
        x = x @ w["merger.proj.weight"].T
        x = _gelu_erf(_layer_norm(x, w["merger.post_projection_norm.weight"], w["merger.post_projection_norm.bias"], 1e-5))
        x = _swiglu(x @ w["merger.gate_proj.weight"].T, x @ w["merger.up_proj.weight"].T, lim)
        return (x @ w["merger.down_proj.weight"].T).astype(dtype)

    def read(self, data):
        """A file's bytes read: a still picture as an `Image`; an animated picture or a video file as a `Video`."""
        video = None if hasattr(data, "convert") else frames_of(data)
        if video is not None:
            return Video(self.read_video(*video, max_tokens=self.video_tokens))
        pixel_values, grid = self.patches(data)
        return Image(self.tower(pixel_values, grid), grid, self.merge)

    @property
    def video_tokens(self):
        """The most tokens a video is made into here: half the boot's context, or the host's bound."""
        return max(VIDEO["min_image_tokens"], self.card.max_context // 2 if self.card is not None else 4 * HOST_MAX_TOKENS)

    # ── a video: frames sampled, two frames a temporal patch, each pair read as a picture of its own ──────────────
    @staticmethod
    def sample_frames(total, fps, target=VIDEO["fps"], max_frames=VIDEO["max_frames"]):
        """Hugging Face's (`Glm5NextVideoProcessor.sample_frames`): the indices of the frames read — a frame each 1/`target`
        seconds by the frames' timestamps, evened out over the video when that gives too few or too many, every index
        once, the last repeated to make their number even."""
        duration = total / fps
        max_seconds, extract = int(duration), min(int(duration * target), max_frames)
        stamps = [i / fps for i in range(total)]
        if total < extract:
            idx = np.linspace(0, total - 1, extract, dtype=int).tolist()
        else:
            idx, now = [], 0.0
            for i in range(total):
                if stamps[i] >= now:
                    now += 1.0 / target
                    idx.append(i)
                    if now >= max_seconds:
                        break
        if len(idx) < extract:
            a, b = (idx[0], idx[-1]) if idx else (0, max(total - 1, 0))
            idx = np.linspace(a, b, extract, dtype=int).tolist()
        elif len(idx) > extract:
            idx = np.linspace(0, total - 1, extract, dtype=int).tolist()
        seen, uniq = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        if len(uniq) & 1:
            uniq.append(uniq[-1])
        return uniq

    def video_patches(self, frames, fps, max_tokens=None):
        """`frames` (a sequence of pictures: arrays h x w x 3 of uint8, or PIL images) at `fps` a second -> (pixel_values:
        patches x 1176 float32, grid (t, h, w), each temporal patch's time in seconds), as Hugging Face's video processor
        makes them: the frames sampled, every one fitted into one canvas whose tokens all together lie within the bounds
        (its aspect kept, the rest zeros), two consecutive frames a temporal patch."""
        from PIL import Image as PILImage
        idx = self.sample_frames(len(frames), fps)
        if not idx:
            raise Refused("a video of %d frames at %.3g a second is too short to read a frame from" % (len(frames), fps))
        pics = [f if hasattr(f, "convert") else PILImage.fromarray(np.asarray(f)) for f in (frames[i] for i in idx)]
        pics = [p.convert("RGB") for p in pics]
        n, height, width = len(pics), pics[0].height, pics[0].width
        factor = self.patch * self.merge
        limit = min(VIDEO["max_image_tokens"], max_tokens or VIDEO["max_image_tokens"])
        th, tw = smart_resize(n, height, width, self.temporal, factor, VIDEO["min_image_tokens"], limit)
        per_token = self.temporal * factor ** 2
        scale = min(th / height, tw / width)
        if n * height * width >= per_token * VIDEO["min_image_tokens"]:
            scale = min(1.0, scale)
        ch, cw = max(1, min(th, math.floor(height * scale))), max(1, min(tw, math.floor(width * scale)))
        x = np.zeros((n, th, tw, 3), dtype=np.float32)
        for k, p in enumerate(pics):
            if (p.height, p.width) != (ch, cw):
                p = p.resize((cw, ch), PILImage.BICUBIC)
            x[k, :ch, :cw] = np.asarray(p, dtype=np.float32)
        x = (x / 255.0 - np.asarray(PROCESSOR["image_mean"], np.float32)) / np.asarray(PROCESSOR["image_std"], np.float32)
        x = x.transpose(0, 3, 1, 2)                                                   # frames, channels, h, w
        if n % self.temporal:
            x = np.concatenate([x, np.repeat(x[-1:], self.temporal - n % self.temporal, 0)])
        gt, gh, gw, p, m, T = len(x) // self.temporal, th // self.patch, tw // self.patch, self.patch, self.merge, self.temporal
        x = x.reshape(gt, T, 3, gh // m, m, p, gw // m, m, p).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        times = [idx[i] / fps for i in range(0, len(idx), self.temporal)][:gt]
        return x.reshape(gt * gh * gw, 3 * T * p * p).astype(np.float32), (gt, gh, gw), times

    def read_video(self, frames, fps, max_tokens=None):
        """A video's frames read: a list of (`Image`, time in seconds) a pair of frames — each pair the tower reads on its
        own, as Hugging Face's `get_video_features` does — in order."""
        pixel_values, (gt, gh, gw), times = self.video_patches(frames, fps, max_tokens)
        per = gh * gw
        return [(Image(self.tower(pixel_values[i * per:(i + 1) * per], (1, gh, gw)), (1, gh, gw), self.merge), times[i])
                for i in range(gt)]


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _rotate_half(x):
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate([-b, a], -1)


def _swiglu(gate, up, limit):
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    return gate / (1.0 + np.exp(-gate)) * up


_ERF = np.frompyfunc(math.erf, 1, 1)


def _gelu_erf(x):
    return (0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)).astype(x.dtype))).astype(x.dtype)
