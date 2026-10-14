"""AN IMAGE AS A QWEN 3.5 MODEL READS IT — the picture made into patches, the vision tower over them, and the positions
its tokens take in the conversation.

    v = Vision(folder)                          # a Qwen 3.5 model folder whose global.pkl carries model.visual.*
    image = v.read(png_or_jpeg_bytes)           # .rows (tokens x the text model's width, float32), .grid (t, h, w)
    tokens = chat-framed text with image.tokens image pads where the picture goes
    video = v.read(gif_or_mp4_bytes)            # a Video: .pairs, each pair of frames an Image and its time in seconds

The tower runs on the card through ai_qwen_3's three words over nn's (▶ qwen3_5.py `vision_rows`) where the boot made
room for it (`picture_pixels`), its weights the pack's codes as they are; the host computes only what a picture's size
decides — its patches, each patch's row and column and its rows of the position table. Else it runs here, in numpy
(float32): the reference, checked against Hugging Face's (`test/src_vision_reference.py`, the same steps in float64), its
weights decoded once when the first picture comes, as floats (about 1.8 GB of RAM). ▶ `test/src_runtime_vision_card.py`
for the card's against Hugging Face's.

The picture is prepared as Hugging Face's processor prepares it — resized to multiples of 32 within the pixel bounds
(Pillow's antialiased bicubic, which torchvision's follows), scaled to [-1, 1], the still picture as two frames, cut into
16 x 16 patches in 2 x 2-block order — except that the most pixels are the boot's `picture_pixels` on the card (4,194,304
by default: a picture of 2048 x 2048, 16,384 patches, 4,096 tokens) and 1,048,576 in numpy, against the model's
16,777,216: the tower's attention is every patch to every patch, its time the square of the patches.

A video is read as Hugging Face's video processor and model read one: frames taken at 2 a second (4 to 768 of them,
evenly over the video), every frame resized to one canvas whose pixels over all the frames lie within the repositories'
bounds, two consecutive frames a temporal patch, and each pair read by the tower on its own (its attention within the
pair, as the model's own segments are) — a picture of two different frames, whose time is the middle of its two. A
moving picture Pillow reads (an animated GIF, WebP or PNG) is read as it is; any other video file through PyAV, where it
is installed (`frames_of`).
"""
import io
import json
import math
import os
import pickle

import numpy as np

from .model_folder import Refused

# the Qwen 3.5 repositories' preprocessor_config.json, which the packs do not carry (one for the three sizes)
PROCESSOR = {"min_pixels": 65536, "max_pixels": 16777216, "image_mean": (0.5, 0.5, 0.5), "image_std": (0.5, 0.5, 0.5)}
HOST_MAX_PIXELS = 1 << 20
# the repositories' video_preprocessor_config.json (the pixels of all the frames together) and Hugging Face's
# Qwen3VLVideoProcessor (the rate and the number of frames)
VIDEO = {"fps": 2.0, "min_frames": 4, "max_frames": 768, "min_pixels": 4096, "max_pixels": 25165824}


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Hugging Face's: both sides a multiple of `factor`, the area within the bounds, the aspect kept."""
    if max(height, width) / min(height, width) > 200:
        raise Refused("a picture more than 200 times as long as it is wide")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def smart_resize_video(frames, height, width, temporal, factor, min_pixels, max_pixels):
    """Hugging Face's for a video: both sides a multiple of `factor`, the pixels of every frame together within the
    bounds, the aspect kept."""
    if height < factor or width < factor:
        raise Refused("a video smaller than %d x %d" % (factor, factor))
    if max(height, width) / min(height, width) > 200:
        raise Refused("a video more than 200 times as long as it is wide")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    t = math.ceil(frames / temporal) * temporal
    if t * h * w > max_pixels:
        beta = math.sqrt(frames * height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif t * h * w < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


class Image:
    """A picture read: its tokens' rows, its grid of patches (t, h, w), and how many tokens it is."""

    def __init__(self, rows, grid, merge):
        self.rows = rows
        self.grid = tuple(int(g) for g in grid)
        self.tokens = self.grid[0] * (self.grid[1] // merge) * (self.grid[2] // merge)


class Vision:
    mrope = True                    # its tokens' positions are M-RoPE's three (▶ `positions`)

    def __init__(self, folder, card=None):
        config = json.load(open(os.path.join(folder.path, "config.json")))
        vc = config.get("vision_config")
        if not vc or vc.get("model_type", "").split("_")[0] != "qwen3":
            raise Refused("%s has no Qwen 3.5 vision tower" % folder.path)
        self.folder = folder
        self.hidden, self.heads = vc["hidden_size"], vc["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.patch, self.temporal, self.merge = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
        self.depth, self.out = vc["depth"], vc["out_hidden_size"]
        self.side = int(round(vc["num_position_embeddings"] ** 0.5))
        self.theta = float((vc.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        self.pad, self.video_pad = config.get("image_token_id"), config.get("video_token_id")
        self.pads = (self.pad, self.video_pad)        # a picture's pads, and a video's pairs'
        self.start, self.end = config.get("vision_start_token_id"), config.get("vision_end_token_id")
        self._w = None
        # the tower on the card, where the boot put it there (▶ qwen3_5.py `vision_rows`): its weights are the card's, and
        # the host keeps only the position table — else the tower in numpy, here
        self.card = card
        self.max_pixels = min(PROCESSOR["max_pixels"], card.VIS_ROWS * self.patch ** 2 if card else HOST_MAX_PIXELS)

    # ── the picture into patches ────────────────────────────────────────────────────────────────────────────
    def patches(self, data):
        """`data` (a picture's file bytes, or a PIL image) -> (pixel_values: patches x 1536 float32, grid (1, h, w))."""
        from PIL import Image as PILImage
        img = data if hasattr(data, "convert") else PILImage.open(io.BytesIO(data))
        img = img.convert("RGB")
        factor = self.patch * self.merge
        h, w = smart_resize(img.height, img.width, factor, PROCESSOR["min_pixels"], self.max_pixels)
        if (h, w) != (img.height, img.width):
            img = img.resize((w, h), PILImage.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.asarray(PROCESSOR["image_mean"], np.float32)) / np.asarray(PROCESSOR["image_std"], np.float32)
        x = x.transpose(2, 0, 1)                                                      # channels, h, w
        gh, gw, p, m = h // self.patch, w // self.patch, self.patch, self.merge
        x = x.reshape(3, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)  # blocks, in-block, c, p, p
        x = np.repeat(x[:, :, :, :, :, None], self.temporal, axis=5)                   # the still picture, two frames
        return x.reshape(gh * gw, 3 * self.temporal * p * p).astype(np.float32), (1, gh, gw)

    # ── the tower ───────────────────────────────────────────────────────────────────────────────────────────
    def _weights(self):
        """The tower's weights from global.pkl, decoded once, as floats."""
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

    def _m(self, name):
        return self._weights()[name]

    def _positions(self, gh, gw):
        m = self.merge
        rows, cols = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        shape = (gh // m, m, gw // m, m)
        return (rows.reshape(shape).transpose(0, 2, 1, 3).ravel().astype(np.float64),
                cols.reshape(shape).transpose(0, 2, 1, 3).ravel().astype(np.float64))

    def _table(self, rows, cols, gh, gw):
        def axis(i, n):
            src = i * (self.side - 1) / max(n - 1, 1)
            f = np.floor(src)
            taps = np.stack([f, f + 1], 1)
            return np.clip(taps, 0, self.side - 1).astype(np.int64), np.clip(1 - np.abs(src[:, None] - taps), 0, None)
        ht, hw = axis(rows, gh)
        wt, ww = axis(cols, gw)
        idx = (ht[:, :, None] * self.side + wt[:, None, :]).reshape(-1, 4)
        wgt = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4).astype(np.float32)
        table = self.card.POS_TABLE if self.card is not None else self._m("pos_embed.weight")
        return (table[idx] * wgt[:, :, None]).sum(1)

    def tower(self, pixel_values, grid, host=False):
        """pixel_values (patches x 1536) -> the tokens' rows (patches / 4 x the text model's width), float32: on the card
        where the boot holds the tower, else (or `host`) in numpy."""
        _, gh, gw = grid
        rows, cols = self._positions(gh, gw)
        if self.card is not None and not host:
            rowcol = np.stack([rows, cols], 1).astype(np.uint32)
            return self.card.vision_rows(pixel_values, rowcol, self._table(rows, cols, gh, gw))
        hid, heads, hd = self.hidden, self.heads, self.head_dim
        x = pixel_values @ self._m("patch_embed.proj.weight").reshape(hid, -1).T + self._m("patch_embed.proj.bias")
        x = x + self._table(rows, cols, gh, gw)
        inv = 1.0 / self.theta ** (np.arange(0, hd // 2, 2, dtype=np.float64) / (hd // 2))
        f = np.concatenate([rows[:, None] * inv, cols[:, None] * inv], 1)
        f = np.concatenate([f, f], 1)
        cos, sin = np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)
        n = len(x)
        for i in range(self.depth):
            p = "blocks.%d." % i
            h = _layer_norm(x, self._m(p + "norm1.weight"), self._m(p + "norm1.bias"))
            qkv = (h @ self._m(p + "attn.qkv.weight").T + self._m(p + "attn.qkv.bias")).reshape(n, 3, heads, hd)
            att = np.empty((n, heads, hd), np.float32)
            for e in range(heads):                                   # a head at a time: n x n scores
                q, k, v = qkv[:, 0, e], qkv[:, 1, e], qkv[:, 2, e]
                q = q * cos + _rotate_half(q) * sin
                k = k * cos + _rotate_half(k) * sin
                s = (q @ k.T) / math.sqrt(hd)
                s = np.exp(s - s.max(1, keepdims=True))
                att[:, e] = (s / s.sum(1, keepdims=True)) @ v
            x = x + att.reshape(n, hid) @ self._m(p + "attn.proj.weight").T + self._m(p + "attn.proj.bias")
            h = _layer_norm(x, self._m(p + "norm2.weight"), self._m(p + "norm2.bias"))
            h = _gelu_tanh(h @ self._m(p + "mlp.linear_fc1.weight").T + self._m(p + "mlp.linear_fc1.bias"))
            x = x + h @ self._m(p + "mlp.linear_fc2.weight").T + self._m(p + "mlp.linear_fc2.bias")
        x = _layer_norm(x, self._m("merger.norm.weight"), self._m("merger.norm.bias")).reshape(-1, hid * self.merge ** 2)
        x = _gelu_erf(x @ self._m("merger.linear_fc1.weight").T + self._m("merger.linear_fc1.bias"))
        return (x @ self._m("merger.linear_fc2.weight").T + self._m("merger.linear_fc2.bias")).astype(np.float32)

    def read(self, data):
        """A file's bytes read: a still picture as an `Image`; an animated picture or a video file as a `Video`."""
        video = None if hasattr(data, "convert") else frames_of(data)
        if video is not None:
            return Video(self.read_video(*video, max_tokens=self.video_tokens))
        pixel_values, grid = self.patches(data)
        return Image(self.tower(pixel_values, grid), grid, self.merge)

    @property
    def video_tokens(self):
        """The most tokens a video is made into here: half the boot's context, or four of the host's pictures."""
        per_picture = HOST_MAX_PIXELS // (self.patch * self.merge) ** 2
        return max(16, self.card.max_context // 2 if self.card is not None else 4 * per_picture)

    # ── a video: frames sampled, two frames a temporal patch, each pair read as a picture of its own ──────────────
    @staticmethod
    def sample_frames(total, fps, target=VIDEO["fps"]):
        """Hugging Face's (`Qwen3VLVideoProcessor.sample_frames`): the indices of the frames read — `target` a second of
        the video, at least 4 and at most 768 and never more than it has, evenly from the first to the last."""
        n = min(max(int(total / fps * target), VIDEO["min_frames"]), VIDEO["max_frames"], total)
        return np.linspace(0, total - 1, n).round().astype(int).tolist()

    def video_patches(self, frames, fps, max_tokens=None):
        """`frames` (a sequence of pictures: arrays h x w x 3 of uint8, or PIL images) at `fps` a second -> (pixel_values:
        patches x 1536 float32, grid (t, h, w), each temporal patch's time in seconds), as Hugging Face's video processor
        makes them: the frames sampled, every one resized to one canvas (multiples of 32) whose pixels over all the frames
        lie within the bounds — and, here, a frame's within the pictures' (`picture_pixels`), the video's tokens within
        `max_tokens` — the last frame repeated to make their number even, two consecutive frames a temporal patch, its
        time the middle of its two frames'."""
        from PIL import Image as PILImage
        idx = self.sample_frames(len(frames), fps)
        if not idx:
            raise Refused("a video of %d frames is too short to read a frame from" % len(frames))
        pics = [f if hasattr(f, "convert") else PILImage.fromarray(np.asarray(f)) for f in (frames[i] for i in idx)]
        pics = [p.convert("RGB") for p in pics]
        n, factor, T = len(pics), self.patch * self.merge, self.temporal
        t = math.ceil(n / T) * T
        limit = min(VIDEO["max_pixels"], t * self.max_pixels, (max_tokens or VIDEO["max_pixels"]) * T * factor ** 2)
        h, w = smart_resize_video(n, pics[0].height, pics[0].width, T, factor, VIDEO["min_pixels"], limit)
        x = np.stack([np.asarray(p if (p.height, p.width) == (h, w) else p.resize((w, h), PILImage.BICUBIC), np.float32)
                      for p in pics])
        mean, std = (np.asarray(PROCESSOR[k], np.float32) for k in ("image_mean", "image_std"))
        x = (x / 255.0 - mean) / std
        x = x.transpose(0, 3, 1, 2)                                                   # frames, channels, h, w
        if n % T:
            x = np.concatenate([x, np.repeat(x[-1:], T - n % T, 0)])
        gt, gh, gw, p, m = len(x) // T, h // self.patch, w // self.patch, self.patch, self.merge
        x = x.reshape(gt, T, 3, gh // m, m, p, gw // m, m, p).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        at = idx + [idx[-1]] * (-len(idx) % T)
        times = [(at[k] + at[k + T - 1]) / 2 / fps for k in range(0, len(at), T)]
        return x.reshape(gt * gh * gw, 3 * T * p * p).astype(np.float32), (gt, gh, gw), times

    def read_video(self, frames, fps, max_tokens=None):
        """A video's frames read: a list of (`Image`, time in seconds) a pair of frames — each pair the tower reads on its
        own, as Hugging Face's tower does within its segments — in order."""
        pixel_values, (gt, gh, gw), times = self.video_patches(frames, fps, max_tokens)
        per = gh * gw
        return [(Image(self.tower(pixel_values[i * per:(i + 1) * per], (1, gh, gw)), (1, gh, gw), self.merge), times[i])
                for i in range(gt)]


class Video:
    """A video read: each pair of its frames an `Image` and its time in seconds, in order; its tokens all of theirs."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.tokens = sum(im.tokens for im, _ in pairs)
        self.grid = pairs[0][0].grid if pairs else (1, 0, 0)


def frames_of(data):
    """A file's frames and their rate when it is moving pictures — an animated GIF, WebP or PNG through Pillow, any other
    video through PyAV where it is installed (requirements-nonfree.txt) — or None for a still picture."""
    from PIL import Image as PILImage, UnidentifiedImageError
    try:
        img = PILImage.open(io.BytesIO(data))
    except UnidentifiedImageError:
        img = None
    if img is not None:
        n = getattr(img, "n_frames", 1)
        if n <= 1:
            return None
        frames, ms = [], 0.0
        for i in range(n):
            img.seek(i)
            frames.append(np.asarray(img.convert("RGB")))
            ms += float(img.info.get("duration") or 100)
        return frames, n / (ms / 1000.0)
    try:
        import av
    except ImportError:
        raise Refused("not a picture Pillow reads; a video file is read through PyAV, which is not installed: "
                      "`pip install -r requirements-nonfree.txt` (in the Docker image, its scripts' install_nonfree)")
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise Refused("a video with no frames")
    return frames, fps


def decoded(rec):
    """A record of global.pkl decoded to its values, in its original shape (float64)."""
    from nn_unpack_model import decode_record
    data, lut = np.asarray(rec["vbr_data"]).tobytes(), np.asarray(rec["vbr_lut"]).tobytes()
    r = {"scalars": {k: v for k, v in rec.items() if k not in ("vbr_data", "vbr_lut")}, "record_offset": 0,
         "record_nbytes": len(data) + len(lut),
         "fields": {"vbr_data": {"rel": 0, "nbytes": len(data)}, "vbr_lut": {"rel": len(data), "nbytes": len(lut)}}}
    return decode_record(r, data + lut)


def positions(tokens, images, pads, merge, start=0):
    """Every token's three rotary positions (time, row, column) and the next text position, as Hugging Face's
    `get_rope_index` makes them: a text token's three are its own position; a run of pads is an image — `images` in
    order, (index of its first pad, grid); a video's pair of frames one of its own — whose tokens take (p, p + row,
    p + column) from the position p it starts at, the text after it going on from p + max(rows, columns) of its merged
    grid. `pads` the pad ids (an image's, a video's); `start` is where the first token sits. Answers (n x 3 uint32, the
    next position)."""
    pads = set(np.atleast_1d(pads).tolist())
    out = np.empty((len(tokens), 3), dtype=np.int64)
    p, i = start, 0
    for first, grid in images:
        while i < first:
            out[i] = p
            p += 1
            i += 1
        _, gh, gw = grid
        h, w = gh // merge, gw // merge
        r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        n = h * w
        if first < i or first + n > len(tokens) or any(t not in pads for t in tokens[first:first + n]):
            raise Refused("an image's %d pads are not where it says they start (%d)" % (n, first))
        out[first:first + n, 0] = p
        out[first:first + n, 1] = p + r.ravel()
        out[first:first + n, 2] = p + c.ravel()
        p += max(h, w)
        i = first + n
    while i < len(tokens):
        out[i] = p
        p += 1
        i += 1
    return out.astype(np.uint32), p


def _layer_norm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _rotate_half(x):
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate([-b, a], -1)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x ** 3)))


_ERF = np.frompyfunc(math.erf, 1, 1)


def _gelu_erf(x):
    return (0.5 * x * (1.0 + _ERF(x / np.float32(math.sqrt(2.0))).astype(np.float32))).astype(np.float32)
