"""A PICTURE AS DEEPSEEK-V4.1 READS IT — the picture made into patches, its vision tower over them on the card, and the
span of rows its tokens are.

    v = model.vision()                          # a DeepSeek-V4.1 boot that holds the tower (▶ deepseek4.py `_vision_plan`)
    image = v.read(png_or_jpeg_bytes)           # .rows (the span's tokens x the text model's width, float32), .tokens
    tokens = chat-framed text with image.tokens `<｜deepseek_image｜>` between the turn's text blocks

The release's code (`inference/image_processor.py`, `inference/vision.py`, `inference/model.py`) as written here:

  ① the picture's plan (`plan_image_grid`): a picture under `min_pixels` (544²) enlarged to it, its aspect kept; its sides
    up to whole 14-pixel patches; and, where its tokens would pass `max_image_tokens` (1,024), the largest size that fits;
  ② the picture padded to that size, centred on grey 127 (Pillow's `ImageOps.pad`), scaled to [-1, 1] and rounded to
    bfloat16, cut into 14 x 14 patches in reading order, a patch's channels one after another;
  ③ the tower on the card (▶ the package's `vision_*` verbs): the patch embedding, 32 blocks, the last norm and the aligner
    — a row a 3 x 3 block of patches, the picture's last rows and columns padded to whole blocks;
  ④ the span: `IMAGE_START`, each row of blocks then `IMAGE_NEW_LINE`, `IMAGE_END` — every position the image token, the
    blocks' rows the aligner's and the three markers the model's learned rows. The text model routes the span's tokens
    by each layer's `bias_vl` and leaves them out of the n-gram memory (▶ deepseek4.py `prefill`).
"""
import io
import math

import numpy as np

from .model_folder import Refused
from .vision import Image


def plan_image_grid(width, height, patch, ratio, max_tokens, min_pixels, max_wh_ratio=None):
    """The release's `plan_image_grid` and `safe_resize`: (n_llm_h, n_llm_w, best_height, best_width) for a picture of
    `width` x `height` — its tokens' grid and the size it is padded to."""
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    best_w, best_h = math.ceil(width / patch) * patch, math.ceil(height / patch) * patch
    grid = lambda bh, bw: (math.ceil((bh // patch) / ratio), math.ceil((bw // patch) / ratio))
    count = lambda h, w: h * (w + 1) + 2
    n_h, n_w = grid(best_h, best_w)
    if count(n_h, n_w) > max_tokens:
        r = height / width
        max_w = math.sqrt((max_tokens - 2) / r + 0.25) - 0.5
        max_h, cell = max_w * r, patch * ratio
        if max_w < 1.0:                                           # very tall: one column
            best_h, best_w = (max_tokens - 2) // 2 * cell, cell
        elif max_h < 1.0:                                         # very wide: one row
            best_h, best_w = cell, (max_tokens - 3) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            best_h = math.floor(height * beta / patch) * patch
            best_w = math.floor(width * beta / patch) * patch
        n_h, n_w = grid(best_h, best_w)
        if count(n_h, n_w) > max_tokens:
            raise Refused("a picture of %d x %d does not fit %d tokens" % (width, height, max_tokens))
    return n_h, n_w, best_h, best_w


def bfloat16(x):
    """float32 values rounded to bfloat16's (to nearest, ties to even), as float32."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    b = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return b.view(np.float32)


class DeepSeekVision:
    mrope = False                   # its tokens keep their places in the cache: only their rows go in
    max_video_tokens = 0            # pictures only

    def __init__(self, card):
        vc = card.VC
        self.card = card
        self.patch, self.merge = vc["patch_size"], vc["downsample_ratio"]
        self.min_pixels, self.max_wh_ratio = int(vc["min_pixels"]), vc.get("max_wh_ratio")
        self.max_tokens = card.VIS_TOKENS
        self.pad = int(card.IMAGE_TOKEN)
        self.pads = (self.pad,)

    def patches(self, data):
        """`data` (a picture's file bytes, or a PIL image) -> (patches: n x 588 float32, n_vit_h, n_vit_w, n_llm_h,
        n_llm_w), as the release's `load_image` makes them."""
        from PIL import Image as PILImage, ImageOps
        img = data if hasattr(data, "convert") else PILImage.open(io.BytesIO(data))
        img = img.convert("RGB")
        p = self.patch
        n_h, n_w, bh, bw = plan_image_grid(img.width, img.height, p, self.merge, self.max_tokens, self.min_pixels,
                                           self.max_wh_ratio)
        if self.max_wh_ratio is not None and img.width >= self.max_wh_ratio * img.height:
            img = img.resize((bw, bh))
        else:
            img = ImageOps.pad(img, (bw, bh), color=(127, 127, 127))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        x = bfloat16((x - np.float32(0.5)) / np.float32(0.5))
        gh, gw = bh // p, bw // p
        x = x.reshape(3, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, 3 * p * p)
        return x, gh, gw, n_h, n_w

    def span(self, aligned, n_h, n_w):
        """The span's rows: the start, each row of blocks then a newline, the end — `aligned` the blocks' rows in order."""
        c = self.card
        rows = [c.VIS_MARKS["image_start"]]
        for i in range(n_h):
            rows += list(aligned[i * n_w:(i + 1) * n_w]) + [c.VIS_MARKS["image_newline"]]
        rows.append(c.VIS_MARKS["image_end"])
        return np.stack(rows).astype(np.float32)

    def read(self, data):
        """A file's bytes read as a still picture: an `Image` whose rows are its span's, a row a token."""
        try:
            from PIL import Image as PILImage
            img = data if hasattr(data, "convert") else PILImage.open(io.BytesIO(data))
            if getattr(img, "n_frames", 1) > 1:
                raise Refused("DeepSeek-V4.1 reads still pictures, not an animation or a video")
        except Refused:
            raise
        except Exception:                                         # noqa: BLE001 — not a picture Pillow reads
            raise Refused("DeepSeek-V4.1 reads still pictures; this file is not one Pillow reads")
        x, gh, gw, n_h, n_w = self.patches(img)
        aligned = self.card.vision_rows(x, gh, gw)
        image = Image(self.span(aligned, n_h, n_w), (1, self.merge * n_h, self.merge * n_w), self.merge)
        image.tokens = n_h * (n_w + 1) + 2
        return image
