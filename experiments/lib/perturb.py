"""Deterministic image conditions. All operate on PIL RGB images and are identical for both classes."""
import hashlib
import io
import re

import numpy as np
from PIL import Image

SOCIAL_SCALE = (0.7, 1.0)
SOCIAL_Q = (70, 100)


def resize(img, f):
    """Scale by factor f (bicubic; PIL's resize antialiases when downscaling)."""
    w, h = img.size
    return img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BICUBIC)


def jpeg(img, q):
    """JPEG round trip at quality q (libjpeg defaults, 4:2:0)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def seed_from_key(key, base_seed=0):
    """Stable per-image seed from any string key (e.g. sha256) and a base seed."""
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{key}".encode()).digest()[:8], "little")


def social_sim(img, key="", seed=0):
    """Random resize in [0.7, 1.0] then JPEG q in [70, 100]; parameters seeded per image."""
    rng = np.random.default_rng(seed_from_key(key, seed))
    f = float(rng.uniform(*SOCIAL_SCALE))
    q = int(rng.integers(SOCIAL_Q[0], SOCIAL_Q[1] + 1))
    return jpeg(resize(img, f), q)


def jpeg75_resize05(img):
    return jpeg(resize(img, 0.5), 75)


def apply_condition(img, condition, key="", seed=0):
    """condition: 'clean' | 'jpeg75_resize05' | 'resize(0.5)' | 'jpeg(75)' | 'social_sim'."""
    img = img.convert("RGB")
    if condition == "clean":
        return img
    if condition == "jpeg75_resize05":
        return jpeg75_resize05(img)
    if condition == "social_sim":
        return social_sim(img, key, seed)
    m = re.fullmatch(r"(resize|jpeg)\(([\d.]+)\)", condition)
    if m:
        return resize(img, float(m.group(2))) if m.group(1) == "resize" else jpeg(img, float(m.group(2)))
    raise ValueError(f"unknown condition {condition!r}")
