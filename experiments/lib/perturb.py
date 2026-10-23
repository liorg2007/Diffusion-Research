"""Deterministic image conditions. All operate on PIL RGB images and are identical for both classes."""
import hashlib
import io
import re

import numpy as np
from PIL import Image

SOCIAL_SCALE = (0.7, 1.0)
SOCIAL_Q = (70, 100)
MAVALI_MEDIUM = (0.75, 50)              # resize factor, JPEG quality
SCREENSHOT_FACTORS = (0.5, 0.67, 1.25)  # resample factor drawn per image
SCREENSHOT_CROP = (0.02, 0.05)          # border fraction removed from EACH side (one fraction per image)


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


def jr(img, q, f):
    """jpeg(resize(img, f), q); f == 1.0 means no resize, just JPEG (augmentation-grid cell)."""
    return jpeg(img if float(f) == 1.0 else resize(img, f), q)


def mavali_medium(img):
    """Held-out laundering: resize 0.75 then JPEG q50."""
    return jr(img, MAVALI_MEDIUM[1], MAVALI_MEDIUM[0])


def screenshot_sim(img, key="", seed=0):
    """Held-out laundering: resample by a per-image factor in {0.5, 0.67, 1.25} (BILINEAR, 8-bit RGB, no JPEG),
    then remove one seeded fraction in [0.02, 0.05] of the width/height from each side (symmetric border crop).
    A 512 input gives >= 230 px (0.5 x 512 = 256 minus 2 x round(0.05 x 256) = 26), so a 224 center crop still fits."""
    rng = np.random.default_rng(seed_from_key(key, seed))
    f = float(SCREENSHOT_FACTORS[int(rng.integers(len(SCREENSHOT_FACTORS)))])
    frac = float(rng.uniform(*SCREENSHOT_CROP))
    w, h = img.size
    out = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
    w, h = out.size
    dx, dy = int(round(frac * w)), int(round(frac * h))
    return out.crop((dx, dy, w - dx, h - dy))


def apply_condition(img, condition, key="", seed=0):
    """condition: 'clean' | 'jpeg75_resize05' | 'resize(f)' | 'jpeg(q)' | 'social_sim' | 'mavali_medium'
    | 'screenshot_sim' | 'jr(q,f)' (= jpeg(resize(img, f), q); f=1.0: JPEG only)."""
    img = img.convert("RGB")
    if condition == "clean":
        return img
    if condition == "jpeg75_resize05":
        return jpeg75_resize05(img)
    if condition == "social_sim":
        return social_sim(img, key, seed)
    if condition == "mavali_medium":
        return mavali_medium(img)
    if condition == "screenshot_sim":
        return screenshot_sim(img, key, seed)
    m = re.fullmatch(r"jr\(\s*(\d+)\s*,\s*([\d.]+)\s*\)", condition)
    if m:
        return jr(img, int(m.group(1)), float(m.group(2)))
    m = re.fullmatch(r"(resize|jpeg)\(([\d.]+)\)", condition)
    if m:
        return resize(img, float(m.group(2))) if m.group(1) == "resize" else jpeg(img, float(m.group(2)))
    raise ValueError(f"unknown condition {condition!r}")
