"""Synthetic datasets for tests (no downloads). Reals: smooth gradients + mild noise, JPEG (with EXIF) per
'camera'. Fakes: gradients + strong noise, PNG per 'generator'."""
import numpy as np
import pandas as pd
from PIL import Image

from manifest import make_row


def gradient(size, rng, noise):
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = np.stack([xx * rng.uniform(100, 255), yy * rng.uniform(100, 255), (xx + yy) / 2 * 200], -1)
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def exif_bytes(make):
    ex = Image.Exif()
    ex[0x010F] = make
    return ex.tobytes()


def make_dataset(root, n_cams=4, n_real=12, gens=("genA", "genB", "genC"), n_fake=16, size=96,
                 leaky=True, seed=0, n_small=2):
    """leaky=True: reals are JPEG (varying q, EXIF), fakes PNG -> format leakage. leaky=False: all PNG."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cams):
        for i in range(n_real):
            arr = gradient(size, rng, noise=4)
            if leaky:
                p = root / f"real_cam{c}_{i}.jpg"
                Image.fromarray(arr).save(p, quality=80 + 5 * c, exif=exif_bytes(f"CAM{c}"))
            else:
                p = root / f"real_cam{c}_{i}.png"
                Image.fromarray(arr).save(p)
            rows.append(make_row(p, 0, "synthReal", "real", f"cam{c}", "test"))
    for g in gens:
        for i in range(n_fake):
            p = root / f"fake_{g}_{i}.png"
            Image.fromarray(gradient(size, rng, noise=20 if leaky else 4)).save(p)
            rows.append(make_row(p, 1, "synthFake", g, "n/a", "test"))
    for i in range(n_small):  # smaller than any crop used in tests -> must be skipped
        p = root / f"small_{i}.png"
        Image.fromarray(gradient(16, rng, 4)).save(p)
        rows.append(make_row(p, 0, "synthReal", "real", "cam0", "test"))
    return pd.DataFrame(rows)
