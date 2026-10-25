"""Critic check 3 (CPU): pixel-level manipulation check for setting M.
PSNR of jpeg(90) vs clean on the RAISE test crops (orig arm, the 512 crop the run used) and an 8x8 JPEG-grid blockiness ratio
on the raw files (RAISE PNG, VISION native JPEG, Firefly, one PNG generator): ratio = mean |horizontal+vertical grey diff| across
x%8==7|8 boundaries / elsewhere; ~1.0 = no aligned JPEG grid."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import perturb
OUT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "experiments/003-baselines-eval/splits.csv", keep_default_na=False)
L = []
r = d[(d.pool == "raise") & (d.split == "test")].sort_values("sha256").head(40)
ps = []
for p in r.path:
    im = Image.open(ROOT / p).convert("RGB")
    a = np.asarray(im, np.float64); b = np.asarray(perturb.jpeg(im, 90), np.float64)
    ps.append(10 * np.log10(255 ** 2 / ((a - b) ** 2).mean()))
L.append(f"RAISE test crops (n={len(ps)}): PSNR jpeg(90) vs clean median {np.median(ps):.2f} dB (min {min(ps):.2f}, max {max(ps):.2f})")
def block(path):
    g = np.asarray(Image.open(path).convert("L"), np.float64)
    dx = np.abs(np.diff(g, axis=1)).mean(0); dy = np.abs(np.diff(g, axis=0)).mean(1)
    bx = (np.arange(len(dx)) % 8 == 7); by = (np.arange(len(dy)) % 8 == 7)
    return 0.5 * (dx[bx].mean() / dx[~bx].mean() + dy[by].mean() / dy[~by].mean())
for name, q in [("raise", (d.pool == "raise")), ("vision_native", d.pool == "vision_native"),
                ("firefly", d.generator == "firefly"), ("dalle3", d.generator == "dalle3"), ("sd3_large", d.generator == "sd3_large")]:
    rows = d[q & (d.split == "test")].sort_values("sha256").head(30)
    v = [block(ROOT / p) for p in rows.src_path]
    L.append(f"raw-file 8x8 blockiness ratio {name:14s} (n={len(v)}): median {np.median(v):.3f} [p10 {np.percentile(v,10):.3f}, p90 {np.percentile(v,90):.3f}]")
(OUT / "c3_manipulation_pixels.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
