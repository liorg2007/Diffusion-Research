"""Synthetic checks of the statistic battery (run: env/.venv/bin/python experiments/014-statistical-field/selftest.py)."""
import sys
from pathlib import Path
import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

rng = np.random.default_rng(0)
ok = True


def check(name, val, lo, hi):
    global ok
    good = lo <= val <= hi
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name}: {val:.4f} (expected in [{lo}, {hi}])")


def power_law_image(alpha, n=512):
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1
    amp = f ** (-alpha / 2)
    amp[0, 0] = 0
    img = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    return img / img.std() * 30 + 128


# 1. 1/f^alpha noise recovers alpha
for a in (1.0, 2.0, 3.0):
    est = np.mean([run.s1_spectrum(power_law_image(a))["s1_alpha"] for _ in range(4)])
    check(f"S1 alpha recovery (true {a})", est, a - 0.15, a + 0.15)

# 2. Gaussian vs Laplacian white noise: excess kurtosis of Haar detail = kappa/4 at level 1 (sum of 4 iid), kappa/16 at level 2
G = rng.normal(0, 20, (512, 512))
L = rng.laplace(0, 20 / np.sqrt(2), (512, 512))
sg, sl = run.s2_wavelet(G), run.s2_wavelet(L)
check("Gaussian level-1 excess kurtosis (0)", np.mean([sg[f"s2_l1_{o}_kurt"] for o in "HVD"]), -0.08, 0.08)
check("Laplacian level-1 excess kurtosis (3/4)", np.mean([sl[f"s2_l1_{o}_kurt"] for o in "HVD"]), 0.65, 0.85)
check("Laplacian level-2 excess kurtosis (3/16)", np.mean([sl[f"s2_l2_{o}_kurt"] for o in "HVD"]), 0.10, 0.28)
check("Gaussian level-1 log10 variance (log10 20^2 = 2.602)", np.mean([sg[f"s2_l1_{o}_logvar"] for o in "HVD"]), 2.57, 2.63)

# 3. Bayer-mosaic + bilinear demosaic: green residual energy is zero on the R/B sites -> checkerboard contrast ~ 1; plain noisy scene ~ 0
n = 256
scene = np.stack([ndi for ndi in [power_law_image(2.0, n) for _ in range(3)]], -1) * 0.8 + 40 + rng.normal(0, 2, (n, n, 3))
Rm = np.zeros((n, n), bool); Gm = np.zeros((n, n), bool); Bm = np.zeros((n, n), bool)
Rm[0::2, 0::2] = True; Gm[0::2, 1::2] = True; Gm[1::2, 0::2] = True; Bm[1::2, 1::2] = True
from scipy import ndimage as ndi
kg = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
kc = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
dem = np.zeros_like(scene)
for c, m, k in ((0, Rm, kc), (1, Gm, kg), (2, Bm, kc)):
    num = ndi.convolve(np.where(m, scene[..., c], 0.0), k, mode="mirror")
    den = ndi.convolve(m.astype(float), k, mode="mirror")
    dem[..., c] = np.where(m, scene[..., c], num / den)
s_dem, s_raw = run.s4_colour(dem), run.s4_colour(scene)
check("S4 phase_diag, demosaiced Bayer image (~1)", s_dem["s4_phase_diag"], 0.9, 1.0)
check("S4 phase_diag, non-mosaiced image (~0)", s_raw["s4_phase_diag"], 0.0, 0.05)
check("S4 phase_cv, demosaiced > raw", s_dem["s4_phase_cv"] - s_raw["s4_phase_cv"], 0.5, 5)

# 4. Poisson-Gaussian noise: var = a*I + b  -> variance slope per I/255 is a*255
a, b = 0.4, 4.0
yy, xx = np.mgrid[0:512, 0:512]
I = 30 + 190 * (0.5 * xx / 511 + 0.5 * yy / 511)                  # linear ramp, high-pass residual of a ramp is exactly 0
noisy = I + rng.normal(0, 1, I.shape) * np.sqrt(a * I + b)
fit = run.nlf_fit(noisy)
check("S3 variance slope (a*255 = 102)", fit["s3_slope_var"], 102 * 0.85, 102 * 1.15)
check("S3 variance intercept (b = 4)", fit["s3_icpt_var"], -2.0, 12.0)
# noise-free image has ~0 noise; heavier noise scales the slope
fit2 = run.nlf_fit(I + rng.normal(0, 1, I.shape) * np.sqrt(2 * a * I + b))
check("S3 slope doubles when a doubles", fit2["s3_slope_var"] / fit["s3_slope_var"], 1.8, 2.2)

# 5. full battery runs and returns 38 finite numbers on a natural-like image
d = run.image_stats(np.clip(np.stack([power_law_image(2.0) for _ in range(3)], -1), 0, 255))
check("38 finite statistics", float(len(d) == 38 and all(np.isfinite(list(d.values())))), 1, 1)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
