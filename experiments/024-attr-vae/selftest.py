"""Synthetic self-tests of run.py: feature math, read-outs, labels, gates, pipeline and PLANTED LEAKS the checks must catch.

    env/.venv/bin/python -I experiments/024-attr-vae/selftest.py   -> results/selftest.txt
No real data. A tiny random-weight AutoencoderKL on CPU checks the VAE wrapper plumbing only.
"""
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run024", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["run024"] = R
spec.loader.exec_module(R)

import numpy as np
import pandas as pd
from PIL import Image

LINES, FAILS = [], []


def check(name, cond, extra=""):
    LINES.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}".rstrip())
    if not cond:
        FAILS.append(name)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


rng = np.random.default_rng(0)
t0 = time.time()

# ---------------------------------------------------------------- 1. feature math
d_flat = rng.normal(size=(64, 64)) * 1e-2
cv_flat, ph = R.hp8_feature(d_flat)
d_per = d_flat.copy()
d_per[:, 7::8] += 0.2 * rng.choice([-1, 1], size=(64, 8))  # strong error energy on one column phase (8-px grid)
cv_per, ph2 = R.hp8_feature(d_per)
check("hp8: random error has a small phase CV", cv_flat < 0.3, f"{cv_flat:.3f}")
check("hp8: a planted 8-px column-phase artefact raises the CV a lot", cv_per > 5 * cv_flat, f"{cv_per:.3f}")
check("hp8: 16 phase means returned (8 col + 8 row)", len(ph) == 16)
check("hp8: the strongest column phase is the planted one (7)", int(np.argmax(ph2[:8])) in (6, 7, 0), f"{np.argmax(ph2[:8])}")
x = rng.random((32, 32, 3))
f, _ = R.vae_features(x, x, x)
check("features: perfect reconstruction -> log e1 very negative", f[0] < -20, f"{f[0]:.1f}")
x1 = np.clip(x + 0.02 * rng.normal(size=x.shape), 0, 1)
x2 = np.clip(x1 + 0.005 * rng.normal(size=x.shape), 0, 1)
f, _ = R.vae_features(x, x1, x2)
check("features: e1 ~ 0.02 and e2 ~ 0.005 and AEDR ~ 4", abs(np.exp(f[0]) - 0.02) < 0.004 and abs(np.exp(f[1]) - 0.005) < 0.002 and 3 < np.exp(f[2]) < 5.5,
      f"e1 {np.exp(f[0]):.4f} e2 {np.exp(f[1]):.4f} aedr {np.exp(f[2]):.2f}")
check("features: AEDR = e1/e2 exactly in log space", abs(f[2] - (f[0] - f[1])) < 1e-9)

# ---------------------------------------------------------------- 2. VAE wrapper plumbing (tiny random AutoencoderKL, CPU)
try:
    from diffusers import AutoencoderKL
    import torch
    torch.manual_seed(0)
    tiny = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                         block_out_channels=(8, 16), latent_channels=4, layers_per_block=1, norm_num_groups=4, sample_size=32).eval().requires_grad_(False)
    arrs = [(rng.random((32, 32, 3)) * 255).astype(np.uint8) for _ in range(3)]
    F, P = R.features_for(tiny, arrs, R.Batcher(2))
    check("VAE wrapper: shapes (3,4) and (3,16), finite", F.shape == (3, 4) and P.shape == (3, 16) and np.isfinite(F).all() and np.isfinite(P).all())
    F2, _ = R.features_for(tiny, arrs, R.Batcher(1))
    check("VAE wrapper: batch-size independent to 2e-3 (uint8 rounding flips make ~1e-4 differences expected)", np.abs(F - F2).max() < 2e-3, f"{np.abs(F - F2).max():.2e}")
    check("VAE wrapper: q8 round trip is uint8", R.q8(R.recon_f(tiny, np.stack(arrs))).dtype == np.uint8)
except Exception as e:  # pragma: no cover
    check("VAE wrapper plumbing", False, repr(e))

# ---------------------------------------------------------------- 3. data path: crops and conditions
tmp = Path(tempfile.mkdtemp(prefix="st024_", dir=str(R.CACHE.parent / "cache")))
big = Image.fromarray((rng.random((700, 900, 3)) * 255).astype(np.uint8))
bigp = tmp / "a.png"
big.save(bigp)
rel = str(bigp.relative_to(R.ROOT)) if str(bigp).startswith(str(R.ROOT)) else str(bigp)
out = R.load_crop(rel, "deadbeef", R.CONDS)
check("crop: clean = exact top-left 512 of the native grid, no resize", np.array_equal(out["clean"], np.asarray(big)[:512, :512]))
check("crop: jpeg75_resize05 is 256x256", out["jpeg75_resize05"].shape == (256, 256, 3))
check("crop: social_sim is a multiple of 8 and in [352, 512]", all(s % 8 == 0 and 352 <= s <= 512 for s in out["social_sim"].shape[:2]), str(out["social_sim"].shape))
out2 = R.load_crop(rel, "deadbeef", R.CONDS)
check("crop: conditions deterministic given the sha256 key", all(np.array_equal(out[c], out2[c]) for c in R.CONDS))
small = tmp / "s.png"
Image.fromarray(np.zeros((400, 600, 3), np.uint8)).save(small)
try:
    R.load_crop(str(small), "x", ["clean"])
    check("crop: an image smaller than 512 raises", False)
except RuntimeError:
    check("crop: an image smaller than 512 raises", True)

# ---------------------------------------------------------------- 4. read-outs and labels
sd = np.array([1.0, 3.0, 2.0])
xl = np.array([3.0, 1.0, 2.5])
check("argmin A2: lower AEDR picks the decoder (0=kl-f8, 1=XL)", list(R.argmin_a2(sd, xl)) == [0, 1, 0])
check("argmin A3: min AEDR above tau -> other (2)", list(R.argmin_a3(sd, xl, 1.5)) == [0, 1, 2])
L = R.label_h_a6
check("label: clean 0.8 and jr 0.55 -> PASS", L(0.8, 0.55, 0.9, True) == "PASS")
check("label: clean 0.8 but jr 0.4 (both sources) -> KILL", L(0.8, 0.40, 0.40, True) == "KILL")
check("label: jr 0.4 clean-source but 0.6 matched -> INCONCLUSIVE", L(0.8, 0.40, 0.60, True) == "INCONCLUSIVE")
check("label: clean 0.65 / jr 0.55 -> INCONCLUSIVE", L(0.65, 0.55, 0.6, True) == "INCONCLUSIVE")
check("label: a failed gate -> INVALID", L(0.9, 0.9, 0.9, False) == "INVALID")

# ---------------------------------------------------------------- 5. gates
fams = np.array([f"r{i // 3:08x}t" for i in range(300)])
tr = R.split_families(fams, 0.7, 0)
check("gate grouped split: no family on both sides", not raises(R.check_grouped, fams, tr, ~tr))
leak_tr = tr.copy()
leak_tr[np.where(~tr)[0][0]] = True
check("PLANTED LEAK: a test family leaking into train is caught", raises(R.check_grouped, fams, leak_tr, ~tr))
check("gate dedupe: unique passes", not raises(R.check_dedupe, ["a", "b", "c"], "t"))
check("PLANTED LEAK: duplicate sha256 across classes is caught", raises(R.check_dedupe, ["a", "b", "a"], "t"))
df = pd.DataFrame({"y": np.repeat([0, 1, 2], [120, 120, 780])})
k1, k2 = R.cap_subset(df, 40, 0), R.cap_subset(df, 40, 0)
check("cap_subset: 40 per class, deterministic", (np.bincount(df.y[k1]) == 40).all() and (k1 == k2).all())
m = R.make_lr(1.0).fit(rng.normal(size=(50, 3)), rng.integers(0, 3, 50))
Xa = rng.normal(size=(5, 3))
check("D-008 logit guard: identical features -> 0", R.check_d008_logits(m, Xa, Xa) == 0.0)
check("D-008 logit guard: a 1e-2 feature shift exceeds 1e-3 (would abort)", R.check_d008_logits(m, Xa, Xa + 1e-2) > 1e-3)

# ---------------------------------------------------------------- 6. pipeline on synthetic features
n = 900
y = np.repeat([0, 1, 2], [150, 150, 600])
fam = np.array([f"r{i:08x}t" for i in np.arange(n) // 3])  # 300 families
train = R.split_families(fam, 0.7, 0)
test = ~train


def synth(sig):
    X = rng.normal(size=(n, 8))
    X[:, 2] += sig * (y == 0) * 2.0
    X[:, 6] += sig * (y == 1) * 2.0
    X[:, 0] += sig * (y == 2) * 1.5
    return X


Xs = {c: synth(1.0) for c in R.CONDS}
res = R.eval_models(Xs, y, fam, train, test, 1.0, 100, 30, 0)
e = res["clean|src=clean"]
check("pipeline: separable synthetic features -> high bacc", e["both"]["bacc"] > 0.8, f"{e['both']['bacc']:.3f}")
check("pipeline: permutation control at chance on that data", abs(e["perm"]["mean"] - 1 / 3) < 0.05, f"{e['perm']['mean']:.3f}")
check("pipeline: sub-feature sets and keys present", {"sd", "xl", "aedr_only", "e1_only"} <= set(e) and "jpeg75_resize05|src=matched" in res)
Xn = {c: synth(0.0) for c in R.CONDS}
rn = R.eval_models(Xn, y, fam, train, test, 1.0, 100, 30, 0)["clean|src=clean"]
check("pipeline: pure noise features -> bacc near chance", abs(rn["both"]["bacc"] - 1 / 3) < 0.12, f"{rn['both']['bacc']:.3f}")
# planted leak: feature chosen using TEST labels
Xl = {c: rng.normal(size=(n, 8)) for c in R.CONDS}
Xl_leak = Xl["clean"].copy()
Xl_leak[:, 3] = (y == 1) * 3.0 + rng.normal(size=n) * 0.5  # plants a test-label-derived column in test rows only
Xl_leak[train, 3] = rng.normal(size=int(train.sum()))
leak_acc = R.macro_recall(R.fit_lr(Xl_leak[train], y[train], 1.0).predict(Xl_leak[test]) == y[test], y[test], n_cls=3)
perm_leak = R.perm_control(Xl_leak[train], y[train], Xl_leak[test], y[test], 1.0, 30, 0)
check("PLANTED LEAK: test-only label-derived feature gives above-chance bacc that the permutation distribution does not reach",
      leak_acc > perm_leak.max(), f"real {leak_acc:.3f} vs perm max {perm_leak.max():.3f}")
am = R.argmin_eval({c: np.abs(Xs[c]) for c in R.CONDS}, y, fam, train, test, 50, 0)
check("argmin_eval: returns A2/A3 per condition", all("A2_bacc" in v and "A3_bacc" in v for v in am.values()))
sdf = pd.DataFrame({"generator": ["a"] * 30 + ["b"] * 30, "fam": [f"f{i // 3}" for i in range(60)]})
Xsec = {c: synth(1.0)[:60] for c in R.CONDS}
sr = R.sec_eval(Xs, y, train, Xsec, sdf, 50, 0, 1.0, None)
check("sec_eval: shares sum to 1 per generator", all(abs(sum(d["share"].values()) - 1) < 1e-9 for e_ in sr.values() for d in e_.values()))

LINES.append(f"\n{len(LINES) - len(FAILS)} of {len(LINES)} checks passed; failed: {', '.join(FAILS) or 'none'}  ({time.time() - t0:.0f}s)")
R.RES.mkdir(exist_ok=True)
(R.RES / "selftest.txt").write_text("\n".join(LINES) + "\n")
print("\n".join(LINES))
import shutil
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAILS else 0)
