"""Critic check Q1: class-differential blur introduced by du2. Per test image (natives, WA, FBH, 6 gens) and arm
(orig/du2/fov512) x cond (clean / jpeg75_resize05 via 007 load_cond): mean |Laplacian| (gray, 512 input) and spectral
high-frequency energy fractions (radial freq > 0.25 and > 0.125 cyc/px, DC removed, Hann window). Then group medians,
AUROC(stat: fakes vs reals) per arm, and the per-image resampling factor k = short/512 (E1 critic sizes.csv)."""
import sys, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
for p in ("experiments/lib", "experiments/002-baselines", "experiments/004-phase4-features", "experiments/007-worst-env-conformal"):
    sys.path.insert(0, str(ROOT / p))
import os; os.chdir(ROOT)
HERE = Path(__file__).resolve().parent
ARMS = {"orig": "splits_pilot.csv", "du2": "splits_pilot_du2.csv", "fov512": "splits_pilot_fov512.csv"}
CONDS = ["clean", "jpeg75_resize05"]
n = 512
fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
RAD = np.sqrt(fx ** 2 + fy ** 2)
WIN = np.outer(np.hanning(n), np.hanning(n))

def stats(img):
    x = np.asarray(img.convert("L"), np.float64) / 255.0
    lap = float(np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]).mean())
    m = x.shape[0]
    if m != n:   # laundered images are 256: own freq grid
        f1 = np.fft.fftfreq(m); rad = np.sqrt(f1[None, :] ** 2 + f1[:, None] ** 2); win = np.outer(np.hanning(m), np.hanning(m))
    else:
        rad, win = RAD, WIN
    P = np.abs(np.fft.fft2((x - x.mean()) * win)) ** 2
    P[0, 0] = 0; tot = P.sum() + 1e-30
    return lap, float(P[rad > 0.25].sum() / tot), float(P[rad > 0.125].sum() / tot)

def one(a):
    import score_pretrained as sp
    path, sha = a
    out = []
    for c in CONDS:
        out += list(stats(sp.load_cond(path, sha, c)))
    return out

if __name__ == "__main__":
    base = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot" / ARMS["orig"], keep_default_na=False)
    sel = (base.split == "test") & (base.pool.isin(["vision_native", "vision_whatsapp", "vision_facebook_hq"]) |
                                   base.generator.isin(["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]))
    ix = np.where(sel)[0]
    sz = pd.read_csv(ROOT / "experiments/008-scale-sensitivity/pilot/critic_checks/sizes.csv").set_index("sha256")
    rows = []
    for arm, f in ARMS.items():
        df = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot" / f, keep_default_na=False)
        assert (df.sha256.values == base.sha256.values).all()
        with mp.Pool(6) as pool:
            r = pool.map(one, [(df.path[i], df.sha256[i]) for i in ix], chunksize=8)
        for i, v in zip(ix, r):
            g = base.generator[i] if base.label[i] == 1 else {"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}[base.pool[i]]
            for ci, c in enumerate(CONDS):
                rows.append(dict(sha256=base.sha256[i], grp=g, label=int(base.label[i]), device=base.device[i], arm=arm, condition=c,
                                 lap=v[3 * ci], hf25=v[3 * ci + 1], hf125=v[3 * ci + 2]))
    D = pd.DataFrame(rows)
    D["k"] = D.sha256.map(sz.short) / 512
    D.to_csv(HERE / "c2_blur.csv.gz", index=False)
    print(D.groupby("grp").k.median().round(2).to_string())
    from sklearn.metrics import roc_auc_score
    for c in CONDS:
        print(f"\n=== {c}: group medians (lap / hf>0.25 / hf>0.125) ===")
        T = D[D.condition == c].groupby(["grp", "arm"])[["lap", "hf25", "hf125"]].median().unstack("arm")
        print(T.round(4).to_string())
        print(f"--- {c}: AUROC(stat, fakes=1 vs reals of pool), all 360 test fakes; >0.5 = fakes sharper ---")
        for pool in ("native", "WA", "FBH"):
            for st in ("lap", "hf25", "hf125"):
                s = []
                for arm in ARMS:
                    d = D[(D.condition == c) & (D.arm == arm) & ((D.label == 1) | (D.grp == pool))]
                    s.append(f"{arm} {roc_auc_score(d.label, d[st]):.3f}")
                print(f"{pool:6s} {st:5s} " + " | ".join(s))
        # within natives: does the du2/orig sharpness ratio scale with k?
        from scipy.stats import spearmanr
        o = D[(D.condition == c) & (D.arm == "orig")].set_index("sha256"); d2 = D[(D.condition == c) & (D.arm == "du2")].set_index("sha256")
        rr = (d2.hf25 / o.hf25)
        for g in ("native", "WA", "FBH", "fakes"):
            m = (o.label == 1) if g == "fakes" else (o.grp == g)
            print(f"{c} {g}: median hf25 ratio du2/orig {rr[m].median():.3f}; Spearman(k, ratio) {spearmanr(o.k[m], rr[m]).statistic:+.2f}; median k {o.k[m].median():.2f}")
