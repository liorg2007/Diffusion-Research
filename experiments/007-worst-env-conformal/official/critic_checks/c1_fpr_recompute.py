"""Critic check (H2-v2 official): independent re-implementation of the pooled split-conformal LODO x LOPO FPR for B0
(orig + fov512) and A0 (005 orig), Clopper-Pearson bounds, the alpha=0.5% degeneracy of the matched-TPR test, and the
FPR/TPR trade-off needed for worst-platform FPR <= 1%. Read-only; prints to stdout.
Usage: env/.venv/bin/python -I c1_fpr_recompute.py"""
import math, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
ROOT = Path(__file__).resolve().parents[4]
OFF = ROOT / "experiments/007-worst-env-conformal/official"
PL = {"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}

def thr(sorted_s, a):
    n = len(sorted_s); k = math.ceil((n + 1) * (1 - a) - 1e-12)
    return sorted_s[-1] if k > n else sorted_s[k - 1]

def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi

def analyse(tag, S, spl, fakes, a=0.01, wg_check=True):
    T = spl[(spl.label == 0) & (spl.split == "test") & spl.pool.isin(PL)].copy()
    T["plat"] = T.pool.map(PL); T["s"] = S["clean"].loc[T.sha256].to_numpy()
    devs = sorted(T.device.unique()); out = {}; ks = {}; deg = []
    for P in ["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]:
        k = n = 0
        for d in devs:
            if P in ("native", "WA", "FBH"):
                te = T[(T.device == d) & (T.plat == P)].s.to_numpy(); calT = T[(T.device != d) & (T.plat != P)]
            else:
                te = S[P].loc[T[(T.device == d) & (T.plat == "native")].sha256].to_numpy(); calT = T[T.device != d]
            cal = np.sort(calT.s.to_numpy()); t = thr(cal, a)
            k += int((te > t).sum()); n += len(te)
            if wg_check:   # wg@0.5% = max over groups of per-group 0.5% conformal threshold; pooled at smallest grid alpha
                wg = max(thr(np.sort(g.s.to_numpy()), 0.005) for _, g in calT.groupby("plat"))
                deg.append(wg == thr(cal, 0.0005))
        ks[P] = (k, n)
    K = sum(k for k, _ in ks.values()); N = sum(n for _, n in ks.values())
    print(f"[{tag}] pooled@{a:g}: all cells {K}/{N} = {100*K/N:.2f}% CP95 [{100*cp(K,N)[0]:.2f}, {100*cp(K,N)[1]:.2f}]")
    for P, (k, n) in ks.items():
        lo, hi = cp(k, n); print(f"     {P:16s} {k:3d}/{n} = {100*k/n:5.2f}%  CP95 [{lo*100:.2f}, {hi*100:.2f}]")
    if wg_check:
        print(f"     wg@0.5% threshold == pooled@0.05% threshold in {np.mean(deg)*100:.0f}% of {len(deg)} cells (structural tie)")
    return ks

for arm in ("orig", "fov512"):
    L = pd.read_csv(OFF / f"scores_long_{arm}.csv.gz", dtype={"sha256": str})
    spl = pd.read_csv(OFF / f"splits_{arm}.csv", keep_default_na=False, dtype={"vidx": str})
    b = L[L.arm == "B0"]; S = {c: g.set_index("sha256").score for c, g in b.groupby("condition")}
    ks = analyse(f"B0 {arm}", S, spl, None)
    # what pooled alpha gives worst platform <= 1%, and the TPR it costs (clean / jpeg75_resize05, 585 test fakes; non-LOGO)
    T = spl[(spl.label == 0) & (spl.split == "test") & spl.pool.isin(PL)]
    F = spl[spl.label == 1].sha256
    for a in (0.01, 0.005, 0.0025):
        t_all = thr(np.sort(S["clean"].loc[T.sha256].to_numpy()), a)   # deployable: calibrate on all 25 devices
        tpr = {c: float((S[c].loc[F] > t_all).mean()) for c in ("clean", "jpeg75_resize05")}
        flux = spl[(spl.label == 1) & (spl.generator == "flux")].sha256
        print(f"     alpha={a:g}: TPR clean {100*tpr['clean']:.1f}% jpeg75_resize05 {100*tpr['jpeg75_resize05']:.1f}% | FLUX (unseen) clean "
              f"{100*float((S['clean'].loc[flux] > t_all).mean()):.1f}%")
    if arm == "orig":
        for a in (0.005, 0.0025):
            analyse(f"B0 {arm}", S, spl, None, a=a, wg_check=False)

# A0 (005 official orig, LOGO models): pooled@1% FPR per model averaged, as h2v2 does
L5 = pd.read_csv(ROOT / "experiments/005-nuisance-projection/official/orig/scores_long.csv.gz", dtype={"sha256": str})
print("005 scores_long columns:", list(L5.columns), "arms:", sorted(L5.arm.unique())[:12])
