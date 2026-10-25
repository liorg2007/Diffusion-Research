"""Critic check (H2-v2): per-device false positives at pooled@1% (LODO x LOPO, real platforms + jpeg75_resize05) for the
005 official orig arms A0/A1/A2, LOGO models averaged as in h2v2 (count = mean over logo_<g> models).
Usage: env/.venv/bin/python -I c3_per_device_Aarms.py"""
import math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
PL = {"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}
def thr(s, a):
    n = len(s); k = math.ceil((n + 1) * (1 - a) - 1e-12); return s[-1] if k > n else s[k - 1]
L = pd.read_csv(ROOT / "experiments/005-nuisance-projection/official/orig/scores_long.csv.gz", dtype={"sha256": str})
spl = pd.read_csv(ROOT / "experiments/003-baselines-eval/splits.csv", keep_default_na=False, dtype={"vidx": str})
T0 = spl[(spl.label == 0) & (spl.split == "test") & spl.pool.isin(PL)].copy(); T0["plat"] = T0.pool.map(PL)
for arm in ("A0", "A1", "A2"):
    acc = {}
    models = sorted(m for m in L[L.arm == arm].model.unique() if m.startswith("logo_"))
    for m in models:
        g = L[(L.arm == arm) & (L.model == m)]
        S = {c: x.set_index("sha256").score for c, x in g.groupby("condition")}
        T = T0.copy(); T["s"] = S["clean"].loc[T.sha256].to_numpy()
        for d in sorted(T.device.unique()):
            for P in ["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]:
                if P in PL.values():
                    te = T[(T.device == d) & (T.plat == P)].s.to_numpy(); cal = T[(T.device != d) & (T.plat != P)]
                else:
                    if P not in S: continue
                    te = S[P].loc[T[(T.device == d) & (T.plat == "native")].sha256].to_numpy(); cal = T[T.device != d]
                acc[d] = acc.get(d, 0) + (te > thr(np.sort(cal.s.to_numpy()), 0.01)).sum() / len(models)
    s = pd.Series(acc).sort_values(ascending=False)
    print(f"{arm}: mean FP count over {len(models)} LOGO models, total {s.sum():.1f}; top devices: "
          + ", ".join(f"{d} {v:.1f}" for d, v in s.head(4).items()))
