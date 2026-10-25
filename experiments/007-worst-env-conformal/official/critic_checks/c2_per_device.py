"""Critic check (H2-v2): which held-out devices carry the B0 false positives at pooled@1% (LODO x LOPO), orig and fov512.
Usage: env/.venv/bin/python -I c2_per_device.py"""
import math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]; OFF = ROOT / "experiments/007-worst-env-conformal/official"
PL = {"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}
def thr(s, a):
    n = len(s); k = math.ceil((n + 1) * (1 - a) - 1e-12); return s[-1] if k > n else s[k - 1]
for arm in ("orig", "fov512"):
    L = pd.read_csv(OFF / f"scores_long_{arm}.csv.gz", dtype={"sha256": str}); b = L[L.arm == "B0"]
    S = {c: g.set_index("sha256").score for c, g in b.groupby("condition")}
    spl = pd.read_csv(OFF / f"splits_{arm}.csv", keep_default_na=False, dtype={"vidx": str})
    T = spl[(spl.label == 0) & (spl.split == "test") & spl.pool.isin(PL)].copy(); T["plat"] = T.pool.map(PL)
    T["s"] = S["clean"].loc[T.sha256].to_numpy()
    rows = []
    for d in sorted(T.device.unique()):
        r = {"device": d, "camera": T[T.device == d].src_path.iloc[0].split("/")[3] if "/" in T[T.device == d].src_path.iloc[0] else d}
        for P in ["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]:
            if P in PL.values():
                te = T[(T.device == d) & (T.plat == P)].s.to_numpy(); cal = T[(T.device != d) & (T.plat != P)]
            else:
                te = S[P].loc[T[(T.device == d) & (T.plat == "native")].sha256].to_numpy(); cal = T[T.device != d]
            r[P] = int((te > thr(np.sort(cal.s.to_numpy()), 0.01)).sum())
        r["n_per_plat"] = int(((T.device == d) & (T.plat == "native")).sum())
        rows.append(r)
    R = pd.DataFrame(rows); R["total"] = R[["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]].sum(1)
    R = R.sort_values("total", ascending=False)
    print(f"== B0 {arm}: false positives per held-out device at pooled@1% (of {int(R.total.sum())} total)")
    print(R[R.total > 0].to_string(index=False))
    top = R.total.iloc[:2].sum(); print(f"   top-2 devices carry {top}/{int(R.total.sum())}; devices with any FP: {(R.total>0).sum()}/25")
