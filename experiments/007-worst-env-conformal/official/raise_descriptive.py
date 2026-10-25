"""Amendment 13 item 5: RAISE as a DESCRIPTIVE held-out real group for H2-v2 (not used in any verdict).
Thresholds are fit on the 25 test devices' CLEAN VISION reals (1,074; all devices, no LODO) with the same rules as h2v2.py:
M0 (logit 0), M1 (99th pct of the 129 calib-device reals), pooled@a (split conformal), wg@a (max over calibration groups of the
per-group conformal threshold; groups = platform [primary, min 100] or platform x brand [min 50 sensitivity]). They are applied
to the 259 RAISE reals (test 199 + calib 60; RAISE train reals are not scored) under clean and jpeg75_resize05. Flag iff score > t.
  python -I experiments/007-worst-env-conformal/official/raise_descriptive.py orig|fov512
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/007-worst-env-conformal"))
import analyze as A
import h2v2 as H

arm = sys.argv[1]
D = A.Data(HERE / f"splits_{arm}.csv")
H.add_brand(D)
long = pd.read_csv(HERE / f"scores_long_{arm}.csv.gz", dtype={"sha256": str})
sp = pd.read_csv(HERE / f"splits_{arm}.csv", keep_default_na=False)
rz = sp[(sp.label == 0) & (sp.pool == "raise") & sp.split.isin(["test", "calib"])]
T = D.test.reset_index(drop=True)
G_plat = T.plat.to_numpy().astype(object)
lab = H.derive_brand_labels(T, 50)
G_pb = np.array([f"{p}/{lab[(p, b)]}" for p, b in zip(T.plat, T.brand)], dtype=object)
ALPHAS = (0.005, 0.01, 0.02, 0.05)
out = {}
for det, g in long.groupby("arm"):
    S = {c: gc.drop_duplicates("sha256").set_index("sha256").score for c, gc in g.groupby("condition")}
    cl = S["clean"].loc[T.sha256].to_numpy(float)
    th = {"M0": 0.0, "M1": float(np.quantile(S["clean"].loc[D.calib.sha256].to_numpy(float), 0.99, method="higher"))}
    for a in ALPHAS:
        th[f"pooled@{a:g}"] = H.conformal_threshold(np.sort(cl), a)[0]
        for nm, G in (("wg_platform", G_plat), ("wg_platform_brand50", G_pb)):
            th[f"{nm}@{a:g}"] = max(H.conformal_threshold(np.sort(cl[G == q]), a)[0] for q in np.unique(G))
    res = {}
    for cond in ("clean", "jpeg75_resize05"):
        s = S[cond].loc[rz.sha256].to_numpy(float)
        res[cond] = {}
        for k, t in th.items():
            r = A.rate(s > t)
            res[cond][k] = dict(thr=float(t), fpr=r["rate"], k=r["k"], n=r["n"], cp=[r["cp_lo"], r["cp_hi"]])
    out[det] = res
(HERE / f"raise_descriptive_{arm}.json").write_text(json.dumps(out, indent=1))
L = [f"# RAISE descriptive group, arm {arm} (n=259 RAISE reals; thresholds fit on VISION clean reals; NOT in any verdict)", "",
     "| detector | cond | M0 | M1 | pooled@1% | wg_platform@1% | wg_platform_brand50@1% | pooled@5% | wg_platform@5% |", "|---|---|---|---|---|---|---|---|---|"]
for det, res in out.items():
    for cond, r in res.items():
        f = lambda k: f"{100 * r[k]['fpr']:.1f}% [{100 * r[k]['cp'][0]:.1f},{100 * r[k]['cp'][1]:.1f}]"
        L.append(f"| {det} | {cond} | " + " | ".join(f(k) for k in ("M0", "M1", "pooled@0.01", "wg_platform@0.01",
                                                                        "wg_platform_brand50@0.01", "pooled@0.05", "wg_platform@0.05")) + " |")
(HERE / f"raise_descriptive_{arm}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
