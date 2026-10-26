"""c1 (post-hoc): is the k=2 -> k=5 FPR gain a probe effect or a threshold-set effect? And separability at an operating point.
For each held-out S, each training subset T (k=2 and k=5), jpeg75_resize05, F_b8, LOGO-consistent per generator:
  FPR_own   = reading A FPR ('full' model, threshold 1% on T's clean calib reals)  [reproduces run.py]
  FPR_wide  = same probe, threshold from the clean calib reals of ALL 5 non-held-out sources (no held-out data used)
  TPR_own / TPR_wide = macro over 8 gens of TPR at the matching threshold (model_of(g) scores and model_of(g) threshold)
  TPR@5%oracle = macro TPR at the threshold giving 5% FPR on the held-out TEST reals (oracle; separability only)
  dprime = (median fake - median real) / IQR-scale of reals, full model (descriptive)."""
import json, itertools
import numpy as np
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import R, ORDER, ALLGENS, JR, load
D = R.Data(load_feats=False)
ci, cc = D.conds.index(JR), D.conds.index("clean")
Ff = {g: D.idx("test", None, gens=[g], label=1) for g in ALLGENS}
out = {}
for S in ORDER:
    others = [s for s in ORDER if s != S]
    rr = R.src_eval_rows(D, S)
    calW = R.calib_rows(D, others)
    res = {}
    for k in (1, 2, 5):
        rows = []
        for T in itertools.combinations(others, k):
            sc, *_ = load(T)
            cal = R.calib_rows(D, T)
            r = {}
            for nm, cl in (("own", cal), ("wide", calW)):
                thr_full = R.thr_fpr(sc["full"][cc][cal if nm == "own" else calW])
                r[f"fpr_{nm}"] = float((sc["full"][ci][rr] >= thr_full).mean())
                tp = []
                for g in ALLGENS:
                    m = sc[R.model_of(g)]
                    t = R.thr_fpr(m[cc][cl])
                    tp.append((m[ci][Ff[g]] >= t).mean())
                r[f"tpr_{nm}"] = float(np.mean(tp))
            tp5, tp1 = [], []
            for g in ALLGENS:
                m = sc[R.model_of(g)][ci]
                tp5.append((m[Ff[g]] >= R.metrics.threshold_for_fpr(m[rr], 0.05)).mean())
                tp1.append((m[Ff[g]] >= R.metrics.threshold_for_fpr(m[rr], 0.01)).mean())
            r["tpr_at5_oracle"], r["tpr_at1_oracle"] = float(np.mean(tp5)), float(np.mean(tp1))
            f = sc["full"][ci]
            r["real_med"], r["fake_med"] = float(np.median(f[rr])), float(np.median(f[np.concatenate(list(Ff.values()))]))
            r["thr_own_full"] = float(R.thr_fpr(f[cal]) if False else R.thr_fpr(sc["full"][cc][cal]))
            rows.append(r)
        res[k] = {key: float(np.mean([x[key] for x in rows])) for key in rows[0]}
    out[S] = res
mean = {k: {key: float(np.mean([out[S][k][key] for S in ORDER])) for key in out[ORDER[0]][k]} for k in (1, 2, 5)}
out["MEAN"] = mean
json.dump(out, open(R.HERE / "critic" / "c1_thresholds.json", "w"), indent=1)
keys = ["fpr_own", "fpr_wide", "tpr_own", "tpr_wide", "tpr_at5_oracle", "tpr_at1_oracle"]
print("S      k  " + "  ".join(f"{x:>14s}" for x in keys))
for S in ORDER + ["MEAN"]:
    for k in (1, 2, 5):
        print(f"{S:6s} {k}  " + "  ".join(f"{out[S][k][x]:14.3f}" for x in keys))
