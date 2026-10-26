"""c2 (post-hoc): TPR at MATCHED held-out FPR, k=2 vs k=5 (jpeg75_resize05, F_b8, LOGO-consistent).
For each S and generator g (model m = model_of(g)): the k=5 model at its reading-A threshold gives held-out FPR f5_g and TPR t5_g.
Each k=2 model is moved (oracle) to the threshold with the same held-out FPR f5_g; its TPR t2_g is read off. Gain = t5 - mean_k2(t2).
Then 'operating-point move' = TPR_own(k=2) - t2 at k=5's FPR; 'ROC gain' = t5 - t2."""
import sys, itertools, json
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import R, ORDER, ALLGENS, JR, load
D = R.Data(load_feats=False)
ci, cc = D.conds.index(JR), D.conds.index("clean")
Ff = {g: D.idx("test", None, gens=[g], label=1) for g in ALLGENS}
def at_fpr(sr, sf, f):
    t = np.quantile(sr, 1 - f, method="higher") if f > 0 else np.max(sr) + 1e-9
    return float((sf >= t).mean()), float((sr >= t).mean())
out = {}
for S in ORDER:
    others = [s for s in ORDER if s != S]
    rr = R.src_eval_rows(D, S)
    sc5, *_ = load(R.canon(others))
    cal5 = R.calib_rows(D, others)
    t5, t2m, f5s = [], [], []
    for g in ALLGENS:
        m = R.model_of(g)
        thr = R.thr_fpr(sc5[m][cc][cal5])
        f5 = float((sc5[m][ci][rr] >= thr).mean()); f5s.append(f5)
        t5.append(float((sc5[m][ci][Ff[g]] >= thr).mean()))
        t2 = []
        for T in itertools.combinations(others, 2):
            sc2, *_ = load(T)
            t2.append(at_fpr(sc2[m][ci][rr], sc2[m][ci][Ff[g]], f5)[0])
        t2m.append(np.mean(t2))
    out[S] = dict(fpr_k5_macro=float(np.mean(f5s)), tpr_k5=float(np.mean(t5)), tpr_k2_at_matched_fpr=float(np.mean(t2m)))
    out[S]["roc_gain"] = out[S]["tpr_k5"] - out[S]["tpr_k2_at_matched_fpr"]
c1 = json.load(open(Path(__file__).resolve().parent / "c1_thresholds.json"))
for S in ORDER:
    out[S]["tpr_k2_own_thr"] = c1[S]["2"]["tpr_own"]; out[S]["fpr_k2_own_thr"] = c1[S]["2"]["fpr_own"]
out["MEAN"] = {k: float(np.mean([out[S][k] for S in ORDER])) for k in out[ORDER[0]]}
json.dump(out, open(Path(__file__).resolve().parent / "c2_matched_fpr.json", "w"), indent=1)
print(f"{'S':6s} {'FPRk2own':>9s} {'TPRk2own':>9s} {'FPRk5':>7s} {'TPRk5':>7s} {'TPRk2@FPRk5':>12s} {'ROCgain':>8s}")
for S in ORDER + ["MEAN"]:
    o = out[S]
    print(f"{S:6s} {o['fpr_k2_own_thr']:9.3f} {o['tpr_k2_own_thr']:9.3f} {o['fpr_k5_macro']:7.3f} {o['tpr_k5']:7.3f} {o['tpr_k2_at_matched_fpr']:12.3f} {o['roc_gain']:+8.3f}")
