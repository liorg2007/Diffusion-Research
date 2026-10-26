"""c5 (post-hoc): (a) near-duplicate check across sources (cosine of block-12 CLS, clean); (b) is a held-out real's false-alarm score tied to its
SEMANTIC closeness to the fakes? For each held-out test real at k=5: delta = mean cos to its 10 nearest TRAIN fakes - mean cos to its 10 nearest
TRAIN reals of the 5 training sources (block-12 CLS = semantic; the probe uses block 8). Spearman(delta, score) and FPR by delta tertile."""
import sys, json
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import R, ORDER
D = R.Data(load_feats=False)
X = R.load_x(D.df, "clean")[:, 3072:3840]          # block 12 CLS
X = X / np.linalg.norm(X, axis=1, keepdims=True)
sl = pd.read_csv(R.HERE / "results" / "scores_long.csv.gz")
sl = sl[(sl.cond == "jpeg75_resize05") & (sl.role == "test") & (sl.label == 0)]
res = json.load(open(R.HERE / "results" / "results.json"))["result"]["heldout"]
pos = {s: i for i, s in enumerate(D.df.sha256)}
out = {"dups": {}, "sem": {}}
# (a) near-duplicates: every real row vs every real row of a DIFFERENT source, and test reals vs train fakes
src_of = np.array([next((S for S in ORDER if p in R.SRC[S]["eval_pools"] + R.SRC[S]["train_pools"]), "FAKE") for p in D.pool])
reals = np.where(D.lab == 0)[0]
fk_tr = D.idx("train", None, label=1)
for S in ORDER:
    a = np.where((src_of == S) & (D.role == "test"))[0]
    b = reals[(src_of[reals] != S) & (D.role[reals] != "test")]
    c = (X[a] @ X[b].T).max(1)
    cf = (X[a] @ X[fk_tr].T).max(1)
    out["dups"][S] = dict(n=len(a), n_cos_gt_0_95_other_src=int((c > 0.95).sum()), max_other_src=float(c.max()), n_cos_gt_0_95_trainfake=int((cf > 0.95).sum()), max_trainfake=float(cf.max()))
    print("dup", S, out["dups"][S], flush=True)
for S in ORDER:
    others = [s for s in ORDER if s != S]
    T = R.canon(others)
    d = sl[(sl.heldout == S) & sl.pool.isin(R.SRC[S]["eval_pools"])].copy()
    thr = res[S]["+".join(T)]["F_b8"]["conds"]["jpeg75_resize05"]["fpr"]["A"]["full"]["thr"]
    idx = np.array([pos[s] for s in d.sha256])
    tr_real = D.idx("train", [p for s in T for p in R.SRC[s]["train_pools"]], label=0)
    sf = np.sort(X[idx] @ X[fk_tr].T, 1)[:, -10:].mean(1)
    sr = np.sort(X[idx] @ X[tr_real].T, 1)[:, -10:].mean(1)
    d["delta"] = sf - sr
    d["fp"] = d.score >= thr
    rho = spearmanr(d.delta, d.score).statistic
    t = pd.qcut(d.delta, 3, labels=["lo", "mid", "hi"])
    fpr_t = d.groupby(t, observed=True).fp.mean().round(3).to_dict()
    out["sem"][S] = dict(rho_delta_score=float(rho), fpr_by_delta_tertile=fpr_t, median_delta=float(d.delta.median()))
    print(f"{S:6s} rho(delta,score)={rho:+.2f}  FPR by delta tertile {fpr_t}  median delta {d.delta.median():+.3f}", flush=True)
json.dump(out, open(Path(__file__).resolve().parent / "c5_semantic.json", "w"), indent=1)
