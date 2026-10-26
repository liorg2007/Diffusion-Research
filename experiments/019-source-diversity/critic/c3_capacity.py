"""c3 (post-hoc): is separability capped by the linear probe, or by the domain shift?
(a) IN-DISTRIBUTION: F_b8 linear probe trained on ALL 6 sources' train reals (E10 recipe: T-aug, balanced, grouped CV, LOGO), scored on each
    source's TEST reals (macro AUROC, LOGO convention) -> compare with the k=5 held-out AUROC of the same source. The gap = generalisation loss.
(b) NONLINEAR at k=5 held out: an MLP (sklearn, 1 hidden layer 256, early stopping, seed 0) on the same standardised F_b8 rows, 'full' model only
    (trained on the 7 training generators), vs the E10 linear 'full' model, both scored as full-model macro over the 8 generators
    (lenient convention, identical for both)."""
import sys, json, time
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import R, ORDER, ALLGENS, JR, load, auroc
D = R.Data(load_feats=True)
ci = D.conds.index(JR)
Ff = {g: D.idx("test", None, gens=[g], label=1) for g in ALLGENS}
out = {"in_dist": {}, "mlp": {}}
t0 = time.time()
ALL = tuple(ORDER)
sc6 = {m: R.fit_scores(D, "F_b8", ALL, m)[0] for m in R.model_names()}
print(f"k=6 fits done {time.time()-t0:.0f}s", flush=True)
for S in ORDER:
    rr = R.src_eval_rows(D, S)
    a6 = np.mean([auroc(sc6[R.model_of(g)][ci][rr], sc6[R.model_of(g)][ci][Ff[g]]) for g in ALLGENS])
    sc5, *_ = load(R.canon([s for s in ORDER if s != S]))
    a5 = np.mean([auroc(sc5[R.model_of(g)][ci][rr], sc5[R.model_of(g)][ci][Ff[g]]) for g in ALLGENS])
    out["in_dist"][S] = dict(auroc_in_dist_k6=float(a6), auroc_heldout_k5=float(a5))
    print(S, out["in_dist"][S], flush=True)
cols = R.FEATS["F_b8"]
for S in ORDER:
    T = R.canon([s for s in ORDER if s != S])
    rows = R.train_rows(D, T, "full")
    X = np.concatenate([D.X[c][rows][:, cols] for c in R.TAUG]); y = np.concatenate([D.lab[rows]] * len(R.TAUG))
    # balance by sample duplication of the minority (fakes) to mimic class_weight='balanced'
    w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    rep = np.repeat(np.arange(len(y)), np.round(w).astype(int))
    mlp = make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-2, early_stopping=True, random_state=0, max_iter=300))
    mlp.fit(X[rep], y[rep])
    s_m = mlp.predict_proba(D.X[JR][:, cols])[:, 1]
    sc5, *_ = load(T)
    s_l = sc5["full"][ci]
    rr = R.src_eval_rows(D, S)
    am = np.mean([auroc(s_m[rr], s_m[Ff[g]]) for g in ALLGENS]); al = np.mean([auroc(s_l[rr], s_l[Ff[g]]) for g in ALLGENS])
    out["mlp"][S] = dict(linear_full_macro=float(al), mlp_full_macro=float(am))
    print(S, out["mlp"][S], flush=True)
json.dump(out, open(Path(__file__).resolve().parent / "c3_capacity.json", "w"), indent=1)
