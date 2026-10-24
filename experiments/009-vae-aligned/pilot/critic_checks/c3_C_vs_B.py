"""Q4: why does adding aligned fakes hurt B? Per-generator C-B from cached scores; refit C with B's chosen C (no CV),
and C with the aligned fakes down-weighted (total weight = 1/3 of the generator fakes). orig arm. Output: c3_C_vs_B.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
out = {}
for arm in ("orig", "fov512"):
    df, Xr, Xrt, tr, grp = arm_data(arm)
    lab, sp, pool, gen = (df[c].to_numpy() for c in ("label", "split", "pool", "generator"))
    b0 = df.b0_train.to_numpy()
    S = dict(np.load(R.CACHE / f"scores6_{arm}.npz"))
    nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
    o = {}
    for g in R.GENS:
        keep = np.where(b0 & (gen != g))[0]
        fk = np.where(gen == g)[0]
        row = {}
        for ci, c in enumerate(R.CONDS):
            for nm in ("B", "C", "C_all_reals"):
                s = S[f"{nm}__{g}"][ci]; row[f"{nm}_{c}"] = R.auc(s[nat], s[fk])
        Xk, yk = Xr["clean"][keep], lab[keep]
        n_gen_fake = int((yk == 1).sum())
        for tag, Cv, wal in (("C_fixC10", 10.0, 1.0), ("C_fixC0.01", 0.01, 1.0), ("C_fixC10_w", 10.0, n_gen_fake / 3 / len(Xrt["clean"])),
                             ("B_fixC0.01", 0.01, None)):
            if wal is None:
                X, y, w = Xk, yk, np.ones(len(yk))
            else:
                X = np.vstack([Xk, Xrt["clean"]]); y = np.r_[yk, np.ones(len(Xrt["clean"]), int)]; w = np.r_[np.ones(len(yk)), np.full(len(Xrt["clean"]), wal)]
            # balanced class weights computed on the sample weights
            cw = {k: w.sum() / (2 * w[y == k].sum()) for k in (0, 1)}
            m = make_pipeline(StandardScaler(), LogisticRegression(C=Cv, max_iter=3000))
            m.fit(X, y, logisticregression__sample_weight=w * np.array([cw[k] for k in y]))
            for c in R.CONDS:
                s = m.decision_function(Xr[c]); row[f"{tag}_{c}"] = R.auc(s[nat], s[fk])
        o[g] = row
    mac = {k: float(np.mean([o[g][k] for g in R.GENS])) for k in o[R.GENS[0]]}
    out[arm] = dict(per_gen=o, macro=mac)
    print(arm, json.dumps(mac, indent=0))
(OUT / "c3_C_vs_B.json").write_text(json.dumps(out, indent=1))
