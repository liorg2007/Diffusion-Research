"""Critic scratch: per-generator predictions of the primary clean LR (capped, all-CPU features) + A2 argmin rate per generator; MJ-v5 role."""
import sys, importlib.util, numpy as np
from pathlib import Path
HERE = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("r024", HERE / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
main, sec, _ = R.load_tables()
keep = R.cap_subset(main, 120, 0); mdf = main[keep].reset_index(drop=True)
dm = {"sd-vae-ft-mse": "cpu", "sdxl-vae": "cpu"}
X = {c: R.feats_matrix("main", mdf, c, dm) for c in R.CONDS}
y, fam = mdf.y.to_numpy(), mdf.fam.to_numpy()
tr = R.split_families(fam, 0.7, 0); te = ~tr
print("test per generator:", mdf[te].generator.value_counts().to_dict())
m = R.fit_lr(X["clean"][tr], y[tr], 1.0)
for cond in R.CONDS:
    p = m.predict(X[cond][te]); g = mdf[te].generator.to_numpy()
    print(cond, {gg: np.bincount(p[g == gg], minlength=3).tolist() for gg in R.GENS})
print("coef (rows kl-f8,XL,other; cols sd e1,e2,aedr,hp8 | xl e1,e2,aedr,hp8):\n", np.round(m[-1].coef_, 2))
# drop-MJ sensitivity: bacc without midjourney in train+test
nm = mdf.generator.to_numpy() != "midjourney-v5"
m2 = R.fit_lr(X["clean"][tr & nm], y[tr & nm], 1.0); p2 = m2.predict(X["clean"][te & nm])
print("clean bacc w/o MJ:", round(R.macro_recall(p2 == y[te & nm], y[te & nm], n_cls=3), 3))
# e1-only and aedr-only clean (point) for reference of which cue carries 'other'
for name, cols in (("aedr2", [2, 6]), ("e1_2", [0, 4]), ("hp8_2", [3, 7]), ("e1e2_4", [0, 1, 4, 5])):
    mm = R.fit_lr(X["clean"][tr][:, cols], y[tr], 1.0); pp = mm.predict(X["clean"][te][:, cols])
    print(name, "clean bacc", round(R.macro_recall(pp == y[te], y[te], n_cls=3), 3))
