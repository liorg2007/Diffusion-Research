"""Critic scratch: CPU-vs-CPU reproducibility. The 18 guard images that are also in the capped set were extracted twice on CPU
(guard run vs main run). Max |dlogit| of the primary clean LR between the two CPU feature sets, and prediction flips."""
import sys, importlib.util, numpy as np
from pathlib import Path
HERE = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("r024", HERE / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
main, sec, _ = R.load_tables()
mdf = main[R.cap_subset(main, 120, 0)].reset_index(drop=True)
dm = {v: "cpu" for v in R.VAES}
X = {c: R.feats_matrix("main", mdf, c, dm) for c in R.CONDS}
y, fam = mdf.y.to_numpy(), mdf.fam.to_numpy(); tr = R.split_families(fam, 0.7, 0)
g32 = R.pick_guard_images(main); g = g32[g32.sha256.isin(set(mdf.sha256))]
for c in R.CONDS:
    for src in (["clean"] if c == "clean" else ["clean", "matched"]):
        m = R.fit_lr(X["clean" if src == "clean" else c][tr], y[tr], 1.0)
        A = R.feats_matrix("g32", g, c, dm); B = R.feats_matrix("main", g, c, dm)
        print(c, src, "n", len(g), "max|dlogit| cpu-run vs cpu-run", round(R.check_d008_logits(m, A, B), 4),
              "pred flips", int((m.predict(A) != m.predict(B)).sum()))
