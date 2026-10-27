"""Critic scratch (H-A3): same folds / probe / device-cluster bootstrap as run.py on the critic features."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import run as R

T = R.build_table()
z = np.load(HERE / "cache" / "critic_feats.npz")
assert z["sha"].tolist() == T.sha256.tolist()
y, dev = T.platform.to_numpy(), T.device.to_numpy()
folds = R.make_folds(T)
dc, dl = R.dev_codes(dev)
W = R.boot_weights(len(dl))
C = R.CONDS
out = {}

wh = z["wh"].astype(float)
D = np.column_stack([np.log2(wh[:, 0]), np.log2(wh[:, 1]), wh[:, 0] / wh[:, 1], np.log2(wh.max(1))])
pred = np.full(len(y), -1)
for tr, te in folds:
    pred[te] = DecisionTreeClassifier(max_depth=6, random_state=0).fit(D[tr], y[tr]).predict(D[te])
out["dims_decoded_clean"] = R.boot_bacc(y, pred, dc, W)
for p in range(3):
    m = wh[y == p].max(1)
    out[f"long_side_{R.CLASSES[p]}"] = [int(m.min()), int(np.median(m)), int(m.max())]

sets = {"M1_native": {c: {"x": z["m1n"][:, i]} for i, c in enumerate(C)},
        "M1_common1024": {c: {"x": z["m1c"][:, i]} for i, c in enumerate(C)},
        "A_common1024": {c: {"x": z["ac"][:, i]} for i, c in enumerate(C)}}
for name, X in [(k, v) for k, v in sets.items() if len(sys.argv) < 2 or k in sys.argv[1:]]:
    for mode, tc in R.CFG["train_modes"].items():
        key = f"{name}_{mode}"
        R._G[key] = dict(X=X, y=y, g=dev, folds=folds, train_conds=tc, test_conds=C, blocks=["x"])
        t0 = time.time()
        pr, info = R.run_cv(key, workers=4)
        out[key] = {c: R.boot_bacc(y, pr[c], dc, W) for c in C}
        out[key]["confusion"] = {c: R.confusion(y, pr[c]) for c in C}
        out[key]["C"] = [v["C"] for v in info.values()]
        print(key, f"{time.time()-t0:.0f}s", {c: round(out[key][c][0], 3) for c in C}, flush=True)
json.dump(out, open(HERE / "probe_results.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
