"""Critic scratch (descriptive, NOT pre-registered): oracle-C upper bound per embedding at jpeg75_resize05, T-mix, outer folds of 023.
Tests whether the 020 DINOv2 C grid (max 0.1) under-fits the low-dim detector embeddings. Oracle = best C chosen ON TEST -> optimistic bound."""
import os, sys, json, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
import importlib.util
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r023", HERE.parent / "run.py")
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
m, A = r.build_analysis()
tm, prim = list(A.tmix), A.primary
want = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"commfor": [0.1, 1, 10], "bfree": [0.1, 1, 10], "clipdet": [0.1, 1, 10], "cat": [0.1, 1]}
out = {}
for fs, Cs in want.items():
    getX = (lambda c, fs=fs: np.concatenate([A.getX(f, c) for f in fs.split("+")], 1))
    res = {}
    for C in Cs:
        pred = np.full(A.n, -1)
        for f in range(5):
            tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
            X = np.concatenate([getX(c)[tr] for c in tm]); y = np.tile(A.y[tr], len(tm))
            mod = m.fit_model("lr", X, y, C)
            pred[te] = m.logp_full(mod, getX(prim)[te], A.K).argmax(1)
        res[C] = round(m.bacc_simple(A.y, pred, list(range(A.K))), 4)
        print(fs, C, res[C], flush=True)
    out[fs] = res
if "rine" in want or True:
    Z = A.getX("rine", "clean")
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    out["rine_diag"] = dict(dims_always_zero=int((np.abs(Z).max(0) == 0).sum()), rank95=int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1))
print(json.dumps(out))
json.dump(out, open(HERE / ("csweep_" + "_".join(want) + ".json"), "w"), indent=1)
