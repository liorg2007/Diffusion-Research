"""Positive control for the gate (post hoc, descriptive): do the KNOWN sd-vae-ft-mse round-trips of 011 (fov512 arm, PNG, 150 test reals) show the
1/8 peak under the H-D2 score O, and under 011's own F3 periodicity features? -> pilot/positive_control.json.
  env/.venv/bin/python -I experiments/013-latent-grid/positive_control.py"""
import importlib.util, json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run013", HERE / "run.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
R = M.R011()
df = R.e2.load_df("fov512"); te = df[(df.label == 0) & (df.split == "test")].head(150)
a, b = [], []
for sha, p in zip(te.sha256, te.path):
    for lst, path in ((a, p), (b, R.rt_file("fov512", sha))):
        lst.append(M.image_feats(Image.open(path).convert("RGB"), 1.0, 512, lambda x: 0)[[0, 5]])
a, b = np.array(a), np.array(b)
out = dict(n=len(te), O_median_real=float(np.median(a[:, 0])), O_median_roundtrip=float(np.median(b[:, 0])), O_auroc_roundtrip_vs_real=M.auc(a[:, 0], b[:, 0]),
           share_argmax_at_1_8_real=float(a[:, 1].mean()), share_argmax_at_1_8_roundtrip=float(b[:, 1].mean()))
c = ROOT = HERE.parents[1] / "experiments/011-vae-trace/cache/feats/"
p = np.load(c / "f3_plain__fov512__clean.npz"); r = np.load(c / "f3_rt__fov512__clean.npz")
pk = {k: i for i, k in enumerate(p["keys"])}
A, B = p["feats"][[pk[k] for k in r["keys"]]], r["feats"]
out["f3_peak_pair_auroc_fov512_clean_red_channel"] = {n: float(roc_auc_score(np.r_[np.zeros(len(A)), np.ones(len(B))], np.r_[A[:, j], B[:, j]]))
    for n, j in [("x1/8", 32), ("x2/8", 33), ("x3/8", 34), ("x4/8", 35), ("y1/8", 36), ("y2/8", 37), ("y3/8", 38), ("y4/8", 39)]}
M.jdump(HERE / "pilot" / "positive_control.json", out); print(json.dumps(out, indent=1))
