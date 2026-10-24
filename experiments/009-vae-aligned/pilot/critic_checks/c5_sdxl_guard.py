"""Q2: SDXL downstream guard. From the cached guard2 GPU/CPU round-trips (32 images), B0 logits (004 CPU DINOv2 features,
same device for both sets) under clean, resize(0.5), jpeg(75), jpeg75_resize05; |dlogit| vs the spread of B0 logits.
Output: c5_sdxl_guard.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch; torch.set_num_threads(8)
from PIL import Image
from perturb import apply_condition
import features
z = np.load(R.CACHE / "guard2_sdxl-vae.npz")
df = R.load_df("orig")
b0 = R.p4run.b0_probe(df, P.cache_root("pilot"))
bb = R.p4run.secondary_backbone()
out = dict(uint8_absdiff_hist_x1={int(k): int(v) for k, v in zip(*np.unique(np.abs(z["q1g"].astype(int) - z["q1c"].astype(int)), return_counts=True))})
X = P.load_features_aligned(df, "primary", "clean", subdir="pilot")
te = (df.split == "test").to_numpy()
lg_all = b0.decision_function(X[te])
out["B0_logit_test_std"] = float(lg_all.std()); out["B0_logit_test_iqr"] = float(np.subtract(*np.percentile(lg_all, [75, 25])))
for c in ("clean", "resize(0.5)", "jpeg(75)", "jpeg75_resize05"):
    F = [features.embed_images(bb, [apply_condition(Image.fromarray(a), c, key="guard2", seed=R.SEED) for a in z[k]], 8, "cpu")[:, -1536:]
         for k in ("q1g", "q1c")]
    lg = [b0.decision_function(f) for f in F]
    d = np.abs(lg[0] - lg[1])
    out[c] = dict(max_abs_dlogit=float(d.max()), median_abs_dlogit=float(np.median(d)), logit_std_32=float(lg[1].std()),
                  max_over_std=float(d.max() / lg_all.std()))
(OUT / "c5_sdxl_guard.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
