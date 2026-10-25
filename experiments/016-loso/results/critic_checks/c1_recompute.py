"""Critic check c1: independent recomputation of the E7 headline from scores_long.csv.gz + split/role facts.
Does not import run.py's metric code (plain sklearn AUROC, own threshold rule = run.py's documented rule)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parents[2]
ROOT = H.parents[1]
s = pd.read_csv(H / "results/scores_long.csv.gz")
cfg = json.loads((H / "config.json").read_text())
out = {}

def thr1(r):  # same rule as metrics.threshold_for_fpr(0.01)
    r = np.sort(r)[::-1]; k = int(np.floor(0.01 * len(r)))
    return np.nextafter(r[k], np.inf)

EVAL = {"VISION": ["vision_native", "vision_whatsapp", "vision_facebook_hq"], "RAISE": ["raise"], "OI": ["openimages"], "D3": ["d3_real"]}
for arm in ["fov512", "orig"]:
    a = s[s.arm == arm]
    for S in ["VISION", "RAISE", "OI"]:
        for feat in ["F_b8", "F_last"]:
            d = a[(a.heldout == S) & (a.feat == feat)]
            others = [x for x in ["VISION", "RAISE", "OI"] if x != S]
            calp = sum((EVAL[o] for o in others), [])
            cl = d[d.cond == "clean"]
            cal = cl[(cl.role == "calib") & cl.pool.isin(calp) & (cl.label == 0)].score.values
            # sanity: no held-out pool in calib
            assert not cl[(cl.role == "calib") & cl.pool.isin(calp)].pool.isin(EVAL[S]).any()
            tA = thr1(cal)
            j = d[d.cond == "jpeg75_resize05"]
            calB = j[(j.role == "calib") & j.pool.isin(calp) & (j.label == 0)].score.values
            tB = thr1(calB)
            R = j[(j.role == "test") & j.pool.isin(EVAL[S]) & (j.label == 0)].score.values
            aucs = []
            for g in sorted(j[j.label == 1].generator.unique()):
                F = j[(j.role == "test") & (j.label == 1) & (j.generator == g)].score.values
                aucs.append(roc_auc_score(np.r_[np.zeros(len(R)), np.ones(len(F))], np.r_[R, F]))
            key = f"{arm}/{S}/{feat}"
            out[key] = dict(n_calib=len(cal), macro_auroc=float(np.mean(aucs)), fprA=float((R >= tA).mean()), fprB=float((R >= tB).mean()),
                            kA=int((R >= tA).sum()), n=len(R), thrA=float(tA))
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[key].items()})

# OI split facts
sp = pd.read_csv(H / "splits_fov512.csv", keep_default_na=False)
oi = sp[sp.pool == "openimages"]
print("OI split", oi.split.value_counts().to_dict(), "unique sha", oi.sha256.is_unique)
m = pd.read_csv(H / "openimages_manifest.csv")
print("manifest dup sha:", m.sha256.duplicated().sum(), "dup author top:", m.author.value_counts().head(5).to_dict())
# author clustering across OI train/test (same Flickr author in train and test)
mm = m.merge(oi[["sha256", "split"]], on="sha256")
auth = mm.groupby("author").split.apply(set)
shared = auth[auth.apply(lambda x: "train" in x and "test" in x)]
print("authors with images in both OI train and test:", len(shared), "images in test from such authors:",
      int(mm[mm.author.isin(shared.index) & (mm.split == "test")].shape[0]))
out["oi_split"] = oi.split.value_counts().to_dict()
out["oi_authors_shared_train_test"] = int(len(shared))
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
