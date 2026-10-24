"""Does du2m (natives down/up by k=2) remove the class-differential bandwidth cue? Same stats as c2 on in-memory du2m natives."""
import sys, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_blur as B   # stats(), CONDS, paths
import c4_du2m as C4  # build()
import perturb
from sklearn.metrics import roc_auc_score
def one(a):
    src, sha = a; im = C4.build(src, 2.0); out = []
    for c in B.CONDS: out += list(B.stats(perturb.apply_condition(im, c, key=sha, seed=0)))
    return out
if __name__ == "__main__":
    D = pd.read_csv(HERE / "c2_blur.csv.gz")
    df = pd.read_csv(B.ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False)
    rows = df[(df.split == "test") & (df.pool == "vision_native")]
    with mp.Pool(6) as p: r = p.map(one, list(zip(rows.src_path, rows.sha256)), chunksize=8)
    for ci, c in enumerate(B.CONDS):
        M = pd.DataFrame([v[3 * ci:3 * ci + 3] for v in r], columns=["lap", "hf25", "hf125"])
        fk = D[(D.condition == c) & (D.arm == "du2") & (D.label == 1)]
        nd = D[(D.condition == c) & (D.arm == "du2") & (D.grp == "native")]
        for st in ("lap", "hf25", "hf125"):
            y = np.r_[np.zeros(len(M)), np.ones(len(fk))]
            print(f"{c:16s} {st:5s} median natives du2 {nd[st].median():.4f} du2m {M[st].median():.4f} fakes du2 {fk[st].median():.4f} | "
                  f"AUROC(fakes du2 vs natives du2) {roc_auc_score(np.r_[np.zeros(len(nd)), np.ones(len(fk))], np.r_[nd[st], fk[st]]):.3f} -> vs natives du2m {roc_auc_score(y, np.r_[M[st], fk[st]]):.3f}")
