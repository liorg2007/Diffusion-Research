import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False, dtype={"vidx": str})
s = pd.read_csv("experiments/005-nuisance-projection/pilot/scores_long.csv.gz")
gens = sorted(df[df.label == 1].generator.unique())
meta = df.set_index("sha256")
out = []
for arm in ["A0", "A1", "A2"]:
    for cond in ["clean", "jpeg75_resize05"]:
        row = {"arm": arm, "cond": cond}
        for pool in ["vision_native", "vision_whatsapp", "vision_facebook_hq"]:
            reals = set(df[(df.pool == pool) & (df.split == "test")].sha256)
            aucs = []
            for g in gens:
                fk = set(df[df.generator == g].sha256)
                x = s[(s.arm == arm) & (s.condition == cond) & (s.model == f"logo_{g}")]
                x = x[x.sha256.isin(reals | fk)]
                y = x.sha256.isin(fk).astype(int)
                aucs.append(roc_auc_score(y, x.score))
            row[pool] = round(float(np.mean(aucs)), 3)
        out.append(row)
print(pd.DataFrame(out).to_string(index=False))
