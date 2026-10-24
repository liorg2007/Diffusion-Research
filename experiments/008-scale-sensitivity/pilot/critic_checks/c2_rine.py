"""Critic check 2: RINE laundered > clean? spread of scores, DALL-E 3 effect, macro without dalle3, ties."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "experiments/008-scale-sensitivity/pilot/"
sz = pd.read_csv(R + "critic_checks/sizes.csv")
d = pd.read_csv(R + "scores_long.csv.gz"); d = d[d.model == "pretrained"].merge(sz, on="sha256")
for det in ["rine", "communityforensics", "clipdet"]:
    print("=====", det)
    for c in ["clean", "jpeg75_resize05"]:
        for a in ["orig", "fov512"]:
            g = d[(d.detector == det) & (d.condition == c) & (d.arm == a)]
            t = g[g.split == "test"]
            nat = t[t.grp == "vision_native"].score.values
            fk = t[~t.grp.str.startswith("vision")]
            q = lambda x: np.percentile(x, [5, 25, 50, 75, 95]).round(3)
            aucs = {gen: roc_auc_score(np.r_[np.zeros(len(nat)), np.ones((fk.grp == gen).sum())], np.r_[nat, fk[fk.grp == gen].score]) for gen in sorted(fk.grp.unique())}
            m_all = np.mean(list(aucs.values())); m_nod3 = np.mean([v for k, v in aucs.items() if k != "dalle3"])
            cal = g[(g.split == "calib") & (g.grp == "vision_native")].score.sort_values(ascending=False).values
            cala = g[(g.split == "calib")].score.sort_values(ascending=False).values
            print(f"{c:16s} {a:6s} macro={m_all:.3f} macro_noDALLE3={m_nod3:.3f} | natives q5..95 {q(nat)} | fakes q {q(fk.score)} | "
                  f"n_unique_test={t.score.nunique()}/{len(t)} | calib nat top3 {cal[:3].round(3)} | calib all top3 {cala[:3].round(3)}")
