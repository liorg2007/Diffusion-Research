"""Critic check 1: source sizes per pool/generator; per-class score medians; per-generator AUROC per arm/cond."""
import json, sys
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
R = "experiments/008-scale-sensitivity/pilot/"
s = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False, dtype={"vidx": str})
s = s[s.split.isin(["calib", "test"])].copy()
def sz(p):
    with Image.open(p) as im: return im.size
wh = [sz(p) for p in s.src_path]
s["short"] = [min(a) for a in wh]; s["long"] = [max(a) for a in wh]
s["grp"] = np.where(s.pool == "fake", s.generator, s.pool)
print(s.groupby(["grp"]).agg(n=("short", "size"), short_med=("short", "median"), short_min=("short", "min"), short_max=("short", "max"),
                            long_med=("long", "median")).to_string())
print(s[s.pool=="fake"].groupby(["generator","orig_format"]).size())
s[["sha256", "grp", "split", "device", "short", "long"]].to_csv(R + "critic_checks/sizes.csv", index=False)
d = pd.read_csv(R + "scores_long.csv.gz")
d = d[d.model.isin(["pretrained"])].merge(s[["sha256", "grp", "split", "short"]], on="sha256")
t = d[d.split == "test"]
print("\nmedian score by group (test)")
print(t.groupby(["detector", "condition", "arm", "grp"]).score.median().unstack("grp").round(2).to_string())
print("\nper-generator AUROC vs test natives")
rows = []
for (det, c, a), g in t.groupby(["detector", "condition", "arm"]):
    nat = g[g.grp == "vision_native"].score.values
    for gen in sorted(g[g.split == "test"].grp.unique()):
        if gen.startswith("vision"): continue
        f = g[g.grp == gen].score.values
        rows.append(dict(det=det, cond=c, arm=a, gen=gen, auc=roc_auc_score(np.r_[np.zeros(len(nat)), np.ones(len(f))], np.r_[nat, f])))
A = pd.DataFrame(rows).pivot_table(index=["det", "gen"], columns=["cond", "arm"], values="auc").round(3)
print(A.to_string())
