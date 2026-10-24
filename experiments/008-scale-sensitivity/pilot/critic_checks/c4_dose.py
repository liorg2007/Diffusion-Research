"""Critic check 4: dose-response of the arm effect on the fov512 down-scale factor k = short/512 (zoom == resample factor here).
Per-device (natives) and per-image (fakes) delta = score(fov512) - score(orig). Descriptive."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
R = "experiments/008-scale-sensitivity/pilot/"
sz = pd.read_csv(R + "critic_checks/sizes.csv")
d = pd.read_csv(R + "scores_long.csv.gz")
d = d[d.model.isin(["pretrained", "full"])].merge(sz, on="sha256")
d = d[d.split == "test"].copy(); d["device"] = d.device.fillna("fake")
w = d.pivot_table(index=["detector", "condition", "sha256", "grp", "device", "short"], columns="arm", values="score").reset_index()
w["delta"] = w.fov512 - w.orig; w["k"] = w.short / 512
for (det, c), g in w.groupby(["detector", "condition"]):
    nat = g[g.grp == "vision_native"]
    pd_ = nat.groupby("device").agg(k=("k", "median"), dmed=("delta", "median"), n=("delta", "size"))
    fk = g[~g.grp.str.startswith("vision")]
    r_dev = spearmanr(pd_.k, pd_.dmed)[0]
    r_fk = spearmanr(fk.k, fk.delta)[0]
    sd = np.std(np.r_[g.orig.values])
    by = fk.groupby("grp").agg(k=("k", "median"), dmed=("delta", "median"))
    wa = g[g.grp == "vision_whatsapp"].delta.median(); fb = g[g.grp == "vision_facebook_hq"].delta.median()
    print(f"{det:18s} {c:15s} natives: median delta {nat.delta.median():+.2f} (k med {nat.k.median():.1f}); per-device Spearman(k, delta) {r_dev:+.2f} over {len(pd_)} devices "
          f"(k range {pd_.k.min():.1f}-{pd_.k.max():.1f}) | WA {wa:+.2f} (k 1.9) FBH {fb:+.2f} (k 3.0) | fakes Spearman(k,delta) {r_fk:+.2f}; "
          + " ".join(f"{gn[:6]}(k{r.k:.1f}):{r.dmed:+.2f}" for gn, r in by.iterrows()) + f" | score sd(orig) {sd:.2f}")
# low-res native devices (k<2): do they behave like fakes / WA?
print()
for (det, c), g in w.groupby(["detector", "condition"]):
    nat = g[g.grp == "vision_native"]
    lo, hi = nat[nat.k < 2], nat[nat.k >= 4]
    print(f"{det:18s} {c:15s} natives k<2: n={len(lo)} devices={lo.device.nunique()} median delta {lo.delta.median():+.2f} | k>=4: n={len(hi)} devices={hi.device.nunique()} median delta {hi.delta.median():+.2f}")
