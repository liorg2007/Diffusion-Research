"""Critic check c3: fov512 down-scale factor per pool (512 / raw short side) and the OI-held-out score shift vs fakes."""
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
H = Path(__file__).resolve().parents[2]; ROOT = H.parents[1]
sp = pd.read_csv(H / "splits_fov512.csv", keep_default_na=False)
rng = np.random.default_rng(0)
for pool, g in sp.groupby(sp.pool.where(sp.pool != "fake", "fake:" + sp.generator)):
    sub = g.sample(min(40, len(g)), random_state=0)
    ss = []
    for p in sub.src_path:
        try:
            with Image.open(ROOT / p) as im: ss.append(min(im.size))
        except Exception as e:
            pass
    if ss:
        print(f"{pool:28s} n={len(ss):3d} raw short side median {np.median(ss):6.0f}  fov512 scale factor {512/np.median(ss):.2f}")
s = pd.read_csv(H / "results/scores_long.csv.gz")
a = s[(s.arm == "fov512") & (s.heldout == "OI") & (s.feat == "F_b8") & (s.cond == "jpeg75_resize05") & (s.role == "test")]
fk = a[a.label == 1].score; oi = a[a.pool == "openimages"].score
print("OI held-out, jpeg75_resize05: median fake", round(fk.median(), 2), "| median OI real", round(oi.median(), 2),
      "| OI reals above the median fake:", round((oi >= fk.median()).mean(), 3))
for S, pools in (("VISION", ["vision_native", "vision_whatsapp", "vision_facebook_hq"]), ("RAISE", ["raise"])):
    b = s[(s.arm == "fov512") & (s.heldout == S) & (s.feat == "F_b8") & (s.cond == "jpeg75_resize05") & (s.role == "test")]
    print(f"{S} held-out: median fake {b[b.label==1].score.median():.2f} | median held-out real {b[b.pool.isin(pools)].score.median():.2f}")
