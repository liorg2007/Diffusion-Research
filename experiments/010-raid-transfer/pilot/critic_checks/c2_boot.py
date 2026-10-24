"""Critic check 2 (CPU, saved scores): paired bootstrap (same plan as run.py: metrics.bootstrap_weights, 2000, seed 0)
for (a) residual attack drop in canon space = AUROC(clean_canon) - AUROC(eps8_canon), (b) recovery ceiling
= [AUROC(clean_canon) - AUROC(eps8)] / drop, (c) cross-class decomposition CIs, plus B0 score vs original size."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent
P = H.parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import metrics  # noqa: E402

df = pd.read_csv(P / "scores_long.csv.gz", dtype={"id": str})
DETS = ["b0", "communityforensics", "rine", "clipdet"]
GENS = ["gen_0", "gen_1", "gen_2", "gen_3"]
base = df[(df.detector == "b0") & (df.input_set == "clean")].reset_index(drop=True)   # item order (class-major, id)
out = {}
for key, rmask in (("primary", ~base.padded.to_numpy()), ("sensitivity_all", np.ones(len(base), bool))):
    R = np.where((base.cls == "real").to_numpy() & rmask)[0]
    F = np.where((base.cls != "real").to_numpy())[0]
    Wr, Wf = metrics.bootstrap_weights(base.id.to_numpy()[R], base.cls.to_numpy()[F], 2000, 0)
    one_r, one_f = np.ones((1, len(R))), np.ones((1, len(F)))

    def S(det, s):
        d = df[(df.detector == det) & (df.input_set == s)].reset_index(drop=True)
        assert (d.id.to_numpy() == base.id.to_numpy()).all() and (d.cls.to_numpy() == base.cls.to_numpy()).all()
        return d.score.to_numpy()

    def A(sr_set, sf_set, det):
        sr, sf = S(det, sr_set)[R], S(det, sf_set)[F]
        return float(metrics.weighted_auroc(sr, sf, one_r, one_f)[0]), metrics.weighted_auroc(sr, sf, Wr, Wf)

    out[key] = {}
    for det in DETS:
        c, cc, e, ec = A("clean", "clean", det), A("clean_canon", "clean_canon", det), A("eps8", "eps8", det), A("eps8_canon", "eps8_canon", det)
        ro, fo = A("eps8", "clean", det), A("clean", "eps8", det)
        cro, cfo = A("clean_canon", "clean", det), A("clean", "clean_canon", det)
        q = lambda b: [round(float(x), 3) for x in np.percentile(b, [2.5, 97.5])]
        drop_b = c[1] - e[1]
        ok = drop_b > 0.02
        ceil_b = np.where(ok, (cc[1] - e[1]) / np.where(ok, drop_b, 1), np.nan)
        r = dict(residual_drop_canon_space=(round(cc[0] - ec[0], 3), q(cc[1] - ec[1])),
                 recovery_ceiling=(round((cc[0] - e[0]) / (c[0] - e[0]), 3), [round(float(x), 3) for x in np.nanpercentile(ceil_b, [2.5, 97.5])]),
                 drop_from_reals_only=(round(c[0] - ro[0], 3), q(c[1] - ro[1])),
                 drop_from_fakes_only=(round(c[0] - fo[0], 3), q(c[1] - fo[1])),
                 cost_from_reals_only=(round(c[0] - cro[0], 3), q(c[1] - cro[1])),
                 cost_from_fakes_only=(round(c[0] - cfo[0], 3), q(c[1] - cfo[1])))
        # gen_0 excluded (CommFor weakest; native 256)
        out[key][det] = r
        print(key, det, json.dumps(r))

# B0 vs original size (reals): sizes from file headers
m = pd.read_csv(P.parent / "raid_manifest.csv", dtype={"id": str})
orig = m[m.kind == "original"].set_index(["cls", "id"]).path
sz = {(c, i): Image.open(ROOT / orig[(c, i)]).size for c, i in orig.index}
for s in ("clean", "eps8"):
    d = df[(df.detector == "b0") & (df.input_set == s)].copy()
    d["short"] = [min(sz[(c, i)]) for c, i in zip(d.cls, d.id)]
    d["area"] = [sz[(c, i)][0] * sz[(c, i)][1] for c, i in zip(d.cls, d.id)]
    ru = d[(d.cls == "real") & ~d.padded]
    print(f"b0 {s}: unpadded reals frac score>0 {np.mean(ru.score > 0):.2f}; spearman(score, short side) "
          f"{spearmanr(ru.score, ru.short)[0]:.2f} p={spearmanr(ru.score, ru.short)[1]:.3f}; all reals frac>0 {np.mean(d[d.cls=='real'].score>0):.2f}")
    for g in GENS:
        gg = d[d.cls == g]
        print(f"   {g}: short side median {gg.short.median():.0f} [{gg.short.min()}, {gg.short.max()}], frac>0 {np.mean(gg.score > 0):.2f}")
    print("   unpadded reals short side median", ru.short.median(), "range", ru.short.min(), ru.short.max())
(H / "c2_out.json").write_text(json.dumps(out, indent=1))
