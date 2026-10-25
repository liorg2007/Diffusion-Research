"""Critic check 1 (CPU, read-only): shift distributions per class/generator, AUROC recomputation, score correlations.
Reads pilot/scores_long.csv.gz only. Writes critic_checks/c1_scores.json.
  env/.venv/bin/python -I experiments/012-self-ref/pilot/critic_checks/c1_scores.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "scores_long.csv.gz")
GENS = sorted(g for g in d[d.kind == "fake"].generator.dropna().unique())
out = {}


def auc(neg, pos):
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


for arm in ("fov512", "orig"):
    out[arm] = {}
    for cond in ("clean", "jpeg75_resize05", "social_sim"):
        o = out[arm].setdefault(cond, {})
        x = d[(d.arm == arm) & (d.cond == cond)]
        for sc in ("A", "B", "Bu", "blurA", "blurB", "F2p"):
            s = x[x.score == sc]
            nat = s[(s.kind == "real") & (s.pool == "vision_native")].value.to_numpy()
            fk = s[s.kind == "fake"]
            if len(nat) == 0:
                continue
            # shift = -score for shift scores
            sign = -1 if sc not in ("F2p",) else 1
            r = dict(macro_recomputed=float(np.mean([auc(nat, fk[fk.generator == g].value.to_numpy()) for g in GENS])),
                     native_median_shift=float(np.median(sign * nat)), native_iqr=[float(v) for v in np.percentile(sign * nat, [25, 75])],
                     native_frac_pos=float((sign * nat > 0).mean()))
            r["fake_median_shift"] = {g: float(np.median(sign * fk[fk.generator == g].value)) for g in GENS}
            r["fake_frac_pos"] = {g: float((sign * fk[fk.generator == g].value > 0).mean()) for g in GENS}
            tr = s[s.kind == "train_real"].value.to_numpy()
            if len(tr):
                r["train_real_median_shift"] = float(np.median(sign * tr))
            o[sc] = r
        # rank correlations over test natives + fakes between scores (is A just blur/texture?)
        wide = x[x.kind.isin(["real", "fake"]) & ((x.pool == "vision_native") | (x.kind == "fake"))].pivot_table(
            index="sha256", columns="score", values="value")
        cors = {}
        for a, b in (("A", "blurA"), ("A", "texture"), ("B", "blurB"), ("B", "texture"), ("F2p", "texture"), ("A", "F2p"), ("B", "F2p"),
                     ("A", "pixL2"), ("blurA", "texture")):
            if a in wide and b in wide:
                m = wide[[a, b]].dropna()
                cors[f"{a}~{b}"] = float(spearmanr(m[a], m[b])[0])
        o["spearman_natives_plus_fakes"] = cors
(HERE / "c1_scores.json").write_text(json.dumps(out, indent=1))
for arm in out:
    for cond in out[arm]:
        o = out[arm][cond]
        print(arm, cond, "macro B rec", round(o["B"]["macro_recomputed"], 3), "A", round(o["A"]["macro_recomputed"], 3),
              "F2p", round(o["F2p"]["macro_recomputed"], 3))
        for sc in ("A", "B", "blurB"):
            r = o[sc]
            print(f"   {sc}: native med {r['native_median_shift']:.3f} frac+ {r['native_frac_pos']:.2f} | fakes med " +
                  " ".join(f"{g[:6]}={v:.2f}" for g, v in r["fake_median_shift"].items()) + " | frac+ " +
                  " ".join(f"{v:.2f}" for v in r["fake_frac_pos"].values()))
        print("   cors", {k: round(v, 2) for k, v in o["spearman_natives_plus_fakes"].items()})
