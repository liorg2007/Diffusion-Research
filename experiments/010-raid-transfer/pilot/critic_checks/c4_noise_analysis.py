"""Critic check 4 (CPU): analyse c3 controls (primary set) with the run's bootstrap plan; plus content-paired win rate
(fake(id) > real(id)) for every detector/set. Usage: env/.venv/bin/python -I c4_noise_analysis.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[2] / "lib"))
import metrics  # noqa: E402

base = pd.read_csv(H.parent / "scores_long.csv.gz", dtype={"id": str})
base = base[(base.cls != "real") | ~base.padded].drop(columns="padded")
ctl = pd.read_csv(H / "c3_noise_scores.csv", dtype={"id": str}) if (H / "c3_noise_scores.csv").exists() else pd.DataFrame()
df = pd.concat([base, ctl], ignore_index=True)
ref = df[(df.detector == "b0") & (df.input_set == "clean")].sort_values(["cls", "id"])
R = ref[ref.cls == "real"].id.to_numpy()
Fc = ref[ref.cls != "real"].cls.to_numpy()
Wr, Wf = metrics.bootstrap_weights(R, Fc, 2000, 0)
one_r, one_f = np.ones((1, len(R))), np.ones((1, len(Fc)))


def S(det, s):
    d = df[(df.detector == det) & (df.input_set == s)].sort_values(["cls", "id"])
    return d[d.cls == "real"].score.to_numpy(), d[d.cls != "real"].score.to_numpy()


q = lambda b: f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
for det in sorted(set(ctl.detector)) if len(ctl) else []:
    cr, cf = S(det, "clean")
    c = metrics.weighted_auroc(cr, cf, Wr, Wf)
    c0 = metrics.weighted_auroc(cr, cf, one_r, one_f)[0]
    line = [f"{det:20s} clean {c0:.3f}"]
    for s in ("eps8", "signflip", "shuffled"):
        if not len(df[(df.detector == det) & (df.input_set == s)]):
            continue
        sr, sf = S(det, s)
        a = metrics.weighted_auroc(sr, sf, Wr, Wf)
        a0 = metrics.weighted_auroc(sr, sf, one_r, one_f)[0]
        rr = metrics.weighted_auroc(sr, cf, one_r, one_f)[0]
        ff = metrics.weighted_auroc(cr, sf, one_r, one_f)[0]
        line.append(f"{s}: {a0:.3f} drop {c0 - a0:.3f} {q(c - a)} | reals-only {rr:.3f} fakes-only {ff:.3f} | "
                    f"med shift real {np.median(sr - cr):+.2f} fake {np.median(sf - cf):+.2f}")
    print("\n   ".join(line))

print("\ncontent-paired win rate P(score fake(id) > score real(id)), unpadded real ids, all 4 generators (vs pooled AUROC)")
for det in ["b0", "communityforensics", "rine", "clipdet"]:
    out = []
    for s in ("clean", "eps8", "clean_canon", "eps8_canon"):
        d = df[(df.detector == det) & (df.input_set == s)].pivot(index="id", columns="cls", values="score").dropna(subset=["real"])
        out.append(f"{s} {np.mean([(d[g] > d.real).mean() for g in ['gen_0', 'gen_1', 'gen_2', 'gen_3']]):.3f}")
    print(f"{det:20s}", " | ".join(out))
