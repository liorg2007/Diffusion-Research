"""Critic check c2: paired bootstrap decomposition layer vs laundering-augmentation, per real pool, plus twin-paired. Uses E6 cached scores and E6 Plan/TwinPlan (same bootstrap plan as run.py)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import run as E

pairs = [("F_b8|T-aug|all", "F_last|T-aug|all", "layer @ T-aug"),
         ("F_b8|T-clean|all", "F_last|T-clean|all", "layer @ T-clean"),
         ("F_last|T-aug|all", "F_last|T-clean|all", "aug @ last"),
         ("F_b8|T-aug|all", "F_b8|T-clean|all", "aug @ b8"),
         ("F_b8|T-aug|all", "F_last|T-clean|all", "P1 total (vs matched B0)"),
         ("F_cat|T-aug|all", "F_last|T-aug|all", "cat vs last @ T-aug")]
out = {}
for arm in ("orig", "fov512"):
    A = E.Arm(arm, load_feats=False)
    plans = {k: E.Plan(A, E.POOLS[k]) for k in ("primary", "native", "RAISE", "WA")}
    twin = E.TwinPlan(A)
    S = {n: E.load_scores(arm, "std", tuple(n.split("|")))[0] for n in {p for a, b, _ in pairs for p in (a, b)}}
    o = {}
    for cond in ("clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"):
        ci = E.CI[cond]
        oc = {}
        for a, b, lab in pairs:
            r = {}
            for pk, pl in plans.items():
                ga, gb = pl.gen_auc(S[a], ci), pl.gen_auc(S[b], ci)
                r[pk + "_macro8"] = E.diff_summary(E.macro(ga, E.GENS8), E.macro(gb, E.GENS8))
                if pk == "primary":
                    r["primary_flux"] = E.diff_summary(ga[E.FLUX], gb[E.FLUX])
                if pk == "RAISE":
                    r["RAISE_flux"] = E.diff_summary(ga[E.FLUX], gb[E.FLUX])
            ta, tb = twin.auc(S[a], ci), twin.auc(S[b], ci)
            r["twin_pooled"] = E.diff_summary(ta["pooled"], tb["pooled"])
            oc[lab] = r
        o[cond] = oc
    out[arm] = o
(HERE / "c2_layer_vs_aug.json").write_text(json.dumps(out, indent=1))
for arm, o in out.items():
    for cond in ("jpeg75_resize05", "clean", "screenshot_sim", "mavali_medium"):
        print(f"== {arm} {cond}")
        for lab, r in o[cond].items():
            print(f"  {lab:28s} " + " ".join(f"{k}={v['delta']:+.3f}[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]" for k, v in r.items()))
