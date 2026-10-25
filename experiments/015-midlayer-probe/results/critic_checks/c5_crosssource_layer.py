"""Critic check c5: does block 8 transfer better than block 12 across real sources? Paired bootstrap (E6 Plan) of
cross-source LOGO-macro8 / FLUX, F_b8 vs F_last at the same regime, both arms."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import run as E
out = {}
for arm in ("orig", "fov512"):
    A = E.Arm(arm, load_feats=False)
    pv, pr = E.Plan(A, E.POOLS["vision3"]), E.Plan(A, E.POOLS["RAISE"])
    o = {}
    for reg in ("T-aug", "T-clean"):
        for src, pl, lab in (("raise_only", pv, "RAISE->VISION3"), ("vision_only", pr, "VISION->RAISE")):
            a = E.load_scores(arm, "std", ("F_b8", reg, src))[0]
            b = E.load_scores(arm, "std", ("F_last", reg, src))[0]
            for cond in ("clean", "jpeg75_resize05", "screenshot_sim"):
                ci = E.CI[cond]
                ga, gb = pl.gen_auc(a, ci), pl.gen_auc(b, ci)
                o[f"{reg} {lab} {cond}"] = dict(F_b8=float(E.macro(ga, E.GENS8)[0]), F_last=float(E.macro(gb, E.GENS8)[0]),
                                                diff_macro8=E.diff_summary(E.macro(ga, E.GENS8), E.macro(gb, E.GENS8)),
                                                diff_flux=E.diff_summary(ga[E.FLUX], gb[E.FLUX]))
    out[arm] = o
(HERE / "c5_crosssource_layer.json").write_text(json.dumps(out, indent=1))
for arm, o in out.items():
    for k, v in o.items():
        d, f = v["diff_macro8"], v["diff_flux"]
        print(f"{arm:6s} {k:40s} b8 {v['F_b8']:.3f} last {v['F_last']:.3f} d {d['delta']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] flux d {f['delta']:+.3f} [{f['ci_lo']:+.3f},{f['ci_hi']:+.3f}]")
