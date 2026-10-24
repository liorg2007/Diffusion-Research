"""Q4: reference-based (x vs VAE(x)) scores from c2_ref_scores.csv.gz -> c3_reference.json.
Fake = high. Scores: -err (mean |x - VAE(x)| on the 224 centre crop; fakes from an SD-like decoder should reconstruct better),
lapratio (Laplacian of VAE(x) / of x; reals lose more detail), -d_F* (probe score change s(VAE(x)) - s(x); reals should move more),
s_F* (the single-image pair probe, i.e. the transfer reading on the same sample). Reals by device cluster, fakes by image per generator."""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run011", HERE.parents[1] / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
t = pd.read_csv(HERE / "c2_ref_scores.csv.gz", keep_default_na=False)
out = {}
for (arm, c), g in t.groupby(["arm", "cond"]):
    r, f = g[g.label == 0], g[g.label == 1]
    plan = R.e2.Plan(r.device.to_numpy(), f.generator.to_numpy(), n_boot=1000)
    res = {}
    for nm, sc in (("neg_err", lambda x: -x.err), ("lapratio", lambda x: x.lapratio), ("neg_d_F4", lambda x: -x.d_F4), ("neg_d_F2", lambda x: -x.d_F2),
                   ("neg_d_F3", lambda x: -x.d_F3), ("s_F4", lambda x: x.s_F4), ("s_F2", lambda x: x.s_F2), ("s_F3", lambda x: x.s_F3)):
        sr, sf = sc(r).to_numpy(float), sc(f).to_numpy(float)
        res[nm] = dict(macro=R.strip(plan.macro(sr, sf)), sd=R.strip(plan.macro(sr, sf, R.SD_FAM)), other=R.strip(plan.macro(sr, sf, R.OTHER_FAM)),
                       per_gen={k: round(v, 3) for k, v in plan.per_gen(sr, sf).items()})
    res["median_err"] = {"real": float(r.err.median()), **{k: float(v.err.median()) for k, v in f.groupby("generator")}}
    res["median_d_F4"] = {"real": float(r.d_F4.median()), **{k: float(v.d_F4.median()) for k, v in f.groupby("generator")}}
    res["frac_d_F4_pos"] = {"real": float((r.d_F4 > 0).mean()), **{k: float((v.d_F4 > 0).mean()) for k, v in f.groupby("generator")}}
    res["n"] = dict(real=len(r), fake=len(f))
    out[f"{arm}/{c}"] = res
(HERE / "c3_reference.json").write_text(json.dumps(out, indent=1, default=float))
for k, v in out.items():
    print("==", k, v["n"])
    for nm in ("neg_err", "lapratio", "neg_d_F4", "neg_d_F2", "neg_d_F3", "s_F4", "s_F2", "s_F3"):
        x = v[nm]
        print(f"  {nm:9s} macro {x['macro']['value']:.3f} [{x['macro']['ci_lo']:.3f},{x['macro']['ci_hi']:.3f}]  sd {x['sd']['value']:.3f}  other {x['other']['value']:.3f}  {x['per_gen']}")
    print("  median err", {a: round(b, 2) for a, b in v["median_err"].items()})
    print("  median dF4", {a: round(b, 2) for a, b in v["median_d_F4"].items()}, "frac>0", {a: round(b, 2) for a, b in v["frac_d_F4_pos"].items()})
