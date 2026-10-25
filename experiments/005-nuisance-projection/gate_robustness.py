"""Amendment 13 gate robustness for H1: headroom decision and label with Firefly in/out and each generator dropped in turn.
Reads <dir>/results.json (per-generator LOGO AUROCs of the logo_g models, primary reals); macro = mean over the kept generators.
FLUX gain (held-out, 'full' model) and drift reduction are unchanged by the subset. A label that differs from the all-generator
label is 'fragile'.   usage: gate_robustness.py <results_dir> [...]"""
import json, sys
from pathlib import Path
import numpy as np

def label(r, keep):
    A, H = r["arms"], r.get("heldout", {})
    mac = lambda a, c: float(np.mean([A[a][c]["per_generator"][g]["auroc"] for g in keep]))
    h = mac("A1", "jpeg75_resize05")
    gains = {"logo_macro": mac("A2", "jpeg75_resize05") - h}
    for g, v in H.get("A2", {}).get("jpeg75_resize05", {}).items():
        gains[g] = v - H["A1"]["jpeg75_resize05"][g]
    gain = min(gains.values())
    cd, sd = mac("A2", "clean") - mac("A1", "clean"), mac("A2", "screenshot_sim") - mac("A1", "screenshot_sim")
    d1, d2 = A["A1"]["drift"]["mean_drift"], A["A2"]["drift"]["mean_drift"]
    red = (d1 - d2) / d1 if d1 > 0 else float("nan")
    holds = [gain >= 0.03, cd >= -0.01, sd >= 0, bool(np.isfinite(red) and red >= 0.5)]
    v = "PASS" if all(holds) else ("FAIL" if (gain < 0.01 or -cd > 0.02) else "INCONCLUSIVE")
    head = h <= 0.96
    return dict(A1_macro=h, headroom=bool(head), gain_macro=gains["logo_macro"], gain_min=gain, clean_d=cd, ss_d=sd, drift_red=red,
                label_ignoring_headroom=v, label=v if head else "UNINFORMATIVE_NO_HEADROOM")

out = []
for d in sys.argv[1:]:
    r = json.loads((Path(d) / "results.json").read_text())
    gens = r["options"]["generators"]
    rows = [("all " + str(len(gens)), gens), ("no firefly", [g for g in gens if g != "firefly"])]
    rows += [("drop " + g, [x for x in gens if x != g]) for g in gens]
    base = label(r, gens)["label"]
    out.append(f"## {d}  (primary pools {r['options']['primary_pools']}; all-generator label {base})\n")
    out.append("| generators | A1 macro @jr | headroom | gain macro | gain min(macro,FLUX) | clean d | ss d | drift red | label | ignoring headroom | fragile |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, k in rows:
        x = label(r, k)
        out.append(f"| {n} | {x['A1_macro']:.4f} | {x['headroom']} | {x['gain_macro']:+.4f} | {x['gain_min']:+.4f} | {x['clean_d']:+.4f} | "
                   f"{x['ss_d']:+.4f} | {x['drift_red']:.2f} | {x['label']} | {x['label_ignoring_headroom']} | {'FRAGILE' if x['label'] != base else ''} |")
    out.append("")
txt = "\n".join(out); print(txt)
