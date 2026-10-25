"""Run every OFFICIAL H2-v2 analysis for one arm (orig | fov512). Pre-registered: Amendment 4 (platform-only groups at min-group 100 =
primary; platform x brand at min-group 50 = sensitivity), Amendment 13 (Firefly in/out; each generator dropped in turn; twin-disjoint
B0 arm `B0_twinfree` is inside the scores file). Verdict rule = h2v2.py:32 (unchanged analyze.py --v2 --label OFFICIAL).

  python -I experiments/007-worst-env-conformal/official/run_analyses.py orig
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
arm = sys.argv[1]
scores = HERE / f"scores_long_{arm}.csv.gz"
base = pd.read_csv(HERE / f"splits_{arm}.csv", keep_default_na=False, dtype={"vidx": str})
gens = sorted(base.generator[base.label == 1].unique())


def run(tag, splits, groupings, min_group):
    out = HERE / f"{arm}_{tag}"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/007-worst-env-conformal/analyze.py"), "--v2",
                        "--scores", str(scores), "--splits", str(splits), "--out-dir", str(out), "--min-group", str(min_group),
                        "--groupings", *groupings, "--label", "OFFICIAL"], capture_output=True, text=True)
    (out.parent / f"{arm}_{tag}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])


run("primary_platform_min100", HERE / f"splits_{arm}.csv", ["platform"], 100)
run("sens_platform_brand_min50", HERE / f"splits_{arm}.csv", ["platform_brand", "platform"], 50)
for g in gens:
    sp = HERE / f"splits_{arm}_drop_{g}.csv"
    base[~((base.label == 1) & (base.generator == g))].to_csv(sp, index=False)
    run(f"primary_drop_{g}", sp, ["platform"], 100)
    if g == "firefly":
        run("sens_platform_brand_min50_drop_firefly", sp, ["platform_brand", "platform"], 50)
print("done", arm)
