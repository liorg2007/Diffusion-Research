"""Critic check 1 (CPU): recompute label-cell point estimates from scores_long.csv.gz; per-generator drop-one gate margin;
per-image RIGID score shift M vs ii on the RAISE test reals (manipulation check at score level)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import h4
OUT = Path(__file__).resolve().parent
s = pd.read_csv(ROOT / "experiments/006-format-shortcut/official/scores_long.csv.gz")
MAIN = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large"]

def get(arm, cond, sg, feat="post"):
    q = s[(s.arm == arm) & (s.cond == cond) & (s.feat == feat) & (np.isclose(s.sigma, sg))]
    return q

def macro(real, fakes_by_g, gens):
    return np.mean([h4.auroc_point(real, fakes_by_g[g]) for g in gens])

lines = []
for arm in ["orig", "fov512", "fov224"]:
    for sg in [0.01, 0.02, 0.05]:
        cl, jp = get(arm, "clean", sg), get(arm, "jpeg(90)", sg)
        rr_c = cl[(cl.split == "test") & (cl.pool == "raise")].set_index("sha256")
        rr_j = jp[(jp.split == "test") & (jp.pool == "raise")].set_index("sha256").loc[rr_c.index]
        fk = cl[(cl.split == "test") & (cl.label == 1)]
        F = {g: fk[fk.generator == g].score.values for g in MAIN + ["flux"]}
        FL = {g: fk[fk.generator == g].laplacian.values for g in MAIN + ["flux"]}
        a_ii = macro(rr_c.score.values, F, MAIN); a_m = macro(rr_j.score.values, F, MAIN)
        tex_ii = macro(-rr_c.laplacian.values, {g: -v for g, v in FL.items()}, MAIN)  # direction sign: -1 if smoother=fake
        d = rr_j.score.values - rr_c.score.values
        lines.append(f"{arm} s={sg}: AUROC(ii)={a_ii:.3f} AUROC(M)={a_m:.3f} gap={a_m-a_ii:+.3f} | RAISE real score shift jpeg90-clean: "
                     f"mean {d.mean():+.4f} median {np.median(d):+.4f} frac>0 {np.mean(d>0):.2f} (real score sd {rr_c.score.std():.4f}); "
                     f"lap ratio {rr_j.laplacian.mean()/rr_c.laplacian.mean():.3f}")
        if sg == 0.02 or arm == "fov224":
            per = {g: h4.auroc_point(rr_c.score.values, F[g]) for g in MAIN}
            pert = {g: h4.auroc_point(-rr_c.laplacian.values, -FL[g]) for g in MAIN}
            lines.append("   per-gen RIGID(ii) " + " ".join(f"{g[:6]}={v:.3f}" for g, v in per.items()))
            lines.append("   per-gen tex(-lap)(ii) " + " ".join(f"{g[:6]}={v:.3f}" for g, v in pert.items()))
            for drop in MAIN:
                gg = [g for g in MAIN if g != drop]
                r_ = np.mean([per[g] for g in gg]); t_ = np.mean([pert[g] for g in gg])
                lines.append(f"   drop {drop:20s}: RIGID {r_:.3f}  tex(-lap) {t_:.3f}  margin {r_-t_:+.3f}")
(OUT / "c1_recompute.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
