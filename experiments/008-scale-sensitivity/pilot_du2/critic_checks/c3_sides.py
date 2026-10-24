"""Critic check Q1/Q2: from pilot_du2/scores_long.csv.gz only.
(a) 3x3 swap matrix AUROC(reals from arm a, fakes from arm b), macro over 6 gens, per pool -> real-side vs fake-side parts
    of Delta_resample / Delta_zoom.  (b) group median scores per arm.  (c) within natives: Spearman(k, score_du2 - score_orig).
(d) recompute reported point Deltas from scores_long (independent of run_du2.analyze)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[4]; HERE = Path(__file__).resolve().parent
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]
A3 = ["orig", "du2", "fov512"]
L = pd.read_csv(ROOT / "experiments/008-scale-sensitivity/pilot_du2/scores_long.csv.gz")
df = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False).set_index("sha256")
sz = pd.read_csv(ROOT / "experiments/008-scale-sensitivity/pilot/critic_checks/sizes.csv").set_index("sha256")
res = json.loads((ROOT / "experiments/008-scale-sensitivity/pilot_du2/results.json").read_text())
L = L[L.sha256.map(df.split) == "test"]
L["grp"] = np.where(L.sha256.map(df.label) == 1, L.sha256.map(df.generator),
                    L.sha256.map(df.pool).map({"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}))
L["k"] = L.sha256.map(sz.short) / 512
ROWS = [("communityforensics", "pretrained"), ("rine", "pretrained"), ("clipdet", "pretrained"), ("b0_trained_orig", None), ("b0_trained_fov512", None), ("b0", None)]

def vec(det, model, arm, cond, grp):
    if det.startswith("b0_trained_") and det[len("b0_trained_"):] == arm:   # write_long stores X[a][a] as detector "b0"
        det = "b0"
    q = L[(L.detector == det) & (L.arm == arm) & (L.condition == cond) & (L.grp == grp)]
    if model: q = q[q.model == model]
    return q.set_index("sha256").score.sort_index()

def macro(det, cond, pool, ar, af):
    out = []
    for g in GENS:
        m = "pretrained" if det in ("communityforensics", "rine", "clipdet") else f"logo_{g}"
        r, f = vec(det, m, ar, cond, pool), vec(det, m, af, cond, g)
        out.append(roc_auc_score(np.r_[np.zeros(len(r)), np.ones(len(f))], np.r_[r.values, f.values]))
    return float(np.mean(out))

out = {}
for cond in ("clean", "jpeg75_resize05"):
    print(f"\n######## {cond}")
    for det, _ in ROWS:
        for pool in ("native", "WA", "FBH"):
            M = {(a, b): macro(det, cond, pool, a, b) for a in A3 for b in A3}
            dr, dz = M["orig", "orig"] - M["du2", "du2"], M["du2", "du2"] - M["fov512", "fov512"]
            # Delta_resample = fake-side [o,o]-[o,d] + real-side [o,d]-[d,d]; Delta_zoom = fake-side [d,d]-[d,f] + real-side [d,f]-[f,f]
            dr_f, dr_r = M["orig", "orig"] - M["orig", "du2"], M["orig", "du2"] - M["du2", "du2"]
            dz_f, dz_r = M["du2", "du2"] - M["du2", "fov512"], M["du2", "fov512"] - M["fov512", "fov512"]
            out[f"{cond}/{det}/{pool}"] = dict(M={f"{a}R|{b}F": v for (a, b), v in M.items()}, dr=dr, dz=dz, dr_fake=dr_f, dr_real=dr_r, dz_fake=dz_f, dz_real=dz_r)
            rep = res["per_pool"][cond][det][pool]
            chk = abs(dr - rep["delta_resample"]) + abs(dz - rep["delta_zoom"])
            print(f"{det:20s} {pool:6s} AUROC o/d/f {M['orig','orig']:.3f}/{M['du2','du2']:.3f}/{M['fov512','fov512']:.3f} | "
                  f"dR {dr:+.3f} = fake {dr_f:+.3f} + real {dr_r:+.3f} | dZ {dz:+.3f} = fake {dz_f:+.3f} + real {dz_r:+.3f} | "
                  f"realsDU2 vs fakesORIG {M['du2','orig']:.3f} | |recomp-reported| {chk:.1e}")
    print(f"\n--- {cond}: median score per group per arm (pretrained detectors; B0 = logo_dalle2 model) ---")
    for det, m in ROWS[:5]:
        mm = m or "logo_dalle2"
        q = pd.concat([vec(det, mm, a, cond, g).rename("score").to_frame().assign(arm=a, grp=g) for a in A3 for g in L.grp.unique()])
        print(det); print(q.groupby(["grp", "arm"]).score.median().unstack("arm")[A3].round(2).to_string())
    print(f"\n--- {cond}: within test natives, Spearman(k, score_du2 - score_orig) and (k, score_fov512 - score_du2) ---")
    for det, m in ROWS[:5]:
        mm = m or "logo_dalle2"
        q = pd.DataFrame({a: vec(det, mm, a, cond, "native") for a in A3})
        k = q.index.map(sz.short) / 512
        a = spearmanr(k, q.du2 - q.orig).statistic; b = spearmanr(k, q.fov512 - q.du2).statistic
        lo = k < 4; print(f"{det:20s} rho(k, du2-orig) {a:+.2f}  rho(k, fov-du2) {b:+.2f} | median(du2-orig) k<4 (n={lo.sum()}) {np.median((q.du2-q.orig)[lo]):+.2f} vs k>=4 {np.median((q.du2-q.orig)[~lo]):+.2f}")
(HERE / "c3_sides.json").write_text(json.dumps(out, indent=1))
