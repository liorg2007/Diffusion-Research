"""Analyse c4_scores.csv: (ii) du2chk CPU vs cached du2 scores; AUROC native pool for orig / du2 / du2m / fov512 with the
SAME paired cluster bootstrap plan as run_du2 (E1 Plan, seed 0, 2000); U = du2 - du2m (effect of the natives' extra
bandwidth loss), alternative decomposition dR' = orig - du2m, dZ' = du2m - fov512, |dR'|-|dZ'| CI."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import run as R
L = pd.read_csv(HERE.parent / "scores_long.csv.gz")
C = pd.read_csv(HERE / "c4_scores.csv")
df = R.load_df("orig"); sel = R.selection(df)
nat = np.where(sel & (df.label == 0).to_numpy() & (df.split == "test").to_numpy() & (df.pool == "vision_native").to_numpy())[0]
pl = R.Plan(df, nat, sel)
sh = df.sha256.to_numpy()
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
res = {}
def full(det, model, arm, cond):
    d = det if not (det.startswith("b0_trained_") and det.endswith(arm)) else "b0"
    q = L[(L.detector == d) & (L.model == model) & (L.arm == arm) & (L.condition == cond)].set_index("sha256").score
    v = np.full(len(df), np.nan); m = np.isin(sh, q.index); v[m] = q.loc[sh[m]].to_numpy(); return v
for cond in R.CONDS:
    for det in R.DETS + ["b0_trained_orig", "b0_trained_fov512"]:
        isb = det.startswith("b0")
        if not ((C.det == det) & (C.cond == cond) & (C.arm == "du2m")).any():
            print(f"{cond:16s} {det:18s} not scored in du2m (skipped)"); continue
        # (ii) self-check
        q = C[(C.det == det) & (C.cond == cond) & (C.arm == "du2chk")]
        if isb:
            dd = max(float(np.abs(q[q.model == m].set_index("sha256").score - pd.Series(full(det, m, "du2", cond), index=sh).loc[q[q.model == m].sha256].values).max()) for m in q.model.unique())
        else:
            dd = float(np.abs(q.set_index("sha256").score - pd.Series(full(det, "pretrained", "du2", cond), index=sh).loc[q.sha256].values).max())
        def getter(arm, g):
            m = f"logo_{g}" if isb else "pretrained"
            if arm != "du2m":
                return full(det, m, arm, cond)
            v = full(det, m, "du2", cond)                               # fakes: du2 scores
            qq = C[(C.det == det) & (C.cond == cond) & (C.arm == "du2m") & (C.model == m)].set_index("sha256").score
            v[nat] = qq.loc[sh[nat]].to_numpy()                          # natives: du2m
            return v
        A = {a: pl.auroc(lambda g, a=a: getter(a, g)) for a in ("orig", "du2", "du2m", "fov512")}
        pt = {a: A[a][1] for a in A}; bt = {a: A[a][2] for a in A}
        e = dict(selfcheck_cpu_vs_cached_du2_max_abs=dd, auroc=pt, auroc_ci={a: ci(bt[a]) for a in bt},
                 U_du2_minus_du2m=pt["du2"] - pt["du2m"], U_ci=ci(bt["du2"] - bt["du2m"]),
                 dR_alt=pt["orig"] - pt["du2m"], dR_alt_ci=ci(bt["orig"] - bt["du2m"]),
                 dZ_alt=pt["du2m"] - pt["fov512"], dZ_alt_ci=ci(bt["du2m"] - bt["fov512"]),
                 absdiff_alt=abs(pt["orig"] - pt["du2m"]) - abs(pt["du2m"] - pt["fov512"]),
                 absdiff_alt_ci=ci(np.abs(bt["orig"] - bt["du2m"]) - np.abs(bt["du2m"] - bt["fov512"])),
                 per_gen={a: A[a][0] for a in A})
        lo, hi = e["absdiff_alt_ci"]
        e["label_alt"] = "resampling-dominated" if lo > 0 else "zoom-dominated" if hi < 0 else "mixed"
        res[f"{cond}/{det}"] = e
        f = lambda x, c: f"{x:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]"
        print(f"{cond:16s} {det:18s} selfchk {dd:.1e} | AUROC orig {pt['orig']:.3f} du2 {pt['du2']:.3f} du2m {pt['du2m']:.3f} fov {pt['fov512']:.3f} | "
              f"U=du2-du2m {f(e['U_du2_minus_du2m'], e['U_ci'])} | dR' {f(e['dR_alt'], e['dR_alt_ci'])} dZ' {f(e['dZ_alt'], e['dZ_alt_ci'])} | absdiff' {f(e['absdiff_alt'], e['absdiff_alt_ci'])} -> {e['label_alt']}")
(HERE / "c5_du2m_analysis.json").write_text(json.dumps(res, indent=1))
