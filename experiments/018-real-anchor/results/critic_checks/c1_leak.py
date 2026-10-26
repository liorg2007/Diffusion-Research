"""E9 critic c1: is the literal A-off a label leak? fov512 / F_b8 / jpeg75_resize05 / n = 50, 20 draws (E9 anchors).
LOGO macro AUROC over 8 generators (E7 convention). Controls:
  A-off (reals only)     : reproduce E9
  A-off-all              : same per-model score shift applied to reals AND fakes (deployment)
  fakes-only (+shift)    : the same relative shift put on the fakes instead of the reals
  const-c (no anchors)   : subtract a constant c (in units of the reals' score SD) from the reals only
  fake-anchor control    : 'anchors' = 50 random TRAIN-role FAKES (no pipeline info at all), reals-only shift
  train-real anchors     : 'anchors' = 50 train-role reals of the TRAINING sources (delta ~ 0)
Plus: threshold-only ceiling on the E7 ranking (TPR at oracle 1% / 5% FPR on test reals)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import numpy as np
import run as R

ctx = R.get_ctx("fov512"); D = ctx.D; feat = "F_b8"; JR = R.CI_JR; n = 50
out = {}
def macro(sc, S, real_add=None, fake_add=None):
    a = []
    for g in ctx.gens:
        m = ctx.mi[g]
        sr = sc[m, JR, ctx.rr[S]] + (0 if real_add is None else real_add[m])
        sf = sc[m, JR, ctx.ff[g]] + (0 if fake_add is None else fake_add[m])
        a.append(R.auroc_point(sr, sf))
    return float(np.mean(a))

rng0 = np.random.default_rng(12345)
for S in R.SOURCES:
    base, W = R.base_scores(ctx, feat, S)
    st = R.static_stats(ctx, feat, S)
    T = ctx.others(S)
    trrows = ctx.E.train_rows(D, T, "full")
    tr_fakes = trrows[D.lab[trrows] == 1]; tr_reals = trrows[D.lab[trrows] == 0]
    sd_real = float(base[ctx.full, JR, ctx.rr[S]].std())
    margin = float(np.mean([base[ctx.full, JR, ctx.ff[g]].mean() for g in ctx.gens]) - base[ctx.full, JR, ctx.rr[S]].mean())
    r = {k: [] for k in ("A-off", "A-off-all", "fakes-only", "fake-anchor", "train-real-anchor")}
    shifts = []
    for d in range(R.ND):
        anc = ctx.anchor_order(S, d)[:n]
        _, mu_an = R.anchor_means(ctx, feat, anc)
        sh = W @ R.anchor_delta(st["mu"], mu_an); shifts.append(float(sh[ctx.full]))
        r["A-off"].append(macro(base, S, real_add=sh))
        r["A-off-all"].append(macro(base, S, real_add=sh, fake_add=sh))
        r["fakes-only"].append(macro(base, S, fake_add=-sh))
        rng = np.random.default_rng([7, d, R.SOURCES.index(S)])
        fa = rng.choice(tr_fakes, n, replace=False); _, mu_fa = R.anchor_means(ctx, feat, fa)
        r["fake-anchor"].append(macro(base, S, real_add=W @ R.anchor_delta(st["mu"], mu_fa)))
        ta = rng.choice(tr_reals, n, replace=False); _, mu_ta = R.anchor_means(ctx, feat, ta)
        r["train-real-anchor"].append(macro(base, S, real_add=W @ R.anchor_delta(st["mu"], mu_ta)))
    o = {k: dict(mean=float(np.mean(v)), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5))) for k, v in r.items()}
    o["E7"] = macro(base, S)
    o["shift_full_model_mean"] = float(np.mean(shifts)); o["shift_in_real_sd"] = float(np.mean(shifts) / sd_real)
    o["real_sd_full"] = sd_real; o["fake_minus_real_mean_full"] = margin
    curve = {}
    for c in (0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        add = np.full(R.NMODELS, 0.0)
        for m in range(R.NMODELS):
            add[m] = -c * base[m, JR, ctx.rr[S]].std()
        curve[str(c)] = macro(base, S, real_add=add)
    o["const_shift_reals_only_curve(c in real SD)"] = curve
    # threshold-only ceiling on the E7 ranking (full model, oracle threshold on the test reals)
    sr = base[ctx.full, JR, ctx.rr[S]]
    for f in (0.01, 0.05):
        t = np.quantile(sr, 1 - f)
        o[f"E7_full_TPR_at_oracle_FPR{int(f*100)}"] = float(np.mean([(base[ctx.full, JR, ctx.ff[g]] > t).mean() for g in ctx.gens]))
    out[S] = o
    print(S, json.dumps(o), flush=True)
json.dump(out, open(Path(__file__).with_name("c1_leak.json"), "w"), indent=1)
