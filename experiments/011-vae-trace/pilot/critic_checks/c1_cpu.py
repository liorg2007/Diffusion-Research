"""Critic CPU checks on saved E4 outputs: split/group integrity, pairing, F3 peak features, score structure (pooled vs paired),
QF / sharpness / pool dependence of the paired margin. Writes c1_cpu.json next to this file."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
E4 = HERE.parents[1]
spec = importlib.util.spec_from_file_location("run011", E4 / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
R.set_mode(False)
from PIL import Image
out = {}
qf = json.loads((R.CACHE / "qf.json").read_text())
L = pd.read_csv(R.PILOT / "scores_long.csv.gz")
def auc(a, b): return R.point_auc(a, b)
for arm in R.ARMS:
    df, D = R.load_arm(arm)
    o = out.setdefault(arm, {})
    lab, sp, pool, dev = (df[c].to_numpy() for c in ("label", "split", "pool", "device"))
    tr = np.where((lab == 0) & (sp == "train"))[0]; te = np.where((lab == 0) & (sp == "test"))[0]
    grp = R.e2.source_groups(df)
    o["devices_train"] = sorted(set(dev[tr])); o["devices_test_n"] = len(set(dev[te]))
    o["device_overlap_train_test"] = sorted(set(dev[tr]) & set(dev[te]))
    o["group_overlap_train_test"] = len(set(grp[tr]) & set(grp[te]))
    o["pool_counts_test"] = pd.Series(pool[te]).value_counts().to_dict()
    # pairing: rt files exist for each real, and rt is the VAE of the right real (mean abs diff to own real << to a random other real)
    rng = np.random.default_rng(0); sel = rng.choice(te, 40, replace=False)
    own, oth = [], []
    for i in sel:
        a = np.asarray(Image.open(df.path.iloc[i]).convert("RGB"), np.float32)
        b = np.asarray(Image.open(R.rt_file(arm, df.sha256.iloc[i])).convert("RGB"), np.float32)
        j = rng.choice(te); c = np.asarray(Image.open(df.path.iloc[j]).convert("RGB"), np.float32)
        own.append(np.abs(a - b).mean()); oth.append(np.abs(c - b).mean())
    o["pairing_mae_own_vs_other"] = [float(np.median(own)), float(np.median(oth)), float(np.max(own))]
    if arm == "orig":
        from preprocess import jpeg_quality_estimate
        bad = 0
        for i in sel:
            with Image.open(R.rt_file(arm, df.sha256.iloc[i])) as im:
                bad += int(jpeg_quality_estimate(im) != qf[df.sha256.iloc[i]])
        o["test_rt_qf_mismatch_of_40"] = bad
        q = np.array([qf[s] for s in df.sha256.iloc[te]])
        o["qf_by_pool_test"] = {p: [float(np.percentile(q[pool[te] == p], k)) for k in (5, 50, 95)] for p in sorted(set(pool[te]))}
    # F3 per-feature AUROC (rt high) on TEST, clean and laundered; peak features named
    names = []
    for ch in "RGB":
        names += [f"{ch}_bin{b}" for b in range(R.F3_BINS)] + [f"{ch}_pk{ax}{k}" for ax in "xy" for k in R.F3_PEAK_K]
    for c in R.CONDS:
        A, B = D[("f3_plain", c)][te, :R.F3_DIM], D[("f3_rt", c)][te, :R.F3_DIM]
        au = np.array([auc(A[:, k], B[:, k]) for k in range(R.F3_DIM)])
        win = ((B > A).mean(0))
        pk = [k for k, n in enumerate(names) if "_pk" in n]
        o[f"f3_{c}"] = dict(
            peaks={names[k]: dict(auroc=round(float(au[k]), 3), win=round(float(win[k]), 3), med_real=round(float(np.median(A[:, k])), 3),
                                  med_rt=round(float(np.median(B[:, k])), 3)) for k in pk if names[k].startswith("G")},
            top10=[(names[k], round(float(au[k]), 3), round(float(win[k]), 3)) for k in np.argsort(-np.abs(au - .5))[:10]],
            spectrum_bins_G_auroc=[round(float(au[k]), 2) for k in range(32, 64)] if False else [round(float(au[k]), 2) for k in range(40, 72)])
        # texture
        lr, lt = D[("f3_plain", c)][te, R.F3_DIM], D[("f3_rt", c)][te, R.F3_DIM]
        o[f"lap_{c}"] = dict(median_ratio_rt_over_real=float(np.median(lt / lr)), frac_rt_lower=float((lt < lr).mean()),
                             paired_logratio_auroc_note="see c3")
    # score structure: pooled vs paired, effect size; margin vs QF / sharpness / pool
    for fam in ("F2", "F4", "F3"):
        for c in R.CONDS:
            for proto in ("primary", "matched"):
                if proto == "matched" and c == R.CLEAN: continue
                s = L[(L.arm == arm) & (L.family == fam) & (L.train_protocol == proto) & (L.cond == c) & (L.split == "test")]
                r = s[s.kind == "real"].set_index("sha256").score; t = s[s.kind == "rt"].set_index("sha256").score.reindex(r.index)
                d = t - r
                key = f"{fam}_{proto}_{c}"
                rec = dict(pooled=auc(r.values, t.values), win=float((d > 0).mean()), d_mean=float(d.mean()), d_sd=float(d.std()),
                           sd_real=float(r.std()), shift_over_sd_real=float(d.mean() / r.std()))
                lap = pd.Series(D[("f3_plain", R.CLEAN)][te, R.F3_DIM], index=df.sha256.iloc[te].values).reindex(r.index)
                rec["spearman_margin_vs_real_lap"] = float(pd.Series(d.values).corr(pd.Series(lap.values), method="spearman"))
                if arm == "orig":
                    qq = pd.Series([qf[x] for x in r.index], index=r.index)
                    rec["spearman_margin_vs_qf"] = float(pd.Series(d.values).corr(pd.Series(qq.values), method="spearman"))
                # pooled AUROC within real-sharpness quintiles (content-matched)
                qb = pd.qcut(lap.rank(method="first"), 5, labels=False).values
                rec["pooled_within_lap_quintile"] = [round(auc(r.values[qb == b], t.values[qb == b]), 3) for b in range(5)]
                pl = s[s.kind == "real"].set_index("sha256").pool.reindex(r.index).values
                rec["per_pool_win"] = {p: round(float((d.values[pl == p] > 0).mean()), 3) for p in sorted(set(pl))}
                o.setdefault("scores", {})[key] = rec
(HERE / "c1_cpu.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float)[:20000])
