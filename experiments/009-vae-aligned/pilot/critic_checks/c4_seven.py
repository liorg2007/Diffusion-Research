"""Q3: #7 under texture control. orig + fov512, both conditions: Spearman of log e1 / log ratio with log Laplacian,
within-Laplacian-quintile AUROC (equal-weight mean over usable strata), log-Lap-matched AUROC, and AUROC of a texture-
residualized single-pass score (log e1 regressed on log lap over ALL rows of both classes, so no label is used).
Output: c4_seven.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
out = {}
for arm in ("orig", "fov512"):
    df = R.load_df(arm)
    laps = R.lap_table(arm, df)
    sel = R.rows7(df, False)
    nat7 = sel[sel.label == 0]; fak7 = sel[sel.label == 1]
    for c in R.CONDS:
        z = np.load(R.CACHE / "r7" / f"{arm}__{c}__sd-vae-ft-mse.npz")
        pos = {s: i for i, s in enumerate(z["sha"].tolist())}
        o = {}
        rows = pd.concat([nat7, fak7])
        e1 = z["e1"][[pos[s] for s in rows.sha256]]; e2 = z["e2"][[pos[s] for s in rows.sha256]]
        L = np.array([laps[(c, s)] for s in rows.sha256]); y = rows.label.to_numpy(); g = rows.generator.to_numpy()
        le1, lr, ll, le2 = np.log(e1), np.log(e1 / e2), np.log(L), np.log(e2)
        o["spearman_logLap"] = dict(e1=float(spearmanr(le1, ll)[0]), e2=float(spearmanr(le2, ll)[0]), ratio=float(spearmanr(lr, ll)[0]))
        o["spearman_logLap_natives_only"] = dict(e1=float(spearmanr(le1[y == 0], ll[y == 0])[0]), ratio=float(spearmanr(lr[y == 0], ll[y == 0])[0]))
        beta = np.polyfit(ll, le1, 1); res1 = le1 - np.polyval(beta, ll)
        betar = np.polyfit(ll, lr, 1); resr = lr - np.polyval(betar, ll)
        G = R.GENS
        mac = lambda s: float(np.mean([R.auc(s[y == 0], s[(y == 1) & (g == k)]) for k in G]))
        o["macro"] = dict(single=mac(-le1), ratio=mac(-lr), texture_fake_high=mac(ll), single_resid=mac(-res1), ratio_resid=mac(-resr))
        o["per_gen_single_resid"] = {k: R.auc(-res1[y == 0], -res1[(y == 1) & (g == k)]) for k in G}
        o["per_gen_ratio_resid"] = {k: R.auc(-resr[y == 0], -resr[(y == 1) & (g == k)]) for k in G}
        for nm, s in (("single", -le1), ("ratio", -lr)):
            o[f"strata_{nm}"] = R.strata_auc(s, L, y)["mean_auroc_usable"]
            o[f"matched_{nm}"] = R.matched_auc(s[y == 0], L[y == 0], s[y == 1], L[y == 1])
        o["median_lap"] = dict(nat=float(np.median(L[y == 0])), fak={k: float(np.median(L[g == k])) for k in G})
        o["median_e1"] = dict(nat=float(np.median(e1[y == 0])), fak={k: float(np.median(e1[g == k])) for k in G})
        o["median_e2"] = dict(nat=float(np.median(e2[y == 0])), fak={k: float(np.median(e2[g == k])) for k in G})
        o["median_ratio"] = dict(nat=float(np.median((e1 / e2)[y == 0])), fak={k: float(np.median((e1 / e2)[g == k])) for k in G})
        out[f"{arm}/{c}"] = o
(OUT / "c4_seven.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, json.dumps({kk: v[kk] for kk in ("spearman_logLap", "macro", "strata_single", "strata_ratio")}))
    print("   matched", v["matched_single"], v["matched_ratio"]); print("   ratio_resid", {a: round(b, 3) for a, b in v["per_gen_ratio_resid"].items()})
    print("   med ratio", v["median_ratio"]); print("   med e1", v["median_e1"])
