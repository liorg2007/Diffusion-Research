"""c4 (post-hoc): do simple image attributes explain the held-out false alarms (esp. UNS)? k=5 held-out scores from results/scores_long.csv.gz
(jpeg75_resize05; reals scored by the 'full' model, fakes by their LOGO model). Attributes on the CLEAN fov512 crop the model sees:
saturation (HSV S mean), colourfulness (Hasler-Suesstrunk), brightness, RMS contrast, sharpness (log var of Laplacian), flat fraction
(share of 16x16 tiles with luma std < 4: bokeh / sky / studio backdrop), plus for the new sources the fetched-file bpp, aspect and portrait flag."""
import sys, json
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from PIL import Image
from scipy.stats import spearmanr, mannwhitneyu
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[3]
E10 = ROOT / "experiments" / "019-source-diversity"
def attrs(p):
    im = np.asarray(Image.open(ROOT / p).convert("RGB"), np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx, mn = im.max(-1), im.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()
    rg, yb = r - g, 0.5 * (r + g) - b
    colf = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    t = y.reshape(32, 16, 32, 16).std(axis=(1, 3))
    return dict(sat=float(sat), colorful=float(colf), bright=float(y.mean()), contrast=float(y.std()), sharp=float(np.log(lap.var() + 1e-6)),
                flat=float((t < 4).mean()))
if __name__ == "__main__":
    sl = pd.read_csv(E10 / "results" / "scores_long.csv.gz")
    sl = sl[(sl.cond == "jpeg75_resize05") & (sl.role == "test")]
    sp = pd.read_csv(E10 / "splits_fov512.csv", keep_default_na=False, dtype=str).set_index("sha256")
    pool_of = {"VISION": ("vision_native", "vision_whatsapp", "vision_facebook_hq"), "RAISE": ("raise",), "OI": ("openimages",), "UNS": ("unsplash",), "WMC": ("commons",), "HDRP": ("hdrplus",)}
    res = json.load(open(E10 / "results" / "results.json"))["result"]["heldout"]
    rows = []
    for S, pools in pool_of.items():
        d = sl[(sl.heldout == S)]
        T = d.train_sources.iloc[0]
        thr = res[S][T]["F_b8"]["conds"]["jpeg75_resize05"]["fpr"]["A"]["full"]["thr"]
        re = d[(d.label == 0) & d.pool.isin(pools)].assign(S=S, thr=thr)
        rows.append(re)
        if S == "UNS":
            rows.append(d[d.label == 1].assign(S="FAKE", thr=thr))
    A = pd.concat(rows, ignore_index=True)
    A["path"] = sp.loc[A.sha256, "path"].to_numpy()
    with ProcessPoolExecutor(12) as ex:
        at = list(ex.map(attrs, A.path, chunksize=32))
    A = pd.concat([A.reset_index(drop=True), pd.DataFrame(at)], axis=1)
    A["fp"] = (A.score >= A.thr) & (A.label == 0)
    keys = ["sat", "colorful", "bright", "contrast", "sharp", "flat"]
    out = {}
    print("medians by group (reals: held-out test reals at k=5; FAKE = all test fakes)")
    print(A.groupby("S")[keys + ["score"]].median().round(3).to_string())
    print("\nSpearman(score, attribute) within each held-out source's reals; and FP vs non-FP medians (MWU p)")
    for S in list(pool_of):
        d = A[A.S == S]
        out[S] = {}
        line = [f"{S:6s} n={len(d):4d} FP={int(d.fp.sum()):3d}"]
        for k in keys:
            rho = spearmanr(d.score, d[k]).statistic
            p = mannwhitneyu(d[d.fp][k], d[~d.fp][k]).pvalue if d.fp.sum() >= 3 else np.nan
            out[S][k] = dict(rho=float(rho), fp_med=float(d[d.fp][k].median()) if d.fp.any() else None, nonfp_med=float(d[~d.fp][k].median()), mwu_p=float(p))
            line.append(f"{k}:rho={rho:+.2f},p={p:.3f}")
        print(" ".join(line))
    # new-source file attributes
    for S, man, pool in (("UNS", "unsplash_manifest.csv", "unsplash"), ("WMC", "commons_manifest.csv", "commons"), ("HDRP", "hdrplus_manifest.csv", "hdrplus")):
        m = pd.read_csv(E10 / man, dtype={"sha256": str}).set_index("sha256")
        d = A[A.S == S].copy()
        mm = m.loc[d.sha256]
        d["bpp"] = (mm.bytes.to_numpy() * 8) / (mm.width.to_numpy() * mm.height.to_numpy())
        d["aspect"] = np.maximum(mm.width, mm.height).to_numpy() / np.minimum(mm.width, mm.height).to_numpy()
        d["portrait"] = (mm.height.to_numpy() > mm.width.to_numpy())
        print(f"\n{S}: Spearman(score,bpp)={spearmanr(d.score, d.bpp).statistic:+.2f}  (score,aspect)={spearmanr(d.score, d.aspect).statistic:+.2f}; "
              f"FPR portrait {d[d.portrait].fp.mean():.3f} (n={int(d.portrait.sum())}) vs landscape {d[~d.portrait].fp.mean():.3f} (n={int((~d.portrait).sum())})")
        for q, lab in ((d.bpp, "bpp"),):
            t = pd.qcut(q, 3, labels=["lo", "mid", "hi"])
            print(f"  FPR by {lab} tertile:", d.groupby(t, observed=True).fp.mean().round(3).to_dict())
        out[S + "_file"] = dict(rho_bpp=float(spearmanr(d.score, d.bpp).statistic), fpr_portrait=float(d[d.portrait].fp.mean()), fpr_landscape=float(d[~d.portrait].fp.mean()),
                                n_portrait=int(d.portrait.sum()))
    # can the six attributes alone separate UNS reals from fakes, and do they predict UNS FP? (5-fold CV logistic, descriptive)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    for S in pool_of:
        d = A[A.S == S]
        if d.fp.sum() >= 5:
            p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), d[keys], d.fp, cv=5, method="predict_proba")[:, 1]
            out[S]["attr_auc_fp"] = float(roc_auc_score(d.fp, p))
            print(f"{S}: CV AUROC of the 6 attributes for predicting FP among reals = {out[S]['attr_auc_fp']:.3f}")
    json.dump(out, open(Path(__file__).resolve().parent / "c4_attributes.json", "w"), indent=1)
    A.drop(columns=["path"]).to_csv(Path(__file__).resolve().parent / "c4_attributes_rows.csv.gz", index=False)
