"""Critic check c2: anatomy of the held-out OI (and RAISE) false positives, fov512, F_b8, train = all other sources, 'full' model.
Per OI test image: raw size, PIL mode, bytes/pixel, estimated JPEG quality (luma quant table), EXIF presence, score at clean / jpeg75_resize05.
Also: FLUX AUROC recomputed (full model scores both classes), alternative threshold readings, and montages of top-FP vs lowest OI reals."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, spearmanr

H = Path(__file__).resolve().parents[2]
ROOT = H.parents[1]
OUT = Path(__file__).resolve().parent
s = pd.read_csv(H / "results/scores_long.csv.gz")
m = pd.read_csv(H / "openimages_manifest.csv")
res = {}

def thr1(r, f=0.01):
    r = np.sort(r)[::-1]; k = int(np.floor(f * len(r)))
    return np.nextafter(r[k], np.inf)

STD_LUMA = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,
                     24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)

def jpeg_q(im):
    try:
        q = np.array(im.quantization[0], float)  # zigzag order in PIL; compare sums (order-free)
    except Exception:
        return np.nan
    scale = q.sum() / STD_LUMA.sum() * 100
    return float(200 - scale) / 2 if scale <= 100 else float(5000 / scale)

rows = []
for r in m.itertuples():
    im = Image.open(ROOT / r.path)
    exif = im.getexif()
    rows.append(dict(sha256=r.sha256, image_id=r.image_id, w=im.width, h=im.height, mode=im.mode, bpp=r.bytes / (im.width * im.height),
                     q=jpeg_q(im), exif_n=len(exif), make=str(exif.get(271, "")), model=str(exif.get(272, "")), soft=str(exif.get(305, ""))))
meta = pd.DataFrame(rows)
res["oi_raw"] = dict(modes=meta["mode"].value_counts().to_dict(), long_side=meta[["w", "h"]].max(1).value_counts().head(5).to_dict(),
                     short_side_q=meta[["w", "h"]].min(1).quantile([0, .5, 1]).to_dict(), q_quantiles=meta.q.quantile([.05, .25, .5, .75, .95]).round(1).to_dict(),
                     exif_frac=float((meta.exif_n > 0).mean()), with_make=float((meta.make != "").mean()))
print("OI raw:", res["oi_raw"])

a = s[(s.arm == "fov512") & (s.heldout == "OI") & (s.feat == "F_b8")]
pv = a[a.label == 0].pivot_table(index=["sha256", "pool", "role"], columns="cond", values="score").reset_index()
cal = pv[(pv.role == "calib") & pv.pool.isin(["vision_native", "vision_whatsapp", "vision_facebook_hq", "raise"])]
tA = thr1(cal.clean.values)
oi = pv[(pv.pool == "openimages")].merge(meta, on="sha256")
t = oi[oi.role == "test"].copy()
t["fp"] = t.jpeg75_resize05 >= tA
t["fp_clean"] = t.clean >= tA
print("thrA", tA, "OI test FPR jr", t.fp.mean(), "clean", t.fp_clean.mean())
# characteristics FP vs not
for col in ["bpp", "q", "w", "h"]:
    a1, a0 = t[t.fp][col].dropna(), t[~t.fp][col].dropna()
    print(f"{col}: FP median {a1.median():.3f} vs nonFP {a0.median():.3f}; MWU p={mannwhitneyu(a1, a0).pvalue:.3g}; spearman(score)={spearmanr(t[col], t.jpeg75_resize05, nan_policy='omit')[0]:.3f}")
g = t.groupby(t["mode"]).agg(n=("fp", "size"), fp=("fp", "sum"), fpc=("fp_clean", "sum"))
print("by mode (test):\n", g)
allL = oi[oi["mode"] == "L"][["image_id", "role", "clean", "jpeg75_resize05"]]
print("all L-mode OI images (any role):\n", allL.round(2).to_string())
print("portrait vs landscape FPR:", t.groupby(t.h > t.w).fp.agg(["size", "mean"]).to_dict())
print("exif present FPR:", t.groupby(t.exif_n > 0).fp.agg(["size", "mean"]).to_dict())
res["L_mode"] = dict(n_test=int(g.loc["L", "n"]) if "L" in g.index else 0, fp_test=int(g.loc["L", "fp"]) if "L" in g.index else 0,
                     all_roles=allL.round(3).to_dict("records"))
res["fp_vs_nonfp"] = {c: [float(t[t.fp][c].median()), float(t[~t.fp][c].median())] for c in ["bpp", "q"]}
# excluding grayscale: does the kill survive?
tn = t[t["mode"] != "L"]
res["fpr_excl_L"] = float(tn.fp.mean()); print("OI FPR excluding L-mode:", res["fpr_excl_L"], len(tn))
# OI scores vs training reals
print("score quantiles at jr: OI test", t.jpeg75_resize05.quantile([.5, .9, .99]).round(2).to_dict(),
      "| calib VISION", cal[cal.pool != "raise"].jpeg75_resize05.quantile([.5, .9, .99]).round(2).to_dict(),
      "| calib RAISE", cal[cal.pool == "raise"].jpeg75_resize05.quantile([.5, .9, .99]).round(2).to_dict())
# alternative thresholds (OI held out)
fk = a[(a.label == 1) & (a.role == "test") & (a.cond == "jpeg75_resize05")]
alts = {"A pooled (pre-reg)": tA,
        "RAISE calib only": thr1(cal[cal.pool == "raise"].clean.values),
        "VISION calib only": thr1(cal[cal.pool != "raise"].clean.values),
        "max of per-source": max(thr1(cal[cal.pool == "raise"].clean.values), thr1(cal[cal.pool != "raise"].clean.values)),
        "B same-cond pooled": thr1(cal.jpeg75_resize05.values)}
res["alt_thr"] = {}
for k, v in alts.items():
    res["alt_thr"][k] = dict(thr=float(v), oi_fpr=float((t.jpeg75_resize05 >= v).mean()), tpr_fakes=float((fk.score >= v).mean()))
    print("alt thr", k, res["alt_thr"][k])
q90 = np.quantile(t.jpeg75_resize05, 0.90)
res["thr_for_oi_10pct"] = dict(thr=float(q90), tpr=float((fk.score >= q90).mean()))
print("threshold that would give OI 10% FPR:", res["thr_for_oi_10pct"])
# FLUX AUROC recompute (full model both classes)
for S in ["VISION", "RAISE", "OI"]:
    b = s[(s.arm == "fov512") & (s.heldout == S) & (s.feat == "F_b8") & (s.cond == "jpeg75_resize05") & (s.role == "test")]
    pools = {"VISION": ["vision_native", "vision_whatsapp", "vision_facebook_hq"], "RAISE": ["raise"], "OI": ["openimages"]}[S]
    R = b[(b.label == 0) & b.pool.isin(pools)].score.values; F = b[b.generator == "flux"].score.values
    print("FLUX AUROC", S, round(roc_auc_score(np.r_[np.zeros(len(R)), np.ones(len(F))], np.r_[R, F]), 4))
# RAISE held out, same anatomy (scale): RAISE test FP
ar = s[(s.arm == "fov512") & (s.heldout == "RAISE") & (s.feat == "F_b8") & (s.label == 0)].pivot_table(index=["sha256", "pool", "role"], columns="cond", values="score").reset_index()
calr = ar[(ar.role == "calib") & (ar.pool != "raise")]
tr_ = thr1(calr.clean.values)
rt = ar[(ar.role == "test") & (ar.pool == "raise")]
print("RAISE held out: FPR jr", (rt.jpeg75_resize05 >= tr_).mean(), "clean", (rt.clean >= tr_).mean())
# montages: top-12 OI FP, 12 lowest OI test, from fov512 crops
sp = pd.read_csv(H / "splits_fov512.csv", keep_default_na=False).set_index("sha256")
def montage(shas, fn):
    ims = [Image.open(ROOT / sp.loc[x, "path"]).convert("RGB").resize((192, 192)) for x in shas]
    W = Image.new("RGB", (192 * 6, 192 * ((len(ims) + 5) // 6)), "white")
    for i, im in enumerate(ims):
        W.paste(im, (192 * (i % 6), 192 * (i // 6)))
    W.save(OUT / fn)
ts = t.sort_values("jpeg75_resize05", ascending=False)
montage(ts.sha256.head(18).tolist(), "c2_oi_top18_fp.jpg")
montage(ts.sha256.tail(12).tolist(), "c2_oi_lowest12.jpg")
ts[["image_id", "mode", "w", "h", "bpp", "q", "make", "soft", "clean", "jpeg75_resize05", "fp"]].head(32).round(3).to_csv(OUT / "c2_oi_top32.csv", index=False)
print(ts[["image_id", "mode", "w", "h", "bpp", "q", "make", "soft", "clean", "jpeg75_resize05"]].head(32).round(2).to_string())
json.dump(res, open(OUT / "c2_oi_fp_anatomy.json", "w"), indent=1, default=float)

# appended: FPR by JPEG-quality bin and bpp tercile (OI test)
t['qbin'] = pd.cut(t.q, [0, 85, 95, 98, 101])
print("FPR by JPEG quality bin:\n", t.groupby('qbin', observed=True).fp.agg(['size', 'sum', 'mean']).round(3))
t['bppbin'] = pd.qcut(t.bpp, 3)
print("FPR by bpp tercile:\n", t.groupby('bppbin', observed=True).fp.agg(['size', 'sum', 'mean']).round(3))
