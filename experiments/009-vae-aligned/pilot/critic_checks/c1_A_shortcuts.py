"""Q1: is A's AUROC a shortcut? Refit A (CPU, cached features), check reproduction, score test WA/FBH reals (unseen
devices), cross-arm zoom probe, per-pool A, Laplacian / QF / JPEG-grid-phase correlations. Output: c1_A_shortcuts.json"""
import json
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
from PIL import Image

res = {}
auc = R.auc
G = R.GENS
for arm in ("orig", "fov512"):
    df, Xr, Xrt, tr, grp = arm_data(arm)
    lab, sp, pool, gen = (df[c].to_numpy() for c in ("label", "split", "pool", "generator"))
    mA = fitA(Xr, Xrt, tr, grp)
    S = np.load(R.CACHE / f"scores6_{arm}.npz")["A"]
    sA = {c: mA.decision_function(Xr[c]) for c in R.CONDS}
    out = dict(chosen_C=float(mA[-1].C_[0]), max_abs_diff_vs_cached=float(max(np.abs(sA[c] - S[i]).max() for i, c in enumerate(R.CONDS))))
    fak = np.where(lab == 1)[0]
    for c in R.CONDS:
        s = sA[c]
        o = {}
        for p in ("vision_native", "vision_whatsapp", "vision_facebook_hq"):
            te = np.where((lab == 0) & (sp == "test") & (pool == p))[0]
            o[p] = dict(macro_auroc_vs_fakes=float(np.mean([auc(s[te], s[fak[gen[fak] == g]]) for g in G])),
                        median_score=float(np.median(s[te])))
        nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
        thr = np.quantile(s[nat], 0.99)
        o["frac_flagged_at_native_1pctFPR"] = {p: float((s[(lab == 0) & (sp == "test") & (pool == p)] > thr).mean())
                                               for p in ("vision_whatsapp", "vision_facebook_hq")}
        o["frac_fakes_flagged_at_native_1pctFPR"] = {g: float((s[fak[gen[fak] == g]] > thr).mean()) for g in G}
        o["median_score_fakes"] = {g: float(np.median(s[fak[gen[fak] == g]])) for g in G}
        # aligned-train separation (in-sample, for scale): reals vs round-trips per pool
        srt = mA.decision_function(Xrt[c])
        o["in_sample_train_auroc_per_pool"] = {p: auc(s[tr][pool[tr] == p], srt[pool[tr] == p]) for p in np.unique(pool[tr])}
        o["in_sample_median_train_real"] = float(np.median(s[tr]))
        o["in_sample_median_train_aligned"] = float(np.median(srt))
        out[c] = o
    # A trained on natives only (no WA/FBH) -> does the signal depend on the compressed pools?
    trn = tr[pool[tr] == "vision_native"]
    mN = fitA(Xr, Xrt, tr, grp, rows=trn)
    nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
    out["A_native_only"] = {c: dict(macro=float(np.mean([auc(mN.decision_function(Xr[c])[nat], mN.decision_function(Xr[c])[fak[gen[fak] == g]]) for g in G])),
                                    per_gen={g: auc(mN.decision_function(Xr[c])[nat], mN.decision_function(Xr[c])[fak[gen[fak] == g]]) for g in G})
                            for c in R.CONDS}
    for p in ("vision_whatsapp", "vision_facebook_hq"):
        trp = tr[pool[tr] == p]
        mP = fitA(Xr, Xrt, tr, grp, rows=trp)
        out[f"A_{p}_only"] = {c: float(np.mean([auc(mP.decision_function(Xr[c])[nat], mP.decision_function(Xr[c])[fak[gen[fak] == g]]) for g in G])) for c in R.CONDS}
    # texture correlation within classes (clean)
    laps = R.lap_table(arm, df)
    for c in R.CONDS:
        L = np.array([laps[(c, s_)] for s_ in df.sha256])
        out[c]["spearman_A_vs_lap_natives"] = float(spearmanr(sA[c][nat], L[nat])[0])
        out[c]["spearman_A_vs_lap_fakes"] = {g: float(spearmanr(sA[c][fak[gen[fak] == g]], L[fak[gen[fak] == g]])[0]) for g in G}
    res[arm] = out
    res[arm]["_mA"] = mA

# zoom cross-arm: orig-arm A applied to fov512 features (same images, zoomed out + resampled) and vice versa
dfo, Xo, _, _, _ = arm_data("orig")
dff, Xf, _, _, _ = arm_data("fov512")
assert (dfo.sha256.to_numpy() == dff.sha256.to_numpy()).all()
lab, sp, pool, gen = (dfo[c].to_numpy() for c in ("label", "split", "pool", "generator"))
nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
fak = np.where(lab == 1)[0]
z = {}
for trained, mA in (("orig", res["orig"]["_mA"]), ("fov512", res["fov512"]["_mA"])):
    for c in R.CONDS:
        so, sf = mA.decision_function(Xo[c]), mA.decision_function(Xf[c])
        z[f"A_{trained}__{c}"] = dict(
            natives_fov512_vs_orig_auroc_fake_high=auc(so[nat], sf[nat]),     # >0.5: zoomed-out natives look more fake
            fakes_fov512_vs_orig_auroc_fake_high=auc(so[fak], sf[fak]),
            macro_orig_feats=float(np.mean([auc(so[nat], so[fak[gen[fak] == g]]) for g in G])),
            macro_fov512_feats=float(np.mean([auc(sf[nat], sf[fak[gen[fak] == g]]) for g in G])),
            macro_nat_orig_vs_fakes_fov512=float(np.mean([auc(so[nat], sf[fak[gen[fak] == g]]) for g in G])),
            macro_nat_fov512_vs_fakes_orig=float(np.mean([auc(sf[nat], so[fak[gen[fak] == g]]) for g in G])))
res["zoom_cross_arm"] = z

# QF and JPEG grid phase (orig arm): test natives, Firefly fakes, train reals; blockiness phase measured on the crop
def phase(path):
    x = np.asarray(Image.open(path).convert("L"), np.float32)
    dh = np.abs(np.diff(x, axis=1)).mean(0)   # 511 col diffs; boundary between col c and c+1
    dv = np.abs(np.diff(x, axis=0)).mean(1)
    ph = lambda d: [float(d[k::8].mean()) for k in range(8)]
    a, b = ph(dh), ph(dv)
    return int(np.argmax(a)), int(np.argmax(b)), float(max(a) / np.median(a)), float(max(b) / np.median(b))

from preprocess import jpeg_quality_estimate
rows = np.r_[nat, fak[gen[fak] == "firefly"], np.where((lab == 0) & (sp == "train"))[0]]
info = []
for i in rows:
    r = dfo.iloc[i]
    with Image.open(r.src_path) as im:
        q = jpeg_quality_estimate(im)
    info.append((i, q, *phase(r.path)))
info = pd.DataFrame(info, columns=["i", "qf", "ph_x", "ph_y", "str_x", "str_y"])
info["grp"] = np.where(lab[info.i] == 1, "firefly", np.where(sp[info.i] == "train", "train_" + pool[info.i], "test_native"))
# aligned fakes (orig arm JPEG round-trips) phase
tro = np.where((lab == 0) & (sp == "train"))[0]
al = [phase(R.rt_path("orig", "sd-vae-ft-mse", dfo.sha256.iloc[i])) for i in tro]
alp = pd.DataFrame(al, columns=["ph_x", "ph_y", "str_x", "str_y"]); alp["grp"] = "aligned_" + pool[tro]
allp = pd.concat([info.drop(columns=["i", "qf"]), alp])
allp["aligned_phase7"] = (allp.ph_x == 7) & (allp.ph_y == 7)
allp["blocky"] = (allp[["str_x", "str_y"]].min(1) > 1.05)
res["jpeg_phase"] = allp.groupby("grp").agg(n=("ph_x", "size"), frac_phase77=("aligned_phase7", "mean"), frac_blocky=("blocky", "mean"),
                                            med_str_x=("str_x", "median"), med_str_y=("str_y", "median")).reset_index().to_dict("records")
res["qf"] = info.groupby("grp").qf.describe().reset_index().to_dict("records")
sAo = res["orig"]["_mA"].decision_function(Xo["clean"])
tn = info[info.grp == "test_native"]
res["spearman_A_vs_qf_test_natives"] = float(spearmanr(sAo[tn.i], tn.qf)[0])
tn = tn.assign(p77=(tn.ph_x == 7) & (tn.ph_y == 7), s=sAo[tn.i])
res["A_by_phase77_test_natives"] = tn.groupby("p77").s.agg(["size", "median"]).reset_index().to_dict("records")
ff = info[info.grp == "firefly"]
ff = ff.assign(p77=(ff.ph_x == 7) & (ff.ph_y == 7), s=sAo[ff.i])
res["A_by_phase77_firefly"] = ff.groupby("p77").s.agg(["size", "median"]).reset_index().to_dict("records")
for arm in ("orig", "fov512"):
    res[arm].pop("_mA")
(OUT / "c1_A_shortcuts.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float)[:200])
