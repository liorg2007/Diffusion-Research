"""Critic check 1 (CPU, saved scores only): recompute AUROCs, per-class shifts, cross-class decomposition of the drop
and the canon cost, fixed-FPR operating points, CI widths. Usage: env/.venv/bin/python -I c1_scores.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

H = Path(__file__).resolve().parent
P = H.parent
df = pd.read_csv(P / "scores_long.csv.gz", dtype={"id": str})
R = json.loads((P / "results.json").read_text())["results"]
DETS = ["b0", "communityforensics", "rine", "clipdet"]
SETS = ["clean", "clean_canon", "eps8", "eps8_canon", "eps16", "eps16_canon"]


def auc(sr, sf):
    sr, sf = np.asarray(sr, float), np.asarray(sf, float)
    rk = rankdata(np.r_[sr, sf])
    return (rk[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf))


def get(det, s, cls=None, padded=None):
    d = df[(df.detector == det) & (df.input_set == s)]
    if cls == "real":
        d = d[d.cls == "real"]
    elif cls == "fake":
        d = d[d.cls != "real"]
    elif cls:
        d = d[d.cls == cls]
    if padded is not None:
        d = d[d.padded == padded]
    return d.sort_values(["cls", "id"])


out = {}
print("== 1. recompute pooled AUROC primary (unpadded reals) and compare with results.json")
maxdiff = 0
for det in DETS:
    for s in SETS:
        a = auc(get(det, s, "real", False).score, get(det, s, "fake").score)
        b = R[det]["primary"]["auroc"][s]["pooled"]["value"]
        a2 = auc(get(det, s, "real").score, get(det, s, "fake").score)
        b2 = R[det]["sensitivity_all"]["auroc"][s]["pooled"]["value"]
        maxdiff = max(maxdiff, abs(a - b), abs(a2 - b2))
print("max |recomputed - reported| =", maxdiff)
print("fake padded count:", int(df[(df.cls != "real")].padded.sum()), " real padded ids:", df[(df.cls == "real") & df.padded].id.nunique())

print("\n== 2. per-class median score and median paired shift (clean->eps8, clean->clean_canon, eps8->eps8_canon); primary reals")
rows = []
for det in DETS:
    for grp in ["real_unpad", "real_pad", "gen_0", "gen_1", "gen_2", "gen_3"]:
        cls = "real" if grp.startswith("real") else grp
        pad = None if cls != "real" else (grp == "real_pad")
        v = {s: get(det, s, cls, pad).score.to_numpy() for s in SETS}
        r = dict(det=det, grp=grp, n=len(v["clean"]), med_clean=np.median(v["clean"]), med_eps8=np.median(v["eps8"]),
                 med_cc=np.median(v["clean_canon"]), med_e8c=np.median(v["eps8_canon"]),
                 shift_attack=np.median(v["eps8"] - v["clean"]), shift_canon_clean=np.median(v["clean_canon"] - v["clean"]),
                 shift_canon_att=np.median(v["eps8_canon"] - v["eps8"]),
                 p_attack=wilcoxon(v["eps8"] - v["clean"]).pvalue)
        rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
out["per_class"] = t.round(4).to_dict("records")

print("\n== 3. cross-class decomposition (primary). AUROC(reals from set A, fakes from set B)")
dec = []
for det in DETS:
    for A, B, lab in [("clean", "clean", "clean"), ("eps8", "clean", "attack reals only"), ("clean", "eps8", "attack fakes only"),
                      ("eps8", "eps8", "both attacked"), ("clean_canon", "clean", "canon reals only"),
                      ("clean", "clean_canon", "canon fakes only"), ("clean_canon", "clean_canon", "both canon"),
                      ("eps8_canon", "eps8", "canon att reals only"), ("eps8", "eps8_canon", "canon att fakes only"),
                      ("eps8_canon", "eps8_canon", "both att+canon")]:
        a = auc(get(det, A, "real", False).score, get(det, B, "fake").score)
        dec.append(dict(det=det, reals=A, fakes=B, what=lab, auroc=round(a, 3)))
d = pd.DataFrame(dec).pivot(index="what", columns="det", values="auroc")
print(d[DETS].to_string())
out["decomposition"] = dec

print("\n== 4. fixed-FPR operating points: threshold set on CLEAN unpadded reals at FPR 5% / 1% (from the 48), "
      "then FPR on attacked reals and TPR on clean/attacked fakes (descriptive; 48 reals -> 1% = 0 or 1 real)")
op = []
for det in DETS:
    rc = get(det, "clean", "real", False).score.to_numpy()
    for fpr in (0.05, 0.10):
        thr = np.quantile(rc, 1 - fpr)
        r = dict(det=det, target_fpr=fpr, thr=round(thr, 3))
        for s in ["clean", "eps8", "eps16", "clean_canon", "eps8_canon"]:
            r[f"FPR_{s}"] = round(float((get(det, s, "real", False).score.to_numpy() > thr).mean()), 3)
            r[f"TPR_{s}"] = round(float((get(det, s, "fake").score.to_numpy() > thr).mean()), 3)
        op.append(r)
o = pd.DataFrame(op)
print(o.to_string(index=False))
out["fixed_fpr"] = op

print("\n== 5. fraction of items whose score moved toward the other class under eps8 (sign of shift)")
for det in DETS:
    r = get(det, "eps8", "real", False).score.to_numpy() - get(det, "clean", "real", False).score.to_numpy()
    f = get(det, "eps8", "fake").score.to_numpy() - get(det, "clean", "fake").score.to_numpy()
    print(f"{det:20s} reals up(=more fake) {np.mean(r > 0):.2f}  fakes down(=more real) {np.mean(f < 0):.2f}")

print("\n== 6. CI widths, primary vs sensitivity (eps8 drop/recovery/cost)")
for det in DETS:
    for k in ("primary", "sensitivity_all"):
        e = R[det][k]["eps8"]
        w = lambda x: None if x is None else round(x[1] - x[0], 3)
        print(f"{det:20s} {k:16s} cleanCIw {w(R[det][k]['auroc']['clean']['pooled']['ci'])} dropCIw {w(e['drop']['ci'])} "
              f"recCIw {w(e['recovery']['ci'])} costCIw {w(e['canon_clean_cost']['ci'])}")

print("\n== 7. B0: correlation of clean score with original size (reals) and per-class mean")
items = pd.read_csv(P.parent / "raid_manifest.csv", dtype={"id": str})
print(items.columns.tolist())
(H / "c1_out.json").write_text(json.dumps(out, indent=1, default=float))
