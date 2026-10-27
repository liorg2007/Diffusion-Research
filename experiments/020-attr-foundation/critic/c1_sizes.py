"""Critic scratch: native size / scale-factor confound for H-A1 (reads committed OOF + M0 cache; no training)."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]; ROOT = H.parents[1]
CL = ["dalle2","dalle3","firefly","midjourney-v5","stable-diffusion-2","stable-diffusion-xl","sd3_large","flux"]
fo = json.load(open(H/"results/folds_ha1_main.json")); shas = fo["rows"]
m0 = json.load(open(H/"cache/m0_fov512.json")); M = dict(zip(m0["shas"], m0["rows"]))
df = pd.read_csv(ROOT/"experiments/019-source-diversity/splits_fov512.csv", keep_default_na=False).set_index("sha256")
gen = df.loc[shas, "generator"].to_numpy()
w = np.array([M[s]["w"] for s in shas]); h = np.array([M[s]["h"] for s in shas])
short = np.minimum(w, h); scale = 512/short; ar = np.maximum(w,h)/short
out = {}
for reg in ["tmix", "clean"]:
    z = np.load(H/f"results/oof_ha1_main_{reg}_nested_jpeg75_resize05.npz"); lp, y = z["logp"], z["y"]
    assert (np.array(CL)[y] == gen).all()
    out[reg] = (lp, y)
lp, y = out["tmix"]; pred = lp.argmax(1)
print("== native sizes per generator (w x h : n), short side, scale=512/short")
for k, g in enumerate(CL):
    m = y == k
    c = Counter(zip(w[m], h[m]))
    print(f"{g:22s} nsizes={len(c):2d} short={sorted(Counter(short[m]).items())[:8]} top={c.most_common(4)}")
def bacc(yt, yp, classes):
    return np.mean([np.mean(yp[yt == c] == c) for c in classes if (yt == c).any()])
print("\n== OOF T-mix nested @ jpeg75_resize05 stratified")
for reg in ["tmix","clean"]:
    lp, y = out[reg]; pred = lp.argmax(1)
    print(f"[{reg}] overall bacc {bacc(y,pred,range(8)):.3f}")
    sq = (w == 1024) & (h == 1024)
    print(f"  native 1024x1024 rows per class: {dict(Counter(np.array(CL)[y[sq]]))}")
    cls_sq = [k for k in range(8) if (y[sq]==k).sum() >= 10]
    print(f"  1024^2 subset, 8-way argmax, bacc over classes present(n>=10): {bacc(y[sq], pred[sq], cls_sq):.3f}  classes={[CL[k] for k in cls_sq]}")
    sq = sq & np.isin(y, cls_sq); r = lp[sq][:, cls_sq].argmax(1); rp = np.array(cls_sq)[r]
    print(f"  1024^2 subset, argmax restricted to those classes: {bacc(y[sq], rp, cls_sq):.3f}")
    three = [0, 6, 7]; m3 = np.isin(y, three)
    rp3 = np.array(three)[lp[m3][:, three].argmax(1)]
    print(f"  dalle2/sd3/flux restricted argmax (from 8-way OOF): {bacc(y[m3], rp3, three):.3f}; 8-way recall {[round(np.mean(pred[y==k]==k),3) for k in three]}")
    # within SD2 / SDXL: recall by short-side bucket
    for k in [4,5,2,3,1]:
        m = y == k
        rows = []
        for s_ in sorted(set(short[m])):
            mm = m & (short == s_)
            if mm.sum() >= 5: rows.append(f"{s_}:{mm.sum()}:{np.mean(pred[mm]==k):.2f}")
        print(f"  {CL[k]} recall by short side (short:n:recall): {' '.join(rows)}")
    # SD2 vs SDXL binary restricted, within identical native size buckets shared by both
    m2 = np.isin(y, [4,5]); rp = np.array([4,5])[lp[:, [4,5]].argmax(1)]
    sz = np.array([f"{a}x{b}" for a,b in zip(w,h)])
    shared = set(sz[y==4]) & set(sz[y==5])
    ms = m2 & np.isin(sz, list(shared))
    print(f"  SD2 vs SDXL restricted: all {bacc(y[m2], rp[m2], [4,5]):.3f} (n={m2.sum()}); shared-size rows only {bacc(y[ms], rp[ms], [4,5]):.3f} (n={ms.sum()}, {len(shared)} shared sizes)")
    # same-scale-factor groups: does prediction collapse onto generators with the same native size?
    wrong = pred != y
    same_sz = np.array([ (sz[i] in set(sz[y==pred[i]])) for i in range(len(y))])
    print(f"  errors: {wrong.sum()}; share of errors where predicted class has an image of identical native size: {same_sz[wrong].mean():.2f}")
