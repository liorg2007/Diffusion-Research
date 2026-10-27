"""Critic scratch: fresh size-matched probes + scale decodability + unmatched-pair 2-AFC. CPU, fixed C (no tuning on test), committed folds."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]; ROOT = H.parents[1]
sys.path.insert(0, str(ROOT/"experiments/004-phase4-features"))
import p4features as P
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
CL = ["dalle2","dalle3","firefly","midjourney-v5","stable-diffusion-2","stable-diffusion-xl","sd3_large","flux"]
fo = json.load(open(H/"results/folds_ha1_main.json")); shas = fo["rows"]; fold = np.array(fo["fold"])
m0 = json.load(open(H/"cache/m0_fov512.json")); M = dict(zip(m0["shas"], m0["rows"]))
df = pd.read_csv(ROOT/"experiments/019-source-diversity/splits_fov512.csv", keep_default_na=False).set_index("sha256")
y = np.array([CL.index(g) for g in df.loc[shas, "generator"]])
fam = np.array([os.path.basename(p).split(".")[0] for p in df.loc[shas, "src_path"]])
w = np.array([M[s]["w"] for s in shas]); h = np.array([M[s]["h"] for s in shas]); short = np.minimum(w, h)
CONDS = ["clean", "jpeg75_resize05", "social_sim"]
X = {}
for c in CONDS:
    k, x = P.read_shards(c, ROOT/"experiments/004-phase4-features/cache/fov512"); st = dict(zip(k.tolist(), x))
    X[c] = np.stack([st[s] for s in shas]).astype(np.float32)
SL = {"b4": slice(0,1536), "b8": slice(1536,3072), "cat": slice(0,4608)}
rng = np.random.default_rng(0)
def bacc(yt, yp):
    return np.mean([np.mean(yp[yt == c] == c) for c in np.unique(yt)])
def boot(yt, yp, g, n=1000):
    ug, inv = np.unique(g, return_inverse=True); vals = []
    for _ in range(n):
        wts = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv]
        vals.append(np.mean([np.sum(wts*(yt==c)*(yp==c))/max(np.sum(wts*(yt==c)),1e-9) for c in np.unique(yt)]))
    return np.percentile(vals, [2.5, 97.5])
def oof(rows, lab, fs, C=0.01, train=("clean","jpeg75_resize05","social_sim"), test="jpeg75_resize05"):
    pred = np.full(len(rows), -1)
    for f in range(5):
        tr, te = rows[fold[rows] != f], rows[fold[rows] == f]
        ltr = lab[fold[rows] != f]
        Xtr = np.concatenate([X[c][tr][:, SL[fs]] for c in train]); ytr = np.tile(ltr, len(train))
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        pred[fold[rows] == f] = m.predict(sc.transform(X[test][te][:, SL[fs]]))
    return pred
def report(name, rows, lab, fss=("b4","b8","cat")):
    for fs in fss:
        for reg, tr in [("tmix", ("clean","jpeg75_resize05","social_sim")), ("clean-train", ("clean",))]:
            p = oof(rows, lab, fs, train=tr)
            lo, hi = boot(lab, p, fam[rows])
            print(f"{name:52s} {fs:4s} {reg:11s} bacc@jpeg75_resize05 {bacc(lab,p):.3f} [{lo:.3f},{hi:.3f}] chance {1/len(np.unique(lab)):.3f} n={len(rows)}")
# A. 3-way dalle2/sd3/flux (all native 1024x1024)
r = np.where(np.isin(y, [0,6,7]))[0]; report("A 3-way dalle2/sd3/flux (all 1024^2)", r, y[r])
# B. 4-way 1024^2-only (incl. square dalle3)
r = np.where(np.isin(y, [0,1,6,7]) & (w==1024) & (h==1024))[0]; report("B 4-way native-1024^2 only (dalle2/dalle3sq/sd3/flux)", r, y[r], ("cat",))
# C. SD2 vs SDXL (identical 26-size sets)
r = np.where(np.isin(y, [4,5]))[0]; report("C SD2 vs SDXL (same 26 native sizes)", r, y[r], ("b4","cat"))
# D. scale decodability within SD2+SDXL: short side <=704 vs >=896 (resample factor >=0.73 vs <=0.57)
r = np.where(np.isin(y, [4,5]) & ((short <= 704) | (short >= 896)))[0]; lab = (short[r] >= 896).astype(int)
report("D scale bucket (short<=704 vs >=896) within SD2+SDXL", r, lab, ("b4","cat"))
# D2. cross-generator scale: train on SD2 rows, test on SDXL rows (clean only, quick)
for fs in ("b4","cat"):
    for c in ("clean","jpeg75_resize05"):
        a = np.where((y==4) & ((short<=704)|(short>=896)))[0]; b = np.where((y==5) & ((short<=704)|(short>=896)))[0]
        sc = StandardScaler().fit(X[c][a][:, SL[fs]]); m = LogisticRegression(C=0.01, max_iter=300, class_weight="balanced").fit(sc.transform(X[c][a][:, SL[fs]]), (short[a]>=896).astype(int))
        s = m.decision_function(sc.transform(X[c][b][:, SL[fs]]))
        print(f"D2 scale probe SD2->SDXL {fs} {c}: AUROC {roc_auc_score((short[b]>=896).astype(int), s):.3f} (n={len(b)})")
# E. 2-AFC on UNMATCHED pairs (same fold, different families) vs matched, from committed OOF logp
for reg in ("tmix","clean"):
    lp = np.load(H/f"results/oof_ha1_main_{reg}_nested_jpeg75_resize05.npz")["logp"]
    accs_u, accs_m, npairs = [], [], {}
    for A in range(8):
        for B in range(A+1, 8):
            ia, ib = np.where(y==A)[0], np.where(y==B)[0]
            ma = lp[ia, A]-lp[ia, B]; mb = lp[ib, A]-lp[ib, B]
            accs_u.append(np.mean((ma[:,None] > mb[None,:]) + 0.5*(ma[:,None]==mb[None,:])))   # all A x B pairs = binary AUROC on the A-vs-B logit
            fa = {f:i for i,f in zip(ia, fam[ia])}; mt = [(fa[fam[j]], j) for j in ib if fam[j] in fa]
            npairs[f"{CL[A]}|{CL[B]}"] = len(mt)
            if mt:
                I, J = np.array(mt).T; accs_m.append(np.mean((lp[I,A]-lp[I,B]) > (lp[J,A]-lp[J,B])))
    print(f"E [{reg}] 2-AFC macro: matched {np.mean(accs_m):.3f}  ALL cross-family pairs (pairwise AUROC) {np.mean(accs_u):.3f}  min unmatched {np.min(accs_u):.3f}")
print("E matched pairs per type:", {k:v for k,v in npairs.items()}, "total", sum(npairs.values()))
# F. M0-style size-only baseline restricted to 1024^2 classes (should be chance)
