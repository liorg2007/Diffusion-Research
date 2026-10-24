"""H-D2 critic check 4 (post hoc, exploratory, CPU): both SD fakes and JPEG-coded natives carry an 8-px grid (c1). Can the SHAPE of the
grid (decoder vs JPEG blocking) separate them, where its energy cannot? Feature = |DFT| of the 8x8 phase-folded median residual, divided
by std(r) (phase-invariant: 36 unique harmonics, DC dropped) + log z8. Logistic regression (standardised, C=1), 5-fold CV grouped by
device for natives / by image for fakes. Clean orig crops only; test natives vs SD2+SDXL, and vs all 6 generators. -> c4_gridshape.json"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
IDX = sorted({(min(a, 8 - a), min(b, 8 - b)) if a <= 4 else (8 - a, (8 - b) % 8) for a in range(8) for b in range(8)} - {(0, 0)})


def feat(path):
    x = np.asarray(Image.open(path).convert("L"), np.float32)
    r = x - ndimage.median_filter(x, size=3, mode="reflect")
    M = r.reshape(64, 8, 64, 8).mean((0, 2))
    F = np.abs(np.fft.fft2(M - M.mean())) / max(r.std(), 1e-6)
    z8 = M.var() * (r.size / 64) / max(r.var(), 1e-9)
    v = [F[a, b] for a in range(5) for b in range(8) if (a, b) != (0, 0) and not (a in (0, 4) and b > 4)]
    return np.r_[np.log(np.array(v) + 1e-6), np.log(z8)]


def main():
    os.chdir(ROOT)
    import pandas as pd
    o = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv")
    nat = o[(o.label == 0) & (o.split == "test") & (o.pool == "vision_native")]
    fk = o[o.label == 1]
    with ProcessPoolExecutor(8) as ex:
        Xn = np.array(list(ex.map(feat, nat.path, chunksize=4))); Xf = np.array(list(ex.map(feat, fk.path, chunksize=4)))
    out = {"n_features": int(Xn.shape[1])}
    for name, sel in (("SD2+SDXL", fk.generator.isin(["stable-diffusion-2", "stable-diffusion-xl"]).to_numpy()), ("all6", np.ones(len(fk), bool))):
        X = np.vstack([Xn, Xf[sel]]); y = np.r_[np.zeros(len(Xn)), np.ones(sel.sum())]
        g = np.r_[nat.device.astype(str).to_numpy(), fk.sha256.to_numpy()[sel]]
        res = {}
        for fs, cols in (("shape+z8", slice(None)), ("z8_only", slice(-1, None))):
            p = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, g):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr][:, cols], y[tr])
                p[te] = m.predict_proba(X[te][:, cols])[:, 1]
            gens = fk.generator.to_numpy()[sel]
            res[fs] = dict(auroc=round(float(roc_auc_score(y, p)), 3),
                           per_gen={gg: round(float(roc_auc_score(np.r_[np.zeros(len(Xn)), np.ones((gens == gg).sum())],
                                                                     np.r_[p[:len(Xn)], p[len(Xn):][gens == gg]])), 3) for gg in sorted(set(gens))})
        out[name] = res
    (HERE / "c4_gridshape.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))




# ---- JPEG-history-matched variant (appended after the first run): SD fakes JPEG-saved (libjpeg 4:2:0) at the QF of a seeded random test
# native (013 cache/native_qf.json), then the same CV. If the shape AUROC collapses, the shape feature was mostly a JPEG-history detector.
def feat_jm(args):
    path, q = args
    import io
    buf = io.BytesIO(); Image.open(path).convert("RGB").save(buf, "JPEG", quality=int(q)); buf.seek(0)
    tmp = Image.open(buf); tmp.load()
    x = np.asarray(tmp.convert("L"), np.float32)
    r = x - ndimage.median_filter(x, size=3, mode="reflect")
    M = r.reshape(64, 8, 64, 8).mean((0, 2))
    F = np.abs(np.fft.fft2(M - M.mean())) / max(r.std(), 1e-6)
    z8 = M.var() * (r.size / 64) / max(r.var(), 1e-9)
    v = [F[a, b] for a in range(5) for b in range(8) if (a, b) != (0, 0) and not (a in (0, 4) and b > 4)]
    return np.r_[np.log(np.array(v) + 1e-6), np.log(z8)]


def main_jm():
    os.chdir(ROOT)
    import pandas as pd
    qf = json.loads((ROOT / "experiments/013-latent-grid/cache/native_qf.json").read_text()); qs = np.array([qf[k] for k in sorted(qf)])
    o = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv")
    nat = o[(o.label == 0) & (o.split == "test") & (o.pool == "vision_native")]
    fk = o[(o.label == 1) & o.generator.isin(["stable-diffusion-2", "stable-diffusion-xl"])]
    rng = np.random.default_rng(0)
    with ProcessPoolExecutor(8) as ex:
        Xn = np.array(list(ex.map(feat, nat.path, chunksize=4)))
        Xf = np.array(list(ex.map(feat_jm, zip(fk.path, rng.choice(qs, len(fk))), chunksize=4)))
    X = np.vstack([Xn, Xf]); y = np.r_[np.zeros(len(Xn)), np.ones(len(Xf))]
    g = np.r_[nat.device.astype(str).to_numpy(), fk.sha256.to_numpy()]
    res = {}
    for fs, cols in (("shape+z8", slice(None)), ("z8_only", slice(-1, None))):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr][:, cols], y[tr])
            p[te] = m.predict_proba(X[te][:, cols])[:, 1]
        res[fs] = round(float(roc_auc_score(y, p)), 3)
    out = json.loads((HERE / "c4_gridshape.json").read_text()); out["SD2+SDXL_JM_fakes_jpeg_at_native_qf"] = res
    (HERE / "c4_gridshape.json").write_text(json.dumps(out, indent=1)); print(res)


if __name__ == "__main__":
    main_jm() if "--jm" in sys.argv else main()
