"""Latent-stride periodicity check that does not rely on single FFT bins: phase-fold the median residual of the FULL image on the
8x8 latent grid (the VAE was applied to the full 512 crop, so the grid phase is known: origin 0). M_i[a,b] = mean r[a::p, b::p].
  z_i  = var(M_i) * (N/p^2) / var(r)      (~1 for no periodic structure)
  tmpl = corr(M_i - mean(M_i), T) with T = mean over TRAIN round-trips of (M_rt - M_real)  (a fixed-phase VAE fingerprint)
Conditions: clean (p = 8) and jpeg75_resize05 / resize(0.5) (p = 4 after the 0.5 resize, same origin). CPU only -> c4_fold.json."""
import importlib.util, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run011", HERE.parents[1] / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
R.set_mode(False)
from perturb import apply_condition


def fold(im, p):
    x = np.asarray(im.convert("L"), np.float32)
    r = x - ndimage.median_filter(x, size=3, mode="reflect")
    h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
    r = r[:h, :w]
    M = r.reshape(h // p, p, w // p, p).mean((0, 2))
    z = M.var() * (r.size / p ** 2) / max(r.var(), 1e-9)
    return M.ravel(), float(z)


out = {}
for arm in R.ARMS:
    df = R.load_df(arm)
    lab, sp, sha = df.label.to_numpy(), df.split.to_numpy(), df.sha256.to_numpy()
    tr = np.where((lab == 0) & (sp == "train"))[0]
    te = np.where((lab == 0) & (sp == "test"))[0]
    for c, p in (("clean", 8), ("jpeg75_resize05", 4), ("resize(0.5)", 4)):
        def job(i):
            a = apply_condition(Image.open(df.path.iloc[i]).convert("RGB"), c, key=sha[i])
            b = apply_condition(Image.open(R.rt_file(arm, sha[i])).convert("RGB"), c, key=sha[i])
            return fold(a, p), fold(b, p)
        with ThreadPoolExecutor(8) as pool:
            res = list(pool.map(job, np.r_[tr, te]))
        Mr = np.stack([x[0][0] for x in res]); zr = np.array([x[0][1] for x in res])
        Mt = np.stack([x[1][0] for x in res]); zt = np.array([x[1][1] for x in res])
        n = len(tr)
        T = (Mt[:n] - Mr[:n]).mean(0)
        T = T - T.mean()

        def corr(M):
            M = M - M.mean(1, keepdims=True)
            return (M @ T) / (np.linalg.norm(M, axis=1) * np.linalg.norm(T) + 1e-12)
        cr, ct = corr(Mr[n:]), corr(Mt[n:])
        out[f"{arm}/{c}"] = dict(
            z_median_real=float(np.median(zr[n:])), z_median_rt=float(np.median(zt[n:])),
            z_pooled_auroc=R.point_auc(zr[n:], zt[n:]), z_win=float((zt[n:] > zr[n:]).mean()),
            tmpl_pooled_auroc=R.point_auc(cr, ct), tmpl_win=float((ct > cr).mean()),
            tmpl_median_real=float(np.median(cr)), tmpl_median_rt=float(np.median(ct)),
            template_train_amplitude=float(np.abs(T).max()), period=p)
        print(arm, c, {k: round(v, 3) for k, v in out[f"{arm}/{c}"].items()}, flush=True)
(HERE / "c4_fold.json").write_text(json.dumps(out, indent=1))
