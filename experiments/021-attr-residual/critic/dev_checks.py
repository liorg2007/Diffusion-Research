"""Critic checks for H-A2(ii) (reporting only; does not touch the judged pipeline).
A) near-duplicate scenes: thumbnail + high-pass-thumbnail correlation of the 197 canonical natives, within vs across device;
   relation of within-device near-duplication to the true-device PCE (results/prnu_scores.csv, nat arm).
B) M1 texture baseline for devices (the 11 run.py m1_features on full-res natives), leave-one-photo-out logistic regression.
C) direct content probe: single-photo fingerprints. For the most-duplicated within-device pairs (A,B) vs a non-duplicate
   same-device photo C: PCE(A | K_B) vs PCE(A | K_C), plus rho at small shifts relative to zero shift."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
spec = importlib.util.spec_from_file_location("h2run", EXP / "run.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
tab = pd.read_csv(EXP / "results" / "vision_photos.csv")
DEVS = R.DEVS
sets = R.CFG["dev"]["same_model_sets"]


def load(r):
    x = R.canon(R.load_gray_raw(r["nat"]), "nat", r["exif"], r["k_extra"])
    return x


def worker(r):
    x = load(r)
    rgb = np.asarray(Image.open(ROOT / r["nat"]).convert("RGB"))
    if rgb.shape[0] > rgb.shape[1]:
        rgb = np.rot90(rgb, r["k_extra"] if r["k_extra"] in (1, 3) else 1)
    res = R.residual(x)
    m1 = R.m1_features(rgb, x, res)
    th = np.asarray(Image.fromarray(x.astype(np.float32)).resize((128, 96), Image.BILINEAR), np.float64)
    hp = th - ndimage.gaussian_filter(th, 2)
    return m1, th.ravel(), hp.ravel()


def corrmat(A):
    A = A - A.mean(1, keepdims=True); A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A @ A.T


def single_fp(x):
    r = R.residual(x)
    return R.zero_mean_rc(r * x / (x * x + R.EPS))


def corr_surface(W, Y):
    from scipy import fft as sfft
    Y = Y - Y.mean()
    c = sfft.irfft2(np.conj(sfft.rfft2(W)) * sfft.rfft2(Y), s=W.shape) / (np.linalg.norm(W) * np.linalg.norm(Y))
    return c


def probe(args):
    a, b, kind = args
    ra, rb = tab.iloc[a].to_dict(), tab.iloc[b].to_dict()
    xa = load(ra); Wa = R.residual(xa)
    K = single_fp(load(rb))
    c = corr_surface(Wa, xa * K)
    pce, rho0, pz, sh = R.pce_from_corr(c)
    near = {f"{dy},{dx}": float(c[dy % c.shape[0], dx % c.shape[1]] / c[0, 0]) for dy, dx in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 1))}
    return {"A": f"{ra['dev']}_{ra['idx']}", "B": f"{rb['dev']}_{rb['idx']}", "kind": kind, "pce": pce, "rho0": rho0,
            "peak_zero": pz, "rho_shift_over_rho0": near}


if __name__ == "__main__":
    rows = tab.to_dict("records")
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(worker, rows, chunksize=4))
    M1 = np.stack([o[0] for o in out]); TH = np.stack([o[1] for o in out]); HP = np.stack([o[2] for o in out])
    dev = tab.dev.to_numpy(); y = np.array([DEVS.index(d) for d in dev])
    res = {}
    # A) near duplicates
    for name, A in (("thumb", TH), ("highpass_thumb", HP)):
        C = corrmat(A.copy()); np.fill_diagonal(C, np.nan)
        same = dev[:, None] == dev[None]
        iu = np.triu_indices(len(dev), 1)
        cs, cd = C[iu][same[iu]], C[iu][~same[iu]]
        mx_same = np.nanmax(np.where(same, C, np.nan), 1); mx_diff = np.nanmax(np.where(~same, C, np.nan), 1)
        res[name] = {"pairs_same": int(len(cs)), "pairs_diff": int(len(cd)),
                     "same_median": float(np.median(cs)), "diff_median": float(np.median(cd)),
                     "same_n_gt_0.9": int((cs > 0.9).sum()), "diff_n_gt_0.9": int((cd > 0.9).sum()),
                     "same_n_gt_0.7": int((cs > 0.7).sum()), "diff_n_gt_0.7": int((cd > 0.7).sum()),
                     "photos_with_same_dev_max_gt_0.7": int((mx_same > 0.7).sum()),
                     "photos_nn_is_same_device": float(np.mean(mx_same > mx_diff))}
        res[name + "_maxsame"] = mx_same
    sc = pd.read_csv(EXP / "results" / "prnu_scores.csv")
    tru = sc[(sc.arm == "nat") & (sc.dev == sc.cand)].set_index(["dev", "idx"]).pce
    pce_true = np.array([tru[(d, i)] for d, i in zip(tab.dev, tab.idx)])
    from scipy.stats import spearmanr
    for name in ("thumb", "highpass_thumb"):
        mx = res.pop(name + "_maxsame")
        rho, p = spearmanr(mx, np.log(np.maximum(pce_true, 1)))
        hi = mx > np.quantile(mx, 0.8)
        res[name]["spearman_maxsame_vs_logPCEtrue"] = [float(rho), float(p)]
        res[name]["median_PCEtrue_top20pct_dup_vs_rest"] = [float(np.median(pce_true[hi])), float(np.median(pce_true[~hi]))]
        if name == "highpass_thumb":
            C = corrmat(HP.copy()); np.fill_diagonal(C, -9)
            same = dev[:, None] == dev[None]
            Cs = np.where(same, C, -9)
            order = np.dstack(np.unravel_index(np.argsort(-Cs, axis=None), Cs.shape))[0]
            pairs, seen = [], set()
            for a, b in order:
                if a < b and (a, b) not in seen:
                    seen.add((a, b)); pairs.append((int(a), int(b), float(Cs[a, b])))
                if len(pairs) == 4:
                    break
            res["top_dup_pairs"] = [(f"{tab.dev[a]}_{tab.idx[a]}", f"{tab.dev[b]}_{tab.idx[b]}", c) for a, b, c in pairs]
    # B) M1 devices, LOPO
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.empty(len(y), int)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        pred[i] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(M1[m], y[m]).predict(M1[i:i + 1])[0]
    top1 = float(np.mean([(pred[y == d] == d).mean() for d in range(len(DEVS))]))
    wm = []
    for s, ds in sets.items():
        idx = [DEVS.index(d) for d in ds]
        msk = np.isin(y, idx)
        predw = np.empty(msk.sum(), int)
        Xi, yi = M1[msk], y[msk]
        for j in range(len(yi)):
            m = np.arange(len(yi)) != j
            predw[j] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xi[m], yi[m]).predict(Xi[j:j + 1])[0]
        wm.append({"set": s, "acc_macro": float(np.mean([(predw[yi == d] == d).mean() for d in idx]))})
    res["m1_dev"] = {"top1_macro_lopo": top1, "chance": 1 / 14, "within_model_sets": wm}
    # C) direct content probe on the top duplicated pairs
    C = corrmat(HP.copy()); np.fill_diagonal(C, 9)
    jobs = []
    for a_s, b_s, _ in res["top_dup_pairs"]:
        a = int(np.where((tab.dev + "_" + tab.idx.astype(str)) == a_s)[0][0]); b = int(np.where((tab.dev + "_" + tab.idx.astype(str)) == b_s)[0][0])
        same_dev = np.where(tab.dev == tab.dev[a])[0]
        c_ = int(same_dev[np.argmin(C[a, same_dev])])  # least similar photo of the same device
        jobs += [(a, b, "dup_same_device"), (a, c_, "nondup_same_device")]
    with ProcessPoolExecutor(4) as ex:
        res["content_probe_single_photo_fp"] = list(ex.map(probe, jobs))
    (HERE / "dev_checks.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
