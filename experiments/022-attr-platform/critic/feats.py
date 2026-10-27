"""Critic scratch (H-A3): per image, dims; M1 (020 texture_stats, 224 centre crop) on the native-grid input A sees;
and A + M1 after resizing every copy to long side 1024 (common size), each under clean/social_sim/jpeg75_resize05."""
import sys, time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import run as R
import perturb

L = 1024
CONDS = R.CONDS


def texture_stats(a):  # verbatim from experiments/020-attr-foundation/run.py
    x = a.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    lap = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    resid = y - gaussian_filter(y, 1.0)
    mad = np.median(np.abs(resid - np.median(resid)))
    dh, dv = np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0))
    bh = np.arange(dh.shape[1]) % 8 == 7
    bv = np.arange(dv.shape[0]) % 8 == 7
    block = 0.5 * (dh[:, bh].mean() / (dh[:, ~bh].mean() + 1e-6) + dv[bv].mean() / (dv[~bv].mean() + 1e-6))
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-3)
    ys = y.std()
    skew = ((y - y.mean()) ** 3).mean() / (ys ** 3 + 1e-6)
    clipped = float(((x <= 2).any(-1) | (x >= 253).any(-1)).mean())
    return np.array([np.log(lap.var() + 1e-3), np.abs(lap).mean(), np.log(1.4826 * mad + 1e-3), block, sat.mean(), sat.std(),
                     x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean(), ys, skew, clipped], np.float32)


def m1(img, s=224):
    w, h = img.size
    l, t = (w - s) // 2, (h - s) // 2
    return texture_stats(np.asarray(img.convert("RGB").crop((l, t, l + s, t + s))))


def job(args):
    path, sha = args
    with Image.open(R.ROOT / path) as im:
        im.load()
        im = im.convert("RGB")
    w, h = im.size
    imc = perturb.resize(im, L / max(w, h))
    m1n, m1c, ac = [], [], []
    for c in CONDS:
        o = perturb.apply_condition(im, c, key=sha, seed=R.CFG["launder_seed"])
        m1n.append(m1(o))
        oc = perturb.apply_condition(imc, c, key=sha, seed=R.CFG["launder_seed"])
        m1c.append(m1(oc))
        ac.append(R.feat_A_from_Y(R.luma(oc)))
    return sha, np.array([w, h]), np.stack(m1n), np.stack(m1c), np.stack(ac)


if __name__ == "__main__":
    t0 = time.time()
    T = R.build_table()
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(job, list(zip(T.src_path, T.sha256)), chunksize=4))
    assert [r[0] for r in res] == T.sha256.tolist()
    np.savez(HERE / "cache" / "critic_feats.npz", sha=np.array(T.sha256.tolist()), conds=np.array(CONDS),
             wh=np.stack([r[1] for r in res]), m1n=np.stack([r[2] for r in res]), m1c=np.stack([r[3] for r in res]),
             ac=np.stack([r[4] for r in res]))
    print(f"done {len(res)} in {time.time()-t0:.0f}s")
