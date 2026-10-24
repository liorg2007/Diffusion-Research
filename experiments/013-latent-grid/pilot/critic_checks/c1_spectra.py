"""H-D2 critic check 1 (post hoc, CPU): does an 8-px latent grid exist in the data at all, where does its energy sit, and why did O miss it?
 - crop alignment of orig SD2/SDXL crops vs the raw files (phase 0 claim)
 - 2-D periodogram of the median residual (gray), lattice points (k/8, l/8), k,l in 0..4: prominence = log10 P[pt] - median log10 P over a
   9x9 window minus its 3x3 centre; per image and on the group-mean spectrum (Corvi-style incoherent averaging)
 - control lattice shifted by half a lattice step (k/8+1/16, l/8+1/16) -> same statistic, no grid expected
 - phase-fold z for p = 7, 8, 9 (p = 7, 9 are controls) and the 8x8 fold's DFT energy split: axis (k=0 or l=0) vs cross terms
Groups: orig SD2, SDXL, other gens, test natives (orig); fov512 test natives (grid-free reals); 011 fov512 round-trips (positive control, PNG).
  env/.venv/bin/python -I experiments/013-latent-grid/pilot/critic_checks/c1_spectra.py   -> c1_spectra.json"""
import importlib.util, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
N = 512
W2 = np.outer(np.hanning(N), np.hanning(N))
LAT = [(k, l) for k in range(5) for l in range(5) if (k, l) != (0, 0)]


def resid(path):
    x = np.asarray(Image.open(path).convert("L"), np.float32)
    assert x.shape == (N, N), (path, x.shape)
    return x - ndimage.median_filter(x, size=3, mode="reflect")


def prom2d(LP, i, j):
    idx = [(i + a) % N for a in range(-4, 5)], [(j + b) % N for b in range(-4, 5)]
    win = LP[np.ix_(*idx)].copy()
    win[3:6, 3:6] = np.nan
    return float(LP[i, j] - np.nanmedian(win))


def fold(r, p):
    h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
    q = r[:h, :w]
    M = q.reshape(h // p, p, w // p, p).mean((0, 2))
    return M, float(M.var() * (q.size / p ** 2) / max(q.var(), 1e-9))


def feats(path):
    r = resid(path)
    P = np.abs(np.fft.fft2((r - r.mean()) * W2)) ** 2
    LP = np.log10(P + 1e-10)
    lat = np.array([prom2d(LP, k * 64, l * 64) for k, l in LAT])
    ctl = np.array([prom2d(LP, k * 64 + 32, l * 64 + 32) for k, l in LAT])
    M8, z8 = fold(r, 8)
    F = np.abs(np.fft.fft2(M8 - M8.mean())) ** 2
    axis = F[0, :].sum() + F[:, 0].sum()
    frac_axis = float(axis / max(F.sum(), 1e-12))
    # energy per harmonic index max(k,l) folded to 0..4
    kk = np.minimum(np.arange(8), 8 - np.arange(8))
    H = np.zeros(5)
    for a in range(8):
        for b in range(8):
            H[max(kk[a], kk[b])] += F[a, b]
    H = H / max(F.sum(), 1e-12)
    return dict(lat=lat, ctl=ctl, z7=fold(r, 7)[1], z8=z8, z9=fold(r, 9)[1], frac_axis=frac_axis, harm=H[1:], P=P.astype(np.float32))


def group_paths():
    spec = importlib.util.spec_from_file_location("run011", ROOT / "experiments/011-vae-trace/run.py")
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    import pandas as pd
    o = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv")
    f = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot_fov512.csv")
    g = {}
    for gen in ["stable-diffusion-2", "stable-diffusion-xl", "dalle2", "dalle3", "midjourney-v5", "firefly"]:
        g["orig_" + gen] = o[(o.label == 1) & (o.generator == gen)].path.tolist()
    g["orig_natives"] = o[(o.label == 0) & (o.split == "test") & (o.pool == "vision_native")].path.tolist()
    g["fov_natives"] = f[(f.label == 0) & (f.split == "test") & (f.pool == "vision_native")].path.tolist()
    te = R.e2.load_df("fov512"); te = te[(te.label == 0) & (te.split == "test")].head(150)
    g["fov_pc_real"] = te.path.tolist()
    g["fov_pc_vae_rt"] = [str(R.rt_file("fov512", s)) for s in te.sha256]
    g["orig_sd_crops_rows"] = o[(o.label == 1) & o.generator.isin(["stable-diffusion-2", "stable-diffusion-xl"])][["path", "src_path"]].values.tolist()
    return g


def crop_check(rows):
    bad = 0
    for p, sp in rows:
        a = np.asarray(Image.open(ROOT / p).convert("RGB")); b = np.asarray(Image.open(ROOT / sp).convert("RGB"))
        H, W = b.shape[:2]; t, l = (H - N) // 2, (W - N) // 2
        bad += int(not np.array_equal(a, b[t:t + N, l:l + N])) + 0 * (t % 8 + l % 8)
        bad += int((t % 8) or (l % 8)) * 1000
    return dict(n=len(rows), n_mismatch_or_offphase=bad)


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def main():
    import os; os.chdir(ROOT)
    g = group_paths()
    out = {"crop_check_orig_SD": crop_check(g.pop("orig_sd_crops_rows"))}
    res = {}
    with ProcessPoolExecutor(8) as ex:
        for k, ps in g.items():
            res[k] = list(ex.map(feats, ps, chunksize=4))
            print(k, len(ps), flush=True)
    summ = {}
    for k, rs in res.items():
        lat = np.stack([r["lat"] for r in rs]); ctl = np.stack([r["ctl"] for r in rs])
        Pm = np.mean([r["P"] for r in rs], 0); LPm = np.log10(Pm + 1e-10)
        summ[k] = dict(
            n=len(rs),
            median_lat_prom_per_point={f"{a}/8,{b}/8": round(float(v), 3) for (a, b), v in zip(LAT, np.median(lat, 0))},
            median_lat_mean=float(np.median(lat.mean(1))), median_ctl_mean=float(np.median(ctl.mean(1))),
            share_lat_mean_gt_ctl_mean=float((lat.mean(1) > ctl.mean(1)).mean()),
            groupmean_spectrum_lat_prom={f"{a}/8,{b}/8": round(prom2d(LPm, a * 64, b * 64), 3) for a, b in LAT},
            groupmean_spectrum_ctl_prom_mean=float(np.mean([prom2d(LPm, a * 64 + 32, b * 64 + 32) for a, b in LAT])),
            median_z7=float(np.median([r["z7"] for r in rs])), median_z8=float(np.median([r["z8"] for r in rs])),
            median_z9=float(np.median([r["z9"] for r in rs])),
            median_fold8_frac_energy_on_axes=float(np.median([r["frac_axis"] for r in rs])),
            mean_fold8_energy_share_by_harmonic_k1_k4=np.mean([r["harm"] for r in rs], 0).round(3).tolist())
    out["groups"] = summ
    sc = lambda k, f: [f(r) for r in res[k]]
    S = {"latmean": lambda r: r["lat"].mean(), "lat_1/8_axes": lambda r: np.mean([r["lat"][LAT.index((1, 0))], r["lat"][LAT.index((0, 1))]]),
         "lat_4/8_any": lambda r: np.mean([v for (a, b), v in zip(LAT, r["lat"]) if max(a, b) == 4]),
         "lat_cross": lambda r: np.mean([v for (a, b), v in zip(LAT, r["lat"]) if a and b]), "z8": lambda r: r["z8"],
         "z8_minus_mean_z7z9": lambda r: r["z8"] - (r["z7"] + r["z9"]) / 2}
    comps = [("orig_natives", "orig_stable-diffusion-2"), ("orig_natives", "orig_stable-diffusion-xl"),
             ("fov_natives", "orig_stable-diffusion-2"), ("fov_natives", "orig_stable-diffusion-xl"),
             ("fov_pc_real", "fov_pc_vae_rt"), ("orig_natives", "orig_dalle2"), ("orig_natives", "orig_midjourney-v5")]
    out["auroc_fake_high"] = {f"{p} vs {n}": {s: round(auc(sc(n, f), sc(p, f)), 3) for s, f in S.items()} for n, p in comps}
    (HERE / "c1_spectra.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
