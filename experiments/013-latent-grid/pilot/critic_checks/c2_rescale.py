"""H-D2 critic check 2 (post hoc, CPU): does the phase-locked decoder grid (found by c1 with fold z8, not with O) survive resize s and
resize+JPEG75, and is the dyadic s = 0.5 special? Periods p = 8 s are integers for s in {0.5, 0.625, 0.75, 0.875, 1}: 4, 5, 6, 7, 8.
Score d_p = z_p - (z_{p-1} + z_{p+1})/2 (fold energy at the oracle period minus two off-period controls, texture-normalised).
Groups: 011 fov512 PNG round-trips vs their reals (paired positive control, 150), orig SD2+SDXL (240), orig test natives (358, JPEG-coded),
fov512 test natives (358, grid-free). -> c2_rescale.json"""
import importlib.util, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
import sys; sys.path.insert(0, str(ROOT / "experiments" / "lib"))
from perturb import apply_condition
SS = [0.5, 0.625, 0.75, 0.875, 1.0]
CONDS = [(f"resize({s})" if s < 1 else "clean", s) for s in SS] + [(f"jr(75,{s})", s) for s in SS]


def fz(r, p):
    h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
    q = r[:h, :w]
    M = q.reshape(h // p, p, w // p, p).mean((0, 2))
    return float(M.var() * (q.size / p ** 2) / max(q.var(), 1e-9))


def job(path):
    img = Image.open(path).convert("RGB")
    out = []
    for c, s in CONDS:
        x = np.asarray(apply_condition(img, c).convert("L"), np.float32)
        r = x - ndimage.median_filter(x, size=3, mode="reflect")
        p = int(round(8 * s))
        zp, zl, zh = fz(r, p), fz(r, p - 1), fz(r, p + 1)
        out.append([zp - (zl + zh) / 2, zp, fz(r, 8)])
    return out


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def main():
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("run011", ROOT / "experiments/011-vae-trace/run.py")
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    import pandas as pd
    o = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv")
    f = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot_fov512.csv")
    te = R.e2.load_df("fov512"); te = te[(te.label == 0) & (te.split == "test")].head(150)
    G = {"pc_real": te.path.tolist(), "pc_rt": [str(R.rt_file("fov512", s)) for s in te.sha256],
         "sd_orig": o[(o.label == 1) & o.generator.isin(["stable-diffusion-2", "stable-diffusion-xl"])].path.tolist(),
         "nat_orig": o[(o.label == 0) & (o.split == "test") & (o.pool == "vision_native")].path.tolist(),
         "nat_fov": f[(f.label == 0) & (f.split == "test") & (f.pool == "vision_native")].path.tolist()}
    A = {}
    with ProcessPoolExecutor(8) as ex:
        for k, ps in G.items():
            A[k] = np.array(list(ex.map(job, ps, chunksize=4)))  # [n, cond, 3]
            print(k, A[k].shape, flush=True)
    out = {}
    for ci, (c, s) in enumerate(CONDS):
        d = {k: A[k][:, ci, :] for k in A}
        out[c] = dict(p=int(round(8 * s)),
                      median_d={k: round(float(np.median(v[:, 0])), 3) for k, v in d.items()},
                      median_zp={k: round(float(np.median(v[:, 1])), 3) for k, v in d.items()},
                      auc_d_pc_rt_vs_real=round(auc(d["pc_real"][:, 0], d["pc_rt"][:, 0]), 3),
                      win_d_pc=round(float((d["pc_rt"][:, 0] > d["pc_real"][:, 0]).mean()), 3),
                      auc_d_sd_vs_nat_fov=round(auc(d["nat_fov"][:, 0], d["sd_orig"][:, 0]), 3),
                      auc_d_sd_vs_nat_orig=round(auc(d["nat_orig"][:, 0], d["sd_orig"][:, 0]), 3),
                      auc_z8_sd_vs_nat_orig=round(auc(d["nat_orig"][:, 2], d["sd_orig"][:, 2]), 3))
        print(c, out[c], flush=True)
    (HERE / "c2_rescale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
