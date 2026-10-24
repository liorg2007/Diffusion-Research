"""H-D2 critic check 3 (post hoc, CPU): the validity gate re-read against a GRID-FREE real reference (fov512 test natives: camera JPEGs
downscaled ~6x, so the capture JPEG grid is gone) instead of JPEG-coded orig natives, for the pre-registered O at 1/8 and for fold z8.
Also z8 vs the image's own z7/z9 (within-image control, no real reference needed). -> c3_gate.json"""
import importlib.util, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("run013", ROOT / "experiments/013-latent-grid/run.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from scipy import ndimage


def job(path):
    im = Image.open(path).convert("RGB")
    v = M.image_feats(im, 1.0, 512, lambda x: 0.0)
    x = np.asarray(im.convert("L"), np.float32); r = x - ndimage.median_filter(x, size=3, mode="reflect")
    return [v[M.FI["O"]], v[M.FI["z8"]], M.fold_z(r, 7), M.fold_z(r, 9)]


def main():
    os.chdir(ROOT)
    import pandas as pd
    o = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot.csv")
    f = pd.read_csv("experiments/004-phase4-features/pilot/splits_pilot_fov512.csv")
    G = {g: o[(o.label == 1) & (o.generator == g)].path.tolist() for g in M.GENS}
    G["nat_orig"] = o[(o.label == 0) & (o.split == "test") & (o.pool == "vision_native")].path.tolist()
    G["nat_fov"] = f[(f.label == 0) & (f.split == "test") & (f.pool == "vision_native")].path.tolist()
    with ProcessPoolExecutor(8) as ex:
        A = {k: np.array(list(ex.map(job, v, chunksize=4))) for k, v in G.items()}
    out = {}
    for ref in ("nat_orig", "nat_fov"):
        tO, tz = np.percentile(A[ref][:, 0], 99), np.percentile(A[ref][:, 1], 99)
        out[f"ref={ref}"] = dict(thr_O=float(tO), thr_z8=float(tz),
                                 share_O_above={k: round(float((v[:, 0] > tO).mean()), 3) for k, v in A.items()},
                                 share_z8_above={k: round(float((v[:, 1] > tz).mean()), 3) for k, v in A.items()})
    out["within_image_z8_gt_1.5x_max_z7_z9"] = {k: round(float((v[:, 1] > 1.5 * np.maximum(v[:, 2], v[:, 3])).mean()), 3) for k, v in A.items()}
    out["median_O_1_8"] = {k: round(float(np.median(v[:, 0])), 3) for k, v in A.items()}
    (HERE / "c3_gate.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
