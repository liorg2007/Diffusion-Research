"""Critic check for H-A2(i) (reporting only): why matched fingerprints give 0.52 at jpeg75_resize05.
Mean-residual NCC can only use patterns that are phase-locked to the crop origin. Test: fingerprints from (0,0) train crops
(seed-0 family split, as run.py), test crops at offsets (0,0), (4,4), (3,5) of the laundered image. Conditions: clean,
jpeg75_resize05, resize(0.5), jpeg(75). Also: per-generator share of fingerprint spectral energy on the period-4/8/16 lattice."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
spec = importlib.util.spec_from_file_location("h2run", EXP / "run.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
CONDS = ["clean", "jpeg75_resize05", "resize(0.5)", "jpeg(75)"]
OFFS = [(0, 0), (4, 4), (3, 5)]
C = 256


def worker(a):
    path, sha = a
    img = Image.open(ROOT / path); img.load(); img = img.convert("RGB")
    out = {}
    for c in CONDS:
        im = R.perturb.apply_condition(img, c, key=sha, seed=0)
        for oy, ox in OFFS:
            g = np.asarray(im.crop((ox, oy, ox + C, oy + C)).convert("L"), np.float64)
            out[(c, oy, ox)] = R.residual(g).astype(np.float32).ravel()
    return out


def lattice_share(F):
    P = np.abs(np.fft.fft2(F.reshape(C, C))) ** 2
    P[0, 0] = 0
    k = np.arange(C)
    res = {}
    for per in (4, 8, 16):
        on = (k % (C // per) == 0)
        m = on[:, None] | on[None, :]
        m2 = on[:, None] & on[None, :]
        res[f"p{per}_rowcol"] = float(P[m].sum() / P.sum())
        res[f"p{per}_grid_points"] = float(P[m2].sum() / P.sum())
    return res


if __name__ == "__main__":
    man = pd.read_csv(ROOT / "data" / "manifest_raw.csv", keep_default_na=False)
    df = man[(man.label == 1) & man.generator.isin(R.GENS)].reset_index(drop=True)
    fam, _ = R.load_caption_ids(df)
    g = df.generator.map({x: i for i, x in enumerate(R.GENS)}).to_numpy()
    tr = R.split_families(fam, 0.7, 0); te = ~tr
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(worker, list(zip(df.path, df.sha256)), chunksize=8))
    out = {"bacc": {}, "recall_offset00": {}, "lattice_share": {}, "native_sizes": {}}
    for gen in R.GENS:
        s = df[df.generator == gen]
        out["native_sizes"][gen] = s.assign(wh=s.width.astype(str) + "x" + s.height.astype(str)).wh.value_counts().head(3).to_dict()
    for c in CONDS:
        Rtr = np.stack([r[(c, 0, 0)] for r in res])[tr]
        F = R.fit_fingerprints(Rtr, g[tr], 8)
        out["lattice_share"][c] = {gen: lattice_share(F[i]) for i, gen in enumerate(R.GENS)}
        for oy, ox in OFFS:
            Rte = np.stack([r[(c, oy, ox)] for r in res])[te]
            pred = R.ncc_scores(Rte, F).argmax(1)
            out["bacc"][f"{c}|off={oy},{ox}"] = float(R.macro_recall(pred == g[te], g[te], n_cls=8))
            if (oy, ox) == (0, 0) or (oy, ox) == (3, 5):
                out["recall_offset00" if (oy, ox) == (0, 0) else "recall_offset35"] = out.get("recall_offset35", {}) if (oy, ox) == (3, 5) else out["recall_offset00"]
                key = "recall_offset00" if (oy, ox) == (0, 0) else "recall_offset35"
                out[key][c] = {gen: float(((pred == i) & (g[te] == i)).sum() / (g[te] == i).sum()) for i, gen in enumerate(R.GENS)}
    (HERE / "gen_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("bacc", "recall_offset00", "recall_offset35", "native_sizes")}, indent=1))
