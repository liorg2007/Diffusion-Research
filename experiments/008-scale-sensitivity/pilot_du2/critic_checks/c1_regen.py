"""Critic check Q3: regenerate du2 (and crop/fov) PNGs with the COMMITTED lib/preprocess.py and compare pixel-wise
with data/processed/{du2512,crop512,fov512}. Temp PNGs in critic_checks/regen_tmp (deleted at the end)."""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/lib"))
import preprocess as P
HERE = Path(__file__).resolve().parent
TMP = HERE / "regen_tmp"; TMP.mkdir(exist_ok=True)
print("git diff HEAD -- lib/preprocess.py:", repr(subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--", "experiments/lib/preprocess.py"], capture_output=True, text=True).stdout))
du = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot_du2.csv", keep_default_na=False)
orig = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False).set_index("sha256")
fov = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot_fov512.csv", keep_default_na=False).set_index("sha256")
assert (du.sha256.values == orig.index.values).all()
du["grp"] = np.where(du.label == 1, du.generator, du.pool)
samp = pd.concat([g.sample(min(6, len(g)), random_state=0) for _, g in du.groupby("grp")])
rows = []
for _, r in samp.iterrows():
    out = {}
    for mode, ref in (("du2", r.path), ("crop", orig.loc[r.sha256, "path"]), ("fov", fov.loc[r.sha256, "path"])):
        dst = TMP / f"{r.sha256}_{mode}.png"
        st, info = P.process_image(str(ROOT / r.src_path), str(dst), 512, mode=mode)
        a = np.asarray(Image.open(dst)).astype(int); b = np.asarray(Image.open(ROOT / ref)).astype(int)
        out[mode] = int(np.abs(a - b).max()) if a.shape == b.shape else -1
    rows.append(dict(grp=r.grp, sha=r.sha256[:12], w=info["orig_width"], h=info["orig_height"], **{f"maxdiff_{k}": v for k, v in out.items()}))
R = pd.DataFrame(rows)
print(R.to_string())
print("n =", len(R), "| all du2 identical:", bool((R.maxdiff_du2 == 0).all()), "| crop:", bool((R.maxdiff_crop == 0).all()), "| fov:", bool((R.maxdiff_fov == 0).all()))
shutil.rmtree(TMP)
