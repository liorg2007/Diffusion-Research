"""Build the RAISE STAND-IN fixture for run_official.py --dry-run --standin-raise. NOT RAISE, NOT DATA, NOT RESULTS.

20 VISION native train-split crops (pilot splits) are re-saved as PNG (orig arm: the 512 crop itself; fov512: same file, a
512 crop is already 512; fov224: short side -> 224 via lib/preprocess.fov_resize) under cache/standin/<arm>/<sha>.png, labelled
pool = "raise_standin", generator "real", 14 as split=test and 6 as calib, b0_train False. Writes
cache/standin/splits_standin_<arm>.csv (rows only for the stand-ins) and prints the 004 commands that extract their features
(run them with --device cpu or when the GPU is free; they are cheap: 20 images x 3 conditions per arm).
  env/.venv/bin/python -I experiments/006-format-shortcut/make_standin.py"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import pandas as pd
from PIL import Image

import preprocess

OC = json.loads((HERE / "config_official.json").read_text())
SI = OC["standin_raise"]
OUT = HERE / SI["dir"]
SPL = ROOT / OC["profiles"]["pilot"]["splits"]["orig"]


def main():
    df = pd.read_csv(SPL, keep_default_na=False, dtype={"vidx": str})
    src = df[(df.split == "train") & (df.pool == OC["pools"]["native"])].sort_values("sha256").head(SI["n"])
    rows = {a: [] for a in ("orig", "fov512", "fov224")}
    for k, r in enumerate(src.itertuples()):
        with Image.open(ROOT / r.path) as im:
            im = im.convert("RGB")
            ims = {"orig": im, "fov512": im, "fov224": preprocess.fov_resize(im, 224)}
        paths = {}
        for a, x in ims.items():
            d = OUT / a
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"tmp_{k}.png"
            x.save(p, "PNG")
            paths[a] = p
        sha = hashlib.sha256(paths["orig"].read_bytes()).hexdigest()       # the stand-in's own key (never a real sha)
        for a, p in paths.items():
            q = p.with_name(f"{sha}.png")
            p.replace(q)
            row = r._asdict()
            row.pop("Index")
            row.update(sha256=sha, path=str(q.relative_to(ROOT)), src_path=str(q.relative_to(ROOT)), label=0, generator="real",
                       pool=SI["pool_name"], camera_or_platform="STAND-IN (not RAISE)", device="", vidx="",
                       split="test" if k < SI["n_test"] else "calib", in_scoring_set=True, b0_train=False, orig_format="PNG",
                       source_dataset="STAND-IN fixture: VISION native re-saved as PNG", license="n/a (fixture)")
            rows[a].append(row)
    for a, rr in rows.items():
        pd.DataFrame(rr, columns=list(df.columns)).to_csv(OUT / f"splits_standin_{a}.csv", index=False)
    print(f"wrote {SI['n']} stand-in rows x 3 arms under {OUT.relative_to(ROOT)}; extract their features with:")
    c = "env/.venv/bin/python -I experiments/004-phase4-features/run.py --no-guard --device cpu"
    for a, conds in (("orig", "clean 'jpeg(90)' jpeg75_resize05"), ("fov512", "clean 'jpeg(90)' jpeg75_resize05"), ("fov224", "clean 'jpeg(90)'")):
        print(f"{c} --splits {SI['dir']}/splits_standin_{a}.csv --cache-subdir standin_{a} --conditions {conds}".replace("--splits ", "--splits experiments/006-format-shortcut/"))


if __name__ == "__main__":
    main()
