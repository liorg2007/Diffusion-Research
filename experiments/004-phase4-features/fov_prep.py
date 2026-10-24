"""fov512 preprocessing (D-013 / Amendment 3): resize the SHORT side to 512 (bicubic, antialiased), then center-crop 512.

  # PILOT (VISION + 6 Synthbuster rows of splits_pilot.csv)
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py \
      --splits experiments/004-phase4-features/pilot/splits_pilot.csv \
      --out-splits experiments/004-phase4-features/pilot/splits_pilot_fov512.csv \
      --report experiments/004-phase4-features/pilot/fov512_prep_report.json
  # OFFICIAL (all rows of 003 splits.csv whose raw file exists; run after env/bootstrap_data.py re-fetched RAISE etc.)
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py \
      --out-splits experiments/004-phase4-features/splits_fov512.csv --report experiments/004-phase4-features/fov512_prep_report.json

  # --no-clobber: safe for the official run, where data/processed/{crop512,fov512} already hold the pilot crops (outside experiments/):
  #   existing files are re-created beside themselves, compared pixel-wise and left untouched (abort if any differ); missing ones are written.
  # du2 (Amendment 10, E1b): fov resize, resize back to the original size, plain centre crop -> data/processed/du2512/images
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py --mode du2 \
      --splits experiments/004-phase4-features/pilot/splits_pilot.csv \
      --out-splits experiments/004-phase4-features/pilot/splits_pilot_du2.csv --report experiments/004-phase4-features/pilot/du2_prep_report.json
  # fov224 (H4 Amendment 6, whole-image arm for RIGID's 224 input: short side -> 224, center crop 224)
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py --size 224 \
      --out-splits experiments/004-phase4-features/splits_fov224.csv --report experiments/004-phase4-features/fov224_prep_report.json
  # (--size N writes data/processed/fov<N>/images; --mode crop re-creates 003's plain 512 centre crops from the raw files
  #  at the splits' own `path` without touching splits.csv: use it after env/bootstrap_data.py re-fetched RAISE/SD3.5/FLUX)

Both classes are processed from the RAW files (`src_path`; EXIF-oriented, metadata stripped, as lib/preprocess.py) into
data/processed/fov512/images/<sha256>.png (git-ignored). The output splits file has the SAME rows and columns as the
input; only `path` points at the fov512 crop. `sha256` stays the raw file's hash, so feature caches stay keyed alike.
Images whose short side is < 512 are NEVER upscaled: they are listed in the report and the run ABORTS (no splits file
written) unless --drop-small is given, in which case those rows are dropped and counted.
Checks: raw file sha256 == row sha256; every output is a 512x512 RGB PNG.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from PIL import Image

import preprocess

SPLITS003 = ROOT / "experiments" / "003-baselines-eval" / "splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "fov512"
SIZE = 512          # default; --size overrides (fov224 arm)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _one(a):
    src, dst = a[:2]
    size, mode = (a[2], a[3]) if len(a) > 2 else (SIZE, "fov")
    noclobber = bool(a[4]) if len(a) > 4 else False
    existing = None
    real_dst = dst
    if noclobber and os.path.exists(dst):       # never overwrite: re-create next to it, compare pixels, delete
        dst = dst[:-4] + ".chk.png"
    status, info = preprocess.process_image(src, dst, size, mode=mode)
    ok_shape = None
    if status == "ok":
        with Image.open(dst) as im:
            ok_shape = list(im.size) + [im.mode]
            if dst != real_dst:
                with Image.open(real_dst) as old:
                    existing = "equal" if (old.size == im.size and np.array_equal(np.asarray(old.convert("RGB")), np.asarray(im.convert("RGB")))) else "differ"
        if dst != real_dst:
            os.remove(dst)
    return dict(status=status, raw_sha=sha256_file(src), shape=ok_shape, existing=existing, **info)


def manifest_preflight(manifest=ROOT / "data" / "manifest_raw.csv", size=SIZE):
    """Short-side census from the raw manifest dims (latent-diffusion excluded, as in 003)."""
    m = pd.read_csv(manifest, keep_default_na=False)
    m = m[m.generator != "latent-diffusion"].copy()
    m["short"] = m[["width", "height"]].astype(int).min(axis=1)
    small = m[m.short < size]
    by = m.groupby("generator").short.agg(["count", "min", "median"]).reset_index()
    return dict(n_manifest_rows_excl_latent_diffusion=len(m), n_short_side_lt_512=int(len(small)),
                short_side_lt_512_by_generator=small.groupby("generator").size().to_dict(),
                min_short_side=int(m.short.min()), by_generator=by.to_dict("records"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(SPLITS003))
    ap.add_argument("--out-splits", required=True)
    ap.add_argument("--out-dir", default=None, help="default data/processed/fov<size> (crop<size> with --mode crop)")
    ap.add_argument("--size", type=int, default=SIZE, help="output side in px (default 512; 224 for the fov224 arm)")
    ap.add_argument("--mode", choices=["fov", "crop", "du2"], default="fov",
                    help="fov: resize short side to --size then centre-crop (default); crop: plain centre crop (003's crops); "
                         "du2 (Amendment 10): fov resize, resize back up to the original size, then plain centre crop")
    ap.add_argument("--report", required=True)
    ap.add_argument("--no-clobber", action="store_true",
                    help="never overwrite an existing output file: re-create it as <sha>.chk.png, compare pixels with the existing "
                         "file, delete the copy, report n_existing_equal / n_existing_differ and ABORT if any differ. New files are written normally")
    ap.add_argument("--drop-small", action="store_true", help="drop (and report) rows with short side < 512 instead of aborting")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.splits, keep_default_na=False, dtype={"vidx": str})
    missing = [p for p in df.src_path if not os.path.exists(p)]
    keep = df[[os.path.exists(p) for p in df.src_path]].copy().reset_index(drop=True)
    size = a.size
    out_dir = Path(a.out_dir).resolve() if a.out_dir else ROOT / "data" / "processed" / f"{a.mode if a.mode in ('crop', 'du2') else 'fov'}{size}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    dst = [str(out_dir / "images" / f"{s}.png") for s in keep.sha256]
    with mp.Pool(a.workers) as pool:
        res = pool.map(_one, [(sp, d, size, a.mode, a.no_clobber) for sp, d in zip(keep.src_path, dst)], chunksize=16)
    R = pd.DataFrame(res)
    R["sha256"] = keep.sha256.to_numpy()
    R["generator"] = keep.generator.to_numpy()
    R["pool"] = keep.pool.to_numpy()
    R.to_csv(out_dir / f"sidecar_{Path(a.out_splits).stem}.csv", index=False)
    small = R[R.status == "skipped_small"]
    bad = R[R.status == "error"]
    rep = dict(mode=(f"fov (short side -> {size}, PIL BICUBIC antialiased, center crop {size})" if a.mode == "fov"
                     else f"du2 (short side -> {size}, resize back up to the original size, PIL BICUBIC, plain centre crop {size})" if a.mode == "du2"
                     else f"crop (plain centre crop {size}, no resize)"), size=size, n_rows_in=len(df),
               n_missing_raw_dropped=len(missing), n_processed=len(keep),
               n_ok=int((R.status == "ok").sum()), n_short_side_lt_512=int(len(small)), n_error=int(len(bad)),
               short_side_lt_512=[dict(sha256=r.sha256, generator=r.generator, pool=r.pool,
                                       size=[r.orig_width, r.orig_height]) for r in small.itertuples()],
               no_clobber=dict(enabled=a.no_clobber, n_existing_equal=int((R.existing == "equal").sum()),
                               n_existing_differ=int((R.existing == "differ").sum()), n_newly_written=int(((R.existing.isna()) & (R.status == "ok")).sum())),
               raw_sha_match=dict(n_match=int((R.raw_sha == R.sha256).sum()), n_total=len(R)),
               shape_ok_512_rgb=int(sum(s == [size, size, "RGB"] for s in R["shape"] if s is not None)),
               manifest_preflight=manifest_preflight(size=size), counts_pool_split=keep.groupby(["pool", "split"]).size()
               .rename("n").reset_index().to_dict("records"), out_dir=str(out_dir.relative_to(ROOT)),
               seconds=time.time() - t0)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("manifest_preflight",)}, indent=1))
    if len(bad):
        raise SystemExit(f"{len(bad)} images failed: {bad.sha256.tolist()[:3]} ...")
    if rep["raw_sha_match"]["n_match"] != len(R):
        raise SystemExit("raw sha256 mismatch: the raw files differ from the splits file")
    if a.no_clobber and int((R.existing == "differ").sum()):
        raise SystemExit(f"ABORT: {int((R.existing == 'differ').sum())} existing output files differ from the re-created ones "
                         "(left untouched; see the report). Ask the owner before overwriting anything.")
    if len(small) and not a.drop_small:
        raise SystemExit(f"ABORT: {len(small)} images have short side < {size} (would need upscaling; see report). "
                         "Re-run with --drop-small to exclude them, or decide a policy.")
    ok = R.status == "ok"
    out = preprocess.rewrite_paths(keep[ok.to_numpy()], out_dir, root=ROOT)
    out.to_csv(a.out_splits, index=False)
    print("wrote", a.out_splits, len(out), "rows")


if __name__ == "__main__":
    main()
