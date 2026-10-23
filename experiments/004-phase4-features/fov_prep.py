"""fov512 preprocessing (D-013 / Amendment 3): resize the SHORT side to 512 (bicubic, antialiased), then center-crop 512.

  # PILOT (VISION + 6 Synthbuster rows of splits_pilot.csv)
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py \
      --splits experiments/004-phase4-features/pilot/splits_pilot.csv \
      --out-splits experiments/004-phase4-features/pilot/splits_pilot_fov512.csv \
      --report experiments/004-phase4-features/pilot/fov512_prep_report.json
  # OFFICIAL (all rows of 003 splits.csv whose raw file exists; run after env/bootstrap_data.py re-fetched RAISE etc.)
  env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py \
      --out-splits experiments/004-phase4-features/splits_fov512.csv --report experiments/004-phase4-features/fov512_prep_report.json

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

import pandas as pd
from PIL import Image

import preprocess

SPLITS003 = ROOT / "experiments" / "003-baselines-eval" / "splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "fov512"
SIZE = 512


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _one(a):
    src, dst = a
    status, info = preprocess.process_image(src, dst, SIZE, mode="fov")
    ok_shape = None
    if status == "ok":
        with Image.open(dst) as im:
            ok_shape = list(im.size) + [im.mode]
    return dict(status=status, raw_sha=sha256_file(src), shape=ok_shape, **info)


def manifest_preflight(manifest=ROOT / "data" / "manifest_raw.csv"):
    """Short-side census from the raw manifest dims (latent-diffusion excluded, as in 003)."""
    m = pd.read_csv(manifest, keep_default_na=False)
    m = m[m.generator != "latent-diffusion"].copy()
    m["short"] = m[["width", "height"]].astype(int).min(axis=1)
    small = m[m.short < SIZE]
    by = m.groupby("generator").short.agg(["count", "min", "median"]).reset_index()
    return dict(n_manifest_rows_excl_latent_diffusion=len(m), n_short_side_lt_512=int(len(small)),
                short_side_lt_512_by_generator=small.groupby("generator").size().to_dict(),
                min_short_side=int(m.short.min()), by_generator=by.to_dict("records"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(SPLITS003))
    ap.add_argument("--out-splits", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--report", required=True)
    ap.add_argument("--drop-small", action="store_true", help="drop (and report) rows with short side < 512 instead of aborting")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.splits, keep_default_na=False, dtype={"vidx": str})
    missing = [p for p in df.src_path if not os.path.exists(p)]
    keep = df[[os.path.exists(p) for p in df.src_path]].copy().reset_index(drop=True)
    out_dir = Path(a.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    dst = [str(out_dir / "images" / f"{s}.png") for s in keep.sha256]
    with mp.Pool(a.workers) as pool:
        res = pool.map(_one, list(zip(keep.src_path, dst)), chunksize=16)
    R = pd.DataFrame(res)
    R["sha256"] = keep.sha256.to_numpy()
    R["generator"] = keep.generator.to_numpy()
    R["pool"] = keep.pool.to_numpy()
    R.to_csv(out_dir / f"sidecar_{Path(a.out_splits).stem}.csv", index=False)
    small = R[R.status == "skipped_small"]
    bad = R[R.status == "error"]
    rep = dict(mode="fov (short side -> 512, PIL BICUBIC antialiased, center crop 512)", n_rows_in=len(df),
               n_missing_raw_dropped=len(missing), n_processed=len(keep),
               n_ok=int((R.status == "ok").sum()), n_short_side_lt_512=int(len(small)), n_error=int(len(bad)),
               short_side_lt_512=[dict(sha256=r.sha256, generator=r.generator, pool=r.pool,
                                       size=[r.orig_width, r.orig_height]) for r in small.itertuples()],
               raw_sha_match=dict(n_match=int((R.raw_sha == R.sha256).sum()), n_total=len(R)),
               shape_ok_512_rgb=int(sum(s == [SIZE, SIZE, "RGB"] for s in R["shape"] if s is not None)),
               manifest_preflight=manifest_preflight(), counts_pool_split=keep.groupby(["pool", "split"]).size()
               .rename("n").reset_index().to_dict("records"), out_dir=str(out_dir.relative_to(ROOT)),
               seconds=time.time() - t0)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("manifest_preflight",)}, indent=1))
    if len(bad):
        raise SystemExit(f"{len(bad)} images failed: {bad.sha256.tolist()[:3]} ...")
    if rep["raw_sha_match"]["n_match"] != len(R):
        raise SystemExit("raw sha256 mismatch: the raw files differ from the splits file")
    if len(small) and not a.drop_small:
        raise SystemExit(f"ABORT: {len(small)} images have short side < 512 (would need upscaling; see report). "
                         "Re-run with --drop-small to exclude them, or decide a policy.")
    ok = R.status == "ok"
    out = preprocess.rewrite_paths(keep[ok.to_numpy()], out_dir, root=ROOT)
    out.to_csv(a.out_splits, index=False)
    print("wrote", a.out_splits, len(out), "rows")


if __name__ == "__main__":
    main()
