"""PILOT prep (Amendment 2): regenerate the 512 crops for the rows of 003 splits.csv whose raw file exists.

  env/.venv/bin/python -I experiments/004-phase4-features/pilot_prep.py

Uses EXACTLY 003 stage_prep's per-image call: preprocess.process_image(src_path, <splits path>, 512)
(same lib function, same args; 4-process pool like 003). Outputs (git-ignored? no: small, tracked):
  pilot/splits_pilot.csv    subset of 003 splits.csv, same columns, same values
  pilot/prep_report.json    counts, checks, seconds

Checks (see notes):
  * raw_sha_match: sha256(raw file at src_path) == row sha256.  NOTE: in 003, `sha256` is the hash of the ORIGINAL file
    (it names the crop file and keys the sidecar); it is NOT the hash of the crop. So sha256(crop) == row sha256 can
    never hold; the raw-file hash is the meaningful data-integrity check.
  * format_match: PIL-detected format of the raw file (this machine) == row orig_format (detected on the Mac).
  * crop: 512x512 RGB PNG; its sha256 is recorded in prep_report.json (crop_sha256) as a reference for later machines.
"""
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

SPLITS003 = ROOT / "experiments" / "003-baselines-eval" / "splits.csv"
OUT = HERE / "pilot"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _one(a):
    from preprocess import process_image      # same function + args as 003 stage_prep._proc_one
    src, dst, row_sha = a
    status, info = process_image(src, dst, 512)
    out = dict(status=status, orig_format_here=info.get("orig_format"), raw_sha=sha256_file(src), crop_sha=None,
               crop_size=None)
    if status == "ok":
        out["crop_sha"] = sha256_file(dst)
        with Image.open(dst) as im:
            out["crop_size"] = list(im.size) + [im.mode]
    return out


def main():
    t0 = time.time()
    df = pd.read_csv(SPLITS003, keep_default_na=False, dtype={"vidx": str})
    keep = df[[os.path.exists(p) for p in df.src_path]].copy().reset_index(drop=True)
    for p in keep.path.unique()[:1]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    jobs = [(r.src_path, r.path, r.sha256) for r in keep.itertuples()]
    with mp.Pool(4) as pool:
        res = pool.map(_one, jobs, chunksize=16)
    R = pd.DataFrame(res)
    keep["raw_sha_ok"] = (R.raw_sha == keep.sha256).to_numpy()
    keep["format_ok"] = (R.orig_format_here == keep.orig_format).to_numpy()
    keep["status"] = R.status.to_numpy()
    keep["crop_ok"] = [s == "ok" and c == [512, 512, "RGB"] for s, c in zip(R.status, R.crop_size)]
    cols = list(df.columns)
    OUT.mkdir(exist_ok=True)
    keep[cols].to_csv(OUT / "splits_pilot.csv", index=False)
    n = len(keep)
    bad_sha = keep[~keep.raw_sha_ok]
    bad_fmt = keep[~keep.format_ok]
    rep = dict(
        label="PILOT", n_splits003=len(df), n_kept=n, n_dropped_missing_raw=len(df) - n,
        n_status_ok=int((keep.status == "ok").sum()), n_crop_ok_512_rgb=int(keep.crop_ok.sum()),
        raw_sha_match=dict(n_match=int(keep.raw_sha_ok.sum()), n_total=n,
                           mismatches=[dict(src_path=r.src_path, orig_format=r.orig_format, generator=r.generator)
                                       for r in bad_sha.itertuples()]),
        format_match=dict(n_match=int(keep.format_ok.sum()), n_total=n,
                          mismatches=[dict(src_path=r.src_path, row_orig_format=r.orig_format, here=f)
                                      for r, f in zip(bad_fmt.itertuples(), R.orig_format_here[~keep.format_ok])]),
        crop_sha_equals_row_sha=dict(n_match=int((R.crop_sha == keep.sha256).sum()), n_total=n,
                                     note="expected ~0: row sha256 is the ORIGINAL file's hash in 003 (names the crop)"),
        crop_sha256=dict(zip(keep.sha256, R.crop_sha)),
        counts_pool_split=keep.groupby(["pool", "split"]).size().rename("n").reset_index().to_dict("records"),
        counts_generator_split=keep.groupby(["generator", "split"]).size().rename("n").reset_index().to_dict("records"),
        orig_format_by_generator=keep.groupby(["generator", "orig_format"]).size().rename("n").reset_index().to_dict("records"),
        seconds=time.time() - t0)
    (OUT / "prep_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("crop_sha256",)}, indent=1))


if __name__ == "__main__":
    main()
