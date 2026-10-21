"""Manifest schema + seeded splits (held-out generator, camera-grouped reals)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["path", "label", "source_dataset", "generator", "camera_or_platform",
           "license", "split", "sha256"]
SPLITS = ("train", "val", "test", "calib")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_row(path, label, source_dataset, generator, camera_or_platform, license, split="", sha256=None):
    """One manifest row. Reals must have generator == 'real'."""
    return dict(path=str(path), label=int(label), source_dataset=source_dataset,
                generator=generator, camera_or_platform=camera_or_platform, license=license,
                split=split, sha256=sha256 or sha256_file(path))


def write_manifest(df, path):
    df[COLUMNS + [c for c in df.columns if c not in COLUMNS]].to_csv(path, index=False)


def read_manifest(path):
    df = pd.read_csv(path, keep_default_na=False, dtype={"sha256": str, "split": str})
    validate_manifest(df)
    return df


def validate_manifest(df):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns {missing}")
    if not set(df.label.unique()) <= {0, 1}:
        raise ValueError("label must be 0 (real) or 1 (fake)")
    if ((df.label == 0) != (df.generator == "real")).any():
        raise ValueError("generator must be 'real' exactly for label 0")
    bad = set(df.split.unique()) - set(SPLITS) - {""}
    if bad:
        raise ValueError(f"unknown split values {bad}")
    if df.path.duplicated().any():
        raise ValueError("duplicate paths in manifest")


def _partition(idx, fracs, rng):
    """Randomly partition idx into named parts with given fractions (order of fracs dict = fill order,
    last part gets the remainder). Parts with frac>0 get >=1 item when len(idx)>=4."""
    idx = np.array(sorted(idx))
    rng.shuffle(idx)
    n, out, start = len(idx), {}, 0
    names = list(fracs)
    for i, k in enumerate(names):
        if i == len(names) - 1:
            cnt = n - start
        else:
            cnt = int(round(fracs[k] * n))
            if n >= 4 and fracs[k] > 0:
                cnt = max(cnt, 1)
            cnt = min(cnt, n - start)
        out[k] = idx[start:start + cnt]
        start += cnt
    return out


def assign_splits(df, holdout_generator, seed=0, val_frac=0.15, test_frac=0.25, calib_frac=0.10):
    """Return a copy of df with `split` filled in.

    * fakes of `holdout_generator` -> ALL in 'test' (never train/val/calib).
    * other fakes -> random train/val/test (stratified per generator).
    * reals -> GROUP split by camera_or_platform: whole cameras go to the test side (~test_frac of reals)
      or the train side; no camera appears in both. Train-side reals are split train/val/calib per image
      (calib = real set used for FPR-calibrated thresholds; it shares cameras with train, which is the
      honest deployment situation: calibrate on known cameras, test on unseen ones).
    """
    validate_manifest(df)
    df = df.copy()
    rng = np.random.default_rng(seed)
    fake_gens = sorted(set(df.generator[df.label == 1]))
    if holdout_generator not in fake_gens:
        raise ValueError(f"holdout generator {holdout_generator!r} not in manifest ({fake_gens})")
    split = pd.Series("", index=df.index, dtype=object)
    split[(df.label == 1) & (df.generator == holdout_generator)] = "test"
    for g in fake_gens:
        if g == holdout_generator:
            continue
        idx = df.index[(df.label == 1) & (df.generator == g)].to_numpy()
        parts = _partition(idx, {"val": val_frac, "test": test_frac, "train": 1.0}, rng)
        for k, v in parts.items():
            split[v] = k
    real = df[df.label == 0]
    cam = real.camera_or_platform.astype(str).replace({"": "unknown", "nan": "unknown"})
    counts = cam.value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("need >=2 distinct camera_or_platform groups among reals for a group split")
    order = list(counts.index)
    rng.shuffle(order)
    test_groups, n_test = [], 0
    for gname in order:
        if test_groups and n_test >= test_frac * len(real):
            break
        if len(test_groups) == len(order) - 1:
            break
        test_groups.append(gname)
        n_test += int(counts[gname])
    is_test = cam.isin(test_groups)
    split[is_test[is_test].index] = "test"
    parts = _partition(is_test[~is_test].index.to_numpy(),
                       {"val": val_frac, "calib": calib_frac, "train": 1.0}, rng)
    for k, v in parts.items():
        split[v] = k
    df["split"] = split
    check_splits(df, holdout_generator)
    return df


def check_splits(df, holdout_generator=None):
    """Raise AssertionError if leakage invariants are violated."""
    assert (df.split != "").all(), "unassigned rows"
    if holdout_generator is not None:
        h = df[df.generator == holdout_generator]
        assert len(h) and (h.split == "test").all(), "held-out generator outside test"
    real = df[df.label == 0]
    cam = real.camera_or_platform.astype(str).replace({"": "unknown", "nan": "unknown"})
    tr = set(cam[real.split.isin(["train", "val", "calib"])])
    te = set(cam[real.split == "test"])
    assert not (tr & te), f"camera overlap between train-side and test: {tr & te}"
    assert (df[df.label == 1].split != "calib").all(), "fakes must not be in calib"
