"""Loader + constants for the Phase 4 DINOv2-base feature cache (no torch import; safe to import from 005-007).

    import sys; sys.path.insert(0, "<repo>/experiments/004-phase4-features")
    from p4features import load_features, load_features_aligned
    shas, X = load_features("primary", "jpeg75_resize05")          # X: float32 [n, 1536]
    X = load_features_aligned(splits_df, "secondary", "clean")     # float32 [len(df), 4608], NaN where not cached

Layout of the stored (secondary) matrix, hidden size H = 768, columns:
    [blk4 CLS | blk4 mean | blk8 CLS | blk8 mean | blk12 CLS | blk12 mean]      (6H = 4608)
'primary' is the LAST 2H columns (= block 12 = hidden_states[-1], CLS + mean): the same tensor B0 (003) uses,
sliced from the same forward pass, so no second pass is needed.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache" / "feats"
SPEC = "hf:dinov2:facebook/dinov2-base"
SEED = 0
INPUT_SIZE = 224
SECONDARY_LAYERS = (4, 8, 12)
PRIMARY_CFG = dict(layers=[-1], use_cls=True, use_mean=True, input_size=INPUT_SIZE, crop_mode="crop")
EVAL_CONDITIONS = ["clean", "jpeg75_resize05", "social_sim", "resize(0.5)", "jpeg(75)", "jpeg(90)",
                   "mavali_medium", "screenshot_sim"]
HELD_OUT_CONDITIONS = ["mavali_medium", "screenshot_sim"]          # never used for fitting anything
GRID_CONDITIONS = [f"jr({q},{f})" for q in (60, 75, 90) for f in (0.5, 0.75, 1.0)]   # split == 'train' only
ALL_CONDITIONS = EVAL_CONDITIONS + GRID_CONDITIONS


def safe(cond):
    return "".join(c if c.isalnum() or c in "-_.+()" else "_" for c in cond)


def cond_dir(cond, cache_dir=None):
    return Path(cache_dir or CACHE) / safe(cond)


def eval_mask(df):
    """Rows that get the 8 eval conditions: in_scoring_set or split in (train, calib) (= all 2,919 rows of splits.csv)."""
    return (df.in_scoring_set.astype(bool) | df.split.isin(["train", "calib"])).to_numpy()


def rows_mask(df, cond):
    """Which rows of splits.csv are extracted under `cond` (grid conditions: split == 'train' only)."""
    if cond in GRID_CONDITIONS:
        return (df.split == "train").to_numpy()
    if cond in EVAL_CONDITIONS:
        return eval_mask(df)
    raise ValueError(f"unknown condition {cond!r}")


def read_shards(cond, cache_dir=None):
    """-> (keys: str array, feats: float32 [n, 6H]) merged over all shards of a condition (later shard wins on dup)."""
    d = cond_dir(cond, cache_dir)
    parts = sorted(glob.glob(str(d / "part_*.npz")))
    if not parts:
        raise FileNotFoundError(f"no cached features for condition {cond!r} in {d}; run run.py first")
    store = {}
    for p in parts:
        z = np.load(p, allow_pickle=False)
        store.update(zip(z["keys"].tolist(), z["feats"]))
    keys = np.array(sorted(store))
    return keys, np.stack([store[k] for k in keys]).astype(np.float32)


def load_features(feature, condition, cache_dir=None):
    """feature: 'primary' (1536-d, = B0 features) | 'secondary' (4608-d, blocks 4/8/12). -> (sha256 array, matrix)."""
    keys, X = read_shards(condition, cache_dir)
    if feature == "secondary":
        return keys, X
    if feature == "primary":
        return keys, np.ascontiguousarray(X[:, -(X.shape[1] // 3):])
    raise ValueError("feature must be 'primary' or 'secondary'")


def load_features_aligned(df, feature, condition, cache_dir=None):
    """Matrix aligned to df rows (df needs a sha256 column); rows without cached features are NaN."""
    keys, X = load_features(feature, condition, cache_dir)
    pos = {k: i for i, k in enumerate(keys.tolist())}
    out = np.full((len(df), X.shape[1]), np.nan, np.float32)
    for r, s in enumerate(df.sha256.tolist()):
        i = pos.get(s)
        if i is not None:
            out[r] = X[i]
    return out


def write_shard(cond, keys, feats, cache_dir=None):
    d = cond_dir(cond, cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    n = len(glob.glob(str(d / "part_*.npz")))
    tmp = d / f"tmp_part_{n:05d}.npz"
    np.savez(tmp, keys=np.array(list(keys)), feats=np.asarray(feats, np.float32))
    os.replace(tmp, d / f"part_{n:05d}.npz")


def cached_keys(cond, cache_dir=None):
    d = cond_dir(cond, cache_dir)
    out = set()
    for p in glob.glob(str(d / "part_*.npz")):
        out.update(np.load(p, allow_pickle=False)["keys"].tolist())
    return out


def check_meta(cond, meta, cache_dir=None):
    """Write meta.json on first use; refuse to mix shards produced under a different backbone/config."""
    p = cond_dir(cond, cache_dir) / "meta.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != meta:
            raise SystemExit(f"cache config mismatch for {cond!r} ({p}); delete that directory to recompute.\n"
                             f"cached: {old}\ncurrent: {meta}")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=1, sort_keys=True))
