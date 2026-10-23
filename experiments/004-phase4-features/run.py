"""Phase 4 shared feature extraction: DINOv2-base primary (= B0) + secondary (blocks 4/8/12) features.

  env/.venv/bin/python -I experiments/004-phase4-features/run.py                 # everything (resumable)
  env/.venv/bin/python -I experiments/004-phase4-features/run.py --conditions clean "jr(60,0.5)"
  env/.venv/bin/python -I experiments/004-phase4-features/run.py --dry-run       # synthetic smoke test, tiny random model
  env/.venv/bin/python -I experiments/004-phase4-features/run.py --verify003     # primary clean == 003 B0 cache

One forward pass per image/condition with layers=(4,8,12), CLS+mean -> 4608-d, stored as float32 shards in
cache/feats/<condition>/part_*.npz keyed by sha256. Primary = last 1536 columns (hidden_states[12] == hidden_states[-1]).
Reads experiments/003-baselines-eval/splits.csv; images from the `path` column (data/processed/crop512/...).
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

import p4features as P

BATCH = 20            # same as 003 (MPS_BATCH) so batch composition matches
CHUNK = 240           # images per shard (12 batches); the resume granularity
FEAT_TOL = 1e-3       # D-008 analogue on features: max_j |dF_j| / max(1, max_i |F_cpu_ij|)  (scaled, see notes.md)
LOGIT_TOL = 1e-3      # D-008 on the B0 probe decision logit (identical to 003)
N_GUARD = 32
SPLITS = ROOT / "experiments" / "003-baselines-eval" / "splits.csv"
CFG003 = ROOT / "experiments" / "003-baselines-eval" / "config.json"
CACHE003 = ROOT / "experiments" / "003-baselines-eval" / "cache" / "b0" / "hf-dinov2-facebook_dinov2-base_clean.npz"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_splits(path=SPLITS):
    return pd.read_csv(path, keep_default_na=False, dtype={"vidx": str}).astype(
        {"in_scoring_set": bool, "b0_train": bool})


def weights_cached():
    return any((Path(os.environ["HF_HOME"]) / "hub" / "models--facebook--dinov2-base" / "snapshots").glob("*/config.json"))


# ----------------------------------------------------------------------------- backbones
def secondary_backbone(model=None, name="hf-dinov2-facebook_dinov2-base"):
    from features import IMAGENET, Backbone, load_backbone
    if model is None:
        return load_backbone(P.SPEC, input_size=P.INPUT_SIZE, layers=P.SECONDARY_LAYERS, use_cls=True, use_mean=True)
    return Backbone(name, model, "hf", *IMAGENET, input_size=P.INPUT_SIZE, crop_mode="crop",
                    layers=P.SECONDARY_LAYERS, use_cls=True, use_mean=True)


def primary_backbone(bb):
    """B0-identical config (layers=(-1,), CLS+mean, 224 center crop) sharing bb's weights."""
    from features import Backbone
    return Backbone(bb.name, bb.model, "hf", bb.mean, bb.std, bb.input_size, "crop", (-1,), True, True)


def check_design_vs_003(bb):
    """By design: the primary pipeline must be B0's. Compare against 003/config.json and the Backbone defaults."""
    pb = primary_backbone(bb)
    cfg = pb.config()
    want = dict(layers=P.PRIMARY_CFG["layers"], use_cls=True, use_mean=True, input_size=224, crop_mode="crop")
    assert {k: cfg[k] for k in want} == want, cfg
    if CFG003.exists():
        b0 = json.loads(CFG003.read_text())["b0"]
        assert (b0["backbone"], b0["input_size"], b0["layers"], b0["use_cls"], b0["use_mean"]) == \
               (P.SPEC, 224, [-1], True, True), b0
        log("design check: primary config == 003 b0 config (backbone, input_size, layers, use_cls, use_mean)")
    assert bb.layers[-1] == 12 and bb.use_cls and bb.use_mean and bb.input_size == 224
    return pb


def meta_for(bb, spec):
    return dict(spec=spec, backbone=bb.config(), seed=P.SEED, batch=BATCH, stored="blocks4/8/12 CLS+mean float32",
                primary="last 2H columns == layers=(-1,) CLS+mean", cond_seed=P.SEED)


# ----------------------------------------------------------------------------- extraction
def load_cond(path, sha, cond):
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        out = apply_condition(im, cond, key=sha, seed=P.SEED)
    if min(out.size) < P.INPUT_SIZE:   # features.Backbone.to_tensor would raise; fail early with context
        raise ValueError(f"{cond}: {path} -> {out.size} is smaller than the {P.INPUT_SIZE} center crop")
    return out


def extract_condition(df, bb, cond, cache, device, pool, spec, limit=None):
    from features import embed_images
    m = P.rows_mask(df, cond)
    rows = df[m]
    if limit:
        rows = rows.iloc[:limit]
    P.check_meta(cond, meta_for(bb, spec), cache)
    have = P.cached_keys(cond, cache)
    todo = rows[~rows.sha256.isin(have)]
    t0 = time.time()
    for s in range(0, len(todo), CHUNK):
        ch = todo.iloc[s:s + CHUNK]
        imgs = list(pool.map(lambda r: load_cond(r[0], r[1], cond), zip(ch.path, ch.sha256)))
        f = embed_images(bb, imgs, BATCH, device)
        if getattr(bb, "force_cpu", False) and device != "cpu":
            raise SystemExit(f"{cond}: accelerator forward failed and fell back to CPU; aborting (D-010)")
        P.write_shard(cond, ch.sha256.tolist(), f, cache)
        log(f"  {cond}: {min(s + CHUNK, len(todo))}/{len(todo)} new ({len(rows) - len(todo)} cached)")
    dt = time.time() - t0
    return dict(n_rows=len(rows), n_new=len(todo), seconds=dt)


# ----------------------------------------------------------------------------- D-008 guard
def guard_subset(df, cond):
    """Fixed 32 images (16 real + 16 fake, seed 0, sorted shas). Eval conditions: same rule as 003's check_subset
    (in_scoring_set rows). Grid conditions: split == 'train' rows."""
    s = df[df.split == "train"] if cond in P.GRID_CONDITIONS else df[df.in_scoring_set]
    r, f = sorted(s.sha256[s.label == 0]), sorted(s.sha256[s.label == 1])
    rng = np.random.default_rng(P.SEED)
    k = N_GUARD // 2
    return ([r[i] for i in rng.choice(len(r), min(k, len(r)), replace=False)] +
            [f[i] for i in rng.choice(len(f), min(k, len(f)), replace=False)])


def b0_probe(df, cache):
    """The B0 probe (003 recipe: fit_probe on b0_train clean primary features, seed 0) for the logit guard."""
    import probe
    try:
        X = P.load_features_aligned(df, "primary", "clean", cache)
    except FileNotFoundError:
        return None
    tr = df.b0_train.to_numpy()
    if not tr.any() or np.isnan(X[tr]).any():
        return None
    return probe.fit_probe(X[tr], df.label.to_numpy()[tr], P.SEED)


def run_guard(df, bb, cond, cache, model, device):
    """Accelerator features (from the cache) vs CPU recomputation (TF32 off) on the fixed subset."""
    from features import embed_images
    sub = guard_subset(df, cond)
    keys, X = P.read_shards(cond, cache)
    pos = {k: i for i, k in enumerate(keys.tolist())}
    sub = [s for s in sub if s in pos]
    if not sub:
        log(f"  guard {cond}: no subset image cached, skipped")
        return None
    cp = Path(cache) / "cpuref" / f"{P.safe(cond)}.npz"
    ref = {}
    if cp.exists():
        z = np.load(cp, allow_pickle=False)
        ref = dict(zip(z["keys"].tolist(), z["feats"]))
    miss = [s for s in sub if s not in ref]
    if miss:
        d = df.set_index("sha256", drop=False).loc[miss]
        imgs = [load_cond(p, s, cond) for p, s in zip(d.path, d.sha256)]
        f = embed_images(bb, imgs, 4, "cpu")
        ref.update(zip(miss, f))
        cp.parent.mkdir(parents=True, exist_ok=True)
        ks = sorted(ref)
        np.savez(cp.with_name("tmp_" + cp.name), keys=np.array(ks), feats=np.stack([ref[k] for k in ks]))
        os.replace(cp.with_name("tmp_" + cp.name), cp)
    A = np.stack([X[pos[s]] for s in sub])
    C = np.stack([ref[s] for s in sub])
    d = np.abs(A - C)
    scaled = float((d / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max())
    out = dict(n=len(sub), device=str(device), max_abs_feature_diff=float(d.max()), max_scaled_feature_diff=scaled,
               feat_tol=FEAT_TOL, primary_max_abs_diff=float(d[:, -(d.shape[1] // 3):].max()))
    if model is not None:
        h = A.shape[1] // 3
        out["max_abs_dlogit"] = float(np.abs(model.decision_function(A[:, -h:]) - model.decision_function(C[:, -h:])).max())
        out["logit_tol"] = LOGIT_TOL
    out["ok"] = bool(scaled <= FEAT_TOL and out.get("max_abs_dlogit", 0.0) <= LOGIT_TOL)
    log(f"  D-008 {cond}: max|dF|={out['max_abs_feature_diff']:.2e} scaled={scaled:.2e}"
        + (f" max|dlogit|={out['max_abs_dlogit']:.2e}" if model is not None else " (no probe yet)")
        + f" -> {'ok' if out['ok'] else 'FAIL'}")
    return out


def update_json(path, key, val):
    p = Path(path)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[key] = val
    p.write_text(json.dumps(cur, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- driver
def run_extraction(df, bb, conditions, cache, device, spec, guard=True, limit=None, workers=4):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    probe_model = None
    with ThreadPoolExecutor(workers) as pool:
        for cond in conditions:
            log(f"condition {cond}")
            t = extract_condition(df, bb, cond, cache, device, pool, spec, limit)
            update_json(cache / "timings.json", cond, t)
            if cond == "clean" or probe_model is None:
                probe_model = b0_probe(df, cache) if guard else None
            if guard:
                g = run_guard(df, bb, cond, cache, probe_model, device)
                if g is not None:
                    update_json(cache / "d008_guard.json", cond, g)
                    if not g["ok"]:
                        raise SystemExit(f"D-008 ABORT: {cond}: {g}")
    return probe_model


def selfcheck_primary(df, bb, pb, device, n=8):
    """primary == slice of the secondary forward (same weights, same batch): must match to float rounding."""
    from features import embed_images
    rows = df.iloc[:n]
    imgs = [load_cond(p, s, "clean") for p, s in zip(rows.path, rows.sha256)]
    a = embed_images(bb, imgs, BATCH, device)
    b = embed_images(pb, imgs, BATCH, device)
    d = float(np.abs(a[:, -b.shape[1]:] - b).max())
    log(f"self-check: primary-config forward vs secondary[:, -{b.shape[1]}:] max|diff|={d:.2e}")
    if d > 1e-5:
        raise SystemExit(f"primary slice != layers=(-1,) forward (diff {d:.3e})")


def verify003(df, cache):
    """Our clean primary features vs the 003 B0 cache (same images). Needs both caches."""
    if not CACHE003.exists():
        print(f"SKIP verify003: {CACHE003} not found (003 cache is git-ignored; produced by 003 run.py b0)")
        return
    z = np.load(CACHE003, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    assert cfg["layers"] == [-1] and cfg["use_cls"] and cfg["use_mean"] and cfg["input_size"] == 224, cfg
    k3 = dict(zip(z["keys"].tolist(), z["feats"]))
    keys, X = P.load_features("primary", "clean", cache)
    common = [i for i, k in enumerate(keys.tolist()) if k in k3]
    A = X[common]
    B = np.stack([k3[keys[i]] for i in common])
    d = np.abs(A - B)
    sc = float((d / np.maximum(1.0, np.abs(B).max(0, keepdims=True))).max())
    print(f"verify003: {len(common)} common images, max|dF|={d.max():.2e}, scaled={sc:.2e} (tol {FEAT_TOL})")
    if sc > FEAT_TOL:
        raise SystemExit("verify003 FAILED")
    print("verify003 OK")


# ----------------------------------------------------------------------------- dry run
def dry_run():
    """4 synthetic 512x512 images, tiny RANDOM Dinov2 (12 layers, hidden 48): exercises extraction, resume, loaders,
    conditions (incl. screenshot_sim and grid), the CUDA/CPU guard and the primary-slice identity. No real data/weights."""
    import torch
    from transformers import Dinov2Config, Dinov2Model
    import hashlib
    scratch = HERE / "_scratch" / f"dryrun_{os.getpid()}"
    scratch.mkdir(parents=True)
    try:
        rng = np.random.default_rng(0)
        rows = []
        for i in range(4):
            yy, xx = np.mgrid[0:512, 0:512] / 512
            base = np.stack([xx * rng.uniform(100, 255), yy * rng.uniform(100, 255), (xx + yy) * 100], -1)
            arr = np.clip(base + rng.normal(0, 8 + 10 * (i % 2), base.shape), 0, 255).astype(np.uint8)
            p = scratch / f"im{i}.png"
            Image.fromarray(arr).save(p)
            rows.append(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), path=str(p), label=i % 2,
                             split="train", in_scoring_set=True, b0_train=True, generator="g", pool="p"))
        df = pd.DataFrame(rows)
        torch.manual_seed(0)
        model = Dinov2Model(Dinov2Config(hidden_size=48, num_hidden_layers=12, num_attention_heads=2,
                                         intermediate_size=96, image_size=224, patch_size=14))
        from features import get_device
        dev = get_device().type
        bb = secondary_backbone(model, name="tiny-dinov2")
        pb = check_design_vs_003(bb)
        selfcheck_primary(df, bb, pb, dev, n=4)
        conds = ["clean", "jpeg75_resize05", "social_sim", "resize(0.5)", "mavali_medium", "screenshot_sim",
                 "jr(60,0.5)", "jr(90,1.0)"]
        cache = scratch / "cache"
        run_extraction(df, bb, conds, cache, dev, "tiny-random-dinov2")
        n0 = bb.n_forward
        run_extraction(df, bb, conds, cache, dev, "tiny-random-dinov2", guard=False)   # resume: nothing recomputed
        assert bb.n_forward == n0, "resume recomputed features"
        for c in conds:
            ks, X = P.load_features("secondary", c, cache)
            kp, Xp = P.load_features("primary", c, cache)
            assert X.shape == (4, 6 * 48) and Xp.shape == (4, 2 * 48) and np.isfinite(X).all()
            assert np.array_equal(Xp, X[:, -96:]) and (ks == kp).all()
            Xa = P.load_features_aligned(df, "primary", c, cache)
            assert np.allclose(Xa, Xp[[kp.tolist().index(s) for s in df.sha256]])
        # primary stored == fresh forward with the layers=(-1,) config on the clean images
        from features import embed_images
        imgs = [load_cond(p, s, "clean") for p, s in zip(df.path, df.sha256)]
        fresh = embed_images(pb, imgs, BATCH, dev)
        _, Xp = P.load_features("primary", "clean", cache)
        order = [P.load_features("primary", "clean", cache)[0].tolist().index(s) for s in df.sha256]
        assert np.abs(Xp[order] - fresh).max() < 1e-5
        g = json.loads((cache / "d008_guard.json").read_text())
        assert all(v["ok"] for v in g.values()) and len(g) == len(conds)
        assert not np.allclose(P.load_features("primary", "clean", cache)[1], P.load_features("primary", "screenshot_sim", cache)[1])
        # real-weights self-test if cached
        if weights_cached():
            log("real DINOv2 weights are cached: running primary-vs-slice self-check on 4 images")
            rb = secondary_backbone()
            selfcheck_primary(df, rb, primary_backbone(rb), dev, n=4)
        else:
            print("SKIP real-weights check: data/hf_cache has no facebook/dinov2-base yet")
        print(f"DRY-RUN OK (device {dev}, guard max scaled diff "
              f"{max(v['max_scaled_feature_diff'] for v in g.values()):.2e})")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            (HERE / "_scratch").rmdir()
        except OSError:
            pass


def write_config(args, splits=None, cache=None):
    cfg = dict(spec=P.SPEC, input_size=P.INPUT_SIZE, primary=P.PRIMARY_CFG, secondary_layers=list(P.SECONDARY_LAYERS),
               secondary_stored="cls+mean per block, 6x768 = 4608-d float32 (primary = last 1536 columns)",
               eval_conditions=P.EVAL_CONDITIONS, held_out_conditions=P.HELD_OUT_CONDITIONS,
               grid_conditions=P.GRID_CONDITIONS, grid_rows="split == 'train'",
               eval_rows="in_scoring_set or split in (train, calib)", seed=P.SEED, batch=BATCH, chunk=CHUNK,
               d008=dict(n=N_GUARD, feat_tol_scaled=FEAT_TOL, logit_tol=LOGIT_TOL, tf32="disabled (features.get_device)",
                         cpu_batch=4), hf_home="data/hf_cache (offline)", splits=str(Path(splits or SPLITS).resolve().relative_to(ROOT)),
               cache=str(Path(cache or P.CACHE).resolve().relative_to(ROOT)) + "/<condition>/part_*.npz (git-ignored)",
               loader="p4features.load_features(feature, condition)" + (f" [subdir={args.cache_subdir!r}]" if args.cache_subdir else ""))
    if args.cache_subdir or args.splits:     # pilot / custom runs never overwrite the main config.json
        out = HERE / "pilot" / "config_pilot.json" if args.cache_subdir == "pilot" else HERE / f"config_{args.cache_subdir or 'custom'}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(cfg, indent=1))
        return
    (HERE / "config.json").write_text(json.dumps(cfg, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of conditions (default: all 17)")
    ap.add_argument("--no-guard", action="store_true", help="skip D-008 CPU guard (not recommended)")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows per condition (debug)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify003", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", default=None, help="splits CSV (default: 003 splits.csv); e.g. pilot/splits_pilot.csv")
    ap.add_argument("--cache-subdir", default=None,
                    help="write/read features in cache/<subdir> instead of cache/feats (e.g. 'pilot')")
    a = ap.parse_args()
    splits = Path(a.splits).resolve() if a.splits else SPLITS
    cache = P.cache_root(a.cache_subdir)
    if a.dry_run:
        return dry_run()
    write_config(a, splits, cache)
    if a.verify003:
        return verify003(load_splits(splits), cache)
    if not weights_cached():
        print("SKIP: facebook/dinov2-base is not in data/hf_cache yet (run env/bootstrap_data.py first)")
        return 2
    df = load_splits(splits)
    miss = [p for p in df.path if not Path(p).exists()]
    if miss:
        print(f"SKIP: {len(miss)}/{len(df)} images from splits.csv are missing (e.g. {miss[0]}); "
              "run 003 `prep` / env/bootstrap_data.py first")
        return 2
    from features import get_device
    dev = get_device(a.device).type
    log(f"device {dev}, {len(df)} rows from {splits.name}, cache {cache}")
    bb = secondary_backbone()
    pb = check_design_vs_003(bb)
    selfcheck_primary(df, bb, pb, dev)
    conds = a.conditions or P.ALL_CONDITIONS
    unknown = [c for c in conds if c not in P.ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; choose from {P.ALL_CONDITIONS}")
    t0 = time.time()
    run_extraction(df, bb, conds, cache, dev, P.SPEC, guard=not a.no_guard, limit=a.limit, workers=a.workers)
    log(f"done in {time.time() - t0:.0f}s")
    verify003(df, cache)   # optional: SKIPs when 003's B0 cache is absent


if __name__ == "__main__":
    sys.exit(main() or 0)
