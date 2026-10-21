"""CLI evaluation harness.

  python run_eval.py --manifest M.csv --out DIR --conditions clean jpeg75_resize05 \
      (--scorer pkg.module:score_fn | --backbone hf:dinov3:<repo> [--layers -1 --use-mean] [--logo])

Manifest `path`s must be readable images (normally the PNGs from preprocess.py). Scorer callables get a list
of PIL images (already perturbed) and return one score per image (higher = more fake).
"""
import argparse
import datetime as dt
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import format_bias_probe
import metrics
import probe
from features import Backbone, extract_features, load_backbone
from manifest import read_manifest
from perturb import apply_condition


def git_info():
    def run(*a):
        return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()
    try:
        return dict(commit=run("rev-parse", "HEAD") or None, dirty=bool(run("status", "--porcelain")))
    except Exception:
        return dict(commit=None, dirty=None)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def import_callable(spec, extra_paths=()):
    sys.path[:0] = [os.getcwd(), *extra_paths]
    mod, _, fn = spec.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(mod), fn)


def score_images(score_fn, df, condition, seed, batch_size=16):
    out = []
    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i + batch_size]
        imgs = [apply_condition(Image.open(p), condition, key=k, seed=seed)
                for p, k in zip(rows.path, rows.sha256)]
        s = np.asarray(score_fn(imgs), float).reshape(-1)
        assert len(s) == len(imgs), "scorer must return one score per image"
        out.append(s)
    return np.concatenate(out) if out else np.zeros(0)


def run(args):
    df = read_manifest(args.manifest)
    if (df.split == "").any():
        raise SystemExit("manifest has unassigned splits; run manifest.assign_splits first")
    if args.limit:
        df = df.groupby(["label", "split"], group_keys=False).head(args.limit).reset_index(drop=True)
    side_path = args.sidecar or os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "sidecar.csv")
    sidecar = pd.read_csv(side_path) if os.path.exists(side_path) else None
    started = dt.datetime.now(dt.timezone.utc)
    res = dict(
        config={k: v for k, v in vars(args).items()}, seed=args.seed, git=git_info(),
        started_utc=started.isoformat(),
        n_counts={"total": len(df), **{f"{s}/{g}": int(n) for (s, g), n in
                                       df.groupby(["split", "generator"]).size().items()}},
        format_bias=format_bias_probe.format_bias_report(df, sidecar, args.seed),
        conditions={})
    test = df[df.split == "test"]
    calib_mask = ((df.split == "calib") & (df.label == 0)).to_numpy()
    test_mask = (df.split == "test").to_numpy()
    train_mask = (df.split == "train").to_numpy()
    bb = None
    if args.backbone:
        bb = (import_callable(args.backbone[3:], args.path)() if args.backbone.startswith("py:")
              else load_backbone(args.backbone, args.input_size, tuple(args.layers), True, args.use_mean))
        assert isinstance(bb, Backbone)

    def feats(cond, mask):
        X = None
        idx = np.where(mask)[0]
        f = extract_features(bb, df.path.iloc[idx], cond, args.exp_dir, args.batch_size,
                             keys=df.sha256.iloc[idx], seed=args.seed)
        X = np.zeros((len(df), f.shape[1]), np.float32)
        X[idx] = f
        return X

    X_train = feats(args.train_condition, train_mask) if bb else None
    for cond in args.conditions:
        if bb:
            X_eval = feats(cond, test_mask | calib_mask)
            r = probe.standard_eval(df, X_train, X_eval, args.seed, args.threshold, args.n_boot)
            if args.logo:
                r["logo"] = probe.logo_eval(df, X_train, X_eval, args.seed, args.threshold, args.n_boot)
            r["train_condition"] = args.train_condition
        else:
            score_fn = import_callable(args.scorer, args.path)
            cal_scores = (score_images(score_fn, df[calib_mask], cond, args.seed, args.batch_size)
                          if calib_mask.any() else None)
            s = score_images(score_fn, test, cond, args.seed, args.batch_size)
            r = metrics.evaluate_scored_frame(test, s, cal_scores, args.threshold, args.n_boot, args.seed)
        res["conditions"][cond] = r
        print(f"[{cond}] AUROC={r['overall']['auroc']['value']:.4f}  "
              f"format-bias AUROC={res['format_bias'].get('auroc', float('nan')):.4f}")
    res["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    return res


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="dir for results.json")
    p.add_argument("--conditions", nargs="+", default=["clean", "jpeg75_resize05"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scorer", help="module:callable, score(list[PIL]) -> np.ndarray")
    g.add_argument("--backbone", help="hf:clip:<id> | hf:dinov2:<id> | hf:dinov3:<id> | timm:<n> | tv:<n> | py:module:factory")
    p.add_argument("--layers", type=int, nargs="+", default=[-1])
    p.add_argument("--use-mean", action="store_true", help="also use mean-pooled patch tokens")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--logo", action="store_true", help="also run leave-one-generator-out")
    p.add_argument("--train-condition", default="clean", help="probe is trained on this condition's features")
    p.add_argument("--exp-dir", default=".", help="feature cache lives in <exp-dir>/cache")
    p.add_argument("--sidecar", help="preprocess sidecar.csv (default: next to manifest)")
    p.add_argument("--path", nargs="*", default=[], help="extra sys.path entries for --scorer/--backbone")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="debug: max rows per (label, split)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
