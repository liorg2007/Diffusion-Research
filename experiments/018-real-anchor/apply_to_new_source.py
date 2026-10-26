"""Apply the FROZEN E9 pipeline (results/choice.json + config.json) to a NEW real pipeline, e.g. the owner's phone photos (confirmatory 4th held-out pipeline).

    env/.venv/bin/python -I experiments/018-real-anchor/apply_to_new_source.py --anchors ANCHORS.npz [--test TEST.npz] [--out OUT.json] [--method A-off|A-proj|A-thr]

Inputs are FEATURES, not images (this script downloads nothing and extracts nothing): DINOv2-base secondary features of the 004 pipeline (4608-d per image,
blocks 4/8/12 CLS+mean; produce them with the E7 `feats` recipe, fov512 preprocessing for the primary arm). ANCHORS.npz needs one array [n, 4608] per T-aug
condition ('clean', 'jpeg75_resize05', 'social_sim'; the same anchor photos processed identically to training); TEST.npz (optional) one array per condition
you want scored. The anchors must be photos of the new pipeline that are NOT in the test set.

What it does (all choices frozen by results/choice.json, nothing is re-chosen here):
  1. refuses to run if config.json changed since the choice was frozen (sha256 check);
  2. trains the E7 probe ('full' model, block 8) on ALL three sources (VISION, RAISE, OI), T-aug, exactly as E7 / E9;
  3. adapts with the anchors (A-off: offset = train-real mean - anchor mean; A-proj: project out that direction and retrain; A-thr: threshold only);
  4. threshold = 1% FPR on the training sources' calib reals' CLEAN scores (A-thr: on the anchors' clean scores);
  5. scores TEST with that threshold and writes thr, scores and (if TEST has a boolean array 'is_fake_<cond>') the FPR / AUROC.
DEPLOYMENT CONVENTION: the images come unlabeled, so A-off shifts EVERY test image (reals and fakes) by the offset. That is a pure operating-point move (AUROC equals
the unadapted probe's); the E9 headline for A-off shifted only the held-out source's reals (literal Amendment 18 reading). Read results/tables.md (A-off-all rows) before
trusting an A-off result on a phone pipeline.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True, help="npz: arrays 'clean', 'jpeg75_resize05', 'social_sim', each [n, 4608] (anchor reals of the new pipeline)")
    ap.add_argument("--test", help="npz: arrays per condition [m, 4608] to score (optional 'is_fake_<cond>' boolean arrays for metrics)")
    ap.add_argument("--method", help="default: the method frozen in results/choice.json")
    ap.add_argument("--feat", default=R.CFG["criteria_feat"])
    ap.add_argument("--out", default=str(HERE / "results" / "new_source_scores.json"))
    a = ap.parse_args()
    ch = json.loads((HERE / "results" / "choice.json").read_text())
    sha = subprocess.run(["sha256sum", str(HERE / "config.json")], capture_output=True, text=True).stdout.split()[0]
    if sha != ch["config_sha256"]:
        raise SystemExit("config.json changed after the choice was frozen: the pipeline is no longer the frozen one")
    method = a.method or ch["method"]
    if a.method and a.method != ch["method"]:
        print(f"WARNING: --method {a.method} differs from the frozen choice {ch['method']}; report it as a non-frozen variant", file=sys.stderr)
    anchors = dict(np.load(a.anchors))
    for c in R.TAUG:
        if c not in anchors or anchors[c].ndim != 2 or anchors[c].shape[1] != 4608:
            raise SystemExit(f"anchors need an array [n, 4608] for condition {c}")
    test = {k: v for k, v in dict(np.load(a.test)).items() if not k.startswith("is_fake_")} if a.test else {}
    labels = {k[len("is_fake_"):]: v.astype(bool) for k, v in dict(np.load(a.test)).items() if k.startswith("is_fake_")} if a.test else {}
    n = anchors["clean"].shape[0]
    print(f"frozen method {method}, feat {a.feat}, n anchors {n}")
    out = R.frozen_pipeline(method, a.feat, anchors, test)
    res = dict(method=method, feat=a.feat, n_anchors=int(n), threshold=out["thr"], delta_norm=out["delta_norm"], conditions={})
    from sklearn.metrics import roc_auc_score
    for c, s in out["scores"].items():
        e = dict(n=int(len(s)), flagged_fraction=float((s >= out["thr"]).mean()), scores=[float(x) for x in s])
        if c in labels:
            y = labels[c]
            e["fpr"] = float((s[~y] >= out["thr"]).mean())
            e["tpr"] = float((s[y] >= out["thr"]).mean()) if y.any() else None
            e["auroc"] = float(roc_auc_score(y, s)) if y.any() and (~y).any() else None
        res["conditions"][c] = e
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({c: {k: v for k, v in e.items() if k != "scores"} for c, e in res["conditions"].items()}, indent=1))
    print("threshold", res["threshold"], "-> wrote", a.out)


if __name__ == "__main__":
    main()
