"""PILOT Part A: score the pilot split with the pretrained detectors (CommFor, RINE, ClipDet) under 3 conditions.

  python -I experiments/007-worst-env-conformal/score_pretrained.py            # all detectors
  python -I experiments/007-worst-env-conformal/score_pretrained.py rine       # one detector

Mirrors experiments/003-baselines-eval/run.py (same scorer functions, same load_cond/apply_condition(key=sha256, seed=0),
batch 20, fixed 32-row CPU-vs-CUDA D-008 guard with abort at 1e-3) but reads the pilot split and writes to
experiments/007-worst-env-conformal/{cache,pilot}. Resumable. Output: pilot/scores_pretrained.csv.gz
(arm, model, condition, sha256, score); arm = detector, model = "pretrained"; higher = more fake (logit). PILOT.
"""
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

CACHE = HERE / "cache"
PILOT = HERE / "pilot"
SPLITS = ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv"
SEED = 0
BATCH = 20
TOL = 1e-3
CONDITIONS = ["clean", "jpeg75_resize05", "screenshot_sim"]
SCORERS = {"communityforensics": ("scorers.communityforensics", "score"),
           "rine": ("scorers.rine", "score"),
           "clipdet": ("scorers.clipdet", "score")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_splits():
    return pd.read_csv(SPLITS, keep_default_na=False, dtype={"vidx": str})


def check_subset(df):
    """Fixed 32 rows (16 real + 16 fake, seed 0) drawn from the pilot test split; cached."""
    p = CACHE / "check_subset.json"
    if p.exists():
        return json.loads(p.read_text())
    s = df[df.split == "test"]
    rng = np.random.default_rng(SEED)
    r = sorted(s.sha256[s.label == 0]); f = sorted(s.sha256[s.label == 1])
    sub = [r[i] for i in rng.choice(len(r), 16, replace=False)] + [f[i] for i in rng.choice(len(f), 16, replace=False)]
    p.write_text(json.dumps(sub))
    return sub


def load_cond(path, sha, cond):
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        return apply_condition(im, cond, key=sha, seed=SEED)


def get_fn(name):
    m, f = SCORERS[name]
    return getattr(importlib.import_module(m), f)


def score_rows(fn, rows, cond, batch=BATCH):
    out = {}
    for i in range(0, len(rows), batch):
        ch = rows.iloc[i:i + batch]
        imgs = [load_cond(p, s, cond) for p, s in zip(ch.path, ch.sha256)]
        sc = np.asarray(fn(imgs), float).reshape(-1)
        assert len(sc) == len(imgs) and np.isfinite(sc).all()
        out.update(zip(ch.sha256, sc))
    return out


def load_scores(name, cond):
    p = CACHE / f"scores_{name}_{cond}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return dict(zip(z["keys"].tolist(), z["vals"].tolist()))
    return {}


def save_scores(name, cond, d):
    p = CACHE / f"scores_{name}_{cond}.npz"
    ks = sorted(d)
    np.savez(str(p) + ".tmp.npz", keys=np.array(ks), vals=np.array([d[k] for k in ks]))
    os.replace(str(p) + ".tmp.npz", p)


def stage_cpuref(name):
    os.environ["DD_DEVICE"] = "cpu"
    df = load_splits().set_index("sha256", drop=False)
    sub = check_subset(df.reset_index(drop=True))
    fn = get_fn(name)
    p = CACHE / f"cpuref_{name}.json"
    ref = json.loads(p.read_text()) if p.exists() else {}
    for cond in CONDITIONS:
        if cond in ref:
            continue
        t = time.time()
        ref[cond] = score_rows(fn, df.loc[sub], cond, batch=4)
        p.write_text(json.dumps(ref))
        log(f"cpuref {name} {cond} done in {time.time() - t:.0f}s")


def stage_score(names):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    df = load_splits()
    dfi = df.set_index("sha256", drop=False)
    sub = check_subset(df)
    rest = df[~df.sha256.isin(sub)]
    guard_p = CACHE / "d008_guard.json"
    guard = json.loads(guard_p.read_text()) if guard_p.exists() else {}
    for name in names:
        rp = CACHE / f"cpuref_{name}.json"
        if not rp.exists() or any(c not in json.loads(rp.read_text()) for c in CONDITIONS):
            log(f"{name}: CPU reference ({len(sub)} imgs x {len(CONDITIONS)} conds)")
            r = subprocess.run([sys.executable, "-I", str(HERE / "score_pretrained.py"), "cpuref", name])
            if r.returncode != 0:
                raise SystemExit(f"cpuref failed for {name}")
        ref = json.loads(rp.read_text())
        os.environ.pop("DD_DEVICE", None)   # auto -> cuda
        fn = get_fn(name)
        for cond in CONDITIONS:
            sc = load_scores(name, cond)
            todo_sub = dfi.loc[[s for s in sub if s not in sc]]
            if len(todo_sub):
                sc.update(score_rows(fn, todo_sub, cond))
                save_scores(name, cond, sc)
            mx = float(np.abs([sc[s] - ref[cond][s] for s in sub]).max())
            guard[f"{name}/{cond}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL))
            guard_p.write_text(json.dumps(guard, indent=1))
            log(f"{name} {cond}: D-008 max|dlogit|={mx:.2e} (tol {TOL})")
            if mx > TOL:
                raise SystemExit(f"D-008 ABORT: {name}/{cond} max |dlogit| {mx:.3e} > {TOL}")
            todo = rest[~rest.sha256.isin(sc)]
            for i in range(0, len(todo), 100):
                sc.update(score_rows(fn, todo.iloc[i:i + 100], cond))
                save_scores(name, cond, sc)
                log(f"{name} {cond}: {len(sc)}/{len(df)}")


def stage_export():
    df = load_splits()
    parts = []
    for name in SCORERS:
        for cond in CONDITIONS:
            sc = load_scores(name, cond)
            assert set(sc) == set(df.sha256), f"incomplete: {name}/{cond} {len(sc)}/{len(df)}"
            parts.append(pd.DataFrame(dict(arm=name, model="pretrained", condition=cond,
                                           sha256=df.sha256.values, score=[sc[s] for s in df.sha256])))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(PILOT / "scores_pretrained.csv.gz", index=False)
    log(f"wrote {PILOT / 'scores_pretrained.csv.gz'} rows={len(out)}")


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True); PILOT.mkdir(exist_ok=True)
    a = sys.argv[1:]
    if a and a[0] == "cpuref":
        stage_cpuref(a[1])
    else:
        names = a or list(SCORERS)
        stage_score(names)
        if set(SCORERS) <= {n for n in SCORERS if all((CACHE / f"scores_{n}_{c}.npz").exists() for c in CONDITIONS)}:
            try:
                stage_export()
            except AssertionError as e:
                log("export skipped:", e)
