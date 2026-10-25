"""OFFICIAL H2-v2 score builder (Amendment 3 item 4 + Amendment 4). Output: official/scores_long_<arm>.csv.gz for arm in {orig, fov512}.

  python -I experiments/007-worst-env-conformal/official/build_scores.py b0 [orig|fov512]        # B0 from 003/004 caches (CPU, seconds)
  python -I experiments/007-worst-env-conformal/official/build_scores.py pretrained <arm> [det]  # CommFor/RINE/ClipDet/B-Free (GPU, resumable)
  python -I experiments/007-worst-env-conformal/official/build_scores.py export <arm>

Rows needed by 007 (h2v2.py), official splits (003 splits.csv for `orig`; 004 splits_fov512.csv for `fov512`; train-split
fakes dropped so TPR is measured on the 585 TEST fakes for every detector, and B0 never sees its own training fakes):
  clean            : 1,074 test reals (25 devices x native/WA/FBH) + 129 calib reals (10 devices x 3 pools, for M1) + 585 test fakes
  jpeg75_resize05  : 358 test natives + 585 test fakes
  screenshot_sim   : 358 test natives                      (synthetic LOPO platform; fakes are not needed)
Pretrained scores: orig reuses experiments/003-baselines-eval/cache (read-only) for every row it covers; anything else is
scored here on the GPU (TF32 off, batch 20) and guarded against a CPU run (batch 4) on a fixed 32-row subset of the NEWLY
scored rows per (detector, condition): abort if max |dlogit| > 1e-3 (D-008). Conditions: perturb.apply_condition(img, cond,
key=sha256, seed=0), as in 003/007 pilot. Score = raw logit, higher = more fake.
"""
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

CACHE = EXP / "cache" / "official"
SEED = 0
BATCH = 20
TOL = 1e-3
CONDS = ["clean", "jpeg75_resize05", "screenshot_sim"]
SCORERS = {"communityforensics": ("scorers.communityforensics", "score"), "bfree": ("scorers.bfree", "score"),
           "rine": ("scorers.rine", "score"), "clipdet": ("scorers.clipdet", "score")}
SPLITS_SRC = {"orig": ROOT / "experiments/003-baselines-eval/splits.csv",
              "fov512": ROOT / "experiments/004-phase4-features/splits_fov512.csv"}
POOLS = ["vision_native", "vision_whatsapp", "vision_facebook_hq"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def official_splits(arm):
    """Split table for `arm` with the train-split fakes removed; written to official/splits_<arm>.csv."""
    df = pd.read_csv(SPLITS_SRC[arm], keep_default_na=False, dtype={"vidx": str})
    df = df[~((df.label == 1) & (df.split == "train"))].reset_index(drop=True)
    df.to_csv(HERE / f"splits_{arm}.csv", index=False)
    return df


def needed(df):
    """{cond: DataFrame of rows needed}."""
    r = df[df.label == 0]
    test = r[(r.split == "test") & r.pool.isin(POOLS)]
    calib = r[(r.split == "calib") & r.pool.isin(POOLS)]
    fk = df[df.label == 1]
    assert len(test) == 1074 and len(calib) == 129 and len(fk) == 585, (len(test), len(calib), len(fk))
    nat = test[test.pool == "vision_native"]
    rz = r[r.pool == "raise"]                      # Amendment 13: RAISE test+calib reals (descriptive group only; 199 + 60)
    rz = rz[rz.split.isin(["test", "calib"])]
    assert len(rz) == 259, len(rz)
    return {"clean": pd.concat([test, calib, rz, fk]), "jpeg75_resize05": pd.concat([nat, rz, fk]), "screenshot_sim": nat}


# ----------------------------------------------------------------------------- B0
def twin_ids(df):
    import re
    return df.src_path.map(lambda p: (re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1]).to_numpy()


def stage_b0(arm):
    """B0 = 003's probe recipe (probe.fit_probe(X[b0_train], y), seed 0) on the cached DINOv2 features of 004 (primary 1536-d =
    003's B0 features). orig: checked against 003's cached B0 scores (max |diff| reported); 003's cached values are used for every
    row they cover. fov512: probe trained on the fov512 clean features. Single model 'full'."""
    import p4features as P
    import probe
    src = pd.read_csv(SPLITS_SRC[arm], keep_default_na=False, dtype={"vidx": str})
    df = official_splits(arm)
    sub = "" if arm == "orig" else "fov512"
    cdir = (P.cache_root(sub or None))
    # model on the FULL split table (b0_train rows include train fakes)
    Xc = P.load_features_aligned(src, "primary", "clean", cache_dir=cdir)
    tr = src.b0_train.to_numpy()
    assert not np.isnan(Xc[tr]).any()
    model = probe.fit_probe(Xc[tr], src.label.to_numpy()[tr], SEED)
    # Amendment 13: twin-disjoint refit = drop train fakes whose RAISE twin id (fake filename = RAISE image id) is a calib/test RAISE real
    rid = twin_ids(src)
    lab, sp, pl = src.label.to_numpy(), src.split.to_numpy(), src.pool.to_numpy()
    held = set(rid[(lab == 0) & (pl == "raise") & np.isin(sp, ["calib", "test"])]) - {""}
    drop = tr & (lab == 1) & np.array([r in held for r in rid])
    tr2 = tr & ~drop
    model2 = probe.fit_probe(Xc[tr2], lab[tr2], SEED)
    info = dict(twin_free=dict(n_train_fakes_dropped=int(drop.sum()), n_train_fakes_before=int((tr & (lab == 1)).sum()),
                               n_train=int(tr2.sum()), C=float(model2[-1].C_[0]),
                               dropped_by_generator=src.generator[drop].value_counts().to_dict()), arm=arm, C=float(model[-1].C_[0]), n_train=int(tr.sum()), n_train_real=int((src.label.to_numpy()[tr] == 0).sum()))
    need = needed(df)
    parts, diffs = [], {}
    for cond in CONDS:
        Xk = P.load_features_aligned(src, "primary", cond, cache_dir=cdir)
        pos = {s: i for i, s in enumerate(src.sha256)}
        rows = need[cond]
        idx = np.array([pos[s] for s in rows.sha256])
        assert not np.isnan(Xk[idx]).any(), f"missing features {arm}/{cond}"
        sc = model.decision_function(Xk[idx])
        if arm == "orig" and cond != "screenshot_sim":
            p = ROOT / f"experiments/003-baselines-eval/cache/scores_b0_dinov2_{cond}.npz"
            z = np.load(p, allow_pickle=False)
            ref = dict(zip(z["keys"].tolist(), z["vals"].tolist()))
            have = np.array([s in ref for s in rows.sha256])
            d = np.abs(sc[have] - np.array([ref[s] for s in rows.sha256[have]]))
            diffs[cond] = dict(n_cached=int(have.sum()), n_new=int((~have).sum()), max_abs_diff_refit_vs_003=float(d.max()))
            sc = np.where(have, [ref.get(s, 0.0) for s in rows.sha256], sc)
        parts.append(pd.DataFrame(dict(arm="B0", model="full", condition=cond, sha256=rows.sha256.values, score=sc)))
        parts.append(pd.DataFrame(dict(arm="B0_twinfree", model="full", condition=cond, sha256=rows.sha256.values,
                                       score=model2.decision_function(Xk[idx]))))
    info["vs_003_cache"] = diffs
    info["d008"] = ("features extracted by 004 under its own D-008 guard: " +
                    str(ROOT / f"experiments/004-phase4-features/cache/{'fov512' if arm == 'fov512' else 'feats'}/d008_guard.json"))
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(CACHE / f"scores_b0_{arm}.csv.gz", index=False)
    (CACHE / f"b0_{arm}_info.json").write_text(json.dumps(info, indent=1))
    log("b0", json.dumps(info))


# ----------------------------------------------------------------------------- pretrained
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


def npz_load(p):
    if Path(p).exists():
        z = np.load(p, allow_pickle=False)
        return dict(zip(z["keys"].tolist(), z["vals"].tolist()))
    return {}


def npz_save(p, d):
    ks = sorted(d)
    np.savez(str(p) + ".tmp.npz", keys=np.array(ks), vals=np.array([d[k] for k in ks]))
    os.replace(str(p) + ".tmp.npz", p)


def cached_003(name, cond):
    """003 GPU score cache (read-only); 003's third condition `social_sim` is NOT screenshot_sim, so only clean/jpeg75 map."""
    if cond == "screenshot_sim":
        return {}
    return npz_load(ROOT / f"experiments/003-baselines-eval/cache/scores_{name}_{cond}.npz")


def guard_subset(arm, cond, new_rows):
    """Fixed seed-0 subset of the NEWLY scored rows: 16 real + 16 fake when both exist, else 32 of what exists."""
    p = CACHE / f"guard_subset_{arm}_{cond}.json"
    if p.exists():
        return json.loads(p.read_text())
    rng = np.random.default_rng(SEED)
    r = sorted(new_rows.sha256[new_rows.label == 0]); f = sorted(new_rows.sha256[new_rows.label == 1])
    if r and f:
        sub = [r[i] for i in rng.choice(len(r), min(16, len(r)), replace=False)] + [f[i] for i in rng.choice(len(f), min(16, len(f)), replace=False)]
    else:
        pool = r or f
        sub = [pool[i] for i in rng.choice(len(pool), min(32, len(pool)), replace=False)]
    p.write_text(json.dumps(sub))
    return sub


def stage_cpuref(arm, name, cond):
    os.environ["DD_DEVICE"] = "cpu"
    df = official_splits(arm)
    dfi = df.set_index("sha256", drop=False)
    sub = json.loads((CACHE / f"guard_subset_{arm}_{cond}.json").read_text())
    p = CACHE / f"cpuref_{arm}_{name}_{cond}.json"
    t = time.time()
    p.write_text(json.dumps(score_rows(get_fn(name), dfi.loc[sub], cond, batch=4)))
    log(f"cpuref {arm}/{name}/{cond} {time.time() - t:.0f}s")


def stage_pretrained(arm, names):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = official_splits(arm)
    dfi = df.set_index("sha256", drop=False)
    need = needed(df)
    gp = CACHE / f"d008_guard_{arm}.json"
    guard = json.loads(gp.read_text()) if gp.exists() else {}
    for name in names:
        for cond in CONDS:
            rows = need[cond]
            sc = npz_load(CACHE / f"scores_{arm}_{name}_{cond}.npz")
            if arm == "orig":
                reuse = {s: v for s, v in cached_003(name, cond).items() if s in set(rows.sha256) and s not in sc}
                sc.update(reuse)
                log(f"{arm}/{name}/{cond}: {len(reuse)} rows reused from 003 cache")
            todo_all = rows[~rows.sha256.isin(sc)]
            # D-008 guard on newly scored rows (fixed subset, computed BEFORE scoring so it is independent of resume state)
            new_all = rows[~rows.sha256.isin(cached_003(name, cond))] if arm == "orig" else rows
            key = f"{name}/{cond}"
            if len(new_all):
                sub = guard_subset(arm, cond, new_all)
                rp = CACHE / f"cpuref_{arm}_{name}_{cond}.json"
                if not rp.exists():
                    os.environ.pop("DD_DEVICE", None)
                    r = subprocess.run([sys.executable, "-I", str(HERE / "build_scores.py"), "cpuref", arm, name, cond])
                    if r.returncode != 0:
                        raise SystemExit(f"cpuref failed {arm}/{name}/{cond}")
                ref = json.loads(rp.read_text())
                os.environ.pop("DD_DEVICE", None)
                fn = get_fn(name)
                miss = [s for s in sub if s not in sc]
                if miss:
                    sc.update(score_rows(fn, dfi.loc[miss], cond))
                    npz_save(CACHE / f"scores_{arm}_{name}_{cond}.npz", sc)
                mx = float(np.abs([sc[s] - ref[s] for s in sub]).max())
                guard[key] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL), n_new_rows=len(new_all))
                gp.write_text(json.dumps(guard, indent=1))
                log(f"{arm}/{name}/{cond}: D-008 max|dlogit|={mx:.2e} (n={len(sub)}, tol {TOL})")
                if mx > TOL:
                    raise SystemExit(f"D-008 ABORT {arm}/{name}/{cond}: {mx:.3e}")
            else:
                guard[key] = dict(note="all rows reused from 003 cache (003 D-008 guard applies)", n_new_rows=0)
                gp.write_text(json.dumps(guard, indent=1))
                fn = get_fn(name)
            todo = rows[~rows.sha256.isin(sc)]
            for i in range(0, len(todo), 100):
                sc.update(score_rows(fn, todo.iloc[i:i + 100], cond))
                npz_save(CACHE / f"scores_{arm}_{name}_{cond}.npz", sc)
                log(f"{arm}/{name}/{cond}: {len(sc)}/{len(rows)}")
            if len(todo) == 0 and not (CACHE / f"scores_{arm}_{name}_{cond}.npz").exists():
                npz_save(CACHE / f"scores_{arm}_{name}_{cond}.npz", sc)
            else:
                npz_save(CACHE / f"scores_{arm}_{name}_{cond}.npz", sc)


def stage_export(arm):
    df = official_splits(arm)
    need = needed(df)
    parts = [pd.read_csv(CACHE / f"scores_b0_{arm}.csv.gz", dtype={"sha256": str})]
    for name in SCORERS:
        for cond in CONDS:
            sc = npz_load(CACHE / f"scores_{arm}_{name}_{cond}.npz")
            rows = need[cond]
            assert set(rows.sha256) <= set(sc), f"incomplete {arm}/{name}/{cond}"
            parts.append(pd.DataFrame(dict(arm=name, model="pretrained", condition=cond, sha256=rows.sha256.values,
                                           score=[sc[s] for s in rows.sha256])))
    out = pd.concat(parts, ignore_index=True)
    assert out.score.notna().all() and np.isfinite(out.score).all()
    out.to_csv(HERE / f"scores_long_{arm}.csv.gz", index=False)
    log(f"wrote scores_long_{arm}.csv.gz rows={len(out)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cpuref":
        stage_cpuref(a[1], a[2], a[3])
    elif a[0] == "b0":
        for arm in (a[1:] or ["orig", "fov512"]):
            stage_b0(arm)
    elif a[0] == "pretrained":
        stage_pretrained(a[1], a[2:] or list(SCORERS))
    elif a[0] == "export":
        stage_export(a[1])
