"""E7 (Amendment 16): leave-one-real-source-out test of the mid-level (block 8) DINOv2 detector. Resumable stages.

    env/.venv/bin/python -I experiments/016-loso/run.py all [--workers 12]
Stages (each skips finished work): splits -> prep -> feats -> fit -> eval
  splits  build splits_{orig,fov512,fov224}.csv (003/004 rows + 600 Open Images + fov224 only: RAID/D3 reals and fakes)
  prep    crops for the NEW rows only, via 004 fov_prep.py --no-clobber (never overwrites a file)
  feats   004 pipeline (blocks 4/8/12 CLS+mean, DINOv2-base) for NEW (image, condition) pairs -> cache/<arm>/; D-008 guard
  fit     probes per (arm, feature, training-source set) with LOGO over fake generators -> cache/scores_*.npz
  eval    metrics, bootstrap, labels -> results/{results.json, tables.md, scores_long.csv.gz}
All design constants are in config.json. Reuses the E6 recipe (015-midlayer-probe/run.py: caption-family roles, groups).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib.util
import itertools
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from scipy.stats import beta

import metrics
import probe
import p4features as P

CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CACHE, RES, PREP = HERE / "cache", HERE / "results", HERE / "_prep"
ARMS = CFG["arms"]
TRAINGENS, D3GENS, FLUX = CFG["train_generators"], CFG["d3_generators"], CFG["test_only_generator"]
FEATS = {k: slice(*v) for k, v in CFG["features"].items()}
FEAT_ORDER = ["F_b8", "F_last"]
SRC = CFG["sources"]
TH = CFG["thresholds"]
NEW_POOLS = ("openimages", "d3_real", "fake_d3")
MANIFEST_OI = HERE / "openimages_manifest.csv"
RAID_MANIFEST = ROOT / "experiments" / "010-raid-transfer" / "raid_manifest.csv"
D3_RE = re.compile(r"(\d+)_(real|gen_\d)\.")


def log(*a):
    print(time.strftime("%H:%M:%S"), "E7", *a, flush=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e6():
    """The E6 module (caption-family roles, groups). Imported by path; its constants come from its own config.json."""
    if "e6run" not in sys.modules:
        sys.modules["e6run"] = _load_module("e6run", ROOT / "experiments" / "015-midlayer-probe" / "run.py")
    return sys.modules["e6run"]


def arm_gens(arm):
    return TRAINGENS + (D3GENS if "D3" in ARMS[arm]["sources"] else [])


def arm_all_gens(arm):
    return arm_gens(arm) + [FLUX]


def model_names(arm):
    return [f"g_{g}" for g in arm_gens(arm)] + ["full"]


def model_of(arm, g):
    return f"g_{g}" if g in arm_gens(arm) else "full"


# ----------------------------------------------------------------------------- stage 1: splits
def oi_rows():
    m = pd.read_csv(MANIFEST_OI)
    assert len(m) == 600 and m.image_id.is_unique and (m.license == "https://creativecommons.org/licenses/by/2.0/").all()
    assert (m[["width", "height"]].min(axis=1) >= 512).all()
    c = CFG["oi_split"]
    ids = sorted(m.image_id)
    perm = np.random.default_rng(c["seed"]).permutation(len(ids))
    sp = {}
    for rank, j in enumerate(perm):
        sp[ids[j]] = "train" if rank < c["n_train"] else "calib" if rank < c["n_train"] + c["n_calib"] else "test"
    assert sum(v == "test" for v in sp.values()) == c["n_test"]
    df = pd.DataFrame(dict(sha256=m.sha256, path="", src_path=m.path, label=0, generator="real", pool="openimages",
                           camera_or_platform="", device="", vidx="", split=[sp[i] for i in m.image_id]))
    df["in_scoring_set"] = df.split == "test"
    df["b0_train"] = False
    df["orig_format"] = "JPEG"
    df["source_dataset"] = "Open Images V7 validation (CVDF copy)"
    df["license"] = "CC BY 2.0"
    return df


def d3_rows():
    m = pd.read_csv(RAID_MANIFEST)
    m = m[m.kind == "original"].copy()
    assert len(m) == 500 and (m.match.astype(str).str.upper() == "TRUE").all()
    ids = sorted(m[m.cls == "real"].id.astype(str))
    assert len(ids) == 100
    perm = np.random.default_rng(SEED).permutation(len(ids))
    fr = CFG["d3_split"]["frac"]
    sp = {ids[j]: ("train" if r < fr[0] * 100 else "calib" if r < (fr[0] + fr[1]) * 100 else "test") for r, j in enumerate(perm)}
    rows = []
    for r in m.itertuples():
        i = str(r.id)
        real = r.cls == "real"
        rows.append(dict(sha256=r.sha256, path="", src_path=r.path, label=0 if real else 1,
                         generator="real" if real else "d3_" + r.cls, pool="d3_real" if real else "fake_d3",
                         camera_or_platform="", device="", vidx="", split=sp[i], in_scoring_set=sp[i] == "test", b0_train=False,
                         orig_format="JPEG", source_dataset="RAID/D3 originals (LAION content)", license="MIT (RAID); upstream [unverified]"))
    return pd.DataFrame(rows)


def stage_splits():
    base = {a: pd.read_csv(ROOT / c["base_splits"], keep_default_na=False, dtype={"vidx": str}) for a, c in ARMS.items()}
    sh = [b.sha256.tolist() for b in base.values()]
    assert sh[0] == sh[1] == sh[2], "003/004 splits files differ in rows/order"
    PREP.mkdir(exist_ok=True)
    new = {"orig": oi_rows(), "fov512": oi_rows(), "fov224": pd.concat([oi_rows(), d3_rows()], ignore_index=True)}
    for a, nd in new.items():
        assert not set(nd.sha256) & set(base[a].sha256), "new rows overlap existing rows"
        nd.to_csv(PREP / f"new_{a}.csv", index=False)
        log(f"splits: {a}: {len(base[a])} existing + {len(nd)} new rows (pools {nd.pool.value_counts().to_dict()})")


# ----------------------------------------------------------------------------- stage 2: crops (004 fov_prep, --no-clobber)
def stage_prep():
    for a, c in ARMS.items():
        out = PREP / f"out_{a}.csv"
        final = HERE / f"splits_{a}.csv"
        if final.exists():
            log(f"prep {a}: {final.name} exists, skipped")
            continue
        cmd = [sys.executable, "-I", str(ROOT / "experiments" / "004-phase4-features" / "fov_prep.py"), "--splits", str(PREP / f"new_{a}.csv"),
               "--out-splits", str(out), "--report", str(RES / f"prep_report_{a}.json"), "--size", str(c["prep_size"]),
               "--mode", c["prep_mode"], "--no-clobber", "--workers", "8"]
        if a == "fov224":
            cmd.append("--drop-small")
        log("prep:", " ".join(cmd[2:]))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (RES / f"prep_{a}.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise SystemExit(f"fov_prep failed for {a}: {r.stderr[-500:]}")
        base = pd.read_csv(ROOT / c["base_splits"], keep_default_na=False, dtype={"vidx": str})
        nd = pd.read_csv(out, keep_default_na=False, dtype={"vidx": str}).astype(base.dtypes.to_dict())
        full = pd.concat([base, nd], ignore_index=True)
        assert full.sha256.is_unique
        full.to_csv(final, index=False)
        rep = json.loads((RES / f"prep_report_{a}.json").read_text())
        log(f"prep {a}: ok {rep['n_ok']}, short<size {rep['n_short_side_lt_512']}, no_clobber {rep['no_clobber']}, wrote {final.name} ({len(full)} rows)")


# ----------------------------------------------------------------------------- data
def caption_family(df):
    """E6 roles for the existing (003/004) rows; new rows: role = split (OI by image, D3 by id family); rid = '' / 'd3_<id>'."""
    new = df.pool.isin(NEW_POOLS).to_numpy()
    role = np.empty(len(df), object)
    rid = np.array([""] * len(df), object)
    b = df[~new].reset_index(drop=True)
    r6, i6, _ = e6().build_roles(b)
    role[~new], rid[~new] = r6, i6
    role[new] = df.split.to_numpy()[new]
    for i in np.where(new)[0]:
        mm = D3_RE.search(os.path.basename(df.src_path.iat[i]))
        if mm:
            rid[i] = "d3_" + mm.group(1)
    return role, rid


class Data:
    """Splits + roles + cached DINOv2 secondary features (4608-d) of one arm: 004 cache for old rows, cache/<arm>/ for new ones."""

    def __init__(self, arm, load_feats=True, splits=None):
        self.arm, self.cfg = arm, ARMS[arm]
        self.df = pd.read_csv(splits or HERE / f"splits_{arm}.csv", keep_default_na=False, dtype={"vidx": str})
        self.role, self.rid = caption_family(self.df)
        self.groups = e6().make_groups(self.df, self.rid)
        self.lab = self.df.label.to_numpy()
        self.gen = self.df.generator.to_numpy()
        self.pool = self.df.pool.to_numpy()
        self.conds = self.cfg["conditions"]
        self.X = {}
        if load_feats:
            for c in self.conds:
                self.X[c] = load_x(arm, self.df, c)

    def idx(self, role=None, pools=None, gens=None, label=None):
        m = np.ones(len(self.df), bool) if role is None else self.role == role
        if pools is not None:
            m &= np.isin(self.pool, pools)
        if gens is not None:
            m &= np.isin(self.gen, gens)
        if label is not None:
            m &= self.lab == label
        return np.where(m)[0]


def read_cache_dir(cond, d):
    try:
        k, x = P.read_shards(cond, d)
    except FileNotFoundError:
        return {}
    return dict(zip(k.tolist(), x))


def load_x(arm, df, cond):
    store = read_cache_dir(cond, P.cache_root(ARMS[arm]["cache004"]))
    store.update(read_cache_dir(cond, CACHE / arm))
    miss = [s for s in df.sha256 if s not in store]
    if miss:
        raise SystemExit(f"{arm}/{cond}: {len(miss)} rows have no cached features (run the feats stage)")
    x = np.stack([store[s] for s in df.sha256]).astype(np.float32)
    assert x.shape[1] == 4608 and np.isfinite(x).all()
    return x


# ----------------------------------------------------------------------------- stage 3: features (004 pipeline) + D-008 guard
def stage_feats(workers=4):
    p4 = _load_module("p4run", ROOT / "experiments" / "004-phase4-features" / "run.py")
    import probe as probe_mod
    from concurrent.futures import ThreadPoolExecutor
    from features import embed_images, get_device
    dev = get_device().type
    bb = p4.secondary_backbone()
    pb = p4.check_design_vs_003(bb)
    guard_all = json.loads((RES / "d008_guard.json").read_text()) if (RES / "d008_guard.json").exists() else {}
    for arm, c in ARMS.items():
        splits = HERE / f"splits_{arm}.csv"
        df = pd.read_csv(splits, keep_default_na=False, dtype={"vidx": str})
        d003 = pd.read_csv(ROOT / "experiments/003-baselines-eval/splits.csv", keep_default_na=False, dtype={"vidx": str})
        cache = CACHE / arm
        cache.mkdir(parents=True, exist_ok=True)
        # probes for the logit guard: B0 (primary = block 12) and block 8, fit on the 003 b0_train rows, clean, from the 004 cache
        x0 = P.load_features_aligned(d003, "secondary", "clean", subdir=c["cache004"])
        tr = d003.b0_train.to_numpy()
        y0 = d003.label.to_numpy()
        probes = {k: probe_mod.fit_probe(x0[tr][:, FEATS[k]], y0[tr], SEED) for k in FEAT_ORDER}
        with ThreadPoolExecutor(workers) as pool:
            for cond in c["conditions"]:
                have004 = set(read_cache_dir(cond, P.cache_root(c["cache004"])))
                todo = df[~df.sha256.isin(have004)]
                key = f"{arm}/{cond}"
                if not len(todo):
                    log(f"feats {key}: all rows in the 004 cache")
                    continue
                if cond in c["conditions"] and cond not in P.EVAL_CONDITIONS:
                    raise SystemExit(f"{cond} is not a 004 eval condition")
                t = p4.extract_condition(todo.assign(in_scoring_set=True), bb, cond, cache, dev, pool, p4.P.SPEC)
                log(f"feats {key}: {t}")
                g = guard(p4, bb, todo, cond, cache, probes, dev, embed_images)
                guard_all[key] = g
                (RES / "d008_guard.json").write_text(json.dumps(guard_all, indent=1, sort_keys=True))
                if not g["ok"]:
                    raise SystemExit(f"D-008 ABORT {key}: {g}")
    log("feats done; D-008 guard max|dlogit| = %.2e over %d (arm, condition) pairs" % (
        max(max(v["max_abs_dlogit_b0"], v["max_abs_dlogit_b8"]) for v in guard_all.values()), len(guard_all)))


def guard(p4, bb, df_new, cond, cache, probes, dev, embed_images, n=32):
    """D-008: 32 fixed NEW images (seed 0, sorted shas; 16+16 real/fake when fakes exist), accelerator features (from the cache) vs CPU
    recomputation (TF32 off, batch 4); also the |dlogit| of the B0 (block 12) and block-8 probes."""
    r, f = sorted(df_new.sha256[df_new.label == 0]), sorted(df_new.sha256[df_new.label == 1])
    rng = np.random.default_rng(SEED)
    if f:
        sub = [r[i] for i in rng.choice(len(r), min(n // 2, len(r)), replace=False)] + [f[i] for i in rng.choice(len(f), min(n // 2, len(f)), replace=False)]
    else:
        sub = [r[i] for i in rng.choice(len(r), min(n, len(r)), replace=False)]
    keys, X = P.read_shards(cond, cache)
    pos = {k: i for i, k in enumerate(keys.tolist())}
    cp = Path(cache) / "cpuref" / f"{P.safe(cond)}.npz"
    ref = {}
    if cp.exists():
        z = np.load(cp, allow_pickle=False)
        ref = dict(zip(z["keys"].tolist(), z["feats"]))
    miss = [s for s in sub if s not in ref]
    if miss:
        d = df_new.set_index("sha256", drop=False).loc[miss]
        imgs = [p4.load_cond(p, s, cond) for p, s in zip(d.path, d.sha256)]
        ref.update(zip(miss, embed_images(bb, imgs, 4, "cpu")))
        cp.parent.mkdir(parents=True, exist_ok=True)
        ks = sorted(ref)
        np.savez(cp.with_name("tmp_" + cp.name), keys=np.array(ks), feats=np.stack([ref[k] for k in ks]))
        os.replace(cp.with_name("tmp_" + cp.name), cp)
    A = np.stack([X[pos[s]] for s in sub])
    C = np.stack([ref[s] for s in sub])
    d = np.abs(A - C)
    scaled = float((d / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max())
    out = dict(n=len(sub), device=dev, max_abs_feature_diff=float(d.max()), max_scaled_feature_diff=scaled, feat_tol=p4.FEAT_TOL, logit_tol=p4.LOGIT_TOL)
    for k, nm in (("F_last", "b0"), ("F_b8", "b8")):
        sl = slice(FEATS[k].start, FEATS[k].stop)
        out[f"max_abs_dlogit_{nm}"] = float(np.abs(probes[k].decision_function(A[:, sl]) - probes[k].decision_function(C[:, sl])).max())
    out["ok"] = bool(scaled <= p4.FEAT_TOL and max(out["max_abs_dlogit_b0"], out["max_abs_dlogit_b8"]) <= p4.LOGIT_TOL)
    log(f"  D-008 {cond}: max|dF|={out['max_abs_feature_diff']:.2e} scaled={scaled:.2e} |dlogit| b0 {out['max_abs_dlogit_b0']:.2e} b8 {out['max_abs_dlogit_b8']:.2e} -> {'ok' if out['ok'] else 'FAIL'}")
    return out


# ----------------------------------------------------------------------------- stage 4: fits
_D = {}                                                       # arm -> Data, filled before fork


def source_sets(arm):
    """Training-source subsets needed: every non-empty proper subset of the arm's sources (held-out source = complement is a superset)."""
    s = ARMS[arm]["sources"]
    out = []
    for k in range(1, len(s)):
        out += [tuple(c) for c in itertools.combinations(s, k)]
    return out


def train_rows(D, srcs, model):
    pools = [p for s in srcs for p in SRC[s]["train_pools"]]
    real = D.idx("train", pools, label=0)
    gens = [g for g in arm_gens(D.arm) if model != f"g_{g}"]
    fake = D.idx("train", None, gens=gens, label=1)
    return np.sort(np.concatenate([real, fake]))


def fit_scores(D, feat, srcs, model):
    rows = train_rows(D, srcs, model)
    cols = FEATS[feat]
    tc = D.cfg["t_aug"]
    X = np.concatenate([D.X[c][rows][:, cols] for c in tc])
    y = np.concatenate([D.lab[rows]] * len(tc))
    g = np.concatenate([D.groups[rows]] * len(tc))
    m = probe.fit_probe(X, y, SEED, groups=g, class_weight="balanced")
    sc = np.stack([m.decision_function(D.X[c][:, cols]) for c in D.conds]).astype(np.float32)
    return sc, float(m[-1].C_[0]), int(len(rows))


def cache_path(arm, feat, srcs):
    return CACHE / f"scores_{arm}__{feat}__{'+'.join(srcs)}.npz"


def _task(args):
    arm, feat, srcs, model = args
    t0 = time.time()
    return args, fit_scores(_D[arm], feat, srcs, model), time.time() - t0


def stage_fit(arms, workers):
    todo = [(a, f, s) for a in arms for s in source_sets(a) for f in FEAT_ORDER if not cache_path(a, f, s).exists()]
    if not todo:
        log("fit: all cached")
        return
    for a in {t[0] for t in todo}:
        _D[a] = Data(a)
        log(f"loaded {a}: roles {pd.Series(_D[a].role).value_counts().to_dict()}")
    tasks = [(a, f, s, m) for a, f, s in todo for m in model_names(a)]
    log(f"fit: {len(todo)} combos, {len(tasks)} probes, {workers} workers")
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(workers) as ex:
        for args, out, dt in ex.map(_task, tasks):
            a, f, s, m = args
            res[args] = out
            log(f"  {a} {f} train={'+'.join(s)} {m}: n_train={out[2]} C={out[1]:.3g} ({dt:.0f}s)")
            full = [res.get((a, f, s, mm)) for mm in model_names(a)]
            if all(x is not None for x in full):
                np.savez(cache_path(a, f, s), scores=np.stack([x[0] for x in full]), C=np.array([x[1] for x in full]),
                         n_train=np.array([x[2] for x in full]))
    log(f"fit done in {time.time() - t0:.0f}s")


def load_scores(arm, feat, srcs):
    z = np.load(cache_path(arm, feat, srcs))
    names = model_names(arm)
    return {m: z["scores"][i] for i, m in enumerate(names)}, z["C"], z["n_train"]      # model -> [conds, N]


# ----------------------------------------------------------------------------- bootstrap, metrics
def summ(arr):
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return [float("nan")] * 2
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def cluster_key(D, i):
    """Bootstrap cluster of a test-role row: 'fam:' (caption family / D3 id: RAISE reals + every fake), 'dev:' (VISION), 'img:'."""
    if D.lab[i] == 1 or D.pool[i] in ("raise", "d3_real"):
        assert D.rid[i], (D.df.src_path.iat[i], "test row without family id")
        return "fam:" + D.rid[i]
    if D.pool[i].startswith("vision"):
        return "dev:" + D.df.device.iat[i]
    return "img:" + D.df.sha256.iat[i]


class Plan:
    """One global cluster-bootstrap plan per arm over all test-role rows (W[0] = ones = point estimate). Within each kind of cluster
    (families / devices / images) K clusters are drawn with replacement, K = number of distinct clusters of that kind; fakes and RAISE/D3
    reals with the same family id move together. The same weights serve every source, method and condition (paired differences)."""

    def __init__(self, D, n_boot=B, seed=SEED):
        self.rows = D.idx("test")
        self.pos = {int(r): j for j, r in enumerate(self.rows)}
        keys = [cluster_key(D, i) for i in self.rows]
        kinds = np.array([k.split(":")[0] for k in keys])
        rng = np.random.default_rng(seed)
        W = np.ones((n_boot + 1, len(self.rows)))
        for kind in ("fam", "dev", "img"):
            m = np.where(kinds == kind)[0]
            if not len(m):
                continue
            uk, code = np.unique(np.array(keys)[m], return_inverse=True)
            K = len(uk)
            cnt = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)]).astype(float)
            W[1:, m] = cnt[:, code]
        self.W = W

    def w(self, rows):
        return self.W[:, [self.pos[int(r)] for r in rows]]


def thr_fpr(scores, fpr=None):
    return metrics.threshold_for_fpr(scores, CFG["fpr_target"] if fpr is None else fpr)


def macro_of(ga, gens):
    return np.mean([ga[g] for g in gens], 0)


def src_eval_rows(D, S):
    return D.idx("test", SRC[S]["eval_pools"], label=0)


def calib_rows(D, srcs):
    return D.idx("calib", [p for s in srcs for p in SRC[s]["eval_pools"]], label=0)


def subsets_of(others):
    return [tuple(c) for k in range(1, len(others) + 1) for c in itertools.combinations(others, k)]


def evaluate_arm(arm):
    D = Data(arm, load_feats=False)
    plan = Plan(D)
    sources = ARMS[arm]["sources"]
    gens = arm_all_gens(arm)
    conds = D.conds
    Rr = {S: src_eval_rows(D, S) for S in sources}
    Ff = {g: D.idx("test", None, gens=[g], label=1) for g in gens}
    WR = {S: plan.w(r) for S, r in Rr.items()}
    WF = {g: plan.w(f) for g, f in Ff.items()}
    for g in gens:
        assert len(Ff[g]) > 0 and (WF[g].sum(1) > 0).all(), (arm, g)
    log(f"{arm}: roles {pd.Series(D.role).value_counts().to_dict()}; test reals { {S: len(r) for S, r in Rr.items()} }; test fakes { {g: len(f) for g, f in Ff.items()} }")
    out = dict(arm=arm, n_test_reals={S: len(r) for S, r in Rr.items()}, n_test_fakes={g: len(f) for g, f in Ff.items()}, heldout={}, owner=None)
    ga_all, long_rows = {}, []
    names = model_names(arm)
    for S in sources:
        others = [s for s in sources if s != S]
        out["heldout"][S] = {}
        for T in subsets_of(others):
            tkey = "+".join(T)
            cal = calib_rows(D, T)
            e = {}
            for feat in FEAT_ORDER:
                SC, Cs, ntr = load_scores(arm, feat, T)
                ef = dict(C=[float(c) for c in Cs], n_train=[int(n) for n in ntr], n_calib=len(cal), conds={})
                for c in conds:
                    ci = conds.index(c)
                    ga = {}
                    for g in gens:
                        s = SC[model_of(arm, g)][ci].astype(np.float64)
                        ga[g] = metrics.weighted_auroc(s[Rr[S]], s[Ff[g]], WR[S], WF[g])
                    ga_all[(S, T, feat, c)] = ga
                    ec = dict(macro=summ(macro_of(ga, gens)), flux=summ(ga[FLUX]), per_gen={g: float(ga[g][0]) for g in gens})
                    # FPR at the 1% threshold, per model (for drop-one-generator robustness); readings A (clean calib) and B (same-condition calib)
                    fp = {}
                    for reading in ("A", "B"):
                        fp[reading] = {}
                        for m in names:
                            sm = SC[m]
                            thr = thr_fpr(sm[conds.index("clean")][cal]) if reading == "A" else thr_fpr(sm[ci][cal])
                            s = sm[ci]
                            hit = (s[Rr[S]] >= thr).astype(float)
                            k, n = int(hit.sum()), len(hit)
                            bfpr = (WR[S] @ hit) / WR[S].sum(1)
                            fp[reading][m] = dict(thr=float(thr), fpr=k / n, k=k, n=n, cp95=clopper_pearson(k, n), boot_ci=[float(v) for v in np.percentile(bfpr[1:], [2.5, 97.5])])
                    ec["fpr"] = fp
                    # TPR (macro over gens) at reading A's full-model threshold, for context
                    thrA = fp["A"]["full"]["thr"]
                    ec["tpr_full_A"] = float(np.mean([(SC["full"][ci][Ff[g]] >= thrA).mean() for g in gens]))
                    if S == "VISION":     # per-pool FPR on the VISION pools (reading A, full model)
                        ec["fpr_pools"] = {}
                        for pk, pools in (("native", ["vision_native"]), ("WA", ["vision_whatsapp"]), ("FBH", ["vision_facebook_hq"])):
                            r = D.idx("test", pools, label=0)
                            k = int((SC["full"][ci][r] >= thrA).sum())
                            ec["fpr_pools"][pk] = dict(fpr=k / len(r), k=k, n=len(r), cp95=clopper_pearson(k, len(r)))
                    ef["conds"][c] = ec
                e[feat] = ef
                if T == tuple(others):          # scores_long: the full-others models only (calib/test rows)
                    keep = np.where(np.isin(D.role, ["calib", "test"]))[0]
                    for c in conds:
                        ci = conds.index(c)
                        sc = np.array([SC[model_of(arm, D.gen[i])][ci][i] if D.lab[i] == 1 else SC["full"][ci][i] for i in keep])
                        long_rows.append(pd.DataFrame(dict(arm=arm, heldout=S, train_sources=tkey, feat=feat, cond=c, row=keep,
                                                           sha256=D.df.sha256.to_numpy()[keep], label=D.lab[keep], generator=D.gen[keep],
                                                           pool=D.pool[keep], role=D.role[keep], score=np.round(sc[:], 4).astype(np.float32))))
            out["heldout"][S][tkey] = e
    out["_ga"], out["_D"], out["_plan"] = ga_all, D, plan
    return out, long_rows


def diff_summary(a, b=None):
    d = a if b is None else a - b
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    return dict(delta=float(d[0]), ci_lo=float(lo), ci_hi=float(hi), p_le0=float((d[1:] <= 0).mean()))


def derived(arm, R, keep, fpr_model="full", reading="A", feat="F_b8"):
    """Pass/kill quantities at the arm's laundering condition for a generator subset `keep` (macro), training on ALL other sources."""
    cfg = ARMS[arm]
    sources, jr = cfg["sources"], cfg["launder_cond"]
    ga = R["_ga"]
    per = {}
    for S in sources:
        T = tuple(s for s in sources if s != S)
        auc = macro_of(ga[(S, T, feat, jr)], keep)
        fp = R["heldout"][S]["+".join(T)][feat]["conds"][jr]["fpr"][reading][fpr_model]
        per[S] = dict(auroc=summ(auc), fpr=fp["fpr"], fpr_cp95=fp["cp95"], fpr_boot_ci=fp["boot_ci"], k=fp["k"], n=fp["n"])
    d_layer = np.mean([macro_of(ga[(S, tuple(s for s in sources if s != S), "F_b8", jr)], keep)
                       - macro_of(ga[(S, tuple(s for s in sources if s != S), "F_last", jr)], keep) for S in sources], 0)
    # learning curve: for each k, per held-out source mean over its k-source training subsets; then mean over held-out sources
    lc = {}
    for k in range(1, len(sources)):
        lc[k] = np.mean([np.mean([macro_of(ga[(S, T, feat, jr)], keep) for T in subsets_of([s for s in sources if s != S]) if len(T) == k], 0)
                         for S in sources], 0)
    kmax = len(sources) - 1
    return dict(per=per, b8_minus_last=diff_summary(d_layer), lc_mean={k: summ(v) for k, v in lc.items()},
                lc_diff_max_vs_1=diff_summary(lc[kmax], lc[1]), lc_diff_steps={f"{k + 1} vs {k}": diff_summary(lc[k + 1], lc[k]) for k in range(1, kmax)},
                mean_auroc=summ(np.mean([macro_of(ga[(S, tuple(s for s in sources if s != S), feat, jr)], keep) for S in sources], 0)))


def label_of(d):
    per = d["per"]
    kill = any(v["auroc"][0] < TH["kill_auroc"] or v["fpr"] > TH["kill_fpr"] for v in per.values())
    ok_src = all(v["auroc"][0] >= TH["pass_auroc"] and v["fpr"] <= TH["pass_fpr"] for v in per.values())
    ok_layer = d["b8_minus_last"]["delta"] >= TH["b8_minus_last"] and d["b8_minus_last"]["ci_lo"] > 0
    ok_lc = d["lc_diff_max_vs_1"]["delta"] > 0 and d["lc_diff_max_vs_1"]["ci_lo"] > 0
    lab = "KILL" if kill else "PASS" if (ok_src and ok_layer and ok_lc) else "INCONCLUSIVE"
    return dict(label=lab, kill=bool(kill), c_sources=bool(ok_src), c_layer=bool(ok_layer), c_learning_curve=bool(ok_lc))


def robustness(arm, R):
    gens = arm_all_gens(arm)
    subsets = {"all": (gens, "full"), f"no {FLUX}": ([g for g in gens if g != FLUX], "full")}
    for g in gens:
        subsets[f"drop {g}"] = ([x for x in gens if x != g], model_of(arm, g))
    res = {}
    for reading in ("A", "B"):
        for nm, (keep, fm) in subsets.items():
            d = derived(arm, R, keep, fm, reading)
            res[f"{reading}|{nm}"] = dict(label=label_of(d), derived=d)
    return res


# ----------------------------------------------------------------------------- stage 5: eval + tables
def run_eval():
    RES.mkdir(exist_ok=True)
    allres, long_all = {}, []
    for arm in ARMS:
        R, lr = evaluate_arm(arm)
        R["robust"] = robustness(arm, R)
        long_all += lr
        D = R["_D"]
        allres[arm] = {k: v for k, v in R.items() if not k.startswith("_")}
        allres[arm]["main"] = {rd: R["robust"][f"{rd}|all"] for rd in ("A", "B")}
        log(f"{arm}: label A={R['robust']['A|all']['label']['label']} B={R['robust']['B|all']['label']['label']}")
    owner_dir = ROOT / CFG["owner"]["dir"]
    n_owner = len(list(owner_dir.glob("*"))) if owner_dir.exists() else 0
    owner = dict(status="pending", dir=CFG["owner"]["dir"], n_images=n_owner, min_images=CFG["owner"]["min_images"],
                 reason="directory missing" if not owner_dir.exists() else "fewer than the required images")
    for a in allres:
        allres[a]["owner"] = owner
    pd.concat(long_all).to_csv(RES / "scores_long.csv.gz", index=False, compression="gzip")
    meta = dict(git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                config=CFG, owner=owner)
    (RES / "results.json").write_text(json.dumps(dict(meta=meta, arms=allres), indent=1, default=float))
    write_tables(allres)
    log("eval done")


def f3(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def gi(d):
    return f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def write_tables(allres):
    L = ["# E7 (016-loso) tables", "",
         "Generated by run.py from results.json. Exploratory (D-016). OWNER photos: PENDING (data/raw/owner_phone/ missing), so labels cover the available sources only "
         "(PASS would be provisional; a KILL stands). AUROC = macro over the arm's generators (LOGO model per train generator, 'full' model for FLUX) of the held-out "
         "source's TEST reals vs the held-out generators' TEST fakes. CIs: 1,000 cluster-bootstrap replicates (devices / caption families / images).", ""]
    for arm, R in allres.items():
        cfg = ARMS[arm]
        sources, conds, jr = cfg["sources"], cfg["conditions"], cfg["launder_cond"]
        L += [f"## Arm `{arm}` ({cfg['role']})", "", f"Test reals {R['n_test_reals']}; test fakes per generator {R['n_test_fakes']}", ""]
        if "note" in cfg:
            L += [f"Note: {cfg['note']}", ""]
        L += [f"### Labels at {jr}, F_b8, T-aug, training on ALL other sources", "",
              "| FPR threshold reading | label | kill? | sources ok (AUROC>=0.85 & FPR<=5%) | b8-last ok | learning curve ok |", "|---|---|---|---|---|---|"]
        for rd in ("A", "B"):
            lb = R["main"][rd]["label"]
            L.append(f"| {rd} ({'clean-calib threshold' if rd == 'A' else 'same-condition calib threshold'}) | **{lb['label']}** | {lb['kill']} | {lb['c_sources']} | {lb['c_layer']} | {lb['c_learning_curve']} |")
        L += ["", f"### Held-out source: AUROC and FPR at {jr} (training on all other sources)", "",
              "| held-out | feat | AUROC [CI] | FPR reading A (k/n) [CP95] {cluster-boot CI} | FPR reading B | TPR@A thr |", "|---|---|---|---|---|---|"]
        for S in sources:
            T = "+".join(s for s in sources if s != S)
            for feat in FEAT_ORDER:
                ec = R["heldout"][S][T][feat]["conds"][jr]
                fa, fb = ec["fpr"]["A"]["full"], ec["fpr"]["B"]["full"]
                f_ = lambda x: f"{x['fpr']:.3f} ({x['k']}/{x['n']}) [{x['cp95'][0]:.3f}, {x['cp95'][1]:.3f}] {{{x['boot_ci'][0]:.3f}, {x['boot_ci'][1]:.3f}}}"
                L.append(f"| {S} (train {T}) | {feat} | {f3(ec['macro'])} | {f_(fa)} | {f_(fb)} | {ec['tpr_full_A']:.3f} |")
        L += ["", "### Held-out source: macro AUROC at every condition (train on all other sources)", "", "| held-out | feat | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for S in sources:
            T = "+".join(s for s in sources if s != S)
            for feat in FEAT_ORDER:
                L.append(f"| {S} | {feat} | " + " | ".join(f3(R['heldout'][S][T][feat]['conds'][c]['macro']) for c in conds) + " |")
        L += ["", "### Held-out source: FLUX AUROC (unseen generator) at every condition", "", "| held-out | feat | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for S in sources:
            T = "+".join(s for s in sources if s != S)
            for feat in FEAT_ORDER:
                L.append(f"| {S} | {feat} | " + " | ".join(f3(R['heldout'][S][T][feat]['conds'][c]['flux']) for c in conds) + " |")
        L += ["", "### FPR on the held-out source at every condition, F_b8 (reading A clean-calib threshold | reading B same-condition threshold), 'full' model", "",
              "| held-out | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for S in sources:
            T = "+".join(s for s in sources if s != S)
            cells = []
            for c in conds:
                fp = R["heldout"][S][T]["F_b8"]["conds"][c]["fpr"]
                cells.append(f"{fp['A']['full']['fpr']:.3f} ({fp['A']['full']['k']}/{fp['A']['full']['n']}) | {fp['B']['full']['fpr']:.3f}")
            L.append(f"| {S} | " + " | ".join(cells) + " |")
        if "VISION" in sources:
            L += ["", f"### VISION held-out, FPR per pool at {jr} (F_b8, reading A)", ""]
            fp = R["heldout"]["VISION"]["+".join(s for s in sources if s != "VISION")]["F_b8"]["conds"][jr]["fpr_pools"]
            L += ["| pool | FPR (k/n) [CP95] |", "|---|---|"] + [f"| {p} | {v['fpr']:.3f} ({v['k']}/{v['n']}) [{v['cp95'][0]:.3f}, {v['cp95'][1]:.3f}] |" for p, v in fp.items()]
        for rd in ("A",):
            d = R["main"][rd]["derived"]
            L += ["", f"### Pass criteria (all generators), reading {rd}", "",
                  f"- mean held-out AUROC F_b8: {f3(d['mean_auroc'])}",
                  f"- mean(F_b8 - F_last) over held-out sources: {gi(d['b8_minus_last'])}",
                  "- learning curve (mean held-out AUROC by number of training sources): " + ", ".join(f"k={k}: {f3(v)}" for k, v in d["lc_mean"].items()),
                  f"- max-k minus 1 source: {gi(d['lc_diff_max_vs_1'])}"]
            for kk, v in d["lc_diff_steps"].items():
                L.append(f"- step {kk}: {gi(v)}")
        L += ["", "### Learning curve per held-out source (macro AUROC at the laundering condition, F_b8 | F_last)", "", "| held-out | train sources | F_b8 | F_last | FPR A (F_b8, full) |", "|---|---|---|---|---|"]
        for S in sources:
            for T, e in R["heldout"][S].items():
                fa = e["F_b8"]["conds"][jr]["fpr"]["A"]["full"]
                L.append(f"| {S} | {T} | {f3(e['F_b8']['conds'][jr]['macro'])} | {f3(e['F_last']['conds'][jr]['macro'])} | {fa['fpr']:.3f} ({fa['k']}/{fa['n']}) |")
        L += ["", "### Drop-one-generator robustness (label of the criteria with one generator dropped from the macros; FPR from the LOGO model that never saw it)", "",
              "| subset | label A | label B | min held-out AUROC | max FPR (A) | b8-last | lc diff |", "|---|---|---|---|---|---|---|"]
        for k, v in R["robust"].items():
            if not k.startswith("A|"):
                continue
            nm = k[2:]
            b = R["robust"]["B|" + nm]["label"]["label"]
            d = v["derived"]
            L.append(f"| {nm} | {v['label']['label']} | {b} | {min(x['auroc'][0] for x in d['per'].values()):.3f} | {max(x['fpr'] for x in d['per'].values()):.3f} | "
                     f"{gi(d['b8_minus_last'])} | {gi(d['lc_diff_max_vs_1'])} |")
        L.append("")
    (RES / "tables.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["splits", "prep", "feats", "fit", "eval", "all"], nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    if a.stage in ("splits", "all"):
        stage_splits()
    if a.stage in ("prep", "all"):
        stage_prep()
    if a.stage in ("feats", "all"):
        stage_feats()
    if a.stage in ("fit", "all"):
        stage_fit(a.arms, a.workers)
    if a.stage in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
