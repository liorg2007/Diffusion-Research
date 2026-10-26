"""E10 (Amendment 19): does real-source DIVERSITY fix the unseen-pipeline problem? Generalises 016-loso/run.py (E7). Resumable stages.

    env/.venv/bin/python -I experiments/019-source-diversity/run.py all [--workers 14]
Stages (each skips finished work): splits -> prep -> feats -> fit -> eval
  splits  016's splits_fov512.csv rows (3,519 = 003/004 + 600 Open Images) + the new sources UNS / WMC / HDRP from the three manifests,
          split by group 300/100/200 (seed 0) -> _prep/new_fov512.csv, results/split_counts.json. `--dry-run`: logic only, writes nothing.
  prep    crops for the NEW rows only, via 004 fov_prep.py --no-clobber (never overwrites a file) -> splits_fov512.csv
  feats   004 pipeline (blocks 4/8/12 CLS+mean, DINOv2-base) for the NEW (image, condition) pairs -> cache/feats/; D-008 guard
  fit     probes per (feature, training-source set, variant) with LOGO over fake generators -> cache/scores_*.npz
            F_b8: every non-empty subset T of the 6 sources with |T| <= 5 (62 sets; a fit on T serves every held-out S not in T)
            F_last: |T| = 5 (6 sets) + the 12 |T| = 4 sets of the drop-one-source labels (config drop_source_flast)
            size-matched control: F_b8, T = the 5 other sources, training reals subsampled by group, 5 seeds (30 sets)
  eval    metrics, bootstrap, labels (Amendment 19) -> results/{results.json, tables.md, scores_long.csv.gz}
Design constants are in config.json. Leak checks (Amendment 19) are asserts: audit_train_rows / audit_calib / audit_eval / audit_fake_rows /
audit_same_prep / audit_taug (called in the fit workers and in every evaluated fold).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd

import metrics
import probe
import p4features as P

CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CACHE, RES, PREP = HERE / "cache", HERE / "results", HERE / "_prep"
ARM = CFG["arm_cfg"]
CONDS, TAUG, JR = ARM["conditions"], ARM["t_aug"], ARM["launder_cond"]
TRAINGENS, FLUX = CFG["train_generators"], CFG["test_only_generator"]
ALLGENS = TRAINGENS + [FLUX]
FEATS = {k: slice(*v) for k, v in CFG["features"].items()}
FEAT_ORDER = ["F_b8", "F_last"]
ORDER = CFG["source_order"]
SRC = CFG["sources"]
NEWS = CFG["new_sources"]
E7S = CFG["e7_sources"]
TH = CFG["thresholds"]
GROUP_POOLS = [p for s in NEWS for p in SRC[s]["train_pools"]]      # pools whose reals are clustered by `group`
NEW_POOLS = (CFG["oi_pool"],) + tuple(GROUP_POOLS)
BASE_SPLITS = ROOT / ARM["base_splits"]
FINAL_SPLITS = HERE / "splits_fov512.csv"
N_CTL = CFG["ctl"]["n_seeds"]
BASE_ROWS_OLD = ARM["n_base_rows_003_004"]


def log(*a):
    print(time.strftime("%H:%M:%S"), "E10", *a, flush=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e7():
    """The E7 module (guard, Clopper-Pearson, summaries, E6 roles). Imported by path, never modified."""
    if "e7run" not in sys.modules:
        sys.modules["e7run"] = _load_module("e7run", ROOT / "experiments" / "016-loso" / "run.py")
    return sys.modules["e7run"]


def canon(srcs, order=None):
    order = order or ORDER
    return tuple(s for s in order if s in set(srcs))


def tkey(T, var=""):
    return "+".join(T) + (f"@{var}" if var else "")


def subsets_of(others):
    return [tuple(c) for k in range(1, len(others) + 1) for c in itertools.combinations(others, k)]


def model_names():
    return [f"g_{g}" for g in TRAINGENS] + ["full"]


def model_of(g):
    return f"g_{g}" if g in TRAINGENS else "full"


def thr_fpr(scores):
    return metrics.threshold_for_fpr(scores, CFG["fpr_target"])


def summ(arr):
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def diff_summary(a, b=None):
    d = a if b is None else a - b
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    return dict(delta=float(d[0]), ci_lo=float(lo), ci_hi=float(hi), p_le0=float((d[1:] <= 0).mean()))


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0:
        return [float("nan")] * 2
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


# ----------------------------------------------------------------------------- stage 1: splits
def split_groups(groups, caps=None, seed=None):
    """Assign WHOLE groups to train / calib / test. `groups`: one group id per image. Groups are visited in the order of a seeded
    permutation of the sorted group ids and put first-fit into train, then calib, then test (cap = n_train / n_calib / n_test images);
    a group that fits nowhere goes to the bucket with the most room left. Returns (list of split names per image, counts dict)."""
    c = CFG["new_split"]
    caps = caps or (c["n_train"], c["n_calib"], c["n_test"])
    seed = c["seed"] if seed is None else seed
    names = ("train", "calib", "test")
    groups = [str(g) for g in groups]
    gs = sorted(set(groups))
    size = Counter(groups)
    perm = np.random.default_rng(seed).permutation(len(gs))
    used = [0, 0, 0]
    where = {}
    for j in perm:
        g = gs[j]
        s = size[g]
        b = next((i for i in range(3) if used[i] + s <= caps[i]), None)
        if b is None:
            b = int(np.argmax([caps[i] - used[i] for i in range(3)]))
        where[g] = b
        used[b] += s
    sp = [names[where[g]] for g in groups]
    return sp, dict(zip(names, used))


def new_rows(dry=False):
    """Rows of the new sources from their manifests (image_id, path, sha256, width, height, license, group)."""
    rows, counts = [], {}
    for S in NEWS:
        mp = HERE / SRC[S]["manifest"]
        if not mp.exists():
            if dry:
                log(f"splits (dry-run): {mp.name} missing, {S} skipped")
                continue
            raise SystemExit(f"{mp} is missing; the fetch_*.py scripts must finish first")
        m = pd.read_csv(mp, keep_default_na=False, dtype=str)
        need = {"image_id", "path", "sha256", "width", "height", "license", "group"}
        assert need <= set(m.columns), (mp.name, need - set(m.columns))
        assert m.sha256.is_unique and m.image_id.is_unique, f"{S}: duplicate image or sha256"
        assert (m.license != "").all() and (m.group != "").all(), f"{S}: empty license or group"
        n0 = len(m)
        short = m[["width", "height"]].astype(int).min(axis=1)
        m = m[short >= ARM["prep_size"]].reset_index(drop=True)
        sp, cnt = split_groups(m.group.tolist())
        df = pd.DataFrame(dict(sha256=m.sha256, path="", src_path=m.path, label=0, generator="real", pool=SRC[S]["train_pools"][0],
                               camera_or_platform="", device="", vidx="", split=sp, in_scoring_set=[s == "test" for s in sp],
                               b0_train=False, orig_format="JPEG", source_dataset=SRC[S]["source_dataset"], license=m.license, group=m.group))
        gsplit = df.groupby("group").split.nunique()
        assert (gsplit == 1).all(), f"{S}: a group was split across roles"
        gcount = df.groupby("split").group.nunique().to_dict()
        counts[S] = dict(n_manifest=n0, n_short_lt_512_dropped=n0 - len(m), n_images=len(df), images=cnt, n_groups=gcount,
                         max_group_size=int(df.group.value_counts().max()))
        rows.append(df)
    return (pd.concat(rows, ignore_index=True) if rows else None), counts


def owner_status():
    d = ROOT / CFG["owner"]["dir"]
    n = len([p for p in d.glob("*") if p.is_file()]) if d.exists() else 0
    return dict(status="pending" if n < CFG["owner"]["min_images"] else "PRESENT-NOT-USED", dir=CFG["owner"]["dir"], n_images=n,
                min_images=CFG["owner"]["min_images"],
                reason="directory missing" if not d.exists() else "fewer than the required images" if n < CFG["owner"]["min_images"]
                else ">= 150 images exist but Amendment 19 defines no OWNER split: lead must decide (see notes.md, Proposed deviations)")


def stage_splits(dry=False):
    base = pd.read_csv(BASE_SPLITS, keep_default_na=False, dtype={"vidx": str})
    assert base.sha256.is_unique
    nd, counts = new_rows(dry)
    log(f"splits: base {len(base)} rows (pools {base.pool.value_counts().to_dict()})")
    if nd is None:
        log("splits (dry-run): no new manifests present")
        return counts
    assert not set(nd.sha256) & set(base.sha256), "new rows overlap existing rows"
    assert nd.sha256.is_unique, "a photo appears in two new sources"
    for S, c in counts.items():
        log(f"splits: {S}: {c}")
    if dry:
        return counts
    PREP.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    nd.to_csv(PREP / "new_fov512.csv", index=False)
    (RES / "split_counts.json").write_text(json.dumps(dict(counts=counts, owner=owner_status()), indent=1))
    return counts


# ----------------------------------------------------------------------------- stage 2: crops
def stage_prep():
    if FINAL_SPLITS.exists():
        log(f"prep: {FINAL_SPLITS.name} exists, skipped")
        return
    out = PREP / "out_fov512.csv"
    cmd = [sys.executable, "-I", str(ROOT / "experiments" / "004-phase4-features" / "fov_prep.py"), "--splits", str(PREP / "new_fov512.csv"),
           "--out-splits", str(out), "--report", str(RES / "prep_report_fov512.json"), "--size", str(ARM["prep_size"]),
           "--mode", ARM["prep_mode"], "--no-clobber", "--workers", "8"]
    log("prep:", " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (RES / "prep_fov512.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"fov_prep failed: {r.stderr[-500:]}")
    base = pd.read_csv(BASE_SPLITS, keep_default_na=False, dtype={"vidx": str})
    base["group"] = ""
    nd = pd.read_csv(out, keep_default_na=False, dtype={"vidx": str, "group": str}).astype(base.dtypes.to_dict())
    full = pd.concat([base, nd], ignore_index=True)
    assert full.sha256.is_unique
    full.to_csv(FINAL_SPLITS, index=False)
    rep = json.loads((RES / "prep_report_fov512.json").read_text())
    log(f"prep: ok {rep['n_ok']}, short<size {rep['n_short_side_lt_512']}, no_clobber {rep['no_clobber']}, wrote {FINAL_SPLITS.name} ({len(full)} rows)")


# ----------------------------------------------------------------------------- data
def roles_and_rid(df):
    """E6 roles for the existing 003/004 rows (caption-family majority rule); every other row (OI and the new sources): role = split.
    rid = caption/family id for the 003/004 rows, '' otherwise."""
    new = df.pool.isin(NEW_POOLS).to_numpy()
    role = np.empty(len(df), object)
    rid = np.array([""] * len(df), object)
    b = df[~new].reset_index(drop=True)
    if (df.pool == CFG["oi_pool"]).any():            # the real splits file: exactly the 2,919 003/004 rows besides OI and the new sources
        assert len(b) == BASE_ROWS_OLD, (len(b), BASE_ROWS_OLD)
    r6, i6, _ = e7().e6().build_roles(b)
    role[~new], rid[~new] = r6, i6
    role[new] = df.split.to_numpy()[new]
    return role, rid


def audit_same_prep(df):
    """Identical transforms for both classes: every row (real or fake, any source) comes from the same fov512 crop directory."""
    p = df.path.to_numpy()
    sha = df.sha256.to_numpy()
    assert all(x == f"data/processed/fov512/images/{s}.png" for x, s in zip(p, sha)), "a row is not a fov512 crop of its own sha256"


class Data:
    """Splits + roles + cached DINOv2 secondary features (4608-d): caches 004 (old rows), 016 (OI) and cache/feats (new rows)."""

    def __init__(self, load_feats=True, splits=None, strict_paths=True):
        self.cfg = ARM
        self.df = pd.read_csv(splits or FINAL_SPLITS, keep_default_na=False, dtype={"vidx": str, "group": str})
        if "group" not in self.df:
            self.df["group"] = ""
        if strict_paths:
            audit_same_prep(self.df)
        self.role, self.rid = roles_and_rid(self.df)
        self.groups = e7().e6().make_groups(self.df, self.rid)
        gp = self.df.pool.isin(GROUP_POOLS).to_numpy()
        self.groups = np.where(gp, ["grp:" + p + ":" + g for p, g in zip(self.df.pool, self.df.group)], self.groups)
        assert not (gp & (self.df.group == "")).any(), "a new-source row without a group"
        self.lab = self.df.label.to_numpy()
        self.gen = self.df.generator.to_numpy()
        self.pool = self.df.pool.to_numpy()
        self.conds = CONDS
        self.X = {}
        if load_feats:
            for c in self.conds:
                self.X[c] = load_x(self.df, c)

    def idx(self, role=None, pools=None, gens=None, label=None):
        m = np.ones(len(self.df), bool) if role is None else self.role == role
        if pools is not None:
            m &= np.isin(self.pool, pools)
        if gens is not None:
            m &= np.isin(self.gen, gens)
        if label is not None:
            m &= self.lab == label
        return np.where(m)[0]


def feature_caches():
    return [P.cache_root(ARM["cache004"]), ROOT / ARM["cache016"], CACHE / "feats"]


def load_x(df, cond):
    store = {}
    for d in feature_caches():
        store.update(e7().read_cache_dir(cond, d))
    miss = [s for s in df.sha256 if s not in store]
    if miss:
        raise SystemExit(f"fov512/{cond}: {len(miss)} rows have no cached features (run the feats stage)")
    x = np.stack([store[s] for s in df.sha256]).astype(np.float32)
    assert x.shape[1] == 4608 and np.isfinite(x).all()
    return x


# ----------------------------------------------------------------------------- stage 3: features (004 pipeline) + D-008 guard
def stage_feats(workers=4):
    p4 = _load_module("p4run", ROOT / "experiments" / "004-phase4-features" / "run.py")
    from concurrent.futures import ThreadPoolExecutor
    from features import embed_images, get_device
    dev = get_device().type
    bb = p4.secondary_backbone()
    p4.check_design_vs_003(bb)
    guard_f = RES / "d008_guard.json"
    guard_all = json.loads(guard_f.read_text()) if guard_f.exists() else {}
    df = pd.read_csv(FINAL_SPLITS, keep_default_na=False, dtype={"vidx": str, "group": str})
    d003 = pd.read_csv(ROOT / "experiments/003-baselines-eval/splits.csv", keep_default_na=False, dtype={"vidx": str})
    cache = CACHE / "feats"
    cache.mkdir(parents=True, exist_ok=True)
    x0 = P.load_features_aligned(d003, "secondary", "clean", subdir=ARM["cache004"])
    tr, y0 = d003.b0_train.to_numpy(), d003.label.to_numpy()
    probes = {k: probe.fit_probe(x0[tr][:, FEATS[k]], y0[tr], SEED) for k in FEAT_ORDER}
    with ThreadPoolExecutor(workers) as pool:
        for cond in CONDS:
            have = set()
            for d in feature_caches()[:2]:
                have |= set(e7().read_cache_dir(cond, d))
            todo = df[~df.sha256.isin(have)]
            key = f"fov512/{cond}"
            if not len(todo):
                log(f"feats {key}: all rows cached")
                continue
            if cond not in P.EVAL_CONDITIONS:
                raise SystemExit(f"{cond} is not a 004 eval condition")
            t = p4.extract_condition(todo.assign(in_scoring_set=True), bb, cond, cache, dev, pool, p4.P.SPEC)
            log(f"feats {key}: {t}")
            g = e7().guard(p4, bb, todo, cond, cache, probes, dev, embed_images)
            guard_all[key] = g
            guard_f.write_text(json.dumps(guard_all, indent=1, sort_keys=True))
            if not g["ok"]:
                raise SystemExit(f"D-008 ABORT {key}: {g}")
    if guard_all:
        log("feats done; D-008 guard max|dlogit| = %.2e over %d condition(s)" % (
            max(max(v["max_abs_dlogit_b0"], v["max_abs_dlogit_b8"]) for v in guard_all.values()), len(guard_all)))


# ----------------------------------------------------------------------------- leak checks (Amendment 19), as asserts
def real_train_counts(D):
    return {s: int(len(D.idx("train", SRC[s]["train_pools"], label=0))) for s in ORDER}


def ctl_target(counts, sources, S):
    """Mean over the k=2 subsets of the other sources of the summed train-role real counts (rounded)."""
    others = [s for s in sources if s != S]
    return int(round(float(np.mean([sum(counts[s] for s in c) for c in itertools.combinations(others, 2)]))))


def subsample_groups(groups, target, seed):
    """Whole-group subsample of `groups` (one id per row) to at most `target` rows: groups in the order of a seeded permutation of the
    sorted unique ids, a group is taken if it does not overshoot. Returns (boolean mask over rows, achieved count)."""
    ug, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    perm = np.random.default_rng([SEED, 7919, seed]).permutation(len(ug))
    chosen = np.zeros(len(ug), bool)
    tot = 0
    for j in perm:
        if tot == target:
            break
        if tot + cnt[j] <= target:
            chosen[j] = True
            tot += int(cnt[j])
    return chosen[inv], tot


def audit_train_rows(D, T, model, rows, S=None):
    """Training rows of a fit on source set T: role=train only; reals only from T's own train pools (so the held-out source, whatever it
    is, contributes nothing: no train, calib or test row); fakes: no FLUX and not the LOGO generator."""
    rows = np.asarray(rows)
    assert len(rows) == len(np.unique(rows)), "duplicate training rows"
    assert (D.role[rows] == "train").all(), "non-train-role row in training"
    pools_T = {p for s in T for p in SRC[s]["train_pools"]}
    real, fake = rows[D.lab[rows] == 0], rows[D.lab[rows] == 1]
    assert set(D.pool[real]) <= pools_T, f"training reals from pools outside T={T}: {set(D.pool[real]) - pools_T}"
    gens = set(D.gen[fake])
    assert FLUX not in gens, "FLUX in training"
    assert gens <= set(TRAINGENS) and (model == "full" or model[2:] not in gens), f"bad training generators for {model}: {gens}"
    if S is not None:
        assert S not in T
        pS = set(SRC[S]["train_pools"] + SRC[S]["eval_pools"])
        assert not pS & set(D.pool[rows]), f"held-out source {S} leaks into training"


def audit_calib(D, S, T, cal):
    """Threshold reals: calib-role reals of the TRAINING sources only."""
    assert S not in T
    assert len(cal) > 0 and (D.role[cal] == "calib").all() and (D.lab[cal] == 0).all(), "threshold rows are not calib reals"
    allowed = {p for s in T for p in SRC[s]["eval_pools"]}
    assert set(D.pool[cal]) <= allowed, f"threshold reals from pools outside the training sources: {set(D.pool[cal]) - allowed}"
    assert not set(D.pool[cal]) & set(SRC[S]["eval_pools"] + SRC[S]["train_pools"]), "held-out source in the threshold reals"


def audit_eval(D, S, rr):
    """The held-out source's rows used for scoring are its TEST reals only."""
    rr = np.asarray(rr)
    assert len(rr) > 0 and (D.role[rr] == "test").all() and (D.lab[rr] == 0).all()
    assert set(D.pool[rr]) <= set(SRC[S]["eval_pools"]), "evaluation reals outside the held-out source"


def fake_test_hash(D):
    rows = np.concatenate([D.idx("test", None, gens=[g], label=1) for g in ALLGENS])
    return hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()


def audit_fake_rows(D, expect):
    assert fake_test_hash(D) == expect, "fake test rows differ between folds"


def audit_taug(lab_rows, tc):
    """Identical transforms for both classes: every class gets exactly the same set of T-aug conditions, once each."""
    assert len(set(tc)) == len(tc) and set(tc) <= set(CONDS)
    cnt = Counter()
    for c in tc:
        for l in (0, 1):
            cnt[(c, l)] += int((np.asarray(lab_rows) == l).sum())
    for l in (0, 1):
        assert len({cnt[(c, l)] for c in tc}) == 1 and cnt[(tc[0], l)] > 0
    return cnt


# ----------------------------------------------------------------------------- stage 4: fits
_D = {}


def train_rows(D, T, model, real_rows=None):
    pools = [p for s in T for p in SRC[s]["train_pools"]]
    real = D.idx("train", pools, label=0) if real_rows is None else np.asarray(real_rows)
    gens = [g for g in TRAINGENS if model != f"g_{g}"]
    fake = D.idx("train", None, gens=gens, label=1)
    return np.sort(np.concatenate([real, fake]))


def ctl_real_rows(D, T, seed):
    """Size-matched control: the training reals of T (the 5 other sources), subsampled by whole groups to the mean real-train count of
    the k=2 subsets of the held-out source's other sources."""
    S = [s for s in ORDER if s not in T]
    assert len(S) == 1, "the control is defined at k = 5 only"
    real = D.idx("train", [p for s in T for p in SRC[s]["train_pools"]], label=0)
    target = ctl_target(real_train_counts(D), ORDER, S[0])
    mask, tot = subsample_groups(D.groups[real], target, seed)
    assert tot >= 0.9 * target, f"size matching achieved {tot} of {target}"
    return real[mask], target


def fit_scores(D, feat, T, model, var=""):
    real_rows = None
    extra = {}
    if var:
        assert var.startswith("ctl")
        real_rows, target = ctl_real_rows(D, T, int(var[3:]))
        extra = dict(target=target, n_real=len(real_rows), real_rows=real_rows)
    rows = train_rows(D, T, model, real_rows)
    audit_train_rows(D, T, model, rows)
    audit_taug(D.lab[rows], TAUG)
    cols = FEATS[feat]
    X = np.concatenate([D.X[c][rows][:, cols] for c in TAUG])
    y = np.concatenate([D.lab[rows]] * len(TAUG))
    g = np.concatenate([D.groups[rows]] * len(TAUG))
    m = probe.fit_probe(X, y, SEED, groups=g, class_weight="balanced")
    sc = np.stack([m.decision_function(D.X[c][:, cols]) for c in D.conds]).astype(np.float32)
    return sc, float(m[-1].C_[0]), int(len(rows)), extra


def cache_path(feat, T, var=""):
    return CACHE / f"scores_{feat}__{'+'.join(T)}{('__' + var) if var else ''}.npz"


def fit_combos():
    """-> list of (feat, T, var): everything Amendment 19 needs (+ the F_last sets of the drop-one-source labels)."""
    out = []
    for k in range(1, len(ORDER)):
        out += [("F_b8", T, "") for T in itertools.combinations(ORDER, k)]
    kmax = len(ORDER) - 1
    for S in ORDER:
        T = canon([s for s in ORDER if s != S])
        out.append(("F_last", T, ""))
        out += [("F_b8", T, f"ctl{i}") for i in range(N_CTL)]
    if CFG["drop_source_flast"]:
        for d in NEWS:
            for S in ORDER:
                if S != d:
                    out.append(("F_last", canon([s for s in ORDER if s not in (S, d)]), ""))
    seen, res = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def _task(args):
    feat, T, model, var = args
    t0 = time.time()
    return args, fit_scores(_D["D"], feat, T, model, var), time.time() - t0


def stage_fit(workers):
    todo = [c for c in fit_combos() if not cache_path(*[c[0], c[1], c[2]]).exists()]
    if not todo:
        log("fit: all cached")
        return
    _D["D"] = Data()
    D = _D["D"]
    log(f"loaded: roles {pd.Series(D.role).value_counts().to_dict()}; train reals per source {real_train_counts(D)}")
    names = model_names()
    tasks = sorted([(f, T, m, v) for f, T, v in todo for m in names], key=lambda t: -len(t[1]))
    log(f"fit: {len(todo)} combos, {len(tasks)} probes, {workers} workers")
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for fu in as_completed(futs):
            args, out, dt = fu.result()
            f, T, m, v = args
            res[args] = out
            log(f"  {f} train={tkey(T, v)} {m}: n_train={out[2]} C={out[1]:.3g} ({dt:.0f}s)")
            full = [res.get((f, T, mm, v)) for mm in names]
            if all(x is not None for x in full) and not cache_path(f, T, v).exists():
                kw = {}
                if v:
                    assert all(np.array_equal(full[0][3]["real_rows"], x[3]["real_rows"]) for x in full)
                    kw = dict(ctl_real_rows=full[0][3]["real_rows"], ctl_target=full[0][3]["target"])
                tmp = cache_path(f, T, v).with_name("tmp_" + cache_path(f, T, v).name)
                np.savez(tmp, scores=np.stack([x[0] for x in full]), C=np.array([x[1] for x in full]),
                         n_train=np.array([x[2] for x in full]), **kw)
                os.replace(tmp, cache_path(f, T, v))
    log(f"fit done in {time.time() - t0:.0f}s")


def file_loader(feat, T, var=""):
    p = cache_path(feat, T, var)
    if not p.exists():
        return None
    z = np.load(p)
    sc = {m: z["scores"][i] for i, m in enumerate(model_names())}
    extra = dict(ctl_target=int(z["ctl_target"]), n_real=int(len(z["ctl_real_rows"]))) if "ctl_real_rows" in z.files else {}
    return sc, z["C"], z["n_train"], extra


# ----------------------------------------------------------------------------- bootstrap plan
def cluster_key(D, i):
    """Bootstrap cluster of a test-role row: 'fam:' (caption family: RAISE reals + every fake), 'dev:' (VISION), 'img:' (OI),
    'g<pool>:<group>' (new sources: photographer / uploader / burst prefix)."""
    if D.lab[i] == 1 or D.pool[i] == "raise":
        assert D.rid[i], (D.df.src_path.iat[i], "test row without family id")
        return "fam:" + D.rid[i]
    if D.pool[i].startswith("vision"):
        return "dev:" + D.df.device.iat[i]
    if D.pool[i] in GROUP_POOLS:
        assert D.df.group.iat[i]
        return f"g{D.pool[i]}:" + D.df.group.iat[i]
    return "img:" + D.df.sha256.iat[i]


class Plan:
    """One global cluster-bootstrap plan over all test-role rows (W[0] = ones = point estimate). Within each kind of cluster
    (fam / dev / img, then one kind per new source) K clusters are drawn with replacement, K = number of distinct clusters of that kind;
    fakes and RAISE reals with the same family id move together. The kinds fam, dev, img are drawn first, in E7's order and with E7's
    seed, so the weights on the E7 rows equal E7's (paired). The same weights serve every source, method and condition."""

    def __init__(self, D, n_boot=B, seed=SEED):
        self.rows = D.idx("test")
        self.pos = {int(r): j for j, r in enumerate(self.rows)}
        keys = [cluster_key(D, i) for i in self.rows]
        kinds = np.array([k.split(":")[0] for k in keys])
        rng = np.random.default_rng(seed)
        W = np.ones((n_boot + 1, len(self.rows)))
        for kind in ["fam", "dev", "img"] + [f"g{p}" for p in GROUP_POOLS]:
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


# ----------------------------------------------------------------------------- evaluation
def macro_of(ga, gens):
    return np.mean([ga[g] for g in gens], 0)


def src_eval_rows(D, S):
    return D.idx("test", SRC[S]["eval_pools"], label=0)


def calib_rows(D, srcs):
    return D.idx("calib", [p for s in srcs for p in SRC[s]["eval_pools"]], label=0)


def evaluate(D, loader, n_boot=B, sources=None, conds=None, jr=None, log_=True):
    """All (held-out S, training subset T, variant) folds -> arrays GA / FP and JSON-able entries. `loader(feat, T, var)` returns
    (scores {model: [conds, N]}, C, n_train, extra) or None. Leak asserts run in every fold."""
    sources = sources or ORDER
    conds = conds or D.conds
    jr = jr or JR
    kmax = len(sources) - 1
    plan = Plan(D, n_boot)
    Rr = {S: src_eval_rows(D, S) for S in sources}
    Ff = {g: D.idx("test", None, gens=[g], label=1) for g in ALLGENS}
    fh = fake_test_hash(D)
    WR = {S: plan.w(r) for S, r in Rr.items()}
    WF = {g: plan.w(f) for g, f in Ff.items()}
    for g in ALLGENS:
        assert len(Ff[g]) > 0 and (WF[g].sum(1) > 0).all(), g
    for S in sources:
        assert len(Rr[S]) > 0 and (WR[S].sum(1) > 0).all(), f"no test reals / empty bootstrap weights for {S}"
    names = model_names()
    GA, FP, audits = {}, {}, []
    out = dict(n_test_reals={S: len(r) for S, r in Rr.items()}, n_test_fakes={g: len(f) for g, f in Ff.items()}, heldout={}, fake_test_hash=fh)
    long_rows = []
    for S in sources:
        others = [s for s in sources if s != S]
        out["heldout"][S] = {}
        folds = [(T, "") for T in subsets_of(others)]
        folds += [(canon(others, sources), f"ctl{i}") for i in range(N_CTL)]
        for T, var in folds:
            if var and not loader("F_b8", T, var):
                continue
            audit_fake_rows(D, fh)
            audit_eval(D, S, Rr[S])
            cal = calib_rows(D, T)
            audit_calib(D, S, T, cal)
            for m in names:                                  # the rows every fit of this fold can have used (recomputed with the same function)
                audit_train_rows(D, T, m, train_rows(D, T, m), S)
            need_all = (len(T) == kmax and not var) or bool(var) or (S in E7S and T == canon([s for s in E7S if s != S], sources))
            for feat in FEAT_ORDER:
                if feat == "F_last" and (var or len(T) < kmax - 1):
                    continue
                got = loader(feat, T, var)
                if got is None:
                    assert feat == "F_last", f"missing fit {feat} {T} {var}"
                    continue
                SC, Cs, ntr, extra = got
                ef = dict(C=[float(c) for c in Cs], n_train=[int(n) for n in ntr], n_calib=len(cal), extra=extra, conds={})
                for c in (conds if need_all else [jr]):
                    ci = conds.index(c)
                    ga = {}
                    for g in ALLGENS:
                        s = SC[model_of(g)][ci].astype(np.float64)
                        ga[g] = metrics.weighted_auroc(s[Rr[S]], s[Ff[g]], WR[S], WF[g])
                    GA[(S, T, var, feat, c)] = ga
                    ec = dict(macro=summ(macro_of(ga, ALLGENS)), flux=summ(ga[FLUX]), per_gen={g: float(ga[g][0]) for g in ALLGENS}, fpr={})
                    for reading in ("A", "B"):
                        ec["fpr"][reading] = {}
                        for m in names:
                            sm = SC[m]
                            thr = thr_fpr(sm[conds.index("clean")][cal]) if reading == "A" else thr_fpr(sm[ci][cal])
                            hit = (sm[ci][Rr[S]] >= thr).astype(float)
                            k, n = int(hit.sum()), len(hit)
                            bf = (WR[S] @ hit) / WR[S].sum(1)
                            FP[(S, T, var, feat, c, reading, m)] = (bf, k, n)
                            ec["fpr"][reading][m] = dict(thr=float(thr), fpr=k / n, k=k, n=n, cp95=clopper_pearson(k, n),
                                                         boot_ci=[float(v) for v in np.percentile(bf[1:], [2.5, 97.5])])
                    thrA = ec["fpr"]["A"]["full"]["thr"]
                    ec["tpr_full_A"] = float(np.mean([(SC["full"][ci][Ff[g]] >= thrA).mean() for g in ALLGENS]))
                    if S == "VISION" and len(T) == kmax and not var:
                        ec["fpr_pools"] = {}
                        for pk, pools in (("native", ["vision_native"]), ("WA", ["vision_whatsapp"]), ("FBH", ["vision_facebook_hq"])):
                            r = D.idx("test", pools, label=0)
                            k = int((SC["full"][ci][r] >= thrA).sum())
                            ec["fpr_pools"][pk] = dict(fpr=k / len(r), k=k, n=len(r), cp95=clopper_pearson(k, len(r)))
                    ef["conds"][c] = ec
                out["heldout"][S].setdefault(tkey(T, var), {})[feat] = ef
                if len(T) == kmax and not var and feat == "F_b8":
                    keep = np.where(np.isin(D.role, ["calib", "test"]))[0]
                    for c in (jr, "clean"):
                        ci = conds.index(c)
                        sc = np.array([SC[model_of(D.gen[i])][ci][i] if D.lab[i] == 1 else SC["full"][ci][i] for i in keep])
                        long_rows.append(pd.DataFrame(dict(heldout=S, train_sources=tkey(T), feat=feat, cond=c, sha256=D.df.sha256.to_numpy()[keep],
                                                           label=D.lab[keep], generator=D.gen[keep], pool=D.pool[keep], role=D.role[keep],
                                                           score=np.round(sc, 4).astype(np.float32))))
            audits.append((S, tkey(T, var)))
    out["audit"] = dict(n_folds_audited=len(audits), checks="train rows role=train and reals only from training sources' train pools; calib reals of the "
                        "training sources only; evaluation reals = held-out TEST reals only; fake test rows hash-identical in every fold; "
                        "T-aug identical for both classes; every row a fov512 crop of its own sha256", fake_test_hash=fh)
    return out, GA, FP, long_rows


def key_arrays(GA, FP, keep, fpr_model, reading, feat, cond):
    def mac(S, T, var="", f=feat, c=cond):
        return np.mean([GA[(S, T, var, f, c)][g] for g in keep], 0)

    def fpa(S, T, var="", f=feat, c=cond):
        return FP[(S, T, var, f, c, reading, fpr_model)]
    return mac, fpa


def derived(GA, FP, pool, keep, fpr_model="full", reading="A", jr=None, use_ctl=True):
    """Amendment 19 quantities at the laundering condition for a source pool (all 6, or 5 for drop-one-source) and a generator subset."""
    jr = jr or JR
    pool = list(pool)
    mac, fpa = key_arrays(GA, FP, keep, fpr_model, reading, "F_b8", jr)
    kmax = len(pool) - 1
    Tf = {S: canon([s for s in pool if s != S], pool) for S in pool}
    T2 = {S: [c for c in subsets_of([s for s in pool if s != S]) if len(c) == 2] for S in pool}
    per = {}
    for S in pool:
        auc = mac(S, Tf[S])
        arr, k, n = fpa(S, Tf[S])
        per[S] = dict(auroc=summ(auc), fpr=float(arr[0]), k=k, n=n, fpr_cp95=clopper_pearson(k, n), fpr_boot_ci=[float(v) for v in np.percentile(arr[1:], [2.5, 97.5])])
    q = dict(per=per, kmax=kmax)
    if all((S, Tf[S], "", "F_last", jr) in GA for S in pool):
        d_layer = np.mean([mac(S, Tf[S]) - mac(S, Tf[S], f="F_last") for S in pool], 0)
        q["b8_minus_last"] = diff_summary(d_layer)
    else:
        q["b8_minus_last"] = None
    lc, lcf = {}, {}
    for k in range(1, kmax + 1):
        lc[k] = np.mean([np.mean([mac(S, T) for T in subsets_of([s for s in pool if s != S]) if len(T) == k], 0) for S in pool], 0)
        lcf[k] = np.mean([np.mean([fpa(S, T)[0] for T in subsets_of([s for s in pool if s != S]) if len(T) == k], 0) for S in pool], 0)
    q["lc_mean"] = {k: summ(v) for k, v in lc.items()}
    q["lc_fpr_mean"] = {k: summ(v) for k, v in lcf.items()}
    q["lc_diff_kmax_vs_k"] = {k: diff_summary(lc[kmax], lc[k]) for k in range(1, kmax)}
    q["mean_auroc"] = summ(np.mean([mac(S, Tf[S]) for S in pool], 0))
    base_auc = {S: np.mean([mac(S, T) for T in T2[S]], 0) for S in pool}
    base_fpr = {S: np.mean([fpa(S, T)[0] for T in T2[S]], 0) for S in pool}
    q["div_auc"] = diff_summary(np.mean([mac(S, Tf[S]) - base_auc[S] for S in pool], 0))
    q["div_fpr"] = diff_summary(np.mean([fpa(S, Tf[S])[0] - base_fpr[S] for S in pool], 0))
    q["per_div"] = {S: dict(auroc_k2=summ(base_auc[S]), fpr_k2=summ(base_fpr[S]), d_auroc=diff_summary(mac(S, Tf[S]), base_auc[S]),
                            d_fpr=diff_summary(fpa(S, Tf[S])[0], base_fpr[S])) for S in pool}
    q["ctl_auc"] = q["ctl_fpr"] = None
    if use_ctl and all((S, Tf[S], f"ctl{i}", "F_b8", jr) in GA for S in pool for i in range(N_CTL)):
        cm = {S: np.mean([mac(S, Tf[S], f"ctl{i}") for i in range(N_CTL)], 0) for S in pool}
        cf = {S: np.mean([fpa(S, Tf[S], f"ctl{i}")[0] for i in range(N_CTL)], 0) for S in pool}
        q["ctl_auc"] = diff_summary(np.mean([cm[S] - base_auc[S] for S in pool], 0))
        q["ctl_fpr"] = diff_summary(np.mean([cf[S] - base_fpr[S] for S in pool], 0))
        q["per_ctl"] = {S: dict(auroc=summ(cm[S]), fpr=summ(cf[S]), d_vs_full_auroc=diff_summary(cm[S], mac(S, Tf[S]))) for S in pool}
    return q


def decide(q):
    """Amendment 19 labels from the derived quantities. All reported; none replaces another."""
    per = q["per"].values()
    gain = TH["diversity_auroc_gain"]
    kill_src = any(v["auroc"][0] < TH["kill_auroc"] or v["fpr"] > TH["kill_fpr"] for v in per)
    ok_src = all(v["auroc"][0] >= TH["pass_auroc"] and v["fpr"] <= TH["pass_fpr"] for v in per)
    helps_auc = q["div_auc"]["delta"] >= gain and q["div_auc"]["ci_lo"] > 0
    helps_fpr = q["div_fpr"]["ci_hi"] < 0
    helps = bool(helps_auc and helps_fpr)
    ctl_ok = None if q["ctl_auc"] is None else bool(q["ctl_auc"]["delta"] >= gain and q["ctl_auc"]["ci_lo"] > 0)
    disc = bool(all(v["auroc"][0] >= TH["discovery_auroc"] and v["fpr"] <= TH["discovery_fpr"] for v in per) and helps)
    lay = q["b8_minus_last"]
    ok_layer = None if lay is None else bool(lay["delta"] >= TH["b8_minus_last"] and lay["ci_lo"] > 0)
    lcs = q["lc_diff_kmax_vs_k"]
    ok_lc = bool(all(v["delta"] > 0 and v["ci_lo"] > 0 for v in lcs.values()))
    ok_lc_vs1 = bool(lcs[1]["delta"] > 0 and lcs[1]["ci_lo"] > 0)
    if kill_src:
        e7lab = "KILL"
    elif ok_layer is None:
        e7lab = "PASS?-needs-F_last" if (ok_src and ok_lc) else "INCONCLUSIVE"
    else:
        e7lab = "PASS" if (ok_src and ok_layer and ok_lc) else "INCONCLUSIVE"
    return dict(DIVERSITY_HELPS=helps, c_auroc=bool(helps_auc), c_fpr=bool(helps_fpr),
                DIVERSITY_NOT_QUANTITY=None if ctl_ok is None else bool(helps and ctl_ok), ctl_auroc_part=ctl_ok,
                DISCOVERY=disc, E7_RULE=e7lab, e7_kill=bool(kill_src), e7_sources_ok=bool(ok_src), e7_layer_ok=ok_layer,
                e7_learning_curve_ok_all_k=ok_lc, e7_learning_curve_ok_vs_k1_only=ok_lc_vs1,
                KILL_OF_LINE=bool(q["div_auc"]["ci_lo"] <= 0 or kill_src))


def robustness(GA, FP, sources):
    res = {"gens": {}, "sources": {}}
    subsets = {"all": (ALLGENS, "full"), f"no {FLUX}": ([g for g in ALLGENS if g != FLUX], "full")}
    for g in ALLGENS:
        subsets[f"drop {g}"] = ([x for x in ALLGENS if x != g], model_of(g))
    for reading in ("A", "B"):
        for nm, (keep, fm) in subsets.items():
            q = derived(GA, FP, sources, keep, fm, reading)
            res["gens"][f"{reading}|{nm}"] = dict(labels=decide(q), q=q)
        for d in NEWS:
            pool = [s for s in sources if s != d]
            q = derived(GA, FP, pool, ALLGENS, "full", reading, use_ctl=False)
            res["sources"][f"{reading}|drop {d}"] = dict(labels=decide(q), q=q)
    return res


def key_secondary(GA, FP, out, reading="A"):
    """VISION / RAISE / OI held out: all 5 other sources vs E7's 2 (the other two E7 sources), paired on identical test rows and weights."""
    res = {}
    pub = None
    pj = ROOT / "experiments" / "016-loso" / "results" / "results.json"
    if pj.exists():
        pub = json.loads(pj.read_text())["arms"]["fov512"]["heldout"]
    for S in E7S:
        T5 = canon([s for s in ORDER if s != S])
        T2 = canon([s for s in E7S if s != S])
        if (S, T5, "", "F_b8", JR) not in GA or (S, T2, "", "F_b8", JR) not in GA:
            continue
        r = {}
        for c in CONDS:
            if (S, T5, "", "F_b8", c) in GA and (S, T2, "", "F_b8", c) in GA:
                a5, a2 = macro_of(GA[(S, T5, "", "F_b8", c)], ALLGENS), macro_of(GA[(S, T2, "", "F_b8", c)], ALLGENS)
                f5, f2 = FP[(S, T5, "", "F_b8", c, reading, "full")], FP[(S, T2, "", "F_b8", c, reading, "full")]
                r[c] = dict(auroc_k5=summ(a5), auroc_e7=summ(a2), d_auroc=diff_summary(a5, a2), fpr_k5=f5[0][0], fpr_e7=f2[0][0],
                            k5=[f5[1], f5[2]], e7=[f2[1], f2[2]], d_fpr=diff_summary(f5[0], f2[0]))
        r["e7_published_jr"] = None
        if pub:
            e = pub[S]["+".join(T2)]["F_b8"]["conds"][JR]
            r["e7_published_jr"] = dict(auroc=e["macro"], fpr_A=e["fpr"]["A"]["full"]["fpr"], fpr_B=e["fpr"]["B"]["full"]["fpr"])
        res[S] = r
    return res


def e7_parity():
    """Max |score difference| of the refit on the E7 source sets vs E7's cached scores on E7's rows (same data, seed, order)."""
    d16 = ROOT / "experiments" / "016-loso" / "cache"
    res = {}
    for feat in FEAT_ORDER:
        for k in (1, 2):
            for T in itertools.combinations(E7S, k):
                a = d16 / f"scores_fov512__{feat}__{'+'.join(T)}.npz"
                mine = cache_path(feat, T)
                if a.exists() and mine.exists():
                    za, zb = np.load(a)["scores"], np.load(mine)["scores"]
                    res[f"{feat} {'+'.join(T)}"] = float(np.abs(za - zb[:, :, :za.shape[2]]).max())
    return res


def run_eval():
    RES.mkdir(exist_ok=True)
    D = Data(load_feats=False)
    log(f"eval: roles {pd.Series(D.role).value_counts().to_dict()}; train reals {real_train_counts(D)}")
    out, GA, FP, long_rows = evaluate(D, file_loader)
    finalize(D, out, GA, FP, long_rows)


def finalize(D, out, GA, FP, long_rows):
    """Robustness, key secondary, parity, files (results.json, tables.md, scores_long.csv.gz)."""
    out["robust"] = robustness(GA, FP, ORDER)
    out["main"] = {rd: out["robust"]["gens"][f"{rd}|all"] for rd in ("A", "B")}
    out["key_secondary"] = {rd: key_secondary(GA, FP, out, rd) for rd in ("A", "B")}
    out["parity_e7"] = e7_parity()
    out["owner"] = owner_status()
    cj = RES / "split_counts.json"
    out["split_counts"] = json.loads(cj.read_text())["counts"] if cj.exists() else None
    counts = real_train_counts(D)
    out["ctl_targets"] = {S: ctl_target(counts, ORDER, S) for S in ORDER}
    out["real_train_counts"] = counts
    out["real_split_counts"] = {S: {r: int(len(D.idx(r, SRC[S]["eval_pools"] if r != "train" else SRC[S]["train_pools"], label=0)))
                                    for r in ("train", "calib", "test")} for S in ORDER}
    out["n_fits"] = dict(combos=len(fit_combos()), probes=len(fit_combos()) * len(model_names()))
    pd.concat(long_rows).to_csv(RES / "scores_long.csv.gz", index=False, compression="gzip")
    meta = dict(git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), config=CFG)
    (RES / "results.json").write_text(json.dumps(dict(meta=meta, result=out), indent=1, default=float))
    write_tables(out)
    A = out["main"]["A"]["labels"]
    log(f"eval done; reading A labels: {A}")


# ----------------------------------------------------------------------------- tables
def f3(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def gi(d):
    return f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def pct(x):
    return f"{100 * x:.1f}%"


def write_tables(R):
    H = R["heldout"]
    L = ["# E10 (019-source-diversity) tables", "",
         "Generated by run.py from results.json. Exploratory (D-016, D-017). AUROC = macro over the 8 generators (LOGO model per train generator, 'full' model "
         "for FLUX) of the held-out source's TEST reals vs the held-out generators' TEST fakes. CIs: 1,000 cluster-bootstrap replicates (clusters: caption family "
         "/ device / image / group). fov512, T-aug, F_b8 unless stated. FPR reading A = 1% threshold on the training sources' CLEAN calib reals, applied at "
         f"{JR}; reading B = same-condition calib. OWNER photos: {R['owner']['status']} ({R['owner']['n_images']} images; {R['owner']['reason']}).", ""]
    if R.get("split_counts"):
        L += ["## New-source splits (images; groups)", "", "| source | manifest | dropped (<512) | train / calib / test images | groups train / calib / test | max group size |", "|---|---|---|---|---|---|"]
        for S, c in R["split_counts"].items():
            i, g = c["images"], c["n_groups"]
            L.append(f"| {S} | {c['n_manifest']} | {c['n_short_lt_512_dropped']} | {i['train']} / {i['calib']} / {i['test']} | "
                     f"{g.get('train', 0)} / {g.get('calib', 0)} / {g.get('test', 0)} | {c['max_group_size']} |")
        L.append("")
    L += ["## Real images per source and role (train-role / calib / test reals; VISION train = natives only)", "", "| source | train | calib | test | size-matched control target |", "|---|---|---|---|---|"]
    for S in ORDER:
        c = R["real_split_counts"][S]
        L.append(f"| {S} | {c['train']} | {c['calib']} | {c['test']} | {R['ctl_targets'][S]} |")
    L += ["", f"Test fakes per generator (identical in every fold, hash {R['fake_test_hash'][:12]}): {R['n_test_fakes']}", ""]
    # labels
    L += ["## Labels (k = 5, fov512, jpeg75_resize05, F_b8; all generators)", "",
          "| reading | DIVERSITY HELPS | (AUROC part / FPR part) | DIVERSITY NOT QUANTITY | DISCOVERY | E7-rule label | KILL of the line |", "|---|---|---|---|---|---|---|"]
    for rd in ("A", "B"):
        lb = R["main"][rd]["labels"]
        L.append(f"| {rd} | **{lb['DIVERSITY_HELPS']}** | {lb['c_auroc']} / {lb['c_fpr']} | {lb['DIVERSITY_NOT_QUANTITY']} (control AUROC part {lb['ctl_auroc_part']}) | "
                 f"**{lb['DISCOVERY']}** | **{lb['E7_RULE']}** (kill {lb['e7_kill']}, sources ok {lb['e7_sources_ok']}, b8-last ok {lb['e7_layer_ok']}, "
                 f"lc ok vs every k<5 {lb['e7_learning_curve_ok_all_k']} / vs k=1 only {lb['e7_learning_curve_ok_vs_k1_only']}) | **{lb['KILL_OF_LINE']}** |")
    q = R["main"]["A"]["q"]
    L += ["", "Quantities (reading A):", "",
          f"- mean over held-out sources of AUROC(k=5) - mean_k=2: {gi(q['div_auc'])}",
          f"- mean over held-out sources of FPR(k=5) - mean_k=2: {gi(q['div_fpr'])}",
          f"- size-matched k=5 (mean of {N_CTL} seeds) minus mean_k=2, AUROC: {gi(q['ctl_auc']) if q['ctl_auc'] else 'n/a'}; FPR: {gi(q['ctl_fpr']) if q['ctl_fpr'] else 'n/a'}",
          f"- mean(F_b8 - F_last) at k=5: {gi(q['b8_minus_last']) if q['b8_minus_last'] else 'n/a'}",
          f"- mean held-out AUROC at k=5: {f3(q['mean_auroc'])}"]
    for k, v in q["lc_diff_kmax_vs_k"].items():
        L.append(f"- AUROC(k=5) - AUROC(k={k}): {gi(v)}")
    # k=5
    full = lambda S: tkey(canon([s for s in ORDER if s != S]))
    L += ["", f"## Held-out source at k = 5 (train on the 5 other sources), {JR}", "",
          "| held-out | feat | AUROC [CI] | FPR reading A (k/n) [CP95] {cluster-boot CI} | FPR reading B | TPR@A thr |", "|---|---|---|---|---|---|"]
    fm = lambda x: f"{x['fpr']:.3f} ({x['k']}/{x['n']}) [{x['cp95'][0]:.3f}, {x['cp95'][1]:.3f}] {{{x['boot_ci'][0]:.3f}, {x['boot_ci'][1]:.3f}}}"
    for S in ORDER:
        for feat in FEAT_ORDER:
            if feat in H[S][full(S)]:
                ec = H[S][full(S)][feat]["conds"][JR]
                L.append(f"| {S} | {feat} | {f3(ec['macro'])} | {fm(ec['fpr']['A']['full'])} | {fm(ec['fpr']['B']['full'])} | {ec['tpr_full_A']:.3f} |")
    L += ["", "## Held-out source at k = 5: macro AUROC / FLUX AUROC / FPR (reading A) at every condition (F_b8)", "",
          "| held-out | " + " | ".join(f"{c} AUROC | {c} FLUX | {c} FPR" for c in CONDS) + " |", "|---|" + "---|" * (3 * len(CONDS))]
    for S in ORDER:
        cells = []
        for c in CONDS:
            ec = H[S][full(S)]["F_b8"]["conds"][c]
            cells += [f3(ec["macro"]), f3(ec["flux"]), f"{ec['fpr']['A']['full']['fpr']:.3f}"]
        L.append(f"| {S} | " + " | ".join(cells) + " |")
    if "fpr_pools" in H["VISION"][full("VISION")]["F_b8"]["conds"][JR]:
        L += ["", f"VISION held out, FPR per pool at {JR} (reading A): " + "; ".join(
            f"{p} {v['fpr']:.3f} ({v['k']}/{v['n']})" for p, v in H["VISION"][full("VISION")]["F_b8"]["conds"][JR]["fpr_pools"].items())]
    # learning curve
    L += ["", f"## Learning curve (mean over the k-source training subsets of each held-out source; {JR}, F_b8)", "",
          "| held-out | " + " | ".join(f"k={k} AUROC" for k in range(1, 6)) + " | " + " | ".join(f"k={k} FPR(A)" for k in range(1, 6)) + " |", "|---|" + "---|" * 10]
    for S in ORDER:
        others = [s for s in ORDER if s != S]
        au, fp = [], []
        for k in range(1, 6):
            Ts = [T for T in subsets_of(others) if len(T) == k]
            ecs = [H[S][tkey(T)]["F_b8"]["conds"][JR] for T in Ts]
            au.append(f"{np.mean([e['macro'][0] for e in ecs]):.3f}")
            fp.append(f"{np.mean([e['fpr']['A']['full']['fpr'] for e in ecs]):.3f}")
        L.append(f"| {S} | " + " | ".join(au) + " | " + " | ".join(fp) + " |")
    L += ["", "Mean over held-out sources (AUROC [CI]; FPR A [CI]):", "", "| k | AUROC | FPR (A) |", "|---|---|---|"]
    for k in range(1, 6):
        L.append(f"| {k} | {f3(q['lc_mean'][k])} | {f3(q['lc_fpr_mean'][k])} |")
    L += ["", "Per held-out source, k=5 vs the k=2 mean (paired, same test rows):", "", "| held-out | AUROC k=2 mean | AUROC k=5 minus k=2 | FPR(A) k=2 mean | FPR k=5 minus k=2 |", "|---|---|---|---|---|"]
    for S, v in q["per_div"].items():
        L.append(f"| {S} | {f3(v['auroc_k2'])} | {gi(v['d_auroc'])} | {f3(v['fpr_k2'])} | {gi(v['d_fpr'])} |")
    # control
    if q.get("ctl_auc") is not None:
        L += ["", f"## Size-matched control (k = 5, training reals subsampled by group to the k=2 mean count; mean of {N_CTL} seeds; {JR})", "",
              "| held-out | target reals | achieved (seeds) | AUROC size-matched | AUROC full k=5 | FPR(A) size-matched | FPR(A) full k=5 |", "|---|---|---|---|---|---|---|"]
        for S in ORDER:
            ex = [H[S][tkey(canon([s for s in ORDER if s != S]), f"ctl{i}")]["F_b8"]["extra"] for i in range(N_CTL)]
            L.append(f"| {S} | {R['ctl_targets'][S]} | {[e['n_real'] for e in ex]} | {f3(q['per_ctl'][S]['auroc'])} | {f3(q['per'][S]['auroc'])} | "
                     f"{f3(q['per_ctl'][S]['fpr'])} | {q['per'][S]['fpr']:.3f} |")
    # key secondary
    L += ["", "## Key secondary: VISION / RAISE / OI held out, 5 training sources vs E7's 2 (paired, identical test rows; F_b8)", ""]
    for rd in ("A", "B"):
        L += [f"Reading {rd}:", "", "| held-out | condition | AUROC k=5 | AUROC E7 (2 sources, refit) | diff | FPR k=5 | FPR E7 | FPR diff |", "|---|---|---|---|---|---|---|---|"]
        for S, r in R["key_secondary"][rd].items():
            for c in CONDS:
                if c in r:
                    v = r[c]
                    L.append(f"| {S} | {c} | {f3(v['auroc_k5'])} | {f3(v['auroc_e7'])} | {gi(v['d_auroc'])} | {v['fpr_k5']:.3f} ({v['k5'][0]}/{v['k5'][1]}) | "
                             f"{v['fpr_e7']:.3f} ({v['e7'][0]}/{v['e7'][1]}) | {gi(v['d_fpr'])} |")
        L.append("")
    L += ["E7 as published (016 results.json, jpeg75_resize05): " + "; ".join(
        f"{S}: AUROC {f3(r['e7_published_jr']['auroc'])}, FPR A {r['e7_published_jr']['fpr_A']:.3f}" for S, r in R["key_secondary"]["A"].items() if r.get("e7_published_jr")) +
          f". Parity of the refits with E7's cached scores (max |score diff|): {R['parity_e7']}", ""]
    # robustness
    L += ["## Robustness: drop-one-generator (FPR from the LOGO model that never saw the dropped generator)", "",
          "| subset | A: helps / discovery / E7 / kill-line | B: helps / discovery / E7 / kill-line | min AUROC (k=5) | max FPR(A) (k=5) | AUROC gain k=5-k=2 | FPR gain | ctl AUROC part |", "|---|---|---|---|---|---|---|---|"]
    for key, v in R["robust"]["gens"].items():
        if not key.startswith("A|"):
            continue
        nm = key[2:]
        a, b = v["labels"], R["robust"]["gens"]["B|" + nm]["labels"]
        qq = v["q"]
        L.append(f"| {nm} | {a['DIVERSITY_HELPS']} / {a['DISCOVERY']} / {a['E7_RULE']} / {a['KILL_OF_LINE']} | {b['DIVERSITY_HELPS']} / {b['DISCOVERY']} / {b['E7_RULE']} / {b['KILL_OF_LINE']} | "
                 f"{min(x['auroc'][0] for x in qq['per'].values()):.3f} | {max(x['fpr'] for x in qq['per'].values()):.3f} | {gi(qq['div_auc'])} | {gi(qq['div_fpr'])} | {a['ctl_auroc_part']} |")
    L += ["", "## Robustness: drop-one-source (each NEW source removed from the pool entirely; k_max = 4, k=2 baseline from the remaining 4; no size-matched control)", "",
          "| pool | reading | helps / discovery / E7 / kill-line | min AUROC (k_max) | max FPR | AUROC gain | FPR gain | b8-last |", "|---|---|---|---|---|---|---|---|"]
    for key, v in R["robust"]["sources"].items():
        rd, nm = key.split("|")
        a, qq = v["labels"], v["q"]
        L.append(f"| {nm} | {rd} | {a['DIVERSITY_HELPS']} / {a['DISCOVERY']} / {a['E7_RULE']} / {a['KILL_OF_LINE']} | {min(x['auroc'][0] for x in qq['per'].values()):.3f} | "
                 f"{max(x['fpr'] for x in qq['per'].values()):.3f} | {gi(qq['div_auc'])} | {gi(qq['div_fpr'])} | {gi(qq['b8_minus_last']) if qq['b8_minus_last'] else 'n/a'} |")
    # per-subset detail
    L += ["", f"## Every (held-out, training subset): macro AUROC [CI] and FPR reading A at {JR} (F_b8)", "", "| held-out | train sources | k | n_train (full model) | AUROC | FPR (A) |", "|---|---|---|---|---|---|"]
    for S in ORDER:
        for tk, e in H[S].items():
            ec = e["F_b8"]["conds"][JR]
            T = tk.split("@")[0].split("+")
            L.append(f"| {S} | {tk} | {len(T)} | {e['F_b8']['n_train'][-1]} | {f3(ec['macro'])} | {ec['fpr']['A']['full']['fpr']:.3f} ({ec['fpr']['A']['full']['k']}/{ec['fpr']['A']['full']['n']}) |")
    L += ["", f"## Leak audit", "", f"{R['audit']['n_folds_audited']} folds audited: {R['audit']['checks']}.", ""]
    (RES / "tables.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["splits", "prep", "feats", "fit", "eval", "all"], nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="splits stage only: run the logic on the manifests that exist, write nothing")
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    if a.stage in ("splits", "all"):
        stage_splits(a.dry_run)
    if a.dry_run:
        return
    if a.stage in ("prep", "all"):
        stage_prep()
    if a.stage in ("feats", "all"):
        stage_feats()
    if a.stage in ("fit", "all"):
        stage_fit(a.workers)
    if a.stage in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
