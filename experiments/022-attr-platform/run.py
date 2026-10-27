"""H-A3 (D-018 cycle): is the platform history (VISION native / WhatsApp / Facebook-HQ) identifiable on content-identical triplets?

    env/.venv/bin/python -I experiments/022-attr-platform/run.py all
Stages: data -> featA (native-grid DCT features, cached) -> cv (A, B, M0) -> perm -> arms (camera model) -> report
Everything is frozen by config.json + docs/08 (H-A3, section 3) + docs/09. selftest.py imports this module.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import fft as sfft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import perturb
import p4features as P

CFG = json.loads((HERE / "config.json").read_text())
SEED, NBOOT = CFG["seed"], CFG["n_boot"]
RES, CACHE = HERE / "results", HERE / "cache"
CONDS = CFG["conditions"]
CLASSES = CFG["data"]["class_names"]
CHANCE = 1 / 3
CS = np.logspace(-4, 1, 6)
BLOCKS = {k: slice(*v) for k, v in CFG["method_B"]["blocks"].items()}
MA = CFG["method_A"]
HB = MA["hist"]
NB = int(round((HB["hi"] - HB["lo"]) / HB["bin"]))
S_MIN, S_MAX = MA["periodicity"]["s_min"], MA["periodicity"]["s_max"]
PHI_CAP = MA["periodicity"]["phi_block_cap"]


def log(*a):
    print(time.strftime("%H:%M:%S"), "H-A3", *a, flush=True)


# ============================================================================== method A: native-grid DCT statistics
def zigzag_order(n=8):
    """JPEG zig-zag: list of (row, col); index 0 = DC, 1 = (0,1), 2 = (1,0), 3 = (2,0), 4 = (1,1), 5 = (0,2) ..."""
    out = []
    for s in range(2 * n - 1):
        cells = [(r, s - r) for r in range(n) if 0 <= s - r < n]
        out += cells if s % 2 else cells[::-1]
    return out


ZZ = zigzag_order()
AC = ZZ[MA["zigzag_ac"][0]:MA["zigzag_ac"][1] + 1]
assert len(AC) == 15


def luma(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def block_dct(Y):
    """Orthonormal 8x8 DCT-II of (Y-128) on the native grid (origin top-left, cropped to multiples of 8) -> [nblocks, 8, 8]."""
    H, W = (Y.shape[0] // 8) * 8, (Y.shape[1] // 8) * 8
    if H == 0 or W == 0:
        raise ValueError(f"image {Y.shape} smaller than one 8x8 block")
    b = Y[:H, :W] - 128.0
    b = b.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return sfft.dctn(b, type=2, norm="ortho", axes=(1, 2), workers=1)


def periodicity(x):
    """x: [n] DCT coefficient values of one frequency. phi(s) = |mean exp(2 pi i x / s)| for s = 1..S_MAX; contrast
    C(s) = phi(s) - (phi(s-1)+phi(s+1))/2 (a quantisation comb of spacing s peaks at frequency 1/s, a smooth density does not).
    -> (s_hat, C(s_hat)); s_hat = largest s in [S_MIN, S_MAX] with C(s) >= 0.8 max C (harmonics 1/(s/m) tie with the fundamental)."""
    if len(x) > PHI_CAP:
        x = x[:: int(np.ceil(len(x) / PHI_CAP))]
    s = np.arange(1, S_MAX + 2, dtype=np.float64)
    ang = (2 * np.pi / s)[None, :] * x[:, None].astype(np.float64)
    phi = np.hypot(np.cos(ang).mean(0), np.sin(ang).mean(0))          # index j <-> s = j+1
    c = phi[1:-1] - 0.5 * (phi[:-2] + phi[2:])                         # index j <-> s = j+2
    c = c[: S_MAX - S_MIN + 1]
    cm = c.max()
    cand = np.where(c >= 0.8 * cm)[0] if cm > 0 else np.arange(len(c))
    j = int(cand.max())
    return float(S_MIN + j), float(c[j])


def feat_A_from_Y(Y):
    """990-d vector: 15 x 64 normalised coefficient histograms | 15 s_hat | 15 contrast peaks."""
    D = block_dct(Y)
    hist, sh, pk = [], [], []
    for (r, c) in AC:
        x = D[:, r, c]
        idx = np.clip(np.floor((x - HB["lo"]) / HB["bin"]).astype(np.int64), 0, NB - 1)
        h = np.bincount(idx, minlength=NB).astype(np.float64)
        hist.append(h / h.sum())
        s, p = periodicity(x)
        sh.append(s)
        pk.append(p)
    return np.concatenate(hist + [np.array(sh), np.array(pk)]).astype(np.float32)


def decode_cond(path, cond, key):
    """Decoded pixels of the raw JPEG (no EXIF transpose, no metadata read) -> the project laundering code, identical for all classes."""
    with Image.open(path) as im:
        im.load()
        return perturb.apply_condition(im, cond, key=key, seed=CFG["launder_seed"])


def _extract(job):
    path, sha, cond = job
    return sha, feat_A_from_Y(luma(decode_cond(ROOT / path, cond, sha)))


def extract_A(table, cond, workers, cache_dir=None, tag="A"):
    """-> [n, 990] aligned to table rows; per-condition npz cache keyed by raw-file sha256."""
    cache_dir = Path(cache_dir or CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{tag}_{cond}.npz"
    store = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        store = dict(zip(z["keys"].tolist(), z["feats"]))
    todo = [(r.src_path, r.sha256, cond) for r in table.itertuples() if r.sha256 not in store]
    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            for k, v in ex.map(_extract, todo, chunksize=4):
                store[k] = v
        ks = sorted(store)
        np.savez(f, keys=np.array(ks), feats=np.stack([store[k] for k in ks]))
        log(f"featA {cond}: {len(todo)} images in {time.time() - t0:.0f}s")
    X = np.stack([store[s] for s in table.sha256]).astype(np.float32)
    assert X.shape[1] == MA["n_features"] and np.isfinite(X).all()
    return X


# ============================================================================== data
def build_table():
    """485 content-identical triplets (3 copies each) from the 004 fov512 splits; D13 excluded. Columns per row: sha256 (raw file),
    src_path, platform (0/1/2), device, vidx, photo (device_vidx), model (folder name)."""
    d = pd.read_csv(ROOT / CFG["data"]["splits"], keep_default_na=False, dtype={"vidx": str})
    pools = {v: int(k) for k, v in CFG["data"]["pools"].items()}
    v = d[d.pool.isin(pools)].copy()
    v["platform"] = v.pool.map(pools)
    v = v[~v.device.isin(CFG["data"]["exclude_devices"])].copy()
    v["photo"] = v.device + "_" + v.vidx
    v["model"] = v.src_path.map(lambda p: p.split("/")[3].split("_", 1)[1])
    cnt = v.groupby("photo").platform.nunique()
    assert (cnt == 3).all() and len(cnt) == CFG["data"]["expected_triplets"], (len(cnt), cnt.value_counts().to_dict())
    assert (v.groupby("photo").size() == 3).all()
    v = v.sort_values(["device", "vidx", "platform"]).reset_index(drop=True)
    return v[["sha256", "src_path", "platform", "device", "vidx", "photo", "model"]]


def gate_dedupe(table):
    """Gate 5: no duplicate raw-file sha256 across classes (recomputed from the files)."""
    h = [hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in table.src_path]
    assert h == table.sha256.tolist(), "table sha256 != file contents"
    if len(set(h)) != len(h):
        dup = pd.Series(h)[pd.Series(h).duplicated(keep=False)]
        raise AssertionError(f"duplicate sha256 across rows: {len(dup)} rows")
    return len(h)


def make_folds(table, k=5):
    dev = table.device.to_numpy()
    folds = list(GroupKFold(n_splits=k).split(np.zeros(len(table)), table.platform.to_numpy(), dev))
    assert_no_group_overlap(folds, dev)
    return folds


def assert_no_group_overlap(folds, groups):
    seen = set()
    for tr, te in folds:
        g_tr, g_te = set(groups[tr]), set(groups[te])
        assert not (g_tr & g_te), f"group leak: {sorted(g_tr & g_te)[:5]} in train and test"
        assert not (g_te & seen), "a group is in two test folds"
        seen |= g_te


# ============================================================================== probe
def bacc(y, yhat, k=3):
    return float(np.mean([(yhat[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def fit_lr(X, y, g, C=None, seed=SEED):
    """Standardiser + multinomial L2 logistic (balanced). C=None: chosen by inner GroupKFold(5) balanced accuracy on the given rows only.
    -> (pipeline, inner_cv_bacc, C)."""
    def mk(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
    if C is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mk(C).fit(X, y), float("nan"), float(C)
    inner = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g))
    sc = np.zeros(len(CS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c in enumerate(CS):
            sc[i] = np.mean([bacc(y[te], mk(c).fit(X[tr], y[tr]).predict(X[te])) for tr, te in inner])
        j = int(np.argmax(sc))                    # ties -> smallest C (first)
        return mk(CS[j]).fit(X, y), float(sc[j]), float(CS[j])


def fit_select(Xb, y, g, fixed=None):
    """Xb: dict block -> train matrix. Several blocks: the block with the best inner-CV bacc wins. fixed=(block, C) skips the search."""
    if fixed is not None:
        b, C = fixed
        m, _, C = fit_lr(Xb[b], y, g, C=C)
        return b, m, float("nan"), C
    best = None
    for b, X in Xb.items():
        m, s, C = fit_lr(X, y, g)
        if best is None or s > best[2]:
            best = (b, m, s, C)
    return best


# parallel fold jobs; _G is filled before the fork
_G = {}


def cv_fold(job):
    key, f, perm_i, fixed = job
    G = _G[key]
    tr, te = G["folds"][f]
    y = G["y"] if perm_i is None else G["perm"][perm_i]
    tc = G["train_conds"]
    Xb = {b: np.concatenate([G["X"][c][b][tr] for c in tc]) for b in G["blocks"]}
    ytr = np.concatenate([y[tr]] * len(tc))
    gtr = np.concatenate([G["g"][tr]] * len(tc))
    b, m, s, C = fit_select(Xb, ytr, gtr, fixed)
    return (key, f, perm_i), dict(block=b, inner_bacc=s, C=C, pred={c: m.predict(G["X"][c][b][te]).astype(np.int8) for c in G["test_conds"]})


def run_cv(key, jobs_perm=None, workers=None, fixed=None):
    """Out-of-fold predictions {cond: yhat[n]} (perm_i None) or {perm_i: {cond: yhat}}; plus per-fold info."""
    G = _G[key]
    n = len(G["y"])
    perms = [None] if jobs_perm is None else list(jobs_perm)
    jobs = [(key, f, p, fixed) for p in perms for f in range(len(G["folds"]))]
    out = {p: {c: np.full(n, -1, np.int8) for c in G["test_conds"]} for p in perms}
    info = {}
    if workers == 1:
        res = list(map(cv_fold, jobs))
    else:
        with ProcessPoolExecutor(workers or CFG["workers"], mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(cv_fold, jobs))
    for (k, f, p), r in res:
        te = G["folds"][f][1]
        for c, yh in r["pred"].items():
            out[p][c][te] = yh
        info[(p, f)] = {k2: v for k2, v in r.items() if k2 != "pred"}
    for p in perms:
        for c in G["test_conds"]:
            assert (out[p][c] >= 0).all()
    return (out[None] if jobs_perm is None else out), info


# ============================================================================== metrics
def dev_codes(devs):
    u = sorted(set(devs))
    return np.array([u.index(d) for d in devs]), u


def boot_weights(ndev, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    W = np.ones((nboot + 1, ndev))
    for b in range(1, nboot + 1):
        W[b] = np.bincount(rng.integers(0, ndev, ndev), minlength=ndev)
    return W


def boot_bacc(y, yhat, dc, W):
    """Device-cluster bootstrap of 3-way balanced accuracy -> [point, lo, hi] (95% percentile)."""
    nd = W.shape[1]
    tot, cor = np.zeros((nd, 3)), np.zeros((nd, 3))
    np.add.at(tot, (dc, y), 1)
    np.add.at(cor, (dc, y), (yhat == y).astype(float))
    arr = ((W @ cor) / (W @ tot)).mean(1)
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def confusion(y, yhat):
    M = np.zeros((3, 3), int)
    np.add.at(M, (y, yhat), 1)
    return M


def summarize(y, yhat, table, W, dc):
    M = confusion(y, yhat)
    per_dev = {}
    for d in sorted(set(table.device)):
        m = (table.device == d).to_numpy()
        per_dev[d] = dict(bacc=bacc(y[m], yhat[m]), acc=float((y[m] == yhat[m]).mean()), model=table.model[m].iat[0])
    return dict(bacc=boot_bacc(y, yhat, dc, W), confusion=M.tolist(), recall=(M / M.sum(1, keepdims=True)).round(4).tolist(),
                majority_rate=float(np.bincount(y).max() / len(y)), per_device=per_dev)


# ============================================================================== M0 (file metadata)
def m0_raw(table):
    rows = []
    for r in table.itertuples():
        p = ROOT / r.src_path
        with Image.open(p) as im:
            w, h = im.size
            qt = getattr(im, "quantization", None) or {}
            fmt = im.format
        qs = [np.array(qt[k], float) for k in sorted(qt)] or [np.zeros(1)]
        qh = hashlib.sha1(json.dumps({str(k): list(map(int, v)) for k, v in sorted(qt.items())}).encode()).hexdigest()[:12]
        nb = p.stat().st_size
        rows.append(dict(w=w, h=h, aspect=w / h, bytes=nb, bpp=nb / (w * h), q0=qs[0].mean(), q1=qs[-1].mean(), qh=qh, fmt=fmt))
    return pd.DataFrame(rows)


def m0_tensor(table):
    """The same features for the preprocessed fov512 tensors: width, height, aspect, format only (bytes / quantisation tables do not exist)."""
    n = len(table)
    return pd.DataFrame(dict(w=[512] * n, h=[512] * n, aspect=[1.0] * n, bytes=0.0, bpp=0.0, q0=0.0, q1=0.0, qh="none", fmt="PNG"))


def m0_matrix(df, hashes):
    base = np.column_stack([np.log2(df.w), np.log2(df.h), df.aspect, df.bytes, df.bpp, df.q0, df.q1])
    oh = np.column_stack([(df.qh.to_numpy() == h).astype(float) for h in hashes]) if len(hashes) else np.zeros((len(df), 0))
    return np.hstack([base, oh])


def m0_cv(feat, y, folds):
    pred = np.full(len(y), -1)
    for tr, te in folds:
        hashes = sorted(set(feat.qh.iloc[tr]))
        clf = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(m0_matrix(feat.iloc[tr], hashes), y[tr])
        pred[te] = clf.predict(m0_matrix(feat.iloc[te], hashes))
    assert (pred >= 0).all()
    return pred


# ============================================================================== label
def decide_label(r):
    """r[(method)][(mode)][cond] = point bacc. Primary mode from config."""
    t = CFG["thresholds"]
    out = {}
    for mode in ("tmix", "clean"):
        A, B = r["A"][mode], r["B"][mode]
        pas = any(m["clean"] >= t["pass_clean"] and m["social_sim"] >= t["pass_social_sim"] for m in (A, B))
        kill = A["clean"] < t["kill_A_clean"] and B["clean"] < t["kill_B_clean"]
        lab = "PASS" if pas else "KILL" if kill else "INCONCLUSIVE"
        fail = lab == "KILL" or (lab == "INCONCLUSIVE" and max(A["clean"], B["clean"]) < t["pass_clean"])
        out[mode] = dict(label=lab, counts_as_failure=bool(fail), A_clean=A["clean"], A_social_sim=A["social_sim"],
                         B_clean=B["clean"], B_social_sim=B["social_sim"])
    return out


# ============================================================================== feature loading for B and the arms
def load_store(cond, dirs):
    store = {}
    for d in dirs:
        try:
            k, x = P.read_shards(cond, d)
        except FileNotFoundError:
            continue
        store.update(zip(k.tolist(), x))
    return store


def load_B(table, cond):
    store = load_store(cond, [P.cache_root("fov512")])
    miss = [s for s in table.sha256 if s not in store]
    if miss:
        raise SystemExit(f"B/{cond}: {len(miss)} rows without cached features")
    X = np.stack([store[s] for s in table.sha256]).astype(np.float32)
    assert X.shape[1] == 4608 and np.isfinite(X).all()
    return {b: np.ascontiguousarray(X[:, sl]) for b, sl in BLOCKS.items()}


# ============================================================================== camera-model arms (exploratory, no label)
def vision_model_transfer(X, table, n_splits=None, seed=SEED, C=None):
    """X: [n, d] native-image features aligned with table (native rows only). For each ordered pair (A, B) of devices of one model:
    train 'is model M' on A vs half of the non-M devices, test B vs the other half (AUROC); control = a device C of another model as
    the positive vs other-model negatives."""
    n_splits = n_splits or CFG["camera_model_arm"]["vision"]["n_neg_splits"]
    C = C or CFG["camera_model_arm"]["vision"]["C"]
    devs = sorted(set(table.device))
    mod = table.groupby("device").model.first().to_dict()
    sets = {}
    for d in devs:
        sets.setdefault(mod[d], []).append(d)
    rng = np.random.default_rng(seed)
    dv = table.device.to_numpy()
    res = []
    for m, ds in sorted(sets.items()):
        if len(ds) < 2:
            continue
        others = [d for d in devs if mod[d] != m]
        for a in ds:
            for b in ds:
                if a == b:
                    continue
                aucs, ctrl = [], []
                for _ in range(n_splits):
                    perm = rng.permutation(len(others))
                    neg_tr = [others[i] for i in perm[: len(others) // 2]]
                    neg_te = [others[i] for i in perm[len(others) // 2:]]
                    itr = np.isin(dv, [a] + neg_tr)
                    ytr = (dv[itr] == a).astype(int)
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(X[itr], ytr)
                    ite = np.isin(dv, [b] + neg_te)
                    aucs.append(roc_auc_score((dv[ite] == b).astype(int), clf.decision_function(X[ite])))
                    c = neg_te[rng.integers(len(neg_te))]          # control: a held-out device of ANOTHER model is the positive

                    negc = [d for d in neg_te if mod[d] != mod[c]]
                    itc = np.isin(dv, [c] + negc)
                    ctrl.append(roc_auc_score((dv[itc] == c).astype(int), clf.decision_function(X[itc])))
                res.append(dict(model=m, train_device=a, test_device=b, auroc=float(np.mean(aucs)), auroc_sd=float(np.std(aucs)),
                                control_auroc=float(np.mean(ctrl)), n_neg_devices=len(others)))
    return res


def hdrplus_table():
    m = pd.read_csv(ROOT / "experiments/019-source-diversity/hdrplus_manifest.csv", dtype={"group": str})
    mp_ = CFG["camera_model_arm"]["hdrplus"]["merge"]

    # the trailing word of 'make model' is a marketing name or a codename; codenames are merged (docs/08 sec. 0)
    out = []
    for s in m.camera_make_model:
        mdl = s.split(" ", 1)[1]
        out.append(mp_.get(mdl, mdl))
    m["model"] = out
    return m


def hdrplus_features(m):
    store = load_store("clean", [P.cache_root("fov512"), ROOT / "experiments/016-loso/cache/fov512", ROOT / "experiments/019-source-diversity/cache/feats"])
    miss = [s for s in m.sha256 if s not in store]
    if miss:
        raise SystemExit(f"HDR+: {len(miss)} rows without cached features")
    X = np.stack([store[s] for s in m.sha256]).astype(np.float32)
    return {b: np.ascontiguousarray(X[:, sl]) for b, sl in BLOCKS.items()}


def hdrplus_cv(X, y, groups, k=6):
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    assert_no_group_overlap(folds, groups)
    pred = np.full(len(y), -1)
    Cs = []
    for tr, te in folds:
        m, _, C = fit_lr(X[tr], y[tr], groups[tr])
        pred[te] = m.predict(X[te])
        Cs.append(C)
    return pred, Cs, folds


def boot_cluster_bacc(y, yhat, groups, k, nboot=NBOOT, seed=SEED):
    """Cluster bootstrap of k-way balanced accuracy (clusters = groups)."""
    ug = sorted(set(groups))
    gi = np.array([ug.index(g) for g in groups])
    tot, cor = np.zeros((len(ug), k)), np.zeros((len(ug), k))
    np.add.at(tot, (gi, y), 1)
    np.add.at(cor, (gi, y), (yhat == y).astype(float))
    rng = np.random.default_rng(seed)
    arr = []
    for b in range(nboot + 1):
        w = np.ones(len(ug)) if b == 0 else np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        t = w @ tot
        arr.append(np.mean((w @ cor)[t > 0] / t[t > 0]))
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


# ============================================================================== stages
def stage_all(workers):
    RES.mkdir(exist_ok=True)
    T0 = time.time()
    table = build_table()
    n_files = gate_dedupe(table)
    y = table.platform.to_numpy()
    dev = table.device.to_numpy()
    dc, dlist = dev_codes(dev)
    folds = make_folds(table)
    W = boot_weights(len(dlist))
    log(f"data: {len(table)} images = {table.photo.nunique()} triplets, {len(dlist)} devices, sha256 unique over {n_files} files; fold test devices:",
        [sorted(set(dev[te])) for _, te in folds])
    results = dict(config=CFG, n_images=len(table), n_triplets=int(table.photo.nunique()), n_devices=len(dlist),
                   folds=[sorted(set(dev[te])) for _, te in folds], chance=CHANCE, gates={})
    results["gates"]["sha256_unique_files"] = n_files

    # ---- features
    XA = {c: {"A": extract_A(table, c, workers)} for c in CONDS}
    XB = {c: load_B(table, c) for c in CONDS}
    results["gates"]["B_cache_rows_found"] = {c: len(table) for c in CONDS}

    # ---- main CV: methods x train mode
    methods = {"A": (XA, ["A"]), "B": (XB, list(BLOCKS))}
    preds, chosen = {}, {}
    for mname, (Xc, blocks) in methods.items():
        for mode, tc in CFG["train_modes"].items():
            key = f"{mname}_{mode}"
            _G[key] = dict(X=Xc, y=y, g=dev, folds=folds, train_conds=tc, test_conds=CONDS, blocks=blocks)
            t0 = time.time()
            pr, info = run_cv(key, workers=workers)
            preds[key] = pr
            chosen[key] = {f: info[(None, f)] for f in range(len(folds))}
            log(f"cv {key}: {time.time() - t0:.0f}s; " + " ".join(f"{c}={bacc(y, pr[c]):.3f}" for c in CONDS)
                + "; fold choices " + str([(v["block"], v["C"]) for v in chosen[key].values()]))
    results["cv"] = {}
    point = {"A": {}, "B": {}}
    for key, pr in preds.items():
        m, mode = key.split("_")
        results["cv"][key] = {c: summarize(y, pr[c], table, W, dc) for c in CONDS}
        results["cv"][key]["folds"] = {str(f): v for f, v in chosen[key].items()}
        point[m][mode] = {c: results["cv"][key][c]["bacc"][0] for c in CONDS}

    # ---- M0
    mraw, mten = m0_raw(table), m0_tensor(table)
    results["M0_raw"] = summarize(y, m0_cv(mraw, y, folds), table, W, dc)
    results["M0_preprocessed_tensor"] = summarize(y, m0_cv(mten, y, folds), table, W, dc)
    results["M0_raw"]["n_distinct_qt_hashes"] = int(mraw.qh.nunique())
    results["M0_raw"]["qt_hash_by_platform"] = {CLASSES[p]: mraw.qh[y == p].value_counts().head(5).to_dict() for p in range(3)}
    log(f"M0 raw bacc {results['M0_raw']['bacc']}; M0 on preprocessed tensors {results['M0_preprocessed_tensor']['bacc'][0]:.3f}")

    # ---- label
    results["label"] = decide_label(point)
    log("label:", json.dumps(results["label"]))
    results["primary_label"] = results["label"][CFG["primary_train_mode"]]["label"]
    json.dump(results, open(RES / "results.json", "w"), indent=1, default=float)

    # ---- permutation control (primary protocol, fixed C)
    nperm = CFG["n_perm"]
    rng = np.random.default_rng(SEED + 1)
    photos = table.photo.to_numpy()
    idx_by_photo = pd.Series(np.arange(len(table))).groupby(photos).apply(list).to_dict()
    perms = []
    for _ in range(nperm):
        yp = y.copy()
        for ix in idx_by_photo.values():
            yp[ix] = rng.permutation(y[ix])
        perms.append(yp)
    results["permutation"] = {}
    for mname, (Xc, blocks) in methods.items():
        key = f"{mname}_{CFG['primary_train_mode']}"
        ch = chosen[key]
        Cm = float(np.median([v["C"] for v in ch.values()]))
        blk = max(set(v["block"] for v in ch.values()), key=[v["block"] for v in ch.values()].count)
        pkey = key + "_perm"
        _G[pkey] = dict(_G[key], perm=perms, blocks=[blk] if mname == "B" else blocks)
        t0 = time.time()
        pr, _ = run_cv(pkey, jobs_perm=range(nperm), workers=workers, fixed=(blk, Cm))
        sc = {c: np.array([bacc(perms[i], pr[i][c]) for i in range(nperm)]) for c in CONDS}
        real = {c: point[mname][CFG["primary_train_mode"]][c] for c in CONDS}
        results["permutation"][mname] = {c: dict(real=real[c], perm_mean=float(sc[c].mean()), perm_sd=float(sc[c].std()),
                                               perm_q025=float(np.percentile(sc[c], 2.5)), perm_q975=float(np.percentile(sc[c], 97.5)),
                                               perm_max=float(sc[c].max()), p_value=float((1 + (sc[c] >= real[c]).sum()) / (nperm + 1)),
                                               chance_ok=bool(abs(sc[c].mean() - CHANCE) <= 0.02)) for c in CONDS}
        results["permutation"][mname]["fixed"] = dict(C=Cm, block=blk)
        log(f"perm {mname}: {time.time() - t0:.0f}s " + " ".join(f"{c}: mean {sc[c].mean():.3f} max {sc[c].max():.3f} (real {real[c]:.3f})" for c in CONDS))
    results["gates"]["permutation_ok"] = all(v[c]["chance_ok"] for v in results["permutation"].values() for c in CONDS)
    json.dump(results, open(RES / "results.json", "w"), indent=1, default=float)

    # ---- camera-model arms
    arms = {}
    nat = (table.platform == 0).to_numpy()
    tn = table[nat].reset_index(drop=True)
    Xn = XB["clean"]["b4"][nat]
    t0 = time.time()
    vr = vision_model_transfer(Xn, tn)
    aucs = np.array([r["auroc"] for r in vr])
    ctr = np.array([r["control_auroc"] for r in vr])
    rngb = np.random.default_rng(SEED)
    bs = [float(aucs[rngb.integers(0, len(aucs), len(aucs))].mean()) for _ in range(NBOOT)]
    bc = [float(ctr[rngb.integers(0, len(ctr), len(ctr))].mean()) for _ in range(NBOOT)]
    arms["vision_same_model_transfer"] = dict(pairs=vr, mean_auroc=[float(aucs.mean())] + np.percentile(bs, [2.5, 97.5]).tolist(),
                                              mean_control_auroc=[float(ctr.mean())] + np.percentile(bc, [2.5, 97.5]).tolist(),
                                              note="CI = bootstrap over ordered device pairs only; exploratory, no label")
    log(f"vision same-model transfer: mean AUROC {aucs.mean():.3f}, control {ctr.mean():.3f} ({time.time() - t0:.0f}s)")
    hm = hdrplus_table()
    XH = hdrplus_features(hm)
    names = sorted(hm.model.unique())
    yh = hm.model.map({n: i for i, n in enumerate(names)}).to_numpy()
    gh = hm.group.to_numpy()
    hres = dict(models=names, counts=hm.model.value_counts().to_dict(), n_groups=int(hm.group.nunique()), chance=1 / len(names),
                majority_rate=float(np.bincount(yh).max() / len(yh)))
    for b in ("b4", "b8", "b12"):
        pred, Cs, hf = hdrplus_cv(XH[b], yh, gh, len(names))
        M = np.zeros((len(names),) * 2, int)
        np.add.at(M, (yh, pred), 1)
        hres[b] = dict(bacc=boot_cluster_bacc(yh, pred, gh, len(names)), acc=float((pred == yh).mean()), confusion=M.tolist(), C=Cs)
        log(f"HDR+ {b}: bacc {hres[b]['bacc']}")
    # permutation (group-level label shuffle) on b4 with fixed C
    folds_h = list(GroupKFold(n_splits=5).split(XH["b4"], yh, gh))
    Cm = float(np.median(hres["b4"]["C"]))
    ug = sorted(set(gh))
    rngp = np.random.default_rng(SEED + 2)
    pb = []
    for _ in range(CFG["camera_model_arm"]["hdrplus"]["n_perm"]):
        mp_g = dict(zip(ug, rngp.permutation(ug)))                 # permute whole burst groups' model labels (keeps group structure)
        yp = np.array([yh[np.where(gh == mp_g[g])[0][0]] for g in gh])
        pr = np.full(len(yp), -1)
        for tr, te in folds_h:
            m, _, _ = fit_lr(XH["b4"][tr], yp[tr], gh[tr], C=Cm)
            pr[te] = m.predict(XH["b4"][te])
        pb.append(bacc(yp, pr, len(names)))
    hres["permutation_b4"] = dict(mean=float(np.mean(pb)), q025=float(np.percentile(pb, 2.5)), q975=float(np.percentile(pb, 97.5)), max=float(np.max(pb)), n=len(pb), C=Cm)
    arms["hdrplus_model_id"] = hres
    results["camera_model_arm"] = arms
    results["runtime_s"] = time.time() - T0
    json.dump(results, open(RES / "results.json", "w"), indent=1, default=float)
    write_tables(results)
    log(f"done in {time.time() - T0:.0f}s")


def fmt(ci):
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def write_tables(r):
    L = ["# H-A3 platform history: tables (generated by run.py from results/results.json)", "",
         f"{r['n_triplets']} content-identical triplets ({r['n_images']} images, {r['n_devices']} devices, D13 excluded); chance 3-way bacc = 0.333; "
         "5-fold GroupKFold by device; CI = device-cluster bootstrap (1,000). Majority-class rate 0.333.", ""]
    for mode in ("tmix", "clean"):
        L += [f"## Train mode: {mode} ({'PRIMARY, label' if mode == CFG['primary_train_mode'] else 'sensitivity'})", "",
              "| method | clean | social_sim | jpeg75_resize05 |", "|---|---|---|---|"]
        for m in ("A", "B"):
            L.append(f"| {m} | " + " | ".join(fmt(r["cv"][f"{m}_{mode}"][c]["bacc"]) for c in CONDS) + " |")
        L += ["", f"Label ({mode}): `{json.dumps(r['label'][mode])}`", ""]
    L += ["## M0 (file size + quantisation table etc., decision tree), raw files, clean only", "",
          f"- M0 raw: {fmt(r['M0_raw']['bacc'])}; distinct quantisation-table hashes: {r['M0_raw']['n_distinct_qt_hashes']}",
          f"- M0 on preprocessed tensors (w, h, aspect, format; constant): {fmt(r['M0_preprocessed_tensor']['bacc'])}", ""]
    L += ["## Confusion matrices (rows = true, cols = predicted: native, whatsapp, facebook_hq), primary mode", ""]
    for m in ("A", "B"):
        for c in CONDS:
            L.append(f"- {m} {c}: " + str(r["cv"][f"{m}_{CFG['primary_train_mode']}"][c]["confusion"]))
    L += ["", "## Fold choices (primary mode)", ""]
    for m in ("A", "B"):
        L.append(f"- {m}: " + str([(v["block"], v["C"]) for v in r["cv"][f"{m}_{CFG['primary_train_mode']}"]["folds"].values()]))
    L += ["", "## Permutation control (200 within-triplet label shuffles, fixed C)", "", "| method | cond | real | perm mean | perm 2.5-97.5% | perm max | p | chance_ok |", "|---|---|---|---|---|---|---|---|"]
    for m, v in r["permutation"].items():
        for c in CONDS:
            q = v[c]
            L.append(f"| {m} | {c} | {q['real']:.3f} | {q['perm_mean']:.3f} | {q['perm_q025']:.3f}-{q['perm_q975']:.3f} | {q['perm_max']:.3f} | {q['p_value']:.4f} | {q['chance_ok']} |")
    L += ["", "## Per-device bacc, primary mode (A | B), clean / social_sim", "", "| device | model | A clean | A social | B clean | B social |", "|---|---|---|---|---|---|"]
    ka, kb = f"A_{CFG['primary_train_mode']}", f"B_{CFG['primary_train_mode']}"
    for d, v in r["cv"][ka]["clean"]["per_device"].items():
        L.append(f"| {d} | {v['model']} | {v['bacc']:.2f} | {r['cv'][ka]['social_sim']['per_device'][d]['bacc']:.2f} | "
                 f"{r['cv'][kb]['clean']['per_device'][d]['bacc']:.2f} | {r['cv'][kb]['social_sim']['per_device'][d]['bacc']:.2f} |")
    arm = r["camera_model_arm"]
    v = arm["vision_same_model_transfer"]
    L += ["", "## Exploratory camera-model arm (no label)", "",
          f"VISION same-model device-pair transfer (b4, native, clean): mean AUROC {fmt(v['mean_auroc'])}; control (different-model positive) {fmt(v['mean_control_auroc'])}; chance 0.5. CI over ordered pairs only.", "",
          "| model | train dev | test dev | AUROC | control AUROC |", "|---|---|---|---|---|"]
    for p in v["pairs"]:
        L.append(f"| {p['model']} | {p['train_device']} | {p['test_device']} | {p['auroc']:.3f} | {p['control_auroc']:.3f} |")
    h = arm["hdrplus_model_id"]
    L += ["", f"HDR+ model ID ({len(h['models'])} merged models {h['models']}, {h['n_groups']} burst groups; chance {h['chance']:.3f}, majority rate {h['majority_rate']:.3f}):", "",
          "| block | bacc | acc |", "|---|---|---|"]
    for b in ("b4", "b8", "b12"):
        L.append(f"| {b} | {fmt(h[b]['bacc'])} | {h[b]['acc']:.3f} |")
    pp = h["permutation_b4"]
    L += ["", f"HDR+ b4 permutation (group-level label shuffles, n={pp['n']}): mean {pp['mean']:.3f}, 2.5-97.5% {pp['q025']:.3f}-{pp['q975']:.3f}, max {pp['max']:.3f}", "",
          f"Model counts: {h['counts']}", ""]
    (RES / "tables.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all"])
    ap.add_argument("--workers", type=int, default=CFG["workers"])
    a = ap.parse_args()
    stage_all(a.workers)
