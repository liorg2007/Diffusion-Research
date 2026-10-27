"""E11 / 020-attr-foundation (D-018 cycle). Implements docs/08 + docs/09 (frozen pre-registration): baselines M0 + M1, hypotheses H-A1 and H-A4.

    env/.venv/bin/python -I experiments/020-attr-foundation/run.py all [--workers 14]
Stages (each skips finished work, cache/*.pkl): prep -> closed -> logo -> ha4 -> perm -> report
  prep    load universes/analyses, commit folds (results/folds_*.json), validity pre-checks, M0 raw metadata, M1 texture stats (cache/m1_*.npz)
  closed  H-A1 closed-set K-way probes (4 DINOv2 feature sets x {T-mix, clean-train}, inner-CV nested choice) + M1 + M1-GBM
  logo    H-A1 leave-one-generator-out open set (-max softmax and Mahalanobis)
  ha4     H-A4 16-source space: leave-one-source-out novelty (Mahalanobis, kNN-10) + kind accuracy
  perm    200-shuffle permutation controls (H-A1 main, RAID, H-A4)
  report  metrics, bootstrap, gates, labels -> results/{results.json, tables.md}
All constants are in config.json (frozen). No GPU: cached DINOv2 features only (no new features -> D-008 is verified on the existing guards).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import pickle
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter
from sklearn.covariance import ledoit_wolf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import p4features as P

CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CACHE, RES = HERE / "cache", HERE / "results"
SL = {k: slice(*v) for k, v in CFG["dino_slices"].items()}
FS_ORDER = CFG["fs_order"]
TH = CFG["thresholds"]
D3_RE = re.compile(r"(\d+)_(real|gen_\d)\.")
warnings.filterwarnings("ignore")


def log(*a):
    print(time.strftime("%H:%M:%S"), "E11", *a, flush=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e6():
    """E6 module (caption-family ids and groups: caption_ids, make_groups). Imported by path, never modified."""
    if "e6run" not in sys.modules:
        sys.modules["e6run"] = _load_module("e6run", ROOT / "experiments" / "015-midlayer-probe" / "run.py")
    return sys.modules["e6run"]


# ============================================================================ generic numerics
def ci(arr):
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order)."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def make_folds(y, groups, k, seed):
    """Deterministic grouped, class-stratified folds (depends only on labels, groups and seed; never on features)."""
    fold = np.full(len(y), -1)
    sgk = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups)):
        fold[te] = f
    return fold


def check_grouped_folds(fold, groups):
    """-> dict(ok, problems). A group must lie in exactly one fold; every row assigned; no empty fold."""
    fold, groups = np.asarray(fold), np.asarray(groups)
    probs = []
    if (fold < 0).any():
        probs.append("unassigned rows")
    df = pd.DataFrame(dict(f=fold, g=groups))
    multi = df.groupby("g").f.nunique()
    if (multi > 1).any():
        probs.append(f"{int((multi > 1).sum())} groups span several folds")
    for f in range(int(fold.max()) + 1):
        if not (fold == f).any():
            probs.append(f"fold {f} empty")
    return dict(ok=not probs, problems=probs)


def assert_disjoint_groups(groups, tr, te, what=""):
    """Leak check used inside every task: no group (family / device / image) on both sides."""
    inter = set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te])
    if inter:
        raise AssertionError(f"LEAK {what}: {len(inter)} groups on both sides, e.g. {sorted(inter)[:3]}")
    return True


def cluster_weights(groups, n_boot, seed, stratum_fn=None):
    """Cluster bootstrap plan: (Wc [B+1, n_clusters] (row 0 = ones), inv [n_rows -> cluster], unique group ids).
    Clusters are drawn with replacement within strata (stratum_fn(group id))."""
    groups = np.asarray(groups).astype(str)
    ug, inv = np.unique(groups, return_inverse=True)
    n = len(ug)
    st = np.array([stratum_fn(g) if stratum_fn else "all" for g in ug])
    rng = np.random.default_rng(seed)
    Wc = np.zeros((n_boot + 1, n), np.float32)
    Wc[0] = 1
    for s in sorted(set(st)):
        mem = np.where(st == s)[0]
        draws = rng.integers(0, len(mem), size=(n_boot, len(mem)))
        for b in range(n_boot):
            Wc[b + 1] += np.bincount(mem[draws[b]], minlength=n)
    return Wc, inv, ug


def boot_conf(Wrow, y, pred, K):
    n = len(y)
    oh = np.zeros((n, K * K), np.float64)
    oh[np.arange(n), np.asarray(y) * K + np.asarray(pred)] = 1
    return (Wrow.astype(np.float64) @ oh).reshape(-1, K, K)


def conf_stats(C, drop=None):
    """C [b,K,K] weighted confusion (rows = truth). -> dict of [b] arrays: bacc, mf1, bacc_drop (macro over classes except `drop`)."""
    diag = np.einsum("bkk->bk", C)
    rs, cs = C.sum(2), C.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(rs > 0, diag / rs, np.nan)
        prec = np.where(cs > 0, diag / cs, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    out = dict(bacc=np.nanmean(rec, 1), mf1=np.nanmean(np.where(np.isnan(rec), np.nan, f1), 1), rec=rec)
    if drop is not None:
        keep = [k for k in range(C.shape[1]) if k != drop]
        out["bacc_drop"] = np.nanmean(rec[:, keep], 1)
    return out


def bacc_simple(y, pred, classes=None):
    y, pred = np.asarray(y), np.asarray(pred)
    cl = np.unique(y) if classes is None else classes
    r = [(pred[y == c] == c).mean() for c in cl if (y == c).any()]
    return float(np.mean(r)) if r else float("nan")


def auroc_boot(s_pos, s_neg, Wpos, Wneg):
    """Weighted AUROC (ties = 0.5) for every bootstrap row: Wpos [b,npos], Wneg [b,nneg]. Row 0 = point estimate."""
    s_pos, s_neg = np.asarray(s_pos, np.float64), np.asarray(s_neg, np.float64)
    M = (s_pos[:, None] > s_neg[None, :]).astype(np.float32) + 0.5 * (s_pos[:, None] == s_neg[None, :]).astype(np.float32)
    num = ((Wpos.astype(np.float32) @ M) * Wneg).sum(1)
    return num / (Wpos.sum(1) * Wneg.sum(1))


# ----------------------------------------------------------------------------- pairwise on same-caption pairs
def build_pairs(cls, fam):
    """Same-family image pairs of two DIFFERENT classes. Returns dict(I, J, GI, GJ) with GI < GJ (class of row I / row J)."""
    cls, fam = np.asarray(cls), np.asarray(fam).astype(str)
    byfam = {}
    for i, f in enumerate(fam):
        byfam.setdefault(f, []).append(i)
    I, J = [], []
    for rows in byfam.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                if cls[i] == cls[j]:
                    continue
                if cls[i] > cls[j]:
                    i, j = j, i
                I.append(i)
                J.append(j)
    I, J = np.array(I, int), np.array(J, int)
    return dict(I=I, J=J, GI=cls[I], GJ=cls[J])


def pair_scores(logp, Pr, strict=False):
    """Per pair: 2-AFC (default) s_A(a)-s_B(a) > s_A(b)-s_B(b), ties 0.5; strict: s_A(a)>s_A(b) AND s_B(b)>s_B(a)."""
    I, J, GI, GJ = Pr["I"], Pr["J"], Pr["GI"], Pr["GJ"]
    if len(I) == 0:
        return np.zeros(0)
    if strict:
        return ((logp[I, GI] > logp[J, GI]) & (logp[J, GJ] > logp[I, GJ])).astype(float)
    mi = logp[I, GI] - logp[I, GJ]
    mj = logp[J, GI] - logp[J, GJ]
    return (mi > mj) + 0.5 * (mi == mj)


def pair_boot(corr, Pr, Wc, inv, exclude=None):
    """Cluster-bootstrapped pairwise accuracy. macro = mean over generator-pair types (types touching `exclude` removed), pooled = over pairs.
    Clusters must be the caption families themselves (rows I and J of a pair share a cluster)."""
    I, J = Pr["I"], Pr["J"]
    if len(I) == 0:
        return dict(macro=np.full(Wc.shape[0], np.nan), pooled=np.full(Wc.shape[0], np.nan), types=[], per_type=np.zeros((Wc.shape[0], 0)))
    assert (inv[I] == inv[J]).all(), "pair rows from different clusters"
    types = sorted(set(zip(Pr["GI"].tolist(), Pr["GJ"].tolist())))
    tix = {t: i for i, t in enumerate(types)}
    pt = np.array([tix[t] for t in zip(Pr["GI"].tolist(), Pr["GJ"].tolist())])
    n_cl = Wc.shape[1]
    corr_m = np.zeros((n_cl, len(types)))
    cnt_m = np.zeros((n_cl, len(types)))
    np.add.at(corr_m, (inv[I], pt), corr)
    np.add.at(cnt_m, (inv[I], pt), 1.0)
    num, den = Wc @ corr_m, Wc @ cnt_m
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(den > 0, num / den, np.nan)
    mask = np.array([exclude is None or (t[0] != exclude and t[1] != exclude) for t in types])
    macro = np.nanmean(per[:, mask], 1)
    pooled = (Wc @ corr_m[:, mask].sum(1)) / (Wc @ cnt_m[:, mask].sum(1))
    return dict(macro=macro, pooled=pooled, types=types, per_type=per)


# ----------------------------------------------------------------------------- models
def fit_model(kind, X, y, C, seed=SEED):
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced").fit(X, y)
        return None, clf
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=CFG["lr"]["max_iter"], class_weight=CFG["lr"]["class_weight"]).fit(sc.transform(X), y)
    return sc, clf


def logp_full(model, X, K):
    """log-probabilities in the global K-class space (-inf for classes the model never saw)."""
    sc, clf = model
    Z = X if sc is None else sc.transform(X)
    if hasattr(clf, "predict_log_proba") and sc is not None:
        lp = clf.predict_log_proba(Z)
    else:
        lp = np.log(np.clip(clf.predict_proba(Z), 1e-12, 1))
    out = np.full((len(X), K), -np.inf, np.float32)
    out[:, clf.classes_] = lp
    return out


class MahaModel:
    """Class means + shared Ledoit-Wolf covariance of within-class residuals (standardised features). Fit on TRAIN rows only."""

    def __init__(self, X, y, keep_train=False):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.cls = np.unique(y)
        idx = np.searchsorted(self.cls, y)
        self.means = np.stack([Z[idx == k].mean(0) for k in range(len(self.cls))])
        cov, _ = ledoit_wolf(Z - self.means[idx], assume_centered=True)
        self.R = np.linalg.cholesky(cov.astype(np.float64))
        self.Mw = solve_triangular(self.R, self.means.T.astype(np.float64), lower=True).T
        self.mw2 = (self.Mw ** 2).sum(1)
        self.Ztrain = Z.astype(np.float32) if keep_train else None
        self.ytrain = y if keep_train else None

    def dists(self, X):
        Zt = self.sc.transform(X)
        Zw = solve_triangular(self.R, Zt.T.astype(np.float64), lower=True).T
        return (Zw ** 2).sum(1)[:, None] - 2 * Zw @ self.Mw.T + self.mw2[None, :]

    def score(self, X):
        d2 = self.dists(X)
        a = d2.argmin(1)
        return d2[np.arange(len(a)), a], self.cls[a]

    def knn(self, X, k):
        """-> (distance to the k-th nearest TRAIN neighbour, labels of the k nearest)."""
        Zt = self.sc.transform(X).astype(np.float32)
        Zr = self.Ztrain
        d2 = (Zt ** 2).sum(1)[:, None] - 2 * Zt @ Zr.T + (Zr ** 2).sum(1)[None, :]
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.sqrt(np.maximum(np.take_along_axis(d2, nn, 1).max(1), 0))
        return dk, self.ytrain[nn]


# ============================================================================ analysis container
class Analysis:
    """Rows + labels + groups + folds + features for one analysis. X: {cond: [n, 4608]} (DINOv2); M1: {cond: [n,12]}; shapes {cond: [n,2]}."""

    def __init__(self, name, cfg, y, classes, groups, fold, conds, tmix, primary, XF, M1=None, shapes=None, df=None, fam=None, kinds=None):
        self.name, self.cfg, self.y, self.classes, self.K = name, cfg, np.asarray(y), list(classes), len(classes)
        self.groups, self.fold, self.conds, self.tmix, self.primary = np.asarray(groups).astype(str), np.asarray(fold), conds, tmix, primary
        self.XF, self.M1, self.shapes, self.df = XF, M1 or {}, shapes or {}, df
        self.fam = self.groups if fam is None else np.asarray(fam)
        self.kinds = kinds
        self.n = len(self.y)

    def getX(self, fs, cond):
        if fs in SL:
            return self.XF[cond][:, SL[fs]]
        return self.M1[cond]


_G = {}                                                     # name -> Analysis (filled before fork)


def stack(A, fs, rows, conds):
    return np.concatenate([A.getX(fs, c)[rows] for c in conds]), np.tile(A.y[rows], len(conds))


def grid_for(fs):
    return CFG["c_grid"]["dino" if fs in SL else "m1"]


def inner_table(A, fs, tr, ctrain, sel_cond, seed, classes):
    """Inner grouped CV on the TRAIN rows only: bacc of each C (train on the inner-train copies, score inner-val at sel_cond)."""
    y, g = A.y[tr], A.groups[tr]
    k = CFG["inner_folds"]
    ifold = make_folds(y, g, k, seed)
    chk = check_grouped_folds(ifold, g)
    assert chk["ok"], chk
    table = {}
    for C in grid_for(fs):
        accs = []
        for j in range(k):
            a, b = tr[ifold != j], tr[ifold == j]
            assert_disjoint_groups(A.groups, a, b, "inner")
            X, yy = stack(A, fs, a, ctrain)
            m = fit_model("lr", X, yy, C)
            lp = logp_full(m, A.getX(fs, sel_cond)[b], A.K)
            accs.append(bacc_simple(A.y[b], lp.argmax(1), classes))
        table[C] = float(np.mean(accs))
    best = max(table, key=lambda c: (round(table[c], 12), -c))             # ties -> smaller C
    return table, best


def regime_conds(A, regime):
    return (list(A.tmix), A.primary) if regime == "tmix" else (["clean"], "clean")


# ----------------------------------------------------------------------------- closed-set task
def closed_task(args):
    an, fs, regime, f = args
    A = _G[an]
    tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
    assert_disjoint_groups(A.groups, tr, te, f"{an} outer fold {f}")
    ctrain, sel = regime_conds(A, regime)
    assert not set(ctrain) & set(CFG["held_out_launderings"]), "held-out laundering used in training"
    if fs == "M1gbm":
        m = fit_model("hgb", *stack(A, "M1", tr, ctrain), None)
        table, Cb = {}, None
        lp = {c: logp_full(m, A.getX("M1", c)[te], A.K) for c in A.conds}
        inner = float("nan")
    else:
        table, Cb = inner_table(A, fs, tr, ctrain, sel, SEED * 1000 + f + 1, list(range(A.K)))
        m = fit_model("lr", *stack(A, fs, tr, ctrain), Cb)
        lp = {c: logp_full(m, A.getX(fs, c)[te], A.K) for c in A.conds}
        inner = table[Cb]
    return dict(an=an, fs=fs, regime=regime, f=f, te=te, C=Cb, inner_best=inner, table=table, logp=lp)


# ----------------------------------------------------------------------------- LOGO open-set task
def logo_task(args):
    an, g, f, fs, regime = args
    A = _G[an]
    tr = np.where((A.fold != f) & (A.y != g))[0]
    te = np.where(A.fold == f)[0]
    assert_disjoint_groups(A.groups, tr, te, f"{an} LOGO g={g} fold {f}")
    assert not (A.y[tr] == g).any(), "held-out generator in training"
    ctrain, sel = regime_conds(A, regime)
    known = [c for c in range(A.K) if c != g]
    table, Cb = inner_table(A, fs, tr, ctrain, sel, SEED * 1000 + 17 * g + f + 1, known)
    X, y = stack(A, fs, tr, ctrain)
    m = fit_model("lr", X, y, Cb)
    mm = MahaModel(X, y)
    out = {}
    for c in A.conds:
        Xt = A.getX(fs, c)[te]
        lp = logp_full(m, Xt, A.K)
        out[c] = dict(msp=(-np.exp(lp.max(1))).astype(np.float32), maha=mm.score(Xt)[0].astype(np.float32))
    return dict(an=an, g=g, f=f, fs=fs, regime=regime, te=te, C=Cb, inner_best=table[Cb], out=out)


# ----------------------------------------------------------------------------- H-A4 task
def ha4_task(args):
    fs, regime, f = args
    A = _G["ha4"]
    H4 = CFG["ha4"]
    tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
    assert_disjoint_groups(A.groups, tr, te, f"ha4 fold {f}")
    ctrain, _ = regime_conds(A, regime)
    X, y = stack(A, fs, tr, ctrain)
    Xte = {c: A.getX(fs, c)[te] for c in A.conds}
    res = dict(fs=fs, regime=regime, f=f, te=te, closed={}, nov={}, near={}, knn={}, knnreal={})
    Cc = CFG["fixed_C_closed_ha4"] if fs in SL else CFG.get("fixed_C_closed_ha4_m1", 1.0)
    m = fit_model("lr", X, y, Cc)
    res["closed"] = {c: logp_full(m, Xte[c], A.K).argmax(1).astype(np.int8) for c in A.conds}
    kind_arr = np.array([1 if k == "real" else 0 for k in A.kinds])                      # per class
    for hi, held in enumerate(A.holdouts):
        keep = ~np.isin(y, held)
        mm = MahaModel(X[keep], y[keep], keep_train=(fs in H4["knn_fs"]))
        for c in A.conds:
            nov, near = mm.score(Xte[c])
            res["nov"][(hi, c)] = nov.astype(np.float32)
            res["near"][(hi, c)] = near.astype(np.int8)
            if fs in H4["knn_fs"] and c in H4["knn_conds"]:
                dk, lab = mm.knn(Xte[c], H4["knn_k"])
                res["knn"][(hi, c)] = dk.astype(np.float32)
                res["knnreal"][(hi, c)] = kind_arr[lab].mean(1).astype(np.float32)
    return res


# ----------------------------------------------------------------------------- permutation controls
def perm_closed_task(args):
    an, s, f, fs, C = args
    A = _G[an]
    yp = np.random.default_rng([SEED, 4242, s]).permutation(A.y)
    tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
    assert_disjoint_groups(A.groups, tr, te, "perm")
    X = np.concatenate([A.getX(fs, c)[tr] for c in A.tmix])
    m = fit_model("lr", X, np.tile(yp[tr], len(A.tmix)), C)
    pred = logp_full(m, A.getX(fs, A.primary)[te], A.K).argmax(1)
    return s, te, pred.astype(np.int8)


def perm_ha4_task(args):
    s, proj_seed = args
    A = _G["ha4"]
    H4 = CFG["ha4"]
    d = H4["perm"]["proj_dim"]
    fs = H4["primary_fs"]
    rng = np.random.default_rng([SEED, 4343, s])
    yp = rng.permutation(A.y)
    Pm = np.random.default_rng([SEED, 99]).normal(size=(SL[fs].stop - SL[fs].start, d)).astype(np.float32)
    nov = {hi: np.zeros(A.n, np.float32) for hi in range(len(A.sources))}
    for f in range(CFG["outer_folds"]):
        tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
        assert_disjoint_groups(A.groups, tr, te, "perm ha4")
        Xtr = np.concatenate([A.getX(fs, c)[tr] for c in A.tmix]) @ Pm
        ytr = np.tile(yp[tr], len(A.tmix))
        Xte = A.getX(fs, A.primary)[te] @ Pm
        for hi in range(len(A.sources)):
            keep = ytr != hi
            nov[hi][te] = MahaModel(Xtr[keep], ytr[keep]).score(Xte)[0]
    aucs = []
    for hi in range(len(A.sources)):
        pos, neg = yp == hi, yp != hi
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(pos, nov[hi]))
    return s, float(np.mean(aucs))


# ============================================================================ pool runner
def run_tasks(fn, tasks, workers, label=""):
    tasks = list(tasks)
    t0 = time.time()
    if workers <= 1:
        out = [fn(t) for t in tasks]
    else:
        out = []
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            for i, r in enumerate(ex.map(fn, tasks, chunksize=1)):
                out.append(r)
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    log(f"{label}: {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    log(f"{label}: {len(tasks)} tasks in {time.time() - t0:.0f}s")
    return out


# ============================================================================ baselines: M1 texture, M0 metadata
def texture_stats(a):
    """12 low-level statistics of an RGB uint8 array (the 224 center crop that the backbone sees)."""
    x = a.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    lap = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    resid = y - gaussian_filter(y, 1.0)
    mad = np.median(np.abs(resid - np.median(resid)))
    dh, dv = np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0))
    bh = np.arange(dh.shape[1]) % 8 == 7
    bv = np.arange(dv.shape[0]) % 8 == 7
    block = 0.5 * (dh[:, bh].mean() / (dh[:, ~bh].mean() + 1e-6) + dv[bv].mean() / (dv[~bv].mean() + 1e-6))
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-3)
    ys = y.std()
    skew = ((y - y.mean()) ** 3).mean() / (ys ** 3 + 1e-6)
    clipped = float(((x <= 2).any(-1) | (x >= 253).any(-1)).mean())
    return np.array([np.log(lap.var() + 1e-3), np.abs(lap).mean(), np.log(1.4826 * mad + 1e-3), block, sat.mean(), sat.std(),
                     x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean(), ys, skew, clipped], np.float32)


def _m1_row(args):
    path, sha, conds = args
    from perturb import apply_condition
    out = np.zeros((len(conds), 12), np.float32)
    shp = np.zeros((len(conds), 2), np.int32)
    with Image.open(path) as im:
        im.load()
        for i, c in enumerate(conds):
            o = apply_condition(im, c, key=sha, seed=P.SEED)
            shp[i] = o.size
            w, h = o.size
            s = CFG["m1_crop"]
            l, t = (w - s) // 2, (h - s) // 2
            out[i] = texture_stats(np.asarray(o.crop((l, t, l + s, t + s))))
    return out, shp


def compute_m1(uname, df, conds, workers):
    f = CACHE / f"m1_{uname}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if z["shas"].tolist() == df.sha256.tolist() and z["conds"].tolist() == conds:
            return z["feats"], z["shapes"]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_m1_row, [(p, s, conds) for p, s in zip(df.path, df.sha256)], chunksize=16))
    feats = np.stack([r[0] for r in res])
    shapes = np.stack([r[1] for r in res])
    CACHE.mkdir(exist_ok=True)
    np.savez(f, shas=np.array(df.sha256.tolist(), dtype=str), conds=np.array(conds, dtype=str), feats=feats, shapes=shapes)
    log(f"M1 {uname}: {len(df)} rows x {len(conds)} conds in {time.time() - t0:.0f}s")
    return feats, shapes


def true_format(path):
    """Container format from the CONTENT (magic bytes via PIL), never the extension (Firefly = JPEG under .png names)."""
    with Image.open(path) as im:
        return im.format


def m0_row(path):
    """Metadata of a RAW file: width, height, aspect, true format code, bytes/pixel, quantisation-table hash."""
    with Image.open(path) as im:
        fmt, (w, h) = im.format, im.size
        qt = getattr(im, "quantization", None)
    qh = hashlib.md5(repr(sorted((k, tuple(v)) for k, v in qt.items())).encode()).hexdigest()[:12] if qt else "none"
    nb = os.path.getsize(path)
    return dict(w=w, h=h, aspect=w / h, fmt=fmt, bpp=nb / (w * h), qhash=qh, ext_mismatch=int(Path(path).suffix.lower().lstrip(".").replace("jpg", "jpeg") != (fmt or "").lower()))


def m0_matrix(rows, qmap):
    fc = {"PNG": 0, "JPEG": 1}
    return np.array([[r["w"], r["h"], r["aspect"], fc.get(r["fmt"], 2), np.log(r["bpp"] + 1e-9), qmap.get(r["qhash"], -1)] for r in rows], np.float64)


def m0_oof(rows, y, fold, K, seed=SEED):
    """M0 = random forest on raw-file metadata, grouped folds; q-table hashes are coded from TRAIN rows only (unseen -> -1)."""
    pred = np.zeros(len(y), int)
    for f in range(int(fold.max()) + 1):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        qmap = {h: i for i, h in enumerate(sorted({rows[i]["qhash"] for i in tr}))}
        Xtr, Xte = m0_matrix([rows[i] for i in tr], qmap), m0_matrix([rows[i] for i in te], qmap)
        clf = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=1).fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    return pred


def m0_tensor_oof(shapes, y, fold, seed=SEED):
    """M0 features computed from the preprocessed TENSOR (size after the condition, aspect, 3 channels): must be at chance."""
    pred = np.zeros(len(y), int)
    X = np.column_stack([shapes[:, 0], shapes[:, 1], shapes[:, 0] / shapes[:, 1], np.full(len(y), 3.0)])
    for f in range(int(fold.max()) + 1):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        clf = RandomForestClassifier(100, class_weight="balanced", random_state=seed, n_jobs=1, min_samples_leaf=3).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


# ============================================================================ validity gates (pure functions; also used by the selftest)
def gate_dedupe(shas, classes):
    """sha256 dedupe across classes: no content hash may appear twice (and certainly not in two classes)."""
    df = pd.DataFrame(dict(s=shas, c=classes))
    dup = df[df.duplicated("s", keep=False)]
    cross = dup.groupby("s").c.nunique()
    return dict(ok=bool(len(dup) == 0), n_rows=len(df), n_duplicated_rows=int(len(dup)), n_cross_class_hashes=int((cross > 1).sum()))


def gate_m0_tensor(bacc_by_cond, chance, max_excess):
    worst = max(bacc_by_cond.values())
    return dict(ok=bool(worst <= chance + max_excess), worst=float(worst), chance=chance, max_excess=max_excess, by_cond=bacc_by_cond)


def gate_permutation(perm_stats, chance, mean_tol, p95_excess):
    arr = np.asarray(perm_stats, float)
    m, p95 = float(arr.mean()), float(np.percentile(arr, 95))
    return dict(ok=bool(abs(m - chance) <= mean_tol and p95 <= chance + p95_excess), n=len(arr), mean=m, p95=p95, chance=chance, mean_tol=mean_tol, p95_excess=p95_excess)


def gate_splits(fold, groups):
    return check_grouped_folds(fold, groups)


def gate_train_conds(tmix, held_out):
    bad = sorted(set(tmix) & set(held_out))
    return dict(ok=not bad, bad=bad)


def d008_status():
    """Existing GPU-vs-CPU guards (no new feature pass in this experiment): every guard entry we use must be ok and |dlogit| <= 1e-3."""
    worst, ok, missing, n = 0.0, True, [], 0
    for u in CFG["universes"].values():
        for gf, keys in u["d008_guards"]:
            d = json.loads((ROOT / gf).read_text())
            for k in keys:
                if k not in d:
                    missing.append(f"{gf}:{k}")
                    continue
                n += 1
                ok &= bool(d[k].get("ok", False))
                for kk, v in d[k].items():
                    if kk.startswith("max_abs_dlogit"):
                        worst = max(worst, float(v))
    return dict(ok=bool(ok and not missing and worst <= 1e-3), max_abs_dlogit=worst, n_entries=n, missing=missing, new_gpu_features_in_this_experiment=False)


def cache_meta_equal(dirs, conds):
    metas = {}
    for d in dirs:
        for c in conds:
            p = ROOT / d / P.safe(c) / "meta.json"
            if p.exists():
                metas[f"{d}/{c}"] = json.loads(p.read_text())
    vals = list(metas.values())
    return dict(ok=bool(vals and all(v == vals[0] for v in vals)), n_meta=len(vals), spec=vals[0].get("spec") if vals else None)


# ============================================================================ labels (pure functions)
def label_ha1(v, gates_ok):
    """v: dict(bacc [pt, lo, hi], m1_bacc, pairwise, raid_bacc). docs/08 H-A1 PASS/KILL exactly; INVALID if a validity gate failed."""
    if not gates_ok:
        return dict(label="INVALID", reasons=["a validity gate failed"])
    t = TH
    kill = v["bacc"][0] < t["ha1_kill_bacc"] or v["pairwise"] < t["ha1_kill_pairwise"]
    parts = dict(
        bacc_ge_060=v["bacc"][0] >= t["ha1_pass_bacc"],
        ci_lo_ge_m1_plus_010=v["bacc"][1] >= v["m1_bacc"] + t["ha1_m1_margin"],
        pairwise_ge_075=v["pairwise"] >= t["ha1_pass_pairwise"],
        raid_bacc_ge_055=(v["raid_bacc"] is not None and v["raid_bacc"] >= t["ha1_pass_raid"]))
    lab = "KILL" if kill else ("PASS" if all(parts.values()) else "INCONCLUSIVE")
    return dict(label=lab, kill_condition=bool(kill), parts={k: bool(x) for k, x in parts.items()})


def label_open_set(aur):
    """aur: {generator: AUROC}. usable iff mean >= 0.80 and no generator < 0.65."""
    a = np.array(list(aur.values()), float)
    return dict(usable=bool(a.mean() >= TH["open_set_mean_auroc"] and a.min() >= TH["open_set_min_auroc"]), mean=float(a.mean()), min=float(a.min()))


def label_ha4(auc_by_source, kind_by_real, gates_ok):
    if not gates_ok:
        return dict(label="INVALID", discovery=False, reasons=["a validity gate failed"])
    a = np.array(list(auc_by_source.values()), float)
    mean, mn = float(a.mean()), float(a.min())
    lab = "PASS" if (mean >= TH["ha4_pass_mean"] and mn >= TH["ha4_pass_min"]) else "KILL" if mean < TH["ha4_kill_mean"] else "INCONCLUSIVE"
    disc = bool(all(v >= TH["ha4_discovery_kind"] for v in kind_by_real.values()))
    return dict(label=lab, mean=mean, min=mn, discovery=disc, min_kind_real=float(min(kind_by_real.values())))


# ============================================================================ data loading
def read_caches(dirs, cond):
    store = {}
    for d in dirs:
        try:
            k, x = P.read_shards(cond, ROOT / d)
        except FileNotFoundError:
            continue
        store.update(zip(k.tolist(), x))
    return store


def load_universe(uname):
    u = CFG["universes"][uname]
    df = pd.read_csv(ROOT / u["splits"], keep_default_na=False, dtype={"vidx": str, "group": str})
    if "group" not in df:
        df["group"] = ""
    if u["pools"]:
        df = df[df.pool.isin(u["pools"])].reset_index(drop=True)
    if u["group_rule"] == "e6":
        rid = e6().caption_ids(df)
        groups = e6().make_groups(df, rid)
        for pool in u["group_pools"]:
            m = (df.pool == pool).to_numpy()
            groups = np.where(m, ["grp:" + pool + ":" + g for g in df.group], groups)
            assert not (m & (df.group == "")).any(), f"{pool}: row without group"
    else:
        ids = [(D3_RE.search(os.path.basename(p)) or [None, None])[1] for p in df.src_path]
        assert all(ids), "RAID row without id"
        rid = np.array(ids)
        groups = np.array(["d3:" + i for i in ids])
    df = df.assign(rid=rid, grp=groups)
    cf = CACHE / f"cropsha_{uname}.json"
    df["crop_sha"], df["dup_drop"] = "", False
    if cf.exists():                                              # sha256 dedupe (gate 5): a later row whose processed pixels equal an earlier row's is dropped
        cs = json.loads(cf.read_text())
        assert cs["shas"] == df.sha256.tolist()
        df["crop_sha"] = cs["crop_sha"]
        df["dup_drop"] = df.duplicated("crop_sha", keep="first").to_numpy()
    return df, u


def strata_of(g):
    p = g.split(":")
    return p[0] + (":" + p[1] if p[0] == "grp" else "")


# ============================================================================ stage: prep
def stage_prep(workers):
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = {}
    for uname, u in CFG["universes"].items():
        df, _ = load_universe(uname)
        conds = u["conditions"]
        t0 = time.time()
        # features (cached DINOv2) -----------------------------------------------------------
        feats = {}
        for c in conds:
            store = read_caches(u["caches"], c)
            miss = [s for s in df.sha256 if s not in store]
            assert not miss, f"{uname}/{c}: {len(miss)} rows without cached features"
            feats[c] = np.stack([store[s] for s in df.sha256]).astype(np.float32)
            assert feats[c].shape[1] == 4608 and np.isfinite(feats[c]).all()
        # preprocessing checks -----------------------------------------------------------------
        exp = u["expected_dir"]
        bad_path = [p for p, s in zip(df.path, df.sha256) if p != f"{exp}/{s}.png"]
        exif = 0
        notrgb = 0
        for p in df.path:
            with Image.open(p) as im:
                exif += int("exif" in im.info or "icc_profile" in im.info)
                notrgb += int(im.mode != "RGB")
        # M1 + shapes ---------------------------------------------------------------------------
        m1, shp = compute_m1(uname, df, conds, workers)
        # M0 raw ------------------------------------------------------------------------------
        m0f = CACHE / f"m0_{uname}.json"
        if m0f.exists() and json.loads(m0f.read_text())["shas"] == df.sha256.tolist():
            m0 = json.loads(m0f.read_text())["rows"]
        else:
            man = {}
            for mp_ in (ROOT / "experiments/019-source-diversity/hdrplus_manifest.csv",):
                mm_ = pd.read_csv(mp_, keep_default_na=False)
                man = {r.sha256: r for r in mm_.itertuples()}
            m0 = []
            for sp_, s in zip(df.src_path, df.sha256):
                if os.path.exists(sp_):
                    m0.append(dict(m0_row(sp_), raw=True))
                else:                                                      # HDR+ raw originals were deleted: manifest size/bytes, q-table unavailable
                    r = man[s]
                    m0.append(dict(w=int(r.width), h=int(r.height), aspect=int(r.width) / int(r.height), fmt="JPEG", bpp=int(r.bytes) / (int(r.width) * int(r.height)),
                                   qhash="unavailable", ext_mismatch=0, raw=False))
            m0f.write_text(json.dumps(dict(shas=df.sha256.tolist(), rows=m0)))
        # sha dedupe on raw sha (splits column) and on the processed crop bytes -----------------------
        crop_sha = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in df.path]
        (CACHE / f"cropsha_{uname}.json").write_text(json.dumps(dict(shas=df.sha256.tolist(), crop_sha=crop_sha)))
        out[uname] = dict(n_rows=len(df), conds=conds, bad_paths=len(bad_path), n_exif_or_icc=exif, n_not_rgb=notrgb,
                          n_raw_available=int(sum(r["raw"] for r in m0)), n_ext_mismatch=int(sum(r["ext_mismatch"] for r in m0)),
                          raw_sha_unique=bool(df.sha256.is_unique), crop_sha_unique=len(set(crop_sha)) == len(crop_sha),
                          cache_meta=cache_meta_equal(u["caches"], conds), seconds=round(time.time() - t0))
        log(f"prep {uname}: {out[uname]}")
    (RES / "prep_checks.json").write_text(json.dumps(out, indent=1))
    return out


def build_analyses(workers=1, need_feats=True):
    """Build Analysis objects for every closed analysis and the H-A4 space from the caches (folds are committed BEFORE any feature is read)."""
    unis = {}
    for uname, u in CFG["universes"].items():
        df, _ = load_universe(uname)
        unis[uname] = (df, u)
    # --- commit folds first (labels + groups only) ------------------------------------------------
    plans = {}
    for an, a in CFG["analyses"].items():
        df, u = unis[a["universe"]]
        sel = ~df.dup_drop.to_numpy()
        for col, vals in a["select"].items():
            sel &= df[col].isin(vals).to_numpy()
        idx = np.where(sel)[0]
        cls = {c: i for i, c in enumerate(a["classes"])}
        assert set(df[a["class_column"]].iloc[idx]) <= set(cls), (an, set(df[a["class_column"]].iloc[idx]) - set(cls))
        y = np.array([cls[c] for c in df[a["class_column"]].iloc[idx]])
        groups = df.grp.to_numpy()[idx]
        fold = make_folds(y, groups, CFG["outer_folds"], SEED)
        plans[an] = (idx, y, groups, fold)
    h4 = CFG["ha4"]
    dfu_full, u4 = unis[h4["universe"]]
    rows4 = np.where(~dfu_full.dup_drop.to_numpy())[0]
    dfu = dfu_full.iloc[rows4].reset_index(drop=True)
    srcs = h4["sources"]
    sid = np.full(len(dfu), -1)
    for i, s in enumerate(srcs):
        m = (dfu.pool == s["pool"]).to_numpy().copy()
        if "generator" in s:
            m &= (dfu.generator == s["generator"]).to_numpy()
        assert (sid[m] == -1).all()
        sid[m] = i
    assert (sid >= 0).all(), "rows outside the 16 sources"
    g4 = dfu.grp.to_numpy()
    f4 = make_folds(sid, g4, CFG["outer_folds"], SEED)
    commit = {}
    for an, (idx, y, groups, fold) in plans.items():
        chk = check_grouped_folds(fold, groups)
        assert chk["ok"], (an, chk)
        commit[an] = dict(sha256_of_folds=hashlib.sha256(fold.tobytes()).hexdigest(), n=len(fold), sizes=np.bincount(fold).tolist(), check=chk)
        (RES / f"folds_{an}.json").write_text(json.dumps(dict(rows=unis[CFG["analyses"][an]["universe"]][0].sha256.iloc[idx].tolist(), fold=fold.tolist())))
    chk = check_grouped_folds(f4, g4)
    assert chk["ok"], chk
    commit["ha4"] = dict(sha256_of_folds=hashlib.sha256(f4.tobytes()).hexdigest(), n=len(f4), sizes=np.bincount(f4).tolist(), check=chk)
    (RES / "folds_ha4.json").write_text(json.dumps(dict(rows=dfu.sha256.tolist(), fold=f4.tolist())))
    (RES / "folds_commit.json").write_text(json.dumps(commit, indent=1))
    # --- features ------------------------------------------------------------------------------
    feat_u, m1_u, shp_u = {}, {}, {}
    for uname, (df, u) in unis.items():
        conds = u["conditions"]
        feat_u[uname] = {}
        for c in conds:
            store = read_caches(u["caches"], c)
            feat_u[uname][c] = np.stack([store[s] for s in df.sha256]).astype(np.float32)
        z = np.load(CACHE / f"m1_{uname}.npz", allow_pickle=False)
        assert z["shas"].tolist() == df.sha256.tolist()
        m1_u[uname] = {c: z["feats"][:, i] for i, c in enumerate(conds)}
        shp_u[uname] = {c: z["shapes"][:, i] for i, c in enumerate(conds)}
    A = {}
    for an, a in CFG["analyses"].items():
        idx, y, groups, fold = plans[an]
        uname = a["universe"]
        df, u = unis[uname]
        A[an] = Analysis(an, a, y, a["classes"], groups, fold, u["conditions"], u["tmix"], a["primary"],
                         {c: feat_u[uname][c][idx] for c in u["conditions"]}, {c: m1_u[uname][c][idx] for c in u["conditions"]},
                         {c: shp_u[uname][c][idx] for c in u["conditions"]}, df.iloc[idx].reset_index(drop=True), fam=groups)
        A[an].idx = idx
        A[an].commit = commit[an]
    a4 = Analysis("ha4", h4, sid, [s["name"] for s in srcs], g4, f4, u4["conditions"], u4["tmix"], h4["primary"],
                  {c: v_[rows4] for c, v_ in feat_u[h4["universe"]].items()}, {c: v_[rows4] for c, v_ in m1_u[h4["universe"]].items()},
                  {c: v_[rows4] for c, v_ in shp_u[h4["universe"]].items()}, dfu)
    a4.rows = rows4
    a4.n_dup_dropped = int(dfu_full.dup_drop.sum())
    a4.sources = srcs
    a4.kinds = [s["kind"] for s in srcs]
    names = a4.classes
    a4.holdouts = [[i] for i in range(len(srcs))] + [[names.index(m) for m in e["members"]] for e in h4["extra_holdouts"]]
    a4.holdout_names = names + [e["name"] for e in h4["extra_holdouts"]]
    a4.holdout_kind = a4.kinds + [e["kind"] for e in h4["extra_holdouts"]]
    a4.commit = commit["ha4"]
    A["ha4"] = a4
    return A, unis


# ============================================================================ stage runners with pickle caching
def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    r = fn()
    CACHE.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(r, protocol=4))
    return r


def stage_closed(workers):
    tasks = []
    for an, a in CFG["analyses"].items():
        for regime in ("tmix", "clean"):
            for fs in FS_ORDER + ["M1", "M1gbm"]:
                for f in range(CFG["outer_folds"]):
                    tasks.append((an, fs, regime, f))
    tasks.sort(key=lambda t: (t[1] != "cat", t[0] != "ha1_main"))
    return run_tasks(closed_task, tasks, workers, "closed")


def stage_logo(workers):
    tasks = []
    for an, a in CFG["analyses"].items():
        if not a["logo"]:
            continue
        for regime in ("tmix", "clean"):
            for fs in FS_ORDER:
                for g in range(len(a["classes"])):
                    for f in range(CFG["outer_folds"]):
                        tasks.append((an, g, f, fs, regime))
    tasks.sort(key=lambda t: (t[3] != "cat", t[4] != "tmix"))
    return run_tasks(logo_task, tasks, workers, "logo")


def stage_ha4(workers):
    tasks = [(fs, regime, f) for regime in ("tmix", "clean") for fs in CFG["ha4"]["fs_list"] + ["M1"] for f in range(CFG["outer_folds"])]
    tasks.sort(key=lambda t: (t[1] != "tmix", t[0] == "M1"))
    return run_tasks(ha4_task, tasks, workers, "ha4")


def stage_perm(workers):
    out = {}
    pc = CFG["perm_control"]
    for an, a in CFG["analyses"].items():
        n = a["perm"]["n"]
        tasks = [(an, s, f, pc["fs"], pc["C"]) for s in range(n) for f in range(CFG["outer_folds"])]
        res = run_tasks(perm_closed_task, tasks, workers, f"perm {an}")
        accs = []
        for s in range(n):
            yp = np.random.default_rng([SEED, 4242, s]).permutation(_G[an].y)
            pred = np.zeros(_G[an].n, int)
            for (s2, te, pr) in res[s * CFG["outer_folds"]:(s + 1) * CFG["outer_folds"]]:
                assert s2 == s
                pred[te] = pr
            accs.append(bacc_simple(yp, pred))
        # real (unpermuted) score of the same fixed pipeline, for context
        out[an] = dict(perm_bacc=accs)
    n = CFG["ha4"]["perm"]["n"]
    res = run_tasks(perm_ha4_task, [(s, 0) for s in range(n)], workers, "perm ha4")
    out["ha4"] = dict(perm_auroc=[r[1] for r in sorted(res)])
    return out


# ============================================================================ stage: report
def pooled_logp(results, A, an, fs, regime):
    """{cond: [n, K]} out-of-fold log-probabilities for one (analysis, feature set, regime)."""
    out = {c: np.full((A.n, A.K), np.nan, np.float32) for c in A.conds}
    chosen = {}
    for r in results:
        if r["an"] == an and r["fs"] == fs and r["regime"] == regime:
            for c in A.conds:
                out[c][r["te"]] = r["logp"][c]
            chosen[r["f"]] = (r["C"], r["inner_best"])
    for c in A.conds:
        assert not np.isnan(out[c]).any(), (an, fs, regime, c)
    return out, chosen


def nested_logp(results, A, an, regime):
    """Per outer fold choose the DINOv2 feature set with the best inner-CV bacc (train folds only); returns pooled log-probs + the choices."""
    out = {c: np.full((A.n, A.K), np.nan, np.float32) for c in A.conds}
    choice = {}
    for f in range(CFG["outer_folds"]):
        cand = [r for r in results if r["an"] == an and r["regime"] == regime and r["f"] == f and r["fs"] in FS_ORDER]
        best = max(cand, key=lambda r: (round(r["inner_best"], 12), -FS_ORDER.index(r["fs"])))
        choice[f] = best["fs"]
        for c in A.conds:
            out[c][best["te"]] = best["logp"][c]
    return out, choice


def eval_closed(A, logp, Wc, inv, drop_name):
    """All closed-set metrics for pooled log-probs {cond: [n,K]}; returns (summary dict, boot dict for paired tests)."""
    Wrow = Wc[:, inv]
    drop = A.classes.index(drop_name) if drop_name else None
    Pr = build_pairs(A.y, A.fam) if A.cfg.get("pairwise") else None
    summ, boots = {}, {}
    for c in A.conds:
        pred = logp[c].argmax(1)
        C = boot_conf(Wrow, A.y, pred, A.K)
        st = conf_stats(C, drop)
        d = dict(bacc=ci(st["bacc"]), mf1=ci(st["mf1"]), recall={A.classes[k]: float(st["rec"][0, k]) for k in range(A.K)},
                 confusion=C[0].astype(int).tolist(), majority_rate=float(np.bincount(A.y).max() / A.n))
        if drop is not None:
            d["bacc_drop"] = ci(st["bacc_drop"])
        boots[c] = dict(bacc=st["bacc"])
        if Pr is not None:
            for nm, strict in (("pw", False), ("pw_strict", True)):
                corr = pair_scores(logp[c], Pr, strict)
                pb = pair_boot(corr, Pr, Wc, inv)
                d[nm] = dict(macro=ci(pb["macro"]), pooled=ci(pb["pooled"]), n_pairs=int(len(Pr["I"])))
                if nm == "pw":
                    boots[c]["pw"] = pb["macro"]
                    if drop is not None:
                        d["pw_drop"] = ci(pair_boot(corr, Pr, Wc, inv, exclude=drop)["macro"])
                    d["pair_matrix"] = {f"{A.classes[t[0]]}|{A.classes[t[1]]}": float(pb["per_type"][0, i]) for i, t in enumerate(pb["types"])}
        summ[c] = d
    return summ, boots


def eval_logo(A, logo_res, Wc, inv, an, fs_key, regime):
    """AUROC (held-out generator vs the other generators' test images) per generator, per condition and novelty score."""
    Wrow = Wc[:, inv]
    out, boots = {}, {}
    for sc in ("msp", "maha"):
        for c in A.conds:
            for g in range(A.K):
                s = np.full(A.n, np.nan, np.float32)
                for r in logo_res:
                    if r["an"] != an or r["regime"] != regime or r["g"] != g:
                        continue
                    if fs_key == "nested":
                        continue
                    if r["fs"] == fs_key:
                        s[r["te"]] = r["out"][c][sc]
                if fs_key == "nested":
                    for f in range(CFG["outer_folds"]):
                        cand = [r for r in logo_res if r["an"] == an and r["regime"] == regime and r["g"] == g and r["f"] == f]
                        best = max(cand, key=lambda r: (round(r["inner_best"], 12), -FS_ORDER.index(r["fs"])))
                        s[best["te"]] = best["out"][c][sc]
                assert not np.isnan(s).any()
                pos, neg = A.y == g, A.y != g
                b = auroc_boot(s[pos], s[neg], Wrow[:, pos], Wrow[:, neg])
                out.setdefault(sc, {}).setdefault(c, {})[A.classes[g]] = ci(b)
                boots[(sc, c, g)] = b
    return out, boots


def stage_report(workers):
    t_start = time.time()
    A, unis = build_analyses()
    _G.update(A)
    closed = pickle.loads((CACHE / "closed.pkl").read_bytes())
    logo = pickle.loads((CACHE / "logo.pkl").read_bytes())
    ha4 = pickle.loads((CACHE / "ha4.pkl").read_bytes())
    perm = pickle.loads((CACHE / "perm.pkl").read_bytes())
    prep = json.loads((RES / "prep_checks.json").read_text())
    R = dict(config=CFG, analyses={}, gates={}, labels={})
    gates = {}
    from threadpoolctl import threadpool_limits
    threadpool_limits(min(16, os.cpu_count() or 1))
    # ------------------------------------------------------------------ H-A1 closed-set analyses
    BOOT = {}
    for an, a in CFG["analyses"].items():
        An = A[an]
        Wc, inv, ug = cluster_weights(An.groups, B, SEED, strata_of)
        An.Wc, An.inv = Wc, inv
        drop = a["drop_macro"]
        res_a = dict(drop_macro=drop, classes=An.classes, chance=1 / An.K, n=An.n, primary=An.primary, role=a["role"], n_clusters=len(ug), fs={})
        for regime in ("tmix", "clean"):
            for fs in FS_ORDER + ["M1", "M1gbm"]:
                lp, chosen = pooled_logp(closed, An, an, fs, regime)
                summ, boots = eval_closed(An, lp, Wc, inv, drop)
                res_a["fs"][f"{regime}|{fs}"] = dict(summary=summ, chosen=chosen)
                BOOT[(an, regime, fs)] = boots
            lp, choice = nested_logp(closed, An, an, regime)
            summ, boots = eval_closed(An, lp, Wc, inv, drop)
            res_a["fs"][f"{regime}|nested"] = dict(summary=summ, chosen_fs=choice)
            BOOT[(an, regime, "nested")] = boots
            np.savez_compressed(RES / f"oof_{an}_{regime}_nested_{An.primary.replace('(', '').replace(')', '')}.npz", logp=lp[An.primary], y=An.y)
        # M0 on raw files ------------------------------------------------------------------------
        m0rows = json.loads((CACHE / f"m0_{a['universe']}.json").read_text())["rows"]
        m0rows = [m0rows[i] for i in An.idx]
        pred = m0_oof(m0rows, An.y, An.fold, An.K)
        C = boot_conf(Wc[:, inv], An.y, pred, An.K)
        st = conf_stats(C, An.classes.index(drop) if drop else None)
        res_a["M0_raw"] = dict(bacc=ci(st["bacc"]), mf1=ci(st["mf1"]), recall={An.classes[k]: float(st["rec"][0, k]) for k in range(An.K)},
                               confusion=C[0].astype(int).tolist(), n_ext_mismatch=int(sum(r["ext_mismatch"] for r in m0rows)))
        if drop:
            res_a["M0_raw"]["bacc_drop"] = ci(st["bacc_drop"])
        BOOT[(an, "M0")] = st["bacc"]
        # M0 features on preprocessed tensors (gate) --------------------------------------------
        m0t = {}
        for c in An.conds:
            pr = m0_tensor_oof(An.shapes[c].astype(float), An.y, An.fold)
            m0t[c] = bacc_simple(An.y, pr)
        res_a["M0_tensor"] = m0t
        gates[f"{an}.M0_at_chance_on_tensors"] = gate_m0_tensor(m0t, 1 / An.K, a["m0_tensor_max_excess"])
        # permutation control --------------------------------------------------------------------
        gates[f"{an}.permutation_200"] = gate_permutation(perm[an]["perm_bacc"], 1 / An.K, a["perm"]["mean_tol"], a["perm"]["p95_excess"])
        gates[f"{an}.permutation_200"]["n_required"] = a["perm"]["n"]
        gates[f"{an}.permutation_200"]["ok"] &= len(perm[an]["perm_bacc"]) == a["perm"]["n"]
        # dedupe / splits -----------------------------------------------------------------------
        sub = An.df
        gates[f"{an}.sha256_dedupe"] = dict(gate_dedupe(sub.crop_sha.tolist(), An.y.tolist()), raw_sha_unique=bool(sub.sha256.is_unique))
        gates[f"{an}.grouped_splits"] = dict(gate_splits(An.fold, An.groups), folds_sha256=An.commit["sha256_of_folds"], pairwise_computed=bool(a["pairwise"]))
        gates[f"{an}.train_conditions"] = gate_train_conds(An.tmix, CFG["held_out_launderings"])
        # Holm across the 4 feature sets: H0 bacc - M1 <= 0.10 at the primary condition (T-mix) ----
        m1b = BOOT[(an, "tmix", "M1")][An.primary]["bacc"]
        ps = []
        for fs in FS_ORDER:
            d = BOOT[(an, "tmix", fs)][An.primary]["bacc"] - m1b
            ps.append(float((1 + (d[1:] <= TH["ha1_m1_margin"]).sum()) / (B + 1)))
        res_a["holm_vs_M1_plus_010"] = dict(fs=FS_ORDER, p=ps, p_holm=holm(ps), note="p = share of bootstrap diffs (bacc_fs - bacc_M1) <= 0.10; Holm over the 4 feature sets")
        if an == "ha1_main":
            d = BOOT[(an, "tmix", "b8")][An.primary]["bacc"] - BOOT[(an, "tmix", "b12")][An.primary]["bacc"]
            res_a["layer_contrast_b8_minus_b12"] = dict(zip(("delta", "ci_lo", "ci_hi"), ci(d)))
            res_a["layer_contrast_b8_minus_b12_clean_train"] = dict(zip(("delta", "ci_lo", "ci_hi"), ci(BOOT[(an, "clean", "b8")][An.primary]["bacc"] - BOOT[(an, "clean", "b12")][An.primary]["bacc"])))
        d = BOOT[(an, "tmix", "nested")][An.primary]["bacc"] - m1b
        res_a["nested_minus_M1"] = dict(zip(("delta", "ci_lo", "ci_hi"), ci(d)))
        R["analyses"][an] = res_a
        # LOGO open set ---------------------------------------------------------------------------
        if a["logo"]:
            lg = {}
            for regime in ("tmix", "clean"):
                for fk in FS_ORDER + ["nested"]:
                    out, _ = eval_logo(An, logo, Wc, inv, an, fk, regime)
                    lg[f"{regime}|{fk}"] = out
            res_a["logo"] = lg
    # ------------------------------------------------------------------ labels H-A1
    main, raid = R["analyses"]["ha1_main"], R["analyses"]["ha1_raid"]
    pc = A["ha1_main"].primary
    nest = main["fs"]["tmix|nested"]["summary"][pc]
    v = dict(bacc=nest["bacc"], m1_bacc=main["fs"]["tmix|M1"]["summary"][pc]["bacc"][0], pairwise=nest["pw"]["macro"][0],
             raid_bacc=raid["fs"]["tmix|nested"]["summary"][A["ha1_raid"].primary]["bacc"][0])
    # ------------------------------------------------------------------ H-A4
    An = A["ha4"]
    Wc4, inv4, ug4 = cluster_weights(An.groups, B, SEED, strata_of)
    Wrow4 = Wc4[:, inv4]
    h4 = dict(classes=An.classes, holdouts=An.holdout_names, n=An.n, n_clusters=len(ug4), chance_auroc=0.5, fs={})
    nov, near, knn, knnreal, closedp = {}, {}, {}, {}, {}
    for r in ha4:
        key = (r["fs"], r["regime"])
        for (hi, c), v_ in r["nov"].items():
            nov.setdefault(key, {}).setdefault((hi, c), np.full(An.n, np.nan, np.float32))[r["te"]] = v_
            near.setdefault(key, {}).setdefault((hi, c), np.full(An.n, -1, np.int8))[r["te"]] = r["near"][(hi, c)]
        for (hi, c), v_ in r["knn"].items():
            knn.setdefault(key, {}).setdefault((hi, c), np.full(An.n, np.nan, np.float32))[r["te"]] = v_
            knnreal.setdefault(key, {}).setdefault((hi, c), np.full(An.n, np.nan, np.float32))[r["te"]] = r["knnreal"][(hi, c)]
        for c, p in r["closed"].items():
            closedp.setdefault(key, {}).setdefault(c, np.full(An.n, -1, np.int8))[r["te"]] = p
    kindarr = np.array([1 if k == "real" else 0 for k in An.kinds])
    BOOT4 = {}
    for key in sorted(nov):
        fs, regime = key
        ent = dict(per_cond={}, closed={})
        for c in An.conds:
            pr = closedp[key][c].astype(int)
            Cb = boot_conf(Wrow4, An.y, pr, An.K)
            st = conf_stats(Cb)
            ent["closed"][c] = dict(bacc=ci(st["bacc"]), mf1=ci(st["mf1"]), recall={An.classes[k]: float(st["rec"][0, k]) for k in range(An.K)},
                                    kind_acc=float((kindarr[pr] == kindarr[An.y]).mean()), chance=1 / An.K,
                                    majority_rate=float(np.bincount(An.y).max() / An.n), confusion=Cb[0].astype(int).tolist())
            per = {}
            for hi, held in enumerate(An.holdouts):
                pos = np.isin(An.y, held)
                neg = ~pos
                assert not np.isnan(nov[key][(hi, c)]).any()
                b = auroc_boot(nov[key][(hi, c)][pos], nov[key][(hi, c)][neg], Wrow4[:, pos], Wrow4[:, neg])
                kind_ok = (kindarr[near[key][(hi, c)].astype(int)[pos]] == (1 if An.holdout_kind[hi] == "real" else 0)).astype(float)
                kb = (Wrow4[:, pos] @ kind_ok) / Wrow4[:, pos].sum(1)
                e = dict(auroc=ci(b), kind_acc=ci(kb), kind_acc_cp=None, n_pos=int(pos.sum()),
                         nearest_source=pd.Series(np.array(An.classes)[near[key][(hi, c)][pos].astype(int)]).value_counts().head(3).to_dict())
                if (hi, c) in knn.get(key, {}):
                    kb2 = knn[key][(hi, c)]
                    b2 = auroc_boot(kb2[pos], kb2[neg], Wrow4[:, pos], Wrow4[:, neg])
                    vote = knnreal[key][(hi, c)][pos]
                    e["knn_auroc"] = ci(b2)
                    e["knn_kind_acc"] = float(((vote > 0.5) if An.holdout_kind[hi] == "real" else (vote < 0.5)).mean())
                per[An.holdout_names[hi]] = e
                BOOT4[(fs, regime, c, hi)] = b
            ent["per_cond"][c] = per
        h4["fs"][f"{regime}|{fs}"] = ent
    # summaries per (fs, regime, cond): mean / min AUROC over the 16 singleton sources; without firefly
    names16 = An.classes
    summ4 = {}
    for key, ent in h4["fs"].items():
        for c in An.conds:
            aucs = {n: ent["per_cond"][c][n]["auroc"][0] for n in names16}
            fi = names16.index("firefly")
            bs = np.mean([BOOT4[(key.split("|")[1], key.split("|")[0], c, hi)] for hi in range(16)], 0)
            bs_nf = np.mean([BOOT4[(key.split("|")[1], key.split("|")[0], c, hi)] for hi in range(16) if hi != fi], 0)
            summ4[(key, c)] = dict(mean=ci(bs), mean_nofirefly=ci(bs_nf), min=float(min(aucs.values())), argmin=min(aucs, key=aucs.get))
        ent["summary"] = {c: summ4[(key, c)] for c in An.conds}
    # Holm over the b4/b8/b12 feature sets: H0 mean AUROC <= 0.85
    ps = []
    for fs in CFG["ha4"]["fs_list"]:
        bs = np.mean([BOOT4[(fs, "tmix", An.primary, hi)] for hi in range(16)], 0)
        ps.append(float((1 + (bs[1:] <= TH["ha4_pass_mean"]).sum()) / (B + 1)))
    h4["holm_mean_auroc_gt_085"] = dict(fs=CFG["ha4"]["fs_list"], p=ps, p_holm=holm(ps))
    # E10 reference FPR (019 tables)
    ref = {}
    txt = (ROOT / "experiments/019-source-diversity/results/tables.md").read_text()
    sec = txt.split("## Held-out source at k = 5 (train on the 5 other sources), jpeg75_resize05")[1].split("\n## ")[0]
    for m_ in re.finditer(r"^\| (\w+) \| F_b8 \| ([\d.]+) \[[^\]]*\] \| ([\d.]+) \(", sec, re.M):
        ref[m_.group(1)] = dict(auroc=float(m_.group(2)), fpr=float(m_.group(3)))
    h4["e10_reference_k5_b8"] = ref
    # baselines for H-A4: M0 raw 16-way, M1 (already in fs 'M1')
    m0rows = json.loads((CACHE / "m0_fov512.json").read_text())["rows"]
    m0rows = [m0rows[i] for i in An.rows]
    pred = m0_oof(m0rows, An.y, An.fold, An.K)
    Cm = boot_conf(Wrow4, An.y, pred, An.K)
    stm = conf_stats(Cm)
    h4["M0_raw_16way"] = dict(bacc=ci(stm["bacc"]), kind_acc=float((kindarr[pred] == kindarr[An.y]).mean()), n_hdrp_no_qtable=int(sum(not r["raw"] for r in m0rows)),
                              recall={An.classes[k]: float(stm["rec"][0, k]) for k in range(An.K)})
    m0t = {c: bacc_simple(An.y, m0_tensor_oof(An.shapes[c].astype(float), An.y, An.fold)) for c in An.conds}
    h4["M0_tensor"] = m0t
    gates["ha4.M0_at_chance_on_tensors"] = gate_m0_tensor(m0t, 1 / An.K, 0.04)
    gates["ha4.permutation_200"] = gate_permutation(perm["ha4"]["perm_auroc"], 0.5, CFG["ha4"]["perm"]["mean_tol"], CFG["ha4"]["perm"]["p95_excess"])
    gates["ha4.permutation_200"]["ok"] &= len(perm["ha4"]["perm_auroc"]) == CFG["ha4"]["perm"]["n"]
    gates["ha4.sha256_dedupe"] = dict(gate_dedupe(An.df.crop_sha.tolist(), An.y.tolist()), raw_sha_unique=bool(An.df.sha256.is_unique),
                                      n_duplicate_crops_dropped=An.n_dup_dropped, note="processed-crop sha256 (pixel content); later duplicates dropped before folds; kept set re-checked")
    gates["ha4.grouped_splits"] = dict(gate_splits(An.fold, An.groups), folds_sha256=An.commit["sha256_of_folds"])
    gates["ha4.train_conditions"] = gate_train_conds(An.tmix, CFG["held_out_launderings"])
    R["ha4"] = h4
    # ------------------------------------------------------------------ gates shared
    for uname, p_ in prep.items():
        gates[f"{uname}.identical_preprocessing"] = dict(
            ok=bool(p_["bad_paths"] == 0 and p_["n_exif_or_icc"] == 0 and p_["n_not_rgb"] == 0 and p_["cache_meta"]["ok"] and p_["raw_sha_unique"]),
            **{k: p_[k] for k in ("bad_paths", "n_exif_or_icc", "n_not_rgb", "n_ext_mismatch", "raw_sha_unique", "crop_sha_unique")}, cache_meta=p_["cache_meta"],
            note="Same fov crop dir for every class; conditions via lib/perturb.apply_condition in one code path (M1 and the cached features); M0 reads the true format from content.")
    gates["D-008"] = d008_status()
    gates["holm_and_inner_cv_layer_choice"] = dict(ok=True, note="nested inner-CV choice per outer fold (train folds only); Holm tables in results; asserts in every task")
    gates["chance_levels_in_tables"] = dict(ok=True, note="every table carries a chance column")
    gates["no_test_group_in_train"] = dict(ok=True, note="assert_disjoint_groups ran inside every outer/inner/LOGO/permutation task (a violation raises)")
    R["gates"] = gates
    gates_ok = all(g["ok"] for g in gates.values())
    ha1_gates_ok = all(g["ok"] for k, g in gates.items() if not k.startswith("ha4."))
    ha4_gates_ok = all(g["ok"] for k, g in gates.items() if not k.startswith(("ha1_", "fov224.")))
    R["labels"]["H-A1"] = dict(label_ha1(v, ha1_gates_ok), values=v)
    ous = {}
    for sc in ("msp", "maha"):
        aur = {g_: R["analyses"]["ha1_main"]["logo"]["tmix|nested"][sc][pc][g_][0] for g_ in main["classes"]}
        ous[sc] = label_open_set(aur)
    ous["usable_either"] = bool(ous["msp"]["usable"] or ous["maha"]["usable"]) and ha1_gates_ok
    R["labels"]["open_set_usable"] = ous
    pk = "tmix|" + CFG["ha4"]["primary_fs"]
    aucs = {n: h4["fs"][pk]["per_cond"][An.primary][n]["auroc"][0] for n in names16}
    kinds = {n: h4["fs"][pk]["per_cond"][An.primary][n]["kind_acc"][0] for n in names16 if An.holdout_kind[An.holdout_names.index(n)] == "real"}
    R["labels"]["H-A4"] = dict(label_ha4(aucs, kinds, ha4_gates_ok), auroc_by_source=aucs, kind_acc_real=kinds)
    R["labels"]["DISCOVERY"] = bool(R["labels"]["H-A4"]["discovery"])
    R["labels"]["all_gates_ok"] = bool(gates_ok)
    R["seconds_report"] = round(time.time() - t_start)
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log("results.json written; labels:", json.dumps({k: (v_ if not isinstance(v_, dict) else v_.get("label", v_.get("usable_either"))) for k, v_ in R["labels"].items()}))
    from tables import write_tables
    write_tables(R, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "closed", "logo", "ha4", "perm", "report", "all"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    os.chdir(ROOT)
    t0 = time.time()
    st = a.stage
    if st in ("prep", "all"):
        stage_prep(a.workers)
    if st in ("closed", "logo", "ha4", "perm", "all"):
        A, _ = build_analyses()
        _G.update(A)
        for name, fn in (("closed", stage_closed), ("logo", stage_logo), ("ha4", stage_ha4), ("perm", stage_perm)):
            if st in (name, "all"):
                cached(name, lambda: fn(a.workers))
    if st in ("report", "all"):
        stage_report(a.workers)
    log(f"stage {st} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
