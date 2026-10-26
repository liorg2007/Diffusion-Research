"""E8 (Amendment 17): embedding geometry and a source-invariant diffusion direction. Resumable stages.

    env/.venv/bin/python -I experiments/017-embedding-geometry/run.py all [--workers 14]
Stages: m1 (C2ST fits) -> m2fit (probe normals) -> m3u (source subspaces) -> m3fit (probes on projected features) -> texture -> analyze -> map -> report
Everything reuses the E7 (016-loso) splits, E6 caption-family roles/groups, cached 004 features, probe recipe (lib/probe.fit_probe, balanced,
grouped CV, T-aug) and cluster-bootstrap Plan. CPU only. Constants in config.json. Pure helper functions (no data access) are at the top so
that selftest.py can test them on synthetic data.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib.util
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(HERE))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import metrics
import probe

CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CACHE, RES = HERE / "cache", HERE / "results"
SOURCES, EVAL, TAUG, JR = CFG["sources"], CFG["eval_conds"], CFG["t_aug"], CFG["launder_cond"]
FS = {k: slice(*v) for k, v in CFG["feature_slices"].items()}
GRID = CFG["m3"]["extra_pcs_grid"]
RD = CFG["readings"]
PAIRS = [("VISION", "RAISE"), ("VISION", "OI"), ("RAISE", "OI"), ("VISION", "FAKE"), ("RAISE", "FAKE"), ("OI", "FAKE")]


def log(*a):
    print(time.strftime("%H:%M:%S"), "E8", *a, flush=True)


# ============================================================================ pure helpers (synthetic-testable)
def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def probe_normal_raw(model):
    """Normal of a fitted pipeline [StandardScaler, LogisticRegression] in RAW feature space (logit = w . x + b)."""
    sc, clf = model[0], model[-1]
    return clf.coef_[0] / sc.scale_


def to_common(w_raw, sigma):
    """Express a raw-space normal in the standardized coordinates z = (x - mu) / sigma of a common scaler."""
    return w_raw * sigma


def safe_sigma(X):
    s = X.std(0)
    s[s == 0] = 1.0
    return s


def twin_direction(Xfake, Xreal_of_fake):
    """d_twin (raw space) = mean over (fake, its real twin) pairs of x_fake - x_real."""
    return (np.asarray(Xfake, float) - np.asarray(Xreal_of_fake, float)).mean(0)


def source_subspace(Z, src, cl, n_extra):
    """Orthonormal source-nuisance basis U [d, k], k = 2 (n_sources - 1) + n_extra (rank permitting), estimated from TRAINING REALS only.
    Columns, in order: (1) the (n_sources-1) shrinkage-LDA directions between the source means; (2) the (n_sources-1) between-source
    mean-difference directions (the rank-(n_sources-1) PCA of the source means; needed because an LDA direction Sigma^-1 dmu is not dmu when
    the within-source covariance is anisotropic); (3) the top `n_extra` principal directions of the source-balanced matrix of cluster means
    (cluster = device / caption family / image, centred on the balanced grand mean), taken in the orthogonal complement of (1)+(2).
    Z: [n, d] standardized features; src: [n] source labels; cl: [n] cluster ids."""
    src, cl = np.asarray(src).astype(str), np.asarray(cl).astype(str)
    srcs = sorted(set(src.tolist()))
    ns = len(srcs)
    Q = np.zeros((Z.shape[1], 0))

    def add(Q, V, n):
        V = V - (V @ Q) @ Q.T if Q.shape[1] else V
        _, sv, Vt = np.linalg.svd(V, full_matrices=False)
        keep = min(n, int((sv > 1e-8 * max(sv[0], 1e-300)).sum()))
        return np.concatenate([Q, Vt[:keep].T], axis=1)

    if ns > 1:
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=ns - 1).fit(Z, src)
        W = lda.scalings_[:, :ns - 1]
        Q, _ = np.linalg.qr(W)
        Msrc = np.stack([Z[src == s].mean(0) for s in srcs])
        Q = add(Q, Msrc - Msrc.mean(0), ns - 1)
    if n_extra > 0:
        keys = np.array([f"{s}|{c}" for s, c in zip(src, cl)])
        uk, code = np.unique(keys, return_inverse=True)
        M = np.zeros((len(uk), Z.shape[1]))
        np.add.at(M, code, Z)
        M /= np.bincount(code)[:, None]
        us = np.array([k.split("|")[0] for k in uk])
        w = np.zeros(len(uk))
        for s in srcs:
            m = us == s
            w[m] = 1.0 / (ns * m.sum())
        mu = (w[:, None] * M).sum(0)
        Q = add(Q, np.sqrt(w)[:, None] * (M - mu), n_extra)
    return Q


def project_out(Z, U):
    return Z - (Z @ U) @ U.T if U.shape[1] else Z


def kdim(n_train_sources, j):
    """Removed dimensions for j extra PCs: (n-1) LDA + (n-1) mean-difference + j."""
    return 2 * (n_train_sources - 1) + j


def choose_k(aucs, tol):
    """Smallest k (dict k -> calib source-ID AUROC) whose AUROC is within `tol` of the minimum over the grid."""
    m = min(aucs.values())
    return min(k for k, v in aucs.items() if v <= m + tol)


def r3_label(per):
    """per[S] = dict(auc, fpr, auc7, fpr7). PASS-R3: every source AUROC >= 0.85 and FPR <= 5%. KILL: no source improves by >= 0.03 AUROC
    or >= 5 points of FPR over E7. Else INCONCLUSIVE."""
    ok = all(v["auc"] >= RD["R3_auroc_min"] and v["fpr"] <= RD["R3_fpr_max"] for v in per.values())
    imp = {S: (v["auc"] - v["auc7"] >= RD["kill_auroc_gain"]) or (v["fpr7"] - v["fpr"] >= RD["kill_fpr_gain"]) for S, v in per.items()}
    lab = "PASS-R3" if ok else "KILL" if not any(imp.values()) else "INCONCLUSIVE"
    return dict(label=lab, all_ok=bool(ok), improved=imp)


# ============================================================================ E7 plumbing
_E7 = {}


def e7():
    if "m" not in _E7:
        spec = importlib.util.spec_from_file_location("e7run", ROOT / "experiments" / "016-loso" / "run.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["e7run"] = m
        spec.loader.exec_module(m)
        _E7["m"] = m
    return _E7["m"]


class Arm:
    """E7 splits/roles/groups of one arm + the cached features of the three T-aug conditions (4608-d)."""

    def __init__(self, arm):
        E = e7()
        self.arm = arm
        self.D = E.Data(arm, load_feats=False)
        self.X = {c: E.load_x(arm, self.D.df, c) for c in TAUG}
        self.plan = None

    def src_pools(self, S):
        return e7().SRC[S]["train_pools"]


_A = {}


def get_arm(arm):
    if arm not in _A:
        _A[arm] = Arm(arm)
    return _A[arm]


def taug_stack(A, rows, sl):
    X = np.concatenate([A.X[c][rows][:, sl] for c in TAUG])
    y = np.concatenate([A.D.lab[rows]] * len(TAUG))
    g = np.concatenate([A.D.groups[rows]] * len(TAUG))
    return X, y, g


def global_weights(D, n_boot=B, seed=SEED):
    """Cluster-bootstrap multiplicities [B+1, N] over ALL rows; clusters = D.groups (device / caption family / image), drawn within kind."""
    keys = np.asarray(D.groups)
    kinds = np.array([k.split(":")[0] for k in keys])
    rng = np.random.default_rng(seed)
    W = np.ones((n_boot + 1, len(keys)))
    for kind in ("dev", "rid", "img"):
        m = np.where(kinds == kind)[0]
        if not len(m):
            continue
        uk, code = np.unique(keys[m], return_inverse=True)
        K = len(uk)
        cnt = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)]).astype(float)
        W[1:, m] = cnt[:, code]
    return W


def run_tasks(fn, tasks, workers):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, out in enumerate(ex.map(fn, tasks)):
            log(f"  [{i + 1}/{len(tasks)}] {out}")
    log(f"{len(tasks)} tasks in {time.time() - t0:.0f}s")


# ============================================================================ M1: C2ST
def m1_rows(A, name):
    D = A.D
    if name == "FAKE":
        return np.where(D.lab == 1)[0]
    return np.where(np.isin(D.pool, A.src_pools(name)))[0]


def m1_path(arm, feat, cond, pair, seed, sub):
    return CACHE / "m1" / f"{arm}__{feat}__{cond}__{pair[0]}-{pair[1]}__s{seed}__m{sub}.npz"


def m1_tasks():
    tasks = []
    for arm in CFG["arms"]:
        for feat, fc in CFG["m1"]["feats"].items():
            for cond in EVAL:
                for pair in PAIRS:
                    for s in range(fc["seeds"]):
                        tasks.append((arm, feat, cond, pair, s, -1))
                    for m in range(fc["matched_subsamples"]):
                        tasks.append((arm, feat, cond, pair, 0, m))
    return [t for t in tasks if not m1_path(*t).exists()]


def m1_task(args):
    arm, feat, cond, pair, seed, sub = args
    t0 = time.time()
    A = get_arm(arm)
    ra, rb = m1_rows(A, pair[0]), m1_rows(A, pair[1])
    if sub >= 0:
        rng = np.random.default_rng([SEED, 100 + sub, PAIRS.index(pair)])
        n = CFG["m1"]["matched_n"]
        ra, rb = np.sort(rng.choice(ra, n, replace=False)), np.sort(rng.choice(rb, n, replace=False))
    rows = np.concatenate([ra, rb])
    y = np.r_[np.zeros(len(ra), int), np.ones(len(rb), int)]
    X = A.X[cond][rows][:, FS[feat]]
    g = A.D.groups[rows]
    oof = np.zeros(len(rows))
    cv = StratifiedGroupKFold(CFG["m1"]["n_splits"], shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, g):
        m = probe.fit_probe(X[tr], y[tr], SEED, groups=g[tr], class_weight="balanced")
        oof[te] = m.decision_function(X[te])
    p = m1_path(*args)
    p.parent.mkdir(exist_ok=True)
    np.savez(p, rows=rows, y=y, oof=oof)
    return f"m1 {p.stem} auc={roc_auc_score(y, oof):.3f} ({time.time() - t0:.0f}s)"


def m1_analyze(arm):
    A = get_arm(arm)
    W = global_weights(A.D)
    out = {}
    for feat, fc in CFG["m1"]["feats"].items():
        for cond in EVAL:
            for mode in ("full", "matched"):
                n_rep = fc["seeds"] if mode == "full" else fc["matched_subsamples"]
                if n_rep == 0:
                    continue
                pa = {}
                for pair in PAIRS:
                    arr = []
                    for r in range(n_rep):
                        z = np.load(m1_path(arm, feat, cond, pair, r if mode == "full" else 0, -1 if mode == "full" else r))
                        rows, y, s = z["rows"], z["y"], z["oof"]
                        arr.append(metrics.weighted_auroc(s[y == 0], s[y == 1], W[:, rows[y == 0]], W[:, rows[y == 1]]))
                    pa["-".join(pair)] = np.mean(arr, 0)
                rr = np.mean([pa["VISION-RAISE"], pa["VISION-OI"], pa["RAISE-OI"]], 0)
                rf = np.mean([pa["VISION-FAKE"], pa["RAISE-FAKE"], pa["OI-FAKE"]], 0)
                ratio = rr / rf
                out[f"{feat}|{cond}|{mode}"] = dict(
                    pairs={k: e7().summ(v) for k, v in pa.items()}, real_real_mean=e7().summ(rr), real_fake_mean=e7().summ(rf),
                    ratio=e7().summ(ratio), R1_entangled=bool(ratio[0] >= RD["R1_ratio"]),
                    ratio_per_pair=dict(zip(["VISION-RAISE", "VISION-OI", "RAISE-OI"],
                                            [float(pa[k][0] / rf[0]) for k in ["VISION-RAISE", "VISION-OI", "RAISE-OI"]])))
    return out


# ============================================================================ M2: directions
def m2_path(arm, feat, S, model):
    return CACHE / "m2" / f"{arm}__{feat}__{S}__{model}.npz"


def m2_tasks():
    E = e7()
    t = []
    for arm in CFG["arms"]:
        for feat in CFG["m2"]["feats"]:
            for S in SOURCES:
                for model in E.model_names(arm):
                    if not m2_path(arm, feat, S, model).exists():
                        t.append((arm, feat, S, model))
    return t


def m2_task(args):
    arm, feat, S, model = args
    t0 = time.time()
    A = get_arm(arm)
    rows = e7().train_rows(A.D, (S,), model)
    X, y, g = taug_stack(A, rows, FS[feat])
    m = probe.fit_probe(X, y, SEED, groups=g, class_weight="balanced")
    p = m2_path(*args)
    p.parent.mkdir(exist_ok=True)
    np.savez(p, w=probe_normal_raw(m), scale=m[0].scale_, C=float(m[-1].C_[0]), n=len(rows))
    return f"m2 {p.stem} n={len(rows)} C={m[-1].C_[0]:.3g} ({time.time() - t0:.0f}s)"


def twin_pairs(A):
    D = A.D
    real = D.idx("train", ["raise"], label=0)
    r2i = {D.rid[i]: i for i in real}
    fk = D.idx("train", None, label=1)
    pr = [(f, r2i[D.rid[f]]) for f in fk if D.rid[f] in r2i]
    return np.array([p[0] for p in pr]), np.array([p[1] for p in pr])


def twin_dir(A, sl, conds, F, R, drop=None):
    keep = np.ones(len(F), bool) if drop is None else A.D.gen[F] != drop
    return np.mean([twin_direction(A.X[c][F[keep]][:, sl], A.X[c][R[keep]][:, sl]) for c in conds], 0)


def subsets_of_gens(arm):
    E = e7()
    gens = E.arm_all_gens(arm)
    sub = {"all": (gens, "full"), f"no {E.FLUX}": ([g for g in gens if g != E.FLUX], "full")}
    for g in gens:
        sub[f"drop {g}"] = ([x for x in gens if x != g], E.model_of(arm, g))
    return sub


class Ctx:
    """Evaluation context of one arm: E7 test rows, bootstrap plan and weights."""

    def __init__(self, arm):
        E = e7()
        self.arm, self.A = arm, get_arm(arm)
        self.D = self.A.D
        self.plan = E.Plan(self.D)
        self.gens = E.arm_all_gens(arm)
        self.names = E.model_names(arm)
        self.Rr = {S: E.src_eval_rows(self.D, S) for S in SOURCES}
        self.Ff = {g: self.D.idx("test", None, gens=[g], label=1) for g in self.gens}
        self.WR = {S: self.plan.w(r) for S, r in self.Rr.items()}
        self.WF = {g: self.plan.w(f) for g, f in self.Ff.items()}


def eval_scores(ctx, S, SC, cal, cidx):
    """SC: model -> [n_conds, N]; cidx: cond -> index. Returns {cond: {ga: {gen: [B+1]}, fp: {reading: {model: {...}}}}} for EVAL conditions.
    Same logic as E7.evaluate_arm (AUROC of S's test reals vs each generator's test fakes under that generator's LOGO model; FPR at the
    1% threshold set on `cal` reals: reading A = clean scores, B = same-condition scores)."""
    E = e7()
    res = {}
    for c in EVAL:
        ci = cidx[c]
        ga = {}
        for g in ctx.gens:
            s = SC[E.model_of(ctx.arm, g)][ci].astype(np.float64)
            ga[g] = metrics.weighted_auroc(s[ctx.Rr[S]], s[ctx.Ff[g]], ctx.WR[S], ctx.WF[g])
        fp = {}
        for rd in ("A", "B"):
            fp[rd] = {}
            for m in ctx.names:
                sm = SC[m]
                thr = E.thr_fpr(sm[cidx["clean"]][cal]) if rd == "A" else E.thr_fpr(sm[ci][cal])
                hit = (sm[ci][ctx.Rr[S]] >= thr).astype(float)
                k, n = int(hit.sum()), len(hit)
                bf = (ctx.WR[S] @ hit) / ctx.WR[S].sum(1)
                fp[rd][m] = dict(thr=float(thr), fpr=k / n, k=k, n=n, cp95=E.clopper_pearson(k, n),
                                 boot_ci=[float(v) for v in np.percentile(bf[1:], [2.5, 97.5])])
        res[c] = dict(ga=ga, fp=fp)
    return res


def ev_summary(ev, ctx, keep=None, fpr_model="full", reading="A"):
    E = e7()
    keep = ctx.gens if keep is None else keep
    out = {}
    for c, r in ev.items():
        f = r["fp"][reading][fpr_model]
        out[c] = dict(auroc=E.summ(E.macro_of(r["ga"], keep)), fpr=f["fpr"], k=f["k"], n=f["n"], cp95=f["cp95"], boot_ci=f["boot_ci"],
                      per_gen={g: float(r["ga"][g][0]) for g in ctx.gens})
    return out


def m2_analyze(arm):
    E = e7()
    ctx = Ctx(arm)
    A, D = ctx.A, ctx.D
    F, R = twin_pairs(A)
    gens_tr = E.arm_gens(arm)
    out = dict(n_twin_pairs=int(len(F)), n_twin_reals=int(len(set(R.tolist()))), n_twin_families=int(len(set(D.rid[F].tolist()))), feats={})
    sub = subsets_of_gens(arm)
    jr_idx = {"clean": 0, "jpeg75_resize05": 1}
    cal_raise = E.calib_rows(D, ["RAISE"])
    for feat in CFG["m2"]["feats"]:
        sl = FS[feat]
        fo = {}
        # common scaler for cosines: pooled training rows of all three sources (reals + all train fakes), T-aug stack
        rows_pool = E.train_rows(D, tuple(SOURCES), "full")
        sig_pool = safe_sigma(taug_stack(A, rows_pool, sl)[0])
        # twin scaler (RAISE train reals + paired fakes only; no held-out source information)
        rows_tw = np.unique(np.concatenate([F, R]))
        Xtw = np.concatenate([A.X[c][rows_tw][:, sl] for c in TAUG])
        mu_t, sig_t = Xtw.mean(0), safe_sigma(Xtw)
        dirs = {}                                                   # subset -> {name: vector in pooled-z space}
        raw = {}
        for nm, (keep, fm) in sub.items():
            drop = None if fm == "full" else fm[2:]
            d = {}
            for S in SOURCES:
                z = np.load(m2_path(arm, feat, S, fm))
                d[f"probe:{S}"] = to_common(z["w"], sig_pool)
            dtw = twin_dir(A, sl, TAUG, F, R, drop)
            d["twin"] = dtw / sig_pool
            dirs[nm] = d
            raw[nm] = {**{f"probe:{S}": np.load(m2_path(arm, feat, S, fm))["w"] for S in SOURCES}, "twin": dtw}
        names = ["probe:VISION", "probe:RAISE", "probe:OI", "twin"]

        def cmat(vs):
            return [[cosine(vs[a], vs[b]) for b in names] for a in names]
        fo["names"] = names
        fo["cos_matrix_pooled_z"] = cmat(dirs["all"])
        fo["cos_matrix_raw"] = cmat(raw["all"])
        # twin direction variants (headline = T-aug mean)
        var = {}
        for vn, conds in (("T-aug", TAUG), ("clean", ["clean"]), ("jpeg75_resize05", ["jpeg75_resize05"])):
            dv = twin_dir(A, sl, conds, F, R) / sig_pool
            var[vn] = {S: cosine(dv, dirs["all"][f"probe:{S}"]) for S in SOURCES}
            var[vn]["twin_vs_Taug"] = cosine(dv, dirs["all"]["twin"])
        fo["twin_variants_cos"] = var
        # robustness over generator subsets
        rob = {}
        for nm in sub:
            c3 = [cosine(dirs[nm][f"probe:{a}"], dirs[nm][f"probe:{b}"]) for a, b in itertools.combinations(SOURCES, 2)]
            rob[nm] = dict(min_cos=float(min(c3)), cos_probe_pairs=dict(zip(["VISION-RAISE", "VISION-OI", "RAISE-OI"], c3)))
        # split-half noise reference (b8 only)
        if feat == "F_b8":
            fo["split_half"] = split_half_reference(A, sl, sig_pool, arm)
        # d_twin projection scores: model m = twin direction without the generator of m
        SC = {}
        for m in ctx.names:
            drop = None if m == "full" else m[2:]
            dz = twin_dir(A, sl, TAUG, F, R, drop) / sig_t
            SC[m] = np.stack([((A.X[c][:, sl] - mu_t) / sig_t) @ dz for c in EVAL]).astype(np.float64) if all(c in A.X for c in EVAL) else None
        evs = {}
        ev_all = {}
        for S in SOURCES:
            ev_all[S] = eval_scores(ctx, S, SC, cal_raise, jr_idx)
        for S in SOURCES:
            evs[S] = {rd: ev_summary(ev_all[S], ctx, reading=rd) for rd in ("A", "B")}
        fo["twin_eval"] = evs
        # R2 labels per subset (cosines from probes trained without g; AUROC macro without g)
        for nm, (keep, fm) in sub.items():
            au = {S: float(e7().macro_of(ev_all[S][JR]["ga"], keep)[0]) for S in CFG["heldout_for_dtwin"]}
            rob[nm]["twin_auroc"] = au
            rob[nm]["R2"] = bool(rob[nm]["min_cos"] >= RD["R2_cos_min"] and all(v >= RD["R2_auroc_min"] for v in au.values()))
            rob[nm]["twin_fpr_A"] = {S: ev_all[S][JR]["fp"]["A"][fm]["fpr"] for S in CFG["heldout_for_dtwin"]}
        fo["robust"] = rob
        fo["R2_main"] = rob["all"]["R2"]
        out["feats"][feat] = fo
    return out


def split_half_reference(A, sl, sig_pool, arm):
    """Noise ceiling for a probe-normal cosine: for each source, fit the same probe (fixed C of the full fit, no CV) on two random halves
    of the groups of its own training rows, cosine in pooled-z space; mean over reps. Half the data, so a conservative reference."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    E = e7()
    res = {}
    for S in SOURCES:
        rows = E.train_rows(A.D, (S,), "full")
        g = A.D.groups[rows]
        ug = np.unique(g)
        C = float(np.load(m2_path(arm, "F_b8", S, "full"))["C"])
        rng = np.random.default_rng(SEED)
        cs = []
        for _ in range(CFG["m2"]["split_half_reps"]):
            perm = rng.permutation(ug)
            ha = np.isin(g, perm[: len(ug) // 2])
            ws = []
            for msk in (ha, ~ha):
                X, y, gg = taug_stack(A, rows[msk], sl)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)).fit(X, y)
                ws.append(to_common(probe_normal_raw(m), sig_pool))
            cs.append(cosine(*ws))
        res[S] = dict(mean=float(np.mean(cs)), min=float(np.min(cs)), max=float(np.max(cs)))
    return res


# ============================================================================ M3: source-nuisance removal
def m3u_path(arm, S):
    return CACHE / "m3" / f"u__{arm}__{S}.npz"


def m3s_path(arm, S, j):
    return CACHE / "m3" / f"scores__{arm}__{S}__j{j}.npz"


def others_of(S):
    return tuple(s for s in SOURCES if s != S)


def m3u_task(args):
    arm, S = args
    t0 = time.time()
    E = e7()
    A = get_arm(arm)
    D, sl = A.D, FS["F_b8"]
    T = others_of(S)
    rows = E.train_rows(D, T, "full")
    rows = rows[D.lab[rows] == 0]                                            # reals only: U cannot depend on any fake
    X, _, g = taug_stack(A, rows, sl)
    mu, sd = X.mean(0), safe_sigma(X)
    Z = (X - mu) / sd
    pool_src = {p: s for s in SOURCES for p in E.SRC[s]["train_pools"]}
    src = np.concatenate([[pool_src[p] for p in D.pool[rows]]] * len(TAUG))
    U = source_subspace(Z, src, g, max(GRID))
    # calib source-ID AUROC per k (training sources only): classifier retrained on projected training reals, scored on calib reals
    cal_rows = np.concatenate([D.idx("calib", E.SRC[s]["train_pools"], label=0) for s in T])
    Zc = (np.concatenate([A.X[c][cal_rows][:, sl] for c in TAUG]) - mu) / sd
    srcc = np.concatenate([[pool_src[p] for p in D.pool[cal_rows]]] * len(TAUG))
    ytr = (src == T[1]).astype(int)
    yca = (srcc == T[1]).astype(int)
    aucs = {}
    for j in [None] + GRID:
        k = 0 if j is None else kdim(len(T), j)
        Uk = U[:, :k]
        m = probe.fit_probe(project_out(Z, Uk), ytr, SEED, groups=g, class_weight="balanced")
        aucs[str(k)] = float(roc_auc_score(yca, m.decision_function(project_out(Zc, Uk))))
    p = m3u_path(arm, S)
    p.parent.mkdir(exist_ok=True)
    np.savez(p, mu=mu, sd=sd, U=U, aucs=json.dumps(aucs), n_u=U.shape[1])
    return f"m3u {arm} {S} U{U.shape} calib source-ID AUROC {aucs} ({time.time() - t0:.0f}s)"


def m3_fit_scores(A, S, j, model, mu, sd, U):
    E = e7()
    D, sl = A.D, FS["F_b8"]
    T = others_of(S)
    k = kdim(len(T), j)
    Uk = U[:, :k]
    rows = E.train_rows(D, T, model)
    Xtr = np.concatenate([project_out((A.X[c][rows][:, sl] - mu) / sd, Uk) for c in TAUG])
    y = np.concatenate([D.lab[rows]] * len(TAUG))
    g = np.concatenate([D.groups[rows]] * len(TAUG))
    m = probe.fit_probe(Xtr, y, SEED, groups=g, class_weight="balanced")
    sc = np.stack([m.decision_function(project_out((A.X[c][:, sl] - mu) / sd, Uk)) for c in EVAL]).astype(np.float32)
    return sc, float(m[-1].C_[0]), len(rows)


def m3fit_task(args):
    arm, S, j, model = args
    t0 = time.time()
    z = np.load(m3u_path(arm, S))
    sc, C, n = m3_fit_scores(get_arm(arm), S, j, model, z["mu"], z["sd"], z["U"])
    p = CACHE / "m3" / "parts" / f"{arm}__{S}__j{j}__{model}.npz"
    p.parent.mkdir(exist_ok=True, parents=True)
    np.savez(p, sc=sc, C=C, n=n)
    return f"m3fit {arm} {S} j={j} {model} n={n} C={C:.3g} ({time.time() - t0:.0f}s)"


def m3_assemble():
    E = e7()
    for arm in CFG["arms"]:
        names = E.model_names(arm)
        for S in SOURCES:
            for j in GRID:
                if m3s_path(arm, S, j).exists():
                    continue
                parts = [CACHE / "m3" / "parts" / f"{arm}__{S}__j{j}__{m}.npz" for m in names]
                if all(p.exists() for p in parts):
                    zs = [np.load(p) for p in parts]
                    np.savez(m3s_path(arm, S, j), scores=np.stack([z["sc"] for z in zs]), C=np.array([float(z["C"]) for z in zs]))


def m3_analyze(arm):
    E = e7()
    ctx = Ctx(arm)
    D = ctx.D
    sub = subsets_of_gens(arm)
    out = dict(sources={}, robust={})
    evs, chosen, auc_tab = {}, {}, {}
    for S in SOURCES:
        T = others_of(S)
        cal = E.calib_rows(D, T)
        SC7 = E.load_scores(arm, "F_b8", T)[0]
        ev = {"e7": eval_scores(ctx, S, SC7, cal, {c: D.conds.index(c) for c in EVAL})}
        for j in GRID:
            z = np.load(m3s_path(arm, S, j))
            SC = {m: z["scores"][i] for i, m in enumerate(ctx.names)}
            ev[j] = eval_scores(ctx, S, SC, cal, {"clean": 0, "jpeg75_resize05": 1})
        evs[S] = ev
        aucs = {int(k): v for k, v in json.loads(str(np.load(m3u_path(arm, S))["aucs"])).items()}
        auc_tab[S] = aucs
        cand = {kdim(len(T), j): aucs[kdim(len(T), j)] for j in GRID}
        kch = choose_k(cand, CFG["m3"]["calib_tol"])
        chosen[S] = kch - 2 * (len(T) - 1)                                       # j (number of extra PCs)
    for S in SOURCES:
        T = others_of(S)
        k_of = lambda j: kdim(len(T), j)
        so = dict(calib_source_id_auc=auc_tab[S], chosen_j=int(chosen[S]), chosen_k=int(k_of(chosen[S])), variants={})
        for v in ["e7"] + GRID:
            so["variants"]["e7" if v == "e7" else f"k={k_of(v)}"] = {rd: ev_summary(evs[S][v], ctx, reading=rd) for rd in ("A", "B")}
        base_a = e7().macro_of(evs[S]["e7"][JR]["ga"], ctx.gens)
        so["delta_auroc_vs_e7"] = {f"k={k_of(j)}": E.diff_summary(E.macro_of(evs[S][j][JR]["ga"], ctx.gens), base_a) for j in GRID}
        out["sources"][S] = so
    # labels and drop-one-generator robustness (chosen k per source; uniform-j sensitivity)
    def label_for(jmap, keep, fm, rd):
        per = {}
        for S in SOURCES:
            n_ = evs[S][jmap[S]][JR]
            o_ = evs[S]["e7"][JR]
            per[S] = dict(auc=float(E.macro_of(n_["ga"], keep)[0]), fpr=n_["fp"][rd][fm]["fpr"], auc7=float(E.macro_of(o_["ga"], keep)[0]),
                          fpr7=o_["fp"][rd][fm]["fpr"])
        lb = r3_label(per)
        lb["per"] = per
        return lb
    for rd in ("A", "B"):
        out["robust"][rd] = {nm: label_for(chosen, keep, fm, rd) for nm, (keep, fm) in sub.items()}
    out["uniform_j"] = {rd: {f"j={j}": label_for({S: j for S in SOURCES}, ctx.gens, "full", rd) for j in GRID} for rd in ("A", "B")}
    # sensitivity: kill rule restricted to the two sources that are not content-confounded (VISION, OI)
    for rd in ("A", "B"):
        lb = out["robust"][rd]["all"]
        out.setdefault("excluding_RAISE", {})[rd] = {S: lb["per"][S] for S in ("VISION", "OI")}
    # bootstrap-of-difference for the chosen k, per source (AUROC)
    out["chosen_delta_auroc"] = {S: E.diff_summary(E.macro_of(evs[S][chosen[S]][JR]["ga"], ctx.gens), E.macro_of(evs[S]["e7"][JR]["ga"], ctx.gens)) for S in SOURCES}
    return out


# ============================================================================ texture baseline (Laplacian), E6 definition
def lap_values(arm):
    p = CACHE / f"lap_{arm}.npz"
    if p.exists():
        z = np.load(p)
        return {c: z[c] for c in EVAL}
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from perturb import apply_condition
    E6 = e7().e6()
    D = get_arm(arm).D.df

    def one(args):
        path, sha, c = args
        with Image.open(ROOT / path) as im:
            im.load()
            return E6.lap224(apply_condition(im, c, key=sha, seed=0))
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for c in EVAL:
            out[c] = np.array(list(ex.map(one, [(pth, s, c) for pth, s in zip(D.path, D.sha256)])))
            log(f"lap {arm}/{c} done")
    np.savez(p, **out)
    return out


def texture_analyze(arm):
    E = e7()
    ctx = Ctx(arm)
    D = ctx.D
    lap = lap_values(arm)
    trf = D.idx("train", None, label=1)
    out = {}
    for S in SOURCES:
        cal = E.calib_rows(D, others_of(S))
        out[S] = {}
        for c in EVAL:
            v = lap[c]
            d_cal = float(metrics.weighted_auroc(v[cal], v[trf], np.ones((1, len(cal))), np.ones((1, len(trf))))[0])
            sign = 1.0 if d_cal >= 0.5 else -1.0
            res = {}
            for nm, sg in (("chosen", sign), ("fake_higher", 1.0), ("fake_lower", -1.0)):
                ga = {g: metrics.weighted_auroc(sg * v[ctx.Rr[S]], sg * v[ctx.Ff[g]], ctx.WR[S], ctx.WF[g]) for g in ctx.gens}
                res[nm] = E.summ(E.macro_of(ga, ctx.gens))
            out[S][c] = dict(calib_auroc_fake_higher=d_cal, sign=sign, **res)
    return out


# ============================================================================ M4 map
def map_analyze(arm):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    A = get_arm(arm)
    D = A.D
    sl = FS["F_b8"]
    pool_src = {p: s for s in SOURCES for p in e7().SRC[s]["train_pools"]}
    cat = np.array([pool_src.get(p, "FAKE") for p in D.pool])
    keep = np.where(np.isin(D.pool, sum([e7().SRC[s]["train_pools"] for s in SOURCES], []) + ["fake"]))[0]
    rng = np.random.default_rng(SEED)
    mu, sd = A.X["clean"][keep][:, sl].mean(0), safe_sigma(A.X["clean"][keep][:, sl])
    Z = {c: (A.X[c][keep][:, sl] - mu) / sd for c in EVAL}
    pca = PCA(2, random_state=SEED).fit(np.concatenate([Z[c] for c in EVAL]))
    out = dict(rows=keep.tolist(), cat=cat[keep].tolist(), label=D.lab[keep].tolist(), pca_var=pca.explained_variance_ratio_.tolist(), pca={}, tsne={})
    ts = rng.choice(len(keep), min(CFG["map"]["max_points"], len(keep)), replace=False)
    out["tsne_idx"] = ts.tolist()
    for c in EVAL:
        out["pca"][c] = pca.transform(Z[c]).tolist()
        pc50 = PCA(50, random_state=SEED).fit_transform(Z[c][ts])
        out["tsne"][c] = TSNE(2, perplexity=CFG["map"]["tsne_perplexity"], random_state=SEED, init="pca").fit_transform(pc50).tolist()
    # descriptive: share of PC1/PC2 variance explained by source identity among reals (eta^2 on the PCA coordinates)
    ev = {}
    for c in EVAL:
        P = np.array(out["pca"][c])
        r = np.array(out["label"]) == 0
        eta = []
        for d in range(2):
            x, s_ = P[r, d], np.array(out["cat"])[r]
            tot = ((x - x.mean()) ** 2).sum()
            btw = sum((s_ == s).sum() * (x[s_ == s].mean() - x.mean()) ** 2 for s in SOURCES)
            eta.append(float(btw / tot))
        ev[c] = eta
    out["eta2_source_among_reals_pc1_pc2"] = ev
    return out


# ============================================================================ main
def stage_all(stages, workers):
    arms = CFG["arms"]
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    for a in arms:
        get_arm(a)                                                          # load before fork
    e7()
    if "m1" in stages:
        t = m1_tasks()
        log(f"m1: {len(t)} C2ST tasks")
        if t:
            run_tasks(m1_task, t, workers)
    if "m2fit" in stages:
        t = m2_tasks()
        log(f"m2fit: {len(t)} probes")
        if t:
            run_tasks(m2_task, t, workers)
    if "m3u" in stages:
        t = [(a, S) for a in arms for S in SOURCES if not m3u_path(a, S).exists()]
        log(f"m3u: {len(t)} subspace tasks")
        if t:
            run_tasks(m3u_task, t, workers)
    if "m3fit" in stages:
        E = e7()
        t = [(a, S, j, m) for a in arms for S in SOURCES for j in GRID if not m3s_path(a, S, j).exists() for m in E.model_names(a)
             if not (CACHE / "m3" / "parts" / f"{a}__{S}__j{j}__{m}.npz").exists()]
        log(f"m3fit: {len(t)} probes")
        if t:
            run_tasks(m3fit_task, t, workers)
        m3_assemble()
    if "analyze" in stages:
        for a in arms:
            for nm, fn in (("m1", m1_analyze), ("m2", m2_analyze), ("m3", m3_analyze), ("texture", texture_analyze)):
                p = RES / f"part_{nm}_{a}.json"
                t0 = time.time()
                r = fn(a)
                p.write_text(json.dumps(r, indent=1, default=float))
                log(f"analyze {nm} {a} ({time.time() - t0:.0f}s)")
    if "map" in stages:
        a = CFG["primary_arm"]
        p = RES / f"part_map_{a}.json"
        if not p.exists():
            p.write_text(json.dumps(map_analyze(a), default=float))
        log("map done")
    if "report" in stages:
        import report
        report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "m1", "m2fit", "m3u", "m3fit", "analyze", "map", "report"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    st = ["m1", "m2fit", "m3u", "m3fit", "analyze", "map", "report"] if a.stage == "all" else [a.stage]
    stage_all(st, a.workers)


if __name__ == "__main__":
    main()
