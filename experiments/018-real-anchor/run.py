"""E9 (Amendment 18): per-pipeline real-anchor adaptation of the E7 (016-loso) mid-level DINOv2 detector. Resumable stages.

    env/.venv/bin/python -I experiments/018-real-anchor/run.py all [--workers 14]
Stages (each skips finished work):
  probes  refit the E7 probes (arm x feat x held-out source x 8 models) and store raw-space weights; checked against E7's cached scores
  choose  VISION fold ONLY, fov512 / F_b8 / n = 50: A-off vs A-proj -> results/choice.json (FROZEN here, before RAISE/OI adapted numbers exist)
  proj    A-proj retrained probes for every (arm, feat, source, n, draw)       -> cache/proj/
  eval    metrics for every method                                              -> cache/eval/
  report  results/{results.json, tables.md, scores_long.csv.gz, ncurve.png}
Reuses E7 (016-loso) splits, roles, groups, probe recipe, cluster-bootstrap Plan and E8 (017) Arm/standardization conventions. CPU only.
Pure helper functions (no data access) are at the top so that selftest.py can test them on synthetic data.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))

import numpy as np

import metrics
import probe

CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CACHE, RES = HERE / "cache", HERE / "results"
CONDS, TAUG, JR = CFG["conditions"], CFG["t_aug"], CFG["launder_cond"]
CI_JR, CI_CLEAN = CONDS.index(JR), CONDS.index("clean")
FS = {k: slice(*v) for k, v in CFG["feature_slices"].items()}
SOURCES, NS, ND = CFG["sources"], CFG["n_anchor"], CFG["n_draws"]
CRIT = CFG["criteria"]
METHODS = ["A-thr", "A-off", "A-off-all", "A-off-pc", "A-proj"]
NMODELS = 8


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " E9 " + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        RES.mkdir(exist_ok=True)
        with open(RES / "run.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ============================================================================ pure helpers (synthetic-testable)
def safe_sigma(X):
    s = X.std(0)
    s[s == 0] = 1.0
    return s


def anchor_delta(mu_train, mu_anchor):
    """Offset to ADD to the held-out source's features: (mean of training reals) - (mean of anchor reals)."""
    return np.asarray(mu_train, float) - np.asarray(mu_anchor, float)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def project_dir(Z, v):
    """Remove the component along the unit vector v from every row of Z."""
    return Z - np.outer(Z @ v, v)


def thr_fpr(scores, fpr=None):
    return metrics.threshold_for_fpr(scores, CFG["fpr_target"] if fpr is None else fpr)


def auroc_point(sr, sf):
    return float(metrics.weighted_auroc(sr, sf, np.ones((1, len(sr))), np.ones((1, len(sf))))[0])


def op_point_w(sr, sfs, WR, WFs, thr):
    """Weighted operating point at a fixed threshold: macro TPR (equal weight per generator) and FPR, one value per bootstrap replicate."""
    tpr = np.mean([(WF @ (sf >= thr)) / WF.sum(1) for sf, WF in zip(sfs, WFs)], 0)
    fpr = (WR @ (np.asarray(sr) >= thr)) / WR.sum(1)
    return tpr, fpr


def fpr_at_tpr_w(sr, sfs, WR, WFs, target):
    """Per bootstrap replicate: FPR of the score at the highest threshold whose macro TPR (equal weight per generator) reaches target[b].
    sr: reals' scores [nr]; sfs: list of fakes' scores per generator; WR [B+1, nr]; WFs: list of [B+1, n_g]; target [B+1]."""
    G = len(sfs)
    s_all = np.concatenate(sfs)
    Wm = np.concatenate([WF / (WF.sum(1, keepdims=True) * G) for WF in WFs], 1)
    o = np.argsort(-s_all, kind="stable")
    ss = s_all[o]
    cum = np.cumsum(Wm[:, o], 1)
    ok = cum >= (target[:, None] - 1e-12)
    k = np.where(ok.any(1), ok.argmax(1), len(ss) - 1)
    thr = ss[k]
    sr = np.asarray(sr, float)
    o_r = np.argsort(-sr, kind="stable")
    CW = np.cumsum(WR[:, o_r], 1)
    cnt = np.searchsorted(-sr[o_r], -thr, side="right")                 # number of reals with score >= thr_b
    fpr = np.where(cnt > 0, CW[np.arange(len(thr)), np.maximum(cnt - 1, 0)], 0.0) / WR.sum(1)
    return np.where(target <= 1e-12, 0.0, fpr)


def choose_method(tab, tie):
    """tab[method] = dict(auroc, fpr) (draw means on the VISION fold). Higher AUROC wins; within `tie` of each other the lower FPR wins."""
    (a, ta), (b, tb) = list(tab.items())
    if abs(ta["auroc"] - tb["auroc"]) < tie:
        return a if ta["fpr"] <= tb["fpr"] else b
    return a if ta["auroc"] > tb["auroc"] else b


def label_of(per, gain_lb):
    """per[S] = dict(auroc, fpr) (draw means); gain_lb[S] = CI lower bound of the gain over A-thr. Amendment 18 rule."""
    n_gain = sum(1 for v in gain_lb.values() if v > 0)
    ok_src = all(v["auroc"] >= CRIT["pass_auroc"] and v["fpr"] <= CRIT["pass_fpr"] for v in per.values())
    kill_src = any(v["auroc"] < CRIT["kill_auroc"] or v["fpr"] > CRIT["kill_fpr"] for v in per.values())
    no_gain = n_gain == 0
    lab = "KILL" if (kill_src or no_gain) else "PASS" if (ok_src and n_gain >= CRIT["gain_sources_for_pass"]) else "INCONCLUSIVE"
    return dict(label=lab, kill_src=bool(kill_src), no_gain=bool(no_gain), sources_ok=bool(ok_src), n_gain_sources=int(n_gain))


# ============================================================================ E7 / E8 plumbing
_M = {}


def _load(name, path):
    if name not in _M:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _M[name] = m
    return _M[name]


def E7():
    return _load("e7run", ROOT / "experiments" / "016-loso" / "run.py")


def E8():
    _load("e8run", ROOT / "experiments" / "017-embedding-geometry" / "run.py")
    return _M["e8run"]


class Ctx:
    """One arm: E7 splits/roles/groups, cached features of all five conditions, E7's global cluster-bootstrap Plan, per-source row sets."""

    def __init__(self, arm):
        E = E7()
        self.arm, self.E = arm, E
        self.D = E.Data(arm, load_feats=False)
        self.X = {c: E.load_x(arm, self.D.df, c) for c in CONDS}
        self.plan = E.Plan(self.D)
        self.gens = E.arm_all_gens(arm)
        self.models = E.model_names(arm)
        assert len(self.models) == NMODELS
        self.mi = {g: self.models.index(E.model_of(arm, g)) for g in self.gens}
        self.full = self.models.index("full")
        D = self.D
        self.rr = {S: E.src_eval_rows(D, S) for S in SOURCES}
        self.ff = {g: D.idx("test", None, gens=[g], label=1) for g in self.gens}
        self.ff_all = np.concatenate([self.ff[g] for g in self.gens])
        self.WR = {S: self.plan.w(r) for S, r in self.rr.items()}
        self.WF = {g: self.plan.w(f) for g, f in self.ff.items()}
        self.pool = {S: np.sort(np.concatenate([D.idx(role, E.SRC[S]["train_pools"], label=0) for role in ("train", "calib")])) for S in SOURCES}
        self.base = {}
        self.static = {}

    def others(self, S):
        return tuple(s for s in SOURCES if s != S)

    def taug(self, rows, sl):
        X = np.concatenate([self.X[c][rows][:, sl] for c in TAUG])
        y = np.concatenate([self.D.lab[rows]] * len(TAUG))
        g = np.concatenate([self.D.groups[rows]] * len(TAUG))
        return X, y, g

    def anchor_order(self, S, d):
        rng = np.random.default_rng([SEED, 9, SOURCES.index(S), d])
        return self.pool[S][rng.permutation(len(self.pool[S]))]

    def subsets(self, full):
        s = {"all": (list(self.gens), self.full)}
        if full:
            s["no flux"] = ([g for g in self.gens if g != self.E.FLUX], self.full)
            for g in self.gens:
                s[f"drop {g}"] = ([x for x in self.gens if x != g], self.mi[g])
        return s


_CTX = {}


def get_ctx(arm):
    if arm not in _CTX:
        _CTX[arm] = Ctx(arm)
    return _CTX[arm]


# ============================================================================ stage: probes
def probe_path(arm, feat, S):
    return CACHE / f"probe__{arm}__{feat}__{S}.npz"


def fit_raw(X, y, g):
    m = probe.fit_probe(X, y, SEED, groups=g, class_weight="balanced")
    w = m[-1].coef_[0] / m[0].scale_
    b = m[-1].intercept_[0] - (m[0].mean_ * w).sum()
    return w.astype(np.float64), float(b), float(m[-1].C_[0]), m


def probe_task(args):
    arm, feat, S = args
    t0 = time.time()
    ctx = get_ctx(arm)
    sl = FS[feat]
    T = ctx.others(S)
    W, bs, Cs = [], [], []
    for name in ctx.models:
        rows = ctx.E.train_rows(ctx.D, T, name)
        X, y, g = ctx.taug(rows, sl)
        w, b, C, _ = fit_raw(X, y, g)
        W.append(w); bs.append(b); Cs.append(C)
    np.savez(probe_path(arm, feat, S), W=np.stack(W), b=np.array(bs), C=np.array(Cs))
    return f"probe {arm} {feat} {S} ({time.time() - t0:.0f}s)"


def base_scores(ctx, feat, S):
    """Unadapted E7 scores [8 models, 5 conds, N] (float64) from the stored raw-space weights."""
    key = (feat, S)
    if key not in ctx.base:
        z = np.load(probe_path(ctx.arm, feat, S))
        sl = FS[feat]
        sc = np.empty((NMODELS, len(CONDS), len(ctx.D.df)))
        for ci, c in enumerate(CONDS):
            sc[:, ci, :] = (ctx.X[c][:, sl].astype(np.float64) @ z["W"].T + z["b"]).T
        ctx.base[key] = (sc, z["W"])
    return ctx.base[key]


def check_against_e7(arm, feat, S, tol=1e-3):
    """Max |difference| between our raw-space probes' scores and E7's cached scores (all 8 models, all 5 conditions, all rows)."""
    ctx = get_ctx(arm)
    T = ctx.others(S)
    sc, _ = base_scores(ctx, feat, S)
    SC7 = ctx.E.load_scores(arm, feat, T)[0]
    d = max(float(np.abs(SC7[m] - sc[i]).max()) for i, m in enumerate(ctx.models))
    return d


def stage_probes(workers):
    todo = [(a, f, S) for a in CFG["arms"] for f in CFG["feats"] for S in SOURCES if not probe_path(a, f, S).exists()]
    for a in CFG["arms"]:
        get_ctx(a)
    if todo:
        log(f"probes: {len(todo)} tasks")
        run_pool(probe_task, todo, workers)
    for a in CFG["arms"]:
        for f in CFG["feats"]:
            for S in SOURCES:
                d = check_against_e7(a, f, S)
                log(f"probes: {a} {f} held-out {S}: max |score - E7 cached score| = {d:.2e}")
                assert d < 1e-3, ("probe refit does not reproduce E7", a, f, S, d)


def run_pool(fn, tasks, workers):
    t0 = time.time()
    if workers <= 1 or len(tasks) == 1:
        for i, t in enumerate(tasks):
            log(f"  [{i + 1}/{len(tasks)}] {fn(t)}")
    else:
        with ProcessPoolExecutor(workers) as ex:
            for i, out in enumerate(ex.map(fn, tasks, chunksize=1)):
                if (i + 1) % max(1, len(tasks) // 20) == 0 or i + 1 == len(tasks):
                    log(f"  [{i + 1}/{len(tasks)}] {out}")
    log(f"{len(tasks)} tasks in {time.time() - t0:.0f}s")


# ============================================================================ stage: proj (A-proj retraining)
def proj_path(arm, feat, S, n, d):
    return CACHE / "proj" / f"{arm}__{feat}__{S}__n{n}__d{d}.npz"


def static_stats(ctx, feat, S):
    """Training-reals statistics of the held-out source's training set (T-aug stack, raw space): per-condition means, T-aug mean/std."""
    key = (feat, S)
    if key not in ctx.static:
        sl = FS[feat]
        T = ctx.others(S)
        rows = ctx.E.train_rows(ctx.D, T, "full")
        reals = rows[ctx.D.lab[rows] == 0]
        mu_pc = np.stack([ctx.X[c][reals][:, sl].astype(np.float64).mean(0) for c in CONDS])
        Xs = np.concatenate([ctx.X[c][reals][:, sl] for c in TAUG]).astype(np.float64)
        ctx.static[key] = dict(mu_pc=mu_pc, mu=mu_pc[[CONDS.index(c) for c in TAUG]].mean(0), sd=safe_sigma(Xs), mu_stack=Xs.mean(0))
    return ctx.static[key]


def anchor_means(ctx, feat, anc):
    sl = FS[feat]
    mu_pc = np.stack([ctx.X[c][anc][:, sl].astype(np.float64).mean(0) for c in CONDS])
    return mu_pc, mu_pc[[CONDS.index(c) for c in TAUG]].mean(0)


def proj_task(args):
    arm, feat, S, n, d = args
    t0 = time.time()
    ctx = get_ctx(arm)
    sl = FS[feat]
    st = static_stats(ctx, feat, S)
    anc = ctx.anchor_order(S, d)[:n]
    _, mu_an = anchor_means(ctx, feat, anc)
    mu, sd = st["mu_stack"], st["sd"]                                    # standardization of the training reals (T-aug stack)
    v = unit((mu_an - mu) / sd)                                          # unit vector of the standardized offset (sign irrelevant)
    sc = np.empty((NMODELS, len(CONDS), len(ctx.D.df)), np.float32)
    Cs = []
    T = ctx.others(S)
    Zc = {c: project_dir((ctx.X[c][:, sl].astype(np.float64) - mu) / sd, v) for c in CONDS}
    for mi, name in enumerate(ctx.models):
        rows = ctx.E.train_rows(ctx.D, T, name)
        y = np.concatenate([ctx.D.lab[rows]] * len(TAUG))
        g = np.concatenate([ctx.D.groups[rows]] * len(TAUG))
        Ztr = np.concatenate([Zc[c][rows] for c in TAUG])
        m = probe.fit_probe(Ztr, y, SEED, groups=g, class_weight="balanced")
        Cs.append(float(m[-1].C_[0]))
        for ci, c in enumerate(CONDS):
            sc[mi, ci] = m.decision_function(Zc[c])
    p = proj_path(arm, feat, S, n, d)
    p.parent.mkdir(exist_ok=True)
    np.savez(p, sc=sc, C=np.array(Cs))
    return f"proj {p.stem} ({time.time() - t0:.0f}s)"


def proj_tasks(arms, feats, sources, ns):
    return [(a, f, S, n, d) for a in arms for f in feats for S in sources for n in ns for d in range(ND) if not proj_path(a, f, S, n, d).exists()]


# ============================================================================ stage: eval
def eval_path(arm, feat, S, n, method):
    return CACHE / "eval" / f"{arm}__{feat}__{S}__n{n}__{method}.pkl"


def thr_arrays(sc, ref_rows):
    """Thresholds [8 models, 5 conds] from the reference reals: reading A (clean scores at every condition) and reading B (same condition)."""
    A = np.zeros((NMODELS, len(CONDS)))
    Bt = np.zeros((NMODELS, len(CONDS)))
    for m in range(NMODELS):
        A[m, :] = thr_fpr(sc[m, CI_CLEAN, ref_rows])
        for ci in range(len(CONDS)):
            Bt[m, ci] = thr_fpr(sc[m, ci, ref_rows])
    return A, Bt


def bundle(ctx, S, sc, thrA, thrB, subs, boot_auc, athr_ops):
    """All metrics of one score tensor sc [8, 5, N] (reals of S = test reals; fakes = test fakes of each generator, scored by the generator's LOGO model)."""
    rr, ff, WR, WF = ctx.rr[S], ctx.ff, ctx.WR[S], ctx.WF
    nc, gens = len(CONDS), ctx.gens
    auc = np.zeros((nc, len(gens)))
    fprA, fprB = np.zeros((nc, NMODELS)), np.zeros((nc, NMODELS))
    tprA, tprB = np.zeros(nc), np.zeros(nc)
    for ci in range(nc):
        for gi, g in enumerate(gens):
            m = ctx.mi[g]
            auc[ci, gi] = auroc_point(sc[m, ci, rr], sc[m, ci, ff[g]])
        for m in range(NMODELS):
            fprA[ci, m] = (sc[m, ci, rr] >= thrA[m, ci]).mean()
            fprB[ci, m] = (sc[m, ci, rr] >= thrB[m, ci]).mean()
        tprA[ci] = np.mean([(sc[ctx.full, ci, ff[g]] >= thrA[ctx.full, ci]).mean() for g in gens])
        tprB[ci] = np.mean([(sc[ctx.full, ci, ff[g]] >= thrB[ctx.full, ci]).mean() for g in gens])
    out = dict(auc=auc, fprA=fprA, fprB=fprB, tprA=tprA, tprB=tprB)
    hit = (sc[ctx.full, CI_JR, rr] >= thrA[ctx.full, CI_JR]).astype(float)
    out["boot_fpr"] = (WR @ hit) / WR.sum(1)
    if boot_auc:
        out["boot_auc"] = np.mean([metrics.weighted_auroc(sc[ctx.mi[g], CI_JR, rr], sc[ctx.mi[g], CI_JR, ff[g]], WR, WF[g]) for g in gens], 0)
    out["sub"] = {}
    for name, (keep, fm) in subs.items():
        r = dict(auc=float(np.mean([auc[CI_JR, gens.index(g)] for g in keep])), fpr=float(fprA[CI_JR, fm]))
        if athr_ops is not None:
            sfs = [sc[fm, CI_JR, ff[g]] for g in keep]
            tpr_t, fpr_t = athr_ops[name]
            fm_fpr = fpr_at_tpr_w(sc[fm, CI_JR, rr], sfs, WR, [WF[g] for g in keep], tpr_t)
            gb = fpr_t - fm_fpr
            r["gain_boot"], r["gain"] = gb, float(gb[0])
            r["athr_tpr"], r["athr_fpr"], r["m_fpr_at_tpr"] = float(tpr_t[0]), float(fpr_t[0]), float(fm_fpr[0])
        out["sub"][name] = r
    return out


def stack_draws(bs):
    """List of per-draw bundles -> arrays over draws; bootstrap vectors are averaged over draws."""
    o = dict(auc=np.stack([b["auc"] for b in bs]), fprA=np.stack([b["fprA"] for b in bs]), fprB=np.stack([b["fprB"] for b in bs]),
             tprA=np.stack([b["tprA"] for b in bs]), tprB=np.stack([b["tprB"] for b in bs]),
             boot_fpr=np.mean([b["boot_fpr"] for b in bs], 0), sub={})
    if "boot_auc" in bs[0]:
        o["boot_auc"] = np.mean([b["boot_auc"] for b in bs], 0)
    for name in bs[0]["sub"]:
        r = dict(auc=np.array([b["sub"][name]["auc"] for b in bs]), fpr=np.array([b["sub"][name]["fpr"] for b in bs]))
        if "gain" in bs[0]["sub"][name]:
            r["gain"] = np.array([b["sub"][name]["gain"] for b in bs])
            r["gain_boot"] = np.mean([b["sub"][name]["gain_boot"] for b in bs], 0)
            for k in ("athr_tpr", "athr_fpr", "m_fpr_at_tpr"):
                r[k] = np.array([b["sub"][name][k] for b in bs])
        o["sub"][name] = r
    return o


def eval_task(args):
    arm, feat, S, n, methods = args
    t0 = time.time()
    ctx = get_ctx(arm)
    sl = FS[feat]
    base, W = base_scores(ctx, feat, S)
    T = ctx.others(S)
    cal = ctx.E.calib_rows(ctx.D, T)
    full_subs = (arm == CFG["criteria_arm"] and feat == CFG["criteria_feat"] and n == CFG["criteria_n"])
    subs = ctx.subsets(full_subs)
    thrA_cal, thrB_cal = thr_arrays(base, cal)
    if n == 0:                                              # E7 reference: no anchors, no draws
        b = bundle(ctx, S, base, thrA_cal, thrB_cal, subs, True, None)
        pickle.dump(stack_draws([b]), open(eval_path(arm, feat, S, 0, "E7"), "wb"))
        return f"eval {arm} {feat} {S} E7 ({time.time() - t0:.0f}s)"
    st = static_stats(ctx, feat, S)
    e7ref = pickle.load(open(eval_path(arm, feat, S, 0, "E7"), "rb"))
    res = {m: [] for m in methods}
    for d in range(ND):
        anc = ctx.anchor_order(S, d)[:n]
        mu_an_pc, mu_an = anchor_means(ctx, feat, anc)
        delta = anchor_delta(st["mu"], mu_an)
        shift = W @ delta                                    # score shift per model (the probe is linear)
        thrA_anc, thrB_anc = thr_arrays(base, anc)
        # A-thr operating point per subset (weighted, JR, reading A)
        athr_ops = {}
        for name, (keep, fm) in subs.items():
            athr_ops[name] = op_point_w(base[fm, CI_JR, ctx.rr[S]], [base[fm, CI_JR, ctx.ff[g]] for g in keep], ctx.WR[S], [ctx.WF[g] for g in keep],
                                        thrA_anc[fm, CI_JR])
        for m in methods:
            ops = athr_ops
            if m == "A-thr":
                sc, tA, tB, ba, ops = base, thrA_anc, thrB_anc, False, None
            elif m == "A-off":
                sc = base.copy()
                sc[:, :, ctx.rr[S]] += shift[:, None, None]
                tA, tB, ba = thrA_cal, thrB_cal, True
            elif m == "A-off-all":
                sc = base.copy()
                sc[:, :, ctx.rr[S]] += shift[:, None, None]
                sc[:, :, ctx.ff_all] += shift[:, None, None]
                tA, tB, ba = thrA_cal, thrB_cal, False
            elif m == "A-off-pc":
                sc = base.copy()
                for ci in range(len(CONDS)):
                    sc[:, ci, ctx.rr[S]] += (W @ anchor_delta(st["mu_pc"][ci], mu_an_pc[ci]))[:, None]
                tA, tB, ba = thrA_cal, thrB_cal, False
            elif m == "A-proj":
                sc = np.load(proj_path(arm, feat, S, n, d))["sc"].astype(np.float64)
                tA, tB = thr_arrays(sc, cal)
                ba = True
            else:
                raise ValueError(m)
            bd = bundle(ctx, S, sc, tA, tB, subs, ba, ops)
            if not ba:                                       # AUROC identical to E7 (A-thr, A-off-all) or point-only (A-off-pc)
                if m in ("A-thr", "A-off-all"):
                    bd["boot_auc"] = e7ref["boot_auc"]
            res[m].append(bd)
    for m in methods:
        o = stack_draws(res[m])
        if "boot_auc" not in o and m != "A-off-pc":
            o["boot_auc"] = e7ref["boot_auc"]
        pickle.dump(o, open(eval_path(arm, feat, S, n, m), "wb"))
    return f"eval {arm} {feat} {S} n={n} {methods} ({time.time() - t0:.0f}s)"


def eval_tasks(arms, feats, sources, ns, methods):
    out = []
    for a in arms:
        for f in feats:
            for S in sources:
                for n in ns:
                    if n == 0:
                        if not eval_path(a, f, S, 0, "E7").exists():
                            out.append((a, f, S, 0, ["E7"]))
                        continue
                    todo = [m for m in methods if not eval_path(a, f, S, n, m).exists()]
                    if todo:
                        out.append((a, f, S, n, todo))
    return out


# ============================================================================ stage: choose (VISION fold only)
def load_eval(arm, feat, S, n, method):
    return pickle.load(open(eval_path(arm, feat, S, n, method), "rb"))


def stage_choose(workers):
    c = CFG["choice"]
    assert c["fold"] == "VISION"
    p = RES / "choice.json"
    if p.exists():
        log("choose: choice.json already frozen:", json.loads(p.read_text())["method"])
        return json.loads(p.read_text())
    leak = [f.name for f in (CACHE).rglob("*") if f.is_file() and any(f"__{S}__" in f.name for S in ("RAISE", "OI")) and
            (f.parent.name in ("proj", "eval"))]
    assert not leak, ("RAISE/OI adapted caches exist before the choice", leak[:5])
    for a in CFG["arms"]:
        get_ctx(a)
    t = proj_tasks([c["arm"]], [c["feat"]], [c["fold"]], [c["n"]])
    if t:
        log(f"choose: VISION fold only: {len(t)} A-proj probe sets")
        run_pool(proj_task, t, workers)
    e = eval_tasks([c["arm"]], [c["feat"]], [c["fold"]], [0], [])
    if e:
        run_pool(eval_task, e, 1)
    e = eval_tasks([c["arm"]], [c["feat"]], [c["fold"]], [c["n"]], c["candidates"])
    if e:
        run_pool(eval_task, e, 1)
    tab, detail = {}, {}
    for m in c["candidates"]:
        r = load_eval(c["arm"], c["feat"], c["fold"], c["n"], m)
        auc = r["auc"][:, CI_JR, :].mean(1)                                        # macro over the 8 generators, per draw
        fpr = r["fprA"][:, CI_JR, ctx_full()]
        tab[m] = dict(auroc=float(auc.mean()), fpr=float(fpr.mean()))
        detail[m] = dict(auroc_mean=float(auc.mean()), auroc_p025_p975=[float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5))],
                         fpr_mean=float(fpr.mean()), fpr_p025_p975=[float(np.percentile(fpr, 2.5)), float(np.percentile(fpr, 97.5))])
    ch = choose_method(tab, c["tie_auroc"])
    out = dict(method=ch, fold=c["fold"], setting=dict(arm=c["arm"], feat=c["feat"], cond=c["cond"], n=c["n"]), rule=c["rule"], candidates=detail,
               frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), frozen_epoch=time.time(),
               config_sha256=subprocess.run(["sha256sum", str(HERE / "config.json")], capture_output=True, text=True).stdout.split()[0],
               note="chosen on the VISION fold only; no RAISE/OI adapted (A-off / A-proj / A-thr) result existed at this moment")
    RES.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    log(f"choose: VISION fold {json.dumps(detail)} -> CHOSEN {ch} (frozen in results/choice.json)")
    return out


def ctx_full():
    return NMODELS - 1                                                           # models[-1] == "full"


# ============================================================================ stage: report
def pct(a):
    return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def ci(boot):
    return [float(boot[0]), float(np.percentile(boot[1:], 2.5)), float(np.percentile(boot[1:], 97.5))]


def summarize(r, ctx_gens):
    """Per-condition summary of a stacked eval pickle: draw mean [and 2.5/97.5 percentiles over draws] of macro AUROC, FLUX AUROC, FPR (full model, A and B), TPR."""
    out = {}
    fl = ctx_gens.index(E7().FLUX)
    for ci_, c in enumerate(CONDS):
        out[c] = dict(auroc=pct(r["auc"][:, ci_, :].mean(1)), flux=pct(r["auc"][:, ci_, fl]), fpr_A=pct(r["fprA"][:, ci_, ctx_full()]),
                      fpr_B=pct(r["fprB"][:, ci_, ctx_full()]), tpr_A=pct(r["tprA"][:, ci_]), per_gen=[float(x) for x in r["auc"][:, ci_, :].mean(0)])
    jr = dict(fpr_boot_ci=ci(r["boot_fpr"]))
    if "boot_auc" in r:
        jr["auroc_boot_ci"] = ci(r["boot_auc"])
    out["_jr"] = jr
    out["_sub"] = {}
    for name, s in r["sub"].items():
        d = dict(auroc=float(s["auc"].mean()), fpr=float(s["fpr"].mean()), fpr_p975=float(np.percentile(s["fpr"], 97.5)))
        if "gain" in s:
            d.update(gain=float(s["gain"].mean()), gain_ci=ci(s["gain_boot"]), gain_draw_p=[float(np.percentile(s["gain"], 2.5)), float(np.percentile(s["gain"], 97.5))],
                     athr_tpr=float(s["athr_tpr"].mean()), athr_fpr=float(s["athr_fpr"].mean()), m_fpr_at_tpr=float(s["m_fpr_at_tpr"].mean()))
        out["_sub"][name] = d
    return out


def stage_report():
    ch = json.loads((RES / "choice.json").read_text())
    chosen = ch["method"]
    allm = ["E7"] + METHODS
    R = {}
    gens = {a: get_ctx(a).gens for a in CFG["arms"]}
    for a in CFG["arms"]:
        R[a] = {}
        for f in CFG["feats"]:
            R[a][f] = {}
            for S in SOURCES:
                R[a][f][S] = {"E7": {"0": summarize(load_eval(a, f, S, 0, "E7"), gens[a])}}
                for m in METHODS:
                    R[a][f][S][m] = {str(n): summarize(load_eval(a, f, S, n, m), gens[a]) for n in NS}
    # criteria
    crit = {}
    cn = str(CFG["criteria_n"])
    for m in ["A-off", "A-proj", "A-off-all", "A-off-pc"]:
        crit[m] = {}
        for a in CFG["arms"]:
            crit[m][a] = {}
            for f in CFG["feats"]:
                names = list(R[a][f]["VISION"][m][cn]["_sub"].keys()) if (a == "fov512" and f == "F_b8") else ["all"]
                crit[m][a][f] = {}
                for nm in names:
                    per, glb = {}, {}
                    for S in SOURCES:
                        s = R[a][f][S][m][cn]["_sub"][nm]
                        per[S] = dict(auroc=s["auroc"], fpr=s["fpr"])
                        glb[S] = s["gain_ci"][1] if "gain_ci" in s else float("nan")
                    lab = label_of(per, {k: (v if v == v else -1.0) for k, v in glb.items()})
                    crit[m][a][f][nm] = dict(per=per, gain_ci_lb=glb, label=lab)
    # protocol check: choice frozen before any RAISE/OI adapted cache
    ts = ch["frozen_epoch"]
    first = min((f.stat().st_mtime for f in CACHE.rglob("*.*") if f.parent.name in ("proj", "eval") and any(f"__{S}__" in f.name for S in ("RAISE", "OI"))), default=None)
    proto = dict(choice_frozen_epoch=ts, first_raise_oi_adapted_cache_epoch=first, order_ok=bool(first is None or first > ts))
    assert proto["order_ok"], proto
    # E7 reproduction check against E7's stored results
    e7r = json.loads((ROOT / "experiments/016-loso/results/results.json").read_text())["arms"]["fov512"]["heldout"]
    repro = {}
    for S in SOURCES:
        T = "+".join(s for s in SOURCES if s != S)
        ec = e7r[S][T]["F_b8"]["conds"][JR]
        mine = R["fov512"]["F_b8"][S]["E7"]["0"][JR]
        repro[S] = dict(auroc_e7=ec["macro"][0], auroc_here=mine["auroc"][0], fpr_e7=ec["fpr"]["A"]["full"]["fpr"], fpr_here=mine["fpr_A"][0])
        assert abs(repro[S]["auroc_e7"] - repro[S]["auroc_here"]) < 1e-4 and abs(repro[S]["fpr_e7"] - repro[S]["fpr_here"]) < 1e-9, repro[S]
    meta = dict(git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), config=CFG)
    out = dict(meta=meta, choice=ch, protocol=proto, e7_reproduction=repro, criteria=crit, chosen_label=crit[chosen]["fov512"]["F_b8"]["all"]["label"],
               results=R, owner=dict(status="pending", dir=CFG["owner"]["dir"]))
    RES.mkdir(exist_ok=True)
    (RES / "results.json").write_text(json.dumps(out, indent=1, default=float))
    write_tables(out, chosen)
    write_scores_long(chosen)
    write_ncurve(out, chosen)
    log("report done; chosen", chosen, "label", out["chosen_label"])


def write_scores_long(chosen):
    import pandas as pd
    rows = []
    for a in CFG["arms"]:
        ctx = get_ctx(a)
        D = ctx.D
        keep = np.where(np.isin(D.role, ["calib", "test"]))[0]
        for f in CFG["feats"]:
            for S in SOURCES:
                base, W = base_scores(ctx, f, S)
                st = static_stats(ctx, f, S)
                anc = ctx.anchor_order(S, 0)[:CFG["criteria_n"]]
                _, mu_an = anchor_means(ctx, f, anc)
                shift = W @ anchor_delta(st["mu"], mu_an)
                sc_off = base.copy()
                sc_off[:, :, ctx.rr[S]] += shift[:, None, None]
                sc_proj = np.load(proj_path(a, f, S, CFG["criteria_n"], 0))["sc"].astype(np.float64)
                for mname, sc in (("E7(=A-thr scores)", base), ("A-off", sc_off), ("A-proj", sc_proj)):
                    for ci_, c in enumerate(CONDS):
                        s = np.array([sc[ctx.mi[D.gen[i]] if D.lab[i] == 1 else ctx.full, ci_, i] for i in keep])
                        rows.append(pd.DataFrame(dict(arm=a, feat=f, heldout=S, method=mname, cond=c, row=keep, sha256=D.df.sha256.to_numpy()[keep],
                                                      label=D.lab[keep], generator=D.gen[keep], pool=D.pool[keep], role=D.role[keep],
                                                      score=np.round(s, 4).astype(np.float32))))
    pd.concat(rows).to_csv(RES / "scores_long.csv.gz", index=False, compression="gzip")


def f3(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def pc(v):
    return f"{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"


def write_ncurve(out, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = out["results"]["fov512"]["F_b8"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    cols = {"A-thr": "#888888", "A-off": "#1f77b4", "A-proj": "#d62728"}
    for j, S in enumerate(SOURCES):
        for m, col in cols.items():
            lw = 2.4 if m == chosen else 1.3
            for i, key in enumerate(("auroc", "fpr_A")):
                v = np.array([R[S][m][str(n)][JR][key] for n in NS])
                ax[i, j].plot(NS, v[:, 0], "-o", color=col, lw=lw, label=m + (" (chosen)" if m == chosen else ""))
                ax[i, j].fill_between(NS, v[:, 1], v[:, 2], color=col, alpha=0.12 if m == chosen else 0.06)
        e = R[S]["E7"]["0"][JR]
        ax[0, j].axhline(e["auroc"][0], color="k", ls=":", lw=1, label="E7 unadapted")
        ax[1, j].axhline(e["fpr_A"][0], color="k", ls=":", lw=1)
        ax[0, j].axhline(CRIT["pass_auroc"], color="g", ls="--", lw=0.8)
        ax[1, j].axhline(CRIT["pass_fpr"], color="g", ls="--", lw=0.8)
        ax[1, j].axhline(CRIT["kill_fpr"], color="r", ls="--", lw=0.8)
        ax[0, j].set_title(f"held-out {S}")
        ax[1, j].set_xlabel("anchor reals n")
        ax[1, j].set_xscale("log")
        ax[1, j].set_xticks(NS)
        ax[1, j].set_xticklabels(NS)
    ax[0, 0].set_ylabel("macro AUROC (jpeg75_resize05)")
    ax[1, 0].set_ylabel("FPR, full model, reading A")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("E9 n-curve: fov512, block 8 (mean over 20 anchor draws, band = 2.5-97.5 percentile over draws; green = PASS bar, red = KILL bar)")
    fig.tight_layout()
    fig.savefig(RES / "ncurve.png", dpi=130)
    plt.close(fig)


def write_tables(out, chosen):
    R, C = out["results"], out["criteria"]
    L = ["# E9 (018-real-anchor) tables", "",
         "Generated by run.py from results.json. Exploratory (D-016). Held-out source S: S's TEST reals vs held-out-generator TEST fakes (E7 protocol), anchors = n train/calib-role reals "
         "of S (20 seeded nested draws, group-disjoint from S's test reals). Mean over draws [2.5, 97.5 percentile over draws]; {cluster-bootstrap CI of the draw mean, 1,000 replicates}. "
         "FPR = full model, threshold at 1% FPR of the training calib reals (reading A: clean scores; A-thr: the anchors' scores). OWNER photos: PENDING.", "",
         f"## Method choice (VISION fold only, fov512, F_b8, jpeg75_resize05, n = 50): **{chosen}**", "", "| method | AUROC mean [p2.5, p97.5 over draws] | FPR mean [p2.5, p97.5] |", "|---|---|---|"]
    for m, d in out["choice"]["candidates"].items():
        L.append(f"| {m} | {d['auroc_mean']:.3f} [{d['auroc_p025_p975'][0]:.3f}, {d['auroc_p025_p975'][1]:.3f}] | {100 * d['fpr_mean']:.2f}% [{100 * d['fpr_p025_p975'][0]:.1f}, {100 * d['fpr_p025_p975'][1]:.1f}] |")
    L += ["", f"Frozen {out['choice']['frozen_at']} (results/choice.json, config sha256 {out['choice']['config_sha256'][:12]}); protocol check: first RAISE/OI adapted cache file is later than the freeze: {out['protocol']['order_ok']}.", ""]
    L += ["## E7 reproduction (fov512 F_b8 jpeg75_resize05)", "", "| held-out | AUROC E7 | AUROC here | FPR E7 | FPR here |", "|---|---|---|---|---|"]
    for S, d in out["e7_reproduction"].items():
        L.append(f"| {S} | {d['auroc_e7']:.4f} | {d['auroc_here']:.4f} | {d['fpr_e7']:.4f} | {d['fpr_here']:.4f} |")
    for a in CFG["arms"]:
        for f in CFG["feats"]:
            role = "PRIMARY (criteria)" if (a, f) == (CFG["criteria_arm"], CFG["criteria_feat"]) else "reported"
            L += ["", f"## Arm `{a}`, {f} ({role}): n = {cn_()} at jpeg75_resize05", "",
                  "| held-out | method | AUROC | FPR (reading A) | FPR (reading B) | TPR@A thr | gain over A-thr at matched TPR (CI) |", "|---|---|---|---|---|---|---|"]
            for S in SOURCES:
                e = R[a][f][S]["E7"]["0"][JR]
                L.append(f"| {S} | E7 unadapted | {f3(e['auroc'])} {{{e['auroc'][0]:.3f}}} | {pc(e['fpr_A'])} | {pc(e['fpr_B'])} | {e['tpr_A'][0]:.3f} | - |")
                for m in METHODS:
                    r = R[a][f][S][m][cn_()]
                    e = r[JR]
                    g = r["_sub"]["all"]
                    gs = f"{100 * g['gain']:+.2f} pt [{100 * g['gain_ci'][1]:+.2f}, {100 * g['gain_ci'][2]:+.2f}]" if "gain_ci" in g else "-"
                    ab = r["_jr"].get("auroc_boot_ci")
                    ab = f" {{{ab[1]:.3f}, {ab[2]:.3f}}}" if ab else ""
                    L.append(f"| {S} | {m}{' (chosen)' if m == chosen else ''} | {f3(e['auroc'])}{ab} | {pc(e['fpr_A'])} {{{100 * r['_jr']['fpr_boot_ci'][1]:.1f}, {100 * r['_jr']['fpr_boot_ci'][2]:.1f}}} | {pc(e['fpr_B'])} | {e['tpr_A'][0]:.3f} | {gs} |")
            L += ["", "Labels (Amendment 18 rule, point estimates over draws, gain CI lower bound > 0):", "", "| method | label | sources ok | kill by AUROC/FPR | no gain | sources with gain |", "|---|---|---|---|---|---|"]
            for m in ["A-off", "A-proj", "A-off-all", "A-off-pc"]:
                lb = C[m][a][f]["all"]["label"]
                L.append(f"| {m}{' (chosen)' if m == chosen else ''} | **{lb['label']}** | {lb['sources_ok']} | {lb['kill_src']} | {lb['no_gain']} | {lb['n_gain_sources']}/3 |")
    # n-curve
    L += ["", "## n-curve, fov512 F_b8 at jpeg75_resize05 (mean over draws [p2.5, p97.5]); see ncurve.png", "", "| held-out | method | " + " | ".join(f"n={n} AUROC | n={n} FPR" for n in NS) + " |", "|---|---|" + "---|---|" * len(NS)]
    for S in SOURCES:
        for m in METHODS:
            cells = []
            for n in NS:
                e = R["fov512"]["F_b8"][S][m][str(n)][JR]
                cells += [f3(e["auroc"]), pc(e["fpr_A"])]
            L.append(f"| {S} | {m} | " + " | ".join(cells) + " |")
    L += ["", "Gain over A-thr (FPR points at matched TPR; CI of the draw mean), fov512 F_b8, jpeg75_resize05:", "", "| held-out | method | " + " | ".join(f"n={n}" for n in NS) + " |", "|---|---|" + "---|" * len(NS)]
    for S in SOURCES:
        for m in ["A-off", "A-proj", "A-off-all", "A-off-pc"]:
            cells = []
            for n in NS:
                g = R["fov512"]["F_b8"][S][m][str(n)]["_sub"]["all"]
                cells.append(f"{100 * g['gain']:+.2f} [{100 * g['gain_ci'][1]:+.2f}, {100 * g['gain_ci'][2]:+.2f}]" if "gain_ci" in g else "-")
            L.append(f"| {S} | {m} | " + " | ".join(cells) + " |")
    # launderings
    for a in CFG["arms"]:
        L += ["", f"## Held-out launderings, arm `{a}`, F_b8, n = {cn_()}: macro AUROC | FPR (reading A) for E7 / A-thr / {chosen}", "",
              "| held-out | method | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
        for S in SOURCES:
            for m in ["E7", "A-thr", chosen]:
                r = R[a]["F_b8"][S][m]["0" if m == "E7" else cn_()]
                L.append(f"| {S} | {m} | " + " | ".join(f"{r[c]['auroc'][0]:.3f} / {100 * r[c]['fpr_A'][0]:.1f}%" for c in CONDS) + " |")
        L += ["", f"FLUX AUROC (unseen generator), arm `{a}`, F_b8, n = {cn_()}:", "", "| held-out | method | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
        for S in SOURCES:
            for m in ["E7", chosen]:
                r = R[a]["F_b8"][S][m]["0" if m == "E7" else cn_()]
                L.append(f"| {S} | {m} | " + " | ".join(f"{r[c]['flux'][0]:.3f}" for c in CONDS) + " |")
    # robustness
    L += ["", f"## Drop-one-generator robustness (fov512 F_b8 n = {cn_()}, jpeg75_resize05; FPR from the LOGO model that never saw the dropped generator)", ""]
    for m in [chosen] + [x for x in ["A-off", "A-proj"] if x != chosen]:
        L += [f"### {m}{' (chosen)' if m == chosen else ''}", "", "| subset | label | min AUROC | max FPR | gain lower bounds (VISION, RAISE, OI) |", "|---|---|---|---|---|"]
        for nm, d in C[m]["fov512"]["F_b8"].items():
            L.append(f"| {nm} | {d['label']['label']} | {min(v['auroc'] for v in d['per'].values()):.3f} | {100 * max(v['fpr'] for v in d['per'].values()):.1f}% | "
                     + ", ".join(f"{100 * d['gain_ci_lb'][S]:+.2f}" for S in SOURCES) + " |")
        L.append("")
    (RES / "tables.md").write_text("\n".join(L))


def cn_():
    return str(CFG["criteria_n"])


# ============================================================================ frozen pipeline for a NEW real source (used by apply_to_new_source.py)
def frozen_pipeline(method, feat, anchor_X, test_X=None):
    """Frozen E9 pipeline for a NEW pipeline: train on ALL three sources (E7 recipe, 'full' model, arm fov512), adapt with the new pipeline's anchor reals.
    anchor_X / test_X: dict condition -> [n, 4608] DINOv2 secondary features (004 pipeline) for at least the T-aug conditions (anchors) and any conditions (test).
    Returns dict(thr, scores {cond: array}, delta_norm). DEPLOYMENT convention: unlabeled images from the new pipeline are ALL corrected (A-off shifts every test image),
    so for A-off this is a pure operating-point shift; A-proj projects every image."""
    arm = CFG["criteria_arm"]
    ctx = get_ctx(arm)
    E = ctx.E
    sl = FS[feat]
    T = tuple(SOURCES)
    rows = E.train_rows(ctx.D, T, "full")
    cal = E.calib_rows(ctx.D, T)
    reals = rows[ctx.D.lab[rows] == 0]
    Xs = np.concatenate([ctx.X[c][reals][:, sl] for c in TAUG]).astype(np.float64)
    mu, sd = Xs.mean(0), safe_sigma(Xs)
    mu_an = np.mean([np.asarray(anchor_X[c])[:, sl].astype(np.float64).mean(0) for c in TAUG], 0)
    delta = anchor_delta(mu, mu_an)
    test_X = test_X or {}
    out = dict(delta_norm=float(np.linalg.norm(delta)), method=method, scores={})
    if method in ("A-off", "A-off-all"):
        X, y, g = ctx.taug(rows, sl)
        w, b, C, _ = fit_raw(X, y, g)
        s_cal = ctx.X["clean"][cal][:, sl].astype(np.float64) @ w + b
        out["thr"] = float(thr_fpr(s_cal))
        for c, Xt in test_X.items():
            out["scores"][c] = (np.asarray(Xt)[:, sl].astype(np.float64) + delta) @ w + b
    elif method == "A-proj":
        v = unit((mu_an - mu) / sd)
        Z = project_dir((np.concatenate([ctx.X[c][rows][:, sl] for c in TAUG]).astype(np.float64) - mu) / sd, v)
        y = np.concatenate([ctx.D.lab[rows]] * len(TAUG))
        g = np.concatenate([ctx.D.groups[rows]] * len(TAUG))
        m = probe.fit_probe(Z, y, SEED, groups=g, class_weight="balanced")
        out["thr"] = float(thr_fpr(m.decision_function(project_dir((ctx.X["clean"][cal][:, sl].astype(np.float64) - mu) / sd, v))))
        for c, Xt in test_X.items():
            out["scores"][c] = m.decision_function(project_dir((np.asarray(Xt)[:, sl].astype(np.float64) - mu) / sd, v))
    elif method == "A-thr":
        X, y, g = ctx.taug(rows, sl)
        w, b, C, _ = fit_raw(X, y, g)
        out["thr"] = float(thr_fpr(np.asarray(anchor_X["clean"])[:, sl].astype(np.float64) @ w + b))
        for c, Xt in test_X.items():
            out["scores"][c] = np.asarray(Xt)[:, sl].astype(np.float64) @ w + b
    else:
        raise ValueError(method)
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probes", "choose", "proj", "eval", "report", "all"], nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    (CACHE / "eval").mkdir(parents=True, exist_ok=True)
    (CACHE / "proj").mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    for arm in CFG["arms"]:
        get_ctx(arm)
    if a.stage in ("probes", "choose", "proj", "eval", "report", "all"):
        stage_probes(a.workers)
    if a.stage in ("choose", "proj", "eval", "report", "all"):
        stage_choose(a.workers)
    if a.stage in ("proj", "eval", "report", "all"):
        t = proj_tasks(CFG["arms"], CFG["feats"], SOURCES, NS)
        if t:
            log(f"proj: {len(t)} A-proj probe sets (8 models each)")
            run_pool(proj_task, t, a.workers)
    if a.stage in ("eval", "report", "all"):
        e = eval_tasks(CFG["arms"], CFG["feats"], SOURCES, [0], [])
        if e:
            run_pool(eval_task, e, a.workers)
        e = eval_tasks(CFG["arms"], CFG["feats"], SOURCES, NS, METHODS)
        if e:
            log(f"eval: {len(e)} tasks")
            run_pool(eval_task, e, a.workers)
    if a.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
