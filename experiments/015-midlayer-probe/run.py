"""E6 (Amendment 15): mid-level DINOv2 linear probe vs B0, twin-disjoint, laundering-augmented training. Resumable.

    env/.venv/bin/python experiments/015-midlayer-probe/run.py all [--workers 8]
Stages: fit (probes, cached scores in cache/), texture (Laplacian baseline, cache/lap_*.npz), eval (results/).
All design constants are in config.json (frozen before any result). CPU only: cached 004 features, no GPU feature pass.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
from p4features import load_features_aligned

CFG = json.loads((HERE / "config.json").read_text())
SEED = CFG["seed"]
B = CFG["n_boot"]
CACHE = HERE / "cache"
RES = HERE / "results"
CONDS = CFG["eval_conditions"]
CI = {c: i for i, c in enumerate(CONDS)}
TRAINGENS = CFG["train_generators"]
FLUX = CFG["test_only_generator"]
GENS8 = TRAINGENS + [FLUX]
MODELS = [f"g_{g}" for g in TRAINGENS] + ["full"]
FEATS = {k: slice(*v) for k, v in CFG["features"].items()}
REGIMES = CFG["regimes"]
SOURCES = list(CFG["sources"])
TH = CFG["p1_gains_thresholds"]
POOLS = {"native": ["vision_native"], "WA": ["vision_whatsapp"], "FBH": ["vision_facebook_hq"], "RAISE": ["raise"],
         "primary": ["vision_native", "raise"], "vision3": ["vision_native", "vision_whatsapp", "vision_facebook_hq"],
         "allreals": ["vision_native", "vision_whatsapp", "vision_facebook_hq", "raise"]}
RID_RE = re.compile(r"(r[0-9a-f]{8,}t)")


def log(*a):
    print(time.strftime("%H:%M:%S"), "E6", *a, flush=True)


# ----------------------------------------------------------------------------- data, roles, groups
def load_splits(arm):
    return pd.read_csv(ROOT / CFG["arms"][arm]["splits"], keep_default_na=False, dtype={"vidx": str})


def caption_ids(df):
    """r<hex>t id of the file name (RAISE real and every fake prompted with its caption), '' when none."""
    return np.array([(RID_RE.search(os.path.basename(p)) or [None, ""])[1] for p in df.src_path])


def build_roles(df):
    """-> (role, rid, side). role in {train, calib, test, drop}. Rows without a caption id keep their official split.
    A caption family (all rows sharing an id, any generator, any class) is placed WHOLE on the train side or the eval side
    (calib/test), by the majority of its official rows (ties -> eval); rows whose official split is on the other side are
    dropped. So no caption id is shared between train and calib/test. Calib and test may share ids (neither trains)."""
    sp = df.split.to_numpy()
    rid = caption_ids(df)
    role = np.where(rid == "", sp, "drop").astype(object)
    side = {}
    has = rid != ""
    for r in np.unique(rid[has]):
        m = rid == r
        ntr = int((sp[m] == "train").sum())
        nev = int(m.sum()) - ntr
        side[r] = "train" if ntr > nev else "eval"
    for i in np.where(has)[0]:
        s = side[rid[i]]
        if (s == "train") == (sp[i] == "train"):
            role[i] = sp[i]
    return role, rid, side


def make_groups(df, rid):
    dev, sha = df.device.to_numpy(), df.sha256.to_numpy()
    return np.array([("dev:" + d) if d else (("rid:" + r) if r else "img:" + s) for d, r, s in zip(dev, rid, sha)])


class Arm:
    """Splits + roles + cached DINOv2 features (secondary, 4608-d) for one arm."""

    def __init__(self, name, load_feats=True):
        self.name = name
        self.df = load_splits(name)
        self.role, self.rid, self.side = build_roles(self.df)
        self.groups = make_groups(self.df, self.rid)
        self.lab = self.df.label.to_numpy()
        self.gen = self.df.generator.to_numpy()
        self.pool = self.df.pool.to_numpy()
        self.X = {}
        if load_feats:
            for c in CONDS:
                x = load_features_aligned(self.df, "secondary", c, subdir=CFG["arms"][name]["subdir"])
                assert x.shape[1] == 4608 and not np.isnan(x).any(), (name, c)
                self.X[c] = x

    def idx(self, role, pools=None, gens=None, label=None):
        m = self.role == role
        if pools is not None:
            m &= np.isin(self.pool, pools)
        if gens is not None:
            m &= np.isin(self.gen, gens)
        if label is not None:
            m &= self.lab == label
        return np.where(m)[0]


def train_rows(A, source, model):
    """Row indices used to fit `model` ('g_<gen>' = leave that generator out; 'full' = all 7) for a real source."""
    pools = {"all": ["vision_native", "raise"], "raise_only": ["raise"], "vision_only": ["vision_native"]}[source]
    real = A.idx("train", pools, label=0)
    gens = [g for g in TRAINGENS if model != f"g_{g}"]
    fake = A.idx("train", ["fake"], gens=gens, label=1)
    return np.sort(np.concatenate([real, fake]))


# ----------------------------------------------------------------------------- fitting
_ARM = {}                                                    # filled before fork


def fit_scores(A, feat, regime, source, model):
    rows = train_rows(A, source, model)
    cols = FEATS[feat]
    tc = REGIMES[regime]
    X = np.concatenate([A.X[c][rows][:, cols] for c in tc])
    y = np.concatenate([A.lab[rows]] * len(tc))
    g = np.concatenate([A.groups[rows]] * len(tc))             # copies share the group of their source image
    m = probe.fit_probe(X, y, SEED, groups=g, class_weight="balanced")
    sc = np.stack([m.decision_function(A.X[c][:, cols]) for c in CONDS]).astype(np.float32)
    return sc, float(m[-1].C_[0]), int(len(rows))


def fit_literal003(A, model):
    """003 recipe: fit_probe on b0_train clean last-block features, default CV, unweighted; LOGO = minus generator."""
    b0 = A.df.b0_train.to_numpy()
    gname = model[2:] if model.startswith("g_") else None
    rows = np.where(b0 & (A.gen != gname))[0]
    cols = FEATS["F_last"]
    m = probe.fit_probe(A.X["clean"][rows][:, cols], A.lab[rows], SEED)
    sc = np.stack([m.decision_function(A.X[c][:, cols]) for c in CONDS]).astype(np.float32)
    return sc, float(m[-1].C_[0]), int(len(rows))


def _task(args):
    arm, kind, key, model = args
    A = _ARM[arm]
    t0 = time.time()
    out = fit_literal003(A, model) if kind == "lit" else fit_scores(A, *key, model)
    return args, out, time.time() - t0


def combos():
    return [(f, r, s) for f in FEATS for r in REGIMES for s in SOURCES]


def cache_path(arm, kind, key):
    return CACHE / (f"scores_{arm}__lit003.npz" if kind == "lit" else f"scores_{arm}__{key[0]}__{key[1]}__{key[2]}.npz")


def stage_fit(arms, workers):
    todo = []
    for arm in arms:
        for kind, key in [("lit", None)] + [("std", c) for c in combos()]:
            if not cache_path(arm, kind, key).exists():
                todo.append((arm, kind, key))
    if not todo:
        log("fit: all cached")
        return
    for arm in {t[0] for t in todo}:
        _ARM[arm] = Arm(arm)
        log(f"loaded {arm}: roles {pd.Series(_ARM[arm].role).value_counts().to_dict()}")
    tasks = [(arm, kind, key, m) for arm, kind, key in todo for m in MODELS]
    log(f"fit: {len(todo)} combos, {len(tasks)} probes, {workers} workers")
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for args, out, dt in ex.map(_task, tasks):
            arm, kind, key, model = args
            res[(arm, kind, key, model)] = out
            full = [res.get((arm, kind, key, m)) for m in MODELS]
            log(f"  {arm} {kind} {key} {model}: n_train={out[2]} C={out[1]:.3g} ({dt:.0f}s)")
            if all(f is not None for f in full):
                np.savez(cache_path(arm, kind, key), scores=np.stack([f[0] for f in full]),
                         C=np.array([f[1] for f in full]), n_train=np.array([f[2] for f in full]))
    log(f"fit done in {time.time() - t0:.0f}s")


def load_scores(arm, kind, key):
    z = np.load(cache_path(arm, kind, key))
    return {m: z["scores"][i] for i, m in enumerate(MODELS)}, z["C"], z["n_train"]       # model -> [conds, N]


# ----------------------------------------------------------------------------- texture baseline
def lap224(im):
    from preprocess import center_crop
    x = np.asarray(center_crop(im.convert("RGB"), 224), np.float32) / 255.0
    g = x.mean(2)
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def lap_values(arm, conds=("clean", "jpeg75_resize05")):
    from PIL import Image
    from perturb import apply_condition
    p = CACHE / f"lap_{arm}.npz"
    if p.exists():
        z = np.load(p)
        return {c: z[c] for c in conds}
    df = load_splits(arm)

    def one(args):
        path, sha, c = args
        with Image.open(ROOT / path) as im:
            im.load()
            return lap224(apply_condition(im, c, key=sha, seed=0))
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for c in conds:
            out[c] = np.array(list(ex.map(one, [(pth, s, c) for pth, s in zip(df.path, df.sha256)])))
            log(f"lap {arm}/{c} done")
    np.savez(p, **out)
    return out


# ----------------------------------------------------------------------------- bootstrap plans and metrics
def summ(arr):
    """arr [B+1] (row 0 = point estimate) -> [value, lo, hi]."""
    lo, hi = np.percentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def model_of(g):
    return f"g_{g}" if g in TRAINGENS else "full"


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return [float("nan")] * 2
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


class Plan:
    """Cluster bootstrap for one real set: reals by cluster (device for VISION, image for RAISE), fakes of the test role by image
    within generator (lib/metrics.bootstrap_weights). Row 0 of the weights is all ones (point estimate). Shared by all methods."""

    def __init__(self, A, pools, gens=GENS8, n_boot=B, seed=SEED):
        self.A = A
        self.R = A.idx("test", pools, label=0)
        self.gens = list(gens)
        self.F = {g: A.idx("test", ["fake"], gens=[g]) for g in self.gens}
        strat = np.concatenate([[g] * len(self.F[g]) for g in self.gens])
        dev, sha = A.df.device.to_numpy(), A.df.sha256.to_numpy()
        clus = [dev[i] if dev[i] else "img:" + sha[i] for i in self.R]
        Wr, Wf = metrics.bootstrap_weights(clus, strat, n_boot, seed)
        self.Wr = np.vstack([np.ones((1, Wr.shape[1])), Wr])
        self.Wf = np.vstack([np.ones((1, Wf.shape[1])), Wf])
        self.cols, o = {}, 0
        for g in self.gens:
            self.cols[g] = slice(o, o + len(self.F[g]))
            o += len(self.F[g])
        self.n_real = len(self.R)

    def gen_auc(self, S, ci):
        """S: model -> [conds, N]. -> {g: [B+1] AUROC of g's fakes vs this real set under g's LOGO (FLUX: full) model."""
        out = {}
        for g in self.gens:
            s = S[model_of(g)][ci].astype(np.float64)
            out[g] = metrics.weighted_auroc(s[self.R], s[self.F[g]], self.Wr, self.Wf[:, self.cols[g]])
        return out


def macro(ga, gens):
    return np.mean([ga[g] for g in gens], 0)


class TwinPlan:
    """RAISE test reals and the test fakes sharing their caption id. Bootstrap over caption ids (real + its fakes together)."""

    def __init__(self, A, n_boot=B, seed=SEED):
        self.A = A
        real = A.idx("test", ["raise"], label=0)
        r2i = {A.rid[i]: i for i in real if A.rid[i]}
        fk = np.array([i for i in A.idx("test", ["fake"]) if A.rid[i] in r2i], int)
        self.ids = sorted({A.rid[i] for i in fk})
        pos = {r: k for k, r in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        W = np.stack([np.bincount(rng.integers(0, len(self.ids), len(self.ids)), minlength=len(self.ids)) for _ in range(n_boot)])
        self.W = np.vstack([np.ones((1, len(self.ids))), W]).astype(float)
        self.per = {}
        for g in GENS8:
            f = fk[A.gen[fk] == g]
            ids_g = sorted({A.rid[i] for i in f})
            self.per[g] = dict(f=f, fid=np.array([pos[A.rid[i]] for i in f]), rids=ids_g, r=np.array([r2i[r] for r in ids_g]),
                               rpos=np.array([pos[r] for r in ids_g]), r_of_f=np.array([ids_g.index(A.rid[i]) for i in f]))
        self.n_pairs = {g: len(v["f"]) for g, v in self.per.items()}
        self.n_ids = len(self.ids)

    def auc(self, S, ci):
        """-> dict(pooled=[B+1] (pairs-weighted mean of per-generator AUROCs), per_gen={g: [B+1]}, win=matched-pair win rate)."""
        per, num, den, win = {}, 0.0, 0, 0.0
        for g, v in self.per.items():
            if not len(v["f"]):
                continue
            s = S[model_of(g)][ci].astype(np.float64)
            sr, sf = s[v["r"]], s[v["f"]]
            per[g] = metrics.weighted_auroc(sr, sf, self.W[:, v["rpos"]], self.W[:, v["fid"]])
            n = len(v["f"])
            num = num + per[g] * n
            den += n
            m = sf - sr[v["r_of_f"]]
            win += float(((m > 0) + 0.5 * (m == 0)).sum())
        return dict(pooled=num / den, per_gen=per, win=win / den)


# ----------------------------------------------------------------------------- evaluation
def other_metrics(A, S, ci, R, thr_models):
    """Macro (8 gens) of AP, FPR@95TPR, TPR@1%FPR and balanced accuracy at the pooled-1% calib threshold, primary reals."""
    out = {k: [] for k in ("ap", "fpr_at_95tpr", "tpr_at_1fpr", "bal_acc_1pct")}
    for g in GENS8:
        mdl = model_of(g)
        s = S[mdl][ci].astype(np.float64)
        F = A.idx("test", ["fake"], gens=[g])
        y = np.r_[np.zeros(len(R)), np.ones(len(F))].astype(int)
        sc = np.r_[s[R], s[F]]
        m = metrics.compute_metrics(y, sc, threshold=None, calib_real_scores=None, n_boot=0)
        out["ap"].append(m["ap"]["value"])
        out["fpr_at_95tpr"].append(m["fpr_at_95tpr"]["value"])
        out["tpr_at_1fpr"].append(m["tpr_at_1fpr"]["value"])
        out["bal_acc_1pct"].append(metrics.balanced_accuracy(y, sc, thr_models[mdl]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def calib_thresholds(A, S):
    calib = A.idx("calib", ["vision_native", "vision_whatsapp", "vision_facebook_hq", "raise"], label=0)
    return {m: metrics.threshold_for_fpr(S[m][CI["clean"]][calib], 0.01) for m in MODELS}, len(calib)


def evaluate_arm(arm):
    A = Arm(arm, load_feats=False)
    plans = {k: Plan(A, p) for k, p in POOLS.items()}
    twin = TwinPlan(A)
    log(f"{arm}: roles {pd.Series(A.role).value_counts().to_dict()}; reals/pool test "
        f"{ {k: p.n_real for k, p in plans.items()} }; twin ids {twin.n_ids}, pairs {twin.n_pairs}")
    R = dict(arm=arm, n_real={k: p.n_real for k, p in plans.items()}, n_fake_test={g: len(plans['primary'].F[g]) for g in GENS8},
             twin=dict(n_ids=twin.n_ids, n_pairs=twin.n_pairs), combos={}, _ga={}, _tw={}, _S={})
    entries = [("lit", None)] + [("std", c) for c in combos()]
    for kind, key in entries:
        S, C, ntr = load_scores(arm, kind, key)
        name = "B0lit003|T-clean|all" if kind == "lit" else "|".join(key)
        thr, ncal = calib_thresholds(A, S)
        e = dict(C=[float(c) for c in C], n_train=[int(n) for n in ntr], n_calib_reals=ncal, conds={})
        R["_S"][name] = S
        for c in CONDS:
            ci = CI[c]
            ec = {}
            for pk, pl in plans.items():
                ga = pl.gen_auc(S, ci)
                R["_ga"][(name, c, pk)] = ga
                ec[pk] = dict(macro8=summ(macro(ga, GENS8)), macro7=summ(macro(ga, TRAINGENS)), flux=summ(ga[FLUX]),
                              per_gen={g: float(ga[g][0]) for g in GENS8})
            ec["extra_primary"] = other_metrics(A, S, ci, plans["primary"].R, thr)
            ta = twin.auc(S, ci)
            R["_tw"][(name, c)] = ta
            ec["twin"] = dict(pooled=summ(ta["pooled"]), macro=summ(macro(ta["per_gen"], list(ta["per_gen"]))),
                              win=ta["win"], per_gen={g: float(v[0]) for g, v in ta["per_gen"].items()})
            # FPR at the pooled 1% calib threshold (model 'full'), per pool, Clopper-Pearson
            s = S["full"][ci]
            fp = {}
            for pk in ("native", "WA", "FBH", "RAISE", "primary", "vision3", "allreals"):
                r = plans[pk].R
                k, n = int((s[r] >= thr["full"]).sum()), len(r)
                fp[pk] = dict(fpr=k / n, k=k, n=n, cp95=clopper_pearson(k, n))
            ec["fpr_full_1pct"] = fp
            ec["tpr_full_1pct"] = {g: float((s[plans["primary"].F[g]] >= thr["full"]).mean()) for g in GENS8}
            e["conds"][c] = ec
        e["thr_full_1pct"] = float(thr["full"])
        R["combos"][name] = e
        log(f"  eval {arm} {name}: jr macro8 {e['conds']['jpeg75_resize05']['primary']['macro8'][0]:.4f}")
    return A, plans, twin, R


def diff_summary(a, b):
    d = a - b
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    return dict(delta=float(d[0]), ci_lo=float(lo), ci_hi=float(hi), p_le0=float((d[1:] <= 0).mean()))


def texture_block(A, arm, plans):
    lap = lap_values(arm)
    out = {}
    calib = A.idx("calib", POOLS["allreals"], label=0)
    trf = A.idx("train", ["fake"], label=1)
    for c in ("clean", "jpeg75_resize05"):
        v = lap[c]
        d_cal = float(metrics.weighted_auroc(v[calib], v[trf], np.ones((1, len(calib))), np.ones((1, len(trf))))[0])
        sign = 1.0 if d_cal >= 0.5 else -1.0
        pl = plans["primary"]
        res = {}
        for nm, sg in (("chosen", sign), ("fake_higher", 1.0), ("fake_lower", -1.0)):
            ga = {g: metrics.weighted_auroc(sg * v[pl.R], sg * v[pl.F[g]], pl.Wr, pl.Wf[:, pl.cols[g]]) for g in GENS8}
            res[nm] = dict(macro8=summ(macro(ga, GENS8)), flux=summ(ga[FLUX]), per_gen={g: float(ga[g][0]) for g in GENS8})
        out[c] = dict(calib_auroc_fake_higher=d_cal, sign=sign, **res,
                      best_sign_macro8=max(res["fake_higher"]["macro8"][0], res["fake_lower"]["macro8"][0]))
    return out


def p_labels(R, arm, gens_keep, combo_main="F_b8|T-aug|all", combo_b0="F_last|T-clean|all", b0_name=None):
    """P1/P2 pass/fail on a generator subset used for the macro (FLUX criterion untouched, as in 005 gate_robustness)."""
    ga, tw = R["_ga"], R["_tw"]
    jr, cl = "jpeg75_resize05", "clean"
    out = {}
    for b0lab, b0 in (("matched", combo_b0), ("lit003", "B0lit003|T-clean|all")):
        gm, gb = ga[(combo_main, jr, "primary")], ga[(b0, jr, "primary")]
        dm = diff_summary(macro(gm, gens_keep), macro(gb, gens_keep))
        df_ = diff_summary(gm[FLUX], gb[FLUX])
        cm, cb = ga[(combo_main, cl, "primary")], ga[(b0, cl, "primary")]
        closs = float(macro(cb, gens_keep)[0] - macro(cm, gens_keep)[0])
        ok = dm["delta"] >= TH["gain"] and df_["delta"] >= TH["flux_gain"] and closs <= TH["clean_loss"] \
            and dm["ci_lo"] > 0 and df_["ci_lo"] > 0
        out[f"P1_vs_{b0lab}"] = dict(passed=bool(ok), gain_macro=dm, gain_flux=df_, clean_loss=closs)
    tw_g = tw[(combo_main, jr)]["per_gen"]
    n = R["twin"]["n_pairs"]
    gk = [g for g in gens_keep if g in tw_g]
    twin_v = float(sum(tw_g[g][0] * n[g] for g in gk) / sum(n[g] for g in gk))
    cr = macro(ga[("F_b8|T-aug|raise_only", jr, "vision3")], gens_keep)[0]
    cv = macro(ga[("F_b8|T-aug|vision_only", jr, "RAISE")], gens_keep)[0]
    out["P2"] = dict(twin=twin_v, cross_raise_to_vision=float(cr), cross_vision_to_raise=float(cv),
                     passed=bool(twin_v >= TH["twin"] and cr >= TH["cross"] and cv >= TH["cross"]))
    return out


def run_eval():
    RES.mkdir(exist_ok=True)
    allres, long_rows = {}, []
    for arm in CFG["arms"]:
        A, plans, twin, R = evaluate_arm(arm)
        R["texture"] = texture_block(A, arm, plans)
        # fragility: P1/P2 labels with each generator (and Firefly) dropped from the macro
        subsets = {"all 8": GENS8, "no flux": TRAINGENS, "no firefly": [g for g in GENS8 if g != "firefly"]}
        subsets.update({f"drop {g}": [x for x in GENS8 if x != g] for g in GENS8})
        R["fragility"] = {k: p_labels(R, arm, v) for k, v in subsets.items()}
        base = R["fragility"]["all 8"]
        for k, v in R["fragility"].items():
            for lab in ("P1_vs_matched", "P1_vs_lit003", "P2"):
                v[lab]["fragile"] = bool(v[lab]["passed"] != base[lab]["passed"])
        # scores_long: per splits row (row = index in the arm's splits csv), score of its own LOGO model (reals: 'full')
        for name, S in R["_S"].items():
            f, r, s = (name.split("|") + ["", "", ""])[:3]
            for c in CONDS:
                ci = CI[c]
                sc = np.array([S[model_of(g)][ci][i] if A.lab[i] == 1 else S["full"][ci][i] for i, g in enumerate(A.gen)])
                long_rows.append(pd.DataFrame(dict(arm=arm, combo=name, cond=c, row=np.arange(len(A.df)), label=A.lab,
                                                   generator=A.gen, pool=A.pool, role=A.role, score=np.round(sc, 4).astype(np.float32))))
        allres[arm] = {k: v for k, v in R.items() if not k.startswith("_")}
        if arm == CFG["label_arm"]:
            paired = {}
            for cond in CONDS:
                for pk in ("primary", "native", "WA", "FBH", "RAISE"):
                    for main in ("F_b8|T-aug|all", "F_b8|T-clean|all", "F_cat|T-aug|all", "F_cat|T-clean|all", "F_last|T-aug|all"):
                        for b0 in ("F_last|T-clean|all", "B0lit003|T-clean|all"):
                            gm, gb = R["_ga"][(main, cond, pk)], R["_ga"][(b0, cond, pk)]
                            paired[f"{main} - {b0} | {cond} | {pk}"] = dict(
                                macro8=diff_summary(macro(gm, GENS8), macro(gb, GENS8)),
                                macro7=diff_summary(macro(gm, TRAINGENS), macro(gb, TRAINGENS)),
                                flux=diff_summary(gm[FLUX], gb[FLUX]))
            allres[arm]["paired_vs_B0"] = paired
    pd.concat(long_rows).to_csv(RES / "scores_long.csv.gz", index=False, compression="gzip")
    meta = dict(git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                config=CFG)
    (RES / "results.json").write_text(json.dumps(dict(meta=meta, arms=allres), indent=1, default=float))
    write_tables(allres)
    log("eval done")


# ----------------------------------------------------------------------------- tables
def f3(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def write_tables(allres):
    L = ["# E6 (015-midlayer-probe) tables", "",
         "Generated by run.py from results.json. Exploratory (D-016). Primary reals = test VISION natives + RAISE test reals. "
         "LOGO-macro = 7 train generators (LOGO model) + FLUX (full model). CIs: 1,000 cluster-bootstrap replicates (reals by device / RAISE image, fakes by image per generator).", ""]
    for arm, R in allres.items():
        L += [f"## Arm `{arm}`", "", f"Test reals per set: {R['n_real']}; test fakes per generator: {R['n_fake_test']}; twin ids {R['twin']['n_ids']}, pairs {R['twin']['n_pairs']}", ""]
        names = list(R["combos"])
        for pk, title in (("primary", "primary reals (native + RAISE)"),):
            L += [f"### LOGO-macro AUROC (8 gens), {title}", "", "| combo | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
            for n in names:
                L.append(f"| {n} | " + " | ".join(f3(R["combos"][n]["conds"][c][pk]["macro8"]) for c in CONDS) + " |")
            L += ["", f"### FLUX AUROC (unseen), {title}", "", "| combo | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
            for n in names:
                L.append(f"| {n} | " + " | ".join(f3(R["combos"][n]["conds"][c][pk]["flux"]) for c in CONDS) + " |")
            L += ["", f"### LOGO-macro over the 7 train generators only, {title}", "", "| combo | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
            for n in names:
                L.append(f"| {n} | " + " | ".join(f"{R['combos'][n]['conds'][c][pk]['macro7'][0]:.3f}" for c in CONDS) + " |")
        for src, tgt, title in (("all", None, "per real pool (source all)"),):
            L += ["", "### LOGO-macro AUROC per real pool, source all (jpeg75_resize05 | clean)", "", "| combo | cond | native | WA | FBH | RAISE |", "|---|---|---|---|---|---|"]
            for n in names:
                if not n.endswith("|all") and n != "B0lit003|T-clean|all":
                    continue
                for c in ("jpeg75_resize05", "clean"):
                    L.append(f"| {n} | {c} | " + " | ".join(f"{R['combos'][n]['conds'][c][p]['macro8'][0]:.3f}" for p in ("native", "WA", "FBH", "RAISE")) + " |")
        L += ["", "### Twin-paired AUROC (RAISE test real vs its test caption-twin fakes; pairs-weighted over generators; CI bootstrap over ids) | matched-pair win rate", "",
              "| combo | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
        for n in names:
            if n.endswith("|all") or n == "B0lit003|T-clean|all":
                L.append(f"| {n} | " + " | ".join(f"{f3(R['combos'][n]['conds'][c]['twin']['pooled'])} / {R['combos'][n]['conds'][c]['twin']['win']:.2f}" for c in CONDS) + " |")
        L += ["", "### Cross-source LOGO-macro AUROC (train with one real source, test against the other)", "",
              "| combo | cond | RAISE-only -> native+WA+FBH | VISION-only -> RAISE |", "|---|---|---|---|"]
        for f, r in (("F_b8", "T-aug"), ("F_b8", "T-clean"), ("F_last", "T-aug"), ("F_last", "T-clean"), ("F_cat", "T-aug"), ("F_cat", "T-clean")):
            for c in CONDS:
                a = R["combos"][f"{f}|{r}|raise_only"]["conds"][c]["vision3"]["macro8"]
                b = R["combos"][f"{f}|{r}|vision_only"]["conds"][c]["RAISE"]["macro8"]
                L.append(f"| {f}|{r} | {c} | {f3(a)} | {f3(b)} |")
        L += ["", "### FPR at the pooled 1% calib threshold (model 'full', FLUX unseen), Clopper-Pearson 95% CI", "",
              "| combo | cond | native | WA | FBH | RAISE | all reals |", "|---|---|---|---|---|---|---|"]
        for n in names:
            if not (n.endswith("|all") or n == "B0lit003|T-clean|all"):
                continue
            for c in ("clean", "jpeg75_resize05", "social_sim"):
                fp = R["combos"][n]["conds"][c]["fpr_full_1pct"]
                L.append(f"| {n} | {c} | " + " | ".join(f"{fp[p]['fpr']:.3f} ({fp[p]['k']}/{fp[p]['n']}) [{fp[p]['cp95'][0]:.3f}, {fp[p]['cp95'][1]:.3f}]"
                                                       for p in ("native", "WA", "FBH", "RAISE", "allreals")) + " |")
        L += ["", "### Other metrics (macro over 8 gens, primary reals; bal. acc. at the model's own pooled-1% calib threshold)", "",
              "| combo | cond | AP | FPR@95TPR | TPR@1%FPR | bal.acc@1% |", "|---|---|---|---|---|---|"]
        for n in names:
            if not (n.endswith("|all") or n == "B0lit003|T-clean|all"):
                continue
            for c in ("clean", "jpeg75_resize05", "social_sim"):
                o = R["combos"][n]["conds"][c]["extra_primary"]
                L.append(f"| {n} | {c} | {o['ap']:.3f} | {o['fpr_at_95tpr']:.3f} | {o['tpr_at_1fpr']:.3f} | {o['bal_acc_1pct']:.3f} |")
        L += ["", "### Laplacian texture baseline (macro AUROC over 8 gens, primary reals)", "", "| cond | sign chosen (calib vs train fakes) | chosen | fake=higher | fake=lower | best sign |", "|---|---|---|---|---|---|"]
        for c, t in R["texture"].items():
            L.append(f"| {c} | {t['sign']:+.0f} (calib AUROC {t['calib_auroc_fake_higher']:.3f}) | {f3(t['chosen']['macro8'])} | {f3(t['fake_higher']['macro8'])} | {f3(t['fake_lower']['macro8'])} | {t['best_sign_macro8']:.3f} |")
        L += ["", "### Fragility: labels with generators dropped from the macro (FLUX criterion unchanged)", "",
              "| subset | P1 vs matched B0 | gain macro [CI] | gain FLUX [CI] | clean loss | P1 vs 003-literal B0 | gain macro [CI] | gain FLUX | P2 | twin | cross R->V | cross V->R |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in R["fragility"].items():
            a, b, p2 = v["P1_vs_matched"], v["P1_vs_lit003"], v["P2"]
            gi = lambda d: f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
            L.append(f"| {k} | {'PASS' if a['passed'] else 'FAIL'}{' FRAGILE' if a['fragile'] else ''} | {gi(a['gain_macro'])} | {gi(a['gain_flux'])} | {a['clean_loss']:+.3f} | "
                     f"{'PASS' if b['passed'] else 'FAIL'}{' FRAGILE' if b['fragile'] else ''} | {gi(b['gain_macro'])} | {gi(b['gain_flux'])} | "
                     f"{'PASS' if p2['passed'] else 'FAIL'}{' FRAGILE' if p2['fragile'] else ''} | {p2['twin']:.3f} | {p2['cross_raise_to_vision']:.3f} | {p2['cross_vision_to_raise']:.3f} |")
        if "paired_vs_B0" in R:
            L += ["", "### Paired AUROC differences vs B0 (macro8; FLUX) at jpeg75_resize05 and held-out launderings, primary reals", "",
                  "| comparison | cond | macro8 delta [CI] | FLUX delta [CI] |", "|---|---|---|---|"]
            for k, v in R["paired_vs_B0"].items():
                m, c, pk = k.split(" | ")
                if pk != "primary":
                    continue
                gi = lambda d: f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
                L.append(f"| {m} | {c} | {gi(v['macro8'])} | {gi(v['flux'])} |")
        L.append("")
    (RES / "tables.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval", "all"], nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=list(CFG["arms"]))
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    if a.stage in ("fit", "all"):
        stage_fit(a.arms, a.workers)
    if a.stage in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
