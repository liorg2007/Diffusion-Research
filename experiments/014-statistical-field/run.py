"""E5 (Amendment 14): the statistical field of real vs diffusion images.

    env/.venv/bin/python experiments/014-statistical-field/run.py stats|tests|report|all [--workers 6]

Stages are resumable: `stats` caches one npz per (arm, condition); `tests` caches one pickle per component under cache/tests/;
`report` only reads caches and writes results/{results.json,tables.md,*.png}. Design and operationalisations: config.json, notes.md.
"""
import os
import sys

_STAGE = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # workers are single-threaded; parallelism is across processes

import gzip
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats as sps
from scipy.signal.windows import hann

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE, RES = HERE / "cache", HERE / "results"
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
CFG = json.loads((HERE / "config.json").read_text())
SEED = CFG["seed"]
CONDS = CFG["conditions"]
ARMS = list(CFG["arms"])
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ============================================================================== per-image statistics
def hp(x):
    """Valid-region high-pass residual x - mean(4 neighbours); shape (h-2, w-2)."""
    return x[1:-1, 1:-1] - 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])


def gray_of(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def s1_spectrum(g, fmin=1 / 64, fmax=1 / 4, nbins=16):
    h, w = g.shape
    win = np.outer(hann(h, sym=False), hann(w, sym=False))
    P = np.abs(np.fft.fft2((g - g.mean()) * win)) ** 2
    f = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2).ravel()
    edges = np.geomspace(fmin, fmax, nbins + 1)
    idx = np.digitize(f, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    cnt = np.bincount(idx[ok], minlength=nbins).astype(float)
    pw = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=nbins)
    fc = np.bincount(idx[ok], weights=f[ok], minlength=nbins)
    k = cnt > 0
    lf, lp = np.log10(fc[k] / cnt[k]), np.log10(pw[k] / cnt[k] + 1e-12)
    slope, icpt = np.polyfit(lf, lp, 1)
    return {"s1_alpha": -slope, "s1_resid": float(np.sqrt(np.mean((lp - (slope * lf + icpt)) ** 2)))}


def haar_step(a):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    return (p + q + r + s) / 2, (p + q - r - s) / 2, (p - q + r - s) / 2, (p - q - r + s) / 2


def s2_wavelet(g, levels=4):
    out, a = {}, g
    for l in range(1, levels + 1):
        a, H, V, D = haar_step(a)
        for nm, d in (("H", H), ("V", V), ("D", D)):
            out[f"s2_l{l}_{nm}_logvar"] = float(np.log10(d.var() + 1e-8))
            out[f"s2_l{l}_{nm}_kurt"] = float(sps.kurtosis(d.ravel(), fisher=True))
    return out


def nlf_fit(g, nbins=8):
    """Noise-level function. Returns slope/intercept of sigma (MAD) and of sigma^2 vs mean intensity/255."""
    r = hp(g)
    sm = ndimage.uniform_filter(g, 5)[1:-1, 1:-1]
    order = np.argsort(sm.ravel(), kind="stable")
    rs, ms = r.ravel()[order], sm.ravel()[order]
    chunks = np.array_split(np.arange(rs.size), nbins)
    m = np.array([ms[c].mean() for c in chunks]) / 255.0
    sig = np.array([1.4826 * np.median(np.abs(rs[c] - np.median(rs[c]))) / np.sqrt(1.25) for c in chunks])
    s1, i1 = np.polyfit(m, sig, 1)
    s2, i2 = np.polyfit(m, sig ** 2, 1)
    return {"s3_slope_sd": s1, "s3_icpt_sd": i1, "s3_slope_var": s2, "s3_icpt_var": i2}


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def s4_colour(rgb):
    hr, hg, hb = hp(rgb[..., 0]), hp(rgb[..., 1]), hp(rgb[..., 2])
    e = np.array([(hg[i::2, j::2] ** 2).mean() for i in (0, 1) for j in (0, 1)])    # e00,e01,e10,e11
    tot = e.sum()
    return {"s4_corr_dRG_dBG": corr(hr - hg, hb - hg), "s4_corr_RG": corr(hr, hg), "s4_corr_BG": corr(hb, hg),
            "s4_phase_cv": float(e.std() / e.mean()) if e.mean() > 0 else np.nan,
            "s4_phase_diag": float(abs(e[0] + e[3] - e[1] - e[2]) / tot) if tot > 0 else np.nan}


def s5_local(g, ps=16, levels=32):
    h, w = g.shape
    gh, gw = h // ps, w // ps
    P = g[: gh * ps, : gw * ps].reshape(gh, ps, gw, ps).transpose(0, 2, 1, 3).reshape(gh * gw, ps * ps)
    q = np.clip((P / 256.0 * levels).astype(int), 0, levels - 1)
    cnt = np.bincount((q + np.arange(gh * gw)[:, None] * levels).ravel(), minlength=gh * gw * levels).reshape(gh * gw, levels)
    p = cnt / cnt.sum(1, keepdims=True)
    ent = -(p * np.log2(np.where(p > 0, p, 1))).sum(1)
    lc = P.std(1)
    return {"s5_entropy": float(ent.mean()), "s5_lc_mean": float(lc.mean()), "s5_lc_std": float(lc.std())}


def image_stats(rgb):
    """rgb: uint8 or float array (h, w, 3). Returns an ordered dict of the 38 statistics."""
    rgb = np.asarray(rgb, np.float64)
    g = gray_of(rgb)
    out = {}
    out.update(s1_spectrum(g))
    out.update(s2_wavelet(g))
    out.update(nlf_fit(g))
    out.update(s4_colour(rgb))
    out.update(s5_local(g))
    return out


STAT_NAMES = list(image_stats(np.random.default_rng(0).integers(0, 255, (64, 64, 3))).keys())


def group_of(name):
    if name.startswith("s1_"):
        return "S1"
    if name.startswith("s2_l1_"):
        return "S2L1"
    if name.startswith("s2_"):
        return "S2L2-4"
    return name[:2].upper()


GROUPS = {n: group_of(n) for n in STAT_NAMES}
FINE = set(CFG["fine_scale_F"])


# ============================================================================== stage 1: stats
def _work(args):
    path, sha, cond = args
    from PIL import Image
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        out = apply_condition(im, cond, key=sha, seed=SEED)
    d = image_stats(np.asarray(out.convert("RGB")))
    return [d[k] for k in STAT_NAMES]


def load_df(arm):
    df = pd.read_csv(ROOT / CFG["arms"][arm]["splits"], keep_default_na=False, dtype={"vidx": str})
    df["rid"] = df.src_path.map(lambda p: (re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1])
    df["rpool"] = df.pool.map({"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH", "raise": "RAISE", "fake": "fake"})
    df["rgroup"] = np.where(df.label == 1, "", np.where(df.pool == "raise", "raise_" + df.rid, df.device))
    return df


def stage_stats(workers):
    from multiprocessing import Pool
    (CACHE / "stats").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        df = load_df(arm)
        for cond in CONDS:
            fp = CACHE / "stats" / f"{arm}__{cond}.npz"
            if fp.exists():
                log("stats cached", arm, cond)
                continue
            jobs = [(str(ROOT / p), s, cond) for p, s in zip(df.path, df.sha256)]
            with Pool(workers) as pool:
                rows = pool.map(_work, jobs, chunksize=16)
            np.savez(fp, sha=df.sha256.values.astype(str), X=np.array(rows, np.float64), names=np.array(STAT_NAMES))
            log("stats done", arm, cond, len(rows))
    # long csv
    parts = []
    for arm in ARMS:
        df = load_df(arm)
        for cond in CONDS:
            z = np.load(CACHE / "stats" / f"{arm}__{cond}.npz")
            t = pd.DataFrame(z["X"], columns=STAT_NAMES)
            t.insert(0, "cond", cond), t.insert(0, "arm", arm)
            for c in ("rid", "generator", "rpool", "label", "sha256"):
                t.insert(0, c, df[c].values)
            parts.append(t)
    RES.mkdir(exist_ok=True)
    pd.concat(parts).to_csv(RES / "stats_long.csv.gz", index=False, float_format="%.6g")
    log("wrote stats_long.csv.gz")


# ============================================================================== data access for tests
def get_stats(arm, cond):
    X = np.load(CACHE / "stats" / f"{arm}__{cond}.npz")["X"].copy()
    nan_frac = np.isnan(X).mean(0)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X) | np.isinf(X), med, X)
    return X, nan_frac


def zscore(X, clip=8.0):
    mu, sd = X.mean(0), X.std(0)
    return np.clip((X - mu) / np.where(sd > 0, sd, 1), -clip, clip)


def get_space(space, arm, cond, df):
    if space == "battery":
        return get_stats(arm, cond)[0]
    import p4features as P4
    X = P4.load_features_aligned(df, "secondary", cond, subdir=CFG["arms"][arm]["dino_subdir"])
    if np.isnan(X).any():
        raise RuntimeError(f"NaN DINOv2 features {arm} {cond}")
    H = 1536
    sl = {"dino_blk4": slice(0, H), "dino_blk8": slice(H, 2 * H), "dino_b0": slice(2 * H, 3 * H)}[space]
    return X[:, sl].astype(np.float64)


SPACES = ["battery", "dino_b0", "dino_blk4", "dino_blk8"]
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large", "flux"]
REAL_POOLS = ["all", "native", "WA", "FBH", "RAISE"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pool(workers, fn, jobs):
    from multiprocessing import Pool
    with Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def cached(name, fn):
    fp = CACHE / "tests" / f"{name}.pkl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        log("cached", name)
        return pickle.loads(fp.read_bytes())
    t = time.time()
    r = fn()
    fp.write_bytes(pickle.dumps(r))
    log(f"computed {name} in {time.time() - t:.0f}s")
    return r


# ============================================================================== test 1: atlas of Cliff's delta
def _atlas_job(args):
    from metrics import bootstrap_weights, weighted_auroc
    arm, cond = args
    df = load_df(arm)
    X, nanf = get_stats(arm, cond)
    real = (df.label == 0).values
    ir, ifk = np.where(real)[0], np.where(~real)[0]
    cl = np.where(df.device.values[ir] != "", df.device.values[ir], "raise_" + df.rid.values[ir])
    Wr, Wf = bootstrap_weights(cl, df.generator.values[ifk], 1000, SEED)
    one_r, one_f = np.ones((1, len(ir))), np.ones((1, len(ifk)))
    rows = []
    gen_arr = df.generator.values[ifk]
    for j, name in enumerate(STAT_NAMES):
        sr, sf = X[ir, j], X[ifk, j]
        for g in GENS + ["ALL"]:
            cols = np.ones(len(ifk), bool) if g == "ALL" else gen_arr == g
            for rp in REAL_POOLS:
                rm = np.ones(len(ir), bool) if rp == "all" else (df.rpool.values[ir] == rp)
                if rp == "all":
                    sr_, sf_ = sr, sf[cols]
                    pt = weighted_auroc(sr_, sf_, one_r, one_f[:, cols])[0]
                    bt = weighted_auroc(sr_, sf_, Wr, Wf[:, cols])
                    lo, hi = np.percentile(2 * bt - 1, [2.5, 97.5])
                else:
                    pt = weighted_auroc(sr[rm], sf[cols], np.ones((1, rm.sum())), one_f[:, cols])[0]
                    lo = hi = np.nan
                rows.append((arm, cond, name, g, rp, 2 * pt - 1, lo, hi))
    return rows, nanf.tolist()


def run_atlas(workers):
    def f():
        res = pool(workers, _atlas_job, [(a, c) for a in ARMS for c in CONDS])
        rows = [r for rr, _ in res for r in rr]
        nanf = {f"{a}|{c}": n for (a, c), (_, n) in zip([(a, c) for a in ARMS for c in CONDS], res)}
        return pd.DataFrame(rows, columns=["arm", "cond", "stat", "gen", "reals", "delta", "lo", "hi"]), nanf
    return cached("atlas", f)


# ============================================================================== test 2: MMD
def _mmd_job(args):
    space, arm, cond = args
    df = load_df(arm)
    Z = zscore(get_space(space, arm, cond, df))
    n2 = (Z ** 2).sum(1)
    D2 = np.maximum(n2[:, None] + n2[None, :] - 2 * Z @ Z.T, 0)
    iu = np.triu_indices(len(Z), 1)
    bw2 = np.median(np.sqrt(D2[iu])) ** 2                     # median heuristic: k = exp(-D^2 / (2 med(D)^2))
    K = np.exp(-D2 / (2 * bw2))
    rng = np.random.default_rng(SEED)
    out = []
    real = (df.label == 0).values
    for variant, rmask in (("all", real), ("native+RAISE", real & df.rpool.isin(["native", "RAISE"]).values)):
        for g in GENS:
            idx = np.where(rmask | (df.generator.values == g))[0]
            Ks = K[np.ix_(idx, idx)].copy()
            np.fill_diagonal(Ks, 0.0)
            isf = (df.generator.values[idx] == g)
            nf, nr = int(isf.sum()), int((~isf).sum())
            n = len(idx)
            k1 = Ks.sum(1)
            tot = k1.sum()
            P = CFG["mmd"]["n_perm"]
            V = np.zeros((n, P + 1))
            V[:, 0] = isf
            order = np.argsort(rng.random((n, P)), axis=0)[:nf]
            for c in range(P):
                V[order[:, c], c + 1] = 1.0
            KV = Ks @ V
            sff = (V * KV).sum(0)
            vk = V.T @ k1
            sfr = vk - sff
            srr = tot - 2 * vk + sff
            m = sff / (nf * (nf - 1)) + srr / (nr * (nr - 1)) - 2 * sfr / (nf * nr)
            p = (1 + np.sum(m[1:] >= m[0])) / (P + 1)
            out.append((space, arm, cond, variant, g, float(m[0]), float(p), nf, nr, float(np.sqrt(bw2))))
    return out


def run_mmd(workers):
    def f():
        res = pool(workers, _mmd_job, [(s, a, c) for s in SPACES for a in ARMS for c in CONDS])
        d = pd.DataFrame([r for rr in res for r in rr], columns=["space", "arm", "cond", "reals", "gen", "mmd2", "p", "n_fake", "n_real", "bandwidth"])
        d["p_holm"] = d.groupby(["space", "arm", "cond", "reals"], group_keys=False).p.apply(lambda s: pd.Series(holm(s.values), index=s.index))
        return d
    return cached("mmd", f)


# ============================================================================== test 3: C2ST (device- and generator-held-out)
def make_clf(space):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0 if space == "battery" else 0.01, class_weight="balanced", max_iter=1000)


def fit_score(space, Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1)
    f = lambda A: np.clip((A - mu) / sd, -8, 8)
    clf = make_clf(space).fit(f(Xtr), ytr)
    return clf.decision_function(f(Xte))


def deal(items, K, rng):
    items = list(items)
    rng.shuffle(items)
    return {it: i % K for i, it in enumerate(items)}


def c2st_oof(space, X, df, rmask, twin_disjoint, seed, K=4):
    """Out-of-fold scores for rows (reals in rmask + all fakes). Returns idx (row indices), score, fold."""
    rng = np.random.default_rng(1000 + seed)
    lab, gen, rid = df.label.values, df.generator.values, df.rid.values
    use = np.where(rmask | (lab == 1))[0]
    rf = deal(sorted(set(df.rgroup.values[use][lab[use] == 0])), K, rng)
    gf = deal(sorted(set(gen[use][lab[use] == 1])), K, rng)
    fold = np.array([rf[df.rgroup.values[i]] if lab[i] == 0 else gf[gen[i]] for i in use])
    score = np.full(len(use), np.nan)
    ndrop = 0
    for k in range(K):
        te, tr = fold == k, fold != k
        if twin_disjoint:
            t_real_rids = set(rid[use][te & (lab[use] == 0) & (rid[use] != "")])
            t_fake_rids = set(rid[use][te & (lab[use] == 1)])
            bad = ((lab[use] == 1) & np.isin(rid[use], list(t_real_rids))) | ((lab[use] == 0) & (rid[use] != "") & np.isin(rid[use], list(t_fake_rids)))
            ndrop += int((tr & bad).sum())
            tr = tr & ~bad
        score[te] = fit_score(space, X[use[tr]], lab[use[tr]], X[use[te]])
    return use, score, fold, ndrop


def _c2st_job(args):
    from metrics import grouped_bootstrap_auroc
    from sklearn.metrics import roc_auc_score
    space, arm, cond, rpool, twin_disjoint, R = args
    df = load_df(arm)
    X = get_space(space, arm, cond, df)
    isr = (df.label == 0).values
    rmask = isr if rpool == "all" else (isr & df.rpool.isin(rpool.split("+")).values)
    aucs, pergen, drops, boots = [], [], [], []
    for r in range(R):
        use, sc, fold, nd = c2st_oof(space, X, df, rmask, twin_disjoint, r)
        y = df.label.values[use]
        aucs.append(roc_auc_score(y, sc))
        g = df.generator.values[use]
        pergen.append({gg: roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones((g == gg).sum())], np.r_[sc[y == 0], sc[g == gg]]) for gg in GENS})
        drops.append(nd)
        y = df.label.values[use]
        bt = grouped_bootstrap_auroc(sc[y == 0], sc[y == 1], df.rgroup.values[use][y == 0], df.generator.values[use][y == 1], n_boot=1000, seed=SEED)
        boots.append(bt["boot"])                               # same rows and seed in every repeat -> the same resampling plan
    boot = np.mean(boots, 0)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return dict(space=space, arm=arm, cond=cond, reals=rpool, twin_disjoint=twin_disjoint, auroc=float(np.mean(aucs)), auroc_sd=float(np.std(aucs)),
                aurocs=[float(a) for a in aucs], ci_lo=float(ci_lo), ci_hi=float(ci_hi), per_gen={g: float(np.mean([d[g] for d in pergen])) for g in GENS},
                n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), n_dropped_train=float(np.mean(drops)))


def run_c2st(workers):
    def f():
        jobs = []
        for s in SPACES:
            for a in ARMS:
                for c in CONDS:
                    jobs.append((s, a, c, "all", True, 5))
        for s in ("battery", "dino_b0"):
            for a in ARMS:
                for c in CONDS:
                    jobs.append((s, a, c, "all", False, 3))
                    for rp in ("native", "WA", "FBH", "RAISE", "WA+RAISE"):
                        jobs.append((s, a, c, rp, True, 5))
        return pool(workers, _c2st_job, jobs)
    return cached("c2st", f)


# ============================================================================== test 4: caption twins
def twin_pairs(df):
    real = df[(df.label == 0) & (df.pool == "raise") & (df.rid != "")]
    rmap = dict(zip(real.rid, real.index))
    fk = df[(df.label == 1) & df.rid.isin(rmap)]
    return np.array([rmap[r] for r in fk.rid]), fk.index.values, fk.generator.values, fk.rid.values


def run_twin_sign(workers):
    def f():
        rows = []
        for arm in ARMS:
            df = load_df(arm)
            ri, fi, gens, rids = twin_pairs(df)
            for cond in CONDS:
                X, _ = get_stats(arm, cond)
                D = X[fi] - X[ri]                              # fake - real
                for g in GENS + ["ALL"]:
                    if g == "ALL":
                        t = pd.DataFrame(D, columns=STAT_NAMES)
                        t["rid"] = rids
                        Dg = t.groupby("rid").mean().values
                    else:
                        Dg = D[gens == g]
                    ps, med, win = [], [], []
                    for j in range(len(STAT_NAMES)):
                        d = Dg[:, j]
                        npos, nneg = int((d > 0).sum()), int((d < 0).sum())
                        p = sps.binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg > 0 else 1.0
                        ps.append(p), med.append(float(np.median(d))), win.append(npos / max(1, npos + nneg))
                    ph = holm(ps)
                    for j, nm in enumerate(STAT_NAMES):
                        rows.append((arm, cond, g, nm, len(Dg), win[j], med[j], ps[j], ph[j]))
        return pd.DataFrame(rows, columns=["arm", "cond", "gen", "stat", "n_pairs", "frac_fake_higher", "median_diff", "p", "p_holm"])
    return cached("twin_sign", f)


def _paired_job(args):
    from metrics import weighted_auroc
    from sklearn.metrics import roc_auc_score
    space, arm, cond, R = args
    df = load_df(arm)
    X = get_space(space, arm, cond, df)
    ri, fi, gens, rids = twin_pairs(df)
    uid = sorted(set(rids))
    uid_ix = {u: i for i, u in enumerate(uid)}
    pid = np.array([uid_ix[r] for r in rids])
    ugen = sorted(set(gens))
    real_of_id = {}
    for a, b in zip(pid, ri):
        real_of_id[a] = b
    K = 4
    res, boots = [], []
    rng = np.random.default_rng(SEED)
    B = 1000
    Wid = np.zeros((B, len(uid)))
    for bi in range(B):
        Wid[bi] = np.bincount(rng.integers(0, len(uid), len(uid)), minlength=len(uid))
    for r in range(R):
        rng = np.random.default_rng(2000 + r)
        idf = deal(range(len(uid)), K, rng)
        gf = deal(ugen, K, rng)
        id_fold = np.array([idf[i] for i in range(len(uid))])
        g_fold = np.array([gf[g] for g in gens])
        cells = []
        pergen_num, pergen_den = {g: 0.0 for g in ugen}, {g: 0 for g in ugen}
        for a in range(K):
            for b in range(K):
                trf = (id_fold[pid] != a) & (g_fold != b)
                tef = (id_fold[pid] == a) & (g_fold == b)
                if tef.sum() == 0:
                    continue
                # training reals: one per twin id with id not in fold a
                tr_real_ids = np.array([i for i in range(len(uid)) if id_fold[i] != a])
                Xtr = np.r_[X[[real_of_id[i] for i in tr_real_ids]], X[fi[trf]]]
                ytr = np.r_[np.zeros(len(tr_real_ids)), np.ones(trf.sum())]
                te_ids = np.unique(pid[tef])
                Xte = np.r_[X[[real_of_id[i] for i in te_ids]], X[fi[tef]]]
                sc = fit_score(space, Xtr, ytr, Xte)
                sr, sf = sc[: len(te_ids)], sc[len(te_ids):]
                auc = roc_auc_score(np.r_[np.zeros(len(sr)), np.ones(len(sf))], sc)
                pos = {i: k for k, i in enumerate(te_ids)}
                win = np.mean([sf[k] > sr[pos[pid[tef][k]]] for k in range(len(sf))]) if len(sf) else np.nan
                cells.append(dict(auc=auc, n=int(tef.sum()), win=float(win), sr=sr, sf=sf, te_ids=te_ids, f_ids=pid[tef], gens=gens[tef]))
                for g in np.unique(gens[tef]):
                    m = gens[tef] == g
                    pergen_num[g] += roc_auc_score(np.r_[np.zeros(len(sr)), np.ones(m.sum())], np.r_[sr, sf[m]]) * m.sum()
                    pergen_den[g] += int(m.sum())
        n = sum(c["n"] for c in cells)
        res.append(dict(auc=sum(c["auc"] * c["n"] for c in cells) / n, win=sum(c["win"] * c["n"] for c in cells) / n,
                        per_gen={g: pergen_num[g] / pergen_den[g] for g in ugen if pergen_den[g]}))
        tot, nn = np.zeros(B), 0
        for c in cells:
            tot += weighted_auroc(c["sr"], c["sf"], Wid[:, c["te_ids"]], Wid[:, c["f_ids"]]) * c["n"]
            nn += c["n"]
        boots.append(tot / nn)
    boot = np.mean(boots, 0)                                    # bootstrap over twin ids, same plan in every repeat
    return dict(space=space, arm=arm, cond=cond, auroc=float(np.mean([r["auc"] for r in res])), auroc_sd=float(np.std([r["auc"] for r in res])),
                win_rate=float(np.mean([r["win"] for r in res])), ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                per_gen={g: float(np.mean([r["per_gen"][g] for r in res if g in r["per_gen"]])) for g in ugen},
                n_pairs=int(len(fi)), n_ids=len(uid), n_pairs_per_gen={g: int((gens == g).sum()) for g in ugen})


def run_twin_c2st(workers):
    def f():
        return pool(workers, _paired_job, [(s, a, c, 5) for s in SPACES for a in ARMS for c in CONDS])
    return cached("twin_c2st", f)


def stage_tests(workers):
    run_atlas(workers)
    run_mmd(workers)
    run_c2st(workers)
    run_twin_sign(workers)
    run_twin_c2st(workers)


# ============================================================================== stage 3: report (predictions, tables, figures)



def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(mod=None, t0=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    atlas, nanf = run_atlas(1)
    mmd = run_mmd(1)
    c2 = pd.DataFrame(run_c2st(1))
    tsign = run_twin_sign(1)
    tc2 = pd.DataFrame(run_twin_c2st(1))
    LAUND = ["jpeg75_resize05", "social_sim"]
    A = atlas[(atlas.reals == "all") & atlas.gen.isin(GENS)]
    macro = A.assign(a=A.delta.abs()).groupby(["arm", "cond", "stat"]).a.mean().unstack("stat")          # arm,cond x stat
    grp_of = pd.Series(GROUPS)
    gmean = lambda arm, cond: macro.loc[(arm, cond)].groupby(grp_of).mean()
    res = {"predictions": {}, "deviations_and_interpretations": []}
    T = []                                                                                          # tables.md pieces

    # ---------------- P3
    p3 = {}
    for arm in ARMS:
        d = {}
        mc = macro.loc[(arm, "clean")]
        top5 = mc.sort_values(ascending=False).index[:5].tolist()
        inF = [GROUPS[t] in FINE for t in top5]
        Fmean = mc[[n for n in STAT_NAMES if GROUPS[n] in FINE]].mean()
        nFmean = mc[[n for n in STAT_NAMES if GROUPS[n] not in FINE]].mean()
        d["top5_clean"] = top5
        d["n_top5_in_F"] = int(sum(inF))
        d["meanabs_F_clean"], d["meanabs_notF_clean"] = float(Fmean), float(nFmean)
        d["P3a"] = bool(sum(inF) >= 3 and Fmean > nFmean)
        for cond in LAUND:
            gr = gmean(arm, cond) / gmean(arm, "clean")
            d[f"ratio_{cond}"] = {k: float(v) for k, v in gr.items()}
            d[f"P3b_{cond}"] = bool(min(gr["S1"], gr["S5"]) > max(gr["S2L1"], gr["S3"], gr["S4"]))
        d["P3_" + LAUND[0]] = bool(d["P3a"] and d["P3b_" + LAUND[0]])
        d["group_meanabs"] = {c: {k: float(v) for k, v in gmean(arm, c).items()} for c in CONDS}
        p3[arm] = d
    res["predictions"]["P3"] = {"verdict_orig": p3["orig"]["P3_" + LAUND[0]], **{a: p3[a] for a in ARMS}}

    # ---------------- P1 (twin C2ST)
    def trow(space, arm, cond):
        r = tc2[(tc2.space == space) & (tc2.arm == arm) & (tc2.cond == cond)].iloc[0]
        return r
    p1 = {}
    for arm in ARMS:
        r = trow("battery", arm, "clean")
        p1[arm] = dict(auroc=r.auroc, ci=[r.ci_lo, r.ci_hi], sd_repeats=r.auroc_sd, win_rate=r.win_rate, dino_b0=float(trow("dino_b0", arm, "clean").auroc),
                       confirmed=bool(r.auroc >= 0.75), ci_lower_ge_075=bool(r.ci_lo >= 0.75))
    res["predictions"]["P1"] = {"verdict_orig": p1["orig"]["confirmed"], **p1}

    # ---------------- P2
    def c2row(space, arm, cond, reals="all", td=True):
        q = c2[(c2.space == space) & (c2.arm == arm) & (c2.cond == cond) & (c2.reals == reals) & (c2.twin_disjoint == td)]
        return q.iloc[0]
    p2 = {}
    for arm in ARMS:
        d = {}
        for cond in LAUND:
            b, dn = c2row("battery", arm, cond), c2row("dino_b0", arm, cond)
            mm = {}
            for sp in ("battery", "dino_b0"):
                q = mmd[(mmd.space == sp) & (mmd.arm == arm) & (mmd.cond == cond) & (mmd.reals == "all")]
                mm[sp] = dict(n_sig=int((q.p_holm < 0.01).sum()), min_p_holm=float(q.p_holm.min()))
            d[cond] = dict(battery_c2st=float(b.auroc), dino_b0_c2st=float(dn.auroc), mmd=mm,
                           confirmed=bool(b.auroc < 0.70 and dn.auroc < 0.70 and mm["battery"]["n_sig"] == 0 and mm["dino_b0"]["n_sig"] == 0),
                           parts=dict(battery_lt_070=bool(b.auroc < 0.70), dino_lt_070=bool(dn.auroc < 0.70), mmd_battery_ns=mm["battery"]["n_sig"] == 0,
                                      mmd_dino_ns=mm["dino_b0"]["n_sig"] == 0))
        p2[arm] = d
    res["predictions"]["P2"] = {"verdict_orig_jpeg75_resize05": p2["orig"]["jpeg75_resize05"]["confirmed"], **p2}

    # ---------------- P4
    base = c2row("battery", "orig", "clean", "native")
    p4 = {}
    for cond in CONDS:
        b = c2row("battery", "orig", cond, "native")
        t = c2row("battery", "fov512", cond, "WA+RAISE")
        p4[cond] = dict(base_orig_native=float(b.auroc), treat_fov512_WA_RAISE=float(t.auroc), delta=float(b.auroc - t.auroc), confirmed=bool(b.auroc - t.auroc >= 0.05),
                        variants={f"{a}|{r}": float(c2row("battery", a, cond, r).auroc) for a in ARMS for r in ["native", "WA", "FBH", "RAISE", "WA+RAISE"]})
    res["predictions"]["P4"] = {"verdict_clean": p4["clean"]["confirmed"], **p4}

    # ---------------- tables
    gens_cols = GENS + ["ALL"]
    for arm in ARMS:
        for cond in CONDS:
            a = atlas[(atlas.arm == arm) & (atlas.cond == cond) & (atlas.reals == "all")]
            rows = []
            for st in STAT_NAMES:
                q = a[a.stat == st].set_index("gen")
                cells = [f"{q.loc[g, 'delta']:+.2f}{'*' if (q.loc[g, 'lo'] > 0 or q.loc[g, 'hi'] < 0) else ''}" for g in gens_cols]
                rows.append([st, GROUPS[st]] + cells + [f"{macro.loc[(arm, cond), st]:.2f}"])
            T.append(f"### Atlas of Cliff's delta, arm={arm}, condition={cond} (all reals vs generator; + = fake larger; * = 95% cluster-bootstrap CI excludes 0)\n\n" +
                     md_table(["stat", "grp"] + [g[:12] for g in gens_cols] + ["mean abs (8 gens)"], rows))
    T.append("### Group summary: mean |delta| over 8 generators and statistics of a group; ratio = laundered / clean\n\n" + md_table(
        ["arm", "group", "clean", "jpeg75_resize05", "social_sim", "ratio jpeg75_resize05", "ratio social_sim"],
        [[arm, g] + [f"{gmean(arm, c)[g]:.3f}" for c in CONDS] + [f"{gmean(arm, c)[g] / gmean(arm, 'clean')[g]:.2f}" for c in LAUND] for arm in ARMS for g in ["S1", "S2L1", "S2L2-4", "S3", "S4", "S5"]]))
    top = []
    for arm in ARMS:
        for cond in CONDS:
            s_ = macro.loc[(arm, cond)].sort_values(ascending=False)
            top.append([arm, cond] + [f"{n} ({v:.2f})" for n, v in s_.iloc[:5].items()])
    T.append("### Top-5 statistics by mean |delta| over the 8 generators\n\n" + md_table(["arm", "cond", "1", "2", "3", "4", "5"], top))
    # real-pool breakdown
    rp_rows = []
    for arm in ARMS:
        for cond in CONDS:
            for rp in REAL_POOLS:
                q = atlas[(atlas.arm == arm) & (atlas.cond == cond) & (atlas.reals == rp) & atlas.gen.isin(GENS)]
                gm = q.assign(a=q.delta.abs()).groupby(["gen", "stat"]).a.mean().groupby("stat").mean().groupby(grp_of).mean()
                rp_rows.append([arm, cond, rp] + [f"{gm[g]:.3f}" for g in ["S1", "S2L1", "S2L2-4", "S3", "S4", "S5"]] + [f"{q.assign(a=q.delta.abs()).a.mean():.3f}"])
    T.append("### Mean |delta| by real sub-pool (group means over 8 generators)\n\n" + md_table(["arm", "cond", "reals", "S1", "S2L1", "S2L2-4", "S3", "S4", "S5", "all stats"], rp_rows))
    # MMD
    mrows = []
    for sp in SPACES:
        for arm in ARMS:
            for cond in CONDS:
                for var in ("all", "native+RAISE"):
                    q = mmd[(mmd.space == sp) & (mmd.arm == arm) & (mmd.cond == cond) & (mmd.reals == var)]
                    mrows.append([sp, arm, cond, var, f"{q.mmd2.mean():.4f}", f"{int((q.p_holm < 0.01).sum())}/8", f"{q.p_holm.min():.3f}", f"{q.p.min():.4f}"])
    T.append("### MMD (RBF, median heuristic, 1000 permutations): generators significant at 0.01 after Holm\n\n" + md_table(
        ["space", "arm", "cond", "reals", "mean MMD^2", "n sig (Holm .01)", "min p_holm", "min raw p (floor 0.001)"], mrows))
    # C2ST
    crow = []
    for sp in SPACES:
        for arm in ARMS:
            crow.append([sp, arm] + [f"{c2row(sp, arm, c).auroc:.3f} [{c2row(sp, arm, c).ci_lo:.3f},{c2row(sp, arm, c).ci_hi:.3f}] sd {c2row(sp, arm, c).auroc_sd:.3f}" for c in CONDS])
    T.append("### C2ST AUROC (device- and generator-held-out, twin-disjoint; mean of 5 fold repeats, CI = percentile of the repeat-averaged bootstrap AUROC)\n\n" + md_table(["space", "arm"] + CONDS, crow))
    sens = [[sp, arm, c, f"{c2row(sp, arm, c, 'all', False).auroc:.3f}", f"{c2row(sp, arm, c).auroc:.3f}"] for sp in ("battery", "dino_b0") for arm in ARMS for c in CONDS]
    T.append("### C2ST sensitivity: plain folds (no twin pruning, 3 repeats) vs twin-disjoint\n\n" + md_table(["space", "arm", "cond", "plain", "twin-disjoint"], sens))
    pg = [[sp, arm, c] + [f"{c2row(sp, arm, c).per_gen[g]:.2f}" for g in GENS] for sp in ("battery", "dino_b0") for arm in ARMS for c in CONDS]
    T.append("### C2ST per-generator AUROC (pooled out-of-fold scores, all reals vs one generator)\n\n" + md_table(["space", "arm", "cond"] + [g[:10] for g in GENS], pg))
    prow = [[sp, arm, c] + [f"{c2row(sp, arm, c, r).auroc:.3f}" for r in ["all", "native", "WA", "FBH", "RAISE", "WA+RAISE"]] for sp in ("battery", "dino_b0") for arm in ARMS for c in CONDS]
    T.append("### C2ST by real sub-pool (reals = pool only, all 8 generators as fakes)\n\n" + md_table(["space", "arm", "cond", "all", "native", "WA", "FBH", "RAISE", "WA+RAISE"], prow))
    # twin
    trw = []
    for sp in SPACES:
        for arm in ARMS:
            trw.append([sp, arm] + [f"{trow(sp, arm, c).auroc:.3f} [{trow(sp, arm, c).ci_lo:.3f},{trow(sp, arm, c).ci_hi:.3f}] win {trow(sp, arm, c).win_rate:.2f}" for c in CONDS])
    ex = tc2.iloc[0]
    T.append(f"### Twin-paired C2ST AUROC (RAISE real vs its own caption-twin fakes; {ex.n_pairs} pairs, {ex.n_ids} ids; id- and generator-held-out 4x4 cells; mean of 5 repeats; win = matched-pair rate fake>real)\n\n" +
             md_table(["space", "arm"] + CONDS, trw))
    tpg = [[sp, arm, c] + [f"{trow(sp, arm, c).per_gen.get(g, float('nan')):.2f}" for g in GENS] for sp in ("battery", "dino_b0") for arm in ARMS for c in CONDS]
    T.append("### Twin-paired C2ST per generator\n\n" + md_table(["space", "arm", "cond"] + [g[:10] for g in GENS], tpg))
    srows = []
    for arm in ARMS:
        for cond in CONDS:
            for g in gens_cols:
                q = tsign[(tsign.arm == arm) & (tsign.cond == cond) & (tsign.gen == g)]
                best = q.sort_values("p").iloc[:3]
                srows.append([arm, cond, g, int(q.n_pairs.iloc[0]), int((q.p_holm < 0.01).sum())] + [f"{r.stat} ({r.frac_fake_higher:.2f})" for r in best.itertuples()])
    T.append("### Twin sign tests (fake - real, per statistic; Holm over 38 stats): number significant at 0.01, and the 3 smallest-p statistics (fraction of pairs with fake larger)\n\n" +
             md_table(["arm", "cond", "gen", "pairs", "n sig/38", "1", "2", "3"], srows))
    # predictions
    pt = ["### Pre-registered predictions (computed in code)\n"]
    P1, P2, P3, P4 = (res["predictions"][k] for k in ("P1", "P2", "P3", "P4"))
    pt.append(f"- **P1** (twin battery C2ST >= 0.75, clean): orig {p1['orig']['auroc']:.3f} [{p1['orig']['ci'][0]:.3f},{p1['orig']['ci'][1]:.3f}] -> {'CONFIRMED' if p1['orig']['confirmed'] else 'REFUTED'}; "
              f"fov512 {p1['fov512']['auroc']:.3f} ({'confirmed' if p1['fov512']['confirmed'] else 'refuted'}); twin DINOv2-B0 {p1['orig']['dino_b0']:.3f}.")
    for arm in ARMS:
        for cond in LAUND:
            d = p2[arm][cond]
            pt.append(f"- **P2** ({arm}, {cond}): battery C2ST {d['battery_c2st']:.3f}, DINOv2-B0 C2ST {d['dino_b0_c2st']:.3f}, MMD generators significant (Holm 0.01) battery {d['mmd']['battery']['n_sig']}/8 "
                      f"DINOv2 {d['mmd']['dino_b0']['n_sig']}/8 -> {'CONFIRMED' if d['confirmed'] else 'REFUTED'}" + (" [pre-registered cell]" if arm == "orig" and cond == "jpeg75_resize05" else ""))
    for arm in ARMS:
        d = p3[arm]
        pt.append(f"- **P3** ({arm}): top-5 clean {d['top5_clean']}; in F: {d['n_top5_in_F']}/5; mean|d| F {d['meanabs_F_clean']:.3f} vs not-F {d['meanabs_notF_clean']:.3f} -> P3a {d['P3a']}; "
                  f"ratios(jpeg75_resize05) " + ", ".join(f"{k} {v:.2f}" for k, v in d["ratio_jpeg75_resize05"].items()) + f" -> P3b {d['P3b_jpeg75_resize05']}; "
                  f"social_sim P3b {d['P3b_social_sim']}; P3 = {'CONFIRMED' if d['P3_jpeg75_resize05'] else 'REFUTED'}")
    for cond in CONDS:
        d = p4[cond]
        pt.append(f"- **P4** ({cond}): C2ST battery orig natives {d['base_orig_native']:.3f} - fov512 WA+RAISE {d['treat_fov512_WA_RAISE']:.3f} = {d['delta']:+.3f} -> {'CONFIRMED' if d['confirmed'] else 'REFUTED'}" + (" [pre-registered cell]" if cond == "clean" else ""))
    T.insert(0, "\n".join(pt))
    (RES / "tables.md").write_text("# E5 tables (generated by run.py report)\n\n" + "\n\n".join(T) + "\n")

    # ---------------- results.json
    res["top5_by_meanabs"] = {f"{a}|{c}": macro.loc[(a, c)].sort_values(ascending=False).iloc[:5].round(4).to_dict() for a in ARMS for c in CONDS}
    res["c2st"] = c2.drop(columns=["aurocs"]).to_dict("records")
    res["twin_c2st"] = tc2.to_dict("records")
    res["mmd_summary"] = mmd.groupby(["space", "arm", "cond", "reals"]).apply(lambda q: dict(mean_mmd2=float(q.mmd2.mean()), n_sig_holm_001=int((q.p_holm < 0.01).sum()), min_p_holm=float(q.p_holm.min()))).reset_index(name="v").to_dict("records")
    res["twin_sign_n_sig"] = tsign.groupby(["arm", "cond", "gen"]).apply(lambda q: dict(n_pairs=int(q.n_pairs.iloc[0]), n_sig_holm_001=int((q.p_holm < 0.01).sum()))).reset_index(name="v").to_dict("records")
    res["stat_nan_fraction_max"] = {k: float(np.max(v)) for k, v in nanf.items()}
    res["statistics"] = STAT_NAMES
    res["runtime_s_report"] = time.time() - (t0 or T0)
    atlas.to_csv(RES / "atlas_long.csv.gz", index=False, float_format="%.5g")
    (RES / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    # ---------------- figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 9), sharey=True)
    for ax, cond in zip(axs, ["clean", "jpeg75_resize05"]):
        a = atlas[(atlas.arm == "orig") & (atlas.cond == cond) & (atlas.reals == "all")]
        M = np.array([[a[(a.stat == st) & (a.gen == g)].delta.iloc[0] for g in gens_cols] for st in STAT_NAMES])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(gens_cols)), [g[:9] for g in gens_cols], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"orig, {cond}")
    axs[0].set_yticks(range(len(STAT_NAMES)), STAT_NAMES, fontsize=6)
    fig.colorbar(im, ax=axs, shrink=0.5, label="Cliff delta (+ = fake larger)")
    fig.savefig(RES / "fig1_effect_heatmap_orig.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arm in zip(axs, ARMS):
        w = 0.2
        for i, sp in enumerate(SPACES):
            v = [c2row(sp, arm, c).auroc for c in CONDS]
            e = [[c2row(sp, arm, c).auroc - c2row(sp, arm, c).ci_lo for c in CONDS], [c2row(sp, arm, c).ci_hi - c2row(sp, arm, c).auroc for c in CONDS]]
            ax.bar(np.arange(3) + i * w, v, w, yerr=e, label=sp, capsize=2)
        ax.axhline(0.5, color="k", lw=0.6), ax.axhline(0.70, color="r", ls="--", lw=0.8)
        ax.set_xticks(np.arange(3) + 1.5 * w, CONDS), ax.set_title(f"C2ST AUROC, {arm}"), ax.set_ylim(0.4, 1.0)
    axs[0].legend(fontsize=7)
    fig.savefig(RES / "fig2_c2st_by_space_condition.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arm in zip(axs, ARMS):
        for i, sp in enumerate(SPACES):
            v = [trow(sp, arm, c).auroc for c in CONDS]
            e = [[trow(sp, arm, c).auroc - trow(sp, arm, c).ci_lo for c in CONDS], [trow(sp, arm, c).ci_hi - trow(sp, arm, c).auroc for c in CONDS]]
            ax.bar(np.arange(3) + i * 0.2, v, 0.2, yerr=e, label=sp, capsize=2)
        ax.axhline(0.5, color="k", lw=0.6), ax.axhline(0.75, color="r", ls="--", lw=0.8)
        ax.set_xticks(np.arange(3) + 0.3, CONDS), ax.set_title(f"twin-paired C2ST AUROC, {arm}"), ax.set_ylim(0.4, 1.0)
    axs[0].legend(fontsize=7)
    fig.savefig(RES / "fig3_twin_c2st.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    grs = ["S1", "S2L1", "S2L2-4", "S3", "S4", "S5"]
    for i, c in enumerate(CONDS):
        ax.bar(np.arange(6) + i * 0.27, [gmean("orig", c)[g] for g in grs], 0.27, label=c)
    ax.set_xticks(np.arange(6) + 0.27, grs), ax.set_ylabel("mean |Cliff delta|"), ax.set_title("orig: effect size by statistic group"), ax.legend(fontsize=7)
    fig.savefig(RES / "fig4_group_effects.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("report written")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "tests", "report", "all"])
    ap.add_argument("--workers", type=int, default=CFG["parallel"]["workers"])
    a = ap.parse_args()
    if a.stage in ("stats", "all"):
        stage_stats(a.workers)
    if a.stage in ("tests", "all"):
        stage_tests(a.workers)
    if a.stage in ("report", "all"):
        build_report(sys.modules[__name__], T0)


if __name__ == "__main__":
    main()
