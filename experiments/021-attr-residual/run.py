"""H-A2: training-free residual fingerprints (docs/08 H-A2, docs/09). CPU only.

    env/.venv/bin/python experiments/021-attr-residual/run.py gen        # (i) generator attribution
    env/.venv/bin/python experiments/021-attr-residual/run.py dev        # (ii) VISION device attribution (PRNU/PCE)
    env/.venv/bin/python experiments/021-attr-residual/run.py report     # results.json + tables.md + labels
Heavy intermediates (fingerprint sums, crops) go to --scratch (outside the repo); results/ holds only small files.
All design constants are in config.json (frozen before any result).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import perturb  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text())
RES = HERE / "results"
SEED = CFG["seed"]
RID_RE = re.compile(r"(r[0-9a-f]{8,}t)")  # same pattern as experiments/015-midlayer-probe/run.py (E6 caption family)
GENS = CFG["generators"]
CONDS = CFG["gen"]["conditions"]
DEVS = CFG["dev"]["devices"]
EPS = CFG["dev"]["eps"]
Image.MAX_IMAGE_PIXELS = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================= residual primitives
def wiener_denoise(x, size=3, mode="reflect"):
    """scipy.signal.wiener(x, size) formula (noise = mean local variance), with selectable border mode (scipy: zero padding)."""
    x = np.asarray(x, dtype=np.float64)
    m = ndimage.uniform_filter(x, size, mode=mode)
    v = ndimage.uniform_filter(x * x, size, mode=mode) - m * m
    noise = v.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (x - m) * (1.0 - noise / v) + m
    return np.where(v < noise, m, res)


def zero_mean_rc(r):
    """Subtract row means then column means (afterwards both are zero)."""
    r = r - r.mean(axis=1, keepdims=True)
    return r - r.mean(axis=0, keepdims=True)


def residual(x):
    x = np.asarray(x, dtype=np.float64)
    return zero_mean_rc(x - wiener_denoise(x))


# ============================================================================= bootstrap helpers
def boot_weights(cluster, n_boot, seed, strata=None):
    """(n_boot, n) integer weights: clusters resampled with replacement (within strata if given)."""
    cluster = np.asarray(cluster)
    rng = np.random.default_rng(seed)
    n = len(cluster)
    w = np.zeros((n_boot, n), np.float32)
    groups = [np.arange(n)] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    for g in groups:
        cl, inv = np.unique(cluster[g], return_inverse=True)
        k = len(cl)
        draws = rng.integers(0, k, (n_boot, k))
        counts = np.stack([np.bincount(d, minlength=k) for d in draws])
        w[:, g] = counts[:, inv]
    return w


def macro_recall(correct, cls, w=None, n_cls=None):
    """balanced accuracy = mean over classes of weighted recall (nan classes ignored). w: (n,) or (B, n)."""
    n_cls = int(cls.max()) + 1 if n_cls is None else n_cls
    w2 = np.ones((1, len(cls)), np.float64) if w is None else np.atleast_2d(np.asarray(w, np.float64))
    num = np.stack([(w2 * (cls == c) * correct).sum(1) for c in range(n_cls)], 1)
    den = np.stack([(w2 * (cls == c)).sum(1) for c in range(n_cls)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    out = np.nanmean(r, axis=1)
    return out[0] if w is None or np.ndim(w) == 1 else out


def ci(samples):
    return [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))]


# ============================================================================= (i) generators: scoring
def fit_fingerprints(R, y, K):
    """R: (n, P) float32 residuals; y: (n,) class codes -> (K, P) mean residual per class (one-hot matmul)."""
    oh = np.zeros((K, len(y)), np.float32)
    oh[y, np.arange(len(y))] = 1
    cnt = np.maximum(oh.sum(1, keepdims=True), 1)
    return ((oh @ R) / cnt).astype(np.float32)


def ncc_scores(R, F):
    """normalised correlation <r,f>/(|r||f|): (n, K)."""
    num = R @ F.T
    nr = np.linalg.norm(R, axis=1, keepdims=True)
    nf = np.linalg.norm(F, axis=1)[None]
    return num / np.maximum(nr * nf, 1e-12)


def spectra(R, shape):
    """|FFT2| of each residual, flattened: (n, P)."""
    n = len(R)
    out = np.empty((n, int(np.prod(shape))), np.float32)
    for i in range(n):
        out[i] = np.abs(np.fft.fft2(R[i].reshape(shape))).ravel()
    return out


def spectral_scores(Rtr, ytr, Rte, K, shape):
    Str, Ste = spectra(Rtr, shape), spectra(Rte, shape)
    G = Str.mean(0)
    F = fit_fingerprints(Str, ytr, K) - G
    return ncc_scores(Ste - G, F)


def pairwise_items(S, gen, fam, K):
    """Per-family numerators / denominators of the pairwise metric and per generator-pair totals.
    For family f with generators A<B: share of (a in A, b in B) pairs with margin m = S[:,A]-S[:,B] larger for a than for b."""
    fams = np.unique(fam)
    num, den = np.zeros(len(fams)), np.zeros(len(fams))
    pn, pd_ = np.zeros((K, K)), np.zeros((K, K))
    for fi, f in enumerate(fams):
        idx = np.where(fam == f)[0]
        gens = np.unique(gen[idx])
        for ia, A in enumerate(gens):
            for B in gens[ia + 1:]:
                m = S[:, A] - S[:, B]
                a, b = idx[gen[idx] == A], idx[gen[idx] == B]
                d = m[a][:, None] - m[b][None, :]
                c = (d > 0).sum() + 0.5 * (d == 0).sum()
                num[fi] += c
                den[fi] += d.size
                pn[A, B] += c
                pd_[A, B] += d.size
    return fams, num, den, pn, pd_


def pairwise_boot(num, den, w_fam):
    """ratio of weighted sums; w_fam: (B, n_fam)."""
    return (w_fam * num).sum(1) / np.maximum((w_fam * den).sum(1), 1e-12)


def split_families(fam_ids, frac, seed):
    fams = np.array(sorted(set(fam_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fams))
    n_tr = int(round(frac * len(fams)))
    train = set(fams[perm[:n_tr]])
    return np.array([f in train for f in fam_ids])


def check_grouped(fam, train_mask, test_mask):
    """Gate 3: no caption family on both sides of the split."""
    both = set(fam[train_mask]) & set(fam[test_mask])
    if both:
        raise AssertionError(f"split not grouped: {len(both)} families on both sides")
    if (train_mask & test_mask).any():
        raise AssertionError("row in both train and test")


def check_dedupe(hashes, label):
    """Gate 5: no duplicate content hash (any class)."""
    s = pd.Series(hashes)
    d = s[s.duplicated(keep=False)]
    if len(d):
        raise AssertionError(f"{label}: {len(d)} rows share a content hash")


def perm_control(Rtr, ytr, Rte, yte, K, n_perm, seed):
    """Bacc of the NCC-argmax rule when the TRAIN labels are shuffled (fingerprints from random groups)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    nr = np.linalg.norm(Rte, axis=1, keepdims=True)
    for i in range(n_perm):
        F = fit_fingerprints(Rtr, rng.permutation(ytr), K)
        S = (Rte @ F.T) / np.maximum(nr * np.linalg.norm(F, axis=1)[None], 1e-12)
        out[i] = macro_recall(S.argmax(1) == yte, yte, n_cls=K)
    return out


def eval_generators(R, M1, meta, train, test, cfg, n_boot, n_perm, seed, shape, do_extra=True):
    """R: {cond: (n, P) float32}; M1: {cond: (n, F)}; meta: DataFrame(gen code 'g', family 'fam')."""
    K = len(GENS) if "n_cls" not in cfg else cfg["n_cls"]
    g, fam = meta.g.to_numpy(), meta.fam.to_numpy()
    check_grouped(fam, train, test)
    tr, te = np.where(train)[0], np.where(test)[0]
    yte, fte = g[te], fam[te]
    cl = np.unique(fte, return_inverse=True)[1]
    w = boot_weights(cl, n_boot, seed + 11)
    mult = np.stack([_family_mult(cl, w[b]) for b in range(n_boot)])  # family multiplicities per draw
    out = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for cond in R:
        for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
            fcond = "clean" if src == "clean" else cond
            F = fit_fingerprints(R[fcond][tr], g[tr], K)
            S = ncc_scores(R[cond][te], F)
            pred = S.argmax(1)
            corr = pred == yte
            point = macro_recall(corr, yte, n_cls=K)
            bs = macro_recall(corr, yte, w=w, n_cls=K)
            cm = np.zeros((K, K), int)
            np.add.at(cm, (yte, pred), 1)
            rec = [float(corr[yte == k].mean()) if (yte == k).any() else None for k in range(K)]
            fams_, num, den, pn, pd_ = pairwise_items(S, yte, fte, K)
            pw = pairwise_boot(num, den, mult)
            ent = {"bacc": float(point), "bacc_ci": ci(bs), "acc": float(corr.mean()), "recall": rec, "confusion": cm.tolist(),
                   "pairwise": float(num.sum() / max(den.sum(), 1)), "pairwise_ci": ci(pw),
                   "pairwise_by_pair": (pn / np.maximum(pd_, 1)).tolist(), "n_test": int(len(te))}
            # spectral variant (not judged)
            if do_extra:
                Ss = spectral_scores(R[fcond][tr], g[tr], R[cond][te], K, shape)
                cs = Ss.argmax(1) == yte
                ent["spectral_bacc"] = float(macro_recall(cs, yte, n_cls=K))
                ent["spectral_bacc_ci"] = ci(macro_recall(cs, yte, w=w, n_cls=K))
                # M1 texture baseline, same train/test protocol
                sc = StandardScaler().fit(M1[fcond][tr])
                lr = LogisticRegression(C=cfg["m1_C"], max_iter=3000).fit(sc.transform(M1[fcond][tr]), g[tr])
                c1 = lr.predict(sc.transform(M1[cond][te])) == yte
                ent["m1_bacc"] = float(macro_recall(c1, yte, n_cls=K))
                m1b = macro_recall(c1, yte, w=w, n_cls=K)
                ent["m1_bacc_ci"] = ci(m1b)
                ent["diff_vs_m1"] = float(point - ent["m1_bacc"])
                ent["diff_vs_m1_ci"] = ci(bs - m1b)
                # K=7 without firefly (refit)
                keep_cls = [k for k in range(K) if GENS[k] != "firefly"] if K == len(GENS) else list(range(K))
                trk = np.isin(g[tr], keep_cls)
                tek = np.isin(yte, keep_cls)
                remap = {k: i for i, k in enumerate(keep_cls)}
                F7 = fit_fingerprints(R[fcond][tr][trk], np.array([remap[v] for v in g[tr][trk]]), len(keep_cls))
                S7 = ncc_scores(R[cond][te][tek], F7)
                y7 = np.array([remap[v] for v in yte[tek]])
                ent["bacc_without_firefly"] = float(macro_recall(S7.argmax(1) == y7, y7, n_cls=len(keep_cls)))
                # permutation control
                perm = perm_control(R[fcond][tr], g[tr], R[cond][te], yte, K, n_perm, seed + 5)
                ent["perm"] = {"mean": float(perm.mean()), "ci": ci(perm), "max": float(perm.max()),
                               "p_ge_real": float((1 + (perm >= point).sum()) / (1 + len(perm)))}
            out[f"{cond}|fp={src}"] = ent
    return out


def _family_mult(cl, wrow):
    """multiplicity of each cluster (family) in one bootstrap draw: weight of the first member row."""
    k = cl.max() + 1
    first = np.zeros(k, int)
    seen = np.zeros(k, bool)
    for i, c in enumerate(cl):
        if not seen[c]:
            first[c], seen[c] = i, True
    return wrow[first]


# ============================================================================= (i) generators: feature extraction
def m1_features(rgb, gray, r):
    f = [ndimage.laplace(gray).var()]
    dh, dv = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    bh = np.arange(dh.shape[1]) % 8 == 7
    bv = np.arange(dv.shape[0]) % 8 == 7
    f.append(0.5 * (dh[:, bh].mean() / (dh[:, ~bh].mean() + 1e-6) + dv[bv].mean() / (dv[~bv].mean() + 1e-6)))
    f.append(r.std())
    mx, mn = rgb.max(2).astype(float), rgb.min(2).astype(float)
    f.append(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0).mean())
    f.append(((rgb == 0) | (rgb == 255)).any(2).mean())
    for c in range(3):
        f += [rgb[..., c].mean(), rgb[..., c].std()]
    return np.array(f, np.float64)


def gen_worker(args):
    path, sha, conds, crop = args
    img = Image.open(ROOT / path)
    img.load()
    img = img.convert("RGB")  # format from content (Firefly: JPEG under .png); EXIF never propagated
    out = {}
    for c in conds:
        im = perturb.apply_condition(img, c, key=sha, seed=0)
        if im.size[0] < crop or im.size[1] < crop:
            raise RuntimeError(f"{path} {c}: {im.size} < {crop}")
        cr = im.crop((0, 0, crop, crop))
        rgb = np.asarray(cr)
        gray = np.asarray(cr.convert("L"), dtype=np.float64)
        r = zero_mean_rc(gray - wiener_denoise(gray))
        out[c] = (r.astype(np.float32).ravel(), m1_features(rgb, gray, r), hashlib.sha1(rgb.tobytes()).hexdigest())
    return out


def load_caption_ids(df):
    """Reuse the E6 caption-family code (015 caption_ids); fall back to the identical regex if its imports are unavailable."""
    rid = np.array([(RID_RE.search(Path(p).name) or [None, ""])[1] for p in df.path])
    try:
        spec = importlib.util.spec_from_file_location("e6run", ROOT / "experiments" / "015-midlayer-probe" / "run.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        r6 = m.caption_ids(df.assign(src_path=df.path))
        assert (r6 == rid).all()
        return r6, "015-midlayer-probe/run.py caption_ids"
    except Exception as e:  # pragma: no cover
        return rid, f"regex copy of E6 RID_RE ({type(e).__name__})"


def stage_gen(scratch, workers):
    t0 = time.time()
    man = pd.read_csv(ROOT / "data" / "manifest_raw.csv", keep_default_na=False)
    df = man[(man.label == 1) & man.generator.isin(GENS)].reset_index(drop=True)
    assert len(df) == 1020 and df.generator.nunique() == 8, (len(df), df.generator.nunique())
    rid, rid_src = load_caption_ids(df)
    assert (rid != "").all()
    df["fam"], df["g"] = rid, df.generator.map({g: i for i, g in enumerate(GENS)})
    check_dedupe(df.sha256.tolist(), "gen sha256 (fakes)")
    cache = scratch / "gen_feats.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        R = {c: z[f"R_{c}"] for c in CONDS}
        M1 = {c: z[f"M1_{c}"] for c in CONDS}
        H = {c: z[f"H_{c}"] for c in CONDS}
    else:
        log("gen: extracting crops/residuals", len(df), "images x", len(CONDS), "conditions")
        args = [(p, s, CONDS, CFG["gen"]["crop"]) for p, s in zip(df.path, df.sha256)]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(gen_worker, args, chunksize=8))
        R = {c: np.stack([r[c][0] for r in res]) for c in CONDS}
        M1 = {c: np.stack([r[c][1] for r in res]) for c in CONDS}
        H = {c: np.array([r[c][2] for r in res]) for c in CONDS}
        np.savez(cache, **{f"R_{c}": R[c] for c in CONDS}, **{f"M1_{c}": M1[c] for c in CONDS}, **{f"H_{c}": H[c] for c in CONDS})
    for c in CONDS:  # crop-level duplicate check (identical decoded pixels, any class)
        check_dedupe(H[c].tolist(), f"gen crop hash [{c}]")
    shape = (CFG["gen"]["crop"],) * 2
    n_boot, n_perm = CFG["n_boot"], CFG["n_perm"]
    train = split_families(df.fam.to_numpy(), CFG["gen"]["train_family_frac"], CFG["gen"]["primary_split_seed"])
    test = ~train
    meta = df[["g", "fam"]].copy()
    log(f"gen: {train.sum()} train / {test.sum()} test rows; families {len(set(df.fam[train]))}/{len(set(df.fam[test]))}")
    main = eval_generators(R, M1, meta, train, test, CFG["gen"], n_boot, n_perm, SEED, shape)
    # sensitivity over split seeds (point estimates only)
    sens = {}
    for s in CFG["gen"]["sensitivity_split_seeds"]:
        tr_s = split_families(df.fam.to_numpy(), CFG["gen"]["train_family_frac"], s)
        te_s = ~tr_s
        tri, tei = np.where(tr_s)[0], np.where(te_s)[0]
        for cond in CONDS:
            for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
                fc = "clean" if src == "clean" else cond
                F = fit_fingerprints(R[fc][tri], meta.g.to_numpy()[tri], 8)
                S = ncc_scores(R[cond][tei], F)
                sens.setdefault(f"{cond}|fp={src}", []).append(float(macro_recall(S.argmax(1) == meta.g.to_numpy()[tei], meta.g.to_numpy()[tei], n_cls=8)))
    sens = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)} for k, v in sens.items()}
    # M0 raw-metadata baseline (random forest on dims / bytes-per-pixel / true format) and M0 on preprocessed tensors (constant)
    from sklearn.ensemble import RandomForestClassifier
    X0 = np.c_[df.width, df.height, df.width / df.height, df.bytes / (df.width * df.height), (df.orig_format == "JPEG").astype(int)].astype(float)
    g_ = meta.g.to_numpy()
    rf = RandomForestClassifier(300, random_state=SEED, n_jobs=1).fit(X0[train], g_[train])
    c0 = rf.predict(X0[test]) == g_[test]
    cl = np.unique(df.fam.to_numpy()[test], return_inverse=True)[1]
    w = boot_weights(cl, n_boot, SEED + 11)
    m0 = {"raw_files_bacc": float(macro_recall(c0, g_[test], n_cls=8)), "raw_files_bacc_ci": ci(macro_recall(c0, g_[test], w=w, n_cls=8))}
    from sklearn.linear_model import LogisticRegression
    const = np.zeros((len(df), 1))
    lr = LogisticRegression().fit(const[train], g_[train])
    m0["tensor_bacc"] = float(macro_recall(lr.predict(const[test]) == g_[test], g_[test], n_cls=8))
    m0["tensor_note"] = "preprocessed 256x256 crops have identical shape/dtype/format for every class -> metadata features are constant"
    per_gen_counts = pd.Series(g_[test]).value_counts().sort_index().to_dict()
    out = {"stage": "gen", "main": main, "sensitivity_seeds": sens, "m0": m0, "n_train": int(train.sum()), "n_test": int(test.sum()),
           "n_families_train": int(len(set(df.fam[train]))), "n_families_test": int(len(set(df.fam[test]))),
           "test_per_generator": {GENS[int(k)]: int(v) for k, v in per_gen_counts.items()}, "caption_family_source": rid_src,
           "gates": {"sha256_unique_fakes": True, "crop_hash_unique": True, "split_grouped": True}, "runtime_s": time.time() - t0}
    (RES / "gen_results.json").write_text(json.dumps(out, indent=1))
    log("gen done", round(time.time() - t0), "s")


# ============================================================================= (ii) devices: PRNU / PCE
def pce_stats(W, Y):
    """Normalised circular cross-correlation rho(s) = sum_x W(x) Y(x+s) / (|W||Y|) (Y mean-removed) and the PCE at zero shift.
    Returns (pce_signed, rho0, peak_at_zero(bool), peak_shift)."""
    Y = Y - Y.mean()
    nw, ny = np.sqrt((W.astype(np.float64) ** 2).sum()), np.sqrt((Y.astype(np.float64) ** 2).sum())
    Fw = sfft.rfft2(W)
    Fy = sfft.rfft2(Y)
    c = sfft.irfft2(np.conj(Fw) * Fy, s=W.shape) / (nw * ny)
    return pce_from_corr(c)


def pce_from_corr(c):
    h, w = c.shape
    rho0 = float(c[0, 0])
    sq = c.astype(np.float64) ** 2
    ri = np.r_[0:6, h - 5:h]
    ci_ = np.r_[0:6, w - 5:w]
    noise = (sq.sum() - sq[np.ix_(ri, ci_)].sum()) / (h * w - len(ri) * len(ci_))
    pk = np.unravel_index(int(np.argmax(c)), c.shape)
    return float(np.sign(rho0) * rho0 ** 2 / noise), rho0, bool(pk == (0, 0)), (int(pk[0]), int(pk[1]))


def check_lopo(member_ids, test_id):
    """Gate 5: the test photo (any platform copy) must not contribute to the fingerprint it is scored against."""
    if test_id in set(member_ids):
        raise AssertionError(f"test photo {test_id} is in the fingerprint members")


def prnu_perm_control(W, I, lab, G, n_perm, seed, h, w, eps=EPS, return_real=True, N=None, D=None):
    """Permutation control with a cheap statistic: zero-shift NCC (not PCE) on crops of shape (h, w).
    W, I: (n, h*w) float32 (residual zero-meaned over the crop, intensity). lab: (n,) device codes 0..G-1.
    Train labels are shuffled (group sizes preserved); LOPO fingerprint of the shuffled group; decision argmax over groups; accuracy vs TRUE labels."""
    n, P = W.shape
    N = (W * I).astype(np.float32) if N is None else N
    D = (I * I).astype(np.float32) if D is None else D
    nW = np.array([np.sqrt((np.asarray(W[i], np.float64) ** 2).sum()) for i in range(n)])  # row-wise: a full float64 copy of W per process caused an OOM (bug fix, see notes.md)
    rng = np.random.default_rng(seed)

    def one(labs):
        oh = np.zeros((G, n), np.float32)
        oh[labs, np.arange(n)] = 1
        Ng, Dg = oh @ N, oh @ D
        Kg = np.stack([zero_mean_rc((Ng[g] / (Dg[g] + eps)).reshape(h, w)).ravel() for g in range(G)]).astype(np.float32)
        num = N @ Kg.T
        sy = I @ Kg.T
        sy2 = D @ (Kg ** 2).T
        S = num / (nW[:, None] * np.sqrt(np.maximum(sy2 - sy ** 2 / P, 1e-12)))
        for p in range(n):  # LOPO: own (shuffled) group without the test photo
            g = labs[p]
            Ko = zero_mean_rc(((Ng[g] - N[p]) / (np.maximum(Dg[g] - D[p], 0) + eps)).reshape(h, w)).ravel().astype(np.float32)
            y = I[p] * Ko
            S[p, g] = float((W[p] * y).sum() / (nW[p] * np.sqrt(max(float((y.astype(np.float64) ** 2).sum()) - float(y.sum(dtype=np.float64)) ** 2 / P, 1e-12))))
        return S.argmax(1)

    def macc(pred):
        return float(np.mean([(pred[lab == d] == d).mean() for d in range(G)]))

    real = macc(one(lab)) if return_real else None
    perm = np.array([macc(one(rng.permutation(lab))) for _ in range(n_perm)])
    return real, perm


# ---- VISION table
def vision_table():
    man = pd.read_csv(ROOT / "data" / "manifest_raw.csv", keep_default_na=False)
    v = man[man.source_dataset == "VISION"].copy()
    v["dev"] = v.camera_or_platform.str[:3]
    v["plat"] = v.camera_or_platform.str.split("_", n=1).str[1]
    v["idx"] = v.path.str.extract(r"_(\d+)\.jpg$")[0].astype(int)
    v = v[v.dev.isin(DEVS)]
    nat = v[v.plat == "native"]
    sz = nat.assign(wh=[tuple(sorted((a, b))) for a, b in zip(nat.width, nat.height)]).groupby("dev").wh.agg(lambda s: set(s))
    assert all(s == {(2448, 3264)} for s in sz), sz.to_dict()  # long side 3264, short 2448 for all 14 (portrait files count)
    check_dedupe(v.sha256.tolist(), "VISION sha256 (14 devices, nat+WA+FBH)")
    t = nat[["dev", "idx", "path", "width", "height"]].rename(columns={"path": "nat"})
    for plat, col in (("whatsapp", "wa"), ("facebook_hq", "fbh")):
        s = v[v.plat == plat][["dev", "idx", "path"]].rename(columns={"path": col})
        t = t.merge(s, on=["dev", "idx"], how="left")
    assert t[["wa", "fbh"]].notna().all().all(), "incomplete nat/WA/FBH triplets"
    t["exif"] = [Image.open(ROOT / p).getexif().get(274) or 1 for p in t.nat]
    return t.sort_values(["dev", "idx"]).reset_index(drop=True)


UNDO_EXIF = {1: 0, 2: 0, 3: 2, 6: 1, 8: 3}  # k of np.rot90 (CCW) that undoes the orientation the platform applied to WA/FBH copies


def load_gray_raw(path):
    im = Image.open(ROOT / path)
    im.load()
    return np.asarray(im.convert("L"), dtype=np.float64)  # raw stored pixels (no EXIF transpose)


def canon(arr, plat, exif, k_extra):
    """Canonical (sensor, landscape) orientation. nat: stored pixels; wa/fbh: undo the native EXIF orientation the platform applied.
    If still portrait: rotate by k_extra (1 or 3; resolved per photo for D33, whose portrait natives are stored portrait)."""
    if plat != "nat":
        arr = np.rot90(arr, UNDO_EXIF.get(int(exif), 0))
    if arr.shape[0] > arr.shape[1]:
        arr = np.rot90(arr, k_extra if k_extra in (1, 3) else 1)
    return np.ascontiguousarray(arr)


def acc_worker(args):
    """Accumulate N = sum r*x, D = sum x^2 over the photos of one device in one platform arm; save K (zero-mean) too."""
    scratch, arm, dev, rows = args
    N = D = None
    for r in rows:
        x = canon(load_gray_raw(r["path"]), arm, r["exif"], r["k_extra"])
        rr = residual(x)
        if N is None:
            N, D = np.zeros(x.shape, np.float64), np.zeros(x.shape, np.float64)
        N += rr * x
        D += x * x
    np.save(scratch / f"N_{arm}_{dev}.npy", N.astype(np.float32))
    np.save(scratch / f"D_{arm}_{dev}.npy", D.astype(np.float32))
    np.save(scratch / f"K_{arm}_{dev}.npy", zero_mean_rc(N / (D + EPS)).astype(np.float32))
    return arm, dev, N.shape


def stage_dev_resolve(tab, scratch):
    """Resolve the portrait direction (k_extra) per photo for devices with portrait natives."""
    ke = {}
    c = 1024
    for dev, g in tab.groupby("dev"):
        land = [r.idx for r in g.itertuples() if r.height < r.width]
        port = [r.idx for r in g.itertuples() if r.height > r.width]
        for r in g.itertuples():
            ke[(dev, r.idx)] = 0
        if not port:
            continue
        raw = {r.idx: load_gray_raw(r.nat) for r in g.itertuples()}
        if len(land) < 3:
            log(f"resolve {dev}: only {len(land)} landscape natives -> k_extra=1 for portraits")
            for i in port:
                ke[(dev, i)] = 1
            continue

        def crop(a):
            h, w = a.shape
            return a[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
        Ns, Ds = 0, 0
        for i in land:
            x = crop(raw[i])
            Ns = Ns + residual(x) * x
            Ds = Ds + x * x
        K = zero_mean_rc(Ns / (Ds + EPS))
        for i in port:
            sc = {}
            for k in (1, 3):
                x = crop(np.rot90(raw[i], k))
                Wr = residual(x)
                y = x * K
                sc[k] = float((Wr * (y - y.mean())).sum() / (np.linalg.norm(Wr) * np.linalg.norm(y - y.mean())))
            ke[(dev, i)] = 1 if sc[1] >= sc[3] else 3
            log(f"resolve {dev} idx {i}: ncc k=1 {sc[1]:.4f} k=3 {sc[3]:.4f} -> k_extra {ke[(dev, i)]}")
    return ke


def thumb_corr(a, b):
    from PIL import Image as I
    a = np.asarray(I.fromarray(a.astype(np.float32)).resize((64, 48), I.BILINEAR))
    b = np.asarray(I.fromarray(b.astype(np.float32)).resize((64, 48), I.BILINEAR))
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def precheck_worker(r):
    n = canon(load_gray_raw(r["nat"]), "nat", r["exif"], r["k_extra"])
    out = [n.shape == (2448, 3264)]
    for arm in ("wa", "fbh"):
        a = canon(load_gray_raw(r[arm]), arm, r["exif"], r["k_extra"])
        out += [a.shape, thumb_corr(n, a)]
    return out


def resize_fp(K, hw):
    """Resize a fingerprint (float) to (h, w) with PIL bicubic (antialiased) and re-zero-mean rows/cols."""
    return zero_mean_rc(np.asarray(Image.fromarray(np.asarray(K, np.float32)).resize((hw[1], hw[0]), Image.BICUBIC), np.float64))


def lopo_K(Nd, Dd, x_fp, r_fp):
    """Leave-one-photo-out fingerprint: remove the test photo's own contribution (r*x, x^2) from the device sums."""
    return zero_mean_rc((Nd - (r_fp * x_fp).astype(np.float32)) / (np.maximum(Dd - (x_fp * x_fp).astype(np.float32), 0) + EPS))


def pce_candidates(W_t, x_t, Ks, variants):
    """For each candidate fingerprint (iterable of arrays): best (max signed PCE) over the orientation variants
    (0 = as is, 2 = test residual and image rotated by 180 deg). -> list of (pce, rho0, peak_at_zero, (dy, dx), variant)."""
    out = []
    for K in Ks:
        best = None
        for v in variants:
            Wv, Iv = (W_t, x_t) if v == 0 else (np.rot90(W_t, 2), np.rot90(x_t, 2))
            Wv = np.ascontiguousarray(Wv, dtype=np.float32)
            Y = (np.ascontiguousarray(Iv) * K).astype(np.float32)
            st = pce_stats(Wv, Y)
            if best is None or st[0] > best[0]:
                best = st + (v,)
        out.append(best)
    return out


def test_worker(args):
    scratch, arm, r, pce_thr = args
    fp_arm = "nat" if arm == "nat2wa" else arm
    dev = r["dev"]
    x_fp = canon(load_gray_raw(r[fp_arm]), fp_arm, r["exif"], r["k_extra"])
    r_fp = residual(x_fp)
    Nd = np.load(scratch / f"N_{fp_arm}_{dev}.npy", mmap_mode="r")
    Dd = np.load(scratch / f"D_{fp_arm}_{dev}.npy", mmap_mode="r")
    K_own = lopo_K(Nd, Dd, x_fp, r_fp)
    if arm == "nat2wa":
        x_t = canon(load_gray_raw(r["wa"]), "wa", r["exif"], r["k_extra"])
        W_t = residual(x_t)
        shape = x_t.shape  # (960, 1280)
        rs = lambda K: resize_fp(K, shape)  # noqa: E731
    else:
        x_t, W_t = x_fp, r_fp
        rs = lambda K: np.asarray(K, np.float64)  # noqa: E731
    portrait_raw = arm == "nat" and r["height"] > r["width"]
    variants = [0, 2] if (arm != "nat" or portrait_raw) else [0]
    Ks = (rs(K_own) if d == dev else rs(np.load(scratch / f"K_{fp_arm}_{d}.npy", mmap_mode="r")) for d in DEVS)
    res = pce_candidates(W_t, x_t, Ks, variants)
    return [{"arm": arm, "dev": dev, "idx": r["idx"], "cand": d, "pce": b[0], "rho0": b[1], "peak_zero": b[2],
             "peak_shift": f"{b[3][0]},{b[3][1]}", "variant": b[4]} for d, b in zip(DEVS, res)]


def verify_lopo_subtraction(scratch, tab, dev="D02"):
    """Gate: recompute the device sum explicitly WITHOUT one photo and compare with (sum - that photo's term)."""
    g = tab[tab.dev == dev]
    rows = g.to_dict("records")
    t = rows[0]
    N = D = 0
    for r in rows[1:]:
        x = canon(load_gray_raw(r["nat"]), "nat", r["exif"], r["k_extra"])
        N = N + residual(x) * x
        D = D + x * x
    x = canon(load_gray_raw(t["nat"]), "nat", t["exif"], t["k_extra"])
    Nd = np.load(scratch / f"N_nat_{dev}.npy", mmap_mode="r")
    Dd = np.load(scratch / f"D_nat_{dev}.npy", mmap_mode="r")
    dn = float(np.abs((Nd - (residual(x) * x).astype(np.float32)) - N).max())
    dd = float(np.abs((Dd - (x * x).astype(np.float32)) - D).max())
    return {"device": dev, "left_out_idx": int(t["idx"]), "max_abs_diff_N": dn, "max_abs_diff_D": dd,
            "rel_D": dd / float(np.abs(D).max()), "ok": bool(dd / float(np.abs(D).max()) < 1e-5 and dn / float(np.abs(N).max()) < 1e-4)}


def prep_perm_crop(args):
    scratch, r, c = args
    x = canon(load_gray_raw(r["nat"]), "nat", r["exif"], r["k_extra"])
    h, w = x.shape
    x = x[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    return x.astype(np.float32), residual(x).astype(np.float32)


def perm_worker(args):
    scratch, h, w, lab, seeds_n, seed, G = args
    W, I, N, D = (np.load(scratch / f"perm_{k}.npy", mmap_mode="r") for k in "WIND")
    _, perm = prnu_perm_control(W, I, lab, G, seeds_n, seed, h, w, return_real=False, N=N, D=D)
    return perm


def real_ncc_worker(args):
    scratch, h, w, lab, G = args
    W, I, N, D = (np.load(scratch / f"perm_{k}.npy", mmap_mode="r") for k in "WIND")
    real, _ = prnu_perm_control(W, I, lab, G, 0, 0, h, w, N=N, D=D)
    return real


def stage_dev(scratch, workers):
    t0 = time.time()
    tab = vision_table()
    log(f"dev: {len(tab)} photos (triplets) over {tab.dev.nunique()} devices")
    ke = stage_dev_resolve(tab, scratch)
    tab["k_extra"] = [ke[(d, i)] for d, i in zip(tab.dev, tab.idx)]
    rows = tab.to_dict("records")
    with ProcessPoolExecutor(workers) as ex:
        pc = list(ex.map(precheck_worker, rows, chunksize=4))
    corr = np.array([[p[2], p[4]] for p in pc])
    shapes_ok = all(p[0] for p in pc) and all(p[1] == (960, 1280) and p[3] == (1536, 2048) for p in pc)
    log(f"dev: orientation precheck: native shapes ok & canonical WA/FBH shapes ok = {shapes_ok}; thumbnail corr nat-vs-WA min {corr[:,0].min():.3f}, nat-vs-FBH min {corr[:,1].min():.3f}")
    assert shapes_ok and corr.min() > 0.8, "canonical orientation does not align native and WA/FBH copies"
    # --- stage A: per-device accumulation for the three arms
    jobs = [(scratch, arm, dev, [{"path": r[arm], "exif": r["exif"], "k_extra": r["k_extra"]} for r in rows if r["dev"] == dev])
            for arm in ("nat", "wa", "fbh") for dev in DEVS]
    if not all((scratch / f"K_{a}_{d}.npy").exists() for a in ("nat", "wa", "fbh") for d in DEVS):
        with ProcessPoolExecutor(workers) as ex:
            for a, d, s in ex.map(acc_worker, jobs):
                log("fingerprint sums", a, d, s)
    lopo_chk = verify_lopo_subtraction(scratch, tab)
    log("dev: LOPO subtraction check", lopo_chk)
    assert lopo_chk["ok"], lopo_chk
    # --- stage B: LOPO PCE
    if (RES / "prnu_scores.csv").exists():  # resume (the scoring takes ~20 min); delete the file to recompute
        sc = pd.read_csv(RES / "prnu_scores.csv")
        log("dev: reusing saved scores", len(sc))
    else:
        tasks = [(scratch, arm, r, CFG["dev"]["pce_threshold"]) for arm in CFG["dev"]["platform_arms"] for r in rows]
        tasks.sort(key=lambda t: t[1] != "nat")  # heavy native arm first
        scores = []
        with ProcessPoolExecutor(workers) as ex:
            for i, rr in enumerate(ex.map(test_worker, tasks, chunksize=1)):
                scores += rr
                if i % 100 == 0:
                    log(f"dev: scored {i + 1}/{len(tasks)} photos")
        sc = pd.DataFrame(scores)
        sc.to_csv(RES / "prnu_scores.csv", index=False)
        log("dev: scores saved", len(sc))
    # --- permutation control (cheap statistic, centre crop)
    c = CFG["dev"]["perm_crop"]
    nat_rows = [r for r in rows]
    with ProcessPoolExecutor(workers) as ex:
        crops = list(ex.map(prep_perm_crop, [(scratch, r, c) for r in nat_rows], chunksize=4))
    I = np.stack([a.ravel() for a, _ in crops]).astype(np.float32)
    W = np.stack([zero_mean_rc(b.astype(np.float64)).ravel() for _, b in crops]).astype(np.float32)
    np.save(scratch / "perm_I.npy", I)
    np.save(scratch / "perm_W.npy", W)
    np.save(scratch / "perm_N.npy", W * I)
    np.save(scratch / "perm_D.npy", I * I)
    lab = np.array([DEVS.index(r["dev"]) for r in nat_rows])
    G = len(DEVS)
    n_perm = CFG["n_perm"]
    nproc = min(workers, 12)
    split = [n_perm // nproc + (1 if i < n_perm % nproc else 0) for i in range(nproc)]
    with ProcessPoolExecutor(nproc + 1) as ex:
        fr = ex.submit(real_ncc_worker, (scratch, c, c, lab, G))
        fs = [ex.submit(perm_worker, (scratch, c, c, lab, n, SEED + 100 + i, G)) for i, n in enumerate(split)]
        real_ncc = fr.result()
        perm = np.concatenate([f.result() for f in fs])
    pc_ = {"stat": f"zero-shift NCC on centre {c}x{c} crop of the native, LOPO", "real_top1_macro": float(real_ncc), "perm_mean": float(perm.mean()),
           "perm_ci": ci(perm), "perm_max": float(perm.max()), "n_perm": int(len(perm)), "chance": 1 / G,
           "p_ge_real": float((1 + (perm >= real_ncc).sum()) / (1 + len(perm)))}
    (RES / "dev_perm_control.json").write_text(json.dumps(pc_, indent=1))
    tab.to_csv(RES / "vision_photos.csv", index=False)
    (RES / "dev_meta.json").write_text(json.dumps({"n_photos": len(tab), "orientation_precheck_min_corr": float(corr.min()),
                                                    "lopo_check": lopo_chk, "k_extra_nonzero": {f"{d}_{i}": int(k) for (d, i), k in ke.items() if k}, "runtime_s": time.time() - t0}, indent=1))
    log("dev done", round(time.time() - t0), "s")


# ============================================================================= report
def dev_metrics(sc, tab, n_boot):
    """Per arm: 14-way top-1, within-model accuracy, TPR/FPR at PCE thr, peak-at-zero check; stratified (by device) photo bootstrap."""
    sets = CFG["dev"]["same_model_sets"]
    dev2set = {d: s for s, ds in sets.items() for d in ds}
    thr = CFG["dev"]["pce_threshold"]
    out = {}
    for arm in CFG["dev"]["platform_arms"]:
        s = sc[sc.arm == arm]
        photos = s[["dev", "idx"]].drop_duplicates().reset_index(drop=True)
        piv = s.pivot_table(index=["dev", "idx"], columns="cand", values="pce").reindex(pd.MultiIndex.from_frame(photos))
        P = piv[DEVS].to_numpy()
        true = np.array([DEVS.index(d) for d in photos.dev])
        pred = P.argmax(1)
        c14 = pred == true
        # within-model
        wm = np.full(len(photos), np.nan)
        wm_chance = np.full(len(photos), np.nan)
        for i, d in enumerate(photos.dev):
            if d in dev2set:
                cands = [DEVS.index(x) for x in sets[dev2set[d]]]
                wm[i] = float(cands[int(np.argmax(P[i, cands]))] == true[i])
                wm_chance[i] = 1 / len(cands)
        w = boot_weights(np.arange(len(photos)), n_boot, SEED + 31, strata=true)
        top1 = macro_recall(c14, true, n_cls=len(DEVS))
        top1_b = macro_recall(c14, true, w=w, n_cls=len(DEVS))
        m = ~np.isnan(wm)
        wm_acc = macro_recall(wm[m], true[m], n_cls=len(DEVS))
        wm_b = macro_recall(wm[m], true[m], w=w[:, m], n_cls=len(DEVS))
        per_dev = {d: float(c14[true == i].mean()) for i, d in enumerate(DEVS)}
        wm_dev = {d: float(wm[(true == i) & m].mean()) for i, d in enumerate(DEVS) if d in dev2set}
        # TPR / FPR at PCE thr
        is_true = np.zeros_like(P, bool)
        is_true[np.arange(len(P)), true] = True
        same_model = np.zeros_like(P, bool)
        for i, d in enumerate(photos.dev):
            if d in dev2set:
                for x in sets[dev2set[d]]:
                    if x != d:
                        same_model[i, DEVS.index(x)] = True
        hit = P >= thr
        tpr = float(hit[is_true].mean())
        fpr_all = float(hit[~is_true].mean())
        fpr_same = float(hit[same_model].mean()) if same_model.any() else None
        fpr_diff = float(hit[~is_true & ~same_model].mean())
        # bootstrap TPR/FPR (photo-level, stratified)
        wt = w.astype(np.float64)
        tpr_b = (wt * hit[is_true]).sum(1) / wt.sum(1)
        fpr_b = np.stack([(wt[b][:, None] * (hit & ~is_true)).sum() / (wt[b].sum() * (len(DEVS) - 1)) for b in range(len(wt))])
        # peak-at-zero check on the true device candidate
        pk = s[s.dev == s.cand]
        hi = pk[pk.pce >= thr]
        peak_hi = float(hi.peak_zero.mean()) if len(hi) else None
        pk_all = float(pk.peak_zero.mean())
        # also: false candidates with PCE >= thr -> where is their peak
        fh = s[(s.dev != s.cand) & (s.pce >= thr)]
        out[arm] = {"n_photos": int(len(photos)), "top1": float(top1), "top1_ci": ci(top1_b), "chance_top1": 1 / len(DEVS),
                    "within_model": float(wm_acc), "within_model_ci": ci(wm_b), "chance_within_model": float(np.nanmean(wm_chance)),
                    "top1_per_device": per_dev, "within_model_per_device": wm_dev,
                    "tpr_at_thr": tpr, "tpr_ci": ci(tpr_b), "fpr_all_at_thr": fpr_all, "fpr_ci": ci(fpr_b),
                    "fpr_same_model": fpr_same, "fpr_diff_model": fpr_diff,
                    "peak_zero_true_all": pk_all, "peak_zero_true_pce_ge_thr": peak_hi, "n_true_pce_ge_thr": int(len(hi)),
                    "n_false_pce_ge_thr": int(len(fh)), "false_pce_ge_thr_peak_zero_frac": float(fh.peak_zero.mean()) if len(fh) else None,
                    "median_pce_true": float(np.median(pk.pce)), "median_pce_false": float(np.median(s[s.dev != s.cand].pce)),
                    "confusion": _conf(true, pred, len(DEVS)),
                    "within_model_by_set": {st: float(np.nanmean([wm[(true == DEVS.index(d)) & m].mean() for d in ds])) for st, ds in sets.items()}}
    return out


def _conf(true, pred, K):
    cm = np.zeros((K, K), int)
    np.add.at(cm, (true, pred), 1)
    return cm.tolist()


def stage_report():
    gen = json.loads((RES / "gen_results.json").read_text()) if (RES / "gen_results.json").exists() else None
    dev = None
    perm_c = None
    if (RES / "prnu_scores.csv").exists():
        sc = pd.read_csv(RES / "prnu_scores.csv")
        tab = pd.read_csv(RES / "vision_photos.csv")
        dev = dev_metrics(sc, tab, CFG["n_boot"])
        perm_c = json.loads((RES / "dev_perm_control.json").read_text())
    th = CFG["gen"]
    lab = {}
    if gen:
        cl = gen["main"]["clean|fp=clean"]
        b = cl["bacc"]
        lab["i"] = "PASS-i" if b >= th["pass_i_clean_bacc"] else ("KILL-i" if b < th["kill_i_clean_bacc"] else "INCONCLUSIVE-i")
        lab["i_m1_margin_met"] = bool(cl["diff_vs_m1_ci"][0] >= th["m1_margin"])
    if dev:
        nat = dev["nat"]
        d = CFG["dev"]
        ok = nat["top1"] >= d["pass_ii_native_top1"] and nat["within_model"] >= d["pass_ii_within_model"]
        lab["ii"] = "PASS-ii" if ok else ("KILL-ii" if nat["top1"] < d["kill_ii_native_top1"] else "INCONCLUSIVE-ii")
    # gates
    gates = {}
    if gen:
        pm = [v["perm"] for v in gen["main"].values()]
        gates["gen_perm_at_chance"] = all(abs(p["mean"] - 1 / 8) < 0.02 for p in pm)
        gates["gen_perm_max"] = max(p["max"] for p in pm)
        gates["m0_tensor_at_chance"] = abs(gen["m0"]["tensor_bacc"] - 1 / 8) < 1e-9
    if perm_c:
        gates["dev_perm_at_chance"] = abs(perm_c["perm_mean"] - perm_c["chance"]) < 0.02
    lab["gates"] = gates
    invalid = [k for k in ("gen_perm_at_chance", "m0_tensor_at_chance", "dev_perm_at_chance") if gates.get(k) is False]
    pi, pii = lab.get("i", ""), lab.get("ii", "")
    if invalid:
        lab["overall"] = "INVALID (" + ",".join(invalid) + ")"
    elif pi.startswith("PASS") or pii.startswith("PASS"):
        lab["overall"] = "PASS"
    elif pi.startswith("KILL") and pii.startswith("KILL"):
        lab["overall"] = "KILL"
    else:
        lab["overall"] = "INCONCLUSIVE"
    out = {"labels": lab, "gen": gen, "dev": dev, "dev_perm_control": perm_c}
    (RES / "results.json").write_text(json.dumps(out, indent=1))
    (RES / "tables.md").write_text(render_tables(out))
    log("labels:", json.dumps({k: v for k, v in lab.items() if k != "gates"}))


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def fci(v, c):
    return f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def render_tables(o):
    L = []
    lab = o["labels"]
    L += ["# 021-attr-residual: H-A2 tables", "", f"Labels: (i) **{lab.get('i')}**, (ii) **{lab.get('ii')}**, overall **{lab['overall']}**. "
          f"M1 margin (CI lower bound of bacc minus M1 >= 0.10, clean): {lab.get('i_m1_margin_met')}.", ""]
    g = o["gen"]
    if g:
        L += ["## (i) Generators, K=8, 256x256 top-left crop, chance 0.125",
              f"Train {g['n_train']} / test {g['n_test']} images; families {g['n_families_train']} / {g['n_families_test']}; test images per generator {g['test_per_generator']}.", "",
              "| condition | fingerprint from | bacc [95% CI] | spectral bacc | M1 bacc | bacc - M1 [CI] | bacc w/o Firefly (K=7) | pairwise [CI] | perm mean (max) | split-seed mean+-sd (10) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, e in g["main"].items():
            c, s = k.split("|fp=")
            sv = g["sensitivity_seeds"][k]
            L.append(f"| {c} | {s} | {fci(e['bacc'], e['bacc_ci'])} | {fci(e['spectral_bacc'], e['spectral_bacc_ci'])} | {fci(e['m1_bacc'], e['m1_bacc_ci'])} | "
                     f"{fci(e['diff_vs_m1'], e['diff_vs_m1_ci'])} | {e['bacc_without_firefly']:.3f} (chance 0.143) | {fci(e['pairwise'], e['pairwise_ci'])} | "
                     f"{e['perm']['mean']:.3f} ({e['perm']['max']:.3f}) | {sv['mean']:.3f} +- {sv['sd']:.3f} |")
        L += ["", f"M0 (raw-file metadata, random forest): bacc {fci(g['m0']['raw_files_bacc'], g['m0']['raw_files_bacc_ci'])}; M0 on the preprocessed 256x256 tensors: {g['m0']['tensor_bacc']:.3f} (constant features -> chance). "
              "Pairwise chance 0.5.", "", "### Per-generator recall (clean, clean fingerprints)", "| " + " | ".join(GENS) + " |", "|" + "---|" * 8,
              "| " + " | ".join(f3(x) for x in g["main"]["clean|fp=clean"]["recall"]) + " |", "", "### Confusion matrix, clean (rows true, cols predicted; order as above)"]
        for row, name in zip(g["main"]["clean|fp=clean"]["confusion"], GENS):
            L.append(f"- {name}: {row}")
        L.append("")
    d = o["dev"]
    if d:
        L += ["## (ii) VISION devices, 14-way, full resolution (native), PRNU/PCE, leave-one-photo-out", "",
              "| arm | photos | top-1 [CI] (chance 0.071) | within-model [CI] (chance %.3f) | TPR@PCE60 [CI] | FPR@PCE60 all [CI] | FPR same-model | FPR diff-model | peak at 0 shift (true dev; PCE>=60) | median PCE true / false |" % d["nat"]["chance_within_model"],
              "|---|---|---|---|---|---|---|---|---|---|"]
        names = {"nat": "native -> native", "wa": "WA -> WA", "fbh": "FBH -> FBH", "nat2wa": "native fingerprint resized -> WA"}
        for a, e in d.items():
            L.append(f"| {names[a]} | {e['n_photos']} | {fci(e['top1'], e['top1_ci'])} | {fci(e['within_model'], e['within_model_ci'])} | {fci(e['tpr_at_thr'], e['tpr_ci'])} | "
                     f"{fci(e['fpr_all_at_thr'], e['fpr_ci'])} | {f3(e['fpr_same_model'])} | {f3(e['fpr_diff_model'])} | {f3(e['peak_zero_true_pce_ge_thr'])} (n={e['n_true_pce_ge_thr']}); all true: {e['peak_zero_true_all']:.3f}; false PCE>=60: n={e['n_false_pce_ge_thr']} | {e['median_pce_true']:.1f} / {e['median_pce_false']:.1f} |")
        L += ["", "### Native arm: per-device top-1 and within-model accuracy", "| device | top-1 | within-model |", "|---|---|---|"]
        for dv in DEVS:
            L.append(f"| {dv} | {f3(d['nat']['top1_per_device'][dv])} | {f3(d['nat']['within_model_per_device'].get(dv))} |")
        L += ["", "Within-model by set (native): " + json.dumps({k: round(v, 3) for k, v in d["nat"]["within_model_by_set"].items()}), ""]
        pc = o["dev_perm_control"]
        L += ["## Permutation control (devices)", f"{pc['stat']}: true labels {pc['real_top1_macro']:.3f}; {pc['n_perm']} shuffles of train labels: mean {pc['perm_mean']:.3f}, 95% interval [{pc['perm_ci'][0]:.3f}, {pc['perm_ci'][1]:.3f}], max {pc['perm_max']:.3f} (chance {pc['chance']:.3f}); p(perm >= real) = {pc['p_ge_real']:.4f}.", ""]
    return "\n".join(L)


# ============================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "dev", "report", "all"])
    ap.add_argument("--scratch", default=os.environ.get("H2_SCRATCH", "/tmp/021-scratch"))
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    if a.stage in ("gen", "all"):
        stage_gen(scratch, a.workers)
    if a.stage in ("dev", "all"):
        stage_dev(scratch, a.workers)
    if a.stage in ("report", "all"):
        stage_report()
