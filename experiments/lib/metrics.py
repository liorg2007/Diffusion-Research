"""Detection metrics with seeded bootstrap CIs. Convention: higher score = more likely fake (label 1)."""
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

KEYS = ["auroc", "ap", "bal_acc", "bal_acc_fpr5", "tpr_at_fpr5_calib", "fpr_at_fpr5_calib",
        "fpr_at_95tpr", "tpr_at_1fpr"]


def threshold_for_fpr(real_scores, fpr=0.05):
    """Smallest threshold t (predict fake iff s >= t) with empirical FPR <= fpr on real_scores.
    Ties are handled conservatively: all reals tied with the cut are on the 'real' side."""
    r = np.sort(np.asarray(real_scores, float))[::-1]
    if len(r) == 0:
        return np.nan
    k = int(np.floor(fpr * len(r)))  # allowed false positives
    return -np.inf if k >= len(r) else float(np.nextafter(r[k], np.inf))


def rates_at(y, s, thr):
    """(tpr, fpr) when predicting fake iff s >= thr."""
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    tpr = float((pos >= thr).mean()) if len(pos) else np.nan
    fpr = float((neg >= thr).mean()) if len(neg) else np.nan
    return tpr, fpr


def balanced_accuracy(y, s, thr):
    tpr, fpr = rates_at(y, s, thr)
    return 0.5 * (tpr + (1 - fpr))


def _point_metrics(y, s, threshold, calib):
    out = {k: np.nan for k in KEYS}
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos and nneg:
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)  # one point per distinct score -> ties OK
        out["auroc"] = float(np.trapz(tpr, fpr))
        out["ap"] = float(average_precision_score(y, s))
        out["fpr_at_95tpr"] = float(fpr[np.argmax(tpr >= 0.95)])
        out["tpr_at_1fpr"] = float(tpr[fpr <= 0.01].max())
        if threshold is not None:
            out["bal_acc"] = balanced_accuracy(y, s, threshold)
        if calib is not None and len(calib):
            t = threshold_for_fpr(calib, 0.05)
            out["tpr_at_fpr5_calib"], out["fpr_at_fpr5_calib"] = rates_at(y, s, t)
            out["bal_acc_fpr5"] = 0.5 * (out["tpr_at_fpr5_calib"] + 1 - out["fpr_at_fpr5_calib"])
    return out


def compute_metrics(y, s, threshold=0.5, calib_real_scores=None, n_boot=1000, seed=0, alpha=0.05):
    """Returns {metric: {'value','ci_lo','ci_hi'}, 'n_pos','n_neg','warnings'}.
    Bootstrap is stratified by class (and resamples the calibration reals too), seeded."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    calib = None if calib_real_scores is None else np.asarray(calib_real_scores, float)
    pt = _point_metrics(y, s, threshold, calib)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    warns = []
    if npos < 2 or nneg < 2:
        warns.append("fewer than 2 samples in a class: CIs undefined")
    elif nneg < 100:
        warns.append(f"only {nneg} reals: tpr_at_1fpr means 0 false positives; coarse FPR resolution")
    if calib is not None and len(calib) < 20:
        warns.append(f"only {len(calib)} calibration reals: FPR=5% threshold is unreliable")
    boots = {k: [] for k in KEYS}
    if n_boot and npos >= 2 and nneg >= 2:
        rng = np.random.default_rng(seed)
        ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ip, npos), rng.choice(ineg, nneg)])
            c = None if calib is None else rng.choice(calib, len(calib))
            b = _point_metrics(y[idx], s[idx], threshold, c)
            for k in KEYS:
                boots[k].append(b[k])
    res = {}
    for k in KEYS:
        v = np.array(boots[k], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = (np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                      if len(v) and not np.isnan(v).all() else (np.nan, np.nan))
        res[k] = dict(value=pt[k], ci_lo=float(lo), ci_hi=float(hi))
    res.update(n_pos=npos, n_neg=nneg, warnings=warns)
    return res


def evaluate_scored_frame(df, scores, calib_real_scores=None, threshold=0.5, n_boot=1000, seed=0):
    """df: test rows with `label`, `generator`; scores aligned. Per-generator metrics use that
    generator's fakes vs ALL test reals."""
    y, s = df.label.to_numpy(), np.asarray(scores, float)
    out = {"overall": compute_metrics(y, s, threshold, calib_real_scores, n_boot, seed), "per_generator": {}}
    reals = y == 0
    for g in sorted(set(df.generator[df.label == 1])):
        m = reals | (df.generator.to_numpy() == g)
        out["per_generator"][g] = compute_metrics(y[m], s[m], threshold, calib_real_scores, n_boot, seed)
    return out


# ----------------------------------------------------------------------------- grouped (cluster) bootstrap
def bootstrap_weights(real_cluster, fake_stratum=None, n_boot=1000, seed=0):
    """Resampling plan as multiplicity matrices (Wr [n_boot, n_real], Wf [n_boot, n_fake]), seeded.

    Reals are resampled BY CLUSTER (e.g. device id; reals sharing a cluster id move together, a real with a unique id
    is its own cluster): K clusters are drawn with replacement, K = number of distinct clusters. Fakes are resampled
    BY IMAGE, stratified: inside each stratum (e.g. generator) n_s images are drawn with replacement, so every
    generator keeps its sample size. fake_stratum=None -> one stratum. Wr[b, i] is how often real i is drawn in
    replicate b. Use the SAME (Wr, Wf) for every arm/condition to get paired comparisons."""
    rng = np.random.default_rng(seed)
    rc = np.asarray(real_cluster)
    _, code = np.unique(rc, return_inverse=True)
    K = int(code.max()) + 1 if len(code) else 0
    cnt = np.zeros((n_boot, K))
    draws = rng.integers(0, K, (n_boot, K))
    for b in range(n_boot):
        cnt[b] = np.bincount(draws[b], minlength=K)
    Wr = cnt[:, code]
    if fake_stratum is None:
        return Wr, None
    fs = np.asarray(fake_stratum)
    Wf = np.zeros((n_boot, len(fs)))
    for g in sorted(set(fs.tolist())):
        cols = np.where(fs == g)[0]
        d = rng.integers(0, len(cols), (n_boot, len(cols)))
        for b in range(n_boot):
            Wf[b, cols] = np.bincount(d[b], minlength=len(cols))
    return Wr, Wf


def weighted_auroc(sr, sf, Wr, Wf):
    """AUROC (fakes above reals, ties 0.5) of every bootstrap replicate: sum_ij Wr_i Wf_j [1(sf_j>sr_i)+.5 1(=)] /
    (sum Wr * sum Wf). sr [nr], sf [nf], Wr [B, nr], Wf [B, nf] -> [B]. Weights may be the all-ones row (point value)."""
    sr, sf = np.asarray(sr, float), np.asarray(sf, float)
    M = (sf[None, :] > sr[:, None]).astype(float) + 0.5 * (sf[None, :] == sr[:, None])
    num = np.einsum("bj,bj->b", Wr @ M, Wf)
    return num / (Wr.sum(1) * Wf.sum(1))


def _plan(real_cluster, fake_stratum, n_fake, n_boot, seed):
    fs = np.zeros(n_fake, int) if fake_stratum is None else fake_stratum
    return bootstrap_weights(real_cluster, fs, n_boot, seed)


def grouped_bootstrap_auroc(sr, sf, real_cluster, fake_stratum=None, n_boot=1000, seed=0, alpha=0.05):
    """AUROC with a cluster-bootstrap CI (reals by cluster, fakes by image within stratum). Returns
    dict(value, ci_lo, ci_hi, boot) with `boot` the [n_boot] replicate AUROCs (percentile CI)."""
    sr, sf = np.asarray(sr, float), np.asarray(sf, float)
    Wr, Wf = _plan(real_cluster, fake_stratum, len(sf), n_boot, seed)
    value = float(weighted_auroc(sr, sf, np.ones((1, len(sr))), np.ones((1, len(sf))))[0])
    boot = weighted_auroc(sr, sf, Wr, Wf)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(value=value, ci_lo=float(lo), ci_hi=float(hi), boot=boot)


def grouped_bootstrap_auroc_diff(sr_a, sf_a, sr_b, sf_b, real_cluster, fake_stratum=None, n_boot=1000, seed=0,
                                 alpha=0.05):
    """Paired AUROC(A) - AUROC(B): the SAME resampled reals/fakes (same clusters, same stratified images) are used for
    both score sets, so shared sampling noise cancels. A and B score the same rows (same order). Returns
    dict(delta, ci_lo, ci_hi, p_le0 = share of replicates with diff <= 0, boot)."""
    sr_a, sf_a, sr_b, sf_b = (np.asarray(x, float) for x in (sr_a, sf_a, sr_b, sf_b))
    if sr_a.shape != sr_b.shape or sf_a.shape != sf_b.shape:
        raise ValueError("A and B must score the same rows")
    Wr, Wf = _plan(real_cluster, fake_stratum, len(sf_a), n_boot, seed)
    one_r, one_f = np.ones((1, len(sr_a))), np.ones((1, len(sf_a)))
    delta = float(weighted_auroc(sr_a, sf_a, one_r, one_f)[0] - weighted_auroc(sr_b, sf_b, one_r, one_f)[0])
    boot = weighted_auroc(sr_a, sf_a, Wr, Wf) - weighted_auroc(sr_b, sf_b, Wr, Wf)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(delta=delta, ci_lo=float(lo), ci_hi=float(hi), p_le0=float((boot <= 0).mean()), boot=boot)
