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
