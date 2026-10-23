"""Linear probe + leave-one-generator-out (LOGO) evaluation."""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import KEYS, compute_metrics

CS = np.logspace(-4, 1, 6)


def fit_probe(X, y, seed=0, Cs=CS, folds=5, groups=None, class_weight=None):
    """Standardize + L2 logistic regression; C picked by stratified CV (AUROC) on the given train set only.

    groups: optional per-row group ids (e.g. the source image of an augmented copy). When given, CV uses
        StratifiedGroupKFold(shuffle, seed) so all rows of one group stay in ONE fold (no augmented-copy leakage into
        the C selection). The fold count is min(folds, rows of the rarer class, number of groups).
    class_weight: passed to LogisticRegressionCV (e.g. "balanced"). Note the CV scoring (AUROC) is weight-free.
    With groups=None, class_weight=None the behaviour is exactly the original."""
    y = np.asarray(y)
    k = min(folds, int(np.bincount(y).min()))
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have one entry per row")
        k = min(k, len(np.unique(groups)))
    if k < 2:
        raise ValueError("need >=2 samples of each class in train")
    if groups is None:
        cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    else:   # explicit list of (train, test) index arrays: LogisticRegressionCV needs no metadata routing
        cv = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups))
    clf = LogisticRegressionCV(Cs=Cs, cv=cv, scoring="roc_auc", max_iter=2000, random_state=seed,
                               class_weight=class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # sklearn 1.9 deprecation chatter about l1_ratios
        return make_pipeline(StandardScaler(), clf).fit(X, y)


def predict_scores(model, X):
    return model.predict_proba(X)[:, 1]


def standard_eval(df, X_train, X_eval, seed=0, threshold=0.5, n_boot=1000):
    """Train on split=='train' (features X_train), evaluate on split=='test' (features X_eval),
    thresholds calibrated on split=='calib' reals. X_* aligned with df rows."""
    from metrics import evaluate_scored_frame
    sp = df.split.to_numpy()
    tr, te, ca = sp == "train", sp == "test", (sp == "calib") & (df.label.to_numpy() == 0)
    m = fit_probe(X_train[tr], df.label.to_numpy()[tr], seed)
    res = evaluate_scored_frame(df[te], predict_scores(m, X_eval[te]),
                                predict_scores(m, X_eval[ca]) if ca.any() else None, threshold, n_boot, seed)
    res["chosen_C"] = float(m[-1].C_[0])
    return res


def logo_eval(df, X_train, X_eval, seed=0, threshold=0.5, n_boot=1000, generators=None):
    """Leave-one-generator-out. For each fake generator g: train on train-split reals + train-split fakes of
    all OTHER generators (the manifest's held-out generator is never in train anyway); test on ALL of g's
    fakes vs test-split reals; thresholds calibrated on calib reals. Returns {'folds': {g: metrics}, 'macro': {...}}."""
    lab, gen, sp = df.label.to_numpy(), df.generator.to_numpy(), df.split.to_numpy()
    gens = generators or sorted(set(gen[lab == 1]))
    folds = {}
    for g in gens:
        tr = (sp == "train") & (gen != g)
        te = ((lab == 0) & (sp == "test")) | (gen == g)
        ca = (lab == 0) & (sp == "calib")
        if len(set(lab[tr])) < 2:
            folds[g] = dict(error="train set lacks a class")
            continue
        m = fit_probe(X_train[tr], lab[tr], seed)
        r = compute_metrics(lab[te], predict_scores(m, X_eval[te]), threshold,
                            predict_scores(m, X_eval[ca]) if ca.any() else None, n_boot, seed)
        r["chosen_C"] = float(m[-1].C_[0])
        r["n_train"] = int(tr.sum())
        folds[g] = r
    ok = [f for f in folds.values() if "error" not in f]
    macro = {k: float(np.nanmean([f[k]["value"] for f in ok])) if ok else float("nan") for k in KEYS}
    return dict(folds=folds, macro=macro)
