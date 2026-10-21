import _path  # noqa
import numpy as np

from metrics import (balanced_accuracy, compute_metrics, evaluate_scored_frame, rates_at, threshold_for_fpr)

import pandas as pd


def V(m, k):
    return m[k]["value"]


def test_hand_computed_basic():
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    m = compute_metrics(y, s, threshold=0.5, n_boot=0)
    assert abs(V(m, "auroc") - 0.75) < 1e-12
    assert abs(V(m, "ap") - (0.5 * 1 + 0.5 * 2 / 3)) < 1e-12
    assert abs(V(m, "bal_acc") - 0.75) < 1e-12  # tpr .5, tnr 1
    assert abs(balanced_accuracy(np.array(y), np.array(s), 0.35) - 0.75) < 1e-12  # tpr 1, tnr .5
    assert m["n_pos"] == 2 and m["n_neg"] == 2


def test_perfect_and_inverted():
    y = np.r_[np.zeros(50), np.ones(50)]
    m = compute_metrics(y, np.r_[np.arange(50), np.arange(50) + 100], n_boot=0)
    assert V(m, "auroc") == 1 and V(m, "ap") == 1 and V(m, "fpr_at_95tpr") == 0 and V(m, "tpr_at_1fpr") == 1
    m = compute_metrics(y, -np.r_[np.arange(50), np.arange(50) + 100], n_boot=0)
    assert V(m, "auroc") == 0


def test_ties():
    assert V(compute_metrics([0, 1], [.5, .5], n_boot=0), "auroc") == 0.5
    assert V(compute_metrics([0, 0, 1, 1], [.3] * 4, n_boot=0), "auroc") == 0.5
    # partial ties: pos {1,2,2}, neg {2,0}: pairs (p,n): 1>2 no,1>0 yes,2=2 half,2>0 yes,x2 -> (1+ .5+1+.5+1)/6=4/6
    m = compute_metrics([1, 1, 1, 0, 0], [1, 2, 2, 2, 0], n_boot=0)
    assert abs(V(m, "auroc") - 4 / 6) < 1e-12


def test_fpr_at_95_tpr_and_tpr_at_1fpr():
    # 20 pos scores 20..39, 25 neg scores 0..24. 95% TPR needs thr 21 -> negs 21..24 -> FPR 4/25
    y = np.r_[np.ones(20), np.zeros(25)]
    s = np.r_[np.arange(20, 40), np.arange(25)].astype(float)
    assert abs(V(compute_metrics(y, s, n_boot=0), "fpr_at_95tpr") - 0.16) < 1e-12
    # 200 neg 0..199, 100 pos at 300+, 100 pos at 50 -> thr 198 allows FPR=1%, TPR=.5
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[np.arange(200), 300 + np.arange(100), np.full(100, 50)].astype(float)
    assert abs(V(compute_metrics(y, s, n_boot=0), "tpr_at_1fpr") - 0.5) < 1e-12


def test_calibrated_threshold():
    reals = np.arange(100) / 100.0
    t = threshold_for_fpr(reals, 0.05)
    assert rates_at(np.zeros(100), reals, t)[1] == 0.05  # exactly 5 of 100 above
    # tied calibration reals: conservative, FPR 0 rather than 100%
    assert rates_at(np.zeros(100), np.full(100, .3), threshold_for_fpr(np.full(100, .3)))[1] == 0
    # tiny calib set (n=10, k=0): threshold just above max
    assert rates_at(np.zeros(10), np.arange(10.), threshold_for_fpr(np.arange(10.)))[1] == 0
    y = np.r_[np.zeros(100), np.ones(10)]
    s = np.r_[reals, [0.99, 0.97, 0.96, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]]
    m = compute_metrics(y, s, calib_real_scores=reals, n_boot=0)
    assert abs(V(m, "fpr_at_fpr5_calib") - 0.05) < 1e-12
    assert abs(V(m, "tpr_at_fpr5_calib") - 0.3) < 1e-12  # 0.99, .97, .96 are > .94
    assert abs(V(m, "bal_acc_fpr5") - 0.5 * (0.3 + 0.95)) < 1e-12


def test_single_class_and_tiny():
    m = compute_metrics([1, 1, 1], [.1, .2, .3], n_boot=50)
    assert np.isnan(V(m, "auroc")) and np.isnan(m["auroc"]["ci_lo"]) and m["warnings"]
    m = compute_metrics([0, 1], [.2, .9], n_boot=50)  # 1 per class: value defined, CI refused
    assert V(m, "auroc") == 1 and np.isnan(m["auroc"]["ci_lo"])
    m = compute_metrics([], [], n_boot=10)
    assert np.isnan(V(m, "auroc"))


def test_bootstrap_seeded_and_sane():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(300), np.ones(300)]
    s = np.r_[rng.normal(0, 1, 300), rng.normal(1, 1, 300)]
    a = compute_metrics(y, s, n_boot=200, seed=7, calib_real_scores=s[:300])
    b = compute_metrics(y, s, n_boot=200, seed=7, calib_real_scores=s[:300])
    assert a == b
    c = compute_metrics(y, s, n_boot=200, seed=8, calib_real_scores=s[:300])
    assert a["auroc"]["ci_lo"] != c["auroc"]["ci_lo"]
    au = a["auroc"]
    assert au["ci_lo"] < au["value"] < au["ci_hi"] and au["ci_hi"] - au["ci_lo"] < 0.15
    from sklearn.metrics import roc_auc_score
    assert abs(au["value"] - roc_auc_score(y, s)) < 1e-12
    assert abs(au["value"] - 0.76) < 0.06  # theory: Phi(1/sqrt2)=0.76


def test_per_generator_uses_all_test_reals():
    df = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1, 1], "generator": ["real"] * 3 + ["a", "a", "b", "b"]})
    s = np.array([.1, .2, .3, .9, .8, .25, .05])
    out = evaluate_scored_frame(df, s, n_boot=0)
    assert out["per_generator"]["a"]["n_neg"] == 3 and out["per_generator"]["a"]["n_pos"] == 2
    assert out["per_generator"]["a"]["auroc"]["value"] == 1.0
    # b: pos {.25,.05} vs neg {.1,.2,.3}: wins: .25>.1,.25>.2 ; .05 none -> 2/6
    assert abs(out["per_generator"]["b"]["auroc"]["value"] - 2 / 6) < 1e-12
    assert out["overall"]["n_pos"] == 4
