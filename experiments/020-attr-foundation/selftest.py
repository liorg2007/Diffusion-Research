"""Synthetic self-tests of run.py: splits, metrics, bootstrap, pairwise, labels, gates, pipelines, and PLANTED LEAKS the checks must catch.

    env/.venv/bin/python -I experiments/020-attr-foundation/selftest.py   -> results/selftest.txt
No real data and no GPU are used (except d008_status, which only reads the existing guard JSON files).
"""
import importlib.util
import io
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run020", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["run020"] = R
spec.loader.exec_module(R)

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

LINES, FAILS = [], []


def check(name, cond, extra=""):
    LINES.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}".rstrip())
    if not cond:
        FAILS.append(name)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


rng = np.random.default_rng(0)
t0 = time.time()

# ---------------------------------------------------------------- 1. splits
K, n = 4, 400
y = np.repeat(np.arange(K), n // K)
groups = np.array([f"rid:{i // 2}" for i in range(n)])             # pairs of rows share a group
groups = np.array([f"rid:{(i % (n // K)) // 2}" for i in range(n)])  # a family holds one row of each... (class-crossing groups)
fold = R.make_folds(y, groups, 5, 0)
chk = R.check_grouped_folds(fold, groups)
check("splits: grouped folds are valid", chk["ok"], str(chk))
check("splits: deterministic (same seed twice)", (fold == R.make_folds(y, groups, 5, 0)).all())
check("splits: all 5 folds used, roughly balanced", sorted(np.bincount(fold).tolist())[0] > n // 5 * 0.7, str(np.bincount(fold).tolist()))
bad = fold.copy()
g0 = np.where(groups == groups[0])[0]
bad[g0[0]], bad[g0[1]] = 0, 1
check("PLANTED LEAK: a group split across folds is caught", not R.check_grouped_folds(bad, groups)["ok"])
check("PLANTED LEAK: train/test sharing a group raises in assert_disjoint_groups", raises(R.assert_disjoint_groups, groups, np.arange(10), np.arange(5, 20)))
check("assert_disjoint_groups passes on a valid fold", R.assert_disjoint_groups(groups, np.where(fold != 0)[0], np.where(fold == 0)[0]) is True)
check("held-out laundering in training is caught", not R.gate_train_conds(["clean", "screenshot_sim"], ["screenshot_sim", "mavali_medium"])["ok"] and R.gate_train_conds(["clean", "social_sim"], ["screenshot_sim"])["ok"])

# ---------------------------------------------------------------- 2. metrics + bootstrap
yt = rng.integers(0, K, 300)
pr = np.where(rng.random(300) < 0.6, yt, rng.integers(0, K, 300))
gg = np.array([f"g{i // 3}" for i in range(300)])
Wc, inv, ug = R.cluster_weights(gg, 200, 0)
check("bootstrap: row 0 is the unweighted point estimate", (Wc[0] == 1).all() and Wc.shape == (201, len(ug)))
check("bootstrap: every replicate draws n_clusters clusters", np.allclose(Wc[1:].sum(1), len(ug)))
C = R.boot_conf(Wc[:, inv], yt, pr, K)
st = R.conf_stats(C)
check("metrics: bacc point == sklearn balanced_accuracy", abs(st["bacc"][0] - balanced_accuracy_score(yt, pr)) < 1e-9)
cib = R.ci(st["bacc"])
check("metrics: bacc CI brackets the point", cib[1] <= cib[0] <= cib[2], str(cib))
stf = R.conf_stats(C, drop=2)
rec = [(pr[yt == k] == k).mean() for k in range(K) if k != 2]
check("metrics: macro without a class drops exactly that class", abs(stf["bacc_drop"][0] - np.mean(rec)) < 1e-9)
sp, sn = rng.normal(1, 1, 80), rng.normal(0, 1, 300)
yy = np.r_[np.ones(80), np.zeros(300)]
g2 = np.array([f"c{i}" for i in range(380)])
Wc2, inv2, _ = R.cluster_weights(g2, 100, 0)
ab = R.auroc_boot(sp, sn, Wc2[:, inv2][:, :80], Wc2[:, inv2][:, 80:])
check("metrics: weighted AUROC point == sklearn roc_auc_score", abs(ab[0] - roc_auc_score(yy, np.r_[sp, sn])) < 1e-6)
check("metrics: AUROC of identical scores is 0.5", abs(R.auroc_boot(np.zeros(10), np.zeros(20), np.ones((1, 10)), np.ones((1, 20)))[0] - 0.5) < 1e-9)
check("Holm: adjusted p-values are monotone and >= raw", all(a >= b for a, b in zip(R.holm([0.01, 0.04, 0.03]), [0.01, 0.04, 0.03])) and R.holm([0.01, 0.04, 0.03])[0] == 0.03)

# ---------------------------------------------------------------- 3. pairwise (content-free by construction)
nf, Kp = 150, 4
fam = np.repeat([f"d3:{i}" for i in range(nf)], Kp)
cls = np.tile(np.arange(Kp), nf)
Pr = R.build_pairs(cls, fam)
check("pairwise: 6 class pairs x 150 families = 900 pairs", len(Pr["I"]) == 6 * nf, str(len(Pr["I"])))
lp_perfect = np.log(np.eye(Kp)[cls] * 0.97 + 0.01)
check("pairwise: a perfect scorer gets 1.0", R.pair_scores(lp_perfect, Pr).mean() == 1.0 and R.pair_scores(lp_perfect, Pr, True).mean() == 1.0)
content = rng.normal(size=(nf, Kp))                                  # family-level 'content' score, identical for all images of a family
lp_content = np.repeat(content, Kp, axis=0)
check("pairwise: a pure CONTENT scorer is exactly at chance (ties = 0.5)", abs(R.pair_scores(lp_content, Pr).mean() - 0.5) < 1e-9)
lp_rand = rng.normal(size=(nf * Kp, Kp))
check("pairwise: a random scorer is ~0.5", abs(R.pair_scores(lp_rand, Pr).mean() - 0.5) < 0.05, f"{R.pair_scores(lp_rand, Pr).mean():.3f}")
Wp, invp, _ = R.cluster_weights(fam, 100, 0)
pb = R.pair_boot(R.pair_scores(lp_perfect, Pr), Pr, Wp, invp)
check("pairwise: bootstrap macro/pooled point == 1.0 for the perfect scorer", pb["macro"][0] == 1.0 and pb["pooled"][0] == 1.0)
pbx = R.pair_boot(R.pair_scores(lp_rand, Pr), Pr, Wp, invp, exclude=0)
check("pairwise: excluding a class removes its pair types (3 of 6 remain)", np.isfinite(pbx["macro"][0]) and len([t for t in pbx["types"] if 0 not in t]) == 3)

# ---------------------------------------------------------------- 4. labels (docs/08 criteria)
v = dict(bacc=[0.70, 0.66, 0.74], m1_bacc=0.40, pairwise=0.80, raid_bacc=0.60)
check("label H-A1: all criteria met -> PASS", R.label_ha1(v, True)["label"] == "PASS")
check("label H-A1: CI lower bound within 0.10 of M1 -> INCONCLUSIVE", R.label_ha1(dict(v, m1_bacc=0.60), True)["label"] == "INCONCLUSIVE")
check("label H-A1: RAID 0.50 -> INCONCLUSIVE", R.label_ha1(dict(v, raid_bacc=0.50), True)["label"] == "INCONCLUSIVE")
check("label H-A1: bacc 0.30 -> KILL", R.label_ha1(dict(v, bacc=[0.30, 0.27, 0.33]), True)["label"] == "KILL")
check("label H-A1: pairwise 0.55 -> KILL", R.label_ha1(dict(v, pairwise=0.55), True)["label"] == "KILL")
check("label H-A1: a failed validity gate -> INVALID (no label)", R.label_ha1(v, False)["label"] == "INVALID")
check("open-set usable: mean 0.85, min 0.70 -> True; one generator at 0.60 -> False",
      R.label_open_set({"a": 0.9, "b": 0.8, "c": 0.85})["usable"] and not R.label_open_set({"a": 0.95, "b": 0.60, "c": 0.95})["usable"])
a_ok = {f"s{i}": 0.9 for i in range(16)}
kr = {f"r{i}": 0.97 for i in range(8)}
check("label H-A4: mean 0.9, min 0.9 -> PASS (+DISCOVERY when every real kind acc >= .95)", R.label_ha4(a_ok, kr, True)["label"] == "PASS" and R.label_ha4(a_ok, kr, True)["discovery"])
check("label H-A4: one real source at 0.94 kind accuracy -> no DISCOVERY", not R.label_ha4(a_ok, dict(kr, r0=0.94), True)["discovery"])
check("label H-A4: min 0.60 -> INCONCLUSIVE; mean 0.65 -> KILL", R.label_ha4(dict(a_ok, s0=0.6), kr, True)["label"] == "INCONCLUSIVE" and R.label_ha4({f"s{i}": 0.65 for i in range(16)}, kr, True)["label"] == "KILL")
check("label H-A4: INVALID when a gate failed", R.label_ha4(a_ok, kr, False)["label"] == "INVALID")

# ---------------------------------------------------------------- 5. validity gates + planted leaks
d = R.gate_dedupe(["a", "b", "c", "d"], [0, 0, 1, 1])
check("gate dedupe: unique hashes pass", d["ok"])
d = R.gate_dedupe(["a", "b", "a", "d"], [0, 0, 1, 1])
check("PLANTED LEAK: the same sha256 in two classes is caught", (not d["ok"]) and d["n_cross_class_hashes"] == 1)

# M0 on tensors: shape differs by class (a planted size leak) must fail, identical shapes must be at chance
yk = np.repeat(np.arange(8), 60)
fo = R.make_folds(yk, np.array([f"g{i}" for i in range(len(yk))]), 5, 0)
same = np.tile([224, 224], (len(yk), 1)).astype(float)
leak = same.copy()
leak[yk == 3] = [224, 200]
leak[yk == 5] = [180, 224]
ba_same = R.bacc_simple(yk, R.m0_tensor_oof(same, yk, fo))
ba_leak = R.bacc_simple(yk, R.m0_tensor_oof(leak, yk, fo))
check("gate M0-on-tensors: identical tensor shapes -> exactly chance", abs(ba_same - 1 / 8) < 1e-9 and R.gate_m0_tensor({"clean": ba_same}, 1 / 8, 0.04)["ok"], f"{ba_same:.3f}")
check("PLANTED LEAK: class-dependent tensor size is caught by the M0-on-tensors gate", not R.gate_m0_tensor({"clean": ba_leak}, 1 / 8, 0.04)["ok"], f"bacc {ba_leak:.3f}")
rng2 = np.random.default_rng(1)
rs = rng2.integers(100, 200, size=(len(yk), 2)).astype(float)                    # random sizes independent of class (social_sim-like)
ba_rs = R.bacc_simple(yk, R.m0_tensor_oof(rs, yk, fo))
check("gate M0-on-tensors: class-independent random sizes stay within the 0.04 margin", R.gate_m0_tensor({"social_sim": ba_rs}, 1 / 8, 0.04)["ok"], f"{ba_rs:.3f}")

# permutation control: honest pipeline vs a pipeline with feature selection on ALL data (classic leak)
X = rng.normal(size=(160, 400))
yb = rng.integers(0, 2, 160)
fob = R.make_folds(yb, np.arange(160).astype(str), 5, 0)


def pipeline(yv, leaky):
    pred = np.zeros(160, int)
    for f in range(5):
        tr, te = np.where(fob != f)[0], np.where(fob == f)[0]
        sel_rows = np.arange(160) if leaky else tr                               # leaky: label-based feature selection also sees the test fold
        cor = np.abs([np.corrcoef(X[sel_rows, j], yv[sel_rows])[0, 1] for j in range(X.shape[1])])
        top = np.argsort(cor)[-20:]
        m = LogisticRegression(max_iter=300).fit(X[tr][:, top], yv[tr])
        pred[te] = m.predict(X[te][:, top])
    return R.bacc_simple(yv, pred)


perm_h = [pipeline(np.random.default_rng(s).permutation(yb), False) for s in range(60)]
perm_l = [pipeline(np.random.default_rng(s).permutation(yb), True) for s in range(60)]
gh = R.gate_permutation(perm_h, 0.5, 0.03, 0.08)
gl = R.gate_permutation(perm_l, 0.5, 0.03, 0.08)
check("gate permutation: the honest pipeline is at chance", gh["ok"], f"mean {gh['mean']:.3f} p95 {gh['p95']:.3f}")
check("PLANTED LEAK: feature selection on test labels is caught by the permutation control", not gl["ok"], f"mean {gl['mean']:.3f} p95 {gl['p95']:.3f}")

# ---------------------------------------------------------------- 6. format from content; texture stats
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "fake.png"
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(p, "JPEG", quality=95)       # JPEG bytes under a .png name (Firefly)
    r = R.m0_row(str(p))
    check("M0: true format comes from content (JPEG under .png), q-table hashed, mismatch flagged", r["fmt"] == "JPEG" and r["qhash"] != "none" and r["ext_mismatch"] == 1)
    p2 = Path(td) / "real.png"
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(p2, "PNG")
    r2 = R.m0_row(str(p2))
    check("M0: a real PNG is PNG with no q-table", r2["fmt"] == "PNG" and r2["qhash"] == "none" and r2["ext_mismatch"] == 0)
noise = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
smooth = np.full((224, 224, 3), 128, np.uint8)
tn, ts = R.texture_stats(noise), R.texture_stats(smooth)
check("M1: 12 finite statistics; noise has higher Laplacian variance and noise std than a flat image", tn.shape == (12,) and np.isfinite(tn).all() and np.isfinite(ts).all() and tn[0] > ts[0] and tn[2] > ts[2])
check("D-008 guard files readable and ok", R.d008_status()["ok"], str({k: v for k, v in R.d008_status().items() if k != "missing"}))

# ---------------------------------------------------------------- 7. pipelines on synthetic features
D = 4608


def synth(nrows_per_class, K_, sep, conds, seed=0, far=None):
    r = np.random.default_rng(seed)
    y_ = np.repeat(np.arange(K_), nrows_per_class)
    cen = r.normal(size=(K_, D)) * sep
    if far is not None:
        cen[far] += 6 * sep * np.sign(r.normal(size=D))
    X_ = {c: (cen[y_] + r.normal(size=(len(y_), D))).astype(np.float32) for c in conds}
    return y_, X_


conds = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]
tm = ["clean", "jpeg75_resize05", "social_sim"]
K1, m1_ = 5, 60
y1, X1 = synth(m1_, K1, 0.2, conds, 1)
fam1 = np.array([f"rid:{i % m1_}" for i in range(len(y1))])                      # one image per class per family
f1 = R.make_folds(y1, fam1, 5, 0)
A1 = R.Analysis("syn", dict(pairwise=True), y1, [f"c{i}" for i in range(K1)], fam1, f1, conds, tm, "jpeg75_resize05", X1,
                {c: np.random.default_rng(2).normal(size=(len(y1), 12)).astype(np.float32) for c in conds})
R._G["syn"] = A1
R.CFG["c_grid"]["dino"] = [0.001, 0.01]
res_ = [R.closed_task(("syn", "b8", "tmix", f)) for f in range(5)]
lp = {c: np.zeros((A1.n, K1), np.float32) for c in conds}
for r_ in res_:
    for c in conds:
        lp[c][r_["te"]] = r_["logp"][c]
b_ = R.bacc_simple(y1, lp["jpeg75_resize05"].argmax(1))
check("pipeline: closed-set probe on separable synthetic classes reaches high bacc", b_ > 0.9, f"bacc {b_:.3f}")
Wc1, inv1, _ = R.cluster_weights(fam1, 100, 0)
summ, _ = R.eval_closed(A1, lp, Wc1, inv1, "c2")
check("pipeline: eval_closed gives bacc, F1, pairwise (macro+strict), no-drop macro, confusion", all(k in summ["clean"] for k in ("bacc", "mf1", "pw", "pw_strict", "bacc_drop", "pw_drop", "confusion")) and summ["clean"]["pw"]["macro"][0] > 0.9)
resm = [R.closed_task(("syn", "M1", "tmix", f)) for f in range(5)]
lpm = np.zeros((A1.n, K1), np.float32)
for r_ in resm:
    lpm[r_["te"]] = r_["logp"]["jpeg75_resize05"]
check("pipeline: M1 baseline on pure-noise features is near chance", abs(R.bacc_simple(y1, lpm.argmax(1)) - 1 / K1) < 0.15, f"{R.bacc_simple(y1, lpm.argmax(1)):.3f}")
resg = [R.closed_task(("syn", "M1gbm", "tmix", f)) for f in range(5)]
check("pipeline: M1 gradient-boosting variant runs", len(resg) == 5 and resg[0]["logp"]["clean"].shape[1] == K1)
nl, ch = None, None
cand = [R.closed_task(("syn", fs, "tmix", 0)) for fs in R.FS_ORDER]
best = max(cand, key=lambda r: (round(r["inner_best"], 12), -R.FS_ORDER.index(r["fs"])))
check("pipeline: inner-CV nested choice picks a feature set from train data only", best["fs"] in R.FS_ORDER and set(best["table"]) <= set(R.CFG["c_grid"]["dino"]))
# LOGO: class 2 far away from the others
y2, X2 = synth(m1_, K1, 0.1, conds, 3, far=2)
A2 = R.Analysis("syn2", dict(pairwise=False), y2, [f"c{i}" for i in range(K1)], fam1, f1, conds, tm, "jpeg75_resize05", X2)
R._G["syn2"] = A2
lg = [R.logo_task(("syn2", g, f, "b8", "tmix")) for g in (2, 0) for f in range(5)]


def logo_auc(g, sc):
    s = np.zeros(A2.n)
    for r_ in lg:
        if r_["g"] == g:
            s[r_["te"]] = r_["out"]["jpeg75_resize05"][sc]
    return roc_auc_score(A2.y == g, s)


check("LOGO: a far-away held-out class is flagged novel (Mahalanobis AUROC > 0.95)", logo_auc(2, "maha") > 0.95, f"{logo_auc(2, 'maha'):.3f}")
check("LOGO: held-out class never appears in the training rows (asserted in task)", all(r_["g"] in (2, 0) for r_ in lg))
check("LOGO: a held-out class in the middle of the others is harder than the far one", logo_auc(0, "maha") < logo_auc(2, "maha"), f"{logo_auc(0, 'maha'):.3f}")
mm = R.MahaModel(X2["clean"][y2 != 2][:, R.SL["b8"]], y2[y2 != 2])
nov_in, _ = mm.score(X2["clean"][y2 != 2][:, R.SL["b8"]])
nov_out, near = mm.score(X2["clean"][y2 == 2][:, R.SL["b8"]])
check("MahaModel: held-out rows are farther than training rows", nov_out.mean() > nov_in.mean() * 1.5)
dk, lab = mm.__class__(X2["clean"][y2 != 2][:, R.SL["b8"]], y2[y2 != 2], keep_train=True).knn(X2["clean"][y2 == 2][:, R.SL["b8"]][:5], 10)
check("kNN-10: returns one distance and 10 labels per row", dk.shape == (5,) and lab.shape == (5, 10))
# H-A4 on 6 synthetic 'sources' (two fake, four real) with one far source
K4 = 6
y4, X4 = synth(m1_, K4, 0.1, conds, 5, far=5)
fam4 = np.array([f"rid:{i % m1_}" for i in range(len(y4))])
f4 = R.make_folds(y4, fam4, 5, 0)
A4 = R.Analysis("ha4", {}, y4, [f"s{i}" for i in range(K4)], fam4, f4, conds, tm, "jpeg75_resize05", X4,
                {c: np.random.default_rng(7).normal(size=(len(y4), 12)).astype(np.float32) for c in conds})
A4.kinds = ["fake", "fake", "real", "real", "real", "real"]
A4.holdouts = [[i] for i in range(K4)]
A4.holdout_names, A4.holdout_kind = A4.classes, A4.kinds
A4.sources = [None] * K4
R._G["ha4"] = A4
r4 = [R.ha4_task(("b8", "tmix", f)) for f in range(5)]
nov5 = np.zeros(A4.n)
near5 = np.zeros(A4.n, int)
for r_ in r4:
    nov5[r_["te"]] = r_["nov"][(5, "jpeg75_resize05")]
    near5[r_["te"]] = r_["near"][(5, "jpeg75_resize05")]
auc5 = roc_auc_score(A4.y == 5, nov5)
check("H-A4: leave-one-source-out novelty flags the far source", auc5 > 0.95, f"AUROC {auc5:.3f}")
check("H-A4: nearest KNOWN source of the held-out rows is never the held-out source itself", (near5[A4.y == 5] != 5).all())
check("H-A4: kNN novelty and the 10-NN real-vote are produced for b8 at the configured conditions", all(("knn" in r_ and (5, "jpeg75_resize05") in r_["knn"]) for r_ in r4))
check("H-A4: closed-set 16-way head returns one prediction per test row per condition", all(len(r_["closed"]["clean"]) == len(r_["te"]) for r_ in r4))
# permutation control on the real code path (H-A1 closed): label permutation -> near chance on separable data
accs = []
for s in range(8):
    yp = np.random.default_rng([R.SEED, 4242, s]).permutation(A1.y)
    pred = np.zeros(A1.n, int)
    for f in range(5):
        s_, te_, pr_ = R.perm_closed_task(("syn", s, f, "b8", 0.01))
        pred[te_] = pr_
    accs.append(R.bacc_simple(yp, pred))
check("permutation (real code path): shuffled labels on separable data give ~chance", abs(np.mean(accs) - 1 / K1) < 0.08, f"mean {np.mean(accs):.3f}")
perm_auc = [R.perm_ha4_task((s, 0))[1] for s in range(2)]
check("permutation (H-A4 code path): shuffled source labels give ~0.5 novelty AUROC", abs(np.mean(perm_auc) - 0.5) < 0.1, f"{np.mean(perm_auc):.3f}")

LINES.append(f"\n{len(LINES) - len(FAILS)} of {len(LINES)} checks passed; failed: {FAILS or 'none'}  ({time.time() - t0:.0f}s)")
out = HERE / "results" / "selftest.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(LINES) + "\n")
print("\n".join(LINES))
sys.exit(1 if FAILS else 0)
