"""Train the two demo models from CACHED DINOv2 features (no GPU, no downloads).

    env/.venv/bin/python -I poc/fingerprint-demo/train_models.py [--fold 0]

(a) attribution_model.npz : 8-way generator probe, DINOv2 fov512 'cat' (blocks 4/8/12 CLS+mean, 4608-d), T-mix training
    (clean + jpeg75_resize05 + social_sim), trained ONLY on the train-role caption families of one fold of
    experiments/020-attr-foundation/results/folds_ha1_main.json. Test-fold images stay honest demo samples.
    C is chosen with 020's own inner grouped CV (e11.inner_table), exactly as in experiments/020-attr-foundation/run.py closed_task.
(b) detector_model.npz    : "is it AI?" probe = E10 (019-source-diversity) F_b8 (block-8 CLS+mean, 1536-d) real-vs-fake probe, T-aug,
    trained on the train-role rows of ALL 6 real sources + the 7 train generators (FLUX is test-only in 019), via
    019's own Data / train_rows / probe.fit_probe. Threshold = 1% FPR on the calib reals' CLEAN scores (reading A).
Everything is stored as npz/json (no pickles). Provenance and held-out metrics go to models/meta.json.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONDS_ALL = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]
TMIX = ["clean", "jpeg75_resize05", "social_sim"]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ (a) attribution
def train_attribution(e11, fold):
    df, u = e11.load_universe("fov512")
    cfg = e11.CFG["analyses"]["ha1_main"]
    sel = ~df.dup_drop.to_numpy() & df.pool.isin(cfg["select"]["pool"]).to_numpy()
    idx = np.where(sel)[0]
    cls = {c: i for i, c in enumerate(cfg["classes"])}
    y = np.array([cls[c] for c in df[cfg["class_column"]].iloc[idx]])
    groups = df.grp.to_numpy()[idx]
    fj = json.loads((C.ROOT / "experiments/020-attr-foundation/results/folds_ha1_main.json").read_text())
    assert fj["rows"] == df.sha256.iloc[idx].tolist(), "row order differs from the committed folds"
    folds = np.array(fj["fold"])
    assert (folds == e11.make_folds(y, groups, e11.CFG["outer_folds"], e11.SEED)).all(), "committed folds not reproduced"
    log(f"attribution: {len(idx)} fakes, folds sizes {np.bincount(folds).tolist()}, using fold {fold} as TEST")
    XF = {}
    for c in CONDS_ALL:
        store = e11.read_caches(u["caches"], c)
        XF[c] = np.stack([store[s] for s in df.sha256.iloc[idx]]).astype(np.float32)
    A = e11.Analysis("ha1_main", cfg, y, cfg["classes"], groups, folds, CONDS_ALL, TMIX, "jpeg75_resize05", XF)
    tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
    e11.assert_disjoint_groups(groups, tr, te, "demo fold")
    # --- C by 020's inner grouped CV on the train rows (cat feature set, T-mix, select at jpeg75_resize05)
    table, Cb = e11.inner_table(A, "cat", tr, TMIX, "jpeg75_resize05", e11.SEED * 1000 + fold + 1, list(range(A.K)))
    log("inner-CV bacc per C:", {k: round(v, 4) for k, v in table.items()}, "-> C =", Cb)
    # --- final model on all train rows (T-mix copies), 020's fit_model
    X, yy = e11.stack(A, "cat", tr, TMIX)
    sc, clf = e11.fit_model("lr", X, yy, Cb)
    assert list(clf.classes_) == list(range(8))
    W, b, mean, scale = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64), sc.mean_.astype(np.float64), sc.scale_.astype(np.float64)

    def probs(x):
        return softmax(((x - mean) / scale) @ W.T + b)

    # --- held-out test metrics (fold `fold`), all 5 conditions
    bacc, per_gen = {}, {}
    cm = None
    for c in CONDS_ALL:
        p = probs(XF[c][te]).argmax(1)
        bacc[c] = e11.bacc_simple(y[te], p, list(range(8)))
        per_gen[c] = {C.GENERATORS[k]: float((p[y[te] == k] == k).mean()) for k in range(8)}
        if c == "jpeg75_resize05":
            cm = np.zeros((8, 8), int)
            for t_, p_ in zip(y[te], p):
                cm[t_, p_] += 1
    log("held-out bacc (fold %d, n=%d):" % (fold, len(te)), {k: round(v, 3) for k, v in bacc.items()})
    # --- msp threshold via inner grouped CV on train rows (OOF), fixed rule
    g_tr, y_tr = groups[tr], y[tr]
    ifold = e11.make_folds(y_tr, g_tr, e11.CFG["inner_folds"], e11.SEED * 1000 + fold + 1)
    oof_msp, oof_ok = [], []
    for j in range(e11.CFG["inner_folds"]):
        a, bb = tr[ifold != j], tr[ifold == j]
        e11.assert_disjoint_groups(groups, a, bb, "msp inner")
        Xa, ya = e11.stack(A, "cat", a, TMIX)
        m = e11.fit_model("lr", Xa, ya, Cb)
        for c in TMIX:
            pr = m[1].predict_proba(m[0].transform(XF[c][bb]))
            oof_msp.append(pr.max(1))
            oof_ok.append(pr.argmax(1) == y[bb])
    oof_msp, oof_ok = np.concatenate(oof_msp), np.concatenate(oof_ok)
    TAU_Q = 0.10
    tau = float(np.quantile(oof_msp[oof_ok], TAU_Q))
    log(f"msp threshold tau={tau:.3f} (10th pct of OOF max-softmax among correct predictions; OOF acc {oof_ok.mean():.3f})")
    unc = {}
    for c in CONDS_ALL:
        pr = probs(XF[c][te])
        msp = pr.max(1)
        ok = pr.argmax(1) == y[te]
        fl = msp < tau
        unc[c] = dict(frac_flagged=float(fl.mean()), acc_flagged=float(ok[fl].mean()) if fl.any() else None,
                      acc_unflagged=float(ok[~fl].mean()) if (~fl).any() else None, acc_all=float(ok.mean()))
    np.savez(C.MODELS / "attribution_model.npz", mean=mean, scale=scale, coef=W, intercept=b, classes=np.array(C.GENERATORS),
             msp_threshold=np.array(tau), C=np.array(Cb), feature_slice=np.array([0, 4608]))
    te_sha = df.sha256.iloc[idx].to_numpy()[te]
    # --- parity vectors for the worker check: cached features + expected probabilities
    meta = dict(
        kind="attribution", fold=fold, n_train=int(len(tr)), n_test=int(len(te)), C=Cb, feature_set="cat (b4/b8/b12 CLS+mean, 4608-d)",
        training="T-mix = clean + jpeg75_resize05 + social_sim copies; StandardScaler + LogisticRegression(C, class_weight=balanced, max_iter=300)",
        inner_cv_bacc=table, classes=C.GENERATORS,
        heldout_bacc_fold=bacc, heldout_per_generator_recall=per_gen,
        confusion_jpeg75_resize05=cm.tolist(),
        msp_threshold=tau, msp_rule="10th percentile of out-of-fold max-softmax among CORRECT predictions (3-fold grouped inner CV on train rows, T-mix conditions pooled)",
        uncertain_on_heldout=unc, oof_accuracy=float(oof_ok.mean()),
        open_set_note="docs/10 section 4: leave-one-generator-out novelty AUROC of the max-softmax score is only 0.768 (below the 0.80 bar): the uncertain flag is NOT a reliable unknown-generator detector.",
        reference_5fold=dict(source="experiments/020-attr-foundation/results/tables.md ('bacc per condition', nested / tmix)",
                             values=dict(clean=0.913, jpeg75_resize05=0.855, social_sim=0.872, screenshot_sim=0.866, mavali_medium=0.813),
                             note="5-fold grouped CV over all 1,020 fakes; our single-fold numbers (about 204 test images) differ by sampling noise"),
        sources=["experiments/020-attr-foundation/run.py", "experiments/020-attr-foundation/config.json",
                 "experiments/020-attr-foundation/results/folds_ha1_main.json", "experiments/020-attr-foundation/results/tables.md",
                 "experiments/004-phase4-features/cache/fov512, experiments/016-loso/cache/fov512, experiments/019-source-diversity/cache/feats (cached DINOv2 features)"],
    )
    test_rows = dict(sha=te_sha.tolist(), y=y[te].tolist())
    return meta, probs, XF, te, df.iloc[idx].reset_index(drop=True), test_rows


# ============================================================================ (b) detector
def train_detector(e10, probe, metrics):
    D = e10.Data()
    T = tuple(e10.ORDER)                                    # all 6 real sources
    rows = e10.train_rows(D, T, "full")                     # 7 train generators (FLUX is test-only), train-role reals of all 6 sources
    e10.audit_train_rows(D, T, "full", rows)
    e10.audit_taug(D.lab[rows], e10.TAUG)
    cols = e10.FEATS["F_b8"]
    X = np.concatenate([D.X[c][rows][:, cols] for c in e10.TAUG])
    y = np.concatenate([D.lab[rows]] * len(e10.TAUG))
    g = np.concatenate([D.groups[rows]] * len(e10.TAUG))
    log(f"detector: {len(rows)} train rows ({int((D.lab[rows]==0).sum())} real / {int((D.lab[rows]==1).sum())} fake) x {len(e10.TAUG)} T-aug copies")
    m = probe.fit_probe(X, y, e10.SEED, groups=g, class_weight="balanced")
    sc, clf = m[0], m[-1]
    mean, scale = sc.mean_.astype(np.float64), sc.scale_.astype(np.float64)
    w, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    Cc = float(clf.C_[0])

    def logit(x):
        return ((x - mean) / scale) @ w + b

    # threshold: reading A = 1% FPR on the training sources' CLEAN calib reals
    cal_pools = sorted({p for s in T for p in e10.SRC[s]["eval_pools"]})
    cal = D.idx("calib", cal_pools, label=0)
    thr = float(metrics.threshold_for_fpr(logit(D.X["clean"][:, cols][cal]), e10.CFG["fpr_target"]))
    log(f"detector C={Cc:.3g}; calib reals {len(cal)}; threshold logit {thr:.3f} (p={sigmoid(thr):.3f})")
    # in-source held-out (test-role) numbers: SEEN real sources, so NOT the unseen-source FPR
    in_src = {}
    from sklearn.metrics import roc_auc_score
    for c in ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim"]:
        Xc = D.X[c][:, cols]
        per_src = {}
        for s in e10.ORDER:
            r = D.idx("test", e10.SRC[s]["eval_pools"], label=0)
            per_src[s] = dict(fpr=float((logit(Xc[r]) >= thr).mean()), n=int(len(r)))
        rr = D.idx("test", None, label=0)
        per_gen = {}
        for gname in e10.ALLGENS:
            f = D.idx("test", None, gens=[gname], label=1)
            per_gen[gname] = dict(tpr=float((logit(Xc[f]) >= thr).mean()), n=int(len(f)))
        ff = D.idx("test", None, label=1)
        auc = float(roc_auc_score(np.r_[np.zeros(len(rr)), np.ones(len(ff))], np.r_[logit(Xc[rr]), logit(Xc[ff])]))
        in_src[c] = dict(fpr_by_seen_source=per_src, tpr_by_generator=per_gen, auroc_seen_sources_pooled=auc,
                         fpr_pooled=float((logit(Xc[rr]) >= thr).mean()))
    log("in-source pooled FPR:", {c: round(v["fpr_pooled"], 3) for c, v in in_src.items()})
    np.savez(C.MODELS / "detector_model.npz", mean=mean, scale=scale, coef=w, intercept=np.array(b), threshold_logit=np.array(thr),
             C=np.array(Cc), feature_slice=np.array([1536, 3072]))
    meta = dict(kind="detector", feature_set="F_b8 (block-8 CLS+mean, 1536-d; columns 1536:3072 of the 4608-d cache)",
                training="E10 'full' model with all 6 sources (k=6): train-role reals of VISION(native)/RAISE/OI/UNS/WMC/HDRP + 7 train generators (no FLUX), T-aug = clean + jpeg75_resize05 + social_sim, probe.fit_probe(balanced, grouped CV for C)",
                C=Cc, n_train_rows=int(len(rows)), n_real_train=int((D.lab[rows] == 0).sum()), n_fake_train=int((D.lab[rows] == 1).sum()),
                threshold_logit=thr, threshold_prob=float(sigmoid(thr)), n_calib_reals=int(len(cal)),
                threshold_rule="reading A: 1% FPR (metrics.threshold_for_fpr, fpr_target 0.01) on the CLEAN scores of the calib reals of the 6 training sources; applied unchanged under every laundering",
                in_source_heldout=in_src,
                in_source_note="Test-role reals of the SAME 6 sources the probe trained on: this is NOT the unseen-source FPR. Unseen-source FPR (leave-one-source-out, k=5): see unseen_source_fpr_k5.",
                sources=["experiments/019-source-diversity/run.py", "experiments/019-source-diversity/config.json",
                         "experiments/019-source-diversity/results/tables.md", "experiments/lib/probe.py"])
    return meta, logit, D, cols


def unseen_fpr_from_019():
    t = C.md_tables(C.ROOT / "experiments/019-source-diversity/results/tables.md")
    tb = C.find_table(t, "Held-out source at k = 5")
    out = {}
    for r in tb["rows"]:
        if r[1] != "F_b8":
            continue
        m = __import__("re").match(r"\s*([\d.]+) \((\d+)/(\d+)\) \[([\d.]+), ([\d.]+)\]", r[3])
        out[r[0]] = dict(auroc=C.num_ci(r[2])[0], fpr_A=float(m.group(1)), k=int(m.group(2)), n=int(m.group(3)),
                         cp95=[float(m.group(4)), float(m.group(5))])
    return dict(values=out, source="experiments/019-source-diversity/results/tables.md ('Held-out source at k = 5', F_b8, jpeg75_resize05, reading A)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    C.MODELS.mkdir(exist_ok=True)
    e11 = load_mod("e11run", "experiments/020-attr-foundation/run.py")
    e10 = load_mod("e10run", "experiments/019-source-diversity/run.py")
    import metrics
    import probe
    ameta, aprobs, XF, te, dfa, test_rows = train_attribution(e11, a.fold)
    dmeta, dlogit, D, cols = train_detector(e10, probe, metrics)
    dmeta["unseen_source_fpr_k5"] = unseen_fpr_from_019()

    # --- eligible demo samples: attribution test fold AND not a detector train/calib row
    sha2i = {s: i for i, s in enumerate(D.df.sha256)}
    elig_f, elig_r = [], []
    te_set = set(test_rows["sha"])
    for s in test_rows["sha"]:
        i = sha2i.get(s)
        if i is not None and D.role[i] == "test":
            r = D.df.iloc[i]
            elig_f.append(dict(sha=s, src_path=r.src_path, generator=r.generator, license=r.license, source_dataset=r.source_dataset))
    for pool in ["vision_native", "raise", "openimages", "unsplash", "commons"]:
        for i in D.idx("test", [pool], label=0):
            r = D.df.iloc[i]
            elig_r.append(dict(sha=r.sha256, src_path=r.src_path, pool=pool, license=r.license, source_dataset=r.source_dataset))
    log(f"eligible demo images: {len(elig_f)} fakes (attr test fold & detector test role), {len(elig_r)} reals")
    (C.MODELS / "eligible.json").write_text(json.dumps(dict(fakes=elig_f, reals=elig_r, fold=a.fold)))

    # --- parity vector: first 20 attribution test rows (cached features), expected outputs
    sh_te = test_rows["sha"]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(sh_te), 20, replace=False).tolist())
    # detector features for those rows: same sha -> same cached 4608-d row (identical caches)
    parity = []
    for k in pick:
        s = sh_te[k]
        i = sha2i[s]
        row = dict(sha=s, path=dfa.path.iloc[te[k]] if False else D.df.path.iloc[i], conds={})
        for c in ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim"]:
            x_a = XF[c][te[k]][None]
            x_d = D.X[c][i][None, cols]
            row["conds"][c] = dict(attr_probs=aprobs(x_a)[0].tolist(), ai_logit=float(dlogit(x_d)[0]))
        parity.append(row)
    (C.MODELS / "parity_expected.json").write_text(json.dumps(parity))
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), attribution=ameta, detector=dmeta,
                licence_note="Trained on Synthbuster (CC BY-NC-SA 4.0), B-Free SD3/FLUX (GRIP non-commercial), RAISE, Unsplash Lite, Commons, VISION, Open Images: the models inherit the non-commercial restriction. Research demo only.")
    (C.MODELS / "meta.json").write_text(json.dumps(meta, indent=1))
    log("wrote", C.rel(C.MODELS / "meta.json"))


if __name__ == "__main__":
    main()
