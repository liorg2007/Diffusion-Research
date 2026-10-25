"""Critic check c2: bridge 003 B0 (0.875 laundered) and the E5 C2ST with the SAME 004 DINOv2 features.

003 protocol: train on b0_train rows (99 VISION natives + 140 RAISE + 435 fakes of 7 generators, FLUX never), test on split=='test'
with reals = VISION native + RAISE, fakes = all test fakes (7 seen generators + FLUX). Model here: run.fit_score (z-score + LR C=0.01,
balanced) - 003 used the same standardize+LR with C=0.01 chosen by CV, no class weighting.
Variants: train condition (clean vs matched laundered), twin-disjoint training, adding WA/FBH copies to training, real pool at test.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e5run", HERE.parents[1] / "run.py")
M = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(M)
sys.path.insert(0, str(M.ROOT / "experiments" / "lib"))
from probe import fit_probe, predict_scores  # noqa: E402

out = []
for arm in ("orig", "fov512"):
    df = M.load_df(arm)
    lab, rp, rid = df.label.values, df.rpool.values, df.rid.values
    b0tr = df.b0_train.astype(str).values == "True"
    test = df.split.values == "test"
    for space in ("dino_b0", "dino_blk8"):
        X = {c: M.get_space(space, arm, c, df) for c in ("clean", "jpeg75_resize05")}
        for test_pool in ("native+RAISE", "WA+RAISE", "RAISE", "native"):
            te = test & ((lab == 1) | np.isin(rp, test_pool.split("+")))
            te_real_rids = set(rid[te & (lab == 0) & (rid != "")])
            for trainset in ("b0_train", "b0_train_twinfree", "train_all_pools", "train_all_pools_twinfree"):
                tr = b0tr.copy() if trainset.startswith("b0_train") else (df.split.values == "train")
                if trainset.endswith("twinfree"):
                    tr &= ~((lab == 1) & np.isin(rid, list(te_real_rids)))
                for trc in ("clean", "jpeg75_resize05"):
                    s = M.fit_score(space, X[trc][tr], lab[tr], X["jpeg75_resize05"][te])
                    row = dict(arm=arm, space=space, test_pool=test_pool, trainset=trainset, train_cond=trc, test_cond="jpeg75_resize05",
                               n_train=int(tr.sum()), n_test_real=int((te & (lab == 0)).sum()), n_test_fake=int((te & (lab == 1)).sum()),
                               auroc=float(roc_auc_score(lab[te], s)))
                    if space == "dino_b0" and trainset == "b0_train" and test_pool == "native+RAISE":
                        m = fit_probe(X[trc][tr], lab[tr])           # exact 003 recipe (CV-chosen C, unweighted)
                        row["auroc_fit_probe"] = float(roc_auc_score(lab[te], predict_scores(m, X["jpeg75_resize05"][te])))
                    out.append(row)
                    print(json.dumps(row), flush=True)
(HERE / "c2_003_bridge.json").write_text(json.dumps(out, indent=1))
