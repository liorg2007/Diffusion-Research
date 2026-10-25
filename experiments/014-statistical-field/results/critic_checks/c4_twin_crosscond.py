"""Critic check c4: twin-paired C2ST (RAISE real vs its own caption-twin fakes, id- and generator-held-out 4x4 cells, as run._paired_job)
with TRAIN on clean features and TEST on jpeg75_resize05 (detector-like, unmatched training), next to the matched version (= E5).
Point estimates only (mean of 5 repeats)."""
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


def paired(space, arm, train_cond, test_cond, R=5, K=4):
    df = M.load_df(arm)
    Xtr_all, Xte_all = M.get_space(space, arm, train_cond, df), M.get_space(space, arm, test_cond, df)
    ri, fi, gens, rids = M.twin_pairs(df)
    uid = sorted(set(rids))
    pid = np.array([uid.index(r) for r in rids])
    real_of_id = {a: b for a, b in zip(pid, ri)}
    ugen = sorted(set(gens))
    aucs, wins = [], []
    for r in range(R):
        rng = np.random.default_rng(2000 + r)
        idf = M.deal(range(len(uid)), K, rng)
        gf = M.deal(ugen, K, rng)
        id_fold = np.array([idf[i] for i in range(len(uid))])
        g_fold = np.array([gf[g] for g in gens])
        num = den = wn = 0.0
        for a in range(K):
            for b in range(K):
                trf = (id_fold[pid] != a) & (g_fold != b)
                tef = (id_fold[pid] == a) & (g_fold == b)
                if tef.sum() == 0:
                    continue
                tr_ids = [i for i in range(len(uid)) if id_fold[i] != a]
                Xtr = np.r_[Xtr_all[[real_of_id[i] for i in tr_ids]], Xtr_all[fi[trf]]]
                ytr = np.r_[np.zeros(len(tr_ids)), np.ones(trf.sum())]
                te_ids = np.unique(pid[tef])
                sc = M.fit_score(space, Xtr, ytr, np.r_[Xte_all[[real_of_id[i] for i in te_ids]], Xte_all[fi[tef]]])
                sr, sf = sc[: len(te_ids)], sc[len(te_ids):]
                auc = roc_auc_score(np.r_[np.zeros(len(sr)), np.ones(len(sf))], sc)
                pos = {i: k for k, i in enumerate(te_ids)}
                w = np.mean([sf[k] > sr[pos[pid[tef][k]]] for k in range(len(sf))])
                num += auc * tef.sum(); wn += w * tef.sum(); den += tef.sum()
        aucs.append(num / den); wins.append(wn / den)
    return float(np.mean(aucs)), float(np.mean(wins))


out = []
for space in ("battery", "dino_b0", "dino_blk8"):
    for arm in ("orig", "fov512"):
        for trc in ("jpeg75_resize05", "clean"):
            a, w = paired(space, arm, trc, "jpeg75_resize05")
            row = dict(space=space, arm=arm, train=trc, test="jpeg75_resize05", auroc=a, win=w)
            out.append(row)
            print(json.dumps(row), flush=True)
(HERE / "c4_twin_crosscond.json").write_text(json.dumps(out, indent=1))
