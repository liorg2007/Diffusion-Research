"""Critic check c1: why does the E5 DINOv2 C2ST (0.940 laundered) exceed 003 B0 laundered (0.875)? Does it survive controls?

Re-uses run.py's own functions (load_df, get_space, fit_score, deal) so folds/model are identical to E5.
Variants (all device- and generator-held-out, 4 folds, 5 repeats, pooled OOF AUROC as in E5):
  pool     : which reals are used (all | native+RAISE (=003 pool) | WA+RAISE | RAISE | WA)
  train    : 'matched' (train on the same condition as test, = E5) | 'clean' (train on clean features, test laundered, = 003)
  prune    : 'twin' (E5 pruning) | 'strict' (additionally drop every training row whose RAISE id occurs anywhere in the test fold,
             i.e. also other-generator fakes sharing a caption with a test fake)
Usage: env/.venv/bin/python .../c1_c2st_controls.py  (CPU, 4 workers)
"""
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
E5 = HERE.parents[1]
spec = importlib.util.spec_from_file_location("e5run", E5 / "run.py")
M = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(M)

_CACHE = {}


def feats(space, arm, cond, df):
    k = (space, arm, cond)
    if k not in _CACHE:
        _CACHE[k] = M.get_space(space, arm, cond, df)
    return _CACHE[k]


def oof(space, arm, cond, pool, train, prune, seed, K=4):
    df = M.load_df(arm)
    Xte_all = feats(space, arm, cond, df)
    Xtr_all = Xte_all if train == "matched" else feats(space, arm, "clean", df)
    lab, gen, rid, rp = df.label.values, df.generator.values, df.rid.values, df.rpool.values
    isr = lab == 0
    rmask = isr if pool == "all" else (isr & np.isin(rp, pool.split("+")))
    rng = np.random.default_rng(1000 + seed)                       # identical dealing to run.c2st_oof
    use = np.where(rmask | (lab == 1))[0]
    rf = M.deal(sorted(set(df.rgroup.values[use][lab[use] == 0])), K, rng)
    gf = M.deal(sorted(set(gen[use][lab[use] == 1])), K, rng)
    fold = np.array([rf[df.rgroup.values[i]] if lab[i] == 0 else gf[gen[i]] for i in use])
    score = np.full(len(use), np.nan)
    per_fold = []
    for k in range(K):
        te, tr = fold == k, fold != k
        L, R = lab[use], rid[use]
        t_real_rids = set(R[te & (L == 0) & (R != "")])
        t_fake_rids = set(R[te & (L == 1)])
        bad = ((L == 1) & np.isin(R, list(t_real_rids))) | ((L == 0) & (R != "") & np.isin(R, list(t_fake_rids)))
        if prune == "strict":
            bad |= (R != "") & np.isin(R, list(t_real_rids | t_fake_rids))
        tr = tr & ~bad
        score[te] = M.fit_score(space, Xtr_all[use[tr]], L[tr], Xte_all[use[te]])
        from sklearn.metrics import roc_auc_score
        per_fold.append(roc_auc_score(L[te], score[te]))
    from sklearn.metrics import roc_auc_score
    y = lab[use]
    pg = {g: roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones((gen[use] == g).sum())], np.r_[score[y == 0], score[gen[use] == g]]) for g in M.GENS}
    return roc_auc_score(y, score), float(np.mean(per_fold)), pg


def job(a):
    space, arm, cond, pool, train, prune = a
    r = [oof(space, arm, cond, pool, train, prune, s) for s in range(5)]
    return dict(space=space, arm=arm, cond=cond, pool=pool, train=train, prune=prune,
                auroc=float(np.mean([x[0] for x in r])), sd=float(np.std([x[0] for x in r])),
                mean_within_fold=float(np.mean([x[1] for x in r])),
                per_gen={g: float(np.mean([x[2][g] for x in r])) for g in M.GENS})


if __name__ == "__main__":
    jobs = []
    for space in ("dino_b0", "dino_blk8"):
        for arm in ("orig", "fov512"):
            for pool in ("all", "native+RAISE", "WA+RAISE", "RAISE", "WA"):
                for train in ("matched", "clean"):
                    jobs.append((space, arm, "jpeg75_resize05", pool, train, "twin"))
                jobs.append((space, arm, "jpeg75_resize05", pool, "matched", "strict"))
                jobs.append((space, arm, "clean", pool, "matched", "twin"))
    jobs.append(("battery", "fov512", "jpeg75_resize05", "WA+RAISE", "matched", "strict"))
    jobs.append(("battery", "fov512", "jpeg75_resize05", "RAISE", "matched", "strict"))
    with Pool(4) as p:
        out = p.map(job, jobs, chunksize=1)
    (HERE / "c1_c2st_controls.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(f"{o['space']:9s} {o['arm']:6s} {o['cond']:16s} {o['pool']:13s} train={o['train']:7s} prune={o['prune']:6s} "
              f"AUROC {o['auroc']:.3f} sd {o['sd']:.3f} within-fold {o['mean_within_fold']:.3f}")
