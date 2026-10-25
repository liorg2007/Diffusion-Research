"""Critic check c3: MMD with CLUSTER-level permutations (VISION device = one cluster; each RAISE image and each fake = own cluster).
E5 permuted single images, which is anticonservative when reals cluster by device. Same kernel/bandwidth as run._mmd_job.
Cells: jpeg75_resize05; orig with reals native+RAISE; fov512 with reals WA+RAISE (scale-matched). 999 permutations, Holm over 8 gens.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e5run", HERE.parents[1] / "run.py")
M = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(M)


def mmd_u(K, f):
    r = ~f
    nf, nr = f.sum(), r.sum()
    if nf < 2 or nr < 2:
        return np.nan
    Kff, Krr, Kfr = K[np.ix_(f, f)], K[np.ix_(r, r)], K[np.ix_(f, r)]
    return Kff.sum() / (nf * (nf - 1)) + Krr.sum() / (nr * (nr - 1)) - 2 * Kfr.sum() / (nf * nr)


out = []
P = 999
for space in ("battery", "dino_b0", "dino_blk8"):
    for arm, pool in (("orig", ["native", "RAISE"]), ("fov512", ["WA", "RAISE"])):
        df = M.load_df(arm)
        Z = M.zscore(M.get_space(space, arm, "jpeg75_resize05", df))
        n2 = (Z ** 2).sum(1)
        D2 = np.maximum(n2[:, None] + n2[None, :] - 2 * Z @ Z.T, 0)
        bw2 = np.median(np.sqrt(D2[np.triu_indices(len(Z), 1)])) ** 2
        K = np.exp(-D2 / (2 * bw2))
        np.fill_diagonal(K, 0.0)
        rng = np.random.default_rng(0)
        rows = []
        for g in M.GENS:
            idx = np.where(((df.label == 0) & df.rpool.isin(pool)).values | (df.generator.values == g))[0]
            Ks = K[np.ix_(idx, idx)]
            isf = df.generator.values[idx] == g
            sid = np.array([str(i) for i in idx], dtype=object)
            cl = np.where(isf, "f" + sid, np.where(df.rpool.values[idx] == "RAISE", "r" + sid, df.device.values[idx].astype(object))).astype(str)
            ucl, inv = np.unique(cl, return_inverse=True)
            clab = np.zeros(len(ucl), bool)
            clab[inv[isf]] = True
            m0 = mmd_u(Ks, isf)
            null = np.array([mmd_u(Ks, rng.permutation(clab)[inv]) for _ in range(P)])
            p = (1 + np.sum(null >= m0)) / (P + 1)
            rows.append(dict(space=space, arm=arm, reals="+".join(pool), gen=g, mmd2=float(m0), p_cluster=float(p), n_clusters=len(ucl)))
        ph = M.holm([r["p_cluster"] for r in rows])
        for r, h in zip(rows, ph):
            r["p_holm"] = float(h)
            print(json.dumps(r), flush=True)
        out += rows
(HERE / "c3_mmd_cluster_perm.json").write_text(json.dumps(out, indent=1))
