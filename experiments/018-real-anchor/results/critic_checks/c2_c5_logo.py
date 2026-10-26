"""E9 critic c2: re-run the E8 critic's c5 few-shot recipe (E8 subspace U_k + the anchor-mean direction projected out of ALL rows,
probe retrained) under the E7/E9 reporting convention: LOGO macro AUROC (8 generators, each scored by the model that never saw it),
plus the full-model macro that c5 reported. fov512 / F_b8 / jpeg75_resize05 / n = 50. Anchor sets: c5's own two picks (train-role
reals, rng seeds 0/1) and E9's draws 0-4 (train+calib pool). The projection is applied to every row (reals and fakes), so it is
label-agnostic; this checks only the reporting convention and the number of draws."""
import os, sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import numpy as np

K = {"RAISE": 3, "OI": 2}; N = 50; ARM = "fov512"; FEAT = "F_b8"

def task(a):
    import run as R
    S, kind, i = a
    ctx = R.get_ctx(ARM); D = ctx.D; sl = R.FS[FEAT]; E = ctx.E
    z = np.load(R.ROOT / "experiments/017-embedding-geometry/cache/m3" / f"u__{ARM}__{S}.npz"); mu, sd, U = z["mu"], z["sd"], z["U"]
    Uk = U[:, :K[S]]
    if kind == "c5":
        trS = D.idx("train", E.SRC[S]["train_pools"], label=0)
        pick = np.random.default_rng(i).choice(trS, min(N, len(trS)), replace=False)
    else:
        pick = ctx.anchor_order(S, i)[:N]
    d = np.concatenate([(ctx.X[c][pick][:, sl] - mu) / sd for c in R.TAUG]).mean(0)
    v = d - Uk @ (Uk.T @ d); v /= np.linalg.norm(v); Uo = np.concatenate([Uk, v[:, None]], 1)
    Z = {c: (lambda Zc: Zc - (Zc @ Uo) @ Uo.T)((ctx.X[c][:, sl].astype(np.float64) - mu) / sd) for c in ("clean", R.JR)}
    T = ctx.others(S); sc = {}
    for mi, name in enumerate(ctx.models):
        rows = E.train_rows(D, T, name)
        y = np.concatenate([D.lab[rows]] * 3); g = np.concatenate([D.groups[rows]] * 3)
        Ztr = np.concatenate([(lambda Zc: Zc - (Zc @ Uo) @ Uo.T)((ctx.X[c][rows][:, sl].astype(np.float64) - mu) / sd) for c in R.TAUG])
        m = R.probe.fit_probe(Ztr, y, R.SEED, groups=g, class_weight="balanced")
        sc[mi] = {c: m.decision_function(Z[c]) for c in Z}
    rr = ctx.rr[S]; cal = E.calib_rows(D, T); f = ctx.full
    logo = float(np.mean([R.auroc_point(sc[ctx.mi[g]][R.JR][rr], sc[ctx.mi[g]][R.JR][ctx.ff[g]]) for g in ctx.gens]))
    full = float(np.mean([R.auroc_point(sc[f][R.JR][rr], sc[f][R.JR][ctx.ff[g]]) for g in ctx.gens]))
    flux = float(R.auroc_point(sc[f][R.JR][rr], sc[f][R.JR][ctx.ff[E.FLUX]]))
    thr = R.thr_fpr(sc[f]["clean"][cal]); thr_anc = R.thr_fpr(sc[f]["clean"][pick])
    fprA = float((sc[f][R.JR][rr] >= thr).mean()); fpr_anc = float((sc[f][R.JR][rr] >= thr_anc).mean())
    tpr_anc = float(np.mean([(sc[f][R.JR][ctx.ff[g]] >= thr_anc).mean() for g in ctx.gens]))
    return dict(S=S, kind=kind, i=i, logo_macro=logo, full_macro=full, flux_full=flux, fpr_A=fprA, fpr_anchor_thr=fpr_anc, tpr_anchor_thr=tpr_anc)

if __name__ == "__main__":
    tasks = [(S, "c5", s) for S in K for s in (0, 1)] + [(S, "e9", d) for S in K for d in range(5)]
    out = []
    with ProcessPoolExecutor(7) as ex:
        for r in ex.map(task, tasks):
            print(json.dumps(r), flush=True); out.append(r)
    json.dump(out, open(Path(__file__).with_name("c2_c5_logo.json"), "w"), indent=1)
