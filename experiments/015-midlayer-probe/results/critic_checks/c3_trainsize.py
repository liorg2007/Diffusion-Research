"""Critic check c3: (A) is P1 an artefact of the shrunken twin-disjoint training set? Refit F_b8/F_last x T-clean/T-aug on the
003 b0_train rows (VISION natives + RAISE + 7 generators, ~2x fakes, includes twins of E6 eval rows -> symmetric leakage), same
groups/balanced/CV, LOGO; evaluate on the E6 test rows with the E6 Plan. (B) RAISE-only cross-source vs number of RAISE train
reals (subsamples of the E6 train RAISE reals; 'full' model only, diagnostic). CPU, 4 workers."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
import run as E
import probe

ARM = "orig"
A = E.Arm(ARM)
b0 = A.df.b0_train.to_numpy()


def fit(rows, feat, regime):
    cols = E.FEATS[feat]
    tc = E.REGIMES[regime]
    X = np.concatenate([A.X[c][rows][:, cols] for c in tc])
    y = np.concatenate([A.lab[rows]] * len(tc))
    g = np.concatenate([A.groups[rows]] * len(tc))
    m = probe.fit_probe(X, y, E.SEED, groups=g, class_weight="balanced")
    return np.stack([m.decision_function(A.X[c][:, cols]) for c in E.CONDS]).astype(np.float32)


def task_a(args):
    feat, regime, model = args
    gname = model[2:] if model.startswith("g_") else None
    rows = np.where(b0 & (A.gen != gname))[0]
    return args, fit(rows, feat, regime), len(rows)


def task_b(args):
    n, seed = args
    rr = A.idx("train", ["raise"], label=0)
    if n == "b0":
        real = np.where(b0 & (A.pool == "raise"))[0]
        fake = np.where(b0 & (A.lab == 1))[0]
    else:
        real = np.random.default_rng(seed).choice(rr, n, replace=False) if n < len(rr) else rr
        fake = A.idx("train", ["fake"], gens=E.TRAINGENS, label=1)
    rows = np.sort(np.concatenate([real, fake]))
    return args, fit(rows, "F_b8", "T-aug"), len(real)


if __name__ == "__main__":
    t0 = time.time()
    ta = [(f, r, m) for f in ("F_b8", "F_last") for r in ("T-clean", "T-aug") for m in E.MODELS]
    tb = [(21, s) for s in range(3)] + [(42, s) for s in range(3)] + [(84, 0), ("b0", 0)]
    with ProcessPoolExecutor(4) as ex:
        ra = {a: (s, n) for a, s, n in ex.map(task_a, ta)}
        rb = {a: (s, n) for a, s, n in ex.map(task_b, tb)}
    out = dict(note=__doc__, A={}, B={})
    pl = E.Plan(A, E.POOLS["primary"])
    plR = E.Plan(A, E.POOLS["RAISE"])
    twin = E.TwinPlan(A)
    S = {}
    for f in ("F_b8", "F_last"):
        for r in ("T-clean", "T-aug"):
            S[f"{f}|{r}"] = {m: ra[(f, r, m)][0] for m in E.MODELS}
    out["A"]["n_train_full"] = int(ra[("F_b8", "T-clean", "full")][1])
    out["A"]["n_fakes_b0"] = int((b0 & (A.lab == 1)).sum())
    for cond in ("clean", "jpeg75_resize05", "screenshot_sim", "mavali_medium"):
        ci = E.CI[cond]
        o = {}
        for k, s in S.items():
            ga, gR = pl.gen_auc(s, ci), plR.gen_auc(s, ci)
            o[k] = dict(macro8=E.summ(E.macro(ga, E.GENS8)), flux=E.summ(ga[E.FLUX]), raise_macro8=float(E.macro(gR, E.GENS8)[0]),
                        twin=float(twin.auc(s, ci)["pooled"][0]))
        ga, gb = pl.gen_auc(S["F_b8|T-aug"], ci), pl.gen_auc(S["F_last|T-clean"], ci)
        o["gain F_b8|T-aug - F_last|T-clean"] = dict(macro8=E.diff_summary(E.macro(ga, E.GENS8), E.macro(gb, E.GENS8)),
                                                     flux=E.diff_summary(ga[E.FLUX], gb[E.FLUX]))
        gc = pl.gen_auc(S["F_last|T-aug"], ci)
        o["layer F_b8|T-aug - F_last|T-aug"] = dict(macro8=E.diff_summary(E.macro(ga, E.GENS8), E.macro(gc, E.GENS8)),
                                                    flux=E.diff_summary(ga[E.FLUX], gc[E.FLUX]))
        out["A"][cond] = o
    Rv = A.idx("test", E.POOLS["vision3"], label=0)
    Rr = A.idx("test", ["raise"], label=0)
    F = A.idx("test", ["fake"], label=1)
    for a, (s, n) in rb.items():
        d = {}
        for cond in ("clean", "jpeg75_resize05"):
            x = s[E.CI[cond]].astype(float)
            d[cond] = dict(fake_vs_vision3=roc_auc_score(np.r_[np.zeros(len(Rv)), np.ones(len(F))], np.r_[x[Rv], x[F]]),
                           fake_vs_raise=roc_auc_score(np.r_[np.zeros(len(Rr)), np.ones(len(F))], np.r_[x[Rr], x[F]]))
        out["B"][f"n={a[0]} seed={a[1]} (n_real={n})"] = d
    (HERE / "c3_trainsize.json").write_text(json.dumps(out, indent=1))
    for cond in ("jpeg75_resize05", "clean"):
        print("== A (003 b0_train rows)", cond)
        for k, v in out["A"][cond].items():
            print("  ", k, json.dumps(v))
    print("== B raise_only F_b8 T-aug full model, vs n RAISE train reals")
    for k, v in out["B"].items():
        print("  ", k, {c: {a: round(b, 3) for a, b in x.items()} for c, x in v.items()})
    print(f"done {time.time() - t0:.0f}s")
