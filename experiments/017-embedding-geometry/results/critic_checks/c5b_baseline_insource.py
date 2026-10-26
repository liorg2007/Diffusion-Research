"""c5b (baseline for c5, no projection, k=0): c4 oracle with only n target-pipeline train-role reals (n = 25, 50, 100; 2 draws), held-out OI and RAISE; also the in-source
(training-source) test macro under the same model, to see whether the extra projection hurts the sources already covered. Full model."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "8"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import run as R
from sklearn.metrics import roc_auc_score
ARM = "fov512"; E = R.e7(); A = R.get_arm(ARM); D = A.D; sl = R.FS["F_b8"]; JR = "jpeg75_resize05"
gens = E.arm_all_gens(ARM); Ff = {g: D.idx("test", None, gens=[g], label=1) for g in gens}
chosen = {"RAISE": 3, "OI": 2}
def auc(s, rr):
    return {g: roc_auc_score(np.r_[np.zeros(len(rr)), np.ones(len(Ff[g]))], np.r_[s[rr], s[Ff[g]]]) for g in gens}
out = {}
for S in ("OI", "RAISE"):
    T = R.others_of(S); z = np.load(R.m3u_path(ARM, S)); mu, sd, U = z["mu"], z["sd"], z["U"]; Uk = U[:, :chosen[S]]
    rows = E.train_rows(D, T, "full")
    Ztr = [((A.X[c][rows][:, sl] - mu) / sd) for c in R.TAUG]
    y = np.concatenate([D.lab[rows]] * 3); g = np.concatenate([D.groups[rows]] * 3)
    trS = D.idx("train", E.SRC[S]["train_pools"], label=0)
    rr = E.src_eval_rows(D, S); cal = E.calib_rows(D, T)
    for n in (0,):
        for seed in (0,):
            pick = np.random.default_rng(seed).choice(trS, min(n, len(trS)), replace=False)
            pass  # d = np.concatenate([(A.X[c][pick][:, sl] - mu) / sd for c in R.TAUG]).mean(0)
            Uo = np.zeros((Uk.shape[0], 0))
            m = R.probe.fit_probe(np.concatenate([R.project_out(Zc, Uo) for Zc in Ztr]), y, R.SEED, groups=g, class_weight="balanced")
            sc = {c: m.decision_function(R.project_out((A.X[c][:, sl] - mu) / sd, Uo)) for c in ("clean", JR)}
            a = auc(sc[JR], rr); thr = E.thr_fpr(sc["clean"][cal])
            ins = {t: float(np.mean(list(auc(sc[JR], E.src_eval_rows(D, t)).values()))) for t in T}
            ins_fpr = {t: float((sc[JR][E.src_eval_rows(D, t)] >= thr).mean()) for t in T}
            r = dict(macro_full=float(np.mean(list(a.values()))), flux=float(a[E.FLUX]), fpr_A=float((sc[JR][rr] >= thr).mean()),
                     insource_macro=ins, insource_fpr=ins_fpr)
            out[f"{S}|n={n}|s={seed}"] = r
            print(S, n, seed, json.dumps(r), flush=True)
json.dump(out, open(Path(__file__).with_name("c5b_baseline_insource.json"), "w"), indent=1)
