"""E8 critic checks (read-only on E8 caches; writes only to results/critic_checks/). fov512, F_b8, jpeg75_resize05 unless stated.
c1 matched split-half: within-source vs cross-source probe-normal cosines on DISJOINT halves (same data size, same C).
c2 functional transfer: AUROC of single-source probe normals d_probe(S1) on S2 test reals vs test fakes (LOGO normals per generator).
c3 subspace capture: share of the held-out source's real-mean offset (training z-space) inside U_k, vs random k/768; alignment with fake offset.
c4 oracle removal: add the held-out source's own offset (estimated from its TRAIN-role reals, disjoint from test) to U and refit (full model).
"""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import numpy as np
import run as R
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ARM = "fov512"
E = R.e7(); A = R.get_arm(ARM); D = A.D; sl = R.FS["F_b8"]; JR = "jpeg75_resize05"
S3 = R.SOURCES
gens = E.arm_all_gens(ARM)
out = {}
rows_pool = E.train_rows(D, tuple(S3), "full")
sig_pool = R.safe_sigma(R.taug_stack(A, rows_pool, sl)[0])

# ---- c1
allrows = rows_pool
ug = np.unique(D.groups[allrows])
rng = np.random.default_rng(1)
within = {s: [] for s in S3}; cross = {f"{a}-{b}": [] for i, a in enumerate(S3) for b in S3[i + 1:]}
for rep in range(6):
    perm = rng.permutation(ug); half = set(perm[: len(ug) // 2].tolist())
    W = {}
    for S in S3:
        rows = E.train_rows(D, (S,), "full")
        C = float(np.load(R.m2_path(ARM, "F_b8", S, "full"))["C"])
        ina = np.array([g in half for g in D.groups[rows]])
        for h, msk in (("A", ina), ("B", ~ina)):
            X, y, _ = R.taug_stack(A, rows[msk], sl)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(X, y)
            W[S, h] = R.to_common(R.probe_normal_raw(m), sig_pool)
    for S in S3:
        within[S].append(R.cosine(W[S, "A"], W[S, "B"]))
    for i, a in enumerate(S3):
        for b in S3[i + 1:]:
            cross[f"{a}-{b}"] += [R.cosine(W[a, "A"], W[b, "B"]), R.cosine(W[a, "B"], W[b, "A"])]
out["c1"] = dict(within={k: [float(np.mean(v)), float(np.std(v))] for k, v in within.items()},
                 cross={k: [float(np.mean(v)), float(np.std(v))] for k, v in cross.items()})
# Spearman-Brown disattenuation of the reported full-data cosines using the E8 split-half reference
sh = {"VISION": 0.335, "RAISE": 0.495, "OI": 0.386}; full = {"VISION-RAISE": 0.351, "VISION-OI": 0.385, "RAISE-OI": 0.294}
rel = {k: 2 * v / (1 + v) for k, v in sh.items()}
out["c1"]["disattenuated_full"] = {k: v / np.sqrt(rel[k.split("-")[0]] * rel[k.split("-")[1]]) for k, v in full.items()}
print("c1", json.dumps(out["c1"], indent=1), flush=True)

# ---- c2
Xj = A.X[JR][:, sl].astype(np.float64)
Ff = {g: D.idx("test", None, gens=[g], label=1) for g in gens}
mat = {}
for S1 in S3:
    for S2 in S3:
        rr = E.src_eval_rows(D, S2)
        au = []
        for g in gens:
            w = np.load(R.m2_path(ARM, "F_b8", S1, E.model_of(ARM, g)))["w"]
            s = Xj @ w
            au.append(roc_auc_score(np.r_[np.zeros(len(rr)), np.ones(len(Ff[g]))], np.r_[s[rr], s[Ff[g]]]))
        mat[f"{S1}->{S2}"] = float(np.mean(au))
out["c2"] = mat
print("c2", json.dumps(mat, indent=1), flush=True)

# ---- c3 / c4
def fit_eval(S, T, mu, sd, U, tag):
    rows = E.train_rows(D, T, "full")
    Xtr = np.concatenate([R.project_out((A.X[c][rows][:, sl] - mu) / sd, U) for c in R.TAUG])
    y = np.concatenate([D.lab[rows]] * 3); g = np.concatenate([D.groups[rows]] * 3)
    m = R.probe.fit_probe(Xtr, y, R.SEED, groups=g, class_weight="balanced")
    sc = {c: m.decision_function(R.project_out((A.X[c][:, sl] - mu) / sd, U)) for c in ("clean", JR)}
    rr = E.src_eval_rows(D, S); cal = E.calib_rows(D, T)
    au = {gg: roc_auc_score(np.r_[np.zeros(len(rr)), np.ones(len(Ff[gg]))], np.r_[sc[JR][rr], sc[JR][Ff[gg]]]) for gg in gens}
    thr = E.thr_fpr(sc["clean"][cal])
    fpr = float((sc[JR][rr] >= thr).mean())
    # held-out real shift along the probe, as a fraction of the train-real -> test-fake margin
    trr = rows[D.lab[rows] == 0]
    fk = np.concatenate([Ff[gg] for gg in gens])
    shift = (np.median(sc[JR][rr]) - np.median(sc[JR][trr])) / (np.median(sc[JR][fk]) - np.median(sc[JR][trr]))
    return dict(tag=tag, k=int(U.shape[1]), macro_full=float(np.mean(list(au.values()))), flux=float(au[E.FLUX]), fpr_A=fpr, shift_frac=float(shift))

c3, c4 = {}, {}
chosen = {"VISION": 4, "RAISE": 3, "OI": 2}
for S in S3:
    T = R.others_of(S)
    z = np.load(R.m3u_path(ARM, S)); mu, sd, U = z["mu"], z["sd"], z["U"]
    zs = lambda rows: np.concatenate([(A.X[c][rows][:, sl] - mu) / sd for c in R.TAUG])
    te = E.src_eval_rows(D, S)
    trS = D.idx("train", E.SRC[S]["train_pools"], label=0)
    dte = zs(te).mean(0); dtr = zs(trS).mean(0)
    rowsT = E.train_rows(D, T, "full"); fk = rowsT[D.lab[rowsT] == 1]
    dfk = zs(fk).mean(0)
    cap = {k: float(np.sum((U[:, :k].T @ dte) ** 2) / np.sum(dte ** 2)) for k in (2, 3, 4, 6, 10)}
    c3[S] = dict(capture=cap, random_baseline={k: k / U.shape[0] for k in cap}, cos_offset_test_vs_train_role=R.cosine(dte, dtr),
                 cos_heldout_offset_vs_fake_offset=R.cosine(dte, dfk), norm_heldout_offset=float(np.linalg.norm(dte)),
                 norm_fake_offset=float(np.linalg.norm(dfk)))
    print("c3", S, json.dumps(c3[S]), flush=True)
    Uk = U[:, :chosen[S]]
    v = dtr - Uk @ (Uk.T @ dtr); v /= np.linalg.norm(v)
    Uo = np.concatenate([Uk, v[:, None]], 1)
    # oracle B: also the top-8 PCs of S's own train-role reals' offset structure (cluster means of S vs T, rank grows) - skipped; one direction only
    c4[S] = [fit_eval(S, T, mu, sd, np.zeros((U.shape[0], 0)), "none"), fit_eval(S, T, mu, sd, Uk, "chosen U_k"),
             fit_eval(S, T, mu, sd, Uo, "U_k + oracle own-offset")]
    print("c4", S, json.dumps(c4[S]), flush=True)
out["c3"], out["c4"] = c3, c4
json.dump(out, open(Path(__file__).with_name("c1_geometry.json"), "w"), indent=1)
