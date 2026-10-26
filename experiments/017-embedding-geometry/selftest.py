"""Self-test of E8 (run: env/.venv/bin/python -I experiments/017-embedding-geometry/selftest.py). Synthetic checks of the geometry helpers,
then plumbing checks on the real data (output also written to results/selftest.txt)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))

import numpy as np
from sklearn.metrics import roc_auc_score

import run
from run import cosine, probe_normal_raw, to_common, safe_sigma, twin_direction, source_subspace, project_out, choose_k, r3_label

OUT, ok_all = [], True


def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    OUT.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    print(OUT[-1], flush=True)


rng = np.random.default_rng(0)
d = 40
unit = lambda v: v / np.linalg.norm(v)

# ---- 1. cosine basics
a = rng.normal(size=d)
check("cosine(a, a) = 1, cosine(a, -a) = -1", abs(cosine(a, a) - 1) < 1e-12 and abs(cosine(a, -a) + 1) < 1e-12)

# ---- 2. shared shift direction: probe normals of different sources agree (cosine ~ 1), and agree with the twin direction
shared = unit(rng.normal(size=d))
n_src = 3
src_mean = [rng.normal(0, 3.0, d) for _ in range(n_src)]             # big source offsets (nuisance)


def make_source(mean, shift_dir, n=600, scale=2.0):
    real = mean + rng.normal(size=(n, d))
    fake = mean + scale * shift_dir + rng.normal(size=(n, d))        # the fake is the SAME real distribution plus the shift
    return real, fake


ws, ds = [], []
for s in range(n_src):
    r, f = make_source(src_mean[s], shared)
    X, y = np.vstack([r, f]), np.r_[np.zeros(len(r), int), np.ones(len(f), int)]
    m = run.probe.fit_probe(X, y, 0, groups=np.arange(len(y)))
    ws.append(probe_normal_raw(m))
    ds.append(twin_direction(f, r))                                  # (fake, real twin) pairs: same row order = twin
cs = [cosine(ws[i], ws[j]) for i in range(n_src) for j in range(i + 1, n_src)]
check("shared shift -> probe-normal cosines ~ 1", min(cs) > 0.95, f"(min {min(cs):.3f})")
check("shared shift -> d_twin ~ shared direction and ~ probe normals", cosine(ds[0], shared) > 0.95 and cosine(ds[0], ws[1]) > 0.9,
      f"({cosine(ds[0], shared):.3f}, {cosine(ds[0], ws[1]):.3f})")
sig = safe_sigma(np.vstack([make_source(src_mean[s], shared)[0] for s in range(n_src)]))
check("to_common keeps direction (cosine of rescaled normals stays high)", cosine(to_common(ws[0], sig), to_common(ws[1], sig)) > 0.9)

# ---- 3. source-specific shifts: normals ~ orthogonal (cosine ~ 0)
dirs = np.linalg.qr(rng.normal(size=(d, n_src)))[0].T                # orthonormal specific directions
ws2 = []
for s in range(n_src):
    r, f = make_source(src_mean[s], dirs[s])
    X, y = np.vstack([r, f]), np.r_[np.zeros(len(r), int), np.ones(len(f), int)]
    ws2.append(probe_normal_raw(run.probe.fit_probe(X, y, 0, groups=np.arange(len(y)))))
cs2 = [abs(cosine(ws2[i], ws2[j])) for i in range(n_src) for j in range(i + 1, n_src)]
check("source-specific shifts -> probe-normal cosines ~ 0", max(cs2) < 0.2, f"(max |cos| {max(cs2):.3f})")

# ---- 4. source-subspace removal improves held-out AUROC.
# Reals of every source = content noise + an offset inside a 2-D "style" subspace N (different per source); fakes = content noise + the shared
# shift. The held-out source's style offset points toward the fake side of the style axes the training sources used for "real".
N = np.linalg.qr(rng.normal(size=(d, 2)))[0]                          # d x 2 style subspace
shift = unit(rng.normal(size=d))
shift -= N @ (N.T @ shift)
shift = unit(shift)


def src_data(style, n=500):
    real = (N @ style) + rng.normal(size=(n, d))
    fake = 1.6 * shift + rng.normal(size=(n, d)) + (N @ style) * 0.0   # fakes carry no source style (generated from prompts)
    return real, fake


train_styles = [np.array([-4.0, 0.0]), np.array([-3.0, -3.0]), np.array([0.0, -4.0])]
held_style = np.array([2.5, 2.0])
Xs, ys, srcs, cls = [], [], [], []
for i, st in enumerate(train_styles):
    r, f = src_data(st)
    Xs += [r]
    ys += [np.zeros(len(r), int)]
    srcs += [np.full(len(r), i)]
    cls += [np.arange(len(r)) + i * 10000]
Rtr = np.vstack(Xs)
Ftr = src_data(train_styles[0])[1]
Xtr = np.vstack([Rtr, Ftr])
ytr = np.r_[np.zeros(len(Rtr), int), np.ones(len(Ftr), int)]
rh, fh = src_data(held_style)
mu, sd = Rtr.mean(0), safe_sigma(Rtr)
Ztr, Zr, Zf = (Xtr - mu) / sd, (rh - mu) / sd, (fh - mu) / sd
m0 = run.probe.fit_probe(Ztr, ytr, 0, groups=np.arange(len(ytr)), class_weight="balanced")
auc0 = roc_auc_score(np.r_[np.zeros(len(Zr)), np.ones(len(Zf))], m0.decision_function(np.vstack([Zr, Zf])))
U = source_subspace((Rtr - mu) / sd, np.concatenate(srcs), np.concatenate(cls), n_extra=2)
check("source_subspace is orthonormal with k = 2(n_src-1)+n_extra", U.shape == (d, 6) and np.allclose(U.T @ U, np.eye(6), atol=1e-8), str(U.shape))
m1_ = run.probe.fit_probe(project_out(Ztr, U), ytr, 0, groups=np.arange(len(ytr)), class_weight="balanced")
auc1 = roc_auc_score(np.r_[np.zeros(len(Zr)), np.ones(len(Zf))], m1_.decision_function(np.vstack([project_out(Zr, U), project_out(Zf, U)])))
check("source removal raises held-out AUROC (style shift)", auc1 > auc0 + 0.1 and auc1 > 0.7, f"({auc0:.3f} -> {auc1:.3f})")
check("U was estimated from training reals only (no held-out row used): style axes explained",
      np.linalg.norm(N.T @ U[:, :4]) > 1.0, f"(|N^T U_src| = {np.linalg.norm(N.T @ U[:, :4]):.2f})")

# ---- 5. no-op cases
check("project_out with k=0 is identity", np.array_equal(project_out(Zr, np.zeros((d, 0))), Zr))
check("choose_k picks the smallest k within tol of the minimum", choose_k({1: 0.9, 2: 0.7, 3: 0.69, 5: 0.5}, 0.02) == 5 and choose_k({1: 0.6, 2: 0.59, 3: 0.58}, 0.02) == 1)

# ---- 6. r3_label on crafted inputs
good = {"V": dict(auc=0.9, fpr=0.02, auc7=0.92, fpr7=0.006), "R": dict(auc=0.88, fpr=0.04, auc7=0.69, fpr7=0.19), "O": dict(auc=0.86, fpr=0.05, auc7=0.81, fpr7=0.16)}
check("r3: PASS-R3", r3_label(good)["label"] == "PASS-R3")
bad = {"V": dict(auc=0.92, fpr=0.006, auc7=0.917, fpr7=0.006), "R": dict(auc=0.70, fpr=0.19, auc7=0.69, fpr7=0.19), "O": dict(auc=0.82, fpr=0.15, auc7=0.81, fpr7=0.16)}
check("r3: KILL (no source improves by >= 0.03 AUROC or >= 5 FPR points)", r3_label(bad)["label"] == "KILL")
mid = {**bad, "O": dict(auc=0.86, fpr=0.06, auc7=0.81, fpr7=0.16)}
check("r3: INCONCLUSIVE (one source improves, not all pass)", r3_label(mid)["label"] == "INCONCLUSIVE")
check("r3 boundary: +0.03 AUROC counts as an improvement (not KILL); +0.029 does (KILL)",
      r3_label({"A": dict(auc=0.78, fpr=0.2, auc7=0.75, fpr7=0.2)})["label"] == "INCONCLUSIVE"
      and r3_label({"A": dict(auc=0.779, fpr=0.2, auc7=0.75, fpr7=0.2)})["label"] == "KILL")

# ---- 7. real-data plumbing
E = run.e7()
for arm in run.CFG["arms"]:
    A = run.get_arm(arm)
    D = A.D
    F, R = run.twin_pairs(A)
    check(f"{arm}: twin pairs are train-role, RAISE real with its own caption id", (D.role[F] == "train").all() and (D.role[R] == "train").all()
          and (D.pool[R] == "raise").all() and (D.rid[F] == D.rid[R]).all(), f"({len(F)} pairs, {len(set(R.tolist()))} reals)")
    for S in run.SOURCES:
        T = run.others_of(S)
        pools_S = set(E.SRC[S]["train_pools"] + E.SRC[S]["eval_pools"])
        rows = E.train_rows(D, T, "full")
        check(f"{arm}: training rows of held-out {S} contain no {S} pool and no non-train role", not (set(D.pool[rows]) & pools_S) and (D.role[rows] == "train").all())
    tr, ev = set(D.rid[(D.role == "train") & (D.rid != "")]), set(D.rid[np.isin(D.role, ["calib", "test"]) & (D.rid != "")])
    check(f"{arm}: no caption id shared by train and calib/test", not (tr & ev))
    # E7 evaluation reproduced from its cached scores
    ctx = run.Ctx(arm)
    S = "OI"
    T = run.others_of(S)
    SC7 = E.load_scores(arm, "F_b8", T)[0]
    ev7 = run.eval_scores(ctx, S, SC7, E.calib_rows(D, T), {c: D.conds.index(c) for c in run.EVAL})
    sm = run.ev_summary(ev7, ctx)[run.JR]
    import json
    r7 = json.loads((HERE.parent / "016-loso" / "results" / "results.json").read_text())["arms"][arm]["heldout"][S]["+".join(T)]["F_b8"]["conds"][run.JR]
    check(f"{arm}: E7 OI AUROC / FPR reproduced by eval_scores", abs(sm["auroc"][0] - r7["macro"][0]) < 1e-9 and abs(sm["fpr"] - r7["fpr"]["A"]["full"]["fpr"]) < 1e-12,
          f"(AUROC {sm['auroc'][0]:.4f} vs {r7['macro'][0]:.4f}, FPR {sm['fpr']:.3f})")

# ---- 8. refit with k=0 projection reproduces E7's cached scores (plumbing of the projected-fit path)
arm = "fov512"
A = run.get_arm(arm)
S, T = "OI", run.others_of("OI")
rows = E.train_rows(A.D, T, "full")
rr = rows[A.D.lab[rows] == 0]
Xr = np.concatenate([A.X[c][rr][:, run.FS["F_b8"]] for c in run.TAUG])
mu_, sd_ = Xr.mean(0), safe_sigma(Xr)
# j=0 already means k = n_src-1 = 1 LDA direction, so the k=0 path is exercised by hand with a zero-column basis
Uz = np.zeros((1536, 0))
Xtr = np.concatenate([project_out((A.X[c][rows][:, run.FS["F_b8"]] - mu_) / sd_, Uz) for c in run.TAUG])
y = np.concatenate([A.D.lab[rows]] * len(run.TAUG))
g = np.concatenate([A.D.groups[rows]] * len(run.TAUG))
m = run.probe.fit_probe(Xtr, y, 0, groups=g, class_weight="balanced")
new = m.decision_function((A.X["jpeg75_resize05"][:, run.FS["F_b8"]] - mu_) / sd_)
old = E.load_scores(arm, "F_b8", T)[0]["full"][A.D.conds.index("jpeg75_resize05")]
check("fov512 k=0 refit on standardized features reproduces E7's cached scores (affine-equivalent)",
      np.corrcoef(new, old)[0, 1] > 0.99999, f"(corr {np.corrcoef(new, old)[0, 1]:.7f}, max|d| {np.abs(new - old).max():.2e})")

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + f"\n{'ALL PASS' if ok_all else 'SOME FAILED'}\n")
print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
