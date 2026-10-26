"""Self-test of E9 (run: env/.venv/bin/python -I experiments/018-real-anchor/selftest.py). Synthetic checks of the adaptation logic, then plumbing checks on
the real data. The real-data checks that touch ADAPTED scores use the VISION fold only (so the self-test can be run before the method choice is frozen);
for RAISE/OI only structural checks (pools, groups, nesting) are made. Output also written to results/selftest.txt."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))

import numpy as np

import run as R

OUT, ok_all = [], True


def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    OUT.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    print(OUT[-1], flush=True)


rng = np.random.default_rng(0)

# ---- 1. pure helpers
v = R.unit(rng.normal(size=30))
Z = rng.normal(size=(100, 30))
check("project_dir removes the direction", np.abs(R.project_dir(Z, v) @ v).max() < 1e-12)
check("anchor_delta = train mean - anchor mean", np.allclose(R.anchor_delta([1, 2], [0.5, 5]), [0.5, -3]))
x = rng.normal(size=300)
check("thr_fpr 1% on 300 reals leaves <= 3 above", (x >= R.thr_fpr(x)).sum() <= 3)
check("thr_fpr 1% on 50 reals = max (0 false positives allowed)", R.thr_fpr(x[:50]) > x[:50].max())
check("choose_method: higher AUROC wins", R.choose_method({"A-off": dict(auroc=0.90, fpr=0.05), "A-proj": dict(auroc=0.92, fpr=0.09)}, 0.002) == "A-proj")
check("choose_method: tie -> lower FPR", R.choose_method({"A-off": dict(auroc=0.9201, fpr=0.05), "A-proj": dict(auroc=0.9199, fpr=0.01)}, 0.002) == "A-proj")
per_ok = {s: dict(auroc=0.92, fpr=0.02) for s in "VRO"}
check("label PASS", R.label_of(per_ok, dict(V=0.01, R=0.02, O=-0.01))["label"] == "PASS")
check("label KILL (AUROC < 0.85)", R.label_of({**per_ok, "R": dict(auroc=0.84, fpr=0.02)}, dict(V=1, R=1, O=1))["label"] == "KILL")
check("label KILL (FPR > 8%)", R.label_of({**per_ok, "R": dict(auroc=0.95, fpr=0.081)}, dict(V=1, R=1, O=1))["label"] == "KILL")
check("label KILL (no gain on any source)", R.label_of(per_ok, dict(V=-0.01, R=0.0, O=-0.2))["label"] == "KILL")
check("label INCONCLUSIVE (AUROC 0.88, gain on 1 source)", R.label_of({**per_ok, "R": dict(auroc=0.88, fpr=0.02)}, dict(V=0.01, R=-1, O=-1))["label"] == "INCONCLUSIVE")
check("label INCONCLUSIVE (gain on 1 of 3 only)", R.label_of(per_ok, dict(V=0.01, R=-1, O=-1))["label"] == "INCONCLUSIVE")

# ---- 2. matched-TPR FPR: weighted vs brute force, and expansion by integer weights
sr = rng.normal(0, 1, 120)
sfs = [rng.normal(1.2, 1, 40), rng.normal(0.8, 1.3, 55), rng.normal(2.0, 1, 30)]
ones_r = np.ones((1, len(sr)))
ones_f = [np.ones((1, len(s))) for s in sfs]
target = np.array([0.6])
got = R.fpr_at_tpr_w(sr, sfs, ones_r, ones_f, target)[0]
cand = np.sort(np.concatenate(sfs))[::-1]
best = None
for t in cand:
    tpr = np.mean([(s >= t).mean() for s in sfs])
    if tpr >= 0.6 - 1e-12:
        best = t
        break
check("fpr_at_tpr_w = brute force at TPR 0.6", abs(got - (sr >= best).mean()) < 1e-12, f"({got:.4f})")
cr = rng.integers(0, 3, size=(1, len(sr))).astype(float)
cf = [rng.integers(0, 3, size=(1, len(s))).astype(float) for s in sfs]
srx = np.repeat(sr, cr[0].astype(int))
sfx = [np.repeat(s, c[0].astype(int)) for s, c in zip(sfs, cf)]
g1 = R.fpr_at_tpr_w(sr, sfs, cr, cf, target)[0]
g2 = R.fpr_at_tpr_w(srx, sfx, np.ones((1, len(srx))), [np.ones((1, len(s))) for s in sfx], target)[0]
check("integer weights == row expansion", abs(g1 - g2) < 1e-12, f"({g1:.4f} vs {g2:.4f})")
tp, fp = R.op_point_w(sr, sfs, ones_r, ones_f, 0.5)
check("op_point_w = direct", abs(tp[0] - np.mean([(s >= 0.5).mean() for s in sfs])) < 1e-12 and abs(fp[0] - (sr >= 0.5).mean()) < 1e-12)
gain_self = fp[0] - R.fpr_at_tpr_w(sr, sfs, ones_r, ones_f, tp)[0]
check("gain of a score over itself at its own operating point = 0", abs(gain_self) < 1e-12, f"({gain_self:.2e})")

# ---- 3. synthetic held-out sources: pure offset -> A-off / A-proj recover in-source AUROC; offset + rotation -> partial
d, ntr, nte, NA = 40, 500, 600, 50
u1 = np.zeros(d); u1[0] = 1.0                         # the diffusion direction
o = np.zeros(d); o[0], o[1] = 2.0, 7.0          # the new pipeline's offset: partly along the diffusion direction (reals look "more fake"), partly in a nuisance direction
sdv = np.ones(d); sdv[2] = 3.0                        # training reals vary strongly along e2


def reals(mean, n, rot=None):
    z = rng.normal(size=(n, d)) * sdv
    if rot is not None:
        z = z @ rot.T
    return mean + z


def fakes(mean, n):
    return mean + 3.0 * u1 + rng.normal(size=(n, d)) * sdv


mA, mB = rng.normal(0, 2, d), rng.normal(0, 2, d)
Xtr = np.vstack([reals(mA, ntr), fakes(mA, ntr), reals(mB, ntr), fakes(mB, ntr)])
ytr = np.r_[np.zeros(ntr), np.ones(ntr), np.zeros(ntr), np.ones(ntr)].astype(int)
mu_tr = Xtr[ytr == 0].mean(0)
w, b, C, mdl = R.fit_raw(Xtr, ytr, np.arange(len(ytr)))
m_fake = (mA + mB) / 2                                  # held-out generator's fakes: no new-pipeline offset
F_test = fakes(m_fake, nte)
mS = (mA + mB) / 2


def auc_off(S_real_test, anchors):
    delta = R.anchor_delta(mu_tr, anchors.mean(0))
    return R.auroc_point(S_real_test @ w + b + w @ delta, F_test @ w + b)


def auc_proj(S_real_test, anchors):
    sd = R.safe_sigma(Xtr[ytr == 0])
    mu = Xtr[ytr == 0].mean(0)
    vv = R.unit((anchors.mean(0) - mu) / sd)
    Z = R.project_dir((Xtr - mu) / sd, vv)
    m = R.probe.fit_probe(Z, ytr, 0, groups=np.arange(len(ytr)), class_weight="balanced")
    return R.auroc_point(m.decision_function(R.project_dir((S_real_test - mu) / sd, vv)), m.decision_function(R.project_dir((F_test - mu) / sd, vv)))


# in-source reference: training-pipeline reals vs the same fakes
ref = R.auroc_point(reals(mS, nte) @ w + b, F_test @ w + b)
S1_test, S1_anc = reals(mS + o, nte), reals(mS + o, NA)
un1 = R.auroc_point(S1_test @ w + b, F_test @ w + b)
a_off1, a_pr1 = auc_off(S1_test, S1_anc), auc_proj(S1_test, S1_anc)
check("pure offset: unadapted AUROC is far below in-source", un1 < ref - 0.15, f"(unadapted {un1:.3f}, in-source {ref:.3f})")
check("pure offset: A-off recovers the in-source AUROC", a_off1 > ref - 0.03, f"(A-off {a_off1:.3f})")
check("pure offset: A-proj recovers the in-source AUROC", a_pr1 > ref - 0.03, f"(A-proj {a_pr1:.3f})")
check("pure offset: A-off on a source WITHOUT an offset changes nothing", abs(auc_off(reals(mS, nte), reals(mS, NA)) - ref) < 0.02)
# offset + rotation: the new pipeline's real variability lies along the detector axis (90 deg rotation in the (u1, e2) plane)
Rm = np.eye(d)
Rm[0, 0] = Rm[2, 2] = 0.0
Rm[0, 2], Rm[2, 0] = -1.0, 1.0
S2_test, S2_anc = reals(mS + o, nte, Rm), reals(mS + o, NA, Rm)
un2 = R.auroc_point(S2_test @ w + b, F_test @ w + b)
a_off2, a_pr2 = auc_off(S2_test, S2_anc), auc_proj(S2_test, S2_anc)
check("offset + rotation: A-off improves on unadapted", a_off2 > un2 + 0.05, f"(unadapted {un2:.3f}, A-off {a_off2:.3f})")
check("offset + rotation: A-off is only PARTIAL (well below in-source)", a_off2 < ref - 0.05, f"(A-off {a_off2:.3f}, in-source {ref:.3f})")
check("offset + rotation: A-proj also only partial", a_pr2 < ref - 0.03, f"(A-proj {a_pr2:.3f})")
# A-off on reals only moves AUROC; A-off-all (everything shifted) leaves AUROC unchanged
delta = R.anchor_delta(mu_tr, S1_anc.mean(0))
all_shift = R.auroc_point(S1_test @ w + b + w @ delta, F_test @ w + b + w @ delta)
check("A-off-all (shift reals and fakes) leaves AUROC unchanged", abs(all_shift - un1) < 1e-12, f"({all_shift:.4f} vs {un1:.4f})")
# threshold at 1% of training calib reals: FPR on the pure-offset source before/after
calib = reals(mS, 400)
thr = R.thr_fpr(calib @ w + b)
f_un = ((S1_test @ w + b) >= thr).mean()
f_off = ((S1_test @ w + b + w @ R.anchor_delta(mu_tr, S1_anc.mean(0))) >= thr).mean()
check("pure offset: FPR at the calib threshold falls from large to ~1-3%", f_un > 0.2 and f_off < 0.04, f"({f_un:.3f} -> {f_off:.3f})")

# ---- 4. real data (VISION adapted; structural only for RAISE/OI)
ctx = R.get_ctx("fov512")
D = ctx.D
for arm in R.CFG["arms"]:
    c = R.get_ctx(arm)
    for S in R.SOURCES:
        pool, te = c.pool[S], c.rr[S]
        ok_roles = set(c.D.role[pool]) <= {"train", "calib"} and (c.D.lab[pool] == 0).all() and np.isin(c.D.pool[pool], c.E.SRC[S]["train_pools"]).all()
        check(f"[{arm}] {S}: anchor pool = train/calib-role reals of the source's train pools ({len(pool)})", ok_roles and len(pool) >= max(R.NS))
        check(f"[{arm}] {S}: anchor pool rows and groups disjoint from the source's test reals", not set(pool) & set(te) and not set(c.D.groups[pool]) & set(c.D.groups[te]))
        o50, o25 = c.anchor_order(S, 3)[:50], c.anchor_order(S, 3)[:25]
        check(f"[{arm}] {S}: draws nested, no repeats, seeded", len(set(o50)) == 50 and (o50[:25] == o25).all() and (c.anchor_order(S, 3)[:50] == o50).all()
              and not (c.anchor_order(S, 4)[:50] == o50).all())
check("both arms draw the same anchor rows", all((ctx.anchor_order(S, 0) == R.get_ctx("orig").anchor_order(S, 0)).all() for S in R.SOURCES))
S = "VISION"
T = ctx.others(S)
rows_tr = ctx.E.train_rows(D, T, "full")
check("VISION fold: training rows never include VISION", not np.isin(D.pool[rows_tr], ["vision_native", "vision_whatsapp", "vision_facebook_hq"]).any())
sl = R.FS["F_b8"]
base, W = R.base_scores(ctx, "F_b8", S)
st = R.static_stats(ctx, "F_b8", S)
reals_tr = rows_tr[D.lab[rows_tr] == 0]
check("training-reals offset reference mean uses only train-role training-source reals", (D.role[reals_tr] == "train").all() and (D.lab[reals_tr] == 0).all())
anc = ctx.anchor_order(S, 0)[:50]
_, mu_an = R.anchor_means(ctx, "F_b8", anc)
delta = R.anchor_delta(st["mu"], mu_an)
shifted = (ctx.X[R.JR][ctx.rr[S]][:, sl].astype(np.float64) + delta) @ W[-1] + np.load(R.probe_path("fov512", "F_b8", S))["b"][-1]
lin = base[-1, R.CI_JR, ctx.rr[S]] + W[-1] @ delta
check("A-off: shifting features == shifting the score by w.delta (linear probe)", np.abs(shifted - lin).max() < 1e-8, f"({np.abs(shifted - lin).max():.1e})")
sd = st["sd"]
v = R.unit((mu_an - st["mu_stack"]) / sd)
Zr = R.project_dir((np.concatenate([ctx.X[c][reals_tr][:, sl] for c in R.TAUG]).astype(np.float64) - st["mu_stack"]) / sd, v)
check("A-proj: projected training reals have zero mean component along the offset direction's unit vector", np.abs(Zr.mean(0) @ v) < 1e-9 and abs(np.linalg.norm(v) - 1) < 1e-12)
check("A-proj direction aligns with the standardized offset", abs(R.unit((mu_an - st["mu_stack"]) / sd) @ ((mu_an - st["mu_stack"]) / sd)) > 0)
off = float(np.linalg.norm((mu_an - st["mu_stack"]) / sd))
check("VISION anchor offset is finite and non-zero", np.isfinite(off) and off > 0, f"(norm {off:.3f})")
# the E7 reproduction itself is asserted in stage_probes (max |score - cached E7 score| < 1e-3); echo it here
for arm in R.CFG["arms"]:
    for S2 in R.SOURCES:
        dd = R.check_against_e7(arm, "F_b8", S2)
        check(f"[{arm}] E7 probe refit reproduces cached E7 scores, held-out {S2}", dd < 1e-3, f"(max diff {dd:.1e})")
# protocol guard: no adapted RAISE/OI cache may exist before choice.json
import json
cp = HERE / "results" / "choice.json"
leak = [f.name for f in (HERE / "cache").rglob("*") if f.is_file() and f.parent.name in ("proj", "eval") and any(f"__{s}__" in f.name for s in ("RAISE", "OI"))]
if cp.exists():
    ts = json.loads(cp.read_text())["frozen_epoch"]
    first = min([f.stat().st_mtime for f in (HERE / "cache").rglob("*") if f.is_file() and f.parent.name in ("proj", "eval") and any(f"__{s}__" in f.name for s in ("RAISE", "OI"))], default=None)
    check("protocol: choice.json frozen before the first RAISE/OI adapted cache file", first is None or first > ts)
else:
    check("protocol: no RAISE/OI adapted cache exists before choice.json", not leak, f"({len(leak)} files)")

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + f"\nALL {'PASS' if ok_all else 'FAIL'}\n")
print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
