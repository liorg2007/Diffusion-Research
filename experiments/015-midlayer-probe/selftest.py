"""Self-test of E6 (run: env/.venv/bin/python -I experiments/015-midlayer-probe/selftest.py). Synthetic checks + real-data design checks.
Output also written to results/selftest.txt."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run

OUT = []
ok_all = True


def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    OUT.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    print(OUT[-1])


rng = np.random.default_rng(0)

# ---- 1. Clopper-Pearson
lo, hi = run.clopper_pearson(0, 50)
check("CP 0/50 upper 0.0711", abs(hi - 0.0711) < 5e-4 and lo == 0.0, f"({lo:.4f},{hi:.4f})")
lo, hi = run.clopper_pearson(5, 50)
check("CP 5/50 [0.0332,0.2181]", abs(lo - 0.0332) < 5e-4 and abs(hi - 0.2181) < 5e-4, f"({lo:.4f},{hi:.4f})")

# ---- 2. roles on a synthetic frame: caption families whole, no id shared train vs calib/test
rows = []
for k in range(40):
    r = f"r{k:08x}t"
    sp_real = rng.choice(["train", "calib", "test"], p=[.4, .15, .45])
    rows.append(dict(src_path=f"raw/raise/{r}.png", label=0, generator="real", pool="raise", device="", split=sp_real, sha256=f"R{k}"))
    for g in ("dalle2", "flux", "sd3_large"):
        sp = "test" if g == "flux" else rng.choice(["train", "test"])
        rows.append(dict(src_path=f"raw/gen/{g}/{r}.png", label=1, generator=g, pool="fake", device="", split=sp, sha256=f"{g}{k}"))
for d in range(6):
    for j in range(5):
        rows.append(dict(src_path=f"raw/vision/D{d}_{j}.jpg", label=0, generator="real", pool="vision_native", device=f"D{d:02d}",
                         split="train" if d < 3 else "test", sha256=f"V{d}{j}"))
sd = pd.DataFrame(rows)
role, rid, side = run.build_roles(sd)
tr, ev = set(rid[(role == "train") & (rid != "")]), set(rid[np.isin(role, ["calib", "test"]) & (rid != "")])
check("synthetic: no caption id shared by train and calib/test", len(tr & ev) == 0, f"(train ids {len(tr)}, eval ids {len(ev)})")
kept = role != "drop"
check("synthetic: kept rows keep their official split", (role[kept] == sd.split.to_numpy()[kept]).all())
check("synthetic: rows without a caption id are untouched", (role[rid == ""] == sd.split.to_numpy()[rid == ""]).all())
fam_ok = all(len({("train" if role[i] == "train" else "eval") for i in np.where((rid == r) & kept)[0]}) <= 1 for r in set(rid) - {""})
check("synthetic: every family is on one side", fam_ok)
grp = run.make_groups(sd, rid)
j = [i for i in range(len(sd)) if rid[i] == rid[0]]
check("groups: a RAISE real and all its fakes share one group", len({grp[i] for i in j}) == 1 and grp[j[0]].startswith("rid:"))
check("groups: VISION copies share their device group", grp[np.where(sd.device == "D00")[0]].tolist() == ["dev:D00"] * 5)

# ---- 3. train_rows: LOGO excludes the generator, sources restrict reals, FLUX never trains
class Fake(run.Arm):
    def __init__(self, df, X):
        self.name, self.df = "syn", df
        self.role, self.rid, self.side = run.build_roles(df)
        self.groups = run.make_groups(df, self.rid)
        self.lab, self.gen, self.pool = df.label.to_numpy(), df.generator.to_numpy(), df.pool.to_numpy()
        self.X = X


# rename synthetic generators to the real list so TRAINGENS applies
sd["generator"] = [("dalle3" if g == "sd3_large" else g) for g in sd.generator]
n = len(sd)
X = {}
for c in run.CONDS:
    x = rng.normal(size=(n, 4608)).astype(np.float32)
    X[c] = x
lab = sd.label.to_numpy()
# plant: fake signal ONLY in block 8 columns (1536:3072), same shift under every condition
for c in run.CONDS:
    X[c][lab == 1, 1536:1700] += 1.2
A = Fake(sd, X)
rows_g = run.train_rows(A, "all", "g_dalle3")
check("LOGO model excludes the left-out generator", not (A.gen[rows_g] == "dalle3").any())
check("FLUX is never a training row", not (A.gen[run.train_rows(A, "all", "full")] == "flux").any())
r_only = run.train_rows(A, "raise_only", "full")
check("raise_only trains on RAISE reals only", set(A.pool[r_only][A.lab[r_only] == 0]) == {"raise"})
v_only = run.train_rows(A, "vision_only", "full")
check("vision_only trains on VISION natives only", set(A.pool[v_only][A.lab[v_only] == 0]) == {"vision_native"})
check("training rows are role==train", (A.role[run.train_rows(A, "all", "full")] == "train").all())

# ---- 4. end-to-end probe: planted block-8 signal found by F_b8 and F_cat, not by F_last (block 12 columns are noise)
sc8, C8, ntr = run.fit_scores(A, "F_b8", "T-aug", "all", "full")
sc12, _, _ = run.fit_scores(A, "F_last", "T-clean", "all", "full")
te = A.idx("test")
a8 = roc_auc_score(A.lab[te], sc8[0][te])
a12 = roc_auc_score(A.lab[te], sc12[0][te])
check("planted block-8 signal: F_b8 AUROC high", a8 > 0.95, f"({a8:.3f})")
check("F_last on the same data is near chance (columns checked)", abs(a12 - 0.5) < 0.2, f"({a12:.3f})")
check("scores shape [conds, N]", sc8.shape == (len(run.CONDS), n))

# ---- 5. Plan + paired bootstrap on this synthetic arm: point estimate == sklearn AUROC; null-diff CI covers 0
SG = ["dalle2", "dalle3", "flux"]
pl = run.Plan(A, run.POOLS["primary"], gens=SG, n_boot=200)
S = {m: sc8 for m in run.MODELS}
ga = pl.gen_auc(S, 0)
g0 = "dalle2"
idx = np.r_[pl.R, pl.F[g0]]
check("weighted AUROC point == sklearn", abs(ga[g0][0] - roc_auc_score(np.r_[np.zeros(len(pl.R)), np.ones(len(pl.F[g0]))], sc8[0][idx])) < 1e-12)
d = run.diff_summary(run.macro(ga, SG), run.macro(ga, SG))
check("paired diff of identical scores is exactly 0 with CI [0,0]", d["delta"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0)
Sperm = {m: sc8[:, rng.permutation(n)] for m in run.MODELS}
gp = pl.gen_auc(Sperm, 0)
dp = run.diff_summary(run.macro(ga, SG), run.macro(gp, SG))
check("signal vs shuffled scores: positive delta, CI > 0", dp["delta"] > 0.2 and dp["ci_lo"] > 0, f"({dp['delta']:.3f})")
mp = run.macro(gp, SG)
check("shuffled scores: macro AUROC near 0.5", abs(mp[0] - 0.5) < 0.12, f"({mp[0]:.3f})")

# ---- 6. TwinPlan: planted pair shift is found, null is 0.5
tp = run.TwinPlan(A)
check("twin plan has pairs and ids", sum(tp.n_pairs.values()) > 0 and tp.n_ids > 0, f"(ids {tp.n_ids}, pairs {sum(tp.n_pairs.values())})")
Sp = {m: np.tile(np.where(A.lab == 1, 1.0, 0.0) + rng.normal(0, 0.1, n), (len(run.CONDS), 1)) for m in run.MODELS}
ta = tp.auc(Sp, 0)
check("twin AUROC with perfectly separating scores = 1, win rate 1", ta["pooled"][0] > 0.999 and ta["win"] > 0.999)
Sn = {m: np.tile(rng.normal(size=n), (len(run.CONDS), 1)) for m in run.MODELS}
tn = tp.auc(Sn, 0)
lo_, hi_ = run.summ(tn["pooled"])[1:]
check("twin AUROC null: CI covers 0.5", lo_ < 0.5 < hi_, f"([{lo_:.2f},{hi_:.2f}])")

# ---- 7. metrics helper: balanced accuracy / threshold
thr = run.metrics.threshold_for_fpr(np.arange(200.0), 0.01)
check("threshold at 1% of 200 calib reals allows 2 false positives", ((np.arange(200.0) >= thr).sum()) == 2)

# ---- 8. real data
for arm in ("orig", "fov512"):
    R = run.Arm(arm)
    df = R.df
    tr, ev = set(R.rid[(R.role == "train") & (R.rid != "")]), set(R.rid[np.isin(R.role, ["calib", "test"]) & (R.rid != "")])
    check(f"[{arm}] real: no caption id in both train and calib/test", len(tr & ev) == 0)
    kept = R.role != "drop"
    check(f"[{arm}] real: kept rows keep the official split", (R.role[kept] == df.split.to_numpy()[kept]).all())
    check(f"[{arm}] real: FLUX has no train rows", not ((R.gen == "flux") & (R.role == "train")).any())
    check(f"[{arm}] real: VISION rows all kept", (R.role[np.char.startswith(R.pool.astype(str), "vision")] != "drop").all())
    vt = R.idx("test", ["vision_native"], label=0)
    vtr = R.idx("train", ["vision_native"], label=0)
    check(f"[{arm}] real: VISION test devices disjoint from train devices", not (set(df.device.to_numpy()[vt]) & set(df.device.to_numpy()[vtr])))
    for c in run.CONDS:
        check(f"[{arm}] real: features cached for every row, {c}", R.X[c].shape == (len(df), 4608) and np.isfinite(R.X[c]).all())
import p4features as P
keys, Xs = P.load_features("secondary", "clean")
_, Xp = P.load_features("primary", "clean")
check("primary (B0) == last 1536 columns of secondary (block 12)", np.array_equal(Xp, Xs[:, 3072:4608]))
check("F_b8 slice is columns 1536:3072 and F_last 3072:4608", (run.FEATS["F_b8"].start, run.FEATS["F_b8"].stop) == (1536, 3072) and
      (run.FEATS["F_last"].start, run.FEATS["F_last"].stop) == (3072, 4608))
check("block 8 differs from block 12 features", not np.allclose(Xs[:, 1536:3072], Xs[:, 3072:4608]))
a, b = run.Arm("orig", False), run.Arm("fov512", False)
check("orig and fov512 splits identical apart from path", (a.df.drop(columns="path").values == b.df.drop(columns="path").values).all())

# ---- 9. Laplacian texture on synthetic images: smooth < noisy
from PIL import Image
smooth = Image.fromarray(np.full((256, 256, 3), 128, np.uint8))
noisy = Image.fromarray(rng.integers(0, 255, (256, 256, 3)).astype(np.uint8))
check("lap224: flat image 0, noisy image large", run.lap224(smooth) < 1e-6 and run.lap224(noisy) > 0.2)

print("ALL PASS" if ok_all else "SOME FAILED")
OUT.append("ALL PASS" if ok_all else "SOME FAILED")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + "\n")
sys.exit(0 if ok_all else 1)
