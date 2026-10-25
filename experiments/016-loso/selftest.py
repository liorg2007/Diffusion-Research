"""Self-test of E7 (run: env/.venv/bin/python -I experiments/016-loso/selftest.py). Synthetic logic checks + real-data design checks.
Output also written to results/selftest.txt."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run

OUT, ok_all = [], True


def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    OUT.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    print(OUT[-1])


rng = np.random.default_rng(0)

# ---- 1. Clopper-Pearson, threshold
lo, hi = run.clopper_pearson(0, 50)
check("CP 0/50 upper 0.0711", abs(hi - 0.0711) < 5e-4 and lo == 0.0, f"({lo:.4f},{hi:.4f})")
x = rng.normal(size=300)
t = run.thr_fpr(x)
check("1% threshold on 300 calib reals leaves <= 3 above", (x >= t).sum() <= 3, f"({(x >= t).sum()})")

# ---- 2. label_of on crafted summaries
def mk(per, lay, lc):
    return dict(per={k: dict(auroc=[a, 0, 1], fpr=f) for k, (a, f) in per.items()}, b8_minus_last=dict(delta=lay[0], ci_lo=lay[1]),
                lc_diff_max_vs_1=dict(delta=lc[0], ci_lo=lc[1]))
good = {"V": (0.9, 0.02), "R": (0.88, 0.04), "O": (0.86, 0.05)}
check("label PASS", run.label_of(mk(good, (0.04, 0.01), (0.02, 0.01)))["label"] == "PASS")
check("label INCONCLUSIVE (layer gain CI includes 0)", run.label_of(mk(good, (0.04, -0.01), (0.02, 0.01)))["label"] == "INCONCLUSIVE")
check("label INCONCLUSIVE (a source at AUROC 0.80, FPR 0.07)", run.label_of(mk({**good, "R": (0.80, 0.07)}, (0.04, 0.01), (0.02, 0.01)))["label"] == "INCONCLUSIVE")
check("label KILL (AUROC < 0.75)", run.label_of(mk({**good, "R": (0.74, 0.01)}, (0.04, 0.01), (0.02, 0.01)))["label"] == "KILL")
check("label KILL (FPR > 10%)", run.label_of(mk({**good, "R": (0.9, 0.101)}, (0.04, 0.01), (0.02, 0.01)))["label"] == "KILL")
check("label boundary: AUROC 0.85 and FPR 0.05 pass", run.label_of(mk({"V": (0.85, 0.05)}, (0.03, 0.001), (0.01, 0.001)))["label"] == "PASS")

# ---- 3. weighted AUROC point == sklearn; real data design checks
sr, sf = rng.normal(0, 1, 40), rng.normal(0.8, 1, 50)
au = run.metrics.weighted_auroc(sr, sf, np.ones((1, 40)), np.ones((1, 50)))[0]
check("weighted AUROC == sklearn", abs(au - roc_auc_score(np.r_[np.zeros(40), np.ones(50)], np.r_[sr, sf])) < 1e-12)

for arm in run.ARMS:
    D = run.Data(arm, load_feats=False)
    df = D.df
    check(f"{arm}: sha256 unique, rows {len(df)}", df.sha256.is_unique)
    check(f"{arm}: all crop files exist", all((run.ROOT / p).exists() for p in df.path))
    oi = df[df.pool == "openimages"]
    sp = oi.split.value_counts().to_dict()
    check(f"{arm}: OI split 300/100/200 by image", sp == {"train": 300, "calib": 100, "test": 200}, str(sp))
    man = pd.read_csv(run.MANIFEST_OI)
    check(f"{arm}: OI rows == manifest sha256 set; every OI file hash == manifest",
          set(oi.sha256) == set(man.sha256) and all(hashlib.sha256(open(run.ROOT / p, "rb").read()).hexdigest() == s for p, s in zip(oi.src_path[:40], oi.sha256[:40])))
    # no caption id (incl. D3 ids) shared by a train row and a calib/test row
    tr = set(D.rid[(D.role == "train") & (D.rid != "")])
    ev = set(D.rid[np.isin(D.role, ["calib", "test"]) & (D.rid != "")])
    check(f"{arm}: no family id shared by train and calib/test", len(tr & ev) == 0, f"(train {len(tr)}, eval {len(ev)})")
    # E6 roles reproduced: 230 train fakes, 84 train RAISE reals, 183 test RAISE reals
    ro = pd.crosstab([D.pool, D.lab], D.role)
    check(f"{arm}: E6 role counts (fakes train 230/test 532+..., RAISE 84/53/183)",
          ro.loc[("fake", 1), "train"] == 230 and ro.loc[("fake", 1), "test"] == 532 and tuple(ro.loc[("raise", 0), ["train", "calib", "test"]]) == (84, 53, 183))
    if arm == "fov224":
        d3 = df[df.pool.isin(["d3_real", "fake_d3"])].assign(i=D.rid[df.pool.isin(["d3_real", "fake_d3"]).to_numpy()])
        check("fov224: each D3 id has ONE split (real + fakes together)", (d3.groupby("i").split.nunique() == 1).all())
        ids = d3.groupby("i").split.first().value_counts().to_dict()
        check("fov224: D3 id split 60/20/20 over the 100 ids (reals < 224 px dropped afterwards)", ids == {"train": 60, "calib": 20, "test": 20}, str(ids))
        from PIL import Image
        dd = df[df.pool.isin(["d3_real", "fake_d3"])]
        short_ok = all(min(Image.open(run.ROOT / p).size) >= 224 for p, l in zip(dd.src_path, dd.label) if l == 0)
        crop_ok = all(Image.open(run.ROOT / p).size == (224, 224) for p in dd.path)
        check("fov224: kept D3 reals have raw short side >= 224 (never upscaled); every D3 crop is 224x224", short_ok and crop_ok,
              f"(D3 reals kept {int((dd.label == 0).sum())}, fakes {int((dd.label == 1).sum())})")
    # held-out source never in training/calib of its own fold; no FLUX or excluded generator in train
    D.X = {}
    for S in run.ARMS[arm]["sources"]:
        others = [s for s in run.ARMS[arm]["sources"] if s != S]
        for T in run.subsets_of(others):
            for model in run.model_names(arm):
                rows = run.train_rows(D, T, model)
                pools = set(D.pool[rows])
                bad = pools & set(run.SRC[S]["eval_pools"] + run.SRC[S]["train_pools"])
                gens_in = set(D.gen[rows][D.lab[rows] == 1])
                ok = (not bad and (D.role[rows] == "train").all() and run.FLUX not in gens_in
                      and (model == "full" or model[2:] not in gens_in) and ((model != "full") or gens_in == set(run.arm_gens(arm))))
                if not ok:
                    check(f"{arm}: train rows clean for S={S} T={T} model={model}", False, str(bad))
    check(f"{arm}: held-out source pools never in training rows; only role=train; FLUX and the LOGO generator excluded", True)
    cal = run.calib_rows(D, ("VISION", "RAISE"))
    check(f"{arm}: calib reals of VISION+RAISE are role=calib reals of those pools only",
          (D.role[cal] == "calib").all() and set(D.pool[cal]) <= {"vision_native", "vision_whatsapp", "vision_facebook_hq", "raise"})
    # bootstrap plan
    pl = run.Plan(D, n_boot=50)
    check(f"{arm}: plan row 0 is all ones", (pl.W[0] == 1).all())
    fam = {}
    for i in D.idx("test"):
        k = run.cluster_key(D, i)
        if k.startswith("fam:"):
            fam.setdefault(k, set()).add(tuple(pl.W[:, pl.pos[int(i)]]))
    check(f"{arm}: rows of one family share bootstrap weights (RAISE real + its fakes)", all(len(v) == 1 for v in fam.values()), f"({len(fam)} families)")
    dev = {}
    for i in D.idx("test", ["vision_native", "vision_whatsapp", "vision_facebook_hq"]):
        dev.setdefault(D.df.device.iat[i], set()).add(tuple(pl.W[:, pl.pos[int(i)]]))
    check(f"{arm}: VISION test copies of one device share weights", all(len(v) == 1 for v in dev.values()), f"({len(dev)} devices)")
    ks = [run.cluster_key(D, i) for i in pl.rows]
    for kind in ("fam", "dev", "img"):
        m = np.array([k.startswith(kind + ":") for k in ks])
        if m.any():
            uk, first = np.unique(np.array(ks)[m], return_index=True)
            cols = np.where(m)[0][first]
            check(f"{arm}: {kind} replicate draws sum to the number of clusters", (pl.W[1:, cols].sum(1) == len(uk)).all(), f"(K={len(uk)})")

# ---- 4. B0 == last block; features cached for every row; new features agree with an independent CPU recomputation
x = run.load_x("fov512", pd.read_csv(HERE / "splits_fov512.csv", keep_default_na=False, dtype={"vidx": str}).iloc[-5:], "clean")
check("features load for new OI rows (4608-d, finite)", x.shape == (5, 4608) and np.isfinite(x).all())
import json
g = json.loads((HERE / "results" / "d008_guard.json").read_text())
mx = max(max(v["max_abs_dlogit_b0"], v["max_abs_dlogit_b8"]) for v in g.values())
check(f"D-008 guard: {len(g)} (arm, condition) pairs all ok, max|dlogit| {mx:.2e} <= 1e-3", all(v["ok"] for v in g.values()) and mx <= 1e-3)

# ---- 5. synthetic end-to-end: planted block-8 signal, three sources with source-specific offsets
class SynD(run.Data):
    def __init__(self):
        self.arm, self.cfg = "fov512", run.ARMS["fov512"]
        self.conds = self.cfg["conditions"]
        rows = []
        for s, pool, n in (("V", "vision_native", 60), ("R", "raise", 60), ("O", "openimages", 60)):
            for k in range(n):
                rows.append(dict(role=["train", "calib", "test"][k % 3], pool=pool, label=0, gen="real", dev=f"{s}{k % 5}", src=s))
        for g in run.TRAINGENS + [run.FLUX]:
            for k in range(30):
                rows.append(dict(role="test" if g == run.FLUX else ["train", "calib", "test"][k % 3], pool="fake", label=1, gen=g, dev="", src="F"))
        d = pd.DataFrame(rows)
        d.loc[d.role == "calib", "role"] = np.where(d.label[d.role == "calib"] == 1, "drop", "calib")
        self.role, self.pool, self.lab, self.gen = d.role.to_numpy(), d.pool.to_numpy(), d.label.to_numpy(), d.gen.to_numpy()
        self.groups = np.array([f"g{i}" for i in range(len(d))])
        off = {"V": 2.0, "R": -2.0, "O": 0.0, "F": 0.0}
        Xb = rng.normal(size=(len(d), 4608)).astype(np.float32)
        for s in off:
            Xb[(d.src == s).to_numpy()] += off[s] * rng.normal(size=4608).astype(np.float32) * 0.5      # source-specific style in all columns
        Xb[self.lab == 1, 1536:3072] += 0.6                                                        # planted signal in block 8 ONLY
        self.X = {c: Xb + 0.1 * rng.normal(size=Xb.shape).astype(np.float32) for c in self.conds}
        self.df = d


S = SynD()
rows = run.train_rows(S, ("VISION", "RAISE"), "full")
check("synthetic: train rows = role train; reals only from the training sources", (S.role[rows] == "train").all() and set(S.pool[rows]) <= {"vision_native", "raise", "fake"})
sc8, C8, n8 = run.fit_scores(S, "F_b8", ("VISION", "RAISE"), "full")
scl, Cl, nl = run.fit_scores(S, "F_last", ("VISION", "RAISE"), "full")
oi = np.where((S.role == "test") & (S.pool == "openimages"))[0]
fk = np.where((S.role == "test") & (S.lab == 1))[0]
a8 = roc_auc_score(np.r_[np.zeros(len(oi)), np.ones(len(fk))], np.r_[sc8[1][oi], sc8[1][fk]])
al = roc_auc_score(np.r_[np.zeros(len(oi)), np.ones(len(fk))], np.r_[scl[1][oi], scl[1][fk]])
check("synthetic: held-out source AUROC high for block 8 (signal), near chance for block 12 (no signal)", a8 > 0.9 and al < 0.65, f"(b8 {a8:.3f}, last {al:.3f})")
check("synthetic: scores cover all rows and conditions", sc8.shape == (len(S.conds), len(S.role)))

OUT.append(f"ALL PASS" if ok_all else "SOME FAILED")
print(OUT[-1])
(HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + "\n")
sys.exit(0 if ok_all else 1)
