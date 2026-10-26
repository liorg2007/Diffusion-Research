"""Self-test of E10 (run: env/.venv/bin/python -I experiments/019-source-diversity/selftest.py). Synthetic logic checks (split, subset
enumeration, size matching, bootstrap, labels, planted leaks) + design checks against the real E7 rows. Output also written to results/selftest.txt."""
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import run

OUT, ok_all = [], True


def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    OUT.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    print(OUT[-1])


def raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    return False


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. split_groups
def mk_groups(n_img, max_size, seed):
    r = np.random.default_rng(seed)
    out, g = [], 0
    while len(out) < n_img:
        s = int(r.integers(1, max_size + 1))
        out += [f"grp{g:04d}"] * min(s, n_img - len(out))
        g += 1
    return out


gr = mk_groups(600, 3, 1)
sp, cnt = run.split_groups(gr)
check("split: 600 images, groups <= 3 -> exactly 300/100/200", cnt == {"train": 300, "calib": 100, "test": 200}, str(cnt))
df = pd.DataFrame(dict(g=gr, s=sp))
check("split: every group lies in exactly ONE split", (df.groupby("g").s.nunique() == 1).all())
check("split: deterministic (same seed twice)", run.split_groups(gr)[0] == sp)
check("split: seed matters", run.split_groups(gr, seed=1)[0] != sp)
perm = np.random.default_rng(5).permutation(600)
sp_p, _ = run.split_groups([gr[i] for i in perm])
check("split: shuffling the image order does not change a group's split", dict(zip([gr[i] for i in perm], sp_p)) == dict(zip(gr, sp)))
gr2 = mk_groups(590, 3, 2)
sp2, cnt2 = run.split_groups(gr2)
check("split: 590 images -> train 300, calib 100, test gets the remainder (190), reported exactly", cnt2 == {"train": 300, "calib": 100, "test": 190}, str(cnt2))
gr3 = mk_groups(600, 12, 3)
sp3, cnt3 = run.split_groups(gr3)
check("split: large groups (up to 12 images): all images assigned, whole groups", sum(cnt3.values()) == 600 and
      (pd.DataFrame(dict(g=gr3, s=sp3)).groupby("g").s.nunique() == 1).all(), str(cnt3))

# ---- new_rows on synthetic manifests (patching run.HERE to a temp dir, never the real one)
import tempfile
with tempfile.TemporaryDirectory() as td:
    old_here = run.HERE
    run.HERE = Path(td)
    for j, S in enumerate(run.NEWS):
        g = mk_groups(603, 3, 10 + j)
        w = [1600] * 603
        h = [1000] * 603
        h[5] = 400                       # one too-small image: must be dropped and reported
        pd.DataFrame(dict(image_id=[f"i{j}_{k}" for k in range(603)], path=[f"data/raw/x/{j}_{k}.jpg" for k in range(603)],
                          sha256=[f"{j}{k:063x}" for k in range(603)], width=w, height=h, license="CC BY", group=g)).to_csv(Path(td) / run.SRC[S]["manifest"], index=False)
    nd, cnts = run.new_rows()
    run.HERE = old_here
check("new_rows: 603 manifest rows, 1 too small -> 602 kept, dropped count reported", all(c["n_manifest"] == 603 and c["n_short_lt_512_dropped"] == 1 and c["n_images"] == 602 for c in cnts.values()))
check("new_rows: 602 images (2 above the caps) -> each bucket within +3 of its cap (overflow groups go to the bucket with most room), roles by whole group, in_scoring_set only for test",
      all(sum(c["images"].values()) == 602 and all(0 <= c["images"][k] - cap <= 3 for k, cap in (("train", 300), ("calib", 100), ("test", 200))) for c in cnts.values()) and (nd.groupby(["pool", "group"]).split.nunique() == 1).all()
      and (nd.in_scoring_set == (nd.split == "test")).all() and set(nd.pool) == set(run.GROUP_POOLS))

# ---------------------------------------------------------------- 2. subset enumeration
combos = run.fit_combos()
check("fits: no duplicate (feat, T, variant)", len(combos) == len(set(combos)))
b8 = [c for c in combos if c[0] == "F_b8" and not c[2]]
check("fits: F_b8 = all 62 non-empty subsets of size 1..5 of the 6 sources", len(b8) == 62 and len({c[1] for c in b8}) == 62 and
      all(1 <= len(c[1]) <= 5 for c in b8))
for S in run.ORDER:
    others = [s for s in run.ORDER if s != S]
    subs = run.subsets_of(others)
    assert len(subs) == 31
    if not all(("F_b8", T, "") in set(combos) for T in subs):
        check(f"fits: all 31 subsets of the other 5 are fitted for held-out {S}", False)
check("fits: for each held-out S all 31 subsets of the other 5 are fitted (186 (S, T) pairs)", True)
last = [c for c in combos if c[0] == "F_last"]
last5 = [c for c in last if len(c[1]) == 5]
check("fits: F_last at k = 5 only for the main labels (6 sets)", len(last5) == 6)
check("fits: + 12 F_last sets at k = 4 for the drop-one-source E7-rule labels (config drop_source_flast)", len(last) == 18 and sum(len(c[1]) == 4 for c in last) == 12)
ctl = [c for c in combos if c[2]]
check("fits: size-matched control = 6 held-out x 5 seeds, F_b8, T = the 5 others", len(ctl) == 30 and all(c[0] == "F_b8" and len(c[1]) == 5 for c in ctl))
check("fits: 110 combos x 8 models = 880 probes", len(combos) == 110 and len(combos) * len(run.model_names()) == 880, f"({len(combos)})")
check("canon: source order fixed", run.canon(["HDRP", "VISION", "OI"]) == ("VISION", "OI", "HDRP"))

# ---------------------------------------------------------------- 3. size matching
counts = {"VISION": 99, "RAISE": 84, "OI": 300, "UNS": 300, "WMC": 300, "HDRP": 300}
tg = {S: run.ctl_target(counts, run.ORDER, S) for S in run.ORDER}
check("size-match target for OI held out = mean over the 10 pairs of the other 5 = 2 x 1083/5 = 433", tg["OI"] == 433, str(tg))
check("size-match target for VISION held out = 2 x 1284/5 = 514 (513.6 rounded)", tg["VISION"] == 514)
grp = np.array([f"g{i // 3}" for i in range(900)])
for sd in range(5):
    m, tot = run.subsample_groups(grp, 433, sd)
    ok = m.sum() == tot and tot <= 433 and tot >= 431 and all((m[grp == g].all() or not m[grp == g].any()) for g in np.unique(grp))
    check(f"size-match seed {sd}: whole groups, achieved {tot} <= 433, and within 2 of the target", ok)
m0, _ = run.subsample_groups(grp, 433, 0)
check("size-match: deterministic, seeds differ", np.array_equal(m0, run.subsample_groups(grp, 433, 0)[0]) and not np.array_equal(m0, run.subsample_groups(grp, 433, 1)[0]))

# ---------------------------------------------------------------- 4. synthetic data
POOLS = {s: run.SRC[s]["train_pools"][0] for s in run.ORDER}


class SynData(run.Data):
    def __init__(self, n_feat=False):
        self.cfg = run.ARM
        self.conds = run.CONDS
        r = np.random.default_rng(1)
        rows = []
        for S in run.ORDER:
            n = 60
            for k in range(n):
                role = ["train", "train", "calib", "test"][k % 4] if k % 4 != 1 or k < 40 else "test"
                pool = POOLS[S]
                dev = f"dev{k % 7}" if S == "VISION" else ""
                grp = f"{S}grp{k // 2}" if S in run.NEWS else ""
                rid = f"c{k % 20}" if S == "RAISE" else ""
                rows.append(dict(role=role, pool=pool, label=0, gen="real", device=dev, group=grp, rid=rid, src=S))
            if S == "VISION":              # laundered copies (WA / FBH) of test natives
                for pool in ("vision_whatsapp", "vision_facebook_hq"):
                    for k in range(60):
                        if k % 4 in (2, 3):
                            rows.append(dict(role=["calib", "test"][k % 4 - 2], pool=pool, label=0, gen="real", device=f"dev{k % 7}", group="", rid="", src=S))
        for g in run.ALLGENS:
            for k in range(30):
                role = "test" if (g == run.FLUX or k % 3 == 2) else "calib" if k % 3 == 1 and False else "train"
                rows.append(dict(role=role, pool="fake", label=1, gen=g, device="", group="", rid=f"c{k % 20}", src="F"))
        d = pd.DataFrame(rows)
        d["sha256"] = [f"{i:064x}" for i in range(len(d))]
        d["src_path"] = d.sha256
        d["path"] = "data/processed/fov512/images/" + d.sha256 + ".png"
        self.df = d
        self.role, self.pool, self.lab, self.gen = d.role.to_numpy(), d.pool.to_numpy(), d.label.to_numpy(), d.gen.to_numpy()
        self.rid = d.rid.to_numpy()
        self.groups = np.array([("grp:" + p + ":" + g) if g else ("dev:" + dv) if dv else "img:" + s for p, g, dv, s in zip(d.pool, d.group, d.device, d.sha256)])
        self.X = {}
        if n_feat:
            Xb = r.normal(size=(len(d), 4608)).astype(np.float32)
            off = {s: r.normal(size=4608).astype(np.float32) for s in run.ORDER}
            for s in run.ORDER:
                Xb[(d.src == s).to_numpy()] += 0.5 * off[s]
            Xb[self.lab == 1, 1536:3072] += 0.6
            self.X = {c: Xb + 0.1 * r.normal(size=Xb.shape).astype(np.float32) for c in self.conds}


S0 = SynData()
check("synthetic data: every source has train/calib/test reals", all(len(S0.idx(r, run.SRC[s]["eval_pools"] if r != "train" else run.SRC[s]["train_pools"], label=0)) > 0
                                                                      for s in run.ORDER for r in ("train", "calib", "test")))

# ---- bootstrap plan
pl = run.Plan(S0, n_boot=40)
check("plan: row 0 all ones", (pl.W[0] == 1).all())
cl = {}
for i in pl.rows:
    cl.setdefault(run.cluster_key(S0, i), set()).add(tuple(pl.W[:, pl.pos[int(i)]]))
check("plan: rows of one cluster (family / device / group) share weights", all(len(v) == 1 for v in cl.values()), f"({len(cl)} clusters)")
ks = [run.cluster_key(S0, i) for i in pl.rows]
for kind in ["fam", "dev", "img"] + [f"g{p}" for p in run.GROUP_POOLS]:
    m = np.array([k.startswith(kind + ":") for k in ks])
    if m.any():
        uk, first = np.unique(np.array(ks)[m], return_index=True)
        check(f"plan: kind {kind}: replicate draws sum to K = {len(uk)}", (pl.W[1:, np.where(m)[0][first]].sum(1) == len(uk)).all())
check("plan: new-source reals are clustered by their GROUP (kind g<pool>), RAISE reals and fakes by family",
      any(k.startswith("gunsplash:UNSgrp") for k in ks) and any(k.startswith("fam:c") for k in ks))
# paired with E7 on the real E7 rows
try:
    e = run.e7().Data("fov512", load_feats=False)
    Dm = run.Data(load_feats=False, splits=run.BASE_SPLITS)
    pe, pm = run.e7().Plan(e, 50), run.Plan(Dm, 50)
    check("plan on the E7 rows is IDENTICAL to E7's plan (paired differences with E7)", np.array_equal(pe.W, pm.W) and (e.role == Dm.role).all() and (e.groups == Dm.groups).all())
except Exception as ex:                                                  # pragma: no cover
    check("plan on the E7 rows identical to E7's", False, repr(ex))

# ---------------------------------------------------------------- 5. planted leaks
Sx = "OI"
T4 = run.canon(["VISION", "RAISE", "UNS", "WMC"])           # held-out OI
rows_ok = run.train_rows(S0, T4, "full")
run.audit_train_rows(S0, T4, "full", rows_ok, S=Sx)
check("clean fold passes all audits", True)
oi_train = S0.idx("train", ["openimages"], label=0)
check("LEAK caught: held-out source's train reals in the training rows", raises(lambda: run.audit_train_rows(S0, T4, "full", np.sort(np.r_[rows_ok, oi_train[:5]]), S=Sx)))
check("LEAK caught (no S given): reals from a pool outside the training sources", raises(lambda: run.audit_train_rows(S0, T4, "full", np.sort(np.r_[rows_ok, oi_train[:1]]))))
oi_calib = S0.idx("calib", ["openimages"], label=0)
check("LEAK caught: held-out source's test or calib reals in the training rows", raises(lambda: run.audit_train_rows(S0, T4, "full", np.sort(np.r_[rows_ok, S0.idx("test", ["openimages"], label=0)[:3]]))))
check("LEAK caught: held-out calib reals in the threshold rows", raises(lambda: run.audit_calib(S0, Sx, T4, np.r_[run.calib_rows(S0, T4), oi_calib[:3]])))
check("LEAK caught: train-role reals used as threshold rows", raises(lambda: run.audit_calib(S0, Sx, T4, np.r_[run.calib_rows(S0, T4), S0.idx("train", ["raise"], label=0)[:3]])))
run.audit_calib(S0, Sx, T4, run.calib_rows(S0, T4))
check("clean threshold rows (calib reals of the training sources) pass", True)
check("LEAK caught: evaluation rows that are not the held-out source's test reals", raises(lambda: run.audit_eval(S0, Sx, np.r_[run.src_eval_rows(S0, Sx), oi_calib[:2]])) and
      raises(lambda: run.audit_eval(S0, Sx, run.src_eval_rows(S0, "RAISE"))))
flux_tr = S0.idx(None, None, gens=[run.FLUX], label=1)[:3]
check("LEAK caught: FLUX fake in training", raises(lambda: run.audit_train_rows(S0, T4, "full", np.sort(np.r_[rows_ok, flux_tr]))))
logo_row = S0.idx("train", None, gens=["dalle2"], label=1)[:2]
check("LEAK caught: the LOGO generator in a g_<gen> fit", raises(lambda: run.audit_train_rows(S0, T4, "g_dalle2", np.sort(run.train_rows(S0, T4, "full")))))
check("LOGO fit excludes its generator", (S0.gen[run.train_rows(S0, T4, "g_dalle2")] != "dalle2").all())
h = run.fake_test_hash(S0)
run.audit_fake_rows(S0, h)
role0 = S0.role.copy()
fk = S0.idx("test", None, gens=["dalle3"], label=1)[0]
S0.role[fk] = "drop"
check("LEAK caught: fake test rows differ between folds", raises(lambda: run.audit_fake_rows(S0, h)))
S0.role[:] = role0
run.audit_taug(S0.lab[rows_ok], run.TAUG)
check("T-aug audit passes for the three configured conditions", True)
check("LEAK caught: a condition applied to one class only / a repeated or unknown condition", raises(lambda: run.audit_taug(S0.lab[rows_ok], ["clean", "clean"])) and
      raises(lambda: run.audit_taug(S0.lab[rows_ok], ["clean", "not_a_condition"])) and raises(lambda: run.audit_taug(np.zeros(10, int), run.TAUG)))
bad = S0.df.copy()
bad.loc[bad.index[0], "path"] = "data/processed/crop512/images/x.png"
check("LEAK caught: a row not prepared as fov512 (different transform)", raises(lambda: run.audit_same_prep(bad)))
run.audit_same_prep(S0.df)
check("same-prep audit passes on clean paths", True)

# ---------------------------------------------------------------- 6. labels on crafted quantities
def dsum(delta, lo, hi):
    return dict(delta=delta, ci_lo=lo, ci_hi=hi, p_le0=0.0)


def mkq(aur=0.93, fpr=0.03, div_auc=(0.08, 0.05, 0.11), div_fpr=(-0.1, -0.15, -0.05), lay=(0.05, 0.02, 0.08), lcgain=(0.1, 0.05, 0.15), ctl=(0.07, 0.04, 0.1), per=None):
    per = per or {s: dict(auroc=[aur, 0, 1], fpr=fpr) for s in run.ORDER}
    return dict(per=per, kmax=5, b8_minus_last=dsum(*lay) if lay else None, lc_diff_kmax_vs_k={k: dsum(*lcgain) for k in range(1, 5)},
                div_auc=dsum(*div_auc), div_fpr=dsum(*div_fpr), ctl_auc=dsum(*ctl) if ctl else None)


L = run.decide(mkq())
check("labels: all-good -> HELPS, NOT_QUANTITY, DISCOVERY, PASS, no kill", L["DIVERSITY_HELPS"] and L["DIVERSITY_NOT_QUANTITY"] and L["DISCOVERY"] and L["E7_RULE"] == "PASS" and not L["KILL_OF_LINE"])
L = run.decide(mkq(div_auc=(0.029, 0.01, 0.05)))
check("labels: AUROC gain 0.029 < 0.03 -> not HELPS, not DISCOVERY", not L["DIVERSITY_HELPS"] and not L["DISCOVERY"])
L = run.decide(mkq(div_auc=(0.05, -0.01, 0.1)))
check("labels: CI lo <= 0 -> not HELPS and KILL OF LINE", not L["DIVERSITY_HELPS"] and L["KILL_OF_LINE"])
L = run.decide(mkq(div_fpr=(-0.1, -0.15, 0.0)))
check("labels: FPR-difference CI hi = 0 (not < 0) -> not HELPS", not L["DIVERSITY_HELPS"] and L["c_auroc"] and not L["c_fpr"])
L = run.decide(mkq(ctl=(0.02, 0.0, 0.04)))
check("labels: size-matched run fails the AUROC part -> HELPS but NOT 'diversity, not quantity'", L["DIVERSITY_HELPS"] and L["DIVERSITY_NOT_QUANTITY"] is False)
L = run.decide(mkq(ctl=None))
check("labels: no control -> NOT_QUANTITY undetermined (None)", L["DIVERSITY_NOT_QUANTITY"] is None)
L = run.decide(mkq(aur=0.89))
check("labels: AUROC 0.89 everywhere -> HELPS but not DISCOVERY; E7 rule PASS (>= 0.85)", L["DIVERSITY_HELPS"] and not L["DISCOVERY"] and L["E7_RULE"] == "PASS")
L = run.decide(mkq(fpr=0.06))
check("labels: FPR 6% -> not DISCOVERY, E7 INCONCLUSIVE (not killed)", not L["DISCOVERY"] and L["E7_RULE"] == "INCONCLUSIVE" and not L["KILL_OF_LINE"])
per = {s: dict(auroc=[0.93, 0, 1], fpr=0.03) for s in run.ORDER}
per["RAISE"] = dict(auroc=[0.74, 0, 1], fpr=0.03)
L = run.decide(mkq(per=per))
check("labels: one source at AUROC 0.74 -> E7 KILL and KILL OF LINE (even if HELPS)", L["E7_RULE"] == "KILL" and L["KILL_OF_LINE"] and L["DIVERSITY_HELPS"])
per["RAISE"] = dict(auroc=[0.93, 0, 1], fpr=0.101)
L = run.decide(mkq(per=per))
check("labels: one source at FPR 10.1% -> E7 KILL and KILL OF LINE", L["E7_RULE"] == "KILL" and L["KILL_OF_LINE"])
per["RAISE"] = dict(auroc=[0.90, 0, 1], fpr=0.05)
check("labels: boundary AUROC 0.90 and FPR 0.05 satisfy DISCOVERY", run.decide(mkq(per=per))["DISCOVERY"])
L = run.decide(mkq(lay=(0.04, -0.01, 0.08)))
check("labels: b8-last CI includes 0 -> E7 INCONCLUSIVE (DIVERSITY HELPS unaffected)", L["E7_RULE"] == "INCONCLUSIVE" and L["DIVERSITY_HELPS"])
q = mkq()
q["lc_diff_kmax_vs_k"][3] = dsum(0.01, -0.005, 0.02)
L = run.decide(q)
check("labels: k=5 not clearly above k=3 -> E7 learning-curve gate fails (INCONCLUSIVE), vs-k=1-only reading still true",
      L["E7_RULE"] == "INCONCLUSIVE" and not L["e7_learning_curve_ok_all_k"] and L["e7_learning_curve_ok_vs_k1_only"])
check("labels: missing F_last -> E7 label flags it instead of guessing", run.decide(mkq(lay=None))["E7_RULE"] == "PASS?-needs-F_last")
check("diff_summary: percentile CI of replicates 1..B, point = replicate 0", run.diff_summary(np.r_[0.5, np.linspace(0, 1, 101)[1:]])["delta"] == 0.5)

# ---------------------------------------------------------------- 7. synthetic end-to-end (signal planted in k; scenario flags)
def make_loader(D, scenario):
    pool_src = {p: s for s in run.ORDER for p in run.SRC[s]["train_pools"] + run.SRC[s]["eval_pools"]}
    real_src = np.array([pool_src.get(p, "F") for p in D.pool])
    n = len(D.df)

    def loader(feat, T, var=""):
        if feat == "F_last" and not (len(T) in (4, 5) and not var):
            return None
        k = 5 if var and scenario != "quantity" else 2 if var else len(T)
        key = f"{feat}{T}{var}"
        r = np.random.default_rng(zlib.crc32(key.encode()))
        if scenario == "flat":
            sig, shift = 1.5, 0.0
        else:
            sig, shift = 0.6 + 0.45 * k, 1.6 * (5 - k) / 4
        if feat == "F_last":
            sig = 0.3
        sc = {}
        for m in run.model_names():
            base = r.normal(size=n) + D.lab * sig
            base += np.where((D.lab == 0) & ~np.isin(real_src, T), shift, 0.0)
            sc[m] = np.stack([base + 0.05 * r.normal(size=n) for _ in run.CONDS]).astype(np.float32)
        extra = dict(ctl_target=100, n_real=100) if var else {}
        return sc, np.ones(len(sc)), np.full(len(sc), 100), extra
    return loader


res = {}
for scen in ("improving", "flat", "quantity"):
    out, GA, FP, lr = run.evaluate(S0, make_loader(S0, scen), n_boot=60)
    rob = run.robustness(GA, FP, run.ORDER)
    res[scen] = (out, rob)
Li = res["improving"][1]["gens"]["A|all"]["labels"]
Lf = res["flat"][1]["gens"]["A|all"]["labels"]
Lq = res["quantity"][1]["gens"]["A|all"]["labels"]
check("end-to-end (planted diversity gain): DIVERSITY HELPS, NOT QUANTITY, DISCOVERY, PASS", Li["DIVERSITY_HELPS"] and Li["DIVERSITY_NOT_QUANTITY"] and Li["DISCOVERY"] and Li["E7_RULE"] == "PASS", str({k: v for k, v in Li.items() if k in ("DIVERSITY_HELPS", "DISCOVERY", "E7_RULE")}))
check("end-to-end (no gain with k): not HELPS, KILL OF LINE", (not Lf["DIVERSITY_HELPS"]) and Lf["KILL_OF_LINE"], str(res['flat'][1]['gens']['A|all']['q']['div_auc']))
check("end-to-end (gain from QUANTITY only: control behaves like k=2): HELPS true but NOT_QUANTITY false", Lq["DIVERSITY_HELPS"] and Lq["DIVERSITY_NOT_QUANTITY"] is False)
qi = res["improving"][1]["gens"]["A|all"]["q"]
check("end-to-end: learning curve increases with k", all(qi["lc_mean"][k + 1][0] > qi["lc_mean"][k][0] for k in range(1, 5)), str([round(qi['lc_mean'][k][0], 3) for k in range(1, 6)]))
check("end-to-end: drop-one-generator robustness has all 10 subsets x 2 readings; drop-one-source 3 pools x 2 readings",
      len(res["improving"][1]["gens"]) == 20 and len(res["improving"][1]["sources"]) == 6)
dsrc = res["improving"][1]["sources"]["A|drop UNS"]
check("end-to-end: drop-one-source recomputes the labels on a 5-source pool (k_max = 4)", dsrc["q"]["kmax"] == 4 and set(dsrc["q"]["per"]) == set(run.ORDER) - {"UNS"} and dsrc["labels"]["DIVERSITY_HELPS"])
check("end-to-end: audit counter covers every fold (6 x (31 + 5 control))", res["improving"][0]["audit"]["n_folds_audited"] == 6 * 36)
ks = run.key_secondary(GA, FP, res["improving"][0], "A")
check("key secondary: VISION/RAISE/OI, k=5 vs the 2-source E7 subset, all 5 conditions, same rows", set(ks) == {"VISION", "RAISE", "OI"} and all(c in ks["OI"] for c in run.CONDS) and ks["OI"][run.JR]["d_auroc"]["delta"] > 0)

# ---- finalize(): results.json / tables.md written to a temp results dir
with tempfile.TemporaryDirectory() as td:
    old_res = run.RES
    run.RES = Path(td)
    out_s, GA_s, FP_s, lr_s = run.evaluate(S0, make_loader(S0, "improving"), n_boot=40)
    run.finalize(S0, out_s, GA_s, FP_s, lr_s)
    run.RES = old_res
    import json
    rj = json.loads((Path(td) / "results.json").read_text())
    tm = (Path(td) / "tables.md").read_text()
    lg = pd.read_csv(Path(td) / "scores_long.csv.gz")
check("finalize: results.json has labels for both readings, robustness, key secondary, parity, ctl targets, owner status",
      all(k in rj["result"] for k in ("main", "robust", "key_secondary", "parity_e7", "ctl_targets", "owner", "audit", "real_split_counts")) and rj["result"]["main"]["A"]["labels"]["DIVERSITY_HELPS"])
check("finalize: tables.md has every section", all(h in tm for h in ("## Labels", "## Held-out source at k = 5", "## Learning curve", "## Size-matched control", "## Key secondary",
                                                                        "## Robustness: drop-one-generator", "## Robustness: drop-one-source", "## Leak audit")), f"({len(tm.splitlines())} lines)")
check("finalize: scores_long.csv.gz written for the k=5 fits (6 held-out x 2 conditions)", set(lg.heldout) == set(run.ORDER) and set(lg.cond) == {run.JR, "clean"})

# ---- planted leaks inside the evaluation loop (must be caught by the asserts)
def leaky_train_rows(D, T, model, real_rows=None):
    pools = [p for s in run.ORDER for p in run.SRC[s]["train_pools"]]          # every source's train reals, incl. the held-out one
    return np.sort(np.concatenate([D.idx("train", pools, label=0), D.idx("train", None, gens=[g for g in run.TRAINGENS if model != f"g_{g}"], label=1)]))


orig_tr, orig_cal = run.train_rows, run.calib_rows
run.train_rows = leaky_train_rows
check("PLANTED LEAK (held-out train reals in training) is caught by evaluate()", raises(lambda: run.evaluate(S0, make_loader(S0, "improving"), n_boot=10)))
run.train_rows = orig_tr
run.calib_rows = lambda D, srcs: D.idx("calib", None, label=0)                # thresholds from ALL calib reals, incl. the held-out source
check("PLANTED LEAK (held-out calib reals in the thresholds) is caught by evaluate()", raises(lambda: run.evaluate(S0, make_loader(S0, "improving"), n_boot=10)))
run.calib_rows = orig_cal
calls = {"n": 0}
base_loader = make_loader(S0, "improving")


def mutating_loader(feat, T, var=""):
    calls["n"] += 1
    if calls["n"] == 40:
        S0.role[S0.idx("test", None, gens=["dalle2"], label=1)[0]] = "drop"        # a fake test row disappears from later folds
    return base_loader(feat, T, var)


check("PLANTED LEAK (fake test rows differ between folds) is caught by evaluate()", raises(lambda: run.evaluate(S0, mutating_loader, n_boot=10)))
S0.role[:] = role0
out, *_ = run.evaluate(S0, make_loader(S0, "improving"), n_boot=10)
check("after removing the plants, evaluate() runs clean", out["audit"]["n_folds_audited"] == 6 * 36)

# ---------------------------------------------------------------- 8. a real fit on synthetic features (plumbing, ctl variant)
SX = SynData(n_feat=True)
T5 = run.canon([s for s in run.ORDER if s != "OI"])
sc, C, ntr, ex = run.fit_scores(SX, "F_b8", T5, "full")
oi = SX.idx("test", ["openimages"], label=0)
fk = SX.idx("test", None, label=1)
from sklearn.metrics import roc_auc_score
au = roc_auc_score(np.r_[np.zeros(len(oi)), np.ones(len(fk))], np.r_[sc[1][oi], sc[1][fk]])
check("fit_scores on synthetic features: shape [conds, N] and planted block-8 signal detected on the held-out source", sc.shape == (len(run.CONDS), len(SX.df)) and au > 0.8, f"(AUROC {au:.3f})")
scl, *_ = run.fit_scores(SX, "F_last", T5, "full")
aul = roc_auc_score(np.r_[np.zeros(len(oi)), np.ones(len(fk))], np.r_[scl[1][oi], scl[1][fk]])
check("fit_scores: block-12 columns carry no planted signal (near chance)", aul < 0.65, f"(AUROC {aul:.3f})")
scc, Cc, ntc, exc = run.fit_scores(SX, "F_b8", T5, "g_dalle2", "ctl0")
tgt = run.ctl_target(run.real_train_counts(SX), run.ORDER, "OI")
check("ctl fit: training reals subsampled by whole groups to <= the k=2 mean target, fakes unchanged",
      exc["n_real"] <= tgt and exc["n_real"] >= 0.9 * tgt and ntc == exc["n_real"] + len(SX.idx("train", None, gens=[g for g in run.TRAINGENS if g != "dalle2"], label=1))
      and len(set(SX.groups[exc["real_rows"]])) < len(set(SX.groups[SX.idx("train", [p for s in T5 for p in run.SRC[s]["train_pools"]], label=0)])), f"(n_real {exc['n_real']}, target {tgt})")
r0, _ = run.ctl_real_rows(SX, T5, 0)
r1, _ = run.ctl_real_rows(SX, T5, 1)
check("ctl rows: deterministic per seed, different across seeds, all train-role reals of T", np.array_equal(r0, run.ctl_real_rows(SX, T5, 0)[0]) and not np.array_equal(r0, r1)
      and (SX.role[r0] == "train").all() and set(SX.pool[r0]) <= {p for s in T5 for p in run.SRC[s]["train_pools"]})

# ---------------------------------------------------------------- 9. real data (E7 rows always; new sources once prepared)
Dr = run.Data(load_feats=False, splits=run.BASE_SPLITS)
ro = pd.crosstab([Dr.pool, Dr.lab], Dr.role)
check("E7 rows: E6 role counts reproduced (train fakes 230, RAISE reals 84/53/183, OI 300/100/200)",
      ro.loc[("fake", 1), "train"] == 230 and tuple(ro.loc[("raise", 0), ["train", "calib", "test"]]) == (84, 53, 183) and tuple(ro.loc[("openimages", 0), ["train", "calib", "test"]]) == (300, 100, 200))
tr = set(Dr.rid[(Dr.role == "train") & (Dr.rid != "")])
ev = set(Dr.rid[np.isin(Dr.role, ["calib", "test"]) & (Dr.rid != "")])
check("E7 rows: no family id shared by train and calib/test", not tr & ev)
sp = run.owner_status()
check("OWNER source: pending unless >= 150 photos exist", sp["status"] in ("pending", "PRESENT-NOT-USED"), str(sp))
if run.FINAL_SPLITS.exists():
    Df = run.Data(load_feats=False)
    for S in run.NEWS:
        pool = run.SRC[S]["train_pools"][0]
        d = Df.df[Df.df.pool == pool]
        cnt = d.split.value_counts().to_dict()
        check(f"{S}: group-disjoint splits, counts {cnt}", (d.groupby("group").split.nunique() == 1).all() and sum(cnt.values()) > 0 and
              (d.groupby("split").size().reindex(["train", "calib", "test"]).fillna(0).to_numpy() <= np.array([300, 100, 200]) + 12).all())
    check("final splits: sha256 unique, E7 rows unchanged (first 3519 rows equal 016's)",
          Df.df.sha256.is_unique and (Df.df.sha256.iloc[:3519].to_numpy() == Dr.df.sha256.to_numpy()).all())
else:
    OUT.append("SKIP real new-source checks: splits_fov512.csv not built yet (prep stage pending)")
    print(OUT[-1])

OUT.append("ALL PASS" if ok_all else "SOME FAILED")
print(OUT[-1])
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + "\n")
sys.exit(0 if ok_all else 1)
