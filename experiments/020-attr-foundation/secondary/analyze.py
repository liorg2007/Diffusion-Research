"""020 secondary (pre-declared DESCRIPTIVE) arms, docs/09 decision 5. No labels are assigned; nothing here changes the frozen 020 verdicts.

  env/.venv/bin/python -I experiments/020-attr-foundation/secondary/analyze.py [--workers 8]
Reuses the frozen 020 code by import (run.py is loaded by path, never modified): closed_task / inner_table / make_folds / cluster_weights /
eval_closed / auroc_boot / pooled_logp / nested_logp. The 020 results directory is NOT written (run.RES is redirected for build_analyses).
(a) Q2-near-twin   SD1.3 vs SD1.4, caption-grouped 5-fold CV, T-mix training, feature sets b8 and cat, eval clean + jpeg75_resize05
(b) Q3-family      frozen 8-way probe (retrained on all 1020 fakes, 020 protocol) applied to SD1.3 / SD1.4 / GLIDE; + 11-way closed-set CV
(c) RAID replication n=500 ids (100 old + 400 new): 4-way, jpeg(75); also without gen_0
"""
import argparse
import importlib.util
import json
import pickle
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES, CACHE = HERE / "results", HERE / "cache"
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = load_mod("e11run", ROOT / "experiments" / "020-attr-foundation" / "run.py")
R.RES = RES / "_tmp_folds"                                    # build_analyses writes fold files here, not into the frozen 020 results
R.RES.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import p4features as P

SEED, B, CFG = R.SEED, R.B, R.CFG
CONDS512 = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]
CONDS224 = ["clean", "jpeg(75)", "jpeg(90)"]
TMIX512 = ["clean", "jpeg75_resize05", "social_sim"]
log = lambda *a: print(time.strftime("%H:%M:%S"), "SEC-A", *a, flush=True)


def stage_cache(name, fn):
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = fn()
    p.write_bytes(pickle.dumps(out))
    return out


def load_new(uni):
    df = pd.read_csv(CACHE / "new_table.csv")
    df = df[df.universe == uni].reset_index(drop=True)
    conds = CONDS512 if uni == "fov512" else CONDS224
    XF = {}
    for c in conds:
        k, x = P.read_shards(c, CACHE / uni)
        pos = {s: i for i, s in enumerate(k.tolist())}
        XF[c] = np.stack([x[pos[s]] for s in df.sha256]).astype(np.float32)
        assert XF[c].shape[1] == 4608 and np.isfinite(XF[c]).all()
    return df, XF


def mk_analysis(name, y, classes, groups, conds, tmix, primary, XF, drop=None):
    cfg = dict(pairwise=True)
    fold = R.make_folds(y, groups, CFG["outer_folds"], SEED)
    chk = R.check_grouped_folds(fold, groups)
    assert chk["ok"], chk
    A = R.Analysis(name, cfg, y, classes, groups, fold, conds, tmix, primary, XF, fam=groups)
    R._G[name] = A
    return A


def run_closed(A, workers, regimes=("tmix", "clean"), fss=None):
    fss = fss or R.FS_ORDER
    tasks = [(A.name, fs, rg, f) for fs in fss for rg in regimes for f in range(CFG["outer_folds"])]
    return R.run_tasks(R.closed_task, tasks, workers, label=A.name)


def sub_ci(arr):
    return R.ci(arr)


def cluster_plan(groups):
    return R.cluster_weights(groups, B, SEED)


def twin_metrics(A, res, fs, regime, Wc, inv):
    lp, ch = R.pooled_logp(res, A, A.name, fs, regime)
    summ, _ = R.eval_closed(A, lp, Wc, inv, None)
    Wrow = Wc[:, inv]
    out = {}
    pos, neg = A.y == 1, A.y == 0
    for c in A.conds:
        s = lp[c][:, 1] - lp[c][:, 0]
        au = R.auroc_boot(s[pos], s[neg], Wrow[:, pos], Wrow[:, neg])
        d = summ[c]
        out[c] = dict(bacc=d["bacc"], auroc=R.ci(au), pw_2afc=d["pw"]["pooled"], pw_strict=d["pw_strict"]["pooled"],
                      recall=d["recall"], confusion=d["confusion"], n=int(A.n))
    return out, {k: [v[0], v[1]] for k, v in ch.items()}, lp


def twin_perm(A, n_perm, workers):
    """Label-swap-within-family null (b8, C=0.01, T-mix, primary cond): chance of the pipeline with caption-matched structure."""
    fams = {}
    for i, g in enumerate(A.groups):
        fams.setdefault(g, []).append(i)
    X = {c: A.getX("b8", c) for c in A.conds}
    tasks = [(s,) for s in range(n_perm)]

    def one(t):
        rng = np.random.default_rng(1000 + t[0])
        y = A.y.copy()
        for rows in fams.values():
            if rng.random() < 0.5 and len(rows) == 2:
                y[rows[0]], y[rows[1]] = y[rows[1]], y[rows[0]]
        out = {}
        for c in ("clean", "jpeg75_resize05"):
            out[c] = []
        pr = {c: np.zeros(A.n, int) for c in out}
        for f in range(CFG["outer_folds"]):
            tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
            Xs = np.concatenate([X[c][tr] for c in A.tmix])
            m = R.fit_model("lr", Xs, np.tile(y[tr], len(A.tmix)), 0.01)
            for c in out:
                pr[c][te] = R.logp_full(m, X[c][te], 2).argmax(1)
        return {c: R.bacc_simple(y, pr[c], [0, 1]) for c in out}

    global _one
    _one = one
    return R.run_tasks(_run_one, tasks, workers, label="twin_perm")


def _run_one(t):
    return _one(t)


def frac_ci(pred, groups, K):
    Wc, inv, ug = cluster_plan(groups)
    oh = np.zeros((len(pred), K))
    oh[np.arange(len(pred)), pred] = 1
    W = Wc[:, inv].astype(np.float64)
    num = W @ oh
    den = W.sum(1, keepdims=True)
    fr = num / den
    return [R.ci(fr[:, k]) for k in range(K)]


def fit_final(A, fs, regime="tmix"):
    """020 protocol on ALL rows: inner grouped 3-fold CV (train rows only) picks C per feature set at the primary condition."""
    ctrain, sel = R.regime_conds(A, regime)
    tr = np.arange(A.n)
    table, Cb = R.inner_table(A, fs, tr, ctrain, sel, SEED * 1000 + 99, list(range(A.K)))
    m = R.fit_model("lr", *R.stack(A, fs, tr, ctrain), Cb)
    return m, Cb, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    W = a.workers
    out = dict(config=dict(seed=SEED, n_boot=B, outer_folds=CFG["outer_folds"], inner_folds=CFG["inner_folds"], c_grid=CFG["c_grid"]["dino"],
                           note="DESCRIPTIVE arms (docs/09 decision 5); no label; 020 verdicts untouched"))
    RES.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    A0, _ = R.build_analyses()
    main8, raid100 = A0["ha1_main"], A0["ha1_raid"]
    # cross-check: folds re-derived here equal the committed 020 folds
    fc = json.loads((ROOT / "experiments/020-attr-foundation/results/folds_commit.json").read_text())
    import hashlib
    out["fold_check"] = {an: bool(hashlib.sha256(A0[an].fold.tobytes()).hexdigest() == fc[an]["sha256_of_folds"]) for an in ("ha1_main", "ha1_raid")}
    assert all(out["fold_check"].values()), out["fold_check"]
    # ---------------------------------------------------------------- new SB images
    nd, nX = load_new("fov512")
    nd["fam"] = "rid:" + nd.rid
    GEN_SB = ["stable-diffusion-1-3", "stable-diffusion-1-4", "glide"]
    old_fam = set(main8.groups)
    out["data"] = dict(n_old_fakes=int(main8.n), old_classes=main8.classes, n_new={g: int((nd.generator == g).sum()) for g in GEN_SB},
                       n_new_families=int(nd.fam.nunique()), n_new_families_shared_with_old_fakes=int(len(set(nd.fam) & old_fam)),
                       glide_deviation="GLIDE raw 256 px, upscaled 2x BICUBIC to 512 before the fov512 pipeline (fov512 never upscales); SD1.3/1.4 are native 512")
    # raw-size census of the old classes (header read only): context for the 'SD1.x -> sd2' reading
    sizes = {}
    for g in main8.classes:
        rows = main8.df[main8.df.generator == g]
        sizes[g] = pd.Series([("x".join(map(str, __import__("PIL.Image", fromlist=["x"]).open(ROOT / p).size))) for p in rows.src_path]).value_counts().head(4).to_dict()
    for g in GEN_SB:
        sizes[g] = {"512x512" if g != "glide" else "256x256": int((nd.generator == g).sum())}
    out["data"]["raw_size_top4"] = sizes

    # ================================================================ (a) near-twin
    log("(a) near-twin")
    m = nd.generator.isin(GEN_SB[:2]).to_numpy()
    tw = nd[m].reset_index(drop=True)
    ytw = (tw.generator == "stable-diffusion-1-4").astype(int).to_numpy()
    A_tw = mk_analysis("twin", ytw, ["sd1.3", "sd1.4"], tw.fam.to_numpy(), CONDS512, TMIX512, "jpeg75_resize05", {c: nX[c][m] for c in CONDS512})
    res_tw = stage_cache("res_twin", lambda: run_closed(A_tw, W))
    Wc, inv, ug = cluster_plan(A_tw.groups)
    out["a_twin"] = dict(n=int(A_tw.n), n_families=int(len(ug)), chance=0.5, label="SD1.3 vs SD1.4, 5-fold StratifiedGroupKFold by caption family", metrics={})
    for rg in ("tmix", "clean"):
        for fs in ("b8", "cat"):
            mt, ch, _ = twin_metrics(A_tw, res_tw, fs, rg, Wc, inv)
            out["a_twin"]["metrics"][f"{rg}/{fs}"] = dict(by_cond=mt, chosen_C_inner_bacc_per_fold=ch)
    perm = stage_cache("perm_twin", lambda: twin_perm(A_tw, 60, W))
    out["a_twin"]["perm_null_b8_tmix"] = {c: dict(mean=float(np.mean([p[c] for p in perm])), p95=float(np.percentile([p[c] for p in perm], 95)),
                                                   max=float(np.max([p[c] for p in perm])), n=len(perm)) for c in ("clean", "jpeg75_resize05")}

    # ================================================================ (b) Q3-family: frozen 8-way probe applied to new images
    log("(b) Q3-family")
    fin = stage_cache("final8", lambda: {fs: fit_final(main8, fs) for fs in R.FS_ORDER})
    best_fs = max(R.FS_ORDER, key=lambda fs: (round(fin[fs][2][fin[fs][1]], 12), -R.FS_ORDER.index(fs)))
    out["b_family"] = dict(classes=main8.classes, nested_choice=dict(fs=best_fs, C=fin[best_fs][1], inner_bacc=fin[best_fs][2][fin[best_fs][1]]),
                           per_fs={fs: dict(C=fin[fs][1], inner_table={str(k): v for k, v in fin[fs][2].items()}) for fs in R.FS_ORDER},
                           frozen_protocol="T-mix (clean, jpeg75_resize05, social_sim), standardised LR class_weight=balanced, C by inner grouped 3-fold CV at jpeg75_resize05, feature set by inner bacc (ties -> b4,b8,b12,cat)",
                           predicted={})
    for g in GEN_SB:
        mm = (nd.generator == g).to_numpy()
        grp = nd.fam.to_numpy()[mm]
        for fs in R.FS_ORDER:
            model = fin[fs][0]
            for c in CONDS512:
                lp = R.logp_full(model, nX[c][mm][:, R.SL[fs]], main8.K)
                pred = lp.argmax(1)
                fr = frac_ci(pred, grp, main8.K)
                out["b_family"]["predicted"].setdefault(g, {}).setdefault(fs, {})[c] = dict(
                    counts={main8.classes[k]: int((pred == k).sum()) for k in range(main8.K)},
                    frac={main8.classes[k]: fr[k] for k in range(main8.K)},
                    mean_prob={main8.classes[k]: float(np.exp(lp)[:, k].mean()) for k in range(main8.K)}, n=int(mm.sum()))
    # 11-way closed set
    log("(b) 11-way")
    cls11 = main8.classes + GEN_SB
    y11 = np.concatenate([main8.y, 8 + np.array([GEN_SB.index(g) for g in nd.generator])])
    g11 = np.concatenate([main8.groups, nd.fam.to_numpy()])
    XF11 = {c: np.concatenate([main8.XF[c], nX[c]]) for c in CONDS512}
    A11 = mk_analysis("closed11", y11, cls11, g11, CONDS512, TMIX512, "jpeg75_resize05", XF11)
    res11 = stage_cache("res_11", lambda: run_closed(A11, W, regimes=("tmix",)))
    Wc11, inv11, _ = cluster_plan(g11)
    lp11, choice = R.nested_logp(res11, A11, "closed11", "tmix")
    s11, _ = R.eval_closed(A11, lp11, Wc11, inv11, None)
    out["b_family"]["closed11"] = dict(n=int(A11.n), chance=1 / 11, nested_choice_per_fold=choice, chance_pair=0.5, by_cond={
        c: dict(bacc=s11[c]["bacc"], mf1=s11[c]["mf1"], recall=s11[c]["recall"], confusion=s11[c]["confusion"], pair_matrix=s11[c]["pair_matrix"])
        for c in CONDS512}, classes=cls11)

    # ================================================================ (c) RAID n=500
    log("(c) RAID")
    rd, rX = load_new("fov224")
    rd["fam"] = "d3:" + rd.rid.astype(str)
    old_ids = set(raid100.groups)
    assert not (set(rd.fam) & old_ids), "new RAID ids overlap the old 100"
    cls4 = raid100.classes
    ynew = np.array([cls4.index(g) for g in rd.generator])
    y5 = np.concatenate([raid100.y, ynew])
    g5 = np.concatenate([raid100.groups, rd.fam.to_numpy()])
    XF5 = {c: np.concatenate([raid100.XF[c], rX[c]]) for c in CONDS224}
    is_new = np.concatenate([np.zeros(raid100.n, bool), np.ones(len(rd), bool)])
    A5 = mk_analysis("raid500", y5, cls4, g5, CONDS224, CONDS224, "jpeg(75)", XF5)
    res5 = stage_cache("res_raid500", lambda: run_closed(A5, W))
    Wc5, inv5, ug5 = cluster_plan(g5)
    out["c_raid"] = dict(n=int(A5.n), n_ids=int(len(ug5)), n_ids_old=100, n_ids_new=int(rd.rid.nunique()), chance_4way=0.25, chance_3way=1 / 3, primary="jpeg(75)",
                         frozen_ref_n100=dict(source="experiments/020-attr-foundation/results/results.json (H-A1 RAID, T-mix nested)"), by_regime={})
    for rg in ("tmix", "clean"):
        lpn, choice = R.nested_logp(res5, A5, "raid500", rg)
        s5, _ = R.eval_closed(A5, lpn, Wc5, inv5, "d3_gen_0")
        ent = dict(nested_choice_per_fold=choice, by_cond={})
        Wrow = Wc5[:, inv5]
        for c in CONDS224:
            pred = lpn[c].argmax(1)
            d = s5[c]
            # restricted 4-way model, gen_1..3 only (critic c3 'restricted'): macro recall over gen_1..3 with the 4-way model
            sub = {}
            for nm, msk in (("old100", ~is_new), ("new400", is_new)):
                Cb_ = R.boot_conf(Wrow * msk[None, :], A5.y, pred, A5.K)
                sub[nm] = R.ci(R.conf_stats(Cb_)["bacc"])
            ent["by_cond"][c] = dict(bacc=d["bacc"], bacc_without_gen0_macro=d["bacc_drop"], mf1=d["mf1"], recall=d["recall"], confusion=d["confusion"],
                                     pw_2afc_macro=d["pw"]["macro"], pw_2afc_without_gen0=d["pw_drop"], pw_strict_macro=d["pw_strict"]["macro"], bacc_subsets=sub)
        out["c_raid"]["by_regime"][rg] = ent
    # fresh 3-way gen_1..3 (no gen_0 rows at all)
    keep = y5 != 0
    A3 = mk_analysis("raid500_g123", y5[keep] - 1, cls4[1:], g5[keep], CONDS224, CONDS224, "jpeg(75)", {c: XF5[c][keep] for c in CONDS224})
    res3 = stage_cache("res_raid500_g123", lambda: run_closed(A3, W))
    Wc3, inv3, _ = cluster_plan(g5[keep])
    out["c_raid"]["fresh_3way_gen123"] = {}
    for rg in ("tmix", "clean"):
        lp3, choice = R.nested_logp(res3, A3, "raid500_g123", rg)
        s3, _ = R.eval_closed(A3, lp3, Wc3, inv3, None)
        out["c_raid"]["fresh_3way_gen123"][rg] = dict(nested_choice_per_fold=choice, n=int(A3.n), by_cond={c: dict(bacc=s3[c]["bacc"], recall=s3[c]["recall"], pw_2afc_macro=s3[c]["pw"]["macro"]) for c in CONDS224})
    # size census of RAID new classes (the critic's gen_0 size cue)
    out["c_raid"]["new_raw_sizes"] = {g: rd[rd.generator == g].apply(lambda r: f"{r.w}x{r.h}", axis=1).value_counts().head(3).to_dict() for g in sorted(set(rd.generator))}
    (RES / "secondary.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log("wrote results/secondary.json")


if __name__ == "__main__":
    main()
