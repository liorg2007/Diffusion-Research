"""H4 OFFICIAL run (docs/04-phase4-plan.md: H4 section + Amendments 3, 5, 6). Code only: the data it needs (RAISE PNG
reals, SD3.5/FLUX fakes) was still on the unreachable grip.unina.it when this was written.

  # official (all rows of 003 splits.csv; aborts if a crop, a feature cache or an arm is missing)
  env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py
  # code-path check on the pilot data (no RAISE: those settings are skipped), tiny subset, n_boot 50 -> cache/official_dryrun/
  env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py --dry-run
  # same + a 20-image stand-in for RAISE (make_standin.py first) so (ii)/(M)/(iii)/(iv) and the gate run -> NOT RESULTS
  env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py --dry-run --standin-raise

The pilot module run.py (frozen behaviour, still `--subdir pilot`) is imported for the RIGID score, the noise generator and
the D-008 guard; its code is not changed. Parameters: config_official.json (+ config.json for sigma/draws/batch).
Outputs: <out>/{results.json, tables.md, scores_long.csv.gz, d008_guard.json, run.log}; score caches in cache/<...>/.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

import h4
import metrics
import probe
import p4features as P
from features import get_device, load_backbone

OC = json.loads((HERE / "config_official.json").read_text())


def _load_pilot_module():
    spec = importlib.util.spec_from_file_location("h4pilot_run", HERE / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load_pilot_module()                    # RIGID score, noise, D-008 guard (unchanged pilot code)
SIGMAS, DRAWS, FVS, PRIMARY_FV = H.SIGMAS, H.DRAWS, H.FVS, H.PRIMARY_FV
SEED = OC["aggregation"]["seed"]
ARMS_ALL = OC["arms"]
ARMC = OC["arm_conditions"]
SET = OC["settings"]
MAIN = OC["generators"]["main"]
TEST_ONLY = OC["generators"]["test_only"]
T0 = time.time()
LOGF = None


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n"); LOGF.flush()


# ----------------------------------------------------------------------------- data
def load_df(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"vidx": str}).astype({"in_scoring_set": bool, "b0_train": bool})


def vec(lab, a_real, a_fake):
    """Per-row array of a setting: reals take their value under the setting's real condition, fakes under its fake condition."""
    return np.where(lab == 0, a_real, a_fake)


def load_aligned_multi(df, cond, subdirs):
    """004 primary features aligned to df rows from one or more cache subdirs (a later subdir fills rows the earlier lacks)."""
    out = None
    for sd in subdirs:
        try:
            X = P.load_features_aligned(df, "primary", cond, subdir=sd)
        except FileNotFoundError:
            continue
        if out is None:
            out = X
        else:
            miss = np.isnan(out).any(1)
            out[miss] = X[miss]
    if out is None:
        raise FileNotFoundError(f"no cached features for {cond!r} in {subdirs}")
    return out


# ----------------------------------------------------------------------------- scoring (RIGID + Laplacian in one image load)
def score_condition(model, dev, p4, tp, paths, shas, cond):
    """-> (scores [n, 2 feature variants, len(SIGMAS)], laplacian [n]); same RIGID computation as the pilot's score_condition."""
    res = np.zeros((len(paths), 2, len(SIGMAS)))
    lap = np.zeros(len(paths))
    pairs = [(si, d) for si in range(len(SIGMAS)) for d in range(DRAWS)]
    for s in range(0, len(paths), H.BATCH):
        ps, sh = paths[s:s + H.BATCH], shas[s:s + H.BATCH]
        x = torch.stack(list(tp.map(lambda a: H.to_unit_tensor(p4.load_cond(a[0], a[1], cond)), zip(ps, sh))))
        lap[s:s + len(ps)] = [h4.laplacian_texture(t.permute(1, 2, 0).numpy()) for t in x]
        f0, fn = H.chunk_feats(model, dev, x, sh, pairs)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        for (si, d), f in fn.items():
            res[s:s + len(ps), :, si] += H.cos_score(f0, f) / DRAWS
        if (s // H.BATCH) % 5 == 0:
            log(f"    {cond}: {min(s + H.BATCH, len(paths))}/{len(paths)}")
    return res, lap


def lap_only(p4, tp, paths, shas, cond):
    return np.array(list(tp.map(lambda a: h4.laplacian_texture(H.to_unit_tensor(p4.load_cond(a[0], a[1], cond)).permute(1, 2, 0).numpy()),
                                zip(paths, shas))))


def plan_rows(df, active, raise_pool):
    """Rows scored per condition (RIGID + Laplacian): test+calib of the two primary real pools and the test fakes;
    `clean` also gets the descriptive WA/FBH test copies. Laplacian only (train fakes, clean): the texture direction."""
    pool, sp, lab, gen = df.pool.to_numpy(), df.split.to_numpy(), df.label.to_numpy(), df.generator.to_numpy()
    nat = OC["pools"]["native"]
    core = active & np.isin(sp, ["test", "calib"]) & (np.isin(pool, [nat, raise_pool]) | (lab == 1))
    derived = active & (sp == "test") & np.isin(pool, OC["pools"]["derived"])
    lap_rows = active & (sp == "train") & (lab == 1) & np.isin(gen, MAIN)
    return core, derived, lap_rows


# ----------------------------------------------------------------------------- B0 (005 A0 recipe, cached features only)
def decide(m, Z):
    out = np.full(len(Z), np.nan)
    ok = np.isfinite(Z).all(1)
    if ok.any():
        out[ok] = m.decision_function(Z[ok])
    return out


def fit_b0(df, X_clean, gens_all):
    """B0 = 005 A0: standardise with the b0_train reals' clean stats, probe.fit_probe(seed 0, groups=sha, balanced).
    One LOGO model per MAIN generator present in training + 'full' (all b0_train rows) for generators with no training rows
    (FLUX). -> dict(mu, sd, models {g or 'full': model}, which {generator: model key})"""
    lab, gen = df.label.to_numpy(), df.generator.to_numpy()
    b0 = np.where(df.b0_train.to_numpy() & np.isfinite(X_clean).all(1))[0]
    if len(b0) < int(df.b0_train.sum()):
        log(f"  B0: {int(df.b0_train.sum()) - len(b0)} b0_train rows without cached features dropped")
    assert not set(gen[b0]) & set(TEST_ONLY), "a test-only generator is in B0's training rows"
    rtr = b0[lab[b0] == 0]
    mu = X_clean[rtr].mean(0).astype(np.float64)
    sd = np.maximum(X_clean[rtr].std(0).astype(np.float64), 1e-6)
    Z = (X_clean - mu) / sd
    sha = df.sha256.to_numpy()
    trained = sorted(set(gen[b0[lab[b0] == 1]]))
    models, which = {}, {}
    for g in [g for g in MAIN if g in trained]:
        keep = b0[gen[b0] != g]
        models[g] = probe.fit_probe(Z[keep], lab[keep], SEED, groups=sha[keep], class_weight="balanced")
        which[g] = g
    models["full"] = probe.fit_probe(Z[b0], lab[b0], SEED, groups=sha[b0], class_weight="balanced")
    for g in gens_all:
        which.setdefault(g, "full")
    return dict(mu=mu, sd=sd, models=models, which=which)


def apply_b0(fit, X, gens_all):
    Z = (X - fit["mu"]) / fit["sd"]
    cache = {}
    out = {}
    for g in gens_all:
        k = fit["which"][g]
        if k not in cache:
            cache[k] = decide(fit["models"][k], Z)
        out[g] = cache[k]
    return out


# ----------------------------------------------------------------------------- analysis
def analyse(C, n_boot):
    """C: dict with df (orig rows), lab/gen/pool/sp arrays, active, raise_pool, arms, SC[arm][cond] [n,2,S], LAP[arm][cond] [n],
    B0[arm][cond][g], B0X[(A,B)][cond][g]. Returns the results dict."""
    d0, lab, gen, pool, sp = C["df"], C["lab"], C["gen"], C["pool"], C["sp"]
    act, rp, arms = C["active"], C["raise_pool"], C["arms"]
    nat = OC["pools"]["native"]
    POOLN = {"native": nat, "raise": rp, "WA": "vision_whatsapp", "FBH": "vision_facebook_hq"}
    test = (sp == "test") & act
    calib = (sp == "calib") & act
    dev, sha = d0.device.to_numpy(), d0.sha256.to_numpy()
    present = [g for g in MAIN + TEST_ONLY if (test & (lab == 1) & (gen == g)).any()]
    main_p = [g for g in MAIN if g in present]
    FF = {"with_firefly": main_p, "without_firefly": [g for g in main_p if g != "firefly"]}
    if "flux" in present:
        FF["with_firefly_plus_flux"] = main_p + ["flux"]
    PRIM_FF = OC["primary_firefly_variant"]
    fm = test & (lab == 1)
    FS = {k: h4.FakeSet(gen, v, fm, n_boot, SEED) for k, v in FF.items() if v}
    FS_GEN = h4.FakeSet(gen, present, fm, n_boot, SEED)
    RS = {}
    for k, pn in POOLN.items():
        R = np.where(test & (lab == 0) & (pool == pn))[0]
        RS[k] = h4.RealSet(R, dev, sha, n_boot, SEED)
    res = dict(plans=dict(n_real_test={k: int(len(v.R)) for k, v in RS.items()},
                          n_clusters={k: v.n_clusters for k, v in RS.items()},
                          n_fake_test={g: int((fm & (gen == g)).sum()) for g in present},
                          generators_present=present, firefly_variants={k: v for k, v in FF.items()},
                          n_boot=n_boot, seed=SEED, raise_pool_name=rp),
               cells={}, cross_scale={}, scale_diff={}, per_pool={}, operating_point={}, notes=[])
    if not len(RS["raise"].R):
        res["notes"].append("no RAISE test reals: settings (ii), (M), (iii), (iv), the gate and the label are NOT_EVALUABLE")
    for g in TEST_ONLY:
        if g not in present:
            res["notes"].append(f"test-only generator {g} not present")
    if "sd3_large" not in present:
        res["notes"].append("sd3_large not present")

    def ready(arm, st):
        S = SET[st]
        return S["real_cond"] in ARMC[arm] and S["fake_cond"] in ARMC[arm] and len(RS[S["reals"]].R) > 0

    def direction(arm, pk, ffg):
        """texture direction: calib reals of pool `pk` vs TRAIN fakes of generators ffg (never test, never FLUX)."""
        L = C["LAP"][arm]["clean"]
        r = L[np.where(calib & (lab == 0) & (pool == POOLN[pk]))[0]]
        f = L[np.where(act & (sp == "train") & (lab == 1) & np.isin(gen, [g for g in ffg if g not in TEST_ONLY]))[0]]
        r, f = r[np.isfinite(r)], f[np.isfinite(f)]
        return h4.texture_direction(r, f) if len(r) and len(f) else float("nan")

    STORE = {}
    res["texture_direction"] = {}
    for arm in arms:
        res["cells"][arm] = {}
        SCa, LAPa, B0a = C["SC"][arm], C["LAP"][arm], C["B0"][arm]
        for fv in FVS:
            vi = FVS.index(fv)
            res["cells"][arm][fv] = {}
            for ffn, fs in FS.items():
                cell = dict(settings={}, residualized={}, contrasts={}, manipulation={}, resid_fit={})
                for st, S in SET.items():
                    if not ready(arm, st):
                        continue
                    rs, rc, fc, pk = RS[S["reals"]], S["real_cond"], S["fake_cond"], S["reals"]
                    rig = {}
                    for si, sg in enumerate(SIGMAS):
                        rig[sg] = h4.macro_auroc(vec(lab, SCa[rc][:, vi, si], SCa[fc][:, vi, si]), rs, fs)
                    lapv = vec(lab, LAPa[rc], LAPa[fc])
                    dr = direction(arm, pk, FF[ffn])
                    res["texture_direction"][f"{arm}/{ffn}/{pk}"] = dr
                    tex = h4.macro_auroc(dr * lapv, rs, fs) if np.isfinite(dr) else (float("nan"), np.full(n_boot, np.nan))
                    if rc in B0a and fc in B0a:
                        b0 = h4.macro_auroc(lambda g: vec(lab, B0a[rc][g], B0a[fc][g]), rs, fs)
                    else:
                        b0 = (float("nan"), np.full(n_boot, np.nan))
                    cidx = np.where(calib & (lab == 0) & (pool == POOLN[pk]))[0]
                    resid = {}
                    for si, sg in enumerate(SIGMAS):
                        try:
                            r_, info = h4.residualize_on_log_texture(SCa[rc][cidx, vi, si], LAPa[rc][cidx],
                                                                     vec(lab, SCa[rc][:, vi, si], SCa[fc][:, vi, si]), lapv)
                            resid[sg] = h4.macro_auroc(r_, rs, fs)
                            cell["resid_fit"][f"{st}/{sg}"] = info
                        except ValueError:
                            resid[sg] = (float("nan"), np.full(n_boot, np.nan))
                    STORE[(arm, fv, ffn, st)] = dict(rig=rig, tex=tex, b0=b0, resid=resid)
                    cell["settings"][st] = dict(
                        desc=S["desc"], n_real=int(len(rs.R)),
                        rigid={str(sg): h4.vc(*rig[sg]) for sg in SIGMAS}, texture=h4.vc(*tex), b0=h4.vc(*b0))
                    cell["residualized"][st] = {str(sg): h4.vc(*resid[sg]) for sg in SIGMAS}
                # ---- contrasts + manipulation check
                pc = OC["primary_contrast"]
                for A, B in [pc] + OC["secondary_contrasts"] + OC["other_contrasts"]:
                    if (arm, fv, ffn, A) not in STORE or (arm, fv, ffn, B) not in STORE:
                        continue
                    a, b = STORE[(arm, fv, ffn, A)], STORE[(arm, fv, ffn, B)]
                    SA, SB = SET[A], SET[B]
                    la = lambda st, S, kind: (LAPa[S["real_cond"]][RS[S["reals"]].R] if kind == "r"
                                              else LAPa[S["fake_cond"]][np.concatenate([fs.F[g] for g in fs.gens])])
                    cell["contrasts"][f"{A}-{B}"] = dict(
                        rigid={str(sg): h4.dc(*a["rig"][sg], *b["rig"][sg]) for sg in SIGMAS},
                        texture=h4.dc(*a["tex"], *b["tex"]), b0=h4.dc(*a["b0"], *b["b0"]),
                        residualized={str(sg): h4.dc(*a["resid"][sg], *b["resid"][sg]) for sg in SIGMAS})
                    cell["manipulation"][f"{A}-{B}"] = dict(
                        laplacian_reals=h4.mean_change(la(A, SA, "r"), la(B, SB, "r")),
                        laplacian_fakes=h4.mean_change(la(A, SA, "f"), la(B, SB, "f")),
                        b0_gap=cell["contrasts"][f"{A}-{B}"]["b0"], texture_gap=cell["contrasts"][f"{A}-{B}"]["texture"])
                # ---- primary label (Amendment 6)
                if all((arm, fv, ffn, s) in STORE for s in pc):
                    m_, ii = STORE[(arm, fv, ffn, pc[0])], STORE[(arm, fv, ffn, pc[1])]
                    auc = {sg: ii["rig"][sg][0] for sg in SIGMAS}
                    cil = {sg: h4.ci(ii["rig"][sg][1])[0] if np.isfinite(ii["rig"][sg][0]) else float("nan") for sg in SIGMAS}
                    gaps = {sg: m_["rig"][sg][0] - ii["rig"][sg][0] for sg in SIGMAS}
                    b0g = m_["b0"][0] - ii["b0"][0]
                    lab_ = h4.official_label(auc, cil, gaps, b0g, ii["tex"][0])
                    cell["primary"] = dict(contrast=f"{pc[0]}-{pc[1]}", **lab_,
                                           auroc_ii={str(sg): auc[sg] for sg in SIGMAS},
                                           gap_by_sigma={str(sg): gaps[sg] for sg in SIGMAS}, b0_gap=float(b0g),
                                           texture_auroc_ii=float(ii["tex"][0]),
                                           is_label_cell=bool(arm == OC["label_arm"] and ffn == PRIM_FF and fv == PRIMARY_FV))
                    log(f"[{arm}/{fv}/{ffn}] best sigma {lab_['best_sigma']}: AUROC(ii)={auc.get(lab_['best_sigma'], float('nan')):.3f} "
                        f"gap(M-ii)={gaps.get(lab_['best_sigma'], float('nan')):+.3f} B0 gap={b0g:+.3f} "
                        f"texture(ii)={ii['tex'][0]:.3f} gate={lab_['gate']['ok']} -> {lab_['label']}")
                else:
                    cell["primary"] = dict(label="NOT_EVALUABLE", note="settings (ii) and/or (M) not available",
                                           is_label_cell=bool(arm == OC["label_arm"] and ffn == PRIM_FF and fv == PRIMARY_FV))
                res["cells"][arm][fv][ffn] = cell
            # ---- per generator (incl. FLUX), once per (arm, fv)
            pg = {}
            for st, S in SET.items():
                if not ready(arm, st):
                    continue
                rs, rc, fc, pk = RS[S["reals"]], S["real_cond"], S["fake_cond"], S["reals"]
                dr = direction(arm, pk, FF["with_firefly"])
                lapv = vec(lab, LAPa[rc], LAPa[fc])
                pg[st] = {}
                for g in present:
                    e = dict(rigid={str(sg): h4.vc(*h4.macro_auroc(vec(lab, SCa[rc][:, vi, si], SCa[fc][:, vi, si]), rs, FS_GEN, [g]))
                                    for si, sg in enumerate(SIGMAS)},
                             texture=h4.vc(*h4.macro_auroc(dr * lapv, rs, FS_GEN, [g])) if np.isfinite(dr) else None)
                    if rc in B0a and fc in B0a:
                        e["b0"] = h4.vc(*h4.macro_auroc(lambda gg: vec(lab, B0a[rc][gg], B0a[fc][gg]), rs, FS_GEN, [g]))
                    pg[st][g] = e
            res["cells"][arm][fv]["per_generator"] = pg
        # ---- per real pool, clean vs clean (descriptive) and 1%-FPR operating point
        res["per_pool"][arm] = {}
        for ffn in ("with_firefly", "without_firefly"):
            if ffn not in FS:
                continue
            o = {}
            for pk, rs in RS.items():
                if not len(rs.R) or "clean" not in ARMC[arm]:
                    continue
                dr = direction(arm, "native" if pk in ("WA", "FBH") else pk, FF[ffn])
                o[pk] = dict(n_real=int(len(rs.R)),
                             b0=h4.vc(*h4.macro_auroc(lambda g: B0a["clean"][g], rs, FS[ffn])) if "clean" in B0a else None,
                             texture=h4.vc(*h4.macro_auroc(dr * LAPa["clean"], rs, FS[ffn])) if np.isfinite(dr) else None)
                for fv in FVS:
                    vi = FVS.index(fv)
                    o[pk][f"rigid_{fv}"] = {str(sg): h4.vc(*h4.macro_auroc(SCa["clean"][:, vi, si], rs, FS[ffn]))
                                            for si, sg in enumerate(SIGMAS)}
            res["per_pool"][arm][ffn] = o
        res["operating_point"][arm] = {}
        for fv in FVS:
            vi = FVS.index(fv)
            op = {}
            for si, sg in enumerate(SIGMAS):
                for pk in ("native", "raise"):
                    cidx = np.where(calib & (lab == 0) & (pool == POOLN[pk]))[0]
                    if not len(cidx) or np.isnan(SCa["clean"][cidx, vi, si]).any():
                        continue
                    thr = metrics.threshold_for_fpr(SCa["clean"][cidx, vi, si], 0.01)
                    e = dict(threshold=thr, n_calib=int(len(cidx)))
                    for st, S in SET.items():
                        if S["reals"] != pk or not ready(arm, st):
                            continue
                        e[f"fpr_{st}"] = float((SCa[S["real_cond"]][RS[pk].R, vi, si] >= thr).mean())
                        for ffn in ("with_firefly", "without_firefly"):
                            Fm = np.concatenate([FS[ffn].F[g] for g in FS[ffn].gens])
                            e[f"tpr_{ffn}_{st}"] = float((SCa[S["fake_cond"]][Fm, vi, si] >= thr).mean())
                    op[f"{sg}/{pk}"] = e
            res["operating_point"][arm][fv] = op
    # ---- cross-scale B0 (Amendment 6.6) and scale differences
    for (A, B), BX in C["B0X"].items():
        key = f"{A}->{B}"
        res["cross_scale"][key] = {}
        for ffn in ("with_firefly", "without_firefly"):
            if ffn not in FS:
                continue
            o = {}
            for st, S in SET.items():
                rc, fc = S["real_cond"], S["fake_cond"]
                if not (ready(B, st) and ready(A, st) and rc in BX and fc in BX and rc in C["B0"][B] and fc in C["B0"][B]
                        and rc in C["B0"][A] and fc in C["B0"][A]):
                    continue
                rs = RS[S["reals"]]
                x = h4.macro_auroc(lambda g: vec(lab, BX[rc][g], BX[fc][g]), rs, FS[ffn])
                wb = h4.macro_auroc(lambda g: vec(lab, C["B0"][B][rc][g], C["B0"][B][fc][g]), rs, FS[ffn])
                wa = h4.macro_auroc(lambda g: vec(lab, C["B0"][A][rc][g], C["B0"][A][fc][g]), rs, FS[ffn])
                o[st] = dict(cross=h4.vc(*x), within_test_arm=h4.vc(*wb), within_train_arm=h4.vc(*wa),
                             cross_minus_within_test_arm=h4.dc(*x, *wb))
            res["cross_scale"][key][ffn] = o
    for fv in FVS:
        res["scale_diff"][fv] = {}
        for ffn in FS:
            for other in [a for a in arms if a != "orig"]:
                if "orig" not in arms:
                    break
                o = {}
                for st in ("i", "ii", "M"):
                    a, b = STORE.get(("orig", fv, ffn, st)), STORE.get((other, fv, ffn, st))
                    if a and b:
                        o[st] = dict(rigid={str(sg): h4.dc(*a["rig"][sg], *b["rig"][sg]) for sg in SIGMAS},
                                     texture=h4.dc(*a["tex"], *b["tex"]), b0=h4.dc(*a["b0"], *b["b0"]))
                if o:
                    res["scale_diff"][fv][f"{ffn}/orig-{other}"] = o
    return res


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- tables
def fv_ci(e):
    if e is None or e.get("value") is None or not np.isfinite(e["value"]):
        return "n/a"
    return f"{e['value']:.3f} [{e['ci'][0]:.3f},{e['ci'][1]:.3f}]"


def fd_ci(e):
    if e is None or not np.isfinite(e["delta"]):
        return "n/a"
    return f"{e['delta']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"


def write_tables(res, meta, path):
    banner = meta["banner"]
    L = [f"# H4 {meta['kind']} (experiment 006, Amendment 6)\n", banner + "\n",
         f"Run: {meta['stamp']} | n_boot={res['plans']['n_boot']} | primary feature variant = {PRIMARY_FV} | label arm = `{OC['label_arm']}` | "
         f"primary Firefly variant = {OC['primary_firefly_variant']}\n",
         f"Test reals: {res['plans']['n_real_test']} | clusters {res['plans']['n_clusters']} | test fakes {res['plans']['n_fake_test']}\n",
         "AUROC = LOGO-macro over the listed generators (each generator's test fakes vs the same reals). 95% CI = cluster bootstrap "
         "(reals by device / RAISE image, fakes by image per generator); gaps are paired on the same resamples where the reals are "
         "the same, on the fakes only otherwise. RIGID: higher = fake, direction never flipped. Texture = |Laplacian| baseline with "
         "its direction set on calib reals vs train fakes.\n"]
    for n in res["notes"]:
        L.append(f"NOTE: {n}\n")
    arms = list(res["cells"])
    for fv in [PRIMARY_FV] + [v for v in FVS if v != PRIMARY_FV]:
        tag = "PRIMARY" if fv == PRIMARY_FV else "SENSITIVITY (pre-final-LayerNorm CLS)"
        L.append(f"\n## Labels, feature variant `{fv}` ({tag}); primary contrast AUROC(M) - AUROC(ii)\n")
        L.append("| arm | Firefly | best sigma | AUROC(ii) | texture AUROC(ii) | gate (CI lo>0.5 / beats texture) | gap (M)-(ii) | B0 gap | label |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            for ffn, c in res["cells"][arm][fv].items():
                if ffn == "per_generator":
                    continue
                p = c["primary"]
                if "gate" not in p or p.get("best_sigma") is None:
                    L.append(f"| {arm} | {ffn} | - | - | - | - | - | - | **{p['label']}**{' (label cell)' if p['is_label_cell'] else ''} |")
                    continue
                bs = str(p["best_sigma"])
                g = p["gate"]
                L.append(f"| {arm} | {ffn} | {bs} | {fv_ci(c['settings']['ii']['rigid'][bs])} | {fv_ci(c['settings']['ii']['texture'])} | "
                         f"{g['ci_lower_gt_half']} / {g['beats_texture']} | {fd_ci(c['contrasts']['M-ii']['rigid'][bs])} | "
                         f"{fd_ci(c['contrasts']['M-ii']['b0'])} | **{p['label']}**{' (label cell)' if p['is_label_cell'] else ''} |")
        for arm in arms:
            for ffn, c in res["cells"][arm][fv].items():
                if ffn == "per_generator":
                    continue
                L.append(f"\n### {fv} / {arm} / {ffn}: AUROC per setting\n")
                L.append("| score | " + " | ".join(f"({s})" for s in c["settings"]) + " |")
                L.append("|---|" + "---|" * len(c["settings"]))
                for sg in SIGMAS:
                    L.append(f"| RIGID s={sg} | " + " | ".join(fv_ci(e["rigid"][str(sg)]) for e in c["settings"].values()) + " |")
                for sg in SIGMAS:
                    L.append(f"| RIGID s={sg} residualised on log-Laplacian | " + " | ".join(fv_ci(c["residualized"][s][str(sg)]) for s in c["settings"]) + " |")
                L.append("| texture baseline | " + " | ".join(fv_ci(e["texture"]) for e in c["settings"].values()) + " |")
                L.append("| B0 (LOGO) | " + " | ".join(fv_ci(e["b0"]) for e in c["settings"].values()) + " |")
                if c["contrasts"]:
                    L.append(f"\nContrasts (A - B), {fv}/{arm}/{ffn}; first = primary, then secondary, then continuity\n")
                    L.append("| contrast | " + " | ".join(f"RIGID s={s}" for s in SIGMAS) + " | texture | B0 | mean Lap reals (A/B ratio) | mean Lap fakes (A/B ratio) |")
                    L.append("|---|" + "---|" * (len(SIGMAS) + 4))
                    for k, e in c["contrasts"].items():
                        m = c["manipulation"][k]
                        L.append(f"| {k} | " + " | ".join(fd_ci(e["rigid"][str(s)]) for s in SIGMAS) +
                                 f" | {fd_ci(e['texture'])} | {fd_ci(e['b0'])} | {m['laplacian_reals']['delta']:+.4f} ({m['laplacian_reals']['ratio']:.2f}x) | "
                                 f"{m['laplacian_fakes']['delta']:+.4f} ({m['laplacian_fakes']['ratio']:.2f}x) |")
        L.append(f"\n## Per generator (incl. FLUX where present), `{fv}`, Firefly in\n")
        for arm in arms:
            pg = res["cells"][arm][fv].get("per_generator", {})
            if not pg:
                continue
            L.append(f"\n### {arm}\n")
            sts = list(pg)
            L.append("| generator | score | " + " | ".join(f"({s})" for s in sts) + " |")
            L.append("|---|---|" + "---|" * len(sts))
            for g in res["plans"]["generators_present"]:
                for nm, f in [(f"RIGID s={s}", lambda e, s=s: e["rigid"][str(s)]) for s in SIGMAS] + [("texture", lambda e: e["texture"]), ("B0", lambda e: e.get("b0"))]:
                    L.append(f"| {g} | {nm} | " + " | ".join(fv_ci(f(pg[s][g])) for s in sts) + " |")
        L.append(f"\n## 1%-FPR operating point of the RIGID score, `{fv}` (calib reals of the pool, clean)\n")
        L.append("| arm | sigma/pool | thr | n_calib | FPR (settings of the pool) | TPR with FF | TPR without FF |")
        L.append("|---|---|---|---|---|---|---|")
        for arm in arms:
            for k, e in res["operating_point"][arm][fv].items():
                fprs = ", ".join(f"({x[4:]}) {v:.3f}" for x, v in e.items() if x.startswith("fpr_"))
                tw = ", ".join(f"({x.split('_', 3)[-1]}) {v:.3f}" for x, v in e.items() if x.startswith("tpr_with_firefly_"))
                to = ", ".join(f"({x.split('_', 3)[-1]}) {v:.3f}" for x, v in e.items() if x.startswith("tpr_without_firefly_"))
                L.append(f"| {arm} | {k} | {e['threshold']:.4g} | {e['n_calib']} | {fprs} | {tw} | {to} |")
        L.append(f"\n## Scale arms, `{fv}`: orig - other arm (descriptive, paired where the reals are shared)\n")
        L.append("| Firefly / pair | setting | " + " | ".join(f"RIGID s={s}" for s in SIGMAS) + " | texture | B0 |")
        L.append("|---|---|" + "---|" * (len(SIGMAS) + 2))
        for k, o in res["scale_diff"][fv].items():
            for st, e in o.items():
                L.append(f"| {k} | ({st}) | " + " | ".join(fd_ci(e["rigid"][str(s)]) for s in SIGMAS) + f" | {fd_ci(e['texture'])} | {fd_ci(e['b0'])} |")
    L.append("\n## Cross-scale B0 (Amendment 6.6; descriptive): probe trained on arm A, tested on arm B\n")
    L.append("| pair | Firefly | setting | cross (A->B) | B0 trained on B (within test arm) | B0 trained on A (within train arm) | cross - within test arm |")
    L.append("|---|---|---|---|---|---|---|")
    for k, o in res["cross_scale"].items():
        for ffn, oo in o.items():
            for st, e in oo.items():
                L.append(f"| {k} | {ffn} | ({st}) | {fv_ci(e['cross'])} | {fv_ci(e['within_test_arm'])} | {fv_ci(e['within_train_arm'])} | {fd_ci(e['cross_minus_within_test_arm'])} |")
    L.append("\n## Per real pool, clean vs clean fakes (descriptive; RAISE row = setting (ii))\n")
    L.append("| arm | Firefly | pool | n_real | B0 | texture | " + " | ".join(f"RIGID({PRIMARY_FV}) s={s}" for s in SIGMAS) + " |")
    L.append("|---|---|---|---|---|---|" + "---|" * len(SIGMAS))
    for arm, oo in res["per_pool"].items():
        for ffn, o in oo.items():
            for pk, e in o.items():
                L.append(f"| {arm} | {ffn} | {pk} | {e['n_real']} | {fv_ci(e['b0'])} | {fv_ci(e['texture'])} | " +
                         " | ".join(fv_ci(e[f"rigid_{PRIMARY_FV}"][str(s)]) for s in SIGMAS) + " |")
    L.append("\n## D-008 guard and pipeline check\n")
    g = meta["d008"]
    L.append(f"n={g['n']}, device {g['device']}, tol (scaled feature diff) {g['feat_tol']}, overall ok={g['ok']}\n")
    L.append("| variant | max abs feat diff | max scaled feat diff | max abs score diff | ok |")
    L.append("|---|---|---|---|---|")
    for v, e in g["variants"].items():
        L.append(f"| {v} | {e['max_abs_feature_diff']:.2e} | {e['max_scaled_feature_diff']:.2e} | {e['max_abs_score_diff']:.2e} | {e['ok']} |")
    p = g["pipeline_vs_004_cache"]
    L.append(f"\nPipeline vs 004 cache (unnoised pre-LN CLS vs primary[:, :768], same 32 images): max abs {p['max_abs_diff']:.2e}, scaled {p['max_scaled_diff']:.2e}, ok={p['ok']}")
    path.write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["official", "pilot"], default="official")
    ap.add_argument("--dry-run", action="store_true", help="pilot data, tiny subset, n_boot 50, output in cache/ (not a result)")
    ap.add_argument("--standin-raise", action="store_true", help="with --dry-run: add the 20-image RAISE stand-in (make_standin.py)")
    ap.add_argument("--allow-missing", action="store_true", help="skip absent arms/crops instead of aborting (implied by --dry-run)")
    ap.add_argument("--out", default=None, help="output dir (default official/ or the dry-run dir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-guard", action="store_true", help="dry-run only: skip the D-008 guard (e.g. on a CPU-only box)")
    a = ap.parse_args()
    if a.skip_guard and not a.dry_run:
        raise SystemExit("--skip-guard is for --dry-run only: the D-008 guard is mandatory in the official run")
    if a.standin_raise and not a.dry_run:
        raise SystemExit("--standin-raise is a dry-run fixture only")
    prof = "pilot" if a.dry_run else a.profile
    PR = OC["profiles"][prof]
    allow = a.allow_missing or a.dry_run
    tag = "official_standin_dryrun" if a.standin_raise else "official_dryrun"
    out_dir = Path(a.out) if a.out else (HERE / "cache" / tag / "out" if a.dry_run else HERE / PR["out_dir"])
    cache = (HERE / "cache" / tag / "scores") if a.dry_run else HERE / "cache" / (out_dir.name + "_scores")
    out_dir.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    LOGF = open(out_dir / "run.log", "a")
    H.LOGF = LOGF
    n_boot = OC["dry_run"]["n_boot"] if a.dry_run else OC["aggregation"]["n_boot"]
    raise_pool = OC["standin_raise"]["pool_name"] if a.standin_raise else OC["pools"]["raise"]
    kind = "DRY-RUN (code-path check, pilot data)" if a.dry_run else "OFFICIAL"
    banner = ("**NOT RESULTS.** Code-path check on the pilot data only; "
              + ("the `raise_standin` pool is 20 VISION natives re-saved as PNG, NOT RAISE." if a.standin_raise else "no RAISE reals: (ii)/(M)/(iii)/(iv) are skipped.")
              if a.dry_run else "Official run on the full 003 splits.")
    log(f"H4 {kind} profile={prof} out={out_dir} n_boot={n_boot} raise pool={raise_pool}")

    # ---- splits per arm
    arms, dfs, subdirs = [], {}, {}
    for arm in ARMS_ALL:
        p = ROOT / PR["splits"][arm]
        if not p.exists():
            if allow:
                log(f"arm {arm}: splits {p.relative_to(ROOT)} missing -> arm skipped"); continue
            raise SystemExit(f"arm {arm}: {p} missing (run the 004 fov_prep step first)")
        if allow and not P.cond_dir("clean", P.cache_root(PR["feature_subdirs"][arm])).exists():
            log(f"arm {arm}: no 004 features in cache/{PR['feature_subdirs'][arm]} -> arm skipped"); continue
        df = load_df(p)
        sd = [PR["feature_subdirs"][arm]]
        if a.standin_raise:
            sp_ = HERE / OC["standin_raise"]["dir"] / f"splits_standin_{arm}.csv"
            if not sp_.exists():
                raise SystemExit(f"{sp_} missing: run make_standin.py first")
            df = pd.concat([df, load_df(sp_)], ignore_index=True)
            sd.append(f"standin_{arm}")
        dfs[arm], subdirs[arm] = df, sd
        arms.append(arm)
    if "orig" not in arms:
        raise SystemExit("the orig arm is required")
    d0 = dfs["orig"]
    for arm in arms:
        assert (dfs[arm].sha256 == d0.sha256).all(), f"arm {arm} must list the same rows in the same order as orig"
    exists = np.ones(len(d0), bool)
    for arm in arms:
        exists &= np.array([Path(p).exists() for p in dfs[arm].path])
    if not exists.all() and not allow:
        bad = d0[~exists]
        raise SystemExit(f"{len(bad)} crops missing in at least one arm (e.g. {bad.path.iloc[0]}): run the 004 prep steps")
    if exists.sum() < len(d0):
        log(f"{int((~exists).sum())} rows without crops in some arm are dropped (allow-missing)")
    lab, gen, pool, sp = d0.label.to_numpy(), d0.generator.to_numpy(), d0.pool.to_numpy(), d0.split.to_numpy()
    assert not ((gen == "flux") & (sp != "test")).any() and not d0.b0_train.to_numpy()[gen == "flux"].any(), "FLUX must be test-only"
    if not allow:      # completeness of the official data
        need = {"raise test reals": ((pool == raise_pool) & (sp == "test")).any(), "sd3_large": (gen == "sd3_large").any(),
                "flux": (gen == "flux").any()}
        if not all(need.values()):
            raise SystemExit(f"official data incomplete: {need}")
    active = exists.copy()
    if a.dry_run:                 # small, device-spread subset
        dr = OC["dry_run"]
        nat = OC["pools"]["native"]
        t = d0[(d0.split == "test") & (d0.pool == nat)].sort_values(["device", "sha256"])
        pick = t.groupby("device").head(2).head(dr["native_test"]).index.tolist()
        pick += d0[(d0.split == "calib") & (d0.pool == nat)].head(dr["native_calib"]).index.tolist()
        for dp in OC["pools"]["derived"]:
            pick += d0[(d0.split == "test") & (d0.pool == dp)].head(dr["derived_test_per_pool"]).index.tolist()
        pick += d0[(d0.split == "test") & (d0.label == 1)].groupby("generator").head(dr["fakes_test_per_generator"]).index.tolist()
        pick += d0[(d0.split == "train") & (d0.label == 1)].groupby("generator").head(dr["fakes_train_per_generator"]).index.tolist()
        pick += d0.index[d0.pool == raise_pool].tolist()
        sub = np.zeros(len(d0), bool); sub[pick] = True
        active &= sub
    log(f"arms {arms}; rows active {int(active.sum())}/{len(d0)}")
    arm_conds = {arm: [c for c in ARMC[arm]] for arm in arms}

    # ---- model + D-008 guard
    dev = get_device(a.device)
    p4 = H.load_004_run()
    if not p4.weights_cached():
        raise SystemExit("facebook/dinov2-base not in data/hf_cache")
    bb = load_backbone(P.SPEC, input_size=224, layers=(-1,), use_cls=True, use_mean=False)
    model = bb.model.to(dev).eval()
    H.args_subdir_for_check.update({"orig": subdirs["orig"][0]})
    if a.skip_guard:
        guard = dict(n=0, device=str(dev), feat_tol=H.FEAT_TOL, ok=None, skipped="dry-run --skip-guard", variants={},
                     pipeline_vs_004_cache=dict(max_abs_diff=float("nan"), max_scaled_diff=float("nan"), ok=None))
    else:
        guard = H.d008_guard(d0, model, dev, p4, out_dir)

    # ---- scores + Laplacian
    core, derived, lap_rows = plan_rows(d0, active, raise_pool)
    SC = {arm: {c: np.full((len(d0), 2, len(SIGMAS)), np.nan) for c in arm_conds[arm]} for arm in arms}
    LAP = {arm: {c: np.full(len(d0), np.nan) for c in arm_conds[arm]} for arm in arms}
    timings = {}
    with ThreadPoolExecutor(4) as tp:
        for arm in arms:
            df = dfs[arm]
            for cond in arm_conds[arm]:
                rows = np.where(core | (derived if cond == "clean" else False))[0]
                shas = df.sha256.to_numpy()[rows].tolist()
                cp = cache / f"{arm}__{P.safe(cond)}.npz"
                if cp.exists() and np.load(cp, allow_pickle=False)["keys"].tolist() == shas:
                    z = np.load(cp, allow_pickle=False)
                    SC[arm][cond][rows], LAP[arm][cond][rows] = z["scores"], z["lap"]
                    log(f"  {arm}/{cond}: loaded {len(rows)} rows from cache")
                else:
                    t = time.time()
                    log(f"  {arm}/{cond}: scoring {len(rows)} images x {len(SIGMAS)} sigmas x {DRAWS} draws")
                    s, lp = score_condition(model, dev, p4, tp, df.path.to_numpy()[rows].tolist(), shas, cond)
                    np.savez(cp, keys=np.array(shas, dtype=str), scores=s, lap=lp)
                    SC[arm][cond][rows], LAP[arm][cond][rows] = s, lp
                    timings[f"{arm}/{cond}"] = time.time() - t
            lr = np.where(lap_rows)[0]
            shas = df.sha256.to_numpy()[lr].tolist()
            cp = cache / f"{arm}__clean__laponly_trainfakes.npz"
            if cp.exists() and np.load(cp, allow_pickle=False)["keys"].tolist() == shas:
                LAP[arm]["clean"][lr] = np.load(cp, allow_pickle=False)["lap"]
            else:
                lp = lap_only(p4, tp, df.path.to_numpy()[lr].tolist(), shas, "clean")
                np.savez(cp, keys=np.array(shas, dtype=str), lap=lp)
                LAP[arm]["clean"][lr] = lp
    del model; torch.cuda.empty_cache()

    # ---- B0 (+ cross-scale)
    gens_all = [g for g in MAIN + TEST_ONLY if (active & (lab == 1) & (gen == g) & (sp == "test")).any()]
    FIT, XF, B0 = {}, {}, {}
    for arm in arms:
        XF[arm] = {}
        for c in arm_conds[arm]:
            try:
                XF[arm][c] = load_aligned_multi(dfs[arm], c, subdirs[arm])
            except FileNotFoundError as e:
                log(f"  B0[{arm}] {c}: {e} -> skipped")
        if "clean" not in XF[arm]:
            raise SystemExit(f"no clean features for arm {arm}: run the 004 extraction first")
        FIT[arm] = fit_b0(dfs[arm], XF[arm]["clean"], gens_all)
        B0[arm] = {c: apply_b0(FIT[arm], X, gens_all) for c, X in XF[arm].items()}
        log(f"B0[{arm}]: models {sorted(FIT[arm]['models'])}, conditions {sorted(B0[arm])}")
    B0X = {}
    for A in arms:
        for B in arms:
            if A != B:
                B0X[(A, B)] = {c: apply_b0(FIT[A], XF[B][c], gens_all) for c in XF[B] if c in XF[A]}
    C = dict(df=d0, lab=lab, gen=gen, pool=pool, sp=sp, active=active, raise_pool=raise_pool, arms=arms,
             SC=SC, LAP=LAP, B0=B0, B0X=B0X)
    res = analyse(C, n_boot)
    meta = dict(stamp=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind, banner=banner, dry_run=a.dry_run, standin_raise=a.standin_raise,
                profile=prof, d008=guard, timings=timings, device=str(dev), subdirs=subdirs, label_arm=OC["label_arm"],
                config="experiments/006-format-shortcut/config_official.json", arms=arms,
                n_rows=int(len(d0)), n_active=int(active.sum()))
    res["meta"] = meta
    (out_dir / "results.json").write_text(json.dumps(sanitize(res), indent=1))
    write_tables(res, meta, out_dir / "tables.md")
    rows = []
    for arm in arms:
        df = dfs[arm]
        for cond in arm_conds[arm]:
            idx = np.where(np.isfinite(LAP[arm][cond]))[0]
            for fi, fv in enumerate(FVS):
                for si, sg in enumerate(SIGMAS):
                    s_ = df.iloc[idx]
                    rows.append(pd.DataFrame(dict(arm=arm, cond=cond, feat=fv, sigma=sg, sha256=s_.sha256.values, label=s_.label.values,
                                                  generator=s_.generator.values, pool=s_.pool.values, device=s_.device.values,
                                                  split=s_.split.values, score=SC[arm][cond][idx, fi, si], laplacian=LAP[arm][cond][idx])))
    pd.concat(rows).to_csv(out_dir / "scores_long.csv.gz", index=False)
    meta["total_seconds"] = time.time() - T0
    (out_dir / "results.json").write_text(json.dumps(sanitize(res), indent=1))
    log(f"done in {time.time() - T0:.0f}s; outputs in {out_dir}")
    if a.dry_run:
        print("DRY-RUN OK")


if __name__ == "__main__":
    sys.exit(main() or 0)
