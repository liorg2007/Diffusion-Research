"""E1b (Amendment 10): separate zoom from resampling with the `du2` arm. PILOT, exploratory.

  env/.venv/bin/python -I experiments/008-scale-sensitivity/run_du2.py            # all stages (resumable) -> pilot_du2/
  env/.venv/bin/python -I experiments/008-scale-sensitivity/run_du2.py cpuref NAME  # internal (CPU reference for D-008)

Arms: orig (003 crop512), du2 (short side -> 512, bicubic back to the original size, plain centre crop 512), fov512.
  Delta_resample = AUROC(orig) - AUROC(du2)   (resampling at fixed zoom)
  Delta_zoom     = AUROC(du2)  - AUROC(fov512) (zoom at fixed resampling);  the sum is E1's Delta exactly.
Reuses E1 (run.py): orig scores from 007 (verified in E1), fov512 scores and B0 models from experiments/008-scale-sensitivity/cache/full.
New here: du2 features (004 cache pilot_du2), du2 scores of CommFor/RINE/ClipDet (D-008 guard), B0 retrained on du2,
fixed-probe B0 on du2, three-arm decomposition with a paired cluster bootstrap, cross-scale 1%-FPR with device-bootstrap CIs.
"""
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R          # noqa: E402  (E1 module: loaders, Plan, scorers; its __main__ block does not run)

import numpy as np       # noqa: E402
import pandas as pd      # noqa: E402

import metrics           # noqa: E402
import p4features as P   # noqa: E402

OUT = HERE / "pilot_du2"
CACHE = R.CACHE          # == cache/full (E1 caches are reused; new files are du2-specific)
ROOT = R.ROOT
SPL_DU2 = ROOT / "experiments/004-phase4-features/pilot/splits_pilot_du2.csv"
R.SPL["du2"] = SPL_DU2
R.SUBDIR["du2"] = "pilot_du2"
A3 = ["orig", "du2", "fov512"]
CONDS, DETS, GENS, POOLS = R.CONDS, R.DETS, R.GENS, R.POOLS
N_BOOT, SEED, TOL = R.N_BOOT, R.SEED, R.TOL
E1RES = HERE / "pilot" / "results.json"
T0 = time.time()
log = R.log


# ----------------------------------------------------------------------------- stages
def stage_lap():
    old = R.ARMS
    R.ARMS = ["du2"]
    try:
        d = R.stage_lap()["du2"]
    finally:
        R.ARMS = old
    return {"orig": np.load(CACHE / "lap_orig.npy"), "fov512": np.load(CACHE / "lap_fov512.npy"), "du2": d}


def stage_b0():
    """-> B0r[arm][model] [2,n] (A0 trained+evaluated in that arm), X[train][test][model] [2,n] (fixed probe `train` on `test`'s features)."""
    r = R.load_r005()
    df = R.load_df("orig")
    F = {a: {c: P.load_features_aligned(df, "primary", c, subdir=R.SUBDIR[a]) for c in CONDS} for a in A3}
    for a in A3:
        for c in CONDS:
            assert not np.isnan(F[a][c]).any(), (a, c)
    r.SPLITS = R.SPL["orig"]
    nat, wa, fb = r.pair_indices(df)
    rtr = r.real_train_indices(df, nat)
    ms = {}
    for a in A3:
        mu = F[a]["clean"][rtr].mean(0).astype(np.float64)
        ms[a] = (mu, np.maximum(F[a]["clean"][rtr].std(0).astype(np.float64), 1e-6))
    models = {a: pickle.loads((CACHE / f"b0_{a}" / "models_A0.pkl").read_bytes()) for a in ("orig", "fov512")}
    # --- retrain A0 on du2 features through 005/run.py (same recipe: grouped CV, balanced), capturing the probes
    d = CACHE / "b0_du2"; d.mkdir(parents=True, exist_ok=True)
    r.OUT, r.CACHE, r.SUBDIR, r.GROUPED_CV, r.BALANCED, r.LABEL = d, d, "pilot_du2", True, True, "E1b-B0"
    r.CONDS, r.CI = list(CONDS), {c: i for i, c in enumerate(CONDS)}
    Zev = {c: ((F["du2"][c] - ms["du2"][0]) / ms["du2"][1]).astype(np.float64) for c in CONDS}
    pk = d / "models_A0.pkl"
    if pk.exists() and (d / "scores_A0.npz").exists():
        models["du2"] = pickle.loads(pk.read_bytes())
    else:
        (d / "scores_A0.npz").unlink(missing_ok=True)
        cap, orig_fit = [], r.probe.fit_probe
        r.probe.fit_probe = lambda *a, **k: (cap.append(orig_fit(*a, **k)), cap[-1])[1]
        try:
            r.run_arm("A0", df, lambda X: X, Zev, {}, aug=False)
        finally:
            r.probe.fit_probe = orig_fit
        assert len(cap) == len(r.MODELS)
        models["du2"] = dict(zip(r.MODELS, cap))
        pk.write_bytes(pickle.dumps(models["du2"]))
    X = {tr: {} for tr in A3}
    for tr in A3:
        for te in A3:
            Z = [((F[te][c] - ms[tr][0]) / ms[tr][1]).astype(np.float64) for c in CONDS]
            X[tr][te] = {m: np.stack([mdl.decision_function(z) for z in Z]).astype(np.float32) for m, mdl in models[tr].items()}
    B0r = {a: X[a][a] for a in A3}
    # consistency with E1's cached retrained-per-arm scores
    for a in ("orig", "fov512"):
        z = np.load(CACHE / f"b0_{a}" / "scores_A0.npz")
        dif = max(float(np.abs(z[m][[0, 1]] - B0r[a][m]).max()) for m in B0r[a])
        assert dif <= 1e-4, f"B0 {a}: re-scored cached probe differs from E1 scores by {dif:.2e}"
        log(f"B0 {a}: fixed-probe re-score == E1 cached A0 scores (max diff {dif:.1e})")
    return B0r, X


def cpuref(name):
    import os
    os.environ["DD_DEVICE"] = "cpu"
    dfd = R.load_df("du2").set_index("sha256", drop=False)
    sub = R.check_subset(R.load_df("orig"))
    fn = R.sp.get_fn(name)
    p = CACHE / f"cpuref_{name}_du2.json"
    ref = json.loads(p.read_text()) if p.exists() else {}
    for c in CONDS:
        if c not in ref:
            ref[c] = R.sp.score_rows(fn, dfd.loc[sub], c, batch=4)
            p.write_text(json.dumps(ref))
            log(f"cpuref {name} {c} done")


def stage_score():
    """-> PS[arm][det][cond] {sha: score} for the rows of R.selection, guard dict (du2 CUDA vs CPU)."""
    import os
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dfo, dfd, dff = R.load_df("orig"), R.load_df("du2"), R.load_df("fov512")
    assert (dfo.sha256.values == dfd.sha256.values).all() and (dfo.sha256.values == dff.sha256.values).all()
    sel = R.selection(dfo)
    shas = dfo.sha256[sel].tolist()
    sub = R.check_subset(dfo)
    dfi = dfd.set_index("sha256", drop=False)
    s007 = pd.read_csv(R.SCORES007)
    PS = {a: {} for a in A3}
    guard = {}
    need_gpu = any(len(R.sp.load_scores(f"{n}_du2", c)) < len(shas) for n in DETS for c in CONDS)
    if need_gpu:
        assert torch.cuda.is_available()
    for name in DETS:
        for a in A3:
            PS[a][name] = {}
        fn = None
        for cond in CONDS:
            q = s007[(s007.arm == name) & (s007.condition == cond)].set_index("sha256").score
            assert set(shas) <= set(q.index)
            PS["orig"][name][cond] = {s: float(q[s]) for s in shas}
            f5 = R.sp.load_scores(f"{name}_fov512", cond)
            assert set(shas) <= set(f5), f"E1 fov512 cache incomplete for {name}/{cond}"
            PS["fov512"][name][cond] = {s: f5[s] for s in shas}
            sc = R.sp.load_scores(f"{name}_du2", cond)
            if len(sc) < len(shas):
                rp = CACHE / f"cpuref_{name}_du2.json"
                if not rp.exists() or any(c not in json.loads(rp.read_text()) for c in CONDS):
                    log(f"{name}: CPU reference ({len(sub)} du2 imgs x {len(CONDS)} conds)")
                    if subprocess.run([sys.executable, "-I", str(HERE / "run_du2.py"), "cpuref", name]).returncode != 0:
                        raise SystemExit(f"cpuref failed for {name}")
                ref = json.loads(rp.read_text())
                os.environ.pop("DD_DEVICE", None)
                fn = fn or R.sp.get_fn(name)
                todo = [s for s in sub if s not in sc]
                if todo:
                    sc.update(R.sp.score_rows(fn, dfi.loc[todo], cond)); R.sp.save_scores(f"{name}_du2", cond, sc)
                mx = float(np.abs([sc[s] - ref[cond][s] for s in sub]).max())
                guard[f"{name}/{cond}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL))
                (OUT / "d008_guard_partial.json").write_text(json.dumps(guard, indent=1))
                log(f"{name} {cond}: D-008 du2 max|dlogit|={mx:.2e} (tol {TOL})")
                if mx > TOL:
                    raise SystemExit(f"D-008 ABORT: {name}/{cond} max |dlogit| {mx:.3e} > {TOL}")
                rest = [s for s in shas if s not in sc]
                for i in range(0, len(rest), 100):
                    sc.update(R.sp.score_rows(fn, dfi.loc[rest[i:i + 100]], cond)); R.sp.save_scores(f"{name}_du2", cond, sc)
                    log(f"{name} {cond}: du2 {len(sc)}/{len(shas)}")
            else:   # fully cached from an earlier (guarded) run: re-read its guard result
                ref = json.loads((CACHE / f"cpuref_{name}_du2.json").read_text())
                mx = float(np.abs([sc[s] - ref[cond][s] for s in sub]).max())
                guard[f"{name}/{cond}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL), from_cache=True)
                if mx > TOL:
                    raise SystemExit(f"D-008 ABORT (cached): {name}/{cond} {mx:.3e}")
            PS["du2"][name][cond] = {s: sc[s] for s in shas}
    return PS, guard


# ----------------------------------------------------------------------------- analysis helpers
def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def boot_thr(cs, Wc, k):
    """Threshold 'flag iff s >= t' that leaves the k-th highest calibration score on the real side, for every bootstrap
    replicate (Wc: [B, n] multiplicities). k=1: calib max + eps; k=2: second-highest + eps."""
    o = np.argsort(-cs, kind="stable")
    cum = np.cumsum(Wc[:, o], 1)
    ok = cum[:, -1] >= k
    idx = np.where(ok, (cum >= k).argmax(1), 0)
    return np.nextafter(cs[o][idx], np.inf)


def point_thr(cs, k):
    return float(np.nextafter(np.sort(cs)[::-1][min(k, len(cs)) - 1], np.inf))


def analyze(df, B0r, X, LAP, PS, guard):
    n = len(df)
    sel = R.selection(df)
    lab, pool, split, gen = df.label.to_numpy(), df.pool.to_numpy(), df.split.to_numpy(), df.generator.to_numpy()
    plans = {nm: R.Plan(df, np.where(sel & (lab == 0) & (split == "test") & (pool == p))[0], sel) for p, nm in POOLS}
    sidx = np.where(sel)[0]

    def v(d):
        o = np.full(n, np.nan); o[sidx] = [d[s] for s in df.sha256.to_numpy()[sidx]]; return o

    S = {a: {c: {} for c in CONDS} for a in A3}
    for a in A3:
        for ci_, c in enumerate(CONDS):
            for d in DETS:
                vv = v(PS[a][d][c]); S[a][c][d] = (lambda g, vv=vv: vv)
            S[a][c]["b0"] = (lambda g, a=a, ci_=ci_: B0r[a][f"logo_{g}"][ci_])
            for tr in A3:
                S[a][c][f"b0_trained_{tr}"] = (lambda g, tr=tr, a=a, ci_=ci_: X[tr][a][f"logo_{g}"][ci_])
    # texture baseline: one sign per (input, cond) from the ORIG arm's train natives vs train fakes (as E1)
    tdir = {}
    for ci_, c in enumerate(CONDS):
        for ii, inp in enumerate(R.INPUTS):
            L0 = LAP["orig"][:, ci_, ii]
            trn = (split == "train") & (pool == "vision_native") & ~np.isnan(L0)
            trf = (split == "train") & (pool == "fake") & ~np.isnan(L0)
            au = float(metrics.weighted_auroc(L0[trn], L0[trf], np.ones((1, trn.sum())), np.ones((1, trf.sum())))[0])
            sg = 1 if au >= 0.5 else -1
            tdir[f"{inp}/{c}"] = dict(auroc_laplacian_train_orig=au, sign=sg)
            for a in A3:
                S[a][c]["texture_" + inp] = (lambda g, vv=sg * LAP[a][:, ci_, ii]: vv)
    PRE = R.INPUTS                                   # CommFor, RINE, ClipDet, B0 (retrained per arm)
    XB0 = [f"b0_trained_{t}" for t in A3]            # fixed probes: orig/fov512 pre-registered in Amendment 10, du2 extra
    TX = ["texture_" + i for i in R.INPUTS]
    names = PRE + XB0 + TX
    kind = {d: ("detector" if d in PRE else "fixed-probe B0 (Amendment 10)" if d in ("b0_trained_orig", "b0_trained_fov512")
                else "extra, not pre-registered") for d in names}
    e1 = json.loads(E1RES.read_text())["primary"]
    res = dict(n_boot=N_BOOT, seed=SEED, d008=guard, texture_direction=tdir, primary={}, per_pool={}, transfer={},
               plan={nm: dict(n_real=int(len(pl.R)), n_clusters=pl.n_clusters) for nm, pl in plans.items()})
    # ---- AUROC per (pool, arm, cond, name) with the bootstrap replicates
    A = {}
    for nm, pl in plans.items():
        for a in A3:
            for c in CONDS:
                for d in names:
                    A[(nm, a, c, d)] = pl.auroc(S[a][c][d])
    for c in CONDS:
        for d in names:
            for nm in plans:
                (po, fo, bo), (pd_, fd, bd), (pf, ff, bf) = (A[(nm, a, c, d)] for a in A3)
                dr, dz = fo - fd, fd - ff
                r_b, z_b = bo - bd, bd - bf
                D_b = np.abs(r_b) - np.abs(z_b)
                e = dict(kind=kind[d], auroc={"orig": fo, "du2": fd, "fov512": ff},
                         auroc_ci={"orig": ci(bo), "du2": ci(bd), "fov512": ci(bf)},
                         delta_resample=dr, delta_resample_ci=ci(r_b), delta_zoom=dz, delta_zoom_ci=ci(z_b),
                         delta_total=fo - ff, delta_total_ci=ci(bo - bf), absdiff=abs(dr) - abs(dz), absdiff_ci=ci(D_b),
                         per_generator={a: A[(nm, a, c, d)][0] for a in A3})
                assert abs((dr + dz) - (fo - ff)) < 1e-12
                if nm == "native":
                    flag = e1.get(c, {}).get(d, {}).get("scale_sensitive")
                    e["e1_scale_sensitive"] = flag
                    e1d = e1.get(c, {}).get(d, {}).get("delta")
                    if e1d is not None:
                        assert abs((fo - ff) - e1d) < 1e-9, f"E1 delta not reproduced for {d}/{c}: {fo - ff} vs {e1d}"
                    if flag:
                        lo, hi = e["absdiff_ci"]
                        e["label"] = "resampling-dominated" if lo > 0 else "zoom-dominated" if hi < 0 else "mixed"
                    else:
                        e["label"] = "no label (not scale-sensitive in E1)" if flag is not None else "no label (not in E1)"
                    res["primary"].setdefault(c, {})[d] = e
                res["per_pool"].setdefault(c, {}).setdefault(d, {})[nm] = e
    # ---- cross-scale 1%-FPR with device-bootstrap CIs (calib devices AND test devices AND fake images), two threshold rules
    calib = np.where(sel & (lab == 0) & (split == "calib") & (pool == "vision_native"))[0]
    Wc = metrics.bootstrap_weights(df.device.to_numpy()[calib], None, N_BOOT, SEED + 1)[0]
    res["transfer_meta"] = dict(n_calib_natives=int(len(calib)), n_calib_devices=int(len(set(df.device.to_numpy()[calib].tolist()))),
                                rules={"max": "calib max + eps (pre-registered in Amendment 7)", "second": "second-highest calib score + eps (critic's variant)"},
                                bootstrap="calib devices, test devices (reals of every pool share the same device draws) and test fake images (stratified per generator) resampled jointly; 95% percentile")
    pl0 = plans["native"]
    fk_n = {g: len(pl0.F[g]) for g in GENS}
    for c in CONDS:
        for d in DETS + ["b0"] + XB0:
            res["transfer"].setdefault(c, {})[d] = {}
            pairs = [(s_, t_) for s_ in A3 for t_ in A3 if (s_ == t_ if d == "b0" else True)]
            per_g = d.startswith("b0")
            for src, tgt in pairs:
                for rule, k in (("max", 1), ("second", 2)):
                    fpr_pt = {nm: [] for nm in plans}; tpr_pt = []; fpr_b = {nm: 0 for nm in plans}; tpr_b = 0; thr_pt = []
                    gl = GENS if per_g else [None]
                    for g in gl:
                        ss, st = np.asarray(S[src][c][d](g), float), np.asarray(S[tgt][c][d](g), float)
                        cs = ss[calib]
                        t0 = point_thr(cs, k); tb = boot_thr(cs, Wc, k); thr_pt.append(t0)
                        for nm, pl in plans.items():
                            fpr_pt[nm].append(float((st[pl.R] >= t0).mean()))
                            fpr_b[nm] = fpr_b[nm] + ((st[pl.R][None, :] >= tb[:, None]) * pl.Wr).sum(1) / pl.Wr.sum(1)
                        gg = GENS if g is None else [g]
                        for g2 in gg:
                            F = pl0.F[g2]; cl = pl0.cols[g2]
                            tpr_pt.append(float((st[F] >= t0).mean()))
                            tpr_b = tpr_b + ((st[F][None, :] >= tb[:, None]) * pl0.Wf[:, cl]).sum(1) / pl0.Wf[:, cl].sum(1)
                    ng = len(gl)
                    o = dict(threshold_mean=float(np.mean(thr_pt)), n_calib=int(len(calib)))
                    for nm in plans:
                        o[f"fpr_{nm}"] = dict(value=float(np.mean(fpr_pt[nm])), ci95=ci(fpr_b[nm] / ng))
                    o["tpr"] = dict(value=float(np.mean(tpr_pt)), ci95=ci(tpr_b / (len(GENS) if not per_g else len(GENS))))
                    res["transfer"][c][d][f"{rule}|thr:{src}->test:{tgt}" + ("(within)" if src == tgt else "(cross)")] = o
    return res, S


def f3(x):
    return f"{x:.3f}"


def tables(res):
    L = ["# E1b (Amendment 10): zoom vs resampling (du2 arm). PILOT, exploratory. Not a Phase 4 verdict.", "",
         f"Data: VISION test natives/WA/FBH ({res['plan']['native']['n_real']} reals, {res['plan']['native']['n_clusters']} devices each) + 6 Synthbuster generators x 60 test fakes. "
         f"Paired cluster bootstrap ({res['n_boot']} resamples, seed {res['seed']}): reals by device, fakes by image per generator. "
         "Delta_resample = orig - du2 (resampling at fixed zoom); Delta_zoom = du2 - fov512 (zoom at fixed resampling); sum = E1 delta. "
         "Label (only for rows E1 flagged scale-sensitive in that condition): resampling-dominated / zoom-dominated if the 95% CI of |Delta_resample| - |Delta_zoom| excludes 0 above/below, else mixed.", ""]
    for c in CONDS:
        L += [f"## Decomposition vs test natives, condition {c}", "",
              "| row | kind | AUROC orig / du2 / fov512 | Delta_resample [CI] | Delta_zoom [CI] | Delta_total [CI] | abs diff [CI] | label |",
              "|---|---|---|---|---|---|---|---|"]
        for d, e in res["primary"][c].items():
            t = lambda a, b: f"{a:+.3f} [{b[0]:+.3f},{b[1]:+.3f}]"
            L.append(f"| {d} | {e['kind']} | {e['auroc']['orig']:.3f} / {e['auroc']['du2']:.3f} / {e['auroc']['fov512']:.3f} | "
                     f"{t(e['delta_resample'], e['delta_resample_ci'])} | {t(e['delta_zoom'], e['delta_zoom_ci'])} | "
                     f"{t(e['delta_total'], e['delta_total_ci'])} | {t(e['absdiff'], e['absdiff_ci'])} | {e['label']} |")
        L.append("")
    L += ["## Per real pool: AUROC orig / du2 / fov512, Delta_resample, Delta_zoom (WA and FBH are the scale-matched pools; no RAISE in the pilot)", ""]
    for c in CONDS:
        L += [f"### {c}", "", "| row | pool | AUROC orig / du2 / fov512 | Delta_resample [CI] | Delta_zoom [CI] |", "|---|---|---|---|---|"]
        for d, ee in res["per_pool"][c].items():
            for p, e in ee.items():
                t = lambda a, b: f"{a:+.3f} [{b[0]:+.3f},{b[1]:+.3f}]"
                L.append(f"| {d} | {p} | {e['auroc']['orig']:.3f} / {e['auroc']['du2']:.3f} / {e['auroc']['fov512']:.3f} | "
                         f"{t(e['delta_resample'], e['delta_resample_ci'])} | {t(e['delta_zoom'], e['delta_zoom_ci'])} |")
        L.append("")
    m = res["transfer_meta"]
    L += ["## Cross-scale 1%-FPR threshold transfer (test FPR / TPR, 95% bootstrap CI over calib devices, test devices and fake images)", "",
          f"Threshold from {m['n_calib_natives']} calib natives ({m['n_calib_devices']} devices) of the source arm, applied to the target arm. Rules: max = calib max + eps (Amendment 7); "
          "second = second-highest + eps (critic's variant). B0 = mean over the 6 LOGO models; `b0` (retrained per arm) has within-arm rows only; "
          "`b0_trained_<arm>` = one fixed probe trained on <arm>, scored on every arm (orig/fov512 pre-registered in Amendment 10, du2 extra).", ""]
    cell = lambda x: f"{x['value']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for c in CONDS:
        for rule in ("max", "second"):
            L += [f"### {c}, threshold rule {rule}", "", "| detector | thr -> test | FPR native | FPR WA | FPR FBH | TPR |", "|---|---|---|---|---|---|"]
            for d, oo in res["transfer"][c].items():
                for k, o in oo.items():
                    if k.startswith(rule + "|"):
                        L.append(f"| {d} | {k.split('|')[1]} | {cell(o['fpr_native'])} | {cell(o['fpr_WA'])} | {cell(o['fpr_FBH'])} | {cell(o['tpr'])} |")
            L.append("")
    L += ["## Texture sign (set on the orig arm's train natives vs train fakes; used in all arms)", "", "| input/cond | AUROC(Lap) train orig | sign |", "|---|---|---|"]
    for k, e in res["texture_direction"].items():
        L.append(f"| {k} | {e['auroc_laplacian_train_orig']:.3f} | {e['sign']:+d} |")
    L += ["", "## D-008", "", "```", json.dumps(res["d008"], indent=1), "```"]
    return "\n".join(L) + "\n"


def write_long(df, S, B0r, X, PS, LAP):
    sel = R.selection(df); ix = np.where(sel)[0]; parts = []
    sh = df.sha256.to_numpy()[ix]
    for a in A3:
        for ci_, c in enumerate(CONDS):
            for ii, d in enumerate(R.INPUTS):
                lap = LAP[a][ix, ci_, ii]
                if d == "b0":
                    for tr in A3:
                        for m, arr in X[tr][a].items():
                            parts.append(pd.DataFrame(dict(detector="b0" if tr == a else f"b0_trained_{tr}", model=m, arm=a, condition=c,
                                                           sha256=sh, score=arr[ci_][ix], laplacian=lap)))
                else:
                    parts.append(pd.DataFrame(dict(detector=d, model="pretrained", arm=a, condition=c, sha256=sh,
                                                   score=S[a][c][d](None)[ix], laplacian=lap)))
    pd.concat(parts, ignore_index=True).to_csv(OUT / "scores_long.csv.gz", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    df = R.load_df("orig")
    log(f"E1b start; rows scored {int(R.selection(df).sum())} of {len(df)}")
    LAP = stage_lap()
    B0r, X = stage_b0()
    PS, g_pre = stage_score()
    feat_guard = json.loads((ROOT / "experiments/004-phase4-features/cache/pilot_du2/d008_guard.json").read_text())
    for c, e in feat_guard.items():
        assert e["ok"] and e["max_abs_dlogit"] <= TOL, f"004 feature guard failed for {c}"
    guard = dict(du2_pretrained_cuda_vs_cpu=g_pre, du2_features_004=feat_guard,
                 reused_without_rescoring="orig = 007 scores (E1 re-score check, pilot/d008_guard.json); fov512 = E1 scores (E1 guard, pilot/d008_guard.json)")
    assert all(e["ok"] for e in g_pre.values())
    res, S = analyze(df, B0r, X, LAP, PS, guard)
    res["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["runtime_s"] = time.time() - T0
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    (OUT / "tables.md").write_text(tables(res))
    (OUT / "d008_guard.json").write_text(json.dumps(guard, indent=1))
    (OUT / "d008_guard_partial.json").unlink(missing_ok=True)
    write_long(df, S, B0r, X, PS, LAP)
    log(f"done in {time.time() - T0:.0f}s -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "cpuref":
        cpuref(sys.argv[2])
    else:
        main()
