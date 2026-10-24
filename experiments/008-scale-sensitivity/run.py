"""E1 (Amendment 7): zoom (scale) sensitivity of CommFor / RINE / ClipDet / B0 on the pilot data. PILOT, exploratory.

  env/.venv/bin/python -I experiments/008-scale-sensitivity/run.py            # full run (resumable) -> pilot/
  env/.venv/bin/python -I experiments/008-scale-sensitivity/run.py --dry      # ~20 images, outputs in cache/dry_out/
  env/.venv/bin/python -I experiments/008-scale-sensitivity/run.py cpuref NAME [--dry]   # internal (CPU reference)

Stages: lap (Laplacian of each detector's actual input) -> b0 (005 A0, grouped CV + balanced, both arms) ->
score (orig reused from 007; fov512 new, D-008 guard) -> analyze. Criteria are those of Amendment 7, unchanged.
"""
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
sys.path.insert(0, str(ROOT / "experiments" / "007-worst-env-conformal"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta, spearmanr

import metrics
import score_pretrained as sp          # 007: load_cond / score_rows / get_fn / save_scores (module globals CACHE set below)

DRY = "--dry" in sys.argv
CACHE = HERE / "cache" / ("dry" if DRY else "full")
OUT = HERE / "cache" / "dry_out" if DRY else HERE / "pilot"
SPL = {"orig": ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv",
       "fov512": ROOT / "experiments/004-phase4-features/pilot/splits_pilot_fov512.csv"}
SCORES007 = ROOT / "experiments/007-worst-env-conformal/pilot/scores_pretrained.csv.gz"
CHECK007 = ROOT / "experiments/007-worst-env-conformal/cache/check_subset.json"
R005 = ROOT / "experiments/005-nuisance-projection/run.py"
V2 = ROOT / "experiments/005-nuisance-projection/pilot_v2"
SUBDIR = {"orig": "pilot", "fov512": "pilot_fov512"}
ARMS = ["orig", "fov512"]
CONDS = ["clean", "jpeg75_resize05"]
DETS = ["communityforensics", "rine", "clipdet"]
INPUTS = DETS + ["b0"]
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]
POOLS = [("vision_native", "native"), ("vision_whatsapp", "WA"), ("vision_facebook_hq", "FBH")]
SEED, N_BOOT, TOL = 0, 2000, 1e-3
T0 = time.time()
sp.CACHE = CACHE


def log(*a):
    print(time.strftime("%H:%M:%S"), "E1", *a, flush=True)


def load_df(arm):
    return pd.read_csv(SPL[arm], keep_default_na=False, dtype={"vidx": str})


def selection(df):
    """Row mask of rows that get a detector score: calib + test (dry: ~20 images)."""
    m = df.split.isin(["calib", "test"]).to_numpy()
    if not DRY:
        return m
    keep = np.zeros(len(df), bool)
    te = df.split.to_numpy() == "test"
    for pool, n in (("vision_native", 4), ("vision_whatsapp", 3), ("vision_facebook_hq", 3)):
        ix = np.where(te & (df.pool.to_numpy() == pool))[0]
        dev = df.device.to_numpy()
        keep[ix[np.unique(dev[ix], return_index=True)[1][:n]]] = True     # one image per device -> several clusters
    for g in GENS:
        keep[np.where(te & (df.generator.to_numpy() == g))[0][:1]] = True
    ca = np.where((df.split.to_numpy() == "calib") & (df.pool.to_numpy() == "vision_native"))[0][:2]
    keep[ca] = True
    return keep


def lap_rows(df):
    """Rows whose Laplacian is computed: the scored rows + the train rows that set the texture direction."""
    m = selection(df)
    tr = (df.split.to_numpy() == "train") & np.isin(df.pool.to_numpy(), ["vision_native", "fake"])
    if DRY:
        ix = np.where(tr & (df.pool.to_numpy() == "vision_native"))[0][:6].tolist() + \
             np.where(tr & (df.pool.to_numpy() == "fake"))[0][:12].tolist()
        tr = np.zeros(len(df), bool); tr[ix] = True
    return m | tr


def check_subset(df):
    if DRY:
        te = np.where(selection(df) & (df.split.to_numpy() == "test"))[0]
        r = [i for i in te if df.label[i] == 0][:4]; f = [i for i in te if df.label[i] == 1][:4]
        return df.sha256.iloc[r + f].tolist()
    return json.loads(CHECK007.read_text())


# ----------------------------------------------------------------------------- Laplacian of each detector's input
def _lap(img):
    x = np.asarray(img.convert("RGB"), np.float32) / 255.0
    g = x.mean(2)
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def _lap_one(a):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
    from preprocess import center_crop
    path, sha = a
    tfs = {"communityforensics": T.Compose([T.Resize(440), T.CenterCrop(384)]),
           "rine": T.CenterCrop(224),
           "clipdet": T.Compose([T.Resize(224, interpolation=InterpolationMode.BICUBIC), T.CenterCrop((224, 224))]),
           "b0": lambda im: center_crop(im, 224)}
    out = np.zeros((len(CONDS), len(INPUTS)), np.float64)
    for ci, c in enumerate(CONDS):
        im = sp.load_cond(path, sha, c)
        for ii, n in enumerate(INPUTS):
            out[ci, ii] = _lap(tfs[n](im))
    return out


def stage_lap():
    """-> {arm: array [n_rows, n_cond, n_input]} (NaN where not computed). Cached."""
    # the transforms above must equal the scorers' pre-normalisation parts
    import scorers.communityforensics as cf, scorers.clipdet as cd
    assert (cf.RESIZE, cf.CROP) == (440, 384)
    res = {}
    for arm in ARMS:
        p = CACHE / f"lap_{arm}.npy"
        df = load_df(arm)
        if p.exists():
            res[arm] = np.load(p); continue
        m = lap_rows(df)
        ix = np.where(m)[0]
        t = time.time()
        with mp.Pool(4) as pool:
            r = pool.map(_lap_one, [(df.path[i], df.sha256[i]) for i in ix], chunksize=16)
        L = np.full((len(df), len(CONDS), len(INPUTS)), np.nan)
        L[ix] = np.stack(r)
        np.save(p, L); res[arm] = L
        log(f"lap {arm}: {len(ix)} rows in {time.time() - t:.0f}s")
    return res


# ----------------------------------------------------------------------------- B0 = 005 arm A0 (LOGO), both arms
def load_r005():
    spec = importlib.util.spec_from_file_location("r005", str(R005))
    m = importlib.util.module_from_spec(spec)
    sys.modules["r005"] = m
    spec.loader.exec_module(m)
    return m


def stage_b0():
    """-> (B0, B0X, chk). B0[arm][model] = [2 conds, n_rows]: 005 A0 trained AND evaluated in the same arm (model in full,
    logo_<g>). B0X[train_arm][test_arm][model]: the probe trained on train_arm's features scored on test_arm's features
    (true cross-zoom deployment of one fixed probe). Dry run: B0 from the 005 pilot_v2 cache; B0X = B0 (no fitting)."""
    import pickle
    out, chk, feats, models, mus = {}, {}, {}, {}, {}
    for arm in ARMS:
        v2 = np.load(V2 / arm / "cache" / "scores_A0.npz")
        if DRY:
            out[arm] = {k: v2[k][:2] for k in v2.files}
            continue
        r = load_r005()
        d = CACHE / f"b0_{arm}"; d.mkdir(parents=True, exist_ok=True)
        r.OUT, r.CACHE, r.SPLITS, r.SUBDIR = d, d, SPL["orig"], SUBDIR[arm]
        r.GROUPED_CV, r.BALANCED, r.LABEL = True, True, "E1-B0"
        df, D = r.load_data(("primary",))
        nat, wa, fb = r.pair_indices(df)
        rtr = r.real_train_indices(df, nat)
        F = D["primary"]
        mu = F["clean"][rtr].mean(0).astype(np.float64)
        sd = np.maximum(F["clean"][rtr].std(0).astype(np.float64), 1e-6)
        feats[arm], mus[arm] = {c: F["ev"][c] for c in r.CONDS}, (mu, sd)
        Zev = {c: ((F["ev"][c] - mu) / sd).astype(np.float64) for c in r.CONDS}
        pk = d / "models_A0.pkl"
        if pk.exists() and (d / "scores_A0.npz").exists():
            models[arm] = pickle.loads(pk.read_bytes())
        else:
            cap, orig_fit = [], r.probe.fit_probe
            r.probe.fit_probe = lambda *a, **k: (cap.append(orig_fit(*a, **k)), cap[-1])[1]
            (d / "scores_A0.npz").unlink(missing_ok=True)
            try:
                r.run_arm("A0", df, lambda X: X, Zev, {}, aug=False)
            finally:
                r.probe.fit_probe = orig_fit
            assert len(cap) == len(r.MODELS)
            models[arm] = dict(zip(r.MODELS, cap))
            pk.write_bytes(pickle.dumps(models[arm]))
        S = np.load(d / "scores_A0.npz")
        out[arm] = {k: S[k][[r.CI[c] for c in CONDS]] for k in S.files}
        chk[arm] = float(max(np.abs(S[k] - v2[k]).max() for k in S.files))
        log(f"B0 {arm}: max|dscore| vs 005/pilot_v2 cache = {chk[arm]:.2e}")
        assert chk[arm] <= 1e-6, f"B0 {arm} is not 005's A0: max|dscore| {chk[arm]:.3e} vs 005/pilot_v2 cache (Amendment 7 requires B0 = 005 A0)"
    if DRY:
        return out, {tr: {te: out[te] if tr == te else out[tr] for te in ARMS} for tr in ARMS}, chk
    r_ci = [r.CI[c] for c in CONDS]
    X = {tr: {} for tr in ARMS}
    for tr in ARMS:
        mu, sd = mus[tr]
        for te in ARMS:
            if tr == te:
                X[tr][te] = out[tr]; continue
            Z = [((feats[te][c] - mu) / sd).astype(np.float64) for c in CONDS]
            X[tr][te] = {m: np.stack([mdl.decision_function(z) for z in Z]).astype(np.float32) for m, mdl in models[tr].items()}
    return out, X, chk


# ----------------------------------------------------------------------------- pretrained scores
def stage_score():
    """-> {arm: {det: {cond: {sha: score}}}}, guard dict."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    dfs = {a: load_df(a) for a in ARMS}
    assert (dfs["orig"].sha256.values == dfs["fov512"].sha256.values).all()
    sel = selection(dfs["orig"])
    shas = dfs["orig"].sha256[sel].tolist()
    dfo, dff = dfs["orig"].set_index("sha256", drop=False), dfs["fov512"].set_index("sha256", drop=False)
    sub = check_subset(dfs["orig"])
    s007 = pd.read_csv(SCORES007)
    guard_p = OUT / "d008_guard.json"
    guard = json.loads(guard_p.read_text()) if guard_p.exists() and not DRY else {}
    guard.setdefault("orig_recheck_vs_007", {}); guard.setdefault("fov512_cuda_vs_cpu", {})
    scores = {"orig": {}, "fov512": {}}
    for name in DETS:
        rp = CACHE / f"cpuref_{name}.json"
        if not rp.exists() or any(c not in json.loads(rp.read_text()) for c in CONDS):
            log(f"{name}: CPU reference ({len(sub)} fov512 imgs x {len(CONDS)} conds)")
            cmd = [sys.executable, "-I", str(HERE / "run.py"), "cpuref", name] + (["--dry"] if DRY else [])
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit(f"cpuref failed for {name}")
        ref = json.loads(rp.read_text())
        os.environ.pop("DD_DEVICE", None)
        fn = sp.get_fn(name)
        scores["orig"][name], scores["fov512"][name] = {}, {}
        for cond in CONDS:
            # ---- orig arm: stored 007 scores, verified by re-scoring the 32-image subset on this GPU
            q = s007[(s007.arm == name) & (s007.condition == cond)].set_index("sha256").score
            assert set(shas) <= set(q.index), f"007 scores do not cover the needed rows for {name}/{cond}"
            re = sp.score_rows(fn, dfo.loc[sub], cond)
            mx = float(np.abs([re[s] - q[s] for s in sub]).max())
            guard["orig_recheck_vs_007"][f"{name}/{cond}"] = dict(max_abs_diff=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL))
            log(f"{name} {cond}: orig re-score vs 007 stored: max|d|={mx:.2e}")
            if mx > TOL:
                raise SystemExit(f"ABORT: orig re-score differs from 007 stored scores by {mx:.3e} for {name}/{cond}")
            scores["orig"][name][cond] = {s: float(q[s]) for s in shas}
            # ---- fov512 arm: new scoring with the D-008 guard
            sc = sp.load_scores(f"{name}_fov512", cond)
            todo = [s for s in sub if s not in sc]
            if todo:
                sc.update(sp.score_rows(fn, dff.loc[todo], cond)); sp.save_scores(f"{name}_fov512", cond, sc)
            mx = float(np.abs([sc[s] - ref[cond][s] for s in sub]).max())
            guard["fov512_cuda_vs_cpu"][f"{name}/{cond}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL))
            guard_p.parent.mkdir(parents=True, exist_ok=True); guard_p.write_text(json.dumps(guard, indent=1))
            log(f"{name} {cond}: D-008 fov512 max|dlogit|={mx:.2e} (tol {TOL})")
            if mx > TOL:
                raise SystemExit(f"D-008 ABORT: {name}/{cond} max |dlogit| {mx:.3e} > {TOL}")
            rest = [s for s in shas if s not in sc]
            for i in range(0, len(rest), 100):
                sc.update(sp.score_rows(fn, dff.loc[rest[i:i + 100]], cond)); sp.save_scores(f"{name}_fov512", cond, sc)
                log(f"{name} {cond}: fov512 {len(sc)}/{len(shas) + len([s for s in sub if s not in shas])}")
            scores["fov512"][name][cond] = {s: sc[s] for s in shas}
    return scores, guard


def stage_cpuref(name):
    os.environ["DD_DEVICE"] = "cpu"
    df = load_df("fov512").set_index("sha256", drop=False)
    sub = check_subset(load_df("orig"))
    fn = sp.get_fn(name)
    p = CACHE / f"cpuref_{name}.json"
    ref = json.loads(p.read_text()) if p.exists() else {}
    for cond in CONDS:
        if cond in ref:
            continue
        t = time.time()
        ref[cond] = sp.score_rows(fn, df.loc[sub], cond, batch=4)
        p.write_text(json.dumps(ref))
        log(f"cpuref {name} {cond} done in {time.time() - t:.0f}s")


# ----------------------------------------------------------------------------- analysis
def cp_ci(k, n, a=0.05):
    k, n = int(round(k)), int(n)
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


class Plan:
    """Cluster-bootstrap plan for the test reals of one pool and the fakes of all 6 generators (shared by arms/detectors)."""

    def __init__(self, df, R, mask):
        gen = df.generator.to_numpy()
        self.R = R
        self.F = {g: np.where((gen == g) & (df.split.to_numpy() == "test") & mask)[0] for g in GENS}
        strat = np.concatenate([[g] * len(self.F[g]) for g in GENS])
        dev = df.device.to_numpy()
        self.Wr, self.Wf = metrics.bootstrap_weights(np.array([dev[i] for i in R]), strat, N_BOOT, SEED)
        self.cols, o = {}, 0
        for g in GENS:
            self.cols[g] = slice(o, o + len(self.F[g])); o += len(self.F[g])
        self.n_clusters = len(set(dev[R].tolist()))
        self.one_r, self.one_f = np.ones((1, len(R))), np.ones((1, o))

    def auroc(self, getS):
        """getS(g) -> full-length score vector for generator g. -> (point per g, macro point, boot macro [B], per-g boot)."""
        pt, boot = {}, {}
        for g in GENS:
            s = np.asarray(getS(g), float)
            pt[g] = float(metrics.weighted_auroc(s[self.R], s[self.F[g]], self.one_r, self.one_f[:, self.cols[g]])[0])
            boot[g] = metrics.weighted_auroc(s[self.R], s[self.F[g]], self.Wr, self.Wf[:, self.cols[g]])
        return pt, float(np.mean(list(pt.values()))), np.mean([boot[g] for g in GENS], 0)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def label_for(delta, lo, hi):
    return bool(abs(delta) >= 0.03 and (lo > 0 or hi < 0))


def analyze(df, B0, B0X, LAP, PS, guard, b0chk):
    """df: orig-order splits. B0: {arm: {model: [cond, n]}}; LAP: {arm: [n, cond, input]}; PS: pretrained scores."""
    n = len(df)
    sel = selection(df)
    lab, pool, split, gen = df.label.to_numpy(), df.pool.to_numpy(), df.split.to_numpy(), df.generator.to_numpy()
    plans = {}
    for p, nm in POOLS:
        R = np.where(sel & (lab == 0) & (split == "test") & (pool == p))[0]
        plans[nm] = Plan(df, R, sel)
    # score accessors: S[arm][cond][det] = function g -> vector [n]
    def vec(d):
        v = np.full(n, np.nan)
        ix = df.index[sel]
        v[ix] = [d[s] for s in df.sha256[sel]]
        return v
    S = {a: {c: {} for c in CONDS} for a in ARMS}
    for a in ARMS:
        for ci_, c in enumerate(CONDS):
            for d in DETS:
                v = vec(PS[a][d][c]); S[a][c][d] = (lambda g, v=v: v)
            S[a][c]["b0"] = (lambda g, a=a, ci_=ci_: B0[a][f"logo_{g}"][ci_])
            for tr in ARMS:   # one fixed probe (trained on arm `tr`) scored on arm a's features
                S[a][c][f"b0_trained_{tr}"] = (lambda g, tr=tr, a=a, ci_=ci_: B0X[tr][a][f"logo_{g}"][ci_])
    # texture baseline: ONE sign per (input, cond), set on the orig arm's (label arm) TRAIN natives vs train fakes and used in
    # both arms (so the texture 'delta' compares one fixed scorer). The per-arm train AUROCs are kept as descriptive output.
    tdir = {}
    for ci_, c in enumerate(CONDS):
        for ii, inp in enumerate(INPUTS):
            for a in ARMS:
                L = LAP[a][:, ci_, ii]
                trn = (split == "train") & (pool == "vision_native") & ~np.isnan(L)
                trf = (split == "train") & (pool == "fake") & ~np.isnan(L)
                au = float(metrics.weighted_auroc(L[trn], L[trf], np.ones((1, trn.sum())), np.ones((1, trf.sum())))[0])
                tdir[f"{inp}/{a}/{c}"] = dict(auroc_laplacian_train=au, n_real=int(trn.sum()), n_fake=int(trf.sum()))
            sgn = 1 if tdir[f"{inp}/orig/{c}"]["auroc_laplacian_train"] >= 0.5 else -1
            for a in ARMS:
                tdir[f"{inp}/{a}/{c}"]["sign"] = sgn
                tdir[f"{inp}/{a}/{c}"]["sign_set_on"] = "orig arm train"
                S[a][c]["texture_" + inp] = (lambda g, v=sgn * LAP[a][:, ci_, ii]: v)
    XB0 = [f"b0_trained_{a}" for a in ARMS]
    allnames = INPUTS + XB0 + ["texture_" + i for i in INPUTS]
    res = dict(n_boot=N_BOOT, seed=SEED, texture_direction=tdir, b0_vs_005_pilot_v2=b0chk, d008=guard,
               plan={nm: dict(n_real=int(len(pl.R)), n_clusters=pl.n_clusters,
                              n_fake={g: int(len(pl.F[g])) for g in GENS}) for nm, pl in plans.items()},
               primary={}, per_pool={}, transfer={}, spearman={})
    # ---- primary + per-pool AUROC
    A = {}
    for nm, pl in plans.items():
        for a in ARMS:
            for c in CONDS:
                for d in allnames:
                    A[(nm, a, c, d)] = pl.auroc(S[a][c][d])
    for c in CONDS:
        for d in allnames:
            for nm in plans:
                po, fo, bo = A[(nm, "orig", c, d)]; pf, ff, bf = A[(nm, "fov512", c, d)]
                dl, dbt = fo - ff, bo - bf
                lo, hi = ci(dbt)
                e = dict(pre_registered=bool(d in INPUTS), kind=("detector" if d in INPUTS else "texture baseline (descriptive)" if d.startswith("texture_") else "extra, not pre-registered"),
                         auroc_orig=fo, auroc_orig_ci=ci(bo), auroc_fov512=ff, auroc_fov512_ci=ci(bf), delta=dl,
                         delta_ci=[lo, hi], per_generator_orig=po, per_generator_fov512=pf)
                if nm == "native":
                    e["scale_sensitive"] = label_for(dl, lo, hi)
                    res["primary"].setdefault(c, {})[d] = e
                else:
                    e["scale_sensitive_rule"] = label_for(dl, lo, hi)
                res["per_pool"].setdefault(c, {}).setdefault(d, {})[nm] = e
    # ---- cross-zoom threshold transfer (detectors only)
    calib_nat = np.where(sel & (lab == 0) & (split == "calib") & (pool == "vision_native"))[0]
    calib_all = np.where(sel & (lab == 0) & (split == "calib"))[0]
    nrm = {nm: plans[nm].R for nm in plans}
    fk = np.concatenate([plans["native"].F[g] for g in GENS])
    for c in CONDS:
        for d in INPUTS + XB0:
            res["transfer"].setdefault(c, {})[d] = {}
            for cal_name, cal in (("calib_natives", calib_nat), ("calib_all_reals", calib_all)):
                for src in ARMS:
                    for tgt in ARMS:
                        if d == "b0" and src != tgt:    # two different retrained probes: not a meaningful pairing (b0_trained_* rows do it right)
                            continue
                        key = f"{cal_name}|thr:{src}->test:{tgt}" + ("(within)" if src == tgt else "(cross)")
                        fpr = {nm: [] for nm in nrm}; tpr = []; thrs = []
                        for g in GENS:     # B0 = LOGO model of g; pretrained: g only selects the fakes
                            ssrc, stgt = np.asarray(S[src][c][d](g), float), np.asarray(S[tgt][c][d](g), float)
                            t = metrics.threshold_for_fpr(ssrc[cal], 0.01)
                            thrs.append(float(t))
                            for nm, R in nrm.items():
                                fpr[nm].append(float((stgt[R] >= t).mean()))
                            tpr.append(float((stgt[plans["native"].F[g]] >= t).mean()))
                        o = dict(threshold_mean=float(np.mean(thrs)), n_calib=int(len(cal)),
                                 pre_registered=bool(cal_name == "calib_natives" and d in INPUTS))
                        for nm, R in nrm.items():
                            f = float(np.mean(fpr[nm])); k = f * len(R)
                            o[f"fpr_{nm}"] = dict(value=f, cp95=cp_ci(k, len(R)), n=int(len(R)), per_model=fpr[nm])
                        tp = float(np.mean(tpr)); nf = len(fk)
                        o["tpr"] = dict(value=tp, cp95=cp_ci(tp * nf, nf), n=int(nf), per_generator=dict(zip(GENS, tpr)))
                        res["transfer"][c][d][key] = o
    # ---- Spearman(score, Laplacian) on test reals (B0: 'full' model scores)
    for c_i, c in enumerate(CONDS):
        for ii, d in enumerate(INPUTS):
            res["spearman"].setdefault(c, {})[d] = {}
            for a in ARMS:
                v = B0[a]["full"][c_i] if d == "b0" else S[a][c][d](None)
                L = LAP[a][:, c_i, ii]
                e = {}
                for nm, R in list(nrm.items()) + [("all3", np.concatenate(list(nrm.values())))]:
                    e[nm] = float(spearmanr(v[R], L[R])[0])
                fa = np.concatenate([plans["native"].F[g] for g in GENS])
                e["fakes"] = float(spearmanr(v[fa], L[fa])[0])
                e["median_lap_native_reals"] = float(np.median(L[nrm["native"]]))
                e["median_lap_fakes"] = float(np.median(L[fa]))
                res["spearman"][c][d][a] = e
    return res, S


# ----------------------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def tables(res):
    L = ["# E1 (Amendment 7): zoom sensitivity of published detectors. PILOT, exploratory. Not a Phase 4 verdict.", "",
         f"Data: VISION test natives/WA/FBH ({res['plan']['native']['n_real']} reals, {res['plan']['native']['n_clusters']} devices each) + "
         f"6 Synthbuster generators x {res['plan']['native']['n_fake']['dalle2']} test fakes. Bootstrap: paired cluster, {res['n_boot']} resamples, seed {res['seed']}. "
         "Scale-sensitive = |delta| >= 0.03 and the 95% CI excludes 0.", ""]
    L += ["## Primary: macro-over-6-generators AUROC vs test natives, delta = orig - fov512", ""]
    for c in CONDS:
        L += [f"### {c}", "", "| detector | AUROC orig [95% CI] | AUROC fov512 [95% CI] | delta [95% CI] | scale-sensitive |", "|---|---|---|---|---|"]
        for d, e in res["primary"][c].items():
            L.append(f"| {d} | {f3(e['auroc_orig'])} [{f3(e['auroc_orig_ci'][0])},{f3(e['auroc_orig_ci'][1])}] | "
                     f"{f3(e['auroc_fov512'])} [{f3(e['auroc_fov512_ci'][0])},{f3(e['auroc_fov512_ci'][1])}] | "
                     f"{e['delta']:+.3f} [{e['delta_ci'][0]:+.3f},{e['delta_ci'][1]:+.3f}] | "
                     f"{'YES' if e['scale_sensitive'] else 'no'} ({'orig' if e['delta'] > 0 else 'fov512'} higher)"
                     f"{'' if e['pre_registered'] else ' [' + e['kind'] + ']'} |")
        L.append("")
    L += ["## Per-real-pool macro AUROC (orig / fov512 / delta [CI])", ""]
    for c in CONDS:
        L += [f"### {c}", "", "| detector | native | WA | FBH |", "|---|---|---|---|"]
        for d, e in res["per_pool"][c].items():
            L.append(f"| {d} | " + " | ".join(f"{f3(e[p]['auroc_orig'])} / {f3(e[p]['auroc_fov512'])} / {e[p]['delta']:+.3f} "
                                              f"[{e[p]['delta_ci'][0]:+.3f},{e[p]['delta_ci'][1]:+.3f}]" for p in ("native", "WA", "FBH")) + " |")
        L.append("")
    L += ["## Cross-zoom 1%-FPR threshold transfer (test FPR with Clopper-Pearson 95%; TPR on 360 test fakes)", "",
          "Threshold from calibration natives (43 reals: the calib max + eps, so nominal FPR is 0 on calib) of the source arm, applied to the target arm. "
          "B0 = mean over the 6 LOGO models (CP CI approximate). Within = source arm == target arm. For `b0` (retrained per arm) only within-arm rows are given; "
          "the rows `b0_trained_<arm>` (one fixed probe scored on the other arm's features) are an extra, NOT PRE-REGISTERED addition.", ""]
    for c in CONDS:
        for cal in ("calib_natives", "calib_all_reals"):
            L += [f"### {c}, thresholds from {cal}" + ("" if cal == "calib_natives" else " (NOT PRE-REGISTERED sensitivity variant)"), "",
                  "| detector | thr -> test | FPR native | FPR WA | FBH | TPR |", "|---|---|---|---|---|---|"]
            for d in INPUTS + [f"b0_trained_{a}" for a in ARMS]:
                for k, o in res["transfer"][c][d].items():
                    if not k.startswith(cal):
                        continue
                    cell = lambda x: f"{x['value']:.3f} [{x['cp95'][0]:.3f},{x['cp95'][1]:.3f}]"
                    L.append(f"| {d}{'' if d in INPUTS else ' (not pre-registered)'} | {k.split('|')[1]} | {cell(o['fpr_native'])} | {cell(o['fpr_WA'])} | {cell(o['fpr_FBH'])} | {cell(o['tpr'])} |")
            L.append("")
    L += ["## Spearman(score, mean|Laplacian| of the detector input) on test reals (B0: 'full' model)", "",
          "| cond | detector | arm | native | WA | FBH | all reals | fakes | median Lap native / fakes |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CONDS:
        for d in INPUTS:
            for a in ARMS:
                e = res["spearman"][c][d][a]
                L.append(f"| {c} | {d} | {a} | {e['native']:+.2f} | {e['WA']:+.2f} | {e['FBH']:+.2f} | {e['all3']:+.2f} | {e['fakes']:+.2f} | "
                         f"{e['median_lap_native_reals']:.4f} / {e['median_lap_fakes']:.4f} |")
    L += ["", "## Texture direction (ONE sign per input/condition, set on the orig arm's TRAIN natives vs train fakes, used in both arms; per-arm AUROC of the raw Laplacian shown for information)", "",
          "| input/arm/cond | AUROC(Lap) on train | sign used |", "|---|---|---|"]
    for k, e in res["texture_direction"].items():
        L.append(f"| {k} | {f3(e['auroc_laplacian_train'])} | {e['sign']:+d} |")
    L += ["", "## D-008", "", "```", json.dumps(res["d008"], indent=1), "```", "",
          f"B0 (fresh fit through 005/run.py code) vs 005/pilot_v2 cached A0 scores: max |d| = {res['b0_vs_005_pilot_v2']}"]
    return "\n".join(L) + "\n"


def write_long(df, S, B0, PS, LAP):
    sel = selection(df)
    parts = []
    for a in ARMS:
        for ci_, c in enumerate(CONDS):
            ix = np.where(sel)[0]
            for ii, d in enumerate(INPUTS):
                lap = LAP[a][ix, ci_, ii]
                if d == "b0":
                    for m in B0[a]:
                        parts.append(pd.DataFrame(dict(detector=d, model=m, arm=a, condition=c, sha256=df.sha256.to_numpy()[ix],
                                                       score=B0[a][m][ci_][ix], laplacian=lap)))
                else:
                    parts.append(pd.DataFrame(dict(detector=d, model="pretrained", arm=a, condition=c, sha256=df.sha256.to_numpy()[ix],
                                                   score=S[a][c][d](None)[ix], laplacian=lap)))
    pd.concat(parts, ignore_index=True).to_csv(OUT / "scores_long.csv.gz", index=False)


def main():
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    df = load_df("orig")
    log(f"{'DRY ' if DRY else ''}start; rows scored {int(selection(df).sum())} of {len(df)}")
    LAP = stage_lap()
    B0, B0X, b0chk = stage_b0()
    PS, guard = stage_score()
    res, S = analyze(df, B0, B0X, LAP, PS, guard, b0chk)
    res["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["dry"] = DRY
    res["runtime_s"] = time.time() - T0
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    (OUT / "tables.md").write_text(tables(res))
    (OUT / "d008_guard.json").write_text(json.dumps(guard, indent=1))
    write_long(df, S, B0, PS, LAP)
    log(f"done in {time.time() - T0:.0f}s -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "cpuref":
        stage_cpuref(sys.argv[2])
    else:
        main()
