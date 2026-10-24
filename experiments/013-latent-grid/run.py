"""H-D2 PILOT: latent-grid periodicity under rescaling (kb/ideas/latent-grid-periodicity-rescale.md + docs/04-phase4-plan.md Amendment 12).

  env/.venv/bin/python -I experiments/013-latent-grid/run.py --stage scan      # CPU: per-image scores for every condition (orig, fov512, JM control)
  env/.venv/bin/python -I experiments/013-latent-grid/run.py --stage e4        # GPU: E4 add-on at jr(75,0.7) (DINOv2 features, probes, D-008 guard)
  env/.venv/bin/python -I experiments/013-latent-grid/run.py --stage analyze   # statistics, pass/fail, tables.md, results.json, scores_long.csv.gz
  (--stage all = scan, e4, analyze; every stage is resumable from cache/.)  Definitions: config.json "score_definitions" (written before running).

Reuses (read-only) 011's run.py (E4 pair setup, F3, PairPlan) and, through it, 009's helpers (Plan, lap224, fit, source_groups).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import metrics
from perturb import apply_condition, jpeg, seed_from_key
from preprocess import jpeg_quality_estimate

CFG = json.loads((HERE / "config.json").read_text())
PILOT, CACHE = HERE / "pilot", HERE / "cache"
SEED, N_BOOT = CFG["seed"], CFG["n_boot"]
ARMS = list(CFG["arms"])
EPS = 1e-10
NEI = (-4, -3, -2, 2, 3, 4)
# condition name -> lib/perturb condition string
COND_STR = {"clean": "clean", "jpeg75": "jr(75,1.0)", "rs050": "resize(0.5)", "rs060": "resize(0.6)", "rs070": "resize(0.7)",
            "rs075": "resize(0.75)", "rs080": "resize(0.8)", "jr050": "jr(75,0.5)", "jr060": "jr(75,0.6)", "jr070": "jr(75,0.7)",
            "jr075": "jr(75,0.75)", "jr080": "jr(75,0.8)", "social_sim": "social_sim"}
CONDS = list(COND_STR)
S_OF = {"clean": 1.0, "jpeg75": 1.0, "rs050": .5, "rs060": .6, "rs070": .7, "rs075": .75, "rs080": .8,
        "jr050": .5, "jr060": .6, "jr070": .7, "jr075": .75, "jr080": .8}
FEATS = ["O", "S", "lap", "z8", "zs", "amax1", "Ox", "Oy"]
FI = {k: i for i, k in enumerate(FEATS)}
NONDYAD = ["jr060", "jr070", "jr080"]
DYAD = ["jr050", "jpeg75"]
SD_FAM = ["stable-diffusion-2", "stable-diffusion-xl"]
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]

_R = None


def R011():
    """Lazy import of 011's run.py (it imports 009 and 004 helpers); never calls its logging (that would write into 011/pilot)."""
    global _R
    if _R is None:
        spec = importlib.util.spec_from_file_location("run011", ROOT / "experiments" / "011-vae-trace" / "run.py")
        _R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_R)
    return _R


def log(*a):
    s = time.strftime("%H:%M:%S") + " H-D2 " + " ".join(str(x) for x in a)
    print(s, flush=True)
    PILOT.mkdir(exist_ok=True)
    with open(PILOT / "run.log", "a") as f:
        f.write(s + "\n")


def jdump(path, obj):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- scores
def logspec(p):
    """log10 power of the Hann-windowed rfft of a mean-removed 1-D profile."""
    p = np.asarray(p, np.float64)
    X = np.fft.rfft((p - p.mean()) * np.hanning(len(p)))
    return np.log10(np.abs(X) ** 2 + EPS)


def prom_curve(L):
    """prom[i] = L[i] - median(L[i+d], d in +-2..4); neighbours outside the array are ignored."""
    n = len(L)
    M = np.full((len(NEI), n), np.nan)
    for j, d in enumerate(NEI):
        lo, hi = max(0, -d), min(n, n - d)
        M[j, lo:hi] = L[lo + d:hi + d]
    return L - np.nanmedian(M, axis=0)


def peak_at(prom, n, f):
    """max prominence over the 3 bins around f*n (rfft bin spacing 1/n)."""
    i0 = int(round(f * n))
    lo, hi = max(i0 - 1, 0), min(i0 + 1, len(prom) - 1)
    return float(np.max(prom[lo:hi + 1])) if lo <= hi else float("nan")


def blind_mask(n, nbins):
    """Bins with 1/16 <= f <= 1/2.5, excluding bins within +-1 of k n/8 (k = 1..4)."""
    i = np.arange(nbins)
    f = i / n
    ok = (f >= 1 / 16 - 1e-12) & (f <= 1 / 2.5 + 1e-12)
    for k in range(1, 5):
        ok &= np.abs(i - k * n / 8) > 1.0
    return ok


def fold_z(r, p):
    """Folded-grid energy of a residual on a p x p phase grid (c4_fold.py): z = var(M) (N/p^2) / var(r); ~1 without periodic structure."""
    h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
    q = r[:h, :w]
    M = q.reshape(h // p, p, w // p, p).mean((0, 2))
    return float(M.var() * (q.size / p ** 2) / max(q.var(), 1e-9))


def image_feats(im, s_pre, in_w, lap_fn):
    """All scores of one conditioned image -> float vector in FEATS order. s_pre = resize history before the condition (1 in orig);
    s_tot = s_pre * output/input width (the oracle knows it)."""
    x = np.asarray(im.convert("L"), np.float32)
    assert x.shape[0] == x.shape[1], x.shape
    r = x - ndimage.median_filter(x, size=3, mode="reflect")
    n = x.shape[1]
    s_tot = s_pre * n / in_w
    px, py = r.mean(0), r.mean(1)
    prx, pry = prom_curve(logspec(px)), prom_curve(logspec(py))
    fstar = 1.0 / (8.0 * s_tot)
    ox, oy = peak_at(prx, n, fstar), peak_at(pry, n, fstar)
    comb = (prx + pry) / 2
    msk = blind_mask(n, len(comb))
    S = float(comb[msk].max())
    band = np.where((np.arange(len(comb)) / n >= 0.05) & (np.arange(len(comb)) / n <= 0.45))[0]
    amax1 = float(abs(band[np.argmax(comb[band])] - n / 8) <= 1.0)
    p_s = 8 * s_tot
    zs = fold_z(r, int(round(p_s))) if abs(p_s - round(p_s)) < 1e-6 and round(p_s) >= 2 else float("nan")
    return np.array([(ox + oy) / 2, S, lap_fn(im), fold_z(r, 8), zs, amax1, ox, oy], np.float64)


# ----------------------------------------------------------------------------- scan
_LAP = None


def _lap(im):
    global _LAP
    if _LAP is None:
        spec = importlib.util.spec_from_file_location("e2run", ROOT / "experiments" / "009-vae-aligned" / "run.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _LAP = m.lap224
    return _LAP(im)


def scan_job(args):
    path, sha, s_pre, jm_qf = args
    img = Image.open(path).convert("RGB")
    img.load()
    if jm_qf is not None:
        img = jpeg(img, jm_qf)
    out = np.full((len(CONDS), len(FEATS)), np.nan)
    for ci, c in enumerate(CONDS):
        im = apply_condition(img, COND_STR[c], key=sha, seed=0)
        out[ci] = image_feats(im, s_pre, img.size[0], _lap)
    return out


def load_rows(arm):
    df = R011().e2.load_df(arm)
    keep = ((df.label == 0) & (df.split == "test")) | (df.label == 1)
    return df[keep].reset_index(drop=True)


def native_qf(df_all):
    p = CACHE / "native_qf.json"
    qf = json.loads(p.read_text()) if p.exists() else {}
    nat = df_all[(df_all.pool == "vision_native") & (df_all.split == "test")]
    for sha, sp in zip(nat.sha256, nat.src_path):
        if sha not in qf:
            with Image.open(sp) as im:
                qf[sha] = jpeg_quality_estimate(im)
    jdump(p, qf)
    return qf


def s_pre_for(arm, rows):
    """orig: 1. fov512: fakes 512/short side of the source file; reals (no latent grid) get the s_pre of a seeded random SD-family fake."""
    if arm == "orig":
        return np.ones(len(rows))
    s = np.zeros(len(rows))
    fk = np.where(rows.label == 1)[0]
    for i in fk:
        with Image.open(rows.src_path.iloc[i]) as im:
            s[i] = 512.0 / min(im.size)
    sd = [i for i in fk if rows.generator.iloc[i] in SD_FAM]
    for i in np.where(rows.label == 0)[0]:
        rng = np.random.default_rng(seed_from_key(rows.sha256.iloc[i], 5))
        s[i] = s[sd[int(rng.integers(len(sd)))]]
    return s


def stage_scan(workers=12):
    CACHE.mkdir(exist_ok=True)
    (CACHE / "scan").mkdir(exist_ok=True)
    df_orig = R011().e2.load_df("orig")
    qf = native_qf(df_orig)
    nat_qfs = np.array([qf[s] for s in sorted(qf)])
    log(f"native QF (test natives) counts: {dict(zip(*np.unique(nat_qfs, return_counts=True)))}")
    for arm in ARMS:
        rows = load_rows(arm)
        s_pre = s_pre_for(arm, rows)
        for variant in (["main", "jm"] if arm == "orig" else ["main"]):
            f = CACHE / "scan" / f"{arm}__{variant}.npz"
            if f.exists():
                continue
            if variant == "main":
                sel = np.arange(len(rows))
                jm = [None] * len(rows)
            else:
                sel = np.where(rows.generator.isin(SD_FAM))[0]
                qkeys = sorted(qf)
                jm = []
                for i in sel:
                    rng = np.random.default_rng(seed_from_key(rows.sha256.iloc[i], 13))
                    jm.append(int(qf[qkeys[int(rng.integers(len(qkeys)))]]))
                assert all(88 <= q <= 100 for q in jm)
            args = [(rows.path.iloc[i], rows.sha256.iloc[i], float(s_pre[i]), q) for i, q in zip(sel, jm)]
            t0 = time.time()
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(scan_job, args, chunksize=8))
            np.savez(f, F=np.stack(res), sha=rows.sha256.to_numpy()[sel].astype(str), s_pre=s_pre[sel],
                     jm_qf=np.array([-1 if q is None else q for q in jm]))
            log(f"scan {arm}/{variant}: {len(args)} images x {len(CONDS)} conditions, {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- stats
def load_scan(arm, variant):
    z = np.load(CACHE / "scan" / f"{arm}__{variant}.npz")
    rows = load_rows(arm).set_index("sha256").loc[z["sha"]].reset_index()
    return rows, z["F"], z


def auc(sr, sf):
    return metrics.weighted_auroc(np.asarray(sr, float), np.asarray(sf, float), np.ones((1, len(sr))), np.ones((1, len(sf))))[0]


def cell(d, nd=3):
    return f"{d['value']:.{nd}f} [{d['ci_lo']:.{nd}f}, {d['ci_hi']:.{nd}f}]"


def strip(d):
    return {k: v for k, v in d.items() if k != "boot"}


class Eval:
    """Cluster-bootstrap evaluation of per-image scores: natives (test, reals by device) vs fakes (by image, stratified per generator)."""

    def __init__(self, rows, F, fake_rows_gen=None):
        self.rows = rows
        lab, pool = rows.label.to_numpy(), rows.pool.to_numpy()
        self.nat = np.where((lab == 0) & (pool == "vision_native"))[0]
        self.fak = np.where(lab == 1)[0]
        self.gen = rows.generator.to_numpy()[self.fak]
        self.F = F
        e2 = R011().e2
        self.plan = e2.Plan(rows.device.to_numpy()[self.nat], self.gen, gens=sorted(set(self.gen)), n_boot=N_BOOT, seed=SEED)

    def col(self, c, f):
        return self.F[:, CONDS.index(c), FI[f]]

    def macro(self, c, f, gens=None, sign=1.0):
        v = sign * self.col(c, f)
        return self.plan.macro(v[self.nat], v[self.fak], gens)

    def per_gen(self, c, f, sign=1.0):
        v = sign * self.col(c, f)
        return self.plan.per_gen(v[self.nat], v[self.fak])


def diffb(a, b):
    d = a["boot"] - b["boot"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(a["value"] - b["value"]), ci_lo=float(lo), ci_hi=float(hi))


def meanb(lst):
    boot = np.mean([x["boot"] for x in lst], 0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(value=float(np.mean([x["value"] for x in lst])), ci_lo=float(lo), ci_hi=float(hi), boot=boot)


def analyze_orig():
    rows, F, _ = load_scan("orig", "main")
    E = Eval(rows, F)
    R = dict(n_natives=len(E.nat), n_native_devices=int(len(set(rows.device.to_numpy()[E.nat]))), n_fakes=len(E.fak))
    ALLG, SDG = sorted(set(E.gen)), SD_FAM
    sd_cols = [i for i, g in enumerate(E.gen) if g in SD_FAM]

    # ---- validity gate (clean, no resize)
    O = E.col("clean", "O")
    thr = float(np.percentile(O[E.nat], 99))
    gate = dict(threshold_99pct_clean_natives=thr, definition="O prominence at 1/8 (+-1 bin) above the 99th percentile of clean test natives")
    for g in SDG + [x for x in ALLG if x not in SDG]:
        ix = E.fak[E.gen == g]
        gate[g] = dict(frac_peak=float((O[ix] > thr).mean()), frac_argmax_at_1_8=float(E.col("clean", "amax1")[ix].mean()), median_O=float(np.median(O[ix])))
    ixsd = E.fak[np.isin(E.gen, SDG)]
    pools = rows.pool.to_numpy()
    gate["SD_pooled"] = dict(frac_peak=float((O[ixsd] > thr).mean()), frac_argmax_at_1_8=float(E.col("clean", "amax1")[ixsd].mean()))
    gate["natives"] = dict(frac_peak=float((O[E.nat] > thr).mean()), frac_argmax_at_1_8=float(E.col("clean", "amax1")[E.nat].mean()), median_O=float(np.median(O[E.nat])))
    for p in ("vision_whatsapp", "vision_facebook_hq"):
        ix = np.where(pools == p)[0]
        gate[p] = dict(frac_peak=float((O[ix] > thr).mean()), frac_argmax_at_1_8=float(E.col("clean", "amax1")[ix].mean()))
    gate["passed"] = bool(gate["SD_pooled"]["frac_peak"] >= 0.60)
    gate["passed_secondary_reading"] = bool(gate["SD_pooled"]["frac_argmax_at_1_8"] >= 0.60)
    R["gate"] = gate

    # ---- main scores per condition
    tab = {}
    for c in CONDS:
        d = dict(s=S_OF.get(c))
        for f in ("O", "S", "z8", "zs"):
            if f == "zs" and np.isnan(E.col(c, f)).mean() > 0.5:
                continue
            for nm, gens in (("sd", SDG), ("all6", None)):
                d[f"{f}_{nm}"] = E.macro(c, f, gens)
        # texture (lap) in the more favourable direction
        for nm, gens in (("sd", SDG), ("all6", None)):
            a = E.macro(c, "lap", gens, +1.0)
            b = E.macro(c, "lap", gens, -1.0)
            d[f"tex_{nm}"] = dict(a if a["value"] >= b["value"] else b, sign=+1 if a["value"] >= b["value"] else -1)
        d["O_per_gen"] = E.per_gen(c, "O")
        d["S_per_gen"] = E.per_gen(c, "S")
        d["O_vs_tex_sd"] = diffb(d["O_sd"], d["tex_sd"])
        d["O_native_median"] = float(np.median(E.col(c, "O")[E.nat]))
        d["O_sdfake_median"] = float(np.median(E.col(c, "O")[ixsd]))
        tab[c] = d
    R["cond"] = tab

    # ---- JM control (SD-family fakes JPEG-saved at a paired native's QF before the condition)
    rj, Fj, zj = load_scan("orig", "jm")
    sd_main = E.fak[np.isin(E.gen, SDG)]
    assert (rj.sha256.to_numpy() == rows.sha256.to_numpy()[sd_main]).all()
    sdgen = rows.generator.to_numpy()[sd_main]
    # plan over the SAME sd fakes (main arrays restricted) so JM and main share replicates
    e2 = R011().e2
    plan_sd = e2.Plan(rows.device.to_numpy()[E.nat], sdgen, gens=SDG, n_boot=N_BOOT, seed=SEED)
    jm = {}
    for c in CONDS:
        for f in ("O", "S"):
            vm = F[:, CONDS.index(c), FI[f]]
            vj = Fj[:, CONDS.index(c), FI[f]]
            a = plan_sd.macro(vm[E.nat], vm[sd_main], SDG)
            b = plan_sd.macro(vm[E.nat], vj, SDG)
            jm.setdefault(c, {})[f] = dict(main=strip(a), jm=strip(b), delta=diffb(b, a))
        jm[c]["tex_jm"] = max((plan_sd.macro(sg * F[:, CONDS.index(c), FI["lap"]][E.nat], sg * Fj[:, CONDS.index(c), FI["lap"]], SDG) for sg in (1.0, -1.0)),
                              key=lambda d: d["value"])
        jm[c]["tex_jm"] = strip(jm[c]["tex_jm"])
    R["jm"] = jm
    R["jm_qf_counts"] = {int(k): int(v) for k, v in zip(*np.unique(zj["jm_qf"], return_counts=True))}

    # ---- pass / fail
    O_sd = {c: tab[c]["O_sd"] for c in CONDS}
    nd = [O_sd[c]["value"] for c in NONDYAD]
    gap = diffb(meanb([O_sd[c] for c in NONDYAD]), meanb([O_sd[c] for c in DYAD]))
    jm_ok = {c: abs(jm[c]["O"]["jm"]["value"] - jm[c]["O"]["main"]["value"]) <= 0.05 for c in NONDYAD}
    tex_ok = {c: tab[c]["O_vs_tex_sd"]["delta"] >= 0.10 for c in NONDYAD}
    Ssd, Sall = tab["social_sim"]["S_sd"]["value"], tab["social_sim"]["S_all6"]["value"]
    # FPR rule: S threshold = 99th pct of clean test natives; share of WA / FBH test copies above it
    S = E.col("clean", "S")
    thr_s = float(np.percentile(S[E.nat], 99))
    fpr = {"threshold_clean": thr_s}
    for p in ("vision_whatsapp", "vision_facebook_hq"):
        ix = np.where(pools == p)[0]
        fpr[p] = dict(clean=float((S[ix] > thr_s).mean()))
        Sg = E.col("social_sim", "S")
        thr_g = float(np.percentile(Sg[E.nat], 99))
        fpr[p]["social_sim_vs_clean_threshold"] = float((Sg[ix] > thr_s).mean())
        fpr[p]["social_sim_vs_social_threshold"] = float((Sg[ix] > thr_g).mean())
        fpr["threshold_social_sim"] = thr_g
    fpr["native_self_fpr_clean"] = float((S[E.nat] > thr_s).mean())
    fpr["max_clean_copy_fpr"] = max(fpr[p]["clean"] for p in ("vision_whatsapp", "vision_facebook_hq"))
    crit = dict(
        gate=gate["passed"],
        O_ge_0p80_at_nondyadic={c: O_sd[c]["value"] >= 0.80 for c in NONDYAD},
        O_le_0p65_at_s050=bool(O_sd["jr050"]["value"] <= 0.65),
        gap_ge_0p15=bool(gap["delta"] >= 0.15),
        jm_within_0p05=jm_ok, beats_texture_by_0p10=tex_ok,
        S_social_sd_ge_0p80=bool(Ssd >= 0.80), S_social_all6_ge_0p70=bool(Sall >= 0.70),
        fpr_le_2pct=bool(fpr["max_clean_copy_fpr"] <= 0.02))
    flat = [crit["gate"], *crit["O_ge_0p80_at_nondyadic"].values(), crit["O_le_0p65_at_s050"], crit["gap_ge_0p15"], *jm_ok.values(),
            *tex_ok.values(), crit["S_social_sd_ge_0p80"], crit["S_social_all6_ge_0p70"], crit["fpr_le_2pct"]]
    fail = bool(all(v < 0.70 for v in nd) or gap["delta"] < 0.05)
    if not gate["passed"]:
        label = "UNINFORMATIVE"
    elif fail:
        label = "FAIL"
    elif all(flat):
        label = "PASS"
    else:
        label = "MIXED"
    R["rule"] = dict(O_sd_nondyadic={c: O_sd[c]["value"] for c in NONDYAD}, O_sd_dyadic={c: O_sd[c]["value"] for c in DYAD},
                     O_sd_s075=O_sd["jr075"]["value"], gap_nondyadic_minus_dyadic=gap, S_social_sd=Ssd, S_social_all6=Sall, fpr=fpr,
                     criteria=crit, fail_rule_triggered=fail, label=label)
    # long scores
    long = []
    for i in range(len(rows)):
        for ci, c in enumerate(CONDS):
            long.append(("orig", "main", c, rows.sha256.iloc[i], rows.pool.iloc[i], rows.generator.iloc[i], rows.device.iloc[i], *F[i, ci, :6]))
    for i in range(len(rj)):
        for ci, c in enumerate(CONDS):
            long.append(("orig", "jm", c, rj.sha256.iloc[i], rj.pool.iloc[i], rj.generator.iloc[i], rj.device.iloc[i], *Fj[i, ci, :6]))
    return R, long


def analyze_fov():
    rows, F, z = load_scan("fov512", "main")
    E = Eval(rows, F)
    s_pre = z["s_pre"]
    out = dict(n_natives=len(E.nat), n_fakes=len(E.fak))
    # predicted collisions of the latent grid for SD-family fakes: f = 1/(8 s_tot) vs the nearest k/8
    sdi = np.where(rows.generator.isin(SD_FAM).to_numpy())[0]
    coll = {}
    for c, s_extra in (("clean", 1.0), ("jr050", 0.5), ("jr070", 0.7), ("jr075", .75)):
        st = s_pre[sdi] * s_extra
        nb = round(512 * s_extra)
        dist = np.min(np.abs(nb / (8 * st)[:, None] - np.arange(1, 5)[None, :] * nb / 8), axis=1)    # bins between f* n and the nearest k n/8
        coll[c] = dict(median_s_tot=float(np.median(st)), s_pre_range=[float(s_pre[sdi].min()), float(s_pre[sdi].max())],
                       share_within_1bin_of_k_over_8=float((dist <= 1.0).mean()))
    out["predicted_collisions_sd"] = coll
    tab = {}
    for c in CONDS:
        d = {}
        for f in ("O", "S", "z8"):
            d[f"{f}_sd"] = strip(E.macro(c, f, SD_FAM))
            d[f"{f}_all6"] = strip(E.macro(c, f, None))
        a = E.macro(c, "lap", SD_FAM, 1.0)
        b = E.macro(c, "lap", SD_FAM, -1.0)
        d["tex_sd"] = strip(a if a["value"] >= b["value"] else b)
        tab[c] = d
    out["cond"] = tab
    long = [("fov512", "main", c, rows.sha256.iloc[i], rows.pool.iloc[i], rows.generator.iloc[i], rows.device.iloc[i], *F[i, ci, :6])
            for i in range(len(rows)) for ci, c in enumerate(CONDS)]
    return out, long


# ----------------------------------------------------------------------------- E4 add-on
E4_COND = "jr(75,0.7)"


def e4_cache(arm, kind):
    return CACHE / "e4" / f"{kind}__{arm}.npz"


def stage_e4():
    import torch
    R = R011()
    e2, p4run = R.e2, R.p4run
    (CACHE / "e4").mkdir(exist_ok=True)
    dev = e2.get_dev().type
    bb = p4run.secondary_backbone()
    p4run.check_design_vs_003(bb)
    import features as FT
    out = dict(condition=E4_COND, arms={})
    probe_models = {}
    for arm in R.ARMS:
        df = e2.load_df(arm)
        pr = R.pair_rows(df)
        shas = pr.sha256.to_numpy()
        pathf = {"plain": dict(zip(df.sha256, df.path)), "rt": {s: R.rt_file(arm, s) for s in shas}}
        for kind in ("plain", "rt"):
            f = e4_cache(arm, kind)
            if f.exists():
                continue
            t0 = time.time()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(4) as pool:
                imgs = list(pool.map(lambda s: p4run.load_cond(pathf[kind][s], s, E4_COND), shas))
            X = [FT.embed_images(bb, imgs[i:i + 240], 20, dev) for i in range(0, len(imgs), 240)]
            if getattr(bb, "force_cpu", False) and dev != "cpu":
                raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
            with ThreadPoolExecutor(4) as pool:
                V = list(pool.map(R.proc_image, imgs))
            np.savez(f, keys=shas.astype(str), sec=np.concatenate(X).astype(np.float32), f3=np.stack(V))
            log(f"e4 features {arm}/{kind}: {len(imgs)} images, {time.time() - t0:.0f}s")
    # ---- probes + pair AUROC (011 setup)
    guard = dict(tol=1e-3, items={}, ok=True)
    for arm in R.ARMS:
        df, D = R.load_arm(arm)
        pos = {s: i for i, s in enumerate(df.sha256)}
        n = len(df)
        for kind in ("plain", "rt"):
            z = np.load(e4_cache(arm, kind))
            sec = np.full((n, z["sec"].shape[1]), np.nan, np.float32)
            f3 = np.full((n, R.F3_DIM + 1), np.nan, np.float32)
            for k, a, b in zip(z["keys"].tolist(), z["sec"], z["f3"]):
                sec[pos[k]], f3[pos[k]] = a, b
            D[("sec_" + kind, "new")], D[("f3_" + kind, "new")] = sec, f3
        lab, sp, dev_, pool_ = (df[c].to_numpy() for c in ("label", "split", "device", "pool"))
        grp = e2.source_groups(df)
        tr = np.where((lab == 0) & (sp == "train"))[0]
        te = np.where((lab == 0) & (sp == "test"))[0]
        plan = R.PairPlan(dev_[te])
        res = dict(n_train_pairs=len(tr), n_test_pairs=len(te), primary={}, matched={}, texture={}, win_primary={}, win_matched={}, per_pool_primary={})
        conds = [R.CLEAN, R.LAUN, "new"]
        for c in conds:
            assert not np.isnan(D[("f3_plain", c)][np.r_[tr, te]]).any() and not np.isnan(D[("sec_rt", c)][np.r_[tr, te]]).any(), (arm, c)
        for fam in R.FAMS:
            def fit_on(train_c):
                key = (arm, fam, train_c)
                if key not in probe_models:
                    X = np.vstack([R.X_of(D, fam, "plain", train_c, tr), R.X_of(D, fam, "rt", train_c, tr)])
                    y = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
                    probe_models[key] = e2.fit(X, y, np.concatenate([grp[tr], grp[tr]]))
                return probe_models[key]
            for proto in ("primary", "matched"):
                for c in conds:
                    if proto == "matched" and c == R.CLEAN:
                        continue
                    m = fit_on(R.CLEAN if proto == "primary" else c)
                    sr = m.decision_function(R.X_of(D, fam, "plain", c, te))
                    st = m.decision_function(R.X_of(D, fam, "rt", c, te))
                    s = plan.stats(sr, st)
                    cn = "jpeg75_resize07" if c == "new" else c
                    res[proto].setdefault(cn, {})[fam] = strip(s["auroc"])
                    (res["win_primary"] if proto == "primary" else res["win_matched"]).setdefault(cn, {})[fam] = strip(s["win"])
                    if proto == "primary":
                        res["per_pool_primary"].setdefault(cn, {})[fam] = {p: float(R.point_auc(sr[pool_[te] == p], st[pool_[te] == p])) for p in sorted(set(pool_[te]))}
        for c in conds:   # texture (direction fixed on train pairs)
            sgn, atr = R.texture_sign(D, tr, c)
            s = plan.stats(sgn * D[("f3_plain", c)][te, R.F3_DIM], sgn * D[("f3_rt", c)][te, R.F3_DIM])
            res["texture"]["jpeg75_resize07" if c == "new" else c] = dict(sign=sgn, pair_auroc=strip(s["auroc"]), win=strip(s["win"]))
        # D-008 guard on the new features: 32 test round-trips, CPU vs the CUDA cache
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(te, 32, replace=False))
        imgs = [p4run.load_cond(R.rt_file(arm, df.sha256.iloc[i]), df.sha256.iloc[i], E4_COND) for i in sel]
        Fc = FT.embed_images(bb, imgs, 4, "cpu")
        Fg = D[("sec_rt", "new")][sel]
        f3 = D[("f3_rt", "new")][sel]
        item = dict(n=len(sel), max_abs_feature_diff=float(np.abs(Fc - Fg).max()))
        dl = 0.0
        for (a, fam, train_c), m in probe_models.items():
            if a != arm or (train_c not in (R.CLEAN, "new")):
                continue
            lg = [m.decision_function(R.fam_matrix(fam, F_, f3)) for F_ in (Fg, Fc)]
            item[f"max_abs_dlogit_{fam}_train_{train_c}"] = float(np.abs(lg[0] - lg[1]).max())
            dl = max(dl, item[f"max_abs_dlogit_{fam}_train_{train_c}"])
        item["max_abs_dlogit"], item["ok"] = dl, bool(dl <= 1e-3)
        guard["items"][f"{arm}/jr(75,0.7)"] = item
        guard["ok"] &= item["ok"]
        log(f"D-008 e4 {arm}: max|dlogit|={dl:.2e} -> {'ok' if item['ok'] else 'FAIL'}")
        out["arms"][arm] = res
    jdump(PILOT / "d008_guard.json", guard)
    if not guard["ok"]:
        raise SystemExit("D-008 ABORT: E4 add-on probe |dlogit| > 1e-3")
    # reference values from 011 (same fits are re-done here; reproduction check)
    r11 = json.loads((ROOT / "experiments/011-vae-trace/pilot/results.json").read_text())
    rep = {}
    for arm in R.ARMS:
        for proto in ("primary", "matched"):
            for fam in R.FAMS:
                a = r11["arms"][arm]["pair_auroc"][proto][R.LAUN][fam]["value"]
                b = out["arms"][arm][proto][R.LAUN][fam]["value"]
                rep[f"{arm}/{proto}/{fam}/jpeg75_resize05"] = dict(e4_011=a, refit_here=b, abs_diff=abs(a - b))
    out["reproduction_vs_011"] = rep
    out["max_reproduction_diff"] = max(v["abs_diff"] for v in rep.values())
    log(f"e4 add-on: max |refit - 011| over F1-F4 / protocols / arms at jpeg75_resize05 = {out['max_reproduction_diff']:.2e}")
    jdump(PILOT / "e4_addon.json", out)


# ----------------------------------------------------------------------------- tables
def write_tables(res):
    L = []
    w = L.append
    ro, rf = res["orig"], res["fov512"]
    w("# H-D2 PILOT: latent-grid periodicity under rescaling\n")
    w(f"Label: **{ro['rule']['label']}**. Natives = {ro['n_natives']} test natives ({ro['n_native_devices']} devices); fakes = {ro['n_fakes']} (120 per generator). "
      "AUROC with fake = high, cluster bootstrap (reals by device, fakes by image per generator, 2000 replicates). SD-family = SD2 + SDXL (macro). Definitions: config.json.\n")
    g = ro["gate"]
    w("\n## Validity gate (clean orig crops, no resize)\n")
    w(f"Threshold = 99th percentile of O at 1/8 over clean test natives = {g['threshold_99pct_clean_natives']:.3f} (log10 units). Gate: >= 60% of SD2+SDXL fakes above it -> **{'PASS' if g['passed'] else 'FAIL (UNINFORMATIVE)'}**.\n")
    w("| group | share with peak at 1/8 (primary) | share with argmax at 1/8 (secondary) | median O |\n|---|---|---|---|")
    for k in SD_FAM + ["SD_pooled", "dalle2", "dalle3", "firefly", "midjourney-v5", "natives", "vision_whatsapp", "vision_facebook_hq"]:
        v = g[k]
        w(f"| {k} | {v['frac_peak']:.3f} | {v['frac_argmax_at_1_8']:.3f} | {v.get('median_O', float('nan')):.3f} |")
    w(f"\nSecondary reading passes: {g['passed_secondary_reading']}.\n")
    w("\n## Oracle-s score O and blind scan S, SD-family vs natives (AUROC)\n")
    w("| condition | s | O (SD) | S (SD) | texture (SD, best dir.) | O - texture | z8 (SD) | zs (SD) | O (all 6) | S (all 6) |\n|---|---|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        d = ro["cond"][c]
        zs = cell(d["zs_sd"]) if "zs_sd" in d else "-"
        dt = d["O_vs_tex_sd"]
        w(f"| {c} | {d['s'] if d['s'] else 'rand 0.7-1'} | {cell(d['O_sd'])} | {cell(d['S_sd'])} | {cell(d['tex_sd'])} | {dt['delta']:+.3f} [{dt['ci_lo']:+.3f}, {dt['ci_hi']:+.3f}] | {cell(d['z8_sd'])} | {zs} | {cell(d['O_all6'])} | {cell(d['S_all6'])} |")
    w("\n## O per generator (point AUROC vs natives)\n")
    w("| condition | " + " | ".join(GENS) + " |\n|---|" + "---|" * len(GENS))
    for c in CONDS:
        w(f"| {c} | " + " | ".join(f"{ro['cond'][c]['O_per_gen'][x]:.3f}" for x in GENS) + " |")
    w("\n## JPEG-history-matched control (SD-family fakes JPEG-saved at a paired native's QF before the condition)\n")
    w("| condition | O main | O JM | delta (JM - main) | S main | S JM | texture JM |\n|---|---|---|---|---|---|---|")
    for c in CONDS:
        j = ro["jm"][c]
        d = j["O"]["delta"]
        w(f"| {c} | {cell(j['O']['main'])} | {cell(j['O']['jm'])} | {d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | {j['S']['main']['value']:.3f} | {j['S']['jm']['value']:.3f} | {j['tex_jm']['value']:.3f} |")
    w(f"\nJM quality factors drawn: {ro['jm_qf_counts']}\n")
    r = ro["rule"]
    w("\n## Pass / fail\n")
    gp = r["gap_nondyadic_minus_dyadic"]
    w(f"- Gap mean O(jr060, jr070, jr080) - mean O(jr050, jpeg75): {gp['delta']:+.3f} [{gp['ci_lo']:+.3f}, {gp['ci_hi']:+.3f}] (pass >= 0.15, fail < 0.05)")
    w(f"- s = 0.75 (jr075, reported only): O = {r['O_sd_s075']:.3f}")
    f = r["fpr"]
    w(f"- FPR of S at the clean-natives 99th-percentile threshold ({f['threshold_clean']:.3f}): WhatsApp {f['vision_whatsapp']['clean']:.3f}, Facebook-HQ {f['vision_facebook_hq']['clean']:.3f} (clean copies; natives themselves {f['native_self_fpr_clean']:.3f}); "
      f"at social_sim vs that threshold: WA {f['vision_whatsapp']['social_sim_vs_clean_threshold']:.3f}, FBH {f['vision_facebook_hq']['social_sim_vs_clean_threshold']:.3f}; vs the social_sim-native threshold: WA {f['vision_whatsapp']['social_sim_vs_social_threshold']:.3f}, FBH {f['vision_facebook_hq']['social_sim_vs_social_threshold']:.3f}")
    w(f"- S at social_sim: SD-family {r['S_social_sd']:.3f}, all 6 {r['S_social_all6']:.3f} (pass >= 0.80 / >= 0.70)")
    w("\nCriteria:\n```\n" + json.dumps(r["criteria"], indent=1) + "\n```")
    w(f"\nFail rule triggered: {r['fail_rule_triggered']}. **Label: {r['label']}**\n")
    w("\n## fov512 (descriptive; fakes are resampled by 512/short side, predicted grid f = 1/(8 s_fov s_extra))\n")
    w("Predicted position of the SD-family latent grid vs the JPEG harmonics k/8:\n")
    w("| extra condition | median s_tot | s_fov range | share of SD fakes within 1 bin of k/8 |\n|---|---|---|---|")
    for c, v in rf["predicted_collisions_sd"].items():
        w(f"| {c} | {v['median_s_tot']:.3f} | {v['s_pre_range'][0]:.2f}-{v['s_pre_range'][1]:.2f} | {v['share_within_1bin_of_k_over_8']:.3f} |")
    w("\n| condition | O (SD) | S (SD) | S (all 6) | texture (SD) | z8 (SD) |\n|---|---|---|---|---|---|")
    for c in CONDS:
        d = rf["cond"][c]
        w(f"| {c} | {cell(d['O_sd'])} | {cell(d['S_sd'])} | {cell(d['S_all6'])} | {cell(d['tex_sd'])} | {cell(d['z8_sd'])} |")
    e = res.get("e4_addon")
    if e:
        w("\n## E4 add-on: pair AUROC at jpeg75_resize07 (jr(75,0.7)) vs 0.5x (011 pair setup, refit here)\n")
        w(f"Reproduction check: max |refit - 011 results.json| at jpeg75_resize05 = {e['max_reproduction_diff']:.2e}. D-008: see d008_guard.json.\n")
        for arm in ("orig", "fov512"):
            a = e["arms"][arm]
            w(f"\n### Arm `{arm}`{' (label arm)' if arm == 'orig' else ''}\n")
            for proto in ("primary", "matched"):
                w(f"\n{proto} training\n")
                w("| family | clean | jpeg75_resize05 | jpeg75_resize07 | win-rate 0.5x | win-rate 0.7x |\n|---|---|---|---|---|---|")
                wk = a["win_primary" if proto == "primary" else "win_matched"]
                for fam in ("F1", "F2", "F3", "F4"):
                    w(f"| {fam} | {cell(a['primary']['clean'][fam])} | {cell(a[proto]['jpeg75_resize05'][fam])} | {cell(a[proto]['jpeg75_resize07'][fam])} | {wk['jpeg75_resize05'][fam]['value']:.3f} | {wk['jpeg75_resize07'][fam]['value']:.3f} |")
                t = a["texture"]
                w(f"| texture | {cell(t['clean']['pair_auroc'])} | {cell(t['jpeg75_resize05']['pair_auroc'])} | {cell(t['jpeg75_resize07']['pair_auroc'])} | {t['jpeg75_resize05']['win']['value']:.3f} | {t['jpeg75_resize07']['win']['value']:.3f} |")
    (PILOT / "tables.md").write_text("\n".join(L) + "\n")


def stage_analyze():
    t0 = time.time()
    ro, long_o = analyze_orig()
    rf, long_f = analyze_fov()
    res = dict(label=ro["rule"]["label"], pilot=True, config=CFG, orig=ro, fov512=rf, n_boot=N_BOOT)
    f = PILOT / "e4_addon.json"
    if f.exists():
        res["e4_addon"] = json.loads(f.read_text())
    g = PILOT / "d008_guard.json"
    if g.exists():
        res["d008"] = json.loads(g.read_text())
    res["runtime_s_analyze"] = time.time() - t0
    jdump(PILOT / "results.json", res)
    pd.DataFrame(long_o + long_f, columns=["arm", "variant", "cond", "sha256", "pool", "generator", "device"] + FEATS[:6]
                 ).to_csv(PILOT / "scores_long.csv.gz", index=False)
    write_tables(res)
    log(f"analysis done in {time.time() - t0:.0f}s; label {res['label']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "e4", "analyze", "all"], required=True)
    a = ap.parse_args()
    log(f"start stage={a.stage}")
    if a.stage in ("scan", "all"):
        stage_scan()
    if a.stage in ("e4", "all"):
        stage_e4()
    if a.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
