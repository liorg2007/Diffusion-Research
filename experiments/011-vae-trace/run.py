"""E4 PILOT (docs/04-phase4-plan.md Amendment 11): how detectable is the sd-vae-ft-mse decoder trace, and does it survive laundering?

  env/.venv/bin/python -I experiments/011-vae-trace/run.py --stage rt        # D-008 VAE guard (reused from 009), TEST round-trips (GPU), 009-cache checks
  env/.venv/bin/python -I experiments/011-vae-trace/run.py --stage feat      # DINOv2 features of all round-trips (004 pipeline) + F3 residual spectra + Laplacian
  env/.venv/bin/python -I experiments/011-vae-trace/run.py --stage analyze   # probes, pair AUROC, transfer, decision rules, probe D-008, tables
  (--stage all runs them in that order; every stage is resumable from cache/.  --dry: tiny subset, outputs in dry/)

Reuses 009's run.py (VAE loading, recon_f/q8, Batcher, save_rt, source_groups, Plan, joint cluster bootstrap helpers, lap224, fit) read-only;
009's TRAIN round-trips (cache/rt) are used after checking them against the Amendment 7-8 rules.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")      # 16 BLAS threads on a loaded box made sklearn fits 25x slower
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import metrics
from preprocess import center_crop, jpeg_quality_estimate

_spec = importlib.util.spec_from_file_location("e2run", ROOT / "experiments" / "009-vae-aligned" / "run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)
P = e2.P
p4run = e2.p4run

CFG = json.loads((HERE / "config.json").read_text())
E2_CACHE = e2.CACHE            # 009's cache: READ ONLY
E2_PILOT = e2.PILOT
VN = CFG["vae"]["name"]
SEED = CFG["seed"]
CONDS = CFG["conditions"]
CLEAN, LAUN = CONDS
GENS = e2.GENS
SD_FAM, OTHER_FAM = e2.SD_FAM, e2.OTHER_FAM
SUBDIR = e2.SUBDIR
ARMS = list(CFG["arms"])
FAMS = ["F1", "F2", "F3", "F4"]
TOL = 1e-3
N_BOOT = CFG["n_boot"]
DRY = False
PILOT = HERE / "pilot"
CACHE = HERE / "cache"
T_START = time.time()

# F3 constants (see config.json "families")
F3_SIZE = 224
F3_BINS = 32
F3_PEAK_K = (1, 2, 3, 4)
F3_NEIGH = (-4, -3, -2, 2, 3, 4)
F3_DIM = 3 * (F3_BINS + 2 * len(F3_PEAK_K))        # 120
EPS = 1e-10


def set_mode(dry):
    global DRY, PILOT, CACHE, N_BOOT
    DRY = dry
    base = HERE / "dry" if dry else HERE
    PILOT, CACHE = base / "pilot", base / "cache"
    PILOT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if dry:
        N_BOOT = 200
    e2.log = log           # 009's Batcher logs through its module global: keep it from writing to 009's run.log


def log(*a):
    s = time.strftime("%H:%M:%S") + " E4 " + " ".join(str(x) for x in a)
    print(s, flush=True)
    PILOT.mkdir(parents=True, exist_ok=True)
    with open(PILOT / "run.log", "a") as f:
        f.write(s + "\n")


def jdump_existing(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def atomic_npz(path, **kw):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- data
def load_df(arm):
    """Arm table. --dry keeps a small deterministic subset (plumbing test only; numbers are meaningless)."""
    df = e2.load_df(arm)
    if not DRY:
        return df
    keep = []
    for (sp, lab, pool), g in df.groupby(["split", "label", "pool"]):
        if lab == 0 and sp in ("train", "test"):
            keep.append(g.sort_values("sha256").head(16 if sp == "train" else 20))
        elif lab == 1:
            for _, gg in g.groupby("generator"):
                keep.append(gg.sort_values("sha256").head(2))
    return pd.concat(keep).sort_index().reset_index(drop=True)


def pair_rows(df):
    return df[(df.label == 0) & df.split.isin(["train", "test"])]


def get_qf(df):
    """Source JPEG quality factor per real (train + test). Cached in OUR cache; train values are compared with 009's qf.json (read-only)."""
    p = CACHE / "qf.json"
    qf = jdump_existing(p)
    for sha, sp in zip(pair_rows(df).sha256, pair_rows(df).src_path):
        if sha not in qf:
            with Image.open(ROOT / sp if not os.path.isabs(sp) else sp) as im:
                q = jpeg_quality_estimate(im)
            if q is None:
                raise SystemExit(f"{sp}: no JPEG quantization tables; cannot set the source quality factor")
            qf[sha] = q
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(qf))
    os.replace(tmp, p)
    return qf


def rt_ext(arm):
    return "jpg" if arm == "orig" else "png"


def rt_file(arm, sha):
    """Our own cache first, else 009's (train round-trips, read-only). Generation always writes to our cache."""
    own = CACHE / "rt" / arm / f"{sha}.{rt_ext(arm)}"
    if own.exists():
        return own
    old = E2_CACHE / "rt" / arm / VN / f"{sha}.{rt_ext(arm)}"
    return old if old.exists() else own


def rt_own_path(arm, sha):
    return CACHE / "rt" / arm / f"{sha}.{rt_ext(arm)}"


# ----------------------------------------------------------------------------- F3: residual spectrum
def f3_channel(x):
    """x: [224,224] float (0..255) -> 32 + 8 values (see config.json)."""
    n = x.shape[0]
    r = x - ndimage.median_filter(x, size=3, mode="reflect")
    Pw = np.abs(np.fft.fft2(r)) ** 2 / r.size              # Pw[fy_idx, fx_idx]
    fr = np.fft.fftfreq(n)
    R = np.hypot(fr[:, None], fr[None, :])
    ok = R <= 0.5 + 1e-12
    b = np.minimum((R[ok] / 0.5 * F3_BINS).astype(int), F3_BINS - 1)
    sums = np.bincount(b, weights=Pw[ok], minlength=F3_BINS)
    cnt = np.bincount(b, minlength=F3_BINS)
    spec = np.log10(sums / np.maximum(cnt, 1) + EPS)
    peaks = []
    for axis in ("x", "y"):
        line = Pw[0, :] if axis == "x" else Pw[:, 0]           # fy = 0 row (x periodicity) / fx = 0 column (y periodicity)
        for k in F3_PEAK_K:
            i = k * n // 8
            neigh = np.mean([line[(i + d) % n] for d in F3_NEIGH])
            peaks.append(np.log10(line[i] + EPS) - np.log10(neigh + EPS))
    return np.concatenate([spec, np.array(peaks)])


def f3_array(a):
    """a: [H,W,3] uint8/float image -> 120-d float32 (center 224 crop). Channels concatenated [R(40), G(40), B(40)]."""
    a = np.asarray(a, np.float32)
    h, w = a.shape[:2]
    t, l = (h - F3_SIZE) // 2, (w - F3_SIZE) // 2
    a = a[t:t + F3_SIZE, l:l + F3_SIZE]
    return np.concatenate([f3_channel(a[:, :, c]) for c in range(3)]).astype(np.float32)


def proc_image(im):
    """PIL -> [121]: 120 F3 values + the Laplacian texture score."""
    im = im.convert("RGB")
    return np.concatenate([f3_array(np.asarray(im)), [e2.lap224(im)]]).astype(np.float32)


def f3_peak_index(channel, axis, k):
    """Column index of the (axis, k) peak value inside one channel's 40-vector (used by the self-test)."""
    return F3_BINS + (0 if axis == "x" else len(F3_PEAK_K)) + F3_PEAK_K.index(k)


# ----------------------------------------------------------------------------- stage rt
def verify_vae_guard():
    """D-008 pixel guard for sd-vae-ft-mse: 009 measured it (CUDA vs CPU, 32 images, max |dpixel| 9.2e-5). Valid for 011 only if the
    weights revision and the round-trip function are the same (config strings compared; recon_f/q8/load_vae ARE 009's functions)."""
    c9 = json.loads((ROOT / "experiments" / "009-vae-aligned" / "config.json").read_text())
    v9 = c9["vaes"][VN]
    assert v9["revision"] == CFG["vae"]["revision"] and v9["repo"] == CFG["vae"]["repo"], "VAE revision differs from 009"
    assert e2.VAES[VN] == v9
    g9 = json.loads((E2_PILOT / "d008_guard.json").read_text())["vae"][VN]
    assert g9["ok"] and max(g9["max_abs_dpixel_x1"], g9["max_abs_dpixel_x2"]) <= TOL, g9
    g = jdump_existing(PILOT / "d008_guard.json")
    g["vae"] = {VN: dict(g9, reused_from="experiments/009-vae-aligned/pilot/d008_guard.json (same weights revision, same recon_f/q8; not re-run)",
                         revision=v9["revision"])}
    e2.jdump(PILOT / "d008_guard.json", g)
    log(f"D-008 VAE guard reused from 009: max|dpixel| x1={g9['max_abs_dpixel_x1']:.2e} x2={g9['max_abs_dpixel_x2']:.2e} (ok)")


def check_009_cache(dfs, qf, vae=None, dev=None, pool=None):
    """Amendment 7-8 rules for 009's TRAIN round-trips (orig: JPEG at the source QF; fov512: PNG) + exact-regeneration consistency."""
    out = {}
    for arm, df in dfs.items():
        tr = df[(df.label == 0) & (df.split == "train")]
        miss, wrong_fmt, wrong_qf, dsz = 0, 0, 0, 0
        for sha in tr.sha256:
            p = E2_CACHE / "rt" / arm / VN / f"{sha}.{rt_ext(arm)}"
            if not p.exists():
                miss += 1
                continue
            with Image.open(p) as im:
                if (im.format == "JPEG") != (arm == "orig") or (im.format == "PNG") != (arm == "fov512"):
                    wrong_fmt += 1
                if im.size != (512, 512):
                    dsz += 1
                if arm == "orig" and jpeg_quality_estimate(im) != qf[sha]:
                    wrong_qf += 1
        r = dict(n_train=len(tr), missing=miss, wrong_format=wrong_fmt, wrong_qf=wrong_qf, wrong_size=dsz)
        if vae is not None and not miss:
            rng = np.random.default_rng(SEED)
            sel = tr.sort_values("sha256").iloc[np.sort(rng.choice(len(tr), min(16, len(tr)), replace=False))]
            arr = e2.load_arr(sel, "clean", pool)
            y = e2.q8(e2.Batcher()(lambda a: e2.recon_f(vae, a), arr))
            mism, mx = [], 0
            for sha, im in zip(sel.sha256, y):
                if arm == "orig":
                    buf = io.BytesIO()
                    Image.fromarray(im).save(buf, "JPEG", quality=int(qf[sha]))
                    buf.seek(0)
                    ours = np.asarray(Image.open(buf).convert("RGB"))
                else:
                    ours = im
                old = np.asarray(Image.open(E2_CACHE / "rt" / arm / VN / f"{sha}.{rt_ext(arm)}").convert("RGB"))
                d = np.abs(ours.astype(int) - old.astype(int))
                mism.append(float((d > 0).mean()))
                mx = max(mx, int(d.max()))
            r.update(regen_n=len(sel), regen_mismatch_frac_mean=float(np.mean(mism)), regen_max_abs_diff_uint8=mx)
        out[arm] = r
        log(f"009 cache check {arm}: {r}")
        if r["missing"] or r["wrong_format"] or r["wrong_qf"] or r["wrong_size"]:
            log(f"WARNING {arm}: 009's train round-trips do not all match the rules; the mismatching files are regenerated into our cache where missing")
    return out


def stage_rt():
    import torch
    verify_vae_guard()
    dev = e2.get_dev()
    dfs = {a: load_df(a) for a in ARMS}
    qf = get_qf(dfs["orig"])
    q9 = json.loads((E2_CACHE / "qf.json").read_text())
    trn = dfs["orig"][(dfs["orig"].label == 0) & (dfs["orig"].split == "train")].sha256
    bad = [s for s in trn if s in q9 and q9[s] != qf[s]]
    assert not bad, f"QF mismatch with 009 for {len(bad)} train reals"
    vae = e2.load_vae(VN, dev)
    pool = ThreadPoolExecutor(4)
    chk = check_009_cache(dfs, qf, vae, dev, pool)
    e2.jdump(PILOT / "check_009_cache.json", chk)
    b = e2.Batcher()
    for arm, df in dfs.items():
        rows = pair_rows(df)
        (CACHE / "rt" / arm).mkdir(parents=True, exist_ok=True)
        todo = rows[[not rt_file(arm, s).exists() for s in rows.sha256]]
        t0 = time.time()
        for s in range(0, len(todo), 48):
            ch = todo.iloc[s:s + 48]
            arr = e2.load_arr(ch, "clean", pool)
            y = e2.q8(b(lambda a: e2.recon_f(vae, a), arr))
            for sha, im in zip(ch.sha256, y):
                e2.save_rt(rt_own_path(arm, sha), im, arm, qf[sha])
            if (s // 48) % 5 == 0:
                log(f"rt {arm}: {s + len(ch)}/{len(todo)}  {time.time() - t0:.0f}s")
        log(f"rt {arm}: {len(todo)} new round-trips ({len(rows) - len(todo)} cached/009), {time.time() - t0:.0f}s, batch {b.bs}")
    del vae
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage feat
def feat_cache(arm, cond, kind):
    return CACHE / "feats" / f"{kind}__{arm}__{cond}.npz"


def stage_feat():
    dev = e2.get_dev().type
    bb = p4run.secondary_backbone()
    p4run.check_design_vs_003(bb)
    for arm in ARMS:
        df = load_df(arm)
        rows = pair_rows(df)
        for cond in CONDS:
            # --- DINOv2 secondary features of every round-trip (train + test), 004 pipeline
            f = feat_cache(arm, cond, "dino_rt")
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                t0 = time.time()
                with ThreadPoolExecutor(4) as pool:
                    imgs = list(pool.map(lambda r: p4run.load_cond(rt_file(arm, r[0]), r[0], cond), [(s,) for s in rows.sha256]))
                X = [__import__("features").embed_images(bb, imgs[i:i + 240], 20, dev) for i in range(0, len(imgs), 240)]
                if getattr(bb, "force_cpu", False) and dev != "cpu":
                    raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
                atomic_npz(f, keys=rows.sha256.to_numpy().astype(str), feats=np.concatenate(X).astype(np.float32))
                log(f"dino_rt {arm}/{cond}: {len(imgs)} round-trips, {time.time() - t0:.0f}s")
            # --- F3 + Laplacian: plain images (all rows of the arm table: reals, fakes) and round-trips
            for kind, shas, pathf in (("plain", df.sha256.to_numpy(), lambda s, d=dict(zip(df.sha256, df.path)): d[s]),
                                      ("rt", rows.sha256.to_numpy(), lambda s, arm=arm: rt_file(arm, s))):
                f = feat_cache(arm, cond, "f3_" + kind)
                if f.exists():
                    continue
                t0 = time.time()
                with ThreadPoolExecutor(4) as pool:
                    V = list(pool.map(lambda s: proc_image(p4run.load_cond(pathf(s), s, cond)), shas))
                atomic_npz(f, keys=np.asarray(shas).astype(str), feats=np.stack(V))
                log(f"f3_{kind} {arm}/{cond}: {len(V)} images, {time.time() - t0:.0f}s")
    # consistency of our train-round-trip DINOv2 features with 009's (same files, same pipeline)
    chk = {}
    for arm in ARMS:
        df = load_df(arm)
        tr = df[(df.label == 0) & (df.split == "train")].sha256.tolist()
        for cond in CONDS:
            z9 = E2_CACHE / "feats" / arm / f"{cond}.npz"
            if not z9.exists():
                continue
            z9 = np.load(z9)
            pos9 = {k: i for i, k in enumerate(z9["keys"].tolist())}
            z = np.load(feat_cache(arm, cond, "dino_rt"))
            pos = {k: i for i, k in enumerate(z["keys"].tolist())}
            sel = [s for s in tr if f"{VN}:{s}" in pos9]
            if sel:
                d = np.abs(z["feats"][[pos[s] for s in sel]] - z9["feats"][[pos9[f"{VN}:{s}"] for s in sel]])
                chk[f"{arm}/{cond}"] = dict(n=len(sel), max_abs_diff_vs_009=float(d.max()))
    e2.jdump(PILOT / "check_009_feats.json", chk)
    log("feature consistency with 009 cache:", chk)


# ----------------------------------------------------------------------------- bootstrap / metrics
class PairPlan:
    """Cluster-bootstrap weights for pairs: pair i (real + own round-trip) has cluster c_i; both members get the same weight."""

    def __init__(self, clusters, n_boot=None, seed=SEED):
        n_boot = n_boot or N_BOOT
        Wr, _ = metrics.bootstrap_weights(np.asarray(clusters), None, n_boot, seed)
        self.W = np.vstack([np.ones((1, Wr.shape[1])), Wr])

    def stats(self, sr, st):
        """sr: real scores, st: own round-trip scores (high = round-trip). -> pooled AUROC and win-rate, with CIs and boot arrays."""
        sr, st = np.asarray(sr, float), np.asarray(st, float)
        au = metrics.weighted_auroc(sr, st, self.W, self.W)
        win = (self.W * ((st > sr) + 0.5 * (st == sr))).sum(1) / self.W.sum(1)
        res = {}
        for nm, v in (("auroc", au), ("win", win)):
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            res[nm] = dict(value=float(v[0]), ci_lo=float(lo), ci_hi=float(hi), boot=v[1:])
        return res


def point_auc(sr, sf):
    return e2.auc(sr, sf)


def rules(best, tex, key_tag=""):
    """Amendment 11 decision rules. best/tex: {cond: {value, ci_lo, ci_hi, boot}} for the best family and the texture baseline (pair AUROC).
    detectable: best clean >= 0.90 and CI lower > 0.85; survives: best laundered >= 0.75 and CI lower > 0.70;
    trace_specific: best - texture >= 0.10 (point; paired CI reported), at clean and at laundered; headline = clean."""
    out = {}
    b, bl = best[CLEAN], best[LAUN]
    out["detectable"] = dict(passed=bool(b["value"] >= 0.90 and b["ci_lo"] > 0.85), value=b["value"], ci_lo=b["ci_lo"], family=b.get("family"))
    out["survives_laundering"] = dict(passed=bool(bl["value"] >= 0.75 and bl["ci_lo"] > 0.70), value=bl["value"], ci_lo=bl["ci_lo"], family=bl.get("family"))
    ts = {}
    for c in CONDS:
        d = best[c]["boot"] - tex[c]["boot"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = best[c]["value"] - tex[c]["value"]
        ts[c] = dict(passed=bool(delta >= 0.10), delta=float(delta), ci_lo=float(lo), ci_hi=float(hi), family=best[c].get("family"))
    out["trace_specific"] = dict(headline_clean=ts[CLEAN]["passed"], by_condition=ts)
    out["deprioritize_vae_trace_methods"] = bool(not out["detectable"]["passed"] or not out["survives_laundering"]["passed"])
    return out


def strip(d):
    return {k: v for k, v in d.items() if k != "boot"}


# ----------------------------------------------------------------------------- stage analyze
def load_arm(arm):
    """Everything the analysis needs for one arm, aligned to the (sub)set arm table df."""
    df = load_df(arm)
    sub = SUBDIR[arm]
    n = len(df)
    pos = {s: i for i, s in enumerate(df.sha256)}
    D = {}
    for c in CONDS:
        D[("sec_plain", c)] = P.load_features_aligned(df, "secondary", c, subdir=sub)            # cached 004 features (reals + fakes)
        z = np.load(feat_cache(arm, c, "dino_rt"))
        m = np.full((n, z["feats"].shape[1]), np.nan, np.float32)
        for k, v in zip(z["keys"].tolist(), z["feats"]):
            if k in pos:
                m[pos[k]] = v
        D[("sec_rt", c)] = m
        for kind in ("plain", "rt"):
            z = np.load(feat_cache(arm, c, "f3_" + kind))
            m = np.full((n, F3_DIM + 1), np.nan, np.float32)
            for k, v in zip(z["keys"].tolist(), z["feats"]):
                if k in pos:
                    m[pos[k]] = v
            D[(f"f3_{kind}", c)] = m
    return df, D


def fam_matrix(fam, sec, f3):
    if fam == "F1":
        return np.ascontiguousarray(sec[:, -1536:])
    if fam == "F2":
        return sec
    if fam == "F3":
        return f3[:, :F3_DIM]
    if fam == "F4":
        return np.hstack([sec, f3[:, :F3_DIM]])
    raise ValueError(fam)


def X_of(D, fam, kind, c, idx):
    return fam_matrix(fam, D[("sec_" + kind, c)][idx], D[("f3_" + kind, c)][idx])


def texture_sign(D, tr, c):
    """+1 if round-trips have the higher Laplacian on train pairs, else -1: score = sign * lap (high = round-trip)."""
    a = point_auc(D[("f3_plain", c)][tr, F3_DIM], D[("f3_rt", c)][tr, F3_DIM])
    return (1 if a >= 0.5 else -1), float(a)


def analyze_arm(arm, models_out, long_rows):
    df, D = load_arm(arm)
    lab, gen, sp, pool, dev_, sha = (df[c].to_numpy() for c in ("label", "generator", "split", "pool", "device", "sha256"))
    grp = e2.source_groups(df)
    tr = np.where((lab == 0) & (sp == "train"))[0]
    te = np.where((lab == 0) & (sp == "test"))[0]
    te_nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
    fak = np.where(lab == 1)[0]
    fak = fak[np.argsort(gen[fak], kind="stable")]
    for c in CONDS:
        for kind in ("plain", "rt"):
            idx = np.r_[tr, te] if kind == "rt" else np.r_[tr, te, fak]
            assert not np.isnan(D[(f"f3_{kind}", c)][idx]).any(), (arm, c, kind, "f3")
            assert not np.isnan(D[("sec_" + kind, c)][idx]).any(), (arm, c, kind, "sec")
    pair_plan = PairPlan(dev_[te])
    tfplan = e2.Plan(dev_[te_nat], gen[fak], n_boot=N_BOOT)
    R = dict(n_train_pairs=len(tr), n_test_pairs=len(te), n_test_devices=int(len(set(dev_[te]))), n_test_natives=len(te_nat),
             n_fakes=len(fak), pair_auroc={}, pair_win={}, per_pool={}, transfer={}, texture={}, train_in_sample={})

    # ---------------- texture baseline (direction fixed on train pairs, per condition)
    tex = {}
    for c in CONDS:
        sgn, atr = texture_sign(D, tr, c)
        s_real, s_rt = sgn * D[("f3_plain", c)][te, F3_DIM], sgn * D[("f3_rt", c)][te, F3_DIM]
        st = pair_plan.stats(s_real, s_rt)
        tex[c] = st["auroc"]
        s_nat = sgn * D[("f3_plain", c)][te_nat, F3_DIM]
        s_fak = sgn * D[("f3_plain", c)][fak, F3_DIM]
        R["texture"][c] = dict(sign=sgn, train_pair_auroc_with_plus_sign=atr, pair_auroc=strip(st["auroc"]), pair_win=strip(st["win"]),
                               transfer_macro=strip(tfplan.macro(s_nat, s_fak)),
                               transfer_sd=strip(tfplan.macro(s_nat, s_fak, SD_FAM)), transfer_other=strip(tfplan.macro(s_nat, s_fak, OTHER_FAM)))
        for kind, ix in (("real", te), ("rt", te)):
            v = sgn * D[("f3_plain" if kind == "real" else "f3_rt", c)][ix, F3_DIM]
            long_rows += [(arm, "texture", "fixed", c, kind, "test", sha[i], "", dev_[i], pool[i], float(x)) for i, x in zip(ix, v)]
        long_rows += [(arm, "texture", "fixed", c, "fake", sp[i], sha[i], gen[i], "", "fake", float(x)) for i, x in zip(fak, s_fak)]

    # ---------------- probes
    pair_boot = {}          # protocol -> cond -> fam -> stats dict
    for proto in ("primary", "matched"):
        R["pair_auroc"][proto], R["pair_win"][proto], R["per_pool"][proto], R["transfer"][proto] = {}, {}, {}, {}
        pair_boot[proto] = {c: {} for c in CONDS}
        for fam in FAMS:
            for c in CONDS:
                train_c = CLEAN if proto == "primary" else c
                if proto == "matched" and c == CLEAN:
                    continue          # identical to primary at clean
                key = (arm, fam, train_c)
                if key not in models_out:
                    X = np.vstack([X_of(D, fam, "plain", train_c, tr), X_of(D, fam, "rt", train_c, tr)])
                    y = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
                    t0 = time.time()
                    models_out[key] = e2.fit(X, y, np.concatenate([grp[tr], grp[tr]]))
                    log(f"{arm}: fit {fam} train_cond={train_c} C={models_out[key][-1].C_[0]:.3g} ({time.time() - t0:.0f}s)")
                m = models_out[key]
                s_real = m.decision_function(X_of(D, fam, "plain", c, te))
                s_rt = m.decision_function(X_of(D, fam, "rt", c, te))
                st = pair_plan.stats(s_real, s_rt)
                pair_boot[proto][c][fam] = st["auroc"]
                R["pair_auroc"][proto].setdefault(c, {})[fam] = strip(st["auroc"])
                R["pair_win"][proto].setdefault(c, {})[fam] = strip(st["win"])
                R["per_pool"][proto].setdefault(c, {})[fam] = {p: point_auc(s_real[pool[te] == p], s_rt[pool[te] == p]) for p in sorted(set(pool[te]))}
                # transfer: pair-trained probe on test natives vs generator fakes
                s_nat = m.decision_function(X_of(D, fam, "plain", c, te_nat))
                s_fak = m.decision_function(X_of(D, fam, "plain", c, fak))
                tr_res = dict(all6=strip(tfplan.macro(s_nat, s_fak)), sd_family=strip(tfplan.macro(s_nat, s_fak, SD_FAM)),
                              other=strip(tfplan.macro(s_nat, s_fak, OTHER_FAM)), per_generator=tfplan.per_gen(s_nat, s_fak))
                R["transfer"][proto].setdefault(c, {})[fam] = tr_res
                if c == CLEAN or proto == "matched":
                    pass
                # in-sample sanity (train pairs)
                if proto == "primary" and c == train_c:
                    s0 = m.decision_function(X_of(D, fam, "plain", c, tr))
                    s1 = m.decision_function(X_of(D, fam, "rt", c, tr))
                    R["train_in_sample"].setdefault(c, {})[fam] = point_auc(s0, s1)
                if not (proto == "matched" and c == CLEAN):
                    tag = f"{proto}"
                    sr_all = m.decision_function(X_of(D, fam, "plain", c, np.r_[tr, te]))
                    srt_all = m.decision_function(X_of(D, fam, "rt", c, np.r_[tr, te]))
                    ii = np.r_[tr, te]
                    long_rows += [(arm, fam, tag, c, "real", sp[i], sha[i], "", dev_[i], pool[i], float(x)) for i, x in zip(ii, sr_all)]
                    long_rows += [(arm, fam, tag, c, "rt", sp[i], sha[i], "", dev_[i], pool[i], float(x)) for i, x in zip(ii, srt_all)]
                    long_rows += [(arm, fam, tag, c, "fake", sp[i], sha[i], gen[i], "", "fake", float(x)) for i, x in zip(fak, s_fak)]
        # matched at clean = primary
        if proto == "matched":
            for key in ("pair_auroc", "pair_win", "per_pool", "transfer"):
                R[key]["matched"][CLEAN] = R[key]["primary"][CLEAN]
            pair_boot["matched"][CLEAN] = pair_boot["primary"][CLEAN]

    # ---------------- best family + decision rules
    R["rules"] = {}
    for proto in ("primary", "matched"):
        best = {}
        for c in CONDS:
            f_best = max(FAMS, key=lambda f: pair_boot[proto][c][f]["value"])
            best[c] = dict(pair_boot[proto][c][f_best], family=f_best)
        R["rules"][proto] = rules(best, tex)
        R["rules"][proto]["best_family"] = {c: best[c]["family"] for c in CONDS}
    return R


def probe_guard(models, arms_data, bb):
    """D-008 for the new features: 32 test round-trips per arm, DINOv2 features on CPU vs the CUDA cache, max |dlogit| of every probe
    (F3/Laplacian are numpy on CPU for both sides, hence identical)."""
    out = dict(tol=TOL, items={}, ok=True)
    for arm in ARMS:
        df, D = arms_data[arm]
        te = np.where((df.label == 0).to_numpy() & (df.split == "test").to_numpy())[0]
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(te, min(32, len(te)), replace=False))
        for c in CONDS:
            imgs = [p4run.load_cond(rt_file(arm, df.sha256.iloc[i]), df.sha256.iloc[i], c) for i in sel]
            Fc = __import__("features").embed_images(bb, imgs, 4, "cpu")
            Fg = D[("sec_rt", c)][sel]
            f3 = D[("f3_rt", c)][sel]
            item = dict(n=len(sel), max_abs_feature_diff=float(np.abs(Fc - Fg).max()),
                        max_scaled_feature_diff=float((np.abs(Fc - Fg) / np.maximum(1.0, np.abs(Fc).max(0, keepdims=True))).max()))
            for fam in FAMS:
                for train_c in CONDS:
                    if (arm, fam, train_c) not in models:
                        continue
                    if train_c != CLEAN and c != train_c:
                        continue
                    m = models[(arm, fam, train_c)]
                    lg = [m.decision_function(fam_matrix(fam, F, f3)) for F in (Fg, Fc)]
                    item[f"max_abs_dlogit_{fam}_train_{train_c}"] = float(np.abs(lg[0] - lg[1]).max())
            dl = max(v for k, v in item.items() if k.startswith("max_abs_dlogit"))
            item["max_abs_dlogit"] = dl
            item["ok"] = bool(dl <= TOL)
            out["items"][f"{arm}/{c}"] = item
            out["ok"] &= item["ok"]
            log(f"D-008 probe {arm}/{c}: max|dlogit|={dl:.2e}, scaled dF={item['max_scaled_feature_diff']:.2e} -> {'ok' if item['ok'] else 'FAIL'}")
    if getattr(bb, "force_cpu", False):
        raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
    return out


def stage_analyze():
    g = jdump_existing(PILOT / "d008_guard.json")
    if not g.get("vae", {}).get(VN, {}).get("ok"):
        raise SystemExit("D-008 VAE guard missing/failed: run --stage rt first")
    models, long_rows = {}, []
    res = dict(label="PILOT", config=CFG, dry=DRY, arms={}, n_boot=N_BOOT)
    arms_data = {}
    for arm in ARMS:
        res["arms"][arm] = analyze_arm(arm, models, long_rows)
        arms_data[arm] = load_arm(arm)
    bb = p4run.secondary_backbone()
    pg = probe_guard(models, arms_data, bb)
    g["probe"] = pg
    e2.jdump(PILOT / "d008_guard.json", g)
    if not pg["ok"]:
        raise SystemExit("D-008 ABORT: probe |dlogit| > 1e-3 for new features; results.json NOT written")
    res["d008"] = g
    res["check_009_cache"] = jdump_existing(PILOT / "check_009_cache.json")
    res["check_009_feats"] = jdump_existing(PILOT / "check_009_feats.json")
    res["runtime_s_analyze_process"] = time.time() - T_START
    e2.jdump(PILOT / "results.json", res)
    pd.DataFrame(long_rows, columns=["arm", "family", "train_protocol", "cond", "kind", "split", "sha256", "generator", "device", "pool", "score"]
                 ).to_csv(PILOT / "scores_long.csv.gz", index=False)
    write_tables(res)
    log(f"analysis done in {time.time() - T_START:.0f}s")


def f3f(x):
    return "-" if x is None else f"{x:.3f}"


def cell(d):
    return f"{d['value']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def write_tables(res):
    L = []
    w = L.append
    w(f"# E4 PILOT: SD-VAE decoder-trace detectability{' (DRY RUN: tiny subset, meaningless numbers)' if res['dry'] else ''}\n")
    w("Pair AUROC = pooled AUROC of probe score for the test round-trips vs their own test reals (cluster bootstrap by device, " + str(res["n_boot"]) + " replicates; a pair shares its cluster). `primary` = probe fit on CLEAN train pairs; `matched` = probe fit on train pairs of the tested condition (sensitivity, not pre-registered).\n")
    for arm in ARMS:
        R = res["arms"][arm]
        w(f"\n## Arm `{arm}`{' (label arm)' if arm == 'orig' else ' (descriptive)'}\n")
        w(f"{R['n_train_pairs']} train pairs, {R['n_test_pairs']} test pairs, {R['n_test_devices']} test devices; transfer on {R['n_test_natives']} natives vs {R['n_fakes']} fakes.\n")
        for proto in ("primary", "matched"):
            w(f"\n### Pair AUROC, {proto} training\n")
            w("| family | clean | jpeg75_resize05 | win-rate clean | win-rate laundered | train in-sample clean |")
            w("|---|---|---|---|---|---|")
            for fam in FAMS:
                a, b = R["pair_auroc"][proto][CLEAN][fam], R["pair_auroc"][proto][LAUN][fam]
                wc, wl = R["pair_win"][proto][CLEAN][fam], R["pair_win"][proto][LAUN][fam]
                w(f"| {fam} | {cell(a)} | {cell(b)} | {wc['value']:.3f} | {wl['value']:.3f} | {R['train_in_sample'].get(CLEAN, {}).get(fam, float('nan')):.3f} |")
            t = R["texture"]
            w(f"| texture (Laplacian, sign {t[CLEAN]['sign']:+d}/{t[LAUN]['sign']:+d}) | {cell(t[CLEAN]['pair_auroc'])} | {cell(t[LAUN]['pair_auroc'])} | {t[CLEAN]['pair_win']['value']:.3f} | {t[LAUN]['pair_win']['value']:.3f} | - |")
            w("\nPer-pool pair AUROC (point):\n")
            w("| family | cond | " + " | ".join(sorted(R["per_pool"][proto][CLEAN]["F1"])) + " |")
            w("|---|---|" + "---|" * len(R["per_pool"][proto][CLEAN]["F1"]))
            for fam in FAMS:
                for c in CONDS:
                    pp = R["per_pool"][proto][c][fam]
                    w(f"| {fam} | {c} | " + " | ".join(f"{pp[k]:.3f}" for k in sorted(pp)) + " |")
        for proto in ("primary", "matched"):
            w(f"\n### Transfer ({proto} probe): test natives vs Synthbuster fakes, macro AUROC\n")
            w("| family | cond | all 6 | SD-family (SD2, SDXL) | other (DALL-E 2/3, Firefly, MJ5) |")
            w("|---|---|---|---|---|")
            for fam in FAMS:
                for c in CONDS:
                    t = R["transfer"][proto][c][fam]
                    w(f"| {fam} | {c} | {cell(t['all6'])} | {cell(t['sd_family'])} | {cell(t['other'])} |")
            for c in CONDS:
                t = R["texture"][c]
                w(f"| texture | {c} | {cell(t['transfer_macro'])} | {cell(t['transfer_sd'])} | {cell(t['transfer_other'])} |")
        for proto in ("primary", "matched"):
            r = R["rules"][proto]
            w(f"\n### Decision rules ({proto} training{', pre-registered reading' if proto == 'primary' else ', sensitivity'})\n")
            w(f"Best family per condition: {r['best_family']}\n")
            d, s, ts = r["detectable"], r["survives_laundering"], r["trace_specific"]
            w(f"- VAE trace detectable (best clean >= 0.90, CI lo > 0.85): **{'YES' if d['passed'] else 'NO'}** ({d['family']}: {d['value']:.3f}, CI lo {d['ci_lo']:.3f})")
            w(f"- Survives laundering (best laundered >= 0.75, CI lo > 0.70): **{'YES' if s['passed'] else 'NO'}** ({s['family']}: {s['value']:.3f}, CI lo {s['ci_lo']:.3f})")
            for c in CONDS:
                x = ts["by_condition"][c]
                w(f"- Trace-specific at {c} (best - texture >= 0.10): **{'YES' if x['passed'] else 'NO'}** (delta {x['delta']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}], {x['family']})")
            w(f"- Headline trace-specific (clean): **{'YES' if ts['headline_clean'] else 'NO'}**")
            w(f"- VAE-trace methods deprioritized in the Phase 5 design: **{'YES' if r['deprioritize_vae_trace_methods'] else 'NO'}**")
    w("\n## D-008\n```\n" + json.dumps(res["d008"], indent=1) + "\n```\n")
    w("\n## Checks of 009's cache\n```\n" + json.dumps(res["check_009_cache"], indent=1) + "\n" + json.dumps(res["check_009_feats"], indent=1) + "\n```\n")
    (PILOT / "tables.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rt", "feat", "analyze", "all"], required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    set_mode(a.dry)
    log(f"start stage={a.stage} dry={a.dry}")
    if a.stage in ("rt", "all"):
        stage_rt()
    if a.stage in ("feat", "all"):
        stage_feat()
    if a.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
