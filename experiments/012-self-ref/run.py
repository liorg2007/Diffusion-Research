"""H-D1 PILOT (kb/ideas/self-ref-trace-contrast.md + docs/04-phase4-plan.md Amendment 12): self-referenced decoder-trace contrast.

  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage feat      # VAE(t') + DINOv2 features of t', VAE(t'), L0 variants (GPU)
  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage vv        # VAE(VAE(x)) features (clean; train reals + test natives)
  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage probes    # w (E4 primary / matched F2, refit + checked), A2, A2L
  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage blur      # matched-blur sigma on train reals, blur features on test
  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage guard     # D-008: GPU vs CPU features of the stored images, |dscore| <= 1e-3
  env/.venv/bin/python -I experiments/012-self-ref/run.py --stage analyze   # scores, AUROC, gates, FPR, blur control, label, tables
  (--stage all runs them in this order; every stage is resumable from cache/.  --dry: tiny subset, outputs in dry/)

Reuses (read-only, imported): 011's run.py (load_arm / fam_matrix / X_of, which load its cached DINOv2 features of the 297 train
round-trips), 009's run.py (VAE loading, recon_f, q8, Batcher, Plan, diff, fit, lap224, source_groups), 004's pipeline (p4run.load_cond,
secondary_backbone), p4features, lib/perturb, lib/metrics.  Nothing is written outside this folder.
"""
import argparse
import importlib.util
import json
import os
import pickle
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
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.stats import spearmanr

import metrics
import perturb

_spec = importlib.util.spec_from_file_location("e4run", ROOT / "experiments" / "011-vae-trace" / "run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
R.set_mode(False)            # 011's own pilot/ and cache/ exist already; this only sets globals (we never write through R)
e2, p4run, P = R.e2, R.p4run, R.P

CFG = json.loads((HERE / "config.json").read_text())
VN = CFG["vae"]["name"]
SEED = CFG["seed"]
N_BOOT = CFG["n_boot"]
CONDS = CFG["conditions"]
CLEAN, LAUN, SOCIAL = CONDS
ARMS = list(CFG["arms"])
LABEL_ARM = CFG["label_arm"]
GENS, SD_FAM, OTHER_FAM = e2.GENS, e2.SD_FAM, e2.OTHER_FAM
POOLS = ["vision_native", "vision_whatsapp", "vision_facebook_hq"]
INPUT = P.INPUT_SIZE
TOL = 1e-3
CHUNK = 96
BLUR_GRID = CFG["blur_control"]["grid"]
DRY = False
PILOT = HERE / "pilot"
CACHE = HERE / "cache"
SHARED = HERE / "cache" / "shared"        # w probes: do not depend on the dry subset
T_START = time.time()


def set_mode(dry):
    global DRY, PILOT, CACHE, N_BOOT
    DRY = dry
    base = HERE / "dry" if dry else HERE
    PILOT, CACHE = base / "pilot", base / "cache"
    for d in (PILOT, CACHE, SHARED):
        d.mkdir(parents=True, exist_ok=True)
    if dry:
        N_BOOT = 200
        (HERE / "dry" / ".gitignore").write_text("*\n!.gitignore\n")


def log(*a):
    s = time.strftime("%H:%M:%S") + " HD1 " + " ".join(str(x) for x in a)
    print(s, flush=True)
    PILOT.mkdir(parents=True, exist_ok=True)
    with open(PILOT / "run.log", "a") as f:
        f.write(s + "\n")


R.log = log
e2.log = log             # 009's Batcher logs through its module global


def jdump(p, o):
    e2.jdump(p, o)


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def atomic_npz(path, **kw):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def stage_time(name, t0):
    p = CACHE / "stage_times.json"
    d = jload(p)
    d[name] = d.get(name, 0.0) + (time.time() - t0)
    jdump(p, d)


# ============================================================================= pure helpers (unit-tested in selftest.py)
def crop8(img):
    """Center-crop to a multiple of 8 on each side (the VAE stride); identity for 512 and 256."""
    w, h = img.size
    nw, nh = w // 8 * 8, h // 8 * 8
    if (nw, nh) == (w, h):
        return img
    l, t = (w - nw) // 2, (h - nh) // 2
    return img.crop((l, t, l + nw, t + nh))


def l0(img, rule="min"):
    """Canonical laundering L0 = jpeg75_resize05 (perturb). rule 'min': upsample back to the pre-L0 size only if the result is shorter
    than the 224 DINOv2 crop; rule 'uniform': always upsample back (Amendment 9 canonicalization)."""
    s = perturb.jpeg75_resize05(img.convert("RGB"))
    if rule == "uniform" or min(s.size) < INPUT:
        s = s.resize(img.size, Image.BICUBIC)
    return s


def blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(float(sigma)))


def match_sigma(grid, means, target):
    """sigma whose mean shift equals `target`: log-linear interpolation at the first sign change of (means - target) scanning from small
    sigma. -> (sigma, status) with status 'ok' | 'out_of_range_high' (blur never reaches the target: largest sigma) | 'out_of_range_low'."""
    g, m = np.asarray(grid, float), np.asarray(means, float)
    d = m - target
    for i in range(len(g)):
        if d[i] == 0:
            return float(g[i]), "ok"
        if i + 1 < len(g) and d[i] * d[i + 1] < 0:
            a = d[i] / (d[i] - d[i + 1])
            return float(np.exp(np.log(g[i]) + a * (np.log(g[i + 1]) - np.log(g[i])))), "ok"
    return (float(g[-1]), "out_of_range_high") if np.all(d < 0) else (float(g[0]), "out_of_range_low")


def device_halves(devices, seed=SEED):
    """Balanced seeded split of the distinct devices into two halves -> (set A, set B)."""
    u = sorted(set(devices))
    perm = np.random.default_rng(seed).permutation(len(u))
    k = len(u) // 2
    return {u[i] for i in perm[:k]}, {u[i] for i in perm[k:]}


def crossfit_fpr(score, dev, pool, label, fpr=0.01, seed=SEED):
    """Device cross-fit. score: fake-high scores of the TEST rows (reals of all pools + fakes), dev/pool/label aligned.
    Threshold = threshold_for_fpr(natives of one device half, fpr), evaluated on the reals (natives, WA, FBH) of the OTHER half.
    -> dict(fpr={pool: pooled FPR}, n={pool: n evaluated}, tpr, thr=[..]) pooled over both folds."""
    score, dev, pool, label = map(np.asarray, (score, dev, pool, label))
    A, B = device_halves(dev[label == 0], seed)
    fp = {p: 0 for p in POOLS}
    n = {p: 0 for p in POOLS}
    tp = nt = 0
    thrs = []
    for cal, ev in ((A, B), (B, A)):
        m_cal = (label == 0) & (pool == "vision_native") & np.isin(dev, list(cal))
        thr = metrics.threshold_for_fpr(score[m_cal], fpr)
        thrs.append(float(thr))
        for p in POOLS:
            m = (label == 0) & (pool == p) & np.isin(dev, list(ev))
            fp[p] += int((score[m] >= thr).sum())
            n[p] += int(m.sum())
        f = label == 1
        tp += int((score[f] >= thr).sum())
        nt += int(f.sum())
    return dict(fpr={p: fp[p] / max(n[p], 1) for p in POOLS}, n=n, fp=fp, tpr=tp / max(nt, 1), thr=thrs)


def calib_fpr(score, pool, label, calib_scores, fpr=0.01):
    """Alternative rule: threshold from the calib-split natives, evaluated on all test reals per pool."""
    thr = metrics.threshold_for_fpr(np.asarray(calib_scores), fpr)
    score, pool, label = map(np.asarray, (score, pool, label))
    return dict(fpr={p: float((score[(label == 0) & (pool == p)] >= thr).mean()) for p in POOLS},
                tpr=float((score[label == 1] >= thr).mean()), thr=float(thr))


def decide_label(s):
    """Computed label from a summary dict (see analyze_arm). Order: validity gates -> UNINFORMATIVE; else PASS/FAIL/INCONCLUSIVE;
    then the matched-blur rule appends 'NOT DIFFUSION-SPECIFIC'."""
    gates = s["gates"]
    if not all(gates.values()):
        base = "UNINFORMATIVE"
        crit = None
    else:
        B, f2 = s["B"], s["f2_laun"]
        crit = {
            "laundered_ge_0.80_ci_lo_ge_0.75_and_f2_plus_0.10": bool(B[LAUN]["value"] >= 0.80 and B[LAUN]["ci_lo"] >= 0.75 and B[LAUN]["value"] >= f2 + 0.10),
            "clean_ge_0.90": bool(B[CLEAN]["value"] >= 0.90),
            "social_ge_0.80": bool(B[SOCIAL]["value"] >= 0.80),
            "non_sd_laundered_ge_0.75": bool(s["nonsd_laun"] >= 0.75),
            "beats_texture_and_pixelL2_by_0.10_every_cond": bool(all(s["d_tex"][c] >= 0.10 and s["d_pix"][c] >= 0.10 for c in CONDS)),
            "fpr_le_2pct_every_cond_pool": bool(s["fpr_max"] <= 0.02),
        }
        if all(crit.values()):
            base = "PASS"
        elif B[LAUN]["value"] < 0.70 or B[LAUN]["value"] <= f2 + 0.03:
            base = "FAIL"
        else:
            base = "INCONCLUSIVE"
    blur_ok = bool(s["blur"]["delta"] >= 0.05 and s["blur"]["ci_lo"] > 0)
    label = base if blur_ok else base + "; NOT DIFFUSION-SPECIFIC"
    return dict(base=base, label=label, criteria=crit, blur_beaten=blur_ok, gates=gates)


# ============================================================================= data
def load_df(arm):
    df = e2.load_df(arm)
    if DRY:
        keep = []
        for (sp, lab, pool, dev), g in df[df.label == 0].groupby(["split", "label", "pool", "device"]):
            if sp in ("train", "test") or (sp == "calib" and pool == "vision_native"):
                keep.append(g.sort_values("sha256").head(1))
        for (sp, gen), g in df[df.label == 1].groupby(["split", "generator"]):
            keep.append(g.sort_values("sha256").head(1))
        df = pd.concat(keep).sort_index()
    return df.reset_index(drop=True)


def set_masks(df):
    lab, sp, pool = df.label.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    tr = (lab == 0) & (sp == "train")
    ca = (lab == 0) & (sp == "calib") & (pool == "vision_native")
    te = (lab == 0) & (sp == "test")
    fk = lab == 1
    return tr, ca, te, fk


def feat_rows(df):
    tr, ca, te, fk = set_masks(df)
    return df[tr | ca | te | fk]


def guard_shas(df):
    """The fixed D-008 subset: 16 test natives + 16 fakes (seeded, sorted by sha); 8+8 in dry."""
    n = 8 if DRY else 16
    rng = np.random.default_rng(SEED)
    nat = sorted(df.sha256[(df.label == 0) & (df.split == "test") & (df.pool == "vision_native")])
    fak = sorted(df.sha256[df.label == 1])
    return ([nat[i] for i in sorted(rng.choice(len(nat), min(n, len(nat)), replace=False))] +
            [fak[i] for i in sorted(rng.choice(len(fak), min(n, len(fak)), replace=False))])


def vae_pass(vae, b, arrs):
    """list of uint8 [H,W,3] -> list of raw uint8 VAE outputs (batches of equal size)."""
    out = [None] * len(arrs)
    groups = {}
    for i, a in enumerate(arrs):
        groups.setdefault(a.shape, []).append(i)
    for ix in groups.values():
        y = e2.q8(b(lambda a: e2.recon_f(vae, a), np.stack([arrs[i] for i in ix])))
        for i, yy in zip(ix, y):
            out[i] = yy
    return out


def embed(bb, imgs, dev):
    import features
    F = features.embed_images(bb, imgs, 20, dev.type)
    if getattr(bb, "force_cpu", False) and dev.type != "cpu":
        raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
    return F.astype(np.float32)


def rms(a, b):
    return float(np.sqrt(np.mean(((a.astype(np.float32) - b.astype(np.float32)) / 255.0) ** 2)))


def load_t(df_rows, cond, pool):
    return list(pool.map(lambda r: crop8(p4run.load_cond(r[0], r[1], cond)), zip(df_rows.path, df_rows.sha256)))


def load_shards(d, prefix):
    """-> dict name -> array concatenated over shards (incl. 'keys'); None if no shard."""
    files = sorted(Path(d).glob(prefix + "*.npz"))
    files = [f for f in files if ".tmp." not in f.name]
    if not files:
        return None
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def align(df, z, names=None):
    """shard dict -> {name: array aligned to df rows (NaN where missing)}."""
    pos = {k: i for i, k in enumerate(z["keys"].tolist())}
    idx = np.array([pos.get(s, -1) for s in df.sha256], int)
    ok = idx >= 0
    out = {}
    for k, v in z.items():
        if k == "keys" or (names and k not in names):
            continue
        m = np.full((len(df),) + v.shape[1:], np.nan, np.float32)
        m[ok] = v[idx[ok]]
        out[k] = m
    out["_present"] = ok
    return out


# ============================================================================= stage feat
def vae_guard():
    """VAE pixel guard of D-008: 009 measured it for this exact weights revision and recon function (CUDA vs CPU, 32 images)."""
    c9 = json.loads((ROOT / "experiments" / "009-vae-aligned" / "config.json").read_text())
    v9 = c9["vaes"][VN]
    assert v9["revision"] == CFG["vae"]["revision"] and v9["repo"] == CFG["vae"]["repo"], "VAE revision differs from 009"
    assert e2.VAES[VN] == v9
    g9 = json.loads((e2.PILOT / "d008_guard.json").read_text())["vae"][VN]
    assert g9["ok"] and max(g9["max_abs_dpixel_x1"], g9["max_abs_dpixel_x2"]) <= TOL, g9
    g = jload(PILOT / "d008_guard.json")
    g["vae"] = {VN: dict(g9, reused_from="experiments/009-vae-aligned/pilot/d008_guard.json (same weights revision, same recon_f/q8; not re-run)",
                         revision=v9["revision"])}
    jdump(PILOT / "d008_guard.json", g)
    log(f"D-008 VAE pixel guard reused from 009: max|dpixel| x1={g9['max_abs_dpixel_x1']:.2e} x2={g9['max_abs_dpixel_x2']:.2e} (ok)")


def stage_feat():
    t0 = time.time()
    vae_guard()
    dev = e2.get_dev()
    vae = e2.load_vae(VN, dev)
    b = e2.Batcher()
    bb = p4run.secondary_backbone()
    p4run.check_design_vs_003(bb)
    pool = ThreadPoolExecutor(4)
    for arm in ARMS:
        df = load_df(arm)
        rows = feat_rows(df)
        guard = set(guard_shas(df))
        for cond in CONDS:
            t1, n_new = time.time(), 0
            for ci, s in enumerate(range(0, len(rows), CHUNK)):
                f = CACHE / "feat" / f"{arm}__{cond}__{ci:03d}.npz"
                f.parent.mkdir(parents=True, exist_ok=True)
                if f.exists():
                    continue
                ch = rows.iloc[s:s + CHUNK]
                T = load_t(ch, cond, pool)
                Y = [Image.fromarray(y) for y in vae_pass(vae, b, [np.asarray(t) for t in T])]
                for sha, y in zip(ch.sha256, Y):
                    if sha in guard:
                        gp = CACHE / "guard" / arm / cond / f"{sha}.png"
                        gp.parent.mkdir(parents=True, exist_ok=True)
                        y.save(gp)
                L = {"t": T, "y": Y}
                for nm, src in (("l0t", T), ("l0y", Y)):
                    L[nm] = list(pool.map(lambda im: l0(im, "min"), src))
                for nm, src in (("l0ut", T), ("l0uy", Y)):
                    L[nm] = list(pool.map(lambda im: l0(im, "uniform"), src))
                out = {k: embed(bb, v, dev) for k, v in L.items()}
                out["l2"] = np.array([rms(np.asarray(t), np.asarray(y)) for t, y in zip(T, Y)], np.float32)
                out["lap"] = np.array([e2.lap224(t) for t in T], np.float32)
                out["side"] = np.array([t.size[0] for t in T], np.int32)
                atomic_npz(f, keys=ch.sha256.to_numpy().astype(str), **out)
                n_new += len(ch)
                if (ci % 5) == 0:
                    log(f"feat {arm}/{cond}: chunk {ci} ({s + len(ch)}/{len(rows)}) {time.time() - t1:.0f}s batch {b.bs}")
            log(f"feat {arm}/{cond}: {n_new} new rows, {time.time() - t1:.0f}s")
    stage_time("feat", t0)


def stage_vv():
    t0 = time.time()
    dev = e2.get_dev()
    vae = e2.load_vae(VN, dev)
    b = e2.Batcher()
    bb = p4run.secondary_backbone()
    pool = ThreadPoolExecutor(4)
    for arm in ARMS:
        df = load_df(arm)
        tr, ca, te, fk = set_masks(df)
        rows = df[tr | (te & (df.pool.to_numpy() == "vision_native"))]
        guard = set(guard_shas(df))
        for ci, s in enumerate(range(0, len(rows), CHUNK)):
            f = CACHE / "vv" / f"{arm}__{ci:03d}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            if f.exists():
                continue
            ch = rows.iloc[s:s + CHUNK]
            T = load_t(ch, CLEAN, pool)
            Y = vae_pass(vae, b, [np.asarray(t) for t in T])
            YY = [Image.fromarray(y) for y in vae_pass(vae, b, Y)]
            for sha, y in zip(ch.sha256, YY):
                if sha in guard:
                    gp = CACHE / "guard" / arm / "vv" / f"{sha}.png"
                    gp.parent.mkdir(parents=True, exist_ok=True)
                    y.save(gp)
            fyy = embed(bb, YY, dev)
            fl0 = embed(bb, list(pool.map(lambda im: l0(im, "min"), YY)), dev)
            atomic_npz(f, keys=ch.sha256.to_numpy().astype(str), yy=fyy, l0yy=fl0)
            log(f"vv {arm}: chunk {ci} ({s + len(ch)}/{len(rows)}) {time.time() - t0:.0f}s")
    stage_time("vv", t0)


# ============================================================================= stage probes
def probe_file(arm, name, shared):
    return (SHARED if shared else CACHE) / f"probe__{arm}__{name}.pkl"


def get_probe(arm, name, build, shared):
    p = probe_file(arm, name, shared)
    if p.exists():
        return pickle.loads(p.read_bytes())
    m = build()
    tmp = Path(str(p) + ".tmp")
    tmp.write_bytes(pickle.dumps(m))
    os.replace(tmp, p)
    return m


def load_probes(arm):
    return {"p": pickle.loads(probe_file(arm, "primary", True).read_bytes()),
            "m": pickle.loads(probe_file(arm, "matched", True).read_bytes()),
            "a2": pickle.loads(probe_file(arm, "a2", False).read_bytes()),
            "a2l": pickle.loads(probe_file(arm, "a2l", False).read_bytes())}


def stage_probes():
    t0 = time.time()
    r011 = json.loads((ROOT / "experiments" / "011-vae-trace" / "pilot" / "results.json").read_text())
    chk = {}
    for arm in ARMS:
        d11, D = R.load_arm(arm)
        lab, sp, gen, pool, dv = (d11[c].to_numpy() for c in ("label", "split", "generator", "pool", "device"))
        grp = e2.source_groups(d11)
        tr = np.where((lab == 0) & (sp == "train"))[0]
        te_nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
        fak = np.where(lab == 1)[0]
        fak = fak[np.argsort(gen[fak], kind="stable")]

        def build(c):
            X = np.vstack([R.X_of(D, "F2", "plain", c, tr), R.X_of(D, "F2", "rt", c, tr)])
            y = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
            t1 = time.time()
            m = e2.fit(X, y, np.concatenate([grp[tr], grp[tr]]))
            log(f"{arm}: fit F2 train_cond={c} C={m[-1].C_[0]:.3g} ({time.time() - t1:.0f}s)")
            return m
        mp = get_probe(arm, "primary", lambda: build(CLEAN), True)
        mm = get_probe(arm, "matched", lambda: build(LAUN), True)
        # reproduction of 011's transfer numbers (F2 primary and matched) from the refit probes
        plan = e2.Plan(dv[te_nat], gen[fak], n_boot=200)
        chk[arm] = {}
        for nm, m, proto in (("primary", mp, "primary"), ("matched", mm, "matched")):
            for c in (CLEAN, LAUN):
                if proto == "matched" and c == CLEAN:
                    continue
                s_nat = m.decision_function(R.X_of(D, "F2", "plain", c, te_nat))
                s_fak = m.decision_function(R.X_of(D, "F2", "plain", c, fak))
                v = plan.macro(s_nat, s_fak)["value"]
                ref = r011["arms"][arm]["transfer"][proto][c]["F2"]["all6"]["value"]
                chk[arm][f"{nm}/{c}"] = dict(refit=v, e4_results_json=ref, abs_diff=abs(v - ref))
        log(f"{arm}: refit-vs-011 transfer check {chk[arm]}")
        assert all(x["abs_diff"] < 1e-6 for x in chk[arm].values()), chk[arm]
        # A2 / A2L from our own train features and second round-trips
        df = load_df(arm)
        trm, _, _, _ = set_masks(df)
        z = align(df, load_shards(CACHE / "feat", f"{arm}__{CLEAN}__"))
        zz = align(df, load_shards(CACHE / "vv", f"{arm}__"))
        idx = np.where(trm)[0]
        assert not np.isnan(z["t"][idx]).any() and not np.isnan(zz["yy"][idx]).any()
        g2 = e2.source_groups(df)[idx]
        y2 = np.r_[np.ones(len(idx), int), np.zeros(len(idx), int)]
        for nm, (a, b_, c_) in (("a2", ("t", "y", "yy")), ("a2l", ("l0t", "l0y", "l0yy"))):
            def build2(a=a, b_=b_, c_=c_):
                d1 = z[b_][idx] - z[a][idx]
                src_y = z[b_][idx]
                d2 = (zz[c_][idx] - src_y)
                return e2.fit(np.vstack([d1, d2]), y2, np.concatenate([g2, g2]))
            get_probe(arm, nm, build2, False)
    jdump(PILOT / "check_probes_vs_011.json", chk)
    stage_time("probes", t0)


# ============================================================================= scores
def compute_scores(Pr, F, idx):
    """F: dict of aligned feature matrices; idx: rows. Returns {name: fake-high scores for idx} using ONE code path for analysis and guard.
    shift = d(round-trip) - d(original); fake-score = -shift (orientation fixed on train pairs)."""
    dp, dm = Pr["p"].decision_function, Pr["m"].decision_function
    out = {
        "A": -(dp(F["y"][idx]) - dp(F["t"][idx])),
        "B": -(dm(F["l0y"][idx]) - dm(F["l0t"][idx])),
        "Bu": -(dm(F["l0uy"][idx]) - dm(F["l0ut"][idx])),
        "A2": -Pr["a2"].decision_function(F["y"][idx] - F["t"][idx]),
        "A2L": -Pr["a2l"].decision_function(F["l0y"][idx] - F["l0t"][idx]),
        "F2p": dp(F["t"][idx]),
        "F2m": dm(F["t"][idx]),
    }
    if "bA" in F:
        out["blurA"] = -(dp(F["bA"][idx]) - dp(F["t"][idx]))
        out["blurB"] = -(dm(F["bB"][idx]) - dm(F["l0t"][idx]))
    return out


# ============================================================================= stage blur
def stage_blur():
    t0 = time.time()
    dev = e2.get_dev()
    bb = p4run.secondary_backbone()
    pool = ThreadPoolExecutor(4)
    for arm in ARMS:
        df = load_df(arm)
        Pr = load_probes(arm)
        tr, ca, te, fk = set_masks(df)
        dpf, dmf = Pr["p"].decision_function, Pr["m"].decision_function
        for cond in CONDS:
            jf = CACHE / "blur" / f"sigma__{arm}__{cond}.json"
            jf.parent.mkdir(parents=True, exist_ok=True)
            z = align(df, load_shards(CACHE / "feat", f"{arm}__{cond}__"))
            if not jf.exists():
                idx = np.where(tr)[0]
                T = load_t(df.iloc[idx], cond, pool)
                tgtA = float(np.mean(dpf(z["y"][idx]) - dpf(z["t"][idx])))
                tgtB = float(np.mean(dmf(z["l0y"][idx]) - dmf(z["l0t"][idx])))
                mA, mB = [], []
                for sg in BLUR_GRID:
                    Bl = list(pool.map(lambda im: blur(im, sg), T))
                    fa = embed(bb, Bl, dev)
                    fb = embed(bb, list(pool.map(lambda im: l0(im, "min"), Bl)), dev)
                    mA.append(float(np.mean(dpf(fa) - dpf(z["t"][idx]))))
                    mB.append(float(np.mean(dmf(fb) - dmf(z["l0t"][idx]))))
                sA, stA = match_sigma(BLUR_GRID, mA, tgtA)
                sB, stB = match_sigma(BLUR_GRID, mB, tgtB)
                jdump(jf, dict(arm=arm, cond=cond, n_train_reals=len(idx), vae_mean_shift_A=tgtA, vae_mean_shift_B=tgtB,
                               grid=BLUR_GRID, blur_mean_shift_A=mA, blur_mean_shift_B=mB, sigma_A=sA, status_A=stA, sigma_B=sB, status_B=stB))
                log(f"blur sigma {arm}/{cond}: A target {tgtA:.3f} -> sigma {sA:.3f} ({stA}); B target {tgtB:.3f} -> sigma {sB:.3f} ({stB})")
            sg = jload(jf)
            rows = df[te & (df.pool.to_numpy() == "vision_native") | fk]
            for ci, s in enumerate(range(0, len(rows), CHUNK)):
                f = CACHE / "blur" / f"feat__{arm}__{cond}__{ci:03d}.npz"
                if f.exists():
                    continue
                ch = rows.iloc[s:s + CHUNK]
                T = load_t(ch, cond, pool)
                fa = embed(bb, list(pool.map(lambda im: blur(im, sg["sigma_A"]), T)), dev)
                fb = embed(bb, list(pool.map(lambda im: l0(blur(im, sg["sigma_B"]), "min"), T)), dev)
                atomic_npz(f, keys=ch.sha256.to_numpy().astype(str), bA=fa, bB=fb)
            log(f"blur feat {arm}/{cond}: done ({len(rows)} rows)")
    stage_time("blur", t0)


# ============================================================================= stage guard
def stage_guard():
    """D-008 for every new GPU feature: DINOv2 features of the stored VAE outputs / blur images recomputed on CPU; |dscore| of every score
    (and the gate-1/A2 quantities from the second round-trips) between the CUDA cache and the CPU recomputation must be <= 1e-3."""
    t0 = time.time()
    g = jload(PILOT / "d008_guard.json")
    if not g.get("vae", {}).get(VN, {}).get("ok"):
        raise SystemExit("VAE guard missing: run --stage feat first")
    bb = p4run.secondary_backbone()
    cpu = __import__("torch").device("cpu")
    pool = ThreadPoolExecutor(4)
    out = dict(tol=TOL, items={}, ok=True)
    for arm in ARMS:
        df = load_df(arm)
        Pr = load_probes(arm)
        gs = guard_shas(df)
        pos = {s: i for i, s in enumerate(df.sha256)}
        gi = np.array([pos[s] for s in gs])
        rows = df.iloc[gi]
        for cond in CONDS:
            z = align(df, load_shards(CACHE / "feat", f"{arm}__{cond}__"))
            zb = load_shards(CACHE / "blur", f"feat__{arm}__{cond}__")
            if zb is not None:
                z.update({k: v for k, v in align(df, zb).items() if k != "_present"})
            T = load_t(rows, cond, pool)
            Y = [Image.open(CACHE / "guard" / arm / cond / f"{s}.png").convert("RGB") for s in gs]
            sg = jload(CACHE / "blur" / f"sigma__{arm}__{cond}.json")
            Lc = {"t": T, "y": Y}
            for nm, src in (("l0t", T), ("l0y", Y)):
                Lc[nm] = [l0(im, "min") for im in src]
            for nm, src in (("l0ut", T), ("l0uy", Y)):
                Lc[nm] = [l0(im, "uniform") for im in src]
            if zb is not None:
                Lc["bA"] = [blur(im, sg["sigma_A"]) for im in T]
                Lc["bB"] = [l0(blur(im, sg["sigma_B"]), "min") for im in T]
            Fc = {k: embed(bb, v, cpu) for k, v in Lc.items()}
            Fg = {k: z[k] for k in Fc}          # the cached CUDA features were computed from these same images (stored VAE outputs)
            Fc_full = {k: np.full((len(df), v.shape[1]), np.nan, np.float32) for k, v in Fc.items()}
            for k, v in Fc.items():
                Fc_full[k][gi] = v
            blur_idx = np.array([i for i in gi if not np.isnan(z["bA"][i]).any()]) if zb is not None else gi
            s_gpu = compute_scores(Pr, {k: z[k] for k in Fc}, blur_idx)
            s_cpu = compute_scores(Pr, Fc_full, blur_idx)
            item = {k: float(np.abs(s_gpu[k] - s_cpu[k]).max()) for k in s_gpu}
            item["max_scaled_feature_diff"] = float(max((np.abs(Fc[k] - Fg[k][gi]) / np.maximum(1.0, np.abs(Fc[k]).max(0, keepdims=True))).max() for k in Fc))
            item["n_images"] = len(gs)
            item["n_blur_rows"] = int(len(blur_idx))
            item["max_abs_dscore"] = max(v for k, v in item.items() if k in s_gpu)
            item["ok"] = bool(item["max_abs_dscore"] <= TOL)
            out["items"][f"{arm}/{cond}"] = item
            out["ok"] &= item["ok"]
            log(f"D-008 {arm}/{cond}: max|dscore|={item['max_abs_dscore']:.2e} scaled dF={item['max_scaled_feature_diff']:.2e} -> {'ok' if item['ok'] else 'FAIL'}")
        # second round-trip features (gate 1 and A2 quantities): guard natives only
        zz = align(df, load_shards(CACHE / "vv", f"{arm}__"))
        z = align(df, load_shards(CACHE / "feat", f"{arm}__{CLEAN}__"))
        nat = [s for s in gs if s in set(df.sha256[(df.label == 0)])]
        ni = np.array([pos[s] for s in nat])
        YY = [Image.open(CACHE / "guard" / arm / "vv" / f"{s}.png").convert("RGB") for s in nat]
        fy = embed(bb, YY, cpu)
        fl = embed(bb, [l0(im, "min") for im in YY], cpu)
        dp = Pr["p"].decision_function
        d2_gpu = dp(zz["yy"][ni]) - dp(z["y"][ni])
        d2_cpu = dp(fy) - dp(z["y"][ni])
        a2_gpu = Pr["a2"].decision_function(zz["yy"][ni] - z["y"][ni])
        a2_cpu = Pr["a2"].decision_function(fy - z["y"][ni])
        a2l_gpu = Pr["a2l"].decision_function(zz["l0yy"][ni] - z["l0y"][ni])
        a2l_cpu = Pr["a2l"].decision_function(fl - z["l0y"][ni])
        item = dict(n_images=len(nat), gate1_delta2=float(np.abs(d2_gpu - d2_cpu).max()), a2_delta2=float(np.abs(a2_gpu - a2_cpu).max()),
                    a2l_delta2=float(np.abs(a2l_gpu - a2l_cpu).max()))
        item["max_abs_dscore"] = max(item["gate1_delta2"], item["a2_delta2"], item["a2l_delta2"])
        item["ok"] = bool(item["max_abs_dscore"] <= TOL)
        out["items"][f"{arm}/vv"] = item
        out["ok"] &= item["ok"]
        log(f"D-008 {arm}/vv: max|dscore|={item['max_abs_dscore']:.2e} -> {'ok' if item['ok'] else 'FAIL'}")
    g["probe"] = out
    jdump(PILOT / "d008_guard.json", g)
    stage_time("guard", t0)
    if not out["ok"]:
        raise SystemExit("D-008 ABORT: |dscore| > 1e-3; analysis will refuse to run")


# ============================================================================= stage analyze
def cell(d):
    return f"{d['value']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


SCORES = ["A", "B", "Bu", "A2", "A2L", "F2p", "F2m", "texture", "pixL2", "blurA", "blurB"]
DESC = {"A": "A: w_p.[phi(VAE t)-phi(t)]", "B": "B: w_m.[phi(L0 VAE t)-phi(L0 t)] (LABEL)", "Bu": "Bu: B, uniform L0", "A2": "A2: diff-vector probe",
        "A2L": "A2L: A2 with L0", "F2p": "pooled F2 (E4 primary probe on phi(t))", "F2m": "F2 matched probe on phi(t)", "texture": "Laplacian texture",
        "pixL2": "pixel L2 round-trip error", "blurA": "A with blur for VAE", "blurB": "B with blur for VAE"}


def analyze_arm(arm, long_rows):
    df = load_df(arm)
    Pr = load_probes(arm)
    lab, gen, sp, pool, dev_, sha = (df[c].to_numpy() for c in ("label", "generator", "split", "pool", "device", "sha256"))
    trm, cam, tem, fkm = set_masks(df)
    te = np.where(tem)[0]
    te_nat = np.where(tem & (pool == "vision_native"))[0]
    fak = np.where(fkm)[0]
    fak = fak[np.argsort(gen[fak], kind="stable")]
    fk_tr = np.where(fkm & (sp == "train"))[0]
    ca = np.where(cam)[0]
    tr = np.where(trm)[0]
    plan = e2.Plan(dev_[te_nat], gen[fak], n_boot=N_BOOT)
    pplans = {p: e2.Plan(dev_[tem & (pool == p)], gen[fak], n_boot=N_BOOT) for p in POOLS}
    for p in POOLS[1:]:
        assert np.array_equal(pplans[p].Wf, pplans[POOLS[0]].Wf) and np.array_equal(pplans[p].Wr, pplans[POOLS[0]].Wr), "pool plans must share draws"
    gens_present = sorted(set(gen[fak]))
    Rz = dict(n_train_reals=len(tr), n_test_reals=len(te), n_test_natives=len(te_nat), n_calib_natives=len(ca), n_fakes=len(fak),
              n_test_devices=int(len(set(dev_[te]))), sigma={}, scores={}, gates={}, gate_detail={}, fpr={}, fpr_calib={}, orientation={}, checks={})
    BOOT = {}
    S = {c: {} for c in CONDS}
    ZZ = align(df, load_shards(CACHE / "vv", f"{arm}__"))
    Z1 = align(df, load_shards(CACHE / "feat", f"{arm}__{CLEAN}__"))
    # ---- scores per condition
    for cond in CONDS:
        z = align(df, load_shards(CACHE / "feat", f"{arm}__{cond}__"))
        zb = load_shards(CACHE / "blur", f"feat__{arm}__{cond}__")
        if zb is not None:
            z.update({k: v for k, v in align(df, zb).items() if k != "_present"})
        rows_ok = np.where(z["_present"])[0]
        assert set(rows_ok) >= set(np.r_[tr, ca, te, fak]), (arm, cond, "missing feature rows")
        F = {k: z[k] for k in ("t", "y", "l0t", "l0y", "l0ut", "l0uy")}
        blur_rows = np.where(~np.isnan(z["bA"]).any(1))[0] if "bA" in z else np.array([], int)
        full = {}
        for nm in ("A", "B", "Bu", "A2", "A2L", "F2p", "F2m"):
            full[nm] = np.full(len(df), np.nan)
        sc = compute_scores(Pr, F, rows_ok)
        for nm in full:
            full[nm][rows_ok] = sc[nm]
        if len(blur_rows):
            Fb = dict(F, bA=z["bA"], bB=z["bB"])
            sb = compute_scores(Pr, Fb, blur_rows)
            full["blurA"], full["blurB"] = np.full(len(df), np.nan), np.full(len(df), np.nan)
            full["blurA"][blur_rows] = sb["blurA"]
            full["blurB"][blur_rows] = sb["blurB"]
        # texture (sign: calib natives vs train-split fakes, 009 convention) and pixel L2
        lap, l2 = z["lap"].astype(float), z["l2"].astype(float)
        a_tex = e2.auc(lap[ca], lap[fk_tr])
        sgn = 1 if a_tex >= 0.5 else -1
        full["texture"] = sgn * lap
        full["pixL2"] = -l2
        Rz["checks"].setdefault("texture_sign", {})[cond] = dict(sign=sgn, auroc_plus_sign_calib_natives_vs_train_fakes=a_tex)
        S[cond] = full
        # consistency of our phi(t') with the 004 cache (clean/laundered/social: where t' == t)
        c004 = P.load_features_aligned(df, "secondary", cond, subdir=R.SUBDIR[arm])
        okr = np.where(~np.isnan(c004).any(1) & (z["side"] > 0))[0]
        same = np.array([i for i in okr if z["side"][i] in (512, 256)], int)
        if len(same):
            Rz["checks"].setdefault("phi_t_vs_004_cache_max_abs_diff", {})[cond] = float(np.abs(F["t"][same] - c004[same]).max())
        # sigma
        sj = jload(CACHE / "blur" / f"sigma__{arm}__{cond}.json")
        if sj:
            Rz["sigma"][cond] = {k: sj[k] for k in ("vae_mean_shift_A", "vae_mean_shift_B", "sigma_A", "status_A", "sigma_B", "status_B")}
    # ---- orientation (fixed on train pairs) and per-score metrics
    dp, dm = Pr["p"].decision_function, Pr["m"].decision_function
    for cond in CONDS:
        zc = align(df, load_shards(CACHE / "feat", f"{arm}__{cond}__"))
        shA = dp(zc["y"][tr]) - dp(zc["t"][tr])
        shB = dm(zc["l0y"][tr]) - dm(zc["l0t"][tr])
        Rz["orientation"][cond] = dict(A_frac_train_reals_positive=float((shA > 0).mean()), A_mean_shift=float(shA.mean()),
                                       B_frac_train_reals_positive=float((shB > 0).mean()), B_mean_shift=float(shB.mean()))
    d1 = Pr["a2"].decision_function(Z1["y"][tr] - Z1["t"][tr])
    d2 = Pr["a2"].decision_function(ZZ["yy"][tr] - Z1["y"][tr])
    Rz["orientation"]["A2_train_in_sample_auroc_delta1_vs_delta2"] = e2.auc(d2, d1)
    Rz["orientation"]["note"] = "shift>0 for a real photo (round-trip = class 1); fake-score = -shift; fixed, never flipped on test"
    for nm in SCORES:
        Rz["scores"][nm] = {}
        for cond in CONDS:
            s = S[cond].get(nm)
            if s is None or np.isnan(s[te_nat]).any() or np.isnan(s[fak]).any():
                continue
            m_all = plan.macro(s[te_nat], s[fak])
            BOOT[(nm, cond)] = m_all
            r = dict(macro=e2.strip(m_all), sd_family=e2.strip(plan.macro(s[te_nat], s[fak], SD_FAM)),
                     other=e2.strip(plan.macro(s[te_nat], s[fak], OTHER_FAM)), per_generator=plan.per_gen(s[te_nat], s[fak]))
            r["per_pool"] = {}
            for p in POOLS:
                ip = np.where(tem & (pool == p))[0]
                if not np.isnan(s[ip]).any():          # blur scores exist for the natives only
                    r["per_pool"][p] = e2.strip(pplans[p].macro(s[ip], s[fak]))
            # flipped orientation (descriptive only)
            r["macro_flipped_orientation"] = 1 - m_all["value"]
            Rz["scores"][nm][cond] = r
            for kind, ix in (("real", te), ("fake", fak), ("calib_native", ca), ("train_real", tr)):
                long_rows += [(arm, nm, cond, kind, sp[i], sha[i], gen[i], dev_[i], pool[i], float(s[i])) for i in ix if not np.isnan(s[i])]
    # ---- gates
    # gate 1 (saturation): primary w, clean, test natives
    D1 = dp(Z1["y"][te_nat]) - dp(Z1["t"][te_nat])
    D2 = dp(ZZ["yy"][te_nat]) - dp(Z1["y"][te_nat])
    g1 = dict(median_w_delta1=float(np.median(D1)), median_w_delta2=float(np.median(D2)), ratio=float(np.median(D2) / np.median(D1)) if np.median(D1) != 0 else None,
              frac_delta1_pos=float((D1 > 0).mean()), frac_delta2_pos=float((D2 > 0).mean()))
    g1["passed"] = bool(np.median(D1) > 0 and np.median(D2) <= 0.5 * np.median(D1))
    # gate 2 (content cancellation)
    g2 = {}
    for cond in CONDS:
        zc = align(df, load_shards(CACHE / "feat", f"{arm}__{cond}__"))
        rA = spearmanr(dp(zc["t"][te_nat]), dp(zc["y"][te_nat]))[0]
        rB = spearmanr(dm(zc["l0t"][te_nat]), dm(zc["l0y"][te_nat]))[0]
        rBall = spearmanr(dm(zc["l0t"][te]), dm(zc["l0y"][te]))[0]
        g2[cond] = dict(A=float(rA), B=float(rB), B_all_test_reals=float(rBall), B_passed=bool(rB >= 0.8), A_passed=bool(rA >= 0.8))
    g2["passed"] = bool(all(g2[c]["B_passed"] for c in CONDS))
    Rz["gate_detail"] = dict(gate1=g1, gate2=g2)
    # gate 5 (validity, Amendments 6/8)
    Bc, Tc = Rz["scores"]["B"][CLEAN]["macro"], Rz["scores"]["texture"][CLEAN]["macro"]
    g5 = dict(B_clean=Bc, texture_clean=Tc, passed=bool(Bc["ci_lo"] > 0.5 and Bc["value"] > Tc["value"]))
    Rz["gate_detail"]["gate5"] = g5
    # ---- deltas
    def dlt(a, b, cond):
        return e2.diff(BOOT[(a, cond)], BOOT[(b, cond)])
    Rz["deltas"] = {c: {f"B-{o}": dlt("B", o, c) for o in ("texture", "pixL2", "F2p", "F2m", "A", "blurB", "Bu") if ("B", c) in BOOT and (o, c) in BOOT} for c in CONDS}
    Rz["deltas"].update({"A_vs_blurA": {c: dlt("A", "blurA", c) for c in CONDS if ("blurA", c) in BOOT}})
    # ---- FPR
    for nm in ("A", "B", "Bu", "A2", "A2L", "F2p", "texture", "pixL2", "blurB"):
        Rz["fpr"][nm], Rz["fpr_calib"][nm] = {}, {}
        for cond in CONDS:
            s = S[cond].get(nm)
            if s is None or np.isnan(s[te]).any() or np.isnan(s[fak]).any():
                continue
            ix = np.r_[te, fak]
            Rz["fpr"][nm][cond] = crossfit_fpr(s[ix], dev_[ix], pool[ix], lab[ix])
            Rz["fpr_calib"][nm][cond] = calib_fpr(s[ix], pool[ix], lab[ix], s[ca])
    # ---- label summary
    B = {c: Rz["scores"]["B"][c]["macro"] for c in CONDS}
    fpr_max = max(max(Rz["fpr"]["B"][c]["fpr"].values()) for c in CONDS)
    summ = dict(gates=dict(g1_saturation=g1["passed"], g2_content_cancellation=g2["passed"], g3_orientation=bool(Rz["orientation"][CLEAN]["A_frac_train_reals_positive"] > 0.5 and
                           Rz["orientation"][LAUN]["B_frac_train_reals_positive"] > 0.5 and Rz["orientation"]["A2_train_in_sample_auroc_delta1_vs_delta2"] > 0.5),
                           g5_validity=g5["passed"]),
                B=B, f2_laun=Rz["scores"]["F2p"][LAUN]["macro"]["value"], nonsd_laun=Rz["scores"]["B"][LAUN]["other"]["value"],
                d_tex={c: Rz["deltas"][c]["B-texture"]["delta"] for c in CONDS}, d_pix={c: Rz["deltas"][c]["B-pixL2"]["delta"] for c in CONDS},
                fpr_max=fpr_max, blur=Rz["deltas"][LAUN]["B-blurB"])
    Rz["summary"] = {k: v for k, v in summ.items()}
    Rz["decision"] = decide_label(summ)
    return Rz


def b0_reference():
    p = ROOT / "experiments" / "005-nuisance-projection" / "pilot" / "results.json"
    r = json.loads(p.read_text())
    return {c: r["arms"]["A0"][c]["macro"] for c in CONDS if c in r["arms"]["A0"]} | dict(source="experiments/005-nuisance-projection/pilot/results.json arms.A0[cond].macro (LOGO-macro, 005 pilot splits)")


def stage_analyze():
    g = jload(PILOT / "d008_guard.json")
    if not g.get("probe", {}).get("ok") or not g.get("vae", {}).get(VN, {}).get("ok"):
        raise SystemExit("D-008 guards missing/failed: run --stage guard first; results.json NOT written")
    res = dict(label="PILOT", config=CFG, dry=DRY, n_boot=N_BOOT, arms={}, d008=g)
    long_rows = []
    for arm in ARMS:
        res["arms"][arm] = analyze_arm(arm, long_rows)
        log(f"{arm}: computed label = {res['arms'][arm]['decision']['label']}")
    res["label_arm"] = LABEL_ARM
    res["computed_label"] = res["arms"][LABEL_ARM]["decision"]["label"]
    res["probe_checks"] = jload(PILOT / "check_probes_vs_011.json")
    try:
        res["b0_logo_reference"] = b0_reference()
    except Exception as e:
        res["b0_logo_reference"] = f"unavailable: {e}"
    res["stage_times_s"] = jload(CACHE / "stage_times.json")
    res["runtime_s_analyze_process"] = time.time() - T_START
    jdump(PILOT / "results.json", res)
    pd.DataFrame(long_rows, columns=["arm", "score", "cond", "kind", "split", "sha256", "generator", "device", "pool", "value"]
                 ).to_csv(PILOT / "scores_long.csv.gz", index=False)
    write_tables(res)
    log(f"analysis done in {time.time() - T_START:.0f}s; computed label ({LABEL_ARM}) = {res['computed_label']}")


def write_tables(res):
    L = []
    w = L.append
    w(f"# H-D1 PILOT self-ref-trace-contrast{' (DRY RUN: tiny subset, meaningless numbers)' if res['dry'] else ''}\n")
    w(f"Computed label ({res['label_arm']} arm, label score B): **{res['computed_label']}**\n")
    w("Fake-high scores; macro AUROC over the 6 Synthbuster generators, test natives (25 devices) vs 120 fakes per generator, cluster bootstrap "
      f"({res['n_boot']} replicates, reals by device, fakes by image per generator, one plan for all scores). Shift scores: fake-score = -(d(round-trip) - d(original)).\n")
    for arm in ARMS:
        Rz = res["arms"][arm]
        dec = Rz["decision"]
        w(f"\n## Arm `{arm}`{' (label arm)' if arm == res['label_arm'] else ''}: {dec['label']}\n")
        w(f"{Rz['n_train_reals']} train reals, {Rz['n_calib_natives']} calib natives, {Rz['n_test_reals']} test reals ({Rz['n_test_natives']} natives, {Rz['n_test_devices']} devices), {Rz['n_fakes']} fakes.\n")
        w("### Macro AUROC [95% CI]\n")
        w("| score | " + " | ".join(CONDS) + " |")
        w("|---|" + "---|" * len(CONDS))
        for nm in SCORES:
            if nm in Rz["scores"] and all(c in Rz["scores"][nm] for c in CONDS):
                w(f"| {DESC[nm]} | " + " | ".join(cell(Rz["scores"][nm][c]["macro"]) for c in CONDS) + " |")
        w("\nFlipped-orientation macro (1 - AUROC; descriptive only): " + "; ".join(
            f"{nm}: " + "/".join(f"{Rz['scores'][nm][c]['macro_flipped_orientation']:.3f}" for c in CONDS) for nm in ("texture", "pixL2", "A", "B") if nm in Rz["scores"]))
        w("\n### SD-family (SD2, SDXL) vs other (DALL-E 2/3, Firefly, MJ5)\n")
        w("| score | cond | SD-family | other |")
        w("|---|---|---|---|")
        for nm in ("A", "B", "Bu", "A2", "A2L", "F2p", "texture", "pixL2", "blurB"):
            for c in CONDS:
                if nm in Rz["scores"] and c in Rz["scores"][nm]:
                    r = Rz["scores"][nm][c]
                    w(f"| {nm} | {c} | {cell(r['sd_family'])} | {cell(r['other'])} |")
        w("\n### Per generator (AUROC, test natives vs that generator)\n")
        w("| score | cond | " + " | ".join(GENS) + " |")
        w("|---|---|" + "---|" * len(GENS))
        for nm in ("A", "B", "F2p", "blurB", "texture", "pixL2"):
            for c in CONDS:
                if nm in Rz["scores"] and c in Rz["scores"][nm]:
                    pg = Rz["scores"][nm][c]["per_generator"]
                    w(f"| {nm} | {c} | " + " | ".join(f"{pg[g]:.3f}" for g in GENS) + " |")
        w("\n### Per real pool (macro AUROC of that pool's test reals vs fakes)\n")
        w("| score | cond | native | WhatsApp | Facebook-HQ |")
        w("|---|---|---|---|---|")
        for nm in ("A", "B", "F2p", "blurB", "texture"):
            for c in CONDS:
                if nm in Rz["scores"] and c in Rz["scores"][nm]:
                    pp = Rz["scores"][nm][c]["per_pool"]
                    w(f"| {nm} | {c} | " + " | ".join(cell(pp[p]) if p in pp else "-" for p in POOLS) + " |")
        w("\n### FPR at the 1% threshold (primary: device cross-fit; test natives / WA / FBH of the other half's devices; pooled over both folds)\n")
        w("| score | cond | FPR native | FPR WA | FPR FBH | TPR (fakes) | FPR native/WA/FBH with calib-split threshold | TPR (calib thr) |")
        w("|---|---|---|---|---|---|---|---|")
        for nm in ("A", "B", "Bu", "A2", "A2L", "F2p", "texture", "pixL2", "blurB"):
            for c in CONDS:
                if c in Rz["fpr"].get(nm, {}):
                    f, fc = Rz["fpr"][nm][c], Rz["fpr_calib"][nm][c]
                    w(f"| {nm} | {c} | {f['fpr'][POOLS[0]]:.3f} | {f['fpr'][POOLS[1]]:.3f} | {f['fpr'][POOLS[2]]:.3f} | {f['tpr']:.3f} | "
                      f"{fc['fpr'][POOLS[0]]:.3f} / {fc['fpr'][POOLS[1]]:.3f} / {fc['fpr'][POOLS[2]]:.3f} | {fc['tpr']:.3f} |")
        w("\n### Matched-blur control and other paired differences (macro AUROC, paired bootstrap)\n")
        w("| contrast | cond | delta [95% CI] |")
        w("|---|---|---|")
        for c in CONDS:
            for k, d in Rz["deltas"][c].items():
                w(f"| {k} | {c} | {d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] |")
        for c, d in Rz["deltas"]["A_vs_blurA"].items():
            w(f"| A-blurA | {c} | {d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] |")
        w("\nBlur sigma (chosen on train reals to match the VAE's mean shift):\n")
        w("| cond | VAE mean shift A | sigma_A | status | VAE mean shift B | sigma_B | status |")
        w("|---|---|---|---|---|---|---|")
        for c, s in Rz["sigma"].items():
            w(f"| {c} | {s['vae_mean_shift_A']:.3f} | {s['sigma_A']:.3f} | {s['status_A']} | {s['vae_mean_shift_B']:.3f} | {s['sigma_B']:.3f} | {s['status_B']} |")
        w("\n### Gates\n")
        g1, g2, g5 = Rz["gate_detail"]["gate1"], Rz["gate_detail"]["gate2"], Rz["gate_detail"]["gate5"]
        w(f"- Gate 1 saturation (clean, test natives, w primary): median w.Delta1 = {g1['median_w_delta1']:.3f}, median w.Delta2 = {g1['median_w_delta2']:.3f}, "
          f"ratio {g1['ratio'] if g1['ratio'] is None else round(g1['ratio'], 3)} (need <= 0.5): **{'PASS' if g1['passed'] else 'FAIL'}**")
        w("- Gate 2 content cancellation, Spearman(w.phi(t), w.phi(VAE t)) on test natives (need >= 0.8; B is the gated score): " +
          "; ".join(f"{c}: A {g2[c]['A']:.3f}, B {g2[c]['B']:.3f}" for c in CONDS) + f" -> **{'PASS' if g2['passed'] else 'FAIL'}**")
        o = Rz["orientation"]
        w(f"- Gate 3 orientation fixed on train pairs: fraction of train reals with positive shift A {o[CLEAN]['A_frac_train_reals_positive']:.3f} (clean), "
          f"B {o[LAUN]['B_frac_train_reals_positive']:.3f} (laundered); A2 in-sample AUROC Delta1 vs Delta2 {o['A2_train_in_sample_auroc_delta1_vs_delta2']:.3f} -> **{'PASS' if dec['gates']['g3_orientation'] else 'FAIL'}**")
        w(f"- Gate 5 validity (Amendments 6/8): B clean {cell(g5['B_clean'])} vs texture clean {g5['texture_clean']['value']:.3f} -> **{'PASS' if g5['passed'] else 'FAIL'}**")
        w("- Gate 4 D-008: see the D-008 section.\n")
        w("### Pass/fail criteria\n")
        if dec["criteria"]:
            for k, v in dec["criteria"].items():
                w(f"- {k}: **{'yes' if v else 'no'}**")
        else:
            w("- not evaluated (a validity gate failed -> UNINFORMATIVE)")
        w(f"- B laundered {cell(Rz['summary']['B'][LAUN])}; pooled F2 laundered {Rz['summary']['f2_laun']:.3f}; non-SD laundered {Rz['summary']['nonsd_laun']:.3f}; max FPR (B, every cond/pool) {Rz['summary']['fpr_max']:.3f}")
        w(f"- Matched-blur: B - blurB at laundered = {Rz['summary']['blur']['delta']:+.3f} [{Rz['summary']['blur']['ci_lo']:+.3f}, {Rz['summary']['blur']['ci_hi']:+.3f}] -> blur {'beaten' if dec['blur_beaten'] else 'NOT beaten'}")
        w(f"- Base label {dec['base']}; **computed label: {dec['label']}**")
        w("\n### Checks\n```\n" + json.dumps(Rz["checks"], indent=1) + "\n```\n")
    w("\n## B0 LOGO reference (005 pilot)\n```\n" + json.dumps(res["b0_logo_reference"], indent=1, default=str) + "\n```\n")
    w("\n## Refit of E4's probes vs 011/results.json\n```\n" + json.dumps(res["probe_checks"], indent=1) + "\n```\n")
    w("\n## D-008\n```\n" + json.dumps(res["d008"], indent=1) + "\n```\n")
    w("\n## Stage times (s)\n```\n" + json.dumps(res["stage_times_s"], indent=1) + "\n```\n")
    (PILOT / "tables.md").write_text("\n".join(L))


STAGES = ["feat", "vv", "probes", "blur", "guard", "analyze"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    set_mode(a.dry)
    log(f"start stage={a.stage} dry={a.dry}")
    todo = STAGES if a.stage == "all" else [a.stage]
    for st in todo:
        globals()["stage_" + st]()


if __name__ == "__main__":
    main()
