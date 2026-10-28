"""H-A6: two-VAE reconstruction signature -> decoder family (kl-f8 / XL / other). Pre-declared NON-COUNTING diagnostic (docs/09 decision 1).

    env/.venv/bin/python -I experiments/024-attr-vae/run.py prep                      # data table, gates (sha256, families, split), writes results/prep.json
    env/.venv/bin/python -I experiments/024-attr-vae/run.py guard                     # D-008 pixel guard (sd-vae) + 32-image CPU/accelerator features
    env/.venv/bin/python -I experiments/024-attr-vae/run.py extract --vae sd-vae-ft-mse --device cuda [--subset cap]
    env/.venv/bin/python -I experiments/024-attr-vae/run.py analyze                   # labels, results.json, tables.md
All design constants are in config.json (frozen before any result).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT_ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_, "data", "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import perturb  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text())
RES = HERE / "results"
CACHE = HERE / "cache"
SEED = CFG["seed"]
GENS = CFG["generators"]
FAMS = CFG["family_names"]
CONDS = CFG["crop"]["conditions"]
CROP = CFG["crop"]["size"]
VAES = list(CFG["vaes"])
FEATS = CFG["features"]["per_vae"]
Image.MAX_IMAGE_PIXELS = None
TOL = 1e-3
BS_MAX = 4


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    RES.mkdir(exist_ok=True)
    with open(RES / "run.log", "a") as f:
        f.write(s + "\n")


def jdump(path, obj):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=default))


def jload(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


# ============================================================================= helpers reused from 021 (same code, imported)
def _load_021():
    spec = importlib.util.spec_from_file_location("run021", ROOT / "experiments" / "021-attr-residual" / "run.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["run021"] = m
    spec.loader.exec_module(m)
    return m


R21 = _load_021()
boot_weights, macro_recall, ci = R21.boot_weights, R21.macro_recall, R21.ci
split_families, check_grouped, check_dedupe = R21.split_families, R21.check_grouped, R21.check_dedupe
m1_features, zero_mean_rc, wiener_denoise = R21.m1_features, R21.zero_mean_rc, R21.wiener_denoise


# ============================================================================= feature math (pure numpy, device independent)
def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def hp8_feature(d):
    """d: (H, W) float error image (H, W multiples of 8). 3x3 high-pass, squared, mean by 8-px phase; CV over phases, rows and cols averaged.
    Returns (cv, phase_means[16] = 8 column phases then 8 row phases)."""
    d = np.asarray(d, np.float64)
    e = (d - ndimage.uniform_filter(d, 3, mode="reflect")) ** 2
    H, W = e.shape
    cols = e.mean(0).reshape(W // 8, 8).mean(0)  # position index mod 8 (anchored at the top-left = native grid)
    rows = e.mean(1).reshape(H // 8, 8).mean(0)
    cv = lambda v: float(v.std() / max(v.mean(), 1e-30))
    return 0.5 * (cv(cols) + cv(rows)), np.concatenate([cols, rows])


def vae_features(x, x1, x2):
    """x, x1, x2: (H, W, 3) arrays in [0,1] (x1, x2 are the quantised round trips). -> (4 log features, 16 phase means)."""
    x, x1, x2 = (np.asarray(a, np.float64) for a in (x, x1, x2))
    e1, e2 = rms(x - x1), rms(x1 - x2)
    cv, ph = hp8_feature((x - x1).mean(2))
    eps = 1e-12
    f = np.array([np.log(e1 + eps), np.log(e2 + eps), np.log((e1 + eps) / (e2 + eps)), np.log(cv + eps)])
    return f, ph


# ============================================================================= data
def load_tables():
    """Main set (1,020 fakes of the 8 generators) + secondary SD1.3/SD1.4 (600)."""
    man = pd.read_csv(ROOT / "data" / "manifest_raw.csv", keep_default_na=False)
    main = man[(man.label == 1) & man.generator.isin(GENS)].reset_index(drop=True)
    assert len(main) == 1020 and main.generator.nunique() == 8, (len(main), main.generator.nunique())
    rid, rid_src = R21.load_caption_ids(main)
    assert (rid != "").all()
    main["fam"] = rid
    main["g"] = main.generator.map({g: i for i, g in enumerate(GENS)})
    main["y"] = main.generator.map(lambda g: FAMS.index(CFG["family_map"][g]))
    sc = CFG["secondary"]["sd13_sd14"]
    sec = pd.read_csv(ROOT / sc["manifest"], keep_default_na=False)
    sec = sec[sec.generator.isin(sc["generators"])].reset_index(drop=True)
    sec["fam"] = sec.caption_id.astype(str)
    return main, sec, rid_src


def cap_subset(main, n_per_class, seed):
    """Seeded stratified (by 3-way family) row subsample, n_per_class per class (all rows of a class with fewer)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(main), bool)
    for k in range(len(FAMS)):
        idx = np.where(main.y.to_numpy() == k)[0]
        pick = idx if len(idx) <= n_per_class else rng.choice(idx, n_per_class, replace=False)
        keep[pick] = True
    return keep


def load_crop(path, sha, conds):
    """Decode (format from content), RGB, EXIF never used; top-left 512 crop on the native grid; condition applied via lib/perturb to the
    crop, identical for all classes; conditioned image cut top-left to a multiple of 8. Returns {cond: uint8 (H,W,3)}."""
    img = Image.open(ROOT / path)
    img.load()
    img = img.convert("RGB")
    w, h = img.size
    if w < CROP or h < CROP:
        raise RuntimeError(f"{path}: {img.size} < {CROP}")
    cr = img.crop((0, 0, CROP, CROP))
    out = {}
    for c in conds:
        im = perturb.apply_condition(cr, c, key=sha, seed=0)
        a = np.asarray(im)
        H, W = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
        out[c] = np.ascontiguousarray(a[:H, :W])
    return out


def base_feats_worker(args):
    """Per image: M1 texture stats + conditioned shape + sha1 of the clean crop pixels (dedupe), for all conditions."""
    path, sha = args
    arrs = load_crop(path, sha, CONDS)
    out = {}
    for c, a in arrs.items():
        gray = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
        r = zero_mean_rc(gray - wiener_denoise(gray))
        out[c] = (m1_features(a, gray, r), np.array(a.shape[:2], float), hashlib.sha1(a.tobytes()).hexdigest())
    return out


def stage_base(df, tag):
    f = CACHE / f"base_{tag}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["keys"]) == list(df.sha256):
            return {c: (z[f"M1_{c}"], z[f"shape_{c}"], z[f"H_{c}"]) for c in CONDS}
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(base_feats_worker, zip(df.path, df.sha256)))
    out = {c: (np.stack([r[c][0] for r in res]), np.stack([r[c][1] for r in res]), np.array([r[c][2] for r in res])) for c in CONDS}
    np.savez(f, keys=np.array(df.sha256), **{f"M1_{c}": out[c][0] for c in CONDS}, **{f"shape_{c}": out[c][1] for c in CONDS},
             **{f"H_{c}": out[c][2] for c in CONDS})
    return out


# ============================================================================= VAE
def get_dev(name):
    import torch
    if name == "cpu":
        nt = int(os.environ.get("NTHREADS", "8"))
        torch.set_num_threads(nt)
        return torch.device("cpu")
    from features import get_device  # disables TF32 on CUDA
    return get_device()


def load_vae(name, dev):
    from diffusers import AutoencoderKL
    c = CFG["vaes"][name]
    v = AutoencoderKL.from_pretrained(c["repo"], revision=c["revision"], local_files_only=True, torch_dtype=None)
    return v.to(dev).eval().requires_grad_(False)


def recon_f(vae, arr):
    """arr [B,H,W,3] uint8 -> float32 [B,H,W,3] in [0,1], clamped, NOT quantised (009 convention; posterior mean)."""
    import torch
    dev = next(vae.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample
        return ((y + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()


def q8(f):
    return np.rint(np.asarray(f, np.float32) * 255.0).clip(0, 255).astype(np.uint8)


class Batcher:
    """Run fn on slices with batch <= bs; on CUDA OOM (shared GPU) free memory, halve the batch / wait, retry."""

    def __init__(self, bs=BS_MAX):
        self.bs = bs

    def __call__(self, fn, arr):
        import torch
        out, i, tries = [], 0, 0
        while i < len(arr):
            try:
                out.append(fn(arr[i:i + self.bs]))
                i += self.bs
                tries = 0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tries += 1
                if self.bs > 1:
                    self.bs = max(1, self.bs // 2)
                    log(f"CUDA OOM -> batch {self.bs}")
                elif tries > 60:
                    raise
                else:
                    log(f"CUDA OOM at batch 1, wait 20 s (try {tries})")
                    time.sleep(20)
        return np.concatenate(out)


def features_for(vae, arrs, batcher=None):
    """arrs: list of uint8 (H,W,3) of one SHAPE -> (n,4) features, (n,16) phase means."""
    b = batcher or Batcher()
    A = np.stack(arrs)
    f1 = b(lambda a: recon_f(vae, a), A)
    q1 = q8(f1)
    f2 = b(lambda a: recon_f(vae, a), q1)
    q2 = q8(f2)
    F, P = [], []
    for i in range(len(A)):
        f, p = vae_features(A[i] / 255.0, q1[i] / 255.0, q2[i] / 255.0)
        F.append(f)
        P.append(p)
    return np.stack(F), np.stack(P)


def feat_file(tag, cond, vn, dev):
    return CACHE / f"f_{tag}_{cond}_{vn}_{dev}.npz"


def extract(df, tag, cond, vn, dev_name, vae, ckpt=64):
    """Features of every row of df for (cond, vae, device); resumable via a keyed npz. Same-shape groups are batched."""
    f = feat_file(tag, cond, vn, dev_name)
    have = {}
    if f.exists():
        z = np.load(f, allow_pickle=True)
        have = {k: (a, p) for k, a, p in zip(z["keys"], z["feats"], z["phases"])}
    todo = df[~df.sha256.isin(have)]
    t0, done = time.time(), 0
    if len(todo):
        log(f"extract {tag}/{cond}/{vn}/{dev_name}: {len(todo)} to do ({len(have)} cached)")
        b = Batcher()
        with ThreadPoolExecutor(3) as ex:
            for s in range(0, len(todo), ckpt):
                chunk = todo.iloc[s:s + ckpt]
                imgs = list(ex.map(lambda r: load_crop(r[0], r[1], [cond])[cond], zip(chunk.path, chunk.sha256)))
                by = {}
                for i, a in enumerate(imgs):
                    by.setdefault(a.shape, []).append(i)
                for shp, ii in by.items():
                    F, P = features_for(vae, [imgs[i] for i in ii], b)
                    for j, i in enumerate(ii):
                        have[chunk.sha256.iat[i]] = (F[j], P[j])
                done += len(chunk)
                keys = list(have)
                np.savez(str(f) + ".tmp.npz", keys=np.array(keys), feats=np.stack([have[k][0] for k in keys]),
                         phases=np.stack([have[k][1] for k in keys]))
                os.replace(str(f) + ".tmp.npz", f)
                log(f"  {tag}/{cond}/{vn}/{dev_name}: {done}/{len(todo)} {time.time() - t0:.0f}s")
    return np.stack([have[s][0] for s in df.sha256]), np.stack([have[s][1] for s in df.sha256])


# ============================================================================= read-outs
def make_lr(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def fit_lr(X, y, C):
    return make_lr(C).fit(X, y)


def argmin_a2(aedr_sd, aedr_xl):
    """No-training decoder rule on the two VAE candidates: 0 (kl-f8) if the sd-vae AEDR is the lower one else 1 (XL)."""
    return (aedr_xl < aedr_sd).astype(int)


def argmin_a3(aedr_sd, aedr_xl, tau):
    """3-way no-training rule: argmin of AEDR over {kl-f8, XL} if the minimum is <= tau else 'other' (code 2)."""
    m = np.minimum(aedr_sd, aedr_xl)
    return np.where(m <= tau, np.where(aedr_sd <= aedr_xl, 0, 1), 2)


def label_h_a6(clean_bacc, jr_clean_src, jr_matched_src, gates_ok):
    """docs/08 H-A6 + docs/09 (non-counting). PASS: clean >= 0.70 and jpeg75_resize05 >= 0.50 (clean-train source).
    KILL: jpeg75_resize05 < 0.45 under BOTH sources. INVALID if a gate failed."""
    if not gates_ok:
        return "INVALID"
    if clean_bacc >= 0.70 and jr_clean_src >= 0.50:
        return "PASS"
    if jr_clean_src < 0.45 and jr_matched_src < 0.45:
        return "KILL"
    return "INCONCLUSIVE"


def perm_control(Xtr, ytr, Xte, yte, C, n_perm, seed, K=3):
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for i in range(n_perm):
        m = fit_lr(Xtr, rng.permutation(ytr), C)
        out[i] = macro_recall(m.predict(Xte) == yte, yte, n_cls=K)
    return out


def check_d008_logits(model, Xa, Xb):
    """Gate D-008 (feature level): max |dlogit| of the fitted model between two feature sets of the same images."""
    return float(np.abs(model.decision_function(Xa) - model.decision_function(Xb)).max())


# ============================================================================= stages
def stage_prep():
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    main, sec, rid_src = load_tables()
    allh = pd.concat([main.sha256, sec.sha256]).tolist()
    check_dedupe(allh, "sha256 across main + SD1.3/1.4")
    base_m = stage_base(main, "main")
    base_s = stage_base(sec, "sec")
    for c in CONDS:
        check_dedupe(np.concatenate([base_m[c][2], base_s[c][2]]).tolist(), f"crop pixel hash [{c}] main+sec")
    train = split_families(main.fam.to_numpy(), CFG["split"]["train_family_frac"], CFG["split"]["primary_split_seed"])
    test = ~train
    check_grouped(main.fam.to_numpy(), train, test)
    info = {"n_main": len(main), "n_sec": len(sec), "caption_family_source": rid_src, "n_train": int(train.sum()), "n_test": int(test.sum()),
            "n_fam_train": int(len(set(main.fam[train]))), "n_fam_test": int(len(set(main.fam[test]))),
            "class_counts_train": {FAMS[k]: int(((main.y == k) & train).sum()) for k in range(3)},
            "class_counts_test": {FAMS[k]: int(((main.y == k) & test).sum()) for k in range(3)},
            "gates": {"sha256_unique": True, "crop_hash_unique": True, "split_grouped": True},
            "sec_families_overlap_main": int(len(set(sec.fam) & set(main.fam)))}
    jdump(RES / "prep.json", info)
    log("prep ok", json.dumps(info))
    return main, sec


def pick_guard_images(main):
    rng = np.random.default_rng(SEED)
    pick = []
    for k, n in ((0, 8), (1, 8), (2, 16)):
        idx = np.sort(np.where(main.y.to_numpy() == k)[0])
        pick += list(rng.choice(idx, n, replace=False))
    return main.iloc[sorted(pick)].reset_index(drop=True)


def stage_guard():
    """D-008. (1) pixel guard for sd-vae-ft-mse (CUDA vs CPU, 32 clean crops, x1 and x2 float outputs, abort if > 1e-3) + sdxl pixel numbers
    (informational; known fail in 009). (2) 32 images x 3 conditions x 2 VAEs: features on CPU and on CUDA saved for the logit-level check
    performed in analyze with the final clean-trained logreg."""
    import torch
    main, _ = load_tables()[:2]
    g32 = pick_guard_images(main)
    cdev, gdev = get_dev("cpu"), get_dev("cuda")
    out = jload(RES / "d008_pixel.json")
    arr = np.stack([load_crop(p, s, ["clean"])["clean"] for p, s in zip(g32.path, g32.sha256)])
    for vn in VAES[:1]:  # sd-vae-ft-mse only (see config d008.pixel)
        if vn in out:
            continue
        v = load_vae(vn, gdev)
        b = Batcher()
        f1g = b(lambda a: recon_f(v, a), arr)
        q1 = q8(f1g)
        f2g = b(lambda a: recon_f(v, a), q1)
        del v
        torch.cuda.empty_cache()
        vc = load_vae(vn, cdev)
        f1c = np.concatenate([recon_f(vc, arr[i:i + 2]) for i in range(0, len(arr), 2)])
        f2c = np.concatenate([recon_f(vc, q1[i:i + 2]) for i in range(0, len(arr), 2)])  # same quantised input on both devices
        d1, d2 = float(np.abs(f1g - f1c).max()), float(np.abs(f2g - f2c).max())
        out[vn] = {"n": len(arr), "max_abs_dpixel_x1": d1, "max_abs_dpixel_x2": d2, "tol": TOL, "ok": bool(max(d1, d2) <= TOL)}
        jdump(RES / "d008_pixel.json", out)
        log(f"D-008 pixel {vn}: x1 {d1:.2e} x2 {d2:.2e} -> {'ok' if out[vn]['ok'] else 'FAIL'}")
        del vc
    if not out["sd-vae-ft-mse"]["ok"]:
        raise SystemExit("D-008 ABORT: sd-vae-ft-mse CUDA vs CPU pixel diff > 1e-3")
    # feature-level files (resumable via extract caches, tag 'g32')
    for dev_name, dev in (("cuda", gdev), ("cpu", cdev)):
        for vn in VAES:
            v = load_vae(vn, dev)
            for c in CONDS:
                extract(g32, "g32", c, vn, dev_name, v, ckpt=4)
            del v
    log("guard features done")


def feats_matrix(tag, df, cond, devmap):
    """(n, 8) = [sd-vae 4 | sdxl 4] features; devmap: {vae: device_name}."""
    parts = []
    for vn in VAES:
        f = feat_file(tag, cond, vn, devmap[vn])
        z = np.load(f, allow_pickle=True)
        d = {k: a for k, a in zip(z["keys"], z["feats"])}
        parts.append(np.stack([d[s] for s in df.sha256]))
    return np.concatenate(parts, 1)


def stage_extract(vae_name, dev_name, subset):
    main, sec, _ = load_tables()
    if subset == "cap":
        n = CFG["fallback_cpu_cap"]["n_per_class_main"]
        keep = cap_subset(main, n, SEED)
        main = main[keep].reset_index(drop=True)
        n2 = CFG["fallback_cpu_cap"]["n_per_generator_sec"]
        rng = np.random.default_rng(SEED)
        parts = []
        for g, d in sec.groupby("generator"):
            parts.append(d.iloc[np.sort(rng.choice(len(d), min(n2, len(d)), replace=False))])
        sec = pd.concat(parts).reset_index(drop=True)
    dev = get_dev(dev_name)
    vae = load_vae(vae_name, dev)
    t0 = time.time()
    for tag, df in (("main", main), ("sec", sec)):
        for c in CONDS:
            extract(df, tag, c, vae_name, dev_name, vae)
    log(f"extract done {vae_name} {dev_name} subset={subset} in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- analysis
COLS = {"both": list(range(8)), "sd": [0, 1, 2, 3], "xl": [4, 5, 6, 7], "aedr_only": [2, 6], "e1_only": [0, 4]}


def eval_models(X, y, fam, train, test, cfg_C, n_boot, n_perm, seed, M1=None, shp=None, do_extra=True):
    """X: {cond: (n, 8)}. Returns results per (cond, source) for the 'both' 8-feature LR, plus sub-feature-sets, baselines."""
    K = 3
    tr, te = np.where(train)[0], np.where(test)[0]
    yte, fte = y[te], fam[te]
    cl = np.unique(fte, return_inverse=True)[1]
    w = boot_weights(cl, n_boot, seed + 11)
    out = {}
    for cond in X:
        for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
            fc = "clean" if src == "clean" else cond
            ent = {}
            for name, cols in COLS.items():
                if not do_extra and name != "both":
                    continue
                m = fit_lr(X[fc][tr][:, cols], y[tr], cfg_C)
                pred = m.predict(X[cond][te][:, cols])
                corr = pred == yte
                ent[name] = {"bacc": float(macro_recall(corr, yte, n_cls=K)), "bacc_ci": ci(macro_recall(corr, yte, w=w, n_cls=K))}
                if name == "both":
                    P = m.predict_proba(X[cond][te][:, cols])
                    cm = np.zeros((K, K), int)
                    np.add.at(cm, (yte, pred), 1)
                    ent[name].update({"recall": [float(corr[yte == k].mean()) for k in range(K)], "confusion": cm.tolist(), "acc": float(corr.mean()),
                                      "pred_share": [float((pred == k).mean()) for k in range(K)]})
                    from sklearn.metrics import roc_auc_score
                    ent[name]["auroc_ovr"] = [float(roc_auc_score(yte == k, P[:, k])) for k in range(K)]
            if do_extra:
                if M1 is not None:
                    m1 = fit_lr(M1[fc][tr], y[tr], cfg_C)
                    c1 = m1.predict(M1[cond][te]) == yte
                    ent["m1"] = {"bacc": float(macro_recall(c1, yte, n_cls=K)), "bacc_ci": ci(macro_recall(c1, yte, w=w, n_cls=K))}
                    b = macro_recall(X_corr(ent, X, y, tr, te, fc, cond, cfg_C), yte, w=w, n_cls=K)
                    ent["diff_vs_m1_ci"] = ci(b - macro_recall(c1, yte, w=w, n_cls=K))
                    ent["diff_vs_m1"] = ent["both"]["bacc"] - ent["m1"]["bacc"]
                if shp is not None:  # M0 on preprocessed tensors: shape of the conditioned crop only
                    m0 = fit_lr(shp[fc][tr], y[tr], cfg_C)
                    ent["m0_tensor"] = {"bacc": float(macro_recall(m0.predict(shp[cond][te]) == yte, yte, n_cls=K))}
                perm = perm_control(X[fc][tr], y[tr], X[cond][te], yte, cfg_C, n_perm, seed + 5)
                ent["perm"] = {"mean": float(perm.mean()), "ci": ci(perm), "max": float(perm.max()),
                               "p_ge_real": float((1 + (perm >= ent["both"]["bacc"]).sum()) / (1 + len(perm)))}
            out[f"{cond}|src={src}"] = ent
    return out


def X_corr(ent, X, y, tr, te, fc, cond, C):
    m = fit_lr(X[fc][tr], y[tr], C)
    return m.predict(X[cond][te]) == y[te]


def argmin_eval(X, y, fam, train, test, n_boot, seed):
    """No-training argmin rules. A2 on SD2/SDXL test images (chance 0.5); A3 3-way (tau = median min-AEDR over the unlabeled train-family pool)."""
    tr, te = np.where(train)[0], np.where(test)[0]
    out = {}
    cl = np.unique(fam[te], return_inverse=True)[1]
    w = boot_weights(cl, n_boot, seed + 11)
    for cond in X:
        sd, xl = X[cond][:, 2], X[cond][:, 6]
        tau = float(np.median(np.minimum(sd[tr], xl[tr])))
        yte = y[te]
        m2 = np.isin(yte, [0, 1])
        c2 = argmin_a2(sd[te], xl[te]) == yte
        a2 = macro_recall(c2[m2], yte[m2], n_cls=2)
        w2 = w[:, m2]
        a2ci = ci(macro_recall(c2[m2], yte[m2], w=w2, n_cls=2))
        c3 = argmin_a3(sd[te], xl[te], tau) == yte
        out[cond] = {"A2_bacc": float(a2), "A2_ci": a2ci, "A3_bacc": float(macro_recall(c3, yte, n_cls=3)),
                     "A3_ci": ci(macro_recall(c3, yte, w=w, n_cls=3)), "tau": tau,
                     "A3_recall": [float(c3[yte == k].mean()) for k in range(3)]}
    return out


def sec_eval(Xmain, y, train, Xsec, sec, n_boot, seed, cfg_C, fam_main_te_pred):
    """Fit on main TRAIN rows; predict SD1.3 / SD1.4. clean-source and matched-source. Share predicted per family, CI over caption families."""
    out = {}
    for cond in Xsec:
        for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
            fc = "clean" if src == "clean" else cond
            m = fit_lr(Xmain[fc][train], y[train], cfg_C)
            pred = m.predict(Xsec[cond])
            ent = {}
            for g, d in sec.groupby("generator"):
                ii = d.index.to_numpy()
                cl = np.unique(d.fam.to_numpy(), return_inverse=True)[1]
                w = boot_weights(cl, n_boot, seed + 17)
                sh = {FAMS[k]: float((pred[ii] == k).mean()) for k in range(3)}
                shci = {FAMS[k]: ci((w * (pred[ii] == k)).sum(1) / w.sum(1)) for k in range(3)}
                ent[g] = {"share": sh, "share_ci": shci, "n": int(len(ii))}
            out[f"{cond}|src={src}"] = ent
    return out


def stage_analyze():
    main, sec, rid_src = load_tables()
    prep = jload(RES / "prep.json")
    assert prep, "run prep first"
    y, fam = main.y.to_numpy(), main.fam.to_numpy()
    base_m = stage_base(main, "main")
    base_s = stage_base(sec, "sec")
    C = 1.0
    n_boot, n_perm = CFG["n_boot"], CFG["n_perm"]
    px = jload(RES / "d008_pixel.json")
    # ---- D-008 feature level, using the final model (clean-train, both VAEs, main train split) on the 32 guard images
    g32 = pick_guard_images(main)
    train = split_families(fam, CFG["split"]["train_family_frac"], CFG["split"]["primary_split_seed"])
    test = ~train
    check_grouped(fam, train, test)

    def dev_of(vn, sdxl_dev):
        return "cuda" if vn == "sd-vae-ft-mse" else sdxl_dev

    # decide sdxl device
    d008 = {"pixel": px, "feature_level": {}}
    # (a) use the accelerator for every VAE if its features were extracted and the logit guard passes
    modes = {}
    for sdxl_dev in ("cuda", "cpu"):
        dm = {v: dev_of(v, sdxl_dev) for v in VAES}
        ok_files = all(feat_file("main", c, v, dm[v]).exists() for c in CONDS for v in VAES)
        modes[sdxl_dev] = ok_files
    log("available feature sets:", modes)
    # logit guard needs the model: fit on the 'cuda' main features (sd-vae + sdxl on cuda) if available
    chosen = None
    if modes["cuda"]:
        dm = {v: dev_of(v, "cuda") for v in VAES}
        Xc = {c: feats_matrix("main", main, c, dm) for c in CONDS}
        m = fit_lr(Xc["clean"][train], y[train], C)
        items = {}
        for c in CONDS:
            Xg = {d: feats_matrix("g32", g32, c, {v: d for v in VAES}) for d in ("cuda", "cpu")}
            items[c] = {"max_abs_dlogit_both": check_d008_logits(m, Xg["cuda"], Xg["cpu"]),
                        "max_abs_dfeature": float(np.abs(Xg["cuda"] - Xg["cpu"]).max()),
                        "max_abs_dfeature_sdvae_cols": float(np.abs(Xg["cuda"][:, :4] - Xg["cpu"][:, :4]).max()),
                        "max_abs_dfeature_sdxl_cols": float(np.abs(Xg["cuda"][:, 4:] - Xg["cpu"][:, 4:]).max())}
            # sd-vae only on the accelerator, sdxl on CPU
            Xmix = np.concatenate([Xg["cuda"][:, :4], Xg["cpu"][:, 4:]], 1)
            items[c]["max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu"] = check_d008_logits(m, Xmix, Xg["cpu"])
        d008["feature_level"]["all_accelerator_vs_cpu"] = items
        ok_all = max(v["max_abs_dlogit_both"] for v in items.values()) <= TOL
        ok_sd = max(v["max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu"] for v in items.values()) <= TOL
        d008["feature_level"]["ok_all_accelerator"] = bool(ok_all)
        d008["feature_level"]["ok_sdvae_accelerator_sdxl_cpu"] = bool(ok_sd)
        if ok_all:
            chosen = "cuda"
    if chosen is None:
        if modes["cpu"]:
            chosen = "cpu"
        else:
            jdump(RES / "d008_guard.json", d008)
            raise SystemExit("D-008: all-accelerator features failed the feature-level guard and no CPU-sdxl features exist: run extract --vae sdxl-vae --device cpu --subset cap")
    jdump(RES / "d008_guard.json", d008)
    d008_ok = (chosen == "cuda")
    if not px.get("sd-vae-ft-mse", {}).get("ok", False):
        d008_ok = False
    log(f"sdxl features from: {chosen}; D-008 ok: {d008_ok}")
    dm = {v: dev_of(v, chosen) for v in VAES}
    capped = chosen == "cpu"
    sd_dev_note = "cuda"
    if capped:  # sd-vae features on the CPU for the capped subset, if complete: then the whole analysis set is CPU and D-008 is moot for it
        keep0 = cap_subset(main, CFG["fallback_cpu_cap"]["n_per_class_main"], SEED)
        try:
            for c in CONDS:
                for tag, dfx in (("main", main[keep0]),):
                    z = np.load(feat_file(tag, c, "sd-vae-ft-mse", "cpu"), allow_pickle=True)
                    assert set(dfx.sha256) <= set(z["keys"])
            dm["sd-vae-ft-mse"] = "cpu"
            sd_dev_note = "cpu"
        except Exception:
            pass
    keep = cap_subset(main, CFG["fallback_cpu_cap"]["n_per_class_main"], SEED) if capped else np.ones(len(main), bool)
    sec_keep = np.ones(len(sec), bool)
    if capped:
        rng = np.random.default_rng(SEED)
        sec_keep[:] = False
        n2 = CFG["fallback_cpu_cap"]["n_per_generator_sec"]
        for g, d in sec.groupby("generator"):
            sec_keep[d.index[np.sort(rng.choice(len(d), min(n2, len(d)), replace=False))]] = True
    mdf = main[keep].reset_index(drop=True)
    sdf = sec[sec_keep].reset_index(drop=True)
    X = {c: feats_matrix("main", mdf, c, dm) for c in CONDS}
    Xsec = {c: feats_matrix("sec", sdf, c, dm) for c in CONDS}
    ym, fm = mdf.y.to_numpy(), mdf.fam.to_numpy()
    tr_m = split_families(fm, CFG["split"]["train_family_frac"], CFG["split"]["primary_split_seed"])
    te_m = ~tr_m
    check_grouped(fm, tr_m, te_m)
    pos = {s: i for i, s in enumerate(main.sha256)}
    ix = np.array([pos[s] for s in mdf.sha256])
    M1 = {c: base_m[c][0][ix] for c in CONDS}
    SHP = {c: base_m[c][1][ix] for c in CONDS}
    if chosen == "cpu" and sd_dev_note == "cpu":
        d008_ok = bool(px.get("sd-vae-ft-mse", {}).get("ok", False))
        d008["note_all_cpu"] = "capped analysis set: both VAEs on CPU -> no accelerator features in the model; the accelerator-vs-CPU feature guard failed for sdxl (0.15 logit) and marginally for sd-vae (1.1e-3..3.0e-3 vs 1e-3), which is why the sd-vae features were recomputed on CPU for the subset"
    elif chosen == "cpu":  # D-008 for the features actually used: sd-vae on the accelerator + sdxl on CPU vs everything on CPU, final (capped) model
        m_fin = fit_lr(X["clean"][tr_m], ym[tr_m], C)
        items = {}
        for c in CONDS:
            Xg = {d: feats_matrix("g32", g32, c, {v: d for v in VAES}) for d in ("cuda", "cpu")}
            Xmix = np.concatenate([Xg["cuda"][:, :4], Xg["cpu"][:, 4:]], 1)
            items[c] = {"max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu": check_d008_logits(m_fin, Xmix, Xg["cpu"]),
                        "max_abs_dfeature_sdvae_cols": float(np.abs(Xg["cuda"][:, :4] - Xg["cpu"][:, :4]).max())}
        d008["feature_level"]["final_model_sdvae_gpu_sdxl_cpu"] = items
        d008_ok = max(v["max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu"] for v in items.values()) <= TOL
        if not px.get("sd-vae-ft-mse", {}).get("ok", False):
            d008_ok = False
        jdump(RES / "d008_guard.json", d008)
        log(f"D-008 (final model, sd-vae GPU + sdxl CPU vs all CPU): ok={d008_ok}")
    log(f"analysis set: {len(mdf)} rows (capped={capped}); train {tr_m.sum()} / test {te_m.sum()}")
    main_res = eval_models(X, ym, fm, tr_m, te_m, C, n_boot, n_perm, SEED, M1=M1, shp=SHP)
    am = argmin_eval(X, ym, fm, tr_m, te_m, n_boot, SEED)
    # sensitivity over split seeds (point estimates, 'both', clean source)
    sens = {}
    for s in CFG["split"]["sensitivity_split_seeds"]:
        tr_s = split_families(fm, CFG["split"]["train_family_frac"], s)
        te_s = ~tr_s
        for cond in CONDS:
            for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
                fc = "clean" if src == "clean" else cond
                m = fit_lr(X[fc][tr_s], ym[tr_s], C)
                sens.setdefault(f"{cond}|src={src}", []).append(float(macro_recall(m.predict(X[cond][te_s]) == ym[te_s], ym[te_s], n_cls=3)))
    sens = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)} for k, v in sens.items()}
    sec_res = sec_eval(X, ym, tr_m, Xsec, sdf.assign(fam=sdf.fam), n_boot, SEED, C, None)
    # reference rates on the main test set for the mapping reading
    ref = {}
    for cond in CONDS:
        for src in (["clean", "matched"] if cond != "clean" else ["clean"]):
            fc = "clean" if src == "clean" else cond
            m = fit_lr(X[fc][tr_m], ym[tr_m], C)
            pred = m.predict(X[cond][te_m])
            yt = ym[te_m]
            ref[f"{cond}|src={src}"] = {"SD2_to_klf8": float((pred[yt == 0] == 0).mean()), "SDXL_to_klf8": float((pred[yt == 1] == 0).mean()),
                                        "other_to_klf8": float((pred[yt == 2] == 0).mean()), "n_test_per_class": [int((yt == k).sum()) for k in range(3)]}
    # full-set sd-vae-only models when capped (the sd-vae features exist for all rows)
    sd_full = None
    if capped:
        pos_all = {s: i for i, s in enumerate(main.sha256)}
        Xf = {c: feats_matrix_sd("main", main, c) for c in CONDS}
        trf = split_families(fam, CFG["split"]["train_family_frac"], CFG["split"]["primary_split_seed"])
        sd_full = eval_models({c: np.concatenate([Xf[c], np.zeros((len(main), 4))], 1) for c in CONDS}, y, fam, trf, ~trf, C, n_boot, 0, SEED, do_extra=False)
    # ---- gates and labels
    perm_ok = all(abs(e["perm"]["mean"] - 1 / 3) < 0.05 for e in main_res.values())
    m0_ok = all(abs(e["m0_tensor"]["bacc"] - 1 / 3) < 0.06 for e in main_res.values())
    gates = {"sha256_unique": prep["gates"]["sha256_unique"], "crop_hash_unique": prep["gates"]["crop_hash_unique"], "split_grouped": True,
             "permutation_at_chance": bool(perm_ok), "m0_on_tensors_at_chance": bool(m0_ok), "d008_ok": bool(d008_ok)}
    gates_ok = all(gates.values())
    cb = main_res["clean|src=clean"]["both"]["bacc"]
    jc = main_res["jpeg75_resize05|src=clean"]["both"]["bacc"]
    jm = main_res["jpeg75_resize05|src=matched"]["both"]["bacc"]
    label = label_h_a6(cb, jc, jm, gates_ok)
    res = {"label": label, "label_inputs": {"clean_bacc": cb, "jpeg75_resize05_clean_src": jc, "jpeg75_resize05_matched_src": jm},
           "gates": gates, "capped": capped, "n_analysis": len(mdf), "sdxl_feature_device": chosen, "sdvae_feature_device_in_capped_set": sd_dev_note, "main": main_res, "argmin": am, "sensitivity_seeds": sens,
           "secondary_sd13_sd14": sec_res, "reference_rates_main_test": ref, "sd_vae_only_full_set": sd_full, "d008": d008,
           "chance": {"bacc_3way": 1 / 3, "A2": 0.5, "majority_class_rate": float((ym == 2).mean())}, "prep": prep}
    jdump(RES / "results.json", res)
    write_tables(res)
    log("analyze done: label", label)


def feats_matrix_sd(tag, df, cond):
    z = np.load(feat_file(tag, cond, "sd-vae-ft-mse", "cuda"), allow_pickle=True)
    d = {k: a for k, a in zip(z["keys"], z["feats"])}
    return np.stack([d[s] for s in df.sha256])


def fmt(e, k="bacc"):
    return f"{e[k]:.3f} [{e[k + '_ci'][0]:.3f}, {e[k + '_ci'][1]:.3f}]"


def write_tables(res):
    L = []
    w = L.append
    w(f"# H-A6 results (non-counting diagnostic)\n\n**Label: {res['label']}**  inputs {json.dumps(res['label_inputs'])}\n")
    w(f"Gates: {json.dumps(res['gates'])}. sdxl features from `{res['sdxl_feature_device']}`, sd-vae features (capped set) from `{res['sdvae_feature_device_in_capped_set']}`; capped={res['capped']}; n={res['n_analysis']}. Chance bacc 0.333.\n")
    w("## 3-way bacc (95% cluster-bootstrap CI over caption families)\n")
    w("| condition | source | 8-feat LR | sd-vae only | sdxl only | AEDR-2 | e1-2 | M1 texture | diff vs M1 [CI] | M0 tensor | perm mean (p) | recalls kl-f8/XL/other |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, e in res["main"].items():
        c, s = k.split("|src=")
        b = e["both"]
        w(f"| {c} | {s} | {fmt(b)} | {fmt(e['sd'])} | {fmt(e['xl'])} | {fmt(e['aedr_only'])} | {fmt(e['e1_only'])} | {fmt(e['m1'])} | "
          f"{e['diff_vs_m1']:+.3f} [{e['diff_vs_m1_ci'][0]:+.3f}, {e['diff_vs_m1_ci'][1]:+.3f}] | {e['m0_tensor']['bacc']:.3f} | "
          f"{e['perm']['mean']:.3f} ({e['perm']['p_ge_real']:.3f}) | {'/'.join(f'{r:.2f}' for r in b['recall'])} |")
    w("\n## Confusion matrices (rows = true kl-f8, XL, other; cols = predicted)\n")
    for k, e in res["main"].items():
        w(f"- {k}: {e['both']['confusion']}  AUROC one-vs-rest {[round(a, 3) for a in e['both']['auroc_ovr']]}")
    w("\n## Argmin rules (no training)\n")
    w("| condition | A2 SD2-vs-SDXL bacc (chance 0.5) | A3 3-way bacc | tau | A3 recalls |\n|---|---|---|---|---|")
    for c, e in res["argmin"].items():
        w(f"| {c} | {e['A2_bacc']:.3f} [{e['A2_ci'][0]:.3f}, {e['A2_ci'][1]:.3f}] | {e['A3_bacc']:.3f} [{e['A3_ci'][0]:.3f}, {e['A3_ci'][1]:.3f}] | {e['tau']:.3f} | {'/'.join(f'{r:.2f}' for r in e['A3_recall'])} |")
    w("\n## Split-seed sensitivity (8-feat LR, point estimates over seeds 1..10)\n")
    for k, v in res["sensitivity_seeds"].items():
        w(f"- {k}: mean {v['mean']:.3f} sd {v['sd']:.3f} range [{v['min']:.3f}, {v['max']:.3f}]")
    w("\n## Secondary: SD1.3 / SD1.4 (not in training); share predicted per family [CI over caption families]\n")
    w("| condition | source | generator | n | kl-f8 | XL | other |\n|---|---|---|---|---|---|---|")
    for k, e in res["secondary_sd13_sd14"].items():
        c, s = k.split("|src=")
        for g, d in e.items():
            cells = " | ".join(f"{d['share'][f]:.2f} [{d['share_ci'][f][0]:.2f}, {d['share_ci'][f][1]:.2f}]" for f in FAMS)
            w(f"| {c} | {s} | {g} | {d['n']} | {cells} |")
    w("\nReference (main test set, same model): share predicted kl-f8 for true SD2 / SDXL / other\n")
    for k, r in res["reference_rates_main_test"].items():
        w(f"- {k}: SD2 {r['SD2_to_klf8']:.2f}, SDXL {r['SDXL_to_klf8']:.2f}, other {r['other_to_klf8']:.2f} (n_test per class {r['n_test_per_class']})")
    if res["sd_vae_only_full_set"]:
        w("\n## sd-vae only, full 1,020-row set (capped run)\n")
        for k, e in res["sd_vae_only_full_set"].items():
            w(f"- {k}: sd-vae 4-feat bacc {fmt(e['both'])}  (8-col model with the sdxl columns zeroed = sd-vae only)")
    w("\n## D-008\n```\n" + json.dumps(res["d008"], indent=1) + "\n```\n")
    (RES / "tables.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "guard", "extract", "analyze"])
    ap.add_argument("--vae", choices=VAES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", default="all", choices=["all", "cap"])
    a = ap.parse_args()
    if a.stage == "prep":
        stage_prep()
    elif a.stage == "guard":
        stage_guard()
    elif a.stage == "extract":
        stage_extract(a.vae, a.device, a.subset)
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
