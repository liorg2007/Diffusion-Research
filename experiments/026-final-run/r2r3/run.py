"""026-final-run/r2r3: R2 (mechanism: what image information carries the mid-level attribution signal) and
R3 (laundering stress map) of docs/11-final-run-plan.md. Pre-registration: docs/11; constants: config.json.

    HF_HOME=data/hf_cache HF_HUB_OFFLINE=1 env/.venv/bin/python -I experiments/026-final-run/r2r3/run.py all [--workers 12]
Stages (each resumable; caches in cache/, git-ignored):
  guards   parity of re-extracted blocks 4/8/12 with the 004 cache (16 images, 5 conditions) + D-008 GPU vs CPU (32 images)
  extract  DINOv2-base features, all 12 blocks CLS+mean: 5 manipulations x 4 conditions (R2) and 30 grid cells (R3)
  fit      T-mix probes: R2 (manip x block x fold), R3 extended-T-mix probes (mid blocks x fold), permutation control
  report   bootstrap, verdicts -> results/results.json, results/tables.md
Data, folds, probe, bootstrap: imported from experiments/020-attr-foundation/run.py (never modified).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

CFG = json.loads((HERE / "config.json").read_text())
CACHE, RES = HERE / "cache", HERE / "results"
SEED, B = CFG["seed"], CFG["n_boot"]
K = len(CFG["classes"])
CHANCE = 1.0 / K
MIDS, LAST = CFG["mid_blocks"], CFG["last_block"]
TRAIN = CFG["train_conds"]
GRID_Q, GRID_F = CFG["grid"]["q"], CFG["grid"]["f"]
GRID_CELLS = [(q, f) for q in GRID_Q for f in GRID_F]
EXT_EXTRA = CFG["extended_tmix_extra"]
MANIPS = list(CFG["manipulations"])
warnings.filterwarnings("ignore")


def log(*a):
    print(time.strftime("%H:%M:%S"), "r2r3", *a, flush=True)


def cell_cond(q, f):
    return f"jr({q},{f})"


GRID_CONDS = [cell_cond(q, f) for q, f in GRID_CELLS]


def safe(s):
    return "".join(c if c.isalnum() or c in "-_.+()" else "_" for c in s)


# ============================================================================ 020 imports
def _load020():
    if "run020" in sys.modules:
        return sys.modules["run020"]
    p = ROOT / "experiments" / "020-attr-foundation" / "run.py"
    spec = importlib.util.spec_from_file_location("run020", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["run020"] = m
    spec.loader.exec_module(m)
    # block-feature names for 020.inner_table/grid_for (they use `fs in SL` to choose the dino C grid); getX is ours
    for b in CFG["blocks"]:
        m.SL[f"blk{b}"] = slice(0, 1)
    assert m.CFG["c_grid"]["dino"] == CFG["c_grid"], "C grid differs from 020"
    return m


R20 = None


def r20():
    global R20
    if R20 is None:
        os.chdir(ROOT)
        R20 = _load020()
    return R20


# ============================================================================ image pipeline
def tile_shuffle(img, key, tile=32):
    from perturb import seed_from_key
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    assert h % tile == 0 and w % tile == 0
    th, tw = h // tile, w // tile
    tiles = a.reshape(th, tile, tw, tile, 3).transpose(0, 2, 1, 3, 4).reshape(th * tw, tile, tile, 3)
    perm = np.random.default_rng(seed_from_key("tileshuffle:" + str(key), 0)).permutation(th * tw)
    out = tiles[perm].reshape(th, tw, tile, tile, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 3)
    return Image.fromarray(np.ascontiguousarray(out))


def _blur(a, sigma=1.5):
    return gaussian_filter(a.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)


def blur(img, sigma=1.5):
    return Image.fromarray(np.clip(np.rint(_blur(np.asarray(img.convert("RGB")), sigma)), 0, 255).astype(np.uint8))


def highpass(img, sigma=1.5):
    a = np.asarray(img.convert("RGB"))
    r = a.astype(np.float32) - _blur(a, sigma) + 128.0
    return Image.fromarray(np.clip(np.rint(r), 0, 255).astype(np.uint8))


def gray3(img):
    g = np.asarray(img.convert("L"))
    return Image.fromarray(np.stack([g, g, g], -1))


def apply_manip(img224, manip, key):
    """Manipulation of the 224 model input. Depends on (image, key) only: never on the class label."""
    if manip == "none":
        return img224.convert("RGB")
    if manip == "a_shuffle":
        return tile_shuffle(img224, key)
    if manip == "b_blur":
        return blur(img224)
    if manip == "c_highpass":
        return highpass(img224)
    if manip == "d_gray":
        return gray3(img224)
    raise ValueError(manip)


def crop224(img, size=224):
    """Laundered image -> 224 centre crop. Short side < 224 (resize factors 0.35/0.25): bicubic upsample to short side 224 first."""
    from preprocess import center_crop
    img = img.convert("RGB")
    m = min(img.size)
    if m < size:
        s = size / m
        img = img.resize((max(size, round(img.size[0] * s)), max(size, round(img.size[1] * s))), Image.BICUBIC)
    return center_crop(img, size)


def model_input(path, sha, cond, manip):
    """The one code path for every class, train and test: PNG -> laundering (perturb.apply_condition) -> 224 crop -> manipulation."""
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        out = apply_condition(im, cond, key=sha, seed=0)
    return apply_manip(crop224(out), manip, sha)


# ============================================================================ embedding
class Embedder:
    """DINOv2-base via features.load_backbone, all 12 blocks, CLS + mean (= the 004 layout per block). Small batches, OOM retry."""

    def __init__(self, bb=None, device=None, batch=None):
        import torch
        self.torch = torch
        if bb is None:
            from features import load_backbone
            bb = load_backbone(CFG["backbone"], input_size=224, layers=tuple(CFG["blocks"]), use_cls=True, use_mean=True)
        self.bb = bb
        self.batch = batch or CFG["batch"]
        self.dev = self.torch.device(device) if device else self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        if self.dev.type == "cuda":
            self.torch.backends.cuda.matmul.allow_tf32 = False
            self.torch.backends.cudnn.allow_tf32 = False
        self.bb.model.to(self.dev)
        self.n_oom = 0

    def _fwd(self, imgs):
        torch = self.torch
        x = torch.stack([self.bb.to_tensor(im) for im in imgs]).to(self.dev)
        with torch.no_grad():
            f = self.bb.forward(x)
        return f.float().cpu().numpy().reshape(len(imgs), len(CFG["blocks"]), -1)

    def embed(self, imgs, max_retry=8):
        out, i, bs, tries = [], 0, self.batch, 0
        while i < len(imgs):
            try:
                out.append(self._fwd(imgs[i:i + bs]))
                i += bs
                tries = 0
            except (RuntimeError, self.torch.cuda.OutOfMemoryError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                self.n_oom += 1
                tries += 1
                if tries > max_retry:
                    raise
                if self.dev.type == "cuda":
                    self.torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                log(f"OOM -> batch {bs}, retry {tries}")
                time.sleep(3 * tries)
        return np.concatenate(out) if out else np.zeros((0, len(CFG["blocks"]), 0), np.float32)


# ============================================================================ data
def load_data():
    """Rows, labels, groups, folds of the 020 H-A1 main analysis (K=8), reproduced and checked against the committed folds."""
    import pandas as pd
    r = r20()
    df, _ = r.load_universe("fov512")
    fj = json.loads((ROOT / CFG["folds_file"]).read_text())
    shas, fold = fj["rows"], np.array(fj["fold"])
    pos = {s: i for i, s in enumerate(df.sha256.tolist())}
    idx = np.array([pos[s] for s in shas])
    d = df.iloc[idx].reset_index(drop=True)
    cls = {c: i for i, c in enumerate(CFG["classes"])}
    assert (d.pool == "fake").all() and set(d.generator) <= set(cls)
    y = np.array([cls[g] for g in d.generator])
    groups = d.grp.to_numpy().astype(str)
    commit = json.loads((ROOT / CFG["folds_commit_file"]).read_text())["ha1_main"]
    assert hashlib.sha256(fold.tobytes()).hexdigest() == commit["sha256_of_folds"], "folds differ from the committed ones"
    assert (r.make_folds(y, groups, CFG["outer_folds"], SEED) == fold).all(), "recomputed folds differ"
    chk = r.check_grouped_folds(fold, groups)
    assert chk["ok"], chk
    assert (~d.dup_drop).all()
    return dict(df=d, y=y, groups=groups, fold=fold, shas=shas, paths=d.path.tolist(), folds_sha=commit["sha256_of_folds"])


# ============================================================================ feature extraction (cached, resumable)
def feat_file(manip, cond):
    return CACHE / f"feat_{manip}_{safe(cond)}.npy"


def extract_file(emb, D, manip, cond, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    f = feat_file(manip, cond)
    if f.exists():
        return False
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        imgs = list(ex.map(lambda ps: model_input(ps[0], ps[1], cond, manip), zip(D["paths"], D["shas"])))
    X = np.concatenate([emb.embed(imgs[i:i + 128]) for i in range(0, len(imgs), 128)])
    assert X.shape == (len(imgs), len(CFG["blocks"]), 1536) and np.isfinite(X).all()
    tmp = f.with_name(f.name + ".tmp.npy")
    np.save(tmp, X.astype(np.float32))
    os.replace(tmp, f)
    log(f"extract {manip} {cond}: {len(imgs)} images in {time.time() - t0:.0f}s (oom retries so far {emb.n_oom})")
    return True


def stage_extract(D):
    CACHE.mkdir(exist_ok=True)
    (CACHE / "rows.json").write_text(json.dumps(dict(shas=D["shas"], cfg=CFG["backbone"], folds_sha=D["folds_sha"])))
    emb = Embedder()
    jobs = [(m, c) for m in MANIPS for c in CFG["r2_conds"]] + [("none", c) for c in GRID_CONDS]
    for m, c in jobs:
        extract_file(emb, D, m, c)
    return emb


def getfeat(manip, cond):
    return np.load(feat_file(manip, cond), mmap_mode="r")


# ============================================================================ guards
def scaled_diff(A, C):
    d = np.abs(A - C)
    return float((d / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max())


def pick_per_class(D, n_per, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(K):
        rows = np.where(D["y"] == k)[0]
        out += sorted(rng.choice(rows, n_per, replace=False).tolist())
    return out


def stage_guards(D):
    import torch
    r = r20()
    torch.set_num_threads(8)
    G = CFG["guards"]
    out = {}
    emb_gpu = Embedder()
    assert emb_gpu.dev.type == "cuda", "no GPU"
    # ---------------- parity with the 004 cache --------------------------------------------------
    rows = pick_per_class(D, G["parity_n"] // K, SEED)
    par = {}
    for cond in G["parity_conds"]:
        store = r.read_caches(G["parity_caches"], cond)
        imgs = [model_input(D["paths"][i], D["shas"][i], cond, "none") for i in rows]
        F = emb_gpu.embed(imgs)                                           # [n, 12, 1536]
        ref = np.stack([store[D["shas"][i]] for i in rows])               # [n, 4608] = blk4|blk8|blk12 (CLS|mean)
        mine = np.concatenate([F[:, b - 1, :] for b in G["parity_blocks"]], 1)
        par[cond] = dict(max_abs=float(np.abs(mine - ref).max()), scaled=scaled_diff(mine, ref))
    par_ok = all(v["scaled"] <= G["parity_tol_scaled"] for v in par.values())
    out["parity"] = dict(n=len(rows), conds=par, tol_scaled=G["parity_tol_scaled"], ok=bool(par_ok))
    log("parity", json.dumps(out["parity"]))
    (RES / "guards.json").write_text(json.dumps(out, indent=1))
    if not par_ok:
        raise SystemExit("PARITY FAILED: stop (do not extract)")
    # ---------------- D-008: GPU vs CPU on 32 images, two CPU runs for the run-to-run spread -----------
    rows32 = pick_per_class(D, G["d008_n"] // K, SEED + 1)
    tr = np.where(D["fold"] != 0)[0]
    store = r.read_caches(G["parity_caches"], CFG["primary"])
    Xtr = np.stack([store[D["shas"][i]][1536:3072] for i in tr])         # blk8 CLS|mean, jpeg75_resize05
    sc, clf = r.fit_model("lr", Xtr, D["y"][tr], 0.01)
    logit = lambda F: clf.decision_function(sc.transform(F[:, 7, :]))
    emb_cpu1 = Embedder(bb=emb_gpu.bb, device="cpu", batch=16)
    res = {}
    for var in G["d008_variants"]:
        manip, cond = var.split("@")
        imgs = [model_input(D["paths"][i], D["shas"][i], cond, manip) for i in rows32]
        emb_cpu1.bb.model.to("cpu")
        torch.set_num_threads(8)
        C1 = emb_cpu1.embed(imgs)
        emb_cpu1.batch = 5
        torch.set_num_threads(4)
        C2 = emb_cpu1.embed(imgs)
        emb_cpu1.batch = 16
        torch.set_num_threads(8)
        emb_gpu.bb.model.to(emb_gpu.dev)
        Gp = emb_gpu.embed(imgs)
        sp_f, sp_l = scaled_diff(C2.reshape(len(rows32), -1), C1.reshape(len(rows32), -1)), float(np.abs(logit(C2) - logit(C1)).max())
        gf, gl = scaled_diff(Gp.reshape(len(rows32), -1), C1.reshape(len(rows32), -1)), float(np.abs(logit(Gp) - logit(C1)).max())
        tf, tl = max(1e-3, 2 * sp_f), max(1e-3, 2 * sp_l)
        res[var] = dict(cpu_spread_feat=sp_f, cpu_spread_logit=sp_l, gpu_cpu_feat=gf, gpu_cpu_logit=gl, tol_feat=tf, tol_logit=tl,
                        ok=bool(gf <= tf and gl <= tl))
        log("D-008", var, json.dumps(res[var]))
    out["d008"] = dict(n=len(rows32), variants=res, ok=all(v["ok"] for v in res.values()),
                       max_gpu_cpu_logit=max(v["gpu_cpu_logit"] for v in res.values()), max_gpu_cpu_feat=max(v["gpu_cpu_feat"] for v in res.values()))
    (RES / "guards.json").write_text(json.dumps(out, indent=1))
    if not out["d008"]["ok"]:
        raise SystemExit("D-008 FAILED: stop")
    return out


# ============================================================================ probe tasks
class View:
    """Duck-typed 020 `Analysis` for the manipulation `manip`: getX(fs='blkN', cond) -> [n, 1536] block features."""

    def __init__(self, D, manip):
        self.y, self.groups, self.K, self.fold, self.manip = D["y"], D["groups"], K, D["fold"], manip

    def getX(self, fs, cond):
        return getfeat(self.manip, cond)[:, int(fs[3:]) - 1, :]


_D = {}


def fit_eval_task(args):
    """(tag, manip, block, fold, train_conds, eval_conds): inner grouped CV for C on the TRAIN folds only, fit, score the test fold."""
    tag, manip, block, f, train_conds, eval_conds = args
    r = r20()
    D = _D["D"]
    A = View(D, manip)
    fs = f"blk{block}"
    tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
    r.assert_disjoint_groups(A.groups, tr, te, f"{tag} {manip} blk{block} fold {f}")
    assert not set(train_conds) & set(CFG["held_out_launderings"]), "held-out laundering used in training"
    table, Cb = r.inner_table(A, fs, tr, list(train_conds), CFG["sel_cond"], SEED * 1000 + f + 1, list(range(K)))
    m = r.fit_model("lr", *r.stack(A, fs, tr, list(train_conds)), Cb)
    lp = {c: r.logp_full(m, A.getX(fs, c)[te], K) for c in eval_conds}
    return dict(tag=tag, manip=manip, block=block, f=f, te=te, C=Cb, inner_best=table[Cb], logp=lp)


def perm_task(args):
    manip, s, f = args
    r = r20()
    D, P = _D["D"], CFG["perm_control"]
    A = View(D, manip)
    yp = np.random.default_rng([SEED, 4242, s]).permutation(A.y)
    tr, te = np.where(A.fold != f)[0], np.where(A.fold == f)[0]
    r.assert_disjoint_groups(A.groups, tr, te, "perm")
    fs = f"blk{P['block']}"
    X = np.concatenate([A.getX(fs, c)[tr] for c in TRAIN])
    m = r.fit_model("lr", X, np.tile(yp[tr], len(TRAIN)), P["C"])
    pred = r.logp_full(m, A.getX(fs, CFG["primary"])[te], K).argmax(1)
    return s, te, pred.astype(np.int8)


def run_cached(name, tasks, fn, workers):
    f = CACHE / f"task_{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    res = r20().run_tasks(fn, tasks, workers, name)
    f.write_bytes(pickle.dumps(res, protocol=4))
    return res


def stage_fit(D, workers):
    _D["D"] = D
    r20()
    out = {}
    for manip in MANIPS:
        tasks = []
        for b in CFG["blocks"]:
            ev = list(CFG["r2_conds"]) + (GRID_CONDS if (manip == "none" and b in MIDS) else [])
            for f in range(CFG["outer_folds"]):
                tasks.append(("r2", manip, b, f, TRAIN, ev))
        out[("r2", manip)] = run_cached(f"r2_{manip}", tasks, fit_eval_task, workers)
    tasks = [("ext", "none", b, f, TRAIN + EXT_EXTRA, list(CFG["r2_conds"]) + GRID_CONDS) for b in MIDS for f in range(CFG["outer_folds"])]
    out[("ext", "none")] = run_cached("ext_none", tasks, fit_eval_task, workers)
    P = CFG["perm_control"]
    perm = {}
    for manip in P["manips"]:
        tasks = [(manip, s, f) for s in range(P["n"]) for f in range(CFG["outer_folds"])]
        perm[manip] = run_cached(f"perm_{manip}", tasks, perm_task, workers)
    return out, perm


# ============================================================================ analysis (pure functions; used by selftest)
def pool_logp(results, block, conds, n):
    out = {c: np.full((n, K), np.nan, np.float32) for c in conds}
    for t in results:
        if t["block"] == block:
            for c in conds:
                out[c][t["te"]] = t["logp"][c]
    for c in conds:
        assert not np.isnan(out[c]).any(), (block, c)
    return out


def nested_mid(results, conds, n, mids=None):
    """Per outer fold choose the block among `mids` (max inner-CV bacc, ties -> lower block) from TRAIN data only."""
    mids = mids or MIDS
    out = {c: np.full((n, K), np.nan, np.float32) for c in conds}
    choice = {}
    folds = sorted({t["f"] for t in results})
    for f in folds:
        cand = [t for t in results if t["f"] == f and t["block"] in mids]
        best = max(cand, key=lambda t: (round(t["inner_best"], 12), -t["block"]))
        choice[f] = (best["block"], best["C"])
        for c in conds:
            out[c][best["te"]] = best["logp"][c]
    for c in conds:
        assert not np.isnan(out[c]).any()
    return out, choice


def bacc_boot(Wrow, y, logp):
    r = r20()
    return r.conf_stats(r.boot_conf(Wrow, y, logp.argmax(1), K))["bacc"]


def ci(arr):
    """[point (replicate 0), 2.5%, 97.5%] over replicates 1..B; NaN replicates ignored."""
    arr = np.asarray(arr, float)
    lo, hi = np.nanpercentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def retained(b_m, b_none, chance=CHANCE):
    return (b_m - chance) / (b_none - chance)


def verdict_threshold(c, thr, direction):
    """c = [pt, lo, hi]. direction 'ge' (claim: value >= thr) or 'lt' (claim: value < thr). -> (confirmed?, ci_robust?)."""
    pt, lo, hi = c
    if direction == "ge":
        return bool(pt >= thr), bool(lo >= thr if pt >= thr else hi < thr)
    return bool(pt < thr), bool(hi < thr if pt < thr else lo >= thr)


# ---- R3 surface metrics: G is [R, nq, nf] (replicates x quality index x resize index; index 0 = mildest)
def break_index(vals, thr):
    """vals [R, L] -> first index with value < thr, L if none."""
    below = vals < thr
    return np.where(below.any(1), below.argmax(1), vals.shape[1])


def slices_of(G):
    """List of (name, vals [R, L]) : 5 resize rows scanning quality, 6 quality columns scanning resize."""
    out = []
    for j, f in enumerate(GRID_F):
        out.append((f"f={f}: scan q", G[:, :, j]))
    for i, q in enumerate(GRID_Q):
        out.append((f"q={q}: scan f", G[:, i, :]))
    return out


def surface_metrics(Gfro, Gext, thr=0.5, cliff=0.20, rise=0.05):
    R = Gfro.shape[0]
    sl_f, sl_e = slices_of(Gfro), slices_of(Gext)
    bf = np.stack([break_index(v, thr) for _, v in sl_f], 1)               # [R, S]
    be = np.stack([break_index(v, thr) for _, v in sl_e], 1)
    lens = np.array([v.shape[1] for _, v in sl_f])
    anyb = (bf < lens) | (be < lens)
    shift = np.where(anyb, be - bf, np.nan).astype(float)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(anyb, shift, np.nan), 1)
    drops = np.stack([(v[:, :-1] - v[:, 1:]).max(1) for _, v in sl_f], 1)
    rises = np.stack([((v[:, 1:] - v[:, :-1]) > rise).sum(1) for _, v in sl_f], 1)
    return dict(n_break_frozen=(Gfro < thr).sum((1, 2)), n_break_ext=(Gext < thr).sum((1, 2)), break_f=bf, break_e=be, shift=shift,
                median_shift=med, max_step_drop=drops.max(1), n_rises=rises.sum(1), slice_names=[n for n, _ in sl_f], slice_len=lens)


# ============================================================================ report
def stage_report(D, fitres, perm, guards):
    r = r20()
    n = len(D["y"])
    Wc, inv, ug = r.cluster_weights(D["groups"], B, SEED, r.strata_of)
    Wrow = Wc[:, inv]
    y = D["y"]
    conds = list(CFG["r2_conds"])
    prim = CFG["primary"]
    R = dict(config=CFG, n=n, n_clusters=len(ug), chance=CHANCE, guards=guards, r2={}, r3={}, gates={})
    BB = {}                                                       # bootstrap arrays
    # ---------------------------------------------------------------- R2
    for manip in MANIPS:
        res = fitres[("r2", manip)]
        prof = {}
        for b in CFG["blocks"]:
            lp = pool_logp(res, b, conds, n)
            for c in conds:
                a = bacc_boot(Wrow, y, lp[c])
                BB[(manip, "blk", b, c)] = a
                prof.setdefault(b, {})[c] = ci(a)
        nm, choice = nested_mid(res, conds, n)
        nest = {}
        for c in conds:
            a = bacc_boot(Wrow, y, nm[c])
            BB[(manip, "nested", c)] = a
            nest[c] = ci(a)
        R["r2"][manip] = dict(profile=prof, nested_mid=nest, chosen={int(f): [int(v[0]), float(v[1])] for f, v in choice.items()})
    nonebase = BB[("none", "nested", prim)]
    prof_none = {b: R["r2"]["none"]["profile"][b][prim][0] for b in MIDS}
    Bstar = max(MIDS, key=lambda b: (round(prof_none[b], 12), -b))
    R["r2"]["B_star"] = int(Bstar)
    verd = {}
    for manip in MANIPS:
        d = R["r2"][manip]
        ret_nested = retained(BB[(manip, "nested", prim)], nonebase)
        ret_fixed = retained(BB[(manip, "blk", Bstar, prim)], BB[("none", "blk", Bstar, prim)])
        orc_m = max(MIDS, key=lambda b: BB[(manip, "blk", b, prim)][0])
        ret_orc = retained(BB[(manip, "blk", orc_m, prim)], BB[("none", "blk", Bstar, prim)])
        ml = {c: ci(BB[(manip, "nested", c)] - BB[(manip, "blk", LAST, c)]) for c in conds}
        d["retained_nested"] = ci(ret_nested)
        d["retained_fixed_Bstar"] = ci(ret_fixed)
        d["retained_oracle_best_mid"] = dict(block=int(orc_m), value=ci(ret_orc))
        d["mid_minus_last"] = ml
        d["last_block_bacc"] = {c: d["profile"][LAST][c] for c in conds}
        d["delta_vs_none_nested"] = {c: ci(BB[(manip, "nested", c)] - BB[("none", "nested", c)]) for c in conds}
    P = CFG["pred_r2"]
    a, b_ = R["r2"]["a_shuffle"], R["r2"]["b_blur"]
    ca = verdict_threshold(a["retained_nested"], P["a_retained_min"], "ge")
    cb = verdict_threshold(b_["retained_nested"], P["b_retained_max"], "lt")
    mlA = a["mid_minus_last"][prim]
    cc = (bool(mlA[0] > 0 and mlA[1] > 0), bool(mlA[1] > 0 or mlA[2] < 0))
    fix_ok = dict(a=verdict_threshold(a["retained_fixed_Bstar"], P["a_retained_min"], "ge")[0], b=verdict_threshold(b_["retained_fixed_Bstar"], P["b_retained_max"], "lt")[0])
    verd = dict(
        a_shuffle_keeps_ge_70=dict(verdict="confirmed" if ca[0] else "refuted", ci_robust=ca[1], retained=a["retained_nested"]),
        b_blur_keeps_lt_50=dict(verdict="confirmed" if cb[0] else "refuted", ci_robust=cb[1], retained=b_["retained_nested"]),
        mid_over_last_persists_under_a=dict(verdict="confirmed" if cc[0] else "refuted", ci_robust=cc[1], delta=mlA,
                                            unmanipulated_delta=R["r2"]["none"]["mid_minus_last"][prim]),
        fixed_block_agrees=fix_ok)
    R["r2"]["verdicts"] = verd
    # ---------------------------------------------------------------- R3
    fro_res, ext_res = fitres[("r2", "none")], fitres[("ext", "none")]
    fro_res_mid = [t for t in fro_res if t["block"] in MIDS]
    fro_lp, fro_choice = nested_mid(fro_res_mid, GRID_CONDS, n)
    ext_lp, ext_choice = nested_mid(ext_res, GRID_CONDS, n)
    G = {}
    for name, lp in (("frozen", fro_lp), ("extended", ext_lp)):
        arr = np.zeros((B + 1, len(GRID_Q), len(GRID_F)))
        for i, q in enumerate(GRID_Q):
            for j, f in enumerate(GRID_F):
                arr[:, i, j] = bacc_boot(Wrow, y, lp[cell_cond(q, f)])
        G[name] = arr
    prim_ext = nested_mid(ext_res, [prim], n)[0][prim]
    R["r3"]["consistency"] = dict(
        frozen_cell_q75_f05_vs_r2_primary=[float(G["frozen"][0, GRID_Q.index(75), GRID_F.index(0.5)]), R["r2"]["none"]["nested_mid"][prim][0]],
        extended_primary_bacc=ci(bacc_boot(Wrow, y, prim_ext)))
    R["r3"]["chosen"] = dict(frozen={int(f): [int(v[0]), float(v[1])] for f, v in fro_choice.items()}, extended={int(f): [int(v[0]), float(v[1])] for f, v in ext_choice.items()})
    R["r3"]["surface"] = {k: [[ci(G[k][:, i, j]) for j in range(len(GRID_F))] for i in range(len(GRID_Q))] for k in G}
    R["r3"]["diff_ext_minus_frozen"] = [[ci(G["extended"][:, i, j] - G["frozen"][:, i, j]) for j in range(len(GRID_F))] for i in range(len(GRID_Q))]
    M = surface_metrics(G["frozen"], G["extended"], CFG["break_threshold"], CFG["pred_r3"]["cliff_max_step_drop"])
    Mx = surface_metrics(G["extended"], G["frozen"], CFG["break_threshold"], CFG["pred_r3"]["cliff_max_step_drop"])
    R["r3"]["metrics"] = dict(
        n_cells_below_frozen=ci(M["n_break_frozen"].astype(float)), n_cells_below_extended=ci(M["n_break_ext"].astype(float)),
        n_cells_below_diff_frozen_minus_ext=ci((M["n_break_frozen"] - M["n_break_ext"]).astype(float)),
        median_shift_steps=ci(M["median_shift"]) if not np.isnan(M["median_shift"][0]) else None,
        median_shift_frac_valid_replicates=float(np.mean(~np.isnan(M["median_shift"]))),
        max_step_drop_frozen=ci(M["max_step_drop"]), max_step_drop_extended=ci(Mx["max_step_drop"]),
        n_rises_gt_005_frozen=int(M["n_rises"][0]), n_rises_gt_005_extended=int(Mx["n_rises"][0]),
        slices=[dict(name=nm, len=int(L), break_idx_frozen=int(M["break_f"][0, s]), break_idx_extended=int(M["break_e"][0, s]),
                     shift=None if np.isnan(M["shift"][0, s]) else int(M["shift"][0, s]))
                for s, (nm, L) in enumerate(zip(M["slice_names"], M["slice_len"]))])
    ms = R["r3"]["metrics"]["median_shift_steps"]
    smooth = bool(M["max_step_drop"][0] <= CFG["pred_r3"]["cliff_max_step_drop"])
    if ms is None:
        shift_v = dict(verdict="undetermined (neither probe breaks in any slice)")
    else:
        shift_v = dict(verdict="confirmed" if ms[0] >= CFG["pred_r3"]["shift_ge_steps"] else "refuted", ci_robust=bool(ms[1] >= 1 if ms[0] >= 1 else ms[2] < 1), median_shift=ms)
    R["r3"]["verdicts"] = dict(smooth_decay=dict(verdict="confirmed" if smooth else "refuted (cliff)", max_step_drop_frozen=R["r3"]["metrics"]["max_step_drop_frozen"]),
                               extended_shift_ge_1_step=shift_v)
    # ---------------------------------------------------------------- gates
    g = R["gates"]
    g["guards_parity"] = dict(ok=guards["parity"]["ok"], max_scaled=max(v["scaled"] for v in guards["parity"]["conds"].values()))
    g["D-008"] = dict(ok=guards["d008"]["ok"], max_gpu_cpu_logit=guards["d008"]["max_gpu_cpu_logit"], max_gpu_cpu_feat=guards["d008"]["max_gpu_cpu_feat"])
    Pc = CFG["perm_control"]
    perm_g = {}
    for manip, res in perm.items():
        accs = []
        for s in range(Pc["n"]):
            yp = np.random.default_rng([SEED, 4242, s]).permutation(y)
            pred = np.zeros(n, int)
            for (s2, te, pr) in res[s * CFG["outer_folds"]:(s + 1) * CFG["outer_folds"]]:
                assert s2 == s
                pred[te] = pr
            accs.append(r.bacc_simple(yp, pred))
        perm_g[manip] = r.gate_permutation(accs, CHANCE, Pc["mean_tol"], Pc["p95_excess"])
    g["permutation"] = dict(ok=all(v["ok"] for v in perm_g.values()), by_manip=perm_g)
    g["sha256_dedupe"] = dict(r.gate_dedupe(D["df"].crop_sha.tolist(), y.tolist()), raw_sha_unique=bool(D["df"].sha256.is_unique))
    g["grouped_splits"] = dict(r.gate_splits(D["fold"], D["groups"]), folds_sha256=D["folds_sha"], recomputed_equal_committed=True)
    g["train_conditions"] = r.gate_train_conds(TRAIN + EXT_EXTRA, CFG["held_out_launderings"])
    cons = R["r3"]["consistency"]
    g["r3_frozen_matches_r2"] = dict(ok=bool(abs(cons["frozen_cell_q75_f05_vs_r2_primary"][0] - cons["frozen_cell_q75_f05_vs_r2_primary"][1]) < 0.01), values=cons["frozen_cell_q75_f05_vs_r2_primary"])
    g["identical_processing"] = dict(ok=True, note="one code path model_input(); manipulation depends on (image, sha256) only; features for all 1020 rows per (manip, cond); train/test share the file; EXIF/ICC absent and RGB checked in 020 prep_checks.json (fov512)")
    for k in ("sha256_dedupe", "grouped_splits"):
        g[k]["ok"] = bool(g[k]["ok"])
    R["all_gates_ok"] = bool(all(v["ok"] for v in g.values()))
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_tables(R)
    log("results.json + tables.md written; gates ok:", R["all_gates_ok"])
    return R


def fmt(c, nd=3):
    return f"{c[0]:.{nd}f} [{c[1]:.{nd}f}, {c[2]:.{nd}f}]"


def write_tables(R):
    L = []
    P = CFG["primary"]
    L += ["# 026-final-run / R2 + R3 tables", "",
          f"Generated by run.py report from results.json. Pre-registration docs/11 (D-020); constants config.json. n = {R['n']} images, {R['n_clusters']} clusters, {B} caption-family bootstrap replicates (020 folds/probe/bootstrap). 8-way bacc, chance {CHANCE:.3f}. Primary condition {P}. Point [95% CI].", ""]
    L += ["## Gates", "", "| gate | ok | detail |", "|---|---|---|"]
    for k, v in R["gates"].items():
        det = {kk: vv for kk, vv in v.items() if kk not in ("ok", "by_manip", "note")}
        if k == "permutation":
            det = {m: dict(mean=round(x["mean"], 4), p95=round(x["p95"], 4), n=x["n"]) for m, x in v["by_manip"].items()}
        L.append(f"| {k} | {'PASS' if v['ok'] else 'FAIL'} | {json.dumps(det, default=str)[:300]} |")
    L += ["", "## R2 verdicts (pre-registered predictions)", ""]
    V = R["r2"]["verdicts"]
    L += [f"- (a) tile shuffle keeps >= 70% of the above-chance bacc: **{V['a_shuffle_keeps_ge_70']['verdict']}** (CI-robust: {V['a_shuffle_keeps_ge_70']['ci_robust']}); retained {fmt(V['a_shuffle_keeps_ge_70']['retained'])}",
          f"- (b) blur sigma 1.5 keeps < 50%: **{V['b_blur_keeps_lt_50']['verdict']}** (CI-robust: {V['b_blur_keeps_lt_50']['ci_robust']}); retained {fmt(V['b_blur_keeps_lt_50']['retained'])}",
          f"- mid > last persists under (a): **{V['mid_over_last_persists_under_a']['verdict']}** (CI-robust: {V['mid_over_last_persists_under_a']['ci_robust']}); delta under (a) {fmt(V['mid_over_last_persists_under_a']['delta'])}; unmanipulated delta {fmt(V['mid_over_last_persists_under_a']['unmanipulated_delta'])}",
          f"- sensitivity, fixed block B* = block {R['r2']['B_star']} (oracle best mid block of the unmanipulated profile) as numerator and denominator: prediction (a) confirmed = {V['fixed_block_agrees']['a']}, prediction (b) confirmed = {V['fixed_block_agrees']['b']}", ""]
    L += ["## R2 summary per manipulation (nested best mid block, T-mix probe retrained per manipulation)", "",
          "retained = (bacc_manip - 1/8) / (bacc_none - 1/8) at the primary condition, same block-selection rule in numerator and denominator.", "",
          f"| manipulation | bacc {P} (nested mid) | retained (nested) | retained (fixed B*={R['r2']['B_star']}) | retained (oracle best mid per manip) | block 12 bacc | mid - last | clean | screenshot_sim (held-out) |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MANIPS:
        d = R["r2"][m]
        orc = d["retained_oracle_best_mid"]
        L.append(f"| {m} | {fmt(d['nested_mid'][P])} | {fmt(d['retained_nested'])} | {fmt(d['retained_fixed_Bstar'])} | blk{orc['block']}: {fmt(orc['value'])} | {fmt(d['last_block_bacc'][P])} | {fmt(d['mid_minus_last'][P])} | {fmt(d['nested_mid']['clean'])} | {fmt(d['nested_mid']['screenshot_sim'])} |")
    L += ["", "Per-block profile at the primary condition (point bacc; CIs in results.json). Mid blocks 3-9 marked *.", "",
          "| block | " + " | ".join(MANIPS) + " |", "|---|" + "---|" * len(MANIPS)]
    for b in CFG["blocks"]:
        L.append(f"| {b}{'*' if b in MIDS else ''} | " + " | ".join(f"{R['r2'][m]['profile'][b][P][0]:.3f}" for m in MANIPS) + " |")
    for c in ("clean", "screenshot_sim"):
        L += ["", f"Per-block profile at {c} (descriptive; point bacc).", "", "| block | " + " | ".join(MANIPS) + " |", "|---|" + "---|" * len(MANIPS)]
        for b in CFG["blocks"]:
            L.append(f"| {b}{'*' if b in MIDS else ''} | " + " | ".join(f"{R['r2'][m]['profile'][b][c][0]:.3f}" for m in MANIPS) + " |")
    L += ["", "Nested choices (fold: block, C): " + "; ".join(f"{m}: " + ", ".join(f"{f}:b{v[0]}/C{v[1]}" for f, v in R["r2"][m]["chosen"].items()) for m in MANIPS), ""]
    # R3
    V3 = R["r3"]["verdicts"]
    Mx = R["r3"]["metrics"]
    L += ["## R3 verdicts", "",
          f"- smooth decay (frozen probe, max single-step bacc drop along any axis slice <= {CFG['pred_r3']['cliff_max_step_drop']}): **{V3['smooth_decay']['verdict']}**; max step drop {fmt(V3['smooth_decay']['max_step_drop_frozen'])}; non-monotone rises > 0.05: frozen {Mx['n_rises_gt_005_frozen']}, extended {Mx['n_rises_gt_005_extended']}",
          f"- extended T-mix moves the breaking point by >= 1 grid step: **{V3['extended_shift_ge_1_step']['verdict']}**" + (f"; median shift {fmt(V3['extended_shift_ge_1_step']['median_shift'], 2)} steps (CI-robust: {V3['extended_shift_ge_1_step']['ci_robust']})" if 'median_shift' in V3['extended_shift_ge_1_step'] else ""),
          f"- cells below 0.5 bacc: frozen {fmt(Mx['n_cells_below_frozen'], 1)}, extended {fmt(Mx['n_cells_below_extended'], 1)}, difference (frozen - extended) {fmt(Mx['n_cells_below_diff_frozen_minus_ext'], 1)} of 30",
          f"- consistency: frozen cell (q75, f0.5) = {R['r3']['consistency']['frozen_cell_q75_f05_vs_r2_primary'][0]:.3f} vs R2 nested {R['r3']['consistency']['frozen_cell_q75_f05_vs_r2_primary'][1]:.3f}; frozen block/C per fold {R['r3']['chosen']['frozen']}; extended {R['r3']['chosen']['extended']}", ""]
    L += ["Breaking-point slices (index of the first step with bacc < 0.5 counted from the mild end; len = never breaks):", "",
          "| slice | length | frozen | extended | shift |", "|---|---|---|---|---|"]
    for s in Mx["slices"]:
        L.append(f"| {s['name']} | {s['len']} | {s['break_idx_frozen']} | {s['break_idx_extended']} | {s['shift']} |")
    for nm, key in (("frozen T-mix best-mid probe", "frozen"), ("extended T-mix probe (adds jr(45,0.5), jr(75,0.35) to training)", "extended")):
        L += ["", f"### bacc surface: {nm}", "", "rows = JPEG quality, columns = resize factor (resize first, then JPEG; f = 0.35 and 0.25 cells are upsampled back to 224 before the crop). * = bacc < 0.5 (point).", "",
              "| q \\ f | " + " | ".join(str(f) for f in GRID_F) + " |", "|---|" + "---|" * len(GRID_F)]
        for i, q in enumerate(GRID_Q):
            L.append(f"| {q} | " + " | ".join(f"{R['r3']['surface'][key][i][j][0]:.3f}{'*' if R['r3']['surface'][key][i][j][0] < 0.5 else ''} [{R['r3']['surface'][key][i][j][1]:.2f},{R['r3']['surface'][key][i][j][2]:.2f}]" for j in range(len(GRID_F))) + " |")
    L += ["", "### Difference (extended - frozen), paired bootstrap", "", "| q \\ f | " + " | ".join(str(f) for f in GRID_F) + " |", "|---|" + "---|" * len(GRID_F)]
    for i, q in enumerate(GRID_Q):
        L.append(f"| {q} | " + " | ".join(f"{R['r3']['diff_ext_minus_frozen'][i][j][0]:+.3f} [{R['r3']['diff_ext_minus_frozen'][i][j][1]:+.2f},{R['r3']['diff_ext_minus_frozen'][i][j][2]:+.2f}]" for j in range(len(GRID_F))) + " |")
    L += ["", "### ASCII surface (point bacc, frozen | extended; '#' < 0.5)", "", "```", "q\\f   " + "".join(f"{f:>7}" for f in GRID_F) + "    |" + "".join(f"{f:>7}" for f in GRID_F)]
    for i, q in enumerate(GRID_Q):
        row = f"{q:>3}   "
        for key in ("frozen", "extended"):
            for j in range(len(GRID_F)):
                v = R["r3"]["surface"][key][i][j][0]
                row += f"{v:6.2f}{'#' if v < 0.5 else ' '}"
            row += "    |" if key == "frozen" else ""
        L.append(row)
    L += ["```", ""]
    (RES / "tables.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["guards", "extract", "fit", "report", "all"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    os.chdir(ROOT)
    D = load_data()
    log("data ok:", len(D["y"]), "rows; folds sha", D["folds_sha"][:12])
    if a.stage in ("guards", "all"):
        stage_guards(D)
    guards = json.loads((RES / "guards.json").read_text())
    if a.stage in ("extract", "all"):
        assert guards["parity"]["ok"] and guards["d008"]["ok"], "guards not passed"
        stage_extract(D)
    if a.stage in ("fit", "report", "all"):
        fitres, perm = stage_fit(D, a.workers)
        if a.stage in ("report", "all"):
            stage_report(D, fitres, perm, guards)


if __name__ == "__main__":
    main()
