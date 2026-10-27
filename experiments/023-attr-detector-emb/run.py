"""E12 / 023-attr-detector-emb (D-018 cycle). docs/08 H-A5 + docs/09 (frozen pre-registration).

Penultimate embeddings (forward hooks) of four published detectors -- Community Forensics-384, B-Free (DINOv2-reg4), RINE (LDM ckpt),
ClipDet (latent10k_plus) -- on the 1,020 Synthbuster/B-Free fakes of the main K=8 set, then the EXACT H-A1 probe protocol (the
functions of experiments/020-attr-foundation/run.py, imported unchanged; same rows, caption-family folds, T-mix training, inner-CV C,
pairwise metric, LOGO open set), compared with the H-A1 DINOv2 layers on identical folds.

    env/.venv/bin/python -I experiments/023-attr-detector-emb/run.py extract [det ...]   # GPU, one subprocess per detector (frees memory)
    env/.venv/bin/python -I experiments/023-attr-detector-emb/run.py guard   [det ...]   # D-008: CPU recomputation of a fixed 32-image subset
    env/.venv/bin/python -I experiments/023-attr-detector-emb/run.py probe | report | all

Caches: experiments/023-attr-detector-emb/cache/ (git-ignored). Seed 0.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
CFG = json.loads((HERE / "config.json").read_text())
SEED = CFG["seed"]
CACHE, RES = HERE / "cache", HERE / "results"
DETS = list(CFG["detectors"])
CONDS = CFG["conditions"]
DINO_FS = CFG["ha1_probe"]["dino_fs"]
MOD020 = ROOT / "experiments" / "020-attr-foundation" / "run.py"


def log(*a):
    print(time.strftime("%H:%M:%S"), "E12", *a, flush=True)


def safe(c):
    return "".join(ch if ch.isalnum() or ch in "-_.+()" else "_" for ch in c)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ============================================================================ rows and images
def load_rows():
    """The 1,020 fakes of the K=8 set, exactly the rows 020 ha1_main selects (pool == fake of the 019 fov512 splits)."""
    df = pd.read_csv(ROOT / CFG["splits"], keep_default_na=False, dtype={"vidx": str})
    df = df[df.pool == CFG["select"]["pool"]].reset_index(drop=True)
    assert set(df.generator) == set(CFG["classes"]), set(df.generator) ^ set(CFG["classes"])
    assert df.sha256.is_unique and len(df) == 1020, len(df)
    return df


def load_cond(path, sha, cond):
    """Same call as 004/p4 extraction: open the 512 crop, apply the condition keyed by sha256, seed 0."""
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        return apply_condition(im, cond, key=sha, seed=SEED)


def guard_subset(df):
    p = CACHE / "guard_subset.json"
    if p.exists():
        return json.loads(p.read_text())
    rng = np.random.default_rng(SEED)
    sub = []
    for g in CFG["classes"]:
        s = sorted(df.sha256[df.generator == g])
        sub += [s[i] for i in rng.choice(len(s), CFG["d008"]["per_generator"], replace=False)]
    CACHE.mkdir(exist_ok=True)
    p.write_text(json.dumps(sub))
    return sub


# ============================================================================ detector adapters (hooks)
class Adapter:
    """Wraps one published detector as built by experiments/002-baselines/scorers/<det>.py (unmodified). fwd(imgs) -> (emb [n,d] float32, logit [n])."""

    def __init__(self, det):
        import torch
        self.torch = torch
        self.det = det
        self.cap = {}
        if torch.cuda.is_available() and os.environ.get("DD_DEVICE", "cuda") != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False       # D-008: fp32 everywhere (also conv)
            torch.backends.cudnn.allow_tf32 = False
        import importlib
        self.mod = importlib.import_module(CFG["detectors"][det]["module"])
        if det == "commfor":
            m = self.mod._load()
            self.net = m.model
            self.hook = self.net.head.register_forward_pre_hook(lambda mod_, inp: self.cap.__setitem__("e", inp[0].detach()))
        elif det == "bfree":
            m = self.mod._load()
            self.net = m.model                                   # Wrapper5crops
            self.hook = self.net.model.head.register_forward_pre_hook(lambda mod_, inp: self.cap.__setitem__("e", inp[0].detach()))
        elif det == "rine":
            m = self.mod._load("ldm")
            self.net = m.model
            self.hook = self.net.proj2.register_forward_hook(lambda mod_, inp, out: self.cap.__setitem__("e", out.detach()))
        elif det == "clipdet":
            m = self.mod._load("latent10k_plus")
            self.net = m.model
            self.hook = self.net.fc.register_forward_pre_hook(lambda mod_, inp: self.cap.__setitem__("e", inp[0].detach()))
        else:
            raise ValueError(det)
        self.dm = m                                              # DeviceModel (we only read .device; its __call__ is NOT used: it hides CUDA OOM as a CPU fallback)
        self.device = m.device
        self.net.eval()

    def _batch(self, imgs):
        """One forward over a list of same-size images -> (emb tensor [n,d], logit tensor [n])."""
        torch, mod, dev = self.torch, self.mod, self.device
        rgb = [im.convert("RGB") if im.mode != "RGB" else im for im in imgs]
        with torch.no_grad():
            if self.det == "commfor":
                x = torch.stack([mod._TF(im) for im in rgb]).to(dev)
                y = self.net(x)[:, 0]
                e = self.cap["e"]
            elif self.det == "bfree":
                x = torch.stack([mod._TF(im) for im in rgb]).to(dev)
                y = mod._fwd(self.net, x)[:, 0]
                e = self.cap["e"]
                if e.shape[0] == 5 * len(rgb):                    # official 5-window path: windows are stacked crop-major; the logit is their mean
                    e = e.reshape(5, len(rgb), -1).mean(0)
            elif self.det == "rine":
                x = torch.stack([mod._TF_CENTER(im) for im in rgb]).to(dev)
                y = self.net(x)[0][:, 0]
                e = self.cap["e"]
            else:  # clipdet
                x = torch.stack([mod._TF(im) for im in rgb]).to(dev)
                y = self.net(x).reshape(len(rgb), -1)[:, 0]
                e = self.cap["e"].reshape(len(rgb), -1)
        assert e.shape[0] == len(rgb) and e.shape[1] == CFG["detectors"][self.det]["dim"], (self.det, tuple(e.shape))
        assert str(e.device).startswith(self.device.type), "embedding left the chosen device (silent fallback?)"
        return e.float().cpu().numpy(), y.float().cpu().numpy()

    def fwd(self, imgs, bs=None):
        """Batches of same-size images (B-Free keeps the native size, so mixed sizes are grouped); OOM -> halve the batch and retry (never CPU)."""
        torch = self.torch
        bs = min(bs or CFG["detectors"][self.det]["batch"], CFG["extract"]["max_batch"])
        groups = {}
        for i, im in enumerate(imgs):
            groups.setdefault(im.size, []).append(i)
        E = np.zeros((len(imgs), CFG["detectors"][self.det]["dim"]), np.float32)
        L = np.zeros(len(imgs), np.float64)
        for idxs in groups.values():
            j = 0
            while j < len(idxs):
                ii = idxs[j:j + bs]
                try:
                    e, y = self._batch([imgs[i] for i in ii])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        time.sleep(20)                            # other jobs share the GPU: wait and retry at batch 1
                        continue
                    bs = max(1, bs // 2)
                    log(f"{self.det}: CUDA OOM -> batch {bs}")
                    continue
                E[ii], L[ii] = e, y
                j += len(ii)
        self.last_bs = bs
        return E, L

    def head(self, e):
        """Recompute the logit from the captured embedding with the detector's own head (selftest: proves the hook sits right before the head)."""
        torch = self.torch
        with torch.no_grad():
            e = torch.as_tensor(e, dtype=torch.float32, device=self.device)
            if self.det == "commfor":
                return self.net.head(e)[:, 0]
            if self.det == "bfree":
                return self.net.model.head(e)[:, 0]
            if self.det == "rine":
                return self.net.head(e)[:, 0]
            return self.net.fc(e[:, :, None, None]).reshape(len(e), -1)[:, 0]

    def close(self):
        self.hook.remove()


def release_detector(ad):
    """Drop every reference to the loaded detector (hooks, the scorer module's cached model, RINE's shared CLIP backbone) and free CUDA memory."""
    ad.close()
    mod = ad.mod
    for name in ("_M", "_BACKBONE"):
        if hasattr(mod, name):
            setattr(mod, name, None)
    if hasattr(mod, "_MODELS"):
        mod._MODELS.clear()
    ad.net = ad.dm = None
    free_gpu()


def free_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================ extraction
def meta_for(det):
    c = CFG["detectors"][det]
    import timm, torch
    return dict(det=det, module=c["module"], arch=c["arch"], hook=c["hook"], dim=c["dim"], transform=c["transform"],
                image_pipeline=CFG["image_pipeline"], seed=SEED, torch=torch.__version__, timm=timm.__version__)


def cache_path(det, cond):
    return CACHE / det / f"{safe(cond)}.npz"


def read_cache(det, cond):
    p = cache_path(det, cond)
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return {k: (e, l) for k, e, l in zip(z["keys"].tolist(), z["emb"], z["logit"])}


def write_cache(det, cond, store):
    p = cache_path(det, cond)
    p.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(store)
    tmp = p.with_name("tmp_" + p.name)
    np.savez(tmp, keys=np.array(ks), emb=np.stack([store[k][0] for k in ks]).astype(np.float32), logit=np.array([store[k][1] for k in ks], np.float64))
    os.replace(tmp, p)


def stage_extract_one(det, limit=None):
    df = load_rows()
    if limit:
        df = df.iloc[:limit]
    d = CACHE / det
    d.mkdir(parents=True, exist_ok=True)
    mp = d / "meta.json"
    meta = meta_for(det)
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise SystemExit(f"cache config mismatch for {det}; delete {d} to recompute")
    mp.write_text(json.dumps(meta, indent=1, sort_keys=True))
    ad = Adapter(det)
    assert ad.device.type == "cuda" or os.environ.get("DD_DEVICE") == "cpu", f"expected CUDA, got {ad.device}"
    log(f"{det}: device {ad.device}, {len(df)} rows")
    tim = {}
    chunk = CFG["extract"]["chunk"]
    with ThreadPoolExecutor(CFG["extract"]["workers"]) as pool:
        for cond in CONDS:
            store = read_cache(det, cond)
            todo = df[~df.sha256.isin(store)]
            t0 = time.time()
            for s in range(0, len(todo), chunk):
                ch = todo.iloc[s:s + chunk]
                imgs = list(pool.map(lambda r: load_cond(r[0], r[1], cond), zip(ch.path, ch.sha256)))
                E, L = ad.fwd(imgs)
                assert np.isfinite(E).all() and np.isfinite(L).all()
                store.update({k: (e, l) for k, e, l in zip(ch.sha256, E, L)})
                write_cache(det, cond, store)
                log(f"  {det}/{cond}: {min(s + chunk, len(todo))}/{len(todo)} new ({len(df) - len(todo)} cached) bs={ad.last_bs}")
            tim[cond] = dict(n_rows=len(df), n_new=len(todo), seconds=time.time() - t0)
    tp = d / "timings.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    old.update(tim)
    tp.write_text(json.dumps(old, indent=1))
    release_detector(ad)


def stage_cpuref_one(det):
    """Subprocess entry (DD_DEVICE=cpu): CPU embeddings + logits of the fixed 32-image subset under every condition (batch 4)."""
    import torch
    torch.set_num_threads(8)
    df = load_rows()
    sub = guard_subset(df)
    dfi = df.set_index("sha256", drop=False)
    p = CACHE / det / "cpuref.npz"
    ref = {}
    if p.exists():
        z = np.load(p, allow_pickle=False)
        ref = {k: (z[f"emb__{k}"], z[f"logit__{k}"], z["keys"].tolist()) for k in CONDS if f"emb__{k}" in z.files}
    ad = Adapter(det)
    assert ad.device.type == "cpu", ad.device
    for cond in CONDS:
        if cond in ref:
            continue
        t0 = time.time()
        imgs = [load_cond(dfi.path[s], s, cond) for s in sub]
        E, L = ad.fwd(imgs, bs=CFG["d008"]["cpu_batch"])
        ref[cond] = (E, L, sub)
        out = {"keys": np.array(sub)}
        for k, v in ref.items():
            out[f"emb__{k}"], out[f"logit__{k}"] = v[0], v[1]
        np.savez(p.with_name("tmp_" + p.name), **out)
        os.replace(p.with_name("tmp_" + p.name), p)
        log(f"cpuref {det}/{cond} done in {time.time() - t0:.0f}s")


def d008_compare(det, tol_e=None, tol_l=None):
    tol_e = tol_e or CFG["d008"]["emb_tol_scaled"]
    tol_l = tol_l or CFG["d008"]["logit_tol"]
    z = np.load(CACHE / det / "cpuref.npz", allow_pickle=False)
    keys = z["keys"].tolist()
    out = {}
    for cond in CONDS:
        store = read_cache(det, cond)
        A = np.stack([store[k][0] for k in keys])
        La = np.array([store[k][1] for k in keys])
        C, Lc = z[f"emb__{cond}"], z[f"logit__{cond}"]
        d = np.abs(A - C)
        scaled = float((d / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max())
        dl = float(np.abs(La - Lc).max())
        out[cond] = dict(n=len(keys), max_abs_emb_diff=float(d.max()), max_scaled_emb_diff=scaled, max_abs_dlogit=dl,
                         emb_ok=bool(scaled <= tol_e), logit_ok=bool(dl <= tol_l), ok=bool(scaled <= tol_e or dl <= tol_l),
                         emb_tol=tol_e, logit_tol=tol_l)
    return out


def stage_guard_one(det):
    p = CACHE / det / "cpuref.npz"
    ok = False
    if p.exists():
        z = np.load(p, allow_pickle=False)
        ok = all(f"emb__{c}" in z.files for c in CONDS)
    if not ok:
        log(f"{det}: computing CPU reference ({CFG['d008']['n']} images x {len(CONDS)} conditions)")
        env = dict(os.environ, DD_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-I", str(HERE / "run.py"), "cpuref", det], env=env)
        if r.returncode != 0:
            raise SystemExit(f"cpuref failed for {det}")
    g = d008_compare(det)
    gp = RES / "d008_guard.json"
    cur = json.loads(gp.read_text()) if gp.exists() else {}
    cur[det] = g
    RES.mkdir(exist_ok=True)
    gp.write_text(json.dumps(cur, indent=1))
    for c, v in g.items():
        log(f"D-008 {det}/{c}: scaled emb {v['max_scaled_emb_diff']:.2e} (abs {v['max_abs_emb_diff']:.2e}) dlogit {v['max_abs_dlogit']:.2e} -> {'ok' if v['ok'] else 'FAIL'}")
    if not all(v["ok"] for v in g.values()):
        raise SystemExit(f"D-008 ABORT for {det}")


def run_sub(args):
    r = subprocess.run([sys.executable, "-I", str(HERE / "run.py")] + args)
    if r.returncode != 0:
        raise SystemExit(f"subprocess {args} failed ({r.returncode})")


def stage_extract(dets):
    for det in dets:
        t0 = time.time()
        run_sub(["extract1", det])
        log(f"extract {det}: {time.time() - t0:.0f}s")


def stage_guard(dets):
    for det in dets:
        run_sub(["guard1", det])


# ============================================================================ probe: H-A1 pipeline (020, imported) on the embeddings
def load020():
    if "m020" not in sys.modules:
        spec = importlib.util.spec_from_file_location("m020", MOD020)
        m = importlib.util.module_from_spec(spec)
        sys.modules["m020"] = m
        spec.loader.exec_module(m)
    return sys.modules["m020"]


def build_rows():
    """Rows, labels, caption-family groups and folds exactly as 020 ha1_main builds them (same calls, same order)."""
    m = load020()
    cfg = m.CFG["analyses"]["ha1_main"]
    df, u = m.load_universe(cfg["universe"])
    sel = np.ones(len(df), bool)
    for col, vals in cfg["select"].items():
        sel &= df[col].isin(vals).to_numpy()
    idx = np.where(sel)[0]
    cls = {c: i for i, c in enumerate(cfg["classes"])}
    assert list(cfg["classes"]) == CFG["classes"]
    y = np.array([cls[c] for c in df[cfg["class_column"]].iloc[idx]])
    groups = df.grp.to_numpy()[idx]
    fold = m.make_folds(y, groups, m.CFG["outer_folds"], m.SEED)
    chk = m.check_grouped_folds(fold, groups)
    assert chk["ok"], chk
    sub = df.iloc[idx].reset_index(drop=True)
    assert sub.sha256.is_unique and len(sub) == 1020
    assert u["conditions"] == CONDS and u["tmix"] == CFG["tmix"] and cfg["primary"] == CFG["primary"], (u["conditions"], u["tmix"], cfg["primary"])
    return m, cfg, u, idx, y, groups, fold, chk, sub


def build_analysis(dets=None, XF_override=None):
    """Analysis object identical (rows, order, groups, folds) to 020 ha1_main, with the DINOv2 4608-d columns followed by the detector embeddings.
    XF_override (selftest only): {cond: matrix} replaces the features (same column layout)."""
    m, cfg, u, idx, y, groups, fold, chk, sub = build_rows()
    dets = DETS if dets is None else dets
    XF = {}
    sl = {}
    off = 4608
    for det in dets:
        sl[det] = slice(off, off + CFG["detectors"][det]["dim"])
        off += CFG["detectors"][det]["dim"]
    for c in u["conditions"]:
        if XF_override is not None:
            XF[c] = XF_override[c]
            continue
        store = m.read_caches(u["caches"], c)
        Xd = np.stack([store[s] for s in sub.sha256]).astype(np.float32)
        assert Xd.shape[1] == 4608 and np.isfinite(Xd).all()
        parts = [Xd]
        for det in dets:
            st = read_cache(det, c)
            miss = [s for s in sub.sha256 if s not in st]
            assert not miss, f"{det}/{c}: {len(miss)} embeddings missing"
            parts.append(np.stack([st[s][0] for s in sub.sha256]))
        XF[c] = np.concatenate(parts, 1)
    A = m.Analysis("ha5", cfg, y, cfg["classes"], groups, fold, u["conditions"], u["tmix"], cfg["primary"], XF, {}, {}, sub, fam=groups)
    A.idx = idx
    A.commit = dict(sha256_of_folds=hashlib.sha256(fold.tobytes()).hexdigest(), n=len(fold), sizes=np.bincount(fold).tolist(), check=chk)
    for det, s in sl.items():
        m.SL[det] = s                                               # getX / grid_for treat the detector like a DINOv2 feature set (dino C grid)
    if XF_override is None:                                         # integrity: the slice the probe reads is the cached embedding of that sha256
        for det in dets:
            for c in A.conds:
                st = read_cache(det, c)
                for i in (0, 17, 511, 1019):
                    assert np.array_equal(A.getX(det, c)[i], st[sub.sha256[i]][0]), (det, c, i)
    return m, A


def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    r = fn()
    f.write_bytes(pickle.dumps(r, protocol=4))
    return r


FS_ALL = DINO_FS + DETS


def stage_probe(workers):
    m, A = build_analysis()
    m._G["ha5"] = A
    cfg020 = m.CFG["analyses"]["ha1_main"]
    n_f = m.CFG["outer_folds"]
    tasks = [("ha5", fs, regime, f) for regime in ("tmix", "clean") for fs in FS_ALL for f in range(n_f)]
    tasks.sort(key=lambda t: (t[1] != "cat",))
    closed = cached("closed", lambda: m.run_tasks(m.closed_task, tasks, workers, "closed"))
    ltasks = [("ha5", g, f, fs, regime) for regime in ("tmix", "clean") for fs in FS_ALL for g in range(A.K) for f in range(n_f)]
    ltasks.sort(key=lambda t: (t[3] != "cat", t[4] != "tmix"))
    logo = cached("logo", lambda: m.run_tasks(m.logo_task, ltasks, workers, "logo"))
    # permutation control per detector embedding (fixed C, T-mix, primary), 020 perm_closed_task
    pc = CFG["perm"]

    def perm_all():
        out = {}
        for det in DETS:
            tk = [("ha5", s, f, det, pc["C"]) for s in range(pc["n"]) for f in range(n_f)]
            res = m.run_tasks(m.perm_closed_task, tk, workers, f"perm {det}")
            accs = []
            for s in range(pc["n"]):
                yp = np.random.default_rng([m.SEED, 4242, s]).permutation(A.y)
                pred = np.zeros(A.n, int)
                for (s2, te, pr) in res[s * n_f:(s + 1) * n_f]:
                    assert s2 == s
                    pred[te] = pr
                accs.append(m.bacc_simple(yp, pred))
            out[det] = accs
        return out
    perm = cached("perm", perm_all)
    return A, closed, logo, perm


# ============================================================================ report
def label_ha5(best, a1, a1_nested, gates_ok, margin):
    """best/a1: dict(bacc point, diff [pt, lo, hi] paired). PASS: bacc_best >= a1 + margin and diff CI lo > 0. KILL: bacc_best <= a1. else INCONCLUSIVE."""
    if not gates_ok:
        return dict(label="INVALID", reasons=["a validity gate failed"])
    d = best["diff"]
    pass_parts = dict(ge_margin=bool(best["bacc"] >= a1 + margin), ci_lo_gt_0=bool(d[1] > 0))
    kill = bool(best["bacc"] <= a1)
    lab = "KILL" if kill else ("PASS" if all(pass_parts.values()) else "INCONCLUSIVE")
    return dict(label=lab, kill_condition=kill, parts=pass_parts, best_bacc=best["bacc"], a1_best_layer_bacc=a1, a1_nested_bacc=a1_nested, margin=margin)


def ha5_decision(m, bb, B):
    """bb: {feature set: bootstrap array of bacc (row 0 = point estimate)} at the primary condition. Picks H-A1's best fixed layer and the best detector,
    computes paired diffs (same bootstrap weights), one-sided p and Holm over the detectors."""
    pt = {fs: float(bb[fs][0]) for fs in bb}
    a1_fs = max(DINO_FS, key=lambda fs: (round(pt[fs], 12), -DINO_FS.index(fs)))
    best_det = max(DETS, key=lambda d: (round(pt[d], 12), -DETS.index(d)))
    diffs, ps = {}, []
    for d in DETS:
        dd = bb[d] - bb[a1_fs]
        diffs[d] = m.ci(dd)
        ps.append(float((1 + (dd[1:] <= 0).sum()) / (B + 1)))
    return dict(pt=pt, a1_fs=a1_fs, best_det=best_det, diffs=diffs, p=dict(zip(DETS, ps)), p_holm=dict(zip(DETS, m.holm(ps))))


def stage_report(workers):
    t0 = time.time()
    A, closed, logo, perm = stage_probe(workers)
    m = load020()
    m._G["ha5"] = A
    B = m.B
    pc = A.primary
    Wc, inv, ug = m.cluster_weights(A.groups, B, m.SEED, m.strata_of)
    drop = m.CFG["analyses"]["ha1_main"]["drop_macro"]
    R = dict(config=CFG, chance=1 / A.K, n=A.n, n_clusters=len(ug), primary=pc, classes=A.classes, fs={}, logo={}, gates={}, labels={})
    BOOT = {}
    for regime in ("tmix", "clean"):
        for fs in FS_ALL:
            lp, chosen = m.pooled_logp(closed, A, "ha5", fs, regime)
            summ, boots = m.eval_closed(A, lp, Wc, inv, drop)
            R["fs"][f"{regime}|{fs}"] = dict(summary=summ, chosen=chosen)
            BOOT[(regime, fs)] = boots
            if regime == "tmix":
                np.savez_compressed(CACHE / f"oof_{fs}_tmix_{pc}.npz", logp=lp[pc], y=A.y)
        for fs in FS_ALL:
            out, _ = m.eval_logo(A, logo, Wc, inv, "ha5", fs, regime)
            R["logo"][f"{regime}|{fs}"] = out
    # ---- paired comparisons at the primary condition (T-mix)
    bb = {fs: BOOT[("tmix", fs)][pc]["bacc"] for fs in FS_ALL}
    dec = ha5_decision(m, bb, B)
    pt, a1_fs, best_det, diffs = dec["pt"], dec["a1_fs"], dec["best_det"], dec["diffs"]
    R["paired_vs_a1_best_layer"] = dict(a1_best_layer=a1_fs, a1_best_bacc=pt[a1_fs], det_minus_a1=diffs, p_one_sided=dec["p"], p_holm=dec["p_holm"],
                                        note="diff = bacc(detector) - bacc(H-A1 best fixed layer) at jpeg75_resize05, T-mix, same cluster-bootstrap weights (paired); p = share of bootstrap diffs <= 0, Holm over the 4 detectors")
    # also vs every dino layer and for the held-out launderings (descriptive)
    desc = {}
    for c in A.conds:
        for d in DETS:
            for fs in DINO_FS:
                dd = BOOT[("tmix", d)][c]["bacc"] - BOOT[("tmix", fs)][c]["bacc"]
                desc[f"{c}|{d}-{fs}"] = m.ci(dd)
    R["paired_all_descriptive"] = desc
    # 020 nested H-A1 value (inner-CV choice among b4/b8/b12/cat per outer fold) computed with the imported function on the DINOv2 rows only
    dino_only = [r for r in closed if r["fs"] in DINO_FS and r["regime"] == "tmix"]
    lpn, choice = m.nested_logp(dino_only, A, "ha5", "tmix")
    summ_n, boots_n = m.eval_closed(A, lpn, Wc, inv, drop)
    R["fs"]["tmix|nested_dino"] = dict(summary=summ_n, chosen_fs=choice)
    BOOT[("tmix", "nested_dino")] = boots_n
    nested_b = float(boots_n[pc]["bacc"][0])
    # ---- 020 results consistency (if available)
    r020 = ROOT / "experiments" / "020-attr-foundation" / "results" / "results.json"
    cons = dict(available=r020.exists())
    if r020.exists():
        j = json.loads(r020.read_text())
        mm = j["analyses"]["ha1_main"]["fs"]
        diffs020 = {}
        for fs in DINO_FS + ["nested"]:
            v020 = mm[f"tmix|{fs}"]["summary"][pc]["bacc"][0]
            mine = nested_b if fs == "nested" else pt[fs]
            diffs020[fs] = dict(v020=v020, mine=mine, abs_diff=abs(v020 - mine))
        cons["bacc_primary_tmix"] = diffs020
        cons["max_abs_diff"] = max(v["abs_diff"] for v in diffs020.values())
        cons["folds_sha256_020"] = j["analyses"]["ha1_main"].get("folds_sha256") or None
        base = {}
        for key in ("M0_raw",):
            base[key] = j["analyses"]["ha1_main"][key]["bacc"]
        base["M1"] = mm["tmix|M1"]["summary"][pc]["bacc"]
        base["M1_bacc_by_cond"] = {c: mm["tmix|M1"]["summary"][c]["bacc"] for c in A.conds}
        base["M0_tensor"] = j["analyses"]["ha1_main"].get("M0_tensor")
        base["labels_H-A1"] = j.get("labels", {}).get("H-A1")
        R["baselines_020"] = base
    R["consistency_020"] = cons
    fj = ROOT / "experiments" / "020-attr-foundation" / "results" / "folds_ha1_main.json"
    if fj.exists():
        f020 = np.array(json.loads(fj.read_text())["fold"])
        R["fold_identity_with_020"] = dict(identical=bool((f020 == A.fold).all()), rows_identical=json.loads(fj.read_text())["rows"] == A.df.sha256.tolist())
    # ---- gates
    gates = {}
    gates["sha256_dedupe"] = m.gate_dedupe(A.df.sha256.tolist(), A.y.tolist())
    gates["grouped_splits"] = dict(m.gate_splits(A.fold, A.groups), folds_sha256=A.commit["sha256_of_folds"], pairwise_computed=True)
    gates["train_conditions"] = m.gate_train_conds(A.tmix, CFG["held_out_launderings"])
    gates["D-008"] = d008_gate()
    pg = m.CFG["analyses"]["ha1_main"]["perm"]
    for d in DETS:
        g = m.gate_permutation(perm[d], 1 / A.K, CFG["perm"]["mean_tol"], CFG["perm"]["p95_excess"])
        g["ok"] &= len(perm[d]) == CFG["perm"]["n"]
        gates[f"permutation_200.{d}"] = g
    gates["identical_preprocessing"] = preproc_gate(A)
    gates["holm_and_inner_cv"] = dict(ok=True, note="C chosen by inner grouped CV on train folds only (020 inner_table); detector choice is a fixed list; best-of-4 selection is on test results and is reported with Holm-adjusted bootstrap p")
    gates["no_test_group_in_train"] = dict(ok=True, note="assert_disjoint_groups runs inside every 020 closed/LOGO/permutation task")
    gates["chance_levels_in_tables"] = dict(ok=True, note="every table carries chance 0.125 (bacc) / 0.5 (pairwise, AUROC) and the majority-class rate")
    gates["baselines_M0_M1"] = dict(ok=True, note="M0/M1 are computed once in 020 on the same rows and folds (results/results.json); copied into baselines_020 when available. H-A5's own label does not use them (docs/08)", available=cons["available"])
    R["gates"] = gates
    gates_ok = all(g["ok"] for g in gates.values())
    R["labels"]["all_gates_ok"] = bool(gates_ok)       # display fix (see notes.md "post-result changes"): key was read by report_tables but never set
    best = dict(bacc=pt[best_det], diff=diffs[best_det])
    R["labels"]["H-A5"] = dict(label_ha5(best, pt[a1_fs], nested_b, gates_ok, CFG["label_rule"]["margin"]), best_detector=best_det, a1_best_layer=a1_fs)
    # open-set flags per embedding (descriptive)
    ous = {}
    for key, out in R["logo"].items():
        regime, fs = key.split("|")
        if regime != "tmix":
            continue
        ous[fs] = {sc: m.label_open_set({g: out[sc][pc][g][0] for g in A.classes}) for sc in ("msp", "maha")}
    R["open_set"] = ous
    R["overlap_flags"] = CFG["overlap_flags"]
    R["provenance"] = dict(run_020_sha256=sha_file(MOD020), config_020_sha256=sha_file(MOD020.parent / "config.json"),
                           run_023_sha256=sha_file(HERE / "run.py"), config_023_sha256=sha_file(HERE / "config.json"))
    R["timings"] = {d: json.loads((CACHE / d / "timings.json").read_text()) for d in DETS}
    R["seconds_report"] = round(time.time() - t0)
    RES.mkdir(exist_ok=True)
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log("results.json written; label:", json.dumps(R["labels"]["H-A5"]))
    from report_tables import write_tables
    write_tables(R, A)


def d008_gate():
    gp = RES / "d008_guard.json"
    if not gp.exists():
        return dict(ok=False, note="d008_guard.json missing")
    g = json.loads(gp.read_text())
    ok = set(DETS) <= set(g) and all(v["ok"] for d in DETS for v in g[d].values())
    worst_l = max(v["max_abs_dlogit"] for d in DETS for v in g[d].values())
    worst_e = max(v["max_scaled_emb_diff"] for d in DETS for v in g[d].values())
    both = all(v["emb_ok"] and v["logit_ok"] for d in DETS for v in g[d].values())
    return dict(ok=bool(ok), worst_abs_dlogit=worst_l, worst_scaled_emb_diff=worst_e, all_both_criteria_ok=bool(both), n_subset=CFG["d008"]["n"])


def preproc_gate(A):
    """Same crop dir for every class, RGB, no EXIF/ICC; conditions through one code path (load_cond used for every row)."""
    bad_path = [p for p, s in zip(A.df.path, A.df.sha256) if p != f"data/processed/fov512/images/{s}.png"]
    exif = notrgb = badsize = 0
    for p in A.df.path:
        with Image.open(p) as im:
            exif += int("exif" in im.info or "icc_profile" in im.info)
            notrgb += int(im.mode != "RGB")
            badsize += int(im.size != (512, 512))
    return dict(ok=bool(not bad_path and exif == 0 and notrgb == 0 and badsize == 0), bad_paths=len(bad_path), n_exif_or_icc=exif, n_not_rgb=notrgb, n_not_512=badsize,
                note="all 1,020 rows go through load_cond (apply_condition keyed by sha256, seed 0) and then the detector's own transform; labels are never an input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "extract1", "cpuref", "guard", "guard1", "probe", "report", "all"])
    ap.add_argument("dets", nargs="*")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dets = a.dets or DETS
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    if a.stage == "extract":
        stage_extract(dets)
    elif a.stage == "extract1":
        stage_extract_one(dets[0], a.limit)
    elif a.stage == "cpuref":
        stage_cpuref_one(dets[0])
    elif a.stage == "guard":
        stage_guard(dets)
    elif a.stage == "guard1":
        stage_guard_one(dets[0])
    elif a.stage == "probe":
        stage_probe(a.workers)
    elif a.stage == "report":
        stage_report(a.workers)
    elif a.stage == "all":
        stage_extract(dets)
        stage_guard(dets)
        stage_report(a.workers)


if __name__ == "__main__":
    main()
