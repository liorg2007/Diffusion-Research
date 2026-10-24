"""E2 PILOT (docs/04-phase4-plan.md Amendment 8): idea #6 aligned-VAE probe, idea #7 VAE double-reconstruction ratio.

  env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage timing   # 20-image timing gate (decides the cut)
  env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage guard    # D-008 VAE CUDA-vs-CPU (32 images); must pass before vae
  env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage vae      # round-trips (train reals) + #7 e1/e2 (test rows)
  env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage feat     # DINOv2 features of the round-trips
  env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage analyze  # probes, metrics, labels, tables, probe D-008
  (--stage all runs them in this order; every stage is resumable from cache/)

Reuses: 004 feature pipeline (secondary backbone, load_cond) + p4features cache, 005's A0 recipe (fit_probe, grouped CV,
balanced; LOGO), lib/metrics cluster bootstrap, lib/preprocess.jpeg_quality_estimate, 006 critic's Laplacian texture.
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
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from PIL import Image

import metrics
import probe
import p4features as P
from preprocess import center_crop, jpeg_quality_estimate

_spec = importlib.util.spec_from_file_location("p4run", ROOT / "experiments" / "004-phase4-features" / "run.py")
p4run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4run)

CFG = json.loads((HERE / "config.json").read_text())
PILOT = HERE / "pilot"
CACHE = HERE / "cache"
SEED = CFG["seed"]
N_BOOT = CFG["n_boot"]
VAES = CFG["vaes"]
CONDS = CFG["conditions"]
GENS = CFG["six_generators"]
SD_FAM, OTHER_FAM = CFG["sd_family"], CFG["other_family"]
SPLITS = {a: ROOT / p for a, p in CFG["arms"].items()}
SUBDIR = {"orig": "pilot", "fov512": "pilot_fov512"}
BS_MAX = 4
BATCH_FEAT = 20
TOL = 1e-3
LABEL = "PILOT"


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + LABEL + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    PILOT.mkdir(exist_ok=True)
    with open(PILOT / "run.log", "a") as f:
        f.write(s + "\n")


def load_df(arm):
    return pd.read_csv(SPLITS[arm], keep_default_na=False, dtype={"vidx": str}).astype(
        {"in_scoring_set": bool, "b0_train": bool})


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


# ----------------------------------------------------------------------------- VAE
def get_dev():
    from features import get_device       # disables TF32 on CUDA
    return get_device()


def load_vae(name, dev):
    from diffusers import AutoencoderKL
    c = VAES[name]
    v = AutoencoderKL.from_pretrained(c["repo"], revision=c["revision"], local_files_only=True, torch_dtype=None)
    return v.to(dev).eval().requires_grad_(False)


def recon_f(vae, arr):
    """arr [B,H,W,3] uint8 (or float already in [0,1]*255 scale as uint8) -> float32 [B,H,W,3] in [0,1], clamped,
    NOT yet quantized. Encode with latent_dist.mode() (posterior mean), decode, (y+1)/2 clamp."""
    import torch
    dev = next(vae.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample
        y = ((y + 1) / 2).clamp(0, 1)
        return y.permute(0, 2, 3, 1).float().cpu().numpy()


def q8(f):
    return np.rint(np.asarray(f, np.float32) * 255.0).clip(0, 255).astype(np.uint8)


class Batcher:
    """Run fn on slices of an array with batch <= bs; on CUDA OOM (shared GPU) halve the batch / wait and retry."""

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
                else:
                    if tries > 60:
                        raise
                    time.sleep(10)
        return np.concatenate(out) if out else np.zeros((0,))


def load_arr(df_rows, cond, pool):
    imgs = list(pool.map(lambda r: np.asarray(p4run.load_cond(r[0], r[1], cond)), zip(df_rows.path, df_rows.sha256)))
    sz = {a.shape for a in imgs}
    assert len(sz) == 1, sz
    return np.stack(imgs)


def rows7(df, cut):
    """#7 test set: split==test natives + fakes (same shas for both arms). cut -> 150 natives / 30 fakes per generator."""
    nat = df[(df.label == 0) & (df.split == "test") & (df.pool == "vision_native")]
    fak = df[(df.label == 1) & (df.split == "test")]
    if cut:
        rng = np.random.default_rng(SEED)
        nat = nat.sort_values("sha256")
        nat = nat.iloc[np.sort(rng.choice(len(nat), 150, replace=False))]
        fak = pd.concat([g.sort_values("sha256").iloc[np.sort(rng.choice(len(g), 30, replace=False))]
                         for _, g in fak.groupby("generator")])
    return pd.concat([nat, fak]).reset_index(drop=True)


def read_timing():
    p = PILOT / "timing.json"
    if not p.exists():
        raise SystemExit("run --stage timing first (the 20-image timing gate decides the pre-registered cut)")
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------- stage timing
def stage_timing():
    import torch
    dev = get_dev()
    df = load_df("orig")
    r7 = rows7(df, False).sort_values("sha256")
    sub = pd.concat([r7[r7.label == 0].head(10), r7[r7.label == 1].head(10)])
    out = dict(device=str(dev), n_images=len(sub), image_size=512, pass_def="encode+decode of one 512x512 image, fp32, TF32 off",
               gpu_note="GPU is shared with other jobs (utilisation measured below)")
    with ThreadPoolExecutor(4) as pool:
        arr = load_arr(sub, "clean", pool)
    assert arr.shape[1:3] == (512, 512)
    for vn in VAES:
        vae = load_vae(vn, dev)
        b = Batcher()
        b(lambda a: recon_f(vae, a), arr[:2])       # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        b(lambda a: recon_f(vae, a), arr)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / len(arr)
        out[vn] = dict(sec_per_pass=dt, batch=b.bs, peak_mem_MiB=torch.cuda.max_memory_allocated() / 2 ** 20)
        log(f"timing {vn}: {dt:.3f} s per 512^2 pass (batch {b.bs})")
        del vae
        torch.cuda.empty_cache()
    try:
        import subprocess
        out["nvidia_smi_at_end"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                                   "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    worst = max(out[v]["sec_per_pass"] for v in VAES)
    out["worst_sec_per_pass"] = worst
    out["limit_s"] = CFG["timing_gate"]["limit_s"]
    out["cut_applied"] = bool(worst > out["limit_s"])
    out["rule"] = "if > 1 s per pass: #7 test set cut to 30 fakes per generator and 150 natives (Amendment 8)"
    jdump(PILOT / "timing.json", out)
    log(f"timing gate: worst {worst:.3f} s/pass -> cut_applied={out['cut_applied']}")


# ----------------------------------------------------------------------------- stage vae
def get_qf(df):
    p = CACHE / "qf.json"
    qf = json.loads(p.read_text()) if p.exists() else {}
    tr = df[(df.label == 0) & (df.split == "train")]
    for sha, sp in zip(tr.sha256, tr.src_path):
        if sha not in qf:
            with Image.open(ROOT / sp if not os.path.isabs(sp) else sp) as im:
                q = jpeg_quality_estimate(im)
            if q is None:
                raise SystemExit(f"{sp}: no JPEG quantization tables; cannot set the source quality factor")
            qf[sha] = q
    p.write_text(json.dumps(qf))
    return qf


def rt_path(arm, vn, sha):
    """orig arm: JPEG re-save at the source QF (Amendment 8). fov512 arm: PNG (Amendment 7-8 clarifications: the fov512 real
    crops are resampled PNGs, so a fresh JPEG would give only the fakes a format cue)."""
    return CACHE / "rt" / arm / vn / f"{sha}.{'jpg' if arm == 'orig' else 'png'}"


def save_rt(path, im_u8, arm, q):
    """Atomic write (tmp + os.replace) so a killed run never leaves a truncated file that the cache counts as done."""
    tmp = Path(str(path) + ".tmp")
    if arm == "orig":
        Image.fromarray(im_u8).save(tmp, "JPEG", quality=int(q))
    else:
        Image.fromarray(im_u8).save(tmp, "PNG")
    os.replace(tmp, path)


def source_groups(df):
    """CV group per row = the SOURCE PHOTO: 'device:vidx' for VISION reals (native, WA and FBH copies of one photo share
    it), the sha256 for everything else (fakes). Amendment 7-8 clarifications."""
    dev, vidx, sha = df.device.to_numpy(), df.vidx.to_numpy(), df.sha256.to_numpy()
    return np.array([f"{d}:{v}" if (d != "" and v != "") else s for d, v, s in zip(dev, vidx, sha)], dtype=object)


def vae_guard_ok():
    g = jdump_existing(PILOT / "d008_guard.json").get("vae", {})
    return len(g) == len(VAES) and all(v.get("ok") for v in g.values())


def stage_vae():
    import torch
    if not vae_guard_ok():
        raise SystemExit("D-008 VAE guard has not passed: run --stage guard first (the bulk VAE stage must follow it)")
    cut = read_timing()["cut_applied"]
    dev = get_dev()
    dfs = {a: load_df(a) for a in SPLITS}
    qf = get_qf(dfs["orig"])
    pool = ThreadPoolExecutor(4)
    for vn in VAES:
        vae = load_vae(vn, dev)
        b = Batcher()
        for arm, df in dfs.items():
            tr = df[(df.label == 0) & (df.split == "train")]
            todo = tr[[not rt_path(arm, vn, s).exists() for s in tr.sha256]]
            (CACHE / "rt" / arm / vn).mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for s in range(0, len(todo), 48):
                ch = todo.iloc[s:s + 48]
                arr = load_arr(ch, "clean", pool)
                y = q8(b(lambda a: recon_f(vae, a), arr))
                for sha, im in zip(ch.sha256, y):
                    save_rt(rt_path(arm, vn, sha), im, arm, qf[sha])
            log(f"rt {vn}/{arm}: {len(todo)} new round-trips ({len(tr) - len(todo)} cached), {time.time() - t0:.0f}s, batch {b.bs}")
            sel = rows7(df, cut)
            for cond in CONDS:
                f = CACHE / "r7" / f"{arm}__{cond}__{vn}.npz"
                f.parent.mkdir(exist_ok=True)
                if f.exists() and set(sel.sha256) <= set(np.load(f)["sha"].tolist()):
                    continue
                t0 = time.time()
                E1, E2, shas = [], [], []
                for s in range(0, len(sel), 48):
                    ch = sel.iloc[s:s + 48]
                    x = load_arr(ch, cond, pool)
                    q1 = q8(b(lambda a: recon_f(vae, a), x))
                    q2 = q8(b(lambda a: recon_f(vae, a), q1))
                    xf, f1, f2 = x.astype(np.float64) / 255, q1.astype(np.float64) / 255, q2.astype(np.float64) / 255
                    E1.append(np.sqrt(((xf - f1) ** 2).reshape(len(x), -1).sum(1)))
                    E2.append(np.sqrt(((f1 - f2) ** 2).reshape(len(x), -1).sum(1)))
                    shas += ch.sha256.tolist()
                np.savez(str(f) + ".tmp.npz", sha=np.array(shas), e1=np.concatenate(E1), e2=np.concatenate(E2))
                os.replace(str(f) + ".tmp.npz", f)
                log(f"r7 {vn}/{arm}/{cond}: {len(sel)} rows, {time.time() - t0:.0f}s, batch {b.bs}")
        del vae
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage guard (VAE part of D-008)
def stage_guard():
    import torch
    dev = get_dev()
    df = load_df("orig")
    r7 = rows7(df, False)
    rng = np.random.default_rng(SEED)
    r, f = sorted(r7.sha256[r7.label == 0]), sorted(r7.sha256[r7.label == 1])
    shas = [r[i] for i in rng.choice(len(r), 16, replace=False)] + [f[i] for i in rng.choice(len(f), 16, replace=False)]
    sub = df.set_index("sha256", drop=False).loc[shas]
    with ThreadPoolExecutor(4) as pool:
        arr = load_arr(sub, "clean", pool)
    g = jdump_existing(PILOT / "d008_guard.json")
    g["vae"] = {}
    for vn in VAES:
        vae = load_vae(vn, dev)
        b = Batcher()
        f1g = b(lambda a: recon_f(vae, a), arr)
        q1 = q8(f1g)
        f2g = b(lambda a: recon_f(vae, a), q1)
        del vae
        torch.cuda.empty_cache()
        vc = load_vae(vn, torch.device("cpu"))
        f1c = np.concatenate([recon_f(vc, arr[i:i + 2]) for i in range(0, len(arr), 2)])
        f2c = np.concatenate([recon_f(vc, q1[i:i + 2]) for i in range(0, len(arr), 2)])   # same quantized input on both devices
        d1, d2 = np.abs(f1g - f1c).max(), np.abs(f2g - f2c).max()
        mism = float((q8(f1g) != q8(f1c)).mean())
        g["vae"][vn] = dict(n=len(arr), max_abs_dpixel_x1=float(d1), max_abs_dpixel_x2=float(d2), tol=TOL,
                            uint8_mismatch_frac_x1=mism, ok=bool(max(d1, d2) <= TOL))
        log(f"D-008 VAE {vn}: max|dpixel| x1={d1:.2e} x2={d2:.2e} uint8 mismatch {mism:.2e} -> {'ok' if g['vae'][vn]['ok'] else 'FAIL'}")
        jdump(PILOT / "d008_guard.json", g)
        del vc
    if not all(v["ok"] for v in g["vae"].values()):
        raise SystemExit("D-008 ABORT: VAE CUDA vs CPU exceeds 1e-3")


def jdump_existing(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


# ----------------------------------------------------------------------------- stage feat
def stage_feat():
    dev = get_dev().type
    bb = p4run.secondary_backbone()
    p4run.check_design_vs_003(bb)
    for arm in SPLITS:
        df = load_df(arm)
        tr = df[(df.label == 0) & (df.split == "train")]
        for cond in CONDS:
            f = CACHE / "feats" / arm / f"{cond}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            if f.exists():
                continue
            keys, imgs = [], []
            t0 = time.time()
            with ThreadPoolExecutor(4) as pool:
                for vn in VAES:
                    paths = [rt_path(arm, vn, s) for s in tr.sha256]
                    imgs += list(pool.map(lambda r: p4run.load_cond(r[0], r[1], cond), zip(paths, tr.sha256)))
                    keys += [f"{vn}:{s}" for s in tr.sha256]
            X = []
            for i in range(0, len(imgs), 240):
                X.append(__import__("features").embed_images(bb, imgs[i:i + 240], BATCH_FEAT, dev))
            if getattr(bb, "force_cpu", False) and dev != "cpu":
                raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
            np.savez(str(f) + ".tmp.npz", keys=np.array(keys), feats=np.concatenate(X).astype(np.float32))
            os.replace(str(f) + ".tmp.npz", f)
            log(f"feats {arm}/{cond}: {len(keys)} round-trips, {time.time() - t0:.0f}s")


def load_rt_feats(arm, cond, shas):
    """[2*len(shas), 1536] primary features, order = VAES x shas (matches stage_feat)."""
    z = np.load(CACHE / "feats" / arm / f"{cond}.npz")
    pos = {k: i for i, k in enumerate(z["keys"].tolist())}
    X = z["feats"][[pos[f"{vn}:{s}"] for vn in VAES for s in shas]]
    return np.ascontiguousarray(X[:, -(X.shape[1] // 3):])


# ----------------------------------------------------------------------------- texture
def lap224(im):
    x = np.asarray(center_crop(im.convert("RGB"), 224), np.float32) / 255.0
    g = x.mean(2)
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def lap_table(arm, df):
    """{(cond, key): lap} for every df row (key = sha) and every round-trip (key = 'vae:sha')."""
    p = CACHE / f"lap_{arm}.csv.gz"
    if p.exists():
        t = pd.read_csv(p)
        return {(c, k): v for c, k, v in zip(t.cond, t.key, t.lap)}
    rows = []
    tr = df[(df.label == 0) & (df.split == "train")]
    jobs = [(c, s, pth) for c in CONDS for s, pth in zip(df.sha256, df.path)]
    jobs += [(c, f"{vn}:{s}", str(rt_path(arm, vn, s))) for c in CONDS for vn in VAES for s in tr.sha256]
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(lambda j: lap224(p4run.load_cond(j[2], j[1].split(":")[-1], j[0])), jobs))
    t = pd.DataFrame(dict(cond=[j[0] for j in jobs], key=[j[1] for j in jobs], lap=res))
    t.to_csv(p, index=False)
    return {(c, k): v for c, k, v in zip(t.cond, t.key, t.lap)}


# ----------------------------------------------------------------------------- bootstrap helpers
class Plan:
    """Cluster-bootstrap plan: reals by cluster, fakes by image stratified per generator (lib/metrics), shared by all
    methods (paired). Row 0 of the weights is the all-ones point estimate."""

    def __init__(self, real_clusters, fake_gen, gens=GENS, n_boot=N_BOOT, seed=SEED):
        self.gens = list(gens)
        fg = np.asarray(fake_gen)
        Wr, Wf = metrics.bootstrap_weights(real_clusters, fg, n_boot, seed)
        self.Wr = np.vstack([np.ones((1, Wr.shape[1])), Wr])
        self.Wf = np.vstack([np.ones((1, Wf.shape[1])), Wf])
        self.cols = {g: np.where(fg == g)[0] for g in self.gens}

    def macro(self, sr_by_g, sf_all, gens=None):
        """sr_by_g: {g: real scores} (or a single array for all g); sf_all: fake scores aligned with fake_gen.
        -> dict(value, ci_lo, ci_hi, boot[n_boot]) macro over `gens`."""
        gens = list(gens or self.gens)
        tot = 0.0
        for g in gens:
            sr = sr_by_g[g] if isinstance(sr_by_g, dict) else sr_by_g
            c = self.cols[g]
            tot = tot + metrics.weighted_auroc(sr, np.asarray(sf_all)[c], self.Wr, self.Wf[:, c])
        tot = tot / len(gens)
        lo, hi = np.percentile(tot[1:], [2.5, 97.5])
        return dict(value=float(tot[0]), ci_lo=float(lo), ci_hi=float(hi), boot=tot[1:])

    def per_gen(self, sr_by_g, sf_all):
        out = {}
        for g in self.gens:
            sr = sr_by_g[g] if isinstance(sr_by_g, dict) else sr_by_g
            c = self.cols[g]
            a = metrics.weighted_auroc(sr, np.asarray(sf_all)[c], self.Wr[:1], self.Wf[:1, c])[0]
            out[g] = float(a)
        return out


def diff(a, b):
    d = a["boot"] - b["boot"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(a["value"] - b["value"]), ci_lo=float(lo), ci_hi=float(hi))


def strip(d):
    return {k: v for k, v in d.items() if k != "boot"}


def auc(sr, sf):
    return float(metrics.weighted_auroc(np.asarray(sr, float), np.asarray(sf, float), np.ones((1, len(sr))), np.ones((1, len(sf))))[0])


def joint_cluster_auc(sr, cr, sf, cf, n_boot=N_BOOT, seed=SEED):
    """AUROC (fake=high) of sf vs sr; cluster bootstrap where reals AND fakes of one cluster (device) move together
    (an aligned fake carries its source real's device)."""
    cl = sorted(set(cr.tolist()) | set(cf.tolist()))
    idx = {c: i for i, c in enumerate(cl)}
    rng = np.random.default_rng(seed)
    K = len(cl)
    cnt = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)])
    Wr = cnt[:, [idx[c] for c in cr]].astype(float)
    Wf = cnt[:, [idx[c] for c in cf]].astype(float)
    b = metrics.weighted_auroc(sr, sf, Wr, Wf)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(value=auc(sr, sf), ci_lo=float(lo), ci_hi=float(hi))


def matched_auc(sr, lr, sf, lf, caliper=0.1):
    """Greedy 1:1 nearest-neighbour matching of fakes to reals on log-Laplacian (caliper in log units); AUROC of the
    matched sets (fake=high score)."""
    lr, lf = np.log(np.asarray(lr)), np.log(np.asarray(lf))
    used = np.zeros(len(lr), bool)
    pr, pf = [], []
    for j in np.argsort(lf):
        d = np.where(used, np.inf, np.abs(lr - lf[j]))
        i = int(np.argmin(d))
        if d[i] <= caliper:
            used[i] = True
            pr.append(i)
            pf.append(j)
    if len(pr) < 10:
        return dict(n_pairs=len(pr), auroc=None)
    return dict(n_pairs=len(pr), auroc=auc(np.asarray(sr)[pr], np.asarray(sf)[pf]),
                mean_abs_dlogLap=float(np.abs(lr[pr] - lf[pf]).mean()))


def strata_auc(s, lap, y, nb=5, min_n=10):
    """Within-quintile (of pooled Laplacian) AUROC; summary = equal-weight mean over strata with >= min_n of each class."""
    q = pd.qcut(pd.Series(lap).rank(method="first"), nb, labels=False).to_numpy()
    rows = []
    for b in range(nb):
        m = q == b
        nr, nf = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        a = auc(s[m & (y == 0)], s[m & (y == 1)]) if nr >= 2 and nf >= 2 else None
        rows.append(dict(bin=b, lap_lo=float(lap[m].min()), lap_hi=float(lap[m].max()), n_real=nr, n_fake=nf, auroc=a,
                         usable=bool(nr >= min_n and nf >= min_n)))
    ok = [r["auroc"] for r in rows if r["usable"]]
    return dict(strata=rows, mean_auroc_usable=float(np.mean(ok)) if ok else None, n_usable=len(ok))


# ----------------------------------------------------------------------------- stage analyze
def fit(X, y, groups):
    return probe.fit_probe(X, y, SEED, groups=groups, class_weight="balanced")


def cpu_feats(arm, cond, keys):
    bb = p4run.secondary_backbone()
    imgs = []
    for k in keys:
        vn, s = k.split(":")
        imgs.append(p4run.load_cond(rt_path(arm, vn, s), s, cond))
    f = __import__("features").embed_images(bb, imgs, 4, "cpu")
    return f


def stage_analyze():
    if not vae_guard_ok():
        raise SystemExit("D-008 VAE guard has not passed: refusing to analyze (run --stage guard)")
    timing = read_timing()
    cut = timing["cut_applied"]
    res = dict(label=LABEL, config=CFG, timing=timing, arms={}, labels={})
    long_rows = []
    probes_for_guard = {}
    for arm in SPLITS:
        df = load_df(arm)
        sub = SUBDIR[arm]
        lab, gen, sp, pool, dev_, sha = (df[c].to_numpy() for c in ("label", "generator", "split", "pool", "device", "sha256"))
        Xr = {c: P.load_features_aligned(df, "primary", c, subdir=sub) for c in CONDS}
        for c in CONDS:
            assert not np.isnan(Xr[c]).any(), (arm, c)
        tr = np.where((lab == 0) & (sp == "train"))[0]
        tr_nonnat = tr[pool[tr] != "vision_native"]
        Xrt = {c: load_rt_feats(arm, c, sha[tr]) for c in CONDS}
        n_al = len(Xrt["clean"])
        assert n_al == 594, n_al
        grp = source_groups(df)       # CV group = source photo (device:vidx); fakes keep their sha256
        al_group = np.tile(grp[tr], len(VAES))     # a round-trip carries its source real's photo group
        b0 = df.b0_train.to_numpy()
        te_nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
        fak = np.where(lab == 1)[0]
        fak = fak[np.argsort(gen[fak], kind="stable")]       # ordered by generator (all 120 per generator, 005 LOGO convention)
        plan6 = Plan(dev_[te_nat], gen[fak])
        R = res["arms"].setdefault(arm, {})

        # ---------------- models (resumable score cache)
        sc = CACHE / f"scores6_{arm}.npz"
        if sc.exists():
            S = {k: v for k, v in np.load(sc).items()}
        else:
            S = {}
            t0 = time.time()
            mA = fit(np.vstack([Xr["clean"][tr], Xrt["clean"]]), np.r_[np.zeros(len(tr)), np.ones(n_al)].astype(int),
                     np.concatenate([grp[tr], al_group]))
            S["A"] = np.stack([mA.decision_function(Xr[c]) for c in CONDS]).astype(np.float32)
            probes_for_guard[(arm, "A")] = mA
            for g in GENS:
                keep = np.where(b0 & (gen != g))[0]
                yk, Xk = lab[keep], Xr["clean"][keep]
                mB = fit(Xk, yk, grp[keep])
                mC = fit(np.vstack([Xk, Xrt["clean"]]), np.r_[yk, np.ones(n_al, int)], np.concatenate([grp[keep], al_group]))
                mD = fit(np.vstack([Xk, Xrt["clean"], Xr["clean"][tr_nonnat]]),
                         np.r_[yk, np.ones(n_al, int), np.zeros(len(tr_nonnat), int)],
                         np.concatenate([grp[keep], al_group, grp[tr_nonnat]]))
                for nm, m in (("B", mB), ("C", mC), ("C_all_reals", mD)):
                    S[f"{nm}__{g}"] = np.stack([m.decision_function(Xr[c]) for c in CONDS]).astype(np.float32)
                log(f"{arm}: LOGO {g} fitted (B C={mB[-1].C_[0]:.3g}, C C={mC[-1].C_[0]:.3g}, C_all C={mD[-1].C_[0]:.3g})")
            keep = np.where(b0)[0]
            mCf = fit(np.vstack([Xr["clean"][keep], Xrt["clean"]]), np.r_[lab[keep], np.ones(n_al, int)],
                      np.concatenate([grp[keep], al_group]))
            probes_for_guard[(arm, "C_full")] = mCf
            np.savez(sc, **S)
            log(f"{arm}: probes fitted in {time.time() - t0:.0f}s")
        if (arm, "A") not in probes_for_guard:       # guard needs the probes themselves: refit if scores were cached
            mA = fit(np.vstack([Xr["clean"][tr], Xrt["clean"]]), np.r_[np.zeros(len(tr)), np.ones(n_al)].astype(int),
                     np.concatenate([grp[tr], al_group]))
            keep = np.where(b0)[0]
            mCf = fit(np.vstack([Xr["clean"][keep], Xrt["clean"]]), np.r_[lab[keep], np.ones(n_al, int)],
                      np.concatenate([grp[keep], al_group]))
            probes_for_guard[(arm, "A")], probes_for_guard[(arm, "C_full")] = mA, mCf

        # ---------------- #6 metrics
        laps = lap_table(arm, df)
        six = {}
        macros = {}
        for ci, c in enumerate(CONDS):
            sA = S["A"][ci]
            mac = {"A": plan6.macro(sA[te_nat], sA[fak])}
            sets = {"A": (sA[te_nat], sA[fak], None)}
            for nm in ("B", "C", "C_all_reals"):
                srg = {g: S[f"{nm}__{g}"][ci][te_nat] for g in GENS}
                sfa = np.empty(len(fak))
                for g in GENS:
                    m = gen[fak] == g
                    sfa[m] = S[f"{nm}__{g}"][ci][fak[m]]
                mac[nm] = plan6.macro(srg, sfa)
                sets[nm] = (srg, sfa, None)
            macros[c] = mac
            six[c] = {}
            for nm, (sr, sf, _) in sets.items():
                six[c][nm] = dict(macro=strip(mac[nm]),
                                  sd_family=strip(plan6.macro(sr, sf, SD_FAM)), other_family=strip(plan6.macro(sr, sf, OTHER_FAM)),
                                  per_generator=plan6.per_gen(sr, sf))
            six[c]["C_minus_B"] = diff(mac["C"], mac["B"])
            six[c]["C_all_reals_minus_B"] = diff(mac["C_all_reals"], mac["B"])
            six[c]["A_minus_B"] = diff(mac["A"], mac["B"])
        R["six"] = six
        # reproduction of 005 pilot_v2 A0
        rep = {}
        p5 = ROOT / "experiments/005-nuisance-projection/pilot_v2" / arm / "results.json"
        if p5.exists():
            r5 = json.loads(p5.read_text())["arms"]["A0"]
            rep = {c: dict(ours=six[c]["B"]["macro"]["value"], e005=r5[c]["macro"]["auroc"],
                           abs_diff=abs(six[c]["B"]["macro"]["value"] - r5[c]["macro"]["auroc"])) for c in CONDS}
        R["B_vs_005_A0"] = rep

        # ---------------- texture check (#6)
        tx = {}
        for c in CONDS:
            lr = np.array([laps[(c, sha[i])] for i in tr])
            la = np.array([laps[(c, f"{vn}:{sha[i]}")] for vn in VAES for i in tr])
            cr, ca = dev_[tr], np.tile(dev_[tr], len(VAES))
            tx[c] = dict(
                auroc_fake_is_lower_texture=joint_cluster_auc(-lr, cr, -la, ca),
                auroc_fake_is_higher_texture=joint_cluster_auc(lr, cr, la, ca),
                median_lap_real=float(np.median(lr)), median_lap_aligned_fake=float(np.median(la)),
                mean_log_lap_ratio_fake_over_real=float(np.mean(np.log(la) - np.tile(np.log(lr), len(VAES)))))
            # A within texture-matched strata on the test set (natives vs the 720 fakes)
            idx = np.r_[te_nat, fak]
            y = np.r_[np.zeros(len(te_nat)), np.ones(len(fak))].astype(int)
            lap = np.array([laps[(c, sha[i])] for i in idx])
            sA = S["A"][CONDS.index(c)][idx]
            tx[c]["A_test_overall_auroc"] = auc(sA[y == 0], sA[y == 1])
            tx[c]["A_test_quintile_strata"] = strata_auc(sA, lap, y)
            tx[c]["A_test_matched_on_logLap"] = matched_auc(sA[y == 0], lap[y == 0], sA[y == 1], lap[y == 1])
            tx[c]["texture_baseline_test_auroc_fake_lower"] = auc(-lap[y == 0], -lap[y == 1])
        R["six_texture_check"] = tx

        # ---------------- #7
        sel = rows7(df, cut)
        nat7 = sel[sel.label == 0].reset_index(drop=True)
        fak7 = sel[sel.label == 1].sort_values(["generator", "sha256"], kind="stable").reset_index(drop=True)
        plan7 = Plan(nat7.device.to_numpy(), fak7.generator.to_numpy())
        calib_nat = df[(df.label == 0) & (df.split == "calib") & (df.pool == "vision_native")].sha256.tolist()
        trfak = df[(df.label == 1) & (df.split == "train")].sha256.tolist()
        seven = {}
        for c in CONDS:
            E = {}
            for vn in VAES:
                z = np.load(CACHE / "r7" / f"{arm}__{c}__{vn}.npz")
                pos = {s: i for i, s in enumerate(z["sha"].tolist())}
                E[vn] = {k: dict(nat=z[k][[pos[s] for s in nat7.sha256]], fak=z[k][[pos[s] for s in fak7.sha256]]) for k in ("e1", "e2")}
            ratio = {vn: {w: E[vn]["e1"][w] / np.maximum(E[vn]["e2"][w], 1e-12) for w in ("nat", "fak")} for vn in VAES}
            sc7 = {}
            sc7["ratio_min_over_vaes"] = {w: -np.min([ratio[vn][w] for vn in VAES], 0) for w in ("nat", "fak")}
            sc7["single_pass_min_over_vaes"] = {w: -np.min([E[vn]["e1"][w] for vn in VAES], 0) for w in ("nat", "fak")}
            for vn in VAES:
                sc7[f"ratio_{vn}"] = {w: -ratio[vn][w] for w in ("nat", "fak")}
                sc7[f"single_{vn}"] = {w: -E[vn]["e1"][w] for w in ("nat", "fak")}
            lp = lambda ss: np.array([laps[(c, s)] for s in ss])
            ln, lf = lp(nat7.sha256), lp(fak7.sha256)
            d_cal = auc(lp(calib_nat), lp(trfak))        # AUROC of lap as the fake score on calib natives vs train fakes
            dirn = 1.0 if d_cal >= 0.5 else -1.0
            sc7["texture_baseline"] = {"nat": dirn * ln, "fak": dirn * lf}
            seven[c] = dict(texture_direction=dict(sign=dirn, auroc_lap_as_fake_score_on_calib_vs_trainfakes=d_cal))
            m = {nm: plan7.macro(v["nat"], v["fak"]) for nm, v in sc7.items()}
            seven[c]["macro"] = {nm: strip(v) for nm, v in m.items()}
            seven[c]["per_generator"] = {nm: plan7.per_gen(v["nat"], v["fak"]) for nm, v in sc7.items()}
            seven[c]["sd_family"] = {nm: strip(plan7.macro(v["nat"], v["fak"], SD_FAM)) for nm, v in sc7.items()}
            seven[c]["other_family"] = {nm: strip(plan7.macro(v["nat"], v["fak"], OTHER_FAM)) for nm, v in sc7.items()}
            seven[c]["ratio_minus_single"] = diff(m["ratio_min_over_vaes"], m["single_pass_min_over_vaes"])
            seven[c]["ratio_minus_texture"] = diff(m["ratio_min_over_vaes"], m["texture_baseline"])
            seven[c]["n"] = dict(natives=len(nat7), fakes=len(fak7))
            for nm, v in sc7.items():
                for w, d_ in (("nat", nat7), ("fak", fak7)):
                    for sh, g_, p_, l_, s_ in zip(d_.sha256, d_.generator, d_.pool, d_.label, v[w]):
                        long_rows.append((arm, c, "7", nm, "", sh, p_, g_, int(l_), float(s_)))
        R["seven"] = seven

        # ---------------- #6 scores_long (test natives + all fakes)
        for ci, c in enumerate(CONDS):
            for nm in ("A", "B", "C", "C_all_reals"):
                models = ["full"] if nm == "A" else GENS
                for mdl in models:
                    arr = S["A"][ci] if nm == "A" else S[f"{nm}__{mdl}"][ci]
                    idxs = list(te_nat) + ([i for i in fak] if nm == "A" else [i for i in fak if gen[i] == mdl])
                    for i in idxs:
                        long_rows.append((arm, c, "6", nm, mdl, sha[i], pool[i], gen[i], int(lab[i]), float(arr[i])))
        log(f"{arm}: analysis done")

    # ---------------- D-008 probe guard BEFORE anything is written; both guards must have passed
    res["d008_probe"] = probe_guard(probes_for_guard)
    if not res["d008_probe"]["ok"]:
        raise SystemExit("D-008 ABORT: probe features/logits CUDA vs CPU exceed 1e-3; results.json NOT written")
    res["labels"] = labels(res)
    jdump(PILOT / "results.json", res)
    cols = ["arm", "cond", "task", "method", "model", "sha256", "pool", "generator", "label", "score"]
    pd.DataFrame(long_rows, columns=cols).to_csv(PILOT / "scores_long.csv.gz", index=False)
    write_tables(res)
    log("results.json / tables.md / scores_long.csv.gz written")


def validity_gate(ci_lo, ci_hi):
    """#7 validity gate (Amendment 6 as clarified in 'Amendment 7-8 clarifications'): valid iff the lower bound of the 95% CI
    of the ratio's clean AUROC is > 0.5. A ratio significantly BELOW chance is therefore UNINFORMATIVE, not FAIL.
    The literal Amendment-8 wording ('CI includes 0.5') is stored too, for transparency; it differs only when ci_hi < 0.5."""
    return dict(valid=bool(ci_lo > 0.5), rule_A6_lower_bound_gt_0p5=bool(ci_lo > 0.5),
                literal_ci_includes_0p5=bool(ci_lo <= 0.5 <= ci_hi))


def labels(res):
    L = {}
    A = res["arms"]["orig"]["six"]["jpeg75_resize05"]["A"]["macro"]
    a = A["value"]
    L["six_A"] = dict(value=a, ci=[A["ci_lo"], A["ci_hi"]], arm="orig", cond="jpeg75_resize05",
                      label=("PILOT PASS" if a >= 0.85 else "PILOT FAIL" if a < 0.75 else "PILOT INCONCLUSIVE (0.75 <= x < 0.85)"))
    B = res["arms"]["orig"]["six"]["jpeg75_resize05"]["B"]["macro"]["value"]
    d = res["arms"]["orig"]["six"]["jpeg75_resize05"]["C_minus_B"]
    if B > 0.96:
        L["six_C_vs_B"] = dict(B_logo_macro_jpeg75_resize05=B, label="no headroom / uninformative (B > 0.96)",
                               descriptive_C_minus_B=d)
    else:
        L["six_C_vs_B"] = dict(B_logo_macro_jpeg75_resize05=B, delta=d, label="PILOT PASS" if d["delta"] >= 0.02 else "PILOT FAIL")
    s7 = res["arms"]["orig"]["seven"]
    r_clean = s7["clean"]["macro"]["ratio_min_over_vaes"]
    cr, cl = s7["clean"]["macro"], s7["jpeg75_resize05"]["macro"]
    c1 = cr["ratio_min_over_vaes"]["value"] >= cr["single_pass_min_over_vaes"]["value"] + 0.05
    c2 = cl["ratio_min_over_vaes"]["value"] >= 0.65
    c3 = (cr["ratio_min_over_vaes"]["value"] > cr["texture_baseline"]["value"]
          and cl["ratio_min_over_vaes"]["value"] > cl["texture_baseline"]["value"])
    gate = validity_gate(r_clean["ci_lo"], r_clean["ci_hi"])
    lab = "PILOT UNINFORMATIVE (clean-AUROC CI lower bound <= 0.5)" if not gate["valid"] else (
        "PILOT PASS" if (c1 and c2 and c3) else "PILOT FAIL")
    L["seven"] = dict(label=lab, validity_gate=gate, arm="orig", ratio_clean=r_clean["value"], ratio_clean_ci=[r_clean["ci_lo"], r_clean["ci_hi"]],
                      single_clean=cr["single_pass_min_over_vaes"]["value"], texture_clean=cr["texture_baseline"]["value"],
                      ratio_laundered=cl["ratio_min_over_vaes"]["value"], single_laundered=cl["single_pass_min_over_vaes"]["value"],
                      texture_laundered=cl["texture_baseline"]["value"],
                      crit_ratio_ge_single_plus_0p05=bool(c1), crit_ratio_laundered_ge_0p65=bool(c2), crit_beats_texture_both_conds=bool(c3))
    return L


def probe_guard(probes):
    """D-008 for the new features: DINOv2 features of 32 aligned fakes, CUDA (cache) vs CPU, per arm and condition; scaled
    |dF| and |dlogit| of probes A and C_full."""
    out = dict(ok=True, tol=TOL, items={})
    g = jdump_existing(PILOT / "d008_guard.json")
    for arm in SPLITS:
        df = load_df(arm)
        tr = df[(df.label == 0) & (df.split == "train")]
        allk = [f"{vn}:{s}" for vn in VAES for s in tr.sha256]
        rng = np.random.default_rng(SEED)
        keys = [allk[i] for i in sorted(rng.choice(len(allk), 32, replace=False))]
        for c in CONDS:
            z = np.load(CACHE / "feats" / arm / f"{c}.npz")
            pos = {k: i for i, k in enumerate(z["keys"].tolist())}
            Fg = z["feats"][[pos[k] for k in keys]]
            Fc = cpu_feats(arm, c, keys)
            d = np.abs(Fg - Fc)
            scaled = float((d / np.maximum(1.0, np.abs(Fc).max(0, keepdims=True))).max())
            h = Fg.shape[1] // 3
            item = dict(max_abs_feature_diff=float(d.max()), max_scaled_feature_diff=scaled)
            for nm in ("A", "C_full"):
                m = probes[(arm, nm)]
                item[f"max_abs_dlogit_{nm}"] = float(np.abs(m.decision_function(Fg[:, -h:]) - m.decision_function(Fc[:, -h:])).max())
            item["ok"] = bool(scaled <= TOL and item["max_abs_dlogit_A"] <= TOL and item["max_abs_dlogit_C_full"] <= TOL)
            out["items"][f"{arm}/{c}"] = item
            out["ok"] &= item["ok"]
            log(f"D-008 probe {arm}/{c}: scaled dF={scaled:.2e} dlogit A={item['max_abs_dlogit_A']:.2e} "
                f"C_full={item['max_abs_dlogit_C_full']:.2e} -> {'ok' if item['ok'] else 'FAIL'}")
    g["probe"] = out
    jdump(PILOT / "d008_guard.json", g)
    return out


# ----------------------------------------------------------------------------- tables
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(m):
    return f"{m['value']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"


def write_tables(res):
    o = []
    w = o.append
    w(f"# E2 {LABEL} tables (experiments/009-vae-aligned). Source: pilot/results.json. PILOT: not a Phase 4 verdict, not counted toward the stop rule.\n")
    w(f"Timing gate: {json.dumps({k: v for k, v in res['timing'].items() if k in ('sd-vae-ft-mse', 'sdxl-vae', 'worst_sec_per_pass', 'cut_applied')})}\n")
    w("## Labels (label arm = orig)\n")
    w("```\n" + json.dumps(res["labels"], indent=1) + "\n```\n")
    for arm in SPLITS:
        R = res["arms"][arm]
        w(f"## #6 macro AUROC over 6 generators vs 358 test natives, arm {arm} (95% cluster-bootstrap CI)\n")
        w("| cond | A (aligned only) | B (=A0 LOGO) | C (B+aligned) | C_all_reals | C-B | A SD-family | A other | B SD-family | B other |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            s = R["six"][c]
            d = s["C_minus_B"]
            w(f"| {c} | {ci(s['A']['macro'])} | {ci(s['B']['macro'])} | {ci(s['C']['macro'])} | {ci(s['C_all_reals']['macro'])} | "
              f"{d['delta']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | {ci(s['A']['sd_family'])} | {ci(s['A']['other_family'])} | "
              f"{ci(s['B']['sd_family'])} | {ci(s['B']['other_family'])} |")
        w("\nPer generator AUROC of A:\n")
        w("| cond | " + " | ".join(GENS) + " |\n|---|" + "---|" * len(GENS))
        for c in CONDS:
            w(f"| {c} | " + " | ".join(f3(R["six"][c]["A"]["per_generator"][g]) for g in GENS) + " |")
        w(f"\nB vs 005 pilot_v2 A0 (reproduction): {json.dumps(R['B_vs_005_A0'])}\n")
        w(f"### Texture check, arm {arm}\n")
        w("| cond | AUROC reals vs aligned (fake=lower Laplacian) | (fake=higher) | median lap real / aligned | A test overall | A within-quintile mean (n usable) | A matched on log-Lap (pairs) | texture baseline test (fake=lower) |")
        w("|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            t = R["six_texture_check"][c]
            ms = t["A_test_matched_on_logLap"]
            w(f"| {c} | {ci(t['auroc_fake_is_lower_texture'])} | {ci(t['auroc_fake_is_higher_texture'])} | "
              f"{t['median_lap_real']:.4f} / {t['median_lap_aligned_fake']:.4f} | {t['A_test_overall_auroc']:.3f} | "
              f"{f3(t['A_test_quintile_strata']['mean_auroc_usable'])} ({t['A_test_quintile_strata']['n_usable']}) | "
              f"{f3(ms['auroc'])} ({ms['n_pairs']}) | {t['texture_baseline_test_auroc_fake_lower']:.3f} |")
        w("\nQuintile strata of A (clean): " + json.dumps(R["six_texture_check"]["clean"]["A_test_quintile_strata"]["strata"]) + "\n")
        w(f"## #7 macro AUROC, arm {arm} (fake-score fixed: -ratio / -single error; n = {R['seven']['clean']['n']})\n")
        w("| cond | ratio (min over VAEs) | single-pass (min) | texture baseline | ratio-single | ratio-texture | ratio SD-fam | ratio other | ratio sd-vae | ratio sdxl-vae |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            s = R["seven"][c]
            m = s["macro"]
            rs, rt = s["ratio_minus_single"], s["ratio_minus_texture"]
            w(f"| {c} | {ci(m['ratio_min_over_vaes'])} | {ci(m['single_pass_min_over_vaes'])} | {ci(m['texture_baseline'])} "
              f"(sign {s['texture_direction']['sign']:+.0f}) | {rs['delta']:+.3f} [{rs['ci_lo']:+.3f}, {rs['ci_hi']:+.3f}] | "
              f"{rt['delta']:+.3f} [{rt['ci_lo']:+.3f}, {rt['ci_hi']:+.3f}] | {f3(s['sd_family']['ratio_min_over_vaes']['value'])} | "
              f"{f3(s['other_family']['ratio_min_over_vaes']['value'])} | {f3(m['ratio_sd-vae-ft-mse']['value'])} | {f3(m['ratio_sdxl-vae']['value'])} |")
        w("")
    w("## D-008\n```\n" + (PILOT / "d008_guard.json").read_text() + "\n```\n")
    (PILOT / "tables.md").write_text("\n".join(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["timing", "vae", "guard", "feat", "analyze", "all"])
    a = ap.parse_args()
    PILOT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    order = ["timing", "guard", "vae", "feat", "analyze"] if a.stage == "all" else [a.stage]
    fns = dict(timing=stage_timing, vae=stage_vae, guard=stage_guard, feat=stage_feat, analyze=stage_analyze)
    for s in order:
        log(f"=== stage {s}")
        fns[s]()


if __name__ == "__main__":
    main()
