"""Smoke test for the 4 baseline scorers. Run:  env/.venv/bin/python experiments/002-baselines/smoke.py [name ...]
Writes experiments/002-baselines/smoke_results.json (merged per scorer name). Does NOT run the evaluation set."""
import glob
import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
OUT = HERE / "smoke_results.json"

TP = ROOT / "third_party"


def synth(n=6, size=512, seed=0):
    """Smooth gradients + blobs + mild noise; plus one non-square image."""
    rng = np.random.default_rng(seed)
    ims = []
    for i in range(n):
        y, x = np.mgrid[0:size, 0:size] / size
        base = np.stack([np.sin(6 * x + i) * .5 + .5, np.cos(5 * y + i) * .5 + .5, (x * y)], -1)
        arr = np.clip(base * 200 + rng.normal(0, 4 + 4 * i, base.shape) + 20, 0, 255).astype(np.uint8)
        ims.append(Image.fromarray(arr))
    ims.append(ims[0].crop((0, 0, 640 if size >= 640 else size, size // 2 + 37)))   # non-square / odd size
    return ims


def run(mod, fn, ims, dev):
    os.environ["DD_DEVICE"] = dev
    for attr in ("_M", "_BACKBONE"):
        if hasattr(mod, attr):
            setattr(mod, attr, None)
    if hasattr(mod, "_MODELS"):
        mod._MODELS.clear()
    t = time.time()
    s = fn(ims)
    return np.asarray(s, float), time.time() - t


def throughput(fn, dev, n=24, size=512):
    os.environ["DD_DEVICE"] = dev
    ims = synth(n, size, seed=1)[:n]
    fn(ims[:2])                                   # warm-up (model already loaded for this device)
    t = time.time()
    fn(ims)
    return n / (time.time() - t)


def refs(name):
    """(images, expected-or-None, note)"""
    if name == "communityforensics":
        ims = [Image.open(p) for p in sorted(glob.glob(str(TP / "Community-Forensics/test_images/*.png")))]
        exp = np.log(np.array([0.9988, 0.9878, 0.9569, 0.9516, 0.7860]) / (1 - np.array([0.9988, 0.9878, 0.9569, 0.9516, 0.7860])))
        return ims, exp, "notebook eval_using_huggingface.ipynb: sigmoid outputs [0.9988,0.9878,0.9569,0.9516,0.7860] (DALL-E 2 test images); compared in logit space, tol 0.05 (4-digit rounding)"
    if name == "bfree":
        import pandas as pd
        d = TP / "B-Free/code/demo_images"
        res = pd.read_csv(d / "results.csv")
        ims = [Image.open(d / f) for f in res.filename]
        return ims, res.iloc[:, 1].to_numpy(float), "demo_images/results.csv (official GPU run), tol 0.05"
    if name.startswith("rine"):
        d = TP / "rine/demo"
        return [Image.open(d / "aadlygmazf.jpg"), Image.open(d / "zzrybysskm.png")], None, "demo real(jpg)/fake(png): no published numbers; expect p(real)<0.5<p(fake) for the ProGAN-4class model"
    if name.startswith("clipdet"):
        ims = [Image.open(p) for p in sorted(glob.glob(str(TP / "Community-Forensics/test_images/*.png")))[:3]] + [Image.open(TP / "rine/demo/aadlygmazf.jpg")]
        return ims, None, "no shipped expected outputs (needs synthbuster download); sanity on 3 DALL-E-2 fakes + 1 real photo"
    raise KeyError(name)


SCORERS = {
    "communityforensics": ("scorers.communityforensics", "score"),
    "bfree": ("scorers.bfree", "score"),
    "rine_ldm": ("scorers.rine", "score"),
    "rine_ldm_tencrop": ("scorers.rine", "score_ldm_tencrop"),
    "rine_4class": ("scorers.rine", "score_4class"),
    "clipdet_latent10k_plus": ("scorers.clipdet", "score"),
    "clipdet_latent10k": ("scorers.clipdet", "score_10k"),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SCORERS)
    res = json.load(open(OUT)) if OUT.exists() else {}
    for name in names:
        modname, fnname = SCORERS[name]
        mod = importlib.import_module(modname)
        fn = getattr(mod, fnname)
        r = {}
        ims = synth()
        sc, scm = {}, {}
        for dev in ("cpu", "mps"):
            sc[dev], t = run(mod, fn, ims, dev)
            r[f"synthetic_{dev}_scores"] = sc[dev].round(5).tolist()
            r[f"synthetic_{dev}_first_call_s"] = round(t, 2)
        r["synthetic_all_finite"] = bool(np.isfinite(sc["cpu"]).all() and np.isfinite(sc["mps"]).all())
        r["synthetic_n"] = len(ims)
        r["mps_vs_cpu_max_abs_diff"] = float(np.abs(sc["cpu"] - sc["mps"]).max())
        r["mps_vs_cpu_max_rel_diff"] = float((np.abs(sc["cpu"] - sc["mps"]) / (np.abs(sc["cpu"]) + 1e-3)).max())
        rims, exp, note = refs(name)
        rc, _ = run(mod, fn, rims, "cpu")
        rm, _ = run(mod, fn, rims, "mps")
        r["reference"] = dict(note=note, cpu=rc.round(5).tolist(), mps=rm.round(5).tolist(),
                              expected=None if exp is None else np.asarray(exp).round(5).tolist(),
                              max_abs_err_cpu=None if exp is None else float(np.abs(rc - exp).max()),
                              max_abs_err_mps=None if exp is None else float(np.abs(rm - exp).max()))
        big = synth(12, 512, seed=3)[:12]             # larger-batch MPS-vs-CPU consistency (guards silent MPS errors)
        bc, _ = run(mod, fn, big, "cpu")
        bm, _ = run(mod, fn, big, "mps")
        r["mps_vs_cpu_12img_max_abs_diff"] = float(np.abs(bc - bm).max())
        r["throughput_img_s_512px"] = {}
        for dev in ("mps", "cpu"):
            n = 16 if (dev == "cpu" and "tencrop" in name) else 24
            run(mod, fn, rims[:1], dev)               # (re)load on this device
            r["throughput_img_s_512px"][dev] = round(throughput(fn, dev, n=n), 2)
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if "scores" not in k}, indent=1), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
