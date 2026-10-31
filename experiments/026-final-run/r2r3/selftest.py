"""Synthetic self-tests of run.py (no real data, no DINOv2 weights, CPU only) incl. PLANTED LEAKS the checks must catch.

    env/.venv/bin/python -I experiments/026-final-run/r2r3/selftest.py   -> results/selftest.txt
"""
import importlib.util
import io
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r2r3run", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["r2r3run"] = R
spec.loader.exec_module(R)

import numpy as np
import pandas as pd
from PIL import Image

LINES, FAILS = [], []


def check(name, cond, extra=""):
    LINES.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}".rstrip())
    if not cond:
        FAILS.append(name)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


t0 = time.time()
r = R.r20()                                   # 020 module (chdir to repo root)
tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 1. manipulations
yy, xx = np.mgrid[0:224, 0:224]
base = np.stack([(128 + 100 * np.sin(xx / 7.0 + k) * np.cos(yy / 11.0)) + rng.normal(0, 25, (224, 224)) for k in range(3)], -1)
img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
a0 = np.asarray(img)
for m in R.MANIPS:
    o = np.asarray(R.apply_manip(img, m, "k"))
    check(f"manip {m}: 224x224x3 uint8", o.shape == (224, 224, 3) and o.dtype == np.uint8)
check("manip none: identity", (np.asarray(R.apply_manip(img, "none", "k")) == a0).all())
# tile shuffle
sh = np.asarray(R.apply_manip(img, "a_shuffle", "k1"))
tiles = lambda a: sorted(map(bytes, a.reshape(7, 32, 7, 32, 3).transpose(0, 2, 1, 3, 4).reshape(49, -1)))
check("shuffle: output is a permutation of the 49 tiles (multiset preserved)", tiles(sh) == tiles(a0))
check("shuffle: differs from the input", (sh != a0).any())
check("shuffle: deterministic per key", (np.asarray(R.apply_manip(img, "a_shuffle", "k1")) == sh).all())
check("shuffle: different keys give different permutations", (np.asarray(R.apply_manip(img, "a_shuffle", "k2")) != sh).any())
idx = Image.fromarray(np.repeat(np.arange(49, dtype=np.uint8).reshape(7, 7)[:, :, None].repeat(32, 0).repeat(32, 1), 3, 2))
check("shuffle: permutation depends on the key only (not on the image)", (np.asarray(R.apply_manip(idx, "a_shuffle", "k1"))[::32, ::32, 0] ==
                                                                          np.asarray(R.apply_manip(idx, "a_shuffle", "k1"))[::32, ::32, 0]).all()
      and len(set(np.asarray(R.apply_manip(idx, "a_shuffle", "k1"))[::32, ::32, 0].ravel().tolist())) == 49)
# blur / highpass / gray
bl = np.asarray(R.apply_manip(img, "b_blur", "k")).astype(float)
hf = lambda a: np.abs(np.diff(a, axis=1)).mean()
check("blur: removes high-frequency energy", hf(bl) < 0.5 * hf(a0.astype(float)), f"{hf(bl):.1f} vs {hf(a0.astype(float)):.1f}")
hp = np.asarray(R.apply_manip(img, "c_highpass", "k")).astype(float)
check("highpass: re-centred to mid-gray (mean ~128)", abs(hp.mean() - 128) < 3, f"mean {hp.mean():.1f}")
exp = np.clip(np.rint(a0 - R._blur(a0, 1.5) + 128), 0, 255)
check("highpass: equals round(img - blur + 128) exactly", (hp == exp).all())
check("highpass: contains little low-frequency energy (blurred residual std << residual std)", R._blur(hp.astype(np.uint8), 4.0).std() < 0.25 * hp.std())
gr = np.asarray(R.apply_manip(img, "d_gray", "k"))
check("gray: three equal channels", (gr[..., 0] == gr[..., 1]).all() and (gr[..., 1] == gr[..., 2]).all())
check("gray: matches PIL L luma", (gr[..., 0] == np.asarray(img.convert("L"))).all())
import inspect
check("manipulations take no label argument", list(inspect.signature(R.apply_manip).parameters) == ["img224", "manip", "key"])

# ------------------------------------------------------------------ 2. 224 crop + laundering path
from preprocess import center_crop
big = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
check("crop224: size>=224 is the plain centre crop (no resample)", (np.asarray(R.crop224(big.resize((256, 256)))) == np.asarray(center_crop(big.resize((256, 256)), 224))).all())
for f in (0.35, 0.25):
    o = R.crop224(__import__("perturb").apply_condition(big, f"jr(30,{f})"))
    check(f"crop224: factor {f} laundered image (<224) is upsampled to a 224x224 input", o.size == (224, 224))
pngs = []
for i in range(8):
    p = tmp / f"im{i}.png"
    Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)).save(p)
    pngs.append(str(p))
from perturb import apply_condition
a1 = np.asarray(apply_condition(Image.open(pngs[0]), "jpeg75_resize05", key="s", seed=0))
a2 = np.asarray(apply_condition(Image.open(pngs[0]), "jr(75,0.5)", key="s", seed=0))
check("grid cell (q75,f0.5) == jpeg75_resize05 (same code path)", (a1 == a2).all())
sizes = {c: R.model_input(pngs[0], "s", c, "none").size for c in R.GRID_CONDS}
check("all 30 grid cells give a 224x224 model input", set(sizes.values()) == {(224, 224)} and len(sizes) == 30)
check("model_input: screenshot_sim / social_sim fit", all(R.model_input(pngs[1], "s1", c, "a_shuffle").size == (224, 224) for c in ("screenshot_sim", "social_sim", "mavali_medium", "clean")))
twin = tmp / "twin.png"
Image.open(pngs[0]).save(twin)
check("IDENTICAL PROCESSING: same pixels + same key through the pipeline -> bit-identical input for any 'class'",
      all((np.asarray(R.model_input(pngs[0], "s", c, m)) == np.asarray(R.model_input(str(twin), "s", c, m))).all() for c in ("clean", "jr(45,0.35)") for m in R.MANIPS))

# ------------------------------------------------------------------ 3. embedder (toy backbone): shapes, batching, OOM retry, caching
import torch
from features import Backbone


class Toy(Backbone):
    def forward(self, x):
        return self.model(torch.nn.functional.adaptive_avg_pool2d(x, 8).flatten(1))


torch.manual_seed(0)
toy = Toy("toy", torch.nn.Linear(192, 12 * 1536), "hf")
emb = R.Embedder(bb=toy, device="cpu", batch=3)
imgs = [R.model_input(p, f"s{i}", "clean", "none") for i, p in enumerate(pngs)]
F1 = emb.embed(imgs)
check("embedder: shape [n, 12, 1536]", F1.shape == (8, 12, 1536))
emb2 = R.Embedder(bb=toy, device="cpu", batch=8)
check("embedder: batch size does not change features", np.abs(emb2.embed(imgs) - F1).max() < 1e-4)
calls = {"n": 0}
orig = toy.forward


def flaky(x):
    calls["n"] += 1
    if calls["n"] == 1:
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
    return orig(x)


toy.forward = flaky
emb3 = R.Embedder(bb=toy, device="cpu", batch=4)
F3 = emb3.embed(imgs)
check("embedder: OOM -> halves the batch and retries, same features", emb3.n_oom == 1 and np.abs(F3 - F1).max() < 1e-4)
toy.forward = lambda x: (_ for _ in ()).throw(RuntimeError("some other failure"))
try:
    R.Embedder(bb=toy, device="cpu").embed(imgs)
    check("embedder: non-OOM errors propagate", False)
except RuntimeError:
    check("embedder: non-OOM errors propagate", True)
toy.forward = orig
R.CACHE = tmp / "cache"
R.CACHE.mkdir()
Dt = dict(paths=pngs, shas=[f"s{i}" for i in range(8)])
check("extract_file: writes the cache file", R.extract_file(emb, Dt, "b_blur", "jr(30,0.25)") is True and R.feat_file("b_blur", "jr(30,0.25)").exists())
check("extract_file: resumable (second call is a no-op)", R.extract_file(emb, Dt, "b_blur", "jr(30,0.25)") is False)
check("getfeat: memmap shape", R.getfeat("b_blur", "jr(30,0.25)").shape == (8, 12, 1536))

# ------------------------------------------------------------------ 4. R3 surface metrics: planted cliff / smooth / shift
nq, nf = len(R.GRID_Q), len(R.GRID_F)
sev = np.add.outer(np.arange(nq), np.arange(nf)).astype(float)               # 0 (mild) .. 9
smooth = (0.9 - 0.075 * sev)[None]                                             # 0.9 .. 0.225 : small steps
cliff = np.where(sev < 5, 0.9, 0.2)[None]
M1 = R.surface_metrics(smooth, smooth)
check("surface: smooth decay -> max step drop 0.075 <= 0.20", abs(M1["max_step_drop"][0] - 0.075) < 1e-9)
M2 = R.surface_metrics(cliff, cliff)
check("surface: planted cliff -> max step drop 0.70 > 0.20", abs(M2["max_step_drop"][0] - 0.70) < 1e-9)
ext = (0.9 - 0.075 * np.maximum(sev - 2, 0))[None]                             # extended probe tolerates 2 more steps of severity
Ms = R.surface_metrics(smooth, ext)
check("surface: extended probe breaks later -> median shift > 0", Ms["median_shift"][0] > 0 and Ms["n_break_ext"][0] < Ms["n_break_frozen"][0],
      f"median shift {Ms['median_shift'][0]}, cells {Ms['n_break_frozen'][0]} -> {Ms['n_break_ext'][0]}")
Mz = R.surface_metrics(smooth, smooth)
check("surface: identical probes -> shift 0, n_break equal", Mz["median_shift"][0] == 0 and Mz["n_break_ext"][0] == Mz["n_break_frozen"][0])
check("surface: no cell breaks -> shift undefined (NaN)", np.isnan(R.surface_metrics(np.full((1, nq, nf), 0.9), np.full((1, nq, nf), 0.9))["median_shift"][0]))
check("break_index: never breaks -> length", R.break_index(np.array([[0.9, 0.8, 0.7]]), 0.5)[0] == 3 and R.break_index(np.array([[0.9, 0.4, 0.7]]), 0.5)[0] == 1)
check("verdict_threshold: ge/lt with CI-robust flags", R.verdict_threshold([0.8, 0.75, 0.85], 0.7, "ge") == (True, True) and R.verdict_threshold([0.72, 0.6, 0.8], 0.7, "ge") == (True, False)
      and R.verdict_threshold([0.3, 0.2, 0.4], 0.5, "lt") == (True, True) and R.verdict_threshold([0.55, 0.4, 0.7], 0.5, "lt") == (False, False))

# ------------------------------------------------------------------ 5. synthetic end-to-end through the real task/report code
nfam, Kc, D_ = 40, 8, 24
n = nfam * Kc
y = np.tile(np.arange(Kc), nfam)
groups = np.array([f"cap:{i // Kc}" for i in range(n)])
fold = r.make_folds(y, groups, 5, 0)
check("synthetic folds valid", r.check_grouped_folds(fold, groups)["ok"])
shas = [f"x{i:05d}" for i in range(n)]
dfs = pd.DataFrame(dict(sha256=shas, crop_sha=[f"c{i}" for i in range(n)], pool="fake"))
Dsyn = dict(df=dfs, y=y, groups=groups, fold=fold, shas=shas, paths=[""] * n, folds_sha="synthetic")
Mcls = np.random.default_rng(1).normal(size=(Kc, D_))
R.CACHE = tmp / "syn"
R.CACHE.mkdir()
R.RES = tmp / "res"
R.RES.mkdir()
R.B = 200
R.CFG["perm_control"]["n"] = 20
R.CFG["perm_control"]["p95_excess"] = 0.06      # 320 synthetic rows (real: 1020 rows, tolerance 0.03 as in 020); sampling noise only


def synth(manip, cond):
    import hashlib
    rg = np.random.default_rng(int(hashlib.sha256(f'{manip}|{cond}'.encode()).hexdigest()[:8], 16))
    amp_mid = {"none": 0.50, "a_shuffle": 0.50, "b_blur": 0.0, "c_highpass": 0.50, "d_gray": 0.40}[manip]
    amp_last = 0.02 if manip != "b_blur" else 0.0
    if cond in R.GRID_CONDS:
        q, f = eval(cond[2:])
        s = R.GRID_Q.index(q) + R.GRID_F.index(f)
        amp_mid *= max(0.05, 1 - 0.09 * s)
    elif cond == CFG_PRIM:
        amp_mid *= 0.9
    X = rg.normal(size=(n, 12, D_)).astype(np.float32)
    for b in range(1, 13):
        X[:, b - 1] += (amp_mid if b in R.MIDS else amp_last if b == 12 else 0.05) * Mcls[y]
    return X


CFG_PRIM = R.CFG["primary"]
base_prim = synth("none", CFG_PRIM)
for m in R.MANIPS:
    for c in R.CFG["r2_conds"]:
        np.save(R.feat_file(m, c), base_prim if (m == "none" and c == CFG_PRIM) else synth(m, c))
for c in R.GRID_CONDS:
    np.save(R.feat_file("none", c), base_prim if c == "jr(75,0.5)" else synth("none", c))   # grid cell (75,0.5) == primary
R._D["D"] = Dsyn
fitres = {}
for m in R.MANIPS:
    tasks = []
    for b in R.CFG["blocks"]:
        ev = list(R.CFG["r2_conds"]) + (R.GRID_CONDS if (m == "none" and b in R.MIDS) else [])
        tasks += [("r2", m, b, f, R.TRAIN, ev) for f in range(5)]
    fitres[("r2", m)] = [R.fit_eval_task(t) for t in tasks]
fitres[("ext", "none")] = [R.fit_eval_task(("ext", "none", b, f, R.TRAIN + R.EXT_EXTRA, list(R.CFG["r2_conds"]) + R.GRID_CONDS)) for b in R.MIDS for f in range(5)]
for m in R.MANIPS:
    pass
# the extended probe trains on grid-cell features: planted check that held-out launderings are rejected
check("PLANTED LEAK: held-out laundering in the training set is refused", raises(R.fit_eval_task, ("x", "none", 8, 0, R.TRAIN + ["screenshot_sim"], ["clean"])))
bad_fold = fold.copy()
g0 = np.where(groups == groups[0])[0]
bad_fold[g0[0]], bad_fold[g0[1]] = 0, 1
bad_fold[g0[2:]] = 2
Dleak = dict(Dsyn, fold=bad_fold)
R._D["D"] = Dleak
check("PLANTED LEAK: a caption family split across folds is caught by the grouped-split check", not r.check_grouped_folds(bad_fold, groups)["ok"])
check("PLANTED LEAK: train/test sharing a group raises inside the probe task", raises(R.fit_eval_task, ("x", "none", 8, 0, R.TRAIN, ["clean"])) or raises(R.fit_eval_task, ("x", "none", 8, 1, R.TRAIN, ["clean"])))
R._D["D"] = Dsyn
perm = {m: [R.perm_task((m, s, f)) for s in range(R.CFG["perm_control"]["n"]) for f in range(5)] for m in R.CFG["perm_control"]["manips"]}
guards = dict(parity=dict(ok=True, conds={"clean": dict(scaled=1e-5)}), d008=dict(ok=True, max_gpu_cpu_logit=1e-5, max_gpu_cpu_feat=1e-5))
res = R.stage_report(Dsyn, fitres, perm, guards)
r2 = res["r2"]
P = R.CFG["primary"]
check("report: tables.md and results.json written", (R.RES / "tables.md").exists() and (R.RES / "results.json").exists())
check("synthetic none: nested mid-block bacc well above chance", r2["none"]["nested_mid"][P][0] > 0.45, f"{r2['none']['nested_mid'][P][0]:.3f}")
check("planted: signal lives in mid blocks (mid - last > 0.2 unmanipulated)", r2["none"]["mid_minus_last"][P][1] > 0.2, str(r2["none"]["mid_minus_last"][P]))
check("planted: shuffle keeps the full signal -> retained ~1 and prediction (a) confirmed", abs(r2["a_shuffle"]["retained_nested"][0] - 1) < 0.15 and r2["verdicts"]["a_shuffle_keeps_ge_70"]["verdict"] == "confirmed",
      str(r2["a_shuffle"]["retained_nested"]))
check("planted: blur destroys the signal -> retained ~0 and prediction (b) confirmed", r2["b_blur"]["retained_nested"][0] < 0.25 and r2["verdicts"]["b_blur_keeps_lt_50"]["verdict"] == "confirmed",
      str(r2["b_blur"]["retained_nested"]))
check("planted: mid > last persists under (a)", r2["verdicts"]["mid_over_last_persists_under_a"]["verdict"] == "confirmed")
check("planted: gray retains less than shuffle but more than blur", r2["b_blur"]["retained_nested"][0] < r2["d_gray"]["retained_nested"][0] < r2["a_shuffle"]["retained_nested"][0] + 0.2,
      str(r2["d_gray"]["retained_nested"]))
check("retained(none) == 1 exactly", abs(r2["none"]["retained_nested"][0] - 1) < 1e-9)
check("B* is a mid block", r2["B_star"] in R.MIDS)
check("permutation gate passes at chance (all 5 manipulations)", res["gates"]["permutation"]["ok"], str({k: round(v["mean"], 3) for k, v in res["gates"]["permutation"]["by_manip"].items()}))
check("frozen R3 probe reproduces R2 primary (consistency gate)", res["gates"]["r3_frozen_matches_r2"]["ok"], str(res["gates"]["r3_frozen_matches_r2"]["values"]))
S = res["r3"]["surface"]["frozen"]
check("planted decay: frozen surface falls with severity (mild corner > severe corner + 0.2)", S[0][0][0] > S[-1][-1][0] + 0.2, f"{S[0][0][0]:.2f} -> {S[-1][-1][0]:.2f}")
check("r3 verdict keys present", set(res["r3"]["verdicts"]) == {"smooth_decay", "extended_shift_ge_1_step"})
check("all gates ok on clean synthetic data", res["all_gates_ok"], str({k: v["ok"] for k, v in res["gates"].items() if not v["ok"]}))

# leak demonstration: duplicated rows straddling folds memorise; grouped folds sit at chance; dedupe gate flags duplicates across classes
nn = 400
Xn = rng.normal(size=(nn // 2, 300)).astype(np.float32).repeat(2, 0)               # every row duplicated
yn = np.repeat(rng.integers(0, 4, nn // 2), 2)
gn = np.repeat([f"g{i}" for i in range(nn // 2)], 2)
leaky_fold = np.tile([0, 1], nn // 2)                                              # duplicates always on opposite sides
good_fold = r.make_folds(yn, gn, 5, 0)


def oof(fold_):
    pred = np.zeros(nn, int)
    for f in range(fold_.max() + 1):
        tr, te = np.where(fold_ != f)[0], np.where(fold_ == f)[0]
        m = r.fit_model("lr", Xn[tr], yn[tr], 10.0)
        pred[te] = r.logp_full(m, Xn[te], 4).argmax(1)
    return r.bacc_simple(yn, pred)


check("PLANTED LEAK: duplicates across the split memorise (bacc >> chance)", oof(leaky_fold) > 0.7, f"{oof(leaky_fold):.2f}")
check("grouped folds on the same data stay at chance", oof(good_fold) < 0.4, f"{oof(good_fold):.2f}")
check("PLANTED LEAK: assert_disjoint_groups raises on the leaky split", raises(r.assert_disjoint_groups, gn, np.where(leaky_fold == 0)[0], np.where(leaky_fold == 1)[0]))
dd = r.gate_dedupe(["a", "b", "a", "c"], [0, 1, 2, 3])
check("PLANTED LEAK: a pixel-duplicate in two classes fails the sha256 dedupe gate", not dd["ok"] and dd["n_cross_class_hashes"] == 1)
pg = r.gate_permutation(np.full(100, 0.4), 0.125, 0.01, 0.03)
check("PLANTED LEAK: permutation gate fails when permuted labels still score above chance", not pg["ok"])

LINES.append(f"\n{len(LINES) - len(FAILS)}/{len(LINES)} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
out = HERE / "results" / "selftest.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(LINES) + "\n")
print("\n".join(LINES))
sys.exit(1 if FAILS else 0)
