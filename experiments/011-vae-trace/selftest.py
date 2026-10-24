"""CPU-only tests for 011 run.py (F3 spectrum on synthetic images, pair bootstrap, decision rules, cache rules). No GPU, no VAE.
  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/011-vae-trace/selftest.py"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run011", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.set_mode(True)          # dry dirs: the self-test never touches pilot/ or cache/
import numpy as np
from PIL import Image


def noise(seed=0, sigma=15.0):
    rng = np.random.default_rng(seed)
    return np.clip(128 + sigma * rng.standard_normal((224, 224, 3)), 0, 255)


def peaks(a):
    v = R.f3_array(a)[:R.F3_BINS + 2 * len(R.F3_PEAK_K)]          # red channel
    n = R.F3_BINS
    pk = {("x", k): v[R.f3_peak_index(0, "x", k)] for k in R.F3_PEAK_K}
    pk.update({("y", k): v[R.f3_peak_index(0, "y", k)] for k in R.F3_PEAK_K})
    return v[:n], pk


def test_f3_shape_and_bins():
    v = R.f3_array(noise().astype(np.uint8))
    assert v.shape == (120,) and v.dtype == np.float32 and np.isfinite(v).all()
    assert R.F3_DIM == 3 * (32 + 8) == 120
    # f = 1/8 is exactly FFT bin 28 on the 224 crop
    assert 1 * 224 // 8 == 28 and abs(np.fft.fftfreq(224)[28] - 1 / 8) < 1e-12
    print("ok f3 shape")


def noise_stats(n=30):
    """Single-FFT-bin prominences of white noise are noisy (chi-square bins): estimate their spread over seeds."""
    P = [peaks(noise(100 + i).round())[1] for i in range(n)]
    return {k: (float(np.mean([p[k] for p in P])), float(np.std([p[k] for p in P]))) for k in P[0]}


NS = None


def z(pk, key):
    global NS
    NS = NS or noise_stats()
    m, sd = NS[key]
    return (pk[key] - m) / sd


def test_periodic_peak_x():
    a = noise()
    xs = np.arange(224)
    a = a + 10.0 * ((xs % 8) < 4)[None, :, None]                   # period-8 stripes along x
    spec, pk = peaks(a.clip(0, 255).round())
    assert pk[("x", 1)] > 1.0, pk                                  # >10x the neighbouring power at f = 1/8
    assert z(pk, ("x", 1)) > 3 and z(pk, ("x", 3)) > 3, {k: round(float(z(pk, k)), 1) for k in pk}   # square wave: odd harmonics 1/8, 3/8
    assert all(abs(z(pk, ("y", k))) < 4 for k in (1, 2, 3)), pk    # nothing along y
    assert abs(z(pk, ("x", 2))) < 4                                # no 2/8 component
    print("ok periodic x:", {k: round(float(v), 2) for k, v in pk.items() if k[0] == 'x'})


def test_periodic_peak_y_and_period4():
    a = noise(1)
    ys = np.arange(224)
    a = a + 10.0 * ((ys % 8) < 4)[:, None, None]
    pk = peaks(a.clip(0, 255).round())[1]
    assert pk[("y", 1)] > 1.0 and z(pk, ("y", 1)) > 3 and abs(z(pk, ("x", 1))) < 4, ({k: round(float(z(pk, k)), 1) for k in pk}, NS)
    b = noise(2)
    xs = np.arange(224)
    b = b + 12.0 * ((xs % 4) < 2)[None, :, None]                    # period 4 -> f = 2/8
    pk = peaks(b.clip(0, 255).round())[1]
    assert pk[("x", 2)] > 1.0 and z(pk, ("x", 2)) > 3 and abs(z(pk, ("x", 1))) < 4, ({k: round(float(z(pk, k)), 1) for k in pk}, NS)
    print("ok periodic y / period 4")


def test_noise_has_no_peaks_and_flat_spectrum():
    spec, pk = peaks(noise(3).round())
    assert all(abs(z(pk, k)) < 4 for k in pk), pk
    assert np.isfinite(spec).all() and spec.shape == (32,)
    print("ok noise")


def test_blur_lowers_highfreq_residual():
    from scipy import ndimage
    a = noise(4)
    bl = np.stack([ndimage.gaussian_filter(a[:, :, c], 1.5) for c in range(3)], 2)
    s0 = peaks(a.round())[0]
    s1 = peaks(bl.round())[0]
    assert s1[24:].mean() < s0[24:].mean() - 0.5
    print("ok blur")


def test_proc_image_and_texture_column():
    im = Image.fromarray(noise(5).astype(np.uint8))
    v = R.proc_image(im)
    assert v.shape == (121,) and v[-1] > 0
    flat = Image.fromarray(np.full((224, 224, 3), 128, np.uint8))
    assert R.proc_image(flat)[-1] < 1e-6 and np.isfinite(R.proc_image(flat)).all()
    print("ok proc_image")


def test_pair_plan():
    rng = np.random.default_rng(0)
    n, ncl = 600, 25
    cl = rng.integers(0, ncl, n)
    base = rng.standard_normal(n)                                  # per-photo content offset shared by the pair
    sr = base + rng.standard_normal(n)
    st = base + rng.standard_normal(n) + 1.5
    pp = R.PairPlan(cl, n_boot=300)
    r = pp.stats(sr, st)
    assert abs(r["auroc"]["value"] - 0.5) > 0.1
    assert r["auroc"]["ci_lo"] < r["auroc"]["value"] < r["auroc"]["ci_hi"]
    assert r["win"]["value"] > r["auroc"]["value"]                  # the shared per-photo offset is removed within a pair
    same = pp.stats(sr, sr)
    assert abs(same["auroc"]["value"] - 0.5) < 1e-9 and abs(same["win"]["value"] - 0.5) < 1e-9
    assert np.allclose(pp.W[0], 1) and pp.W.shape == (301, n)
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(st, sr).statistic / (n * n)
    assert abs(r["auroc"]["value"] - u) < 1e-9                      # point value = plain Mann-Whitney AUROC
    # whole clusters are drawn: rows of one cluster always carry the same multiplicity
    for c in range(ncl):
        w = pp.W[1:, cl == c]
        assert (w == w[:, :1]).all()
    print("ok pair plan", round(r["auroc"]["value"], 3), round(r["win"]["value"], 3))


def mk(v, lo, hi, fam="F2", n=50):
    rng = np.random.default_rng(int(v * 1000))
    return dict(value=v, ci_lo=lo, ci_hi=hi, boot=v + 0.01 * rng.standard_normal(n), family=fam)


def test_rules():
    tex = {R.CLEAN: mk(0.6, .55, .65), R.LAUN: mk(0.55, .5, .6)}
    ok = R.rules({R.CLEAN: mk(0.95, 0.90, 0.97), R.LAUN: mk(0.80, 0.75, 0.85)}, tex)
    assert ok["detectable"]["passed"] and ok["survives_laundering"]["passed"] and ok["trace_specific"]["headline_clean"]
    assert not ok["deprioritize_vae_trace_methods"]
    bad1 = R.rules({R.CLEAN: mk(0.92, 0.84, 0.95), R.LAUN: mk(0.80, 0.75, 0.85)}, tex)      # CI lower bound 0.84 <= 0.85
    assert not bad1["detectable"]["passed"] and bad1["deprioritize_vae_trace_methods"]
    bad2 = R.rules({R.CLEAN: mk(0.95, 0.9, 0.97), R.LAUN: mk(0.74, 0.72, 0.8)}, tex)        # 0.74 < 0.75
    assert not bad2["survives_laundering"]["passed"] and bad2["deprioritize_vae_trace_methods"]
    bad2b = R.rules({R.CLEAN: mk(0.95, 0.9, 0.97), R.LAUN: mk(0.80, 0.69, 0.85)}, tex)      # CI lower bound 0.69 <= 0.70
    assert not bad2b["survives_laundering"]["passed"]
    bad3 = R.rules({R.CLEAN: mk(0.66, 0.6, 0.7), R.LAUN: mk(0.56, 0.5, 0.6)}, {R.CLEAN: mk(0.6, .5, .7), R.LAUN: mk(0.5, .45, .55)})
    assert not bad3["detectable"]["passed"] and not bad3["trace_specific"]["headline_clean"]
    edge = R.rules({R.CLEAN: mk(0.95, 0.9, 0.97), R.LAUN: mk(0.80, 0.75, 0.85)}, {R.CLEAN: mk(0.86, .8, .9), R.LAUN: mk(0.7, .6, .8)})
    assert not edge["trace_specific"]["by_condition"][R.CLEAN]["passed"]                      # gap 0.09 < 0.10
    print("ok rules")


def test_cache_rules_and_paths():
    assert R.rt_ext("orig") == "jpg" and R.rt_ext("fov512") == "png"
    assert str(R.rt_own_path("orig", "ab")).endswith("rt/orig/ab.jpg") and str(R.rt_own_path("fov512", "ab")).endswith("rt/fov512/ab.png")
    dfs = {a: R.e2.load_df(a) for a in R.ARMS}
    qf = {}
    df = dfs["orig"]
    tr = df[(df.label == 0) & (df.split == "train")]
    for s, sp in zip(tr.sha256, tr.src_path):
        with Image.open(R.ROOT / sp) as im:
            qf[s] = R.jpeg_quality_estimate(im)
    chk = R.check_009_cache(dfs, qf)
    for arm in R.ARMS:
        c = chk[arm]
        assert c["n_train"] == 297 and c["missing"] == 0 and c["wrong_format"] == 0 and c["wrong_qf"] == 0 and c["wrong_size"] == 0, c
    q9 = __import__("json").loads((R.E2_CACHE / "qf.json").read_text())
    assert all(q9[s] == qf[s] for s in tr.sha256)
    print("ok 009 cache obeys the Amendment 7-8 rules (orig JPEG at source QF, fov512 PNG, 512x512)")


def test_groups_and_texture_sign():
    df = R.e2.load_df("orig")
    g = R.e2.source_groups(df)
    nat = df[df.pool == "vision_native"].iloc[0]
    same = df[(df.device == nat.device) & (df.vidx == nat.vidx)]
    assert len(same) == 3 and len(set(g[same.index])) == 1
    D = {("f3_plain", "c"): np.zeros((10, 121), np.float32), ("f3_rt", "c"): np.zeros((10, 121), np.float32)}
    D[("f3_plain", "c")][:, 120] = np.arange(10) + 5.0
    D[("f3_rt", "c")][:, 120] = np.arange(10) + 5.0 - 1.0          # round-trips blurrier (lower Laplacian) -> sign -1
    sgn, a = R.texture_sign(D, np.arange(10), "c")
    assert sgn == -1 and a < 0.5
    print("ok groups / texture sign")


def test_family_dims():
    sec = np.zeros((4, 4608), np.float32)
    f3 = np.zeros((4, 121), np.float32)
    assert R.fam_matrix("F1", sec, f3).shape[1] == 1536 and R.fam_matrix("F2", sec, f3).shape[1] == 4608
    assert R.fam_matrix("F3", sec, f3).shape[1] == 120 and R.fam_matrix("F4", sec, f3).shape[1] == 4728
    print("ok family dims")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except Exception:
                import traceback
                traceback.print_exc()
                fails += 1
    print("FAILED" if fails else "ALL OK", fails)
    sys.exit(1 if fails else 0)
