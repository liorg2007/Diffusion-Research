"""CPU-only tests for 012 run.py (pure helpers, label logic, score orientation with mock probes). No GPU, no VAE, no DINOv2.
  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/012-self-ref/selftest.py"""
import importlib.util
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run012", HERE / "run.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
H.set_mode(True)          # dry dirs: the self-test never touches pilot/ or cache/
import numpy as np
from PIL import Image


def rnd_img(w=512, h=512, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 4, w // 4, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((w, h), Image.BICUBIC)


def test_crop8():
    assert H.crop8(rnd_img(512, 512)).size == (512, 512)
    assert H.crop8(rnd_img(256, 256)).size == (256, 256)
    im = rnd_img(358, 366)
    c = H.crop8(im)
    assert c.size == (352, 360) and c.size[0] % 8 == 0 and c.size[1] % 8 == 0
    a, b = np.asarray(im), np.asarray(c)
    assert np.array_equal(a[3:363, 3:355], b)          # centered
    print("ok crop8")


def test_l0():
    im = rnd_img(512, 512)
    a = H.l0(im, "min")
    assert a.size == (256, 256)                          # literal jpeg75_resize05 when it fits the 224 crop
    assert np.array_equal(np.asarray(a), np.asarray(H.perturb.jpeg75_resize05(im)))
    assert H.l0(im, "uniform").size == (512, 512)
    s = rnd_img(256, 256)
    assert H.l0(s, "min").size == (256, 256) and H.l0(s, "uniform").size == (256, 256)    # 128 < 224 -> back up
    m = rnd_img(360, 360)
    assert H.l0(m, "min").size == (360, 360)             # 180 < 224
    m2 = rnd_img(456, 456)
    assert H.l0(m2, "min").size == (228, 228)            # 228 >= 224: stays at half size
    assert np.array_equal(np.asarray(H.l0(im, "min")), np.asarray(H.l0(im, "min")))       # deterministic
    # same function on both sides: identical images -> identical outputs
    assert np.array_equal(np.asarray(H.l0(im.copy(), "uniform")), np.asarray(H.l0(im, "uniform")))
    print("ok l0 sizes and determinism")


def test_blur_monotone():
    im = rnd_img(256, 256, 1)
    x = np.asarray(im).astype(float)

    def hf(i):
        g = np.asarray(i).astype(float).mean(2)
        return np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean()
    v = [hf(H.blur(im, s)) for s in (0.25, 0.5, 1.0, 2.0)]
    assert v[0] > v[1] > v[2] > v[3], v
    assert hf(H.blur(im, 0.25)) <= hf(im) + 1e-9
    print("ok blur monotone", [round(t, 2) for t in v])


def test_match_sigma():
    g = H.BLUR_GRID
    means = [0.1 * i for i in range(len(g))]
    s, st = H.match_sigma(g, means, 0.35)
    assert st == "ok" and g[3] < s < g[4], (s, st)
    lo = np.log(g[3])
    assert abs(np.log(s) - (lo + 0.5 * (np.log(g[4]) - lo))) < 1e-9          # log-linear interpolation at the midpoint
    assert H.match_sigma(g, means, 99) == (g[-1], "out_of_range_high")
    assert H.match_sigma(g, means, -1) == (g[0], "out_of_range_low")
    assert H.match_sigma(g, means, means[5]) == (g[5], "ok")
    print("ok match_sigma")


class Lin:
    """mock probe: decision = X @ w + b (so shift = w.(phi(y)-phi(t)))"""
    def __init__(self, w, b=0.3):
        self.w, self.b = np.asarray(w, float), b

    def decision_function(self, X):
        return np.asarray(X) @ self.w + self.b


def test_scores_orientation_and_identity():
    rng = np.random.default_rng(0)
    d = 8
    w = rng.standard_normal(d)
    Pr = {k: Lin(w) for k in ("p", "m", "a2", "a2l")}
    n = 20
    t = rng.standard_normal((n, d))
    F = {"t": t, "l0t": t + 0.1, "l0ut": t - 0.1}
    # identity VAE: shift exactly 0 for every shift score
    F |= {"y": t.copy(), "l0y": F["l0t"].copy(), "l0uy": F["l0ut"].copy()}
    s = H.compute_scores(Pr, F, np.arange(n))
    for k in ("A", "B", "Bu"):
        assert np.allclose(s[k], 0), k
    for k in ("A2", "A2L"):                       # A2 is a decision on the difference vector itself: 0 vector -> -(bias)
        assert np.allclose(s[k], -0.3), k
    # round-trip moves a REAL image along +w (positive shift): fake-score must be negative, and a 'fake' that barely moves must score higher
    big = t + 2.0 * w / np.linalg.norm(w)
    small = t + 0.05 * w / np.linalg.norm(w)
    for nm, sh in (("big", big), ("small", small)):
        F2 = dict(F, y=sh, l0y=F["l0t"] + (sh - t), l0uy=F["l0ut"] + (sh - t))
        s2 = H.compute_scores(Pr, F2, np.arange(n))
        F[nm] = s2
    assert (F["big"]["A"] < 0).all() and (F["big"]["B"] < 0).all()
    assert (F["small"]["A"] > F["big"]["A"]).all() and (F["small"]["B"] > F["big"]["B"]).all()
    assert np.allclose(F["big"]["A"], -2.0 * np.linalg.norm(w), atol=1e-9)         # shift = w.(2 w/|w|) = 2|w|
    assert np.allclose(F["big"]["F2p"], (t @ w + 0.3))                              # single-image score = +decision
    print("ok score orientation / identity")


def test_crossfit_fpr():
    rng = np.random.default_rng(1)
    devs = [f"D{i:02d}" for i in range(20)]
    rows = []
    for d in devs:
        for p in H.POOLS:
            for _ in range(40):
                rows.append((d, p, 0, rng.standard_normal()))
    for _ in range(300):
        rows.append(("", "fake", 1, rng.standard_normal() + 5.0))
    dev, pool, lab, sc = (np.array([r[i] for r in rows]) for i in range(4))
    sc = sc.astype(float)
    lab = lab.astype(int)
    r = H.crossfit_fpr(sc, dev, pool, lab, 0.01)
    A, B = H.device_halves(dev[lab == 0])
    assert len(A) == 10 and len(B) == 10 and not (A & B)
    for p in H.POOLS:
        assert r["n"][p] == 20 * 40            # every device evaluated exactly once (by the other half's threshold)
        assert r["fpr"][p] < 0.03, r["fpr"]
    assert r["tpr"] > 0.9
    # threshold uses only the calibration half's natives
    m = (lab == 0) & (pool == "vision_native") & np.isin(dev, list(A))
    thr = H.metrics.threshold_for_fpr(sc[m], 0.01)
    assert abs(r["thr"][0] - thr) < 1e-12
    # a constant-offset shift of the evaluation half's reals must change FPR (the threshold really is cross-fitted)
    sc2 = sc.copy()
    sc2[(lab == 0) & np.isin(dev, list(B))] += 5
    r2 = H.crossfit_fpr(sc2, dev, pool, lab, 0.01)
    assert r2["fpr"]["vision_native"] > r["fpr"]["vision_native"]
    # calib rule
    c = H.calib_fpr(sc, pool, lab, rng.standard_normal(43))
    assert 0 <= c["fpr"]["vision_native"] < 0.1
    print("ok crossfit fpr", {p: round(v, 4) for p, v in r["fpr"].items()})


def mk_summary(**kw):
    c = H.CONDS
    base = dict(gates=dict(g1=True, g2=True, g3=True, g5=True),
                B={c[0]: dict(value=0.95, ci_lo=0.9), c[1]: dict(value=0.85, ci_lo=0.8), c[2]: dict(value=0.9, ci_lo=0.85)},
                f2_laun=0.65, nonsd_laun=0.8, d_tex={x: 0.3 for x in c}, d_pix={x: 0.3 for x in c}, fpr_max=0.01,
                blur=dict(delta=0.2, ci_lo=0.1))
    base.update(kw)
    return base


def test_label_logic():
    d = H.decide_label(mk_summary())
    assert d["base"] == "PASS" and d["label"] == "PASS" and d["blur_beaten"]
    assert H.decide_label(mk_summary(gates=dict(g1=False, g2=True)))["base"] == "UNINFORMATIVE"
    f = H.decide_label(mk_summary(B={H.CLEAN: dict(value=0.6, ci_lo=0.5), H.LAUN: dict(value=0.62, ci_lo=0.55), H.SOCIAL: dict(value=0.6, ci_lo=0.5)}))
    assert f["base"] == "FAIL"
    f2 = H.decide_label(mk_summary(f2_laun=0.83))                  # B 0.85 <= f2 + 0.03 -> fail even though >= 0.80
    assert f2["base"] == "FAIL"
    inc = H.decide_label(mk_summary(nonsd_laun=0.6))
    assert inc["base"] == "INCONCLUSIVE" and inc["criteria"]["non_sd_laundered_ge_0.75"] is False
    inc2 = H.decide_label(mk_summary(fpr_max=0.05))
    assert inc2["base"] == "INCONCLUSIVE"
    b1 = H.decide_label(mk_summary(blur=dict(delta=0.04, ci_lo=0.01)))
    assert b1["label"] == "PASS; NOT DIFFUSION-SPECIFIC" and not b1["blur_beaten"]
    b2 = H.decide_label(mk_summary(blur=dict(delta=0.2, ci_lo=-0.01)))
    assert not b2["blur_beaten"]
    b3 = H.decide_label(mk_summary(gates=dict(g1=False), blur=dict(delta=0.0, ci_lo=-0.1)))
    assert b3["label"] == "UNINFORMATIVE; NOT DIFFUSION-SPECIFIC"
    print("ok label logic")


def test_data_wiring():
    df = H.load_df("fov512")
    assert len(df) < 500
    tr, ca, te, fk = H.set_masks(df)
    assert tr.sum() > 0 and ca.sum() > 0 and te.sum() > 0 and fk.sum() > 0
    assert not (tr & te).any() and set(df.device[tr]).isdisjoint(set(df.device[te]))
    g = H.guard_shas(df)
    assert len(g) == len(set(g)) == 16 and set(g) <= set(df.sha256)
    z = dict(keys=np.array(["a", "c"]), x=np.arange(8, dtype=np.float32).reshape(2, 4))
    dfx = type(df)({"sha256": ["a", "b", "c"]})
    al = H.align(dfx, z)
    assert np.isnan(al["x"][1]).all() and (al["x"][2] == [4, 5, 6, 7]).all() and list(al["_present"]) == [True, False, True]
    print("ok data wiring")


def test_rms():
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 51, np.uint8)
    assert abs(H.rms(a, b) - 0.2) < 1e-6 and H.rms(a, a) == 0
    print("ok rms")


if __name__ == "__main__":
    test_crop8()
    test_l0()
    test_blur_monotone()
    test_match_sigma()
    test_scores_orientation_and_identity()
    test_crossfit_fpr()
    test_label_logic()
    test_data_wiring()
    test_rms()
    print("ALL OK")
