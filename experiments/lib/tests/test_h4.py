"""lib/h4.py: texture baseline, residualisation, paired bootstrap plans, validity gate and Amendment-6 label logic."""
import _path  # noqa
import numpy as np

import h4
import metrics


def test_laplacian_texture_flat_smooth_noisy_and_channel_mean():
    flat = np.full((32, 32, 3), 0.4, np.float32)
    assert h4.laplacian_texture(flat) < 1e-6
    yy, xx = np.mgrid[0:32, 0:32] / 32.0
    smooth = np.stack([xx, yy, xx * yy], -1).astype(np.float32)
    noisy = np.clip(smooth + np.random.default_rng(0).normal(0, 0.1, smooth.shape), 0, 1).astype(np.float32)
    assert h4.laplacian_texture(noisy) > 10 * h4.laplacian_texture(smooth)
    # a checkerboard of period 1: |4g - 4 neighbours| = 4 * amplitude at every interior pixel
    cb = (np.indices((16, 16)).sum(0) % 2).astype(np.float32)
    assert abs(h4.laplacian_texture(cb) - 4.0) < 1e-6
    assert abs(h4.laplacian_texture(np.stack([cb] * 3, -1)) - 4.0) < 1e-6


def test_texture_direction_set_on_calibration_only():
    assert h4.texture_direction([1, 2, 3], [4, 5, 6]) == 1.0          # fakes more textured
    assert h4.texture_direction([4, 5, 6], [1, 2, 3]) == -1.0
    assert h4.texture_direction([1, 2], [1, 2]) == 1.0                # tie -> +1 (AUROC 0.5), deterministic


def test_residualize_removes_log_texture_trend_and_uses_fit_rows_only():
    rng = np.random.default_rng(0)
    lap_c = np.exp(rng.uniform(-4, -1, 60))
    sc_c = 0.5 - 0.1 * np.log(lap_c) + rng.normal(0, 1e-3, 60)
    lap_t = np.exp(rng.uniform(-4, -1, 80))
    sc_t = 0.5 - 0.1 * np.log(lap_t) + rng.normal(0, 1e-3, 80)
    r, info = h4.residualize_on_log_texture(sc_c, lap_c, sc_t, lap_t)
    assert abs(info["b"] + 0.1) < 0.01 and info["r2"] > 0.99 and info["n_fit"] == 60
    assert np.abs(r).max() < 0.01 and r.shape == (80,)
    # a pure-texture score has chance AUROC after residualising: fakes just have more texture
    lap_f = lap_t * 5
    sc_f = 0.5 - 0.1 * np.log(lap_f) + rng.normal(0, 1e-3, 80)
    rr, _ = h4.residualize_on_log_texture(sc_c, lap_c, np.r_[sc_t, sc_f], np.r_[lap_t, lap_f])
    assert abs(h4.auroc_point(rr[:80], rr[80:]) - 0.5) < 0.1
    assert h4.auroc_point(sc_t, sc_f) < 0.3               # raw score is inverted by texture
    try:
        h4.residualize_on_log_texture([1.0, 2.0], [0.1, 0.2], [1.0], [0.1])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _toy(n_dev=6, per=5, gens=("a", "b"), nf=20, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    dev, sha, gen, lab = [], [], [], []
    for d in range(n_dev):
        for p in range(per):
            dev.append(f"D{d}"); sha.append(f"r{d}_{p}"); gen.append("real"); lab.append(0)
    for i in range(10):                                   # RAISE-like: no device
        dev.append(""); sha.append(f"x{i}"); gen.append("real"); lab.append(0)
    for g in gens:
        for i in range(nf):
            dev.append(""); sha.append(f"{g}{i}"); gen.append(g); lab.append(1)
    lab = np.array(lab)
    s = rng.normal(0, 1, len(lab)) + shift * lab
    return np.array(dev), np.array(sha), np.array(gen), lab, s


def test_plans_macro_matches_metrics_and_is_paired():
    dev, sha, gen, lab, s = _toy()
    n_boot = 200
    fs = h4.FakeSet(gen, ["a", "b"], lab == 1, n_boot)
    rs_v = h4.RealSet(np.where((lab == 0) & (dev != ""))[0], dev, sha, n_boot)
    rs_r = h4.RealSet(np.where((lab == 0) & (dev == ""))[0], dev, sha, n_boot)
    assert rs_v.n_clusters == 6 and rs_r.n_clusters == 10
    p, b = h4.macro_auroc(s, rs_v, fs)
    exp = np.mean([h4.auroc_point(s[rs_v.R], s[fs.F[g]]) for g in "ab"])
    assert abs(p - exp) < 1e-12 and b.shape == (n_boot,) and 0.5 < p < 1
    lo, hi = h4.ci(b)
    assert lo <= p <= hi or abs(p - lo) < 0.1
    # same reals + same fakes + same scores -> the paired difference is exactly 0 in every replicate
    p2, b2 = h4.macro_auroc(s, rs_v, fs)
    d = h4.dc(p, b, p2, b2)
    assert d["delta"] == 0 and d["ci"] == [0.0, 0.0]
    # two real pools share the fake resampling (Wf), the real side differs
    assert fs.Wf["a"].shape == (n_boot, 20) and np.all(fs.Wf["a"].sum(1) == 20)
    assert rs_v.Wr.shape == (n_boot, len(rs_v.R)) and np.all(rs_v.Wr.sum(1) == 30)
    # callable score (per-generator model) and the NaN guard
    pc, _ = h4.macro_auroc(lambda g: s, rs_v, fs)
    assert abs(pc - p) < 1e-12
    s_nan = s.copy(); s_nan[rs_r.R[0]] = np.nan
    assert np.isnan(h4.macro_auroc(s_nan, rs_r, fs)[0])
    assert np.isnan(h4.macro_auroc(s, h4.RealSet([], dev, sha, n_boot), fs)[0])
    # deterministic given the seed
    fs2 = h4.FakeSet(gen, ["a", "b"], lab == 1, n_boot)
    assert np.array_equal(fs.Wf["b"], fs2.Wf["b"])


def test_validity_gate_and_official_label_branches():
    g = h4.validity_gate(0.55, 0.62, 0.58)
    assert g["ok"] and g["evaluable"]
    assert not h4.validity_gate(0.48, 0.62, 0.58)["ok"] and "CI lower" in h4.validity_gate(0.48, 0.62, 0.58)["reason"]
    assert not h4.validity_gate(0.55, 0.57, 0.58)["beats_texture"]
    assert not h4.validity_gate(None, 0.6, 0.5)["evaluable"] and not h4.validity_gate(0.6, np.nan, 0.5)["ok"]
    auc = {0.01: 0.6, 0.02: 0.7, 0.05: 0.65}
    cil = {0.01: 0.5, 0.02: 0.6, 0.05: 0.55}
    gaps = {0.01: 0.02, 0.02: 0.12, 0.05: 0.0}
    out = h4.official_label(auc, cil, gaps, 0.01, 0.55)
    assert out["best_sigma"] == 0.02 and out["label"] == "SUPPORTED"          # best sigma = highest AUROC(ii)
    assert h4.official_label(auc, cil, gaps, 0.06, 0.55)["label"] == "INCONCLUSIVE"
    assert h4.official_label(auc, cil, {k: 0.01 for k in gaps}, 0.2, 0.55)["label"] == "REFUTED"
    assert h4.official_label(auc, cil, {0.01: 0.05, 0.02: 0.0, 0.05: 0.0}, 0.0, 0.55)["label"] == "INCONCLUSIVE"
    # gate failures override everything, whatever the gap
    assert h4.official_label(auc, cil, gaps, 0.01, 0.75)["label"] == "UNINFORMATIVE"          # texture beats RIGID
    assert h4.official_label(auc, {**cil, 0.02: 0.49}, gaps, 0.01, 0.55)["label"] == "UNINFORMATIVE"
    nan = float("nan")
    assert h4.official_label({s: nan for s in auc}, cil, gaps, 0.01, nan)["label"] == "NOT_EVALUABLE"
    assert h4.official_label(auc, cil, {s: nan for s in gaps}, 0.01, 0.5)["label"] == "NOT_EVALUABLE"
    assert h4.official_label(auc, cil, gaps, nan, 0.55)["label"] == "NOT_EVALUABLE"
    assert h4.best_sigma({0.01: nan}) is None


def test_mean_change_and_vc_dc():
    m = h4.mean_change([2.0, 4.0, np.nan], [1.0, 2.0])
    assert m["delta"] == 1.5 and abs(m["ratio"] - 2.0) < 1e-12
    assert np.isnan(h4.mean_change([], [1.0])["delta"])
    v = h4.vc(0.7, np.linspace(0.6, 0.8, 101))
    assert v["value"] == 0.7 and abs(v["ci"][0] - 0.605) < 1e-9
    assert np.isnan(h4.dc(float("nan"), np.zeros(3), 0.5, np.zeros(3))["delta"])


def test_fakeset_generator_stream_independent_of_other_generators():
    _, _, gen, lab, _ = _toy()
    a = h4.FakeSet(gen, ["a", "b"], lab == 1, 50)
    b = h4.FakeSet(gen, ["b"], lab == 1, 50)
    assert np.array_equal(a.Wf["b"], b.Wf["b"]) and not np.array_equal(a.Wf["a"], a.Wf["b"])


def test_jm_condition_seeded_fraction_and_rows_mask():
    import sys
    from pathlib import Path
    import pandas as pd
    from PIL import Image
    import perturb
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "004-phase4-features"))
    import p4features as P
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    applied = 0
    for i in range(200):
        a = perturb.apply_condition(img, "jm(85,95)", key=f"k{i}", seed=0)
        b = perturb.apply_condition(img, "jm(85,95)", key=f"k{i}", seed=0)
        assert np.array_equal(np.asarray(a), np.asarray(b)) and a.size == img.size
        applied += not np.array_equal(np.asarray(a), np.asarray(img))
    assert 70 < applied < 130                                   # ~50% of the images are compressed
    df = pd.DataFrame(dict(split=["train", "train", "test", "calib"], label=[1, 0, 1, 0]))
    assert P.rows_mask(df, "jm(85,95)").tolist() == [True, False, False, False]
    assert "jm(85,95)" not in P.ALL_CONDITIONS and P.JM_CONDITIONS == ["jm(85,95)"]


def test_realset_named_pools_get_independent_streams():
    dev, sha, gen, lab, s = _toy()
    R = np.where((lab == 0) & (dev == ""))[0]
    a, b = h4.RealSet(R, dev, sha, 40, name="raise"), h4.RealSet(R, dev, sha, 40, name="native")
    c = h4.RealSet(R, dev, sha, 40)
    assert not np.array_equal(a.Wr, b.Wr) and np.array_equal(a.Wr, h4.RealSet(R, dev, sha, 40, name="raise").Wr)
    assert np.array_equal(c.Wr, h4.RealSet(R, dev, sha, 40, name=None).Wr)
