"""Amendment-3 additions to lib: grouped / class-weighted probe, cluster bootstrap, fov512 preprocessing.
The defaults of the pre-existing functions must stay identical."""
import _path  # noqa
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import metrics
import preprocess
import probe


# ----------------------------------------------------------------------------- probe
def _xy(n=120, d=6, seed=0, pos_frac=0.5, shift=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(int)
    X = rng.normal(size=(n, d))
    X[:, 0] += shift * y
    return X, y


def test_fit_probe_default_identical_to_original_recipe():
    X, y = _xy()
    new = probe.fit_probe(X, y, 3)
    clf = LogisticRegressionCV(Cs=probe.CS, cv=StratifiedKFold(5, shuffle=True, random_state=3), scoring="roc_auc",
                               max_iter=2000, random_state=3)
    old = make_pipeline(StandardScaler(), clf).fit(X, y)
    assert np.array_equal(new[-1].coef_, old[-1].coef_) and np.array_equal(new[-1].intercept_, old[-1].intercept_)
    assert new[-1].C_[0] == old[-1].C_[0]


def test_fit_probe_groups_keep_copies_in_one_fold():
    X, y = _xy(60)
    g = np.arange(60)
    Xa, ya, ga = np.tile(X, (4, 1)), np.tile(y, 4), np.tile(g, 4)     # 4 "augmented" copies per source image
    seen = {}
    orig = probe.LogisticRegressionCV

    def spy(*a, **kw):
        seen["cv"] = kw["cv"]
        return orig(*a, **kw)

    probe.LogisticRegressionCV = spy
    try:
        m = probe.fit_probe(Xa + np.random.default_rng(1).normal(0, 0.01, Xa.shape), ya, 0, groups=ga)
    finally:
        probe.LogisticRegressionCV = orig
    folds = seen["cv"]
    assert len(folds) == 5
    tested = []
    for tr, te in folds:
        assert not (set(ga[tr]) & set(ga[te])), "a source image appears on both sides of a CV split"
        tested += list(te)
    assert sorted(tested) == list(range(len(ya)))       # every row is tested exactly once
    assert m[-1].C_[0] in probe.CS
    # plain stratified CV does leak copies across folds (why the option exists)
    leak = [bool(set(ga[tr]) & set(ga[te])) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya)]
    assert any(leak)


def test_fit_probe_groups_validation_and_few_groups():
    X, y = _xy(40)
    try:
        probe.fit_probe(X, y, 0, groups=np.arange(10))
        raise AssertionError("length mismatch must raise")
    except ValueError:
        pass
    g = np.repeat(np.arange(3), 14)[:40]      # only 3 groups -> 3 folds instead of 5
    m = probe.fit_probe(X, y, 0, groups=g)
    assert m[-1].scores_[1].shape[0] == 3


def test_fit_probe_class_weight_changes_the_threshold_not_default():
    X, y = _xy(400, pos_frac=0.1, shift=0.8, seed=2)
    plain = probe.fit_probe(X, y, 0)
    bal = probe.fit_probe(X, y, 0, class_weight="balanced")
    sp, sb = probe.predict_scores(plain, X), probe.predict_scores(bal, X)
    assert sb[y == 1].mean() > sp[y == 1].mean() + 0.05 and bal[-1].class_weight == "balanced"
    assert plain[-1].class_weight is None


# ----------------------------------------------------------------------------- cluster bootstrap
def test_bootstrap_weights_structure_and_seed():
    cl = np.repeat(np.arange(6), [3, 3, 4, 5, 2, 3])
    st = np.array(["a"] * 7 + ["b"] * 5)
    Wr, Wf = metrics.bootstrap_weights(cl, st, n_boot=300, seed=4)
    assert Wr.shape == (300, 20) and Wf.shape == (300, 12)
    for c in range(6):                                    # clusters move together
        cols = np.where(cl == c)[0]
        assert (Wr[:, cols] == Wr[:, [cols[0]]]).all()
    assert (Wf[:, :7].sum(1) == 7).all() and (Wf[:, 7:].sum(1) == 5).all()   # stratified, sizes preserved
    assert Wr.sum(1).min() > 0
    W2 = metrics.bootstrap_weights(cl, st, n_boot=300, seed=4)
    assert np.array_equal(Wr, W2[0]) and np.array_equal(Wf, W2[1])
    assert not np.array_equal(Wr, metrics.bootstrap_weights(cl, st, n_boot=300, seed=5)[0])
    # 6 clusters drawn with replacement: the number of drawn clusters per replicate is 6
    cnt = np.stack([Wr[:, np.where(cl == c)[0][0]] for c in range(6)], 1)
    assert (cnt.sum(1) == 6).all()


def test_weighted_auroc_matches_sklearn_with_ties_and_unit_weights():
    rng = np.random.default_rng(0)
    sr, sf = np.round(rng.normal(0, 1, 50), 1), np.round(rng.normal(1, 1, 30), 1)     # rounding -> ties
    y = np.r_[np.zeros(50), np.ones(30)]
    a = metrics.weighted_auroc(sr, sf, np.ones((1, 50)), np.ones((1, 30)))[0]
    assert abs(a - roc_auc_score(y, np.r_[sr, sf])) < 1e-12
    # integer weights == duplicated rows
    wr, wf = rng.integers(0, 4, 50).astype(float), rng.integers(1, 3, 30).astype(float)
    b = metrics.weighted_auroc(sr, sf, wr[None], wf[None])[0]
    ref = roc_auc_score(np.r_[np.zeros(int(wr.sum())), np.ones(int(wf.sum()))], np.r_[np.repeat(sr, wr.astype(int)), np.repeat(sf, wf.astype(int))])
    assert abs(b - ref) < 1e-12


def test_grouped_bootstrap_ci_is_wider_than_iid_when_clusters_correlate():
    rng = np.random.default_rng(1)
    n_cl, per = 10, 30
    cl = np.repeat(np.arange(n_cl), per)
    dev_shift = rng.normal(0, 1.0, n_cl)                 # strong device effect on real scores
    sr = rng.normal(0, 0.3, n_cl * per) + dev_shift[cl]
    sf = rng.normal(1.0, 0.5, 200)
    g = metrics.grouped_bootstrap_auroc(sr, sf, cl, None, n_boot=400, seed=0)
    iid = metrics.grouped_bootstrap_auroc(sr, sf, np.arange(len(sr)), None, n_boot=400, seed=0)
    assert g["ci_hi"] - g["ci_lo"] > 1.5 * (iid["ci_hi"] - iid["ci_lo"])
    assert abs(g["value"] - iid["value"]) < 1e-12 and g["ci_lo"] < g["value"] < g["ci_hi"]


def test_grouped_bootstrap_diff_is_paired_and_seeded():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(8), 10)
    sr_a, sf_a = rng.normal(0, 1, 80), rng.normal(1.5, 1, 40)
    sr_b, sf_b = sr_a + rng.normal(0, 0.05, 80), sf_a + rng.normal(0, 0.05, 40)
    st = np.repeat(["x", "y"], 20)
    d0 = metrics.grouped_bootstrap_auroc_diff(sr_a, sf_a, sr_a, sf_a, cl, st, n_boot=200, seed=0)
    assert d0["delta"] == 0 and np.all(d0["boot"] == 0) and d0["ci_lo"] == 0 == d0["ci_hi"]
    d = metrics.grouped_bootstrap_auroc_diff(sr_a, sf_a, sr_b, sf_b, cl, st, n_boot=200, seed=0)
    a = metrics.grouped_bootstrap_auroc(sr_a, sf_a, cl, st, n_boot=200, seed=0)
    b = metrics.grouped_bootstrap_auroc(sr_b, sf_b, cl, st, n_boot=200, seed=0)
    assert np.allclose(d["boot"], a["boot"] - b["boot"])      # same resamples for both score sets
    d2 = metrics.grouped_bootstrap_auroc_diff(sr_a, sf_a, sr_b, sf_b, cl, st, n_boot=200, seed=0)
    assert np.array_equal(d["boot"], d2["boot"])
    assert abs(d["delta"] - (a["value"] - b["value"])) < 1e-12


# ----------------------------------------------------------------------------- fov512 preprocessing
def _save_jpeg(path, w, h, seed=0, orientation=None):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] //= 2                                   # left/right asymmetry -> orientation is detectable
    im = Image.fromarray(arr)
    ex = Image.Exif()
    if orientation:
        ex[0x0112] = orientation
    ex[0x010F] = "SomeMake"
    im.save(path, quality=92, exif=ex.tobytes())
    return im


def test_fov_mode_resizes_short_side_then_center_crops(tmp_path):
    src, dst = tmp_path / "a.jpg", tmp_path / "a.png"
    _save_jpeg(src, 1200, 800)
    st, info = preprocess.process_image(str(src), str(dst), 512, mode="fov")
    assert st == "ok" and (info["fov_resized_w"], info["fov_resized_h"]) == (768, 512)
    out = Image.open(dst)
    assert out.size == (512, 512) and out.mode == "RGB" and not out.info.get("exif")
    with Image.open(src) as im:
        ref = im.convert("RGB").resize((768, 512), Image.BICUBIC)
    ref = preprocess.center_crop(ref, 512)
    assert np.array_equal(np.asarray(out), np.asarray(ref))


def test_fov_mode_exif_orientation_and_square_and_equal_size(tmp_path):
    src, dst = tmp_path / "p.jpg", tmp_path / "p.png"
    _save_jpeg(src, 800, 1200, orientation=6)               # stored 800x1200, displayed rotated -> 1200x800
    st, info = preprocess.process_image(str(src), str(dst), 512, mode="fov")
    assert st == "ok" and (info["fov_resized_w"], info["fov_resized_h"]) == (768, 512)   # orientation applied first
    # an image already 512 on the short side and square: pure crop, no resampling
    s2, d2 = tmp_path / "s.png", tmp_path / "s2.png"
    arr = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(arr).save(s2)
    assert preprocess.process_image(str(s2), str(d2), 512, mode="fov")[0] == "ok"
    assert np.array_equal(np.asarray(Image.open(d2)), arr)


def test_fov_mode_never_upscales_and_default_mode_unchanged(tmp_path):
    small = tmp_path / "small.jpg"
    _save_jpeg(small, 900, 400)
    st, info = preprocess.process_image(str(small), str(tmp_path / "x.png"), 512, mode="fov")
    assert st == "skipped_small" and not (tmp_path / "x.png").exists() and info["orig_height"] == 400
    # default call == explicit mode="crop" == crop-only behaviour (no resize)
    big = tmp_path / "big.jpg"
    _save_jpeg(big, 1000, 700)
    assert preprocess.process_image(str(big), str(tmp_path / "d1.png"), 512)[0] == "ok"
    assert preprocess.process_image(str(big), str(tmp_path / "d2.png"), 512, mode="crop")[0] == "ok"
    a, b = np.asarray(Image.open(tmp_path / "d1.png")), np.asarray(Image.open(tmp_path / "d2.png"))
    assert np.array_equal(a, b)
    with Image.open(big) as im:
        assert np.array_equal(a, np.asarray(preprocess.center_crop(im.convert("RGB"), 512)))
    try:
        preprocess.process_image(str(big), str(tmp_path / "e.png"), 512, mode="nope")
        raise AssertionError
    except ValueError:
        pass


def test_preprocess_manifest_fov_and_rewrite_paths(tmp_path):
    rows = []
    for i, (w, h) in enumerate([(1000, 700), (700, 1000), (600, 300)]):
        p = tmp_path / f"i{i}.jpg"
        _save_jpeg(p, w, h, seed=i)
        rows.append(dict(path=str(p), label=i % 2, source_dataset="s", generator="g", camera_or_platform="c",
                         license="l", split="train", sha256=f"sha{i}"))
    df = pd.DataFrame(rows)
    pm, side, rep = preprocess.preprocess_manifest(df, tmp_path / "out", 512, mode="fov")
    assert rep["n_ok"] == 2 and rep["n_skipped_small"] == 1 and rep["mode"] == "fov" and set(side["mode"]) == {"fov"}
    assert list(pm.columns) == list(df.columns) and len(pm) == 2
    _, side0, rep0 = preprocess.preprocess_manifest(df[:2], tmp_path / "out0", 512)       # default sidecar unchanged
    assert "mode" not in side0.columns and "mode" not in rep0 and "fov_resized_w" not in side0.columns
    rw = preprocess.rewrite_paths(df, tmp_path / "out", root=tmp_path)
    assert list(rw.columns) == list(df.columns) and rw.path.tolist() == [f"out/images/sha{i}.png" for i in range(3)]
    assert rw.drop(columns="path").equals(df.drop(columns="path"))


def test_phase4_conditions_leave_224_px_on_512_crops():
    from perturb import apply_condition
    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    conds = (["clean", "jpeg75_resize05", "social_sim", "resize(0.5)", "jpeg(75)", "jpeg(90)", "mavali_medium",
              "screenshot_sim"] + [f"jr({q},{f})" for q in (60, 75, 90) for f in (0.5, 0.75, 1.0)])
    for c in conds:
        keys = [f"k{i}" for i in range(200)] if c in ("social_sim", "screenshot_sim") else ["k"]
        assert min(min(apply_condition(blank, c, key=k, seed=0).size) for k in keys) >= 224, c
