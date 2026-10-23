"""Phase 4 conditions: mavali_medium, screenshot_sim, jr(q,f); existing conditions must stay unchanged."""
import _path  # noqa
import hashlib

import numpy as np
from PIL import Image

import perturb


def _img(seed=0, n=512):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (n, n, 3), dtype=np.uint8))


def _arr(im):
    return np.asarray(im)


def test_existing_conditions_unchanged():
    im = Image.fromarray(np.random.default_rng(123).integers(0, 256, (64, 80, 3), dtype=np.uint8))
    h = lambda o: hashlib.sha256(_arr(o).tobytes()).hexdigest()[:16]
    # golden hashes of the pre-Phase-4 implementation (pure-PIL, libjpeg-independent conditions)
    assert h(perturb.apply_condition(im, "clean")) == "e92873979faa479f"
    assert h(perturb.apply_condition(im, "resize(0.5)")) == "83bdef68f3f897fc"
    assert h(perturb.apply_condition(im, "resize(0.7)")) == "cdd45d2f15030be6"
    # JPEG-based ones: defined by composition of the primitives (independent of libjpeg build)
    assert np.array_equal(_arr(perturb.apply_condition(im, "jpeg75_resize05")), _arr(perturb.jpeg(perturb.resize(im, 0.5), 75)))
    assert np.array_equal(_arr(perturb.apply_condition(im, "jpeg(90)")), _arr(perturb.jpeg(im, 90)))
    rng = np.random.default_rng(perturb.seed_from_key("abc", 0))
    f = float(rng.uniform(0.7, 1.0)); q = int(rng.integers(70, 101))
    assert np.array_equal(_arr(perturb.apply_condition(im, "social_sim", key="abc", seed=0)),
                          _arr(perturb.jpeg(perturb.resize(im, f), q)))
    assert perturb.SOCIAL_SCALE == (0.7, 1.0) and perturb.SOCIAL_Q == (70, 100)


def test_mavali_medium():
    im = _img()
    o = perturb.apply_condition(im, "mavali_medium")
    assert o.size == (384, 384)
    assert np.array_equal(_arr(o), _arr(perturb.jpeg(perturb.resize(im, 0.75), 50)))
    assert np.array_equal(_arr(o), _arr(perturb.apply_condition(im, "mavali_medium", key="other", seed=9)))  # key-free
    # order matters: jpeg first then resize differs
    assert not np.array_equal(_arr(o), _arr(perturb.resize(perturb.jpeg(im, 50), 0.75)))


def test_jr_grid():
    im = _img(n=128)
    for q in (60, 75, 90):
        for f in (0.5, 0.75, 1.0):
            o = perturb.apply_condition(im, f"jr({q},{f})")
            assert o.size == (round(128 * f),) * 2, (q, f)
            assert np.array_equal(_arr(o), _arr(perturb.apply_condition(im, f"jr({q},{f})")))
            exp = perturb.jpeg(im if f == 1.0 else perturb.resize(im, f), q)
            assert np.array_equal(_arr(o), _arr(exp)), (q, f)
    # equivalences with older named conditions
    a = lambda c: _arr(perturb.apply_condition(im, c))
    assert np.array_equal(a("jr(75,0.5)"), a("jpeg75_resize05"))
    assert np.array_equal(a("jr(75,1.0)"), a("jpeg(75)")) and np.array_equal(a("jr(90,1.0)"), a("jpeg(90)"))
    assert np.array_equal(a("jr(50,0.75)"), a("mavali_medium"))
    assert np.array_equal(a("jr( 60 , 0.5 )"), a("jr(60,0.5)"))
    for bad in ("jr(60)", "jr(a,0.5)", "jr(60,0.5,1)"):
        try:
            perturb.apply_condition(im, bad); raise RuntimeError("should fail")
        except ValueError:
            pass


def test_screenshot_sim_seeded_sizes_and_fit():
    im = _img(n=512)
    keys = [f"k{i}" for i in range(80)]
    outs = {k: perturb.apply_condition(im, "screenshot_sim", key=k, seed=0) for k in keys}
    for k in keys[:10]:  # determinism
        assert np.array_equal(_arr(outs[k]), _arr(perturb.apply_condition(im, "screenshot_sim", key=k, seed=0)))
    assert np.array_equal(_arr(outs["k0"]), _arr(perturb.screenshot_sim(im, "k0", 0)))
    assert not np.array_equal(_arr(outs["k0"]), _arr(perturb.apply_condition(im, "screenshot_sim", key="k0", seed=1)))
    base = {256: 0.5, 343: 0.67, 640: 1.25}  # 512 * {0.5, 0.67, 1.25}
    seen = set()
    for k, o in outs.items():
        assert o.mode == "RGB" and o.size[0] == o.size[1]
        # recover the factor: crop removes 2-5% per side from the resampled side s
        cands = [s for s in base if 2 * round(0.02 * s) <= s - o.size[0] <= 2 * round(0.05 * s)]
        assert len(cands) == 1, (k, o.size)
        seen.add(cands[0])
        assert o.size[0] >= 224, o.size  # downstream 224 center crop still fits (worst case 256 -> 230)
    assert seen == {256, 343, 640}  # all three factors occur over 80 keys
    # lossless: with factor 0.5 / 1.25 the output equals a plain bilinear resize + border crop (no JPEG)
    k = next(k for k, o in outs.items() if o.size[0] < 260)
    s = 256; d = (s - outs[k].size[0]) // 2
    ref = im.resize((s, s), Image.BILINEAR).crop((d, d, s - d, s - d))
    assert np.array_equal(_arr(outs[k]), _arr(ref))
    # worst case: 0.5 factor and 5% crop on a 512 image is still >= 224
    assert 256 - 2 * round(0.05 * 256) >= 224
