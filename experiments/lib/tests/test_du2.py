"""Amendment 10 (E1b): preprocess.process_image(mode="du2") = fov_resize -> resize back to the original size -> plain centre crop."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
import preprocess  # noqa: E402


def test_du2_downs_up_then_plain_center_crop(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (800, 1200, 3), dtype=np.uint8)
    src, dst = tmp_path / "a.png", tmp_path / "o.png"
    Image.fromarray(arr).save(src)
    st, info = preprocess.process_image(str(src), str(dst), 512, mode="du2")
    assert st == "ok"
    im = Image.fromarray(arr)
    ref = preprocess.fov_resize(im, 512).resize((1200, 800), Image.BICUBIC)
    ref = preprocess.center_crop(ref, 512)
    assert np.array_equal(np.asarray(Image.open(dst)), np.asarray(ref))
    # same field of view as the plain crop (differs only by resampling), not the fov512 field of view
    crop = preprocess.center_crop(im, 512)
    assert np.asarray(Image.open(dst)).shape == (512, 512, 3)
    assert 0 < np.abs(np.asarray(Image.open(dst)).astype(int) - np.asarray(crop)).mean() < 128


def test_du2_short_side_512_is_identical_to_crop_and_small_is_skipped(tmp_path):
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (512, 700, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    assert preprocess.process_image(str(tmp_path / "b.png"), str(tmp_path / "d.png"), 512, mode="du2")[0] == "ok"
    assert preprocess.process_image(str(tmp_path / "b.png"), str(tmp_path / "c.png"), 512, mode="crop")[0] == "ok"
    assert np.array_equal(np.asarray(Image.open(tmp_path / "d.png")), np.asarray(Image.open(tmp_path / "c.png")))
    Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(tmp_path / "s.png")
    assert preprocess.process_image(str(tmp_path / "s.png"), str(tmp_path / "x.png"), 512, mode="du2")[0] == "skipped_small"
