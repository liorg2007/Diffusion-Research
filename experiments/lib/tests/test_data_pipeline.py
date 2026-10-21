import _path  # noqa
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import perturb
import synth
from manifest import assign_splits, check_splits, make_row, read_manifest, validate_manifest, write_manifest
from preprocess import center_crop, jpeg_quality_estimate, preprocess_manifest


def _manifest():
    rows = [dict(path=f"r{c}_{i}", label=0, source_dataset="x", generator="real", camera_or_platform=f"cam{c}",
                 license="l", split="", sha256=f"s{c}{i}") for c in range(6) for i in range(20)]
    rows += [dict(path=f"f{g}_{i}", label=1, source_dataset="y", generator=g, camera_or_platform="n/a",
                  license="l", split="", sha256=f"t{g}{i}") for g in ("A", "B", "C") for i in range(40)]
    return pd.DataFrame(rows)


def test_group_split_no_camera_overlap_and_holdout():
    df = assign_splits(_manifest(), "C", seed=3)
    real = df[df.label == 0]
    te = set(real.camera_or_platform[real.split == "test"])
    other = set(real.camera_or_platform[real.split != "test"])
    assert te and other and not (te & other)
    c = df[df.generator == "C"]
    assert (c.split == "test").all() and len(c) == 40
    assert not df[(df.generator == "C") & df.split.isin(["train", "val", "calib"])].shape[0]
    for g in "AB":
        assert set(df.split[df.generator == g]) == {"train", "val", "test"}
    assert set(real.split[real.camera_or_platform.isin(other)]) == {"train", "val", "calib"}
    assert not (df.split[df.label == 1] == "calib").any()


def test_split_deterministic_and_seed_sensitive():
    a, b = assign_splits(_manifest(), "C", seed=1), assign_splits(_manifest(), "C", seed=1)
    assert a.equals(b)
    assert not a.split.equals(assign_splits(_manifest(), "C", seed=2).split)


def test_split_checks_catch_leaks():
    df = assign_splits(_manifest(), "C", seed=0)
    bad = df.copy()
    bad.loc[bad.generator == "C", "split"] = "train"
    try:
        check_splits(bad, "C"); raise RuntimeError("should fail")
    except AssertionError:
        pass
    bad = df.copy()
    r = bad[(bad.label == 0) & (bad.split == "test")].index[0]
    cam = bad.loc[r, "camera_or_platform"]
    bad.loc[bad[(bad.camera_or_platform == cam)].index[:1], "split"] = "train"
    try:
        check_splits(bad, "C"); raise RuntimeError("should fail")
    except AssertionError:
        pass
    one_cam = _manifest()
    one_cam["camera_or_platform"] = one_cam.camera_or_platform.where(one_cam.label == 1, "cam0")
    try:
        assign_splits(one_cam, "C"); raise RuntimeError("should fail")
    except ValueError:
        pass


def test_manifest_roundtrip_and_validation(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
    df = pd.DataFrame([make_row(p, 0, "d", "real", "camX", "cc0")])
    write_manifest(df, tmp_path / "m.csv")
    back = read_manifest(tmp_path / "m.csv")
    assert list(back.columns)[:8] == ["path", "label", "source_dataset", "generator", "camera_or_platform",
                                       "license", "split", "sha256"]
    assert len(back.sha256[0]) == 64
    bad = df.copy(); bad["generator"] = "sd"
    try:
        validate_manifest(bad); raise RuntimeError("should fail")
    except ValueError:
        pass


def test_jpeg_quality_estimate(tmp_path):
    arr = synth.gradient(64, np.random.default_rng(0), 5)
    for q in (30, 50, 75, 90, 95):
        p = tmp_path / f"q{q}.jpg"
        Image.fromarray(arr).save(p, quality=q)
        with Image.open(p) as im:
            assert jpeg_quality_estimate(im) == q, q
    Image.fromarray(arr).save(tmp_path / "x.png")
    with Image.open(tmp_path / "x.png") as im:
        assert jpeg_quality_estimate(im) is None


def test_preprocess_strips_exif_crops_skips_and_sidecar(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    df = synth.make_dataset(src, n_cams=2, n_real=2, gens=("g",), n_fake=2, size=96, n_small=2)
    # one JPEG with EXIF orientation 6 + an ICC blob, to check orientation + stripping
    arr = synth.gradient(100, np.random.default_rng(5), 5)[:, :80]
    ex = Image.Exif(); ex[0x010F] = "SECRET"; ex[0x0112] = 6
    p = src / "rot.jpg"
    Image.fromarray(arr).save(p, quality=85, exif=ex.tobytes(), icc_profile=b"\x00" * 32)
    from manifest import sha256_file
    df = pd.concat([df, pd.DataFrame([make_row(p, 0, "d", "real", "cam0", "l")])], ignore_index=True)
    out = tmp_path / "out"
    pm, side, rep = preprocess_manifest(df, out, size=64)
    assert rep["n_skipped_small"] == 2 and rep["n_ok"] == len(df) - 2 and rep["n_error"] == 0
    assert len(pm) == rep["n_ok"] and (out / "sidecar.csv").exists() and (out / "manifest_processed.csv").exists()
    for r in pm.itertuples():
        with Image.open(r.path) as im:
            assert im.size == (64, 64) and im.mode == "RGB" and im.format == "PNG"
            assert len(im.getexif()) == 0 and "icc_profile" not in im.info and "exif" not in im.info
    # pixels = orientation-corrected center crop of source (no resize)
    row = pm[pm.path.str.contains(sha256_file(p))].iloc[0]
    with Image.open(p) as im:
        ref = np.asarray(center_crop(ImageOps.exif_transpose(im).convert("RGB"), 64))
    assert np.array_equal(np.asarray(Image.open(row.path)), ref)
    s = side.set_index("sha256")
    assert s.loc[sha256_file(p), "orig_format"] == "JPEG" and s.loc[sha256_file(p), "orig_jpeg_quality"] == 85
    fk = side[side.label == 1].iloc[0]
    assert fk.orig_format == "PNG" and pd.isna(fk.orig_jpeg_quality) and fk.orig_width == 96
    assert (side.status == "skipped_small").sum() == 2


def _img(seed=0, n=64):
    return Image.fromarray(synth.gradient(n, np.random.default_rng(seed), 10))


def test_perturb_deterministic_and_correct():
    im = _img()
    for c in ("clean", "jpeg75_resize05", "resize(0.5)", "jpeg(75)", "social_sim"):
        a, b = perturb.apply_condition(im, c, key="k"), perturb.apply_condition(im, c, key="k")
        assert np.array_equal(np.asarray(a), np.asarray(b)), c
    assert np.array_equal(np.asarray(perturb.apply_condition(im, "clean")), np.asarray(im))
    j = perturb.apply_condition(im, "jpeg75_resize05")
    assert j.size == (32, 32)
    manual = perturb.jpeg(perturb.resize(im, 0.5), 75)
    assert np.array_equal(np.asarray(j), np.asarray(manual))
    swapped = perturb.resize(perturb.jpeg(im, 75), 0.5)  # wrong order must differ
    assert not np.array_equal(np.asarray(j), np.asarray(swapped))
    d = lambda q: np.abs(np.asarray(perturb.jpeg(im, q), float) - np.asarray(im, float)).mean()
    assert d(95) < d(75) < d(30)
    assert perturb.apply_condition(im, "resize(0.7)").size == (45, 45)


def test_social_sim_seeded_per_image():
    im = _img(n=100)
    sizes = {perturb.social_sim(im, key=f"k{i}", seed=0).size for i in range(30)}
    assert len(sizes) > 5 and all(70 <= s[0] <= 100 for s in sizes)
    a = np.asarray(perturb.social_sim(im, "k1", 0)); b = np.asarray(perturb.social_sim(im, "k1", 0))
    assert np.array_equal(a, b)
    assert perturb.seed_from_key("k1", 0) != perturb.seed_from_key("k1", 1)
    try:
        perturb.apply_condition(im, "bogus"); raise RuntimeError("should fail")
    except ValueError:
        pass
