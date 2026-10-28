"""Make demo samples: ~3 test-fold images per generator + 2 per real source as 768 px JPEG q95 thumbnails.

    env/.venv/bin/python -I poc/fingerprint-demo/make_samples.py     (after train_models.py)
Eligible images = in the TEST fold of the attribution model AND test-role for the detector (models/eligible.json), so neither model saw
them in training. Picks are seeded-random (no look at model outputs). samples/ is git-ignored (NC / no-redistribution licences).
HDR+ is skipped (raw originals deleted).
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_env()
from PIL import Image, ImageOps  # noqa: E402

REAL_NAMES = {"vision_native": "VISION (2010s phones, native)", "raise": "RAISE (DSLR raw)", "openimages": "Open Images",
              "unsplash": "Unsplash", "commons": "Wikimedia Commons"}
PER_GEN, PER_REAL, SHORT = 3, 2, 768


def thumb(src, dst):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    if s > SHORT:
        f = SHORT / s
        im = im.resize((round(w * f), round(h * f)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=95)
    return im.size


def main():
    el = json.loads((C.MODELS / "eligible.json").read_text())
    C.SAMPLES.mkdir(exist_ok=True)
    for p in C.SAMPLES.glob("*.jpg"):
        p.unlink()
    rng = random.Random(0)
    out = []
    for g in C.GENERATORS:
        pool = sorted([x for x in el["fakes"] if x["generator"] == g], key=lambda x: x["sha"])
        for i, x in enumerate(rng.sample(pool, min(PER_GEN, len(pool)))):
            name = f"fake_{g}_{i + 1}.jpg"
            size = thumb(C.ROOT / x["src_path"], C.SAMPLES / name)
            out.append(dict(id=name[:-4], file=name, kind="fake", true=g, true_label=C.GEN_LABELS[g], size=size,
                            dataset=x["source_dataset"], license=x["license"], sha=x["sha"][:12],
                            note="AI-generated; test fold of the attribution probe and test role of the detector."))
    for pool, nice in REAL_NAMES.items():
        rows = sorted([x for x in el["reals"] if x["pool"] == pool], key=lambda x: x["sha"])
        for i, x in enumerate(rng.sample(rows, PER_REAL)):
            name = f"real_{pool}_{i + 1}.jpg"
            size = thumb(C.ROOT / x["src_path"], C.SAMPLES / name)
            out.append(dict(id=name[:-4], file=name, kind="real", true=pool, true_label=nice, size=size,
                            dataset=x["source_dataset"], license=x["license"], sha=x["sha"][:12],
                            note="Real photo; test role of the detector (this source was in its training set)."))
    (C.SAMPLES / "samples.json").write_text(json.dumps(dict(
        samples=out, thumbnail="short side <= 768 px, JPEG q95 (a re-encode of the original: slightly different from the cached 512 crops)",
        skipped="HDR+ (raw originals deleted)", note="Images are NOT redistributable: samples/ is git-ignored."), indent=1))
    print(f"wrote {len(out)} samples to {C.rel(C.SAMPLES)}")


if __name__ == "__main__":
    main()
