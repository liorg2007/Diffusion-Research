"""Identical preprocessing for both classes: RGB, EXIF-orientation applied, metadata stripped by
re-encoding from raw pixels, center crop (NO resize; optional mode="fov": short side -> 512 first), lossless PNG. Format info goes to a sidecar CSV."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from manifest import write_manifest

_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def _table_for_quality(q):
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip((_STD_LUMA * s + 50) // 100, 1, 255)


def jpeg_quality_estimate(img):
    """IJG quality (1-100) whose luminance table best matches the file's; None if no quant tables.
    Order-invariant comparison (sorted tables), so zigzag/natural order does not matter."""
    qt = getattr(img, "quantization", None)
    if not qt:
        return None
    t = np.sort(np.array(qt[min(qt)], dtype=float))
    errs = [np.abs(np.sort(_table_for_quality(q)) - t).sum() for q in range(1, 101)]
    return int(np.argmin(errs)) + 1


def center_crop(img, size):
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def fov_resize(img, size):
    """Resize so the SHORT side equals `size` (long side scaled by the same factor, rounded). PIL BICUBIC; PIL's
    resize widens the kernel when downscaling, i.e. it is antialiased (same function family as perturb.resize).
    Never called for short side < size (that would be an upscale; process_image skips such images)."""
    w, h = img.size
    nw, nh = (size, max(size, round(h * size / w))) if w <= h else (max(size, round(w * size / h)), size)
    return img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.BICUBIC)


def process_image(src, dst, size=512, mode="crop"):
    """Returns (status, info). status in {'ok','skipped_small','error'}.
    mode="crop" (default, D-009): center crop, NO resize. mode="fov" (D-013 `fov512`): resize the short side to
    `size` (bicubic, antialiased) then center-crop `size`x`size`; images with short side < size are NOT upscaled but
    returned as 'skipped_small' (and counted by the caller). Both read the RAW file, EXIF-orient and strip metadata."""
    if mode not in ("crop", "fov"):
        raise ValueError(f"unknown preprocessing mode {mode!r}")
    info = dict(orig_format=None, orig_jpeg_quality=None, orig_width=None, orig_height=None,
                orig_bytes=None, has_quant_table=0)
    try:
        info["orig_bytes"] = os.path.getsize(src)
        with Image.open(src) as im:
            info["orig_format"] = im.format
            info["orig_width"], info["orig_height"] = im.size
            q = jpeg_quality_estimate(im)
            info["orig_jpeg_quality"], info["has_quant_table"] = q, int(q is not None)
            im = ImageOps.exif_transpose(im).convert("RGB")
        if min(im.size) < size:
            return "skipped_small", info
        if mode == "fov":
            im = fov_resize(im, size)
            info["fov_resized_w"], info["fov_resized_h"] = im.size
        arr = np.asarray(center_crop(im, size))
        Image.fromarray(arr).save(dst, "PNG")  # fresh image: no EXIF / ICC / text chunks
        return "ok", info
    except Exception as e:  # corrupt file etc.
        info["error"] = repr(e)
        return "error", info


def preprocess_manifest(df, out_dir, size=512, mode="crop"):
    """Process every row. Returns (processed_manifest, sidecar_df, report).
    The processed manifest has the same schema; `path` points at <out_dir>/images/<sha256>.png and
    `sha256` stays the ORIGINAL file's hash (key for the sidecar). Writes manifest_processed.csv + sidecar.csv."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows, side = [], []
    for r in df.to_dict("records"):
        dst = out_dir / "images" / f"{r['sha256']}.png"
        status, info = process_image(r["path"], dst, size, mode)
        side.append(dict(sha256=r["sha256"], src_path=r["path"], label=r["label"], generator=r["generator"],
                         status=status, crop_size=size, **info) if mode == "crop" else
                    dict(sha256=r["sha256"], src_path=r["path"], label=r["label"], generator=r["generator"],
                         status=status, crop_size=size, mode=mode, **info))
        if status == "ok":
            rows.append({**r, "path": str(dst)})
    sidecar = pd.DataFrame(side)
    sidecar.to_csv(out_dir / "sidecar.csv", index=False)
    pm = pd.DataFrame(rows, columns=list(df.columns))
    write_manifest(pm, out_dir / "manifest_processed.csv")
    report = dict(n_total=len(df), n_ok=int((sidecar.status == "ok").sum()),
                  n_skipped_small=int((sidecar.status == "skipped_small").sum()),
                  n_error=int((sidecar.status == "error").sum()), crop_size=size)
    if mode != "crop":
        report["mode"] = mode
    return pm, sidecar, report


def rewrite_paths(df, out_dir, root=None):
    """Same rows/columns as `df`, with `path` -> <out_dir>/images/<sha256>.png (relative to `root` when given and
    out_dir lies under it; D-011 repo-relative paths). Used to build splits_fov512 from a splits file."""
    out = Path(out_dir) / "images"
    if root is not None:
        out = out.resolve().relative_to(Path(root).resolve())
    res = df.copy()
    res["path"] = [str(out / f"{sha}.png") for sha in df.sha256]
    return res
