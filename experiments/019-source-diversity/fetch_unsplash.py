#!/usr/bin/env python3
"""Fetch 600 Unsplash Lite photos for E10 real source UNS (Amendment 19).

Run:  env/.venv/bin/python -I experiments/019-source-diversity/fetch_unsplash.py
Reads:  data/raw/unsplash/meta/photos.tsv000  (Unsplash Lite Dataset 1.4.0)
Writes: data/raw/unsplash/images/<photo_id>.jpg
        experiments/019-source-diversity/unsplash_manifest.csv
"""
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TABLE = REPO / "data/raw/unsplash/meta/photos.tsv000"
OUT = REPO / "data/raw/unsplash/images"
MANIFEST = HERE / "unsplash_manifest.csv"

TARGET = 600
PER_GROUP = 3
SUBMITTED_BEFORE = pd.Timestamp("2022-01-01")
MIN_SOURCE_SIDE = 1200  # photo_width and photo_height in the table
MIN_SHORT_SIDE = 512    # decoded file
SEED = 0
PAUSE = 0.2
TIMEOUT = 60
RETRIES = 3
MAX_FILE_BYTES = 30_000_000
BYTE_CAP = 800_000_000  # images only; keeps the whole task under 1.2 GB with the table
ALLOWED_HOST = "images.unsplash.com"
LICENSE = "Unsplash Lite Dataset 1.4 terms"
UA = "Diffusion-Research-dataset-fetch/1.0 (academic research; Unsplash Lite 1.4.0)"
QUERY = "w=1600&q=95&fm=jpg"


def host_ok(url):
    p = urlparse(url)
    return p.scheme == "https" and p.netloc == ALLOWED_HOST


def fetch(url):
    """Return (bytes, None) or (None, error). 1 initial try + RETRIES retries."""
    last = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                if not host_ok(r.geturl()):
                    return None, f"redirected off host: {r.geturl()}"
                data = r.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    return None, "file larger than cap"
                return data, None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code != 429 and e.code < 500:
                break  # permanent client error, retrying will not help
        except Exception as e:  # network, timeout
            last = repr(e)
        if attempt < RETRIES:
            time.sleep(2 ** (attempt + 1))
    return None, last


def check_image(path):
    """Full decode with PIL. Return (width, height, None) or (None, None, error)."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None, None, f"format {im.format}"
            im.verify()
        with Image.open(path) as im:
            im.load()  # forces a full decode
            w, h = im.size
        if min(w, h) < MIN_SHORT_SIDE:
            return w, h, f"short side {min(w, h)} < {MIN_SHORT_SIDE}"
        return w, h, None
    except Exception as e:
        return None, None, f"decode error: {e!r}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidates():
    df = pd.read_csv(TABLE, sep="\t", dtype=str, keep_default_na=False)
    df = df.drop_duplicates(subset="photo_id")
    ts = pd.to_datetime(df["photo_submitted_at"], errors="coerce")
    w = pd.to_numeric(df["photo_width"], errors="coerce")
    h = pd.to_numeric(df["photo_height"], errors="coerce")
    ids_ok = df["photo_id"].str.match(r"^[A-Za-z0-9_-]+$")
    url_ok = df["photo_image_url"].map(host_ok)
    keep = (ts < SUBMITTED_BEFORE) & (w >= MIN_SOURCE_SIDE) & (h >= MIN_SOURCE_SIDE) & ids_ok & url_ok
    sub = df[keep].copy()
    sub = sub.sort_values("photo_id", kind="mergesort")
    order = np.random.default_rng(SEED).permutation(len(sub))
    sub = sub.iloc[order]
    print(f"table rows={len(df)} eligible={len(sub)}", flush=True)
    return sub


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cand = load_candidates()

    rows = []
    per_group = Counter()
    failures = []
    total_bytes = 0
    n_net = 0
    for rec in cand.itertuples(index=False):
        if len(rows) >= TARGET:
            break
        group = rec.photographer_username
        if per_group[group] >= PER_GROUP:
            continue
        pid = rec.photo_id
        base = rec.photo_image_url
        url = base + ("&" if "?" in base else "?") + QUERY
        dest = OUT / f"{pid}.jpg"

        w = h = None
        if dest.exists():
            w, h, err = check_image(dest)
            if err:
                dest.unlink()  # stale or broken from an earlier run; refetch
                w = h = None
        if w is None:
            if total_bytes > BYTE_CAP:
                print(f"BYTE CAP reached at {total_bytes} bytes; stopping", flush=True)
                break
            data, err = fetch(url)
            n_net += 1
            time.sleep(PAUSE)
            if err:
                failures.append((pid, "download", err))
                print(f"FAIL {pid}: {err}", flush=True)
                continue
            tmp = OUT / f"{pid}.jpg.part"
            tmp.write_bytes(data)
            w, h, err = check_image(tmp)
            if err:
                tmp.unlink()
                failures.append((pid, "verify", err))
                print(f"SKIP {pid}: {err}", flush=True)
                continue
            os.replace(tmp, dest)

        size = dest.stat().st_size
        total_bytes += size
        per_group[group] += 1
        author = f"{rec.photographer_first_name} {rec.photographer_last_name}".strip()
        rows.append({
            "image_id": pid,
            "path": str(dest.relative_to(REPO)),
            "bytes": size,
            "sha256": sha256_file(dest),
            "width": w,
            "height": h,
            "license": LICENSE,
            "author": author,
            "group": group,
            "submitted_at": rec.photo_submitted_at,
            "url": url,
        })
        if len(rows) % 50 == 0:
            print(f"ok={len(rows)} net_requests={n_net} bytes={total_bytes}", flush=True)

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)

    short = np.array([min(r["width"], r["height"]) for r in rows])
    print(f"photos={len(rows)} target={TARGET}")
    print(f"total_bytes={total_bytes}")
    print(f"distinct_photographers={len({r['group'] for r in rows})}")
    print(f"short_side min={short.min()} median={np.median(short):.1f} max={short.max()}")
    print(f"failures={len(failures)}")
    for fl in failures:
        print("  failure:", *fl)
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256_file(MANIFEST)}")
    if len(rows) < TARGET:
        print("WARNING: target not reached", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
