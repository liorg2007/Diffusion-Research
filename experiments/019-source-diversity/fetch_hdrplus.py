"""Fetch 600 final.jpg images of the Google HDR+ burst dataset (E10, Amendment 19, item HDRP).

Source: official public bucket gs://hdrplusdata (https://hdrplusdata.org/dataset.html), licence CC BY-SA.
Run with:  env/.venv/bin/python -I experiments/019-source-diversity/fetch_hdrplus.py
Outputs (git-ignored data dir):  data/raw/hdrplus/meta/listing.json, data/raw/hdrplus/images/<burst_id>.jpg
Outputs (tracked):               experiments/019-source-diversity/hdrplus_manifest.csv
"""

import base64
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFIX = "20171106/results_20171023/"
LIST_URL = (
    "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o?"
    + urllib.parse.urlencode({"prefix": PREFIX, "fields": "items(name,size,md5Hash,crc32c),nextPageToken"})
)
OBJ_URL = "https://storage.googleapis.com/hdrplusdata/{}"
RAW_DIR = os.path.join(REPO, "data", "raw", "hdrplus")
META_DIR = os.path.join(RAW_DIR, "meta")
IMG_DIR = os.path.join(RAW_DIR, "images")
MANIFEST = os.path.join(REPO, "experiments", "019-source-diversity", "hdrplus_manifest.csv")
TARGET = 600
MIN_SHORT = 512
TIMEOUT = 120
RETRIES = 3
BUDGET_ABORT = 4.5e9
LICENSE = "CC BY-SA (HDR+ burst dataset)"


def fetch(url):
    """GET with timeout and 3 attempts (1 + 2 retries after a short backoff)."""
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "diffusion-research-e10/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:  # network/HTTP error: retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}: {last}")


def list_objects():
    items, token, pages = [], None, 0
    while True:
        url = LIST_URL + (("&pageToken=" + urllib.parse.quote(token)) if token else "")
        page = json.loads(fetch(url))
        items.extend(page.get("items", []))
        pages += 1
        token = page.get("nextPageToken")
        if not token:
            break
    return items, pages


def md5_b64(data):
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def check_image(data):
    """Full decode with PIL. Returns (width, height, make_model) or raises."""
    im = Image.open(io.BytesIO(data))
    im.load()  # full decode, raises on truncation
    w, h = im.size
    make = model = ""
    try:
        ex = im.getexif()
        make = str(ex.get(271, "")).strip().strip("\x00")
        model = str(ex.get(272, "")).strip().strip("\x00")
    except Exception:
        pass
    cam = " ".join(x for x in (make, model) if x)
    return w, h, cam


def main():
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    # 1. Listing
    items, pages = list_objects()
    with open(os.path.join(META_DIR, "listing.json"), "w") as f:
        json.dump({"prefix": PREFIX, "pages": pages, "items": items}, f, indent=1)
    finals = {}
    for it in items:
        name = it["name"]
        if not name.endswith("/final.jpg"):
            continue
        burst = name.split("/")[-2]
        finals[burst] = it
    print(f"listing: {len(items)} objects, {pages} pages, {len(finals)} final.jpg bursts")

    # 2. Order: sort ids, permute with numpy seed 0
    ids = sorted(finals)
    np.random.seed(0)
    order = [ids[i] for i in np.random.permutation(len(ids))]

    # 3. Download until TARGET succeed
    accepted, failures, total_bytes = [], [], 0
    aborted = False
    for burst in order:
        if len(accepted) >= TARGET:
            break
        it = finals[burst]
        size = int(it["size"])
        if total_bytes + size > BUDGET_ABORT:
            aborted = True
            print(f"ABORT: running total would exceed 4.5 GB at {burst}")
            break
        dest = os.path.join(IMG_DIR, f"{burst}.jpg")
        url = OBJ_URL.format(urllib.parse.quote(it["name"]))
        data = None
        if os.path.exists(dest):
            with open(dest, "rb") as f:
                cached = f.read()
            if md5_b64(cached) == it["md5Hash"]:
                data = cached
        err = None
        if data is None:
            for attempt in range(RETRIES):
                try:
                    blob = fetch(url)
                except Exception as e:
                    err = f"download: {e}"
                    continue
                if md5_b64(blob) != it["md5Hash"]:
                    err = "md5 mismatch"
                    continue
                data, err = blob, None
                break
        if data is None:
            failures.append((burst, err))
            print(f"FAIL {burst}: {err}")
            continue
        try:
            w, h, cam = check_image(data)
        except Exception as e:
            failures.append((burst, f"decode: {e}"))
            print(f"FAIL {burst}: decode: {e}")
            continue
        if min(w, h) < MIN_SHORT:
            failures.append((burst, f"short side {min(w, h)} < {MIN_SHORT}"))
            print(f"FAIL {burst}: short side {min(w, h)}")
            continue
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        total_bytes += len(data)
        accepted.append(
            dict(
                image_id=burst,
                path=os.path.relpath(dest, REPO),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                md5_ok=True,
                width=w,
                height=h,
                license=LICENSE,
                group=burst[:4],
                camera_make_model=cam,
                url=url,
            )
        )
        if len(accepted) % 50 == 0:
            print(f"accepted {len(accepted)} (total {total_bytes / 1e9:.3f} GB)")

    with open(os.path.join(META_DIR, "failures.json"), "w") as f:
        json.dump(failures, f, indent=1)

    # 4. Manifest
    fields = ["image_id", "path", "bytes", "sha256", "md5_ok", "width", "height",
              "license", "group", "camera_make_model", "url"]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in accepted:
            w.writerow(row)
    with open(MANIFEST, "rb") as f:
        man_sha = hashlib.sha256(f.read()).hexdigest()

    shorts = sorted(min(r["width"], r["height"]) for r in accepted)
    groups = {r["group"] for r in accepted}
    cams = {}
    for r in accepted:
        cams[r["camera_make_model"] or "(none)"] = cams.get(r["camera_make_model"] or "(none)", 0) + 1

    print("---")
    print(f"accepted={len(accepted)} failures={len(failures)} aborted={aborted}")
    print(f"total_bytes={total_bytes} ({total_bytes / 1e9:.3f} GB)")
    print(f"md5_ok={sum(r['md5_ok'] for r in accepted)}/{len(accepted)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={man_sha}")
    print(f"distinct_groups={len(groups)}")
    print("camera_models:", json.dumps(dict(sorted(cams.items(), key=lambda kv: -kv[1]))))
    if shorts:
        print(f"short_side min={shorts[0]} median={shorts[len(shorts) // 2]}")
    print("failures:", failures)
    return 0 if len(accepted) >= TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
