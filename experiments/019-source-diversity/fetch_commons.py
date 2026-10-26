#!/usr/bin/env python3
"""E10 source WMC: 600 Wikimedia Commons "Quality images" (Amendment 19).

Official MediaWiki API only (commons.wikimedia.org). Licence, MIME, upload date and
short side are checked per file. Thumbnails (1600 px) come from upload.wikimedia.org.
Run with: env/.venv/bin/python -I experiments/019-source-diversity/fetch_commons.py
"""
import csv
import hashlib
import html
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/commons"
META = RAW / "meta"
IMG = RAW / "images"
MANIFEST = REPO / "experiments/019-source-diversity/commons_manifest.csv"

API = "https://commons.wikimedia.org/w/api.php"
UA = "DiffusionResearch-academic/0.1 (non-commercial research script)"
CATEGORY = "Category:Quality_images"
TARGET = 600
DOWNLOAD_PAUSE = 1.0
PER_UPLOADER = 3
CAP_BYTES = 1_600_000_000  # raised from 1 GB for this source (lead decision)
CUTOFF = "2022-01-01T00:00:00Z"
MIN_SHORT_ORIGINAL = 1000
MIN_SHORT_DECODED = 512
THUMB_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}  # both serve the API thumburl

# CC BY, CC BY-SA (any version or port), CC0, Public domain. NC/ND/GFDL/unknown are rejected.
# The lookahead keeps "cc-by-nc" and "cc-by-nd" from matching.
LICENCE_OK = re.compile(r"^(cc[- ]by(-sa)?(?=$|[\s,]|-\d)|cc0\b|public domain$)", re.I)


def http_get(url, timeout=60, retries=3):
    """GET with a fixed User-Agent. 3 retries; HTTP 429 sleeps before retrying."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429:
                try:
                    wait = int(e.headers.get("Retry-After") or 60)
                except ValueError:
                    wait = 60
                time.sleep(wait)
                continue
            if e.code < 500 and e.code != 408:
                break  # client errors are not retried
        except Exception as e:  # network, timeout
            last = repr(e)
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(last)


def api(params):
    q = dict(params, format="json", formatversion="2")
    data = json.loads(http_get(API + "?" + urllib.parse.urlencode(q), timeout=60))
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data


PREFIXES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PER_SLICE = 1000        # pages (entries) per sortkey-prefix slice
MAX_CALLS_SLICE = 40    # safety bound per slice
API_PAUSE = 1.0         # seconds between API calls


def enumerate_candidates(existing):
    """Sortkey-prefix slices so the sample spreads across uploaders (lead decision).
    Merges into `existing` (pageid -> page), de-duplicated by pageid."""
    base = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": CATEGORY,
        "gcmtype": "file",
        "gcmlimit": 500,
        "gcmsort": "sortkey",
        "gcmdir": "ascending",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata|user|timestamp",
        "iiurlwidth": 1600,
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|DateTimeOriginal",
    }
    pages = dict(existing)
    total_calls = 0
    for prefix in PREFIXES:
        cont, seen, calls = {}, 0, 0
        while seen < PER_SLICE and calls < MAX_CALLS_SLICE:
            params = {**base, "gcmstartsortkeyprefix": prefix, **cont}
            data = api(params)
            calls += 1
            total_calls += 1
            time.sleep(API_PAUSE)
            for p in data.get("query", {}).get("pages", []):
                if "pageid" not in p:
                    continue
                seen += 1
                if "imageinfo" not in p or p["pageid"] in pages:
                    continue
                pages[p["pageid"]] = p
            cont = data.get("continue")
            if not cont:
                break
        print(f"slice {prefix}: {seen} entries, {calls} calls, {len(pages)} pages total")
    passing = sum(1 for p in pages.values() if passes_filter(p))
    print(f"enumeration: {total_calls} API calls this run; {len(pages)} pages; {passing} pass filter")
    return pages, total_calls


def licence_of(ii):
    ext = ii.get("extmetadata", {})
    return ext.get("LicenseShortName", {}).get("value", "") or ""


def passes_filter(p):
    if "imageinfo" not in p:
        return False
    ii = p["imageinfo"][0]
    if ii.get("mime") != "image/jpeg":
        return False
    if not ii.get("timestamp", "") < CUTOFF:
        return False
    if min(ii.get("width", 0), ii.get("height", 0)) < MIN_SHORT_ORIGINAL:
        return False
    if not LICENCE_OK.match(licence_of(ii).strip()):
        return False
    if not ii.get("thumburl") or urllib.parse.urlparse(ii["thumburl"]).hostname not in THUMB_HOSTS:
        return False
    return True


def strip_html(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(html.unescape(s).split())


def decode_check(data):
    """Full PIL decode. Returns (width, height) or raises."""
    im = Image.open(io.BytesIO(data))
    if im.format != "JPEG":
        raise ValueError(f"format {im.format}")
    im.verify()
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.size


def fetch_one(p, total_bytes):
    pid = p["pageid"]
    ii = p["imageinfo"][0]
    out = IMG / f"{pid}.jpg"
    if out.exists():
        data = out.read_bytes()  # resume: reuse an existing file, still re-verified below
    else:
        data = http_get(ii["thumburl"], timeout=60, retries=3)
    if total_bytes + len(data) > CAP_BYTES:
        raise RuntimeError("size cap reached")
    try:
        w, h = decode_check(data)
    except Exception as e:
        raise RuntimeError(f"decode: {e!r}")
    if min(w, h) < MIN_SHORT_DECODED:
        raise RuntimeError(f"short side {min(w, h)} < {MIN_SHORT_DECODED}")
    if not out.exists():
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(out)
    return {
        "image_id": pid,
        "title": p["title"],
        "path": str(out.relative_to(REPO)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": w,
        "height": h,
        "license": licence_of(ii).strip(),
        "license_url": ii.get("extmetadata", {}).get("LicenseUrl", {}).get("value", ""),
        "author": strip_html(ii.get("extmetadata", {}).get("Artist", {}).get("value", "")),
        "group": ii.get("user", ""),
        "uploaded_at": ii.get("timestamp", ""),
        "url": ii["thumburl"],
        "descriptionurl": ii.get("descriptionurl", ""),
    }


def main():
    META.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    # 1. Enumerate and save the candidate list.
    cand_path = META / "candidates.json"
    existing = {}
    if cand_path.exists():
        existing = {p["pageid"]: p for p in json.loads(cand_path.read_text())["pages"]}
    merged, calls = enumerate_candidates(existing)
    pages = list(merged.values())
    cand_path.write_text(json.dumps(
        {"category": CATEGORY, "gcmsort": "sortkey", "n": len(pages), "pages": pages},
        ensure_ascii=False))

    # 2. Filter.
    filtered = [p for p in pages if passes_filter(p)]
    filtered.sort(key=lambda p: p["title"])
    print(f"filter: {len(filtered)} pass (jpeg, <2022, short side>=1000, licence, thumb host)")

    # 3. Seeded permutation, walk, cap 3 per uploader, download until 600 succeed.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(filtered))
    per_user = Counter()
    rows, failures, skipped_uploader, total_bytes = [], [], 0, 0
    for k in order:
        if len(rows) >= TARGET:
            break
        p = filtered[k]
        user = p["imageinfo"][0].get("user", "")
        if per_user[user] >= PER_UPLOADER:
            skipped_uploader += 1
            continue
        try:
            row = fetch_one(p, total_bytes)
        except Exception as e:
            failures.append({"pageid": p["pageid"], "title": p["title"], "error": str(e)})
            print(f"FAIL {p['pageid']}: {e}")
            if "size cap" in str(e):
                break
            time.sleep(DOWNLOAD_PAUSE)
            continue
        per_user[user] += 1
        total_bytes += row["bytes"]
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} ok, {total_bytes / 1e6:.0f} MB")
        time.sleep(DOWNLOAD_PAUSE)

    # Remove stale images (not in the final selection), including partial downloads.
    keep = {f"{r['image_id']}.jpg" for r in rows}
    for fp in IMG.iterdir():
        if fp.name not in keep:
            fp.unlink()
            print(f"removed stale file {fp.name}")

    # 4. Manifest.
    cols = ["image_id", "title", "path", "bytes", "sha256", "width", "height", "license",
            "license_url", "author", "group", "uploaded_at", "url", "descriptionurl"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    (META / "download_log.json").write_text(json.dumps(
        {"selected": len(rows), "failures": failures, "skipped_uploader_cap": skipped_uploader,
         "manifest_sha256": manifest_sha}, ensure_ascii=False, indent=1))

    shorts = sorted(min(r["width"], r["height"]) for r in rows)
    print("---- SUMMARY ----")
    print(f"selected: {len(rows)} of target {TARGET}; total bytes: {total_bytes} "
          f"({total_bytes / 1e6:.1f} MB)")
    print(f"licence breakdown: {dict(Counter(r['license'] for r in rows))}")
    print(f"distinct uploaders: {len(set(r['group'] for r in rows))}")
    if shorts:
        print(f"short side min {shorts[0]}, median {shorts[len(shorts) // 2]}")
    print(f"failures: {len(failures)}; skipped by uploader cap: {skipped_uploader}")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")
    print(f"manifest sha256: {manifest_sha}")


if __name__ == "__main__":
    main()
