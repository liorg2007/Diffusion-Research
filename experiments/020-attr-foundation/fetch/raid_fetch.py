"""020 / RAID originals: 400 new ids x 5 classes (real, gen_0..gen_3) under original/, pinned HF commit.
Pool = ids present in all 5 original/ folders, minus ids already in experiments/010-raid-transfer/raid_manifest.csv.
Selection: sorted pool; numpy default_rng(0).permutation; first 400. Each file: sha256 vs HF tree lfs.oid (and size).
Outputs (this dir): raid_new_manifest.csv, raid_fetch_log.json.
"""
import sys, os, io, csv, json, hashlib, time, urllib.request, urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

ROOT = "/home/zera/project/Diffusion-Research"
FD = f"{ROOT}/experiments/020-attr-foundation/fetch"
REV = "36335df31a738119b517cf77388829350fd18e9f"
REPO = "aimagelab/RAID"
CLS = ["real", "gen_0", "gen_1", "gen_2", "gen_3"]
N = 400
CAP = 600_000_000
UA = {"User-Agent": "diffusion-detect-research/1.0 (academic RAID fetch, 020-attr-foundation)"}
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"


def get(url, retries=4):
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404):
                break
            time.sleep(3 * (a + 1))
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise last


def list_folder(cls):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/original/{cls}?limit=1000"
    out = []
    while url:
        body, hdr = get(url)
        out += json.loads(body)
        nxt = None
        link = hdr.get("Link") or ""
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part[part.index("<") + 1:part.index(">")]
        url = nxt
    return out


# 1. listings -> id maps
maps, lfs = {}, {}
for c in CLS:
    ents = [e for e in list_folder(c) if e.get("type") == "file"]
    m = {}
    for e in ents:
        base = e["path"].split("/")[-1]
        suffix = f"_{c}.jpg"
        assert base.endswith(suffix), base
        m[base[: -len(suffix)]] = e
    maps[c] = m
    print(c, "files", len(ents), "ids", len(m), "lfs entries", sum(1 for e in ents if "lfs" in e), flush=True)

common = set.intersection(*(set(maps[c]) for c in CLS))
prev = set()
with open(f"{ROOT}/experiments/010-raid-transfer/raid_manifest.csv") as fh:
    for r in csv.DictReader(fh):
        prev.add(r["id"])
pool = sorted(common - prev)
print("ids in all 5 classes", len(common), "already used (010)", len(prev & common), "pool", len(pool), flush=True)
perm = np.random.default_rng(0).permutation(len(pool))
chosen = sorted(pool[i] for i in perm[:N])
assert len(set(chosen) & prev) == 0 and len(chosen) == N

plan = [(c, i) for i in chosen for c in CLS]
plan_bytes = sum(maps[c][i]["size"] for c, i in plan)
print("planned bytes (originals)", plan_bytes, flush=True)
assert plan_bytes < CAP, "plan over cap"


def job(c, i):
    e = maps[c][i]
    path = e["path"]
    base = path.split("/")[-1]
    dest = f"{ROOT}/data/raw/raid/original/{c}/{base}"
    oid = e["lfs"]["oid"]
    url = RESOLVE + path
    rel = os.path.relpath(dest, ROOT)
    reused = False
    if os.path.isfile(dest) and os.path.getsize(dest) == e["size"]:
        data = open(dest, "rb").read()
        reused = hashlib.sha256(data).hexdigest() == oid
    if not reused:
        data = None
        for a in range(3):
            try:
                data, _ = get(url)
                break
            except Exception as ex:
                last = ex
                time.sleep(2 * (a + 1))
        if data is None:
            return {"path": rel, "error": repr(last)}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dest + ".part", dest)
    sha = hashlib.sha256(data).hexdigest()
    oid_ok = sha == oid and len(data) == e["size"]
    with Image.open(io.BytesIO(data)) as im:
        fmt = im.format
        im.load()
        w, h = im.size
    return {"path": rel, "bytes": len(data), "sha256": sha, "width": w, "height": h, "class": c, "id": i,
            "caption_id": "", "url": url, "hf_lfs_oid": oid, "oid_ok": oid_ok, "pil_format": fmt,
            "reused_on_disk": reused, "error": "" if oid_ok else "oid/size mismatch"}


rows, errs = [], []
with ThreadPoolExecutor(8) as ex:
    futs = [ex.submit(job, c, i) for c, i in plan]
    for k, fu in enumerate(as_completed(futs)):
        r = fu.result()
        (errs if r.get("error") else rows).append(r)
        if (k + 1) % 500 == 0:
            print("done", k + 1, flush=True)
rows.sort(key=lambda r: r.get("path", ""))
print("ok", len(rows), "errors", len(errs), flush=True)
for r in errs:
    print("ERROR", r, flush=True)
cols = ["path", "bytes", "sha256", "width", "height", "class", "id", "caption_id", "url", "hf_lfs_oid", "oid_ok",
        "pil_format", "reused_on_disk"]
man = os.path.join(FD, "raid_new_manifest.csv")
with open(man, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in cols})
print("manifest sha256", hashlib.sha256(open(man, "rb").read()).hexdigest(), flush=True)
print("per class", {c: sum(1 for r in rows if r["class"] == c) for c in CLS},
      "bytes", sum(r["bytes"] for r in rows), "formats", {r["pil_format"] for r in rows}, flush=True)
json.dump({"errors": errs, "n_ok": len(rows), "chosen_ids": chosen, "pool_size": len(pool)},
          open(os.path.join(FD, "raid_fetch_log.json"), "w"), indent=1)
