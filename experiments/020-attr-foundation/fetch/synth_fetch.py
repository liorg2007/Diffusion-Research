"""020 / Synthbuster new files: SD1.3, SD1.4, GLIDE, 300 each, from the Zenodo synthbuster.zip via HTTP Range.
Selection: entries sorted by name; numpy default_rng(0).permutation (fresh rng per folder); first 300.
Each entry: range fetch, CRC32 vs central directory, full PNG decode. Writes data/raw/synthbuster/<folder>/<file>.
Outputs (this dir): synthbuster_new_manifest.csv, synthbuster_prompts_new.csv, synthbuster_fetch_log.json.
"""
import sys, os, io, csv, json, hashlib, zlib, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
FD = "/home/zera/project/Diffusion-Research/experiments/020-attr-foundation/fetch"
sys.path.insert(0, FD)
import synth_common as sc
from synth_common import parse_cd, fetch_entry, RID_RE

ROOT = "/home/zera/project/Diffusion-Research"
URL = "https://zenodo.org/api/records/10066460/files/synthbuster.zip/content"
CD = f"{ROOT}/data/raw/synthbuster/_central_directory/synthbuster.cd"
FOLDERS = ["stable-diffusion-1-3", "stable-diffusion-1-4", "glide"]
N = 300
CAP = 600_000_000  # total cap, bytes (0.6 GB)
PNG_SIG = b"\x89PNG\r\n\x1a\n"

ents = parse_cd(open(CD, "rb").read())
pool = {f: [] for f in FOLDERS}
for e in ents:
    parts = e["name"].split("/")
    if len(parts) == 3 and parts[0] == "synthbuster" and parts[1] in pool and parts[2].endswith(".png"):
        pool[parts[1]].append(e)

chosen = []
for f in FOLDERS:
    es = sorted(pool[f], key=lambda e: e["name"])
    perm = np.random.default_rng(0).permutation(len(es))
    chosen += [(f, es[i]) for i in perm[:N]]
    print(f, "candidates", len(es), "chosen", N, flush=True)

plan_bytes = sum(e["csize"] + 30 + e["nl"] + 256 for _, e in chosen)
print("planned compressed bytes (images)", plan_bytes, flush=True)
assert plan_bytes < CAP * 0.7, "plan too large (leaves 0.18 GB for RAID)"


def job(f, e):
    base = e["name"].split("/")[-1]
    dest = f"{ROOT}/data/raw/synthbuster/{f}/{base}"
    rel = os.path.relpath(dest, ROOT)
    reused = False
    if os.path.isfile(dest) and os.path.getsize(dest) == e["usize"]:
        raw = open(dest, "rb").read()
        crc_ok = (zlib.crc32(raw) & 0xFFFFFFFF) == e["crc"]
        reused = crc_ok
    if not reused:
        last = None
        for attempt in range(3):
            try:
                raw, crc_ok = fetch_entry(URL, e)
                break
            except Exception as ex:
                last = ex
                time.sleep(2 * (attempt + 1))
        else:
            return {"path": rel, "error": repr(last)}
        if not crc_ok:
            return {"path": rel, "error": "CRC32 mismatch"}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, dest)
    png_sig = raw[:8] == PNG_SIG
    with Image.open(io.BytesIO(raw)) as im:
        fmt = im.format
        im.load()  # full decode
        w, h = im.size
    cap = RID_RE.search(base)
    return {"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "width": w, "height": h,
            "generator": f, "caption_id": cap.group(1) if cap else "", "url": URL, "zip_member": e["name"],
            "crc_ok": crc_ok, "png_sig_ok": png_sig, "pil_format": fmt, "reused_on_disk": reused, "error": ""}


rows, errs = [], []
with ThreadPoolExecutor(6) as ex:
    futs = [ex.submit(job, f, e) for f, e in chosen]
    for i, fu in enumerate(as_completed(futs)):
        r = fu.result()
        (errs if r.get("error") else rows).append(r)
        if (i + 1) % 100 == 0:
            print("done", i + 1, "transfer MB", round(sc.TRANSFER[0] / 1e6, 1), flush=True)
rows.sort(key=lambda r: r.get("path", ""))
print("ok", len(rows), "errors", len(errs), "transfer bytes", sc.TRANSFER[0], flush=True)
for r in errs:
    print("ERROR", r)
if sc.TRANSFER[0] > CAP:
    sys.exit("over cap")

man = os.path.join(FD, "synthbuster_new_manifest.csv")
cols = ["path", "bytes", "sha256", "width", "height", "generator", "caption_id", "url", "zip_member",
        "crc_ok", "png_sig_ok", "pil_format", "reused_on_disk"]
with open(man, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in cols})

# prompts: full prompts.csv range-fetched (small), keep rows whose caption id is among the new images
pe = [e for e in ents if e["name"] == "synthbuster/prompts.csv"][0]
praw, pcrc = fetch_entry(URL, pe)
ids = {r["caption_id"] for r in rows}
text = praw.decode("utf-8")
lines = text.splitlines(keepends=True)
header, body = lines[0], lines[1:]
import csv as _c, io as _io
parsed = list(_c.reader(_io.StringIO(text)))
keep = [ln for ln, rr in zip(body, parsed[1:]) if rr and rr[0] in ids]
found = {rr[0] for rr in parsed[1:] if rr and rr[0] in ids}
with open(os.path.join(FD, "synthbuster_prompts_new.csv"), "w", encoding="utf-8", newline="") as fh:
    fh.write(header)
    fh.writelines(keep)
missing_prompt = sorted(ids - found)
print("prompts.csv crc_ok", pcrc, "rows kept", len(keep), "caption ids without prompt row", len(missing_prompt), flush=True)
print("manifest sha256", hashlib.sha256(open(man, "rb").read()).hexdigest(), flush=True)
print("prompts_new sha256", hashlib.sha256(open(os.path.join(FD, "synthbuster_prompts_new.csv"), "rb").read()).hexdigest(), flush=True)
print("per generator", {f: sum(1 for r in rows if r["generator"] == f) for f in FOLDERS},
      "bytes", {f: sum(r["bytes"] for r in rows if r["generator"] == f) for f in FOLDERS}, flush=True)
print("pil formats", {f: sorted({r["pil_format"] for r in rows if r["generator"] == f}) for f in FOLDERS},
      "png_sig_all", all(r["png_sig_ok"] for r in rows), "crc_all", all(r["crc_ok"] for r in rows), flush=True)
print("caption ids unique", len(ids), "of", len(rows), flush=True)
json.dump({"errors": errs, "transfer_bytes": sc.TRANSFER[0], "n_ok": len(rows), "missing_prompt": missing_prompt},
          open(os.path.join(FD, "synthbuster_fetch_log.json"), "w"), indent=1)
