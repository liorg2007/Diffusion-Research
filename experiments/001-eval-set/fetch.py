#!/usr/bin/env python
"""Evaluation-set fetcher (D-005).  Usage: python -I fetch.py plan|run|finalize

from-manifest : re-create EXACTLY the files listed in manifest_raw.csv (sha256-verified), using the committed
           state/plan.json + state/latent_diffusion.json; does not touch state/ or the budget (fresh-machine rebuild).
plan     : select samples (seed 0), read zip central directories via HTTP Range, HEAD VISION files,
           write state/plan.json (exact planned download bytes).
run      : download everything in plan.json (resumable), per-entry CRC32 verified, budget enforced.
finalize : write data/manifest_raw.csv, results.json.

Downloaded files are untrusted data: never executed, only parsed with zipfile/zlib/PIL.
Only official URLs listed in config.json are contacted.
"""
import csv, hashlib, json, os, random, re, shutil, struct, sys, threading, time, zlib
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "001-eval-set"
STATE = EXP / "state"
DATA = ROOT / "data"
RAW = DATA / "raw"
CFG = json.load(open(EXP / "config.json"))
HARD = CFG["budget_hard_stop_bytes"]
RETRIES = CFG["max_retries"]
UA = {"User-Agent": "diffusion-detect-research/1.0 (academic evaluation-set fetch)"}

STATE.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()


class BudgetStop(Exception):
    pass


class AbortRun(Exception):
    pass


# ------------------------------------------------------------------ budget
class Budget:
    def __init__(self):
        self.path = STATE / "budget.json"
        self.d = json.load(open(self.path)) if self.path.exists() else {"total": 0, "by_source": {}}
        self.reserved = 0

    def reserve(self, n):
        with LOCK:
            if self.d["total"] + self.reserved + n > HARD:
                raise BudgetStop(f"budget: total {self.d['total']} + inflight {self.reserved} + {n} > {HARD}")
            self.reserved += n

    def release(self, n):
        with LOCK:
            self.reserved -= n

    def add(self, src, n):
        with LOCK:
            self.d["total"] += n
            self.d["by_source"][src] = self.d["by_source"].get(src, 0) + n
            if self.path is None:  # from-manifest mode: in-memory accounting only
                return
            tmp = str(self.path) + ".tmp"
            json.dump(self.d, open(tmp, "w"))
            os.replace(tmp, self.path)


BUD = Budget()


# ------------------------------------------------------------------ errors
ERRS = {}


def note_failure(step, e):
    key = (step, type(e).__name__ + ":" + str(e)[:60])
    with LOCK:
        ERRS[key] = ERRS.get(key, 0) + 1
        if ERRS[key] >= 2:
            raise AbortRun(f"same error twice at step {step}: {key[1]}")


# ------------------------------------------------------------------ http
def http(url, src, start=None, end=None, suffix=None, method="GET", timeout=60):
    """Returns (bytes, total_len_or_None, content_length). Retries transient errors."""
    h = dict(UA)
    if suffix is not None:
        h["Range"] = f"bytes=-{suffix}"
    elif start is not None:
        h["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=h, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                cl = r.headers.get("Content-Length")
                cr = r.headers.get("Content-Range")
                if method == "HEAD":
                    return b"", None, int(cl) if cl else None
                if "Range" in h and r.status != 206:
                    raise RuntimeError(f"expected 206 got {r.status}")
                data = r.read()
                BUD.add(src, len(data))
                if cl and len(data) != int(cl):
                    raise RuntimeError(f"short read {len(data)}/{cl}")
                total = int(cr.split("/")[-1]) if cr else None
                if start is not None and cr:
                    got_start = int(cr.split()[1].split("-")[0])
                    if got_start != start:
                        raise RuntimeError(f"range start mismatch {got_start}!={start}")
                return data, total, len(data)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 416):
                break
            time.sleep(4 * (attempt + 1))  # 429/5xx: back off harder
        except Exception as e:  # transient
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


# ------------------------------------------------------------------ zip over range
def safe_name(name):
    n = name.replace("\\", "/")
    if n.startswith("/") or re.match(r"^[A-Za-z]:", n):
        return None
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or n.endswith("/"):
        return None
    if any("\x00" in p for p in parts):
        return None
    return parts


def read_central_directory(url, src, cache):
    cache = Path(cache)
    if cache.exists():
        buf = cache.read_bytes()
        total = int(open(str(cache) + ".total").read())
    else:
        tail, total, _ = http(url, src, suffix=262144)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("EOCD not found")
        sig, d1, d2, nd, nt, cdsize, cdoff, clen = struct.unpack("<4sHHHHIIH", tail[i:i + 22])
        if cdoff == 0xFFFFFFFF or nt == 0xFFFF or cdsize == 0xFFFFFFFF:
            loc = tail[i - 20:i]
            lsig, ld, z64off, ldisks = struct.unpack("<4sIQI", loc)
            if lsig != b"PK\x06\x07":
                raise RuntimeError("zip64 locator missing")
            rel = z64off - (total - len(tail))
            if rel >= 0:
                z = tail[rel:rel + 56]
            else:
                z, _, _ = http(url, src, start=z64off, end=z64off + 55)
            zs, zsize, vm, vn, dn, dcd, ndisk, nt, cdsize, cdoff = struct.unpack("<4sQHHIIQQQQ", z[:56])
            if zs != b"PK\x06\x06":
                raise RuntimeError("zip64 EOCD bad")
        buf, _, _ = http(url, src, start=cdoff, end=cdoff + cdsize - 1)
        if len(buf) != cdsize:
            raise RuntimeError("central directory short")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(buf)
        open(str(cache) + ".total", "w").write(str(total))
    return parse_cd(buf), total


def parse_cd(buf):
    ents = []
    p = 0
    while p + 46 <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        (sig, vm, vn, flags, method, mt, md, crc, csize, usize, nl, el, cl, dn, ia, ea, off) = \
            struct.unpack("<4sHHHHHHLLLHHHHHLL", buf[p:p + 46])
        name_raw = buf[p + 46:p + 46 + nl]
        extra = buf[p + 46 + nl:p + 46 + nl + el]
        name = name_raw.decode("utf-8" if flags & 0x800 else "cp437", errors="replace")
        q = 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                z = extra[q + 4:q + 4 + hl]
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
            q += 4 + hl
        ents.append({"name": name, "method": method, "crc": crc, "csize": csize, "usize": usize,
                     "off": off, "nl": nl, "flags": flags})
        p += 46 + nl + el + cl
    return ents


def fetch_entry(url, src, e):
    """Range-fetch local header + data of one zip entry; decompress; verify size + CRC32 vs central dir."""
    pad = 30 + e["nl"] + 256
    want = pad + e["csize"]
    buf, _, _ = http(url, src, start=e["off"], end=e["off"] + want - 1)
    sig, vn, fl, meth, mt, md, crc, cs, us, nl, el = struct.unpack("<4sHHHHHLLLHH", buf[:30])
    if sig != b"PK\x03\x04":
        raise RuntimeError("bad local header")
    ds = 30 + nl + el
    if len(buf) < ds + e["csize"]:
        more, _, _ = http(url, src, start=e["off"] + len(buf), end=e["off"] + ds + e["csize"] - 1)
        buf += more
    comp = buf[ds:ds + e["csize"]]
    if e["method"] == 0:
        raw = comp
    elif e["method"] == 8:
        d = zlib.decompressobj(-15)
        raw = d.decompress(comp) + d.flush()
    else:
        raise RuntimeError(f"unsupported method {e['method']}")
    if len(raw) != e["usize"]:
        raise RuntimeError("size mismatch")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != e["crc"]:
        with LOCK:
            CRC_FAIL.append(e["name"])
        raise RuntimeError("CRC32 mismatch")
    return raw


CRC_FAIL = []


def fetch_entry_retry(url, src, e):
    last = None
    for attempt in range(RETRIES):
        try:
            return fetch_entry(url, src, e)
        except (BudgetStop, AbortRun):
            raise
        except Exception as ex:
            last = ex
            time.sleep(1.5 * (attempt + 1))
    note_failure("zip-entry", last)
    raise last


# ------------------------------------------------------------------ image meta
def img_meta(path):
    from PIL import Image
    with Image.open(path) as im:
        fmt, (w, h) = im.format, im.size
        im.load()  # full decode = integrity check
    return fmt, w, h


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------ done-index (resumability)
class Done:
    def __init__(self):
        self.path = STATE / "done.jsonl"
        self.rows = {}
        if self.path.exists():
            for l in open(self.path):
                r = json.loads(l)
                self.rows[r["path"]] = r

    def ok(self, rel):
        r = self.rows.get(rel)
        p = ROOT / rel
        return r is not None and p.exists() and p.stat().st_size == r["bytes"]

    def add(self, r):
        with LOCK:
            self.rows[r["path"]] = r
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ PLAN
def sample(items, n, rng):
    items = sorted(items)
    return sorted(rng.sample(items, n))


def plan():
    plan = {"vision": [], "zips": {}}
    # ---- VISION
    lines = [l.strip() for l in open(RAW / "vision" / "VISION_files.txt") if "/images/" in l]
    by = {}
    for u in lines:
        m = re.match(r".*/dataset/([^/]+)/images/([^/]+)/(.+)\.jpg$", u)
        dev, kind, stem = m.groups()
        by.setdefault(dev, {}).setdefault(kind, {})[stem] = u
    devs = sorted(by)
    assert len(devs) == 35
    rng = random.Random(CFG["seed"])
    n = CFG["vision"]["n_native"]
    base, extra = divmod(n, len(devs))
    plus = set(rng.sample(devs, extra))
    sel = []
    for dev in devs:
        q = base + (1 if dev in plus else 0)
        for stem in sample(by[dev]["nat"], q, rng):
            idx = stem.split("_")[-1]
            sel.append((dev, idx))
    # mapping check: every selected index must exist in all three kinds with the same 4-digit index
    for dev, idx in sel:
        for kind in CFG["vision"]["kinds"]:
            assert f"{dev.split('_')[0]}_I_{kind}_{idx}" in by[dev][kind], (dev, idx, kind)
    for dev, idx in sel:
        d = dev.split("_")[0]
        for kind, plat in CFG["vision"]["kinds"].items():
            stem = f"{d}_I_{kind}_{idx}"
            url = by[dev][kind][stem]
            plan["vision"].append({"dev": dev, "id": d, "kind": kind, "platform": plat, "stem": stem, "url": url,
                                   "rel": f"data/raw/vision/{dev}/{kind}/{stem}.jpg"})
    print("VISION selected natives:", len(sel), "files:", len(plan["vision"]))

    def head(it):
        _, _, cl = http(it["url"], "vision", method="HEAD")
        it["size"] = cl
        return it
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(head, plan["vision"]))
    vb = {}
    for it in plan["vision"]:
        vb[it["kind"]] = vb.get(it["kind"], 0) + it["size"]
    print("VISION planned bytes by kind:", vb, "sum", sum(vb.values()))

    # ---- zips
    def zinfo(key, url, src, folder_filter=None):
        ents, total = read_central_directory(url, src, RAW / src / "_central_directory" / f"{key}.cd")
        ents = [e for e in ents if safe_name(e["name"]) and safe_name(e["name"])[-1].lower().endswith((".png", ".jpg", ".jpeg"))]
        return ents, total

    zs = {}
    ents, total = zinfo("real_RAISE_1k", CFG["raise_bfree"]["url"], "raise_bfree")
    zs["raise"] = (ents, total)
    ents, total = zinfo("synthbuster", CFG["synthbuster"]["url"], "synthbuster")
    zs["synthbuster"] = (ents, total)
    ents, total = zinfo("sd3_flux", CFG["bfree_gen"]["sd3_flux_url"], "bfree_gen")
    zs["sd3_flux"] = (ents, total)

    def folders(ents):
        c = {}
        for e in ents:
            p = safe_name(e["name"])
            c.setdefault(p[-2] if len(p) > 1 else "", []).append(e)
        return c
    out = {}
    for k, (ents, total) in zs.items():
        f = folders(ents)
        print(k, "zip bytes", total, {a: len(b) for a, b in f.items()})
        print("  sample names:", [e["name"] for e in ents[:2]])
        out[k] = f
    rng = random.Random(CFG["seed"])
    # RAISE
    ents = sorted(sum(out["raise"].values(), []), key=lambda e: e["name"])
    pick = rng.sample(ents, CFG["raise_bfree"]["n"])
    plan["zips"]["raise"] = {"url": CFG["raise_bfree"]["url"], "src": "raise_bfree", "entries": [dict(e, gen="real", folder="RAISE") for e in sorted(pick, key=lambda e: e["name"])]}
    # Synthbuster
    rng = random.Random(CFG["seed"])
    pick = []
    for g in CFG["synthbuster"]["generators"]:
        assert g in out["synthbuster"], (g, list(out["synthbuster"]))
        es = sorted(out["synthbuster"][g], key=lambda e: e["name"])
        pick += [dict(e, gen=g, folder=g) for e in sorted(rng.sample(es, CFG["synthbuster"]["n_per_generator"]), key=lambda e: e["name"])]
    plan["zips"]["synthbuster"] = {"url": CFG["synthbuster"]["url"], "src": "synthbuster", "entries": pick}
    # sd3_flux
    rng = random.Random(CFG["seed"])
    fl = out["sd3_flux"]
    pick = []
    for g, n in (("sd3", CFG["bfree_gen"]["n_sd3"]), ("flux", CFG["bfree_gen"]["n_flux"])):
        cand = [k for k in fl if g in k.lower()]
        assert len(cand) == 1, (g, list(fl))
        es = sorted(fl[cand[0]], key=lambda e: e["name"])
        pick += [dict(e, gen=cand[0], folder=cand[0]) for e in sorted(rng.sample(es, n), key=lambda e: e["name"])]
    plan["zips"]["sd3_flux"] = {"url": CFG["bfree_gen"]["sd3_flux_url"], "src": "bfree_gen", "entries": pick}

    tot = sum(it["size"] for it in plan["vision"])
    for k, z in plan["zips"].items():
        b = sum(e["csize"] + 30 + e["nl"] + 256 for e in z["entries"])
        print(k, len(z["entries"]), "planned bytes", b)
        tot += b
    tot += 127295883 + 154
    print("PLANNED TOTAL incl latent-diffusion:", tot, "already used", BUD.d["total"], "hard", HARD)
    plan["planned_total"] = tot
    json.dump(plan, open(STATE / "plan.json", "w"))


# ------------------------------------------------------------------ RUN
DONE = Done()


def row_base(rel, **kw):
    r = {"path": rel}
    r.update(kw)
    return r


def do_vision(it):
    rel = it["rel"]
    if DONE.ok(rel):
        return
    BUD.reserve(it["size"])
    try:
        last = None
        for attempt in range(RETRIES):
            try:
                data, _, cl = http(it["url"], "vision")
                if len(data) != it["size"]:
                    raise RuntimeError(f"size {len(data)} != HEAD {it['size']}")
                p = ROOT / rel
                write_atomic(p, data)
                fmt, w, h = img_meta(p)
                DONE.add(row_base(rel, label=0, source_dataset="VISION", generator="real",
                                  camera_or_platform=f"{it['id']}_{it['platform']}", license=CFG["vision"]["license"],
                                  split="", sha256=hashlib.sha256(data).hexdigest(), orig_format=fmt, width=w,
                                  height=h, bytes=len(data)))
                return
            except (BudgetStop, AbortRun):
                raise
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        note_failure("vision-file", last)
        FAILED.append((rel, repr(last)))
    finally:
        BUD.release(it["size"])


FAILED = []


def do_zip_entry(z, e, meta):
    sp = safe_name(e["name"])
    rel = f"data/raw/{z['src']}/{e['folder']}/{sp[-1]}"
    if DONE.ok(rel):
        return
    need = e["csize"] + 30 + e["nl"] + 256
    BUD.reserve(need)
    try:
        raw = fetch_entry_retry(z["url"], z["src"], e)
        p = ROOT / rel
        write_atomic(p, raw)
        fmt, w, h = img_meta(p)
        DONE.add(row_base(rel, label=meta["label"], source_dataset=meta["source_dataset"], generator=e["gen"],
                          camera_or_platform=meta["cam"], license=meta["license"], split=meta.get("split", ""),
                          sha256=hashlib.sha256(raw).hexdigest(), orig_format=fmt, width=w, height=h, bytes=len(raw),
                          zip_entry=e["name"], crc32=f"{e['crc']:08x}"))
    except (BudgetStop, AbortRun):
        raise
    except Exception as ex:
        FAILED.append((rel, repr(ex)))
    finally:
        BUD.release(need)


def do_latent_diffusion(rng_seed):
    src = "bfree_gen"
    url = CFG["bfree_gen"]["latent_diffusion_url"]
    d = RAW / src / "_archive"
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / "latent-diffusion.zip"
    info = STATE / "latent_diffusion.json"
    done_all = info.exists() and all(DONE.ok(r) for r in json.load(open(info))["rels"])
    if done_all:
        return
    # checksum.txt (published md5)
    ck, _, _ = http(CFG["bfree_gen"]["checksum_url"], src)
    md5s = {l.split()[1]: l.split()[0] for l in ck.decode().splitlines() if l.strip()}
    want_md5 = md5s["latent-diffusion.zip"]
    size = 127295883
    if not zpath.exists() or zpath.stat().st_size != size:
        BUD.reserve(size)
        try:
            part = d / "latent-diffusion.zip.part"
            have = part.stat().st_size if part.exists() else 0
            for attempt in range(RETRIES):
                try:
                    have = part.stat().st_size if part.exists() else 0
                    h = dict(UA)
                    if have:
                        h["Range"] = f"bytes={have}-"
                    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                        with open(part, "ab") as f:
                            while True:
                                b = r.read(1 << 20)
                                if not b:
                                    break
                                f.write(b)
                                BUD.add(src, len(b))
                    if part.stat().st_size != size:
                        raise RuntimeError(f"size {part.stat().st_size} != {size}")
                    os.replace(part, zpath)
                    break
                except Exception as e:
                    last = e
                    time.sleep(2 * (attempt + 1))
            else:
                note_failure("latent-diffusion", last)
                raise last
        finally:
            BUD.release(size)
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    sha = sha256_file(zpath)
    md5_ok = md5 == want_md5
    import zipfile
    zf = zipfile.ZipFile(zpath)
    infos = [i for i in zf.infolist() if safe_name(i.filename) and safe_name(i.filename)[-1].lower().endswith((".png", ".jpg"))]
    rng = random.Random(rng_seed)
    pick = sorted(rng.sample(sorted(infos, key=lambda i: i.filename), CFG["bfree_gen"]["n_latent_diffusion"]), key=lambda i: i.filename)
    rels = []
    nbad = 0
    for i in infos:  # verify CRC of every entry vs central directory (whole zip is local anyway)
        raw = zf.read(i)
        if (zlib.crc32(raw) & 0xFFFFFFFF) != i.CRC:
            nbad += 1
            CRC_FAIL.append(i.filename)
    for i in pick:
        raw = zf.read(i)
        sp = safe_name(i.filename)
        gen = "latent-diffusion"
        rel = f"data/raw/{src}/{gen}/{sp[-1]}"
        p = ROOT / rel
        write_atomic(p, raw)
        fmt, w, h = img_meta(p)
        DONE.add(row_base(rel, label=1, source_dataset="B-Free extended Synthbuster", generator=gen,
                          camera_or_platform="", license=CFG["bfree_gen"]["license"], split="",
                          sha256=hashlib.sha256(raw).hexdigest(), orig_format=fmt, width=w, height=h, bytes=len(raw),
                          zip_entry=i.filename, crc32=f"{i.CRC:08x}"))
        rels.append(rel)
    zf.close()
    json.dump({"md5": md5, "published_md5": want_md5, "md5_ok": md5_ok, "sha256": sha, "n_entries": len(infos),
               "n_crc_bad": nbad, "rels": rels}, open(info, "w"))
    print("latent-diffusion md5 ok:", md5_ok, "sha256", sha, "entries", len(infos), "crc bad", nbad)
    zpath.unlink()  # keep only the 300 sampled images


def run():
    plan_ = json.load(open(STATE / "plan.json"))
    t0 = time.time()
    meta_raise = dict(label=0, source_dataset="RAISE-1k resized (B-Free extended)", cam="RAISE", license=CFG["raise_bfree"]["license"])
    meta_sb = dict(label=1, source_dataset="Synthbuster", cam="", license=CFG["synthbuster"]["license"])
    meta_bf = dict(label=1, source_dataset="B-Free extended Synthbuster", cam="", license=CFG["bfree_gen"]["license"])
    jobs = []
    # order: zips first (exact budget), latent-diffusion, VISION last
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        futs = []
        for key, meta in (("raise", meta_raise), ("sd3_flux", meta_bf), ("synthbuster", meta_sb)):
            z = plan_["zips"][key]
            for e in z["entries"]:
                m = dict(meta)
                if key == "sd3_flux" and "flux" in e["gen"].lower():
                    m["split"] = CFG["bfree_gen"]["flux_split"]
                futs.append(ex.submit(do_zip_entry, z, e, m))
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 100 == 0:
                print(f"[{time.time()-t0:.0f}s] zip entries {done}/{len(futs)} bytes {BUD.d['total']/1e9:.3f} GB", flush=True)
    do_latent_diffusion(CFG["seed"])
    print("zips done; bytes", BUD.d["total"], flush=True)
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        futs = [ex.submit(do_vision, it) for it in plan_["vision"]]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 100 == 0:
                print(f"[{time.time()-t0:.0f}s] vision {done}/{len(futs)} bytes {BUD.d['total']/1e9:.3f} GB", flush=True)
    print("FAILED:", FAILED)
    json.dump({"failed": FAILED, "crc_fail": CRC_FAIL}, open(STATE / "run_failures.json", "w"))
    print("run complete; total bytes", BUD.d["total"])


# ------------------------------------------------------------------ FROM-MANIFEST (exact rebuild on a fresh machine)
def from_manifest():
    """Download exactly the files in experiments/001-eval-set/manifest_raw.csv that are missing/wrong on disk.
    Selection is NOT re-done: URLs / zip entry names come from the committed state/plan.json, latent-diffusion
    picks from the manifest itself. Every file is checked against the manifest sha256. state/ is not modified."""
    global HARD
    HARD = float("inf")
    BUD.d = {"total": 0, "by_source": {}}
    BUD.path = None
    man = list(csv.DictReader(open(EXP / "manifest_raw.csv")))
    plan_ = json.load(open(STATE / "plan.json"))
    want = {r["path"]: r for r in man}

    def good(rel):
        p = ROOT / rel
        return p.is_file() and p.stat().st_size == int(want[rel]["bytes"]) and sha256_file(p) == want[rel]["sha256"]

    with ThreadPoolExecutor(CFG["workers"]) as ex:
        okmap = dict(zip(want, ex.map(good, want)))
    todo = {rel for rel, ok in okmap.items() if not ok}
    print(f"manifest files {len(want)}, already good {len(want) - len(todo)}, to fetch {len(todo)}", flush=True)
    failed = []

    def put(rel, data):
        if hashlib.sha256(data).hexdigest() != want[rel]["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {rel}")
        write_atomic(ROOT / rel, data)

    def do_v(it):
        last = None
        for attempt in range(RETRIES):
            try:
                data, _, _ = http(it["url"], "vision")
                put(it["rel"], data)
                return
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        failed.append((it["rel"], repr(last)))

    def do_z(z, e, rel):
        try:
            put(rel, fetch_entry_retry(z["url"], z["src"], e))
        except Exception as ex_:
            failed.append((rel, repr(ex_)))

    jobs = []
    vis = [it for it in plan_["vision"] if it["rel"] in todo]
    keymap = {"raise": "real_RAISE_1k", "synthbuster": "synthbuster", "sd3_flux": "sd3_flux"}
    for key, z in plan_["zips"].items():
        mine = []
        for e in z["entries"]:
            rel = f"data/raw/{z['src']}/{e['folder']}/{safe_name(e['name'])[-1]}"
            if rel in todo:
                mine.append((e, rel))
        if not mine:
            continue
        ents, _ = read_central_directory(z["url"], z["src"], RAW / z["src"] / "_central_directory" / f"{keymap[key]}.cd")
        live = {x["name"]: x for x in ents}
        for e, rel in mine:  # refresh offsets from the live central directory, require identical CRC32/size
            f = live.get(e["name"])
            if f is None or f["crc"] != e["crc"] or f["usize"] != e["usize"]:
                failed.append((rel, "zip entry missing or changed upstream"))
                continue
            jobs.append((z, dict(e, off=f["off"], csize=f["csize"], nl=f["nl"], method=f["method"]), rel))
    ld_rels = [r for r in json.load(open(STATE / "latent_diffusion.json"))["rels"] if r in todo]
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        futs = [ex.submit(do_z, z, e, rel) for z, e, rel in jobs] + [ex.submit(do_v, it) for it in vis]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 200 == 0:
                print(f"fetched {i}/{len(futs)} ({BUD.d['total']/1e9:.3f} GB)", flush=True)
    if ld_rels:
        src = "bfree_gen"
        d = RAW / src / "_archive"
        d.mkdir(parents=True, exist_ok=True)
        zpath = d / "latent-diffusion.zip"
        part = d / "latent-diffusion.zip.part"
        info = json.load(open(STATE / "latent_diffusion.json"))
        url = CFG["bfree_gen"]["latent_diffusion_url"]
        size = 127295883
        for attempt in range(RETRIES):
            have = part.stat().st_size if part.exists() else 0
            if have == size:
                break
            try:
                h = dict(UA)
                if have:
                    h["Range"] = f"bytes={have}-"
                with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                    if have and r.status != 206:  # server ignored Range: restart from scratch
                        part.unlink()
                    f = open(part, "ab")
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        BUD.add(src, len(b))
                    f.close()
            except Exception as e:
                print("latent-diffusion download retry:", repr(e), flush=True)
                time.sleep(2 * (attempt + 1))
        if not part.exists() or part.stat().st_size != size:
            failed.append(("latent-diffusion.zip", "download incomplete"))
        else:
            os.replace(part, zpath)
            if sha256_file(zpath) != info["sha256"]:
                failed.append(("latent-diffusion.zip", "sha256 differs from state/latent_diffusion.json"))
            else:
                import zipfile
                with zipfile.ZipFile(zpath) as zf:
                    byname = {}
                    for i in zf.infolist():
                        sp = safe_name(i.filename)
                        if sp:
                            byname.setdefault(sp[-1], []).append(i)
                    for rel in ld_rels:
                        try:
                            c = byname[rel.rsplit("/", 1)[-1]]
                            assert len(c) == 1
                            put(rel, zf.read(c[0]))
                        except Exception as ex_:
                            failed.append((rel, repr(ex_)))
                zpath.unlink()
    bad = [rel for rel in want if not good(rel)]
    print(f"from-manifest done: downloaded {BUD.d['total']} bytes; failed {len(failed)}; still bad/missing {len(bad)}")
    for f in failed[:20]:
        print("  FAIL", f)
    if failed or bad:
        sys.exit(5)


# ------------------------------------------------------------------ FINALIZE
def finalize():
    from collections import Counter, defaultdict
    from PIL import Image
    rows = list(DONE.rows.values())
    rows = [r for r in rows if (ROOT / r["path"]).exists()]
    rows.sort(key=lambda r: r["path"])
    cols = "path,label,source_dataset,generator,camera_or_platform,license,split,sha256,orig_format,width,height,bytes".split(",")
    with open(DATA / "manifest_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    res = {"counts": {}, "total_files": len(rows)}
    summ = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp = r["source_dataset"] + " | " + (r["generator"] if r["label"] == 1 else r["camera_or_platform"].split("_", 1)[-1] if r["source_dataset"] == "VISION" else r["generator"])
        res["counts"][grp] = res["counts"].get(grp, 0) + 1
        s = summ[grp]
        s["fmt"].append(r["orig_format"]); s["w"].append(int(r["width"])); s["h"].append(int(r["height"])); s["bytes"].append(int(r["bytes"]))
        if r["orig_format"] == "JPEG":
            with Image.open(ROOT / r["path"]) as im:
                q = getattr(im, "quantization", None)
                s["q00"].append(q[0][0] if q else -1)
    fm = {}
    for g, s in summ.items():
        n = len(s["w"])
        px = sorted(a * b for a, b in zip(s["w"], s["h"]))
        fm[g] = {
            "n": n, "formats": dict(Counter(s["fmt"])),
            "width_min_max": [min(s["w"]), max(s["w"])], "height_min_max": [min(s["h"]), max(s["h"])],
            "n_unique_wh": len(set(zip(s["w"], s["h"]))),
            "top_wh": [list(k) + [v] for k, v in Counter(zip(s["w"], s["h"])).most_common(3)],
            "megapixels_median": round(px[n // 2] / 1e6, 3),
            "bytes_total": sum(s["bytes"]), "bytes_median": sorted(s["bytes"])[n // 2],
        }
        if s["q00"]:
            fm[g]["jpeg_qtable00_counts_top"] = Counter(s["q00"]).most_common(5)
    res["format_summary"] = fm
    res["dataset_bytes_on_disk"] = sum(int(r["bytes"]) for r in rows)
    res["bytes_downloaded_total"] = BUD.d["total"]
    res["bytes_downloaded_by_source"] = BUD.d["by_source"]
    ld = STATE / "latent_diffusion.json"
    if ld.exists():
        j = json.load(open(ld))
        res["latent_diffusion"] = {k: j[k] for k in ("md5", "published_md5", "md5_ok", "sha256", "n_entries", "n_crc_bad")}
    fails = json.load(open(STATE / "run_failures.json")) if (STATE / "run_failures.json").exists() else {"failed": [], "crc_fail": []}
    res["n_crc_failures"] = len(fails["crc_fail"]) + (res.get("latent_diffusion", {}).get("n_crc_bad", 0))
    res["failed_files"] = fails["failed"]
    res["config"] = CFG
    json.dump(res, open(EXP / "results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("format_summary", "config")}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    try:
        {"plan": plan, "run": run, "finalize": finalize, "from-manifest": from_manifest}[cmd]()
    except BudgetStop as e:
        print("HARD STOP (budget):", e, "total", BUD.d["total"])
        sys.exit(3)
    except AbortRun as e:
        print("ABORT (repeated error):", e)
        sys.exit(4)
