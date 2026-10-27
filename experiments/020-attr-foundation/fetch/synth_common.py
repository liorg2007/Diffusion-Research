"""Shared zip helpers for the 020 Synthbuster/RAID fetch (copied from experiments/001-eval-set/fetch.py,
which cannot be imported without side effects: it writes state/budget.json)."""
import re, struct, zlib, time, threading, urllib.request, urllib.error

TRANSFER = [0]  # compressed bytes received over HTTP (all threads)
_LOCK = threading.Lock()

UA = {"User-Agent": "diffusion-detect-research/1.0 (academic evaluation-set fetch, 020-attr-foundation)"}


def http_range(url, start, end, retries=5):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={start}-{end}"))
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"expected 206 got {r.status}")
                cr = r.headers.get("Content-Range")
                if cr and int(cr.split()[1].split("-")[0]) != start:
                    raise RuntimeError("range start mismatch")
                data = r.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short read {len(data)}")
                with _LOCK:
                    TRANSFER[0] += len(data)
                return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 416):
                break
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def parse_cd(buf):
    ents, p = [], 0
    while p + 46 <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        (sig, vm, vn, flags, method, mt, md, crc, csize, usize, nl, el, cl, dn, ia, ea, off) = \
            struct.unpack("<4sHHHHHHLLLHHHHHLL", buf[p:p + 46])
        name = buf[p + 46:p + 46 + nl].decode("utf-8" if flags & 0x800 else "cp437", errors="replace")
        extra = buf[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                z, k = extra[q + 4:q + 4 + hl], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", z[k:k + 8])[0]; k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", z[k:k + 8])[0]
            q += 4 + hl
        ents.append({"name": name, "method": method, "crc": crc, "csize": csize, "usize": usize,
                     "off": off, "nl": nl, "flags": flags})
        p += 46 + nl + el + cl
    return ents


def fetch_entry(url, e):
    """Range-fetch one zip entry (local header + data), decompress, check size and CRC32 vs central directory."""
    want = 30 + e["nl"] + 256 + e["csize"]
    buf = http_range(url, e["off"], e["off"] + want - 1)
    sig, vn, fl, meth, mt, md, crc, cs, us, nl, el = struct.unpack("<4sHHHHHLLLHH", buf[:30])
    if sig != b"PK\x03\x04":
        raise RuntimeError("bad local header")
    ds = 30 + nl + el
    if len(buf) < ds + e["csize"]:
        buf += http_range(url, e["off"] + len(buf), e["off"] + ds + e["csize"] - 1)
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
    crc_ok = (zlib.crc32(raw) & 0xFFFFFFFF) == e["crc"]
    return raw, crc_ok


RID_RE = re.compile(r"(r[0-9a-f]{8,}t)")
