#!/usr/bin/env python
"""Re-create / verify everything git-ignored in this repo from its ORIGINAL sources (see env/SOURCES.md).

    env/.venv/bin/python -I env/bootstrap_data.py --verify-only   # download nothing, print OK/MISSING/MISMATCH table
    env/.venv/bin/python -I env/bootstrap_data.py                 # fetch whatever is missing/wrong, then verify

Items:  third_party/* git clones at pinned commits | data/weights/* (SHA256 pinned) | RINE ckpt copies |
        DINOv2-base in data/hf_cache | VISION_files.txt | eval images in data/raw (via experiments/001-eval-set/fetch.py
        from-manifest, every file sha256-checked against experiments/001-eval-set/manifest_raw.csv) | data/manifest_raw.csv.
Hashes/commits below are the ones recorded in env/SOURCES.md (cross-checked against that file on every run).
Never "fixes" a hash: a download whose SHA256 differs is discarded and the run aborts.
Resumable: partial downloads live in <name>.part and are continued with HTTP Range.
Exit code 0 = everything OK, 1 = any problem.
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
W = DATA / "weights"
TP = ROOT / "third_party"
EXP = ROOT / "experiments" / "001-eval-set"
SOURCES_MD = ROOT / "env" / "SOURCES.md"
UA = {"User-Agent": "diffusion-detect-research-bootstrap/1.0"}

# HF cache must live in the project (same as the scorers); must be set BEFORE importing huggingface_hub.
os.environ["HF_HOME"] = str(DATA / "hf_cache")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)


class Abort(Exception):
    pass


# ----------------------------------------------------------------------------- pinned facts (from env/SOURCES.md)
GIT = {  # dir under third_party/ -> (url, commit)
    "Community-Forensics": ("https://github.com/JeongsooP/Community-Forensics.git", "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"),
    "B-Free": ("https://github.com/grip-unina/B-Free.git", "c6a9f898782fb466b29af01f21960b67415afb0e"),
    "rine": ("https://github.com/mever-team/rine.git", "9b7fd5857cc205d0412be6aeee0d7611b95bd620"),
    "CLIP": ("https://github.com/openai/CLIP.git", "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"),
    "ClipBased-SyntheticImageDetection": ("https://github.com/grip-unina/ClipBased-SyntheticImageDetection.git",
                                          "c76ef7f5e158c5aba9e55b8b94ab0079720d281e"),
}
CLIPDET = "https://github.com/grip-unina/ClipBased-SyntheticImageDetection"
CLIPDET_COMMIT = GIT["ClipBased-SyntheticImageDetection"][1]
CF_REPO, CF_REV = "OwensLab/commfor-model-384", "6076002bf0d9dd37537f965ee2f06f826c333b61"
LAION_REPO, LAION_REV = "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", "d979a50bf428edbb14b32c4dce7da2d49f7f46ed"
DINO_REPO, DINO_REV = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINO_FILES = {  # file -> (sha256 or None, git blob oid for the small non-LFS files, size)
    "model.safetensors": ("d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841", None, 346345912),
    "config.json": (None, "5df0129f878aa42d71fa74d27a50f382d13ed71e", 548),
    "preprocessor_config.json": (None, "ff5b47c2edcd1d3556d63c01a65d93b58b9efce1", 436),
}
VISION_FILES_SHA = "318ef924c8f5f2403dd4e0ff9708b51c22d53bcee5af81772472c03eb9eadb5e"

# plain-URL downloads: rel path under data/weights -> (urls tried in order, sha256, size)
URL_FILES = {
    "openai-clip-vitl14/ViT-L-14.pt": (
        ["https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"],
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836", 932768134),
    "b-free/BFREE_dino2reg4.zip": (
        ["https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"],
        "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f", 321653488),
    "clipdet/clipdet_latent10k_plus/weights.pth": (  # git-LFS object; recorded raw/main URL first, commit-pinned as fallback
        [f"{CLIPDET}/raw/main/weights/clipdet_latent10k_plus/weights.pth",
         f"{CLIPDET}/raw/{CLIPDET_COMMIT}/weights/clipdet_latent10k_plus/weights.pth"],
        "1c9c9f48a47d975bde67e9924983aa4e4ccda96a15e7e983b6305f0c56d9ee18", 5155),
    "clipdet/clipdet_latent10k/weights.pth": (
        [f"{CLIPDET}/raw/main/weights/clipdet_latent10k/weights.pth",
         f"{CLIPDET}/raw/{CLIPDET_COMMIT}/weights/clipdet_latent10k/weights.pth"],
        "04b022c6990709acb132337d7e432fc57090cec9db3869577111b031d40c19f9", 5227),
}
# files unzipped from the B-Free zip (zip member names are BFREE_dino2reg4/<file>; extracted into data/weights/b-free/)
BFREE_ZIP = "b-free/BFREE_dino2reg4.zip"
BFREE_UNZIPPED = {
    "b-free/BFREE_dino2reg4/model_epoch_best.pth": ("5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947", 346171370),
    "b-free/BFREE_dino2reg4/config.yaml": ("1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40", 153),
}
# HF files downloaded with local_dir=data/weights/<dir> (same layout as the original download, incl. .cache/huggingface)
HF_FILES = {
    "community-forensics/commfor-model-384/model.safetensors": (CF_REPO, CF_REV, "model.safetensors",
        "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387", 87262324),
    "community-forensics/commfor-model-384/config.json": (CF_REPO, CF_REV, "config.json",
        "877416e73aac0fdbc1be723f5ddf674e78a2350305865ad9031564b287b60147", 116),
    "laion-commonpool-vitl14/open_clip_pytorch_model.safetensors": (LAION_REPO, LAION_REV, "open_clip_pytorch_model.safetensors",
        "947d32621fa7da5c2fced632f85b50314ad581396628876f24d5a2e4ddebec0f", 1710517748),
    "laion-commonpool-vitl14/open_clip_config.json": (LAION_REPO, LAION_REV, "open_clip_config.json",
        "48238f62879cfab584edd2de407ba2ca10f897bdc947917b94a7ea9c5d7d4a3d", 471),
}
# RINE: checkpoints live in the git repo (third_party/rine/ckpt/); data/weights/rine/ holds byte-identical copies.
RINE = {
    "model_ldm_trainable.pth": ("13986ffc86e1bd9d57fd640f891358c4f14c0f8aea8e590e52c598541f69fdc1", 42093973),
    "model_4class_trainable.pth": ("6535ff6ecfa88d33c081e561c2cb2dcb594a1006b050eddb72a72ce005ed2ca1", 25298182),
    "model_1class_trainable.pth": ("427621294b40f37f774ef53eff96d205b93c709790ae24f70b4f5687842c5875", 42094118),
    "model_2class_trainable.pth": ("680d301b6d191da85b9dabcebf42cd3df20c5c669fa8320c498f835d02bf5ccb", 1139686),
}
# bytes the eval-image fetcher reads besides the images themselves: 3 zip tails (256 KiB) + central directories,
# latent-diffusion.zip (whole) + checksum.txt
FETCH_OVERHEAD = 3 * 262144 + 183582 + 98279 + 831914
LATENT_ZIP = 127295883 + 154


# ----------------------------------------------------------------------------- helpers
def sha256_file(p, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_oid(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout.strip()


def check_file(p, sha, size=None):
    p = Path(p)
    if not p.is_file():
        return "MISSING", "not found"
    if size is not None and p.stat().st_size != size:
        return "MISMATCH", f"size {p.stat().st_size} != {size}"
    got = sha256_file(p)
    return ("OK", "") if got == sha else ("MISMATCH", f"sha256 {got[:12]}.. != {sha[:12]}..")


def download(urls, dest, sha, size):
    """Resumable download to dest.part (HTTP Range); verified by SHA256 before the atomic rename. Aborts on mismatch."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for url in urls:
        for attempt in range(4):
            try:
                have = part.stat().st_size if part.exists() else 0
                if have > size:
                    part.unlink(); have = 0
                if have < size:
                    h = dict(UA)
                    if have:
                        h["Range"] = f"bytes={have}-"
                    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                        if have and r.status != 206:  # server ignored Range -> restart
                            have = 0
                        with open(part, "ab" if have else "wb") as f:
                            lastp, n = time.time(), have
                            while True:
                                b = r.read(1 << 20)
                                if not b:
                                    break
                                f.write(b); n += len(b)
                                if size > 50e6 and time.time() - lastp > 10:
                                    lastp = time.time()
                                    print(f"    {dest.name}: {n/1e6:.0f}/{size/1e6:.0f} MB", flush=True)
                if part.stat().st_size != size:
                    raise RuntimeError(f"incomplete: {part.stat().st_size}/{size} B")
                got = sha256_file(part)
                if got != sha:
                    part.unlink()
                    raise Abort(f"SHA256 MISMATCH for {dest.name} from {url}: got {got}, expected {sha}. Not fixing the hash; aborting.")
                os.replace(part, dest)
                return
            except Abort:
                raise
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 403, 404):
                    break  # try next URL
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                last = e
                time.sleep(3 * (attempt + 1))
    raise Abort(f"could not download {dest.name}: {last!r}")


def git_state(d):
    """(head, dirty) for a clone, or None if d is not a git repo."""
    if not (d / ".git").exists():
        return None
    head = run(["git", "-C", str(d), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no"])
    return head, bool(dirty)


# ----------------------------------------------------------------------------- items
# Each item: (name, check() -> (status, detail), fetch() or None).  fetch() must leave the item OK or raise Abort.
ITEMS = []


def item(name, check, fetch=None, planned=lambda: 0):
    ITEMS.append(dict(name=name, check=check, fetch=fetch, planned=planned))


# --- SOURCES.md consistency (guards against typos in the constants above)
def check_sources_md():
    txt = SOURCES_MD.read_text()
    need = [c for _, c in GIT.values()] + [CF_REV, LAION_REV, DINO_REV, VISION_FILES_SHA]
    need += [v[1] for v in URL_FILES.values()] + [v[0] for v in BFREE_UNZIPPED.values()] + [v[3] for v in HF_FILES.values()]
    need += [v[0] for v in RINE.values()] + [DINO_FILES["model.safetensors"][0]]
    miss = [n[:12] for n in need if n not in txt]
    return ("OK", f"{len(need)} hashes/commits found in SOURCES.md") if not miss else ("MISMATCH", f"not in SOURCES.md: {miss}")


item("SOURCES.md constants", check_sources_md)


# --- a. git clones
def make_git(name, url, commit):
    d = TP / name

    def check():
        st = git_state(d)
        if st is None:
            return ("MISSING", "no clone") if not d.exists() else ("MISMATCH", "exists but is not a git repo")
        head, dirty = st
        if head != commit:
            return "MISMATCH", f"HEAD {head[:10]} != {commit[:10]}"
        if dirty:
            return "MISMATCH", "tracked files modified"
        return "OK", commit[:10]

    def fetch():
        TP.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            run(["git", "clone", "--quiet", url, str(d)])
        elif git_state(d) is None:
            raise Abort(f"{d} exists but is not a git repo; refusing to touch it")
        elif git_state(d)[1]:
            raise Abort(f"{d} has local modifications; refusing to touch it")
        else:
            run(["git", "-C", str(d), "fetch", "--quiet", "origin"])
        run(["git", "-C", str(d), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", commit])

    item(f"third_party/{name}", check, fetch)


for _n, (_u, _c) in GIT.items():
    make_git(_n, _u, _c)


# --- b. weights from plain URLs
def make_url_file(rel, urls, sha, size):
    p = W / rel

    def fetch():
        download(urls, p, sha, size)

    item(f"weights/{rel}", lambda: check_file(p, sha, size), fetch, planned=lambda: size)


for _rel, (_urls, _sha, _size) in URL_FILES.items():
    make_url_file(_rel, _urls, _sha, _size)


# --- b. B-Free zip contents (unzipped to data/weights/b-free/BFREE_dino2reg4/)
def make_bfree_unzipped(rel, sha, size):
    p = W / rel

    def fetch():
        zp = W / BFREE_ZIP
        st, _ = check_file(zp, URL_FILES[BFREE_ZIP][1], URL_FILES[BFREE_ZIP][2])
        if st != "OK":
            raise Abort(f"{BFREE_ZIP} must be present and verified before unzipping")
        member = "BFREE_dino2reg4/" + p.name  # only the two known members are extracted, by exact name
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z, z.open(member) as src, open(p.with_name(p.name + ".part"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(p.with_name(p.name + ".part"), p)
        st, d = check_file(p, sha, size)
        if st != "OK":
            raise Abort(f"unzipped {rel}: {st} {d}")

    item(f"weights/{rel}", lambda: check_file(p, sha, size), fetch)


for _rel, (_sha, _size) in BFREE_UNZIPPED.items():
    make_bfree_unzipped(_rel, _sha, _size)


# --- b. HF files (local_dir layout, as originally downloaded)
def make_hf(rel, repo, rev, fname, sha, size):
    p = W / rel

    def fetch():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename=fname, revision=rev, local_dir=str(p.parent))
        st, d = check_file(p, sha, size)
        if st != "OK":
            raise Abort(f"HF {repo}@{rev[:8]}/{fname}: {st} {d}. Not fixing the hash; aborting.")

    item(f"weights/{rel}", lambda: check_file(p, sha, size), fetch, planned=lambda: size)


for _rel, _v in HF_FILES.items():
    make_hf(_rel, *_v)


# --- b. RINE: ckpt in the repo + copies in data/weights/rine/
def make_rine(fname, sha, size):
    repo_p = TP / "rine" / "ckpt" / fname
    copy_p = W / "rine" / fname

    def check():
        a = check_file(repo_p, sha, size)
        if a[0] != "OK":
            return a[0], f"third_party/rine/ckpt: {a[1]}"
        return check_file(copy_p, sha, size)

    def fetch():
        st, d = check_file(repo_p, sha, size)
        if st != "OK":
            raise Abort(f"third_party/rine/ckpt/{fname}: {st} {d} (re-clone rine at the pinned commit first)")
        copy_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_p, copy_p.with_name(copy_p.name + ".part"))
        os.replace(copy_p.with_name(copy_p.name + ".part"), copy_p)

    item(f"weights/rine/{fname}", check, fetch)


for _f, (_sha, _size) in RINE.items():
    make_rine(_f, _sha, _size)


# --- c. DINOv2-base in data/hf_cache (HF_HOME), snapshot-pinned
DINO_ROOT = DATA / "hf_cache" / "hub" / "models--facebook--dinov2-base"


def check_dino():
    snap = DINO_ROOT / "snapshots" / DINO_REV
    problems, missing = [], 0
    for fname, (sha, oid, size) in DINO_FILES.items():
        p = snap / fname
        if not p.exists():
            missing += 1
            problems.append(f"{fname} missing")
            continue
        if sha:
            st, d = check_file(p.resolve(), sha, size)
        else:
            st, d = ("OK", "") if p.resolve().stat().st_size == size and git_blob_oid(p) == oid else ("MISMATCH", "blob oid/size differs")
        if st != "OK":
            problems.append(f"{fname}: {st} {d}")
    ref = DINO_ROOT / "refs" / "main"
    if not ref.exists() or ref.read_text().strip() != DINO_REV:
        problems.append("refs/main missing/wrong")  # needed for offline from_pretrained('facebook/dinov2-base')
    if not problems:
        return "OK", DINO_REV[:10]
    return ("MISSING" if missing == len(DINO_FILES) else "MISMATCH"), "; ".join(problems)


def fetch_dino():
    from huggingface_hub import snapshot_download
    snapshot_download(DINO_REPO, revision=DINO_REV, allow_patterns=list(DINO_FILES))  # HF_HOME=data/hf_cache
    ref = DINO_ROOT / "refs" / "main"  # a commit-hash download writes no ref; the original layout has refs/main = pinned commit
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(DINO_REV)
    st, d = check_dino()
    if st != "OK":
        raise Abort(f"DINOv2-base: {st} {d}")


item("hf_cache/dinov2-base", check_dino, fetch_dino, planned=lambda: DINO_FILES["model.safetensors"][2] + 984)


# --- d. eval images
MAN_SRC = EXP / "manifest_raw.csv"
IMG = {"rows": None, "bad": None}


def read_manifest():
    if IMG["rows"] is None:
        IMG["rows"] = list(csv.DictReader(open(MAN_SRC)))
    return IMG["rows"]


def check_images():
    rows = read_manifest()

    def one(r):
        p = ROOT / r["path"]
        if not p.is_file():
            return "MISSING"
        if p.stat().st_size != int(r["bytes"]):
            return "MISMATCH"
        return "OK" if sha256_file(p) == r["sha256"] else "MISMATCH"

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, rows))
    IMG["bad"] = [r for r, s in zip(rows, res) if s != "OK"]
    nm, nx = res.count("MISSING"), res.count("MISMATCH")
    if not nm and not nx:
        return "OK", f"{len(rows)} files, sha256 verified"
    return ("MISSING" if nx == 0 else "MISMATCH"), f"{len(rows)} files: {nm} missing, {nx} wrong"


def planned_images():
    if IMG["bad"] is None:
        check_images()
    bad = {r["path"] for r in IMG["bad"]}
    if not bad:
        return 0
    plan = json.load(open(EXP / "state" / "plan.json"))
    ld = set(json.load(open(EXP / "state" / "latent_diffusion.json"))["rels"])
    tot = sum(it["size"] for it in plan["vision"] if it["rel"] in bad)
    for z in plan["zips"].values():
        for e in z["entries"]:
            rel = f"data/raw/{z['src']}/{e['folder']}/{e['name'].replace(chr(92), '/').split('/')[-1]}"
            if rel in bad:
                tot += e["csize"] + 30 + e["nl"] + 256
    if bad & ld:
        tot += LATENT_ZIP
    return tot + FETCH_OVERHEAD


def fetch_images():
    r = subprocess.run([sys.executable, "-I", str(EXP / "fetch.py"), "from-manifest"])
    if r.returncode != 0:
        raise Abort(f"fetch.py from-manifest exited {r.returncode}")
    st, d = check_images()
    if st != "OK":
        raise Abort(f"eval images: {st} {d}")


item("raw/eval images (manifest)", check_images, fetch_images, planned=planned_images)


def check_vision_list():
    return check_file(DATA / "raw" / "vision" / "VISION_files.txt", VISION_FILES_SHA)


def fetch_vision_list():
    url = json.load(open(EXP / "config.json"))["vision"]["file_list"]
    p = DATA / "raw" / "vision" / "VISION_files.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != VISION_FILES_SHA:
        raise Abort("VISION_files.txt SHA256 differs from SOURCES.md (site changed?). Not fixing the hash; aborting.")
    p.write_bytes(data)


item("raw/vision/VISION_files.txt", check_vision_list, fetch_vision_list, planned=lambda: 3521636)


def check_manifest_copy():
    dst = DATA / "manifest_raw.csv"
    if not dst.is_file():
        return "MISSING", "not found"
    return ("OK", "") if dst.read_bytes() == MAN_SRC.read_bytes() else ("MISMATCH", "differs from committed copy")


def fetch_manifest_copy():
    (DATA).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MAN_SRC, DATA / "manifest_raw.csv")


item("data/manifest_raw.csv", check_manifest_copy, fetch_manifest_copy)


# ----------------------------------------------------------------------------- driver
def table(results):
    w = max(len(n) for n, _, _ in results)
    for n, s, d in results:
        print(f"  {n:<{w}}  {s:<8}  {d}")
    c = {k: sum(1 for _, s, _ in results if s == k) for k in ("OK", "MISSING", "MISMATCH")}
    print(f"  => {c['OK']} OK, {c['MISSING']} MISSING, {c['MISMATCH']} MISMATCH (of {len(results)})")
    return c


def check_all():
    out = []
    for it in ITEMS:
        try:
            s, d = it["check"]()
        except Exception as e:
            s, d = "MISMATCH", f"check error: {e!r}"
        out.append((it["name"], s, d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true", help="download nothing; check everything; exit 1 on any problem")
    args = ap.parse_args()
    print(f"project root: {ROOT}")
    results = check_all()
    print("\nCurrent state:")
    c = table(results)
    if args.verify_only:
        sys.exit(0 if c["OK"] == len(results) else 1)
    todo = [it for it, (_, s, _) in zip(ITEMS, results) if s != "OK" and it["fetch"]]
    if any(n == "SOURCES.md constants" and s != "OK" for n, s, _ in results):
        sys.exit("script constants disagree with env/SOURCES.md; fix the script, not the hashes")
    if not todo:
        print("\nNothing to do.")
        sys.exit(0 if c["OK"] == len(results) else 1)
    planned = sum(it["planned"]() for it in todo)
    n_git = sum(1 for it in todo if it["name"].startswith("third_party/"))
    print(f"\nPlanned download: {planned:,} bytes ({planned/1e9:.3f} GB) for {len(todo)} items "
          f"(+ {n_git} git clone(s), size not counted).", flush=True)
    for it in todo:  # order = ITEMS order: clones -> weights -> rine copies -> dino -> images -> manifest copy
        print(f"\n[fetch] {it['name']}", flush=True)
        try:
            it["fetch"]()
        except Abort as e:
            print(f"ABORT: {e}")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"ABORT: command failed: {e.cmd}\n{e.stderr}")
            sys.exit(1)
    print("\nFinal verification:")
    c = table(check_all())
    sys.exit(0 if c["OK"] == len(ITEMS) else 1)


if __name__ == "__main__":
    main()
