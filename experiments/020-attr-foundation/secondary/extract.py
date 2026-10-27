"""020 secondary arms, step 1: DINOv2-base features for the NEW images with the 004 pipeline (descriptive arms, docs/09 decision 5).

  env/.venv/bin/python -I experiments/020-attr-foundation/secondary/extract.py [prep|feats|guard|parity|all]

Pipeline = 004 exactly: p4run.load_cond (perturb.apply_condition(key=raw sha256, seed 0)), Backbone(layers 4/8/12, CLS+mean, 224 centre crop),
fp32, TF32 off, batch 20, shards via p4features.write_shard. Only deviation of mechanism: the forward loop retries on CUDA OOM (smaller batch)
instead of the sticky CPU fallback of features.embed_images (the GPU is shared with another job); numerics are the same forward.
Images: SD1.3 / SD1.4 (512 px) -> fov512 (no-op resize + centre crop); GLIDE is 256 px -> 2x BICUBIC upscale to 512 (DEVIATION: fov512 never upscales);
RAID new ids -> fov224 (short side -> 224 bicubic, centre crop). All via lib/preprocess.process_image except GLIDE.
Cache: secondary/cache/{fov512,fov224}/<cond>/part_*.npz (git-ignored), keyed by RAW sha256 (as the 004/016 caches).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for p in (ROOT / "experiments" / "lib", ROOT / "experiments" / "004-phase4-features"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import preprocess
import p4features as P

FETCH = ROOT / "experiments" / "020-attr-foundation" / "fetch"
CACHE = HERE / "cache"
IMG = CACHE / "img"
RES = HERE / "results"
RES.mkdir(exist_ok=True)
CONDS512 = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]
CONDS224 = ["clean", "jpeg(75)", "jpeg(90)"]
BATCH = 20


def log(*a):
    print(time.strftime("%H:%M:%S"), "SEC-X", *a, flush=True)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def p4():
    if "p4run" not in sys.modules:
        load_mod("p4run", ROOT / "experiments" / "004-phase4-features" / "run.py")
    return sys.modules["p4run"]


# ----------------------------------------------------------------------------- new-image table
def new_table():
    sb = pd.read_csv(FETCH / "synthbuster_new_manifest.csv")
    sb = pd.DataFrame(dict(sha256=sb.sha256, src_path=sb.path, generator=sb.generator, rid=sb.caption_id, universe="fov512",
                           w=sb.width, h=sb.height))
    rd = pd.read_csv(FETCH / "raid_new_manifest.csv")
    rd = rd[rd["class"] != "real"]
    rd = pd.DataFrame(dict(sha256=rd.sha256, src_path=rd.path, generator="d3_" + rd["class"], rid=rd.id.astype(str), universe="fov224",
                           w=rd.width, h=rd.height))
    df = pd.concat([sb, rd], ignore_index=True)
    df["path"] = [str(IMG / u / f"{s}.png") for u, s in zip(df.universe, df.sha256)]
    return df


def prep_one(a):
    src, dst, uni, sha, size = a
    if os.path.exists(dst):
        return "exists"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        small = min(im.size) < size
    if small and uni == "fov512":                       # GLIDE 256 -> 512: DEVIATION (documented), plain bicubic 2x
        from PIL import ImageOps
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = im.resize((size, size), Image.BICUBIC)
            Image.fromarray(np.asarray(im)).save(dst, "PNG")
        return "upscaled"
    st, _ = preprocess.process_image(src, dst, size, mode="fov")
    assert st == "ok", (src, st)
    return "ok"


def stage_prep():
    df = new_table()
    import hashlib
    bad = [s for s, p in zip(df.sha256, df.src_path) if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
    assert not bad, f"{len(bad)} raw files differ from manifest sha"
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(prep_one, [(s, d, u, sh, 512 if u == "fov512" else 224) for s, d, u, sh in zip(df.src_path, df.path, df.universe, df.sha256)]))
    for p in df.path:
        with Image.open(p) as im:
            assert im.size == ((512, 512) if "/fov512/" in p else (224, 224)) and im.mode == "RGB", p
    df.to_csv(HERE / "cache" / "new_table.csv", index=False)
    log("prep:", pd.Series(res).value_counts().to_dict(), "n", len(df))
    return df


# ----------------------------------------------------------------------------- forward with OOM retry
def embed_retry(bb, imgs, batch, dev):
    import torch
    out, i = [], 0
    while i < len(imgs):
        b = max(1, min(batch, len(imgs) - i))
        for attempt in range(8):
            try:
                x = torch.stack([bb.to_tensor(im) for im in imgs[i:i + b]])
                bb.model.to(dev)
                f = bb.forward(x.to(dev))
                out.append(f.float().cpu().numpy())
                bb.n_forward += len(x)
                i += b
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                b = max(1, b // 2)
                log(f"  CUDA OOM -> batch {b}")
                time.sleep(3)
        else:
            raise SystemExit("repeated CUDA OOM")
    return np.concatenate(out)


def extract_cond(df, bb, cond, cache, dev, pool, spec, chunk=100):
    run = p4()
    P.check_meta(cond, run.meta_for(bb, spec), cache)
    have = P.cached_keys(cond, cache)
    todo = df[~df.sha256.isin(have)]
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        ch = todo.iloc[s:s + chunk]
        imgs = list(pool.map(lambda r: run.load_cond(r[0], r[1], cond), zip(ch.path, ch.sha256)))
        f = embed_retry(bb, imgs, BATCH, dev)
        P.write_shard(cond, ch.sha256.tolist(), f, cache)
    log(f"  {cond}: {len(todo)} new ({len(df) - len(todo)} cached) in {time.time() - t0:.0f}s")


def backbone():
    run = p4()
    from features import get_device
    bb = run.secondary_backbone()
    run.check_design_vs_003(bb)
    return bb, get_device().type, run.P.SPEC


def stage_feats():
    df = pd.read_csv(CACHE / "new_table.csv")
    bb, dev, spec = backbone()
    log("device", dev)
    with ThreadPoolExecutor(4) as pool:
        for uni, conds in (("fov512", CONDS512), ("fov224", CONDS224)):
            d = df[df.universe == uni].reset_index(drop=True)
            for c in conds:
                extract_cond(d, bb, c, CACHE / uni, dev, pool, spec)


# ----------------------------------------------------------------------------- D-008 guard (GPU features in the cache vs CPU recomputation)
def old_probe(uni):
    """Probe for the |dlogit| guard: 020 b8, C=0.01 (perm_control constants). fov512: 8-way fakes, T-mix; fov224: RAID 4-way, T-mix."""
    run = load_mod("e11run", ROOT / "experiments" / "020-attr-foundation" / "run.py") if "e11run" not in sys.modules else sys.modules["e11run"]
    run.RES = HERE / "results" / "_tmp_folds"           # build_analyses writes fold files: keep them out of the frozen 020 results
    run.RES.mkdir(parents=True, exist_ok=True)
    A, _ = run.build_analyses()
    a = A["ha1_main" if uni == "fov512" else "ha1_raid"]
    X = np.concatenate([a.getX("b8", c) for c in a.tmix])
    y = np.tile(a.y, len(a.tmix))
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=300, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


def stage_guard():
    df = pd.read_csv(CACHE / "new_table.csv")
    bb, dev, spec = backbone()
    run = p4()
    from features import get_device
    out = {}
    gp = RES / "d008_guard.json"
    for uni, conds in (("fov512", CONDS512), ("fov224", CONDS224)):
        d = df[df.universe == uni].reset_index(drop=True)
        rng = np.random.default_rng(P.SEED)
        shas = sorted(d.sha256)
        sub = [shas[i] for i in sorted(rng.choice(len(shas), 32, replace=False))]
        sc, clf = old_probe(uni)
        for c in conds:
            keys, X = P.read_shards(c, CACHE / uni)
            pos = {k: i for i, k in enumerate(keys.tolist())}
            cp = CACHE / uni / "cpuref" / f"{P.safe(c)}.npz"
            ref = {}
            if cp.exists():
                z = np.load(cp, allow_pickle=False)
                ref = dict(zip(z["keys"].tolist(), z["feats"]))
            miss = [s for s in sub if s not in ref]
            if miss:
                dd = d.set_index("sha256", drop=False).loc[miss]
                imgs = [run.load_cond(p, s, c) for p, s in zip(dd.path, dd.sha256)]
                from features import embed_images
                ref.update(zip(miss, embed_images(bb, imgs, 4, "cpu")))
                cp.parent.mkdir(parents=True, exist_ok=True)
                ks = sorted(ref)
                np.savez(cp, keys=np.array(ks), feats=np.stack([ref[k] for k in ks]))
            A_ = np.stack([X[pos[s]] for s in sub])
            C_ = np.stack([ref[s] for s in sub])
            dif = np.abs(A_ - C_)
            scaled = float((dif / np.maximum(1.0, np.abs(C_).max(0, keepdims=True))).max())
            sl = slice(1536, 3072)
            dl = float(np.abs(clf.decision_function(sc.transform(A_[:, sl])) - clf.decision_function(sc.transform(C_[:, sl]))).max())
            o = dict(n=len(sub), device=dev, max_abs_feature_diff=float(dif.max()), max_scaled_feature_diff=scaled, max_abs_dlogit_b8=dl,
                     feat_tol=1e-3, logit_tol=1e-3, ok=bool(scaled <= 1e-3 and dl <= 1e-3))
            out[f"{uni}/{c}"] = o
            log(f"  D-008 {uni}/{c}: max|dF|={dif.max():.2e} scaled={scaled:.2e} max|dlogit_b8|={dl:.2e} -> {'ok' if o['ok'] else 'FAIL'}")
    gp.write_text(json.dumps(out, indent=1, sort_keys=True))
    if not all(v["ok"] for v in out.values()):
        raise SystemExit("D-008 ABORT")


# ----------------------------------------------------------------------------- pipeline parity: re-extract existing cached images
def stage_parity():
    run = load_mod("e11run", ROOT / "experiments" / "020-attr-foundation" / "run.py") if "e11run" not in sys.modules else sys.modules["e11run"]
    p4m = p4()
    bb, dev, spec = backbone()
    out = {}
    rng = np.random.default_rng(1)
    with ThreadPoolExecutor(4) as pool:
        for uname, conds in (("fov512", CONDS512), ("fov224", ["clean", "jpeg(75)"])):
            df, u = run.load_universe(uname)
            df = df[~df.dup_drop]
            if uname == "fov512":                                   # 2 per fake generator
                rows = np.concatenate([rng.choice(np.where(df.generator == g)[0], 2, replace=False) for g in sorted(set(df.generator[df.pool == "fake"]))])
            else:
                rows = rng.choice(len(df), 16, replace=False)
            sub = df.iloc[rows]
            # (a) raw -> processed pixel parity (prep code path)
            pix = []
            size = 512 if uname == "fov512" else 224
            for r in sub.itertuples():
                tmp = CACHE / "parity_tmp.png"
                st, _ = preprocess.process_image(r.src_path, tmp, size, mode="fov")
                a = np.asarray(Image.open(tmp)); b = np.asarray(Image.open(ROOT / r.path).convert("RGB"))
                pix.append(bool(st == "ok" and a.shape == b.shape and np.array_equal(a, b)))
            # (b) features
            res = {}
            for c in conds:
                store = run.read_caches(u["caches"], c)
                imgs = list(pool.map(lambda r: p4m.load_cond(ROOT / r[0], r[1], c), zip(sub.path, sub.sha256)))
                f = embed_retry(bb, imgs, BATCH, dev)
                cached = np.stack([store[s] for s in sub.sha256])
                d = np.abs(f - cached)
                sc = float((d / np.maximum(1.0, np.abs(cached).max(0, keepdims=True))).max())
                res[c] = dict(max_abs=float(d.max()), max_scaled=sc, ok=bool(sc <= 1e-3))
                log(f"  parity {uname}/{c}: n={len(sub)} max|dF|={d.max():.2e} scaled={sc:.2e}")
            out[uname] = dict(n=len(sub), generators=sub.generator.tolist(), pixel_parity_raw_to_processed=pix, features=res,
                              ok=bool(all(pix) and all(v["ok"] for v in res.values())))
    (RES / "parity.json").write_text(json.dumps(out, indent=1))
    if not all(v["ok"] for v in out.values()):
        raise SystemExit("PARITY FAILED")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("prep", "all"):
        stage_prep()
    if st in ("feats", "all"):
        stage_feats()
    if st in ("guard", "all"):
        stage_guard()
    if st in ("parity", "all"):
        stage_parity()
