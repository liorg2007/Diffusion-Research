"""Selftest for run.py (synthetic data + planted leaks; no real-data result is read).

    env/.venv/bin/python -I experiments/022-attr-platform/selftest.py   -> results/selftest.txt
"""
import importlib.util
import io
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hrun", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["hrun"] = R
spec.loader.exec_module(R)

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold

LINES, FAILS = [], []


def check(name, ok, detail=""):
    s = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    print(s, flush=True)
    LINES.append(s)
    if not ok:
        FAILS.append(name)


def raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- 1. primitives
def t_primitives():
    zz = R.ZZ
    check("zigzag first cells", zz[:6] == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)], str(zz[:6]))
    from preprocess import _STD_LUMA
    seq = [int(_STD_LUMA[r * 8 + c]) for r, c in zz[:16]]
    check("zigzag order reproduces the standard luminance table sequence", seq == [16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40], str(seq))
    check("15 AC coefficients = zigzag 1..15", len(R.AC) == 15 and R.AC[0] == (0, 1) and R.AC[-1] == zz[15])
    rng = np.random.default_rng(0)
    Y = rng.uniform(0, 255, (24, 40)).astype(np.float32)
    D = R.block_dct(Y)
    from scipy.fft import dct
    ref = dct(dct(Y[8:16, 16:24] - 128.0, type=2, norm="ortho", axis=0), type=2, norm="ortho", axis=1)
    check("block_dct matches separable scipy DCT and block order", np.allclose(D[1 * 5 + 2], ref, atol=1e-3), f"max|d|={np.abs(D[7] - ref).max():.2e}")
    check("block_dct is orthonormal (Parseval)", abs((D ** 2).sum() - ((Y - 128.0) ** 2).sum()) / ((Y - 128.0) ** 2).sum() < 1e-5)
    Y2 = rng.uniform(0, 255, (27, 43)).astype(np.float32)
    check("block_dct crops to multiples of 8 from the top-left", R.block_dct(Y2).shape == (3 * 5, 8, 8))
    # periodicity on planted combs
    ok, det = True, []
    for step in (3, 5, 8, 12, 17, 24):
        x = step * np.round(rng.laplace(0, 3 * step, 60000)) + rng.normal(0, 0.05, 60000)
        s, p = R.periodicity(x)
        det.append((step, s, round(p, 2)))
        ok &= (s == step)
    check("periodicity recovers planted quantisation steps", ok, str(det))
    s, p = R.periodicity(rng.laplace(0, 6, 60000))
    check("periodicity peak is small for a smooth (unquantised) density", p < 0.05, f"peak={p:.3f}")
    # normalised histograms
    img = make_image(rng, 128, 96)
    f = R.feat_A_from_Y(R.luma(img))
    h = f[:15 * R.NB].reshape(15, R.NB)
    check("feature vector: 990 dims, finite, histograms sum to 1", f.shape == (990,) and np.isfinite(f).all() and np.allclose(h.sum(1), 1, atol=1e-5))


def make_image(rng, w, h, noise=3.0, tint=(0, 0, 0)):
    base = np.stack([gaussian_filter(rng.normal(0, 1, (h, w)), s) for s in (6, 3, 12)], -1)
    base = (base - base.mean()) / (base.std() + 1e-9)
    edges = np.sin(np.arange(w)[None, :, None] / rng.uniform(4, 15) + rng.uniform(0, 6)) * 20
    a = 128 + 45 * base + edges + rng.normal(0, noise, (h, w, 3)) + np.array(tint)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def jpg(img, q, path):
    img.save(path, "JPEG", quality=int(q))
    return Image.open(path).convert("RGB")


def t_step_on_real_jpeg():
    rng = np.random.default_rng(1)
    from preprocess import _STD_LUMA, _table_for_quality
    rate = {}
    for q in (50, 75):
        tab = _table_for_quality(q).astype(int)
        want = [tab[r * 8 + c] for r, c in R.AC]
        hit, tot = 0, 0
        with tempfile.TemporaryDirectory() as td:
            for i in range(3):
                dec = jpg(make_image(rng, 512, 384, noise=6), q, Path(td) / "x.jpg")
                D = R.block_dct(R.luma(dec))
                for k, (r, c) in enumerate(R.AC):
                    s, _ = R.periodicity(D[:, r, c])
                    hit += int(s == want[k]) if want[k] <= R.S_MAX - 1 else 0
                    tot += int(want[k] <= R.S_MAX - 1)
        rate[q] = hit / tot
    check("step estimator recovers the true luminance steps on decoded JPEGs (informational, >=30% exact at q50 and q75)", min(rate.values()) >= 0.3, f"exact rate {rate}")


# ---------------------------------------------------------------- 2. synthetic triplets through the real code path
def synth_dataset(td, n_dev=12, n_photo=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dev):
        q = int(rng.integers(86, 97))
        noise = rng.uniform(1.5, 4.0)
        tint = rng.normal(0, 4, 3)
        for p in range(n_photo):
            img = make_image(rng, 320, 240, noise, tint)
            nat = jpg(img, q, Path(td) / f"D{d:02d}_{p}_nat.jpg")
            wa = jpg(nat.resize((160, 120), Image.BICUBIC), 58, Path(td) / f"D{d:02d}_{p}_wa.jpg")
            fb = jpg(nat.resize((256, 192), Image.BICUBIC), 84, Path(td) / f"D{d:02d}_{p}_fb.jpg")
            for plat, nm in enumerate(("nat", "wa", "fb")):
                rows.append(dict(src_path=str(Path(td) / f"D{d:02d}_{p}_{nm}.jpg"), platform=plat, device=f"D{d:02d}", vidx=str(p),
                                 photo=f"D{d:02d}_{p}", model=f"M{d % 6}"))
    t = pd.DataFrame(rows)
    t["sha256"] = [__import__("hashlib").sha256(Path(p).read_bytes()).hexdigest() for p in t.src_path]
    return t


def t_pipeline():
    with tempfile.TemporaryDirectory() as td:
        t = synth_dataset(td)
        y = t.platform.to_numpy()
        dev = t.device.to_numpy()
        folds = R.make_folds(t)
        check("synthetic folds: no device in both train and test, each device tested once", True)  # make_folds asserts internally
        # planted duplicate: a WA row pointing at a native file must trip the dedupe gate (absolute paths: ROOT / abs == abs)
        t2 = t.copy()
        t2.loc[t2.index[1], "src_path"] = t2.src_path.iloc[0]
        t2.loc[t2.index[1], "sha256"] = t2.sha256.iloc[0]
        check("planted duplicate sha256 across classes is caught by the dedupe gate", raises(lambda: R.gate_dedupe(t2)))
        check("dedupe gate passes on the clean synthetic table", R.gate_dedupe(t) == len(t))
        XA = {c: {"A": R.extract_A(t, c, 6, cache_dir=Path(td) / "cache")} for c in R.CONDS}
        check("method A features finite for all conditions", all(np.isfinite(XA[c]["A"]).all() for c in R.CONDS))
        # same laundering code path / determinism: re-extraction of social_sim equals the cache and depends only on the key
        img = Image.open(t.src_path.iloc[0]).convert("RGB")
        a = R.perturb.apply_condition(img, "social_sim", key=t.sha256.iloc[0], seed=0)
        b = R.perturb.apply_condition(img, "social_sim", key=t.sha256.iloc[0], seed=0)
        c_ = R.perturb.apply_condition(img, "social_sim", key=t.sha256.iloc[1], seed=0)
        check("social_sim deterministic per key, differs across keys", np.array_equal(np.asarray(a), np.asarray(b)) and a.size != c_.size or not np.array_equal(np.asarray(a), np.asarray(c_)))
        W = R.boot_weights(len(set(dev)), 200)
        dc, _ = R.dev_codes(dev)
        out = {}
        for mode, tc in R.CFG["train_modes"].items():
            R._G["S_A_" + mode] = dict(X=XA, y=y, g=dev, folds=folds, train_conds=tc, test_conds=R.CONDS, blocks=["A"])
            pr, info = R.run_cv("S_A_" + mode, workers=4)
            out[mode] = {c: R.bacc(y, pr[c]) for c in R.CONDS}
        check("A clean-train, clean test: bacc >= 0.90 on synthetic platform differences", out["clean"]["clean"] >= 0.90, f"{out['clean']}")
        check("A T-mix: clean >= 0.90 and social_sim >= 0.50", out["tmix"]["clean"] >= 0.90 and out["tmix"]["social_sim"] >= 0.5, f"{out['tmix']}")
        # M0
        raw = R.m0_raw(t)
        p0 = R.m0_cv(raw, y, folds)
        check("M0 (size + quantisation table) ~1.0 on synthetic files", R.bacc(y, p0) >= 0.95, f"bacc={R.bacc(y, p0):.3f}")
        pt = R.m0_cv(R.m0_tensor(t), y, folds)
        check("M0 on preprocessed tensors is exactly at chance", abs(R.bacc(y, pt) - 1 / 3) < 1e-9, f"bacc={R.bacc(y, pt):.4f}")
        # bootstrap / summary
        s = R.summarize(y, pr["clean"], t, W, dc)
        lo, pt_, hi = s["bacc"][1], s["bacc"][0], s["bacc"][2]
        check("bootstrap CI brackets the point estimate; confusion sums to n", lo <= pt_ <= hi and sum(map(sum, s["confusion"])) == len(t), str(s["bacc"]))
        check("bacc: perfect=1, constant prediction=1/3", R.bacc(y, y) == 1.0 and abs(R.bacc(y, np.zeros_like(y)) - 1 / 3) < 1e-9)

        # ---------- permutation control (grouped CV, labels shuffled within triplet): must be at chance
        rng = np.random.default_rng(7)
        idx = pd.Series(np.arange(len(t))).groupby(t.photo.to_numpy()).apply(list).to_dict()
        perms = []
        for _ in range(24):
            yp = y.copy()
            for ix in idx.values():
                yp[ix] = rng.permutation(y[ix])
            perms.append(yp)
        R._G["S_perm"] = dict(R._G["S_A_tmix"], perm=perms)
        pr, _ = R.run_cv("S_perm", jobs_perm=range(len(perms)), workers=4, fixed=("A", 0.01))
        pb = np.array([R.bacc(perms[i], pr[i]["clean"]) for i in range(len(perms))])
        check("permutation control (24 shuffles, grouped CV) is at chance", abs(pb.mean() - 1 / 3) < 0.05, f"mean={pb.mean():.3f} sd={pb.std():.3f}")

        # ---------- planted leak 1: a label-carrying feature must be found (pipeline has power)
        Xl = {c: {"A": np.hstack([XA[c]["A"], np.eye(3)[y] * 5])} for c in R.CONDS}
        R._G["S_leak1"] = dict(X=Xl, y=y, g=dev, folds=folds, train_conds=["clean"], test_conds=["clean"], blocks=["A"])
        pr, _ = R.run_cv("S_leak1", workers=4)
        check("planted label feature -> bacc ~1.0 (pipeline can detect signal)", R.bacc(y, pr["clean"]) > 0.99, f"{R.bacc(y, pr['clean']):.3f}")

        # ---------- planted leak 2: content memorisation. Photo-level random labels (all copies of a photo share one random label) and a
        # per-photo content fingerprint feature: a split that puts copies of a photo on both sides leaks, a device-grouped split cannot.
        ph = t.photo.to_numpy()
        up = sorted(set(ph))
        pi = np.array([up.index(p) for p in ph])
        fp = np.random.default_rng(9).normal(0, 1, (len(up), 40))
        Xf = (fp[pi] + np.random.default_rng(5).normal(0, 0.3, (len(t), 40))).astype(np.float32)
        leaky = list(KFold(5, shuffle=True, random_state=0).split(Xf))
        check("group-overlap assertion trips on a leaky (random) split", raises(lambda: R.assert_no_group_overlap(leaky, dev)))
        for nm, fo in (("grouped", folds), ("leaky", leaky)):
            accs = []
            for sd in range(6):                                     # average over label draws (72 photos only)
                yp = np.random.default_rng(10 + sd).integers(0, 3, len(up))[pi]
                R._G["S_fp_" + nm] = dict(X={"clean": {"A": Xf}}, y=yp, g=dev, folds=fo, train_conds=["clean"], test_conds=["clean"], blocks=["A"])
                prr, _ = R.run_cv("S_fp_" + nm, workers=4, fixed=("A", 1.0))
                accs.append(R.bacc(yp, prr["clean"]))
            out[nm] = float(np.mean(accs))
        check("planted content fingerprint + random photo labels: device-grouped CV at chance, leaky CV high (control has power)", out["grouped"] < 0.42 and out["leaky"] > 0.6,
              f"grouped={out['grouped']:.3f} leaky={out['leaky']:.3f}")

        # ---------- method B block selection on synthetic cached-like features: signal only in b8
        rng = np.random.default_rng(11)
        n = len(t)
        base = rng.normal(0, 1, (n, 1536))
        XB = {}
        for c in R.CONDS:
            b4, b8, b12 = [rng.normal(0, 1, (n, 1536)).astype(np.float32) for _ in range(3)]
            b8[:, :20] += np.eye(3)[y] @ rng.normal(0, 2.0, (3, 20))
            XB[c] = {"b4": b4, "b8": b8, "b12": b12}
        R._G["S_B"] = dict(X=XB, y=y, g=dev, folds=folds, train_conds=["clean"], test_conds=["clean"], blocks=["b4", "b8", "b12"])
        pr, info = R.run_cv("S_B", workers=4)
        blocks = [v["block"] for v in info.values()]
        check("method B: inner CV picks the informative block (b8) and reaches bacc >= 0.9", blocks.count("b8") >= 4 and R.bacc(y, pr["clean"]) >= 0.9, f"blocks={blocks} bacc={R.bacc(y, pr['clean']):.3f}")

        # ---------- VISION same-model transfer on planted model features (devices of one model share a signature)
        tn = t[t.platform == 0].reset_index(drop=True)
        rng = np.random.default_rng(13)
        sig = rng.normal(0, 1, (6, 1536))
        dsig = rng.normal(0, 0.6, (len(set(tn.device)), 1536))
        mi = tn.model.str[1:].astype(int).to_numpy()
        di = R.dev_codes(tn.device.to_numpy())[0]
        Xm = (sig[mi] * 0.8 + dsig[di] + rng.normal(0, 1, (len(tn), 1536))).astype(np.float32)
        res = R.vision_model_transfer(Xm, tn, n_splits=4)
        mean_a, mean_c = np.mean([r["auroc"] for r in res]), np.mean([r["control_auroc"] for r in res])
        check("same-model transfer: AUROC high on planted model signature, control near 0.5", mean_a > 0.85 and abs(mean_c - 0.5) < 0.15,
              f"pairs={len(res)} auroc={mean_a:.3f} control={mean_c:.3f}")
        Xr = rng.normal(0, 1, Xm.shape).astype(np.float32)
        res0 = R.vision_model_transfer(Xr, tn, n_splits=4)
        check("same-model transfer on pure noise is ~0.5", abs(np.mean([r["auroc"] for r in res0]) - 0.5) < 0.1, f"{np.mean([r['auroc'] for r in res0]):.3f}")


# ---------------------------------------------------------------- 3. label rule, HDR+ merge, real-data structure (no results)
def t_label_and_structure():
    def mk(ac, asoc, bc, bsoc, ac2=None):
        d = {"A": {"tmix": dict(clean=ac, social_sim=asoc, jpeg75_resize05=0.5), "clean": dict(clean=ac, social_sim=asoc, jpeg75_resize05=0.5)},
             "B": {"tmix": dict(clean=bc, social_sim=bsoc, jpeg75_resize05=0.5), "clean": dict(clean=bc, social_sim=bsoc, jpeg75_resize05=0.5)}}
        return R.decide_label(d)["tmix"]
    check("label: PASS when A >= .80 clean and >= .60 social", mk(.9, .65, .4, .3)["label"] == "PASS")
    check("label: PASS via B", mk(.5, .4, .85, .61)["label"] == "PASS")
    check("label: KILL when A < .60 and B < .50", mk(.55, .4, .45, .4)["label"] == "KILL")
    check("label: INCONCLUSIVE otherwise (clean pass but social fails)", mk(.9, .55, .45, .4)["label"] == "INCONCLUSIVE")
    check("label: INCONCLUSIVE mid-range counts as failure when max clean < .80", mk(.7, .5, .45, .4)["counts_as_failure"] is True)
    hm = R.hdrplus_table()
    check("HDR+ codenames merged to 6 models", sorted(hm.model.unique()) == ["Nexus 5", "Nexus 5X", "Nexus 6", "Nexus 6P", "Pixel", "Pixel XL"] and len(hm) == 600,
          str(hm.model.value_counts().to_dict()))
    t = R.build_table()
    check("real table: 485 triplets, 1455 images, D13 excluded, 34 devices", t.photo.nunique() == 485 and len(t) == 1455 and "D13" not in set(t.device) and t.device.nunique() == 34)
    check("real table: every triplet has one image per platform", (t.groupby("photo").platform.apply(lambda s: sorted(s)) .map(lambda s: s == [0, 1, 2])).all())
    folds = R.make_folds(t)
    check("real folds: 5 device-grouped folds covering all rows, all copies of a photo together",
          len(folds) == 5 and sum(len(te) for _, te in folds) == len(t) and all(t.photo.iloc[te].nunique() * 3 == len(te) for _, te in folds))
    n = R.gate_dedupe(t)
    check("real data: sha256 unique across 1455 raw files (recomputed from disk)", n == 1455)
    XBc = R.load_B(t, "clean")
    check("real B cache: clean block features found for all rows", XBc["b4"].shape == (1455, 1536))
    for c in ("social_sim", "jpeg75_resize05"):
        check(f"real B cache: {c} found for all rows", R.load_B(t, c)["b8"].shape == (1455, 1536))
    # smoke: method A on one real triplet under all conditions (no result is evaluated)
    one = t[t.photo == t.photo.iloc[0]]
    ok = True
    for r in one.itertuples():
        for c in R.CONDS:
            f = R.feat_A_from_Y(R.luma(R.decode_cond(R.ROOT / r.src_path, c, r.sha256)))
            ok &= f.shape == (990,) and bool(np.isfinite(f).all())
    check("real triplet: method A features finite under clean/social_sim/jpeg75_resize05", ok)


if __name__ == "__main__":
    t0 = time.time()
    t_primitives()
    t_step_on_real_jpeg()
    t_pipeline()
    t_label_and_structure()
    LINES.append(f"{len(LINES) - len(FAILS)}/{len(LINES)} passed; failures: {FAILS}; {time.time() - t0:.0f}s")
    print(LINES[-1])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "selftest.txt").write_text("\n".join(LINES) + "\n")
    sys.exit(1 if FAILS else 0)
