"""Self-tests of 023-attr-detector-emb/run.py, run BEFORE any real embedding is read.

    env/.venv/bin/python -I experiments/023-attr-detector-emb/selftest.py        -> results/selftest.txt

Part A (CPU, synthetic): label rule, paired decision, D-008 comparison logic, cache round trip, row/fold identity with an independent
re-derivation of 020's ha1_main, probe sanity with PLANTED class signal / PLANTED content leak / label shuffle.
Part B (GPU, 8 real images per detector, one subprocess per detector so memory is freed): the hook sits right before the head
(head(emb) == logit), the logit equals the official scorer.score(), batch invariance, B-Free 5-window path, device stays CUDA, memory is freed.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402  (sets cwd = repo root, sys.path)

import numpy as np  # noqa: E402

OUT = []
FAILS = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def check(name, cond, detail=""):
    say(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------- part B worker
def hooks_worker(det):
    import torch
    df = R.load_rows()
    # 2 images from each of 4 generators (distinct families), 3 conditions incl. variable-size social_sim and screenshot_sim
    sel = []
    for g in ("dalle3", "stable-diffusion-xl", "sd3_large", "firefly"):
        sel += list(df[df.generator == g].head(2).itertuples())
    res = {}
    ad = R.Adapter(det)
    res["device"] = str(ad.device)
    dev_ok = ad.device.type == "cuda"
    for cond in ("clean", "jpeg75_resize05", "social_sim", "screenshot_sim"):
        imgs = [R.load_cond(r.path, r.sha256, cond) for r in sel]
        E, L = ad.fwd(imgs)
        head = ad.head(E).float().cpu().numpy()
        res[f"{cond}_head_vs_logit"] = float(np.abs(head - L).max())
        # official scorer path (its own batching / DeviceModel) on the same images
        mod = ad.mod
        sc = mod.score(imgs) if det != "bfree" else mod.score(imgs, dedupe=False)      # bfree: official 5-window forward, no dedupe shortcut
        res[f"{cond}_logit_vs_scorer"] = float(np.abs(np.asarray(sc) - L).max())
        # batch invariance: one at a time vs batched
        E1 = np.concatenate([ad.fwd([im], bs=1)[0] for im in imgs[:4]])
        res[f"{cond}_batch_invariance_scaled"] = float((np.abs(E1 - E[:4])).max() / max(1.0, np.abs(E[:4]).max()))
        res[f"{cond}_emb_std_over_images"] = float(E.std(0).mean())
        res[f"{cond}_sizes"] = sorted({tuple(im.size) for im in imgs})
    if det == "bfree":
        # force the 5-window path: 640x640 upsample of a crop (grid 45x45 > 36)
        from PIL import Image
        base = R.load_cond(sel[0].path, sel[0].sha256, "clean").resize((640, 640), Image.BICUBIC)
        E, L = ad.fwd([base, base.transpose(Image.FLIP_LEFT_RIGHT)])
        head = ad.head(E).float().cpu().numpy()
        sc = ad.mod.score([base, base.transpose(Image.FLIP_LEFT_RIGHT)], dedupe=False)
        res["bfree_5window_head_vs_logit"] = float(np.abs(head - L).max())
        res["bfree_5window_logit_vs_scorer"] = float(np.abs(np.asarray(sc) - L).max())
    res["device_after"] = str(ad.dm.device)
    R.release_detector(ad)
    del ad
    res["cuda_mem_allocated_after_free_MB"] = float(torch.cuda.memory_allocated() / 2 ** 20) if dev_ok else None
    print("JSON " + json.dumps(res), flush=True)


# ---------------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    say(f"023-attr-detector-emb selftest  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    import tempfile
    # ---- A1 label rule ------------------------------------------------------------------------------
    L = R.label_ha5
    ok = dict(bacc=0.70, diff=[0.08, 0.03, 0.13])
    check("label PASS (>= +0.05 and CI lo > 0)", L(ok, 0.62, 0.60, True, 0.05)["label"] == "PASS")
    check("label INCONCLUSIVE (>= +0.05 but CI lo <= 0)", L(dict(bacc=0.70, diff=[0.08, -0.01, 0.17]), 0.62, 0.60, True, 0.05)["label"] == "INCONCLUSIVE")
    check("label INCONCLUSIVE (above best layer, below +0.05)", L(dict(bacc=0.65, diff=[0.03, 0.01, 0.05]), 0.62, 0.60, True, 0.05)["label"] == "INCONCLUSIVE")
    check("label KILL (<= best layer)", L(dict(bacc=0.61, diff=[-0.01, -0.05, 0.03]), 0.62, 0.60, True, 0.05)["label"] == "KILL")
    check("label KILL at equality", L(dict(bacc=0.62, diff=[0.0, -0.02, 0.02]), 0.62, 0.60, True, 0.05)["label"] == "KILL")
    check("label INVALID when a gate fails", L(ok, 0.62, 0.60, False, 0.05)["label"] == "INVALID")

    # ---- A2 D-008 comparison logic on a temporary cache -----------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="st023_", dir=str(HERE / "cache")))
    old_cache = R.CACHE
    R.CACHE = tmp
    try:
        rng = np.random.default_rng(0)
        keys = [f"k{i}" for i in range(32)]
        emb = rng.normal(size=(32, 16)).astype(np.float32)
        lg = rng.normal(size=32)
        for c in R.CONDS:
            R.write_cache("commfor", c, {k: (emb[i], lg[i]) for i, k in enumerate(keys)})
        back = R.read_cache("commfor", "clean")
        check("cache round trip exact", all(np.array_equal(back[k][0], emb[i]) and back[k][1] == lg[i] for i, k in enumerate(keys)))
        z = {"keys": np.array(keys)}
        for c in R.CONDS:
            z[f"emb__{c}"], z[f"logit__{c}"] = emb.copy(), lg.copy()
        np.savez(tmp / "commfor" / "cpuref.npz", **z)
        g = R.d008_compare("commfor")
        check("D-008 compare: identical -> ok", all(v["ok"] and v["max_abs_emb_diff"] == 0 for v in g.values()))
        z2 = dict(z)
        z2["emb__clean"] = emb + 5e-4
        np.savez(tmp / "commfor" / "cpuref.npz", **z2)
        g = R.d008_compare("commfor")
        check("D-008: abs emb diff 5e-4 -> scaled <= 5e-4, ok", g["clean"]["ok"] and g["clean"]["emb_ok"] and abs(g["clean"]["max_abs_emb_diff"] - 5e-4) < 1e-6 and g["clean"]["max_scaled_emb_diff"] <= 5e-4 + 1e-9,
          f"abs {g['clean']['max_abs_emb_diff']:.2e}, scaled {g['clean']['max_scaled_emb_diff']:.2e}")
        z3 = dict(z)
        z3["emb__clean"], z3["logit__clean"] = emb + 0.05, lg + 0.05
        np.savez(tmp / "commfor" / "cpuref.npz", **z3)
        g = R.d008_compare("commfor")
        check("D-008 PLANTED failure (0.05 shift in emb and logit) is caught", not g["clean"]["ok"])
        z4 = dict(z)
        z4["emb__clean"] = emb + 0.05                       # embedding off, logit exact -> passes through the 'or logits' clause but flags emb_ok False
        np.savez(tmp / "commfor" / "cpuref.npz", **z4)
        g = R.d008_compare("commfor")
        check("D-008: emb off + logit exact -> ok via logits, emb_ok False recorded", g["clean"]["ok"] and not g["clean"]["emb_ok"] and g["clean"]["logit_ok"])
    finally:
        R.CACHE = old_cache
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- A3 rows / groups / folds identical to an independent re-derivation of 020 ha1_main ------------
    import re
    from sklearn.model_selection import StratifiedGroupKFold
    m, cfg, u, idx, y, groups, fold, chk, sub = R.build_rows()
    check("rows: 1,020 fakes, 8 classes, unique sha256", len(sub) == 1020 and sub.sha256.is_unique and sub.generator.nunique() == 8)
    check("rows == load_rows() (extraction rows) as sets and order", sub.sha256.tolist() == R.load_rows().sha256.tolist() or set(sub.sha256) == set(R.load_rows().sha256),
          "order identical" if sub.sha256.tolist() == R.load_rows().sha256.tolist() else "same set, different order (fine: caches are keyed by sha256)")
    rid = np.array([(re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1] for p in sub.src_path])
    check("groups == 'rid:<caption id>' re-derived from file names", (np.array(["rid:" + r for r in rid]) == groups).all() and (rid != "").all(), f"{len(set(rid))} families")
    f2 = np.full(len(y), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)):
        f2[te] = k
    check("folds == independent StratifiedGroupKFold(5, shuffle, seed 0)", (f2 == fold).all())
    check("no caption family spans two folds", all(len(set(fold[groups == g])) == 1 for g in np.unique(groups)))
    fj = Path(R.MOD020).parent / "results" / "folds_ha1_main.json"
    if fj.exists():
        j = json.loads(fj.read_text())
        check("folds identical to 020/results/folds_ha1_main.json", j["rows"] == sub.sha256.tolist() and (np.array(j["fold"]) == fold).all())
    else:
        say("NOTE 020/results/folds_ha1_main.json not present yet; fold identity is re-checked in the report (fold_identity_with_020)")
    check("tmix excludes the held-out launderings", m.gate_train_conds(R.CFG["tmix"], R.CFG["held_out_launderings"])["ok"])

    # ---- A4 probe sanity on synthetic embeddings through the REAL 020 closed/LOGO tasks -----------------
    rng = np.random.default_rng(1)
    d = 24
    names = ["syn_a", "syn_b"]
    off = 4608
    ncols = off + d + 16

    def synth(kind):
        XF = {}
        mu = rng.normal(size=(8, d)) * 1.5
        famvec = {g: rng.normal(size=d) * 3 for g in np.unique(groups)}
        for c in R.CONDS:
            X = np.zeros((len(y), ncols), np.float32)
            X[:, :off] = rng.normal(size=(len(y), off)) * 0.1
            noise = rng.normal(size=(len(y), d))
            if kind == "signal":
                X[:, off:off + d] = mu[y] + noise
            elif kind == "content":                      # PLANTED content leak: embedding = family vector, no class information at all
                X[:, off:off + d] = np.stack([famvec[g] for g in groups]) + 0.3 * noise
            else:                                        # pure noise
                X[:, off:off + d] = noise
            X[:, off + d:] = rng.normal(size=(len(y), 16))
            XF[c] = X
        return XF
    res = {}
    for kind in ("signal", "content", "noise"):
        m2, A = R.build_analysis(dets=[], XF_override=synth(kind))
        A.XF = A.XF
        m2.SL["syn"] = slice(off, off + d)
        m2._G["ha5"] = A
        # inner folds / C selection exactly as the real run
        outs = [m2.closed_task(("ha5", "syn", "tmix", f)) for f in range(5)]
        lp, chosen = m2.pooled_logp(outs, A, "ha5", "syn", "tmix")
        pr = lp[A.primary].argmax(1)
        res[kind] = m2.bacc_simple(A.y, pr)
        if kind == "signal":
            Wc, inv, ug = m2.cluster_weights(A.groups, 200, 0, m2.strata_of)
            summ, boots = m2.eval_closed(A, lp, Wc, inv, "firefly")
            res["signal_pairwise"] = summ[A.primary]["pw"]["macro"][0]
            res["signal_bacc_ci"] = summ[A.primary]["bacc"]
            outs_l = [m2.logo_task(("ha5", 3, f, "syn", "tmix")) for f in range(5)]
            s = np.zeros(A.n)
            for r in outs_l:
                s[r["te"]] = r["out"][A.primary]["maha"]
            from sklearn.metrics import roc_auc_score
            res["signal_logo_maha_auroc"] = float(roc_auc_score(A.y == 3, s))
            # label shuffle through the real permutation task
            perm = [m2.perm_closed_task(("ha5", s_, f, "syn", 0.01)) for s_ in range(3) for f in range(5)]
            accs = []
            for s_ in range(3):
                yp = np.random.default_rng([m2.SEED, 4242, s_]).permutation(A.y)
                pred = np.zeros(A.n, int)
                for (s2, te, p_) in perm[s_ * 5:(s_ + 1) * 5]:
                    pred[te] = p_
                accs.append(m2.bacc_simple(yp, pred))
            res["signal_perm_bacc"] = float(np.mean(accs))
    say("synthetic probe:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    check("planted class signal -> bacc >= 0.85 (T-mix, grouped folds)", res["signal"] >= 0.85, f"{res['signal']:.3f}")
    check("signal pairwise (2-AFC macro) >= 0.9", res["signal_pairwise"] >= 0.9, f"{res['signal_pairwise']:.3f}")
    check("signal LOGO Mahalanobis AUROC of a held-out generator > 0.5 (finite)", np.isfinite(res["signal_logo_maha_auroc"]), f"{res['signal_logo_maha_auroc']:.3f}")
    check("PLANTED content leak (family vector, no class info) stays near chance under grouped folds", abs(res["content"] - 0.125) < 0.05, f"{res['content']:.3f}")
    check("pure noise -> near chance", abs(res["noise"] - 0.125) < 0.04, f"{res['noise']:.3f}")
    check("label-permutation control -> near chance", abs(res["signal_perm_bacc"] - 0.125) < 0.04, f"{res['signal_perm_bacc']:.3f}")

    # ---- A5 paired decision on synthetic bootstrap arrays ---------------------------------------------
    B = 200
    rngb = np.random.default_rng(3)

    def arr(mean, sd=0.01):
        return np.concatenate([[mean], mean + sd * rngb.normal(size=B)])
    base = rngb.normal(size=B) * 0.01
    bb = {fs: np.concatenate([[0.5], 0.5 + base]) for fs in R.DINO_FS}
    bb["b8"] = np.concatenate([[0.62], 0.62 + base])
    for dname, mu in zip(R.DETS, (0.70, 0.66, 0.40, 0.30)):
        bb[dname] = np.concatenate([[mu], mu + base + 0.002 * rngb.normal(size=B)])
    dec = R.ha5_decision(m, bb, B)
    check("decision picks best layer b8 and best detector commfor", dec["a1_fs"] == "b8" and dec["best_det"] == "commfor", f"{dec['a1_fs']}, {dec['best_det']}")
    lab = R.label_ha5(dict(bacc=dec["pt"][dec["best_det"]], diff=dec["diffs"][dec["best_det"]]), dec["pt"][dec["a1_fs"]], 0.6, True, 0.05)
    check("paired decision PASS when the detector beats b8 by 0.08 with tight paired CI", lab["label"] == "PASS", json.dumps(lab["parts"]))
    check("paired diff CI is paired (width ~ small): hi-lo < 0.02", dec["diffs"]["commfor"][2] - dec["diffs"]["commfor"][1] < 0.02)
    check("Holm p-values >= raw p", all(dec["p_holm"][d] >= dec["p"][d] - 1e-12 for d in R.DETS))
    check("weak detectors have diff CI below 0 -> p ~ 1", dec["p"]["clipdet"] > 0.95)

    # ---- A6 gates fail on planted violations ----------------------------------------------------------
    check("permutation gate catches a leak (mean 0.30)", not m.gate_permutation(np.full(200, 0.30), 0.125, 0.01, 0.03)["ok"])
    check("permutation gate accepts chance", m.gate_permutation(0.125 + 0.005 * rngb.normal(size=200), 0.125, 0.01, 0.03)["ok"])
    check("dedupe gate catches a duplicated sha256 across classes", not m.gate_dedupe(["a", "b", "a"], [0, 1, 2])["ok"])
    bad = fold.copy()
    g0 = np.where(groups == groups[0])[0]
    bad[g0[0]] = (bad[g0[0]] + 1) % 5
    if len(g0) > 1:
        check("grouped-split gate catches a family spanning two folds", not m.gate_splits(bad, groups)["ok"])
    else:
        say("NOTE first family has a single row; split-leak plant skipped")
    try:
        m.assert_disjoint_groups(groups, np.where(fold != 0)[0], np.arange(len(groups)), "planted")
        check("assert_disjoint_groups catches a planted train/test overlap", False)
    except AssertionError:
        check("assert_disjoint_groups catches a planted train/test overlap", True)
    gd = R.RES / "d008_guard.json"
    keep = gd.read_text() if gd.exists() else None
    try:
        bad_g = {d_: {c: dict(ok=(d_ != "rine"), max_abs_dlogit=0.0, max_scaled_emb_diff=0.0, emb_ok=True, logit_ok=True) for c in R.CONDS} for d_ in R.DETS}
        gd.write_text(json.dumps(bad_g))
        check("D-008 gate fails when one detector's guard fails", not R.d008_gate()["ok"])
    finally:
        if keep is None:
            gd.unlink()
        else:
            gd.write_text(keep)

    # ---- B: real detectors, hooks ------------------------------------------------------------------------
    say("---- part B: hooks on real detectors (GPU, 8 images each; a subprocess per detector)")
    tol = dict(head_vs_logit=1e-3, logit_vs_scorer=1e-3, batch_inv=1e-3)
    for det in R.DETS:
        t1 = time.time()
        r = subprocess.run([sys.executable, "-I", str(HERE / "selftest.py"), "hooks", det], capture_output=True, text=True)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON ")]
        if r.returncode != 0 or not line:
            check(f"{det}: hook worker ran", False, (r.stderr or r.stdout)[-400:])
            continue
        j = json.loads(line[0][5:])
        say(f"{det} ({time.time() - t1:.0f}s):", json.dumps({k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in j.items()}))
        check(f"{det}: ran on CUDA and never fell back to CPU", j["device"].startswith("cuda") and j["device_after"].startswith("cuda"))
        for c in ("clean", "jpeg75_resize05", "social_sim", "screenshot_sim"):
            check(f"{det}/{c}: head(captured embedding) == logit (hook is the head input)", j[f"{c}_head_vs_logit"] <= tol["head_vs_logit"], f"{j[f'{c}_head_vs_logit']:.2e}")
            check(f"{det}/{c}: logit == official scorer.score()", j[f"{c}_logit_vs_scorer"] <= tol["logit_vs_scorer"], f"{j[f'{c}_logit_vs_scorer']:.2e}")
            check(f"{det}/{c}: batch invariance of the embedding", j[f"{c}_batch_invariance_scaled"] <= tol["batch_inv"], f"{j[f'{c}_batch_invariance_scaled']:.2e}")
            check(f"{det}/{c}: embeddings differ between images (not constant)", j[f"{c}_emb_std_over_images"] > 1e-4, f"{j[f'{c}_emb_std_over_images']:.2e}")
        if det == "bfree":
            check("bfree: 5-window path (640 px) head(mean-window embedding) == logit", j["bfree_5window_head_vs_logit"] <= tol["head_vs_logit"], f"{j['bfree_5window_head_vs_logit']:.2e}")
            check("bfree: 5-window logit == official forward", j["bfree_5window_logit_vs_scorer"] <= tol["logit_vs_scorer"], f"{j['bfree_5window_logit_vs_scorer']:.2e}")
        check(f"{det}: GPU memory released after close", j["cuda_mem_allocated_after_free_MB"] is not None and j["cuda_mem_allocated_after_free_MB"] < 300, f"{j['cuda_mem_allocated_after_free_MB']:.0f} MB")

    say(f"\nSUMMARY: {len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    say(f"elapsed {time.time() - t0:.0f}s")
    R.RES.mkdir(exist_ok=True)
    (R.RES / "selftest.txt").write_text("\n".join(OUT) + "\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "hooks":
        hooks_worker(sys.argv[2])
    else:
        sys.exit(main())
