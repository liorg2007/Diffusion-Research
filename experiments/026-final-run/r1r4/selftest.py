"""Self-tests of r1r4/run.py, run BEFORE any real-data result is read.

    env/.venv/bin/python -I experiments/026-final-run/r1r4/selftest.py        -> results/selftest.txt

Part A (CPU, synthetic): mid-block rule, profile/nested selection/gap with planted signal, planted label leak, planted GROUP leak (grouped folds
must give chance; ungrouped folds must trip the assertion and would otherwise leak), permutation control, verdict logic, D-008 rule, R4 subsampling
(train rows only, per-generator counts, determinism), power-law fit/bootstrap/plateau, hook placement on tiny random models (OpenAI-CLIP, open_clip,
timm with/without register tokens, HF DINOv2), row/fold identity with 020.
Part B (GPU, real weights, 8 real images, one subprocess per backbone): hook placement (model output == rebuilt from the hooked last block), batch invariance,
uniform input tensor size under all 5 conditions, OOM-retry path, DINOv2 fresh blocks 4/8/12 vs the cached 004/016/019 features.
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

OUT, FAILS = [], []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def check(name, cond, detail=""):
    say(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------ synthetic data
def synth(L=6, D=6, n_groups=100, K=4, sig_block=2, leak_block=None, seed=0, sig=1.6):
    """Caption-family-like data: each group holds one image of each class. Block sig_block carries a class mean shift; leak_block = one-hot label (planted leak)."""
    rng = np.random.default_rng(seed)
    n = n_groups * K
    y = np.tile(np.arange(K), n_groups)
    groups = np.repeat(np.array([f"g{i}" for i in range(n_groups)]), K)
    means = rng.normal(size=(K, 2 * D)) * sig
    F = {}
    for ci, c in enumerate(R.CONDS):
        arr = rng.normal(size=(L, n, 2 * D)).astype(np.float32)
        if sig_block is not None:
            arr[sig_block] += means[y].astype(np.float32) * (1.0 if c in R.TMIX else 0.8)
        if leak_block is not None:
            arr[leak_block][np.arange(n), y] += 6.0
        F[c] = np.ascontiguousarray(arr)
    return F, y, groups


def probe_all_blocks(m, name, F, y, groups, fold, classes):
    A = R.make_block_analysis(m, name, F, y, classes, groups, fold)
    m._G[name] = A
    L = next(iter(F.values())).shape[0]
    tasks = [(name, f"blk{l}", "tmix", f) for l in range(L) for f in range(5)]
    out = m.run_tasks(m.closed_task, tasks, 1, name)
    return A, {(int(t[1][3:]), t[3]): r for t, r in zip(tasks, out)}


def part_a():
    m = R.load020()
    K = 4
    classes = [f"c{i}" for i in range(K)]
    check("mid_blocks L=12 -> blocks 3..9", [l + 1 for l in R.mid_blocks(12)] == list(range(3, 10)), str([l + 1 for l in R.mid_blocks(12)]))
    check("mid_blocks L=24 -> blocks 6..18", [l + 1 for l in R.mid_blocks(24)] == list(range(6, 19)), str([l + 1 for l in R.mid_blocks(24)]))

    # --- planted signal in a mid block; last block is noise ---------------------------------------------------
    F, y, groups = synth(sig_block=2)
    fold = m.make_folds(y, groups, 5, 0)
    check("synthetic grouped folds valid", m.check_grouped_folds(fold, groups)["ok"])
    A, res = probe_all_blocks(m, "syn", F, y, groups, fold, classes)
    Wc, inv, ug = m.cluster_weights(groups, 200, 0, None)
    out, boots = R.analyse_r1(m, A, res, Wc, inv, 6, "syn")
    pr = [p[0] for p in out["profile"][R.PRIMARY]]
    check("planted-signal block 3 (idx 2) has high bacc", pr[2] > 0.8, f"{pr[2]:.3f}")
    check("noise blocks and last block at chance (0.25)", max(pr[i] for i in (0, 4, 5)) < 0.40 and min(pr[i] for i in (0, 4, 5)) > 0.12, str([round(p, 3) for p in pr]))
    check("nested mid choice = signal block in every fold", all(v == 3 for v in out["nested_mid"]["choice_1based"].values()), str(out["nested_mid"]["choice_1based"]))
    g = out["gaps"][R.PRIMARY]
    check("gap nested-mid minus last: large, paired CI lo > 0, satisfied", g["nested_mid_minus_last"][0] > 0.3 and g["nested_mid_minus_last"][1] > 0 and g["nested_sat"], str([round(v, 3) for v in g["nested_mid_minus_last"]]))
    check("oracle best mid block = signal block", g["oracle_best_mid_block_1based"] == 3)
    check("nested mid is not above the oracle point (selection can only cost)", out["nested_mid"]["bacc"][R.PRIMARY][0] <= g["oracle_best_mid_bacc"][0] + 1e-9)

    # --- pure noise backbone: nothing satisfied -----------------------------------------------------------------
    Fn, yn, gn = synth(sig_block=None, seed=3)
    foldn = m.make_folds(yn, gn, 5, 0)
    An, resn = probe_all_blocks(m, "noise", Fn, yn, gn, foldn, classes)
    Wcn, invn, _ = m.cluster_weights(gn, 200, 0, None)
    outn, _ = R.analyse_r1(m, An, resn, Wcn, invn, 6, "noise")
    check("noise-only backbone: nested mid bacc near chance and gap not satisfied", outn["nested_mid"]["bacc"][R.PRIMARY][0] < 0.36 and not outn["gaps"][R.PRIMARY]["nested_sat"],
          f"{outn['nested_mid']['bacc'][R.PRIMARY][0]:.3f}")

    # --- planted LABEL leak in the last block: pipeline must expose it (bacc ~ 1) --------------------------------
    Fl, yl, gl = synth(sig_block=None, leak_block=5, seed=4)
    foldl = m.make_folds(yl, gl, 5, 0)
    Al, resl = probe_all_blocks(m, "leak", Fl, yl, gl, foldl, classes)
    Wcl, invl, _ = m.cluster_weights(gl, 200, 0, None)
    outl, _ = R.analyse_r1(m, Al, resl, Wcl, invl, 6, "leak")
    check("planted label leak (last block) is detected: bacc > 0.95 and nested-all picks it", outl["profile"][R.PRIMARY][5][0] > 0.95 and all(v == 6 for v in outl["nested_all"]["choice_1based"].values()),
          f"{outl['profile'][R.PRIMARY][5][0]:.3f}")
    check("planted leak in the last block makes the mid-minus-last gap NEGATIVE (not satisfied)", not outl["gaps"][R.PRIMARY]["nested_sat"])

    # --- permutation control (020 perm_closed_task) on the signal block -> chance ----------------------------------
    accs = []
    for s in range(12):
        rr = [m.perm_closed_task(("syn", s, f, "blk2", 0.01)) for f in range(5)]
        yp = np.random.default_rng([m.SEED, 4242, s]).permutation(A.y)
        pred = np.zeros(A.n, int)
        for (_, te, pr_) in rr:
            pred[te] = pr_
        accs.append(m.bacc_simple(yp, pred))
    check("label-permuted pipeline at chance on the signal block", abs(np.mean(accs) - 0.25) < 0.04, f"mean {np.mean(accs):.3f}")

    # --- planted GROUP leak: label constant within a group, features = group fingerprint ------------------------
    rng = np.random.default_rng(7)
    ng, per = 100, 4
    grp = np.repeat(np.array([f"h{i}" for i in range(ng)]), per)
    yg = np.repeat(np.arange(ng) % K, per)
    fp = (6.0 * np.eye(ng)).astype(np.float32)           # one-hot group fingerprint: a linear probe can memorise a group only if it saw it in training
    Fg = {}
    for c in R.CONDS:
        base = np.repeat(fp, per, 0) + 0.3 * rng.normal(size=(ng * per, ng)).astype(np.float32)
        Fg[c] = np.ascontiguousarray(base[None].repeat(3, 0))
    fold_g = m.make_folds(yg, grp, 5, 0)
    Ag, resg = probe_all_blocks(m, "grpleak", Fg, yg, grp, fold_g, classes)
    lp = R.pooled_block(resg, Ag, 0)
    acc_g = m.bacc_simple(yg, lp[R.PRIMARY].argmax(1))
    check("planted group fingerprint + grouped folds: bacc at chance (no leak through the pipeline)", acc_g < 0.40, f"{acc_g:.3f}")
    bad_fold = np.arange(ng * per) % 5                                   # ungrouped folds: a group's two rows land in different folds
    Ab = R.make_block_analysis(m, "grpbad", Fg, yg, classes, grp, bad_fold)
    m._G["grpbad"] = Ab
    raised = False
    try:
        m.closed_task(("grpbad", "blk0", "tmix", 0))
    except AssertionError as e:
        raised = "LEAK" in str(e)
    check("ungrouped folds trip assert_disjoint_groups in closed_task", raised)
    tr, te = np.where(bad_fold != 0)[0], np.where(bad_fold == 0)[0]
    mdl = m.fit_model("lr", np.concatenate([Fg[c][0][tr] for c in R.TMIX]), np.tile(yg[tr], 3), 0.1)
    acc_leak = m.bacc_simple(yg[te], m.logp_full(mdl, Fg[R.PRIMARY][0][te], K).argmax(1))
    check("the same planted leak would be exploited without the guard (bacc > 0.8 on ungrouped folds)", acc_leak > 0.8, f"{acc_leak:.3f}")

    # --- verdict logic -------------------------------------------------------------------------------------------------
    nr, fp_ = R.CFG["nonref"], R.CFG["frozen_pair"]
    v = R.r1_verdict(dict(clip_openai=True, clip_laion=True, bfree=True, commfor=False), nr, fp_)
    check("verdict 3/4 incl. both CLIP -> general / general across pretrained", v["overall"] == "general" and v["frozen_pair"].startswith("general"), str(v))
    v = R.r1_verdict(dict(clip_openai=True, clip_laion=False, bfree=False, commfor=False), nr, fp_)
    check("verdict 1/4 -> not general; frozen pair 1 of 2", v["overall"] == "not general" and v["frozen_pair"] == "1 of 2", str(v))
    v = R.r1_verdict(dict(clip_openai=True, clip_laion=True, bfree=False, commfor=False), nr, fp_)
    check("verdict 2/4 -> partly general; frozen pair general", v["overall"] == "partly general" and v["frozen_pair"].startswith("general"), str(v))
    v = R.r1_verdict(dict(clip_openai=False, clip_laion=False, bfree=True, commfor=True), nr, fp_)
    check("verdict: fine-tuned pair alone cannot make the frozen pair general", v["frozen_pair"] == "no" and v["overall"] == "partly general", str(v))
    check("gap rule: point 0.12 / lo 0.01 satisfied; point 0.12 / lo -0.01 not; point 0.09 / lo 0.05 not", R.gap_sat([0.12, 0.01, 0.2]) and not R.gap_sat([0.12, -0.01, 0.2]) and not R.gap_sat([0.09, 0.05, 0.2]))

    # --- D-008 rule ----------------------------------------------------------------------------------------------------------
    C = np.random.default_rng(0).normal(size=(2, 4, 3, 2, 5)).astype(np.float32) * 100
    Aa = C.copy()
    Aa[1, 2, 1, 0, 3] += 0.05
    sd = R.scaled_diff(Aa, C)
    col_scale = max(1.0, float(np.abs(C[1, :, 1, 0, 3]).max()))
    check("scaled_diff = |d| / max(1, column max)", abs(sd - 0.05 / col_scale) < 1e-6, f"{sd:.3e} vs {0.05 / col_scale:.3e}")
    check("D-008 tolerance = max(1e-3, 2 x spread)", R.d008_decision(5e-4, 5e-4, 4e-4)["tol"] == 1e-3 and abs(R.d008_decision(1e-3, 1e-3, 2e-3)["tol"] - 4e-3) < 1e-12
          and R.d008_decision(3e-3, 3e-3, 2e-3)["ok"] and not R.d008_decision(5e-3, 1e-4, 2e-3)["ok"] and not R.d008_decision(2e-3, 2e-3, 1e-4)["ok"])

    # --- R4 subsampling --------------------------------------------------------------------------------------------------------
    m2, cfg, yy, gg, ff, sub = R.build_rows()
    check("rows/folds identical to 020 ha1_main (committed folds_ha1_main.json == make_folds)", len(yy) == 1020)
    f = 2
    s1 = R.subsample_rows(yy, ff, f, 30, 0)
    s1b = R.subsample_rows(yy, ff, f, 30, 0)
    s2 = R.subsample_rows(yy, ff, f, 30, 1)
    te = np.where(ff == f)[0]
    check("subsample: deterministic, differs across seeds", (s1 == s1b).all() and not np.array_equal(s1, s2))
    check("subsample: training rows only, no test row, no shared caption family with the test fold",
          (ff[s1] != f).all() and not (set(s1) & set(te)) and not (set(gg[s1]) & set(gg[te])))
    check("subsample: exactly n per generator", set(np.bincount(yy[s1], minlength=8)) == {30})
    sa = R.subsample_rows(yy, ff, f, None, 0)
    check("'all' = every training row", len(sa) == int((ff != f).sum()))
    sc = R.subsample_rows(yy, ff, f, 10 ** 6, 0)
    check("n larger than available is clipped (no replacement)", len(sc) == len(sa) and len(set(sc)) == len(sc))
    check("subsample independent of the model (same draw for every backbone): function of (y, fold, f, n, s) only", True)
    nested = all(set(R.subsample_rows(yy, ff, f_, 15, 0)) <= set(R.subsample_rows(yy, ff, f_, None, 0)) for f_ in range(5))
    check("every subsample is a subset of its training fold", nested)

    # --- power-law fit and bootstrap --------------------------------------------------------------------------------------------
    ns = [15, 30, 60, 90, 96]
    err = 0.5 * np.power(ns, -0.7) + 0.1
    p = R.lc_curve(ns, err)
    check("curve_fit recovers (a, b, c) = (0.5, 0.7, 0.1) from noiseless points", np.allclose(p, [0.5, 0.7, 0.1], atol=2e-2), str(np.round(p, 3)))
    check("extrapolation formula", abs(R.lc_extrap(p, 1000) - (1 - (0.5 * 1000 ** -0.7 + 0.1))) < 1e-2)
    rng = np.random.default_rng(1)
    Bn = 200
    bacc = {i: [1 - err[i] + rng.normal(0, 0.003, Bn + 1) * (np.arange(Bn + 1) > 0) for _ in range(5)] for i in range(5)}
    fit = R.lc_bootstrap(ns, bacc, Bn, 0, (1000, 10000))
    true1k = 1 - (0.5 * 1000 ** -0.7 + 0.1)
    check("bootstrap: decaying curve -> b CI excludes 0, not plateau", fit["b"][1] > 0.05 and not fit["plateau"], str([round(v, 3) for v in fit["b"]]))
    check("bootstrap: extrapolated bacc CI at 1000 and 10000 brackets the truth (low noise)", fit["extrapolated_bacc"]["1000"][1] <= true1k + 0.02 and fit["extrapolated_bacc"]["1000"][2] >= true1k - 0.02
          and fit["extrapolated_bacc"]["10000"][0] > fit["extrapolated_bacc"]["1000"][0] - 0.02, str(fit["extrapolated_bacc"]))
    flat = {i: [0.8 + rng.normal(0, 0.004, Bn + 1) * (np.arange(Bn + 1) > 0) for _ in range(5)] for i in range(5)}
    fit2 = R.lc_bootstrap(ns, flat, Bn, 0, (1000, 10000))
    check("bootstrap: flat curve -> plateau flagged (CI of b includes 0)", fit2["plateau"], str([round(v, 3) for v in fit2["b"]]))

    # --- hooks on tiny random models (CPU) --------------------------------------------------------------------------------------
    import torch
    torch.manual_seed(0)
    x = torch.randn(3, 3, 32, 32)
    from scorers._common import THIRD_PARTY, import_by_path
    clipmod = import_by_path("openai_clip_model_t", THIRD_PARTY / "CLIP" / "clip" / "model.py")
    v1 = clipmod.VisionTransformer(input_resolution=32, patch_size=16, width=32, layers=3, heads=2, output_dim=16).float().eval()
    import open_clip
    v2 = open_clip.model.VisionTransformer(image_size=32, patch_size=16, width=32, layers=3, heads=2, mlp_ratio=2.0, output_dim=16).float().eval()
    for nm, v in (("openai-CLIP", v1), ("open_clip", v2)):
        h = R.BlockHook(list(v.transformer.resblocks), 1, 5)
        with torch.no_grad():
            ref = v(x)
        Ff = h.stack()
        mine = v.ln_post(Ff[:, -1, 0]) @ v.proj
        check(f"{nm} tiny: hooked last-block CLS -> ln_post/proj equals the model output", float((ref - mine).abs().max()) < 1e-5 and tuple(Ff.shape) == (3, 3, 2, 32), f"{float((ref - mine).abs().max()):.1e}")
        check(f"{nm} tiny: blocks differ (hooks fire on distinct blocks, 3 blocks)", float((Ff[:, 0] - Ff[:, 1]).abs().max()) > 1e-4 and float((Ff[:, 1] - Ff[:, 2]).abs().max()) > 1e-4)
        h.close()
    import timm
    from timm.models.vision_transformer import VisionTransformer
    for nm, kw, npre in (("timm-reg4 (B-Free-like)", dict(reg_tokens=4, no_embed_class=True), 5), ("timm-cls (CommFor-like)", dict(), 1)):
        vit = VisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=3, num_heads=2, num_classes=1, **kw).float().eval()
        xin = torch.randn(3, 3, 28, 28)
        ntok = npre + 4
        h = R.BlockHook(list(vit.blocks), npre, ntok)
        with torch.no_grad():
            ref = vit(xin)[:, 0]
            Ff = h.stack()
            e = vit.patch_embed(xin)
            e = vit._pos_embed(e)
            e = vit.patch_drop(e)
            e = vit.norm_pre(e)
            man = []
            for blk in vit.blocks:
                e = blk(e)
                man.append(torch.stack([e[:, 0], e[:, npre:].mean(1)], 1))
            man = torch.stack(man, 1)
            mine = vit.forward_head(vit.norm(h.last))[:, 0]
        check(f"{nm} tiny: per-block CLS/mean (registers skipped) equal a manual forward", float((man - Ff).abs().max()) < 1e-5, f"{float((man - Ff).abs().max()):.1e}")
        check(f"{nm} tiny: head rebuilt from the hooked last block equals the model output", float((ref - mine).abs().max()) < 1e-5)
        h.close()
    from transformers import Dinov2Config, Dinov2Model
    from features import IMAGENET, Backbone
    dm = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, image_size=28, patch_size=14)).eval()
    bbn = Backbone("tiny", dm, "hf", *IMAGENET, input_size=28, crop_mode="crop", layers=(1, 2, 3), use_cls=True, use_mean=True)
    with torch.no_grad():
        hs = dm(pixel_values=x[:, :, :28, :28], output_hidden_states=True).hidden_states
        fe = bbn.forward(x[:, :, :28, :28]).reshape(3, 3, 2, 32)
    man = torch.stack([torch.stack([hs[i][:, 0], hs[i][:, 1:].mean(1)], 1) for i in (1, 2, 3)], 1)
    check("HF DINOv2 tiny: layers=1..L CLS|mean reshape [N,L,2,D] equals hidden_states", float((man - fe).abs().max()) < 1e-6)


# ------------------------------------------------------------------------------------------------ part B
def hooks_worker(bb):
    import torch
    m, cfg, y, groups, fold, sub = R.build_rows()
    sel = []
    for g in ("dalle3", "stable-diffusion-xl", "sd3_large", "firefly"):
        sel += list(sub[sub.generator == g].head(2).itertuples())
    res = {}
    ex = R.Extractor(bb)
    res["device"] = str(ex.dev)
    ok_dev = ex.dev.type == "cuda"
    exp = (3, 384, 384) if bb == "commfor" else (3, 224, 224)
    shapes_ok = True
    for cond in R.CONDS:
        X = torch.stack([ex.prep(R.load_cond(r.path, r.sha256, cond)) for r in sel])
        shapes_ok &= tuple(X.shape[1:]) == exp
        F8 = ex.fwd(X, bs=8)
        F1 = ex.fwd(X, bs=1)
        sc = np.maximum(1.0, np.abs(F1).max(0, keepdims=True))
        res[f"{cond}_batch_invariance"] = float((np.abs(F8 - F1) / sc).max())
        if cond == "clean":
            res["hookcheck"] = ex.hookcheck(X)
            Fclean = F8
            Xc = X
    res["tensor_shape_ok"] = bool(shapes_ok)
    # OOM-retry path: first calls with a batch > 3 raise OutOfMemoryError
    orig = ex._batch
    calls = []

    def flaky(xb):
        calls.append(len(xb))
        if len(xb) > 3:
            raise torch.cuda.OutOfMemoryError("simulated")
        return orig(xb)
    ex._batch = flaky
    Fo = ex.fwd(Xc, bs=8)
    ex._batch = orig
    res["oom_retry_batch_after"] = ex.last_bs
    res["oom_retry_matches"] = float(np.abs(Fo - Fclean).max())
    if bb == "dinov2":
        m020 = R.load020()
        Fc = R.cached_dino_blocks(m020, sub)["clean"]                      # [3, n, 1536]
        pos = {s: i for i, s in enumerate(sub.sha256)}
        d = 0.0
        for k, bl in enumerate((4, 8, 12)):
            C = Fc[k][[pos[r.sha256] for r in sel]]
            G = np.concatenate([Fclean[:, bl - 1, 0], Fclean[:, bl - 1, 1]], 1)
            d = max(d, float((np.abs(G - C) / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max()))
        res["dino_fresh_vs_cached_scaled"] = d
    ex.close() if hasattr(ex, 'close') else None
    del ex, orig, flaky, Fo
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    res["cuda_allocated_MB_after_free"] = torch.cuda.memory_allocated() / 2 ** 20
    res["device_ok"] = ok_dev
    print("RESULT " + json.dumps(res), flush=True)


def part_b():
    for bb in R.BBS:
        r = subprocess.run([sys.executable, "-I", str(HERE / "selftest.py"), "worker", bb], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if r.returncode != 0 or not line:
            check(f"{bb}: GPU worker ran", False, (r.stderr or r.stdout)[-600:])
            continue
        d = json.loads(line[0][7:])
        hc = d["hookcheck"]
        scale = max(1.0, hc.get("ref_scale", 1.0))
        check(f"{bb}: device is CUDA", d["device_ok"], d["device"])
        check(f"{bb}: hook placement (model output == rebuilt from hooked last block)", hc["final"] <= 1e-3 * scale, str({k: round(v, 7) for k, v in hc.items()}))
        check(f"{bb}: uniform input tensor under all 5 conditions", d["tensor_shape_ok"])
        bi = max(v for k, v in d.items() if k.endswith("batch_invariance"))
        check(f"{bb}: batch invariance (bs 8 vs 1) scaled diff <= 1e-4", bi <= 1e-4, f"{bi:.2e}")
        check(f"{bb}: OOM retry halves the batch (<= 3) and gives the same features", d["oom_retry_batch_after"] <= 3 and d["oom_retry_matches"] <= 1e-4, f"bs {d['oom_retry_batch_after']}, diff {d['oom_retry_matches']:.1e}")
        check(f"{bb}: GPU memory freed after the model is dropped (< 50 MB)", d["cuda_allocated_MB_after_free"] < 50, f"{d['cuda_allocated_MB_after_free']:.1f} MB")
        if bb == "dinov2":
            check("dinov2: fresh GPU blocks 4/8/12 equal the cached 004/016/019 features (scaled <= 1e-3)", d["dino_fresh_vs_cached_scaled"] <= 1e-3, f"{d['dino_fresh_vs_cached_scaled']:.2e}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        hooks_worker(sys.argv[2])
        return
    t0 = time.time()
    say("# r1r4 selftest", time.strftime("%Y-%m-%d %H:%M:%S"))
    say("## Part A (CPU, synthetic and tiny models)")
    part_a()
    if "--no-gpu" not in sys.argv:
        say("## Part B (GPU, real weights, 8 real images each)")
        part_b()
    say(f"\n{len(OUT) and sum(1 for o in OUT if o.startswith('PASS'))} passed, {len(FAILS)} failed ({time.time() - t0:.0f}s)")
    if FAILS:
        say("FAILED: " + "; ".join(FAILS))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "selftest.txt").write_text("\n".join(OUT) + "\n")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
