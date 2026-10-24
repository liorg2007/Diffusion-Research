"""Critic GPU/CPU checks for E4 (011). Nothing is written outside this folder; images stay in memory.
  Q2  orig arm with PNG (raw) round-trips instead of JPEG@source-QF; fov512 with BOTH members JPEG'd at the source QF.
  Q3  laundering components: resize(0.5), jpeg(75), jpeg(90) applied to both pair members; primary (clean probe) and matched.
  BLUR control: real vs Gaussian-blurred real, scored by the clean primary probes.
  Q4  reference-based scoring on test natives vs a 60/generator sample of fakes: recon error |x - VAE(x)| and probe delta
      s(VAE(x)) - s(x), at clean and jpeg75_resize05 (VAE applied to the laundered image).
  env/.venv/bin/python -I experiments/011-vae-trace/pilot/critic_checks/c2_gpu.py
"""
import importlib.util
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
E4 = HERE.parents[1]
spec = importlib.util.spec_from_file_location("run011", E4 / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.set_mode(False)


def log(*a):
    s = time.strftime("%H:%M:%S") + " CRIT " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(HERE / "c2_gpu.log", "a") as f:
        f.write(s + "\n")


R.log = log
R.e2.log = log          # keep 009's Batcher from writing to 011/pilot/run.log
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import features
from perturb import apply_condition

e2, p4run, P = R.e2, R.p4run, R.P
SEED = 0
OUT = {}
dev = e2.get_dev()
bb = p4run.secondary_backbone()
qf = json.loads((R.CACHE / "qf.json").read_text())
FAMS = ["F2", "F3", "F4"]


def cond(im, c, sha):
    return apply_condition(im.convert("RGB"), c, key=sha, seed=P.SEED)


def featurize(imgs):
    sec = np.concatenate([features.embed_images(bb, imgs[i:i + 240], 20, dev.type) for i in range(0, len(imgs), 240)]).astype(np.float32)
    with ThreadPoolExecutor(4) as pool:
        f3 = np.stack(list(pool.map(R.proc_image, imgs)))
    if getattr(bb, "force_cpu", False):
        raise SystemExit("accelerator fell back to CPU")
    return sec, f3


def X(fam, sec, f3):
    return R.fam_matrix(fam, sec, f3)


def pair_stats(sr, st):
    sr, st = np.asarray(sr), np.asarray(st)
    return dict(pooled=round(R.point_auc(sr, st), 4), win=round(float(((st > sr) + 0.5 * (st == sr)).mean()), 4))


def fit(sec_r, f3_r, sec_t, f3_t, grp_tr, fam):
    Xa = np.vstack([X(fam, sec_r, f3_r), X(fam, sec_t, f3_t)])
    y = np.r_[np.zeros(len(sec_r), int), np.ones(len(sec_t), int)]
    return e2.fit(Xa, y, np.concatenate([grp_tr, grp_tr]))


def vae_run(vae, arrs):
    """list of uint8 arrays (same size within a call) -> list of uint8 round-trips (raw q8, i.e. what a PNG would store)."""
    b = e2.Batcher()
    out = []
    for s in range(0, len(arrs), 48):
        out += list(e2.q8(b(lambda a: e2.recon_f(vae, a), np.stack(arrs[s:s + 48]))))
    return out


def jpeg_u8(u8, q):
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


vae = e2.load_vae(R.VN, dev)
REF_ROWS = []
for arm in R.ARMS:
    t0 = time.time()
    df, D = R.load_arm(arm)
    o = OUT.setdefault(arm, {})
    lab, sp, pool, devc, gen, sha = (df[c].to_numpy() for c in ("label", "split", "pool", "device", "generator", "sha256"))
    grp = e2.source_groups(df)
    tr = np.where((lab == 0) & (sp == "train"))[0]
    te = np.where((lab == 0) & (sp == "test"))[0]
    pr = np.r_[tr, te]
    ntr = len(tr)
    real_img = {i: Image.open(df.path.iloc[i]).convert("RGB") for i in pr}
    rt_img = {i: Image.open(R.rt_file(arm, sha[i])).convert("RGB") for i in pr}

    # ---- base: reproduce the reported clean-primary probes; GPU features vs cache sanity
    base = {}
    for fam in FAMS:
        m = fit(D[("sec_plain", "clean")][tr], D[("f3_plain", "clean")][tr], D[("sec_rt", "clean")][tr], D[("f3_rt", "clean")][tr], grp[tr], fam)
        base[fam] = m
        o.setdefault("reproduce_clean", {})[fam] = pair_stats(m.decision_function(X(fam, D[("sec_plain", "clean")][te], D[("f3_plain", "clean")][te])),
                                                            m.decision_function(X(fam, D[("sec_rt", "clean")][te], D[("f3_rt", "clean")][te])))
    chk = [real_img[i] for i in te[:24]]
    s_chk, f_chk = featurize(chk)
    o["feat_recompute_vs_cache_max_abs"] = float(np.abs(s_chk - D[("sec_plain", "clean")][te[:24]]).max())
    o["f3_recompute_vs_cache_max_abs"] = float(np.abs(f_chk - D[("f3_plain", "clean")][te[:24]]).max())
    log(arm, "base", o["reproduce_clean"], o["feat_recompute_vs_cache_max_abs"])

    def run_condition(name, real_imgs, rt_imgs, primary_models=base):
        """real_imgs/rt_imgs: lists aligned with pr (train first). Returns primary + matched pair stats on test."""
        sr, fr = featurize(real_imgs)
        st, ft = featurize(rt_imgs)
        res = {}
        for fam in FAMS:
            m = primary_models[fam]
            res.setdefault("primary", {})[fam] = pair_stats(m.decision_function(X(fam, sr[ntr:], fr[ntr:])), m.decision_function(X(fam, st[ntr:], ft[ntr:])))
            mm = fit(sr[:ntr], fr[:ntr], st[:ntr], ft[:ntr], grp[tr], fam)
            res.setdefault("matched", {})[fam] = pair_stats(mm.decision_function(X(fam, sr[ntr:], fr[ntr:])), mm.decision_function(X(fam, st[ntr:], ft[ntr:])))
        lr, lt = fr[ntr:, R.F3_DIM], ft[ntr:, R.F3_DIM]
        res["lap_rt_lower_frac"] = float((lt < lr).mean())
        res["lap_median_ratio"] = float(np.median(lt / lr))
        log(arm, name, res)
        return res, (sr, fr, st, ft)

    # ---- Q3: laundering components (pre-registered round-trips: orig JPEG@QF, fov512 PNG)
    for c in ("jpeg75_resize05", "resize(0.5)", "jpeg(75)", "jpeg(90)"):
        res, _ = run_condition(c, [cond(real_img[i], c, sha[i]) for i in pr], [cond(rt_img[i], c, sha[i]) for i in pr])
        o.setdefault("Q3_components", {})[c] = res

    # ---- BLUR control: real vs Gaussian-blurred real (no VAE), clean primary probes; test only (train = real vs VAE as before)
    sr_te, fr_te = D[("sec_plain", "clean")][te], D[("f3_plain", "clean")][te]
    for rad in (0.5, 0.8):
        bl = [real_img[i].filter(ImageFilter.GaussianBlur(rad)) for i in te]
        sb, fb = featurize(bl)
        r = {fam: pair_stats(base[fam].decision_function(X(fam, sr_te, fr_te)), base[fam].decision_function(X(fam, sb, fb))) for fam in FAMS}
        r["lap_median_ratio"] = float(np.median(fb[:, R.F3_DIM] / fr_te[:, R.F3_DIM]))
        o.setdefault("blur_control", {})[f"gauss{rad}"] = r
        log(arm, "blur", rad, r)

    # ---- Q2
    if arm == "orig":
        raw = vae_run(vae, [np.asarray(real_img[i]) for i in pr])
        # consistency: JPEG@QF of the raw output == cached file (regeneration check on 32)
        mx = 0
        for k in range(0, len(pr), max(1, len(pr) // 32)):
            mx = max(mx, int(np.abs(jpeg_u8(raw[k], qf[sha[pr[k]]]).astype(int) - np.asarray(rt_img[pr[k]]).astype(int)).max()))
        o["Q2_regen_jpeg_vs_cache_max_abs"] = mx
        png_rt = [Image.fromarray(u) for u in raw]
        q2 = {}
        res_c, (sr, fr, st, ft) = run_condition("Q2 orig PNG-rt clean", [real_img[i] for i in pr], png_rt)
        # primary for laundered = probe fit on clean PNG pairs
        prim = {fam: fit(sr[:ntr], fr[:ntr], st[:ntr], ft[:ntr], grp[tr], fam) for fam in FAMS}
        q2["clean"] = res_c
        q2["jpeg75_resize05"], _ = run_condition("Q2 orig PNG-rt laundered", [cond(real_img[i], "jpeg75_resize05", sha[i]) for i in pr],
                                                 [cond(im, "jpeg75_resize05", sha[i]) for im, i in zip(png_rt, pr)], primary_models=prim)
        o["Q2_png_roundtrips"] = q2
        orig_png_rt = {i: im for i, im in zip(pr, png_rt)}
    else:
        # reverse: fov512 with BOTH members JPEG'd at the source QF (as if the arm had been JPEG-coded at the source quality)
        rj = [Image.fromarray(jpeg_u8(np.asarray(real_img[i]), qf[sha[i]])) for i in pr]
        tj = [Image.fromarray(jpeg_u8(np.asarray(rt_img[i]), qf[sha[i]])) for i in pr]
        res_c, (sr, fr, st, ft) = run_condition("Q2 fov512 both JPEG@QF", rj, tj)
        o["Q2_fov512_bothJPEGatQF"] = res_c
        tj_only = [Image.fromarray(jpeg_u8(np.asarray(rt_img[i]), qf[sha[i]])) for i in pr]
        res_c2, _ = run_condition("Q2 fov512 rt-only JPEG@QF", [real_img[i] for i in pr], tj_only)
        o["Q2_fov512_rtonlyJPEGatQF"] = res_c2

    # ---- Q4 reference-based: test natives vs fakes (60 per generator, seeded), clean and laundered
    rng = np.random.default_rng(SEED)
    nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
    fk = []
    for g in R.GENS:
        ii = np.where((lab == 1) & (gen == g))[0]
        fk += sorted(rng.choice(ii, 60, replace=False).tolist())
    fk = np.array(fk)
    for c in ("clean", "jpeg75_resize05"):
        xs_nat = [cond(real_img[i], c, sha[i]) for i in nat]
        xs_fk = [cond(Image.open(df.path.iloc[i]), c, sha[i]) for i in fk]
        if c == "clean" and arm == "orig":
            ys_nat = [np.asarray(orig_png_rt[i]) for i in nat]             # raw VAE output (no JPEG re-save)
        elif c == "clean":
            ys_nat = [np.asarray(rt_img[i]) for i in nat]                  # fov512 rt are PNG = raw
        else:
            ys_nat = vae_run(vae, [np.asarray(x) for x in xs_nat])
        ys_fk = vae_run(vae, [np.asarray(x) for x in xs_fk])
        xs = xs_nat + xs_fk
        ys = [Image.fromarray(np.asarray(y)) for y in ys_nat] + [Image.fromarray(np.asarray(y)) for y in ys_fk]
        sx, fx = featurize(xs)
        sy, fy = featurize(ys)
        err = []
        for x, y in zip(xs, ys):
            a = np.asarray(R.center_crop(x, 224), np.float32)
            b = np.asarray(R.center_crop(y, 224), np.float32)
            err.append(float(np.abs(a - b).mean()))
        err = np.array(err)
        lapratio = fy[:, R.F3_DIM] / fx[:, R.F3_DIM]
        rows = dict(idx=np.r_[nat, fk], label=np.r_[np.zeros(len(nat), int), np.ones(len(fk), int)], err=err, lapratio=lapratio)
        for fam in FAMS:
            m = base[fam]
            s0 = m.decision_function(X(fam, sx, fx))
            s1 = m.decision_function(X(fam, sy, fy))
            rows[f"s_{fam}"] = s0
            rows[f"d_{fam}"] = s1 - s0
        t = pd.DataFrame(rows)
        t["arm"], t["cond"], t["generator"], t["device"] = arm, c, gen[t.idx], devc[t.idx]
        REF_ROWS.append(t)
        log(arm, "Q4", c, "done", f"{time.time() - t0:.0f}s")
    log(arm, "arm done", f"{time.time() - t0:.0f}s")
    (HERE / "c2_gpu.json").write_text(json.dumps(OUT, indent=1, default=float))

pd.concat(REF_ROWS).to_csv(HERE / "c2_ref_scores.csv.gz", index=False)
(HERE / "c2_gpu.json").write_text(json.dumps(OUT, indent=1, default=float))
log("all done")
